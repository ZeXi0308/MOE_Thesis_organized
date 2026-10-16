"""vLLM 0.26 eager worker, per-router policy and explicit scheduler phases.

Install with worker_cls='skip_worker.ExpertSkipWorker', moe_backend='triton',
enforce_eager=True, compilation_config={'mode': 0, 'cudagraph_mode': 'NONE'}.
Policy is disabled during model initialization. RPCs switch policies outside
measurement intervals. Trace/logits capture is diagnostic, never timed as TPOT.
"""
import importlib.metadata
import copy
import time
import types
from collections import Counter

from vllm.v1.worker.gpu_worker import Worker


def native_drop_count(spec):
    """Validate the native-k action without silently accepting mass constraints."""
    unknown=set(spec)-{'mode','drop_j','j','backend','count_routes'}
    if unknown:
        raise ValueError(f'global_native does not use these fields: {sorted(unknown)}')
    if 'drop_j' in spec and 'j' in spec and spec['drop_j']!=spec['j']:
        raise ValueError('drop_j and j disagree')
    value=spec.get('drop_j',spec.get('j',1))
    if isinstance(value,bool) or not isinstance(value,int) or value<0:
        raise ValueError('native drop_j must be a nonnegative integer')
    if not isinstance(spec.get('count_routes',True),bool):
        raise ValueError('count_routes must be boolean')
    return value


def call_native_reduced_topk(router,original,args,kwargs,drop_j):
    """Synchronous eager-only override; restore even if native routing raises."""
    original_k=router.top_k
    if getattr(router,'_routing_replay_out',None) is not None:
        raise ValueError('native reduced-k does not support fixed-width routing replay buffers')
    router.top_k=max(1,original_k-drop_j)
    try:
        result=original(*args,**kwargs)
        if result[1].shape[-1]!=router.top_k:
            raise RuntimeError('router did not honor its temporary top_k')
        return result
    finally:
        router.top_k=original_k


def scheduler_phase(output, requests):
    scheduled = output.num_scheduled_tokens
    cached = output.scheduled_cached_reqs
    current = dict(zip(cached.req_ids, cached.num_computed_tokens))
    new = {item.req_id: item for item in output.scheduled_new_reqs}
    prefill = decode = 0
    for rid, count in scheduled.items():
        state = new[rid] if rid in new else requests[rid]
        computed = state.num_computed_tokens if rid in new else current[rid]
        npre = min(count, max(0, len(state.prompt_token_ids)-computed))
        prefill += npre
        decode += count-npre
    phase = 'mixed' if prefill and decode else ('prefill' if prefill else 'decode')
    return phase, {'prefill_tokens':prefill, 'decode_tokens':decode, 'requests':len(scheduled)}


class ExpertSkipWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        if importlib.metadata.version('vllm').split('+')[0] != '0.26.0':
            raise RuntimeError('This integration targets vLLM 0.26.0')
        vc = self.vllm_config
        if not vc.model_config.enforce_eager:
            raise ValueError('First integration requires enforce_eager; graph path is evaluated separately')
        if any(getattr(vc.parallel_config,key,1)!=1 for key in
               ('tensor_parallel_size','pipeline_parallel_size','data_parallel_size')):
            raise ValueError('Single-GPU TP=PP=DP=1 required')
        if any(getattr(vc.parallel_config,key,False) for key in
               ('enable_expert_parallel','enable_eplb','enable_dbo')) or vc.lora_config is not None:
            raise ValueError('EP/EPLB/DBO/LoRA are not supported')
        super().load_model(load_dummy_weights=load_dummy_weights)
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
        from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
        self._skip_policy = None
        self._skip_spec = None
        self._skip_native_j = None
        self._skip_alignment = None
        self._skip_phase = 'startup'
        self._skip_step = 0
        self._skip_scope = 'decode'
        self._skip_trace_limit = 0
        self._skip_trace = []
        self._skip_times = []
        self._skip_timing = False
        self._skip_phase_counts = Counter()
        self._skip_install = {}
        self._skip_bindings = []
        self._skip_routers = {}
        for name, runner in self.get_model().named_modules():
            if not isinstance(runner, MoERunner):
                continue
            routed=runner.routed_experts
            experts=getattr(getattr(routed.quant_method,'moe_kernel',None),'fused_experts',None)
            if type(experts) is not TritonExperts:
                raise TypeError(f'{name}: expected TritonExperts, got {type(experts).__name__}')
            E=runner.moe_config.num_experts
            if runner.moe_config.num_local_experts!=E or runner.router.eplb_state is not None:
                raise ValueError('expert-sharded routing is not supported')
            if not hasattr(runner.router,'renormalize'):
                raise ValueError('router must expose its renormalization contract')
            tensors={key:{'shape':list(t.shape),'bytes':t.numel()*t.element_size()}
                     for key,t in routed.named_parameters(recurse=False)
                     if t.ndim>=1 and t.shape[0]==E and (key.startswith('w13') or key.startswith('w2'))}
            if not {'w13_weight','w2_weight'}.issubset(tensors):
                raise ValueError('could not enumerate actual expert weights')
            per_expert=sum(row['bytes'] for row in tensors.values())//E
            renorm=bool(runner.router.renormalize)
            prepare_finalize=getattr(getattr(routed.quant_method,'moe_kernel',None),'prepare_finalize',None)
            native_dynamic_k=(type(prepare_finalize) is MoEPrepareAndFinalizeNoDPEPModular
                and type(runner.router).__name__ in ('FusedTopKRouter','GroupedTopKRouter','FusedTopKBiasRouter'))
            self._skip_install[name]={'num_experts':E,'top_k':runner.router.top_k,
                'renormalize':renorm,'expert_tensors':tensors,'weight_bytes_per_expert':per_expert,
                'backend':type(experts).__name__,'router':type(runner.router).__name__,
                'prepare_finalize':type(prepare_finalize).__name__,'native_dynamic_k_supported':native_dynamic_k,
                'native_postglobal_equivalence_expected':getattr(runner.router,'e_score_correction_bias',None) is None}
            original=runner.router.select_experts
            def wrapped(router,*args,_original=original,_name=name,_e=E,_renorm=renorm,_bytes=per_expert,**kwargs):
                eligible=self._skip_scope=='all' or self._skip_phase=='decode'
                original_k=router.top_k
                native_active=eligible and self._skip_native_j is not None
                if native_active:
                    weights,ids=call_native_reduced_topk(router,_original,args,kwargs,self._skip_native_j)
                else:
                    weights,ids=_original(*args,**kwargs)
                before_ids,before_weights=ids,weights
                if self._skip_policy is not None and eligible:
                    weights,ids=self._skip_policy(_name,weights,ids,_e,_renorm,_bytes)
                if (eligible or self._skip_native_j is not None) and len(self._skip_trace)<self._skip_trace_limit:
                    self._skip_trace.append({'layer':_name,'step':self._skip_step,
                        'phase':self._skip_phase,'batch_size':ids.shape[0],'num_experts':_e,
                        'policy_applied':eligible and self._skip_spec is not None,
                        'execution_mode':'global_native' if native_active else
                            (self._skip_spec.get('mode','dense') if eligible and self._skip_spec else 'native'),
                        'original_top_k':original_k,'executed_top_k':ids.shape[1],
                        'before_route_semantics':'native_reduced_route' if native_active else 'original_topk_route',
                        'top_k':ids.shape[1],'topk_ids':before_ids.detach().cpu().tolist(),
                        'topk_weights':before_weights.detach().float().cpu().tolist(),
                        'post_topk_ids':ids.detach().cpu().tolist(),
                        'post_topk_weights':weights.detach().float().cpu().tolist(),
                        'source':'vllm_actual_select_experts','forward_completed':False})
                return weights,ids
            runner.router.select_experts=types.MethodType(wrapped,runner.router)
            self._skip_bindings.append((runner.router,original))
            self._skip_routers[name]=runner.router
        if not self._skip_install:
            raise RuntimeError('No MoE layers found')

    def execute_model(self,scheduler_output):
        if not scheduler_output.total_num_scheduled_tokens:
            return super().execute_model(scheduler_output)
        self._skip_phase, counts=scheduler_phase(scheduler_output,self.model_runner.requests)
        self._skip_step += 1
        self._skip_phase_counts[self._skip_phase]+=1
        trace_start=len(self._skip_trace)
        timing=self._skip_timing and self._skip_phase=='decode'
        if timing:
            import torch
            start=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            start.record()
            cpu_start=time.perf_counter()
        try:
            result=super().execute_model(scheduler_output)
            if timing:
                end.record()
                self._skip_times.append((self._skip_step,counts,start,end,time.perf_counter()-cpu_start))
            for row in self._skip_trace[trace_start:]:
                row['forward_completed']=True
            return result
        finally:
            self._skip_phase='idle'

    def skip_set_policy(self,spec=None,scope='decode',trace_limit=0,timing=False):
        from expert_skip import ExpertSkipPolicy,SkipConfig,install_safe_alignment
        if scope not in ('decode','all'):
            raise ValueError('scope must be decode or all')
        actual_spec=copy.deepcopy(spec)
        native_j=None
        if spec is not None and spec.get('mode')=='global_native':
            native_j=native_drop_count(spec)
            unsupported=[name for name,meta in self._skip_install.items()
                         if not meta['native_dynamic_k_supported']]
            if unsupported:
                raise ValueError(f'native reduced-k requires inspected no-DP/EP modular path: {unsupported}')
            counter_config=SkipConfig(mode='dense',backend=spec.get('backend','torch'))
            policy=ExpertSkipPolicy(counter_config) if spec.get('count_routes',True) else None
        else:
            policy=None if spec is None else ExpertSkipPolicy(SkipConfig(**spec))
        if self._skip_alignment is not None:
            self._skip_alignment.restore()
            self._skip_alignment=None
        self._skip_policy=policy
        self._skip_spec=actual_spec
        self._skip_native_j=native_j
        if spec is not None and spec.get('mode','dense') not in ('dense','global_native'):
            self._skip_alignment=install_safe_alignment()
        self._skip_scope=scope
        self._skip_trace_limit=int(trace_limit)
        self._skip_trace=[]
        self._skip_step=0
        self._skip_times=[]
        self._skip_timing=bool(timing)
        self._skip_phase_counts=Counter()
        return {'layers':self.skip_execution_metadata(),'spec':actual_spec,'scope':scope,'trace_limit':trace_limit,
                'timing':timing,'startup_excluded':True}

    def skip_execution_metadata(self):
        rows={}
        for name,original in self._skip_install.items():
            meta=dict(original)
            meta['original_top_k']=original['top_k']
            router=getattr(self,'_skip_routers',{}).get(name)
            meta['current_router_top_k']=router.top_k if router is not None else None
            meta['eligible_top_k']=(max(1,original['top_k']-self._skip_native_j)
                                    if self._skip_native_j is not None else original['top_k'])
            meta['execution_mode']='native' if self._skip_spec is None else self._skip_spec.get('mode','dense')
            meta['scope']=self._skip_scope
            if self._skip_native_j is not None:
                meta['counter_before_semantics']='actual_native_reduced_route; not dense counterfactual'
                meta['removed_gate_mass_measured']=False
                meta['logical_route_counters_enabled']=self._skip_policy is not None
                meta['effective_drop_j']=original['top_k']-meta['eligible_top_k']
            rows[name]=meta
        return rows

    def skip_snapshot(self,reset=False):
        import torch
        torch.cuda.synchronize()
        rows=[{'step':step,**counts,'model_gpu_ms':start.elapsed_time(end),
               'model_host_enqueue_s':host} for step,counts,start,end,host in self._skip_times]
        counters={} if self._skip_policy is None else self._skip_policy.snapshot(reset=reset)
        if self._skip_native_j is not None:
            if not counters:
                counters={'layers':{},'totals':{}}
            counters['execution_spec']=copy.deepcopy(self._skip_spec)
            counters['logical_route_counters_enabled']=self._skip_policy is not None
            counters['counter_before_semantics']='actual_native_reduced_route; not dense counterfactual'
            counters['removed_gate_mass_measured']=False
            counters['skipped_slots_measured']=False
            # Dense accounting observes only the executed reduced routes. Its
            # zero post-drop mass is not a measurement against original top-k.
            for values in ([counters['totals'],*counters['layers'].values()]
                           if self._skip_policy is not None else []):
                for key in ('skipped_slots','removed_gate_mass_sum','removed_gate_mass_max','tokens_with_skips'):
                    values[key]=None
        result={'layers':counters,
                'phase_steps':dict(self._skip_phase_counts),'decode_model_times':rows,
                'trace':self._skip_trace,'spec':copy.deepcopy(self._skip_spec),
                'execution':self.skip_execution_metadata()}
        if reset:
            self._skip_trace=[];self._skip_times=[];self._skip_phase_counts=Counter()
        return result

    def skip_capture_control(self,command):
        from logits_capture import LogitsCapture
        cap=getattr(self,'_skip_logits',None)
        if command=='restore':
            if cap is not None:cap.restore();self._skip_logits=None
            return {'restored':True}
        if cap is None:
            cap=LogitsCapture(self.get_model()).install();self._skip_logits=cap
        if command=='finish':return cap.finish()
        if command not in ('baseline','candidate'):raise ValueError(command)
        cap.begin(baseline=command=='baseline')
        return {'mode':command}
