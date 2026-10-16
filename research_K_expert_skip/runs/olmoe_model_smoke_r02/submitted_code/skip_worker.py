"""vLLM 0.26 eager worker, per-router policy and explicit scheduler phases.

Install with worker_cls='skip_worker.ExpertSkipWorker', moe_backend='triton',
enforce_eager=True, compilation_config={'mode': 0, 'cudagraph_mode': 'NONE'}.
Policy is disabled during model initialization. RPCs switch policies outside
measurement intervals. Trace/logits capture is diagnostic, never timed as TPOT.
"""
import importlib.metadata
import time
import types
from collections import Counter

from vllm.v1.worker.gpu_worker import Worker


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
        self._skip_policy = None
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
            self._skip_install[name]={'num_experts':E,'top_k':runner.router.top_k,
                'renormalize':renorm,'expert_tensors':tensors,'weight_bytes_per_expert':per_expert,
                'backend':type(experts).__name__,'router':type(runner.router).__name__}
            original=runner.router.select_experts
            def wrapped(router,*args,_original=original,_name=name,_e=E,_renorm=renorm,_bytes=per_expert,**kwargs):
                weights,ids=_original(*args,**kwargs)
                eligible=self._skip_scope=='all' or self._skip_phase=='decode'
                before_ids,before_weights=ids,weights
                if self._skip_policy is not None and eligible:
                    weights,ids=self._skip_policy(_name,weights,ids,_e,_renorm,_bytes)
                if eligible and len(self._skip_trace)<self._skip_trace_limit:
                    self._skip_trace.append({'layer':_name,'step':self._skip_step,
                        'phase':self._skip_phase,'batch_size':ids.shape[0],'num_experts':_e,
                        'top_k':ids.shape[1],'topk_ids':before_ids.detach().cpu().tolist(),
                        'topk_weights':before_weights.detach().float().cpu().tolist(),
                        'post_topk_ids':ids.detach().cpu().tolist(),
                        'post_topk_weights':weights.detach().float().cpu().tolist(),
                        'source':'vllm_actual_select_experts','forward_completed':False})
                return weights,ids
            runner.router.select_experts=types.MethodType(wrapped,runner.router)
            self._skip_bindings.append((runner.router,original))
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
        if self._skip_alignment is not None:
            self._skip_alignment.restore()
            self._skip_alignment=None
        self._skip_policy=None if spec is None else ExpertSkipPolicy(SkipConfig(**spec))
        if spec is not None and spec.get('mode','dense')!='dense':
            self._skip_alignment=install_safe_alignment()
        self._skip_scope=scope
        self._skip_trace_limit=int(trace_limit)
        self._skip_trace=[]
        self._skip_step=0
        self._skip_times=[]
        self._skip_timing=bool(timing)
        self._skip_phase_counts=Counter()
        return {'layers':self._skip_install,'spec':spec,'scope':scope,'trace_limit':trace_limit,
                'timing':timing,'startup_excluded':True}

    def skip_snapshot(self,reset=False):
        import torch
        torch.cuda.synchronize()
        rows=[{'step':step,**counts,'model_gpu_ms':start.elapsed_time(end),
               'model_host_enqueue_s':host} for step,counts,start,end,host in self._skip_times]
        result={'layers':{} if self._skip_policy is None else self._skip_policy.snapshot(reset=reset),
                'phase_steps':dict(self._skip_phase_counts),'decode_model_times':rows,
                'trace':self._skip_trace,'spec':None if self._skip_policy is None else str(self._skip_policy.config)}
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
