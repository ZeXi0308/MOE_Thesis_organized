"""Experimental fixed-arm full-decode CUDA graph integration for vLLM 0.26.

Set K_SKIP_GRAPH_CONFIG to JSON before constructing LLM, for example:
{"spec":{"mode":"marginal","tail_rank":4,"gate_threshold":0.15,
         "mass_budget":0.25,"backend":"triton"},
 "count_routes":true,"log_runtime":true}
Use enforce_eager=False, compilation mode=0, cudagraph_mode=FULL_DECODE_ONLY.
One process owns one immutable arm. Prefill remains original routing/native
alignment. Runtime padding stays routed but is excluded from policy decisions
and logical counters; those counters consequently exclude padding traffic.
An expert unique among real rows may still be selected by a dummy row. Thus
real-row singleton counts do not prove physical marginal savings in padded
buckets; actual native kernel traffic can retain those expert weights.
"""
import copy
import importlib.metadata
import json
import os
import types
from collections import Counter

from vllm.v1.worker.gpu_worker import Worker
from skip_worker import ExpertSkipWorker,call_native_reduced_topk,native_drop_count


def _full_decode_context():
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context,is_forward_context_available
    return (is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode==CUDAGraphMode.FULL)


class ExpertSkipGraphWorker(ExpertSkipWorker):
    def load_model(self,*,load_dummy_weights=False):
        if importlib.metadata.version('vllm').split('+')[0]!='0.26.0':
            raise RuntimeError('This graph integration targets vLLM 0.26.0')
        raw=os.environ.get('K_SKIP_GRAPH_CONFIG')
        if raw is None:
            raise ValueError('K_SKIP_GRAPH_CONFIG must specify the fixed process arm')
        settings=json.loads(raw)
        unknown=set(settings)-{'spec','count_routes','log_runtime'}
        if unknown:
            raise ValueError(f'unknown graph settings: {sorted(unknown)}')
        self._graph_count_routes=settings.get('count_routes',True)
        self._graph_log_runtime=settings.get('log_runtime',False)
        if not isinstance(self._graph_count_routes,bool) or not isinstance(self._graph_log_runtime,bool):
            raise ValueError('count_routes and log_runtime must be booleans')
        fixed_spec=copy.deepcopy(settings.get('spec'))
        vc=self.vllm_config
        if vc.model_config.enforce_eager or int(vc.compilation_config.mode)!=0:
            raise ValueError('Graph worker requires enforce_eager=False and compilation mode=0')
        if str(vc.compilation_config.cudagraph_mode)!='FULL_DECODE_ONLY':
            raise ValueError('Graph worker requires FULL_DECODE_ONLY')
        if any(getattr(vc.parallel_config,key,1)!=1 for key in
               ('tensor_parallel_size','pipeline_parallel_size','data_parallel_size')):
            raise ValueError('Single GPU TP=PP=DP=1 required')
        if any(getattr(vc.parallel_config,key,False) for key in
               ('enable_expert_parallel','enable_eplb','enable_dbo')) or vc.lora_config is not None:
            raise ValueError('EP/EPLB/DBO/LoRA are not supported')
        if vc.speculative_config is not None:
            raise ValueError('Initial graph integration excludes speculative decoding')
        # Deliberately bypass the eager-only parent load_model. Graph capture
        # happens later in compile_or_warm_up_model, after hooks are installed.
        Worker.load_model(self,load_dummy_weights=load_dummy_weights)
        import torch
        from expert_skip import ExpertSkipPolicy,SkipConfig,install_safe_alignment
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
        from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
        self._skip_spec=fixed_spec
        self._skip_native_j=(native_drop_count(fixed_spec)
                             if fixed_spec and fixed_spec.get('mode')=='global_native' else None)
        if fixed_spec and 'count_routes' in fixed_spec and fixed_spec['count_routes']!=self._graph_count_routes:
            raise ValueError('native spec count_routes and graph count_routes disagree')
        policy_spec=({'mode':'dense','backend':(fixed_spec or {}).get('backend','triton')}
                     if fixed_spec is None or self._skip_native_j is not None else
                     {**fixed_spec,'backend':fixed_spec.get('backend','triton')})
        policy_config=SkipConfig(**policy_spec)
        if policy_config.backend not in ('triton','triton_multi'):
            raise ValueError('Graph integration requires the Triton policy backend')
        needs_action=fixed_spec is not None and fixed_spec.get('mode','dense') not in ('dense','global_native')
        self._skip_policy=(ExpertSkipPolicy(policy_config,collect_stats=self._graph_count_routes)
                           if needs_action or self._graph_count_routes else None)
        self._skip_alignment=install_safe_alignment(enable_predicate=lambda:
            _full_decode_context() or self._skip_phase=='decode') if needs_action else None
        self._skip_phase='startup'
        self._skip_step=0
        self._skip_scope='decode'
        self._skip_trace_limit=0
        self._skip_trace=[]
        self._skip_times=[]
        self._skip_timing=False
        self._skip_phase_counts=Counter()
        self._skip_install={}
        self._skip_bindings=[]
        self._skip_routers={}
        self._graph_runtime_rows=[]
        # A stable pointer read at replay time. Dummy capture starts at zero;
        # no actions or stats are recorded for dummy rows.
        self._graph_actual_tokens=torch.zeros((),device=self.device,dtype=torch.int32)
        for name,runner in self.get_model().named_modules():
            if not isinstance(runner,MoERunner):
                continue
            routed=runner.routed_experts
            kernel=getattr(routed.quant_method,'moe_kernel',None)
            experts=getattr(kernel,'fused_experts',None)
            prepare=getattr(kernel,'prepare_finalize',None)
            if type(experts) is not TritonExperts or type(prepare) is not MoEPrepareAndFinalizeNoDPEPModular:
                raise TypeError(f'{name}: graph integration requires TritonExperts + NoDPEPModular')
            E=runner.moe_config.num_experts
            if runner.moe_config.num_local_experts!=E or runner.router.eplb_state is not None:
                raise ValueError('expert-sharded routing is not supported')
            if not hasattr(runner.router,'renormalize'):
                raise ValueError('router must expose its renormalization contract')
            native_supported=type(runner.router).__name__ in ('FusedTopKRouter','GroupedTopKRouter','FusedTopKBiasRouter')
            if self._skip_native_j is not None and not native_supported:
                raise ValueError('native reduced-k requires an inspected dynamic-k router')
            tensors={key:{'shape':list(value.shape),'bytes':value.numel()*value.element_size()}
                     for key,value in routed.named_parameters(recurse=False)
                     if value.ndim>=1 and value.shape[0]==E and (key.startswith('w13') or key.startswith('w2'))}
            if not {'w13_weight','w2_weight'}.issubset(tensors):
                raise ValueError('could not enumerate actual expert weights')
            per_expert=sum(row['bytes'] for row in tensors.values())//E
            renorm=bool(runner.router.renormalize)
            self._skip_install[name]={'num_experts':E,'top_k':runner.router.top_k,
                'renormalize':renorm,'expert_tensors':tensors,'weight_bytes_per_expert':per_expert,
                'backend':type(experts).__name__,'router':type(runner.router).__name__,
                'prepare_finalize':type(prepare).__name__,'native_dynamic_k_supported':native_supported}
            original=runner.router.select_experts
            def wrapped(router,*args,_original=original,_name=name,_e=E,_renorm=renorm,_bytes=per_expert,**kwargs):
                eligible=_full_decode_context() or self._skip_phase=='decode'
                if eligible and self._skip_native_j is not None:
                    weights,ids=call_native_reduced_topk(router,_original,args,kwargs,self._skip_native_j)
                else:
                    weights,ids=_original(*args,**kwargs)
                if eligible and self._skip_policy is not None:
                    weights,ids=self._skip_policy(_name,weights,ids,_e,_renorm,_bytes,
                                                  active_tokens=self._graph_actual_tokens)
                return weights,ids
            runner.router.select_experts=types.MethodType(wrapped,runner.router)
            self._skip_bindings.append((runner.router,original))
            self._skip_routers[name]=runner.router
        if not self._skip_install:
            raise RuntimeError('No MoE layers found')
        # Persistent tensors must be initialized before capture. In particular,
        # a zeros() first recorded inside a graph would reset that layer's
        # cumulative counter on every replay of the first captured bucket.
        warmed=set()
        self._graph_policy_warmup_shapes=[]
        buckets=vc.compilation_config.cudagraph_capture_sizes or []
        for name,meta in self._skip_install.items():
            E=meta['num_experts']
            k=max(1,meta['top_k']-self._skip_native_j) if self._skip_native_j is not None else meta['top_k']
            if self._skip_alignment is not None:
                self._skip_alignment.prepare_maps(E,self.device)
            if self._skip_policy is None:
                continue
            if self._graph_count_routes:
                self._skip_policy._counter(name,self.device)
            for batch in buckets:
                shape=(batch,k,E,meta['renormalize'],meta['weight_bytes_per_expert'])
                if shape in warmed:
                    continue
                weights=torch.full((batch,k),1./k,device=self.device,dtype=torch.float32)
                ids=torch.arange(k,device=self.device,dtype=torch.int32)[None,:].expand(batch,-1).contiguous()
                self._skip_policy(name,weights,ids,E,meta['renormalize'],meta['weight_bytes_per_expert'],
                                  active_tokens=self._graph_actual_tokens)
                warmed.add(shape)
                self._graph_policy_warmup_shapes.append(list(shape))
        torch.cuda.synchronize()
        determine=self.model_runner._determine_batch_execution_and_padding
        def observed(runner,*args,**kwargs):
            result=determine(*args,**kwargs)
            if self._graph_log_runtime:
                self._graph_runtime_rows.append({'step':self._skip_step,'phase':self._skip_phase,
                    'actual_tokens':kwargs.get('num_tokens',args[0] if args else None),'padded_tokens':result[1].num_tokens,
                    'runtime_mode':str(result[0])})
            return result
        self.model_runner._determine_batch_execution_and_padding=types.MethodType(observed,self.model_runner)

    def execute_model(self,scheduler_output):
        # Same stream as replay. This is one device fill per scheduled step,
        # with no device->host copy or synchronization. Wall serving time
        # includes this fill; inherited model_gpu_ms starts after the fill.
        if self._skip_policy is not None:
            self._graph_actual_tokens.fill_(scheduler_output.total_num_scheduled_tokens)
        return ExpertSkipWorker.execute_model(self,scheduler_output)

    def skip_set_policy(self,spec=None,scope='decode',trace_limit=0,timing=False):
        if spec!=self._skip_spec or scope!='decode':
            raise ValueError('graph arm is immutable; create a new process to change spec/scope')
        if trace_limit:
            raise ValueError('Python route traces do not run during graph replay; use counters/runtime metadata')
        self.skip_snapshot(reset=True)
        self._skip_timing=bool(timing)
        return {'spec':copy.deepcopy(self._skip_spec),'scope':'decode','fixed_arm':True,
                'layers':self.skip_execution_metadata(),'count_routes':self._graph_count_routes}

    def skip_snapshot(self,reset=False):
        result=ExpertSkipWorker.skip_snapshot(self,reset=reset)
        wrapper=self.model_runner.model
        entries=getattr(wrapper,'concrete_cudagraph_entries',{})
        result['graph']={'fixed_arm':True,'compilation_mode':0,'cudagraph_mode':'FULL_DECODE_ONLY',
            'count_routes':self._graph_count_routes,'runtime_rows':list(self._graph_runtime_rows),
            'policy_warmup_shapes':self._graph_policy_warmup_shapes,
            'model_gpu_ms_excludes_actual_token_fill':True,
            'captured_token_buckets':sorted({key.num_tokens for key in entries}),
            'padding_semantics':'padded routes pass unchanged; excluded from policy and logical footprint counters'}
        if reset:
            self._graph_runtime_rows=[]
        return result
