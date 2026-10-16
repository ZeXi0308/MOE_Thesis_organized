"""Process-local vLLM 0.26 Triton modular quality hook, eager single-GPU only.

Patch the class in this process, never site-packages files. Install in the
worker before model execution; context manager restores the original methods.
Layer IDs are stable within one worker via explicit bind() or first-use order.
Coverage counters must be checked: a backend bypassing Triton is not evaluated.
"""
import contextvars,functools,inspect
from quality_transform import transform


class QualityHook:
    def __init__(self,policy,layer_policies=None,observer=None):
        self.policy=policy;self.layer_policies=layer_policies or {}
        self.observer=observer
        self.context=contextvars.ContextVar('tiered_quality_context',default=None)
        self.labels={};self.calls={};self.active=False

    def bind(self,expert,layer_name):
        self.labels[id(expert)]=layer_name

    def __enter__(self):
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
        if getattr(TritonExperts.apply,'_tiered_hook',False):raise RuntimeError('Hook already installed')
        self.cls=TritonExperts;self.apply=TritonExperts.apply;self.reduce=TritonExperts.moe_sum
        signature=inspect.signature(self.apply);hook=self
        @functools.wraps(self.apply)
        def apply(expert,*args,**kwargs):
            b=signature.bind(expert,*args,**kwargs);b.apply_defaults();v=b.arguments
            if v['expert_map'] is not None:raise RuntimeError('Quality emulation requires single GPU, no expert map')
            if v['apply_router_weight_on_input']:raise RuntimeError('Input-weighted experts need a separate semantic check')
            name=hook.labels.setdefault(id(expert),f'layer_{len(hook.labels)}')
            token=hook.context.set((name,v['topk_weights'],v['topk_ids'],v['global_num_experts']))
            try:return hook.apply(expert,*args,**kwargs)
            finally:hook.context.reset(token)
        apply._tiered_hook=True
        def reduce(expert,inputs,output):
            context=hook.context.get()
            if context is None:raise RuntimeError('moe_sum outside intercepted apply; unsupported path')
            name,weights,ids,experts=context
            if hook.observer is not None:hook.observer(name,inputs,weights,ids,experts)
            selected=hook.layer_policies.get(name,hook.policy)
            x=transform(inputs,weights,ids,experts,selected)
            hook.calls[name]=hook.calls.get(name,0)+1
            if selected.mode=='I':return hook.reduce(expert,x,output)
            # Reference owner partial reduction; grouped BF16 is its baseline.
            output.copy_(x.float().sum(1).to(output.dtype))
        self.cls.apply=apply;self.cls.moe_sum=reduce;self.active=True
        return self

    def __exit__(self,*exc):
        self.cls.apply=self.apply;self.cls.moe_sum=self.reduce;self.active=False

    def require_coverage(self,layers=None):
        if not self.calls or (layers is not None and len(self.calls)!=layers):
            raise RuntimeError(f'Unexpected quality hook coverage: {self.calls}')
        return dict(self.calls)
