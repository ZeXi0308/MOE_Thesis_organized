"""Worker RPCs for eager single-GPU corpus quality runs."""


class QualityWorkerExtension:
    """Use vLLM's named extension API; no callable/pickle RPC serialization."""
    def tiered_set_policy(self,spec,expected_layers=16,layer_policies=None,profile=False):
        return set_policy(self,spec,expected_layers,layer_policies,profile)

    def tiered_finish_policy(self,expected_layers=16):
        return finish_policy(self,expected_layers)

    def tiered_capture_control(self,command):
        return capture_control(self,command)


def set_policy(worker,spec,expected_layers,layer_policies=None,profile=False):
    import vllm_quality_hook as module
    from quality_transform import Policy
    from rank_profile import RankProfile
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    old=getattr(module,'_ACTIVE',None)
    if old is not None:
        old.__exit__(None,None,None);module._ACTIVE=None
    if spec is None:return {'installed':False}
    observer=RankProfile() if profile else None
    hook=module.QualityHook(Policy(**spec),
        {name:Policy(**p) for name,p in (layer_policies or {}).items()},observer)
    backends={}
    for name,layer in worker.get_model().named_modules():
        kernel=getattr(getattr(layer,'quant_method',None),'moe_kernel',None)
        expert=getattr(kernel,'fused_experts',None)
        if expert is None:continue
        backends[name]=type(expert).__name__
        if type(expert) is not TritonExperts:
            raise RuntimeError(f'Unsupported backend {name}: {type(expert)}')
        hook.bind(expert,name)
    if len(backends)!=expected_layers:
        raise RuntimeError(f'Expected {expected_layers} MoE layers: {backends}')
    unknown=set(layer_policies or {})-set(backends)
    if unknown:raise ValueError(f'Unknown layer overrides: {unknown}')
    hook.__enter__();module._ACTIVE=hook
    return {'installed':True,'backends':backends}


def finish_policy(worker,expected_layers):
    import vllm_quality_hook as module
    hook=getattr(module,'_ACTIVE',None)
    if hook is None:return {'coverage':{},'profile':{}}
    return {'coverage':hook.require_coverage(expected_layers),
            'profile':hook.observer.finish() if hook.observer is not None else {}}


def capture_control(worker,command):
    import logits_capture as module
    cap=getattr(module,'_ACTIVE',None)
    if command=='restore':
        if cap is not None:cap.restore();module._ACTIVE=None
        return {'restored':True}
    if cap is None:
        cap=module.LogitsCapture(worker.get_model()).install();module._ACTIVE=cap
    if command=='finish':return cap.finish()
    if command not in ['baseline','candidate']:raise ValueError(command)
    cap.begin(baseline=command=='baseline')
    return {'mode':command}
