"""One-shot pre-action state diagnostic for pinned synchronous OLMoE/vLLM."""
import hashlib
import json
import time
from recovery_kv_fingerprint import request_kv_digest


def capture(scheduler, config, block_size, output):
    import torch
    manager=scheduler.kv_cache_manager
    if manager.enable_caching or manager.num_kv_cache_groups!=1 or block_size!=16:
        raise ValueError('requires pinned unshared KV mode')
    owned=manager.coordinator.single_type_managers[0].req_to_blocks
    context=config.compilation_config.static_forward_context
    names=[f'model.layers.{i}.self_attn.attn' for i in range(16)]
    pool=manager.block_pool
    canonical=lambda rid:rid.rsplit('-',1)[0]
    requests={}
    for rid,r in scheduler.requests.items():
        if r.sampling_params.temperature!=0:
            raise ValueError('stochastic sampling requires additional RNG state capture')
        requests[canonical(rid)]=dict(computed=r.num_computed_tokens,prompt=r.num_prompt_tokens,
            output=r.num_output_tokens,max_tokens=r.max_tokens,preemptions=r.num_preemptions,
            status=r.status.name,blocks=[b.block_id for b in owned.get(rid,())],
            token_sha256=hashlib.sha256(json.dumps(list(r.all_token_ids)).encode()).hexdigest())
    state=dict(requests=requests,running=[canonical(r.request_id) for r in scheduler.running],
               waiting=[canonical(r.request_id) for r in scheduler.waiting],free=pool.get_num_free_blocks())
    started=time.perf_counter();torch.cuda.synchronize()
    layers={}
    for name in names:
        layer=context[name];cache=layer.kv_cache
        if cache.ndim!=4 or cache.shape[0]!=len(pool.blocks) or cache.shape[2]!=16 or cache.dtype!=torch.bfloat16:
            raise ValueError(f'unsupported actual cache layout {name}: {cache.shape}')
        layers[name]=dict(shape=list(cache.shape),stride=list(cache.stride()),requests={})
        for rid,r in scheduler.requests.items():
            layers[name]['requests'][canonical(rid)]=request_kv_digest(cache,
                [b.block_id for b in owned.get(rid,())],r.num_computed_tokens,block_size=block_size)
    torch.cuda.synchronize()
    result=dict(status='CAPTURED',step=329,state=state,layers=layers,
                diagnostic_seconds=time.perf_counter()-started,
                scope='Actual logical valid KV bytes and request state; not full engine checkpoint, '
                      'not hidden-state/RNG/allocator equivalence, not performance measurement.')
    with output.open('x') as f:json.dump(result,f,indent=2)
