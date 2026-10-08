"""One logical prefix before store / after observed load completion. Diagnostic."""
import hashlib
import json
import time
from pathlib import Path
from recovery_kv_fingerprint import request_kv_digest


def capture(engine, request_id, tokens):
    import torch
    scheduler=engine.engine_core.engine_core.scheduler
    manager=scheduler.kv_cache_manager
    if manager.enable_caching or manager.num_kv_cache_groups!=1:
        raise ValueError('Requires one unshared KV group')
    req=scheduler.requests[request_id]
    owned=manager.coordinator.single_type_managers[0].req_to_blocks[request_id]
    blocks=[b.block_id for b in owned]
    if tokens<=0 or tokens%16 or len(blocks)<tokens//16:
        raise ValueError('Missing complete logical prefix blocks')
    started=time.perf_counter();torch.cuda.synchronize()
    context=engine.vllm_config.compilation_config.static_forward_context
    layers={}
    for i in range(16):
        name=f'model.layers.{i}.self_attn.attn';cache=context[name].kv_cache
        if cache.ndim!=4 or cache.shape[0]!=len(manager.block_pool.blocks) or cache.dtype!=torch.bfloat16:
            raise ValueError('Unqualified cross-layer cache view: '+name)
        layers[name]=dict(shape=list(cache.shape),stride=list(cache.stride()),
                         digest=request_kv_digest(cache,blocks[:tokens//16],tokens))
    torch.cuda.synchronize()
    return dict(request=request_id,tokens=tokens,blocks=blocks[:tokens//16],layers=layers,
                computed=req.num_computed_tokens,output=req.num_output_tokens,
                token_prefix_sha256=hashlib.sha256(json.dumps(list(req.all_token_ids[:tokens])).encode()).hexdigest(),
                diagnostic_seconds=time.perf_counter()-started)


def install(engine, action, transfers, output, capture_fn=capture):
    original=engine.step;was_local='step' in vars(engine);index=0
    data=dict(status='UNRUN',before=None,after=None)
    def step(*args,**kwargs):
        nonlocal index
        n=index
        result=original(*args,**kwargs)
        index+=1
        if n==328:
            event=next(e for e in action['events'] if e['event']=='select')
            tokens=event['save_cap']//16*16
            data['before']=capture_fn(engine,action['selected'],tokens)
            data['before']['call']=n
            data['status']='WAITING_FOR_LOAD'
        completed=[j for e in transfers['completed_jobs'] for j in e['jobs']
                   if j['request']==action['selected'] and not j['is_store']]
        if data['before'] is not None and data['after'] is None and completed:
            data['after']=capture_fn(engine,action['selected'],data['before']['tokens'])
            data['after']['call']=n;data['load_job']=completed[0]['job_id']
            a,b=data['before'],data['after']
            data['token_identity_equal']=a['token_prefix_sha256']==b['token_prefix_sha256']
            data['layer_equal']={name:v['digest']==b['layers'][name]['digest'] for name,v in a['layers'].items()}
            data['status']='MATCH' if data['token_identity_equal'] and all(data['layer_equal'].values()) else 'MISMATCH'
        return result
    engine.step=step
    def uninstall():
        if was_local:engine.step=original
        else:delattr(engine,'step')
        data['calls']=index
        data['scope']='Synchronized CPU copies/hash perturb execution. Logical full-chunk prefix only; no performance claim, untouched requests or uncomputed tail certification.'
        Path(output).write_text(json.dumps(data,indent=2)+'\n')
        return data
    return data,uninstall
