#!/usr/bin/env python3
"""One natural-EOS cell, reusing B's request measurement and native engine setup."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def sampling_for_request(config, source):
    params=dict(config.get('sampling', {'temperature':0.0}))
    if 'seed' in params:
        # W3 changes ordering only: preserve the same document's RNG seed.
        identity=json.dumps(source['document_id'],sort_keys=True,separators=(',',':'))
        offset=int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)
        params['seed']=(params['seed']+offset) % (2**31-1)
    return params


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',choices=['native','horizon','full-budget','mc','mc-load'],required=True)
    p.add_argument('--warmup-profile',choices=['legacy','broad'],default='legacy')
    p.add_argument('--lease-victim',type=int,choices=(0,1),default=1)
    p.add_argument('--stall-order',type=int,choices=(0,1),default=1)
    p.add_argument('--estimate-hr',type=int,choices=(0,1),default=0)
    p.add_argument('--ha',type=int,default=64);p.add_argument('--hr',type=int,default=32)
    p.add_argument('--kv-gib',type=int,default=48);p.add_argument('--cap',type=int,default=256)
    p.add_argument('--arrival-rate',type=float,default=10.)
    p.add_argument('--max-seconds',type=int,default=240)
    p.add_argument('--b-root',type=Path,default=Path('/root/autodl-tmp/moe-b-recovery-order-20261004'))
    args=p.parse_args()
    if args.arrival_rate<=0:raise ValueError('Positive rate required')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    dump(out/'status.json',dict(status='INITIALIZING',start_unix_s=time.time()))
    dump(out/'command.json',sys.argv)
    sys.path.insert(0,str(args.b_root/'pkg'))
    import request_measurement as measurement
    from native_offload_observer import drain,install as observe
    from native_capture import set_empty_admission_cap
    from horizon_sched import install,Config
    source=inspect.getsource(measurement.measure_episode)
    anchor='                row["token_times_s"].extend([received] * len(added))'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'\n                if added and _hz_adapter is not None:\n                    _hz_adapter.receipt(row["internal_request_id"], origin + received)')
    sampling_anchor='SamplingParams(n=1, temperature=0.0, max_tokens=limits[rid],'
    assert source.count(sampling_anchor)==1
    source=source.replace(sampling_anchor,
        'SamplingParams(n=1, **_hz_sampling(config, sources[index]), max_tokens=limits[rid],')
    namespace=dict(measurement.__dict__,_hz_adapter=None,_hz_sampling=sampling_for_request)
    exec(compile(source,'<B-measurement-with-receipts>','exec'),namespace)
    capture=namespace['measure_episode']
    payload=json.loads(args.inputs.read_text());config=payload['config'];work=payload['workload']
    work['arrival_traces_s']={'steady':[i/args.arrival_rate for i in range(len(work['source_requests']))]}
    config.update(ignore_eos=False,min_tokens=0,cap=args.cap,output_tokens=1024,output_tokens_by_request={})
    dump(out/'inputs.json',payload)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_BATCH_INVARIANT='0',VLLM_USE_SIMPLE_KV_OFFLOAD='0')
    engine=adapter=None;undo_observer=lambda:None
    try:
        import torch,vllm
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        kwargs=dict(model=config['model_path'],dtype='bfloat16',seed=20261016,
            max_model_len=4096,max_num_seqs=256,max_num_batched_tokens=1024,
            gpu_memory_utilization=.90,enable_chunked_prefill=True,enable_prefix_caching=False,
            scheduling_policy='fcfs',async_scheduling=False,kv_cache_memory_bytes=args.kv_gib*2**30,
            scheduler_reserve_full_isl=True,stream_interval=1,enforce_eager=False,
            enable_return_routed_experts=False,kv_offloading_size=16,kv_offloading_backend='native',
            kv_transfer_config={'kv_connector_extra_config':{'offload_prompt_only':False}})
        dump(out/'engine_args.json',kwargs)
        dump(out/'environment.json',dict(torch=torch.__version__,vllm=vllm.__version__,
            code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
            original_measurement_sha256=hashlib.sha256(Path(measurement.__file__).read_bytes()).hexdigest(),
            measurement_compiled_sha256=hashlib.sha256(source.encode()).hexdigest()))
        start=time.perf_counter();engine=LLMEngine.from_engine_args(EngineArgs(**kwargs),enable_multiprocessing=False)
        scheduler=engine.engine_core.engine_core.scheduler
        dump(out/'init.json',dict(seconds=time.perf_counter()-start,scheduler_type=type(scheduler).__name__,
            num_gpu_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks))
        print('APPLICATION_WARMUP',flush=True)
        if args.warmup_profile=='broad':
            # A short-prefix cohort survives long enough to reach cap256 decode.
            # This is startup work, not a quality or serving measurement.
            count=256
            if len(work['source_requests'])<count:raise ValueError('Broad warmup needs 256 inputs')
            w=dict(work,source_requests=work['source_requests'][:count],
                   actual_prompt_token_ids=[ids[-64:] for ids in work['actual_prompt_token_ids'][:count]],
                   arrival_traces_s={'steady':[0.]*count})
            c=dict(config,output_tokens=64,output_tokens_by_request={},cap=count,
                   ignore_eos=True,min_tokens=64)
            dump(out/'warmup-broad-inputs.json',dict(workload=w,config=c,
                 semantics='Last64 prompt tokens; fixed64 outputs only for kernel shape warmup. Measurement retains original full prompts and natural EOS.'))
            set_empty_admission_cap(engine,count)
            raw=capture(engine,w,c,'steady',1.,'warmup-broad',max_seconds=120)
            dump(out/'warmup-broad.json',raw)
            if raw['status']!='COMPLETE':raise RuntimeError('Broad warmup failed: '+str(raw['error']))
        for idx,(count,cap) in enumerate(((32,32),(2,2))):
            w=dict(work,source_requests=work['source_requests'][:count],
                   actual_prompt_token_ids=work['actual_prompt_token_ids'][:count],
                   arrival_traces_s={'steady':[0.]*count})
            c=dict(config,output_tokens=16,cap=cap)
            set_empty_admission_cap(engine,cap)
            raw=capture(engine,w,c,'steady',1.,'warmup'+str(idx),max_seconds=120)
            dump(out/f'warmup-{idx}.json',raw)
            if raw['status']!='COMPLETE':raise RuntimeError('Warmup failed: '+str(raw['error']))
        dump(out/'warmup-drain.json',drain(engine))
        if not engine.reset_prefix_cache(reset_connector=True):raise RuntimeError('Warmup reset failed')
        set_empty_admission_cap(engine,args.cap)
        cfg=Config(H_a=args.ha,H_r=args.hr,lease_victim=bool(args.lease_victim),
                   stall_order=bool(args.stall_order),estimate_Hr=bool(args.estimate_hr)) if args.policy=='horizon' else Config(0,0,False,False,False)
        if args.policy in ('full-budget','mc','mc-load'):
            from baselines import install_baseline
            adapter=install_baseline(scheduler,args.policy,args.cap)
        else:
            adapter=install(scheduler,cfg)
        namespace['_hz_adapter']=adapter
        obs,undo_observer=observe(scheduler.connector,diagnostic=True)
        dump(out/'status.json',dict(status='MEASURING',start_unix_s=time.time(),policy=args.policy,config=vars(cfg)))
        print('MEASUREMENT',flush=True)
        raw=capture(engine,work,config,'steady',1.,'measured',max_seconds=args.max_seconds,record_preemptions=True)
        dump(out/'raw.json',raw)
        dump(out/'offload-drain.json',drain(engine))
        undo_observer();undo_observer=lambda:None
        dump(out/'offload.json',obs)
        dump(out/'horizon.json',adapter.uninstall())
        dump(out/'status.json',dict(status=raw['status'],error=raw['error'],end_unix_s=time.time()))
        print('CELL_RESULT',raw['status'],flush=True)
        return 0 if raw['status']=='COMPLETE' else 2
    except BaseException as e:
        dump(out/'status.json',dict(status='ERROR',error=repr(e),end_unix_s=time.time()))
        raise
    finally:
        undo_observer()
        if adapter is not None:dump(out/'horizon.json',adapter.uninstall())
        if engine is not None:engine.engine_core.shutdown()


if __name__=='__main__':raise SystemExit(main())
