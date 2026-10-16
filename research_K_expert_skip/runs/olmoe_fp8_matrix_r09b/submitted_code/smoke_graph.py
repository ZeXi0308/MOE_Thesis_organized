"""Separate-process eager/graph parity, with explicit real/captured batches.

Run through run_locked.py. The supervisor starts two serialized children while
the launcher retains the GPU lock. This script does not acquire another lock.
Passing assertions verifies this selected model/arm only; it is not a speedup
measurement. Child partial results remain on failure.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_SPEC={'mode':'marginal','tail_rank':4,'gate_threshold':.15,
              'mass_budget':.25,'backend':'triton'}


def capture_sizes(batches, explicit=None):
    if not batches or len(set(batches))!=len(batches) or any(b<1 or b>64 for b in batches):
        raise ValueError('batches must be unique integers in 1..64')
    buckets=explicit if explicit is not None else [2**i for i in range((max(batches)-1).bit_length()+1)]
    if (not buckets or buckets!=sorted(set(buckets)) or buckets[0]<1 or buckets[-1]>64
            or max(batches)>buckets[-1]):
        raise ValueError('capture sizes must increase uniquely in 1..64 and cover all batches')
    return buckets


def child_command(args,kind,path,spec):
    native=kind.startswith('native_')
    execution=kind.removeprefix('native_')
    argv=[sys.executable,str(Path(__file__).resolve()),'--model',args.model,
          '--dtype',args.dtype,'--gpu-memory-utilization',str(args.gpu_memory_utilization),
          '--output',str(path),'--spec-json',json.dumps(None if native else spec),
          '--child',execution,'--batches',*map(str,args.batches),
          '--capture-sizes',*map(str,args.capture_sizes),
          '--no-count-routes' if native or not args.count_routes else '--count-routes']
    if args.quantization is not None:
        argv.extend(['--quantization',args.quantization])
    return argv


def child(args,spec):
    graph=args.child=='graph'
    # Enqueue the whole cohort before stepping; an MP engine may start early
    # while LLM.generate is still submitting later requests.
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['K_SKIP_GRAPH_CONFIG']=json.dumps({'spec':spec,'count_routes':args.count_routes,'log_runtime':True})
    from vllm import LLM,SamplingParams
    cc={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY' if graph else 'NONE'}
    if graph:
        cc.update(cudagraph_capture_sizes=args.capture_sizes,max_cudagraph_capture_size=max(args.capture_sizes))
    llm=LLM(model=args.model,dtype=args.dtype,quantization=args.quantization,tensor_parallel_size=1,
        enforce_eager=not graph,compilation_config=cc,
        worker_cls='skip_graph_worker.ExpertSkipGraphWorker' if graph else 'skip_worker.ExpertSkipWorker',
        moe_backend='triton',enable_prefix_caching=False,max_model_len=512,max_num_seqs=max(args.capture_sizes),
        max_num_batched_tokens=max(512,128*max(args.batches)),gpu_memory_utilization=args.gpu_memory_utilization,seed=1701,
        async_scheduling=False,enable_chunked_prefill=False)
    output=Path(args.output)
    rows=[]
    # Native eager is truly policy-free. Native+dense comparison uses a separate
    # dense counting process, so the no-counter control never installs a policy.
    install_spec=spec
    if not graph and not args.count_routes and spec is not None and spec.get('mode')=='dense':
        install_spec=None
    for batch in args.batches:
        padded=next(size for size in args.capture_sizes if size>=batch)
        installed=llm.collective_rpc('skip_set_policy',args=(install_spec,'decode',0,False))
        prompts=[f'Explain why repeated measurements matter in science. Give example number {i+1}.'
                 for i in range(batch)]
        results=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=6,ignore_eos=True),use_tqdm=False)
        snapshots=llm.collective_rpc('skip_snapshot')
        row={'batch':batch,'target_bucket':padded,'installation':installed,'worker':snapshots,
             'outputs':[{'tokens':list(r.outputs[0].token_ids),'text':r.outputs[0].text} for r in results]}
        rows.append(row)
        output.write_text(json.dumps({'status':'RUNNING','execution':args.child,'spec':spec,
                                     'args':vars(args),'rows':rows},indent=2))
        for snapshot in snapshots:
            assert snapshot['phase_steps'].get('prefill',0)==1,snapshot['phase_steps']
            assert snapshot['phase_steps'].get('mixed',0)==0,snapshot['phase_steps']
            assert snapshot['phase_steps'].get('decode',0)==5,snapshot['phase_steps']
            assert all(len(r.outputs[0].token_ids)==6 for r in results)
            layers=len(snapshot['execution'])
            if graph:
                runtime=[r for r in snapshot['graph']['runtime_rows'] if r['phase']=='decode']
                assert runtime,'no real decode dispatcher observations'
                assert all(r['runtime_mode']=='FULL' for r in runtime),runtime
                assert all(r['actual_tokens']==batch and r['padded_tokens']==padded for r in runtime),runtime
                assert padded in snapshot['graph']['captured_token_buckets']
                expected_tokens=sum(r['actual_tokens'] for r in runtime)*layers
                expected_events=len(runtime)*layers
            else:
                expected_events=snapshot['phase_steps']['decode']*layers
                expected_tokens=expected_events*batch
            # Padded rows must not inflate token or router-event accounting.
            if args.count_routes:
                totals=snapshot['layers']['totals']
                assert totals['events']==expected_events,(totals,expected_events)
                assert totals['tokens']==expected_tokens,(totals,expected_tokens)
            assert all(meta['current_router_top_k']==meta['original_top_k']
                       for meta in snapshot['execution'].values())
    output.write_text(json.dumps({'status':'PASS','execution':args.child,'spec':spec,'args':vars(args),
                                 'rows':rows,'completed_unix':time.time()},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--dtype',default='bfloat16')
    p.add_argument('--quantization')
    p.add_argument('--gpu-memory-utilization',type=float,default=.35)
    p.add_argument('--spec-json',default=json.dumps(DEFAULT_SPEC))
    p.add_argument('--child',choices=['eager','graph'])
    p.add_argument('--count-routes',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--batches',type=int,nargs='+',default=[3,5])
    p.add_argument('--capture-sizes',type=int,nargs='+')
    args=p.parse_args()
    try:
        args.capture_sizes=capture_sizes(args.batches,args.capture_sizes)
    except ValueError as exc:
        p.error(str(exc))
    if not 0 < args.gpu_memory_utilization < 1:
        p.error('--gpu-memory-utilization must be between 0 and 1')
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    spec=json.loads(args.spec_json)
    if spec is None and args.count_routes:
        p.error('native control requires --no-count-routes; use mode=dense for counters')
    if not args.count_routes and spec is not None and spec.get('mode') not in ('dense','global_native'):
        p.error('the eager action-policy control currently requires counters')
    # This validation run needs diagnostics enabled in both executions.
    if spec and spec.get('mode')=='global_native':
        spec={**spec,'count_routes':args.count_routes}
    if args.child:
        try:
            child(args,spec)
        except Exception as exc:
            previous=json.loads(output.read_text()) if output.exists() else {}
            previous.update(status='FAILED',args=vars(args),execution=args.child,spec=spec,
                            error=f'{type(exc).__name__}: {exc}')
            output.write_text(json.dumps(previous,indent=2))
            raise
        return
    paths={kind:output.with_name(output.stem+f'_{kind}'+output.suffix) for kind in ['eager','graph']}
    if spec is not None and spec.get('mode')=='dense':
        paths['native_graph']=output.with_name(output.stem+'_native_graph'+output.suffix)
        paths['native_eager']=output.with_name(output.stem+'_native_eager'+output.suffix)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('child output path already exists')
    report={'status':'RUNNING','spec':spec,'args':vars(args),
            'children':{key:str(path) for key,path in paths.items()}}
    output.write_text(json.dumps(report,indent=2))
    try:
        for kind,path in paths.items():
            subprocess.run(child_command(args,kind,path,spec),check=True)
        eager=json.loads(paths['eager'].read_text())
        graph=json.loads(paths['graph'].read_text())
        for execution in ('graph','eager'):
            if 'native_'+execution in paths:
                native=json.loads(paths['native_'+execution].read_text())
                counted=graph if execution=='graph' else eager
                matched=all(a['outputs']==b['outputs'] for a,b in zip(native['rows'],counted['rows']))
                report['native_dense_'+execution+'_identity_checked']=matched
                assert matched,f'native {execution} / dense instrumented {execution} outputs differ'
        comparisons=[]
        for erow,grow in zip(eager['rows'],graph['rows']):
            differences=[]
            if args.count_routes:
                for ew,gw in zip(erow['worker'],grow['worker']):
                    ec,gc=ew['layers']['totals'],gw['layers']['totals']
                    for key in ['tokens','events','slots_before','slots_after','distinct_before','distinct_after',
                                'logical_weight_bytes_before','logical_weight_bytes_after']:
                        if ec[key]!=gc[key]:
                            differences.append({'key':key,'eager':ec[key],'graph':gc[key]})
            comparisons.append({'batch':erow['batch'],'outputs_equal':erow['outputs']==grow['outputs'],
                                'counter_differences':differences})
        report['eager_graph_comparisons']=comparisons
        for erow,grow in zip(eager['rows'],graph['rows']):
            assert erow['batch']==grow['batch']
            assert erow['outputs']==grow['outputs'],'eager/graph deterministic outputs differ'
            for ew,gw in zip(erow['worker'],grow['worker']) if args.count_routes else []:
                ec,gc=ew['layers']['totals'],gw['layers']['totals']
                for key in ['tokens','events','slots_before','slots_after','distinct_before','distinct_after',
                            'logical_weight_bytes_before','logical_weight_bytes_after']:
                    assert ec[key]==gc[key],(key,ec[key],gc[key])
        report.update(status='PASS',completed_unix=time.time(),
            scope='fixed-arm graph replay at explicit batches/capture sizes; eager output/count parity; no speedup claim')
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
