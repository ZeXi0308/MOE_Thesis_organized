"""Small, native-only BF16 Triton MoE tuning grid for vLLM 0.26.

Run GPU work only through run_locked.py. --plan-only is CPU-only and checks
actual native route coverage without importing torch/vLLM. Real local OLMoE
layer-0 checkpoint weights are assembled read-only; activations are synthetic
standard-normal BF16 and are explicitly not captured model activations.

The objective is native top-k expert-block latency, not post-skip latency.
All comparison arms must use the SAME emitted VLLM_TUNED_CONFIG_FOLDER in a
fresh engine. CUDA graph timing includes alignment, both GEMMs, activation,
and reduction. It excludes router selection, CPU dispatch, and policy work.
Repeated cache conditioning is a declared approximation, not proof of cold
DRAM. A selected tile is a kernel tuning result, not a service speedup claim.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack,contextmanager
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import time


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def append_json(path,value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def mode_from_spec(spec):
    if spec is None:return 'native'
    if isinstance(spec,dict):return spec.get('mode','dense')
    if isinstance(spec,str):
        match=re.search(r"mode=['\"]([^'\"]+)",spec)
        return match.group(1) if match else None
    return None


def iter_traces(node,mode=None,location='$'):
    """Carry arm/spec provenance into nested worker trace lists."""
    if isinstance(node,list):
        for index,item in enumerate(node):
            yield from iter_traces(item,mode,f'{location}[{index}]')
    elif isinstance(node,dict):
        if 'arm' in node:
            mode=node['arm'] if node['arm'] in ('native','dense') else str(node['arm'])
        if 'spec' in node:
            mode=mode_from_spec(node['spec'])
        if 'execution_mode' in node:
            mode=node['execution_mode']
        if 'topk_ids' in node and 'topk_weights' in node:
            yield node,mode,location
            return
        for key,value in node.items():
            if isinstance(value,(dict,list)):
                yield from iter_traces(value,mode,f'{location}.{key}')


def load_native_routes(paths,layer,experts,top_k):
    selected=defaultdict(list)
    rejected=Counter()
    sources=[]
    layer_pattern=re.compile(rf'(^|\.)layers\.{layer}\.')
    for path in paths:
        path=Path(path)
        if path.suffix=='.jsonl':
            data=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            data=json.loads(path.read_text())
        sources.append({'path':str(path.resolve()),'sha256':sha256_file(path)})
        for row,mode,location in iter_traces(data):
            reason=None
            if mode not in ('native','dense'):reason='not_explicit_native_or_dense'
            elif row.get('phase')!='decode':reason='not_decode'
            elif not layer_pattern.search(row.get('layer','')):reason='other_layer'
            elif row.get('source')!='vllm_actual_select_experts':reason='not_actual_vllm_route'
            elif row.get('forward_completed') is not True:reason='incomplete_forward'
            if reason:
                rejected[reason]+=1
                continue
            ids,weights=row['topk_ids'],row['topk_weights']
            batch=len(ids)
            if (not batch or len(weights)!=batch or row.get('num_experts')!=experts
                    or any(len(a)!=top_k or len(b)!=top_k for a,b in zip(ids,weights))):
                raise ValueError(f'{path}:{location}: native routing shape does not match model')
            for token_ids,token_weights in zip(ids,weights):
                if (len(set(token_ids))!=top_k or any(not isinstance(e,int) or e<0 or e>=experts for e in token_ids)
                    or any(not math.isfinite(w) or w<0 for w in token_weights)):
                    raise ValueError(f'{path}:{location}: invalid native routes')
            if 'post_topk_ids' in row and row['post_topk_ids']!=ids:
                raise ValueError(f'{path}:{location}: reported native route has post-select drops')
            fingerprint=hashlib.sha256(json.dumps([ids,weights],separators=(',',':')).encode()).hexdigest()
            selected[batch].append({'layer':row['layer'],'step':row.get('step'),'batch':batch,
                'num_experts':experts,'top_k':top_k,'topk_ids':ids,'topk_weights':weights,
                'native_mode':mode,'source_file':str(path.resolve()),'source_json_path':location,
                'route_sha256':fingerprint,'distinct_experts':len({e for token in ids for e in token})})
    return dict(selected),{'sources':sources,'rejected':dict(rejected)}


def sample_routes(routes,batches,limit):
    result={}
    for batch in batches:
        rows=routes.get(batch,[])
        if not rows:
            raise ValueError(f'No complete native layer route for B={batch}; capture this batch, do not synthesize/concatenate routes')
        if len(rows)<=limit:
            result[batch]=rows
        else:
            indexes=([len(rows)//2] if limit==1 else
                     [round(i*(len(rows)-1)/(limit-1)) for i in range(limit)])
            result[batch]=[rows[i] for i in indexes]
    return result


def small_grid():
    return [dict(BLOCK_SIZE_M=m,BLOCK_SIZE_N=n,BLOCK_SIZE_K=k,GROUP_SIZE_M=1,
                 SPLIT_K=1,num_warps=4,num_stages=3)
            for m,n,k in itertools.product((16,32),(64,128),(64,128))]


def unique_configs(configs):
    result=[];seen=set()
    for cfg in configs:
        key=json.dumps(cfg,sort_keys=True)
        if key not in seen:
            result.append(dict(cfg));seen.add(key)
    return result


def read_model_config(model):
    model=Path(model).resolve()
    if not model.is_dir():raise ValueError('--model must be an existing local checkpoint directory')
    config=json.loads((model/'config.json').read_text())
    if config.get('model_type')!='olmoe' or config.get('quantization_config'):
        raise ValueError('This minimal tuner supports unquantized OLMoE checkpoints only')
    return model,config


def load_layer_weights(model,config,layer):
    import torch
    from safetensors import safe_open
    E,D,I=config['num_experts'],config['hidden_size'],config['intermediate_size']
    keys=[f'model.layers.{layer}.mlp.experts.{e}.{projection}.weight'
          for e in range(E) for projection in ('gate_proj','up_proj','down_proj')]
    index=model/'model.safetensors.index.json'
    if index.exists():
        mapping=json.loads(index.read_text())['weight_map']
    else:
        mapping={}
        for path in sorted(model.glob('*.safetensors')):
            with safe_open(str(path),framework='pt',device='cpu') as handle:
                for key in handle.keys():mapping[key]=path.name
    missing=[key for key in keys if key not in mapping]
    if missing:raise ValueError(f'missing standard OLMoE expert keys, first: {missing[:3]}')
    w1=torch.empty((E,2*I,D),dtype=torch.bfloat16)
    w2=torch.empty((E,D,I),dtype=torch.bfloat16)
    dtypes=set()
    files=sorted({mapping[key] for key in keys})
    with ExitStack() as stack:
        handles={file:stack.enter_context(safe_open(str(model/file),framework='pt',device='cpu')) for file in files}
        for e in range(E):
            for projection,target,shape in [('gate_proj',w1[e,:I],(I,D)),
                                             ('up_proj',w1[e,I:],(I,D)),('down_proj',w2[e],(D,I))]:
                key=f'model.layers.{layer}.mlp.experts.{e}.{projection}.weight'
                source=handles[mapping[key]].get_tensor(key)
                if tuple(source.shape)!=shape:raise ValueError(f'{key}: {source.shape} != {shape}')
                if source.dtype not in (torch.bfloat16,torch.float16,torch.float32):
                    raise ValueError(f'{key}: unsupported checkpoint dtype {source.dtype}')
                dtypes.add(str(source.dtype));target.copy_(source)
    def tensor_hash(tensor):
        return hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()
    meta={'model_path':str(model),'layer':layer,'source_files':files,'source_dtypes':sorted(dtypes),
          'execution_dtype':'torch.bfloat16','w1_shape':list(w1.shape),'w2_shape':list(w2.shape),
          'w1_sha256':tensor_hash(w1),'w2_sha256':tensor_hash(w2),
          'config_sha256':sha256_file(model/'config.json'),
          'weight_layout':'w1[e]=concat(gate_proj,up_proj) rows; w2[e]=down_proj, TP=1',
          'expert_weight_bytes':(2*I*D+D*I)*2}
    return w1,w2,meta


@contextmanager
def safe_override(package,config):
    # v0.26 override_config lacks try/finally after yield. Preserve its previous
    # process-local value even if a candidate compilation/resource check fails.
    previous=package.get_config()
    try:
        with package.override_config(config):yield
    finally:
        package._config=previous


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--traces',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[1,8,32,64])
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--events-per-batch',type=int,default=3)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--replays-per-sample',type=int,default=10)
    p.add_argument('--evict-mib',type=int,default=256)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--max-relative-l2',type=float,default=.02)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.layer!=0:raise ValueError('This minimal entry point intentionally tunes first layer only')
    if min(args.events_per_batch,args.rounds,args.replays_per_sample)<=0 or args.evict_mib<0:
        raise ValueError('positive event/round/replay counts and nonnegative eviction size required')
    if len(set(args.batches))!=len(args.batches) or any(b<1 or b>64 for b in args.batches):
        raise ValueError('--batches must be distinct integers in [1,64]')
    if not math.isfinite(args.max_relative_l2) or args.max_relative_l2<=0:
        raise ValueError('--max-relative-l2 must be finite and positive')
    model,config=read_model_config(args.model)
    E,K,D,I=(config[key] for key in ('num_experts','num_experts_per_tok','hidden_size','intermediate_size'))
    routes,provenance=load_native_routes(args.traces,args.layer,E,K)
    selected=sample_routes(routes,args.batches,args.events_per_batch)
    args.output_dir.mkdir(parents=True)
    plan={'status':'CPU_PLAN' if args.plan_only else 'PREPARING','model':str(model),
          'shape':{'experts':E,'top_k':K,'hidden':D,'intermediate':I},'layer':args.layer,
          'grid':small_grid(),'include_current_native_config':True,'selected_routes':selected,
          'route_provenance':provenance,'rounds':args.rounds,'replays_per_sample':args.replays_per_sample,
          'seed':args.seed,'cache_conditioning':{'buffer_mib':args.evict_mib,
               'method':'read-modify-write separate CUDA graph before each measured MoE replay' if args.evict_mib else 'none; repeated same-layer hot-cache replay',
               'eviction_time_in_objective':False,'guarantees_cold_DRAM':False},
          'activation_source':'synthetic standard-normal BF16; not captured model activations',
          'selection_target':'native top-k only; equal-trace mean of per-trace median round latencies',
          'scope':'kernel tuning only; all arms share output configs; service remeasurement required'}
    write_json(args.output_dir/'plan.json',plan)
    if args.plan_only:
        print('CPU_PLAN',str(args.output_dir/'plan.json'),flush=True);return
    if importlib.metadata.version('vllm').split('+')[0]!='0.26.0':
        raise RuntimeError('This tuner requires vLLM 0.26.0')
    import torch
    import triton
    import vllm.model_executor.layers.fused_moe as package
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts,get_config_file_name,get_moe_configs,try_get_optimal_moe_config)
    if getattr(package,'get_config')() is not None:
        raise RuntimeError('process already has a MoE override config')
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    raw_path=args.output_dir/'samples.jsonl'
    state={'status':'RUNNING','plan':'plan.json','results':{},'candidate_checks':[],
           'environment':{'torch':torch.__version__,'triton':triton.__version__,'vllm':importlib.metadata.version('vllm'),
                          'gpu':torch.cuda.get_device_name(),'existing_tuned_folder':os.environ.get('VLLM_TUNED_CONFIG_FOLDER')}}
    try:
        w1,w2,weight_meta=load_layer_weights(model,config,args.layer)
        write_json(args.output_dir/'weights.json',weight_meta)
        w1,w2=w1.cuda(),w2.cuda()
        defaults={b:dict(try_get_optimal_moe_config(w1.shape,w2.shape,K,None,b)) for b in args.batches}
        original_map=get_moe_configs(E,I,None)
        emitted={int(b):dict(cfg) for b,cfg in (original_map or {}).items()}
        # Preserve stock matrix points and default transition boundaries;
        # these anchors are NOT tuned measurements. File lookup is nearest-B.
        anchors={1,2,4,8,16,32,64,65,96,97,128,129,512,513,1024,2048,4096,8192,E*129-1,E*129}
        for batch in sorted(anchors):
            if batch not in emitted:
                emitted[batch]=dict(try_get_optimal_moe_config(w1.shape,w2.shape,K,None,batch))
        state['untimed_preserved_anchor_batches']=sorted(anchors-set(args.batches))
        state['current_native_configs']=defaults
        evict_graph=None
        if args.evict_mib:
            scratch=torch.zeros(args.evict_mib*1024*1024//4,device='cuda',dtype=torch.float32)
            scratch.add_(1.)
            torch.cuda.synchronize()
            evict_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(evict_graph):scratch.add_(1.)
        batch_order=list(args.batches);rng.shuffle(batch_order)
        state['batch_order']=batch_order
        for batch in batch_order:
            cfgs=unique_configs([defaults[batch],*small_grid()])
            events=[]
            for route_index,row in enumerate(selected[batch]):
                generator=torch.Generator(device='cuda').manual_seed(args.seed+batch*1009+route_index)
                x=torch.randn(batch,D,generator=generator,device='cuda',dtype=torch.bfloat16)
                ids=torch.tensor(row['topk_ids'],device='cuda',dtype=torch.int32)
                weights=torch.tensor(row['topk_weights'],device='cuda',dtype=torch.float32)
                with safe_override(package,defaults[batch]):
                    reference=fused_experts(x,w1,w2,weights,ids,global_num_experts=E)
                events.append((x,weights,ids,reference))
            graphs={};valid=[]
            build_order=list(range(len(cfgs)));rng.shuffle(build_order)
            for candidate in build_order:
                ok=True
                temporary=[]
                for route_index,(x,weights,ids,reference) in enumerate(events):
                    check={'batch':batch,'candidate':candidate,'config':cfgs[candidate],'trace_index':route_index}
                    try:
                        with safe_override(package,cfgs[candidate]):
                            for _ in range(3):
                                output=fused_experts(x,w1,w2,weights,ids,global_num_experts=E)
                            torch.cuda.synchronize()
                            error=float((output.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12))
                            finite=bool(torch.isfinite(output).all())
                            if not finite or error>args.max_relative_l2:
                                check.update(status='NUMERICAL_MISMATCH',relative_l2=error if math.isfinite(error) else None,finite=finite)
                                ok=False
                            else:
                                graph=torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    graph_output=fused_experts(x,w1,w2,weights,ids,global_num_experts=E)
                                graph.replay()
                                graph_error=float((graph_output.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12))
                                graph_finite=bool(torch.isfinite(graph_output).all())
                                if not graph_finite or graph_error>args.max_relative_l2:
                                    check.update(status='GRAPH_NUMERICAL_MISMATCH',relative_l2=error,
                                        graph_relative_l2=graph_error if math.isfinite(graph_error) else None,
                                        graph_finite=graph_finite)
                                    ok=False
                                else:
                                    temporary.append((route_index,graph,graph_output))
                                    check.update(status='VALID',relative_l2=error,finite=True,
                                        graph_relative_l2=graph_error,graph_finite=True)
                    except Exception as exc:
                        check.update(status='COMPILE_OR_RESOURCE_ERROR',error=f'{type(exc).__name__}: {exc}')
                        ok=False
                        # CUDA execution faults may poison subsequent timings.
                        if isinstance(exc,RuntimeError) and any(s in str(exc).lower() for s in
                                ('illegal memory','device-side assert','unspecified launch failure')):
                            state['candidate_checks'].append(check)
                            write_json(args.output_dir/'summary.json',state)
                            raise
                    state['candidate_checks'].append(check)
                    write_json(args.output_dir/'summary.json',state)
                    if not ok:break
                if ok:
                    valid.append(candidate)
                    for route_index,graph,output in temporary:graphs[(candidate,route_index)]=(graph,output)
            if 0 not in valid:
                raise RuntimeError(f'current native config failed at B={batch}; cannot select against an invalid reference')
            timings=defaultdict(list)
            for repeat in range(args.rounds):
                order=list(itertools.product(valid,range(len(events))));rng.shuffle(order)
                for order_index,(candidate,route_index) in enumerate(order):
                    graph,_=graphs[(candidate,route_index)]
                    # One conditioning/warmup cycle, excluded from samples.
                    if evict_graph:evict_graph.replay()
                    graph.replay()
                    pairs=[]
                    for _ in range(args.replays_per_sample):
                        if evict_graph:evict_graph.replay()
                        begin,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                        begin.record();graph.replay();end.record();pairs.append((begin,end))
                    pairs[-1][1].synchronize()
                    values=[begin.elapsed_time(end)*1000 for begin,end in pairs]
                    sample={'batch':batch,'round':repeat,'order_index':order_index,'candidate':candidate,
                            'config':cfgs[candidate],'trace_index':route_index,
                            'route_sha256':selected[batch][route_index]['route_sha256'],
                            'replay_us':values,'median_replay_us':statistics.median(values)}
                    append_json(raw_path,sample)
                    timings[(candidate,route_index)].append(sample['median_replay_us'])
            scores={candidate:statistics.mean(statistics.median(timings[(candidate,trace)])
                       for trace in range(len(events))) for candidate in valid}
            winner=min(valid,key=lambda candidate:(scores[candidate],candidate))
            emitted[batch]=cfgs[winner]
            state['results'][batch]={'candidates':cfgs,'valid_candidates':valid,'score_us':scores,
                'selected_candidate':winner,'selected_config':cfgs[winner],
                'stock_candidate':0,'selected_vs_stock_ratio':scores[winner]/scores[0],
                'trace_count':len(events),'selection_uses_postskip_data':False}
            write_json(args.output_dir/'summary.json',state)
            del graphs,events
        folder=args.output_dir/'tuned_configs';folder.mkdir()
        filename=get_config_file_name(E,I,None)
        payload={'triton_version':triton.__version__,**{str(b):cfg for b,cfg in sorted(emitted.items())}}
        write_json(folder/filename,payload)
        state.update(status='PASS',completed_unix=time.time(),tuned_config_folder=str(folder.resolve()),
            config_filename=filename,scope='native kernel tuning; use same folder for every service arm and rerun service measurements',
            uncertainty='rounds/traces characterize this microbenchmark; token/graph samples are not independent service runs')
        write_json(args.output_dir/'summary.json',state)
        print('PASS',str(folder.resolve()),flush=True)
    except Exception as exc:
        state.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',failed_unix=time.time())
        write_json(args.output_dir/'summary.json',state)
        raise


if __name__=='__main__':main()
