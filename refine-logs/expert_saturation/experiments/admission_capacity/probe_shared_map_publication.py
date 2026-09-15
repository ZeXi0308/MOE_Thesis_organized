"""Single-layer conditional X invocation. GPU use requires the shared executor window.

Preparation is CPU-only. GPU mode uses synthetic values and recorded routes,
restores the recorded pre-call private state before every invocation, and calls
the frozen planner/ordered_dispatch/oneshot. It is not a request replay.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

SOURCE_FILES = ('run_shared_pool_pager.py', 'shared_pool_plan.py', 'full_stage_plan.py',
                'analyze_layer_budget.py', 'wisp_expert_groups.py')
MODES = ('blocking', 'execution_only', 'all_async')
EXPERT_BYTES = 12582912


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_module(directory):
    sys.path.insert(0, str(directory.resolve()))
    return importlib.import_module('run_shared_pool_pager')


def symbolic_apply(contents, plan):
    result = dict(contents)
    for c in plan['d2d']:
        result[c['dst']] = result[c['src']]
    for c in plan['h2d']:
        result[c['dst']] = c['expert']
    return result


def prepare(repo, case_file, source_dir):
    selected = json.loads(case_file.read_text())
    trace = repo / selected['source']
    if digest(trace) != selected['source_sha256']:
        raise ValueError('recorded trace changed')
    module = source_module(source_dir)
    wanted = {c['call_id']: name for name, c in selected['cases'].items()}
    previous, ever, cases = {}, {}, {}
    for line in trace.open():
        r = json.loads(line); layer = r['layer_name']
        if r['call_id'] in wanted:
            name = wanted[r['call_id']]; c = deepcopy(selected['cases'][name])
            before = previous[layer]; entry = before['shared_plan']['final_state']
            if before['status'] != 'complete' or before['execution'] != 'shared_pool_oneshot':
                raise ValueError('no exact completed same-layer predecessor')
            p = module.plan_shared_pool(c['active_experts'], entry, c['shared_plan']['layer_index'])
            if p != c['shared_plan'] or p != r['shared_plan']:
                raise ValueError('predecessor does not reproduce recorded plan')
            contents = {p['layer_index']*21+slot: int(e) for e, slot in entry['expert_to_slot'].items()}
            once = symbolic_apply(contents, p); twice = symbolic_apply(once, p)
            bad = lambda x: [e for e in c['active_experts'] if x.get(p['expert_map_device'][e]) != e]
            if bad(once) or any(once[p['layer_index']*21+s] != e
                               for s,e in enumerate(p['final_state']['slot_to_expert']) if e != -1):
                raise ValueError('symbolic physical contents mismatch')
            c.update(entry_state=entry, previous_call_id=before['call_id'],
                     entry_ever_loaded=sorted(ever.get(layer, set())), context=r['context'],
                     plan_reproduced=True, first_call_wrong_experts=bad(once),
                     repeated_without_reset_wrong_experts=bad(twice))
            cases[name] = c
        ever.setdefault(layer, set()).update(r.get('loaded_experts', []))
        previous[layer] = r
        if len(cases) == len(wanted):
            break
    if set(cases) != set(selected['cases']):
        raise ValueError('selected calls missing')
    return dict(status='CPU_PREPARED_GPU_UNRUN', cases=cases,
                trace_sha256=selected['source_sha256'], selected_sha256=digest(case_file),
                source_hashes={n: digest(source_dir/n) for n in SOURCE_FILES},
                GPU_runs=0, scope='Recorded pre-state/route with synthetic numerical values; conditional invocation only',
                resource_bytes=dict(pool=384*EXPERT_BYTES, host_master=64*EXPERT_BYTES,
                    qualification_reference_gpu=64*EXPERT_BYTES, banks_each_host_gpu=28672),
                primary_order=list(MODES)+list(reversed(MODES)))


def check_occupancy(allowed_pid=None):
    import subprocess
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=10, check=True)
    pids = [int(v.strip()) for v in p.stdout.splitlines() if v.strip()]
    if any(pid != allowed_pid for pid in pids):
        raise RuntimeError('other GPU compute process present')
    q = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
    return dict(compute_pids=pids, gpu_snapshot=q.stdout.splitlines())


def run_gpu(args, prepared, module):
    import os
    import resource
    from threading import Lock
    import torch
    import vllm
    from shared_map_staging import SharedMapPair, install
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    if not torch.__version__.startswith('2.11.') or not vllm.__version__.startswith('0.26.'):
        raise RuntimeError('probe targets the frozen PyTorch 2.11 / vLLM 0.26 runtime')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    torch.manual_seed(53036)
    masters = tuple(torch.empty(s, dtype=torch.bfloat16, pin_memory=True)
                    for s in ((64,2048,2048),(64,2048,1024)))
    for master in masters:
        for expert in master:
            expert.copy_(torch.randn(expert.shape,dtype=torch.float32).mul_(.02))
    pools = tuple(torch.empty((384,*m.shape[1:]), dtype=m.dtype, device=device) for m in masters)
    for pool in pools:
        pool.fill_(float('nan'))
    banks = [SharedMapPair(torch,num_experts=64,private_cap=21,pool_cap=384,device=device) for _ in range(16)]
    original = module.oneshot
    method = SimpleNamespace(moe_quant_config=FUSED_MOE_UNQUANTIZED_CONFIG)
    all_rows = []
    with (args.output_dir/'samples.jsonl').open('x') as journal, torch.inference_mode():
        for name, c in prepared['cases'].items():
            layer = SimpleNamespace(layer_name=c['layer_name'], activation=MoEActivation.SILU,
                                    apply_router_weight_on_input=False)
            state = SimpleNamespace(num_experts=64, cap_experts=21, cpu_w13=masters[0],cpu_w2=masters[1],
                                    expert_map_device=torch.empty(64,dtype=torch.int32,device=device))
            entry = dict(state=state,layer_name=c['layer_name'])
            runtime = SimpleNamespace(torch=torch,layers={id(layer):entry},shared_weights=pools,
                private_cap=21,shared_mode='oneshot',shared_qualification=False,group_retention='none',
                summary=None,_flush_error=None,measurement=True,validation_enabled=False,
                next_call_id=c['call_id'],context=c['context'],records=[],events=[],kernel=fused_experts,
                expert_bytes=lambda state: EXPERT_BYTES,shared_call_lock=Lock(),shared_transitions=[],
                shared_stream=torch.cuda.current_stream().cuda_stream,shared_last_event=torch.cuda.Event())
            runtime.shared_last_event.record()
            base=c['shared_plan']['layer_index']*21
            state.scratch_w13=pools[0][base:base+21];state.scratch_w2=pools[1][base:base+21]
            x=torch.randn(c['rows'],2048,dtype=torch.float32).to(dtype=torch.bfloat16,device=device)
            cpu_ids=torch.tensor(c['row_topk_experts'],dtype=torch.long)
            logits=torch.randn(c['rows'],64,dtype=torch.float32).mul_(.1).sub_(2)
            logits.scatter_(1,cpu_ids,torch.linspace(2,1,8).expand(c['rows'],8))
            if not torch.equal(logits.topk(8,dim=-1).indices,cpu_ids):
                raise RuntimeError('synthetic router scores do not realize recorded top-k order')
            # OLMoE norm_topk_prob=false: gather from softmax64, no softmax8 renormalization.
            weights=logits.softmax(-1).gather(1,cpu_ids).to(device)
            ids=cpu_ids.to(dtype=torch.int32,device=device)

            def reset():
                torch.cuda.synchronize(); start=time.perf_counter_ns()
                meta=deepcopy(c['entry_state'])
                for key in ('slot_to_expert','lru_tick','lru_clock'):
                    setattr(state,key,meta[key])
                state.expert_to_slot={int(e):s for e,s in meta['expert_to_slot'].items()}
                state.stats_forward=state.stats_hits=state.stats_miss=state.stats_evict=0
                entry['ever_loaded']=set(c['entry_ever_loaded'])
                for e,s in state.expert_to_slot.items():
                    for pool,master in zip(pools,masters):
                        pool[base+s].copy_(master[e],non_blocking=True)
                state.expert_map_device.copy_(torch.tensor(meta['expert_map_device'],dtype=torch.int32,device=device))
                runtime.records.clear();runtime.events.clear()
                torch.cuda.synchronize()
                return (time.perf_counter_ns()-start)/1e6

            def invoke():
                started=time.perf_counter_ns()
                y=module.ordered_dispatch(runtime,None,method,layer,x,weights,ids)
                returned=time.perf_counter_ns();torch.cuda.synchronize();finished=time.perf_counter_ns()
                return y,dict(host_return_ms=(returned-started)/1e6,
                    post_return_drain_ms=(finished-returned)/1e6,complete_call_ms=(finished-started)/1e6)

            def mode_set(mode):
                torch.cuda.synchronize(); module.oneshot=original
                if mode != 'blocking':
                    install(module,runtime,{id(layer):banks[c['shared_plan']['layer_index']]},mode=mode)

            # Full resident reference is a numerical qualification, outside timing.
            refs=tuple(t.to(device) for t in masters)
            reference=fused_experts(hidden_states=x,w1=refs[0],w2=refs[1],topk_weights=weights,
                topk_ids=ids,global_num_experts=64,expert_map=None,activation=layer.activation,
                quant_config=method.moe_quant_config,apply_router_weight_on_input=False)
            torch.cuda.synchronize();reference_cpu=reference.cpu();del refs,reference
            torch.cuda.empty_cache()
            for mode in MODES:
                mode_set(mode);reset();y,_=invoke()
                if not bool(torch.isfinite(y).all()) or not torch.equal(y.cpu().view(torch.uint8),reference_cpu.view(torch.uint8)):
                    raise RuntimeError('full-reference numerical mismatch')
                contents={base+int(s):int(e) for e,s in c['shared_plan']['final_state']['expert_to_slot'].items()}
                contents.update({c['shared_plan']['expert_map_device'][e]:e for e in c['active_experts']})
                for pos,e in contents.items():
                    for pool,master in zip(pools,masters):
                        if not torch.equal(pool[pos].cpu().view(torch.uint8),master[e].view(torch.uint8)):
                            raise RuntimeError('physical expert content mismatch')
            for block,mode in enumerate(prepared['primary_order']):
                environment=check_occupancy(os.getpid());mode_set(mode)
                for repeat in range(args.warmup+args.samples):
                    reset_ms=reset();y,timing=invoke()
                    record=runtime.records[-1]
                    maps=state.expert_map_device.cpu().tolist()
                    if (record['shared_plan'] != c['shared_plan'] or maps != c['shared_plan']['final_state']['expert_map_device']
                            or not torch.equal(y.cpu().view(torch.uint8),reference_cpu.view(torch.uint8))):
                        raise RuntimeError('trial state/plan/output mismatch')
                    row=dict(case=name,block=block,mode=mode,repeat=repeat,
                        phase='warmup' if repeat<args.warmup else 'measurement',reset_ms=reset_ms,**timing,
                        host_apply_ms=record['host_apply_ms'],host_plan_ms=record['host_plan_ms'],
                        route_to_host_ms=record['route_to_host_ms'],weight_copy_bytes=record['weight_copy_bytes'],
                        d2d_copy_bytes=record['d2d_copy_bytes'],environment=environment)
                    journal.write(json.dumps(row)+'\n');journal.flush();all_rows.append(row)
            # Profiler is a separate diagnostic, never the primary timing sample.
            for mode in MODES:
                mode_set(mode);reset()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as prof:
                    with torch.profiler.record_function('conditional_X_call'):
                        invoke()
                prof.export_chrome_trace(str(args.output_dir/f'{name}-{mode}-diagnostic.json'))
        torch.cuda.synchronize()
        return dict(status='COMPLETE_CONDITIONAL_INVOCATION',samples=len(all_rows),
            torch_version=torch.__version__,vllm_version=vllm.__version__,cuda_version=torch.version.cuda,
            peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),
            linux_host_hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            pinned_master_bytes=sum(t.numel()*t.element_size() for t in masters),
            scope=prepared['scope'],resource_bytes=prepared['resource_bytes'],
            baseline='unmodified frozen oneshot plus the same allocated banks; candidate wrapper cost included',
            timing='route/plan/copies/maps/kernel host call plus final drain; reset separately; profiler diagnostic separate',
            request_level_benefit=None,postflight=check_occupancy(os.getpid()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','gpu'))
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--repo-root',type=Path);p.add_argument('--case-file',type=Path)
    p.add_argument('--prepared',type=Path);p.add_argument('--lock-file',type=Path)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--samples',type=int,default=12);p.add_argument('--warmup',type=int,default=3)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    if args.action=='prepare':
        if args.repo_root is None or args.case_file is None:p.error('prepare requires repo-root and case-file')
        result=prepare(args.repo_root,args.case_file,args.source_dir)
        (args.output_dir/'prepared.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:dict(previous=v['previous_call_id'],exact=v['plan_reproduced'],
            repeated_without_reset_wrong_experts=v['repeated_without_reset_wrong_experts']) for k,v in result['cases'].items()}))
        return
    if args.prepared is None or args.lock_file is None:p.error('GPU requires prepared input and shared lock file')
    if args.samples<1 or args.warmup<1:p.error('positive sample/warmup counts required')
    import fcntl
    result=dict(status='ABORT_BEFORE_GPU_INITIALIZATION')
    try:
        with args.lock_file.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            preflight=check_occupancy()
            (args.output_dir/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
            prepared=json.loads(args.prepared.read_text())
            if any(digest(args.source_dir/n)!=h for n,h in prepared['source_hashes'].items()):
                raise RuntimeError('frozen source mismatch')
            result['status']='FAILED_OR_INCOMPLETE'
            result=run_gpu(args,prepared,source_module(args.source_dir))
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        (args.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
