#!/usr/bin/env python3
"""Actual BF16 OLMoE-shape layer probe, not complete-serving performance.

One shared GPU group lock covers correctness and timing. Both timed methods
copy identical cold expert chunks from pinned CPU memory on every invocation.
The complete GPU weights exist only for the numerical reference. They are not
part of the serving resource claim. Synthetic fixed states are not route oracles.
"""
import argparse
import fcntl
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

p = argparse.ArgumentParser()
p.add_argument("--src-dir", type=Path, default=Path(__file__).parent / "wave_src")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--repeats", type=int, default=8)
args = p.parse_args()
assert not args.output.exists()
receipt = dict(status="WAITING_FOR_GROUP_LOCK", cases=[], timing=[], started_unix_s=time.time())
def save():
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
save()
lock = open('/root/autodl-tmp/moe-research-gpu.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX)
try:
    procs = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    uuid = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'], text=True).strip()
    assert not procs and uuid == 'GPU-273c48a6-cbbb-f4e4-ae0e-78f1876a0f51', (procs,uuid)
    receipt.update(status="RUNNING", acquired_unix_s=time.time(), gpu_uuid=uuid); save()
    sys.path.insert(0, str(args.src_dir.resolve()))
    import torch
    import vllm
    from deferred_expert_reduction import DeferredExpertReduction
    fm = importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
    from vllm.model_executor.layers.fused_moe import override_config
    assert vllm.__version__ == '0.26.0'
    torch.manual_seed(19)
    E,H,I,K = 64,2048,1024,8
    w1 = torch.randn((E,2*I,H), device='cuda', dtype=torch.bfloat16) * 0.02
    w2 = torch.randn((E,H,I), device='cuda', dtype=torch.bfloat16) * 0.02

    def compare(actual, reference):
        delta = actual.float()-reference.float()
        return dict(bit_equal=torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)),
                    maxabs=delta.abs().max().item(), relative_l2=(delta.norm()/reference.float().norm()).item(),
                    allfinite=bool(torch.isfinite(actual).all()))

    def prepare(m, seed, cap):
        torch.manual_seed(seed)
        x = torch.randn((m,H), device='cuda', dtype=torch.bfloat16)
        ids = torch.randn((m,E), device='cuda').topk(K,dim=1).indices.to(torch.int32)
        weights = torch.rand((m,K), device='cuda');weights = weights / weights.sum(-1,keepdim=True)
        active = sorted(set(ids.cpu().reshape(-1).tolist()))
        waves=[]
        for start in range(0,len(active),cap):
            experts=active[start:start+cap]
            # Reverse slot assignment so global expert id is never the slot id.
            slots={e:len(experts)-1-i for i,e in enumerate(experts)}
            mapping=torch.full((E,),-1,device='cuda',dtype=torch.int32)
            a=torch.zeros((cap,2*I,H),dtype=torch.bfloat16,pin_memory=True)
            b=torch.zeros((cap,H,I),dtype=torch.bfloat16,pin_memory=True)
            for e,slot in slots.items():
                a[slot].copy_(w1[e]);b[slot].copy_(w2[e]);mapping[e]=slot
            waves.append((experts,mapping,a,b))
        s1=torch.empty((cap,2*I,H),device='cuda',dtype=torch.bfloat16)
        s2=torch.empty((cap,H,I),device='cuda',dtype=torch.bfloat16)
        return x,ids,weights,active,waves,s1,s2

    def run(data, deferred, reverse=False, poison=False):
        x,ids,weights,active,waves,s1,s2=data
        acc=None;out=None;written=torch.zeros_like(ids,dtype=torch.bool)
        if deferred:
            acc=DeferredExpertReduction(x,weights,ids,s1,s2,activation=fm.MoEActivation.SILU,global_num_experts=E)
            if poison:
                acc.fc1.fill_(float('nan'));acc.act.fill_(float('nan'));acc.contributions.fill_(float('nan'))
        for experts,mapping,a,b in (list(reversed(waves)) if reverse else waves):
            s1.copy_(a,non_blocking=True);s2.copy_(b,non_blocking=True)
            if deferred:
                old=acc.contributions[written].clone() if poison else None
                acc.wave(s1,s2,mapping,experts)
                if poison:
                    assert torch.equal(old.view(torch.uint8),acc.contributions[written].view(torch.uint8))
                    selected=mapping[ids.long()]>=0
                    assert not bool((written & selected).any())
                    written |= selected
                    assert bool(torch.isnan(acc.contributions[~written]).all())
                    assert bool(torch.isfinite(acc.contributions[written]).all())
            else:
                y=fm.fused_experts(x,s1,s2,weights,ids,global_num_experts=E,expert_map=mapping)
                if out is None:out=y
                else:out.add_(y)
        if acc is not None:
            out=acc.finish(active)
            if poison:assert bool(written.all())
        return out

    # Original shape-selected config and common explicit config are separate cases.
    for m,seed,cap in ((1,0,24),(16,0,24),(257,17,20),(2048,17,23)):
        data=prepare(m,seed,cap);x,ids,weights,active,waves,s1,s2=data
        config=fm.try_get_optimal_moe_config(s1.shape,s2.shape,K,fm._get_config_dtype_str(dtype=x.dtype),m)
        for fixed in (False,True):
            from contextlib import nullcontext
            with (override_config(config) if fixed else nullcontext()):
                ref=fm.fused_experts(x,w1,w2,weights,ids,global_num_experts=E)
                old=run(data,False);new=run(data,True,poison=True);rev=run(data,True,reverse=True,poison=True)
                case=dict(rows=m,seed=seed,cap=cap,waves=len(waves),fixed_config=fixed,config=config,
                          old_vs_reference=compare(old,ref),new_vs_reference=compare(new,ref),
                          reverse_vs_forward=compare(rev,new),poison_and_slot_preservation_pass=True)
                receipt['cases'].append(case);save()
                assert case['new_vs_reference']['allfinite'] and case['reverse_vs_forward']['bit_equal']
                if fixed:assert case['new_vs_reference']['bit_equal'],case
        # Alternating complete layer invocations including actual H2D of padded chunks.
        for method in (False,True):
            run(data,method);run(data,method)
        torch.cuda.synchronize()
        samples={False:[],True:[]}
        for repeat in range(args.repeats):
            for method in ((False,True) if repeat%2==0 else (True,False)):
                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();result=run(data,method);b.record();b.synchronize()
                samples[method].append(dict(wall_s=time.perf_counter()-start,gpu_ms=a.elapsed_time(b),
                                            extra_peak_bytes=torch.cuda.max_memory_allocated()-base))
                del result
        receipt['timing'].append(dict(rows=m,cap=cap,waves=len(waves),repeats=args.repeats,
            copied_bytes_per_call=sum(a.numel()*2+b.numel()*2 for _,_,a,b in waves),
            scope='cold layer: pinned H2D + native group execution + reduction; all groups padded to cap; not serving',
            baseline=samples[False],deferred=samples[True],
            mean_wall_ratio=statistics.mean(r['wall_s'] for r in samples[True])/statistics.mean(r['wall_s'] for r in samples[False])))
        save();del data,x,ids,weights,waves,s1,s2,ref,old,new,rev
    receipt['status']='PASS'
except BaseException as exc:
    receipt.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    raise
finally:
    receipt['finished_unix_s']=time.time();save()
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
