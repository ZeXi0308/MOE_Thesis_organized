"""Native Triton MoE checks for -1 slots, including absent-expert NaN poison."""
import argparse
import json
import pathlib
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=pathlib.Path(a.output)
    if out.exists():raise FileExistsError(out)
    import torch
    from expert_skip import ExpertSkipPolicy,SkipConfig,install_safe_alignment
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    torch.manual_seed(1701)
    E,K,D,H=64,8,128,128
    w1=torch.randn(E,2*H,D,device='cuda',dtype=torch.bfloat16)*.03
    w2=torch.randn(E,D,H,device='cuda',dtype=torch.bfloat16)*.03
    patch=install_safe_alignment();rows=[]
    try:
        for B in [1,2,8,32,64]:
            x=torch.randn(B,D,device='cuda',dtype=torch.bfloat16)
            ids=((torch.arange(B,device='cuda')[:,None]*5+torch.arange(K,device='cuda'))%E).int()
            w=torch.arange(K,0,-1,device='cuda',dtype=torch.float32)[None,:].expand(B,-1).contiguous()/36
            for mode in ['global','gate','marginal']:
                spec=dict(mode=mode,drop_j=4,tail_rank=4,gate_threshold=.15,mass_budget=.4,backend='torch')
                policy=ExpertSkipPolicy(SkipConfig(**spec))
                ww,ii=policy('test',w,ids,E,True,w1[0].numel()*2+w2[0].numel()*2)
                got=fused_experts(x,w1,w2,ww,ii,global_num_experts=E)
                ref=torch.zeros(B,D,device='cuda',dtype=torch.float32)
                for t in range(B):
                    for slot in range(K):
                        e=int(ii[t,slot])
                        if e<0:continue
                        h=torch.nn.functional.linear(x[t].float(),w1[e].float())
                        a0,a1=h.chunk(2)
                        ref[t]+=torch.nn.functional.linear(torch.nn.functional.silu(a0)*a1,w2[e].float())*ww[t,slot]
                err=float((got.float()-ref).norm()/ref.norm().clamp_min(1e-12))
                active=set(ii[ii>=0].cpu().tolist())
                absent=sorted(set(range(E))-active)
                poisoned1=w1.clone();poisoned2=w2.clone()
                poisoned1[absent]=float('nan');poisoned2[absent]=float('nan')
                poison=fused_experts(x,poisoned1,poisoned2,ww,ii,global_num_experts=E)
                finite=bool(torch.isfinite(poison).all())
                exact=bool(torch.equal(poison,got))
                row={'batch':B,'spec':spec,'skipped_slots':int((ii<0).sum()),
                     'distinct_before':len(set(ids.flatten().cpu().tolist())),
                     'distinct_after':len(active),'relative_l2_to_fp32_reference':err,
                     'absent_experts_poisoned':len(absent),'poison_output_finite':finite,
                     'poison_equals_unpoisoned':exact,'counters':policy.snapshot()}
                rows.append(row)
                out.write_text(json.dumps({'status':'RUNNING','rows':rows},indent=2))
                if err>.03 or not finite or not exact:
                    raise AssertionError(row)
        out.write_text(json.dumps({'status':'PASS','scope':'synthetic weights; BF16 Triton numerical and absent-weight poison test, not service performance or quality',
                                   'rows':rows,'completed_unix':time.time()},indent=2))
        print('PASS',len(rows),flush=True)
    finally:patch.restore()


if __name__=='__main__':main()
