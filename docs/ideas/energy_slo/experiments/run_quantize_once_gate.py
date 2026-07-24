#!/usr/bin/env python3
"""QuantizeOnce MoE layer-stage existence test on real experts."""
from __future__ import annotations

import argparse, csv, json, random, re, statistics
from pathlib import Path
from typing import Any, Sequence

import torch
from route_row_policy import DualResidentExpertMLP, RuntimeCounters, _fp8_quantize_per_tensor, require_cuda_fp8

EXPERT_PATH=re.compile(r"(?:^|\.)layers\.(\d+)\..*experts\.(\d+)$")

def parse_ints(v:str)->tuple[int,...]: return tuple(int(x) for x in v.split(','))
def args()->argparse.Namespace:
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model-path',type=Path,required=True); p.add_argument('--layer',type=int,default=0)
 p.add_argument('--active-experts',type=int,default=8); p.add_argument('--top-k',type=int,default=8)
 p.add_argument('--tokens',type=parse_ints,default=(8,16,32,64,128,256)); p.add_argument('--blocks',type=int,default=30)
 p.add_argument('--warmups',type=int,default=3); p.add_argument('--bootstrap',type=int,default=1000)
 p.add_argument('--seed',type=int,default=20260723); p.add_argument('--output-dir',type=Path,required=True); return p.parse_args()

def find(model:Any,layer:int,count:int)->list[Any]:
 found={}
 for name,module in model.named_modules():
  m=EXPERT_PATH.search(name)
  if m and int(m.group(1))==layer and (hasattr(module,'gate_proj') or hasattr(module,'w1')): found[int(m.group(2))]=module
 if len(found)<count: raise RuntimeError(f'need {count} experts, found {len(found)}')
 return [found[i] for i in sorted(found)[:count]]

def event_us(fn:Any)->float:
 s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record(); y=fn(); e.record(); e.synchronize()
 if not isinstance(y,torch.Tensor) or y.numel()==0: raise RuntimeError('invalid output')
 return float(s.elapsed_time(e))*1000

def pct(v:Sequence[float],p:float)->float:
 a=sorted(v); x=(len(a)-1)*p; lo=int(x); hi=min(lo+1,len(a)-1); return a[lo]*(hi-x)+a[hi]*(x-lo)

def bootstrap(rows:list[dict[str,object]],tokens:int,arm:str,n:int,seed:int)->dict[str,float]:
 sample=[r for r in rows if int(r['tokens'])==tokens]
 def metric(sel:list[dict[str,object]])->float:
  bf=statistics.mean(float(r['bf16_us']) for r in sel); other=statistics.mean(float(r[f'{arm}_us']) for r in sel); return (bf-other)/bf
 point=metric(sample); rng=random.Random(seed); values=[metric([sample[rng.randrange(len(sample))] for _ in sample]) for _ in range(n)]
 return {'speedup':point,'ci95_low':pct(values,.025),'ci95_high':pct(values,.975)}

def main()->None:
 a=args()
 if a.output_dir.exists(): raise RuntimeError('refusing overwrite')
 if a.top_k>a.active_experts: raise ValueError('top-k cannot exceed active experts')
 require_cuda_fp8('cuda:0',probe_kernel=True)
 from transformers import AutoModelForCausalLM
 model=AutoModelForCausalLM.from_pretrained(str(a.model_path),torch_dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True).eval()
 originals=find(model,a.layer,a.active_experts); counters=RuntimeCounters(); experts=[DualResidentExpertMLP(e,counters).eval() for e in originals]
 hidden=experts[0].input_features; gen=torch.Generator(device='cuda'); gen.manual_seed(a.seed)
 inputs={t:torch.randn(t,hidden,generator=gen,device='cuda',dtype=torch.bfloat16) for t in a.tokens}
 indices={}
 for t in a.tokens:
  route=torch.tensor([[(token+slot)%a.active_experts for slot in range(a.top_k)] for token in range(t)],device='cuda')
  indices[t]=[torch.nonzero(route==eid,as_tuple=False)[:,0].contiguous() for eid in range(a.active_experts)]

 def down(expert:Any,intermediate:torch.Tensor)->torch.Tensor:
  q,s=_fp8_quantize_per_tensor(intermediate)
  return torch._scaled_mm(q,expert.down.weight_fp8_t,scale_a=s,scale_b=expert.down.weight_scale,out_dtype=torch.bfloat16,use_fast_accum=True)
 def shared(expert:Any,x:torch.Tensor)->torch.Tensor:
  q,s=_fp8_quantize_per_tensor(x)
  g=torch._scaled_mm(q,expert.gate.weight_fp8_t,scale_a=s,scale_b=expert.gate.weight_scale,out_dtype=torch.bfloat16,use_fast_accum=True)
  u=torch._scaled_mm(q,expert.up.weight_fp8_t,scale_a=s,scale_b=expert.up.weight_scale,out_dtype=torch.bfloat16,use_fast_accum=True)
  return down(expert,expert.original_expert.act_fn(g)*u)
 def stage(tokens:int,arm:str)->torch.Tensor:
  x=inputs[tokens]; result=torch.zeros_like(x)
  global_q=global_s=None
  if arm=='quantize_once': global_q,global_s=_fp8_quantize_per_tensor(x)
  for eid,expert in enumerate(experts):
   idx=indices[tokens][eid]
   if idx.numel()==0:
    continue
   if arm=='bf16': y=expert.original_expert(torch.index_select(x,0,idx))
   elif arm=='per_expert': y=shared(expert,torch.index_select(x,0,idx))
   else:
    q=torch.index_select(global_q,0,idx)
    g=torch._scaled_mm(q,expert.gate.weight_fp8_t,scale_a=global_s,scale_b=expert.gate.weight_scale,out_dtype=torch.bfloat16,use_fast_accum=True)
    u=torch._scaled_mm(q,expert.up.weight_fp8_t,scale_a=global_s,scale_b=expert.up.weight_scale,out_dtype=torch.bfloat16,use_fast_accum=True)
    y=down(expert,expert.original_expert.act_fn(g)*u)
   result.index_add_(0,idx,y/a.top_k)
  return result
 arms=('bf16','per_expert','quantize_once'); raw=[]; rng=random.Random(a.seed+1)
 with torch.inference_mode():
  for t in a.tokens:
   for arm in arms:
    for _ in range(a.warmups): stage(t,arm)
  torch.cuda.synchronize()
  for block in range(a.blocks):
   order=list(a.tokens); rng.shuffle(order)
   for t in order:
    arm_order=list(arms); rng.shuffle(arm_order); measured={arm:event_us(lambda arm=arm,t=t:stage(t,arm)) for arm in arm_order}
    raw.append({'block':block,'tokens':t,**{f'{arm}_us':measured[arm] for arm in arms}})
  quality={}
  for t in a.tokens:
   ref=stage(t,'bf16').float()
   quality[str(t)]={arm:float(((stage(t,arm).float()-ref).square().sum()/ref.square().sum().clamp_min(1e-12)).item()) for arm in ('per_expert','quantize_once')}
 cells=[]
 for t in a.tokens:
  cells.append({'tokens':t,'rows_per_expert':t*a.top_k/a.active_experts,'per_expert':bootstrap(raw,t,'per_expert',a.bootstrap,a.seed+t),'quantize_once':bootstrap(raw,t,'quantize_once',a.bootstrap,a.seed+1000+t),'quality':quality[str(t)]})
 positive=[c for c in cells if c['quantize_once']['ci95_low']>.05]
 verdict='QUANTIZE_ONCE_EXISTS' if positive else 'NO_GO_QUANTIZE_ONCE_NO_FAST_REGION'
 summary={'verdict':verdict,'gpu':torch.cuda.get_device_name(0),'layer':a.layer,'active_experts':a.active_experts,'top_k':a.top_k,'cells':cells,'boundary':'REAL_EXPERT_STAGE_SYNTHETIC_UNIFORM_ROUTE_NOT_SERVING_NOT_ENERGY'}
 a.output_dir.mkdir(parents=True)
 with (a.output_dir/'timings.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(raw[0])); w.writeheader(); w.writerows(raw)
 (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
