"""Standalone native-layout KV gather/D2H/H2D/scatter probe; not a serving result."""
import argparse,fcntl,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--lock',default='/root/autodl-tmp/moe-research-gpu.lock');p.add_argument('--trials',type=int,default=7);a=p.parse_args()
out=Path(a.output)
if out.exists():raise SystemExit('refuse overwrite')
out.parent.mkdir(parents=True,exist_ok=True)
result={'status':'STARTING','rows':[]}
lock=open(a.lock,'a')
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 check=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
 result['gpu_processes_before']=check.stdout
 if check.stdout.strip():raise RuntimeError('ABORT_GPU_BUSY')
 result['gpu_inventory']=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
 import torch
 result['torch']=torch.__version__
 # Match captured FLASH_ATTN logical shape and physical stride; full KV pool.
 layers=[torch.empty((6657,16,16,256),dtype=torch.bfloat16,device='cuda').transpose(1,2) for _ in range(16)]
 block_values=(torch.arange(6657,device='cuda')%127).to(torch.bfloat16).view(-1,1,1,1)
 for i,x in enumerate(layers):x.copy_(block_values);x.add_(i)
 assert list(layers[0].stride())==[65536,256,4096,1]
 for blocks in [205,207,256]:
  ids=torch.randperm(6657,device='cuda',generator=torch.Generator(device='cuda').manual_seed(730+blocks))[:blocks]
  host=torch.empty((16,blocks,16,16,256),dtype=torch.bfloat16,pin_memory=True)
  stage=torch.empty((blocks,16,16,256),dtype=torch.bfloat16,device='cuda')
  # One warmup is retained separately; all seven measured repeats are retained.
  for trial in range(-1,a.trials):
   torch.cuda.synchronize();begin=time.perf_counter()
   for i,x in enumerate(layers):
    torch.index_select(x,0,ids,out=stage)
    host[i].copy_(stage,non_blocking=True)
   torch.cuda.synchronize();offload=time.perf_counter()-begin
   for x in layers:x.index_fill_(0,ids,0)
   torch.cuda.synchronize();begin=time.perf_counter()
   for i,x in enumerate(layers):
    stage.copy_(host[i],non_blocking=True)
    x.index_copy_(0,ids,stage)
   torch.cuda.synchronize();restore=time.perf_counter()-begin
   # Verify original known per-block/layer contents, outside all timed regions.
   expected=block_values.index_select(0,ids)
   good=all(bool(torch.all(x.index_select(0,ids)==expected+i).item()) for i,x in enumerate(layers))
   result['rows'].append(dict(blocks=blocks,trial=trial,warmup=trial<0,bytes=host.numel()*host.element_size(),offload_s=offload,restore_s=restore,roundtrip_s=offload+restore,correct=good))
   if not good:raise RuntimeError('KV_ROUNDTRIP_MISMATCH')
  del host,stage
 result['status']='COMPLETE'
 result['scope']='Isolated copy primitive with synthetic payload and real captured layout. Includes Python dispatch, gather/scatter, D2H/H2D and synchronize. Excludes allocation, engine integration, concurrent decode and full-request effects; full-block padding charged.'
except Exception as e:
 result['status']='ABORTED_OR_FAILED';result['error']=repr(e)
 raise
finally:
 out.write_text(json.dumps(result,indent=2)+'\n')
