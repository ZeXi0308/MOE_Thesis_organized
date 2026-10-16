"""CPU interpreter reproducer for E2M1 division ties; not GPU evidence."""
import os
os.environ['TRITON_INTERPRET']='1'
import argparse,json,pathlib
import torch,triton,codec,codec_reference as ref
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
results=[]
for seed in range(5):
    for d in [16,48,512]:
        torch.manual_seed(seed)
        x=torch.randn(7,d,dtype=torch.bfloat16);x[0,0]=1000;x[1]*=1e-6
        q,s,g=ref.nvfp4_quant(x);cq=torch.empty_like(q);cs=torch.empty_like(s)
        codec._nvq[(7,)](x,cq,cs,g,d,triton.next_power_of_2(d))
        assert torch.equal(q,cq),(seed,d,'payload')
        assert torch.equal(s.view(torch.uint8),cs.view(torch.uint8)),(seed,d,'scales')
        y=torch.empty_like(x,dtype=torch.float32) # interpreter BF16 stores truncate; validate FP32 arithmetic here
        codec._nvdq[(7,)](cq,cs,g,y,d,triton.next_power_of_2(d))
        assert torch.equal(y,ref.nvfp4_dequant(q,s,g,dtype=torch.float32)),(seed,d,'dequant')
        results.append({'seed':seed,'d':d})
pathlib.Path(a.output).write_text(json.dumps({'status':'PASS','scope':'CPU Triton interpreter: payload/scales/FP32 dequant; BF16 store requires GPU verification','cases':results},indent=2))
print('PASS',len(results),'cases')
