import json,argparse,pathlib
import torch
from quality_transform import Policy,owner_ids,transform,qdq
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
# Uneven expert ownership is deliberately different from floor(id*EP/E).
assert owner_ids(torch.arange(10),10,3).tolist()==[0,0,0,0,1,1,1,2,2,2]
torch.manual_seed(17)
x=torch.randn(5,4,32,dtype=torch.bfloat16)
w=torch.tensor([[.1,.4,.2,.3]]).expand(5,-1)
ids=torch.tensor([[0,3,6,9]]).expand(5,-1)
assert transform(x,w,ids,10,Policy()).data_ptr()==x.data_ptr()
# Gate order is not position order. Exactly slot 1 must get the FP8 lane.
y=transform(x,w,ids,10,Policy(precision='tiered',head=1))
torch.testing.assert_close(y[:,1],qdq(x[:,1],'fp8'),rtol=0,atol=0)
torch.testing.assert_close(y[:,[0,2,3]],qdq(x[:,[0,2,3]],'nvfp4'),rtol=0,atol=0)
grouped=transform(x,w,ids,10,Policy(mode='II',ep_size=3))
expected=torch.stack([(x[:,0].float()+x[:,1].float()).to(x.dtype),x[:,2],x[:,3]],dim=1)
assert torch.equal(grouped,expected)
for mode in ['I','II']:
 for prec in ['bf16','fp8','nvfp4','tiered']:
  y=transform(x,w,ids,10,Policy(mode=mode,precision=prec,ep_size=3))
  assert torch.isfinite(y).all()
pathlib.Path(a.output).write_text(json.dumps({'status':'PASS','scope':'CPU transform semantics only; vLLM runtime hook unverified','checks':['uneven vLLM linear mapping','identity mode I BF16','unsorted routing weights','independent grouped sums','all 8 mode/precision combinations']},indent=2))
print('PASS')
