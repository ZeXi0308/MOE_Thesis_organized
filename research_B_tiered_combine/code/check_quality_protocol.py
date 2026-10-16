"""Independent token accounting, payload arithmetic, and rank-share checks."""
import argparse,json,pathlib
import torch
from quality_protocol import token_windows,default_arms,validate_arms,payload_bytes
from rank_profile import RankProfile

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
for count in [2,3,511,512,513,1023,1024,2053]:
    tokens=list(range(count));windows=list(token_windows(tokens,512))
    assert [token for window in windows for token in window['token_ids'][1:]]==tokens[1:]
    assert all(2<=len(window['token_ids'])<=512 for window in windows)
assert list(token_windows([1],512))==[]
for k in [8,16]:validate_arms(default_arms(k))
try:validate_arms([{'name':'candidate','baseline':'missing'}])
except ValueError:pass
else:raise AssertionError('Missing baseline accepted')
# Independently enumerated 2 tokens, 8 units, d=32: 8 head rows and 8 tail rows.
policy={'mode':'I','precision':'tiered','head':4}
assert payload_bytes(policy,2,32,8)==8*(32+4)+8*(16+2)+4
observer=RankProfile()
weighted=torch.tensor([[[3.,4.],[0.,2.],[0.,0.]],[[0.,1.],[0.,3.],[0.,0.]]])
gates=torch.tensor([[.2,.7,.1],[.5,.4,.1]])
ids=torch.tensor([[2,0,1],[0,2,1]])
observer('layer',weighted,gates,ids,3)
observer('layer',weighted,gates,ids,3)
r=observer.finish()['layer']
assert r['tokens']==4 and r['calls']==2 and r['zero_norm_tokens']==0
# First token gate order gives norm [2,5,0], second gives [1,3,0].
torch.testing.assert_close(torch.tensor(r['norm_sum_by_gate_rank']),torch.tensor([6.,16.,0.]))
torch.testing.assert_close(torch.tensor(r['norm_share_sum_by_gate_rank']),2*torch.tensor([2/7+1/4,5/7+3/4,0.]))
torch.testing.assert_close(torch.tensor(r['energy_share_sum_by_gate_rank']),2*torch.tensor([4/29+1/10,25/29+9/10,0.]))
pathlib.Path(a.output).write_text(json.dumps({'status':'PASS',
    'scope':'CPU token accounting, rank-share definition, logical payload arithmetic; no worker/GPU validation',
    'profile_example':r},indent=2))
print('PASS')
