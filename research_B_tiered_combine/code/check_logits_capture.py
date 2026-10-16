"""CPU check: compare to independent categorical KL and reject call mismatch."""
import argparse,json,pathlib,torch
from logits_capture import LogitsCapture
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
class Model:
    def compute_logits(self,x):return x
m=Model();c=LogitsCapture(m).install()
x=torch.tensor([[0.,1.,-1.],[.5,-.5,2.]])
y=x+torch.tensor([[.1,-.2,.3],[-.3,.1,.2]])
c.begin(True);m.compute_logits(x);c.finish()
c.begin();m.compute_logits(y);r=c.finish()
expected=torch.distributions.kl_divergence(torch.distributions.Categorical(logits=x),torch.distributions.Categorical(logits=y))
torch.testing.assert_close(torch.tensor(r['token_kl']),expected,atol=1e-6,rtol=1e-5)
c.begin();m.compute_logits(x);identity=c.finish();assert max(abs(z) for z in identity['token_kl'])==0
c.begin()
try:c.finish()
except RuntimeError:pass
else:raise AssertionError('Empty capture accepted')
c.restore();assert torch.equal(m.compute_logits(x),x)
pathlib.Path(a.output).write_text(json.dumps({'status':'PASS','scope':'CPU numerical/call-coverage check; worker integration still unverified','kl':r},indent=2))
print('PASS')
