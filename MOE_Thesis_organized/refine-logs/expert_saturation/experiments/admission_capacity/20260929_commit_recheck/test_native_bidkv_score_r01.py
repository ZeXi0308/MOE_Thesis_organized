"""Compare adapted scores and ties against the pinned official implementation."""
import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'candidate_native_bidkv_score_r01/pkg'))
from bidkv_score import bidkv_score, bidkv_order

source = (ROOT/'bidkv_pinned_sources_20261002/selector.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == '3ad8af893f3693a5ffbe3b9367153cca14fed50fb1b195cdbab924b1e5849a99'
tree = ast.parse(source)
record = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UtilityCandidateScore')
original = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BidkvVictimSelector')
names = {'_rank_candidates','_score_request','_compute_utility','_compute_completion','_output_tokens'}
original.body = [n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in names]
module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),record,original],type_ignores=[]))
env = dict(dataclass=dataclass)
exec(compile(module,'<five reviewed methods from pinned BidKV>','exec'),env)
ref = env['BidkvVictimSelector']()
ref.config=NS(utility_completion_weight=.5,utility_preempt_weight=.3,utility_epsilon=1e-6,utility_default_max_tokens=1024)
requests=[]
for computed in [-1,0,151,3072]:
    for output in [0,26,588,1024,1200]:
        for maximum in [None,0,1024,2048]:
            for preemptions in [0,1,4]:
                req=NS(request_id=str(len(requests)),num_computed_tokens=computed,
                       num_output_tokens=output,max_tokens=maximum,
                       num_preemptions=preemptions,arrival_time=float(len(requests)%3))
                expected=ref._score_request(req,req.request_id)
                actual=bidkv_score(req)
                assert actual==dict(utility=expected.utility,computed_tokens=expected.tokens_freed,
                                    completion=expected.completion,num_preemptions=expected.num_preemptions)
                requests.append(req)
# A real output-token list takes precedence over the fallback count.
req=NS(request_id='ids',num_computed_tokens=999,output_token_ids=[1,2,3],
       num_output_tokens=500,max_tokens=1024,num_preemptions=2,arrival_time=0)
assert bidkv_score(req)['utility']==ref._score_request(req,'ids').utility
requests.append(req)
official,_=ref._rank_candidates(requests)
adapted=sorted([dict(request=r.request_id,arrival_time=r.arrival_time,bidkv=bidkv_score(r)) for r in requests],key=bidkv_order)
assert [c.request_id for c in official]==[c['request'] for c in adapted]
print('PASS:',len(requests),'official score/tie comparisons; same-input baseline GPU UNRUN')
