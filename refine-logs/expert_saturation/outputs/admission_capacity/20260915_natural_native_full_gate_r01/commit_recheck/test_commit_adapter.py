"""CPU component tests of candidate's actual begin/schedule function bodies.

AST changes only closure bindings to globals. Native execution is a stub; this
does not qualify vLLM installation, GPU jobs or asynchronous connector behavior.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import dataclass
from commit_disposition import choose_commit_action


@dataclass
class View:
    request_id: str
    pure_decode: bool = True
    remaining_blocks: int = 2
    status: str = "RUNNING"


source=Path(__file__).with_name('staged_store_rotation_candidate.py').read_text()
tree=ast.parse(source)
install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install')
functions=[n for n in install.body if isinstance(n,ast.FunctionDef) and n.name in ('begin','schedule')]
class BindGlobals(ast.NodeTransformer):
    def visit_Nonlocal(self,node):
        return ast.copy_location(ast.Global(names=node.names),node)
code=compile(ast.fix_missing_locations(BindGlobals().visit(ast.Module(body=functions,type_ignores=[]))),'<actual-adapter-bodies>','exec')


def run(enabled=True,free=5,reason='READY',inflight_load=False):
    victim=NS(request_id='victim',num_output_tokens=10)
    target=NS(request_id='target',num_output_tokens=4,status=NS(name='PREEMPTED'),is_finished=lambda:False)
    class Queue(list):
        def remove_request(self,r):self.remove(r)
        def prepend_request(self,r):self.insert(0,r)
    preempt_calls=[];rotations=[];remaining={'target':2}
    scheduler=NS(requests={'victim':victim,'target':target},running=[victim],
        waiting=Queue([target]),skipped_waiting=[],_preempt_request=lambda r,t:preempt_calls.append(r.request_id))
    tracker=NS(note_rotation_applied=lambda:rotations.append(True),note_preempted=lambda *a:None,note_resumed=lambda *a:None)
    pool=NS(get_num_free_blocks=lambda:free)
    n=dict(scheduler=scheduler,tracker=tracker,pool=pool,
        view=lambda r:View(r.request_id,remaining_blocks=remaining.get(r.request_id,0)),
        commit_reason=lambda *a,**k:reason,choose_commit_action=choose_commit_action,
        recheck_commit_funding=enabled,open_population=False,cohort={'victim','target'},diagnostic=False,
        expected_requests=32,step=1,phase=None,pending_flush=set(),cancelled=False,
        plan=NS(victim=NS(request_id='victim'),target=NS(request_id='target')),
        protected=None,output_start=None,save=True,store_scope='native_full',
        cs=NS(_req_status={'victim':NS(transfer_jobs={7})},_jobs={7:NS(is_store=not inflight_load)}),
        data={'events':[],'applied_rotations':0,'direct_resumes':0,'status':'INSTALLED'})
    exec(code,n)
    def native():
        preempted=[];n['begin'](preempted,0)
        if n['protected'] is not None:
            remaining['target']=0;target.status.name='RUNNING'
        meta=NS(store_jobs={},load_jobs={},jobs_to_flush=set(n['pending_flush']))
        return NS(kv_connector_metadata=meta,preempted_req_ids={r.request_id for r in preempted},
            num_scheduled_tokens={'target':1} if n['protected'] else {},
            scheduled_cached_reqs=NS(resumed_req_ids=['target'] if n['protected'] else []))
    n['native']=native
    result=n['schedule']()
    return n,scheduler,preempt_calls,rotations,result


n,s,p,r,o=run()
assert not p and not r and len(s.running)==1
assert n['data']['direct_resumes']==1 and n['data']['applied_rotations']==0
assert n['plan'] is None and n['protected'].request_id=='target'
assert not o.kv_connector_metadata.jobs_to_flush and not o.preempted_req_ids
assert s._rotation_target=='target' and s._rotation_forced_count==0
for kwargs in ({'enabled':False},{'free':1}):
    n,s,p,r,o=run(**kwargs)
    assert p==['victim'] and len(r)==1 and o.preempted_req_ids=={'victim'}
    assert o.kv_connector_metadata.jobs_to_flush=={7}
    assert n['data']['applied_rotations']==1 and n['data']['direct_resumes']==0
    assert n['plan'] is None
n,s,p,r,o=run(reason='CANCEL_TARGET_CHANGED')
assert not p and not r and n['protected'] is None and n['plan'] is None
try:
    run(inflight_load=True)
except RuntimeError as e:
    assert 'in-flight load' in str(e)
else:
    raise AssertionError('pending victim load accepted')
print('PASS: direct retention/promotion/counts/plan cleanup; default-off and insufficient funding swap+flush; cancellation and in-flight-load rejection.')
