"""Exact frozen return-wrapper regression; no runtime/transfer reimplementation."""
import ast
import copy
import hashlib
from pathlib import Path
from types import FunctionType, SimpleNamespace as NS

import exchange_guarded as guard

PATH=Path(__file__).resolve().parents[1]/'pkg/staged_store_rotation.py'
SOURCE=PATH.read_bytes()
assert hashlib.sha256(SOURCE).hexdigest()==guard.STAGED_SHA
tree=ast.parse(SOURCE);modified=guard._instrument(tree)


class RemoveOnePredicate(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_If(self,node):
        self.generic_visit(node)
        if (isinstance(node.test,ast.BoolOp) and len(node.test.values)==2
                and isinstance(node.test.values[1],ast.UnaryOp)
                and isinstance(node.test.values[1].operand,ast.Call)
                and isinstance(node.test.values[1].operand.func,ast.Name)
                and node.test.values[1].operand.func.id==guard.CALLBACK):
            node.test=node.test.values[0];self.count+=1
        return node


remove=RemoveOnePredicate()
restored=remove.visit(copy.deepcopy(modified))
assert remove.count==1 and ast.dump(restored,include_attributes=True)==ast.dump(tree,include_attributes=True)
code=guard._nested_schedule(compile(SOURCE,str(PATH),'exec',dont_inherit=True))


def cell(value):
    def capture():return value
    return capture.__closure__[0]


def fixture(*,patch=True,mode='exchange_once',mutate=None,native_error=False):
    donor=NS(request_id='donor',status=NS(name='PREEMPTED'),num_preemptions=2)
    cs=NS(_req_status={'donor':NS(req=donor)})
    scheduler=NS(_rotation_forced_count=1,requests={'donor':donor},connector=NS(connector_scheduler=cs))
    result=NS(num_scheduled_tokens={},preempted_req_ids={'donor'},
        scheduled_cached_reqs=NS(resumed_req_ids=set()),
        kv_connector_metadata=NS(store_jobs={},load_jobs={71:object()},jobs_to_flush={17}))
    action=dict(kind='action',host_perf_s=101.,donor='donor',native_preempt_called=True,
        forced_count=1,donor_num_preemptions=2,pending_store_jobs=[17])
    exchange=dict(mode=mode,action_count=1,events=[action])
    data=dict(applied_rotations=0,self_preempt_continuations=[],victim_decisions=[],capacity_deferrals=[],
        lease_peer_admissions=[],events=[],oldest_anchor_count=0,oldest_episode_count=0,oldest_retired_count=0)
    calls=[]
    def native(*args,**kwargs):
        calls.append('native_metadata_constructed')
        if native_error:raise ValueError('original native error')
        if mutate:mutate(scheduler,exchange,data,result,bindings)
        return result
    bindings={name:None for name in code.co_freevars}
    bindings.update(scheduler=scheduler,data=data,native=native,lease_enabled=False,
        observe_oldest_result=lambda:None,protected=None,open_population=True,
        recovery_min_outputs=1,phase=None,store_scope='native_full',diagnostic=False,
        lease_step_decode_ids=set(),oldest_admission_mode='native',allow_forced_rotations=False,
        tracker=NS(note_preempted=lambda *args:calls.append('native_preempt_noted'),
                   note_resumed=lambda *args:calls.append('native_resume_noted')),step=972)
    cells={name:cell(value) for name,value in bindings.items()}
    original=FunctionType(code,{'time':NS(perf_counter=lambda:100.)},'schedule',(),
        tuple(cells[name] for name in code.co_freevars))
    # Preserve the real outer closure shape; guard must replace only this slot.
    def outer_factory(old):
        def outer():return old()
        return outer
    outer=outer_factory(original);scheduler.schedule=outer
    if patch:receipt,undo=guard._attach_guard(scheduler,exchange)
    else:receipt,undo=None,lambda:None
    return NS(s=scheduler,result=result,exchange=exchange,data=data,calls=calls,receipt=receipt,
        undo=undo,original=original,outer=outer,cells=cells)


old_clock=guard.time.perf_counter;guard.time.perf_counter=lambda:102.
try:
    # Reproduce the actual defect in the full frozen staged wrapper: metadata
    # exists locally, but the return guard throws before its caller receives it.
    f=fixture(patch=False)
    try:f.s.schedule()
    except RuntimeError as error:assert str(error)==guard.ERROR
    else:raise AssertionError('Failed-run guard no longer reproduced')
    assert f.calls==['native_metadata_constructed','native_preempt_noted','native_resume_noted']
    assert f.data['status']=='ERROR'
    # New predicate returns that very same SchedulerOutput, leaves all counters
    # and metadata identities intact, and keeps the next existing guard active.
    f=fixture();assert f.s.schedule() is f.result
    assert f.receipt['checks'][0]['allowed'] and f.s._rotation_forced_count==1
    assert f.cells['allow_forced_rotations'].cell_contents is False
    assert f.data['applied_rotations']==0 and f.exchange['action_count']==1
    assert f.data['status']=='EXECUTING' and f.cells['step'].cell_contents==973
    f.undo();f.undo();assert f.s.schedule is f.outer
    assert f.outer.__closure__[0].cell_contents is f.original
    try:f.s.schedule()
    except RuntimeError as error:assert str(error)==guard.ERROR
    else:raise AssertionError('Uninstall did not restore original guard')

    cases={
        'wrong donor':lambda s,e,d,r,b:setattr(r,'preempted_req_ids',{'other'}),
        'extra preempt':lambda s,e,d,r,b:setattr(r,'preempted_req_ids',{'donor','other'}),
        'forced2':lambda s,e,d,r,b:setattr(s,'_rotation_forced_count',2),
        'old action':lambda s,e,d,r,b:e['events'][0].update(host_perf_s=99.),
        'future action':lambda s,e,d,r,b:e['events'][0].update(host_perf_s=103.),
        'other rotation':lambda s,e,d,r,b:d.update(applied_rotations=1),
        'missing flush':lambda s,e,d,r,b:setattr(r.kv_connector_metadata,'jobs_to_flush',set()),
        'wrong episode':lambda s,e,d,r,b:setattr(s.requests['donor'],'num_preemptions',3),
        'no action':lambda s,e,d,r,b:e.update(action_count=0),
    }
    for name,mutate in cases.items():
        f=fixture(mutate=mutate)
        try:f.s.schedule()
        except RuntimeError as error:assert str(error)==guard.ERROR,(name,str(error))
        else:raise AssertionError('Overbroad exception: '+name)
        assert not f.receipt['checks'][0]['allowed'];f.undo()
    f=fixture(mode='stall8')
    try:f.s.schedule()
    except RuntimeError as error:assert str(error)==guard.ERROR
    else:raise AssertionError('Baseline unexpectedly authorized')
    f.undo()
    f=fixture(native_error=True)
    try:f.s.schedule()
    except ValueError as error:assert str(error)=='original native error'
    else:raise AssertionError('Native error swallowed')
    assert not f.receipt['checks'];f.undo()
finally:guard.time.perf_counter=old_clock

print('PASS: full frozen staged wrapper reproduces initial metadata-return failure; exact single guard predicate AST delta; one same-call matched donor returns identical output; baseline/foreign/stale/extra/missing-flush actions still reject; counters unchanged; native error and idempotent uninstall preserved. CPU only.')
