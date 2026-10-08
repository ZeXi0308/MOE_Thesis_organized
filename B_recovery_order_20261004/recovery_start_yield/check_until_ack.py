"""Bounded real native-fragment checks for the two-entry ACK-yield adapter."""
import ast
import hashlib
from pathlib import Path
import runpy
from types import SimpleNamespace as NS

import yield_until_ack as A

B=Path(__file__).resolve().parents[1]
fixture_path=Path(__file__).with_name('check_cpu.py')
assert hashlib.sha256(fixture_path.read_bytes()).hexdigest()=='d9bebc87905129401c357869547699de3524610e484732f4925ec336e442979c'
F=runpy.run_path(str(fixture_path))  # Includes full AST removal equality.
fragment=F['fragment']; Queue=F['Queue']
source=B/'native_sources/offloading_scheduler_20261007.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='89ac26a80fbc29b9bcaa5a0daba88fb6309247f9bb053e48d2d61efa7d9d66f1'
method=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='update_connector_output')
loop=next(n for n in method.body if isinstance(n,ast.For) and ast.unparse(n.iter)=='meta.completed_jobs.items()')
ack_tree=ast.parse('def native_ack(self,meta):\n    pass\n');ack_tree.body[0].body=[loop]
ast.fix_missing_locations(ack_tree);namespace={}
exec(compile(ack_tree,str(source),'exec'),namespace)
native_ack=namespace['native_ack']


def fixture(mode='yield_ack'):
    s,m,one,cs,head,early,events,refs=F['fixture']()
    cs._stale_job_threshold=0;cs._block_id_to_pending_jobs={}
    early.is_finished=lambda:False
    cs._req_status['early'].req_context=object()
    def complete_load(keys,context):
        assert keys=={'prefix'} and context is cs._req_status['early'].req_context
        refs[0]-=1
    cs.manager.complete_load=complete_load
    slot=A.H._native_slot(s.schedule,s);old=slot.cell_contents
    ticks=iter(range(100,300))
    data,undo=A._attach(s,m,one,cs,mode,{},fragment,now=lambda:next(ticks))
    return s,m,one,cs,head,early,events,refs,data,undo,slot,old


def allocate_ids(events): return [e[1] for e in events if e[0]=='allocate']


for mode in ('native','yield_ack'):
    s,m,one,cs,head,early,events,refs,data,undo,slot,old=fixture(mode)
    wrapper=s.schedule
    assert s.schedule(16,[])==[7]
    assert refs==[1] and cs._req_status['early'].transfer_jobs=={7}
    assert allocate_ids(events)==(['early','head'] if mode=='native' else ['early'])
    assert data['executed_breaks']==int(mode=='yield_ack')
    assert s.schedule(16,[])==[]
    assert refs==[1]
    assert allocate_ids(events)==(['early','head'] if mode=='native' else ['early'])
    assert data['executed_breaks']==(0 if mode=='native' else 2)
    assert s.schedule(16,[])==[]
    assert allocate_ids(events)==['early','head'] and refs==[1]
    assert data['action_count']==int(mode=='yield_ack')
    assert data['requested_breaks']==data['executed_breaks']
    if mode=='yield_ack':
        assert [e['step'] for e in data['events'] if e['kind']=='actual_break_executed']==[1,2]
        assert data['events'][-1]['reason']=='TWO_ROUND_LIMIT'
    else:
        assert not any(e['kind']=='actual_break_requested' for e in data['events'])
    s.schedule(16,[])
    assert data['executed_breaks']==(0 if mode=='native' else 2)
    undo();undo()
    assert slot.cell_contents is old and s.schedule is wrapper and data['status']=='UNINSTALLED'

# Execute the actual native complete_load/deletion block before the next entry.
s,m,one,cs,head,early,events,refs,data,undo,slot,old=fixture()
s.schedule(16,[]);cs._jobs[7].keys={'prefix'}
native_ack(cs,NS(completed_jobs={7:1}))
assert refs==[0] and 7 not in cs._jobs and not cs._req_status['early'].transfer_jobs
s.schedule(16,[])
assert allocate_ids(events)==['early','head'] and data['executed_breaks']==1
assert next(e for e in data['events'] if e['kind']=='ack_entry_observation')['all_original_loads_ack_retired']
assert data['events'][-1]['reason']=='ALL_ORIGINAL_LOADS_ACK_RETIRED'
undo()

# Another native head may progress, then this same target can consume the second
# break in the same next schedule round; no queue mutation by the adapter.
s,m,one,cs,head,early,events,refs,data,undo,slot,old=fixture()
s.schedule(16,[])
other=F['F']['request']('other',0.5)
s.waiting.prepend_request(other)
cs._req_status['other']=NS(req=other,transfer_jobs=set(),group_states=[NS(offload_keys=list(range(4)))])
s.schedule(32,[])
assert allocate_ids(events)==['early','other'] and [r.request_id for r in s.waiting]==['head']
assert any(e['kind']=='native_other_head_pass_through' and e['native_head']=='other' for e in data['events'])
assert data['executed_breaks']==2 and refs==[1]
s.schedule(16,[])
assert allocate_ids(events)==['early','other','head'] and refs==[1]
undo()

# Reset, inconsistent retirement, new WAITING and target ownership changes must
# never masquerade as a successful scheduler ACK or justify another break.
for change,reason in (
    (lambda s,m,one,cs,head:setattr(cs,'_stale_job_threshold',20),'UNKNOWN_RESET'),
    (lambda s,m,one,cs,head:cs._jobs.clear(),'UNKNOWN_JOB_STATE'),
    (lambda s,m,one,cs,head:setattr(head,'status',NS(name='WAITING')),'NEW_WAITER_PRESENT'),
    (lambda s,m,one,cs,head:setattr(head,'num_computed_tokens',1),'TARGET_NOT_CLEAN_RECOVERY')):
    s,m,one,cs,head,early,events,refs,data,undo,slot,old=fixture()
    s.schedule(16,[]);change(s,m,one,cs,head)
    # Observe the real schedule-entry callback directly; this check deliberately
    # does not ask the fake allocator to execute a corrupt/reset fixture state.
    callback=slot.cell_contents.__func__.__globals__[A.H.CALLBACK]
    callback(s,'entry',16,[])
    assert data['events'][-1]['reason']==reason and data['executed_breaks']==1 and refs==[1]
    if reason.startswith('UNKNOWN'):
        assert not data['events'][-2]['all_original_loads_ack_retired']
    undo()

# Native exceptions still propagate; uninstall restores the original closure.
s,m,one,cs,head,early,events,refs,data,undo,slot,old=fixture()
def fail(*args,**kwargs): raise ValueError('native allocator failed')
m.allocate_slots=fail
try:s.schedule(16,[])
except ValueError as error:assert str(error)=='native allocator failed'
else:raise AssertionError('Swallowed native error')
undo();assert slot.cell_contents is old and refs==[0]
print('PASS: unchanged native AST/fragments; two-round maximum; actual native ACK retirement; other-head pass and native merge; reset/ownership safety; references unchanged; errors and uninstall. GPU_UNRUN')
