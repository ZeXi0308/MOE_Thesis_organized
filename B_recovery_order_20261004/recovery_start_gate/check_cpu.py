"""CPU-only native allocator/waiting AST checks for the bounded startup gate."""
import ast
import copy
import hashlib
from pathlib import Path
import runpy
import sys
from types import MethodType,SimpleNamespace as NS

import start_gate as G

B=Path(__file__).resolve().parents[1]
for relative,digest in (
    ('recovery_start_yield/check_cpu.py','d9bebc87905129401c357869547699de3524610e484732f4925ec336e442979c'),
    ('tail_reservation/check_cpu.py','f51a0470879bf09fa363d6e06f4fa71ad3e098f5fcf2f1cf8c2537d70ed4808c')):
    assert hashlib.sha256((B/relative).read_bytes()).hexdigest()==digest
sys.path.insert(0,str(B/'recovery_start_yield'))
F=runpy.run_path(str(B/'recovery_start_yield/check_cpu.py'))
sys.path.insert(0,str(B/'tail_reservation'))
T=runpy.run_path(str(B/'tail_reservation/check_cpu.py'))
tree=G._instrument(F['full_tree'])
assert ast.dump(F['RemoveCallbacks']().visit(copy.deepcopy(tree)),include_attributes=True)==ast.dump(F['full_tree'],include_attributes=True)
compile(tree,str(F['F']['SCHEDULER']),'exec',dont_inherit=True)

# Use real pinned capacity-query/allocator methods and the previous exact native
# waiting/async/metadata fixture. Only its model-specific empty-KV stub changes.
fragment=copy.deepcopy(F['fragment'])
matches=0
for n in ast.walk(fragment):
    if isinstance(n,ast.Assign) and ast.unparse(n)=='new_computed_blocks = ()':
        n.value=ast.copy_location(ast.parse('self.kv_cache_manager.empty_kv_cache_blocks',mode='eval').body,n.value)
        matches+=1
assert matches==1
ast.fix_missing_locations(fragment)


def fixture(mode='wait_release',free=300):
    s,_,_,cs,head,early,events,refs=F['fixture']()
    m,one,_=T['fixture'](free);s.kv_cache_manager=m
    cs._stale_job_threshold=0
    for r in (head,early):r.num_prompt_tokens=64
    running=F['F']['request']('running',-1)
    running.status=NS(name='RUNNING');running.is_finished=lambda:running.status.name.startswith('FINISHED')
    s.running=[running];one.req_to_blocks['running']=[NS(is_null=False,ref_cnt=1)]*2
    old=G.H._native_slot(s.schedule,s).cell_contents
    env=dict(old.__func__.__globals__)
    exec(compile(fragment,str(F['F']['SCHEDULER']),'exec',dont_inherit=True),env)
    native=MethodType(env['schedule'],s)
    def wrapper():
        def schedule(*args,**kwargs):return native(*args,**kwargs)
        return schedule
    s.schedule=wrapper();slot=G.H._native_slot(s.schedule,s);original=slot.cell_contents
    data,undo=G._attach(s,m,one,cs,mode,{},fragment)
    return s,m,one,cs,head,early,running,events,refs,data,undo,slot,original


# Exact native legality is max(full_need, slot_need+reserve), not full+reserve.
for free,fit in ((3,False),(4,False),(5,True)):
    s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture(free=free)
    before=(dict(one.req_to_blocks),dict(one.num_cached_block),m.block_pool.get_num_free_blocks())
    row,reason=G._fit(m,one,cs,early,0,32,True,0,0,0,m.empty_kv_cache_blocks,3)
    assert row['full_fit_blocks']==4 and row['slot_blocks']==2 and row['required_free_blocks']==5
    assert (reason is None)==fit and before==(dict(one.req_to_blocks),dict(one.num_cached_block),m.block_pool.get_num_free_blocks())
    assert not m.block_pool.calls and not m.coordinator.skipped_calls and refs==[0]
    result=m.allocate_slots(early,0,num_external_computed_tokens=32,delay_cache_blocks=True,
                            full_sequence_must_fit=True,reserved_blocks=3)
    assert (result is not None)==fit
    undo()

# Exactly 16 actual breaks including the selected entry; entry17 runs native.
s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture()
for _ in range(16):
    assert s.schedule(16,[])==[]
    assert not m.block_pool.calls and refs==[0] and not cs._jobs
assert data['executed_breaks']==data['requested_breaks']==16 and data['action_count']==1
assert s.schedule(16,[])==[7]
assert data['events'][-1]['reason']=='BLOCK_EPOCH_LIMIT' and refs==[1]
assert data['executed_breaks']==16 and len(one.req_to_blocks['early'])==2
undo();undo();assert slot.cell_contents is old

# A genuinely finished original cohort object releases even if its held blocks
# remain visible; merely leaving RUNNING or being preempted does not qualify.
for status,finished in (('PREEMPTED',False),('RUNNING',False),('FINISHED_LENGTH_CAPPED',True)):
    s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture()
    s.schedule(16,[]);s.running=[];running.status=NS(name=status)
    s.schedule(16,[])
    if finished:
        release=data['events'][-1]
        assert release['reason']=='COHORT_REQUEST_FINISHED' and release['cohort_finished'][0]['held_gpu_blocks']==2
        assert release['free_gpu_blocks']==300 and data['executed_breaks']==1 and refs==[1]
    else:
        assert data['executed_breaks']==2 and refs==[0]
    undo()

# Other native heads progress; the same selected target may then break. No extra
# lookup/allocation/CPU LOAD reference is introduced by the observer.
s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture()
s.schedule(16,[])
other=F['F']['request']('other',-0.5);other.num_prompt_tokens=64
s.waiting.prepend_request(other)
s.schedule(32,[])
assert s.running[-1] is other and s.waiting.peek_request() is early and refs==[0]
assert data['executed_breaks']==2 and any(e['kind']=='native_other_head_pass_through' for e in data['events'])
undo()

# Native mode is shadow only and its original allocation/ref path runs once.
s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture('native')
assert s.schedule(16,[])==[7] and refs==[1]
assert data['action_count']==data['executed_breaks']==data['requested_breaks']==0
s.schedule(16,[])
assert data['events'][-1]['kind']=='release'
undo();assert slot.cell_contents is old

for change,reason in (
    (lambda s,m,one,cs,head,early:setattr(cs,'_stale_job_threshold',99),'CACHE_RESET'),
    (lambda s,m,one,cs,head,early:setattr(head,'status',NS(name='WAITING')),'NEW_WAITER_PRESENT'),
    (lambda s,m,one,cs,head,early:cs._req_status['early'].transfer_jobs.add(9),'TARGET_NO_LONGER_UNPREPARED')):
    s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture()
    s.schedule(16,[]);change(s,m,one,cs,head,early)
    callback=slot.cell_contents.__func__.__globals__[G.H.CALLBACK]
    callback(s,'entry',16,[])
    assert data['events'][-1]['reason']==reason and data['executed_breaks']==1 and refs==[0]
    undo()

# Native allocator errors are not swallowed; uninstall restores its closure.
s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture('native')
def fail(*args,**kwargs):raise ValueError('native allocator error')
m.allocate_slots=fail
try:s.schedule(16,[])
except ValueError as error:assert str(error)=='native allocator error'
else:raise AssertionError('Native exception swallowed')
undo();assert slot.cell_contents is old and refs==[0]
print('PASS: pinned native read-only fit agrees with actual allocator; AST reversible; 16-entry bound; true cohort finish vs preemption/removal; native merge/pass-through; no early LOAD ownership; safety release, native errors and uninstall. GPU_UNRUN')
