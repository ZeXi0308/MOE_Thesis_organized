"""Bounded CPU check of added raw snapshots; frozen action semantics unchanged."""
import ast
import copy
import hashlib
from pathlib import Path
import runpy
from types import MethodType, SimpleNamespace as NS

import observe_progress as O

path = Path(__file__).with_name('check_cpu.py')
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'da808515e9cbeda7f0b15535ccd58c2d0a615fb9dda4dea2b2b5714bef6ad8a3'
F = runpy.run_path(str(path))
tree = O._instrument(copy.deepcopy(F['F']['full_tree']))
stripped = F['F']['RemoveCallbacks']().visit(copy.deepcopy(tree))
assert ast.dump(stripped, include_attributes=True) == ast.dump(F['F']['full_tree'], include_attributes=True)
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
         and n.func.id == O.G.H.CALLBACK]
assert sum(isinstance(n.args[-1], ast.Name) and n.args[-1].id == 'num_scheduled_tokens' for n in calls) == 2


class Inflight(list):
    def add(self, request):
        if all(r is not request for r in self):
            self.append(request)


def fixture(mode, observed=True):
    s,m,one,cs,head,early,running,events,refs,_,old_undo,slot,_ = F['fixture'](mode)
    old_undo()
    fragment = copy.deepcopy(F['fragment'])
    # Supply an actual function local. It deliberately differs from C and I.
    fragment.body[0].body[:0] = ast.parse('num_scheduled_tokens = self.cpu_scheduled_tokens').body
    ast.fix_missing_locations(fragment)
    running.num_tokens=32; running.num_computed_tokens=31
    s.cpu_scheduled_tokens={'running':1}
    s._inflight_prefills=Inflight([running])  # Same physical owner in both sets.
    s.requests={r.request_id:r for r in (head,early,running)}
    for r in s.requests.values():
        r.num_output_placeholders=0; r.spec_token_ids=[]
    one.req_to_blocks['running']=[NS(block_id=100+i,is_null=False,ref_cnt=1) for i in range(2)]
    old=slot.cell_contents; env=dict(old.__func__.__globals__)
    exec(compile(fragment,str(F['F']['F']['SCHEDULER']),'exec',dont_inherit=True),env)
    native=MethodType(env['schedule'],s)
    def outer():
        def schedule(*args,**kwargs):return native(*args,**kwargs)
        return schedule
    s.schedule=outer(); slot=O.G.H._native_slot(s.schedule,s); original=slot.cell_contents
    module=O.G if observed else F['G']
    data,undo=module._attach(s,m,one,cs,mode,{},fragment)
    return s,m,one,cs,head,early,running,events,refs,data,undo,slot,original


def action_signature(data):
    return (data['action_count'],data['requested_breaks'],data['executed_breaks'],
            [(e['kind'],e.get('step'),e.get('reason'),e.get('ordinal')) for e in data['events']])


for mode in ('native','wait_release'):
    pairs=[fixture(mode,observed) for observed in (False,True)]
    outputs=[]
    for s,m,one,cs,head,early,running,events,refs,data,undo,slot,original in pairs:
        out=[s.schedule(16,[])]
        if mode=='wait_release':
            # Completion is only a release signal. Its held blocks stay at 2;
            # the first target preallocate after release must still be sampled.
            running.status=NS(name='FINISHED_LENGTH_CAPPED');s.running=[];s._inflight_prefills=Inflight()
            s.cpu_scheduled_tokens={}
            out.append(s.schedule(16,[]))
        outputs.append((out,refs[:],action_signature(data)))
        if 'progress_observation' in data:
            obs=data['progress_observation'];snapshots=obs['snapshots']
            assert len(snapshots)==(1 if mode=='native' else 2)
            first=snapshots[0];assert first['status']=='KNOWN'
            rows=[dict(zip(first['columns'],r)) for r in first['requests']]
            assert len(rows)==1 and rows[0]['is_running'] and rows[0]['is_inflight']
            assert rows[0]['scheduled_tokens_this_step']==1 and rows[0]['computed_tokens']==31
            assert first['free_gpu_blocks']==300 and first['native_reserved_blocks']==0
            assert first['snapshot_s']>=0 and first['allocation_has_not_executed']
            last=snapshots[-1]
            assert last['native_continuation_permitted'] and not last['original_break_return']
            assert obs['outcome']=='NATIVE_ATTEMPT_BOUNDARY_OBSERVED'
            assert 'capacity_fit' not in last and 'fit' not in last
            if mode=='wait_release':
                assert last['step']==2 and last['requests']==[]
                assert any(e.get('reason')=='COHORT_REQUEST_FINISHED' for e in data['events'])
        undo();undo();assert slot.cell_contents is original
    assert outputs[0]==outputs[1]

# Full bound includes a final snapshot when the frozen callback returns False
# silently after entry has released the gate.
s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture('wait_release')
for _ in range(17):s.schedule(16,[])
obs=data['progress_observation']
assert data['executed_breaks']==16 and len(obs['snapshots'])==17
assert obs['snapshots'][-1]['step']==17 and obs['snapshots'][-1]['native_continuation_permitted']
undo();assert slot.cell_contents is old

# An entry can release while the native preemption gate prevents reaching the
# target until a later entry. Snapshot step must not reuse the release's step.
s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture('wait_release')
s.schedule(16,[])
running.status=NS(name='FINISHED_LENGTH_CAPPED');s.running=[];s._inflight_prefills=Inflight()
s.cpu_scheduled_tokens={}
s.schedule(16,[running])
assert next(e for e in data['events'] if e['kind']=='release')['step']==2
assert len(data['progress_observation']['snapshots'])==1
s.schedule(16,[])
assert data['progress_observation']['snapshots'][-1]['step']==3
undo();assert slot.cell_contents is old

# Unsupported ownership is explicit UNKNOWN, without changing the policy's
# action or swallowing a subsequent native exception.
s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture('wait_release')
one.req_to_blocks['running'][0].ref_cnt=2
s.schedule(16,[])
assert data['executed_breaks']==1
assert data['progress_observation']['snapshots'][0]['status']=='UNKNOWN'
undo();assert data['progress_observation']['outcome']=='NO_NATIVE_ATTEMPT_BOUNDARY_OBSERVED'
s,m,one,cs,head,early,running,events,refs,data,undo,slot,old=fixture('native')
def fail(*args,**kwargs):raise ValueError('original allocator error')
m.allocate_slots=fail
try:s.schedule(16,[])
except ValueError as error:assert str(error)=='original allocator error'
else:raise AssertionError('Native exception swallowed')
undo();assert slot.cell_contents is old and refs==[0]
print('PASS: added native Q argument/AST reversibility; original actions unchanged; union ownership dedup; first post-release attempt captured; 16+1 bound; finished is not fit; UNKNOWN and uninstall/native errors. CPU only; GPU_UNRUN.')
