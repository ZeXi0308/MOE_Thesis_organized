"""Only the new failure latch/next-begin lifetime; no model or GPU."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import exchange_after_failure as policy

B = Path(__file__).resolve().parents[1]
path = B/'recovery_capacity_exchange/check_cpu.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '7624c04786a48e2e4d610d76b17ea665784c5bcca728031b07e6aee0c187dc08'
tree = ast.parse(path.read_text())
# Reuse fixture definitions only; never rerun the old runtime/copy tests.
end = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=='original_clock' for t in n.targets))
tree.body = tree.body[:end]
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCHED' for t in n.targets):
        n.value = ast.parse("Path(os.environ['B_GUARD_TEST_SCHEDULER_SOURCE']) if 'B_GUARD_TEST_SCHEDULER_SOURCE' in os.environ else "+ast.unparse(n.value),mode='eval').body
    if isinstance(n,ast.FunctionDef) and n.name=='fixture':
        ret = next(x for x in n.body if isinstance(x,ast.Return))
        ret.value.keywords.extend([ast.keyword(arg='manager',value=ast.Name(id='manager',ctx=ast.Load())),
                                   ast.keyword(arg='single',value=ast.Name(id='single',ctx=ast.Load()))])
ns = dict(__file__=str(path),os=os)
exec(compile(ast.fix_missing_locations(tree),str(path)+'[fixture-only]','exec'),ns)
obs_path = B/'capacity_handoff/observe_tail.py'
assert hashlib.sha256(obs_path.read_bytes()).hexdigest() == policy.OBSERVER_SHA
spec=importlib.util.spec_from_file_location('failure_real_observer',obs_path)
observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)


def fake_qualified_install(scheduler,mode='stall8',*,last_receipts,selective):
    # Production qualification and runtime guard are frozen/r03-executed. The
    # new install wiring is exercised, with only these prior checks doubled.
    return _attach(scheduler,scheduler.kv_cache_manager,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0],
        scheduler.connector.connector_scheduler,mode,last_receipts,
        scheduler.fixture_baseline,lambda:scheduler.fixture_baseline)


def fixture(mode='exchange_once'):
    c=ns['fixture']();c.undo()
    old=c.t.request_id;rid='measured/'+policy.CONTROLLED_SOURCE_REQUEST+'-cpu'
    c.t.request_id=rid
    for mapping in (c.s.requests,c.owned,c.cs._req_status,c.receipts):mapping[rid]=mapping.pop(old)
    c.manager.coordinator=NS(single_type_managers=[c.single])
    c.s.kv_cache_manager=c.manager;c.s.connector=NS(connector_scheduler=c.cs)
    c.s.fixture_baseline=c.baseline
    c.alloc_calls=0;c.alloc_result=None;c.alloc_error=None
    def allocator(*args,**kwargs):
        c.alloc_calls+=1
        if c.alloc_error:raise c.alloc_error
        return c.alloc_result
    c.manager.allocate_slots=allocator
    observation,c.undo_observer=observer._install(c.s,dict(exact_layout=True,block_size=16))
    c.t.status=ns['STATUS'].RUNNING;c.s.waiting.remove_request(c.t)
    c.s._preempt_request(c.t,10.)  # Enter the real observer's recovering population.
    c.original_allocate=c.manager.allocate_slots
    with patch.object(policy.FROZEN,'install',fake_qualified_install),patch.object(policy.RUNTIME,'_attach_guard',lambda *args:({},lambda:None)):
        c.data,c.undo=policy.install(c.s,mode,last_receipts=c.receipts,selective={},allocation_observation=observation)
    c.observation=observation
    return c


def fail(c, **overrides):
    args=dict(num_new_tokens=0,num_external_computed_tokens=64,
        delay_cache_blocks=True,full_sequence_must_fit=True)
    args.update(overrides)
    return c.manager.allocate_slots(c.t,**args)


def close(c):
    c.undo();c.undo();assert c.manager.allocate_slots is c.original_allocate
    assert c.data['failure_trigger']['status']=='UNINSTALLED'
    c.undo_observer()


with patch.object(policy.FROZEN.time,'perf_counter',lambda:100.):
    for mode in ('stall8','exchange_once'):
        c=fixture(mode);c.s.schedule()
        assert c.data['action_count']==0 and c.data['failure_trigger']['first_failure'] is None
        assert fail(c) is None and c.alloc_calls==1
        record=c.data['failure_trigger']['first_failure']
        assert record['latched'] and record['allocation_observation'] is c.observation['events'][-1]
        # Change only present peer growth after the failure: donor is recalculated.
        c.peer.num_tokens=65;c.peer.num_tokens_with_spec=65;c.peer.num_computed_tokens=64
        c.s.schedule()
        decision=next(e for e in c.data['events'] if e['kind']=='decision')
        assert decision['donor']=='larger' and decision['target']==c.t.request_id
        assert decision['step']==record['failure_step']+1 and decision['trigger_failure'] is record
        assert c.data['action_count']==(1 if mode=='exchange_once' else 0)
        if mode=='exchange_once':
            assert c.s._rotation_forced_count==1
            c.receipts[c.t.request_id]=99.;c.s.schedule()
            assert next(e for e in c.data['events'] if e['kind']=='release')['reason']=='FIRST_CLIENT_RECEIPT'
        for _ in range(2):c.s.schedule()
        assert len([e for e in c.data['events'] if e['kind']=='decision'])==1
        close(c)

    for condition in ('new_receipt','new_episode','new_object','new_waiter','priority','pending_load','no_donor'):
        c=fixture();c.s.schedule();fail(c)
        if condition=='new_receipt':c.receipts[c.t.request_id]=99.
        if condition=='new_episode':c.t.num_preemptions+=1
        if condition=='new_object':c.s.requests[c.t.request_id]=NS()
        if condition=='new_waiter':
            r=ns['Request']('new',16,0,ns['STATUS'].WAITING,2.);c.s.waiting.append(r)
        if condition=='priority':
            r=ns['Request']('barrier',16,0,ns['STATUS'].PREEMPTED,2.);r.priority=1;c.s.waiting.insert(0,r)
        if condition=='pending_load':
            c.cs._req_status[c.t.request_id].transfer_jobs.add(91);c.cs._jobs[91]=NS(req_id=c.t.request_id,is_store=False)
        if condition=='no_donor':c.pool.free=0;c.pool.free_block_queue.num_free_blocks=0
        c.s.schedule();assert c.data['action_count']==0,condition
        # A later otherwise usable failure never rearms this one-shot trigger.
        c.pool.free=4;c.pool.free_block_queue.num_free_blocks=4
        c.s.schedule();assert c.data['action_count']==0;close(c)

    c=fixture();c.s.schedule();c.alloc_result=object();assert fail(c) is c.alloc_result
    assert c.data['failure_trigger']['first_failure'] is None
    c.alloc_result=None;fail(c,num_external_computed_tokens=0,delay_cache_blocks=False,num_new_tokens=160)
    assert c.data['failure_trigger']['first_failure'] is None
    c.alloc_error=ValueError('native allocation error')
    try:fail(c)
    except ValueError as error:assert str(error)=='native allocation error'
    else:raise AssertionError('Native error swallowed')
    assert c.data['failure_trigger']['first_failure'] is None;close(c)

    # Q1/16-entry bound stays the original exchange policy's bound.
    c=fixture();c.s.schedule();fail(c);c.s.schedule()
    for _ in range(15):c.s.schedule()
    assert not any(e['kind']=='release' for e in c.data['events'])
    c.s.schedule();assert c.data['events'][-1]['reason']=='SIXTEEN_ROUND_LIMIT';close(c)
print('PASS: real allocation observer failure latch, exact next begin, fresh donor arithmetic, same-object/episode/receipt, priority/new WAITING/pending guards, baseline shadow, exact-once native call, failure propagation, Q1/16-entry/uninstall. CPU only; controlled development ID; GPU_UNRUN.')
