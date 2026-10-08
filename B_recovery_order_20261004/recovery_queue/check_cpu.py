"""Bounded CPU checks using the pinned native waiting-loop AST; no vLLM/CUDA."""
import ast
import copy
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import observe_waiting as obs

B = Path(__file__).resolve().parents[1]
SCHEDULER = B.parent/'MOE_Thesis_organized/refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/liveness_pinned_sources_20260930/scheduler.py'
assert hashlib.sha256(SCHEDULER.read_bytes()).hexdigest() == '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941'
for relative, digest in obs.PINS.items():
    assert hashlib.sha256((B/relative).read_bytes()).hexdigest() == digest
builder_ast = ast.parse((B/'pkg/rotation_native.py').read_text())
builder = next(n for n in builder_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'patched_schedule_tree')
namespace = {'ast': ast}
exec(compile(ast.Module(body=[builder], type_ignores=[]), '<pinned-builder>', 'exec'), namespace)
full_tree = namespace['patched_schedule_tree'](SCHEDULER.read_text())
instrumented = obs._instrument(full_tree)


class RemoveObservations(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == obs.CALLBACK:
            return None
        return self.generic_visit(node)


assert ast.dump(RemoveObservations().visit(copy.deepcopy(instrumented)), include_attributes=True) == ast.dump(full_tree, include_attributes=True)
# Also compile and compare the complete actual patched native method, not just the fixture.
full_namespace = {'SchedulerOutput': object}
exec(compile(full_tree, str(SCHEDULER), 'exec', dont_inherit=True), full_namespace)
full_native = MethodType(full_namespace['schedule'], NS())
assert obs._same_code(obs._compile(full_tree, full_native).__func__.__code__, full_native.__func__.__code__)

# Execute the exact decision prefix from native schedule (including actual queue/cap
# and patched protection guards), ending before model-specific lookup/allocation.
full_fn = full_tree.body[0]
outer = copy.deepcopy(next(n for n in full_fn.body if isinstance(n, ast.If)
    and ast.unparse(n.test).startswith('len(preempted_reqs) == self._rotation_forced_count')))
loop = next(n for n in outer.body if isinstance(n, ast.While))
outer.body = outer.body[:outer.body.index(loop)+1]
cut = next(i for i,n in enumerate(loop.body) if isinstance(n, ast.Assign)
           and ast.unparse(n) == 'request_id = request.request_id')
loop.body = loop.body[:cut+1] + ast.parse('return request_id').body
fragment = ast.parse('def schedule(self, token_budget, preempted_reqs):\n    pass\n')
fragment.body[0].body = [outer] + ast.parse('return None').body
ast.fix_missing_locations(fragment)


class Queue(list):
    def peek_request(self):
        if getattr(self, 'fail', False): raise ValueError('native queue exception')
        return self[0]


def request(rid, arrival, history=64):
    return NS(request_id=rid, priority=0, arrival_time=arrival, status=NS(name='PREEMPTED'),
              num_tokens=history, num_computed_tokens=0, num_in_flight_tokens=0,
              has_encoder_inputs=False, num_preemptions=1, skip_reading_prefix_cache=False)


def fixture():
    head, target = request('head', 2), request('anchor', 1)
    scheduler = NS(waiting=Queue([head,target]), skipped_waiting=Queue(), running=[],
        num_waiting_for_streaming_input=0, max_num_running_reqs=4,
        _rotation_forced_count=0, _rotation_lease_enabled=False, _rotation_target=None,
        _pause_state=NS(name='UNPAUSED'))
    # The native comparison uses the enum member; our one object serves that role.
    enum = NS(UNPAUSED=scheduler._pause_state)
    scheduler._select_waiting_queue_for_scheduling = lambda: scheduler.skipped_waiting or scheduler.waiting or None
    scheduler._rotation_waiting_queue = lambda q:q
    scheduler._inflight_prefill_reserved_blocks = lambda:0
    single = NS(block_size=16, req_to_blocks={})
    manager = NS(max_model_len=4096, block_pool=NS(free_block_queue=NS(num_free_blocks=8)))
    policy = NS(blocks={i:NS(ref_cnt=0, is_ready=True) for i in range(4)})
    cs = NS(manager=NS(_policy=policy), _chunks_being_loaded=None, _req_status={r.request_id:NS(req=r, transfer_jobs=set(),
        group_states=[NS(offload_keys=list(range(4)))]) for r in [head,target]})
    env = dict(PauseState=enum, create_request_queue=lambda policy:Queue())
    scheduler.policy = 'FCFS'
    exec(compile(fragment, str(SCHEDULER), 'exec', dont_inherit=True), env)
    native = MethodType(env['schedule'], scheduler)
    def existing_wrapper():
        def schedule(*args, **kwargs): return native(*args, **kwargs)
        return schedule
    def observer_wrapper(old):
        def call(*args, **kwargs): return old(*args, **kwargs)
        return call
    scheduler.schedule = observer_wrapper(existing_wrapper())
    source = dict(mode='native', selected_request=None, events=[])
    return scheduler, source, manager, single, cs


s,source,manager,single,cs = fixture()
wrapper = s.schedule
slot = obs._native_slot(wrapper,s); old_native = slot.cell_contents
data, uninstall = obs._attach(s,source,manager,single,cs,fragment)
try: obs._attach(s,source,manager,single,cs,fragment)
except RuntimeError: pass
else: raise AssertionError('Duplicate instrumentation accepted')
assert s.schedule is wrapper and s.schedule(8,[]) == 'head' and not data['events']
source['selected_request']='anchor'; source['events'].append(dict(kind='selected'))
before_queues = (list(s.waiting),list(s.skipped_waiting)); before_host = dict(cs.manager._policy.blocks)
assert s.schedule(8,[]) == 'head'
assert [e['kind'] for e in data['events']] == ['waiting_entry','queue_decision']
row = data['events'][-1]
assert row['native_head']=='head' and row['oldest_arrival_fit']=='anchor' and row['shadow_gate_open']
assert all(c['eligible'] for c in row['candidates'])
assert before_queues == (list(s.waiting),list(s.skipped_waiting)) and before_host == cs.manager._policy.blocks
for budget, preempted, expected in [(0,[],'TOKEN_BUDGET_EMPTY'),(8,[s.waiting[0]],'PREEMPTED_THIS_STEP')]:
    n=len(data['events']); assert s.schedule(budget,preempted) is None
    assert len(data['events'])==n+1 and data['events'][-1]['entry_gate']==expected
s.running=[None]*4
assert s.schedule(8,[]) is None and data['events'][-1]['entry_gate']=='RUNNING_CAP'
s.running=[]
s._pause_state=NS(name='PAUSED')
assert s.schedule(8,[]) is None and data['events'][-1]['entry_gate']=='PAUSED'
s._pause_state=old_native.__func__.__globals__['PauseState'].UNPAUSED

# MISS remains a valid recompute source. Pending/unknown and physical shortfall do not.
target=s.waiting[1]
for loading in (None,set(),{9}):
    cs._chunks_being_loaded=loading
    assert obs._candidate(target,manager,single,cs,0,8)['conservative_fit']
cs._chunks_being_loaded={1}
conflict=obs._candidate(target,manager,single,cs,0,8)
assert not conflict['conservative_fit'] and conflict['context']['global_load_conflict_key_indices']==[1]
assert conflict['context']['global_load_conflict_keys']==['1'] and 'MATCHED_PREFIX_IN_GLOBAL_LOAD' in conflict['reasons']
del cs._chunks_being_loaded
unknown=obs._candidate(target,manager,single,cs,0,8)
assert not unknown['conservative_fit'] and 'UNKNOWN_GLOBAL_LOAD_STATE' in unknown['reasons']
cs._chunks_being_loaded=None; target.skip_reading_prefix_cache=True
disabled=obs._candidate(target,manager,single,cs,0,8)
assert not disabled['conservative_fit'] and disabled['context']['state']=='UNSUPPORTED_PREFIX_READ'
target.skip_reading_prefix_cache=False
cs.manager._policy.blocks.clear()
cs._chunks_being_loaded={1}  # A MISS has no matched prefix/global-load dependency.
miss=obs._candidate(s.waiting[1],manager,single,cs,0,8)
assert miss['context']['state']=='MISS' and miss['conservative_fit']
cs._chunks_being_loaded=None
cs.manager._policy.blocks[0]=NS(is_ready=False,ref_cnt=-1)
assert not obs._candidate(s.waiting[1],manager,single,cs,0,8)['conservative_fit']
cs.manager._policy.blocks.clear(); cs._req_status['anchor'].group_states[0].offload_keys=[]
assert not obs._candidate(s.waiting[1],manager,single,cs,0,8)['context']['known']
cs._req_status['anchor'].group_states[0].offload_keys=list(range(4))
short=obs._candidate(s.waiting[1],manager,single,cs,2,4)
assert short['memory_fit'] and not short['native_reservation_pass']

# Native errors propagate unchanged; release stops observations; uninstall preserves outer wrappers.
s.waiting.fail=True
try: s.schedule(8,[])
except ValueError as error: assert str(error)=='native queue exception'
else: raise AssertionError('Native exception was swallowed')
s.waiting.fail=False
source['events'].append(dict(kind='release')); count=len(data['events'])
assert s.schedule(8,[])=='head' and len(data['events'])==count
uninstall(); uninstall()
assert s.schedule is wrapper and slot.cell_contents is old_native and s.schedule(8,[])=='head'

# Code drift and ambiguity must fail before replacing the slot.
s,source,manager,single,cs=fixture(); native=obs._native_slot(s.schedule,s).cell_contents
changed=copy.deepcopy(fragment); changed.body[0].body.append(ast.Pass()); ast.fix_missing_locations(changed)
# Unreachable pass need not alter bytecode; a changed literal return definitely does.
changed.body[0].body[-2]=ast.Return(value=ast.Constant(value='drift')); ast.fix_missing_locations(changed)
try: obs._attach(s,source,manager,single,cs,changed)
except RuntimeError as error: assert 'does not match' in str(error)
else: raise AssertionError('Code drift accepted')
assert obs._native_slot(s.schedule,s).cell_contents is native

s,source,manager,single,cs=fixture()
source['selected_request']='anchor'
data,uninstall=obs._attach(s,source,manager,single,cs,fragment)
s._inflight_prefill_reserved_blocks=lambda:None
assert s.schedule(8,[])=='head' and data['status']=='UNVERIFIED'
assert len(data['observation_errors'])==1
uninstall()
print('PASS: pinned full AST unchanged except two callbacks; complete code identity; native decision fragment; gates, shadow/no mutations, MISS/pending/unknown, exact wrapper lifecycle, exceptions and drift')
