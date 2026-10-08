"""Bounded retry-gate checks on the frozen native waiting-loop CPU fixture."""
import ast
import copy
import hashlib
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as NS

import retry_defer as probe

B = Path(__file__).resolve().parents[1]
fixture_path = B/'recovery_queue/check_cpu.py'
assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == '702e80971cbff2c193980dd29d82aa48d38584634b73bd96db681ddcb6000506'
sys.path.insert(0, str(fixture_path.parent))
frozen = runpy.run_path(str(fixture_path))
fragment, full_tree = frozen['fragment'], frozen['full_tree']


def is_callback(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == probe.H.CALLBACK)


class RemoveGate(ast.NodeTransformer):
    def visit_Expr(self, node):
        return None if is_callback(node.value) else self.generic_visit(node)

    def visit_If(self, node):
        if is_callback(node.test):
            assert len(node.body) == 2 and isinstance(node.body[-1], ast.Break)
            assert is_callback(node.body[0].value) and not node.orelse
            return None
        return self.generic_visit(node)


instrumented = probe._instrument(full_tree)
assert ast.dump(RemoveGate().visit(copy.deepcopy(instrumented)), include_attributes=True) == ast.dump(full_tree, include_attributes=True)
compile(instrumented, '<full-native-with-retry-gate>', 'exec')


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


def peer(rid):
    r = NS(request_id=rid, status=NS(name='RUNNING'), finished=False)
    r.is_finished = lambda: r.finished
    return r


def setup(mode='defer_once', prior_change=None):
    s, _, manager, single, cs = frozen['fixture']()
    head = s.waiting[0]
    head.num_preemptions = 2
    peers = [peer('running-a'), peer('running-b')]
    s.running = peers[:]
    for r in peers:
        single.req_to_blocks[r.request_id] = [NS(is_null=False, ref_cnt=1)]
    prior = dict(request=head.request_id, qualified=True, residency_outputs=3,
                 bidkv=dict(num_preemptions=1))
    if prior_change:
        prior_change(prior)
    decision = dict(step=9, host_perf_counter_s=99., failed_request='other-running',
                    selected=head.request_id, native_tail=head.request_id, changed=False,
                    candidates=[prior])
    selective = dict(victim_decisions=[decision])
    clock = Clock()
    wrapper = s.schedule
    slot = probe.H._native_slot(wrapper, s)
    old = slot.cell_contents
    data, undo = probe._attach(s, manager, single, cs, mode, selective, fragment, clock)
    return NS(s=s, manager=manager, single=single, cs=cs, head=head, peers=peers,
              selective=selective, prior=prior, data=data, undo=undo, clock=clock,
              wrapper=wrapper, slot=slot, old=old)


def run(x, budget=8, preempted=None):
    return x.s.schedule(budget, [] if preempted is None else preempted)


def opportunities(x):
    return [e for e in x.data['events'] if e['kind'] == 'legal_retry_opportunity']


def last_release(x):
    return [e for e in x.data['events'] if e['kind'] == 'release'][-1]


# Execute the native decision fragment: the only changed result is the actual
# waiting-loop break. Existing queue, blocks, Host refs and outer wrappers stay.
for mode in ('native', 'defer_once'):
    x = setup(mode)
    waiting = list(x.s.waiting)
    owned = dict(x.single.req_to_blocks)
    host = dict(x.cs.manager._policy.blocks)
    assert run(x) == ('head' if mode == 'native' else None)
    e = opportunities(x)[0]
    expected = int(mode == 'defer_once')
    assert x.data['action_count'] == x.data['executed_breaks'] == expected
    assert e['requested_breaks'] == e['executed_breaks'] == expected
    assert e['final_action'] == ('SHADOW_ONLY' if mode == 'native' else 'WAITING_LOOP_BREAK')
    assert x.s.schedule is x.wrapper and x.s.waiting == waiting
    assert x.single.req_to_blocks == owned and x.cs.manager._policy.blocks == host
    assert all(b.ref_cnt == 0 for b in host.values())
    assert x.manager.block_pool.free_block_queue.num_free_blocks == 8
    x.clock.value = 100.499
    assert run(x) == ('head' if mode == 'native' else None)
    assert len(opportunities(x)) == 1 and x.data['action_count'] == expected
    x.clock.value = 100.5
    assert run(x) == 'head'
    if expected:
        assert last_release(x)['reason'] == 'DEADLINE'
        assert e['extra_gate_observed_s'] == .5 and e['deadline_observation_lateness_s'] == 0
        assert x.data['executed_breaks'] == 2
    # No later retry can consume a second budget, even with fresh victim evidence.
    x.head.num_preemptions += 1
    x.selective['victim_decisions'].append(dict(x.selective['victim_decisions'][0],
        candidates=[dict(x.prior, bidkv=dict(num_preemptions=2))]))
    assert run(x) == 'head' and len(opportunities(x)) == 1
    assert x.data['action_count'] == expected
    x.undo(); x.undo()
    assert x.s.schedule is x.wrapper and x.slot.cell_contents is x.old and run(x) == 'head'

# Qualification must identify output from a resumed residence and the same
# successful preemption episode; a rejection must leave the one-shot budget.
for change in (
    lambda r:r.update(qualified=False),
    lambda r:r.update(residency_outputs=0),
    lambda r:r.update(residency_outputs=None),
    lambda r:r.update(bidkv=dict(num_preemptions=0)),
    lambda r:r.update(bidkv=dict(num_preemptions=2)),
):
    x = setup(prior_change=change)
    assert run(x) == 'head' and not opportunities(x) and x.data['action_count'] == 0
    assert x.data['skip_counts']['NO_PREVIOUS_RESUMED_OUTPUT_PREEMPTION'] == 1
    valid = dict(x.prior, qualified=True, residency_outputs=1, bidkv=dict(num_preemptions=1))
    x.selective['victim_decisions'].append(dict(x.selective['victim_decisions'][0], candidates=[valid]))
    assert run(x) is None and x.data['action_count'] == 1
    x.undo()

# A closed native gate or failed conservative fit does not consume an action.
for gate in ('budget', 'preempted', 'capacity', 'host_pending'):
    x = setup()
    if gate == 'capacity': x.manager.block_pool.free_block_queue.num_free_blocks = 0
    if gate == 'host_pending': x.cs.manager._policy.blocks[0].is_ready = False
    result = run(x, budget=0 if gate == 'budget' else 8,
                 preempted=[x.head] if gate == 'preempted' else [])
    assert result == (None if gate in ('budget', 'preempted') else 'head')
    assert not opportunities(x) and x.data['action_count'] == 0
    x.manager.block_pool.free_block_queue.num_free_blocks = 8
    x.cs.manager._policy.blocks[0].is_ready = True
    assert run(x) is None and x.data['action_count'] == 1
    x.undo()

# Removal or completion alone is insufficient: require a captured cohort member
# to be terminal AND have returned its native KV ownership.
x = setup(); assert run(x) is None
completed = x.peers[0]
x.s.running.remove(completed)
x.single.req_to_blocks.pop(completed.request_id)
x.clock.value = 100.1
assert run(x) is None and not any(e['kind'] == 'release' for e in x.data['events'])
completed.finished = True; completed.status = NS(name='FINISHED_LENGTH_CAPPED')
x.single.req_to_blocks[completed.request_id] = [NS(is_null=False, ref_cnt=1)]
x.clock.value = 100.2
assert run(x) is None and not any(e['kind'] == 'release' for e in x.data['events'])
x.single.req_to_blocks.pop(completed.request_id)
x.clock.value = 100.3
assert run(x) == 'head'
assert last_release(x)['reason'] == 'COHORT_FINISHED_AND_FREED'
assert last_release(x)['completed_released'] == [dict(request='running-a', status='FINISHED_LENGTH_CAPPED')]
x.undo()

# Existing native changes cancel only the extra gate; no queue/request/ref is
# undone. Empty running is a release even when removed peers are not terminal.
def new_waiter(x):
    r = frozen['request']('new', 10); r.status = NS(name='WAITING')
    x.s.waiting.append(r)


for change, reason in (
    (lambda x:x.s.running.clear(), 'RUNNING_EMPTY'),
    (lambda x:setattr(x.s, '_rotation_target', x.head.request_id), 'BASELINE_OVERRIDE'),
    (new_waiter, 'NEW_WAITER'),
    (lambda x:x.cs._req_status[x.head.request_id].transfer_jobs.add(73), 'TARGET_NATIVE_STATE_CHANGED'),
    (lambda x:x.single.req_to_blocks.update(head=[NS(is_null=False, ref_cnt=1)]), 'TARGET_NATIVE_STATE_CHANGED'),
    (lambda x:setattr(x.head, 'status', NS(name='WAITING_FOR_REMOTE_KVS')), 'TARGET_LEFT_PREEMPTED_QUEUE'),
):
    x = setup(); assert run(x) is None
    change(x); x.clock.value = 100.1
    assert run(x) == 'head' and last_release(x)['reason'] == reason
    assert x.data['action_count'] == 1 and x.data['executed_breaks'] == 1
    if reason == 'TARGET_NATIVE_STATE_CHANGED' and 73 in x.cs._req_status['head'].transfer_jobs:
        assert x.cs._req_status['head'].transfer_jobs == {73}
    x.undo()

# The entry callback expires the extra gate even when this step's original
# waiting gate cannot run. It does not force allocation/scheduling through it.
x = setup(); assert run(x) is None
x.clock.value = 100.6
assert run(x, budget=0, preempted=[x.head]) is None
assert last_release(x)['reason'] == 'DEADLINE'
assert abs(opportunities(x)[0]['deadline_observation_lateness_s']-.1) < 1e-12
x.undo()

# Native exceptions pass through while active; uninstall restores the exact
# closure and reports an active gate's cancellation once, without a new action.
x = setup(); assert run(x) is None
x.s.waiting.fail = True
try:
    run(x)
except ValueError as error:
    assert str(error) == 'native queue exception'
else:
    raise AssertionError('Native exception swallowed')
assert x.data['executed_breaks'] == 1
x.undo(); count = len(x.data['events']); x.undo()
assert len(x.data['events']) == count and last_release(x)['reason'] == 'UNINSTALL_WITH_ACTIVE_GATE'
assert x.slot.cell_contents is x.old and x.s.schedule is x.wrapper
x.s.waiting.fail = False
assert run(x) == 'head'

print('PASS: unchanged native AST after removing two insertions; executed-break footprint; prior resumed-output episode; exact 500ms expiry; finished-and-freed provenance; native/new-waiter/transfer fallbacks; one action; exception and idempotent uninstall')
