"""CPU check of passing an earlier native skipped head before the gated target."""
import ast
import copy
import hashlib
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as NS

import retry_defer_persistent as probe

B = Path(__file__).resolve().parents[1]
p = B/'recovery_queue/check_cpu.py'
assert hashlib.sha256(p.read_bytes()).hexdigest() == '702e80971cbff2c193980dd29d82aa48d38584634b73bd96db681ddcb6000506'
sys.path.insert(0, str(p.parent))
f = runpy.run_path(str(p))
assert probe.adapted_source.replace(probe.NEW, probe.OLD) == probe.source
assert probe.adapted_source.count(probe.NEW) == 1

# Retain the exact native blocked-request pop/local-prepend/continue branch and
# the original post-loop skipped merge. Stop before lookup/allocation with the
# same return-request-ID continuation stub as the frozen fixture.
tree = copy.deepcopy(f['fragment'])
outer = tree.body[0].body[0]
native_outer = next(n for n in f['full_tree'].body[0].body if isinstance(n, ast.If)
                   and ast.unparse(n.test).startswith('len(preempted_reqs) == self._rotation_forced_count'))
loop = next(n for n in outer.body if isinstance(n, ast.While))
native_loop = next(n for n in native_outer.body if isinstance(n, ast.While))
blocked = next(n for n in native_loop.body if isinstance(n, ast.If)
               and ast.unparse(n.test).startswith('self._is_blocked_waiting_status'))
loop.body.insert(len(loop.body)-1, copy.deepcopy(blocked))
merge = next(n for n in native_outer.body if isinstance(n, ast.If)
             and ast.unparse(n.test) == 'step_skipped_waiting')
outer.body.append(copy.deepcopy(merge))
ast.fix_missing_locations(tree)


def prepend(self, request):
    self.insert(0, request)


def prepend_many(self, requests):
    self[:0] = list(requests)


f['Queue'].prepend_request = prepend
f['Queue'].prepend_requests = prepend_many
f['Queue'].pop_request = lambda self: self.pop(0)


def setup(mode):
    s, _, manager, single, cs = f['fixture']()
    target = s.waiting[0]; target.num_preemptions = 2
    running = NS(request_id='running', status=NS(name='RUNNING'), is_finished=lambda:False)
    s.running = [running]
    s._is_blocked_waiting_status = lambda status: status.name == 'WAITING_FOR_REMOTE_KVS'
    s._try_promote_blocked_waiting_request = lambda request:False
    slot = probe.H._native_slot(s.schedule, s)
    old = slot.cell_contents
    slot.cell_contents = probe.H._compile(tree, old)
    old = slot.cell_contents
    # The exact native branch has a debug call before moving a pending LOAD.
    old.__func__.__globals__['RequestStatus'] = NS(WAITING_FOR_REMOTE_KVS=NS(name='WAITING_FOR_REMOTE_KVS'))
    old.__func__.__globals__['logger'] = NS(debug=lambda *args:None)
    clock = [100.]
    selective = dict(victim_decisions=[dict(step=1, host_perf_counter_s=99.,
        selected=target.request_id, native_tail=target.request_id, changed=False,
        failed_request='other', candidates=[dict(request=target.request_id,
            qualified=True, residency_outputs=7, bidkv=dict(num_preemptions=1))])])
    data, undo = probe._attach(s, manager, single, cs, mode, selective, tree, lambda:clock[0])
    return s, single, cs, target, clock, data, undo, slot, old


for mode in ('native', 'defer_once'):
    s, single, cs, target, clock, data, undo, slot, old = setup(mode)
    assert s.schedule(8, []) == ('head' if mode == 'native' else None)
    if mode == 'native':
        assert data['action_count'] == 0 and data['events'][0]['final_action'] == 'SHADOW_ONLY'
        undo(); continue
    pending = f['request']('earlier-load', 0)
    pending.status = old.__func__.__globals__['RequestStatus'].WAITING_FOR_REMOTE_KVS
    s.skipped_waiting.append(pending)
    before = (list(s.waiting), list(s.skipped_waiting), dict(single.req_to_blocks), dict(cs.manager._policy.blocks))
    clock[0] = 100.1
    # Native processes the earlier skipped request into its local queue, then
    # reaches the same target and executes another break. Its normal exit merges
    # the skipped peer back; the adapter itself does not reorder either queue.
    assert s.schedule(8, []) is None
    passes = [e for e in data['events'] if e['kind'] == 'native_queue_pass_through']
    assert len(passes) == 1 and passes[0]['actual_head'] == 'earlier-load'
    assert passes[0]['actual_queue'] == 'skipped_waiting' and passes[0]['skipped_order'] == ['earlier-load']
    assert data['action_count'] == 1 and data['executed_breaks'] == 2
    assert not any(e['kind'] == 'release' for e in data['events'])
    assert before == (list(s.waiting), list(s.skipped_waiting), dict(single.req_to_blocks), dict(cs.manager._policy.blocks))
    assert all(b.ref_cnt == 0 for b in cs.manager._policy.blocks.values())
    clock[0] = 100.5
    assert s.schedule(8, []) == 'head'
    assert [e for e in data['events'] if e['kind'] == 'release'][-1]['reason'] == 'DEADLINE'
    assert data['action_count'] == 1 and data['executed_breaks'] == 2
    undo(); undo(); assert slot.cell_contents is old

# Active matching is by the actual native request identity, not the public
# queue's identity or emptiness. This setup move is performed only by the CPU
# fixture, while the adapter leaves both queues untouched.
s, single, cs, target, clock, data, undo, slot, old = setup('defer_once')
assert s.schedule(8, []) is None
s.waiting.remove(target); s.skipped_waiting.append(target)
before = (list(s.waiting), list(s.skipped_waiting))
clock[0] = 100.2
assert s.schedule(8, []) is None and data['executed_breaks'] == 2
assert before == (list(s.waiting), list(s.skipped_waiting))
assert not any(e['kind'] == 'release' for e in data['events'])
undo()

print('PASS: exact one-branch source adapter; native shadow; earlier native skipped progress then target break; original skipped merge; no adapter queue/ref mutation; unchanged 500ms deadline and one action; uninstall')
