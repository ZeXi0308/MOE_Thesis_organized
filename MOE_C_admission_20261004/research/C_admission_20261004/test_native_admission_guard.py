"""CPU check of the pinned waiting cap guard, not a full scheduler replay.

Applicability: install from a drained cell; retain the complete unique unfinished
admitted cap; all running/recovery requests belong to that set; no streaming;
the set never exceeds 192. Engine compilation capacity remains 256. The test
does not establish equivalence outside these invariants or replay GPU work.
"""
import ast
import copy
from pathlib import Path
import unittest

import test_admission_probe as fixture


def native_guard():
    path = Path(__file__).with_name('vendor') / 'scheduler.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    schedule = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    matches = [node for node in ast.walk(schedule) if isinstance(node, ast.While)
        and len(node.body) >= 2 and isinstance(node.body[0], ast.Assign)
        and any(isinstance(n, ast.Name) and n.id == 'num_running' for n in node.body[0].targets)
        and isinstance(node.body[1], ast.If)
        and any(isinstance(n, ast.Attribute) and n.attr == 'max_num_running_reqs'
                for n in ast.walk(node.body[1].test))]
    assert len(matches) == 1, 'Pinned native waiting cap guard changed'
    loop = copy.deepcopy(matches[0])
    assert len(loop.body[1].body) == 1 and isinstance(loop.body[1].body[0], ast.Break)
    # Preserve the actual waiting-loop condition, count expression, and break.
    # Stop immediately after this guard, before any request is examined.
    loop.body = loop.body[:2] + [ast.Return(value=ast.Constant(value=True))]
    wrapper = ast.parse('def guard(self, token_budget=1024):\n    return False\n')
    wrapper.body[0].body.insert(0, loop)
    namespace = {}
    exec(compile(ast.fix_missing_locations(wrapper), str(path), 'exec'), namespace)
    return namespace['guard']


class NativeAdmissionGuardTest(unittest.TestCase):
    def test_full_cap_and_recovery_slots(self):
        guard = native_guard()
        status = fixture.Status
        for running_count, recovery_kinds in ((192, ()), (190, ('preempted', 'ready')),
                                               (191, ('pending',))):
            with self.subTest(running=running_count, recovery=recovery_kinds):
                running = [fixture.Request(f'running{i}', status.RUNNING, computed=16, output=1)
                           for i in range(running_count)]
                recovery = [fixture.Request(kind,
                    status.PREEMPTED if kind == 'preempted' else status.WAITING_FOR_REMOTE_KVS,
                    preemptions=1, computed=0 if kind == 'preempted' else 16)
                    for kind in recovery_kinds]
                new = fixture.Request('new')
                scheduler = fixture.scheduler(running + recovery + [new])
                scheduler.running = running
                scheduler.waiting, scheduler.skipped_waiting = recovery + [new], []
                scheduler.num_waiting_for_streaming_input = 0
                if 'ready' in recovery_kinds:
                    scheduler.finished_recving_kv_req_ids.add('ready')
                gate = fixture.probe.Gate(scheduler, cap=192, kv_floor=0, probe_enabled=False)
                for request in running + recovery:
                    saved_status = request.status
                    # Register the successful original admission, then preserve
                    # its later preemption/remote state without releasing the cap.
                    if saved_status == status.PREEMPTED:
                        request.status = status.RUNNING
                    gate.admitted(request)
                    request.status = saved_status
                gate.begin()
                self.assertEqual(gate.state()['active'], 192)
                self.assertTrue({r.request_id for r in running} <= gate.admitted_ids)

                outcomes = []
                for native_cap in (192, 256):
                    scheduler.max_num_running_reqs = native_cap
                    outcomes.append(guard(scheduler))
                self.assertEqual(outcomes, [running_count < 192, True])

                # Native slots alone do not account for remote/preempted members.
                # Complete admission accounting still denies the 193rd request.
                check = fixture.GateTests.before(gate, new)
                self.assertTrue(check['fit']['fits'])
                self.assertTrue(check['row']['denied'])
                self.assertEqual(check['row']['reason'], 'cap')
                for request in recovery:
                    self.assertFalse(gate.is_new(request))
                    self.assertIsNone(gate.early(request))  # Even behind a cap FIFO barrier.
                    old_check = fixture.GateTests.before(gate, request)
                    self.assertIsNone(old_check['row'])
                    self.assertTrue(old_check['fit']['fits'])
                self.assertEqual(gate.state()['active'], 192)


if __name__ == '__main__':
    unittest.main()
