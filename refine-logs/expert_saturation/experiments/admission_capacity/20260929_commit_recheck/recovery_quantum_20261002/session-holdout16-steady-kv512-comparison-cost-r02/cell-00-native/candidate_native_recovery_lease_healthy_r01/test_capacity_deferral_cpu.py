"""Bounded CPU closure for the pinned native allocation-failure branch."""
import ast
from pathlib import Path
import sys
import textwrap
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'pkg'))
from rotation_native import patched_schedule_tree  # noqa: E402
from staged_store_rotation import _capacity_finish_remaining  # noqa: E402

PINNED = HERE.parent.parent / 'liveness_pinned_sources_20260930' / 'scheduler.py'


def pinned_failure_loop():
    tree = patched_schedule_tree(PINNED.read_text())
    compile(tree, str(PINNED), 'exec')
    outer = [node for node in ast.walk(tree) if isinstance(node, ast.While)
             and ast.unparse(node.test) == 'req_index < len(self.running) and token_budget > 0']
    assert len(outer) == 1
    retry = [node for node in ast.walk(outer[0]) if isinstance(node, ast.While)
             and ast.unparse(node.test) == 'True'
             and 'self.kv_cache_manager.allocate_slots' in ast.unparse(node)]
    failed = [node for node in outer[0].body if isinstance(node, ast.If)
              and ast.unparse(node.test) == 'new_blocks is None']
    assert len(retry) == len(failed) == 1
    copied = ast.unparse(ast.Module(body=[retry[0], failed[0]], type_ignores=[]))
    source = (
        'def execute(self, prefix):\n'
        '    req_index = 1\n'
        '    token_budget = 2\n'
        '    scheduled_running_reqs = [prefix]\n'
        '    num_scheduled_tokens = {prefix.request_id: 1}\n'
        '    req_to_new_blocks = {prefix.request_id: object()}\n'
        '    scheduled_spec_decode_tokens = {}\n'
        '    scheduled_encoder_inputs = {}\n'
        '    encoder_compute_budget = 0\n'
        '    preempted_reqs = []\n'
        '    scheduled_timestamp = 0\n'
        '    while req_index < len(self.running) and token_budget > 0:\n'
        '        request = self.running[req_index]\n'
        '        deferred_current = False\n'
        '        num_new_tokens = 1\n'
        + textwrap.indent(copied, '        ') + '\n'
        '        scheduled_running_reqs.append(request)\n'
        '        num_scheduled_tokens[request.request_id] = 1\n'
        '        req_index += 1\n'
        '    return (req_index, num_scheduled_tokens,\n'
        '            [r.request_id for r in preempted_reqs])\n')
    class SchedulingPolicy:
        PRIORITY = object()
    namespace = {'SchedulingPolicy': SchedulingPolicy}
    exec(compile(source, 'pinned allocation failure closure', 'exec'), namespace)
    return namespace['execute']


class Req:
    def __init__(self, request_id):
        self.request_id = request_id


class Manager:
    def allocate_slots(self, request, num_new_tokens, *, num_lookahead_tokens):
        return None if request.request_id == 'b' else object()


class Scheduler:
    def __init__(self, mode):
        self.running = [Req('a'), Req('b'), Req('c')]
        self.kv_cache_manager = Manager()
        self.num_lookahead_tokens = 0
        self.policy = None
        self._rotation_full_running_enabled = False
        self.mode = mode
        self.visited = []

    def _rotation_try_capacity_deferral(self, request, req_index, scheduled,
                                        planned, new_tokens, preempted):
        self.visited.append(request.request_id)
        return self.mode == 'prefix_work' and request.request_id == 'b' and bool(scheduled)

    def _rotation_pick_victim(self, req_index, scheduled):
        return len(self.running) - 1

    def _rotation_continue_after_self_preempt(self, request, req_index):
        return False

    def _preempt_request(self, request, timestamp):
        pass


class DeferralClosure(unittest.TestCase):
    def test_defer_skips_preemption_and_visits_next_while_preserving_prefix(self):
        scheduler = Scheduler('prefix_work')
        index, plan, preempted = pinned_failure_loop()(scheduler, scheduler.running[0])
        self.assertEqual((index, plan, preempted), (3, {'a': 1, 'c': 1}, []))
        self.assertEqual(scheduler.visited, ['b'])

    def test_off_keeps_native_preemption_path(self):
        scheduler = Scheduler('off')
        index, plan, preempted = pinned_failure_loop()(scheduler, scheduler.running[0])
        self.assertEqual((index, plan, preempted), (1, {'a': 1}, ['c', 'b']))

    def test_hard_cap_boundary_and_unknown_fallback(self):
        self.assertEqual(_capacity_finish_remaining(8, 12, 60, 4, 16), 4)
        self.assertIsNone(_capacity_finish_remaining(8, 13, 60, 4, 16))
        self.assertIsNone(_capacity_finish_remaining(8, None, 60, 4, 16))
        self.assertIsNone(_capacity_finish_remaining(8, 8, 60, 4, 16))


if __name__ == '__main__':
    unittest.main()
