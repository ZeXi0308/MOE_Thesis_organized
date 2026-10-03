"""Small CPU closure for the pinned FCFS prefix-rollback AST branch."""

import ast
from pathlib import Path
import sys
import textwrap
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pkg"))
from rotation_native import patched_schedule_tree  # noqa: E402
from staged_store_rotation import (  # noqa: E402
    _bidkv_full_running_choice, _full_running_indices,
    _native_victim_current_guard,
)

PINNED = HERE.parent / "liveness_pinned_sources_20260930" / "scheduler.py"


def fcfs_branch():
    tree = patched_schedule_tree(PINNED.read_text())
    compile(tree, str(PINNED), "exec")
    policies = [node for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "self.policy == SchedulingPolicy.PRIORITY"]
    assert len(policies) == 1
    branch = ast.unparse(ast.Module(body=policies[0].orelse, type_ignores=[]))
    source = ("def execute(self, req_index, token_budget, scheduled_running_reqs, "
              "num_scheduled_tokens, req_to_new_blocks, "
              "scheduled_spec_decode_tokens, scheduled_encoder_inputs, "
              "encoder_compute_budget):\n"
              + textwrap.indent(branch, "    ") + "\n"
              "    return (preempted_req, req_index, token_budget, "
              "scheduled_running_reqs, num_scheduled_tokens, req_to_new_blocks, "
              "scheduled_spec_decode_tokens, scheduled_encoder_inputs, "
              "encoder_compute_budget)\n")
    namespace = {}
    exec(compile(source, "pinned FCFS branch", "exec"), namespace)
    return namespace["execute"]


class Request:
    def __init__(self, name):
        self.request_id = name

    def get_num_encoder_embeds(self, index):
        return index


class Scheduler:
    def __init__(self, running, choice, full_running):
        self.running = list(running)
        self.choice = choice
        self._rotation_full_running_enabled = full_running
        self.rollback = []

    def _rotation_pick_victim(self, req_index, scheduled_running_reqs):
        return self.choice

    def _rotation_note_prefix_rollback(self, *args):
        self.rollback.append(args)


class FullRunningClosure(unittest.TestCase):
    def test_prefix_revokes_current_step_plan_and_refunds_budget(self):
        a, b, c = (Request(name) for name in "abc")
        scheduler = Scheduler([a, b, c], choice=0, full_running=True)
        planned = [a]
        tokens = {"a": 2}
        blocks = {"a": object()}
        spec = {"a": [7]}
        encoder = {"a": [3]}
        result = fcfs_branch()(scheduler, 1, 5, planned, tokens, blocks,
                               spec, encoder, 4)
        victim, index, budget, *_rest, encoder_budget = result
        self.assertIs(victim, a)
        self.assertEqual([r.request_id for r in scheduler.running], ["b", "c"])
        self.assertEqual((index, budget, encoder_budget), (0, 7, 7))
        self.assertEqual((planned, tokens, blocks, spec, encoder), ([], {}, {}, {}, {}))
        self.assertEqual(len(scheduler.rollback), 1)
        self.assertEqual(scheduler.rollback[0][1:], (2, 1, 0, True, 1, 3))

    def test_suffix_and_disabled_mode_keep_prior_plan(self):
        a, b, c = (Request(name) for name in "abc")
        scheduler = Scheduler([a, b, c], choice=2, full_running=False)
        planned = [a]
        tokens = {"a": 1}
        blocks = {"a": object()}
        result = fcfs_branch()(scheduler, 1, 5, planned, tokens, blocks,
                               {}, {}, 4)
        self.assertIs(result[0], c)
        self.assertEqual((result[1], result[2]), (1, 5))
        self.assertEqual([r.request_id for r in scheduler.running], ["a", "b"])
        self.assertEqual((planned, tokens), ([a], {"a": 1}))
        self.assertFalse(scheduler.rollback)
        self.assertEqual(_full_running_indices([a, b, c], 1, [a]), [0, 1, 2])
        self.assertEqual(_full_running_indices([a, b, c], 1, []), [1, 2])

    def test_unknown_and_current_guard_use_explicit_failed_index(self):
        rows = [dict(index=index, arrival_time=index, request=str(index),
                     bidkv=dict(utility=utility))
                for index, utility in ((0, 2), (1, 9), (2, 1))]
        # The failed current is index 1 even though prefix index 0 comes first.
        selected, unrestricted, eligible = _bidkv_full_running_choice(
            rows, False, current_index=1, native_tail_index=2, guard=True)
        self.assertEqual((selected, unrestricted, eligible), (0, 1, True))
        self.assertEqual(_bidkv_full_running_choice(
            rows, True, current_index=1, native_tail_index=2, guard=True),
            (2, 2, False))
        suffix = rows[1:]
        self.assertEqual(_native_victim_current_guard(
            suffix, "bidkv_score", False, True), (2, 1, True))


if __name__ == "__main__":
    unittest.main()
