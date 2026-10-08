"""Bounded resource/state checks; no simulated time or claimed GPU performance."""
import ast
from dataclasses import replace
from pathlib import Path
import random
import unittest

from completion_headroom import CompletionHeadroom, DecodeState, patched_schedule_tree


class HeadroomChecks(unittest.TestCase):
    def test_measured_first_window_and_negative_control(self):
        leader = DecodeState("leader", 3868, 3072, 797, 1024, 242)
        peers = [DecodeState(str(i), 3840, 3072, 769, 1024, 240) for i in range(3)]
        policy = CompletionHeadroom(16)
        decision = policy.decide([leader, *peers], 16)
        self.assertEqual(decision["reserve"], 14)
        self.assertEqual(decision["held"], ["2"])
        # Allocating all three peers, as the negative control does, spends
        # the leader's completion guarantee even though this step still fits.
        self.assertLess(16 - sum(r.next_blocks(16) for r in [leader, *peers]), 14)

    def test_final_token_not_cached_and_late_activation_rejected(self):
        request = DecodeState("a", 15, 8, 8, 9, 1)
        decision = CompletionHeadroom(16).decide([request], 0)
        self.assertEqual(decision["scheduled"], ["a"])
        self.assertEqual(decision["reserve"], 0)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            CompletionHeadroom(16).decide([replace(request, max_output=10)], 0)

    def test_homogeneous_bound_transitions_complete_without_discard(self):
        rng = random.Random(712)
        for trial in range(150):
            b = rng.choice([2, 4, 16])
            rows = []
            prompt, maximum = rng.randint(1, 24), rng.randint(2, 40)
            for i in range(rng.randint(2, 12)):
                output = rng.randint(1, maximum - 1)
                computed = prompt + output - 1
                rows.append(DecodeState(str(i), computed, prompt, output, maximum,
                                        (computed + b - 1) // b))
            leader = min(rows, key=lambda r: r.max_output - r.output)
            free = leader.remaining_blocks(b) + rng.randint(0, 3)
            capacity = free + sum(r.allocated for r in rows)
            work = sum(r.max_output - r.output for r in rows)
            policy, executed = CompletionHeadroom(b), 0
            while rows:
                decision = policy.decide(rows, free)
                self.assertTrue(decision["scheduled"], trial)
                previous = {r.request_id: r for r in rows}
                next_rows = []
                for r in rows:
                    if r.request_id in decision["scheduled"]:
                        additional = r.next_blocks(b)
                        free -= additional
                        self.assertGreaterEqual(free, 0, trial)
                        r = replace(r, computed=r.computed + 1, output=r.output + 1,
                                    allocated=r.allocated + additional)
                        executed += 1
                        if r.output == r.max_output:
                            free += r.allocated
                            continue
                    else:
                        self.assertEqual(r, previous[r.request_id])
                    next_rows.append(r)
                self.assertEqual(capacity, free + sum(r.allocated for r in next_rows))
                rows = next_rows
                self.assertLessEqual(executed, work)
            self.assertEqual(executed, work)
            self.assertEqual(free, capacity)

    def test_one_completion_does_not_certify_arbitrary_future_bounds(self):
        # Retained counterexample from the initial heterogeneous property run:
        # a 9-block pool cannot hold request a's 10-block terminal KV. The
        # short request can finish; that does not certify the rest of a cohort.
        policy = CompletionHeadroom(4)
        long = DecodeState("a", 17, 1, 17, 39, 5)
        short = DecodeState("b", 14, 14, 1, 2, 4)
        first = policy.decide([long, short], 0)
        self.assertIn("b", first["scheduled"])
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            policy.decide([replace(long, computed=18, output=18)], 4)

    def test_patch_only_inserts_two_native_skips(self):
        source = Path("/private/tmp/moe-native-v026-recovery-source/scheduler.py").read_text()
        patched = patched_schedule_tree(source)
        compile(patched, "pinned-scheduler.py", "exec")
        fn = patched.body[0]
        for node in ast.walk(fn):
            if isinstance(node, ast.While) and ast.unparse(node.test) == "req_index < len(self.running) and token_budget > 0":
                self.assertIn("self._headroom_held", ast.unparse(node.body[1]))
                del node.body[1]
            elif isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp) and "self._headroom_closed" in ast.unparse(node.test):
                node.test = node.test.values[0]
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
        self.assertEqual(ast.dump(fn), ast.dump(original))


if __name__ == "__main__":
    unittest.main()
