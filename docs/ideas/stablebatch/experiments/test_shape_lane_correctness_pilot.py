import unittest

import run_shape_lane_correctness_pilot as runner


class ShapeLaneCorrectnessTests(unittest.TestCase):
    def test_companion_segments_are_disjoint_and_deterministic(self):
        identities = [f"row-{index:02d}" for index in range(30)]
        first = runner.select_companion_segments(identities, "seed", "cell", 7, 3)
        second = runner.select_companion_segments(identities, "seed", "cell", 7, 3)
        self.assertEqual(first, second)
        self.assertEqual([len(row) for row in first], [7, 7, 7])
        self.assertEqual(len({item for row in first for item in row}), 21)

    def test_companion_segments_fail_when_pool_is_too_small(self):
        with self.assertRaises(runner.ProtocolError):
            runner.select_companion_segments(["a"] * 20, "seed", "cell", 7, 3)

    def test_rank_order_is_rotation(self):
        self.assertEqual(runner.rank_order(8, 0, 0), list(range(8)))
        self.assertEqual(runner.rank_order(8, 2, 1), [3, 4, 5, 6, 7, 0, 1, 2])

    def test_classify_pass(self):
        config = {
            "research_boundary": "bounded",
            "gate": {"minimum_eligible_cells": 2, "minimum_distinct_victims": 2},
            "decision": {"pass": "PASS", "kill": "KILL", "invalid": "INVALID"},
        }
        rows = [
            {
                "cell_key": "a",
                "victim_id": "v1",
                "prior_sensitive_hashes": [{}],
                "within_context_repeat_raw_bitwise": True,
                "cross_context_raw_bitwise": True,
                "cross_context_target_moe_bitwise": True,
                "cross_context_downstream_routes_equal": True,
                "cross_context_final_logits_bitwise": True,
            },
            {
                "cell_key": "b",
                "victim_id": "v2",
                "prior_sensitive_hashes": [{}],
                "within_context_repeat_raw_bitwise": True,
                "cross_context_raw_bitwise": True,
                "cross_context_target_moe_bitwise": True,
                "cross_context_downstream_routes_equal": True,
                "cross_context_final_logits_bitwise": True,
            },
        ]
        self.assertEqual(runner.classify_results(rows, [], config)["verdict"], "PASS")

    def test_classify_kill_on_one_raw_mismatch(self):
        config = {
            "research_boundary": "bounded",
            "gate": {"minimum_eligible_cells": 1, "minimum_distinct_victims": 1},
            "decision": {"pass": "PASS", "kill": "KILL", "invalid": "INVALID"},
        }
        row = {
            "cell_key": "a",
            "victim_id": "v1",
            "prior_sensitive_hashes": [{}],
            "within_context_repeat_raw_bitwise": True,
            "cross_context_raw_bitwise": False,
            "cross_context_target_moe_bitwise": True,
            "cross_context_downstream_routes_equal": True,
            "cross_context_final_logits_bitwise": True,
        }
        self.assertEqual(runner.classify_results([row], [], config)["verdict"], "KILL")

    def test_classify_invalid_on_coverage(self):
        config = {
            "research_boundary": "bounded",
            "gate": {"minimum_eligible_cells": 2, "minimum_distinct_victims": 2},
            "decision": {"pass": "PASS", "kill": "KILL", "invalid": "INVALID"},
        }
        self.assertEqual(runner.classify_results([], [], config)["verdict"], "INVALID")


if __name__ == "__main__":
    unittest.main()

