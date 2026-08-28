from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("analyze_n0b_repeat_divergence.py")
SPEC = importlib.util.spec_from_file_location("n0b_repeat_localizer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RepeatDivergenceLocalizationTest(unittest.TestCase):
    def test_topk_set_localization_ignores_order_only_changes(self) -> None:
        left = np.asarray([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
        right = np.asarray([[[[2, 1], [3, 9]], [[5, 6], [7, 8]]]])
        report = MODULE._route_differences(left, right)
        self.assertFalse(report["raw_route_exact"])
        self.assertFalse(report["topk_set_exact"])
        self.assertEqual(report["changed_raw_route_positions"], 2)
        self.assertEqual(report["changed_topk_set_positions"], 1)
        self.assertEqual(
            report["first_topk_set_divergence_by_request"], {"0": [0, 1]}
        )

    def test_first_token_difference_is_request_aligned(self) -> None:
        left = {
            "request_metrics": [
                {"token_ids": [1, 2, 3]},
                {"token_ids": [4, 5, 6]},
            ]
        }
        right = {
            "request_metrics": [
                {"token_ids": [1, 9, 3]},
                {"token_ids": [4, 5, 7]},
            ]
        }
        self.assertEqual(
            MODULE._first_token_differences(left, right),
            [
                {
                    "request_row": 0,
                    "first_output_token_index": 1,
                    "changed_output_token_count": 1,
                },
                {
                    "request_row": 1,
                    "first_output_token_index": 2,
                    "changed_output_token_count": 1,
                },
            ],
        )

    def test_saved_route_step_maps_to_next_output_token(self) -> None:
        immediate = MODULE._align_route_to_output([0, 3], 0)
        self.assertEqual(immediate["route_forward_produced_output_token_index"], 1)
        self.assertFalse(
            immediate["captured_route_forward_no_later_than_output_divergence"]
        )
        preceding = MODULE._align_route_to_output([4, 8], 6)
        self.assertEqual(preceding["route_forward_produced_output_token_index"], 5)
        self.assertTrue(
            preceding["captured_route_forward_no_later_than_output_divergence"]
        )


if __name__ == "__main__":
    unittest.main()
