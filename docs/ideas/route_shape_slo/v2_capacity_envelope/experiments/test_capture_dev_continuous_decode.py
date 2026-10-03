#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wrapper = load_module(
    HERE / "capture_dev_continuous_decode.py", "capture_dev_continuous_decode"
)
comparator = load_module(
    HERE / "compare_serial_batched_router_logits.py",
    "compare_serial_batched_router_logits",
)


class AssignmentAuditTest(unittest.TestCase):
    def test_topk_order_is_ignored_but_expert_assignment_is_not(self) -> None:
        expected = [
            {"layer": 7, "selected_experts": np.asarray([[45, 1, 48, 23, 39, 56, 21, 41]])}
        ]
        permuted = [
            {"layer": 7, "selected_experts": np.asarray([[45, 1, 48, 23, 56, 39, 21, 41]])}
        ]
        changed = [
            {"layer": 7, "selected_experts": np.asarray([[45, 1, 48, 23, 56, 40, 21, 41]])}
        ]
        signature = wrapper.assignment_route_signature
        self.assertEqual(signature(expected, 0), signature(permuted, 0))
        self.assertNotEqual(signature(expected, 0), signature(changed, 0))

    def test_router_logit_summary_keeps_token_and_topk_boundaries_separate(self) -> None:
        token = {
            "request_id": "req-0",
            "decode_step": 0,
            "input_token_id": 11,
            "predicted_next_token_id": 12,
        }
        base = {
            "tokens": [token],
            "router": [
                {
                    **token,
                    "layer": 0,
                    "router_logits": [3.0, 2.0, 1.0],
                    "selected_experts": [0, 1],
                    "topk_margins": {
                        "top1_minus_top2": 1.0,
                        "within_selected": [1.0],
                        "selection_boundary": 1.0,
                    },
                }
            ],
        }
        changed = {
            "tokens": [dict(token)],
            "router": [
                {
                    **token,
                    "layer": 0,
                    "router_logits": [3.0, 0.9, 1.1],
                    "selected_experts": [0, 2],
                    "topk_margins": {
                        "top1_minus_top2": 1.9,
                        "within_selected": [1.9],
                        "selection_boundary": 0.2,
                    },
                }
            ],
        }
        summary = comparator.summarize_trace_pair(base, changed)
        self.assertTrue(summary["tokens"]["full_token_parity"])
        self.assertEqual(
            summary["expert_assignment"]["multiset_match_fraction"], 0.0
        )
        self.assertEqual(
            summary["expert_assignment"][
                "swapped_expert_order_crossing_coverage"
            ],
            1.0,
        )
        self.assertEqual(
            summary["expert_assignment"][
                "material_swapped_expert_gap_change_coverage"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            summary["topk_margins"]["selection_boundary"]["max_abs_delta"],
            0.8,
        )

    def test_unrelated_logit_drift_does_not_explain_assignment_change(self) -> None:
        common = {
            "request_id": "req-0",
            "decode_step": 0,
            "layer": 0,
            "input_token_id": 11,
            "predicted_next_token_id": 12,
            "selection_boundary_logit_margin": 1.0,
        }
        comparison = comparator.compare_records(
            [
                {
                    **common,
                    "router_logits": [3.0, 2.0, 1.0, 0.0],
                    "selected_experts": [0, 1],
                }
            ],
            [
                {
                    **common,
                    # Only unrelated expert 3 moves.  The declared 1 -> 2
                    # assignment swap is therefore not explained by logits.
                    "router_logits": [3.0, 2.0, 1.0, 0.5],
                    "selected_experts": [0, 2],
                }
            ],
            logit_atol=1e-6,
            near_tie_margin=1e-2,
            include_logits=False,
        )
        self.assertEqual(comparison["material_logit_difference_rows"], 1)
        self.assertEqual(
            comparison["swapped_expert_order_crossing_coverage"], 0.0
        )
        self.assertEqual(
            comparator.classify(comparison, stable=True),
            "INCONCLUSIVE_ASSIGNMENT_NOT_EXPLAINED_BY_SWAPPED_LOGITS",
        )


if __name__ == "__main__":
    unittest.main()
