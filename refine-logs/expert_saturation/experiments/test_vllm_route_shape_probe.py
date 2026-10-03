from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_vllm_route_shape_probe.py")
SPEC = importlib.util.spec_from_file_location("route_shape_probe", MODULE_PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

COMPARE_PATH = Path(__file__).with_name("compare_vllm_route_probe_runs.py")
COMPARE_SPEC = importlib.util.spec_from_file_location("compare_route_probe", COMPARE_PATH)
assert COMPARE_SPEC and COMPARE_SPEC.loader
compare = importlib.util.module_from_spec(COMPARE_SPEC)
COMPARE_SPEC.loader.exec_module(compare)


class RouteShapeMetricTest(unittest.TestCase):
    def test_concentrated_routes_have_higher_pressure(self) -> None:
        balanced = np.array(
            [
                [[0, 1], [2, 3]],
                [[2, 3], [0, 1]],
            ]
        )
        concentrated = np.zeros_like(balanced)

        balanced_metrics = probe.summarize_routes([balanced, balanced], 4)
        concentrated_metrics = probe.summarize_routes([concentrated, concentrated], 4)

        self.assertLess(
            balanced_metrics["max_layer_step_concentration"],
            concentrated_metrics["max_layer_step_concentration"],
        )
        self.assertGreater(
            balanced_metrics["mean_normalized_entropy"],
            concentrated_metrics["mean_normalized_entropy"],
        )
        self.assertGreater(
            balanced_metrics["mean_layer_working_set_fraction"],
            concentrated_metrics["mean_layer_working_set_fraction"],
        )
        self.assertEqual(
            concentrated_metrics["per_request_exact_route_match_fraction"], 1.0
        )
        self.assertEqual(
            balanced_metrics["per_request_exact_route_match_fraction"], 0.0
        )

    def test_shape_and_expert_range_fail_closed(self) -> None:
        good = np.zeros((2, 2, 2), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            probe.summarize_routes([good, np.zeros((3, 2, 2))], 4)
        bad = good.copy()
        bad[0, 0, 0] = 4
        with self.assertRaisesRegex(ValueError, "outside"):
            probe.summarize_routes([bad], 4)

    def test_cell_summary_selects_observational_extremes(self) -> None:
        rows = []
        for index, concentration in enumerate((0.25, 0.50, 0.75)):
            rows.append(
                {
                    "batch_id": f"b{index}",
                    "prompt_length": 128,
                    "batch_size": 4,
                    "timing": {"request_tpot_p95_ms": 2.0 + index},
                    "route": {
                        "max_layer_step_concentration": concentration,
                        "mean_layer_working_set_fraction": 0.5 - index * 0.1,
                    },
                }
            )
        summary = probe.build_cell_summaries(rows)[0]
        self.assertEqual(summary["low_concentration_batch_id"], "b0")
        self.assertEqual(summary["high_concentration_batch_id"], "b2")
        self.assertAlmostEqual(summary["observational_high_minus_low_tpot_pct"], 100.0)

    def test_telemetry_comparison_fails_on_token_drift(self) -> None:
        config = {field: 1 for field in compare.MATCHED_CONFIG_FIELDS}
        config.update(
            {
                "prompt_lengths": [128],
                "batch_sizes": [4],
                "groups": 1,
                "within_process_repeats": 1,
            }
        )
        config["capture_routes"] = False
        row = {
            "prompt_length": 128,
            "batch_size": 4,
            "group": 0,
            "within_process_repeat": 0,
            "prompt_token_ids_sha256": "same",
            "request_metrics": [{"token_ids": [1, 2]}],
            "timing": {"wall_ms": 10.0, "request_tpot_p95_ms": 2.0},
        }
        on_config = dict(config, capture_routes=True)
        on_row = dict(row, request_metrics=[{"token_ids": [1, 3]}])
        report = compare.compare_runs(on_config, config, [on_row], [row], 5.0)
        self.assertEqual(report["status"], "INVALID_TELEMETRY_PAIR")
        self.assertFalse(report["token_parity"])

    def test_telemetry_comparison_rejects_identically_partial_runs(self) -> None:
        config = {field: 1 for field in compare.MATCHED_CONFIG_FIELDS}
        config.update(
            {
                "prompt_lengths": [128],
                "batch_sizes": [4],
                "groups": 2,
                "within_process_repeats": 1,
                "capture_routes": False,
            }
        )
        row = {
            "prompt_length": 128,
            "batch_size": 4,
            "group": 0,
            "within_process_repeat": 0,
            "prompt_token_ids_sha256": "same",
            "request_metrics": [{"token_ids": [1, 2]}],
            "timing": {"wall_ms": 10.0, "request_tpot_p95_ms": 2.0},
        }
        report = compare.compare_runs(
            dict(config, capture_routes=True), config, [row], [row], 5.0
        )
        self.assertEqual(report["status"], "INVALID_TELEMETRY_PAIR")
        self.assertEqual(report["incomplete_on"], [[128, 4, 1, 0]])

    def test_timing_denominator_fails_closed_on_short_generation(self) -> None:
        completion = SimpleNamespace(token_ids=[1, 2, 3], finish_reason="length")
        metrics = SimpleNamespace(
            num_generation_tokens=3,
            last_token_ts=4.0,
            first_token_ts=1.0,
            first_token_latency=0.5,
            scheduled_ts=1.0,
            queued_ts=0.5,
        )
        output = SimpleNamespace(
            request_id="request-0", metrics=metrics, outputs=[completion]
        )
        with self.assertRaisesRegex(ValueError, "denominator"):
            probe.summarize_timings([output], 1.0, expected_output_tokens=4)


if __name__ == "__main__":
    unittest.main()
