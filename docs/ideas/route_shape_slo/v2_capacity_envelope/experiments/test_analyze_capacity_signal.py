#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
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


analysis = load_module(HERE / "analyze_capacity_signal.py", "analyze_capacity_signal")


def windows(shared_identity: bool = False):
    rows = []
    for episode_index, (episode, regime) in enumerate(
        (("steady", "steady"), ("bursty", "bursty"))
    ):
        for index in range(10):
            route = float((index * 3 + episode_index) % 7 + 1)
            identity = "shared" if shared_identity else f"{episode}-request"
            rows.append(
                {
                    "model": "allenai/OLMoE-1B-7B-0924",
                    "model_revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
                    "episode_id": episode,
                    "arrival_regime": regime,
                    "window_id": f"{episode}-{index}",
                    "window_start_us": str(index * 100),
                    "window_end_us": str(index * 100 + 50),
                    "feature_available_at_us": str(index * 100 + 50),
                    "request_ids": json.dumps([identity]),
                    "document_ids": json.dumps([f"{identity}-document"]),
                    "step_service_ms": str(8.0 + 0.3 * index + 0.4 * route),
                    "serial_route_conformance": (
                        "PASS_EXPERT_ASSIGNMENT_MULTISET"
                    ),
                    "serial_route_identity_match_fraction": "1.0",
                    "batch_dependent_route_observed": "false",
                    "custom_waiting_count": str(episode_index * 2),
                    "running_sequences": "4",
                    "arrived_active_sequences": str(4 + episode_index * 2),
                    "active_tokens": "4",
                    "batch_tokens": "4",
                    "mean_logical_kv": str(32 + index),
                    "max_logical_kv": str(40 + index),
                    "mean_physical_kv": str(40 + index),
                    "max_physical_kv": str(40 + index),
                    "left_padding_ratio": "0.1",
                    "arrival_count": str(1 + episode_index),
                    "arrival_rate_per_s": str(20 + 5 * episode_index),
                    "decode_step": str(index),
                    "route_max_expert_load": str(route),
                    "route_max_mean": str(1.0 + route / 10),
                    "route_cv": str(route / 20),
                    "active_experts": "24",
                    "route_hhi": str(0.03 + route / 1000),
                    "top1_share": str(0.05 + route / 100),
                    "hotspot_persistence": str((index % 3) / 2),
                    "cross_layer_max_pressure": str(1.0 + route / 8),
                    "cross_layer_mean_pressure": str(1.0 + route / 12),
                    "route_shape_ewma": str(1.0 + route / 15),
                    "route_shape_delta": str((route - 4) / 10),
                    "expert_identity_turnover": str((index + 1) % 2),
                    "top1_share_persistence": "0.75",
                    "serial_route_conformance": (
                        "PASS_EXPERT_ASSIGNMENT_MULTISET"
                    ),
                    "serial_route_identity_match_fraction": "1.0",
                    "batch_dependent_route_observed": "false",
                }
            )
    return rows


class CapacitySignalTest(unittest.TestCase):
    CONFIG = {
        "analysis_contract": {
            "model": {
                "id": "allenai/OLMoE-1B-7B-0924",
                "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
            },
            "target_quantile": 0.95,
            "dangerous_underprediction_margin_fraction": 0.10,
            "quantile_l2_alpha": 1.0,
            "ridge_alpha": 1.0,
            "signal_thresholds": {
                "promising_pinball_improvement": 0.03,
                "promising_dangerous_reduction": 0.10,
                "weak_pinball_improvement": 0.01,
            },
        },
        "action": {
            "candidate_if_stage_d_is_authorized": "running_set_budget",
            "stage_d_authorized": False,
        },
        "telemetry_overhead_contract": {"max_relative_overhead": 0.02},
    }
    OVERHEAD = {
        "schema": "route-capacity-envelope-telemetry-overhead-v1",
        "status": "TELEMETRY_OVERHEAD_OK",
        "model": {
            "id": "allenai/OLMoE-1B-7B-0924",
            "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
            "dtype": "bfloat16",
        },
        "token_output_match": True,
        "logit_output_match": True,
        "on_route_trace_stable": True,
        "completion_trace_match": True,
        "same_requests": True,
        "same_arrival_trace": False,
        "arrival_policy_applied_to_timing": False,
        "same_seed": True,
        "same_batch_schedule": True,
        "same_decode_steps": True,
        "same_dtype": True,
        "model_call_relative_overhead": 0.01,
        "loop_wall_relative_overhead": 0.01,
        "max_relative_overhead": 0.02,
    }

    def test_causal_cutoff_rejects_future_available_feature(self) -> None:
        rows = windows()
        rows[0]["feature_available_at_us"] = "51"
        with self.assertRaisesRegex(analysis.AnalysisError, "after window t"):
            analysis.align_next_window(rows)

    def test_episode_split_rejects_request_document_overlap(self) -> None:
        aligned = analysis.align_next_window(windows(shared_identity=True))
        with self.assertRaisesRegex(analysis.AnalysisError, "overlaps"):
            analysis.validate_episode_split(aligned)

    def test_m2_is_mandatory_expert_load_baseline_and_m4_is_future_only(self) -> None:
        self.assertEqual(
            self.CONFIG["action"]["candidate_if_stage_d_is_authorized"],
            "running_set_budget",
        )
        self.assertFalse(self.CONFIG["action"]["stage_d_authorized"])
        aligned = analysis.align_next_window(windows())
        result = analysis.analyze(aligned, self.CONFIG, self.OVERHEAD)
        methods = result["method_contract"]
        self.assertTrue(
            set(methods["M1_workload_only"])
            < set(methods["M2_workload_plus_expert_load"])
        )
        self.assertTrue(
            set(methods["M2_workload_plus_expert_load"])
            < set(methods["M3_workload_expert_load_plus_historical_route"])
        )
        self.assertFalse(
            any(name.startswith("future_") for name in methods["M3_workload_expert_load_plus_historical_route"])
        )
        self.assertTrue(
            any(name.startswith("future_") for name in methods["M4_future_route_latency_diagnostic"])
        )
        self.assertEqual(
            set(methods["M3_workload_expert_load_plus_historical_route"])
            - set(methods["M2_workload_plus_expert_load"]),
            set(analysis.HISTORICAL_ROUTE_FEATURES),
        )
        self.assertTrue(
            set(methods["M3_workload_expert_load_plus_historical_route"])
            < set(methods["M4_future_route_latency_diagnostic"])
        )
        self.assertIn(result["verdict"], {
            "PROMISING_SINGLE_MODEL",
            "WEAK_SIGNAL_FOLD_INTO_DEPA",
            "USE_TOKEN_OR_EXPERT_LOAD_CONTROLLER",
        })

    def test_batch_dependent_route_overrides_signal_interpretation(self) -> None:
        raw = windows()
        for row in raw:
            row["serial_route_conformance"] = "BATCH_DEPENDENT"
            row["serial_route_identity_match_fraction"] = "0.98"
            row["batch_dependent_route_observed"] = "true"
        result = analysis.analyze(
            analysis.align_next_window(raw), self.CONFIG, self.OVERHEAD
        )
        self.assertEqual(result["verdict"], "PIVOT_TO_EXECUTION_CONFORMANCE")
        self.assertFalse(result["action_oracle_authorized"])

    def test_p95_quantile_fit_reaches_known_lp_optimum(self) -> None:
        x = np.asarray(
            [
                [0.126, -0.132], [0.640, 0.105], [-0.536, 0.362],
                [1.304, 0.947], [-0.704, -1.265], [-0.623, 0.041],
                [-2.325, -0.219], [-1.246, -0.732],
            ],
            dtype=float,
        )
        y = np.asarray(
            [2.385, 2.661, 2.049, 2.357, 2.118, 2.678, 4.238, 3.004],
            dtype=float,
        )
        coefficients = analysis.fit_quantile_linear(x, y, 0.95)
        residual = y - analysis.predict_linear(coefficients, x)
        objective = np.where(
            residual >= 0,
            0.95 * residual,
            0.05 * -residual,
        ).sum()
        self.assertAlmostEqual(float(objective), 0.17408122085074887, places=8)


if __name__ == "__main__":
    unittest.main()
