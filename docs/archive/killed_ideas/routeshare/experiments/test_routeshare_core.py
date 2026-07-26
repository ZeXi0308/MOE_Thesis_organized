from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PATH = Path(__file__).with_name("routeshare_core.py")
SPEC = importlib.util.spec_from_file_location("routeshare_core", PATH)
assert SPEC and SPEC.loader
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


class RouteShareCoreTests(unittest.TestCase):
    def test_route_generation_closes_identity_and_topk(self) -> None:
        scenario = core.make_scenario(
            split="calibration",
            tokens_per_tenant=16,
            top_k=8,
            num_experts=64,
            overlap_fraction=0.5,
            histogram_regime="skewed",
            seed=3,
        )
        self.assertEqual(scenario.selected_experts.shape, (32, 8))
        self.assertTrue(
            all(len(np.unique(row)) == 8 for row in scenario.selected_experts)
        )
        self.assertEqual(set(np.unique(scenario.tenant_ids)), {0, 1})

    def test_overlap_changes_union_without_changing_rows(self) -> None:
        common = dict(
            split="calibration",
            tokens_per_tenant=32,
            top_k=8,
            num_experts=64,
            histogram_regime="balanced",
            seed=1,
        )
        low = core.make_scenario(overlap_fraction=0.0, **common)
        high = core.make_scenario(overlap_fraction=1.0, **common)
        low_features = core.scenario_features(low)
        high_features = core.scenario_features(high)
        self.assertEqual(low_features["total_rows"], high_features["total_rows"])
        self.assertGreater(low_features["active_experts"], high_features["active_experts"])

    def test_row_bin_accounting_equals_active_experts(self) -> None:
        scenario = core.make_scenario(
            split="sealed",
            tokens_per_tenant=64,
            top_k=16,
            num_experts=32,
            overlap_fraction=1.0,
            histogram_regime="balanced",
            seed=100,
        )
        features = core.scenario_features(scenario)
        binned = sum(int(features[f"experts_bin_{label}"]) for label in core.ROW_BINS)
        self.assertEqual(binned, features["active_experts"])

    def test_llm_jp_histogram_intervention_is_not_silently_identical(self) -> None:
        common = dict(
            split="calibration",
            tokens_per_tenant=32,
            top_k=16,
            num_experts=32,
            overlap_fraction=1.0,
            seed=2,
        )
        balanced = core.make_scenario(histogram_regime="balanced", **common)
        skewed = core.make_scenario(histogram_regime="skewed", **common)
        balanced_features = core.scenario_features(balanced)
        skewed_features = core.scenario_features(skewed)
        self.assertEqual(balanced_features["active_experts"], 32)
        self.assertEqual(skewed_features["active_experts"], 32)
        self.assertNotEqual(
            balanced_features["row_count_cv"], skewed_features["row_count_cv"]
        )

    def test_cost_model_fit_and_gap_recovery(self) -> None:
        rows = []
        for active in range(2, 12):
            row = {
                "total_rows": 128.0,
                "active_experts": float(active),
                **{f"experts_bin_{label}": 0.0 for label in core.ROW_BINS},
            }
            row["experts_bin_9-16"] = float(active)
            row["coalition_latency_ms"] = 1.0 + 0.2 * active
            rows.append(row)
        beta = core.fit_linear_cost(rows, "m1_rows_active")
        prediction = core.predict_linear_cost(rows, "m1_rows_active", beta)
        truth = np.asarray([row["coalition_latency_ms"] for row in rows])
        self.assertLess(float(np.abs(prediction - truth).max()), 1e-10)
        recovery = core.squared_error_gap_recovery(
            truth,
            np.full_like(truth, truth.mean()),
            prediction,
        )
        self.assertAlmostEqual(recovery, 1.0)


if __name__ == "__main__":
    unittest.main()
