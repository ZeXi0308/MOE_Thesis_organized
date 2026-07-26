from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


PATH = Path(__file__).with_name("analyze_routeshare_gate0.py")
SPEC = importlib.util.spec_from_file_location("analyze_routeshare_gate0", PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class RouteShareAnalysisTests(unittest.TestCase):
    def test_null_histogram_intervention_is_excluded(self) -> None:
        base = {
            "model_key": "llm_jp",
            "layer_id": 0,
            "tokens_per_tenant": 8,
            "overlap_fraction": 0.0,
            "seed": 100,
            "total_rows": 256,
            "active_experts": 32,
            "max_rows_per_expert": 8,
            "row_count_cv": 0.0,
            "experts_bin_5-8": 32,
        }
        rows = pd.DataFrame(
            [
                {**base, "histogram_regime": "balanced", "route_sha256": "a", "coalition_latency_ms": 2.0},
                {**base, "histogram_regime": "skewed", "route_sha256": "b", "coalition_latency_ms": 2.2},
            ]
        )
        self.assertEqual(len(analysis.matched_histogram_contrasts(rows)), 0)

    def test_real_histogram_intervention_is_retained(self) -> None:
        base = {
            "model_key": "olmoe",
            "layer_id": 0,
            "tokens_per_tenant": 8,
            "overlap_fraction": 1.0,
            "seed": 100,
            "total_rows": 128,
            "active_experts": 16,
            "experts_bin_5-8": 16,
        }
        rows = pd.DataFrame(
            [
                {**base, "histogram_regime": "balanced", "route_sha256": "a", "max_rows_per_expert": 8, "row_count_cv": 0.1, "coalition_latency_ms": 2.0},
                {**base, "histogram_regime": "skewed", "route_sha256": "b", "max_rows_per_expert": 20, "row_count_cv": 0.8, "coalition_latency_ms": 2.4},
            ]
        )
        values = analysis.matched_histogram_contrasts(rows)
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(float(values[0]), 0.4 / 2.2)


if __name__ == "__main__":
    unittest.main()
