from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("run_historical_budget_screen.py")
SPEC = importlib.util.spec_from_file_location("bridge_screen", MODULE_PATH)
assert SPEC and SPEC.loader
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class HistoricalBudgetScreenTests(unittest.TestCase):
    def test_upper_cvar_uses_document_tail(self) -> None:
        values = np.arange(1.0, 11.0)
        self.assertAlmostEqual(bridge.upper_cvar(values, 0.20), 9.5)

    def test_ridge_predict_handles_constant_and_collinear_columns(self) -> None:
        fit_x = np.array(
            [[1.0, 2.0, 7.0], [2.0, 4.0, 7.0], [3.0, 6.0, 7.0], [4.0, 8.0, 7.0]]
        )
        fit_y = np.array([1.0, 2.0, 3.0, 4.0])
        target_x = np.array([[5.0, 10.0, 7.0]])
        prediction = bridge.ridge_predict(fit_x, fit_y, target_x, alpha=0.0)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertAlmostEqual(float(prediction[0]), 5.0, places=8)

    def test_top_budget_protection_beats_random_for_perfect_scores(self) -> None:
        harm = np.arange(1.0, 41.0)
        scores = harm.copy()
        rows = pd.DataFrame(
            bridge.evaluate_policy(
                "arrival_same_prompt",
                "arrival_lexical",
                harm,
                scores,
                budgets=[0.25],
                point_random_trials=1000,
                bootstrap=100,
                bootstrap_random_trials=32,
                seed=7,
            )
        )
        cvar = rows[rows["metric"] == "cvar90"].iloc[0]
        self.assertEqual(cvar["protected_count"], 10)
        self.assertLess(cvar["predicted_metric"], cvar["random_metric_mean"])
        self.assertAlmostEqual(cvar["oracle_headroom_recovery"], 1.0)

    def test_zero_information_has_no_systematic_headroom(self) -> None:
        rng = np.random.default_rng(11)
        harm = np.linspace(0.01, 1.0, 80)
        scores = rng.normal(size=80)
        rows = pd.DataFrame(
            bridge.evaluate_policy(
                "arrival_same_prompt",
                "arrival_lexical",
                harm,
                scores,
                budgets=[0.25],
                point_random_trials=2000,
                bootstrap=100,
                bootstrap_random_trials=32,
                seed=13,
            )
        )
        cvar = rows[rows["metric"] == "cvar90"].iloc[0]
        self.assertLess(cvar["relative_reduction_ci_low"], 0.10)

    def test_validation_rejects_duplicate_ids_and_nonfinite(self) -> None:
        columns = bridge.FEATURE_GROUPS["arrival_lexical"]
        frame = pd.DataFrame(
            {
                "sample_id": [1, 1],
                "label": [0.1, 0.2],
                **{column: [1.0, 2.0] for column in columns},
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bridge.validate_frame(frame, columns, "label", "bad")
        frame["sample_id"] = [1, 2]
        frame.loc[0, columns[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bridge.validate_frame(frame, columns, "label", "bad")

    def test_locus_gate_requires_two_passing_budget_points(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "locus": "arrival_same_prompt",
                    "feature_group": "arrival_lexical",
                    "metric": "cvar90",
                    "budget_fraction": budget,
                    "relative_reduction_ci_low": lcb,
                    "oracle_headroom_recovery": recovery,
                    "predicted_metric": predicted,
                }
                for budget, lcb, recovery, predicted in [
                    (0.10, 0.11, 0.31, 1.0),
                    (0.25, 0.12, 0.35, 0.8),
                    (0.50, 0.05, 0.50, 0.6),
                ]
            ]
        )
        passed, budgets = bridge.locus_pass(frame, "arrival_same_prompt")
        self.assertTrue(passed)
        self.assertEqual(budgets, [0.10, 0.25])


if __name__ == "__main__":
    unittest.main()
