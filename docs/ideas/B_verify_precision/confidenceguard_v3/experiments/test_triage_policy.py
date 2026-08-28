from __future__ import annotations

import unittest

import numpy as np

from triage_policy import (
    AuditState,
    FEATURE_NAMES,
    FrozenConfidenceGuard,
    FrozenRidgeTriage,
    TriagePolicyError,
    audit_phase,
    budget_matched_hash_periods,
    calibration_stability,
    confidence_guard_stability,
    common_audit_phase,
    cvar,
    fit_frozen_ridge,
    hash_control_period,
)


def row(offset: float) -> dict[str, float]:
    return {name: float(index + offset) for index, name in enumerate(FEATURE_NAMES)}


class TriagePolicyTests(unittest.TestCase):
    def test_cvar_is_worst_tail(self) -> None:
        self.assertEqual(cvar([1.0, 2.0, 100.0], 0.1), 100.0)
        self.assertEqual(cvar([1.0, 2.0, 3.0, 4.0], 0.5), 3.5)

    def test_ridge_roundtrip_and_periods(self) -> None:
        rows = [row(float(i)) for i in range(12)]
        labels = [10.0 ** (i / 8.0) for i in range(12)]
        model = fit_frozen_ridge(rows, labels)
        restored = FrozenRidgeTriage.from_dict(model.to_dict())
        self.assertAlmostEqual(model.score(rows[3]), restored.score(rows[3]))
        self.assertEqual(model.period(rows[0]), 8)
        self.assertEqual(model.period(rows[-1]), 2)

    def test_ridge_rejects_string_numeric_and_wrong_order(self) -> None:
        rows = [row(float(i)) for i in range(12)]
        model = fit_frozen_ridge(rows, [float(i + 1) for i in range(12)]).to_dict()
        model["intercept"] = "0.0"
        with self.assertRaises(TriagePolicyError):
            FrozenRidgeTriage.from_dict(model)

    def test_hash_controls_are_deterministic(self) -> None:
        digest = "a" * 64
        self.assertEqual(hash_control_period(digest), hash_control_period(digest))
        self.assertEqual(audit_phase("triage", digest, 8), audit_phase("triage", digest, 8))
        self.assertEqual(common_audit_phase(digest, 8), common_audit_phase(digest, 8))

    def test_budget_matched_hash_preserves_exact_period_multiset(self) -> None:
        documents = [f"{index:064x}" for index in range(1, 7)]
        periods = [2, 2, 4, 4, 8, 8]
        mapped = budget_matched_hash_periods(
            documents, periods, model_key="olmoe", split="sealed"
        )
        self.assertEqual(set(mapped), set(documents))
        self.assertEqual(sorted(mapped.values()), sorted(periods))

    def test_calibration_stability_reports_probabilities(self) -> None:
        rows = [row(float(i)) for i in range(16)]
        labels = [10.0 ** (i / 8.0) for i in range(16)]
        result = calibration_stability(
            rows, labels, alpha=1.0, repeats=100, seed=7
        )
        self.assertEqual(len(result["assignment_probabilities"]), 16)
        self.assertGreater(result["spearman_point"], 0.9)

    def test_confidence_guard_roundtrip_and_binary_stability(self) -> None:
        rows = [row(float(i)) for i in range(16)]
        labels = [10.0 ** (i / 8.0) for i in range(16)]
        result = confidence_guard_stability(
            rows,
            labels,
            alpha=1.0,
            repeats=100,
            seed=9,
            safe_probability_min=0.8,
            risk_probability_max=0.2,
        )
        guard = FrozenConfidenceGuard.from_dict(result["frozen_guard"])
        restored = FrozenConfidenceGuard.from_dict(guard.to_dict())
        self.assertEqual(len(result["binary_assignment_probabilities"]), 16)
        self.assertEqual(len(result["calibration_safe_probabilities"]), 16)
        self.assertEqual(guard.period(rows[0]), restored.period(rows[0]))
        self.assertIn(guard.period(rows[0]), {2, 4, 8})
        self.assertGreater(result["spearman_point"], 0.9)

    def test_audit_accounts_both_branches_and_lockout(self) -> None:
        state = AuditState(period=4, phase=0, lockout_following_steps=3)
        self.assertEqual(state.decision(0), "audit")
        self.assertEqual(state.record_audit(1.0, 0.5), "high")
        for step in range(1, 4):
            self.assertEqual(state.decision(step), "lockout_high")
            state.record_single("lockout_high")
        counters = state.counters()
        self.assertEqual(counters["audit_events"], 1)
        self.assertEqual(counters["high_forward_calls"], 4)
        self.assertEqual(counters["low_forward_calls"], 1)
        self.assertEqual(counters["cache_clone_events"], 2)
        self.assertEqual(counters["served_high_steps"], 4)

    def test_max_unaudited_interval_forces_audit(self) -> None:
        state = AuditState(period=8, phase=7, max_unaudited_steps=8)
        for step in range(7):
            self.assertEqual(state.decision(step), "low")
            state.record_single("low")
        self.assertEqual(state.decision(7), "audit")

    def test_missing_or_nonfinite_features_fail(self) -> None:
        rows = [row(float(i)) for i in range(12)]
        broken = dict(rows[0])
        broken.pop(FEATURE_NAMES[0])
        rows[0] = broken
        with self.assertRaises(TriagePolicyError):
            fit_frozen_ridge(rows, np.ones(12))


if __name__ == "__main__":
    unittest.main()
