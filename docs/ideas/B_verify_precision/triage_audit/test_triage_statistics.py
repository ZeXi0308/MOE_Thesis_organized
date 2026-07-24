from __future__ import annotations

import unittest

from triage_statistics import StatisticsError, analyze_model, cross_model_decision, holm_adjust


def synthetic_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(32):
        digest = f"{index + 1:064x}"
        for policy in ("triage_2_4_8", "hash_budget_matched_2_4_8", "fixed_2", "fixed_4", "fixed_8"):
            quality = {"triage_2_4_8": 0.8, "hash_budget_matched_2_4_8": 1.0, "fixed_2": 0.7, "fixed_4": 0.9, "fixed_8": 1.2}[policy]
            calls = {"triage_2_4_8": 35, "hash_budget_matched_2_4_8": 40, "fixed_2": 50, "fixed_4": 42, "fixed_8": 34}[policy]
            rows.append({
                "policy": policy,
                "text_sha256": digest,
                "document_cvar90_kl": quality + index * 1e-4,
                "total_candidate_forward_calls": calls,
                "dangerous_step_recall": 0.9 if policy == "triage_2_4_8" else 0.88,
                "threshold_violation_fraction": 0.02 if policy == "triage_2_4_8" else 0.03,
            })
    return rows


class StatisticsTests(unittest.TestCase):
    def test_holm_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
        self.assertLessEqual(adjusted["a"], adjusted["b"])
        self.assertLessEqual(adjusted["b"], adjusted["c"])

    def test_positive_fixture_passes(self) -> None:
        result = analyze_model(synthetic_rows(), bootstrap_repeats=100, seed=3)
        self.assertTrue(result["model_go"])
        decision = cross_model_decision({"olmoe": result, "llmjp": result})
        self.assertTrue(decision["go"])

    def test_missing_policy_fails(self) -> None:
        rows = [row for row in synthetic_rows() if row["policy"] != "fixed_8"]
        with self.assertRaises(StatisticsError):
            analyze_model(rows, bootstrap_repeats=100, seed=3)

    def test_zero_over_zero_quality_is_neutral(self) -> None:
        rows = synthetic_rows()
        for row in rows:
            if row["policy"] in {"triage_2_4_8", "hash_budget_matched_2_4_8"}:
                row["document_cvar90_kl"] = 0.0
        result = analyze_model(rows, bootstrap_repeats=100, seed=3)
        self.assertEqual(result["intervals"]["quality_ratio_vs_hash"]["point"], 1.0)

    def test_out_of_range_probability_fails(self) -> None:
        rows = synthetic_rows()
        rows[0]["dangerous_step_recall"] = 1.1
        with self.assertRaises(StatisticsError):
            analyze_model(rows, bootstrap_repeats=100, seed=3)


if __name__ == "__main__":
    unittest.main()
