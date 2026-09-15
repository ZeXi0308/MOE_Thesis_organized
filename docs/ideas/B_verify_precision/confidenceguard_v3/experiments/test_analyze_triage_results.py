from __future__ import annotations

import unittest

from types import SimpleNamespace

import torch

from analyze_triage_results import AnalysisError, analyze, validate_complete_raw_results
from test_triage_statistics import synthetic_rows
from triage_executor import execute_policy_trajectory
from triage_policy import FEATURE_NAMES, FrozenConfidenceGuard, common_audit_phase, fit_frozen_ridge


class AnalyzeResultsTests(unittest.TestCase):
    def test_cross_model_decision(self) -> None:
        rows = []
        for model_key in ("olmoe", "llmjp"):
            rows.extend({**row, "model_key": model_key, "split": "sealed"} for row in synthetic_rows())
        config = {
            "seed": 4,
            "evidence_boundary": "test",
            "dataset": {"sealed_documents": 32},
            "statistics": {
                "bootstrap_replicates": 100,
                "effect_estimator": "median_of_document_level_paired_effects",
                "pareto_effect_estimator": "policy_level_mean_points_rebuilt_per_document_bootstrap",
            },
        }
        self.assertTrue(analyze(rows, config)["cross_model"]["go"])

    def test_rejects_wrong_split(self) -> None:
        rows = [{**row, "model_key": "olmoe", "split": "calibration"} for row in synthetic_rows()]
        config = {
            "seed": 4,
            "dataset": {"sealed_documents": 32},
            "statistics": {
                "bootstrap_replicates": 100,
                "effect_estimator": "median_of_document_level_paired_effects",
                "pareto_effect_estimator": "policy_level_mean_points_rebuilt_per_document_bootstrap",
            },
        }
        with self.assertRaises(AnalysisError):
            analyze(rows, config)

    def test_integrity_gate_rejects_missing_arms(self) -> None:
        config = {"dataset": {"sealed_documents": 1, "decode_steps": 2}}
        lock = {
            "schema_version": "triage-calibration-lock-v2",
            "models": {"olmoe": {"audit_threshold": 0.1}, "llmjp": {"audit_threshold": 0.1}},
        }
        with self.assertRaises(AnalysisError):
            validate_complete_raw_results([], config, lock)

    def test_integrity_gate_accepts_executor_generated_rows(self) -> None:
        digest = "1" * 64
        config = {
            "dataset": {"sealed_documents": 1, "decode_steps": 2},
            "controller": {"max_unaudited_steps": 8, "lockout_following_steps": 1},
        }
        lock = {
            "schema_version": "triage-calibration-lock-v2",
            "models": {"olmoe": {"audit_threshold": 0.1}, "llmjp": {"audit_threshold": 0.1}},
        }

        def forward(offset: float):
            def call(token, cache):
                old = cache[0][0]
                new = torch.cat([old, torch.full((1, 1, 1, 2), offset)], dim=-2)
                return SimpleNamespace(
                    logits=torch.tensor([[[offset, -offset]]]),
                    past_key_values=((new,),),
                )
            return call

        all_rows = []
        for model_key in ("olmoe", "llmjp"):
            all_rows.append({
                "model_key": model_key,
                "split": "sealed",
                "text_sha256": digest,
                "policy": "always_bf16",
                "period": None,
                "phase": None,
                "document_cvar90_kl": 0.0,
                "physical_low_forward_calls": 0,
                "steps": [],
            })
            for policy, period in (
                ("always_low", None),
                ("triage_2_4_8", 2),
                ("hash_budget_matched_2_4_8", 2),
                ("fixed_2", 2),
                ("fixed_4", 4),
                ("fixed_8", 8),
                ("full_shadow", 1),
            ):
                phase = common_audit_phase(digest, period) if period is not None else None
                summary, steps = execute_policy_trajectory(
                    policy=policy,
                    initial_cache=((torch.zeros(1, 1, 1, 2),),),
                    decode_tokens=torch.ones(1, 2, dtype=torch.long),
                    reference_logits=torch.tensor([[1.0, -1.0]] * 2),
                    high_forward=forward(1.0),
                    low_forward=forward(-1.0),
                    discrepancy_threshold=0.1,
                    period=period,
                    phase=phase,
                    lockout_following_steps=0 if policy == "full_shadow" else 1,
                )
                all_rows.append({
                    "model_key": model_key,
                    "split": "sealed",
                    "text_sha256": digest,
                    "policy": policy,
                    "period": period,
                    "phase": phase,
                    **summary,
                    "steps": steps,
                })
        validate_complete_raw_results(all_rows, config, lock)

        feature_row = {name: float(index) for index, name in enumerate(FEATURE_NAMES)}
        calibration_rows = [
            {name: value + float(offset) for name, value in feature_row.items()}
            for offset in range(12)
        ]
        model = fit_frozen_ridge(calibration_rows, [float(index + 1) for index in range(12)])
        safe_cut = model.score(feature_row) - 1.0
        guard = FrozenConfidenceGuard(
            point_model=model,
            bootstrap_models=tuple(model for _ in range(100)),
            safe_cuts=tuple(safe_cut for _ in range(100)),
            safe_probability_min=0.8,
            risk_probability_max=0.2,
        )
        for result_row in all_rows:
            result_row["features"] = feature_row
            result_row["safe_probability"] = 0.0
        v3_lock = {
            "schema_version": "confidence-guard-calibration-lock-v3",
            "models": {
                key: {
                    "audit_threshold": 0.1,
                    "confidence_guard": {"frozen_guard": guard.to_dict()},
                }
                for key in ("olmoe", "llmjp")
            },
        }
        validate_complete_raw_results(all_rows, config, v3_lock)


if __name__ == "__main__":
    unittest.main()
