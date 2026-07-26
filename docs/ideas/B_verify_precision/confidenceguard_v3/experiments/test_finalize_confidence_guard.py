from __future__ import annotations

import unittest

from finalize_confidence_guard import ConfidenceGuardCalibrationError, finalize
from triage_policy import FEATURE_NAMES


class FinalizeConfidenceGuardTests(unittest.TestCase):
    def _rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for model_key in ("olmoe", "llmjp"):
            for index in range(32):
                rows.append(
                    {
                        "model_key": model_key,
                        "split": "calibration",
                        "text_sha256": f"{index + 1:064x}",
                        "features": {
                            name: float(index + feature_index * 0.01)
                            for feature_index, name in enumerate(FEATURE_NAMES)
                        },
                        "same_state_discrepancies": [0.001 * (index + 1)] * 4,
                    }
                )
        return rows

    def _config(self) -> dict[str, object]:
        return {
            "schema_version": "confidence-guard-v3-design",
            "seed": 7,
            "dataset": {"calibration_documents": 32, "decode_steps": 4},
            "models": {"olmoe": {}, "llmjp": {}},
            "predictor": {
                "name": "bootstrap_confidence_guard",
                "ridge_alpha": 1.0,
                "bootstrap_replicates": 100,
                "safe_probability_min": 0.8,
                "risk_probability_max": 0.2,
            },
            "calibration_stability": {
                "median_binary_assignment_probability_min": 0.8,
                "fraction_documents_probability_ge_0_6_min": 0.8,
                "spearman_lcb_min_exclusive": 0.0,
            },
            "controller": {"audit_threshold_quantile": 0.9},
            "reformulation": {"origin_raw_calibration_sha256": "a" * 64},
        }

    def test_valid_reformulation_lock_is_source_bound(self) -> None:
        lock = finalize(self._rows(), self._config(), raw_sha256="a" * 64)
        self.assertEqual(lock["schema_version"], "confidence-guard-calibration-lock-v3")
        self.assertTrue(lock["reformulation_gate_all_models_pass"])
        self.assertTrue(lock["calibration_numbers_are_exploratory"])
        for model in lock["models"].values():
            self.assertEqual(
                len(model["confidence_guard"]["frozen_guard"]["bootstrap_models"]),
                100,
            )

    def test_origin_hash_mismatch_fails(self) -> None:
        with self.assertRaises(ConfidenceGuardCalibrationError):
            finalize(self._rows(), self._config(), raw_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
