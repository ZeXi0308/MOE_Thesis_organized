from __future__ import annotations

import unittest

from finalize_calibration import CalibrationError, finalize
from triage_policy import FEATURE_NAMES


def fixture_config() -> dict:
    return {
        "schema_version": "triage-audit-v2-design",
        "seed": 7,
        "dataset": {"calibration_documents": 12, "decode_steps": 4},
        "models": {"olmoe": {}, "llmjp": {}},
        "predictor": {"ridge_alpha": 1.0},
        "controller": {"audit_threshold_quantile": 0.9},
        "calibration_stability": {
            "bootstrap_replicates": 100,
            "median_assignment_probability_min": 0.0,
            "fraction_documents_probability_ge_0_6_min": 0.0,
            "spearman_lcb_min_exclusive": -2.0,
        },
    }


def fixture_rows() -> list[dict[str, object]]:
    rows = []
    for model_key in ("olmoe", "llmjp"):
        for index in range(12):
            rows.append({
                "model_key": model_key,
                "split": "calibration",
                "text_sha256": f"{index + 1:064x}",
                "features": {name: index + feature_index * 0.01 for feature_index, name in enumerate(FEATURE_NAMES)},
                "same_state_discrepancies": [0.01 + index * 0.001] * 4,
            })
    return rows


class FinalizeCalibrationTests(unittest.TestCase):
    def test_two_model_lock(self) -> None:
        result = finalize(fixture_rows(), fixture_config(), raw_sha256="a" * 64)
        self.assertTrue(result["h0_all_models_pass"])
        self.assertGreater(result["models"]["olmoe"]["audit_threshold"], 0)

    def test_rejects_non_calibration_row(self) -> None:
        rows = fixture_rows()
        rows[0]["split"] = "sealed"
        with self.assertRaises(CalibrationError):
            finalize(rows, fixture_config(), raw_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
