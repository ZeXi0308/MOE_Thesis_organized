#!/usr/bin/env python3

import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_gate.py")
SPEC = importlib.util.spec_from_file_location("cpr_moe_run_gate", MODULE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class FormulaTests(unittest.TestCase):
    def test_exact_relative_improvement(self):
        self.assertAlmostEqual(gate.exact_relative_improvement(0.5, 0.6875, 0.2), 1 / 24)

    def test_required_fraction_inverts_formula(self):
        required = gate.required_exposed_fraction(0.5, 0.6875, 0.05)
        self.assertAlmostEqual(required, 4 / 17)
        self.assertAlmostEqual(gate.exact_relative_improvement(0.5, 0.6875, required), 0.05)

    def test_rejects_candidate_worse_than_baseline(self):
        with self.assertRaises(gate.EvidenceError):
            gate.exact_relative_improvement(0.5, 0.4, 0.2)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = {
            "uniform_fp8": {"policy": "uniform_fp8", "byte_saving": 0.5, "kl_ci_high": 0.004},
            "fp8top6": {"policy": "fp8top6", "byte_saving": 0.5625, "kl_ci_high": 0.009},
            "fp8top2": {"policy": "fp8top2", "byte_saving": 0.6875, "kl_ci_high": 0.052},
        }

    def test_quality_free_selection_is_most_optimistic(self):
        selected = gate.choose_max_saving(self.rows, "fp8top")
        self.assertEqual(selected["policy"], "fp8top2")

    def test_quality_budget_is_ci_high_not_mean(self):
        selected = gate.choose_under_quality_budget(self.rows, "fp8top", 0.01)
        self.assertEqual(selected["policy"], "fp8top6")
        self.assertIsNone(gate.choose_under_quality_budget(self.rows, "fp8top", 0.008))


class EvidenceValidationTests(unittest.TestCase):
    def _write_quality_fixture(self, root: Path, model_key: str, saving: float = 0.6875):
        summary = root / f"{model_key}.csv"
        metadata = root / f"{model_key}.json"
        rows = [
            ["uniform_fp8", 0.5, 0.004, 0.003, 0.005],
            ["fp8top_tail", saving, 0.03, 0.02, 0.04],
        ]
        with summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["policy", "byte_saving", "mean_kl", "kl_ci_low", "kl_ci_high"])
            writer.writerows(rows)
        metadata.write_text(
            json.dumps(
                {
                    "model": model_key,
                    "model_key": model_key,
                    "top_k": 8,
                    "samples": 128,
                    "byte_saving": {"uniform_fp8": 0.5, "fp8top_tail": saving},
                    "evidence_boundary": "single-forward evidence. No decode-loop and no communication claim.",
                }
            ),
            encoding="utf-8",
        )
        return {
            "model_key": model_key,
            "summary_csv": summary.name,
            "metadata_json": metadata.name,
        }

    def test_cross_model_and_fails_when_one_model_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            quality_inputs = [
                self._write_quality_fixture(root, "pass_model", 0.75),
                self._write_quality_fixture(root, "fail_model", 0.6875),
            ]
            codec_path = root / "codec.json"
            codec_path.write_text(
                json.dumps(
                    {
                        "gpu": "fixture",
                        "source": "fixture p95 sample arrays not re-fetched",
                        "evidence_boundary": "fixture only",
                        "incremental_fp8_to_int4": {"n_serving_cells": 1, "viable_count": 0},
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "schema_version": 1,
                "experiment_id": "fixture",
                "hypothesis": "fixture",
                "baseline_policy": "uniform_fp8",
                "candidate_prefix": "fp8top",
                "exposed_return_fraction_grid": [0.2],
                "exposed_return_fraction_max": 0.2,
                "min_relative_improvement": 0.05,
                "quality_kl_ci_high_budgets": [0.04],
                "quality_inputs": quality_inputs,
                "codec_metadata_json": codec_path.name,
            }
            result = gate.analyze(root, config)
            self.assertFalse(result["primary_gate"]["pass"])
            self.assertEqual(
                result["decision"]["code"],
                "NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_0_20",
            )

    def test_csv_metadata_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = self._write_quality_fixture(root, "mismatch")
            metadata_path = root / spec["metadata_json"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["byte_saving"]["fp8top_tail"] = 0.625
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(gate.EvidenceError):
                gate.read_quality_input(root, spec)


if __name__ == "__main__":
    unittest.main()
