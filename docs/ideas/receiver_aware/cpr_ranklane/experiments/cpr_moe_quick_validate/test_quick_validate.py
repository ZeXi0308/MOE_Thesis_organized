from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import decide, validate_run_artifacts
from run_experiment import (
    EXPECTED_QUANTIZATION_CONTRACT,
    EXPECTED_QUALITY_RUNTIME_CONTRACT,
    REPO_ROOT,
    paired_bootstrap_ci,
    run_quality,
    sha256_file,
    summarize_samples,
    wire_time_us,
)


class QuickValidateTests(unittest.TestCase):
    def test_sample_summary_reports_distribution(self) -> None:
        result = summarize_samples(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mean_us"], 2.5)
        self.assertGreaterEqual(result["p99_us"], result["p95_us"])
        self.assertGreaterEqual(result["p95_us"], result["p50_us"])

    def test_bootstrap_is_deterministic(self) -> None:
        values = np.arange(1.0, 17.0)
        first = paired_bootstrap_ci(values, 200, 7)
        second = paired_bootstrap_ci(values, 200, 7)
        self.assertEqual(first, second)
        self.assertGreater(first[0], 0.0)

    def test_wire_time(self) -> None:
        self.assertAlmostEqual(wire_time_us(25_000, 200.0), 1.0)

    def test_quality_gate_uses_paired_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = []
            producer = root / "producer.py"
            producer.write_text("# frozen producer\n", encoding="utf-8")
            dependency_paths = {
                "capture_moe.py": REPO_ROOT / "experiments/shared/capture_moe.py",
                "fake_quant.py": REPO_ROOT / "experiments/shared/fake_quant.py",
                "metrics.py": REPO_ROOT / "experiments/shared/metrics.py",
                "modeling.py": REPO_ROOT / "experiments/shared/modeling.py",
                "policies.py": REPO_ROOT / "experiments/shared/policies.py",
                "prompts.py": REPO_ROOT / "experiments/shared/prompts.py",
            }
            for name in ("m1", "m2"):
                run_identity = {
                    "model": f"org/{name}",
                    "model_key": name,
                    "model_revision": "a" * 40,
                    "dataset": "fixture",
                    "split": "test",
                    "samples": 16,
                    "offset": 0,
                    "seq_len": 8,
                    "dtype": "bfloat16",
                    "producer_seed": 1,
                }
                path = root / f"{name}.csv"
                pd.DataFrame(
                    {
                        "sample_id": list(range(16)),
                        "rank1_int4__kl": np.linspace(0.10, 0.20, 16),
                        "rankk_int4__kl": np.linspace(0.001, 0.002, 16),
                    }
                ).to_csv(path, index=False)
                provenance = root / f"{name}.provenance.json"
                provenance.write_text(
                    json.dumps(
                        {
                            "schema_version": "rank-quality-int4-provenance-v1",
                            "attestation": "PRODUCER_EMITTED_DURING_FORWARD_RUN",
                            "run_identity": run_identity,
                            "runtime_environment": {
                                "gpu": "NVIDIA GeForce RTX 5090",
                                "compute_capability": [12, 0],
                                "torch": "2.8.0+cu128",
                                "torch_cuda": "12.8",
                            },
                            "producer_sha256": sha256_file(producer),
                            "dependency_sha256": {
                                key: sha256_file(value)
                                for key, value in sorted(dependency_paths.items())
                            },
                            "per_document_sha256": sha256_file(path),
                            "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                        }
                    ),
                    encoding="utf-8",
                )
                models.append(
                    {
                        "name": name,
                        "run_identity": run_identity,
                        "per_document_csv": str(path),
                        "provenance_json": str(provenance),
                    }
                )
            config = {
                "seed": 1,
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "quality": {
                    "decision_mode": "int4",
                    "producer": str(producer),
                    "runtime_contract": EXPECTED_QUALITY_RUNTIME_CONTRACT,
                    "bootstrap_repeats": 200,
                    "head_tail_ratio_threshold": 5.0,
                    "paired_difference_lcb_threshold": 0.0,
                    "models": models,
                },
            }
            output = root / "out"
            output.mkdir()
            decision = run_quality(config, output, smoke=False)
            self.assertTrue(decision["all_models_passed"])
            raw = pd.read_csv(output / "quality_paired_raw.csv")
            self.assertEqual(len(raw), 32)
            self.assertIn("paired_head_minus_tail_kl", raw.columns)

    def test_analysis_never_promotes_single_gpu_to_ep_go(self) -> None:
        result = decide(
            {
                "status": "COMPLETE",
                "decision_mode": "int4",
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "source_provenance": [{"model": "m"}],
                "all_models_passed": True,
            },
            {
                "status": "COMPLETE",
                "decision_mode": "int4",
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "primary_mode_passed": True,
            },
        )
        self.assertEqual(
            result["verdict"], "NOT_FALSIFIED_SINGLE_GPU_BLOCKED_EP_RETURN_PATH_GATE"
        )
        self.assertEqual(result["ep_return_path_gate"], "BLOCKED_NOT_TESTABLE_ON_SINGLE_GPU")

    def test_codec_failure_is_not_relabeled_as_ep_result(self) -> None:
        result = decide(
            {
                "status": "COMPLETE",
                "decision_mode": "int4",
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "source_provenance": [{"model": "m"}],
                "all_models_passed": True,
            },
            {
                "status": "COMPLETE",
                "decision_mode": "int4",
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "primary_mode_passed": False,
            },
        )
        self.assertEqual(result["verdict"], "NO_GO_CURRENT_UNFUSED_INT4_CODEC_PATH")
        self.assertEqual(result["ep_return_path_gate"], "BLOCKED_NOT_TESTABLE_ON_SINGLE_GPU")

    def test_failed_status_cannot_be_promoted(self) -> None:
        result = decide(
            {
                "status": "FAILED",
                "decision_mode": "int4",
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "source_provenance": [{"model": "m"}],
                "all_models_passed": True,
            },
            {
                "status": "COMPLETE",
                "decision_mode": "int4",
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "primary_mode_passed": True,
            },
        )
        self.assertEqual(result["verdict"], "INVALID_DECISION_STATUS")

    def test_int8_characterization_cannot_replace_int4_gate(self) -> None:
        result = decide(
            {
                "status": "COMPLETE",
                "decision_mode": "int4",
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "source_provenance": [{"model": "m"}],
                "all_models_passed": True,
            },
            {
                "status": "COMPLETE",
                "decision_mode": "int4",
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "primary_mode_passed": False,
                "mode_gates": {
                    "int8": {"passed": True},
                    "int4": {"passed": False},
                },
            },
        )
        self.assertEqual(result["verdict"], "NO_GO_CURRENT_UNFUSED_INT4_CODEC_PATH")

    def test_string_boolean_cannot_pass_quality_gate(self) -> None:
        result = decide(
            {
                "status": "COMPLETE",
                "decision_mode": "int4",
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "source_provenance": [{"model": "m"}],
                "all_models_passed": "false",
            },
            {
                "status": "COMPLETE",
                "decision_mode": "int4",
                "quantization_contract": EXPECTED_QUANTIZATION_CONTRACT,
                "primary_mode_passed": True,
            },
        )
        self.assertEqual(result["verdict"], "INVALID_QUALITY_CONTRACT")

    def test_missing_manifest_is_invalid(self) -> None:
        self.assertIn(
            "missing run_manifest.json",
            validate_run_artifacts(None, {"status": "COMPLETE"}, {"status": "COMPLETE"}),
        )


if __name__ == "__main__":
    unittest.main()
