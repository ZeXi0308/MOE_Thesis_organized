from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    from .decide_fjrc_corrected_two_model import (
        TwoModelDecisionError,
        decide,
        write_atomic,
    )
except ImportError:  # pragma: no cover
    from decide_fjrc_corrected_two_model import TwoModelDecisionError, decide, write_atomic  # type: ignore


REQUIRED = {
    "config.yaml",
    "metrics.json",
    "raw_results.jsonl",
    "environment.json",
    "source_manifest.json",
    "stdout.log",
    "summary.md",
}


def make_bundle(path: Path, model: str, status: str) -> None:
    path.mkdir()
    gates = {
        "holdout_q_risk_nondegenerate": True,
        "effect_size": status == "PASS",
        "paired_bootstrap_lower_gt_zero": status == "PASS",
        "minimum_strict_action_flips": status == "PASS",
        "r_not_worse_on_primary": True,
    }
    values = {
        "config.yaml": {"model": model, "run_class": "CPU_DRY_RUN"},
        "metrics.json": {
            "schema_version": "fjrc-corrected-level1-replay-v1",
            "status": "LOGICAL_TRACE_REPLAY_ONLY",
            "aggregate": {"Q": {"request_count": 32}, "R": {"request_count": 32}},
            "decision": {"status": status, "gates": gates},
            "deadline_calibration": {
                "status": "FROZEN_FROM_SELECTION_Q_ONLY",
                "r_outcomes_read_for_selection": False,
            },
        },
        "environment.json": {"cuda_execution": False, "gpu_measurement": False},
        "source_manifest.json": {
            "sources": {"core.py": "a" * 64},
            "inputs": {"lut_sha256": "b" * 64},
            "protocol_binding": {
                "scientific_boundary": "LOGICAL_TRACE_REPLAY_NOT_NETWORK_OR_SERVING_MEASUREMENT"
            },
        },
    }
    for name in REQUIRED:
        if name in values:
            (path / name).write_text(json.dumps(values[name]), encoding="utf-8")
        else:
            (path / name).write_text("test\n", encoding="utf-8")


class TwoModelDecisionTests(unittest.TestCase):
    def test_and_rule_rejects_one_model_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            olmoe, llmjp = root / "olmoe", root / "llmjp"
            make_bundle(olmoe, "olmoe", "PASS")
            make_bundle(llmjp, "llmjp", "FAIL")
            value = decide(olmoe, llmjp, expected_run_class="CPU_DRY_RUN")
            self.assertEqual(value["status"], "FAIL")
            self.assertFalse(value["pooling"])

    def test_source_drift_blocks(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            olmoe, llmjp = root / "olmoe", root / "llmjp"
            make_bundle(olmoe, "olmoe", "PASS")
            make_bundle(llmjp, "llmjp", "PASS")
            manifest = json.loads((llmjp / "source_manifest.json").read_text())
            manifest["sources"]["core.py"] = "c" * 64
            (llmjp / "source_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(TwoModelDecisionError, "same source"):
                decide(olmoe, llmjp, expected_run_class="CPU_DRY_RUN")

    def test_gate_surface_drift_blocks(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            olmoe, llmjp = root / "olmoe", root / "llmjp"
            make_bundle(olmoe, "olmoe", "PASS")
            make_bundle(llmjp, "llmjp", "PASS")
            metrics = json.loads((llmjp / "metrics.json").read_text())
            metrics["decision"]["gates"].pop("minimum_strict_action_flips")
            (llmjp / "metrics.json").write_text(json.dumps(metrics))
            with self.assertRaisesRegex(TwoModelDecisionError, "gate surface"):
                decide(olmoe, llmjp, expected_run_class="CPU_DRY_RUN")

    def test_atomic_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "decision.json"
            base = {
                "schema_version": "test",
                "status": "FAIL",
            }
            import hashlib

            canonical = json.dumps(base, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
            value = {**base, "artifact_sha256": hashlib.sha256(canonical).hexdigest()}
            write_atomic(output, value)
            with self.assertRaisesRegex(TwoModelDecisionError, "already exists"):
                write_atomic(output, value)


if __name__ == "__main__":
    unittest.main()
