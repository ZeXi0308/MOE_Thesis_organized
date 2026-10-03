from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from .preflight_fjrc_corrected_gpu import (
        PreflightError,
        _new_report,
        _write_report,
        plan_artifact_state,
        run_preflight,
        validate_disk_capacity,
    )
except ImportError:  # pragma: no cover
    from preflight_fjrc_corrected_gpu import (  # type: ignore
        PreflightError,
        _new_report,
        _write_report,
        plan_artifact_state,
        run_preflight,
        validate_disk_capacity,
    )


class FJRCGPUPreflightTests(unittest.TestCase):
    @staticmethod
    def artifact_config():
        return {
            "artifacts": {
                "route_root": "$REMOTE_ROOT/clean_v2/routes/calibration",
                "route_state_root": "$REMOTE_ROOT/state",
                "lut": "$REMOTE_ROOT/clean_v2/fjrc/lut.json",
                "native_dry_run_outputs": {
                    "olmoe": "$REMOTE_ROOT/clean_v2/fjrc/dry/olmoe",
                    "llmjp": "$REMOTE_ROOT/clean_v2/fjrc/dry/llmjp",
                },
            }
        }

    def test_static_bundle_passes_only_with_exact_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.py"
            source.write_text("answer = 42\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            config = root / "config.json"
            config.write_text(
                json.dumps({"reviewed_sources": {"source.py": digest}}), encoding="utf-8"
            )
            report = run_preflight(
                mode="static",
                repo_root=root,
                remote_root=root,
                config_path=config,
                deep_model_hash=False,
            )
            self.assertEqual(report["status"], "READY")
            source.write_text("answer = 43\n", encoding="utf-8")
            report = run_preflight(
                mode="static",
                repo_root=root,
                remote_root=root,
                config_path=config,
                deep_model_hash=False,
            )
            self.assertEqual(report["status"], "BLOCKED")
            self.assertIn("reviewed_source_manifest", report["blockers"])

    def test_report_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "report.json"
            _write_report(output, {"status": "READY"})
            self.assertEqual(json.loads(output.read_text())["status"], "READY")
            with self.assertRaisesRegex(PreflightError, "already exists"):
                _write_report(output, {"status": "BLOCKED"})

    def test_unknown_mode_blocks(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "mode"):
                run_preflight(
                    mode="wrong",
                    repo_root=root,
                    remote_root=root,
                    config_path=config,
                    deep_model_hash=False,
                )

    def test_empty_remote_state_plans_exact_capture_and_dry_run_actions(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = _new_report("gpu", root, root / "config.json")
            plan_artifact_state(root, self.artifact_config(), report)
            self.assertEqual(
                report["planned_actions"],
                [
                    "capture_route_olmoe",
                    "capture_route_llmjp",
                    "capture_primitive_lut",
                    "native_cpu_dry_run_olmoe",
                    "native_cpu_dry_run_llmjp",
                ],
            )
            self.assertFalse(report["blockers"])

    def test_orphan_one_shot_ledger_is_a_hard_block(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state"
            state.mkdir()
            (state / "route_calibration_olmoe_consumption.json").write_text("{}")
            report = _new_report("gpu", root, root / "config.json")
            plan_artifact_state(root, self.artifact_config(), report)
            self.assertIn("route_olmoe_one_shot_state", report["blockers"])
            self.assertNotIn("capture_route_olmoe", report["planned_actions"])

    def test_disk_threshold_is_action_aware(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = {
                "environment": {
                    "minimum_free_disk_gib_by_action": {
                        "route_capture": 10**9,
                        "lut_and_replay": 0,
                        "validation_only": 0,
                    }
                }
            }
            route_report = _new_report("gpu", root, root / "config.json")
            route_report["planned_actions"] = ["capture_route_olmoe"]
            validate_disk_capacity(root, config, route_report)
            self.assertIn("free_disk", route_report["blockers"])
            replay_report = _new_report("gpu", root, root / "config.json")
            replay_report["planned_actions"] = ["capture_primitive_lut"]
            validate_disk_capacity(root, config, replay_report)
            self.assertNotIn("free_disk", replay_report["blockers"])
            self.assertEqual(replay_report["checks"][-1]["detail"]["action_class"], "lut_and_replay")


if __name__ == "__main__":
    unittest.main()
