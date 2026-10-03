"""Small CPU fixtures for the remaining-grid identity gate."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gate_g64_perf_remaining_grid_r02 as gate


ROOT = Path(__file__).parent


class RemainingGridGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot = json.loads((ROOT / "G64_PERF_NATIVE_EAGER_LTR_T30Q10_PLAN_R01_20260930.json").read_text())
        cls.remaining = json.loads((ROOT / "G64_PERF_REMAINING_GRID_PLAN_R02_20260930.json").read_text())

    def test_exact_remaining_plan_is_disjoint_from_pilot(self):
        gate.validate_remaining_plan(self.remaining, self.pilot)

    def test_consumed_arm_and_output_identity_are_rejected(self):
        plan = deepcopy(self.remaining)
        plan["cells"][0]["arm"] = "ltr_t30_q10"
        with self.assertRaisesRegex(ValueError, "exactly the three unrun"):
            gate.validate_remaining_plan(plan, self.pilot)
        plan = deepcopy(self.remaining)
        plan["cells"][0]["output_dir"] = self.pilot["cells"][2]["output_dir"]
        with self.assertRaisesRegex(ValueError, "consumed output/session"):
            gate.validate_remaining_plan(plan, self.pilot)

    def test_resource_drift_is_rejected(self):
        plan = deepcopy(self.remaining)
        plan["approved_host_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "changes GPU, lock, model, runtime, or host"):
            gate.validate_remaining_plan(plan, self.pilot)

    def test_remote_marker_and_output_reuse_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "candidate_g64_perf_r01"
            package.mkdir()
            for arm in gate.PILOT_ARMS:
                (package / f"launch-once-{arm}").mkdir()
            lock = root / "lock"
            lock.write_bytes(b"")
            plan = deepcopy(self.remaining)
            plan["session_dir"] = str(root / "new-session")
            plan["lock_path"] = str(lock)
            plan["expected_lock_device_inode"] = f"{lock.stat().st_dev}:{lock.stat().st_ino}"
            for cell in plan["cells"]:
                cell["package_dir"] = str(package)
                cell["output_dir"] = str(root / cell["arm"])
            pilot_session = root / "pilot-session"
            with patch.object(gate, "REMOTE_PILOT_SESSION", pilot_session):
                gate.validate_remote_state(plan, pilot_session, package)
                (package / "launch-once-ltr_t30_q1").mkdir()
                with self.assertRaisesRegex(ValueError, "identity already consumed"):
                    gate.validate_remote_state(plan, pilot_session, package)
                (package / "launch-once-ltr_t30_q1").rmdir()
                (root / "ltr_t200_q1").mkdir()
                with self.assertRaisesRegex(ValueError, "group identity already consumed"):
                    gate.validate_remote_state(plan, pilot_session, package)

    def test_pilot_receipt_and_archive_bytes_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "pilot"
            session.mkdir()
            plan_path = session / "plan.json"
            plan_path.write_text(json.dumps(self.pilot))
            pilot_plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            receipt = {"status": "CELLS_COMPLETE", "plan_sha256": pilot_plan_sha,
                       "held_lock_device_inode": self.pilot["expected_lock_device_inode"],
                       "cells": []}
            evidence = {}
            for index, planned in enumerate(self.pilot["cells"]):
                arm = planned["arm"]
                cell = session / f"cell-{index:02d}-{arm}"
                archive = cell / "archive"
                archive.mkdir(parents=True)
                for name in ("raw.json", "config.json", "status.json"):
                    (archive / name).write_text(f"{arm}:{name}")
                hashes = gate.archive_hashes(archive)
                map_path = cell / "output_sha256.json"
                map_path.write_text(json.dumps(hashes))
                evidence[arm] = {"archive_sha256_map_sha256": gate.sha256_file(map_path),
                                 "raw_sha256": hashes["raw.json"]}
                receipt["cells"].append({
                    "arm": arm, "output_dir": planned["output_dir"],
                    "argv": [planned["package_dir"] + "/pkg/run.sh", arm,
                             planned["output_dir"]],
                    "launch_status": "FINISHED", "exit_code": 0, "timed_out": False,
                    "archive_status": "VERIFIED", "gpu_process_state_after": "EMPTY",
                    "gpu_process_rows_after": [],
                })
            receipt_path = session / "receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            audit = {"status": "PILOT_THREE_ARM_COMPLETE_PARTIAL_TQ_GRID",
                     "calibration_status": "NOT_SELECTABLE_PARTIAL_GRID",
                     "plan_sha256": pilot_plan_sha,
                     "receipt_sha256": gate.sha256_file(receipt_path),
                     "package_manifest_sha256": gate.PACKAGE_MANIFEST_SHA,
                     "metrics": {arm: {"status": "COMPLETE", "expected_requests": 64,
                                       "completed": 64, "failed": 0, "unfinished": 0}
                                 for arm in gate.PILOT_ARMS},
                     "cells": evidence}
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            with patch.object(gate, "PILOT_PLAN_SHA", pilot_plan_sha):
                gate.validate_pilot(session, audit_path, gate.sha256_file(audit_path))
                (session / "cell-00-native_full_native/archive/raw.json").write_text("changed")
                with self.assertRaisesRegex(ValueError, "archive readback differs"):
                    gate.validate_pilot(session, audit_path, gate.sha256_file(audit_path))


if __name__ == "__main__":
    unittest.main()
