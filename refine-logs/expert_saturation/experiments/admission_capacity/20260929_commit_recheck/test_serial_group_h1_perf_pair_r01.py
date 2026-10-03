"""CPU checks for the frozen H1 two-block adjacent-pair controller."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import serial_group_h1_perf_pair_r01 as controller


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class PairPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.private = self.root / "private"
        self.source = self.root / "source"
        self.private.mkdir()
        self.source.mkdir()
        self.lock = self.root / "gpu.lock"
        self.lock.touch()
        st = self.lock.stat()
        self.verifier = self.root / "verifier.py"
        self.verifier.write_bytes(b"# pinned model verifier fixture\n")
        self.packages = {}
        manifest_bytes = b'{"fixture": "same payload mapping"}\n'
        for block in (1, 2):
            package = self.root / controller.PACKAGE_NAMES[block]
            (package / "pkg").mkdir(parents=True)
            (package / "manifest.json").write_bytes(manifest_bytes)
            (package / "pkg" / "run.sh").write_text("#!/bin/sh\n")
            self.packages[block] = package
        self.protocol = self.root / "paired-protocol.json"
        self.protocol.write_text(json.dumps({
            "schema_version": 1,
            "qualified_package_manifest_sha256": digest(manifest_bytes),
            "blocks": [{"block": b, "order": list(controller.FROZEN_ORDERS[b])}
                       for b in (1, 2)],
        }))
        self.qualification = self.root / "qualification-audit.json"
        self.qualification_report = {
            "schema_version": 1, "status": "OBSERVED_DIRECT_ACTION_CHAIN",
            "qualification": "OBSERVED_NATIVE_DIRECT_AND_LATER_OUTPUT",
            "plan_sha256": controller.QUALIFICATION_PLAN_SHA256,
            "package_manifest_sha256": digest(manifest_bytes),
            "requests_completed": 128,
            "direct_action": {"status": "OBSERVED_DIRECT_ACTION_CHAIN",
                              "direct_commits": 1,
                              "chains": [{"native_admission": "ASYNC_LOAD_ADMITTED",
                                          "native_ready_s": 12.0,
                                          "first_observed_output_after_ready_s": 13.0}]},
        }
        self.write_qualification()
        self.sessions = {b: self.root / f"session-b{b}" for b in (1, 2)}
        self.outputs = {
            1: (self.root / "b1-off", self.root / "b1-on"),
            2: (self.root / "b2-on", self.root / "b2-off"),
        }
        patches = {
            "PACKAGE_SHA256": digest(manifest_bytes),
            "PROTOCOL_SHA256": digest(self.protocol.read_bytes()),
            "QUALIFICATION_AUDIT_SHA256": digest(self.qualification.read_bytes()),
            "EXPECTED_PRIVATE_PYTHON": sys.executable,
            "EXPECTED_MODEL_CACHE": str(self.private),
            "EXPECTED_SHARED_MODEL_SOURCE_CACHE": str(self.source),
            "MODEL_VERIFIER_PATH": str(self.verifier),
            "MODEL_VERIFIER_SHA256": digest(self.verifier.read_bytes()),
            "EXPECTED_LOCK_PATH": str(self.lock),
            "EXPECTED_LOCK_INODE": f"{st.st_dev}:{st.st_ino}",
            "EXPECTED_SESSIONS": {b: str(self.sessions[b]) for b in (1, 2)},
            "EXPECTED_OUTPUTS": {b: tuple(str(p) for p in self.outputs[b]) for b in (1, 2)},
            "EXPECTED_PACKAGE_ROOT": str(self.root),
        }
        for key, value in patches.items():
            item = patch.object(controller, key, value)
            item.start()
            self.addCleanup(item.stop)

    def write_qualification(self):
        self.qualification.write_text(json.dumps(self.qualification_report))

    def plan(self, block):
        order = controller.FROZEN_ORDERS[block]
        return {
            "schema_version": 1, "block_index": block,
            "authorization_reference": "fixture authorized pair, qualification passed",
            "authorized_gpu_uuid": controller.EXPECTED_GPU_UUID,
            "approved_host_bytes": controller.EXPECTED_HOST_BYTES,
            "approved_total_wall_seconds": 3200,
            "lock_path": str(self.lock),
            "expected_lock_device_inode": controller.EXPECTED_LOCK_INODE,
            "session_dir": str(self.sessions[block]),
            "python": sys.executable,
            "hf_cache_dir": str(self.private),
            "shared_model_source_cache": str(self.source),
            "model_revision": controller.EXPECTED_MODEL_REVISION,
            "model_verifier_path": str(self.verifier),
            "model_verifier_sha256": controller.MODEL_VERIFIER_SHA256,
            "cgroup_memory_max_file": "/sys/fs/cgroup/memory.max",
            "protocol_path": str(self.protocol),
            "protocol_sha256": controller.PROTOCOL_SHA256,
            "qualification_audit_path": str(self.qualification),
            "qualification_audit_sha256": controller.QUALIFICATION_AUDIT_SHA256,
            "cells": [{"arm": arm, "package_dir": str(self.packages[block]),
                       "output_dir": str(self.outputs[block][index]),
                       "max_wall_seconds": 900}
                      for index, arm in enumerate(order)],
        }

    def test_both_blocks_have_distinct_frozen_orders_and_cli(self):
        for block in (1, 2):
            with self.subTest(block=block):
                plan = self.plan(block)
                self.assertIs(controller.validate_plan(plan), plan)
                self.assertEqual([cell["arm"] for cell in plan["cells"]],
                                 list(controller.FROZEN_ORDERS[block]))
                for cell in plan["cells"]:
                    gate = cell["arm"].rsplit("_", 1)[1]
                    self.assertEqual(controller.performance_argv(
                        self.packages[block], cell["arm"], Path(cell["output_dir"])),
                        [str(self.packages[block] / "pkg" / "run.sh"),
                         "eager", "performance", gate, cell["output_dir"]])

    def test_wrong_order_package_or_budget_rejected_before_gpu(self):
        plan = self.plan(1)
        plan["cells"].reverse()
        with self.assertRaisesRegex(ValueError, "frozen H1 pair order"):
            controller.validate_plan(plan)
        plan = self.plan(1)
        plan["cells"][0]["package_dir"] = str(self.packages[2])
        with self.assertRaisesRegex(ValueError, "package identity mismatch"):
            controller.validate_plan(plan)
        plan = self.plan(1)
        plan["cells"][0]["max_wall_seconds"] = 899
        with self.assertRaisesRegex(ValueError, "frozen 900 s"):
            controller.validate_plan(plan)
        plan = self.plan(1)
        plan["approved_total_wall_seconds"] = 3199
        with self.assertRaisesRegex(ValueError, "fixed 3200 s"):
            controller.validate_plan(plan)

    def test_no_action_or_audit_drift_rejected_before_gpu(self):
        plan = self.plan(1)
        self.qualification_report["status"] = "NO_ACTION"
        self.qualification_report["qualification"] = "NO_ACTION_STOP"
        self.qualification_report["direct_action"] = {
            "status": "NO_ACTION", "direct_commits": 0, "chains": []}
        self.write_qualification()
        new_sha = digest(self.qualification.read_bytes())
        with patch.object(controller, "QUALIFICATION_AUDIT_SHA256", new_sha):
            plan["qualification_audit_sha256"] = new_sha
            with self.assertRaisesRegex(ValueError, "has not succeeded"):
                controller.validate_plan(plan)
        with self.assertRaisesRegex(ValueError, "qualification_audit SHA-256 differs"):
            controller.validate_plan(self.plan(1))

    def test_first_archive_reserves_second_full_cell(self):
        later = [{"max_wall_seconds": 900}]
        self.assertEqual(controller.archive_budget_seconds(1200, later), 235)
        self.assertLessEqual(controller.archive_budget_seconds(965, later), 0)
        self.assertEqual(controller.archive_budget_seconds(1200, []), 1180)


if __name__ == "__main__":
    unittest.main()
