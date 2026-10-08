"""CPU gates for the SHA-pinned H128 transfer controller."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import serial_group_h128_perf_guard_r02 as controller


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class H128TransferGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.private = self.root / "private"
        self.source = self.root / "source"
        self.private.mkdir()
        self.source.mkdir()
        self.lock = self.root / "gpu.lock"
        self.lock.touch()
        self.verifier = self.root / "verify_model.py"
        self.verifier.write_bytes(b"# fixture\n")
        self.package = self.root / controller.PACKAGE_NAME
        (self.package / "pkg").mkdir(parents=True)
        (self.package / "manifest.json").write_bytes(b"fixture manifest")
        (self.package / "pkg" / "run.sh").write_text("#!/bin/sh\n")
        self.selection = self.root / "g64-selection.json"
        self.selection.write_bytes((Path(__file__).parent /
            "A_G64_GUARDED_FOUR_POINT_SELECTION_20260930.json").read_bytes())
        held = self.lock.stat()
        self.plan = {
            "schema_version": 1,
            "authorization_reference": "fixture authorization",
            "authorized_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "approved_host_bytes": 90 * 1024**3,
            "approved_total_wall_seconds": 4200,
            "lock_path": str(self.lock),
            "expected_lock_device_inode": f"{held.st_dev}:{held.st_ino}",
            "session_dir": str(self.root / "new-session"),
            "python": sys.executable,
            "hf_cache_dir": str(self.private),
            "shared_model_source_cache": str(self.source),
            "model_revision": controller.EXPECTED_MODEL_REVISION,
            "model_verifier_path": str(self.verifier),
            "model_verifier_sha256": digest(self.verifier.read_bytes()),
            "cgroup_memory_max_file": "/sys/fs/cgroup/memory.max",
            "g64_selection_path": str(self.selection),
            "g64_selection_sha256": controller.G64_SELECTION_SHA256,
            "selected_ltr_arm": controller.G64_SELECTED_ARM,
            "cells": [{
                "arm": arm,
                "package_dir": str(self.package),
                "output_dir": str(self.root / f"output-{arm}"),
                "max_wall_seconds": 900,
            } for arm in controller.FROZEN_ARMS],
        }
        for name, value in (
            ("EXPECTED_PRIVATE_PYTHON", sys.executable),
            ("EXPECTED_MODEL_CACHE", str(self.private)),
            ("EXPECTED_SHARED_MODEL_SOURCE_CACHE", str(self.source)),
            ("MODEL_VERIFIER_PATH", str(self.verifier)),
            ("MODEL_VERIFIER_SHA256", digest(self.verifier.read_bytes())),
            ("PACKAGE_SHA256", digest((self.package / "manifest.json").read_bytes())),
        ):
            item = patch.object(controller, name, value)
            item.start()
            self.addCleanup(item.stop)

    def test_exact_three_arm_plan_accepts_pinned_selection(self):
        self.assertIs(controller.validate_plan(self.plan), self.plan)
        self.assertEqual(digest(controller.verify_g64_selection(self.plan)),
                         controller.G64_SELECTION_SHA256)

    def test_wrong_selected_arm_and_reordered_cells_fail(self):
        self.plan["selected_ltr_arm"] = "ltr_t200_q10"
        with self.assertRaisesRegex(ValueError, "H128 LTR arm differs"):
            controller.validate_plan(self.plan)
        self.plan["selected_ltr_arm"] = controller.G64_SELECTED_ARM
        self.plan["cells"][1], self.plan["cells"][2] = self.plan["cells"][2], self.plan["cells"][1]
        with self.assertRaisesRegex(ValueError, "arm order differs"):
            controller.validate_plan(self.plan)

    def test_selection_receipt_bytes_and_semantics_fail_closed(self):
        self.selection.write_bytes(self.selection.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "receipt bytes differ"):
            controller.validate_plan(self.plan)
        report = json.loads(self.selection.read_text())
        report["selected_arm"] = "ltr_t30_q1"
        changed = json.dumps(report).encode()
        self.selection.write_bytes(changed)
        with patch.object(controller, "G64_SELECTION_SHA256", digest(changed)):
            self.plan["g64_selection_sha256"] = digest(changed)
            with self.assertRaisesRegex(ValueError, "selected and eligible arm mismatch"):
                controller.validate_plan(self.plan)

    def test_plan_cannot_replace_frozen_selection_sha(self):
        self.plan["g64_selection_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "G64 selection SHA differs"):
            controller.validate_plan(self.plan)

    def test_selection_symlink_is_rejected_even_if_bytes_match(self):
        link = self.root / "selection-link.json"
        link.symlink_to(self.selection.name)
        self.plan["g64_selection_path"] = str(link)
        with self.assertRaisesRegex(ValueError, "receipt missing or symlinked"):
            controller.validate_plan(self.plan)


if __name__ == "__main__":
    unittest.main()
