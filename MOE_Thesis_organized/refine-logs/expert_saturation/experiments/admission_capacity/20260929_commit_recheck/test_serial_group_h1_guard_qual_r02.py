"""CPU checks for the single H1 guard qualification controller."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import serial_group_h1_guard_qual_r02 as controller


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class H1QualificationPlanTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.private = self.root / "private"
        self.source = self.root / "source"
        self.private.mkdir()
        self.source.mkdir()
        self.lock = self.root / "gpu.lock"
        self.lock.touch()
        lock_stat = self.lock.stat()
        self.package = self.root / controller.PACKAGE_NAME
        (self.package / "pkg").mkdir(parents=True)
        (self.package / "manifest.json").write_bytes(b"fixture manifest")
        (self.package / "pkg" / "run.sh").write_text("#!/bin/sh\n")
        self.verifier = self.root / "verifier.py"
        self.verifier.write_bytes(b"# model verifier fixture\n")
        self.plan = {
            "schema_version": 1,
            "authorization_reference": "fixture authorized diagnostic",
            "authorized_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "approved_host_bytes": 90 * 1024**3,
            "approved_total_wall_seconds": 2100,
            "lock_path": str(self.lock),
            "expected_lock_device_inode": f"{lock_stat.st_dev}:{lock_stat.st_ino}",
            "session_dir": str(self.root / "new-session"),
            "python": sys.executable,
            "hf_cache_dir": str(self.private),
            "shared_model_source_cache": str(self.source),
            "model_revision": controller.EXPECTED_MODEL_REVISION,
            "model_verifier_path": str(self.verifier),
            "model_verifier_sha256": digest(self.verifier.read_bytes()),
            "cgroup_memory_max_file": "/sys/fs/cgroup/memory.max",
            "cells": [{
                "arm": "eager_diagnostic_on",
                "package_dir": str(self.package),
                "output_dir": str(self.root / "new-output"),
                "max_wall_seconds": 900,
            }],
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

    def test_only_diagnostic_on_is_valid_and_cli_has_four_arguments(self):
        self.assertIs(controller.validate_plan(self.plan), self.plan)
        output = Path(self.plan["cells"][0]["output_dir"])
        self.assertEqual(controller.qualification_argv(self.package, output), [
            str(self.package / "pkg" / "run.sh"), "eager", "diagnostic", "on", str(output)])

    def test_performance_or_second_cell_fails_before_gpu(self):
        self.plan["cells"][0]["arm"] = "eager_performance_on"
        with self.assertRaisesRegex(ValueError, "unknown H1 qualification arm"):
            controller.validate_plan(self.plan)
        self.plan["cells"][0]["arm"] = "eager_diagnostic_on"
        self.plan["cells"].append(dict(self.plan["cells"][0]))
        with self.assertRaisesRegex(ValueError, "exactly one diagnostic cell"):
            controller.validate_plan(self.plan)

    def test_budget_must_fund_model_rehash_and_full_cell(self):
        self.plan["approved_total_wall_seconds"] = 1800
        with self.assertRaisesRegex(ValueError, "cannot fund model hashing"):
            controller.validate_plan(self.plan)


if __name__ == "__main__":
    unittest.main()
