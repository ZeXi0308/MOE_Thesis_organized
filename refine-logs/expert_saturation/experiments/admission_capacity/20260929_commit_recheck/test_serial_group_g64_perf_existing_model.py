"""CPU gates for the existing-model G64 performance controller."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import serial_group_g64_perf_existing_model as controller


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ExistingModelGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.private = self.root / "private"
        self.revision = "fixture-revision"
        self.metadata = {name: ("metadata:" + name).encode()
                         for name in controller.MODEL_METADATA_SHA256}
        self.shards = {name: ("shard:" + name).encode()
                       for name in controller.MODEL_SHARDS}
        for cache in (self.source, self.private):
            repo = cache / "hub" / controller.MODEL_REPO_NAME
            snapshot = repo / "snapshots" / self.revision
            blobs = repo / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            for name, data in {**self.metadata, **self.shards}.items():
                blob = blobs / digest(data)
                blob.write_bytes(data)
                (snapshot / name).symlink_to(Path("../../blobs") / blob.name)
        self.plan = {
            "shared_model_source_cache": str(self.source),
            "hf_cache_dir": str(self.private),
            "model_revision": self.revision,
        }
        self.session = self.root / "session"
        self.session.mkdir()
        self.patches = [
            patch.object(controller, "MODEL_METADATA_SHA256",
                         {name: digest(data) for name, data in self.metadata.items()}),
            patch.object(controller, "MODEL_SHARDS",
                         {name: (len(data), digest(data)) for name, data in self.shards.items()}),
            patch.object(controller, "filesystem_device",
                         side_effect=lambda path: 2 if "private" in str(path) else 1),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def private_blob(self, name):
        return (self.private / "hub" / controller.MODEL_REPO_NAME /
                "snapshots" / self.revision / name).resolve()

    def rehash(self):
        return controller.verify_existing_private_model_under_lock(
            self.plan, self.session, time.monotonic() + 600)

    def test_rehashes_existing_private_model_without_writing_to_it(self):
        repo = self.private / "hub" / controller.MODEL_REPO_NAME
        before = [(str(path.relative_to(repo)), path.lstat().st_mtime_ns)
                  for path in sorted(repo.rglob("*"))]
        result = self.rehash()
        after = [(str(path.relative_to(repo)), path.lstat().st_mtime_ns)
                 for path in sorted(repo.rglob("*"))]
        self.assertEqual(result["status"], "VERIFIED_PRIVATE_MODEL_VIEW")
        self.assertEqual(result["access_mode"], "existing_read_only")
        self.assertEqual(len(result["metadata_sha256"]), 6)
        self.assertEqual(len(result["shards"]), 3)
        self.assertEqual(before, after)

    def test_same_size_corrupt_shard_fails_closed(self):
        name = next(iter(self.shards))
        blob = self.private_blob(name)
        blob.write_bytes(b"x" * len(self.shards[name]))
        with self.assertRaisesRegex(ValueError, "shard size/SHA-256 differs"):
            self.rehash()
        self.assertEqual(
            json.loads((self.session / "model-private-view-status.json").read_text())["status"],
            "INCOMPLETE",
        )

    def test_noncritical_private_cache_metadata_does_not_mask_frozen_hashes(self):
        repo = self.private / "hub" / controller.MODEL_REPO_NAME
        (repo / ".no_exist").mkdir()
        (repo / ".no_exist" / "optional-file").write_bytes(b"cached absence")
        result = self.rehash()
        self.assertEqual(result["status"], "VERIFIED_PRIVATE_MODEL_VIEW")
        self.assertNotEqual(result["private_tree_bytes"], result["source_tree_bytes"])

    def test_private_snapshot_link_must_stay_inside_private_repo(self):
        name = next(iter(self.shards))
        link = (self.private / "hub" / controller.MODEL_REPO_NAME /
                "snapshots" / self.revision / name)
        link.unlink()
        outside = self.private / "hub" / "escape"
        outside.write_bytes(self.shards[name])
        link.symlink_to("../../../escape")
        with self.assertRaisesRegex(ValueError, "link escapes its own repo"):
            self.rehash()

    def test_private_blob_must_have_independent_inode(self):
        name = next(iter(self.shards))
        private_blob = self.private_blob(name)
        source_blob = (self.source / "hub" / controller.MODEL_REPO_NAME /
                       "snapshots" / self.revision / name).resolve()
        private_blob.unlink()
        os.link(source_blob, private_blob)
        with self.assertRaisesRegex(ValueError, "independent regular inode"):
            self.rehash()

    def test_runtime_helpers_live_in_session_not_model_cache(self):
        controller.prepare_runtime_cache(self.session)
        self.plan["authorized_gpu_uuid"] = "GPU-00000000-0000-0000-0000-000000000000"
        env = controller.private_model_env(self.plan, self.session)
        self.assertEqual(env["HF_HOME"], str(self.private))
        self.assertEqual(env["HF_HUB_CACHE"], str(self.private / "hub"))
        for name in ("TMPDIR", "VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR"):
            self.assertTrue(env[name].startswith(str(self.session / "runtime-cache")))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")


class PlanAndLockGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lock = self.root / "gpu.lock"
        self.lock.touch()
        lock_stat = self.lock.stat()
        self.lock_id = f"{lock_stat.st_dev}:{lock_stat.st_ino}"
        self.package = self.root / controller.PACKAGE_NAME
        (self.package / "pkg").mkdir(parents=True)
        (self.package / "pkg" / "run.sh").write_text("#!/bin/sh\n")
        (self.package / "manifest.json").write_bytes(b"frozen fixture manifest")
        self.verifier = self.root / "verifier.py"
        self.verifier.write_bytes(b"# verifier fixture\n")
        self.private = self.root / "private"
        self.source = self.root / "source"
        self.private.mkdir()
        self.source.mkdir()
        self.plan = {
            "schema_version": 1,
            "authorization_reference": "fixture authorization",
            "authorized_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "approved_host_bytes": 96636764160,
            "approved_total_wall_seconds": 1100,
            "lock_path": str(self.lock),
            "expected_lock_device_inode": self.lock_id,
            "session_dir": str(self.root / "new-session"),
            "python": sys.executable,
            "hf_cache_dir": str(self.private),
            "shared_model_source_cache": str(self.source),
            "model_revision": controller.EXPECTED_MODEL_REVISION,
            "model_verifier_path": str(self.verifier),
            "model_verifier_sha256": digest(self.verifier.read_bytes()),
            "cgroup_memory_max_file": "/sys/fs/cgroup/memory.max",
            "cells": [
                {"arm": arm, "package_dir": str(self.package),
                 "output_dir": str(self.root / f"new-output-{arm}"),
                 "max_wall_seconds": 10}
                for arm in ("native_full_native", "eager", "ltr_t30_q10")
            ],
        }
        self.patches = [
            patch.object(controller, "EXPECTED_PRIVATE_PYTHON", sys.executable),
            patch.object(controller, "EXPECTED_MODEL_CACHE", str(self.private)),
            patch.object(controller, "EXPECTED_SHARED_MODEL_SOURCE_CACHE", str(self.source)),
            patch.object(controller, "MODEL_VERIFIER_PATH", str(self.verifier)),
            patch.object(controller, "MODEL_VERIFIER_SHA256", digest(self.verifier.read_bytes())),
            patch.object(controller, "PACKAGE_SHA256", digest((self.package / "manifest.json").read_bytes())),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_accepts_three_frozen_distinct_arms(self):
        self.assertIs(controller.validate_plan(self.plan), self.plan)

    def test_plan_bytes_require_explicit_sha_pin(self):
        path = self.root / "plan.json"
        path.write_text(json.dumps(self.plan))
        argv = ["controller", str(path), "--expected-plan-sha256", "0" * 64,
                "--validate-only"]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "plan SHA-256 differs"):
                controller.main()
        argv[3] = digest(path.read_bytes())
        with patch.object(sys, "argv", argv):
            self.assertEqual(controller.main(), 0)

    def test_group_budget_must_include_model_rehash_and_archive(self):
        self.plan["approved_total_wall_seconds"] = 1084
        with self.assertRaisesRegex(ValueError, "cannot fund model hashing"):
            controller.validate_plan(self.plan)

    def test_repeated_arm_fails_before_gpu(self):
        self.plan["cells"][2]["arm"] = "eager"
        with self.assertRaisesRegex(ValueError, "repeats a performance arm"):
            controller.validate_plan(self.plan)

    def test_existing_model_is_required(self):
        self.private.rmdir()
        with self.assertRaisesRegex(ValueError, "existing A private model cache missing"):
            controller.validate_plan(self.plan)

    def test_manifest_mutation_fails_before_gpu(self):
        (self.package / "manifest.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            controller.validate_plan(self.plan)

    def test_model_cache_cannot_be_an_output_destination(self):
        self.plan["cells"][0]["output_dir"] = str(self.private / "bad-output")
        with self.assertRaisesRegex(ValueError, "must not enter a model cache"):
            controller.validate_plan(self.plan)

    def test_busy_common_lock_creates_no_session(self):
        fd = os.open(self.lock, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                controller.run_session(self.plan, "0" * 64, b"{}")
            self.assertFalse(Path(self.plan["session_dir"]).exists())
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
