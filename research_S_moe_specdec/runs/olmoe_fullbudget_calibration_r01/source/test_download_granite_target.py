"""Tiny local fixtures only; no model assets, network or GPU operations."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download_draft as common
import download_granite_target as granite


class GraniteTargetChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="granite_target_test_", dir="/private/tmp" if Path("/private/tmp").is_dir() else None)
        self.root = Path(self.temp.name).resolve()
        self.models = self.root / "models"
        self.models.mkdir()
        self.target = self.models / "granite3b" / granite.REVISION / "target"
        self.reuse = self.root / "read_only_B"
        self.reuse.mkdir()
        self.model_patch = patch.object(common, "MODELS", self.models)
        self.model_patch.start()

    def tearDown(self):
        self.model_patch.stop()
        self.temp.cleanup()

    def entry(self, name="asset.bin", payload=b"fixed fixture"):
        return {"name": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                "git_blob_sha1": None, "repo_id": granite.REPO, "revision": granite.REVISION,
                "url": "https://example.invalid/never-contacted"}

    def test_copy_is_independent_regular_file_and_B_unchanged(self):
        entry = self.entry()
        source = self.reuse / entry["name"]
        source.write_bytes(b"fixed fixture")
        before = source.stat()
        jobs = granite.plan_transfer([entry], self.target, self.reuse, set())
        granite.copy_verified(jobs[0]["source"], self.target, entry)
        dest = self.target / entry["name"]
        self.assertFalse(dest.is_symlink())
        self.assertNotEqual(dest.stat().st_ino, source.stat().st_ino)
        self.assertEqual(source.read_bytes(), b"fixed fixture")
        self.assertEqual(source.stat().st_mtime_ns, before.st_mtime_ns)
        dest.write_bytes(b"independent")
        self.assertEqual(source.read_bytes(), b"fixed fixture")

    def test_all_sources_checked_before_copy(self):
        good, bad = self.entry("good"), self.entry("bad")
        (self.reuse / "good").write_bytes(b"fixed fixture")
        (self.reuse / "bad").write_bytes(b"bad")
        with self.assertRaises(ValueError):
            granite.plan_transfer([good, bad], self.target, self.reuse, set())
        self.assertFalse(self.target.exists())

    def test_source_changes_after_preflight_are_not_published(self):
        entry = self.entry()
        source = self.reuse / entry["name"]
        source.write_bytes(b"fixed fixture")
        granite.plan_transfer([entry], self.target, self.reuse, set())
        source.write_bytes(b"other fixture")
        with self.assertRaises(ValueError):
            granite.copy_verified(source, self.target, entry)
        self.assertFalse((self.target / entry["name"]).exists())

    def test_traversal_and_symlinks_rejected(self):
        outside = self.root / "outside"
        outside.write_bytes(b"fixed fixture")
        (self.reuse / "link").symlink_to(outside)
        for name in ("../outside", str(outside), "a\\b", "link"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                granite.source_file(self.reuse, name)
        self.target.mkdir(parents=True)
        (self.target / "asset.bin.part").symlink_to(outside)
        with self.assertRaises(ValueError):
            granite.plan_transfer([self.entry()], self.target, self.reuse, {"asset.bin"})
        self.assertEqual(outside.read_bytes(), b"fixed fixture")

    def test_missing_reuse_is_not_silently_downloaded(self):
        with self.assertRaises(ValueError):
            granite.plan_transfer([self.entry()], self.target, self.reuse, set())
        jobs = granite.plan_transfer([self.entry()], self.target, self.reuse, {"asset.bin"})
        self.assertEqual(jobs[0]["kind"], "pinned_network_download")

    def test_bad_existing_destination_is_not_overwritten(self):
        self.target.mkdir(parents=True)
        (self.target / "asset.bin").write_bytes(b"bad")
        with self.assertRaises(ValueError):
            granite.plan_transfer([self.entry()], self.target, self.reuse, {"asset.bin"})
        self.assertEqual((self.target / "asset.bin").read_bytes(), b"bad")

    def test_resume_uses_existing_common_range_downloader(self):
        payload = b"0123456789abcdef"
        entry = self.entry(payload=payload)
        self.target.mkdir(parents=True)
        (self.target / "asset.bin.part").write_bytes(payload[:6])
        response = io.BytesIO(payload[6:])
        response.status = 206
        response.headers = {"Content-Range": f"bytes 6-15/{len(payload)}"}
        with patch.object(common, "request", return_value=response) as request:
            self.assertEqual(common.download(entry, self.target), entry["sha256"])
        request.assert_called_once_with(entry["url"], 6)
        self.assertEqual((self.target / "asset.bin").read_bytes(), payload)
        self.assertFalse((self.target / "asset.bin.part").exists())

    def test_pinned_metadata_and_upstream_mismatch(self):
        source = json.loads((common.PROJECT / "configs/granite3b_target_source.json").read_text())
        entries = granite.entries_from_source(source, common.ENDPOINTS[0], self.target)
        actual = copy.deepcopy(entries)
        for entry in actual:
            if entry.get("git_blob_sha1"):
                entry["sha256"] = None
        granite.check_inventory(entries, actual)
        actual[-1]["size"] += 1
        with self.assertRaises(ValueError):
            granite.check_inventory(entries, actual)
        source["revision"] = "0" * 40
        with self.assertRaises(ValueError):
            granite.entries_from_source(source, common.ENDPOINTS[0], self.target)


if __name__ == "__main__":
    unittest.main()
