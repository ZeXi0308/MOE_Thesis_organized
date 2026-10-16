"""CPU checks for the read-only asset gate; all files are temporary fixtures."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import check_qwen3_awq_ready as gate


class AssetReadyTests(unittest.TestCase):
    def test_required_hashes_gate_ready_and_optional_absence_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_bytes(b"{}")
            entries = [{"name":"config.json", "bytes":2,
                        "sha256":hashlib.sha256(b"{}").hexdigest()},
                       {"name":"README.md", "bytes":2, "sha256":"0"*64,
                        "required_for_native_load":False}]
            result = gate.check_files(root, entries)
            self.assertTrue(result["ready"])
            self.assertEqual(result["files"][1]["status"], "missing")
            (root / "config.json").write_bytes(b"[]")  # Same size, wrong content.
            result = gate.check_files(root, entries)
            self.assertFalse(result["ready"])
            self.assertEqual(result["files"][0]["status"], "hash_or_read_error")
            self.assertEqual((root / "config.json").read_bytes(), b"[]")

    def test_missing_weight_defers_large_hash_and_never_promotes_prior_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weight1").write_bytes(b"large")
            entries = [{"name":"weight1", "size":5, "sha256":hashlib.sha256(b"large").hexdigest(),
                        "observed_hash_verified":True},
                       {"name":"weight2", "size":5, "sha256":hashlib.sha256(b"large").hexdigest()}]
            with patch.object(gate, "SMALL_FILE_BYTES", 2), patch.object(gate.common, "verify") as verify:
                result = gate.check_files(root, entries)
            self.assertFalse(result["ready"])
            verify.assert_not_called()
            self.assertEqual(result["files"][0]["status"], "hash_deferred_until_required_sizes_complete")
            self.assertEqual(result["files"][1]["status"], "missing")
            (root / "weight2").write_bytes(b"large")
            self.assertTrue(gate.check_files(root, entries)["ready"])


if __name__ == "__main__":
    unittest.main()
