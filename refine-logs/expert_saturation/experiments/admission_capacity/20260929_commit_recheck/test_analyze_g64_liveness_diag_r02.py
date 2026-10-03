"""Small CPU checks for the exact diagnostic branch inference and readback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_g64_liveness_diag_r02 import AuditError, infer_native_refusal, verify_archive


def requirement(call, stage, required, free):
    return {"allocation_call": call, "native_stage_hint": stage,
            "required_blocks_returned": required, "free_blocks_at_call": free}


class DiagnosticAnalysisTest(unittest.TestCase):
    def test_full_sequence_precheck_refusal(self):
        allocation = {"call": 7, "returned_none": True,
                      "actual_kwargs": {"full_sequence_must_fit": True, "reserved_blocks": 2}}
        rows = [requirement(7, "full_sequence_precheck", 8, 6)]
        result = infer_native_refusal(allocation, rows, 0)
        self.assertEqual(result["branch"], "FULL_SEQUENCE_FIT_REFUSAL")

    def test_reserved_blocks_are_decisive_only_when_raw_free_fits(self):
        allocation = {"call": 9, "returned_none": True,
                      "actual_kwargs": {"full_sequence_must_fit": True, "reserved_blocks": 1}}
        rows = [requirement(9, "full_sequence_precheck", 5, 6),
                requirement(9, "incremental_or_other", 6, 6)]
        result = infer_native_refusal(allocation, rows, 0)
        self.assertEqual(result["branch"], "INCREMENTAL_FREE_MINUS_RESERVED_REFUSAL")
        self.assertTrue(result["incremental_check"]["reserved_is_decisive"])
        rows[-1]["required_blocks_returned"] = 7
        result = infer_native_refusal(allocation, rows, 0)
        self.assertFalse(result["incremental_check"]["reserved_is_decisive"])

    def test_missing_or_contradictory_native_rows_remain_unknown(self):
        allocation = {"call": 1, "returned_none": True,
                      "actual_kwargs": {"full_sequence_must_fit": True, "reserved_blocks": 1}}
        self.assertEqual(infer_native_refusal(allocation, [], 0)["branch"], "UNKNOWN")
        rows = [requirement(1, "full_sequence_precheck", 8, 6),
                requirement(1, "incremental_or_other", 1, 6)]
        self.assertEqual(infer_native_refusal(allocation, rows, 0)["branch"], "UNKNOWN")

    def test_archive_hash_readback_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            cell = Path(directory)
            archive = cell / "archive"
            archive.mkdir()
            file = archive / "no-progress-observer.json"
            file.write_text('{"status":"SNAPSHOT_CAPTURED"}\n')
            expected = hashlib.sha256(file.read_bytes()).hexdigest()
            (cell / "output_sha256.json").write_text(json.dumps({file.name: expected}))
            _, hashes, _ = verify_archive(cell)
            self.assertEqual(hashes[file.name], expected)
            file.write_text('{"status":"CHANGED"}\n')
            with self.assertRaises(AuditError):
                verify_archive(cell)


if __name__ == "__main__":
    unittest.main()
