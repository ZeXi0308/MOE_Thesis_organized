"""Deterministic cohort selection and prepared-input compatibility checks."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from run_capacity import arrival_traces, main, select_source_requests, validate_prepared


class InputSelectionTest(unittest.TestCase):
    def setUp(self):
        self.source = dict(max_prompt_tokens=8, requests=[
            dict(request_id=f"r{i}", prompt_token_count=n) for i, n in enumerate((1, 4, 2, 4, 4, 3, 4))])
        self.config = dict(prompt_tokens=4, requests=2, arrival_gap_s=0.1, burst_size=1)

    def workload(self, offset=0):
        return dict(source_requests=select_source_requests(self.source, 4, 2, offset),
                    actual_prompt_token_ids=[[1, 2, 3, 4], [5, 6, 7, 8]],
                    arrival_traces_s=arrival_traces(2, 0.1, 1))

    def test_old_default_and_config_without_offset(self):
        self.assertEqual([r["request_id"] for r in select_source_requests(self.source, 4, 2)], ["r1", "r3"])
        validate_prepared(self.workload(), self.config, self.source)

    def test_offset_is_applied_after_length_filtering(self):
        self.assertEqual([r["request_id"] for r in select_source_requests(self.source, 4, 2, 2)], ["r4", "r6"])
        validate_prepared(self.workload(2), dict(self.config, request_offset=2), self.source)

    def test_negative_and_out_of_bounds_offsets_fail(self):
        for offset in (-1, 3, 4):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                select_source_requests(self.source, 4, 2, offset)

    def test_prepared_selection_drift_and_cli_override_fail(self):
        with self.assertRaisesRegex(ValueError, "prepared cohort differs"):
            validate_prepared(self.workload(), dict(self.config, request_offset=2), self.source)
        argv = ["run_capacity.py", "--prepared-dir", "/unused-prepared", "--output-dir", "/unused-output", "--request-offset", "2"]
        with patch.object(sys, "argv", argv), patch.object(Path, "read_text") as read, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
