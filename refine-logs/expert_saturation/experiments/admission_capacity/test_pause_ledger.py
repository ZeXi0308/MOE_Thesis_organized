import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from analyze_pause_ledger import analyze, read_raw


def fixture():
    return dict(status="INCOMPLETE", error="retained failure", observation_end_s=6,
        internal_to_source={"internal-a": "a"},
        requests=[dict(request_id="a", internal_request_id="internal-a", status="completed",
                       arrival_s=0, admission_s=0.1, token_times_s=[1, 6], output_token_ids=[4, 5]),
                  dict(request_id="failed", status="failed", arrival_s=0,
                       token_times_s=[], output_token_ids=[]),
                  dict(request_id="future", status="unfinished", arrival_s=8,
                       token_times_s=[], output_token_ids=[])],
        scheduler_steps=[dict(step=0, start_s=5, scheduled=[dict(request_id="a",
            internal_request_id="internal-a", recompute_tokens=10)])],
        engine_steps=[dict(scheduler_step_start=0, scheduler_step_end=1,
            start_s=5, returned_s=5.5, completed=True)],
        preemption_events=[dict(victim_internal_request_id="internal-a")])


class PauseLedgerTests(unittest.TestCase):
    def test_gap_partition_retains_failed_future_and_submission_boundary(self):
        result = analyze(fixture())
        self.assertEqual((result["n_planned"], result["n_arrived"]), (3, 2))
        self.assertEqual(result["request_status_counts"], dict(completed=1, failed=1, unfinished=1))
        row = result["requests"][0]
        self.assertEqual((row["submission_s"], row["first_service_s"]), (0.1, 5))
        self.assertEqual(row["partition"], dict(before_first_recompute_call_s=4,
            recompute_calls_span_s=0.5, after_last_recompute_call_s=0.5, recompute_step_ids=[0]))
        self.assertEqual(sum(row["partition"][k] for k in row["partition"] if k.endswith("_s")), row["max_itl_s"])

    def test_missing_recovery_is_not_zero_cost(self):
        raw = fixture()
        raw["engine_steps"][0]["completed"] = False
        result = analyze(raw)
        self.assertIsNone(result["requests"][0]["partition"])
        self.assertEqual(result["partition_unavailable_for_preempted_requests"], ["a"])

    def test_call_crossing_receipt_boundary_is_not_silently_truncated(self):
        raw = fixture()
        raw["engine_steps"][0]["start_s"] = 0.5
        self.assertIsNone(analyze(raw)["requests"][0]["partition"])

    def test_corrupt_identity_and_step_ownership_are_rejected(self):
        for kind in ("request", "step", "owner", "time", "tokens"):
            raw = copy.deepcopy(fixture())
            if kind == "request": raw["scheduler_steps"][0]["scheduled"][0]["request_id"] = "failed"
            if kind == "step": raw["scheduler_steps"][0]["step"] = 2
            if kind == "owner": raw["engine_steps"].append(copy.deepcopy(raw["engine_steps"][0]))
            if kind == "time": raw["requests"][0]["token_times_s"] = [6, 1]
            if kind == "tokens": raw["requests"][0]["output_token_ids"] = [4]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                analyze(raw)

    def test_gzip_and_plain_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain, compressed = Path(tmp) / "raw.json", Path(tmp) / "raw.json.gz"
            plain.write_text(json.dumps(fixture()))
            with gzip.open(compressed, "wt") as stream:
                json.dump(fixture(), stream)
            self.assertEqual(analyze(read_raw(plain)), analyze(read_raw(compressed)))


if __name__ == "__main__":
    unittest.main()
