"""Targeted accounting fixtures; no vLLM/GPU or policy implementation."""
import json
from pathlib import Path
import tempfile
import unittest
from analyze_repeated_kv_service import analyze, diagnostic, eligibility, performance


def request(rid, tokens, times, completion):
    return dict(request_id=rid, document_id=rid, status="completed", arrival_s=0,
        completion_s=completion, output_token_ids=tokens, token_times_s=times,
        stop_reason="stop", prompt_token_ids_sha256="prompt", max_output_tokens=8)


class Accounting(unittest.TestCase):
    def test_terminal_receipt_and_zero_output(self):
        raw = dict(status="COMPLETE", observation_end_s=5, requests=[
            request("a", [], [], 4), request("b", [9], [1], 5)])
        out = performance(raw)
        self.assertEqual(out["mean_completion_s"], 4.5)
        self.assertEqual(out["requests"][1]["last_output_s"], 1)
        self.assertIsNone(out["requests"][0]["ttft_s"])
        self.assertIsNone(out["max_engine_return_gap_s"])
        self.assertEqual(out["ttft_request_count"], 1)

    def test_loaded_prefix_pending_store_and_boundary(self):
        cfg = dict(protect_progress_fraction=.9, max_absences_per_request=8, min_residency_steps=30)
        snap = dict(step=2, host_perf_counter_s=103, free_blocks=3, protected_id=None, protected_reserve=0,
            running_ids=["v"], waiting_ids=["r"], skipped_ids=[], cohort_active=True, plan_victim=None, plan_target=None,
            requests={"r": dict(computed=0, prompt=32, output=1, max_output=64, status="PREEMPTED", held_blocks=0),
                "v": dict(computed=34, prompt=32, output=3, max_output=64, status="RUNNING", held_blocks=3)},
            tracker=dict(config=cfg, absence_count={}, resident_since={}, absent_since={"r": 1}, last_swap_step=-100))
        store = dict(job_id=1, before_perf_s=102.5, after_perf_s=102.6, is_store=True, request="r", accepted=True)
        load = dict(job_id=2, before_perf_s=103.2, after_perf_s=103.3, is_store=False, request="r", accepted=True)
        offload = dict(dispatch=[store, load], completed_jobs=[dict(time_s=104, jobs=[dict(job_id=1)])])
        e = eligibility(snap, "r", 100, offload["dispatch"], {"1": 104})
        self.assertTrue(e["direct"])
        self.assertEqual(e["pending_store_jobs"], [1])
        snap["protected_id"], snap["protected_reserve"] = "other", 1
        self.assertFalse(eligibility(snap, "r", 100, offload["dispatch"], {"1": 104})["direct"])
        snap["protected_id"], snap["protected_reserve"] = None, 0
        raw = dict(measurement_origin_perf_counter_s=100, observation_end_s=10,
            internal_to_source={"r": "source"}, requests=[request("source", [1, 2], [1, 8], None)],
            engine_steps=[dict(completed=True, scheduler_step_start=2, scheduler_step_end=3, start_s=3, returned_s=3.5),
                dict(completed=True, scheduler_step_start=3, scheduler_step_end=4, start_s=5, returned_s=8)],
            scheduler_steps=[dict(step=3, scheduled=[dict(internal_request_id="r", computed_adjustment=32,
                scheduled_tokens=1, recompute_tokens=1)])],
            preemption_events=[dict(victim_internal_request_id="r", attempted_step=step, host_perf_counter_s=t,
                victim_state=dict(output_tokens=n, computed_tokens=c, block_counts=[3]), victim_state_after=dict(computed_tokens=0))
                for step, t, n, c in [(1, 102, 1, 32), (4, 109, 2, 33)]])
        out = diagnostic(raw, dict(eligibility_snapshots=[snap]), offload)
        segment = out["segments"][0]
        self.assertAlmostEqual(segment["S"]["time_s"], 3.2)
        self.assertEqual(segment["S_host_engine_call"]["time_s"], 3)
        self.assertEqual((segment["L_s"], segment["F_s"]), (1, 8))
        self.assertEqual(segment["confirmed_recompute_tokens"], 1)
        self.assertEqual(segment["positive_computed_adjustments"], 32)
        self.assertEqual(out["counts"]["one_two_output_reinvalidated"], 1)

    def test_outer_failure_and_unrun_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ("off", "on"):
                folder = root/("block0-"+arm)
                folder.mkdir()
                (folder/"status.json").write_text(json.dumps(dict(status="INCOMPLETE" if arm == "on" else "COMPLETE")))
                (folder/"raw.json").write_text(json.dumps(dict(status="COMPLETE", observation_end_s=5,
                    requests=[request("a", [1], [1], 5)])))
            out = analyze(root)
            self.assertEqual(len(out["cells"]), 6)
            self.assertEqual(out["cells"]["diag-off"]["status"], "UNRUN")
            self.assertEqual(out["cells"]["block0-on"]["status"], "INCOMPLETE")
            self.assertEqual(out["performance_comparisons"][0]["status"], "NOT_COMPARABLE")


if __name__ == "__main__":
    unittest.main()
