"""Synthetic calibration fixtures exercise boundaries, not performance claims."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_calibration import analyze_episode, analyze_run
from cost_model import CalibrationTable


def fixture():
    steps = []
    for index in range(4):
        steps.append({"step": index, "episode": "wave0", "phase": "decode" if index else "prefill",
            "decode_batch": 2 if index else 0, "scheduled_tokens": 2,
            "scheduled_tokens_by_request": {"a": 1, "b": 1},
            "scheduled_drafts_by_request": {"a": 2}, "observed_drafts_by_request": {"a": 2},
            "accepted_by_request": {"a": 1}, "proposed_before_by_request": {"a": 3, "b": 0},
            "proposal_budget_next": 3, "control_ns": 1000, "acceptance_update_ns": 2000,
            "drafts_available_before": 3, "truncated_by_controller": 1,
            "ewma_censored_requests": ["a"] if index == 3 else [],
            "decision": {"gamma": 2, "expected_tokens_per_request": 2.0, "decision_cpu_us": .25},
            "alpha_after": .5})
    return {"label": "wave0", "host_steps": [
        {"index": i, "host_step_ms": value, "elapsed_s": i + 1.0}
        for i, value in enumerate([100, 30, 10, 5])], "plugin": {"steps": steps}}


class CalibrationBuildTests(unittest.TestCase):
    def test_full_step_calibration_keeps_cold_decode_spike(self):
        wave = analyze_episode(fixture(), 2, 2, 3, "synthetic")
        self.assertEqual(wave["eligible_steps"], 2)
        self.assertEqual(wave["mean_step_ms"], 20)
        self.assertEqual(wave["step_ms_max"], 30)
        self.assertEqual(wave["excluded_steps"]["terminal_or_output_budget_censored"], 1)

    def test_no_draft_terminal_detected_by_runner_timestamp(self):
        data = fixture()
        data["plugin"]["steps"][-1]["ewma_censored_requests"] = []
        data["requests"] = [{"finished": True, "chunks": [{"elapsed_s": 4.0}]}]
        wave = analyze_episode(data, 2, 2, 3, "synthetic")
        self.assertEqual(wave["eligible_steps"], 2)

    def test_alignment_and_actual_batch_required(self):
        data = fixture()
        data["host_steps"].pop()
        with self.assertRaises(ValueError):
            analyze_episode(data, 2, 2, 3, "synthetic")
        data = fixture()
        data["plugin"]["steps"][1]["decode_batch"] = 1
        self.assertEqual(analyze_episode(data, 2, 2, 3, "synthetic")["mean_step_ms"], 10)

    def test_f5_keeps_zero_proposal_population(self):
        wave = analyze_episode(fixture(), 2, 2, 3, "synthetic")
        stats = wave["acceptance"]
        self.assertEqual(stats["eligible_requests"], 4)
        self.assertEqual(stats["zero_scheduled_requests"], 2)
        self.assertEqual(stats["accepted_drafts"], 2)
        self.assertEqual(stats["tested_positions"], 4)
        self.assertEqual(wave["progress"][0]["mean_output_tokens"], 1.5)

    def test_run_is_one_table_repeat_and_budget_is_preserved(self):
        wave0 = fixture()
        wave1 = copy.deepcopy(wave0)
        for host in wave1["host_steps"]:
            host["host_step_ms"] *= 2
        data = {"batch": 2, "gamma": 2, "proposal_budget": 3, "mode": "static", "label": "run0",
                "episodes": [wave0, wave1]}
        result = analyze_run(data, "synthetic")
        self.assertEqual(result["calibration"]["total_step_ms"], 30)
        self.assertEqual(result["calibration"]["eligible_steps"], 4)
        table = CalibrationTable([result["calibration"]])
        self.assertEqual(table.estimate(2, 2, draft_gamma=3).total_step_ms, 30)
        with self.assertRaises(KeyError):
            table.estimate(2, 2, draft_gamma=7)
        self.assertEqual(result["acceptance"][0]["availability_survival"], .5)
        self.assertEqual(result["control"][0]["drafts_truncated_by_controller"], 6)

    def test_trace_cost_not_admitted(self):
        with self.assertRaises(ValueError):
            analyze_run({"batch": 2, "gamma": 2, "trace_enabled": True}, "synthetic")
        with self.assertRaises(ValueError):
            analyze_run({"batch": 2, "gamma": 2, "trace_enabled": False}, "synthetic",
                        {"native_route_return": True})

    def test_execution_configuration_reaches_calibration_rows(self):
        result = analyze_run({"batch": 2, "gamma": 2, "proposal_budget": 3, "episodes": [fixture()]},
                             "synthetic", {"execution_mode": "compiled", "dtype": "bfloat16",
                             "kernel_backend": '{"moe_backend":"triton"}'})
        self.assertEqual(result["calibration"]["execution_mode"], "compiled")
        self.assertEqual(result["acceptance"][0]["execution_mode"], "compiled")
        self.assertEqual(result["control"][0]["execution_mode"], "compiled")

    def test_admission_and_missing_timers_preserved(self):
        wave = fixture()
        for step in wave["plugin"]["steps"]:
            step.pop("control_ns", None)
        result = analyze_run({"batch": 2, "gamma": 2, "proposal_budget": 3, "episodes": [wave]},
                             "synthetic", {"admission_mode": "queued", "engine_max_num_seqs": 16})
        self.assertEqual(result["calibration"]["admission_mode"], "queued")
        self.assertEqual(result["calibration"]["engine_max_num_seqs"], "16")
        self.assertIsNone(result["control"][0]["control_us_mean"])
        self.assertEqual(result["control"][0]["control_us_observed_steps"], 0)


if __name__ == "__main__":
    unittest.main()
