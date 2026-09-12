"""Synthetic bundles only: accounting/identity tests, never capacity evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_capacity import analyze, report


class AnalyzeCapacityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-capacity-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plans = [dict(repeat=0, regime="steady", cap=cap, telemetry=on)
                      for cap in (1, 2) for on in (False, True)]
        self.write("config.json", dict(fixture_kind="synthetic_unit_test_not_scientific_data",
            run_order=self.plans, requests=2, burst_size=2, arrival_gap_s=1,
            ttft_slo_s=5, tpot_slo_s=5, cap_schedule=None))
        self.write("workload.json", dict(source_requests=[dict(request_id=f"r{i}", document_id=f"d{i}")
            for i in range(2)], actual_prompt_token_ids=[[1, 2], [3, 4]],
            arrival_traces_s=dict(steady=[0, 1], bursty=[0, 1])))
        for index, plan in enumerate(self.plans):
            rows = [dict(request_id=f"r{i}", document_id=f"d{i}", arrival_s=i,
                admission_s=i, completion_s=i + 2, prompt_tokens=2,
                prompt_token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                output_token_ids=[7, 8], token_times_s=[i + 1, i + 2], status="completed")
                for i, ids in enumerate([[1, 2], [3, 4]])]
            steps = [dict(request_ids=[f"r{i}"], itl_s=[1],
                          pressure=[dict(U=0.5, C=2)] if plan["telemetry"] else []) for i in range(2)]
            self.write(f"cell-{index:03d}.json", dict(plan=plan, status="COMPLETE", requests=rows,
                observation_end_s=5, steps=steps))
        self.write("curves.csv", "this file is deliberately not trusted")

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def cell(self, index):
        return json.loads((self.root / f"cell-{index:03d}.json").read_text())

    def test_frozen_request_document_arrival_and_prompt_identity(self):
        original = self.cell(2)
        for key, bad_value in (("request_id", "alien"), ("document_id", "other"),
                               ("arrival_s", 0.25), ("prompt_token_ids_sha256", "wrong")):
            with self.subTest(key=key):
                bad = json.loads(json.dumps(original))
                bad["requests"][0][key] = bad_value
                self.write("cell-002.json", bad)
                result = analyze(self.root)
                self.assertEqual(result["verdict"], "INVALID_EXPERIMENT")
                self.assertEqual(result["cells"][2]["state"], "INVALID")
                self.assertFalse(result["complete_scan"])
        self.write("cell-002.json", original)

    def test_missing_cell_and_failed_request_keep_denominator_and_partial_state(self):
        (self.root / "cell-001.json").unlink()
        failed = self.cell(0)
        failed.update(status="INCOMPLETE")
        failed["requests"][1].update(status="failed", completion_s=None)
        self.write("cell-000.json", failed)
        result = analyze(self.root)
        self.assertEqual(result["verdict"], "PARTIAL_MEASUREMENT")
        self.assertEqual(result["cells"][1]["state"], "MISSING")
        metrics = result["cells"][0]["metrics"]
        self.assertEqual((metrics["n_arrived"], metrics["n_failed"], metrics["slo_attainment"]), (2, 1, 0.5))
        self.assertEqual(metrics["goodput_rps"], 0.2)
        group = result["static_comparisons"][0]
        self.assertEqual(group["observed_best_cap"], 2)
        self.assertFalse(group["complete_off_scan"])
        self.assertEqual([c["cap"] for c in group["candidates"]], [2])

    def test_on_future_tokens_membership_and_timing_are_diagnostics(self):
        on = self.cell(1)
        for row in on["requests"]:
            row["token_times_s"] = [v + 0.5 for v in row["token_times_s"]]
            row["completion_s"] += 0.5
            row["output_token_ids"][1] = 9
        on["observation_end_s"] = 6
        on["steps"] = [dict(request_ids=["r0", "r1"], itl_s=[1.5, 1.5], pressure=[dict(U=1, C=1)])]
        self.write("cell-001.json", on)
        result = analyze(self.root)
        self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
        pair = result["telemetry_pairs"][0]
        self.assertFalse(pair["membership_sequence_equal"])
        self.assertEqual(pair["differing_request_ids"]["output_token_ids"], ["r0", "r1"])
        self.assertEqual(pair["on_minus_off"]["wall_time_s"], 1)
        self.assertEqual(pair["on_minus_off"]["ttft_p50_s"], 0.5)
        self.assertNotIn("own_on_pressure", result["cells"][0])
        self.assertEqual(result["cells"][1]["own_on_pressure"]["U"]["mean"], 1)
        group = result["static_comparisons"][0]
        self.assertEqual(group["adjacent_cap_deltas"][0]["high_minus_low"],
                         dict(goodput_rps=0, ttft_p50_s=0, tpot_p50_s=0, n_slo_pass=0))
        self.assertTrue(group["calibration_hints"][0].startswith("ALL_ATTAIN_ONE"))
        self.assertIn("synthetic_unit_test_not_scientific_data", report(result))
        self.assertEqual(result["action_increment_verdict"], "NO_ACTION_INCREMENT_ESTABLISHED")

    def test_no_cells_remains_unrun_without_tokenized_workload(self):
        for path in self.root.glob("cell-*.json"):
            path.unlink()
        self.write("workload.json", dict(source_requests=[]))
        result = analyze(self.root)
        self.assertEqual(result["verdict"], "UNRUN")
        self.assertEqual(result["identity_validation"], "not_checked_no_cells")
        self.assertEqual(result["static_comparisons"], [])
        self.assertFalse(result["complete_scan"])

    def test_actual_width_queue_and_helper_exposure_without_treating_drain_as_invalid(self):
        for index, plan in enumerate(self.plans):
            raw = self.cell(index)
            for step in raw["steps"]:
                step.update(ordinary=dict(actual_active=1, decode_requests=1, waiting_requests=0,
                                          future_requests=0, target_cap=plan["cap"]), telemetry_s=0.25)
            for row in raw["requests"]:
                row["admission_s"] += 0.25
            self.write(f"cell-{index:03d}.json", raw)
        result = analyze(self.root)
        exposure = result["cells"][2]["execution_exposure"]
        self.assertEqual(exposure["distributions"]["actual_active"]["counts"], {1: 2})
        self.assertEqual(exposure["target_snapshot"]["at_target_steps"], 0)
        self.assertEqual(exposure["static_cap_reached"], "NOT_REACHED")
        self.assertEqual(exposure["distributions"]["future_requests"]["counts"], {0: 2})
        self.assertEqual(exposure["waiting_positive_steps"], 0)
        self.assertEqual(exposure["waiting_sample_fraction"], 0)
        self.assertEqual(exposure["admitted_request_queue_s"]["n_positive"], 2)
        self.assertEqual(exposure["admitted_request_queue_s"]["mean"], 0.25)
        self.assertEqual(exposure["slo_checks"]["joint"]["complete_cohort_hint"], "ALL_PASS")
        self.assertEqual(exposure["telemetry_helper"]["total_s"], 0.5)
        self.assertEqual(exposure["telemetry_helper"]["episode_wall_fraction"], 0.1)
        self.assertEqual(result["cells"][2]["metrics"]["observation_duration_s"], 5)
        self.assertEqual(result["cells"][2]["metrics"]["goodput_rps"], 0.4)
        hints = result["static_comparisons"][0]["calibration_hints"]
        for prefix in ("NO_WAITING_AT_DECODE_SNAPSHOTS", "TARGET_NOT_OBSERVED cap=2", "SAME_MAX_DECODE_WIDTH"):
            self.assertTrue(any(h.startswith(prefix) for h in hints), hints)
        raw = self.cell(0)
        raw["steps"][0]["ordinary"].update(actual_active=2, decode_requests=2, waiting_requests=1)
        self.write("cell-000.json", raw)
        config = json.loads((self.root / "config.json").read_text())
        config["cap_schedule"] = [[0, 2], [1, 1]]
        self.write("config.json", config)
        result = analyze(self.root)
        self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
        exposure = result["cells"][0]["execution_exposure"]
        self.assertEqual(exposure["static_cap_reached"], "NOT_APPLICABLE_DYNAMIC_CAP")
        self.assertEqual(exposure["target_snapshot"]["above_target_steps"], 1)
        self.assertEqual(exposure["waiting_positive_steps"], 1)
        self.assertIn("helper only", report(result))

    def test_legacy_missing_or_partial_step_fields_are_not_measured_zero(self):
        raw = self.cell(0)
        raw["steps"][0].update(ordinary=dict(actual_active=1, decode_requests=1, waiting_requests=0,
                                            target_cap=1), telemetry_s=0.25)
        self.write("cell-000.json", raw)
        result = analyze(self.root)
        partial = result["cells"][0]["execution_exposure"]
        missing = result["cells"][2]["execution_exposure"]
        self.assertEqual(partial["distributions"]["actual_active"]["status"], "PARTIAL")
        self.assertEqual(partial["telemetry_helper"]["status"], "PARTIAL")
        self.assertIsNone(partial["telemetry_helper"]["total_s"])
        self.assertEqual(missing["distributions"]["waiting_requests"]["status"], "UNMEASURED")
        self.assertEqual(missing["static_cap_reached"], "UNAVAILABLE")
        self.assertIsNone(partial["waiting_sample_fraction"])
        self.assertIsNone(missing["waiting_sample_fraction"])
        self.assertIsNone(missing["waiting_positive_steps"])
        self.assertIsNone(missing["target_snapshot"]["at_target_steps"])
        self.assertIsNone(missing["telemetry_helper"]["episode_wall_fraction"])
        hints = result["static_comparisons"][0]["calibration_hints"]
        self.assertTrue(any(h.startswith("EXPOSURE_UNMEASURED_OR_PARTIAL") for h in hints))
        self.assertFalse(any(h.startswith("NO_WAITING_AT_DECODE_SNAPSHOTS") for h in hints))

    def test_required_post_cell_checks_keep_metrics_but_exclude_unverified_cells(self):
        config = json.loads((self.root / "config.json").read_text())
        config["cell_checks_required"] = True
        self.write("config.json", config)
        for index in range(4):
            self.write(f"checks-{index:03d}.json", dict(status="PASS", gpu_after={}))
        for status in (None, "FAILED", "PASS"):
            with self.subTest(status=status):
                if status is None:
                    (self.root / "checks-002.json").unlink()
                else:
                    self.write("checks-002.json", dict(status=status, reason="test-only isolation result", gpu_after={}))
                result = analyze(self.root)
                cell = result["cells"][2]
                self.assertEqual(cell["metrics"]["n_completed"], 2)
                self.assertEqual(cell["state"], "COMPLETE" if status == "PASS" else "INVALID")
                candidates = result["static_comparisons"][0]["candidates"]
                self.assertEqual([c["cap"] for c in candidates], [1, 2] if status == "PASS" else [1])
                if status != "PASS":
                    self.assertIn("post-cell isolation check", " ".join(cell["issues"]))


if __name__ == "__main__":
    unittest.main()
