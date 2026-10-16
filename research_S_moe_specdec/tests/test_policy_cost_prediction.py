"""Small CPU event fixtures; no synthetic research output is written."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import analyze_policy_cost_prediction as analysis
from cost_model import CalibrationTable
from plot_results import REGIME_FIELDS


def fixture():
    requests = [{"request_id": name, "source_index": index, "finished": True,
                 "chunks": [{"elapsed_s": 1.0, "tokens": 1}, {"elapsed_s": 3.0, "tokens": 1}]}
                for index, name in enumerate(("a", "b"))]
    steps = [{"step": index, "episode": "wave0", "phase": "decode", "control_batch": 2,
        "decode_batch": 2, "scheduled_requests": 2, "scheduled_tokens": 2,
        "scheduled_decode_request_ids": ["a", "b"], "scheduled_tokens_by_request": {"a": 1, "b": 1},
        "scheduled_drafts_by_request": {}, "accepted_by_request": {},
        "proposed_before_by_request": {"a": 0, "b": 0}, "proposal_budget_next": 7,
        "scheduled_drafts": 0, "observed_drafts": 0, "gamma": 3,
        "ewma_censored_requests": [],  # Client terminal at step2 is still detected.
        "decision": {"gamma": 3, "batch": 2, "total_step_ms": 4.0, "source": "measured_cell"}}
        for index in range(3)]
    record = {"label": "perf-r1-b2", "batch": 2, "gamma": 7, "proposal_budget": 7,
        "mode": "adaptive", "all_finished": True, "trace_enabled": False,
        "episodes": [{"label": "wave0", "requests": requests,
            "host_steps": [{"index": index, "elapsed_s": float(index + 1), "host_step_ms": duration}
                           for index, duration in enumerate((5., 3., 100.))],
            "plugin": {"config": {"mode": "adaptive", "proposal_budget": "max", "max_gamma": 7}, "steps": steps}}]}
    table = CalibrationTable([{"batch": 2, "gamma": 3, "proposal_budget": 7, "total_step_ms": 4.0}])
    return record, table


class CostPredictionTests(unittest.TestCase):
    def test_same_step_metrics_terminal_and_batch_mismatch(self):
        record, table = fixture()
        run, diagnostics, rows = analysis.analyze_record(record, "fixture.json", {}, table)
        self.assertEqual(run["status"], "complete")
        self.assertEqual(run["n"], 2)
        self.assertEqual((run["predicted_mean_ms"], run["observed_mean_ms"], run["bias_ms"], run["mae_ms"], run["rmse_ms"]), (4., 4., 0., 1., 1.))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["observed_host_step_ms"], 100.)
        self.assertIn("terminal_or_output_budget_censored", rows[-1]["exclusion_reasons"])
        self.assertEqual(rows[-1]["terminal_recovered_count"], 2)
        self.assertEqual(diagnostics[0]["n"], 2)
        # Keep the cost pair, but exclude a decision made for another B.
        altered = copy.deepcopy(record)
        step = altered["episodes"][0]["plugin"]["steps"][1]
        step["control_batch"] = step["decision"]["batch"] = 3
        expanded = CalibrationTable([{"batch": batch, "gamma": 3, "proposal_budget": 7, "total_step_ms": 4.0} for batch in (2, 3)])
        run, _, rows = analysis.analyze_record(altered, "fixture.json", {}, expanded)
        self.assertEqual(run["n"], 1)
        self.assertIn("precontrol_actual_batch_mismatch", rows[1]["exclusion_reasons"])
        altered["episodes"][0]["host_steps"][1]["index"] = 99
        with self.assertRaisesRegex(ValueError, "index mismatch"):
            analysis.analyze_record(altered, "fixture.json", {}, expanded)

    def test_missing_native_predictions_stay_pending_not_zero(self):
        record, table = fixture()
        for step in record["episodes"][0]["plugin"]["steps"]:
            step["decision"].pop("total_step_ms")
        for mode in ("adaptive", "native"):
            record["mode"] = mode
            record["episodes"][0]["plugin"]["config"]["mode"] = mode
            run, _, rows = analysis.analyze_record(record, "fixture.json", {}, table)
            self.assertEqual(run["status"], "pending")
            self.assertEqual(run["n"], 0)
            self.assertIsNone(run["mae_ms"])
            self.assertTrue(all(row["predicted_total_step_ms"] is None for row in rows))
            self.assertEqual(len(rows), 3)
        record, table = fixture()
        record["episodes"][0]["host_steps"][0].pop("host_step_ms")
        run, _, rows = analysis.analyze_record(record, "fixture.json", {}, table)
        self.assertEqual(run["status"], "pending")
        self.assertIsNone(rows[0]["observed_host_step_ms"])
        self.assertEqual(run["n"], 1)

    def test_budget_execution_domain_and_logged_lookup_must_match(self):
        regime = {key: "fixture" for key in REGIME_FIELDS}
        regime.update(execution_mode="compiled", admission_mode="wave", engine_max_num_seqs="64")
        table = CalibrationTable([{"batch": 2, "gamma": 3, "proposal_budget": 7, "total_step_ms": 4.0}], regime)
        analysis.validate_table(table, regime, 7)
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            analysis.validate_table(table, regime, 3)
        with self.assertRaisesRegex(ValueError, "execution-domain mismatch"):
            analysis.validate_table(table, {**regime, "execution_mode": "eager"}, 7)
        record, _ = fixture()
        record["episodes"][0]["plugin"]["steps"][0]["decision"]["total_step_ms"] = 9.
        with self.assertRaisesRegex(ValueError, "differs from the executed calibration lookup"):
            analysis.analyze_record(record, "fixture.json", {}, table)


if __name__ == "__main__":
    unittest.main()
