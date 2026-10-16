"""Temporary CPU fixtures only; no research measurements or GPU execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import plot_policy_comparisons as plot


def fixture(path, *, reduced=False):
    regime = {key: "CPU_fixture" for key in plot.REGIME_FIELDS}
    regime.update(model="CPU_fixture_model", draft_source="ngram", admission_mode="wave",
                  execution_mode="eager", measurement_kind="fixed_length_generation")
    runs, controls, coverage = [], [], []
    for index, role in enumerate(plot.ROLES):
        for batch in (1, 4, 16, 64):
            repeats = [0] if role == "native_threshold" and batch == 4 else [0, 1]
            coverage.append({"role": role, "batch": batch, "expected_run_records": 2,
                "observed_run_records": len(repeats), "repeats": repeats,
                "status": "measured" if len(repeats) == 2 else "pending"})
            for repeat in repeats:
                source = f"/CPU_fixture/{role}/perf-r{repeat}-b{batch}.json"
                energy = None if role == "native_unrestricted" and batch == 16 and repeat == 1 else .008 * (index + 1) + batch * .0001 + repeat * .001
                row = {**regime, "source": source, "source_sha256": "a" * 64,
                    "sample_indices_sha256": "b" * 64, "evidence_kind": "measured_run",
                    "role": role, "batch": batch, "repeat": repeat,
                    "effective_tpot_ms": 4 + index + batch * .05 + repeat * .4,
                    "population_counter_joules_per_token": energy,
                    "population_counter_status": "validated_cumulative_counter_span" if energy else "unavailable",
                    "joules_per_token": energy * .9 if energy else None}
                runs.append(row)
                controls.append({**{k: row[k] for k in (*plot.REGIME_FIELDS, "source", "sample_indices_sha256", "role", "batch", "repeat")},
                    "phase": "decode", "steps": 50,
                    "policy_choose_cpu_us_mean": 6 + .02 * batch + repeat if role == "adaptive" else None,
                    "policy_choose_cpu_us_observed_steps": 48 if role == "adaptive" else 0,
                    "discarded_available_token_fraction": (.2 + .002 * batch + repeat * .01) if role == "adaptive" else None if role == "ar" else 0,
                    "drafts_available_before": 0 if role == "ar" else 500,
                    "truncated_by_controller": 100 if role == "adaptive" else 0})
    obj = {"test_fixture": True, "status": "pending" if reduced else "pending_or_partial",
        "protocol": {"batches": [1, 4, 16, 64], "repeats": 2},
        "runs": runs, "control_runs": controls, "coverage": coverage}
    if reduced:
        obj.update(domain="CPU_fixture_domain", execution_domain={"draft_model": "CPU_fixture_draft"}, F3=[], F4=[], F6=[])
    else:
        obj.update(summary=[], control_summary=[])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


class PolicyPlotTests(unittest.TestCase):
    def test_both_comparison_schemas_raw_values_and_median_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for reduced in (False, True):
                path = fixture(root / str(reduced) / "comparison.json", reduced=reduced)
                groups, raw, _ = plot.load_comparison(path)
                self.assertEqual(len(groups), 1)
                cells = plot.prepare(groups, raw)
                def cell(metric, role, batch):
                    return next(r for r in cells if (r["metric"], r["role"], r["batch"]) == (metric, role, batch))
                self.assertEqual(cell("effective_tpot_ms", "native_threshold", 4)["observed_values"], 1)
                self.assertIsNone(cell("effective_tpot_ms", "native_threshold", 4)["median"])
                self.assertIsNone(cell("population_counter_joules_per_token", "native_unrestricted", 16)["median"])
                self.assertIsNone(cell("policy_choose_cpu_us_mean", "ar", 1)["median"])
                tex = plot.document(next(iter(groups.values())), cells, ["effective_tpot_ms"])
                self.assertIn("Native table", tex)
                self.assertIn("Native threshold", tex)
                self.assertIn("(4,nan)", tex)
                self.assertIn("unbounded coords=jump", tex)
                self.assertIn("SYNTHETIC CPU FIXTURE", tex)
                self.assertNotIn("error bars", tex)

    def test_comparisons_never_pool_and_missing_input_stays_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = fixture(root / "a/comparison.json"), fixture(root / "b/comparison.json", reduced=True)
            result = plot.analyze([first, second], root / "out", no_plot=True)
            self.assertEqual(len(result["groups"]), 2)
            self.assertEqual(result["figure_artifacts"], [])
            self.assertTrue(result["fixture_only"])
            pending = plot.analyze([root / "missing"], root / "pending")
            self.assertEqual(pending["status"], "pending_no_measurements")
            self.assertEqual(pending["figure_artifacts"], [])

    def test_invalid_counter_or_duplicate_repeat_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fixture(Path(tmp) / "comparison.json")
            obj = json.loads(path.read_text())
            obj["runs"][0]["population_counter_status"] = "unvalidated"
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "counter-span energy lacks"):
                plot.load_comparison(path)
            fixture(path)
            obj = json.loads(path.read_text())
            duplicate = copy.deepcopy(obj["runs"][0])
            duplicate["source"] += ".duplicate"
            obj["runs"].append(duplicate)
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, "duplicate role/batch/repeat"):
                plot.load_comparison(path)


if __name__ == "__main__":
    unittest.main()
