"""CPU tests with synthetic fixtures; these are not experimental results."""
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cost_model import (AcceptanceEWMA, AdaptiveGammaPolicy, CalibrationTable,
                        LayerWorkingSet, WorkingSetCostModel, expected_distinct,
                        expected_distinct_slot_approx, expected_tokens_geometric)
from analyze_routes import (analyze_exact, analyze_histograms, load_records,
                            parse_exact_routes, probability_estimate, replay_windows)
from plot_results import load_runs, summarize, _context


class ProbabilityTests(unittest.TestCase):
    def test_single_token_topk_is_exact(self):
        q = [8 / 64] * 64
        self.assertAlmostEqual(expected_distinct(q, 1, 0), 8)
        self.assertLess(expected_distinct_slot_approx([1 / 64] * 64, 1, 0, 8), 8)
        self.assertAlmostEqual(expected_distinct(q, 1, 4), 64 * (1 - (7 / 8) ** 5))

    def test_extremes_and_saturation(self):
        self.assertEqual(expected_distinct([1, 0, 1, 0], 100, 10), 2)
        self.assertAlmostEqual(expected_distinct([.1] * 10, 1000, 0), 10)
        for args in (([math.nan], 1, 0), ([1.1], 1, 0), ([.2], 0, 0)):
            with self.assertRaises(ValueError):
                expected_distinct(*args)

    def test_geometric_boundaries(self):
        self.assertEqual(expected_tokens_geometric(1, 7), 8)
        self.assertEqual(expected_tokens_geometric(0, 7), 1)
        self.assertAlmostEqual(expected_tokens_geometric(.7, 4), 2.7731)

    def test_empirical_survival_differs_from_geometric(self):
        estimator = AcceptanceEWMA(decay=0)
        estimator.update([3, 0], [3, 3])
        self.assertEqual(estimator.prefix_survival(3), [.5, .5, .5])
        self.assertEqual(estimator.expected_tokens(3), 2.5)
        self.assertEqual(estimator.alpha, .75)  # 3 accepts / 4 tested positions
        self.assertNotEqual(estimator.expected_tokens(3), expected_tokens_geometric(.75, 3))

    def test_variable_draft_lengths_and_no_draft(self):
        estimator = AcceptanceEWMA(decay=0)
        estimator.update([1, 3, 0], [1, 3, 0])
        self.assertEqual(estimator.expected_tokens(3), 4)
        self.assertEqual(estimator.exposures, {1: 2, 2: 1, 3: 1})
        before = estimator.snapshot()
        estimator.update([0], [0])
        self.assertEqual(before, estimator.snapshot())
        with self.assertRaises(ValueError):
            estimator.update([2], [1])

    def test_count_ratio_distinguishes_varying_step_exposure(self):
        weighted = AcceptanceEWMA(alpha=0, decay=.5, prior_tested_count=0)
        legacy = AcceptanceEWMA(alpha=0, decay=.5, alpha_estimator="legacy_step_ratio", prior_tested_count=0)
        # First step rejects immediately: 0/1. Next accepts all three: 3/3.
        # Decayed count ratio = 1.5/(.25+1.5), rather than EMA(0,1)=.5.
        for estimator in (weighted, legacy):
            estimator.update([0], [7])
            estimator.update([3], [3])
        self.assertAlmostEqual(weighted.alpha, 6/7)
        self.assertAlmostEqual(legacy.alpha, .5)
        self.assertEqual(weighted.snapshot()["alpha_counts"]["observed_tested"], 4)
        self.assertEqual(weighted.snapshot()["alpha_counts"]["observed_accepted"], 3)

    def test_same_accepts_different_tested_counts(self):
        full = AcceptanceEWMA(decay=.5, prior_tested_count=0)
        partial = AcceptanceEWMA(decay=.5, prior_tested_count=0)
        for _ in range(2):
            full.update([1], [1])
            partial.update([1], [7])  # accepted token plus first rejected position
        self.assertEqual(full.observed_accepted, partial.observed_accepted)
        self.assertEqual(full.alpha, 1)
        self.assertEqual(partial.alpha, .5)
        self.assertEqual(partial.observed_tested, 4)  # remaining draft slots are untested

    def test_count_prior_and_first_exposure(self):
        estimator = AcceptanceEWMA(alpha=.7, decay=.9)
        before = estimator.snapshot()
        estimator.update([], [])
        estimator.update([0], [0])
        estimator.update_availability([0], 1)
        self.assertEqual(before["alpha_counts"], estimator.snapshot()["alpha_counts"])
        self.assertEqual(estimator.alpha, .7)
        estimator.update([3], [7])
        self.assertAlmostEqual(estimator.alpha, (.9*.7 + .1*3) / (.9*1 + .1*4))
        self.assertEqual(estimator.snapshot()["alpha_prior"], {"alpha":.7,"tested_count":1.,"accepted_count":.7})
        no_prior = AcceptanceEWMA(alpha=.7, decay=.9, prior_tested_count=0)
        no_prior.update([3], [7])
        self.assertAlmostEqual(no_prior.alpha, .75)

    def test_count_estimator_boundaries_and_legacy_reproduction(self):
        for alpha, accepts in ((0, [0]), (1, [3])):
            estimator = AcceptanceEWMA(alpha=alpha, decay=0)
            estimator.update(accepts, [3])
            self.assertEqual(estimator.alpha, alpha)
        legacy = AcceptanceEWMA(alpha=.7, decay=.9, alpha_estimator="legacy_step_ratio")
        legacy.update([3], [7])
        self.assertAlmostEqual(legacy.alpha, .9*.7+.1*.75)
        for kwargs in ({"alpha_estimator":"unknown"}, {"prior_tested_count":-1}, {"prior_tested_count":math.nan}):
            with self.assertRaises(ValueError):
                AcceptanceEWMA(**kwargs)

    def test_no_available_drafts_implies_one_output(self):
        estimator = AcceptanceEWMA(alpha=1)
        estimator.update_availability([0, 0, 0], proposal_budget=7)
        self.assertEqual(estimator.expected_tokens(7), 1)

    def test_availability_observed_before_truncation(self):
        estimator = AcceptanceEWMA(alpha=1, decay=0)
        estimator.update_availability([3, 0], proposal_budget=3)
        self.assertEqual(estimator.expected_tokens(3), 2.5)
        # Selecting gamma=1 reveals no evidence about availability of ranks 2,3.
        estimator.update_availability([1, 1], proposal_budget=1)
        self.assertEqual(estimator.availability[3], .5)


class CalibrationTests(unittest.TestCase):
    def test_complete_cost_changes_optimum(self):
        # Verification is faster for gamma=3, but full drafting makes it worse.
        table = CalibrationTable([
            {"batch": 1, "gamma": 0, "total_step_ms": 2},
            {"batch": 1, "gamma": 3, "verify_ms": 3, "draft_ms": 8,
             "components_nonoverlap": True}])
        decision = AdaptiveGammaPolicy(table, AcceptanceEWMA(alpha=1)).choose(1)
        self.assertEqual(decision.gamma, 0)
        self.assertEqual(len(decision.candidates), 2)

    def test_no_silent_missing_draft_cost(self):
        with self.assertRaises(ValueError):
            CalibrationTable([{"batch": 1, "gamma": 1, "verify_ms": 2}])
        with self.assertRaises(ValueError):
            CalibrationTable([{"batch": 1, "gamma": 1, "verify_ms": 2, "draft_ms": 3}])

    def test_postproposal_replacement_retains_full_cost(self):
        table = CalibrationTable([
            {"batch": 1, "gamma": 0, "total_step_ms": 2, "draft_ms": 0},
            {"batch": 1, "gamma": 3, "total_step_ms": 10, "draft_ms": 7}],
            {"draft_cost_nonoverlap": True})
        self.assertEqual(table.estimate(1, 0, draft_gamma=3).total_step_ms, 9)
        self.assertEqual(AdaptiveGammaPolicy(table, AcceptanceEWMA(alpha=1)).choose(1, draft_gamma=3).gamma, 3)

    def test_postproposal_budget_specific_timings(self):
        table = CalibrationTable([
            {"batch": 1, "gamma": 0, "proposal_budget": 3, "total_step_ms": 9},
            {"batch": 1, "gamma": 3, "proposal_budget": 3, "total_step_ms": 10}])
        self.assertEqual(table.estimate(1, 0, 3).total_step_ms, 9)
        self.assertEqual(AdaptiveGammaPolicy(table).choose(1, draft_gamma=3).gamma, 3)

    def test_postproposal_does_not_invent_draft_components(self):
        table = CalibrationTable([{"batch": 1, "gamma": 0, "total_step_ms": 2},
                                  {"batch": 1, "gamma": 3, "total_step_ms": 10}])
        with self.assertRaises(KeyError):
            table.estimate(1, 0, draft_gamma=3)

    def test_interpolate_repeats_no_extrapolation(self):
        table = CalibrationTable([{"batch": 1, "gamma": 0, "total_step_ms": 2},
                                  {"batch": 1, "gamma": 0, "total_step_ms": 4},
                                  {"batch": 3, "gamma": 0, "total_step_ms": 7}])
        self.assertEqual(table.lookup(1, 0).total_step_ms, 3)
        self.assertEqual(table.lookup(2, 0).total_step_ms, 5)
        self.assertIn("interpolation", table.lookup(2, 0).source)
        with self.assertRaises(KeyError):
            table.lookup(4, 0)
        table.update(1, 0, 5, decay=.5)
        self.assertEqual(table.lookup(1, 0).total_step_ms, 4)

    def test_mixed_regimes_rejected(self):
        with self.assertRaises(ValueError):
            CalibrationTable([{"batch": 1, "gamma": 0, "total_step_ms": 1, "model": "A"},
                              {"batch": 1, "gamma": 1, "total_step_ms": 1, "model": "B"}])
        with self.assertRaises(ValueError):
            CalibrationTable([{"batch": 1, "gamma": 0, "total_step_ms": 1, "execution_mode": "eager"},
                              {"batch": 1, "gamma": 1, "total_step_ms": 1, "execution_mode": "compiled"}])

    def test_logical_bytes_not_traffic(self):
        model = WorkingSetCostModel([LayerWorkingSet([.5, .5], 100)], 100)
        row = model.predict(1, 0, draft_ms=2, attention_ms=3)
        self.assertEqual(row["logical_weight_working_set_bytes"], 100)
        self.assertEqual(row["total_step_ms"], 6)
        self.assertEqual(row["evidence_kind"], "analytic_prediction_not_measured")


class TraceTests(unittest.TestCase):
    @staticmethod
    def fixture():
        return [{"sample_id": str(s), "layer": 0, "token_position": t,
                 "rank": 1, "expert_id": s % 2}
                for s in range(6) for t in range(8)]

    def test_exact_routes_and_correlation(self):
        routes = parse_exact_routes(self.fixture(), 2)
        probabilities = probability_estimate(routes["0"], 2)
        self.assertEqual(probabilities["q_e"], [.5, .5])
        self.assertEqual(set(replay_windows(routes["0"], 1, 3)), {1})
        self.assertGreater(expected_distinct(probabilities["q_e"], 1, 3), 1)

    def test_duplicate_route_rejected(self):
        with self.assertRaises(ValueError):
            parse_exact_routes([self.fixture()[0], self.fixture()[0]], 2)

    def test_fit_holdout_are_disjoint_and_correction_fit_only(self):
        output, rows = analyze_exact(self.fixture(), 2, [1, 64], [0, 3])
        self.assertFalse(set(output["fit_samples"]) & set(output["holdout_samples"]))
        self.assertEqual(len(output["fit_samples"]) + len(output["holdout_samples"]), 6)
        single = [r for r in rows if r["batch"] == 1 and r["gamma"] == 3]
        self.assertTrue(all(r["fit_corrected_distinct"] == 1 for r in single))
        self.assertTrue(all(r["window_count"] == 0 for r in rows if r["batch"] == 64))

    def test_histogram_native_schema_and_phase_filter(self):
        records = [{"model_phase": "target", "phase": "decode", "layer": "l0",
                    "tokens": 2, "expert_token_counts": [2, 1, 1], "step": 1},
                   {"model_phase": "draft", "phase": "decode", "layer": "l0",
                    "tokens": 1, "expert_token_counts": [0, 0, 1]}]
        output, comparison = analyze_histograms(records, 3, True)
        self.assertEqual(output["layers"][0]["q_e"], [1, .5, .5])
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison[0]["observed_distinct"], 3)
        with self.assertRaises(ValueError):
            analyze_histograms(records, 3, False)

    def test_wave_json_routes_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "waves.json"
            path.write_text(json.dumps({"episodes": [{"plugin": {"routes": [{"layer": 1}]}},
                                                     {"plugin": {"routes": [{"layer": 2}]}}]}))
            self.assertEqual([r["layer"] for r in load_records(path)], [1, 2])


class PerformanceAnalysisTests(unittest.TestCase):
    def test_admission_and_capacity_have_separate_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.json"
            path.write_text(json.dumps([
                {"batch": 16, "gamma": 0, "tokens_per_second": 10, "engine_max_num_seqs": 64},
                {"batch": 16, "gamma": 3, "tokens_per_second": 20, "engine_max_num_seqs": 64, "admission_mode": "queued"},
                {"batch": 16, "gamma": 3, "tokens_per_second": 20, "engine_max_num_seqs": 16}]))
            runs, _ = load_runs([path])
            self.assertEqual(runs[0]["admission_mode"], "wave")
            self.assertTrue(all(row["speedup"] is None for row in summarize(runs) if row["gamma"] == 3))
        with self.assertRaises(ValueError):
            CalibrationTable([{"batch": 1, "gamma": 0, "total_step_ms": 1},
                              {"batch": 1, "gamma": 1, "total_step_ms": 1, "admission_mode": "queued"}])
        with self.assertRaises(ValueError):
            CalibrationTable([{"batch": 1, "gamma": 0, "total_step_ms": 1, "engine_max_num_seqs": 64},
                              {"batch": 1, "gamma": 1, "total_step_ms": 1, "engine_max_num_seqs": 16}])

    def test_execution_modes_have_separate_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.json"
            path.write_text(json.dumps([
                {"batch": 1, "gamma": 0, "tokens_per_second": 10, "execution_mode": "eager"},
                {"batch": 1, "gamma": 3, "tokens_per_second": 20, "execution_mode": "compiled"}]))
            runs, _ = load_runs([path])
            summaries = summarize(runs)
            self.assertIsNone(next(r for r in summaries if r["execution_mode"] == "compiled")["speedup"])

    def test_engine_configuration_and_native_route_return(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perf-r0-b1.json"
            engine = Path(directory) / "engine_args.json"
            path.write_text(json.dumps({"batch": 1, "gamma": 0, "tokens_per_second": 10, "trace_enabled": False}))
            config = {"enforce_eager": True, "dtype": "bfloat16", "quantization": None,
                      "kv_cache_memory_bytes": 1024, "max_model_len": 4096,
                      "kernel_config": {"moe_backend": "triton", "other": 1},
                      "attention_config": {"backend": "FLASH_ATTN"}}
            engine.write_text(json.dumps(config))
            runs, excluded = load_runs([path])
            self.assertEqual(runs[0]["execution_mode"], "eager")
            self.assertEqual(runs[0]["quantization"], "none")
            self.assertEqual(runs[0]["kv_cache_memory_bytes"], "1024")
            self.assertEqual(json.loads(runs[0]["kernel_backend"])["kernel_config"]["moe_backend"], "triton")
            config["enable_return_routed_experts"] = True
            engine.write_text(json.dumps(config))
            runs, excluded = load_runs([path])
            self.assertFalse(runs)
            self.assertEqual(excluded[0]["reason"], "route instrumentation enabled")

    def test_run_values_preserved_and_diagnostic_excluded(self):
        records = [{"label": f"perf-r{i}-b1", "batch": 1, "gamma": gamma,
                    "proposal_budget": gamma, "tokens_per_second": tps,
                    "all_finished": True, "mode": "native", "model": "test",
                    "draft_source": "ngram", "measurement_kind": "fixed_length_generation",
                    "episodes": [{"requests": [{"effective_tpot_ms": 10 / tps}]}]}
                   for gamma, values in ((0, [10, 20]), (3, [20, 40]))
                   for i, tps in enumerate(values)]
        records.extend([{**records[0], "trace_enabled": True},
                        {**records[0], "label": "warmup"},
                        {**records[0], "all_finished": False}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(records))
            runs, excluded = load_runs([path])
            self.assertEqual(len(runs), 4)
            self.assertEqual(len(excluded), 3)
            rows = summarize(runs)
            gamma3 = next(row for row in rows if row["gamma"] == 3)
            self.assertEqual(gamma3["speedup"], 2)
            self.assertEqual(gamma3["n_runs"], 2)
            self.assertEqual(json.loads(gamma3["tokens_per_second_raw"]), [20, 40])
            self.assertEqual(gamma3["tpot_speedup"], 2)

    def test_missing_baseline_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps({"batch": 1, "gamma": 3, "tokens_per_second": 20}))
            runs, _ = load_runs([path])
            self.assertIsNone(summarize(runs)[0]["speedup"])


if __name__ == "__main__":
    unittest.main()
