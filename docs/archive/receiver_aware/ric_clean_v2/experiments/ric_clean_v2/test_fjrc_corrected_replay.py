from __future__ import annotations

from dataclasses import replace
import unittest

try:
    from .fjrc_corrected_level1 import ServiceLUT, select_holdout_scenarios, select_split_scenarios
    from .test_fjrc_corrected_level1 import joins_fixture
    from .fjrc_corrected_replay import (
        Q_ONLY_POLICIES,
        ReplayConfig,
        ReplayError,
        aggregate_campaign,
        aggregate_baselines,
        calibrate_deadline_on_selection,
        choose_policy_action,
        decide_two_model,
        evaluate_baselines,
        materialize_replay,
        negative_controls,
        optimize_information,
        paired_bootstrap,
        receiver_state,
        run_campaign,
        simulate,
        validate_scenario,
    )
except ImportError:  # pragma: no cover
    from fjrc_corrected_level1 import ServiceLUT, select_holdout_scenarios, select_split_scenarios  # type: ignore
    from test_fjrc_corrected_level1 import joins_fixture  # type: ignore
    from fjrc_corrected_replay import (  # type: ignore
        Q_ONLY_POLICIES,
        ReplayConfig,
        ReplayError,
        aggregate_campaign,
        aggregate_baselines,
        calibrate_deadline_on_selection,
        choose_policy_action,
        decide_two_model,
        evaluate_baselines,
        materialize_replay,
        negative_controls,
        optimize_information,
        paired_bootstrap,
        receiver_state,
        run_campaign,
        simulate,
        validate_scenario,
    )


def service() -> ServiceLUT:
    return ServiceLUT("olmoe", 1.0, 0.25, 2.0, 0.5, 3.25, "a" * 64)


def scenarios(config: ReplayConfig | None = None):
    native = select_holdout_scenarios("olmoe", joins_fixture(), service())
    frozen = config or ReplayConfig(bootstrap_replicates=100)
    return [materialize_replay(value, service(), frozen) for value in native]


class CorrectedReplayTests(unittest.TestCase):
    def test_materialization_has_heterogeneous_timing_and_identical_q(self):
        scenario = scenarios()[0]
        validate_scenario(scenario)
        self.assertEqual(receiver_state(scenario, 0), receiver_state(scenario, 1))
        self.assertNotEqual(scenario.joins[0].arrival_us, scenario.joins[1].arrival_us)
        self.assertNotEqual(scenario.joins[0].deadline_us, scenario.joins[1].deadline_us)
        self.assertIn("SYNTHETIC", scenario.timing_source)

    def test_four_stage_ledger_and_exact_census(self):
        scenario = scenarios()[0]
        result = simulate(scenario, 0, scenario.candidate_task_ids[0])
        accounting = result["accounting"]
        for stage in ("pack_count", "cut_count", "unpack_count"):
            self.assertEqual(accounting[stage], accounting["task_universe"])
        self.assertEqual(accounting["combine_count"], 2)
        prior = result["prior_events"]
        for left, right in zip(prior, prior[1:]):
            self.assertLessEqual(left["unpack_end_us"], right["pack_start_us"])
        for event in result["task_events"]:
            self.assertLessEqual(event["pack_start_us"], event["pack_end_us"])
            self.assertLessEqual(event["pack_end_us"], event["cut_start_us"])
            self.assertLessEqual(event["cut_end_us"], event["unpack_start_us"])

    def test_q_only_baselines_cannot_branch_on_world(self):
        scenario = scenarios()[0]
        for policy in Q_ONLY_POLICIES:
            self.assertEqual(
                choose_policy_action(scenario, 0, policy),
                choose_policy_action(scenario, 1, policy),
            )
        self.assertEqual(set(evaluate_baselines(scenario)), set(Q_ONLY_POLICIES) | {"join_credit"})

    def test_negative_controls_are_zero_value(self):
        controls = negative_controls(scenarios()[0])
        self.assertTrue(controls["shuffled_key"]["passed"])
        self.assertTrue(controls["equal_phase"]["passed"])

    def test_information_oracle_never_makes_exact_optimum_worse(self):
        for scenario in scenarios()[:3]:
            report = optimize_information(scenario)
            self.assertEqual(report["q_map_fingerprints"][0], report["q_map_fingerprints"][1])
            self.assertLessEqual(report["R"]["metrics"].objective, report["Q"]["metrics"].objective)

    def test_campaign_has_16_pairs_and_32_requests(self):
        reports = [optimize_information(value) for value in scenarios()]
        aggregate = aggregate_campaign(reports)
        self.assertEqual(aggregate["Q"].request_count, 32)
        self.assertEqual(aggregate["R"].request_count, 32)
        with self.assertRaisesRegex(ReplayError, "16"):
            aggregate_campaign(reports[:-1])

    def test_cluster_bootstrap_is_deterministic(self):
        reports = [optimize_information(value) for value in scenarios()]
        first = paired_bootstrap(reports, replicates=50, seed=7)
        second = paired_bootstrap(reports, replicates=50, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["unit"], "matched_request_pair")

    def test_deadline_calibration_reads_q_only_selection(self):
        selection = select_split_scenarios("olmoe", joins_fixture(), service(), split="selection")
        effective, audit = calibrate_deadline_on_selection(
            selection,
            service(),
            ReplayConfig(bootstrap_replicates=10),
        )
        self.assertEqual(effective.deadline_factor, audit["selected_deadline_factor"])
        self.assertFalse(audit["r_outcomes_read_for_selection"])
        self.assertGreaterEqual(audit["selected_q_miss_rate"], 0.20)
        self.assertLessEqual(audit["selected_q_miss_rate"], 0.80)

    def test_two_model_gate_is_and_without_pooling(self):
        passed = {"status": "PASS"}
        failed = {"status": "FAIL"}
        self.assertEqual(decide_two_model({"olmoe": passed, "llmjp": passed})["status"], "PASS")
        self.assertEqual(decide_two_model({"olmoe": passed, "llmjp": failed})["status"], "FAIL")
        with self.assertRaisesRegex(ReplayError, "exactly"):
            decide_two_model({"olmoe": passed})

    def test_future_multiset_drift_is_rejected(self):
        scenario = scenarios()[0]
        prior1 = next(
            iter(set(scenario.worlds[1].prior_task_ids) - set(scenario.worlds[0].prior_task_ids))
        )
        changed = tuple(
            replace(task, ready_us=task.ready_us + 0.1) if task.task_id == prior1 else task
            for task in scenario.tasks
        )
        with self.assertRaisesRegex(ReplayError, "future work/resource multiset"):
            validate_scenario(replace(scenario, tasks=changed))

    def test_small_campaign_dry_run(self):
        config = ReplayConfig(bootstrap_replicates=20)
        report = run_campaign(scenarios(config), config)
        self.assertEqual(report["status"], "LOGICAL_TRACE_REPLAY_ONLY")
        self.assertEqual(report["aggregate"]["Q"].request_count, 32)
        self.assertEqual(len(report["baseline_reports"]), 16)
        self.assertEqual(len(report["baseline_aggregate"]), 6)
        self.assertEqual(len(report["negative_controls"]), 16)


if __name__ == "__main__":
    unittest.main()
