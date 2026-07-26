from __future__ import annotations

from dataclasses import replace
import unittest

try:
    from .fjrc_corrected_level0 import (
        CorrectedFJRCError,
        Join,
        PriorCompletion,
        Scenario,
        Task,
        World,
        equal_phase_control,
        fanout1_control,
        legal_actions,
        observation,
        optimize_arm,
        optimize_information_lattice,
        shuffled_key_control,
        simulate,
        validate_scenario,
    )
except ImportError:  # pragma: no cover
    from fjrc_corrected_level0 import (  # type: ignore
        CorrectedFJRCError,
        Join,
        PriorCompletion,
        Scenario,
        Task,
        World,
        equal_phase_control,
        fanout1_control,
        legal_actions,
        observation,
        optimize_arm,
        optimize_information_lattice,
        shuffled_key_control,
        simulate,
        validate_scenario,
    )


def fixture() -> Scenario:
    # All tasks target one receiver. pa/pb have identical resource signatures;
    # each world swaps which identity completed before t0 and which remains as
    # fixed-after work. Candidate tasks are identical across worlds.
    tasks = (
        Task("pa", "ja", "ra", 2, 0, 8.0, 1.0),
        Task("pb", "jb", "rb", 2, 0, 8.0, 1.0),
        Task("a", "ja", "ra", 0, 0, 0.0, 2.0),
        Task("b", "jb", "rb", 1, 0, 0.0, 2.0),
    )
    joins = (
        Join("ja", "ra", 0, 4.0, 0.5, ("pa", "a")),
        Join("jb", "rb", 0, 4.0, 0.5, ("pb", "b")),
    )
    worlds = (
        World("w0", (PriorCompletion("pa", -2.0, -1.0),)),
        World("w1", (PriorCompletion("pb", -2.0, -1.0),)),
    )
    return Scenario(
        "corrected-level0",
        0.0,
        ((0, 0.0),),
        tasks,
        joins,
        ("a", "b"),
        worlds,
    )


def fanout1_fixture() -> Scenario:
    tasks = (
        Task("a", "ja", "ra", 0, 0, 0.0, 2.0),
        Task("b", "jb", "rb", 1, 0, 0.0, 2.0),
    )
    joins = (
        Join("ja", "ra", 0, 10.0, 0.5, ("a",)),
        Join("jb", "rb", 0, 10.0, 0.5, ("b",)),
    )
    return Scenario(
        "fanout1",
        0.0,
        ((0, 0.0),),
        tasks,
        joins,
        ("a", "b"),
        (World("f0", ()), World("f1", ())),
    )


class CorrectedFJRCLevel0Tests(unittest.TestCase):
    def test_scenario_is_reachable_and_actions_are_receiver_candidates(self):
        scenario = fixture()
        validate_scenario(scenario)
        self.assertEqual(legal_actions(scenario), ("a", "b"))

    def test_q_is_identical_and_j_differs(self):
        scenario = fixture()
        self.assertEqual(observation(scenario, 0, "Q"), observation(scenario, 1, "Q"))
        self.assertNotEqual(observation(scenario, 0, "J"), observation(scenario, 1, "J"))
        self.assertNotIn("completed_siblings", observation(scenario, 0, "Q"))
        self.assertNotIn("queue_map", observation(scenario, 0, "J"))

    def test_each_world_completes_full_task_universe_once(self):
        scenario = fixture()
        for world in (0, 1):
            result = simulate(scenario, world, "a")
            accounting = result["accounting"]
            self.assertEqual(accounting["task_universe"], 4)
            self.assertEqual(accounting["prior_count"], 1)
            self.assertEqual(accounting["post_t0_count"], 3)
            self.assertEqual(accounting["unique_completion_count"], 4)
            self.assertEqual(accounting["combine_count"], 2)

    def test_information_lattice_enforces_primary_estimand(self):
        report = optimize_information_lattice(fixture())
        self.assertFalse(report["arms"]["Q"]["observation_distinguishes_worlds"])
        self.assertTrue(report["arms"]["R"]["observation_distinguishes_worlds"])
        self.assertLessEqual(report["arms"]["R"]["metrics"].objective, report["arms"]["Q"]["metrics"].objective)
        self.assertTrue(report["q_to_r_strict_first_action_flip"])
        self.assertEqual(report["arms"]["Q"]["metrics"].miss_rate, 0.75)
        self.assertEqual(report["arms"]["R"]["metrics"].miss_rate, 0.5)
        self.assertEqual(report["arms"]["R"]["selected_policy"], ("a", "b"))

    def test_shuffled_key_has_zero_increment_over_q(self):
        self.assertTrue(shuffled_key_control(fixture())["passed"])

    def test_equal_phase_and_fanout1_controls_are_zero(self):
        self.assertTrue(equal_phase_control(fixture())["passed"])
        self.assertTrue(fanout1_control(fanout1_fixture())["passed"])

    def test_q_policy_cannot_branch_but_r_may(self):
        q = optimize_arm(fixture(), "Q")
        r = optimize_arm(fixture(), "R")
        self.assertTrue(all(policy[0] == policy[1] for policy in q["optimal_policies"]))
        self.assertLessEqual(r["metrics"].objective, q["metrics"].objective)

    def test_future_work_signature_drift_is_rejected(self):
        scenario = fixture()
        changed = tuple(
            replace(task, ready_us=9.0) if task.task_id == "pb" else task for task in scenario.tasks
        )
        with self.assertRaisesRegex(CorrectedFJRCError, "future work/resource multiset"):
            validate_scenario(replace(scenario, tasks=changed))

    def test_prior_completion_after_t0_is_rejected(self):
        scenario = fixture()
        bad = World("bad", (PriorCompletion("pa", 0.0, 1.0),))
        with self.assertRaisesRegex(CorrectedFJRCError, "after t0"):
            validate_scenario(replace(scenario, worlds=(bad, scenario.worlds[1])))

    def test_candidate_or_prior_double_role_is_rejected(self):
        scenario = fixture()
        bad = World("bad", (PriorCompletion("a", -3.0, -1.0),))
        with self.assertRaisesRegex(CorrectedFJRCError, "prior completion identity"):
            validate_scenario(replace(scenario, worlds=(bad, scenario.worlds[1])))

    def test_same_join_phase_is_rejected_for_primary_fixture(self):
        scenario = fixture()
        same = replace(scenario.worlds[0], world_id="same")
        with self.assertRaisesRegex(CorrectedFJRCError, "do not differ in join phase"):
            validate_scenario(replace(scenario, worlds=(scenario.worlds[0], same)))

    def test_multiple_receiver_resources_fail_closed(self):
        scenario = fixture()
        joins = (scenario.joins[0], replace(scenario.joins[1], receiver_rank=1))
        tasks = tuple(
            replace(task, receiver_rank=1) if task.join_id == "jb" else task for task in scenario.tasks
        )
        broken = replace(scenario, tasks=tasks, joins=joins, receiver_available_us=((0, 0.0), (1, 0.0)))
        with self.assertRaisesRegex(CorrectedFJRCError, "exactly one receiver"):
            validate_scenario(broken)


if __name__ == "__main__":
    unittest.main()
