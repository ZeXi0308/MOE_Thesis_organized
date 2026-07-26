from __future__ import annotations

from dataclasses import replace
import unittest

try:
    from .fjrc_oracle_core import (
        BackgroundJob,
        Contribution,
        FJRCError,
        Join,
        Scenario,
        World,
        aggregate_16_pair_reports,
        empirical_cvar90,
        enumerate_actions,
        equal_map_negative_control,
        fanout1_negative_control,
        optimize_information_arms,
        simulate,
        uninformative_key_negative_control,
    )
except ImportError:  # pragma: no cover
    from fjrc_oracle_core import (  # type: ignore
        BackgroundJob,
        Contribution,
        FJRCError,
        Join,
        Scenario,
        World,
        aggregate_16_pair_reports,
        empirical_cvar90,
        enumerate_actions,
        equal_map_negative_control,
        fanout1_negative_control,
        optimize_information_arms,
        simulate,
        uninformative_key_negative_control,
    )


def fixture() -> Scenario:
    # Two 2-way fork-join requests and two common senders.  Each action bit
    # chooses whether request A or B goes first on that sender.
    tasks = (
        Contribution("a0", "ja", "ra", 0, 0, 0, 1, 1, 2, "decision"),
        Contribution("b0", "jb", "rb", 0, 1, 0, 1, 1, 2, "decision"),
        Contribution("a1", "ja", "ra", 1, 0, 0, 1, 1, 2, "decision"),
        Contribution("b1", "jb", "rb", 1, 1, 0, 1, 1, 2, "decision"),
    )
    joins = (
        Join("ja", "ra", 0, 0, 8, 1, ("a0", "a1")),
        Join("jb", "rb", 1, 0, 8, 1, ("b0", "b1")),
    )
    # The same reachable backlog history is assigned to opposite receiver
    # identities.  The deadline makes the first service choice actionable.
    w0 = World("w0", (BackgroundJob("hot", 0, -1, 3),))
    w1 = World("w1", (BackgroundJob("hot", 1, -1, 3),))
    return Scenario("fixture", 0, tasks, joins, (w0, w1))


def fanout1_fixture() -> Scenario:
    tasks = (
        Contribution("a0", "ja0", "ra0", 0, 0, 0, 1, 1, 1, "decision"),
        Contribution("b0", "jb0", "rb0", 0, 1, 0, 1, 1, 1, "decision"),
    )
    joins = tuple(
        Join(f"j{task.task_id}", f"r{task.task_id}", task.receiver_rank, 0, 100, 1, (task.task_id,))
        for task in tasks
    )
    worlds = (
        World("f0", (BackgroundJob("hot", 0, -1, 2),)),
        World("f1", (BackgroundJob("hot", 1, -1, 2),)),
    )
    return Scenario("fanout1", 0, tasks, joins, worlds, kind="fanout1_control")


def renamed(scenario: Scenario, index: int) -> Scenario:
    suffix = f"-{index:02d}"
    task_ids = {task.task_id: task.task_id + suffix for task in scenario.contributions}
    joins = tuple(
        replace(
            join,
            join_id=join.join_id + suffix,
            request_id=join.request_id + suffix,
            sibling_task_ids=tuple(task_ids[value] for value in join.sibling_task_ids),
        )
        for join in scenario.joins
    )
    join_ids = {old.join_id: new.join_id for old, new in zip(scenario.joins, joins)}
    request_ids = {old.request_id: new.request_id for old, new in zip(scenario.joins, joins)}
    tasks = tuple(
        replace(
            task,
            task_id=task_ids[task.task_id],
            join_id=join_ids[task.join_id],
            request_id=request_ids[task.request_id],
        )
        for task in scenario.contributions
    )
    worlds = tuple(
        replace(
            world,
            world_id=world.world_id + suffix,
            background_jobs=tuple(replace(job, job_id=job.job_id + suffix) for job in world.background_jobs),
        )
        for world in scenario.worlds
    )
    return replace(scenario, scenario_id=scenario.scenario_id + suffix, contributions=tasks, joins=joins, worlds=worlds)


class FJRCOracleCoreTests(unittest.TestCase):
    def test_action_vector_is_binary_per_common_sender(self):
        actions = enumerate_actions(fixture())
        self.assertEqual(len(actions), 4)
        self.assertTrue(all(len(action) == 2 for action in actions))
        with self.assertRaisesRegex(FJRCError, "cover every decision sender"):
            simulate(fixture(), 0, ((0, "a0"),))
        with self.assertRaisesRegex(FJRCError, "outside its sender"):
            simulate(fixture(), 0, ((0, "not-a-task"), (1, "a1")))

    def test_full_stage_accounting_and_join_close(self):
        scenario = fixture()
        result = simulate(scenario, 0, enumerate_actions(scenario)[0])
        self.assertEqual(
            result["accounting"],
            {
                "foreground_contributions": 4,
                "pack_count": 4,
                "cut_count": 4,
                "unpack_count": 4,
                "combine_count": 2,
            },
        )
        self.assertEqual({row["join_id"] for row in result["combine_events"]}, {"ja", "jb"})
        for row in result["combine_events"]:
            self.assertGreaterEqual(row["combine_start_us"], row["join_ready_us"])
            self.assertGreater(row["join_close_us"], row["combine_start_us"])

    def test_future_siblings_are_not_omitted(self):
        scenario = fixture()
        future = (
            Contribution("af", "ja", "ra", 2, 0, 5, 1, 1, 1, "fixed_after"),
            Contribution("bf", "jb", "rb", 3, 1, 6, 1, 1, 1, "fixed_after"),
        )
        joins = (
            replace(scenario.joins[0], deadline_us=30, sibling_task_ids=("a0", "a1", "af")),
            replace(scenario.joins[1], deadline_us=30, sibling_task_ids=("b0", "b1", "bf")),
        )
        expanded = replace(scenario, contributions=scenario.contributions + future, joins=joins)
        result = simulate(expanded, 0, enumerate_actions(expanded)[0])
        self.assertEqual(result["accounting"]["foreground_contributions"], 6)
        self.assertEqual(result["accounting"]["pack_count"], 6)
        self.assertEqual(result["accounting"]["unpack_count"], 6)
        self.assertTrue({"af", "bf"}.issubset({row["task_id"] for row in result["sender_events"]}))

    def test_receiver_queue_uses_arrival_then_task_identity(self):
        scenario = fixture()
        result = simulate(scenario, 0, (((0, "a0"), (1, "a1"))))
        receiver_zero = [
            row for row in result["receiver_events"] if row["receiver_rank"] == 0
        ]
        self.assertEqual(receiver_zero[0]["identity"], "background")
        self.assertEqual([row["identity"] for row in receiver_zero[1:]], ["a0", "a1"])
        self.assertGreaterEqual(receiver_zero[1]["unpack_start_us"], receiver_zero[0]["unpack_end_us"])

    def test_r0_enumeration_can_branch_b_cannot(self):
        report = optimize_information_arms(fixture())
        for policy in report["B"]["optimal_policies"]:
            self.assertEqual(policy[0], policy[1])
        self.assertLessEqual(report["R0"]["metrics"].objective, report["B"]["metrics"].objective)
        self.assertTrue(report["actionable"])

    def test_equal_map_and_uninformative_key_controls_are_zero(self):
        equal = equal_map_negative_control(fixture())
        shuffled = uninformative_key_negative_control(fixture())
        self.assertTrue(equal["passed"])
        self.assertTrue(shuffled["passed"])

    def test_explicit_fanout1_negative_control(self):
        control = fanout1_negative_control(fanout1_fixture())
        self.assertTrue(control["passed"])

    def test_cvar90_uses_four_of_32_requests(self):
        values = [0.0] * 28 + [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(empirical_cvar90(values), 2.5)

    def test_missing_sibling_is_rejected(self):
        scenario = fixture()
        broken_join = replace(scenario.joins[0], sibling_task_ids=("a0",))
        broken = replace(scenario, joins=(broken_join, scenario.joins[1]))
        with self.assertRaisesRegex(FJRCError, "fork-join fanout|union of full sibling"):
            enumerate_actions(broken)

    def test_world_history_multiset_drift_is_rejected(self):
        scenario = fixture()
        bad_world = World("bad", (BackgroundJob("hot", 1, -1, 8),))
        with self.assertRaisesRegex(FJRCError, "history multiset"):
            enumerate_actions(replace(scenario, worlds=(scenario.worlds[0], bad_world)))

    def test_decision_sender_requires_two_requests(self):
        scenario = fixture()
        corrupted = tuple(
            replace(task, sender_rank=0) if task.task_id == "a1" else
            replace(task, sender_rank=1) if task.task_id == "b0" else task
            for task in scenario.contributions
        )
        with self.assertRaisesRegex(FJRCError, "one decision task from each request"):
            enumerate_actions(replace(scenario, contributions=corrupted))


if __name__ == "__main__":
    unittest.main()
