from __future__ import annotations

from dataclasses import replace
import copy
import unittest

try:
    from . import phasemap_oracle_core as core
except ImportError:  # pragma: no cover
    import phasemap_oracle_core as core  # type: ignore


def _primary_tasks() -> tuple[core.Contribution, ...]:
    rows = [
        core.Contribution("a0", "ja", "ra", 0, 0, 1.0, 0.1, 1.0, True),
        core.Contribution("b0", "jb", "rb", 0, 1, 1.0, 0.1, 1.0, True),
        core.Contribution("a1", "ja", "ra", 1, 0, 1.0, 0.1, 1.0, True),
        core.Contribution("b1", "jb", "rb", 1, 1, 1.0, 0.1, 1.0, True),
    ]
    for index in range(2, 6):
        rows.append(core.Contribution(f"a{index}", "ja", "ra", index, 0, 1, 0.1, 1, False))
        rows.append(core.Contribution(f"b{index}", "jb", "rb", index, 1, 1, 0.1, 1, False))
    return tuple(rows)


def _receiver_jobs(
    *, q_bit: int, j_bit: int, equal_q: bool = False, equal_j: bool = False
) -> tuple[core.ReceiverJob, ...]:
    depths = (
        {0: 8, 1: 8}
        if equal_q
        else ({0: 8, 1: 16} if q_bit == 0 else {0: 16, 1: 8})
    )
    phases = {0: 1, 1: 1} if equal_j else ({0: 1, 1: 4} if j_bit == 0 else {0: 4, 1: 1})
    jobs: list[core.ReceiverJob] = []
    for receiver, prefix in ((0, "a"), (1, "b")):
        pending = {f"{prefix}{index}" for index in range(2, 2 + phases[receiver])}
        committed = [f"{prefix}{index}" for index in range(2, 6) if f"{prefix}{index}" not in pending]
        for ordinal, task_id in enumerate(committed):
            jobs.append(core.ReceiverJob(f"commit:{task_id}", receiver, -30 + 2 * ordinal, 1, task_id))
        unfinished: list[str | None] = sorted(pending)
        unfinished.extend([None] * (depths[receiver] - len(unfinished)))
        for ordinal, task_id in enumerate(unfinished):
            job_id = task_id if task_id is not None else f"bg:{receiver}:{q_bit}:{j_bit}:{ordinal}"
            jobs.append(
                core.ReceiverJob(
                    f"unfinished:{job_id}", receiver, -0.5 + ordinal * 0.001, 1, task_id
                )
            )
    return tuple(jobs)


def _sender_history(tasks: tuple[core.Contribution, ...]) -> tuple[core.SenderHistoryEvent, ...]:
    return tuple(
        core.SenderHistoryEvent(task.task_id, task.sender_rank, -40.0 - index)
        for index, task in enumerate(sorted(tasks, key=lambda value: value.task_id))
        if not task.is_decision
    )


def primary_fixture(kind: core.ScenarioKind = "primary") -> core.Scenario:
    equal_q = kind == "equal_q"
    equal_j = kind == "equal_j"
    tasks = _primary_tasks()
    worlds = tuple(
        core.World(
            f"w{q}{j}",
            q,
            j,
            _receiver_jobs(q_bit=q, j_bit=j, equal_q=equal_q, equal_j=equal_j),
            _sender_history(tasks),
            "UNINFORMATIVE-J" if kind == "shuffled_key" else None,
        )
        for q in (0, 1)
        for j in (0, 1)
    )
    joins = (
        core.Join("ja", "ra", 0, -10, 7.0, 0.5, tuple(f"a{i}" for i in range(6))),
        core.Join("jb", "rb", 1, -10, 7.0, 0.5, tuple(f"b{i}" for i in range(6))),
    )
    return core.Scenario(f"fixture:{kind}", tasks, joins, worlds, kind)


def fanout1_fixture() -> core.Scenario:
    tasks = (
        core.Contribution("a", "ja", "ra", 0, 0, 1, 0.1, 1, True),
        core.Contribution("b", "jb", "rb", 0, 1, 1, 0.1, 1, True),
    )
    joins = (
        core.Join("ja", "ra", 0, -10, 7, 0.5, ("a",)),
        core.Join("jb", "rb", 1, -10, 7, 0.5, ("b",)),
    )
    worlds = []
    for q in (0, 1):
        for j in (0, 1):
            depths = {0: 8, 1: 16} if q == 0 else {0: 16, 1: 8}
            jobs = tuple(
                core.ReceiverJob(f"bg:{q}:{j}:{receiver}:{ordinal}", receiver, -0.5 + ordinal * 0.001, 1)
                for receiver in (0, 1)
                for ordinal in range(depths[receiver])
            )
            worlds.append(core.World(f"f{q}{j}", q, j, jobs, ()))
    return core.Scenario("fanout1", tasks, joins, tuple(worlds), "fanout1")


def no_conflict_fixture() -> core.Scenario:
    tasks = (
        core.Contribution("a", "ja", "ra", 2, 0, 1, 0.1, 1, False),
        core.Contribution("b", "jb", "rb", 3, 1, 1, 0.1, 1, False),
    )
    joins = (
        core.Join("ja", "ra", 0, -10, 7, 0.5, ("a",)),
        core.Join("jb", "rb", 1, -10, 7, 0.5, ("b",)),
    )
    worlds = tuple(
        core.World(
            f"n{q}{j}", q, j,
            (
                core.ReceiverJob("a-job", 0, -10, 1, "a"),
                core.ReceiverJob("b-job", 1, -10, 1, "b"),
            ),
            _sender_history(tasks),
        )
        for q in (0, 1) for j in (0, 1)
    )
    return core.Scenario("no-conflict", tasks, joins, worlds, "no_conflict")


def _clone_report(report: dict[str, object], index: int) -> dict[str, object]:
    cloned = copy.deepcopy(report)
    cloned["scenario_id"] = f"scenario-{index:02d}"
    for arm in ("B0", "Q", "J", "R", "C"):
        arm_report = (
            cloned["ceiling"] if arm == "C" else cloned["arms"][arm]  # type: ignore[index]
        )
        metrics = arm_report["metrics"]  # type: ignore[index]

        def rename(rows: tuple[tuple[str, float], ...]) -> tuple[tuple[str, float], ...]:
            return tuple((f"{key}-{index:02d}", value) for key, value in rows)

        arm_report["metrics"] = replace(  # type: ignore[index]
            metrics,
            expected_miss_by_request=rename(metrics.expected_miss_by_request),
            expected_tardiness_by_request=rename(metrics.expected_tardiness_by_request),
            expected_join_close_by_request=rename(metrics.expected_join_close_by_request),
        )
    return cloned


def _aggregate_from_one(report: dict[str, object]) -> dict[str, object]:
    return core.aggregate_16_pair_reports([_clone_report(report, index) for index in range(16)])


class PhaseMapOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.primary = primary_fixture()
        cls.primary_report = core.optimize_information_lattice(cls.primary)

    def test_primary_information_lattice_is_exactly_1_2_2_4(self):
        self.assertEqual(len(core.enumerate_actions(self.primary)), 4)
        counts = {
            arm: len(core.observation_partitions(self.primary, arm))
            for arm in ("B0", "Q", "J", "R")
        }
        self.assertEqual(counts, {"B0": 1, "Q": 2, "J": 2, "R": 4})

    def test_full_future_c_is_diagnostic_and_no_worse_than_r(self):
        ceiling = self.primary_report["ceiling"]
        self.assertTrue(ceiling["diagnostic_only"])
        self.assertTrue(ceiling["sees_full_future"])
        self.assertEqual(ceiling["observation_class_count"], 4)
        self.assertEqual(ceiling["policy_count"], 256)
        self.assertTrue(
            core.lexicographic_no_worse(
                ceiling["lexicographic_minima"],
                self.primary_report["arms"]["R"]["lexicographic_minima"],
            )
        )

    def test_lexicographic_tolerance_band_is_stagewise(self):
        minima, identities = core.lexicographic_optimal_rows(
            [
                ("exact-first", (1.0, 2.0, 0.0)),
                ("near-first-better-second", (1.0 + 5e-11, 1.0, 9.0)),
            ]
        )
        self.assertEqual(identities, ("near-first-better-second",))
        self.assertEqual(minima[0], 1.0)
        self.assertEqual(minima[1], 1.0)

    def test_q_and_j_cross_world_invariants(self):
        worlds = {(world.q_bit, world.j_bit): world for world in self.primary.worlds}
        for q in (0, 1):
            self.assertEqual(
                core.observation_key(self.primary, worlds[(q, 0)], "Q"),
                core.observation_key(self.primary, worlds[(q, 1)], "Q"),
            )
        for j in (0, 1):
            self.assertEqual(
                core.observation_key(self.primary, worlds[(0, j)], "J"),
                core.observation_key(self.primary, worlds[(1, j)], "J"),
            )

    def test_full_stage_and_native_sibling_accounting(self):
        result = core.simulate(self.primary, 0, core.enumerate_actions(self.primary)[0])
        self.assertEqual(result["accounting"]["native_siblings"], 12)
        self.assertEqual(result["accounting"]["decision_pack_count"], 4)
        self.assertEqual(result["accounting"]["decision_cut_count"], 4)
        self.assertEqual(result["accounting"]["combine_count"], 2)
        foreground = {
            row["task_id"] for row in result["receiver_events"] if row["task_id"] is not None
        }
        self.assertEqual(foreground, {task.task_id for task in self.primary.contributions})

    def test_missing_and_duplicate_sibling_fail_closed(self):
        broken_join = replace(self.primary.joins[0], sibling_task_ids=self.primary.joins[0].sibling_task_ids[:-1])
        with self.assertRaisesRegex(core.PhaseMapError, "phase-carrier|full sibling census"):
            core.enumerate_actions(replace(self.primary, joins=(broken_join, self.primary.joins[1])))
        jobs = self.primary.worlds[0].receiver_jobs
        duplicated = replace(self.primary.worlds[0], receiver_jobs=jobs + (jobs[0],))
        with self.assertRaisesRegex(core.PhaseMapError, "job identity is duplicated"):
            core.enumerate_actions(replace(self.primary, worlds=(duplicated,) + self.primary.worlds[1:]))

    def test_sender_history_and_receiver_identity_leak_fail_closed(self):
        history = list(self.primary.worlds[0].sender_history)
        history[0] = replace(history[0], send_complete_us=history[0].send_complete_us - 1)
        leaked = replace(self.primary.worlds[0], sender_history=tuple(history))
        with self.assertRaisesRegex(core.PhaseMapError, "sender history leaks"):
            core.enumerate_actions(replace(self.primary, worlds=(leaked,) + self.primary.worlds[1:]))
        jobs = list(self.primary.worlds[0].receiver_jobs)
        carrier_index = next(index for index, job in enumerate(jobs) if job.task_id is not None)
        jobs[carrier_index] = replace(jobs[carrier_index], receiver_rank=1 - jobs[carrier_index].receiver_rank)
        wrong = replace(self.primary.worlds[0], receiver_jobs=tuple(jobs))
        with self.assertRaisesRegex(core.PhaseMapError, "output receiver"):
            core.enumerate_actions(replace(self.primary, worlds=(wrong,) + self.primary.worlds[1:]))

    def test_phase_flip_cannot_change_q_observation(self):
        target = self.primary.worlds[1]
        extra = core.ReceiverJob("extra", 0, -0.2, 1)
        drifted = replace(target, receiver_jobs=target.receiver_jobs + (extra,))
        with self.assertRaisesRegex(core.PhaseMapError, "join phase changed.*Q"):
            core.enumerate_actions(replace(self.primary, worlds=(self.primary.worlds[0], drifted) + self.primary.worlds[2:]))

    def test_policy_obeys_each_information_partition(self):
        for arm in ("B0", "Q", "J", "R"):
            report = self.primary_report["arms"][arm]
            policy = dict(report["selected_canonical_policy"])
            for key, indices in core.observation_partitions(self.primary, arm):
                self.assertIn(key, policy)
                self.assertGreaterEqual(len(indices), 1)
        self.assertEqual(self.primary_report["arms"]["R"]["policy_count"], 256)

    def test_worlds_fold_to_native_requests_by_quarters(self):
        metrics = self.primary_report["arms"]["R"]["metrics"]
        self.assertEqual(metrics.request_count, 2)
        self.assertTrue(
            all(value in {0.0, 0.25, 0.5, 0.75, 1.0} for _, value in metrics.expected_miss_by_request)
        )

    def test_tied_r_policy_cannot_be_counted_as_strict_flip(self):
        action = core.enumerate_actions(self.primary)[0]
        worlds = {core.observation_key(self.primary, world, "R") for world in self.primary.worlds}
        policy_a = tuple((key, action) for key in sorted(worlds))
        policy_b = tuple(reversed(policy_a))
        report = {"optimal_policies": (policy_a, policy_b)}
        self.assertFalse(core._dual_conditioned_flip(self.primary, report))

    def test_aggregate_is_32_requests_not_128_world_samples(self):
        aggregate = _aggregate_from_one(self.primary_report)
        self.assertEqual(aggregate["native_request_count"], 32)
        self.assertEqual(aggregate["counterfactual_world_count"], 128)
        self.assertTrue(aggregate["worlds_are_folded_not_samples"])
        self.assertEqual(aggregate["arms"]["R"].request_count, 32)
        self.assertEqual(aggregate["ceiling_C"].request_count, 32)
        with self.assertRaisesRegex(core.PhaseMapError, "exactly 16"):
            core.aggregate_16_pair_reports([self.primary_report] * 128)

    def test_best_single_uses_cvar_before_mean_tardiness(self):
        reports = [_clone_report(self.primary_report, index) for index in range(16)]
        for index, report in enumerate(reports):
            for arm in ("Q", "J"):
                arm_report = report["arms"][arm]
                metrics = arm_report["metrics"]
                request_ids = tuple(key for key, _value in metrics.expected_miss_by_request)
                if arm == "Q":
                    values = (1.0, 1.0) if index < 2 else (0.0, 0.0)
                else:
                    values = (0.2, 0.2)
                arm_report["metrics"] = replace(
                    metrics,
                    expected_miss_count=0.0,
                    miss_rate=0.0,
                    expected_tardiness_sum=sum(values),
                    mean_normalized_tardiness=sum(values) / 2.0,
                    expected_miss_by_request=tuple((key, 0.0) for key in request_ids),
                    expected_tardiness_by_request=tuple(zip(request_ids, values)),
                )
        aggregate = core.aggregate_16_pair_reports(reports)
        self.assertLess(
            aggregate["arms"]["Q"].mean_normalized_tardiness,
            aggregate["arms"]["J"].mean_normalized_tardiness,
        )
        self.assertGreater(
            aggregate["arms"]["Q"].cvar90_normalized_tardiness,
            aggregate["arms"]["J"].cvar90_normalized_tardiness,
        )
        self.assertEqual(aggregate["best_single_arm"], "J")
        self.assertEqual(aggregate["best_single_comparator"], core.BEST_SINGLE_COMPARATOR)

    def test_all_frozen_negative_control_relations(self):
        fixtures = {
            "equal_q": primary_fixture("equal_q"),
            "equal_j": primary_fixture("equal_j"),
            "fanout1": fanout1_fixture(),
            "no_conflict": no_conflict_fixture(),
            "shuffled_key": primary_fixture("shuffled_key"),
        }
        for name, scenario in fixtures.items():
            with self.subTest(name=name):
                report = core.optimize_information_lattice(scenario)
                aggregate = _aggregate_from_one(report)
                self.assertTrue(core.validate_control(name, aggregate)["passed"])

    def test_predecision_jobs_cannot_be_smuggled_in_from_future(self):
        jobs = list(self.primary.worlds[0].receiver_jobs)
        jobs[0] = replace(jobs[0], arrival_us=0.1)
        future = replace(self.primary.worlds[0], receiver_jobs=tuple(jobs))
        with self.assertRaisesRegex(core.PhaseMapError, "arrive before t0"):
            core.enumerate_actions(replace(self.primary, worlds=(future,) + self.primary.worlds[1:]))


if __name__ == "__main__":
    unittest.main()
