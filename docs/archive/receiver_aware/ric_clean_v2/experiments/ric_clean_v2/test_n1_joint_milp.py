from __future__ import annotations

from dataclasses import replace
import importlib.util
import unittest

try:
    from .n1_joint_milp import (
        ContributionIdentity,
        N1CoreError,
        PairSpec,
        PriorEvent,
        ServiceTuple,
        Task,
        WorldSpec,
        empirical_cvar,
        observation_fingerprint,
        replay_prior_history,
        r0_flip,
        solve_independent_enumeration,
        solve_joint_milp,
        validate_matched_pair,
    )
except ImportError:
    from n1_joint_milp import (  # type: ignore
        ContributionIdentity,
        N1CoreError,
        PairSpec,
        PriorEvent,
        ServiceTuple,
        Task,
        WorldSpec,
        empirical_cvar,
        observation_fingerprint,
        replay_prior_history,
        r0_flip,
        solve_independent_enumeration,
        solve_joint_milp,
        validate_matched_pair,
    )


HAS_SCIPY = importlib.util.find_spec("scipy") is not None


def _task(task_id: str, join: str, slot: int, sender: int) -> Task:
    return Task(
        task_id=task_id,
        identity=ContributionIdentity(
            request_id=f"req-{join}",
            layer_id=0,
            token_id=0 if join == "j1" else 1,
            token_block_id=0 if join == "j1" else 1,
            topk_slot=slot,
            expert_id=slot,
            sender_rank=sender,
            receiver_rank=0,
            epoch=0,
        ),
        join_key=join,
        top_k=2,
        release_us=0.0,
        payload_bytes=16,
        service=ServiceTuple(0.5, 0.5, 0.5, 0.5),
        join_arrival_us=0.0,
        closure_budget_us=100.0,
    )


def _prefix(task_id: str, offset: float) -> tuple[PriorEvent, ...]:
    return (
        PriorEvent(task_id, "pack", offset, offset + 0.5),
        PriorEvent(task_id, "shared_cut", offset + 0.5, offset + 1.0),
        PriorEvent(task_id, "unpack", offset + 1.0, offset + 1.5),
    )


def fixture_pair() -> PairSpec:
    a = _task("a", "j1", 0, 0)
    b = _task("b", "j2", 0, 0)
    p1 = _task("p1", "j1", 1, 1)
    p2 = _task("p2", "j2", 1, 1)
    common = _prefix("p1", 0.0) + _prefix("p2", 1.5)
    # Canonical ordering is by completion timestamp, then stage, then task ID.
    common = tuple(sorted(common, key=lambda event: (event.end_us, ("pack", "shared_cut", "unpack", "receiver_apply").index(event.stage), event.task_id)))
    w0_history = tuple(sorted(common + (PriorEvent("p1", "receiver_apply", 4.0, 4.5),), key=lambda event: (event.end_us, ("pack", "shared_cut", "unpack", "receiver_apply").index(event.stage), event.task_id)))
    w1_history = tuple(sorted(common + (PriorEvent("p2", "receiver_apply", 4.0, 4.5),), key=lambda event: (event.end_us, ("pack", "shared_cut", "unpack", "receiver_apply").index(event.stage), event.task_id)))
    return PairSpec(
        pair_id="fixture-pair",
        tasks=(a, b, p1, p2),
        worlds=(
            WorldSpec("w0", w0_history, {"j1": 1, "j2": 3}, {"p2": 20.0}),
            WorldSpec("w1", w1_history, {"j1": 3, "j2": 1}, {"p1": 20.0}),
        ),
        current_ready_task_ids=("a", "b"),
        flip_candidate_task_ids=("a", "b"),
        target_sender_rank=0,
        decision_time_us=5.0,
    )


class N1JointMilpTests(unittest.TestCase):
    def test_prior_histories_are_reachable_and_views_are_closed(self) -> None:
        pair = fixture_pair()
        states = validate_matched_pair(pair)
        self.assertNotEqual(states[0].join_masks, states[1].join_masks)
        self.assertEqual(
            observation_fingerprint(pair, pair.worlds[0], "S"),
            observation_fingerprint(pair, pair.worlds[1], "S"),
        )
        self.assertEqual(
            observation_fingerprint(pair, pair.worlds[0], "B"),
            observation_fingerprint(pair, pair.worlds[1], "B"),
        )
        self.assertNotEqual(
            observation_fingerprint(pair, pair.worlds[0], "R0"),
            observation_fingerprint(pair, pair.worlds[1], "R0"),
        )

    def test_identity_relabel_or_unreachable_history_is_blocked(self) -> None:
        pair = fixture_pair()
        bad_event = replace(pair.worlds[0].prior_history[-1], task_id="not-a-task")
        bad_world = replace(pair.worlds[0], prior_history=pair.worlds[0].prior_history[:-1] + (bad_event,))
        with self.assertRaisesRegex(N1CoreError, "INVALID_HISTORY_EVENT"):
            replay_prior_history(replace(pair, worlds=(bad_world, pair.worlds[1])), bad_world)

    def test_ru_cvar_is_exact_for_fractional_tail_mass(self) -> None:
        self.assertEqual(empirical_cvar([1.0, 2.0, 3.0, 10.0]), 10.0)
        self.assertAlmostEqual(empirical_cvar([1.0] * 99 + [10.0]), 10.0)

    def test_enumeration_enforces_joint_nonanticipativity_and_r0_flip(self) -> None:
        pair = fixture_pair()
        blind = solve_independent_enumeration(pair, "B")
        receiver = solve_independent_enumeration(pair, "R0")
        self.assertEqual(blind.world_actions[0], blind.world_actions[1])
        self.assertEqual(receiver.first_action_sets, (frozenset({"a"}), frozenset({"b"})))
        self.assertTrue(r0_flip(receiver))

    @unittest.skipUnless(HAS_SCIPY, "scipy.optimize.milp unavailable")
    def test_joint_milp_matches_independent_enumeration(self) -> None:
        pair = fixture_pair()
        for level in ("S", "B", "R0"):
            milp = solve_joint_milp(pair, level)
            reference = solve_independent_enumeration(pair, level)
            self.assertEqual(milp.objective, reference.objective)
            self.assertTrue(all(stage.status == "OPTIMAL" for stage in milp.solver_stages))
            self.assertTrue(all(stage.relative_gap <= 1e-6 for stage in milp.solver_stages))
        self.assertTrue(r0_flip(solve_joint_milp(pair, "R0")))


if __name__ == "__main__":
    unittest.main()
