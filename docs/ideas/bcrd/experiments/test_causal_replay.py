from __future__ import annotations

import unittest

try:
    from .core import Contribution, CurvePoint, ProtocolError, ReplayConfig, ServiceCatalog, simulate_assignment
    from .policies import (
        AssignmentState,
        GreedyCompletionPolicy,
        LeastLoadPolicy,
        OnlineContributionView,
        simulate_online_policy,
    )
except ImportError:
    from core import Contribution, CurvePoint, ProtocolError, ReplayConfig, ServiceCatalog, simulate_assignment
    from policies import (
        AssignmentState,
        GreedyCompletionPolicy,
        LeastLoadPolicy,
        OnlineContributionView,
        simulate_online_policy,
    )


def catalog() -> ServiceCatalog:
    return ServiceCatalog(
        {("m", 0): [CurvePoint(1, 10, 11), CurvePoint(2, 14, 15), CurvePoint(4, 20, 22)]}
    )


def row(
    request: str,
    *,
    ready: float,
    expert: int = 0,
    rank: int = 1,
    source: int = 0,
    deadline: float = 1000.0,
    legal: tuple[int, ...] = (),
) -> Contribution:
    return Contribution(
        "m",
        "decode",
        request,
        0,
        ready,
        deadline,
        0,
        0,
        rank,
        expert,
        1.0,
        source,
        input_event_id=f"{request}:event",
        token_id=0,
        decode_step=0,
        legal_replica_set=legal,
    )


class _RecordingPolicy:
    name = "recording"
    remote_latency_us = 0.0

    def __init__(self) -> None:
        self.joinable: list[int] = []
        self.predicted_completion: list[float] = []

    def choose(self, item, state, service_catalog):
        self.asserted_causal_view = isinstance(item, OnlineContributionView)
        self.hidden_suffix_absent = not any(
            hasattr(item, name)
            for name in (
                "dispatch_end_us",
                "expert_start_us",
                "expert_end_us",
                "combine_end_us",
            )
        )
        self.engine_absent = not hasattr(state, "engine")
        self.joinable.append(state.joinable_rows(item, 0))
        self.predicted_completion.append(state.predict(item, 0)["completion_us"])
        return 0


class CausalReplayTest(unittest.TestCase):
    def test_singleton_pays_full_hold(self) -> None:
        result = simulate_assignment([row("r", ready=0)], [0], catalog(), ReplayConfig(2, hold_us=100))
        self.assertEqual(result["contribution_completion_us"][0], 110.0)
        self.assertEqual(result["batch_records"][0]["seal_us"], 100.0)

    def test_arrival_after_seal_is_not_batched(self) -> None:
        rows = [row("a", ready=0), row("b", ready=101)]
        result = simulate_assignment(rows, [0, 0], catalog(), ReplayConfig(2, hold_us=100))
        self.assertEqual(result["launches"], 2)
        self.assertEqual([record["indices"] for record in result["batch_records"]], [[0], [1]])
        self.assertEqual(result["contribution_completion_us"][0], 110.0)
        self.assertEqual(result["contribution_completion_us"][1], 211.0)

    def test_same_timestamp_arrivals_precede_zero_hold_seal(self) -> None:
        rows = [row("a", ready=0), row("b", ready=0)]
        result = simulate_assignment(rows, [0, 0], catalog(), ReplayConfig(2, hold_us=0))
        self.assertEqual(result["launches"], 1)
        self.assertEqual(result["batch_records"][0]["indices"], [0, 1])

    def test_max_batch_never_overflows_on_same_timestamp_arrivals(self) -> None:
        rows = [row("a", ready=0), row("b", ready=0), row("c", ready=0)]
        result = simulate_assignment(
            rows,
            [0, 0, 0],
            catalog(),
            ReplayConfig(2, hold_us=100, max_batch_rows=2),
        )
        self.assertEqual(result["launches"], 2)
        self.assertEqual(sorted(len(record["indices"]) for record in result["batch_records"]), [1, 2])
        self.assertTrue(
            all(len(record["indices"]) <= 2 for record in result["batch_records"])
        )

    def test_rows_are_removed_after_launch(self) -> None:
        policy = _RecordingPolicy()
        rows = [row("a", ready=0), row("b", ready=100)]
        assignments, result = simulate_online_policy(
            rows, policy, catalog(), ReplayConfig(2, hold_us=0)
        )
        self.assertEqual(assignments, [0, 0])
        self.assertEqual(policy.joinable, [0, 0])
        self.assertEqual(
            policy.predicted_completion,
            [result["contribution_completion_us"][0], result["contribution_completion_us"][1]],
        )
        self.assertEqual(result["launches"], 2)

    def test_online_policy_never_reads_future_suffix(self) -> None:
        prefix = [row("a", ready=0), row("b", ready=100)]
        full = prefix + [row("future", ready=200)]
        short_policy = _RecordingPolicy()
        long_policy = _RecordingPolicy()
        short_assignments, _ = simulate_online_policy(
            prefix, short_policy, catalog(), ReplayConfig(2, hold_us=20)
        )
        long_assignments, _ = simulate_online_policy(
            full, long_policy, catalog(), ReplayConfig(2, hold_us=20)
        )
        self.assertEqual(short_assignments, long_assignments[: len(prefix)])
        self.assertEqual(
            short_policy.predicted_completion,
            long_policy.predicted_completion[: len(prefix)],
        )
        self.assertTrue(short_policy.asserted_causal_view)
        self.assertTrue(short_policy.hidden_suffix_absent)
        self.assertTrue(short_policy.engine_absent)

    def test_least_load_is_distinct_from_min_predicted_finish(self) -> None:
        item = OnlineContributionView.from_contribution(row("current", ready=0))
        state = AssignmentState(
            2,
            (12.0, 14.0),
            item.contribution_id,
            {
                0: {"completion_us": 22.0, "batch_rows": 1.0},
                1: {"completion_us": 14.0, "batch_rows": 2.0},
            },
            0.0,
        )
        self.assertEqual(LeastLoadPolicy().choose(item, state, catalog()), 0)
        self.assertEqual(GreedyCompletionPolicy().choose(item, state, catalog()), 1)

    def test_remote_latency_is_charged_once(self) -> None:
        result = simulate_assignment(
            [row("r", ready=0, source=0)],
            [1],
            catalog(),
            ReplayConfig(2, remote_latency_us=5),
        )
        self.assertEqual(result["contribution_completion_us"][0], 15.0)
        self.assertEqual(result["remote_assignments"], 1)

    def test_controller_cost_advances_deadline_and_slo(self) -> None:
        item = row("r", ready=0, deadline=12)
        free = simulate_assignment([item], [0], catalog(), ReplayConfig(2))
        taxed = simulate_assignment(
            [item], [0], catalog(), ReplayConfig(2, controller_latency_us=5)
        )
        self.assertEqual(free["on_time"], 1)
        self.assertEqual(taxed["on_time"], 0)
        self.assertEqual(taxed["contribution_completion_us"][0], 15.0)

    def test_hold_is_capped_by_remaining_deadline_slack(self) -> None:
        result = simulate_assignment(
            [row("tight", ready=0, deadline=5)],
            [0],
            catalog(),
            ReplayConfig(2, hold_us=100),
        )
        self.assertEqual(result["batch_records"][0]["seal_us"], 0.0)
        self.assertEqual(result["batch_records"][0]["seal_reason"], "deadline")

    def test_deadline_cap_reserves_seal_and_launch_costs(self) -> None:
        result = simulate_assignment(
            [row("tight", ready=0, deadline=20)],
            [0],
            catalog(),
            ReplayConfig(
                2,
                hold_us=100,
                seal_cost_us=5,
                launch_cost_us=5,
            ),
        )
        self.assertEqual(result["batch_records"][0]["seal_us"], 0.0)
        self.assertEqual(result["batch_records"][0]["seal_reason"], "deadline")
        self.assertEqual(result["contribution_completion_us"][0], 20.0)
        self.assertEqual(result["on_time"], 1)

    def test_tighter_joiner_advances_open_batch_deadline_cap(self) -> None:
        rows = [
            row("loose", ready=0, deadline=100),
            row("tight", ready=10, deadline=25),
        ]
        result = simulate_assignment(
            rows,
            [0, 0],
            catalog(),
            ReplayConfig(2, hold_us=50),
        )
        self.assertEqual(result["launches"], 1)
        self.assertEqual(result["batch_records"][0]["seal_us"], 11.0)
        self.assertEqual(result["batch_records"][0]["seal_reason"], "deadline")
        self.assertEqual(result["contribution_completion_us"][1], 25.0)

    def test_illegal_per_expert_target_fails_for_replay(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "illegal replica"):
            simulate_assignment(
                [row("r", ready=0, legal=(0,))], [1], catalog(), ReplayConfig(2)
            )

    def test_per_queue_holds_are_independent(self) -> None:
        rows = [row("a", ready=0, expert=0), row("b", ready=0, expert=1)]
        result = simulate_assignment(
            rows,
            [0, 0],
            catalog(),
            ReplayConfig(2, hold_by_queue={(0, 0): 0.0, (0, 1): 100.0}),
        )
        seals = {record["expert_id"]: record["seal_us"] for record in result["batch_records"]}
        self.assertEqual(seals, {0: 0.0, 1: 100.0})

    def test_edf_tie_break_sorts_request_ids(self) -> None:
        rows = [
            row("z", ready=0, expert=0),
            row("a", ready=0, expert=0),
            row("b", ready=0, expert=1),
            row("c", ready=0, expert=1),
        ]
        result = simulate_assignment(rows, [0, 0, 0, 0], catalog(), ReplayConfig(2))
        self.assertEqual([record["expert_id"] for record in result["batch_records"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
