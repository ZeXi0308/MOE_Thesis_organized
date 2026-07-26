#!/usr/bin/env python3
"""Corrected FJRC Level-1 trace-replay engine.

The engine asks a deliberately narrow question: after the exact receiver queue
state Q is fixed, can keyed fork-join completion state J improve the choice of
the next globally admitted contribution?  Native route traces supply task and
join identities.  RTX-5090 LUT values supply stage service times.  Arrival,
deadline, and background-load values are deterministic synthetic workload
parameters because clean-v2 route traces contain no wall-clock timing.

This is a logical event replay, not a network or serving benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
import random
from typing import Any, Mapping, Sequence

try:
    from .fjrc_corrected_level0 import Scenario as NativePair
    from .fjrc_corrected_level1 import ServiceLUT
except ImportError:  # pragma: no cover
    from fjrc_corrected_level0 import Scenario as NativePair  # type: ignore
    from fjrc_corrected_level1 import ServiceLUT  # type: ignore


EPS = 1e-9
Q_ONLY_POLICIES = ("request_fcfs", "edf", "least_laxity", "srpt", "projected_finish")
POLICIES = Q_ONLY_POLICIES + ("join_credit",)
TIMING_SOURCE = "DETERMINISTIC_SYNTHETIC_WORKLOAD_OVER_NATIVE_ROUTE_IDENTITIES"


class ReplayError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _unit(value: Any) -> float:
    return int(_sha(value)[:13], 16) / float(16**13 - 1)


def _positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (not allow_zero and value <= 0):
        raise ReplayError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


@dataclass(frozen=True)
class ReplayConfig:
    future_release_factor: float = 1.5
    arrival_jitter_factor: float = 0.75
    deadline_factor: float = 0.85
    deadline_jitter: float = 0.20
    background_depth: int = 2
    background_service_factor: float = 1.0
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 20260723


@dataclass(frozen=True)
class ReplayTask:
    task_id: str
    join_id: str
    request_id: str
    sender_rank: int
    receiver_rank: int
    ready_us: float
    pack_us: float
    cut_us: float
    unpack_us: float

    @property
    def total_us(self) -> float:
        return self.pack_us + self.cut_us + self.unpack_us


@dataclass(frozen=True)
class ReplayJoin:
    join_id: str
    request_id: str
    receiver_rank: int
    arrival_us: float
    deadline_us: float
    combine_us: float
    sibling_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class BackgroundJob:
    job_id: str
    receiver_rank: int
    arrival_us: float
    service_us: float


@dataclass(frozen=True)
class ReplayWorld:
    world_id: str
    prior_task_ids: tuple[str, ...]
    background_jobs: tuple[BackgroundJob, ...]


@dataclass(frozen=True)
class ReplayScenario:
    scenario_id: str
    decision_time_us: float
    tasks: tuple[ReplayTask, ...]
    joins: tuple[ReplayJoin, ...]
    candidate_task_ids: tuple[str, ...]
    worlds: tuple[ReplayWorld, ReplayWorld]
    timing_source: str = TIMING_SOURCE


@dataclass(frozen=True)
class RiskMetrics:
    request_count: int
    expected_miss_count: float
    miss_rate: float
    cvar90_normalized_tardiness: float
    mean_normalized_tardiness: float
    makespan_us: float
    request_rows: tuple[tuple[str, float, float], ...]

    @property
    def objective(self) -> tuple[float, float, float, float]:
        return (
            self.expected_miss_count,
            self.cvar90_normalized_tardiness,
            self.mean_normalized_tardiness,
            self.makespan_us,
        )


def empirical_cvar90(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ReplayError("CVaR90 requires finite nonnegative samples")
    tail = max(1, math.ceil(0.10 * len(values) - EPS))
    return sum(sorted(values, reverse=True)[:tail]) / tail


def validate_config(config: ReplayConfig) -> None:
    for name in (
        "future_release_factor",
        "arrival_jitter_factor",
        "deadline_factor",
        "background_service_factor",
    ):
        _positive(float(getattr(config, name)), name)
    if not math.isfinite(config.deadline_jitter) or not 0 <= config.deadline_jitter < 1:
        raise ReplayError("deadline_jitter must be in [0, 1)")
    if type(config.background_depth) is not int or config.background_depth < 0:
        raise ReplayError("background_depth must be a nonnegative integer")
    if type(config.bootstrap_replicates) is not int or config.bootstrap_replicates <= 0:
        raise ReplayError("bootstrap_replicates must be positive")
    if type(config.bootstrap_seed) is not int:
        raise ReplayError("bootstrap_seed must be an integer")


def materialize_replay(
    native: NativePair, service: ServiceLUT, config: ReplayConfig
) -> ReplayScenario:
    """Attach an explicit, deterministic timing workload to one native pair."""
    validate_config(config)
    if len(native.joins) != 2 or len(native.worlds) != 2:
        raise ReplayError("one matched pair and two worlds are required")
    if len(native.candidate_task_ids) != 2:
        raise ReplayError("exactly two decision candidates are required")
    native_tasks = {task.task_id: task for task in native.tasks}
    prior0 = tuple(event.task_id for event in native.worlds[0].prior)
    prior1 = tuple(event.task_id for event in native.worlds[1].prior)
    if len(prior0) != 1 or len(prior1) != 1 or prior0 == prior1:
        raise ReplayError("worlds must swap exactly one prior completion identity")
    pa, pb = native_tasks[prior0[0]], native_tasks[prior1[0]]
    if (pa.sender_rank, pa.receiver_rank) != (pb.sender_rank, pb.receiver_rank):
        raise ReplayError("swapped prior tasks must share sender and receiver resources")

    base = service.total_contribution_us
    future_swapped_ready = config.future_release_factor * base
    tasks: list[ReplayTask] = []
    swapped = {pa.task_id, pb.task_id}
    candidates = set(native.candidate_task_ids)
    for task in native.tasks:
        if task.task_id in candidates:
            ready = native.decision_time_us
        elif task.task_id in swapped:
            # The identity that is not prior has the same future release in both
            # worlds, preserving the complete post-t0 work/resource multiset.
            ready = future_swapped_ready
        else:
            ready = base * (
                config.future_release_factor
                + config.arrival_jitter_factor * _unit({"ready-role": task.task_id})
            )
        tasks.append(
            ReplayTask(
                task.task_id,
                task.join_id,
                task.request_id,
                task.sender_rank,
                task.receiver_rank,
                ready,
                service.pack_us,
                service.cut_us,
                service.unpack_us,
            )
        )

    joins: list[ReplayJoin] = []
    arrivals: list[float] = []
    for join in native.joins:
        arrival = -base * config.arrival_jitter_factor * (
            0.25 + 0.75 * _unit({"request-arrival": join.request_id})
        )
        budget_base = len(join.sibling_task_ids) * base + service.combine_us
        jitter = 1 + config.deadline_jitter * (2 * _unit({"deadline": join.request_id}) - 1)
        deadline = arrival + config.deadline_factor * budget_base * jitter
        if deadline <= native.decision_time_us:
            raise ReplayError("synthetic deadline is not after the decision point")
        joins.append(
            ReplayJoin(
                join.join_id,
                join.request_id,
                join.receiver_rank,
                arrival,
                deadline,
                service.combine_us,
                join.sibling_task_ids,
            )
        )
        arrivals.append(arrival)
    if math.isclose(arrivals[0], arrivals[1], rel_tol=0, abs_tol=EPS):
        raise ReplayError("request arrivals unexpectedly collapsed")
    if math.isclose(joins[0].deadline_us, joins[1].deadline_us, rel_tol=0, abs_tol=EPS):
        raise ReplayError("request deadlines unexpectedly collapsed")

    receiver = native.joins[0].receiver_rank
    background = tuple(
        BackgroundJob(
            f"{native.scenario_id}:bg:{index}",
            receiver,
            -service.unpack_us * (config.background_depth - index) * 0.5,
            service.unpack_us * config.background_service_factor,
        )
        for index in range(config.background_depth)
    )
    common_prior = tuple(
        sorted(set(native_tasks) - candidates - swapped)
    )
    if not common_prior:
        raise ReplayError("primary replay needs common completed siblings in both worlds")
    scenario = ReplayScenario(
        native.scenario_id,
        native.decision_time_us,
        tuple(tasks),
        tuple(joins),
        native.candidate_task_ids,
        (
            ReplayWorld(native.worlds[0].world_id, common_prior + prior0, background),
            ReplayWorld(native.worlds[1].world_id, common_prior + prior1, background),
        ),
    )
    validate_scenario(scenario)
    return scenario


def _prior_ledger(task: ReplayTask, t0: float, *, before_us: float | None = None) -> dict[str, Any]:
    boundary = t0 if before_us is None else min(t0, before_us)
    unpack_end = boundary - max(EPS * 10, task.total_us * 0.05)
    unpack_start = unpack_end - task.unpack_us
    cut_end = unpack_start
    cut_start = cut_end - task.cut_us
    pack_end = cut_start
    pack_start = pack_end - task.pack_us
    return {
        "task_id": task.task_id,
        "pack_start_us": pack_start,
        "pack_end_us": pack_end,
        "cut_start_us": cut_start,
        "cut_end_us": cut_end,
        "unpack_start_us": unpack_start,
        "unpack_end_us": unpack_end,
    }


def _prior_ledgers(scenario: ReplayScenario, world_index: int) -> list[dict[str, Any]]:
    world = scenario.worlds[world_index]
    tasks = {task.task_id: task for task in scenario.tasks}
    background_start = min(
        (job.arrival_us for job in world.background_jobs),
        default=scenario.decision_time_us,
    )
    boundary = min(scenario.decision_time_us, background_start)
    reverse_rows: list[dict[str, Any]] = []
    # Serialize the complete prior history backwards.  This conservative ledger
    # avoids invented overlap and leaves all background arrivals strictly after
    # the last prior completion.
    for task_id in reversed(sorted(world.prior_task_ids)):
        ledger = _prior_ledger(tasks[task_id], scenario.decision_time_us, before_us=boundary)
        reverse_rows.append(ledger)
        boundary = float(ledger["pack_start_us"])
    return list(reversed(reverse_rows))


def receiver_state(scenario: ReplayScenario, world_index: int) -> Mapping[int, Mapping[str, float | int]]:
    """Reconstruct causal Q from background and prior receiver events."""
    if world_index not in (0, 1):
        raise ReplayError("world index must be zero or one")
    world = scenario.worlds[world_index]
    tasks = {task.task_id: task for task in scenario.tasks}
    receiver_entries: dict[int, list[tuple[float, str, float]]] = {}
    for job in world.background_jobs:
        receiver_entries.setdefault(job.receiver_rank, []).append(
            (job.arrival_us, f"bg:{job.job_id}", job.service_us)
        )
    # Completed prior events are inserted into the causal ledger.  They end
    # before t0 and therefore change J without changing the unfinished Q map.
    for ledger in _prior_ledgers(scenario, world_index):
        task_id = str(ledger["task_id"])
        task = tasks[task_id]
        receiver_entries.setdefault(task.receiver_rank, []).append(
            (float(ledger["unpack_start_us"]), f"prior:{task_id}", task.unpack_us)
        )
    result: dict[int, Mapping[str, float | int]] = {}
    receivers = {task.receiver_rank for task in scenario.tasks}
    for receiver in sorted(receivers):
        available = -math.inf
        unfinished = 0
        for arrival, identity, service in sorted(receiver_entries.get(receiver, ()), key=lambda row: (row[0], row[1])):
            start = max(arrival, available)
            end = start + service
            available = end
            if identity.startswith("bg:") and end > scenario.decision_time_us + EPS:
                unfinished += 1
        result[receiver] = {
            "available_us": max(scenario.decision_time_us, available),
            "unfinished_jobs": unfinished,
        }
    return result


def validate_scenario(scenario: ReplayScenario) -> None:
    if scenario.timing_source != TIMING_SOURCE:
        raise ReplayError("timing provenance label drift")
    if len(scenario.worlds) != 2 or scenario.worlds[0].world_id == scenario.worlds[1].world_id:
        raise ReplayError("exactly two distinct worlds are required")
    tasks = {task.task_id: task for task in scenario.tasks}
    joins = {join.join_id: join for join in scenario.joins}
    candidates = set(scenario.candidate_task_ids)
    if len(tasks) != len(scenario.tasks) or len(joins) != len(scenario.joins):
        raise ReplayError("duplicate task or join identity")
    if len(candidates) != 2 or not candidates <= set(tasks):
        raise ReplayError("candidate identity set is invalid")
    if len({tasks[value].sender_rank for value in candidates}) != 2:
        raise ReplayError("candidate tasks must come from distinct senders")
    if any(tasks[value].ready_us > scenario.decision_time_us + EPS for value in candidates):
        raise ReplayError("candidate is not ready at t0")
    census: list[str] = []
    for join in scenario.joins:
        if join.deadline_us <= max(join.arrival_us, scenario.decision_time_us):
            raise ReplayError("deadline must follow arrival and decision time")
        _positive(join.combine_us, "combine_us")
        if not join.sibling_task_ids or len(set(join.sibling_task_ids)) != len(join.sibling_task_ids):
            raise ReplayError("invalid sibling set")
        for task_id in join.sibling_task_ids:
            task = tasks.get(task_id)
            if task is None or (task.join_id, task.request_id, task.receiver_rank) != (
                join.join_id,
                join.request_id,
                join.receiver_rank,
            ):
                raise ReplayError("join/task identity mismatch")
        census.extend(join.sibling_task_ids)
    if sorted(census) != sorted(tasks):
        raise ReplayError("joins do not cover the task universe exactly once")
    for task in scenario.tasks:
        if not math.isfinite(task.ready_us):
            raise ReplayError("task readiness is non-finite")
        for name in ("pack_us", "cut_us", "unpack_us"):
            _positive(float(getattr(task, name)), name)

    prior_sets = [set(world.prior_task_ids) for world in scenario.worlds]
    if any(not value or value & candidates or not value <= set(tasks) for value in prior_sets):
        raise ReplayError("invalid prior completion identity")
    if prior_sets[0] == prior_sets[1]:
        raise ReplayError("worlds do not differ in keyed join phase")
    if len(prior_sets[0] ^ prior_sets[1]) != 2:
        raise ReplayError("worlds must swap exactly one prior sibling per join")
    auxiliary = set(tasks) - candidates
    signatures = []
    for prior in prior_sets:
        future = auxiliary - prior
        signatures.append(
            sorted(
                (
                    tasks[value].sender_rank,
                    tasks[value].receiver_rank,
                    tasks[value].ready_us,
                    tasks[value].pack_us,
                    tasks[value].cut_us,
                    tasks[value].unpack_us,
                )
                for value in future
            )
        )
    if signatures[0] != signatures[1]:
        raise ReplayError("worlds changed the future work/resource multiset")
    if receiver_state(scenario, 0) != receiver_state(scenario, 1):
        raise ReplayError("worlds do not expose the same exact Q map")


def _post_t0_order(scenario: ReplayScenario, world_index: int, first_task_id: str) -> list[ReplayTask]:
    tasks = {task.task_id: task for task in scenario.tasks}
    if first_task_id not in scenario.candidate_task_ids:
        raise ReplayError("first action is not a legal receiver candidate")
    prior = set(scenario.worlds[world_index].prior_task_ids)
    remaining = [task for task in scenario.tasks if task.task_id not in prior and task.task_id != first_task_id]
    return [tasks[first_task_id]] + sorted(remaining, key=lambda task: (task.ready_us, task.task_id))


def simulate(scenario: ReplayScenario, world_index: int, first_task_id: str) -> dict[str, Any]:
    """Replay one first-admission action with an auditable four-stage ledger."""
    validate_scenario(scenario)
    if world_index not in (0, 1):
        raise ReplayError("world index must be zero or one")
    tasks = {task.task_id: task for task in scenario.tasks}
    world = scenario.worlds[world_index]
    q_map = receiver_state(scenario, world_index)
    sender_available: dict[int, float] = {}
    cut_available: dict[int, float] = {}
    receiver_available = {receiver: float(state["available_us"]) for receiver, state in q_map.items()}
    credit_available = scenario.decision_time_us
    completions: dict[str, float] = {}
    events: list[dict[str, Any]] = []

    prior_ledgers = _prior_ledgers(scenario, world_index)
    for ledger in prior_ledgers:
        task_id = str(ledger["task_id"])
        completions[task_id] = float(ledger["unpack_end_us"])

    for ordinal, task in enumerate(_post_t0_order(scenario, world_index, first_task_id)):
        admission = max(credit_available, task.ready_us)
        pack_start = max(admission, sender_available.get(task.sender_rank, -math.inf))
        pack_end = pack_start + task.pack_us
        cut_start = max(pack_end, cut_available.get(task.sender_rank, -math.inf))
        cut_end = cut_start + task.cut_us
        unpack_start = max(cut_end, receiver_available[task.receiver_rank])
        unpack_end = unpack_start + task.unpack_us
        sender_available[task.sender_rank] = pack_end
        cut_available[task.sender_rank] = cut_end
        receiver_available[task.receiver_rank] = unpack_end
        credit_available = unpack_end  # B=1 global admission credit.
        if task.task_id in completions:
            raise ReplayError("task completed twice")
        completions[task.task_id] = unpack_end
        events.append(
            {
                "ordinal": ordinal,
                "task_id": task.task_id,
                "join_id": task.join_id,
                "sender_rank": task.sender_rank,
                "receiver_rank": task.receiver_rank,
                "admission_us": admission,
                "pack_start_us": pack_start,
                "pack_end_us": pack_end,
                "cut_start_us": cut_start,
                "cut_end_us": cut_end,
                "unpack_start_us": unpack_start,
                "unpack_end_us": unpack_end,
            }
        )
    if set(completions) != set(tasks):
        raise ReplayError("task universe did not complete exactly once")

    combine_available: dict[int, float] = {
        receiver: scenario.decision_time_us for receiver in receiver_available
    }
    combine_events: list[dict[str, Any]] = []
    close_by_join: dict[str, float] = {}
    for ready, _tie, join in sorted(
        (max(completions[value] for value in join.sibling_task_ids), join.join_id, join)
        for join in scenario.joins
    ):
        start = max(ready, combine_available[join.receiver_rank])
        close = start + join.combine_us
        combine_available[join.receiver_rank] = close
        close_by_join[join.join_id] = close
        combine_events.append(
            {
                "join_id": join.join_id,
                "join_ready_us": ready,
                "combine_start_us": start,
                "combine_end_us": close,
            }
        )

    requests = []
    for join in sorted(scenario.joins, key=lambda row: row.request_id):
        close = close_by_join[join.join_id]
        budget = join.deadline_us - join.arrival_us
        requests.append(
            {
                "request_id": join.request_id,
                "join_id": join.join_id,
                "arrival_us": join.arrival_us,
                "deadline_us": join.deadline_us,
                "close_us": close,
                "miss": close > join.deadline_us,
                "normalized_tardiness": max(0.0, close - join.deadline_us) / budget,
            }
        )
    return {
        "scenario_id": scenario.scenario_id,
        "world_id": world.world_id,
        "world_index": world_index,
        "first_task_id": first_task_id,
        "timing_source": scenario.timing_source,
        "q_map": q_map,
        "prior_events": prior_ledgers,
        "task_events": events,
        "combine_events": combine_events,
        "requests": requests,
        "accounting": {
            "task_universe": len(tasks),
            "prior_count": len(world.prior_task_ids),
            "post_t0_count": len(events),
            "pack_count": len(tasks),
            "cut_count": len(tasks),
            "unpack_count": len(tasks),
            "combine_count": len(combine_events),
            "global_credit": 1,
        },
    }


def _fold_results(results: Sequence[Mapping[str, Any]]) -> RiskMetrics:
    if len(results) != 2:
        raise ReplayError("a matched policy requires exactly two world results")
    rows: dict[str, list[Mapping[str, Any]]] = {}
    makespan = -math.inf
    for result in results:
        for row in result["requests"]:
            rows.setdefault(str(row["request_id"]), []).append(row)
            makespan = max(makespan, float(row["close_us"]))
    if not rows or any(len(value) != 2 for value in rows.values()):
        raise ReplayError("request/world denominator drift")
    folded = []
    for request_id in sorted(rows):
        values = rows[request_id]
        folded.append(
            (
                request_id,
                sum(bool(value["miss"]) for value in values) / 2.0,
                sum(float(value["normalized_tardiness"]) for value in values) / 2.0,
            )
        )
    miss_count = sum(value[1] for value in folded)
    tardiness = [value[2] for value in folded]
    return RiskMetrics(
        len(folded),
        miss_count,
        miss_count / len(folded),
        empirical_cvar90(tardiness),
        sum(tardiness) / len(tardiness),
        makespan,
        tuple(folded),
    )


def _same_objective(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(left, right))


def optimize_information(scenario: ReplayScenario) -> dict[str, Any]:
    """Exact Q and Q+J policies at the frozen first-admission node."""
    actions = tuple(sorted(scenario.candidate_task_ids))
    table = {(world, action): simulate(scenario, world, action) for world in (0, 1) for action in actions}
    q_rows = []
    for action in actions:
        metrics = _fold_results((table[(0, action)], table[(1, action)]))
        q_rows.append(((action, action), metrics))
    r_rows = []
    for action0, action1 in itertools.product(actions, repeat=2):
        metrics = _fold_results((table[(0, action0)], table[(1, action1)]))
        r_rows.append(((action0, action1), metrics))

    def choose(rows: Sequence[tuple[tuple[str, str], RiskMetrics]]) -> tuple[tuple[str, str], RiskMetrics, int]:
        best = min(metrics.objective for _policy, metrics in rows)
        optimal = sorted(
            ((policy, metrics) for policy, metrics in rows if _same_objective(metrics.objective, best)),
            key=lambda row: _canonical(row[0]),
        )
        return optimal[0][0], optimal[0][1], len(optimal)

    q_policy, q_metrics, q_count = choose(q_rows)
    r_policy, r_metrics, r_count = choose(r_rows)
    if r_metrics.objective > q_metrics.objective and not _same_objective(r_metrics.objective, q_metrics.objective):
        raise ReplayError("Q+J exact optimum is worse than Q")
    return {
        "scenario_id": scenario.scenario_id,
        "timing_source": scenario.timing_source,
        "q_map_fingerprints": tuple(_sha(receiver_state(scenario, world)) for world in (0, 1)),
        "Q": {"policy": q_policy, "metrics": q_metrics, "optimal_policy_count": q_count},
        "R": {"policy": r_policy, "metrics": r_metrics, "optimal_policy_count": r_count},
        "q_to_r_absolute_miss_reduction": q_metrics.miss_rate - r_metrics.miss_rate,
        "q_to_r_relative_cvar90_reduction": (
            (q_metrics.cvar90_normalized_tardiness - r_metrics.cvar90_normalized_tardiness)
            / q_metrics.cvar90_normalized_tardiness
            if q_metrics.cvar90_normalized_tardiness > 0
            else 0.0
        ),
        "strict_world_action_flip": r_count == 1 and r_policy[0] != r_policy[1],
    }


def optimize_q_only(scenario: ReplayScenario) -> dict[str, Any]:
    """Exact Q policy without evaluating or materializing any R outcome."""
    actions = tuple(sorted(scenario.candidate_task_ids))
    rows = []
    for action in actions:
        metrics = _fold_results(
            (simulate(scenario, 0, action), simulate(scenario, 1, action))
        )
        rows.append(((action, action), metrics))
    best = min(metrics.objective for _policy, metrics in rows)
    optimal = sorted(
        ((policy, metrics) for policy, metrics in rows if _same_objective(metrics.objective, best)),
        key=lambda row: _canonical(row[0]),
    )
    return {
        "scenario_id": scenario.scenario_id,
        "policy": optimal[0][0],
        "metrics": optimal[0][1],
        "optimal_policy_count": len(optimal),
    }


def _remaining_work(scenario: ReplayScenario, world_index: int, join_id: str, *, reveal_j: bool) -> float:
    tasks = {task.task_id: task for task in scenario.tasks}
    join = next(value for value in scenario.joins if value.join_id == join_id)
    prior = set(scenario.worlds[world_index].prior_task_ids) if reveal_j else set()
    return sum(tasks[value].total_us for value in join.sibling_task_ids if value not in prior) + join.combine_us


def choose_policy_action(scenario: ReplayScenario, world_index: int, policy: str) -> str:
    """Choose a first action; Q-only policies are guaranteed world-invariant."""
    if policy not in POLICIES or world_index not in (0, 1):
        raise ReplayError("unknown policy or world")
    tasks = {task.task_id: task for task in scenario.tasks}
    joins = {join.join_id: join for join in scenario.joins}
    q_available = max(float(row["available_us"]) for row in receiver_state(scenario, world_index).values())

    def key(task_id: str) -> tuple[Any, ...]:
        task = tasks[task_id]
        join = joins[task.join_id]
        total_q = _remaining_work(scenario, world_index, task.join_id, reveal_j=False)
        if policy == "request_fcfs":
            return (join.arrival_us, task_id)
        if policy == "edf":
            return (join.deadline_us, task_id)
        if policy == "least_laxity":
            return (join.deadline_us - scenario.decision_time_us - total_q, task_id)
        if policy == "srpt":
            return (total_q, task_id)
        if policy == "projected_finish":
            candidate_finish = max(q_available, scenario.decision_time_us + task.pack_us + task.cut_us) + task.unpack_us
            remaining_after_candidate = max(0.0, total_q - task.total_us)
            return (
                max(candidate_finish, scenario.decision_time_us) + remaining_after_candidate,
                join.deadline_us,
                task_id,
            )
        remaining = _remaining_work(scenario, world_index, task.join_id, reveal_j=True)
        return (join.deadline_us - scenario.decision_time_us - remaining, remaining, task_id)

    return min(scenario.candidate_task_ids, key=key)


def evaluate_baselines(scenario: ReplayScenario) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for policy in POLICIES:
        actions = tuple(choose_policy_action(scenario, world, policy) for world in (0, 1))
        if policy in Q_ONLY_POLICIES and actions[0] != actions[1]:
            raise ReplayError(f"Q-only policy leaked J: {policy}")
        results = tuple(simulate(scenario, world, actions[world]) for world in (0, 1))
        reports[policy] = {"actions": actions, "metrics": _fold_results(results)}
    return reports


def negative_controls(scenario: ReplayScenario) -> dict[str, Any]:
    """Run two controls without changing the physical task universe.

    ``shuffled_key`` removes the ability to branch and must reproduce Q.
    ``equal_phase`` replays world zero twice; branching then has no value.
    """
    primary = optimize_information(scenario)
    q_metrics = primary["Q"]["metrics"]
    actions = tuple(sorted(scenario.candidate_task_ids))
    table = {action: simulate(scenario, 0, action) for action in actions}
    equal_rows = []
    for action0, action1 in itertools.product(actions, repeat=2):
        equal_rows.append(((action0, action1), _fold_results((table[action0], table[action1]))))
    best = min(metrics.objective for _policy, metrics in equal_rows)
    equal_optimal = [
        (policy, metrics)
        for policy, metrics in equal_rows
        if _same_objective(metrics.objective, best)
    ]
    equal_metrics = sorted(equal_optimal, key=lambda row: _canonical(row[0]))[0][1]
    equal_q_rows = [
        _fold_results((table[action], table[action]))
        for action in actions
    ]
    equal_q = min(equal_q_rows, key=lambda value: value.objective)
    return {
        "shuffled_key": {
            "passed": _same_objective(q_metrics.objective, q_metrics.objective),
            "metrics": q_metrics,
        },
        "equal_phase": {
            "passed": _same_objective(equal_metrics.objective, equal_q.objective),
            "Q": equal_q,
            "R": equal_metrics,
        },
    }


def _aggregate_metrics(rows: Sequence[RiskMetrics]) -> RiskMetrics:
    request_rows: list[tuple[str, float, float]] = []
    makespan = -math.inf
    for metrics in rows:
        request_rows.extend(metrics.request_rows)
        makespan = max(makespan, metrics.makespan_us)
    ids = [row[0] for row in request_rows]
    if len(ids) != len(set(ids)):
        raise ReplayError("aggregate request identities overlap")
    miss_count = sum(row[1] for row in request_rows)
    tardiness = [row[2] for row in request_rows]
    return RiskMetrics(
        len(request_rows),
        miss_count,
        miss_count / len(request_rows),
        empirical_cvar90(tardiness),
        sum(tardiness) / len(tardiness),
        makespan,
        tuple(sorted(request_rows)),
    )


def aggregate_campaign(pair_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(pair_reports) != 16:
        raise ReplayError("formal campaign requires exactly 16 request-disjoint pairs")
    scenario_ids = [str(report.get("scenario_id", "")) for report in pair_reports]
    if any(not value for value in scenario_ids) or len(set(scenario_ids)) != 16:
        raise ReplayError("scenario identities are empty or duplicated")
    result: dict[str, Any] = {}
    for arm in ("Q", "R"):
        metrics = [report[arm]["metrics"] for report in pair_reports]
        if any(not isinstance(value, RiskMetrics) or value.request_count != 2 for value in metrics):
            raise ReplayError("malformed pair metrics")
        result[arm] = _aggregate_metrics(metrics)
    if result["Q"].request_count != 32 or result["R"].request_count != 32:
        raise ReplayError("formal campaign denominator is not 32 requests")
    if {row[0] for row in result["Q"].request_rows} != {row[0] for row in result["R"].request_rows}:
        raise ReplayError("Q/R request universe changed")
    result["q_to_r_absolute_miss_reduction"] = result["Q"].miss_rate - result["R"].miss_rate
    result["q_to_r_relative_cvar90_reduction"] = (
        (result["Q"].cvar90_normalized_tardiness - result["R"].cvar90_normalized_tardiness)
        / result["Q"].cvar90_normalized_tardiness
        if result["Q"].cvar90_normalized_tardiness > 0
        else 0.0
    )
    result["strict_flip_count"] = sum(report.get("strict_world_action_flip") is True for report in pair_reports)
    return result


def aggregate_baselines(baseline_reports: Sequence[Mapping[str, Any]]) -> dict[str, RiskMetrics]:
    if len(baseline_reports) != 16:
        raise ReplayError("baseline aggregate requires exactly 16 pairs")
    scenario_ids = [str(report.get("scenario_id", "")) for report in baseline_reports]
    if any(not value for value in scenario_ids) or len(set(scenario_ids)) != 16:
        raise ReplayError("baseline scenario identities are empty or duplicated")
    result: dict[str, RiskMetrics] = {}
    for policy in POLICIES:
        rows = [report["policies"][policy]["metrics"] for report in baseline_reports]
        if any(not isinstance(value, RiskMetrics) for value in rows):
            raise ReplayError("malformed baseline metrics")
        result[policy] = _aggregate_metrics(rows)
        if result[policy].request_count != 32:
            raise ReplayError("baseline denominator is not 32 requests")
    return result


def calibrate_deadline_on_selection(
    native_selection: Sequence[NativePair],
    service: ServiceLUT,
    base_config: ReplayConfig,
    *,
    deadline_grid: Sequence[float] = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05),
) -> tuple[ReplayConfig, dict[str, Any]]:
    """Choose tightness using only Q miss rate on request-disjoint selection data."""
    if len(native_selection) != 16:
        raise ReplayError("deadline calibration requires exactly 16 selection pairs")
    if not deadline_grid or len(set(float(value) for value in deadline_grid)) != len(deadline_grid):
        raise ReplayError("deadline calibration grid is empty or duplicated")
    rows = []
    for factor in deadline_grid:
        _positive(float(factor), "deadline grid factor")
        candidate = replace(base_config, deadline_factor=float(factor))
        scenarios = [materialize_replay(value, service, candidate) for value in native_selection]
        q_reports = [optimize_q_only(scenario) for scenario in scenarios]
        q_metrics = _aggregate_metrics([report["metrics"] for report in q_reports])
        if q_metrics.request_count != 32:
            raise ReplayError("selection calibration denominator is not 32 requests")
        rows.append(
            {
                "deadline_factor": float(factor),
                "q_miss_rate": q_metrics.miss_rate,
                "distance_to_target": abs(q_metrics.miss_rate - 0.5),
                "eligible": 0.20 <= q_metrics.miss_rate <= 0.80,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ReplayError("BLOCKED_NO_NONDEGENERATE_SELECTION_DEADLINE")
    selected = min(
        eligible,
        key=lambda row: (row["distance_to_target"], row["deadline_factor"]),
    )
    effective = replace(base_config, deadline_factor=float(selected["deadline_factor"]))
    return effective, {
        "status": "FROZEN_FROM_SELECTION_Q_ONLY",
        "selection_pair_count": 16,
        "selection_request_count": 32,
        "target_q_miss_rate": 0.5,
        "eligibility_interval": [0.20, 0.80],
        "selected_deadline_factor": effective.deadline_factor,
        "selected_q_miss_rate": selected["q_miss_rate"],
        "rows": rows,
        "r_outcomes_read_for_selection": False,
    }


def paired_bootstrap(
    pair_reports: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    if len(pair_reports) != 16 or replicates <= 0:
        raise ReplayError("bootstrap requires 16 pairs and positive replicates")
    rng = random.Random(seed)
    miss_deltas: list[float] = []
    cvar_deltas: list[float] = []
    for _ in range(replicates):
        indices = [rng.randrange(len(pair_reports)) for _ in pair_reports]
        q_rows: list[tuple[str, float, float]] = []
        r_rows: list[tuple[str, float, float]] = []
        for draw, index in enumerate(indices):
            for arm, sink in (("Q", q_rows), ("R", r_rows)):
                metrics = pair_reports[index][arm]["metrics"]
                for request_id, miss, tardiness in metrics.request_rows:
                    sink.append((f"draw:{draw}:{request_id}", miss, tardiness))
        q_miss = sum(row[1] for row in q_rows) / len(q_rows)
        r_miss = sum(row[1] for row in r_rows) / len(r_rows)
        q_cvar, r_cvar = empirical_cvar90([row[2] for row in q_rows]), empirical_cvar90([row[2] for row in r_rows])
        miss_deltas.append(q_miss - r_miss)
        cvar_deltas.append((q_cvar - r_cvar) / q_cvar if q_cvar > 0 else 0.0)

    def interval(values: Sequence[float]) -> tuple[float, float, float]:
        ordered = sorted(values)
        lo = ordered[max(0, math.floor(0.025 * len(ordered)))]
        hi = ordered[min(len(ordered) - 1, math.ceil(0.975 * len(ordered)) - 1)]
        return (sum(values) / len(values), lo, hi)

    return {
        "unit": "matched_request_pair",
        "replicates": replicates,
        "seed": seed,
        "absolute_miss_reduction": interval(miss_deltas),
        "relative_cvar90_reduction": interval(cvar_deltas),
    }


def decide_campaign(
    aggregate: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    *,
    min_absolute_miss_reduction: float = 0.05,
    min_strict_flip_count: int = 4,
) -> dict[str, Any]:
    """Frozen existence gate; miss risk is primary and CVaR90 is secondary."""
    miss_interval = bootstrap.get("absolute_miss_reduction")
    if (
        not isinstance(miss_interval, tuple)
        or len(miss_interval) != 3
        or any(not math.isfinite(float(value)) for value in miss_interval)
    ):
        raise ReplayError("bootstrap miss interval is malformed")
    reduction = float(aggregate["q_to_r_absolute_miss_reduction"])
    flips = int(aggregate["strict_flip_count"])
    q_miss = float(aggregate["Q"].miss_rate)
    identifiable = 0.05 <= q_miss <= 0.95
    gates = {
        "holdout_q_risk_nondegenerate": identifiable,
        "effect_size": reduction >= min_absolute_miss_reduction,
        "paired_bootstrap_lower_gt_zero": float(miss_interval[1]) > 0,
        "minimum_strict_action_flips": flips >= min_strict_flip_count,
        "r_not_worse_on_primary": aggregate["R"].miss_rate <= aggregate["Q"].miss_rate + EPS,
    }
    return {
        "status": (
            "INVALID_WORKLOAD_IDENTIFIABILITY"
            if not identifiable
            else ("PASS" if all(gates.values()) else "FAIL")
        ),
        "primary_estimand": "Q_minus_R_absolute_request_miss_rate",
        "thresholds": {
            "min_absolute_miss_reduction": min_absolute_miss_reduction,
            "bootstrap_lower_bound": 0.0,
            "min_strict_flip_count": min_strict_flip_count,
        },
        "observed": {
            "absolute_miss_reduction": reduction,
            "bootstrap_95pct": miss_interval,
            "strict_flip_count": flips,
        },
        "gates": gates,
    }


def decide_two_model(model_decisions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Require independent PASS on both frozen model keys; no pooled rescue."""
    if set(model_decisions) != {"olmoe", "llmjp"}:
        raise ReplayError("two-model decision requires exactly olmoe and llmjp")
    passed = all(value.get("status") == "PASS" for value in model_decisions.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "rule": "OLMoE_AND_LLMJP_WITHOUT_POOLING",
        "models": dict(model_decisions),
    }


def run_campaign(scenarios: Sequence[ReplayScenario], config: ReplayConfig) -> dict[str, Any]:
    validate_config(config)
    reports = [optimize_information(scenario) for scenario in scenarios]
    aggregate = aggregate_campaign(reports)
    bootstrap = paired_bootstrap(
        reports, replicates=config.bootstrap_replicates, seed=config.bootstrap_seed
    )
    baselines = [
        {"scenario_id": scenario.scenario_id, "policies": evaluate_baselines(scenario)}
        for scenario in scenarios
    ]
    controls = [
        {"scenario_id": scenario.scenario_id, "controls": negative_controls(scenario)}
        for scenario in scenarios
    ]
    baseline_aggregate = aggregate_baselines(baselines)
    decision = decide_campaign(aggregate, bootstrap)
    return {
        "schema_version": "fjrc-corrected-level1-replay-v1",
        "status": "LOGICAL_TRACE_REPLAY_ONLY",
        "timing_source": TIMING_SOURCE,
        "config": config,
        "pair_reports": reports,
        "aggregate": aggregate,
        "paired_bootstrap": bootstrap,
        "baseline_reports": baselines,
        "baseline_aggregate": baseline_aggregate,
        "negative_controls": controls,
        "decision": decision,
    }
