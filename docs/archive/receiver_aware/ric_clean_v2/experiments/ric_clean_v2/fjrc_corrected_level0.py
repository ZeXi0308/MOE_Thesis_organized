#!/usr/bin/env python3
"""Corrected FJRC Level-0 information-isolation reference.

This is a CPU-only, normalized-service oracle.  It tests whether keyed join
phase (J) adds decision value after exact receiver queue state (Q) is already
known.  It is not a network, GPU, or serving simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from typing import Any, Mapping, Sequence


EPS = 1e-9
ARMS = ("B0", "Q", "J", "R")


class CorrectedFJRCError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class Task:
    task_id: str
    join_id: str
    request_id: str
    sender_rank: int
    receiver_rank: int
    ready_us: float
    service_us: float


@dataclass(frozen=True)
class Join:
    join_id: str
    request_id: str
    receiver_rank: int
    deadline_us: float
    combine_us: float
    sibling_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class PriorCompletion:
    task_id: str
    start_us: float
    completion_us: float


@dataclass(frozen=True)
class World:
    world_id: str
    prior: tuple[PriorCompletion, ...]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    decision_time_us: float
    receiver_available_us: tuple[tuple[int, float], ...]
    tasks: tuple[Task, ...]
    joins: tuple[Join, ...]
    candidate_task_ids: tuple[str, ...]
    worlds: tuple[World, World]


@dataclass(frozen=True)
class Metrics:
    miss_rate: float
    mean_tardiness: float
    makespan_us: float
    miss_vector: tuple[tuple[str, bool], ...]
    close_vector: tuple[tuple[str, float], ...]

    @property
    def objective(self) -> tuple[float, float, float]:
        return (self.miss_rate, self.mean_tardiness, self.makespan_us)


def _finite_nonnegative(value: float, name: str, *, positive: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise CorrectedFJRCError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def validate_scenario(scenario: Scenario, *, require_phase_difference: bool = True) -> None:
    _finite_nonnegative(scenario.decision_time_us, "decision_time_us")
    tasks = {task.task_id: task for task in scenario.tasks}
    joins = {join.join_id: join for join in scenario.joins}
    candidates = set(scenario.candidate_task_ids)
    if len(tasks) != len(scenario.tasks) or len(joins) != len(scenario.joins):
        raise CorrectedFJRCError("duplicate task or join identity")
    if len(scenario.worlds) != 2 or scenario.worlds[0].world_id == scenario.worlds[1].world_id:
        raise CorrectedFJRCError("exactly two distinct worlds are required")
    if len(candidates) != len(scenario.candidate_task_ids) or not candidates <= set(tasks):
        raise CorrectedFJRCError("candidate identity set is invalid")
    if len(candidates) < 2:
        raise CorrectedFJRCError("at least two candidates are required")
    candidate_rows = [tasks[task_id] for task_id in candidates]
    if len({task.sender_rank for task in candidate_rows}) < 2 or len({task.join_id for task in candidate_rows}) < 2:
        raise CorrectedFJRCError("candidates must span at least two senders and two joins")
    if any(task.ready_us > scenario.decision_time_us + EPS for task in candidate_rows):
        raise CorrectedFJRCError("a decision candidate is not ready at t0")

    sibling_census: list[str] = []
    for join in scenario.joins:
        _finite_nonnegative(join.deadline_us, "deadline_us", positive=True)
        _finite_nonnegative(join.combine_us, "combine_us", positive=True)
        if not join.sibling_task_ids or len(set(join.sibling_task_ids)) != len(join.sibling_task_ids):
            raise CorrectedFJRCError("join sibling set is empty or duplicated")
        for task_id in join.sibling_task_ids:
            task = tasks.get(task_id)
            if task is None or (task.join_id, task.request_id, task.receiver_rank) != (
                join.join_id,
                join.request_id,
                join.receiver_rank,
            ):
                raise CorrectedFJRCError("join/task identity mismatch")
        sibling_census.extend(join.sibling_task_ids)
    if sorted(sibling_census) != sorted(tasks):
        raise CorrectedFJRCError("full task universe is not exactly covered by joins")

    for task in scenario.tasks:
        if not task.task_id or not task.join_id or not task.request_id:
            raise CorrectedFJRCError("empty task identity")
        _finite_nonnegative(task.ready_us, "ready_us")
        _finite_nonnegative(task.service_us, "service_us", positive=True)

    queue_map = dict(scenario.receiver_available_us)
    receiver_set = {join.receiver_rank for join in scenario.joins}
    if len(receiver_set) != 1:
        raise CorrectedFJRCError("Level-0 reference supports exactly one receiver resource")
    if len(queue_map) != len(scenario.receiver_available_us) or set(queue_map) != receiver_set:
        raise CorrectedFJRCError("receiver queue map is incomplete or duplicated")
    for value in queue_map.values():
        if not math.isfinite(value) or value + EPS < scenario.decision_time_us:
            raise CorrectedFJRCError("receiver availability predates decision time")

    auxiliary = set(tasks) - candidates
    prior_sets: list[set[str]] = []
    future_signatures: list[list[tuple[int, int, float, float]]] = []
    for world in scenario.worlds:
        prior_ids: set[str] = set()
        for event in world.prior:
            task = tasks.get(event.task_id)
            if task is None or event.task_id in candidates or event.task_id in prior_ids:
                raise CorrectedFJRCError("prior completion identity is invalid")
            if not math.isfinite(event.start_us) or not math.isfinite(event.completion_us):
                raise CorrectedFJRCError("non-finite prior history")
            if event.start_us + task.service_us > event.completion_us + EPS:
                raise CorrectedFJRCError("prior history cannot contain task service")
            if event.completion_us > scenario.decision_time_us + EPS:
                raise CorrectedFJRCError("prior completion occurs after t0")
            prior_ids.add(event.task_id)
        if require_phase_difference and (not prior_ids or prior_ids >= auxiliary):
            raise CorrectedFJRCError("each primary world needs prior and fixed-after auxiliary tasks")
        future_ids = auxiliary - prior_ids
        # Identities may swap, but work/resource/release signatures must match.
        future_signatures.append(
            sorted(
                (tasks[value].sender_rank, tasks[value].receiver_rank, tasks[value].ready_us, tasks[value].service_us)
                for value in future_ids
            )
        )
        if prior_ids | future_ids != auxiliary or prior_ids & future_ids:
            raise CorrectedFJRCError("past/future partition does not cover auxiliary tasks exactly once")
        prior_sets.append(prior_ids)
    if future_signatures[0] != future_signatures[1]:
        raise CorrectedFJRCError("future work/resource multiset drift")
    if require_phase_difference and prior_sets[0] == prior_sets[1]:
        raise CorrectedFJRCError("worlds do not differ in join phase")


def observation(
    scenario: Scenario,
    world_index: int,
    arm: str,
    *,
    require_phase_difference: bool = True,
) -> Mapping[str, Any]:
    validate_scenario(scenario, require_phase_difference=require_phase_difference)
    if world_index not in (0, 1) or arm not in ARMS:
        raise CorrectedFJRCError("invalid world or information arm")
    tasks = {task.task_id: task for task in scenario.tasks}
    world = scenario.worlds[world_index]
    base: dict[str, Any] = {
        "decision_time_us": scenario.decision_time_us,
        "candidates": sorted(
            (
                task_id,
                tasks[task_id].join_id,
                tasks[task_id].request_id,
                tasks[task_id].sender_rank,
                tasks[task_id].receiver_rank,
                tasks[task_id].service_us,
            )
            for task_id in scenario.candidate_task_ids
        ),
        "deadlines": sorted((join.join_id, join.deadline_us) for join in scenario.joins),
        "join_sizes": sorted((join.join_id, len(join.sibling_task_ids)) for join in scenario.joins),
    }
    if arm in {"Q", "R"}:
        base["queue_map"] = sorted(scenario.receiver_available_us)
    if arm in {"J", "R"}:
        prior = {event.task_id for event in world.prior}
        base["completed_siblings"] = sorted(
            (join.join_id, tuple(sorted(prior & set(join.sibling_task_ids))))
            for join in scenario.joins
        )
    return base


def observation_fingerprint(
    scenario: Scenario,
    world_index: int,
    arm: str,
    *,
    require_phase_difference: bool = True,
) -> str:
    return _sha(
        observation(
            scenario,
            world_index,
            arm,
            require_phase_difference=require_phase_difference,
        )
    )


def legal_actions(scenario: Scenario, *, require_phase_difference: bool = True) -> tuple[str, ...]:
    validate_scenario(scenario, require_phase_difference=require_phase_difference)
    return tuple(sorted(scenario.candidate_task_ids))


def _post_t0_order(scenario: Scenario, world_index: int, first_task_id: str) -> list[Task]:
    tasks = {task.task_id: task for task in scenario.tasks}
    world = scenario.worlds[world_index]
    prior = {event.task_id for event in world.prior}
    remaining = [task for task in scenario.tasks if task.task_id not in prior and task.task_id != first_task_id]
    if first_task_id not in scenario.candidate_task_ids:
        raise CorrectedFJRCError("first action is not a decision candidate")
    return [tasks[first_task_id]] + sorted(remaining, key=lambda task: (task.ready_us, task.task_id))


def simulate(
    scenario: Scenario,
    world_index: int,
    first_task_id: str,
    *,
    require_phase_difference: bool = True,
) -> dict[str, Any]:
    validate_scenario(scenario, require_phase_difference=require_phase_difference)
    if world_index not in (0, 1):
        raise CorrectedFJRCError("world index must be 0 or 1")
    tasks = {task.task_id: task for task in scenario.tasks}
    world = scenario.worlds[world_index]
    prior_completion = {event.task_id: event.completion_us for event in world.prior}
    receiver_available = dict(scenario.receiver_available_us)
    completions = dict(prior_completion)
    events: list[dict[str, Any]] = []
    now = scenario.decision_time_us
    for ordinal, task in enumerate(_post_t0_order(scenario, world_index, first_task_id)):
        now = max(now, task.ready_us)
        start = max(now, receiver_available[task.receiver_rank])
        end = start + task.service_us
        receiver_available[task.receiver_rank] = end
        now = end  # B=1: release the next credit after this receiver service completes.
        if task.task_id in completions:
            raise CorrectedFJRCError("task completed more than once")
        completions[task.task_id] = end
        events.append(
            {
                "ordinal": ordinal,
                "task_id": task.task_id,
                "join_id": task.join_id,
                "receiver_rank": task.receiver_rank,
                "start_us": start,
                "completion_us": end,
            }
        )
    if set(completions) != set(tasks):
        raise CorrectedFJRCError("full task universe did not complete exactly once")

    combine_available: dict[int, float] = {receiver: scenario.decision_time_us for receiver in receiver_available}
    join_ready = sorted(
        (
            max(completions[value] for value in join.sibling_task_ids),
            join.join_id,
            join,
        )
        for join in scenario.joins
    )
    closes: dict[str, float] = {}
    combine_events: list[dict[str, Any]] = []
    for ready, _tie, join in join_ready:
        start = max(ready, combine_available[join.receiver_rank])
        close = start + join.combine_us
        combine_available[join.receiver_rank] = close
        closes[join.join_id] = close
        combine_events.append({"join_id": join.join_id, "ready_us": ready, "close_us": close})
    request_rows = []
    for join in sorted(scenario.joins, key=lambda item: item.request_id):
        close = closes[join.join_id]
        request_rows.append(
            {
                "request_id": join.request_id,
                "join_id": join.join_id,
                "close_us": close,
                "deadline_us": join.deadline_us,
                "miss": close > join.deadline_us,
                "tardiness_us": max(0.0, close - join.deadline_us),
            }
        )
    return {
        "world_id": world.world_id,
        "first_task_id": first_task_id,
        "events": events,
        "combine_events": combine_events,
        "requests": request_rows,
        "accounting": {
            "task_universe": len(tasks),
            "prior_count": len(prior_completion),
            "post_t0_count": len(events),
            "unique_completion_count": len(completions),
            "combine_count": len(combine_events),
        },
    }


def _metrics(results: Sequence[Mapping[str, Any]]) -> Metrics:
    if len(results) != 2:
        raise CorrectedFJRCError("paired policy needs exactly two world results")
    rows: dict[str, list[Mapping[str, Any]]] = {}
    makespan = 0.0
    for result in results:
        for row in result["requests"]:
            rows.setdefault(str(row["request_id"]), []).append(row)
            makespan = max(makespan, float(row["close_us"]))
    if not rows or any(len(value) != 2 for value in rows.values()):
        raise CorrectedFJRCError("request/world denominator drift")
    miss_vector = []
    close_vector = []
    tardiness = []
    for request_id in sorted(rows):
        values = rows[request_id]
        miss_vector.append((request_id, any(bool(row["miss"]) for row in values)))
        close_vector.append((request_id, sum(float(row["close_us"]) for row in values) / 2.0))
        tardiness.append(sum(float(row["tardiness_us"]) for row in values) / 2.0)
    # Primary risk folds worlds before averaging requests.
    miss_rate = sum(sum(bool(row["miss"]) for row in rows[key]) / 2.0 for key in rows) / len(rows)
    return Metrics(miss_rate, sum(tardiness) / len(tardiness), makespan, tuple(miss_vector), tuple(close_vector))


def _equal_objective(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(left, right))


def optimize_arm(
    scenario: Scenario,
    arm: str,
    *,
    force_uninformative: bool = False,
    require_phase_difference: bool = True,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise CorrectedFJRCError("unknown information arm")
    actions = legal_actions(scenario, require_phase_difference=require_phase_difference)
    table = {
        (world, action): simulate(
            scenario,
            world,
            action,
            require_phase_difference=require_phase_difference,
        )
        for world in (0, 1)
        for action in actions
    }
    labels = (
        ("UNINFORMATIVE", "UNINFORMATIVE")
        if force_uninformative
        else tuple(
            observation_fingerprint(
                scenario,
                world,
                arm,
                require_phase_difference=require_phase_difference,
            )
            for world in (0, 1)
        )
    )
    policies = []
    for action0, action1 in itertools.product(actions, repeat=2):
        if labels[0] == labels[1] and action0 != action1:
            continue
        metrics = _metrics((table[(0, action0)], table[(1, action1)]))
        policies.append(((action0, action1), metrics))
    best_objective = min(metrics.objective for _, metrics in policies)
    optimal = tuple(sorted(policy for policy, metrics in policies if _equal_objective(metrics.objective, best_objective)))
    selected = optimal[0]
    selected_metrics = next(metrics for policy, metrics in policies if policy == selected)
    return {
        "arm": arm,
        "observation_labels": labels,
        "observation_distinguishes_worlds": labels[0] != labels[1],
        "optimal_policies": optimal,
        "selected_policy": selected,
        "unique": len(optimal) == 1,
        "strict_world_action_flip": len(optimal) == 1 and optimal[0][0] != optimal[0][1],
        "metrics": selected_metrics,
    }


def optimize_information_lattice(scenario: Scenario) -> dict[str, Any]:
    reports = {arm: optimize_arm(scenario, arm) for arm in ARMS}
    if reports["B0"]["observation_distinguishes_worlds"] or reports["Q"]["observation_distinguishes_worlds"]:
        raise CorrectedFJRCError("B0/Q leaked keyed join phase")
    if not reports["J"]["observation_distinguishes_worlds"] or not reports["R"]["observation_distinguishes_worlds"]:
        raise CorrectedFJRCError("J/R failed to expose keyed join phase")
    q = reports["Q"]["metrics"]
    r = reports["R"]["metrics"]
    if r.objective > q.objective and not _equal_objective(r.objective, q.objective):
        raise CorrectedFJRCError("additional join information worsened the exact optimum")
    return {
        "scenario_id": scenario.scenario_id,
        "arms": reports,
        "q_to_r_absolute_miss_reduction": q.miss_rate - r.miss_rate,
        "q_to_r_mean_tardiness_reduction": q.mean_tardiness - r.mean_tardiness,
        "q_to_r_strict_first_action_flip": reports["R"]["strict_world_action_flip"],
    }


def equal_phase_control(scenario: Scenario) -> dict[str, Any]:
    world0 = scenario.worlds[0]
    controlled = replace(
        scenario,
        scenario_id=f"{scenario.scenario_id}:equal-phase",
        worlds=(world0, replace(world0, world_id=f"{world0.world_id}:copy")),
    )
    informative = optimize_arm(controlled, "R", require_phase_difference=False)
    baseline = optimize_arm(controlled, "Q", require_phase_difference=False)
    passed = (
        _equal_objective(informative["metrics"].objective, baseline["metrics"].objective)
        and informative["strict_world_action_flip"] is False
    )
    return {"name": "equal_phase_partition", "passed": passed}


def shuffled_key_control(scenario: Scenario) -> dict[str, Any]:
    q = optimize_arm(scenario, "Q")
    shuffled = optimize_arm(scenario, "R", force_uninformative=True)
    passed = _equal_objective(q["metrics"].objective, shuffled["metrics"].objective) and not shuffled[
        "strict_world_action_flip"
    ]
    return {"name": "shuffled_key", "passed": passed, "Q": q, "shuffled_R": shuffled}


def fanout1_control(scenario: Scenario) -> dict[str, Any]:
    validate_scenario(scenario, require_phase_difference=False)
    if any(len(join.sibling_task_ids) != 1 for join in scenario.joins):
        raise CorrectedFJRCError("fanout1 control contains a multi-sibling join")
    q = optimize_arm(scenario, "Q", require_phase_difference=False)
    r = optimize_arm(scenario, "R", require_phase_difference=False)
    passed = (
        _equal_objective(q["metrics"].objective, r["metrics"].objective)
        and not r["observation_distinguishes_worlds"]
        and not r["strict_world_action_flip"]
    )
    return {"name": "fanout1", "passed": passed, "Q": q, "R": r}
