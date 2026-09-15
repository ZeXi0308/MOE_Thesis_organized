#!/usr/bin/env python3
"""Pure four-world oracle for the frozen PhaseMap-MILP gate.

The module intentionally does not read route, LUT, selection, or holdout files.
It consumes a fully materialized two-request episode, replays its reachable
receiver histories, enforces the B0/Q/J/R non-anticipativity partitions, and
folds counterfactual worlds back onto native requests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Iterable, Literal, Mapping, Sequence


EPS = 1e-9
LEX_REL_TOLERANCE = 1e-10
LEX_ABS_TOLERANCE = 1e-10
BEST_SINGLE_COMPARATOR = (
    "expected_miss_count",
    "cvar90_normalized_tardiness",
    "mean_normalized_tardiness",
    "expected_join_close_sum",
    "arm_identity",
)
Arm = Literal["B0", "Q", "J", "R"]
ScenarioKind = Literal[
    "primary", "equal_q", "equal_j", "fanout1", "no_conflict", "shuffled_key"
]
Action = tuple[tuple[int, str], ...]
Policy = tuple[tuple[str, Action], ...]


class PhaseMapError(RuntimeError):
    """A frozen protocol, causality, or accounting invariant failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise PhaseMapError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class Contribution:
    task_id: str
    join_id: str
    request_id: str
    sender_rank: int
    receiver_rank: int
    pack_us: float
    cut_us: float
    unpack_us: float
    is_decision: bool


@dataclass(frozen=True)
class Join:
    join_id: str
    request_id: str
    receiver_rank: int
    request_arrival_us: float
    deadline_us: float
    combine_us: float
    sibling_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReceiverJob:
    job_id: str
    receiver_rank: int
    arrival_us: float
    service_us: float
    task_id: str | None = None


@dataclass(frozen=True)
class SenderHistoryEvent:
    task_id: str
    sender_rank: int
    send_complete_us: float
    receiver_commit_ack: bool = False


@dataclass(frozen=True)
class World:
    world_id: str
    q_bit: int
    j_bit: int
    receiver_jobs: tuple[ReceiverJob, ...]
    sender_history: tuple[SenderHistoryEvent, ...]
    j_observation_override: str | None = None


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    contributions: tuple[Contribution, ...]
    joins: tuple[Join, ...]
    worlds: tuple[World, ...]
    kind: ScenarioKind = "primary"
    low_depth: int = 8
    high_depth: int = 16


@dataclass(frozen=True)
class PairMetrics:
    request_count: int
    expected_miss_count: float
    miss_rate: float
    expected_tardiness_sum: float
    mean_normalized_tardiness: float
    expected_join_close_sum: float
    expected_miss_by_request: tuple[tuple[str, float], ...]
    expected_tardiness_by_request: tuple[tuple[str, float], ...]
    expected_join_close_by_request: tuple[tuple[str, float], ...]

    @property
    def objective(self) -> tuple[float, float, float]:
        return (
            self.expected_miss_count,
            self.expected_tardiness_sum,
            self.expected_join_close_sum,
        )


@dataclass(frozen=True)
class AggregateMetrics:
    request_count: int
    expected_miss_count: float
    miss_rate: float
    cvar90_normalized_tardiness: float
    mean_normalized_tardiness: float
    expected_join_close_sum: float
    expected_miss_by_request: tuple[tuple[str, float], ...]
    expected_tardiness_by_request: tuple[tuple[str, float], ...]

    @property
    def objective(self) -> tuple[float, float, float, float]:
        return (
            self.expected_miss_count,
            self.cvar90_normalized_tardiness,
            self.mean_normalized_tardiness * self.request_count,
            self.expected_join_close_sum,
        )


def empirical_cvar90(values: Sequence[float]) -> float:
    if not values:
        raise PhaseMapError("CVaR90 requires native requests")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise PhaseMapError("invalid normalized tardiness")
    tail = max(1, math.ceil(0.1 * len(values) - 1e-12))
    return sum(sorted(values, reverse=True)[:tail]) / tail


def lexicographic_tolerance(minimum: float) -> float:
    """Return the frozen tolerance band for one numeric lexicographic stage."""
    if not math.isfinite(minimum):
        raise PhaseMapError("lexicographic minimum is not finite")
    return max(LEX_ABS_TOLERANCE, abs(minimum) * LEX_REL_TOLERANCE)


def lexicographic_optimal_rows(
    rows: Sequence[tuple[Any, Sequence[float]]],
) -> tuple[tuple[float, ...], tuple[Any, ...]]:
    """Return stage minima and all tolerance-banded lexicographic optima."""
    if not rows:
        raise PhaseMapError("empty lexicographic candidate set")
    width = len(rows[0][1])
    if width == 0:
        raise PhaseMapError("empty lexicographic objective")
    candidates = list(rows)
    for _identity, objective in candidates:
        if len(objective) != width or any(
            not math.isfinite(float(value)) for value in objective
        ):
            raise PhaseMapError("malformed lexicographic objective")
    minima: list[float] = []
    for index in range(width):
        minimum = min(float(objective[index]) for _identity, objective in candidates)
        upper = minimum + lexicographic_tolerance(minimum)
        candidates = [
            (identity, objective)
            for identity, objective in candidates
            if float(objective[index]) <= upper
        ]
        minima.append(minimum)
    return tuple(minima), tuple(identity for identity, _objective in candidates)


def lexicographic_no_worse(left: Sequence[float], right: Sequence[float]) -> bool:
    """Return whether left is lexicographically no worse within frozen bands."""
    if len(left) != len(right):
        raise PhaseMapError("lexicographic objectives have different widths")
    for left_value, right_value in zip(left, right):
        tolerance = max(
            lexicographic_tolerance(float(left_value)),
            lexicographic_tolerance(float(right_value)),
        )
        if float(left_value) < float(right_value) - tolerance:
            return True
        if float(left_value) > float(right_value) + tolerance:
            return False
    return True


def _decision_senders(scenario: Scenario) -> tuple[int, ...]:
    return tuple(sorted({task.sender_rank for task in scenario.contributions if task.is_decision}))


def enumerate_actions(scenario: Scenario) -> tuple[Action, ...]:
    _validate_scenario(scenario)
    senders = _decision_senders(scenario)
    if not senders:
        return ((),)
    choices: list[tuple[tuple[int, str], ...]] = []
    for sender in senders:
        task_ids = sorted(
            task.task_id
            for task in scenario.contributions
            if task.is_decision and task.sender_rank == sender
        )
        choices.append(tuple((sender, task_id) for task_id in task_ids))
    return tuple(tuple(value) for value in itertools.product(*choices))


def _validate_action(scenario: Scenario, action: Action) -> None:
    senders = _decision_senders(scenario)
    action_map = dict(action)
    if len(action_map) != len(action) or tuple(sorted(action_map)) != senders:
        raise PhaseMapError("action must cover every decision sender exactly once")
    for sender in senders:
        legal = {
            task.task_id
            for task in scenario.contributions
            if task.is_decision and task.sender_rank == sender
        }
        if action_map[sender] not in legal:
            raise PhaseMapError("action selected a task outside its sender-local pair")


def _receiver_replay(
    jobs: Sequence[ReceiverJob],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    events: list[dict[str, Any]] = []
    completion: dict[str, float] = {}
    available: dict[int, float] = {}
    for job in sorted(jobs, key=lambda value: (value.receiver_rank, value.arrival_us, value.job_id)):
        start = max(job.arrival_us, available.get(job.receiver_rank, -math.inf))
        end = start + job.service_us
        available[job.receiver_rank] = end
        events.append(
            {
                "job_id": job.job_id,
                "task_id": job.task_id,
                "receiver_rank": job.receiver_rank,
                "arrival_us": job.arrival_us,
                "start_us": start,
                "end_us": end,
                "service_us": job.service_us,
            }
        )
        if job.task_id is not None:
            if job.task_id in completion:
                raise PhaseMapError("foreground receiver task executed twice")
            completion[job.task_id] = end
    return events, completion


def _world_state(scenario: Scenario, world: World) -> dict[str, Any]:
    events, completion = _receiver_replay(world.receiver_jobs)
    task_by_id = {task.task_id: task for task in scenario.contributions}
    q_state: dict[str, dict[str, float | int]] = {}
    receiver_set = sorted({task.receiver_rank for task in scenario.contributions})
    for receiver in receiver_set:
        unfinished = [row for row in events if row["receiver_rank"] == receiver and row["end_us"] > 0]
        q_state[str(receiver)] = {
            "depth": len(unfinished),
            "unfinished_work_us": sum(
                row["end_us"] - max(0.0, row["start_us"]) for row in unfinished
            ),
            "availability_us": max([0.0] + [float(row["end_us"]) for row in unfinished]),
        }
    j_state: dict[str, dict[str, Any]] = {}
    for join in sorted(scenario.joins, key=lambda value: value.join_id):
        committed: list[str] = []
        queued: list[str] = []
        decision: list[str] = []
        for task_id in join.sibling_task_ids:
            task = task_by_id[task_id]
            if task.is_decision:
                decision.append(task_id)
            elif completion[task_id] <= EPS:
                committed.append(task_id)
            else:
                queued.append(task_id)
        j_state[join.join_id] = {
            "committed": sorted(committed),
            "queued": sorted(queued),
            "decision_ready": sorted(decision),
            "deficit": len(queued) + len(decision),
        }
    raw_j_hash = _sha(j_state)
    effective_j = world.j_observation_override or raw_j_hash
    q_multiset = sorted(q_state.values(), key=_canonical)
    j_multiset = sorted(
        (
            {
                "committed_count": len(value["committed"]),
                "queued_count": len(value["queued"]),
                "decision_ready_count": len(value["decision_ready"]),
                "deficit": value["deficit"],
            }
            for value in j_state.values()
        ),
        key=_canonical,
    )
    return {
        "events": events,
        "carrier_completion": completion,
        "q_state": q_state,
        "j_state": j_state,
        "q_hash": _sha(q_state),
        "q_multiset_hash": _sha(q_multiset),
        "raw_j_hash": raw_j_hash,
        "raw_j_multiset_hash": _sha(j_multiset),
        "effective_j_hash": effective_j,
    }


def _validate_scenario(scenario: Scenario) -> None:
    if type(scenario.low_depth) is not int or type(scenario.high_depth) is not int:
        raise PhaseMapError("queue depths must be integers")
    if scenario.low_depth < 0 or scenario.high_depth < scenario.low_depth:
        raise PhaseMapError("invalid frozen low/high queue depths")
    if len(scenario.worlds) != 4:
        raise PhaseMapError("exactly four 2x2 worlds are required")
    world_cells = {(world.q_bit, world.j_bit) for world in scenario.worlds}
    if world_cells != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise PhaseMapError("worlds must cover the complete q-by-j Cartesian grid")
    if len({world.world_id for world in scenario.worlds}) != 4:
        raise PhaseMapError("world identity is missing or duplicated")
    tasks = {task.task_id: task for task in scenario.contributions}
    joins = {join.join_id: join for join in scenario.joins}
    if not tasks or len(tasks) != len(scenario.contributions):
        raise PhaseMapError("task identity is empty or duplicated")
    if len(joins) != len(scenario.joins) or len(joins) != 2:
        raise PhaseMapError("each episode requires exactly two unique joins")
    if len({join.request_id for join in scenario.joins}) != 2:
        raise PhaseMapError("each join must belong to a distinct native request")
    if scenario.kind not in {
        "primary", "equal_q", "equal_j", "fanout1", "no_conflict", "shuffled_key"
    }:
        raise PhaseMapError("unknown scenario kind")
    sibling_census: list[str] = []
    for task in scenario.contributions:
        for value, name in (
            (task.pack_us, "pack_us"), (task.cut_us, "cut_us"), (task.unpack_us, "unpack_us")
        ):
            _positive(value, name)
        if task.join_id not in joins:
            raise PhaseMapError("task references an absent join")
    for join in scenario.joins:
        _positive(join.combine_us, "combine_us")
        if not math.isfinite(join.request_arrival_us) or join.deadline_us <= join.request_arrival_us:
            raise PhaseMapError("deadline must follow a finite request arrival")
        if not join.sibling_task_ids or len(set(join.sibling_task_ids)) != len(join.sibling_task_ids):
            raise PhaseMapError("join sibling set is empty or duplicated")
        if scenario.kind == "fanout1" and len(join.sibling_task_ids) != 1:
            raise PhaseMapError("fanout1 control must have one sibling per join")
        for task_id in join.sibling_task_ids:
            task = tasks.get(task_id)
            if task is None:
                raise PhaseMapError("join references a missing sibling")
            if (
                task.join_id != join.join_id
                or task.request_id != join.request_id
                or task.receiver_rank != join.receiver_rank
            ):
                raise PhaseMapError("full sibling identity disagrees with its join")
        sibling_census.extend(join.sibling_task_ids)
        if scenario.kind in {"primary", "equal_q", "equal_j", "shuffled_key"}:
            carrier_count = sum(not tasks[task_id].is_decision for task_id in join.sibling_task_ids)
            if carrier_count < 4:
                raise PhaseMapError("PhaseMap join needs at least four phase-carrier siblings")
    if sorted(sibling_census) != sorted(tasks):
        raise PhaseMapError("task universe is not exactly the full sibling census")
    if scenario.kind in {"primary", "equal_q", "equal_j", "shuffled_key"} and len(
        {join.receiver_rank for join in scenario.joins}
    ) != 2:
        raise PhaseMapError("paired requests must have distinct output receivers")

    senders = _decision_senders(scenario)
    required = 0 if scenario.kind == "no_conflict" else (1 if scenario.kind == "fanout1" else 2)
    if len(senders) != required:
        raise PhaseMapError(f"{scenario.kind} requires exactly {required} decision sender(s)")
    request_set = {join.request_id for join in scenario.joins}
    for sender in senders:
        rows = [task for task in scenario.contributions if task.is_decision and task.sender_rank == sender]
        if len(rows) != 2 or {task.request_id for task in rows} != request_set:
            raise PhaseMapError("each decision sender needs one ready task from each request")
    if scenario.kind == "primary" and len(enumerate_actions_unchecked(scenario)) != 4:
        raise PhaseMapError("primary action space must contain four joint actions")

    nondecision = {task.task_id for task in scenario.contributions if not task.is_decision}
    decision = {task.task_id for task in scenario.contributions if task.is_decision}
    row1_services = {task.unpack_us for task in scenario.contributions if task.is_decision}
    sender_history_hashes: set[str] = set()
    for world in scenario.worlds:
        history_tasks: set[str] = set()
        canonical_history = []
        for event in world.sender_history:
            if event.task_id in history_tasks or event.task_id not in nondecision:
                raise PhaseMapError("sender history task census is duplicated or incomplete")
            history_tasks.add(event.task_id)
            task = tasks[event.task_id]
            if event.sender_rank != task.sender_rank:
                raise PhaseMapError("sender history uses the wrong expert sender rank")
            if not math.isfinite(event.send_complete_us) or event.send_complete_us >= 0:
                raise PhaseMapError("phase carrier must be send-complete before t0")
            if event.receiver_commit_ack:
                raise PhaseMapError("sender history leaks receiver commit ACK")
            canonical_history.append(
                (event.task_id, event.sender_rank, event.send_complete_us, event.receiver_commit_ack)
            )
        if history_tasks != nondecision:
            raise PhaseMapError("sender history must cover every phase carrier exactly once")
        sender_history_hashes.add(_sha(sorted(canonical_history)))
        seen_jobs: set[str] = set()
        seen_tasks: set[str] = set()
        for job in world.receiver_jobs:
            if job.job_id in seen_jobs:
                raise PhaseMapError("receiver job identity is duplicated")
            seen_jobs.add(job.job_id)
            if not math.isfinite(job.arrival_us) or job.arrival_us >= 0:
                raise PhaseMapError("pre-decision receiver job must arrive before t0")
            _positive(job.service_us, "receiver job service")
            row1_services.add(job.service_us)
            if job.task_id is not None:
                if job.task_id in seen_tasks or job.task_id not in nondecision:
                    raise PhaseMapError("carrier census is duplicated, absent, or marked decision")
                if tasks[job.task_id].receiver_rank != job.receiver_rank:
                    raise PhaseMapError("receiver key is not the task output receiver")
                seen_tasks.add(job.task_id)
        if seen_tasks != nondecision or seen_tasks & decision:
            raise PhaseMapError("every non-decision sibling must appear exactly once in each world")
    if len(sender_history_hashes) != 1:
        raise PhaseMapError("sender history leaks the receiver join phase")
    if len(row1_services) != 1:
        raise PhaseMapError("all receiver jobs must use one frozen row-1 unpack service")

    states = {(world.q_bit, world.j_bit): _world_state(scenario, world) for world in scenario.worlds}
    for q_bit in (0, 1):
        if states[(q_bit, 0)]["q_hash"] != states[(q_bit, 1)]["q_hash"]:
            raise PhaseMapError("flipping join phase changed the keyed Q observation")
    for j_bit in (0, 1):
        if states[(0, j_bit)]["effective_j_hash"] != states[(1, j_bit)]["effective_j_hash"]:
            raise PhaseMapError("flipping queue map changed the keyed J observation")
    if len({state["q_multiset_hash"] for state in states.values()}) != 1:
        raise PhaseMapError("unkeyed Q multiset drifted across matched worlds")
    if len({state["raw_j_multiset_hash"] for state in states.values()}) != 1:
        raise PhaseMapError("unkeyed J multiset drifted across matched worlds")

    joins_by_request = sorted(scenario.joins, key=lambda value: value.request_id)
    receiver_a, receiver_b = (join.receiver_rank for join in joins_by_request)
    if scenario.kind != "no_conflict":
        for world in scenario.worlds:
            state = states[(world.q_bit, world.j_bit)]
            expected_depths = (
                {receiver_a: scenario.low_depth, receiver_b: scenario.low_depth}
                if scenario.kind == "equal_q"
                else (
                    {receiver_a: scenario.low_depth, receiver_b: scenario.high_depth}
                    if world.q_bit == 0
                    else {receiver_a: scenario.high_depth, receiver_b: scenario.low_depth}
                )
            )
            actual_depths = {
                receiver: int(state["q_state"][str(receiver)]["depth"])
                for receiver in (receiver_a, receiver_b)
            }
            if actual_depths != expected_depths:
                raise PhaseMapError("receiver unfinished depth disagrees with the frozen q-bit map")
    if scenario.kind in {"primary", "equal_q", "equal_j", "shuffled_key"}:
        for world in scenario.worlds:
            state = states[(world.q_bit, world.j_bit)]
            queued = {
                join.request_id: len(state["j_state"][join.join_id]["queued"])
                for join in joins_by_request
            }
            expected = (
                {joins_by_request[0].request_id: 1, joins_by_request[1].request_id: 1}
                if scenario.kind == "equal_j"
                else (
                    {joins_by_request[0].request_id: 1, joins_by_request[1].request_id: 4}
                    if world.j_bit == 0
                    else {joins_by_request[0].request_id: 4, joins_by_request[1].request_id: 1}
                )
            )
            if queued != expected:
                raise PhaseMapError("receiver carrier census disagrees with the frozen near/far phase")


def enumerate_actions_unchecked(scenario: Scenario) -> tuple[Action, ...]:
    senders = _decision_senders(scenario)
    if not senders:
        return ((),)
    choices = []
    for sender in senders:
        choices.append(tuple(
            (sender, task.task_id)
            for task in sorted(scenario.contributions, key=lambda value: value.task_id)
            if task.is_decision and task.sender_rank == sender
        ))
    return tuple(tuple(value) for value in itertools.product(*choices))


def observation_key(scenario: Scenario, world: World, arm: Arm) -> str:
    _validate_scenario(scenario)
    state = _world_state(scenario, world)
    if arm == "B0":
        return "B0:UNKEYED"
    if arm == "Q":
        return f"Q:{state['q_hash']}"
    if arm == "J":
        return f"J:{state['effective_j_hash']}"
    if arm == "R":
        return f"R:{state['q_hash']}:{state['effective_j_hash']}"
    raise PhaseMapError("unknown information arm")


def causal_receiver_state(scenario: Scenario, world: World) -> dict[str, Any]:
    """Return the replay-derived causal Q/J state used by simple baselines."""
    _validate_scenario(scenario)
    state = _world_state(scenario, world)
    return {"q_state": state["q_state"], "j_state": state["j_state"]}


def observation_partitions(scenario: Scenario, arm: Arm) -> tuple[tuple[str, tuple[int, ...]], ...]:
    _validate_scenario(scenario)
    grouped: dict[str, list[int]] = {}
    for index, world in enumerate(scenario.worlds):
        grouped.setdefault(observation_key(scenario, world, arm), []).append(index)
    return tuple((key, tuple(indices)) for key, indices in sorted(grouped.items()))


def _sender_arrivals(scenario: Scenario, action: Action) -> tuple[dict[str, float], list[dict[str, Any]]]:
    _validate_action(scenario, action)
    action_map = dict(action)
    arrivals: dict[str, float] = {}
    events: list[dict[str, Any]] = []
    for sender in _decision_senders(scenario):
        tasks = sorted(
            (task for task in scenario.contributions if task.is_decision and task.sender_rank == sender),
            key=lambda value: (value.task_id != action_map[sender], value.task_id),
        )
        available = 0.0
        for task in tasks:
            pack_start = available
            pack_end = pack_start + task.pack_us
            cut_end = pack_end + task.cut_us
            arrivals[task.task_id] = cut_end
            available = cut_end
            events.append(
                {
                    "task_id": task.task_id,
                    "sender_rank": sender,
                    "pack_start_us": pack_start,
                    "pack_end_us": pack_end,
                    "cut_end_us": cut_end,
                }
            )
    return arrivals, events


def simulate(scenario: Scenario, world_index: int, action: Action) -> dict[str, Any]:
    _validate_scenario(scenario)
    if world_index not in range(4):
        raise PhaseMapError("world index must be in [0,3]")
    _validate_action(scenario, action)
    world = scenario.worlds[world_index]
    task_by_id = {task.task_id: task for task in scenario.contributions}
    decision_arrivals, sender_events = _sender_arrivals(scenario, action)
    receiver_jobs = list(world.receiver_jobs)
    for task_id, arrival in decision_arrivals.items():
        task = task_by_id[task_id]
        receiver_jobs.append(
            ReceiverJob(
                job_id=f"decision:{task_id}",
                task_id=task_id,
                receiver_rank=task.receiver_rank,
                arrival_us=arrival,
                service_us=task.unpack_us,
            )
        )
    receiver_events, completion = _receiver_replay(receiver_jobs)
    if set(completion) != set(task_by_id):
        raise PhaseMapError("foreground completion census differs from full sibling universe")

    ready_by_receiver: dict[int, list[tuple[float, str, Join]]] = {}
    for join in scenario.joins:
        ready = max(completion[task_id] for task_id in join.sibling_task_ids)
        ready_by_receiver.setdefault(join.receiver_rank, []).append((ready, join.join_id, join))
    combine_events: list[dict[str, Any]] = []
    close_by_join: dict[str, float] = {}
    for receiver, rows in sorted(ready_by_receiver.items()):
        available = -math.inf
        for ready, _tie, join in sorted(rows, key=lambda value: (value[0], value[1])):
            start = max(ready, available)
            close = start + join.combine_us
            available = close
            close_by_join[join.join_id] = close
            combine_events.append(
                {
                    "join_id": join.join_id,
                    "receiver_rank": receiver,
                    "join_ready_us": ready,
                    "combine_start_us": start,
                    "join_close_us": close,
                }
            )
    request_rows = []
    for join in sorted(scenario.joins, key=lambda value: value.request_id):
        close = close_by_join[join.join_id]
        normalized = max(0.0, close - join.deadline_us) / (
            join.deadline_us - join.request_arrival_us
        )
        request_rows.append(
            {
                "request_id": join.request_id,
                "join_id": join.join_id,
                "join_close_us": close,
                "deadline_us": join.deadline_us,
                "miss": close > join.deadline_us,
                "normalized_tardiness": normalized,
            }
        )
    post_t0_carriers = sum(
        row["task_id"] is not None
        and not task_by_id[str(row["task_id"])].is_decision
        and row["end_us"] > 0
        for row in receiver_events
    )
    return {
        "scenario_id": scenario.scenario_id,
        "world_id": world.world_id,
        "q_bit": world.q_bit,
        "j_bit": world.j_bit,
        "action": action,
        "sender_events": sender_events,
        "receiver_events": receiver_events,
        "combine_events": combine_events,
        "requests": request_rows,
        "accounting": {
            "native_siblings": len(task_by_id),
            "decision_pack_count": len(sender_events),
            "decision_cut_count": len(sender_events),
            "post_t0_foreground_unpack_count": post_t0_carriers + len(sender_events),
            "combine_count": len(combine_events),
        },
    }


def _policy_metrics(results: Sequence[Mapping[str, Any]]) -> PairMetrics:
    if len(results) != 4:
        raise PhaseMapError("one policy requires exactly four world results")
    cells = {(int(row["q_bit"]), int(row["j_bit"])) for row in results}
    if cells != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise PhaseMapError("policy results do not cover the 2x2 worlds")
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        for row in result["requests"]:
            by_request.setdefault(str(row["request_id"]), []).append(row)
    if len(by_request) != 2 or any(len(rows) != 4 for rows in by_request.values()):
        raise PhaseMapError("native request denominator drifted while folding worlds")
    misses = []
    tardiness = []
    closes = []
    for request_id in sorted(by_request):
        rows = by_request[request_id]
        misses.append((request_id, sum(bool(row["miss"]) for row in rows) / 4.0))
        tardiness.append(
            (request_id, sum(float(row["normalized_tardiness"]) for row in rows) / 4.0)
        )
        closes.append((request_id, sum(float(row["join_close_us"]) for row in rows) / 4.0))
    miss_count = sum(value for _, value in misses)
    tardiness_sum = sum(value for _, value in tardiness)
    close_sum = sum(value for _, value in closes)
    return PairMetrics(
        request_count=2,
        expected_miss_count=miss_count,
        miss_rate=miss_count / 2.0,
        expected_tardiness_sum=tardiness_sum,
        mean_normalized_tardiness=tardiness_sum / 2.0,
        expected_join_close_sum=close_sum,
        expected_miss_by_request=tuple(misses),
        expected_tardiness_by_request=tuple(tardiness),
        expected_join_close_by_request=tuple(closes),
    )


def fold_four_world_results(results: Sequence[Mapping[str, Any]]) -> PairMetrics:
    """Public denominator-preserving fold for causal baseline policies."""
    return _policy_metrics(results)


def _objective_equal(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(
            a,
            b,
            rel_tol=LEX_REL_TOLERANCE,
            abs_tol=LEX_ABS_TOLERANCE,
        )
        for a, b in zip(left, right)
    )


def _policies(scenario: Scenario, arm: Arm) -> Iterable[Policy]:
    classes = tuple(key for key, _ in observation_partitions(scenario, arm))
    actions = enumerate_actions(scenario)
    for choices in itertools.product(actions, repeat=len(classes)):
        yield tuple(zip(classes, choices))


def _evaluate_policy(scenario: Scenario, arm: Arm, policy: Policy) -> PairMetrics:
    policy_map = dict(policy)
    expected_keys = {key for key, _ in observation_partitions(scenario, arm)}
    if set(policy_map) != expected_keys or len(policy_map) != len(policy):
        raise PhaseMapError("policy does not cover each observation class exactly once")
    results = []
    for index, world in enumerate(scenario.worlds):
        results.append(simulate(scenario, index, policy_map[observation_key(scenario, world, arm)]))
    return _policy_metrics(results)


def enumerate_policy_metrics(scenario: Scenario, arm: Arm) -> tuple[tuple[Policy, PairMetrics], ...]:
    """Materialize the finite policy table for an independent MILP cross-check."""
    _validate_scenario(scenario)
    return tuple((policy, _evaluate_policy(scenario, arm, policy)) for policy in _policies(scenario, arm))


def optimize_arm(scenario: Scenario, arm: Arm) -> dict[str, Any]:
    rows = list(enumerate_policy_metrics(scenario, arm))
    if not rows:
        raise PhaseMapError("empty policy space")
    minima, optimal_rows = lexicographic_optimal_rows(
        [(policy, metrics.objective) for policy, metrics in rows]
    )
    optimal = tuple(sorted(optimal_rows, key=_canonical))
    selected = optimal[0]
    metrics = next(value for policy, value in rows if policy == selected)
    return {
        "arm": arm,
        "observation_class_count": len(observation_partitions(scenario, arm)),
        "policy_count": len(rows),
        "lexicographic_minima": minima,
        "lexicographic_tolerance": {
            "relative": LEX_REL_TOLERANCE,
            "absolute": LEX_ABS_TOLERANCE,
        },
        "metrics": metrics,
        "optimal_policies": optimal,
        "selected_canonical_policy": selected,
        "unique": len(optimal) == 1,
    }


def optimize_full_future_ceiling(scenario: Scenario) -> dict[str, Any]:
    """Enumerate the full-future C arm as a diagnostic-only ceiling."""
    _validate_scenario(scenario)
    actions = enumerate_actions(scenario)
    class_keys = tuple(f"C:{world.world_id}" for world in scenario.worlds)
    rows: list[tuple[Policy, PairMetrics]] = []
    for choices in itertools.product(actions, repeat=len(scenario.worlds)):
        policy = tuple(zip(class_keys, choices))
        results = [
            simulate(scenario, index, choices[index]) for index in range(len(scenario.worlds))
        ]
        rows.append((policy, _policy_metrics(results)))
    minima, optimal_rows = lexicographic_optimal_rows(
        [(policy, metrics.objective) for policy, metrics in rows]
    )
    optimal = tuple(sorted(optimal_rows, key=_canonical))
    selected = optimal[0]
    metrics = next(value for policy, value in rows if policy == selected)
    return {
        "arm": "C",
        "diagnostic_only": True,
        "sees_full_future": True,
        "observation_class_count": len(scenario.worlds),
        "policy_count": len(rows),
        "lexicographic_minima": minima,
        "lexicographic_tolerance": {
            "relative": LEX_REL_TOLERANCE,
            "absolute": LEX_ABS_TOLERANCE,
        },
        "metrics": metrics,
        "optimal_policies": optimal,
        "selected_canonical_policy": selected,
        "unique": len(optimal) == 1,
    }


def _dual_conditioned_flip(scenario: Scenario, r_report: Mapping[str, Any]) -> bool:
    policies = r_report["optimal_policies"]
    if len(policies) != 1:
        return False
    policy_map = dict(policies[0])
    actions = {
        (world.q_bit, world.j_bit): policy_map[observation_key(scenario, world, "R")]
        for world in scenario.worlds
    }
    q_condition = any(actions[(q, 0)] != actions[(q, 1)] for q in (0, 1))
    j_condition = any(actions[(0, j)] != actions[(1, j)] for j in (0, 1))
    return q_condition and j_condition


def optimize_information_lattice(scenario: Scenario) -> dict[str, Any]:
    _validate_scenario(scenario)
    arms = {arm: optimize_arm(scenario, arm) for arm in ("B0", "Q", "J", "R")}
    if scenario.kind == "primary":
        counts = {arm: arms[arm]["observation_class_count"] for arm in arms}
        if counts != {"B0": 1, "Q": 2, "J": 2, "R": 4}:
            raise PhaseMapError("primary information lattice is not 1/2/2/4")
    action_patterns = set()
    for action in enumerate_actions(scenario):
        pattern = []
        for world_index in range(4):
            result = simulate(scenario, world_index, action)
            pattern.append(tuple(bool(row["miss"]) for row in result["requests"]))
        action_patterns.add(tuple(pattern))
    ceiling = optimize_full_future_ceiling(scenario)
    if not lexicographic_no_worse(
        ceiling["lexicographic_minima"], arms["R"]["lexicographic_minima"]
    ):
        raise PhaseMapError("full-future ceiling is worse than the receiver-aware arm")
    return {
        "scenario_id": scenario.scenario_id,
        "kind": scenario.kind,
        "action_count": len(enumerate_actions(scenario)),
        "arms": arms,
        "ceiling": ceiling,
        "actionable": len(action_patterns) > 1,
        "dual_conditioned_strict_interaction_flip": _dual_conditioned_flip(scenario, arms["R"]),
    }


def _aggregate_arm(pair_reports: Sequence[Mapping[str, Any]], arm: str) -> AggregateMetrics:
    misses: dict[str, float] = {}
    tardiness: dict[str, float] = {}
    closes: dict[str, float] = {}
    for report in pair_reports:
        arm_report = report["ceiling"] if arm == "C" else report["arms"][arm]
        policies = arm_report["optimal_policies"]
        if not policies or arm_report["selected_canonical_policy"] != min(policies, key=_canonical):
            raise PhaseMapError("pair arm did not use identity-only canonical tie breaking")
        metrics = arm_report["metrics"]
        if not isinstance(metrics, PairMetrics) or metrics.request_count != 2:
            raise PhaseMapError("pair metrics have the wrong type or denominator")
        for request_id, value in metrics.expected_miss_by_request:
            if request_id in misses or value not in {0.0, 0.25, 0.5, 0.75, 1.0}:
                raise PhaseMapError("request duplicated or worlds were not folded by quarters")
            misses[request_id] = value
        for request_id, value in metrics.expected_tardiness_by_request:
            if request_id in tardiness or not math.isfinite(value) or value < 0:
                raise PhaseMapError("invalid folded request tardiness")
            tardiness[request_id] = value
        for request_id, value in metrics.expected_join_close_by_request:
            if request_id in closes or not math.isfinite(value):
                raise PhaseMapError("invalid folded join-close accounting")
            closes[request_id] = value
    if len(misses) != 32 or set(misses) != set(tardiness) or set(misses) != set(closes):
        raise PhaseMapError("aggregate requires exactly 32 distinct native requests")
    request_ids = sorted(misses)
    z_values = [tardiness[request_id] for request_id in request_ids]
    miss_count = sum(misses.values())
    return AggregateMetrics(
        request_count=32,
        expected_miss_count=miss_count,
        miss_rate=miss_count / 32.0,
        cvar90_normalized_tardiness=empirical_cvar90(z_values),
        mean_normalized_tardiness=sum(z_values) / 32.0,
        expected_join_close_sum=sum(closes.values()),
        expected_miss_by_request=tuple((key, misses[key]) for key in request_ids),
        expected_tardiness_by_request=tuple((key, tardiness[key]) for key in request_ids),
    )


def aggregate_pair_metrics(metrics_rows: Sequence[PairMetrics]) -> AggregateMetrics:
    """Aggregate exactly sixteen pair metrics without treating worlds as samples."""
    if len(metrics_rows) != 16:
        raise PhaseMapError("aggregate requires exactly 16 pair metrics")
    wrappers = []
    for index, metrics in enumerate(metrics_rows):
        empty_policy: Policy = ()
        wrappers.append(
            {
                "scenario_id": f"metric-wrapper-{index:02d}",
                "arms": {
                    "B0": {
                        "metrics": metrics,
                        "optimal_policies": (empty_policy,),
                        "selected_canonical_policy": empty_policy,
                    }
                },
            }
        )
    return _aggregate_arm(wrappers, "B0")


def aggregate_16_pair_reports(pair_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(pair_reports) != 16:
        raise PhaseMapError("aggregate requires exactly 16 fixed pair reports")
    scenario_ids = [report.get("scenario_id") for report in pair_reports]
    if any(not isinstance(value, str) or not value for value in scenario_ids) or len(set(scenario_ids)) != 16:
        raise PhaseMapError("pair scenario identity is missing or duplicated")
    arms = {arm: _aggregate_arm(pair_reports, arm) for arm in ("B0", "Q", "J", "R")}
    ceiling = _aggregate_arm(pair_reports, "C")
    best_single_minima, best_single_rows = lexicographic_optimal_rows(
        [(arm, arms[arm].objective) for arm in ("Q", "J")]
    )
    best_single_name = min(best_single_rows)
    best_single = arms[best_single_name]
    r_metrics = arms["R"]
    absolute = best_single.miss_rate - r_metrics.miss_rate
    relative = absolute / best_single.miss_rate if best_single.miss_rate > 0 else None
    cvar_relative = (
        (best_single.cvar90_normalized_tardiness - r_metrics.cvar90_normalized_tardiness)
        / best_single.cvar90_normalized_tardiness
        if best_single.cvar90_normalized_tardiness > 0
        else None
    )
    actionable_count = sum(report.get("actionable") is True for report in pair_reports)
    flip_count = sum(
        report.get("dual_conditioned_strict_interaction_flip") is True for report in pair_reports
    )
    return {
        "schema_version": "phasemap-four-world-aggregate-v1",
        "pair_count": 16,
        "native_request_count": 32,
        "counterfactual_world_count": 128,
        "worlds_are_folded_not_samples": True,
        "arms": arms,
        "ceiling_C": ceiling,
        "best_single_comparator": BEST_SINGLE_COMPARATOR,
        "best_single_lexicographic_minima": best_single_minima,
        "best_single_arm": best_single_name,
        "absolute_miss_reduction": absolute,
        "relative_miss_reduction": relative,
        "relative_cvar90_reduction": cvar_relative,
        "actionable_pairs": actionable_count,
        "actionable_rate": actionable_count / 16.0,
        "strict_interaction_flip_pairs": flip_count,
        "strict_interaction_flip_rate": flip_count / 16.0,
    }


def validate_control(control_name: str, aggregate: Mapping[str, Any]) -> dict[str, Any]:
    arms = aggregate.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"B0", "Q", "J", "R"}:
        raise PhaseMapError("control aggregate lacks the complete information lattice")

    def equal(left: str, right: str) -> bool:
        a = arms[left]
        b = arms[right]
        if not isinstance(a, AggregateMetrics) or not isinstance(b, AggregateMetrics):
            raise PhaseMapError("control arm metric type mismatch")
        return _objective_equal(a.objective, b.objective) and math.isclose(
            a.cvar90_normalized_tardiness,
            b.cvar90_normalized_tardiness,
            rel_tol=1e-10,
            abs_tol=1e-10,
        )

    if control_name == "equal_q":
        passed = equal("R", "J")
    elif control_name in {"equal_j", "fanout1", "shuffled_key"}:
        passed = equal("R", "Q") and equal("J", "B0")
    elif control_name == "no_conflict":
        passed = all(equal("B0", arm) for arm in ("Q", "J", "R"))
    else:
        raise PhaseMapError("unknown frozen negative control")
    passed = passed and aggregate.get("strict_interaction_flip_rate") == 0.0
    return {"name": control_name, "passed": passed}
