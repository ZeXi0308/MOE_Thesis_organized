#!/usr/bin/env python3
"""Pure event-model oracle for the frozen FJRC pilot.

This module consumes already-materialized tasks, services, deadlines and two
reachable receiver-background worlds.  It deliberately does not read route
artifacts, measure a GPU, select arrival/deadline parameters, or emit a
scientific result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from typing import Any, Iterable, Mapping, Sequence


EPS = 1e-9
PHASES = {"fixed_before", "decision", "fixed_after"}


class FJRCError(RuntimeError):
    """A frozen protocol invariant or accounting identity failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite_nonnegative(value: float, name: str, *, positive: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise FJRCError(f"{name} must be {'positive' if positive else 'non-negative'} and finite")


@dataclass(frozen=True)
class Contribution:
    task_id: str
    join_id: str
    request_id: str
    sender_rank: int
    receiver_rank: int
    release_us: float
    pack_us: float
    cut_us: float
    unpack_us: float
    phase: str
    fixed_ordinal: int = 0


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
class BackgroundJob:
    job_id: str
    receiver_rank: int
    arrival_us: float
    service_us: float


@dataclass(frozen=True)
class World:
    world_id: str
    background_jobs: tuple[BackgroundJob, ...]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    decision_time_us: float
    contributions: tuple[Contribution, ...]
    joins: tuple[Join, ...]
    worlds: tuple[World, World]
    kind: str = "primary"


@dataclass(frozen=True)
class RiskMetrics:
    request_count: int
    expected_miss_count: float
    miss_rate: float
    cvar90_normalized_tardiness: float
    mean_normalized_tardiness: float
    makespan_us: float
    expected_miss_by_request: tuple[tuple[str, float], ...]
    expected_tardiness_by_request: tuple[tuple[str, float], ...]

    @property
    def objective(self) -> tuple[float, float, float, float]:
        return (
            self.expected_miss_count,
            self.cvar90_normalized_tardiness,
            self.mean_normalized_tardiness,
            self.makespan_us,
        )


def empirical_cvar90(values: Sequence[float]) -> float:
    """Mean of the worst ceil(10%) samples (4 samples for frozen n=32)."""
    if not values:
        raise FJRCError("CVaR90 needs at least one request")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise FJRCError("invalid normalized tardiness")
    tail = max(1, math.ceil(0.10 * len(values) - 1e-12))
    return sum(sorted(values, reverse=True)[:tail]) / tail


def _validate_scenario(scenario: Scenario) -> None:
    _finite_nonnegative(scenario.decision_time_us, "decision_time_us")
    if len(scenario.worlds) != 2 or scenario.worlds[0].world_id == scenario.worlds[1].world_id:
        raise FJRCError("exactly two distinctly identified matched worlds are required")
    tasks = {task.task_id: task for task in scenario.contributions}
    joins = {join.join_id: join for join in scenario.joins}
    if len(tasks) != len(scenario.contributions) or len(joins) != len(scenario.joins):
        raise FJRCError("duplicate task or join identity")
    if len({join.request_id for join in scenario.joins}) != len(scenario.joins):
        raise FJRCError("the pilot requires one selected join per request")
    if scenario.kind not in {"primary", "fanout1_control"}:
        raise FJRCError("unknown scenario kind")
    if scenario.kind == "primary" and len(scenario.joins) != 2:
        raise FJRCError("a primary matched pair must contain exactly two requests")
    if scenario.kind == "fanout1_control" and any(
        len(join.sibling_task_ids) != 1 for join in scenario.joins
    ):
        raise FJRCError("fanout1 control contains a multi-sibling join")
    sibling_census: list[str] = []
    for join in scenario.joins:
        _finite_nonnegative(join.combine_us, "combine_us", positive=True)
        if join.deadline_us <= join.request_arrival_us:
            raise FJRCError("deadline must follow request arrival")
        if not join.sibling_task_ids or len(set(join.sibling_task_ids)) != len(join.sibling_task_ids):
            raise FJRCError("join must contain a non-empty unique full sibling set")
        for task_id in join.sibling_task_ids:
            task = tasks.get(task_id)
            if task is None:
                raise FJRCError("join references an absent sibling")
            if (
                task.join_id != join.join_id
                or task.request_id != join.request_id
                or task.receiver_rank != join.receiver_rank
            ):
                raise FJRCError("sibling identity disagrees with its join")
        sibling_census.extend(join.sibling_task_ids)
    if scenario.kind == "primary" and any(len(join.sibling_task_ids) < 2 for join in scenario.joins):
        raise FJRCError("primary FJRC joins must retain fork-join fanout")
    if sorted(sibling_census) != sorted(tasks):
        raise FJRCError("task universe is not exactly the union of full sibling sets")

    decision_by_sender: dict[int, list[Contribution]] = {}
    for task in scenario.contributions:
        if task.phase not in PHASES:
            raise FJRCError("invalid contribution phase")
        for name, value in (
            ("release_us", task.release_us),
            ("pack_us", task.pack_us),
            ("cut_us", task.cut_us),
            ("unpack_us", task.unpack_us),
        ):
            if name == "release_us":
                if not math.isfinite(value):
                    raise FJRCError("release_us must be finite")
            else:
                _finite_nonnegative(value, name, positive=True)
        if task.phase == "decision":
            if task.release_us > scenario.decision_time_us + EPS:
                raise FJRCError("decision contribution is not ready at t0")
            decision_by_sender.setdefault(task.sender_rank, []).append(task)
    minimum_senders = 2 if scenario.kind == "primary" else 1
    if len(decision_by_sender) < minimum_senders:
        raise FJRCError(f"{scenario.kind} requires at least {minimum_senders} common decision sender(s)")
    for sender, pair in decision_by_sender.items():
        if len(pair) != 2 or len({task.request_id for task in pair}) != 2:
            raise FJRCError(f"sender {sender} must have one decision task from each request")
    if scenario.kind == "primary" and len(
        {frozenset(task.request_id for task in pair) for pair in decision_by_sender.values()}
    ) != 1:
        raise FJRCError("all primary common senders must compete on the same request pair")

    allowed_receivers = {task.receiver_rank for task in scenario.contributions}
    world_signatures = []
    for world in scenario.worlds:
        seen: set[str] = set()
        per_receiver: dict[int, list[tuple[str, float, float]]] = {}
        for job in world.background_jobs:
            if job.job_id in seen:
                raise FJRCError("duplicate background job identity")
            seen.add(job.job_id)
            if job.receiver_rank not in allowed_receivers:
                raise FJRCError("background history mapped outside foreground receiver identities")
            if not math.isfinite(job.arrival_us):
                raise FJRCError("background arrival must be finite")
            if job.arrival_us > scenario.decision_time_us + EPS:
                raise FJRCError("receiver background history is not pre-decision causal state")
            _finite_nonnegative(job.service_us, "background service", positive=True)
            per_receiver.setdefault(job.receiver_rank, []).append(
                (job.job_id, job.arrival_us, job.service_us)
            )
        # Histories may move between receiver identities, but their multiset and
        # every background job identity/service/arrival must remain unchanged.
        world_signatures.append(sorted(tuple(sorted(per_receiver.get(receiver, []))) for receiver in allowed_receivers))
    if world_signatures[0] != world_signatures[1]:
        raise FJRCError("matched worlds do not preserve the background-history multiset")


def decision_senders(scenario: Scenario) -> tuple[int, ...]:
    return tuple(sorted({task.sender_rank for task in scenario.contributions if task.phase == "decision"}))


def enumerate_actions(scenario: Scenario) -> tuple[tuple[tuple[int, str], ...], ...]:
    """Enumerate the binary joint first-admission vector across common senders."""
    _validate_scenario(scenario)
    choices = []
    for sender in decision_senders(scenario):
        task_ids = sorted(
            task.task_id
            for task in scenario.contributions
            if task.phase == "decision" and task.sender_rank == sender
        )
        choices.append(tuple((sender, task_id) for task_id in task_ids))
    return tuple(tuple(vector) for vector in itertools.product(*choices))


def _validate_action(scenario: Scenario, action: tuple[tuple[int, str], ...]) -> None:
    action_map = dict(action)
    senders = decision_senders(scenario)
    if len(action_map) != len(action) or tuple(sorted(action_map)) != senders:
        raise FJRCError("action does not cover every decision sender exactly once")
    legal = {
        sender: {
            task.task_id
            for task in scenario.contributions
            if task.phase == "decision" and task.sender_rank == sender
        }
        for sender in senders
    }
    if any(action_map[sender] not in legal[sender] for sender in senders):
        raise FJRCError("action selects a task outside its sender decision pair")


def _sender_orders(
    scenario: Scenario, action: tuple[tuple[int, str], ...]
) -> dict[int, list[Contribution]]:
    _validate_action(scenario, action)
    action_map = dict(action)
    grouped: dict[int, list[Contribution]] = {}
    for task in scenario.contributions:
        grouped.setdefault(task.sender_rank, []).append(task)
    orders: dict[int, list[Contribution]] = {}
    for sender, tasks in grouped.items():
        before = sorted(
            (task for task in tasks if task.phase == "fixed_before"),
            key=lambda task: (task.fixed_ordinal, task.release_us, task.task_id),
        )
        decision = [task for task in tasks if task.phase == "decision"]
        after = sorted(
            (task for task in tasks if task.phase == "fixed_after"),
            key=lambda task: (task.fixed_ordinal, task.release_us, task.task_id),
        )
        if decision:
            first_id = action_map.get(sender)
            if first_id not in {task.task_id for task in decision}:
                raise FJRCError("action selects a task outside its sender decision pair")
            decision.sort(key=lambda task: (task.task_id != first_id, task.task_id))
        elif sender in action_map:
            raise FJRCError("action includes a non-decision sender")
        orders[sender] = before + decision + after
    return orders


def _sender_ledger(
    scenario: Scenario, action: tuple[tuple[int, str], ...]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    arrivals: dict[str, float] = {}
    events: list[dict[str, Any]] = []
    for sender, tasks in sorted(_sender_orders(scenario, action).items()):
        available = -math.inf
        for task in tasks:
            if task.phase == "decision":
                earliest = max(task.release_us, scenario.decision_time_us)
            else:
                earliest = task.release_us
            pack_start = max(earliest, available)
            pack_end = pack_start + task.pack_us
            cut_start = pack_end
            cut_end = cut_start + task.cut_us
            if pack_start + EPS < available:
                raise FJRCError("sender overlap")
            events.append(
                {
                    "task_id": task.task_id,
                    "sender_rank": sender,
                    "pack_start_us": pack_start,
                    "pack_end_us": pack_end,
                    "cut_start_us": cut_start,
                    "cut_end_us": cut_end,
                }
            )
            arrivals[task.task_id] = cut_end
            available = cut_end
    return arrivals, events


def _observation_fingerprint(scenario: Scenario, world: World) -> str:
    """Keyed receiver availability causally present immediately before t0."""
    # fixed_before sender arrivals are action-independent foreground history.
    arbitrary = tuple(
        (
            sender,
            min(
                task.task_id
                for task in scenario.contributions
                if task.phase == "decision" and task.sender_rank == sender
            ),
        )
        for sender in decision_senders(scenario)
    )
    arrivals, _ = _sender_ledger(scenario, arbitrary)
    task_by_id = {task.task_id: task for task in scenario.contributions}
    queued: dict[int, list[tuple[float, str, float]]] = {}
    for job in world.background_jobs:
        if job.arrival_us <= scenario.decision_time_us + EPS:
            queued.setdefault(job.receiver_rank, []).append((job.arrival_us, f"bg:{job.job_id}", job.service_us))
    for task_id, arrival in arrivals.items():
        task = task_by_id[task_id]
        if task.phase == "fixed_before" and arrival <= scenario.decision_time_us + EPS:
            queued.setdefault(task.receiver_rank, []).append((arrival, f"fg:{task_id}", task.unpack_us))
    receiver_set = sorted(
        {task.receiver_rank for task in scenario.contributions}
        | {job.receiver_rank for job in world.background_jobs}
    )
    state = {}
    for receiver in receiver_set:
        available = -math.inf
        unfinished = 0
        for arrival, identity, service in sorted(queued.get(receiver, [])):
            start = max(arrival, available)
            available = start + service
            if available > scenario.decision_time_us + EPS:
                unfinished += 1
        state[str(receiver)] = {
            "available_us": max(scenario.decision_time_us, available),
            "unfinished_jobs": unfinished,
        }
    return _sha(state)


def simulate(
    scenario: Scenario,
    world_index: int,
    action: tuple[tuple[int, str], ...],
) -> dict[str, Any]:
    """Run one world and return a fully auditable stage ledger."""
    _validate_scenario(scenario)
    if world_index not in (0, 1):
        raise FJRCError("world index must be zero or one")
    _validate_action(scenario, action)
    world = scenario.worlds[world_index]
    task_by_id = {task.task_id: task for task in scenario.contributions}
    arrivals, sender_events = _sender_ledger(scenario, action)

    receiver_inputs: dict[int, list[tuple[float, str, str, float]]] = {}
    for job in world.background_jobs:
        receiver_inputs.setdefault(job.receiver_rank, []).append(
            (job.arrival_us, f"bg:{job.job_id}", "background", job.service_us)
        )
    for task_id, arrival in arrivals.items():
        task = task_by_id[task_id]
        receiver_inputs.setdefault(task.receiver_rank, []).append(
            (arrival, f"fg:{task_id}", task_id, task.unpack_us)
        )

    unpack_completion: dict[str, float] = {}
    receiver_events: list[dict[str, Any]] = []
    for receiver, entries in sorted(receiver_inputs.items()):
        available = -math.inf
        for arrival, tie_id, identity, service in sorted(entries, key=lambda row: (row[0], row[1])):
            start = max(arrival, available)
            end = start + service
            if start + EPS < available:
                raise FJRCError("receiver unpack overlap")
            receiver_events.append(
                {
                    "identity": identity,
                    "receiver_rank": receiver,
                    "arrival_us": arrival,
                    "unpack_start_us": start,
                    "unpack_end_us": end,
                    "service_us": service,
                }
            )
            if identity != "background":
                if identity in unpack_completion:
                    raise FJRCError("foreground contribution unpacked twice")
                unpack_completion[identity] = end
            available = end
    if set(unpack_completion) != set(task_by_id):
        raise FJRCError("foreground unpack census mismatch")

    ready_by_receiver: dict[int, list[tuple[float, str, Join]]] = {}
    for join in scenario.joins:
        ready = max(unpack_completion[task_id] for task_id in join.sibling_task_ids)
        ready_by_receiver.setdefault(join.receiver_rank, []).append((ready, join.join_id, join))
    close_by_join: dict[str, float] = {}
    combine_events: list[dict[str, Any]] = []
    for receiver, ready_joins in sorted(ready_by_receiver.items()):
        available = -math.inf
        for ready, _tie, join in sorted(ready_joins, key=lambda row: (row[0], row[1])):
            start = max(ready, available)
            close = start + join.combine_us
            combine_events.append(
                {
                    "join_id": join.join_id,
                    "request_id": join.request_id,
                    "receiver_rank": receiver,
                    "join_ready_us": ready,
                    "combine_start_us": start,
                    "join_close_us": close,
                }
            )
            close_by_join[join.join_id] = close
            available = close

    # Accounting identities: every foreground contribution has one pack/cut,
    # one unpack, and every selected join has exactly one combine.
    if len(sender_events) != len(task_by_id) or {row["task_id"] for row in sender_events} != set(task_by_id):
        raise FJRCError("pack/cut census mismatch")
    if len([row for row in receiver_events if row["identity"] != "background"]) != len(task_by_id):
        raise FJRCError("unpack census mismatch")
    if len(combine_events) != len(scenario.joins) or set(close_by_join) != {join.join_id for join in scenario.joins}:
        raise FJRCError("combine census mismatch")

    requests = []
    for join in sorted(scenario.joins, key=lambda item: item.request_id):
        close = close_by_join[join.join_id]
        budget = join.deadline_us - join.request_arrival_us
        normalized = max(0.0, close - join.deadline_us) / max(budget, EPS)
        requests.append(
            {
                "request_id": join.request_id,
                "join_id": join.join_id,
                "join_close_us": close,
                "deadline_us": join.deadline_us,
                "miss": close > join.deadline_us,
                "normalized_tardiness": normalized,
            }
        )
    return {
        "scenario_id": scenario.scenario_id,
        "world_id": world.world_id,
        "world_index": world_index,
        "action": list(action),
        "receiver_observation_fingerprint": _observation_fingerprint(scenario, world),
        "sender_events": sender_events,
        "receiver_events": receiver_events,
        "combine_events": combine_events,
        "requests": requests,
        "accounting": {
            "foreground_contributions": len(task_by_id),
            "pack_count": len(sender_events),
            "cut_count": len(sender_events),
            "unpack_count": len(task_by_id),
            "combine_count": len(combine_events),
        },
    }


def policy_metrics(results: Sequence[Mapping[str, Any]]) -> RiskMetrics:
    """Average matched-world outcomes per native request, preserving its denominator."""
    if len(results) != 2:
        raise FJRCError("a policy requires exactly one result from each matched world")
    by_request: dict[str, list[Mapping[str, Any]]] = {}
    makespan = -math.inf
    for result in results:
        for row in result["requests"]:
            by_request.setdefault(str(row["request_id"]), []).append(row)
            makespan = max(makespan, float(row["join_close_us"]))
    if not by_request or any(len(rows) != 2 for rows in by_request.values()):
        raise FJRCError("request denominator or matched-world census drift")
    expected_miss = []
    expected_tardiness = []
    for request_id in sorted(by_request):
        rows = by_request[request_id]
        expected_miss.append((request_id, sum(bool(row["miss"]) for row in rows) / 2.0))
        expected_tardiness.append(
            (request_id, sum(float(row["normalized_tardiness"]) for row in rows) / 2.0)
        )
    tardiness_values = [value for _, value in expected_tardiness]
    miss_count = sum(value for _, value in expected_miss)
    return RiskMetrics(
        request_count=len(by_request),
        expected_miss_count=miss_count,
        miss_rate=miss_count / len(by_request),
        cvar90_normalized_tardiness=empirical_cvar90(tardiness_values),
        mean_normalized_tardiness=sum(tardiness_values) / len(tardiness_values),
        makespan_us=makespan,
        expected_miss_by_request=tuple(expected_miss),
        expected_tardiness_by_request=tuple(expected_tardiness),
    )


def _objective_equal(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(left, right))


def _best_policies(
    candidates: Iterable[tuple[tuple[tuple[tuple[int, str], ...], ...], RiskMetrics]]
) -> tuple[
    RiskMetrics,
    tuple[tuple[tuple[tuple[int, str], ...], ...], ...],
    tuple[tuple[tuple[int, str], ...], ...],
]:
    rows = list(candidates)
    if not rows:
        raise FJRCError("empty policy space")
    best_objective = min(metrics.objective for _, metrics in rows)
    optimal = tuple(sorted(
        (policy for policy, metrics in rows if _objective_equal(metrics.objective, best_objective)),
        key=lambda policy: _canonical(policy),
    ))
    selected_policy = optimal[0]
    selected_metrics = next(metrics for policy, metrics in rows if policy == selected_policy)
    return selected_metrics, optimal, selected_policy


def optimize_information_arms(
    scenario: Scenario,
    *,
    observation_labels: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Enumerate exact B and R0 policies at the single joint decision node.

    B always shares one action across worlds.  R0 shares actions only between
    worlds with the same causal keyed-state observation.  ``observation_labels``
    is reserved for frozen negative-control partitions; it changes information,
    never the simulated receiver state.
    """
    actions = enumerate_actions(scenario)
    table = {
        (world, action): simulate(scenario, world, action)
        for world in (0, 1)
        for action in actions
    }
    b_candidates = []
    for action in actions:
        policy = (action, action)
        b_candidates.append((policy, policy_metrics((table[(0, action)], table[(1, action)]))))
    b_metrics, b_optimal, b_selected = _best_policies(b_candidates)

    labels = observation_labels or tuple(
        table[(world, actions[0])]["receiver_observation_fingerprint"] for world in (0, 1)
    )
    if len(labels) != 2:
        raise FJRCError("two observation labels are required")
    r_candidates = []
    for action0 in actions:
        for action1 in actions:
            if labels[0] == labels[1] and action0 != action1:
                continue
            policy = (action0, action1)
            r_candidates.append(
                (policy, policy_metrics((table[(0, action0)], table[(1, action1)])))
            )
    r_metrics, r_optimal, r_selected = _best_policies(r_candidates)
    if r_metrics.objective > b_metrics.objective and not _objective_equal(r_metrics.objective, b_metrics.objective):
        raise FJRCError("more receiver information made the exact optimum worse")

    miss_patterns = {
        tuple(
            bool(row["miss"])
            for world in (0, 1)
            for row in table[(world, action)]["requests"]
        )
        for action in actions
    }
    actionable = len(miss_patterns) > 1
    unique_b = len(b_optimal) == 1
    unique_r = len(r_optimal) == 1
    flip = unique_r and r_optimal[0][0] != r_optimal[0][1]
    absolute_gap = b_metrics.miss_rate - r_metrics.miss_rate
    relative_gap = absolute_gap / b_metrics.miss_rate if b_metrics.miss_rate > 0 else 0.0
    cvar_gap = (
        (b_metrics.cvar90_normalized_tardiness - r_metrics.cvar90_normalized_tardiness)
        / b_metrics.cvar90_normalized_tardiness
        if b_metrics.cvar90_normalized_tardiness > 0
        else 0.0
    )
    return {
        "scenario_id": scenario.scenario_id,
        "action_count": len(actions),
        "actions": actions,
        "observation_labels": labels,
        "B": {
            "metrics": b_metrics,
            "optimal_policies": b_optimal,
            "selected_canonical_policy": b_selected,
            "unique": unique_b,
        },
        "R0": {
            "metrics": r_metrics,
            "optimal_policies": r_optimal,
            "selected_canonical_policy": r_selected,
            "unique": unique_r,
        },
        "absolute_miss_rate_gap": absolute_gap,
        "relative_miss_rate_reduction": relative_gap,
        "relative_cvar90_reduction": cvar_gap,
        "actionable": actionable,
        "unique_optimal_joint_action_flip": flip,
    }


def equal_map_negative_control(scenario: Scenario) -> dict[str, Any]:
    """Replay world zero twice; keyed information must then have zero value."""
    world0 = scenario.worlds[0]
    controlled = replace(
        scenario,
        scenario_id=f"{scenario.scenario_id}:equal-map",
        worlds=(world0, replace(world0, world_id=f"{world0.world_id}:copy")),
    )
    report = optimize_information_arms(controlled)
    passed = (
        abs(report["absolute_miss_rate_gap"]) <= EPS
        and abs(report["relative_cvar90_reduction"]) <= EPS
        and report["unique_optimal_joint_action_flip"] is False
    )
    return {"name": "equal_map", "passed": passed, "report": report}


def uninformative_key_negative_control(scenario: Scenario) -> dict[str, Any]:
    """Destroy keyed identity by assigning both worlds one observation label."""
    report = optimize_information_arms(scenario, observation_labels=("UNKEYED", "UNKEYED"))
    passed = (
        abs(report["absolute_miss_rate_gap"]) <= EPS
        and abs(report["relative_cvar90_reduction"]) <= EPS
        and report["unique_optimal_joint_action_flip"] is False
    )
    return {"name": "shuffled_credit_key_uninformative_partition", "passed": passed, "report": report}


def fanout1_negative_control(scenario: Scenario) -> dict[str, Any]:
    """Evaluate an explicit fanout-one control scenario without changing its task universe."""
    if scenario.kind != "fanout1_control":
        raise FJRCError("fanout1 negative control requires an explicit fanout1_control scenario")
    report = optimize_information_arms(scenario)
    passed = (
        abs(report["absolute_miss_rate_gap"]) <= EPS
        and abs(report["relative_cvar90_reduction"]) <= EPS
        and report["unique_optimal_joint_action_flip"] is False
    )
    return {"name": "fanout1", "passed": passed, "report": report}


def _unwrap_pair_report(value: Mapping[str, Any]) -> Mapping[str, Any]:
    report = value.get("report", value)
    if not isinstance(report, Mapping):
        raise FJRCError("pair report wrapper is malformed")
    return report


def _validate_canonical_arm(report: Mapping[str, Any], arm: str) -> RiskMetrics:
    value = report.get(arm)
    if not isinstance(value, Mapping):
        raise FJRCError(f"pair report lacks {arm} arm")
    policies = value.get("optimal_policies")
    selected = value.get("selected_canonical_policy")
    metrics = value.get("metrics")
    if not isinstance(policies, tuple) or not policies:
        raise FJRCError(f"{arm} optimal-policy set is missing")
    canonical = min(policies, key=lambda policy: _canonical(policy))
    if selected != canonical:
        raise FJRCError(f"{arm} selected policy is not canonical-minimum")
    if value.get("unique") is not (len(policies) == 1):
        raise FJRCError(f"{arm} uniqueness flag disagrees with the pre-canonical set")
    if not isinstance(metrics, RiskMetrics):
        raise FJRCError(f"{arm} metrics have the wrong type")
    if metrics.request_count != 2:
        raise FJRCError("each fixed matched pair must contain two native requests")
    return metrics


def _aggregate_arm_metrics(metrics_rows: Sequence[RiskMetrics]) -> RiskMetrics:
    if len(metrics_rows) != 16:
        raise FJRCError("aggregate requires exactly 16 fixed pair reports")
    misses: dict[str, float] = {}
    tardiness: dict[str, float] = {}
    makespan = -math.inf
    for metrics in metrics_rows:
        if metrics.request_count != 2:
            raise FJRCError("pair request denominator drift")
        if {key for key, _ in metrics.expected_miss_by_request} != {
            key for key, _ in metrics.expected_tardiness_by_request
        }:
            raise FJRCError("miss/tardiness request identity mismatch")
        for request_id, value in metrics.expected_miss_by_request:
            if request_id in misses or not math.isfinite(value) or value not in {0.0, 0.5, 1.0}:
                raise FJRCError("duplicate request or invalid two-world miss probability")
            misses[request_id] = value
        for request_id, value in metrics.expected_tardiness_by_request:
            if request_id in tardiness or not math.isfinite(value) or value < 0:
                raise FJRCError("duplicate request or invalid folded tardiness")
            tardiness[request_id] = value
        if not math.isfinite(metrics.makespan_us):
            raise FJRCError("invalid pair makespan")
        makespan = max(makespan, metrics.makespan_us)
    if len(misses) != 32 or set(misses) != set(tardiness):
        raise FJRCError("aggregate denominator is not 32 distinct native requests")
    request_ids = sorted(misses)
    miss_count = sum(misses.values())
    z_values = [tardiness[request_id] for request_id in request_ids]
    return RiskMetrics(
        request_count=32,
        expected_miss_count=miss_count,
        miss_rate=miss_count / 32.0,
        cvar90_normalized_tardiness=empirical_cvar90(z_values),
        mean_normalized_tardiness=sum(z_values) / 32.0,
        makespan_us=makespan,
        expected_miss_by_request=tuple((request_id, misses[request_id]) for request_id in request_ids),
        expected_tardiness_by_request=tuple((request_id, tardiness[request_id]) for request_id in request_ids),
    )


def _aggregate_report_set(pair_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(pair_reports) != 16:
        raise FJRCError("aggregate requires exactly 16 fixed pair reports")
    reports = [_unwrap_pair_report(value) for value in pair_reports]
    scenario_ids = [report.get("scenario_id") for report in reports]
    if any(not isinstance(value, str) or not value for value in scenario_ids) or len(set(scenario_ids)) != 16:
        raise FJRCError("pair scenario identity is missing or duplicated")
    b_rows = [_validate_canonical_arm(report, "B") for report in reports]
    r_rows = [_validate_canonical_arm(report, "R0") for report in reports]
    for b_metrics, r_metrics in zip(b_rows, r_rows):
        if {key for key, _ in b_metrics.expected_miss_by_request} != {
            key for key, _ in r_metrics.expected_miss_by_request
        }:
            raise FJRCError("B/R0 changed the native request universe")
    b_aggregate = _aggregate_arm_metrics(b_rows)
    r_aggregate = _aggregate_arm_metrics(r_rows)
    actionable_count = sum(report.get("actionable") is True for report in reports)
    flip_count = 0
    for report in reports:
        r0 = report["R0"]
        strict_flip = (
            len(r0["optimal_policies"]) == 1
            and r0["optimal_policies"][0][0] != r0["optimal_policies"][0][1]
        )
        if report.get("unique_optimal_joint_action_flip") is not strict_flip:
            raise FJRCError("flip flag disagrees with the pre-canonical optimal set")
        flip_count += strict_flip
    absolute = b_aggregate.miss_rate - r_aggregate.miss_rate
    relative = absolute / b_aggregate.miss_rate if b_aggregate.miss_rate > 0 else None
    cvar_absolute = (
        b_aggregate.cvar90_normalized_tardiness - r_aggregate.cvar90_normalized_tardiness
    )
    cvar_relative = (
        cvar_absolute / b_aggregate.cvar90_normalized_tardiness
        if b_aggregate.cvar90_normalized_tardiness > 0 else None
    )
    return {
        "pair_count": 16,
        "native_request_count": 32,
        "worlds_are_folded_not_samples": True,
        "B": b_aggregate,
        "R0": r_aggregate,
        "absolute_miss_reduction": absolute,
        "relative_miss_reduction": relative,
        "relative_cvar90_reduction": cvar_relative,
        "actionable_pairs": actionable_count,
        "actionable_rate": actionable_count / 16.0,
        "strict_unique_flip_pairs": flip_count,
        "strict_unique_flip_rate": flip_count / 16.0,
    }


def aggregate_16_pair_reports(
    pair_reports: Sequence[Mapping[str, Any]],
    *,
    negative_controls: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate one model's frozen 16 pairs without expanding two worlds to 64 samples."""
    main = _aggregate_report_set(pair_reports)
    required_controls = {"equal_map", "fanout1", "shuffled_uninformative_key"}
    if set(negative_controls) != required_controls:
        raise FJRCError("all and only the three frozen negative controls are required")
    controls: dict[str, Any] = {}
    for name in sorted(required_controls):
        aggregate = _aggregate_report_set(negative_controls[name])
        passed = (
            abs(aggregate["absolute_miss_reduction"]) <= EPS
            and (
                aggregate["B"].cvar90_normalized_tardiness
                - aggregate["R0"].cvar90_normalized_tardiness
            ) == 0.0
            and aggregate["strict_unique_flip_rate"] == 0.0
        )
        controls[name] = {"passed": passed, "aggregate": aggregate}

    gates = {
        "relative_miss_reduction": (
            main["relative_miss_reduction"] is not None
            and main["relative_miss_reduction"] >= 0.10
        ),
        "absolute_miss_reduction": main["absolute_miss_reduction"] >= 0.02,
        "relative_cvar90_reduction": (
            main["relative_cvar90_reduction"] is not None
            and main["relative_cvar90_reduction"] >= 0.05
        ),
        "actionable_rate": main["actionable_rate"] >= 0.50,
        "strict_unique_flip_rate": main["strict_unique_flip_rate"] >= 0.25,
        "negative_controls": all(value["passed"] for value in controls.values()),
    }
    return {
        "schema_version": "fjrc-aggregate-16-pairs-v1",
        "status": "EXPLORATORY_ORACLE_AGGREGATE_NOT_SCIENTIFIC_RESULT",
        "scientific_result": False,
        "main": main,
        "negative_controls": controls,
        "within_model_oracle_gates": gates,
        "within_model_oracle_gates_pass": all(gates.values()),
        "remaining_runner_gates": ["two_model_AND", "best_simple_baseline_capture_lt_90pct"],
    }
