from __future__ import annotations

"""Development-only exact full request-DAG pilot for FrontierCredit-MoE.

This module deliberately lives beside, rather than inside, the frozen BCRD
Gate-2 implementation.  It tests one narrow mechanism question on deterministic
CPU fixtures: can a policy that observes the *currently revealed* top-k join
frontier choose whole-ready-queue flush/hold actions better than queue-local
baselines?  It is simulation-only and never emits a formal serving verdict.
"""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from typing import Mapping, Protocol, Sequence

try:
    from .core import (
        Contribution,
        CurvePoint,
        ProtocolError,
        ServiceCatalog,
        validate_causal_route_v3,
    )
except ImportError:
    from core import (  # type: ignore
        Contribution,
        CurvePoint,
        ProtocolError,
        ServiceCatalog,
        validate_causal_route_v3,
    )


EPS = 1e-12
MODEL = "frontiercredit-pilot"
SIMPLE_POLICY_NAMES = (
    "immediate",
    "edf",
    "max_rows",
    "queue_local_credit",
)
REFERENCE_SIMPLE_POLICY = "queue_local_credit"
DECISION_THRESHOLDS = {
    "minimum_eligible_cells": 2,
    "minimum_identity_sham_applicable_eligible_cells": 2,
    "maximum_fixed_simple_median_capture": 0.90,
    "minimum_frontier_increment_vs_reference": 0.10,
    "minimum_frontier_identity_gap": 0.10,
    "deadline_miss_delta_must_be_nonpositive": True,
}

NodeId = str
GroupKey = tuple[str, int, int]  # request_id, decode_step, layer_id
QueueKey = tuple[int, int, int]  # target_replica, layer_id, expert_id


@dataclass(frozen=True, order=True)
class RequestSpec:
    request_id: str
    arrival_us: float
    deadline_us: float

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not math.isfinite(self.arrival_us) or self.arrival_us < 0:
            raise ValueError("arrival_us must be finite and non-negative")
        if not math.isfinite(self.deadline_us) or self.deadline_us < self.arrival_us:
            raise ValueError("deadline_us must be finite and >= arrival_us")


@dataclass(frozen=True, order=True)
class NodeSpec:
    node_id: NodeId
    request_id: str
    decode_step: int
    layer_id: int
    topk_slot: int
    expert_id: int
    target_replica: int
    deadline_us: float

    @property
    def group_key(self) -> GroupKey:
        return (self.request_id, self.decode_step, self.layer_id)

    @property
    def queue_key(self) -> QueueKey:
        return (self.target_replica, self.layer_id, self.expert_id)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    requests: tuple[RequestSpec, ...]
    contributions: tuple[Contribution, ...]
    nodes: tuple[NodeSpec, ...]
    replicas: int = 2
    decode_steps: int = 2
    layers: int = 2
    top_k: int = 2
    tick_us: float = 1.0
    max_hold_us: float = 3.0
    combine_us: float = 1.0
    launch_cost_us: float = 0.0

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.replicas <= 0 or self.decode_steps <= 0 or self.layers <= 0 or self.top_k <= 0:
            raise ValueError("episode dimensions must be positive")
        if self.tick_us <= 0 or self.max_hold_us <= 0 or self.combine_us < 0:
            raise ValueError("invalid episode timing")
        if self.launch_cost_us < 0:
            raise ValueError("launch_cost_us must be non-negative")
        if len({request.request_id for request in self.requests}) != len(self.requests):
            raise ValueError("request ids must be unique")
        if len({node.node_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("node ids must be unique")
        request_ids = {request.request_id for request in self.requests}
        if {node.request_id for node in self.nodes} != request_ids:
            raise ValueError("every request must have nodes and no unknown request is allowed")
        expected_groups = {
            (request.request_id, step, layer)
            for request in self.requests
            for step in range(self.decode_steps)
            for layer in range(self.layers)
        }
        grouped: dict[GroupKey, list[NodeSpec]] = {}
        for node in self.nodes:
            if not 0 <= node.target_replica < self.replicas:
                raise ValueError("target replica is outside the fixed executor set")
            grouped.setdefault(node.group_key, []).append(node)
        if set(grouped) != expected_groups:
            raise ValueError("episode does not contain the complete request DAG")
        for key, members in grouped.items():
            if sorted(node.topk_slot for node in members) != list(range(self.top_k)):
                raise ValueError(f"group {key} does not contain exactly one of each top-k slot")

    @property
    def request_map(self) -> dict[str, RequestSpec]:
        return {request.request_id: request for request in self.requests}

    @property
    def node_map(self) -> dict[NodeId, NodeSpec]:
        return {node.node_id: node for node in self.nodes}

    @property
    def group_map(self) -> dict[GroupKey, tuple[NodeId, ...]]:
        grouped: dict[GroupKey, list[NodeId]] = {}
        for node in self.nodes:
            grouped.setdefault(node.group_key, []).append(node.node_id)
        return {key: tuple(sorted(values)) for key, values in grouped.items()}


@dataclass(frozen=True, order=True)
class ReadyNode:
    node_id: NodeId
    ready_since_us: float


@dataclass(frozen=True)
class RunningBatch:
    queue_key: QueueKey
    node_ids: tuple[NodeId, ...]
    start_us: float
    finish_us: float
    service_us: float


@dataclass(frozen=True, order=True)
class PendingRelease:
    release_us: float
    group_key: GroupKey


@dataclass(frozen=True)
class SimulationState:
    now_us: float
    arrived_requests: tuple[str, ...] = ()
    revealed_at: tuple[tuple[NodeId, float], ...] = ()
    ready: tuple[ReadyNode, ...] = ()
    running: tuple[RunningBatch | None, ...] = ()
    completed_at: tuple[tuple[NodeId, float], ...] = ()
    joined_groups: tuple[GroupKey, ...] = ()
    pending_releases: tuple[PendingRelease, ...] = ()
    request_completed_at: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, order=True)
class QueueView:
    queue_key: QueueKey
    rows: int
    ready_since_us: float
    min_deadline_us: float
    service_us: float
    singleton_service_us: float
    batch_saving_us: float


@dataclass(frozen=True, order=True)
class FrontierMemberView:
    observed_group_key: GroupKey
    queue_key: QueueKey | None
    status: str
    predicted_finish_us: float


@dataclass(frozen=True, order=True)
class FrontierView:
    observed_group_key: GroupKey
    deadline_us: float
    members: tuple[FrontierMemberView, ...]
    complete_observation: bool


@dataclass(frozen=True)
class DecisionView:
    now_us: float
    tick_us: float
    max_hold_us: float
    combine_us: float
    hold_allowed: bool
    queues: tuple[QueueView, ...]
    frontiers: tuple[FrontierView, ...]


class QueuePolicy(Protocol):
    name: str

    def choose(self, view: DecisionView) -> QueueKey | None:
        """Return a whole-ready queue to flush, or None for one bounded hold."""


@dataclass(frozen=True)
class ImmediatePolicy:
    name: str = "immediate"

    def choose(self, view: DecisionView) -> QueueKey | None:
        return min(view.queues, key=lambda q: (q.ready_since_us, q.queue_key)).queue_key


@dataclass(frozen=True)
class EDFPolicy:
    name: str = "edf"

    def choose(self, view: DecisionView) -> QueueKey | None:
        return min(
            view.queues,
            key=lambda q: (q.min_deadline_us, q.ready_since_us, q.queue_key),
        ).queue_key


def _must_launch(view: DecisionView, queue: QueueView) -> bool:
    age = view.now_us - queue.ready_since_us
    urgent = (
        queue.min_deadline_us - view.now_us
        <= queue.service_us + view.combine_us + view.tick_us + EPS
    )
    return queue.rows >= 2 or age + EPS >= view.max_hold_us or urgent


@dataclass(frozen=True)
class MaxRowsPolicy:
    name: str = "max_rows"

    def choose(self, view: DecisionView) -> QueueKey | None:
        if view.hold_allowed and not any(_must_launch(view, q) for q in view.queues):
            return None
        return max(
            view.queues,
            key=lambda q: (
                q.rows,
                view.now_us - q.ready_since_us,
                -q.min_deadline_us,
                tuple(-part for part in q.queue_key),
            ),
        ).queue_key


@dataclass(frozen=True)
class QueueLocalCreditPolicy:
    name: str = "queue_local_credit"

    def choose(self, view: DecisionView) -> QueueKey | None:
        if view.hold_allowed and not any(_must_launch(view, q) for q in view.queues):
            return None
        return min(
            view.queues,
            key=lambda q: (
                -q.batch_saving_us,
                -q.rows,
                q.min_deadline_us,
                q.ready_since_us,
                q.queue_key,
            ),
        ).queue_key


@dataclass(frozen=True)
class FrontierCreditPolicy:
    name: str = "frontier_credit"

    def choose(self, view: DecisionView) -> QueueKey | None:
        if view.hold_allowed and not any(_must_launch(view, q) for q in view.queues):
            return None

        def rank(queue: QueueView) -> tuple[float, float, str, int, QueueKey]:
            score = 0.0
            priority_deadline = math.inf
            priority_group = "~"
            for frontier in view.frontiers:
                if not frontier.complete_observation:
                    continue
                selected = [
                    member
                    for member in frontier.members
                    if member.queue_key == queue.queue_key and member.status == "ready"
                ]
                if not selected:
                    continue
                current_frontier = max(member.predicted_finish_us for member in frontier.members)
                delayed_frontier = max(
                    member.predicted_finish_us + view.tick_us
                    if member.queue_key == queue.queue_key and member.status == "ready"
                    else member.predicted_finish_us
                    for member in frontier.members
                )
                advance = max(delayed_frontier - current_frontier, 0.0)
                if advance <= EPS:
                    continue
                slack = max(frontier.deadline_us - current_frontier, view.tick_us)
                score += advance / slack
                marker = str(frontier.observed_group_key)
                if (frontier.deadline_us, marker) < (priority_deadline, priority_group):
                    priority_deadline, priority_group = frontier.deadline_us, marker
            normalized = score / max(queue.service_us, EPS)
            return (-normalized, priority_deadline, priority_group, -queue.rows, queue.queue_key)

        return min(view.queues, key=rank).queue_key


@dataclass(frozen=True)
class OracleTail:
    flow_us: float
    total_tardiness_us: float
    deadline_misses: int
    launches: int
    total_service_us: float
    request_completion_us: tuple[tuple[str, float], ...]
    actions: tuple[str, ...]

    @property
    def objective(self) -> tuple[object, ...]:
        return (
            self.flow_us,
            self.total_tardiness_us,
            self.deadline_misses,
            self.launches,
            self.total_service_us,
            self.actions,
        )


def _sorted_pairs(values: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(key), float(value)) for key, value in values.items()))


def _initial_state(episode: Episode) -> SimulationState:
    first = min(request.arrival_us for request in episode.requests)
    return SimulationState(now_us=first, running=tuple(None for _ in range(episode.replicas)))


def _reveal_group(
    episode: Episode,
    group_key: GroupKey,
    now_us: float,
    revealed: dict[NodeId, float],
    ready: dict[NodeId, float],
) -> None:
    for node_id in episode.group_map[group_key]:
        if node_id in revealed:
            raise ProtocolError(f"node {node_id} was revealed more than once")
        revealed[node_id] = now_us
        ready[node_id] = now_us


def _settle(episode: Episode, state: SimulationState) -> SimulationState:
    """Process every event at state.now_us using one deterministic order."""

    while True:
        changed = False
        now = state.now_us
        arrived = set(state.arrived_requests)
        revealed = dict(state.revealed_at)
        ready = {item.node_id: item.ready_since_us for item in state.ready}
        running = list(state.running)
        completed = dict(state.completed_at)
        joined = set(state.joined_groups)
        releases = list(state.pending_releases)
        request_completed = dict(state.request_completed_at)

        # Match the existing causal engine: executor finishes precede same-time
        # releases and arrivals.
        for replica, batch in enumerate(running):
            if batch is None or batch.finish_us > now + EPS:
                continue
            for node_id in batch.node_ids:
                if node_id in completed:
                    raise ProtocolError(f"node {node_id} completed more than once")
                completed[node_id] = now
            running[replica] = None
            changed = True

        due_releases = [release for release in releases if release.release_us <= now + EPS]
        if due_releases:
            releases = [release for release in releases if release not in due_releases]
            for release in sorted(due_releases):
                request_id, step, layer = release.group_key
                if layer + 1 < episode.layers:
                    successor = (request_id, step, layer + 1)
                    _reveal_group(episode, successor, now, revealed, ready)
                elif step + 1 < episode.decode_steps:
                    successor = (request_id, step + 1, 0)
                    _reveal_group(episode, successor, now, revealed, ready)
                else:
                    if request_id in request_completed:
                        raise ProtocolError(f"request {request_id} completed more than once")
                    request_completed[request_id] = now
                changed = True

        request_map = episode.request_map
        due_arrivals = [
            request
            for request in episode.requests
            if request.request_id not in arrived and request.arrival_us <= now + EPS
        ]
        for request in sorted(due_arrivals):
            arrived.add(request.request_id)
            _reveal_group(
                episode,
                (request.request_id, 0, 0),
                now,
                revealed,
                ready,
            )
            changed = True

        pending_groups = {release.group_key for release in releases}
        for group_key, node_ids in episode.group_map.items():
            if group_key in joined or group_key in pending_groups:
                continue
            if all(node_id in completed for node_id in node_ids):
                joined.add(group_key)
                releases.append(PendingRelease(now + episode.combine_us, group_key))
                changed = True

        next_state = SimulationState(
            now_us=now,
            arrived_requests=tuple(sorted(arrived)),
            revealed_at=_sorted_pairs(revealed),
            ready=tuple(ReadyNode(node_id, ready[node_id]) for node_id in sorted(ready)),
            running=tuple(running),
            completed_at=_sorted_pairs(completed),
            joined_groups=tuple(sorted(joined)),
            pending_releases=tuple(sorted(releases)),
            request_completed_at=_sorted_pairs(request_completed),
        )
        if not changed:
            return next_state
        state = next_state


def _ready_queues(episode: Episode, state: SimulationState) -> dict[QueueKey, tuple[ReadyNode, ...]]:
    node_map = episode.node_map
    grouped: dict[QueueKey, list[ReadyNode]] = {}
    for item in state.ready:
        grouped.setdefault(node_map[item.node_id].queue_key, []).append(item)
    return {key: tuple(sorted(values)) for key, values in grouped.items()}


def _eligible_queues(episode: Episode, state: SimulationState) -> dict[QueueKey, tuple[ReadyNode, ...]]:
    return {
        key: values
        for key, values in _ready_queues(episode, state).items()
        if state.running[key[0]] is None
    }


def _decision_queues(episode: Episode, state: SimulationState) -> dict[QueueKey, tuple[ReadyNode, ...]]:
    """Return the complete declared action set across every idle executor.

    Do not prune higher-numbered replicas.  Launching there while leaving a
    lower-numbered replica idle can interact with a subsequent bounded hold, so
    the two choice orders are not equivalent for every legal schedule.
    """

    return _eligible_queues(episode, state)


def _next_external_event(episode: Episode, state: SimulationState) -> float | None:
    values: list[float] = []
    arrived = set(state.arrived_requests)
    values.extend(
        request.arrival_us
        for request in episode.requests
        if request.request_id not in arrived and request.arrival_us > state.now_us + EPS
    )
    values.extend(
        batch.finish_us
        for batch in state.running
        if batch is not None and batch.finish_us > state.now_us + EPS
    )
    values.extend(
        release.release_us
        for release in state.pending_releases
        if release.release_us > state.now_us + EPS
    )
    return min(values) if values else None


def _hold_allowed(episode: Episode, state: SimulationState) -> bool:
    eligible = _eligible_queues(episode, state)
    return bool(eligible) and all(
        state.now_us - min(item.ready_since_us for item in items) < episode.max_hold_us - EPS
        for items in eligible.values()
    )


def _advance_without_action(episode: Episode, state: SimulationState) -> SimulationState:
    next_time = _next_external_event(episode, state)
    if next_time is None:
        raise ProtocolError("INVALID_DAG_DEADLOCK: no legal action or future event")
    return _settle(episode, replace(state, now_us=next_time))


def _hold_once(episode: Episode, state: SimulationState) -> SimulationState:
    if not _hold_allowed(episode, state):
        raise ProtocolError("bounded hold is illegal after a queue timeout")
    candidates: list[float] = []
    external = _next_external_event(episode, state)
    if external is not None:
        candidates.append(external)
    for items in _eligible_queues(episode, state).values():
        candidates.append(min(item.ready_since_us for item in items) + episode.max_hold_us)
    future = [value for value in candidates if value > state.now_us + EPS]
    if not future:
        raise ProtocolError("hold failed to advance time")
    return _settle(episode, replace(state, now_us=min(future)))


def _flush_queue(
    episode: Episode,
    state: SimulationState,
    catalog: ServiceCatalog,
    queue_key: QueueKey,
) -> tuple[SimulationState, float, tuple[NodeId, ...]]:
    eligible = _eligible_queues(episode, state)
    if queue_key not in eligible:
        raise ProtocolError(f"queue {queue_key} is not a legal whole-ready flush")
    replica, layer, _expert = queue_key
    items = eligible[queue_key]
    node_ids = tuple(item.node_id for item in items)
    service_us = catalog.estimate_us(MODEL, layer, len(node_ids))
    ready = tuple(item for item in state.ready if item.node_id not in set(node_ids))
    running = list(state.running)
    if running[replica] is not None:
        raise ProtocolError("flush selected a busy executor")
    start = state.now_us + episode.launch_cost_us
    running[replica] = RunningBatch(
        queue_key=queue_key,
        node_ids=node_ids,
        start_us=start,
        finish_us=start + service_us,
        service_us=service_us,
    )
    return replace(state, ready=ready, running=tuple(running)), service_us, node_ids


def _terminal(episode: Episode, state: SimulationState) -> bool:
    return len(state.request_completed_at) == len(episode.requests)


def _metrics(episode: Episode, state: SimulationState) -> dict[str, object]:
    if not _terminal(episode, state):
        raise ProtocolError("metrics require a complete request DAG")
    completion = dict(state.request_completed_at)
    requests = episode.request_map
    flow = sum(completion[key] - requests[key].arrival_us for key in completion)
    tardiness = sum(max(0.0, completion[key] - requests[key].deadline_us) for key in completion)
    misses = sum(completion[key] > requests[key].deadline_us + EPS for key in completion)
    completed_nodes = dict(state.completed_at)
    revealed = dict(state.revealed_at)
    expected = set(episode.node_map)
    if set(completed_nodes) != expected or set(revealed) != expected:
        raise ProtocolError("full-DAG replay did not reveal and complete every node exactly once")
    return {
        "flow_us": flow,
        "total_tardiness_us": tardiness,
        "deadline_misses": misses,
        "request_completion_us": completion,
        "node_completion_us": completed_nodes,
        "node_revealed_us": revealed,
    }


def _observation_map(
    episode: Episode,
    state: SimulationState,
    *,
    sham: bool,
) -> dict[NodeId, GroupKey]:
    """Build a causally visible sibling-identity map.

    The sham rotates slot-1 membership only among request groups that are
    already revealed at the same decode-step/layer.  With fewer than two such
    groups it is intentionally a no-op: inventing a second real request label
    would leak a future arrival.
    """

    node_map = episode.node_map
    joined = set(state.joined_groups)
    revealed_nodes = tuple(
        sorted(
            node_id
            for node_id in dict(state.revealed_at)
            if node_map[node_id].group_key not in joined
        )
    )
    if not sham:
        return {node_id: node_map[node_id].group_key for node_id in revealed_nodes}

    visible_by_stage: dict[tuple[int, int], list[str]] = {}
    for node_id in revealed_nodes:
        node = node_map[node_id]
        visible_by_stage.setdefault((node.decode_step, node.layer_id), []).append(
            node.request_id
        )
    visible_by_stage = {
        stage: sorted(set(request_ids))
        for stage, request_ids in visible_by_stage.items()
    }

    observed: dict[NodeId, GroupKey] = {}
    for node_id in revealed_nodes:
        node = node_map[node_id]
        cohort = visible_by_stage[(node.decode_step, node.layer_id)]
        observed_request = node.request_id
        if node.topk_slot == 1 and len(cohort) >= 2:
            current = cohort.index(node.request_id)
            observed_request = cohort[(current + 1) % len(cohort)]
        observed[node_id] = (observed_request, node.decode_step, node.layer_id)
    visible_request_ids = {node_map[node_id].request_id for node_id in revealed_nodes}
    if not {group_key[0] for group_key in observed.values()} <= visible_request_ids:
        raise ProtocolError("identity sham exposed an unrevealed request identity")
    return observed


def _build_view(
    episode: Episode,
    state: SimulationState,
    catalog: ServiceCatalog,
    *,
    sham: bool,
) -> DecisionView:
    node_map = episode.node_map
    all_queue_nodes = _eligible_queues(episode, state)
    decision_queue_keys = set(_decision_queues(episode, state))
    all_queue_views: list[QueueView] = []
    for queue_key, items in sorted(all_queue_nodes.items()):
        _replica, layer, _expert = queue_key
        rows = len(items)
        service = catalog.estimate_us(MODEL, layer, rows)
        singleton = catalog.estimate_us(MODEL, layer, 1)
        all_queue_views.append(
            QueueView(
                queue_key=queue_key,
                rows=rows,
                ready_since_us=min(item.ready_since_us for item in items),
                min_deadline_us=min(node_map[item.node_id].deadline_us for item in items),
                service_us=service,
                singleton_service_us=singleton,
                batch_saving_us=max(rows * singleton - service, 0.0),
            )
        )

    ready_lookup = {item.node_id: item for item in state.ready}
    completed = dict(state.completed_at)
    joined = set(state.joined_groups)
    running_by_node: dict[NodeId, RunningBatch] = {}
    for batch in state.running:
        if batch is not None:
            for node_id in batch.node_ids:
                running_by_node[node_id] = batch
    queue_view_map = {queue.queue_key: queue for queue in all_queue_views}
    observed = _observation_map(episode, state, sham=sham)
    grouped_members: dict[GroupKey, list[FrontierMemberView]] = {}
    grouped_deadlines: dict[GroupKey, list[float]] = {}
    revealed_nodes = set(dict(state.revealed_at))
    for node_id in sorted(revealed_nodes):
        node = node_map[node_id]
        if node.group_key in joined:
            continue
        observed_key = observed[node_id]
        if node_id in completed:
            member = FrontierMemberView(observed_key, None, "completed", completed[node_id])
        elif node_id in running_by_node:
            batch = running_by_node[node_id]
            member = FrontierMemberView(observed_key, batch.queue_key, "running", batch.finish_us)
        elif node_id in ready_lookup:
            queue_key = node.queue_key
            if queue_key in queue_view_map:
                predicted = state.now_us + episode.launch_cost_us + queue_view_map[queue_key].service_us
            else:
                running = state.running[queue_key[0]]
                if running is None:
                    raise ProtocolError("ready node disappeared from an idle eligible queue")
                rows = len(_ready_queues(episode, state)[queue_key])
                predicted = (
                    running.finish_us
                    + episode.launch_cost_us
                    + catalog.estimate_us(MODEL, node.layer_id, rows)
                )
            member = FrontierMemberView(observed_key, queue_key, "ready", predicted)
        else:
            raise ProtocolError("revealed active node is neither ready, running, nor completed")
        grouped_members.setdefault(observed_key, []).append(member)
        grouped_deadlines.setdefault(observed_key, []).append(node.deadline_us)

    frontiers = tuple(
        FrontierView(
            observed_group_key=key,
            deadline_us=min(grouped_deadlines[key]),
            members=tuple(
                sorted(
                    members,
                    key=lambda member: (
                        member.observed_group_key,
                        member.status,
                        (-1, -1, -1) if member.queue_key is None else member.queue_key,
                        member.predicted_finish_us,
                    ),
                )
            ),
            complete_observation=len(members) == episode.top_k,
        )
        for key, members in sorted(grouped_members.items())
    )
    return DecisionView(
        now_us=state.now_us,
        tick_us=episode.tick_us,
        max_hold_us=episode.max_hold_us,
        combine_us=episode.combine_us,
        hold_allowed=_hold_allowed(episode, state),
        queues=tuple(
            queue for queue in all_queue_views if queue.queue_key in decision_queue_keys
        ),
        frontiers=frontiers,
    )


def simulate_policy(
    episode: Episode,
    catalog: ServiceCatalog,
    policy: QueuePolicy,
    *,
    sham: bool = False,
) -> dict[str, object]:
    state = _settle(episode, _initial_state(episode))
    actions: list[dict[str, object]] = []
    total_service = 0.0
    iterations = 0
    sham_permuted_decisions = 0
    sham_permuted_members = 0
    while not _terminal(episode, state):
        eligible = _decision_queues(episode, state)
        if not eligible:
            state = _advance_without_action(episode, state)
            continue
        if sham:
            actual_identity = _observation_map(episode, state, sham=False)
            sham_identity = _observation_map(episode, state, sham=True)
            changed_members = sum(
                actual_identity[node_id] != sham_identity[node_id]
                for node_id in actual_identity
            )
            if changed_members:
                sham_permuted_decisions += 1
                sham_permuted_members += changed_members
        view = _build_view(episode, state, catalog, sham=sham)
        queue_key = policy.choose(view)
        if queue_key is None:
            if not view.hold_allowed:
                raise ProtocolError(f"policy {policy.name} held after timeout")
            before = state.now_us
            state = _hold_once(episode, state)
            actions.append({"kind": "hold", "start_us": before, "end_us": state.now_us})
        else:
            if queue_key not in eligible:
                raise ProtocolError(f"policy {policy.name} chose a non-eligible queue")
            before = state.now_us
            state, service, node_ids = _flush_queue(episode, state, catalog, queue_key)
            total_service += service
            actions.append(
                {
                    "kind": "flush",
                    "time_us": before,
                    "queue_key": list(queue_key),
                    "node_ids": list(node_ids),
                    "rows": len(node_ids),
                    "service_us": service,
                }
            )
            state = _settle(episode, state)
        iterations += 1
        if iterations > 100_000:
            raise ProtocolError("policy replay failed to terminate")
    metrics = _metrics(episode, state)
    return {
        "policy": policy.name,
        "identity_mode": "sham" if sham else "actual",
        "status": "COMPLETE_SIMULATION_ONLY",
        "evaluation_type": "simulation_only",
        "scientific_result_eligible": False,
        **metrics,
        "launches": sum(action["kind"] == "flush" for action in actions),
        "total_service_us": total_service,
        "actions": actions,
        "sham_permuted_decisions": sham_permuted_decisions,
        "sham_permuted_members": sham_permuted_members,
    }


def _replay_oracle_actions(
    episode: Episode,
    catalog: ServiceCatalog,
    action_tokens: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Replay an Oracle token trace into a physical, auditable action ledger."""

    state = _settle(episode, _initial_state(episode))
    ledger: list[dict[str, object]] = []
    for token in action_tokens:
        while not _terminal(episode, state) and not _decision_queues(episode, state):
            state = _advance_without_action(episode, state)
        if _terminal(episode, state):
            raise ProtocolError("Oracle trace contains actions after terminal completion")
        eligible = _decision_queues(episode, state)
        if token == "HOLD":
            before = state.now_us
            state = _hold_once(episode, state)
            ledger.append({"kind": "hold", "start_us": before, "end_us": state.now_us})
            continue
        prefix = "FLUSH:("
        if not token.startswith(prefix) or not token.endswith(")"):
            raise ProtocolError(f"invalid Oracle action token: {token}")
        parts = token[len(prefix) : -1].split(",")
        if len(parts) != 3:
            raise ProtocolError(f"invalid Oracle queue token: {token}")
        queue_key = tuple(int(part.strip()) for part in parts)
        if queue_key not in eligible:
            raise ProtocolError(f"Oracle replay selected an ineligible queue: {queue_key}")
        before = state.now_us
        state, service, node_ids = _flush_queue(episode, state, catalog, queue_key)
        ledger.append(
            {
                "kind": "flush",
                "time_us": before,
                "queue_key": list(queue_key),
                "node_ids": list(node_ids),
                "rows": len(node_ids),
                "service_us": service,
            }
        )
        state = _settle(episode, state)

    while not _terminal(episode, state):
        if _decision_queues(episode, state):
            raise ProtocolError("Oracle trace ended while a decision action remained")
        state = _advance_without_action(episode, state)
    return ledger, _metrics(episode, state)


def solve_exact_oracle(
    episode: Episode,
    catalog: ServiceCatalog,
    *,
    max_states: int = 500_000,
) -> dict[str, object]:
    evaluated = 0

    @lru_cache(maxsize=None)
    def solve(raw_state: SimulationState) -> OracleTail:
        nonlocal evaluated
        state = _settle(episode, raw_state)
        if state != raw_state:
            return solve(state)
        evaluated += 1
        if evaluated > max_states:
            raise ProtocolError("UNSOLVED_EXACT_STATE_LIMIT")
        if _terminal(episode, state):
            metrics = _metrics(episode, state)
            return OracleTail(
                flow_us=float(metrics["flow_us"]),
                total_tardiness_us=float(metrics["total_tardiness_us"]),
                deadline_misses=int(metrics["deadline_misses"]),
                launches=0,
                total_service_us=0.0,
                request_completion_us=tuple(sorted(dict(metrics["request_completion_us"]).items())),
                actions=(),
            )

        eligible = _decision_queues(episode, state)
        if not eligible:
            return solve(_advance_without_action(episode, state))

        candidates: list[OracleTail] = []
        for queue_key in sorted(eligible):
            next_state, service, _node_ids = _flush_queue(episode, state, catalog, queue_key)
            tail = solve(next_state)
            candidates.append(
                replace(
                    tail,
                    launches=tail.launches + 1,
                    total_service_us=tail.total_service_us + service,
                    actions=(f"FLUSH:{queue_key}",) + tail.actions,
                )
            )
        if _hold_allowed(episode, state):
            tail = solve(_hold_once(episode, state))
            candidates.append(replace(tail, actions=("HOLD",) + tail.actions))
        return min(candidates, key=lambda result: result.objective)

    initial = _settle(episode, _initial_state(episode))
    result = solve(initial)
    action_ledger, replay_metrics = _replay_oracle_actions(episode, catalog, result.actions)
    if (
        float(replay_metrics["flow_us"]) != result.flow_us
        or float(replay_metrics["total_tardiness_us"]) != result.total_tardiness_us
        or int(replay_metrics["deadline_misses"]) != result.deadline_misses
    ):
        raise ProtocolError("Oracle action ledger does not reproduce the solved objective")
    return {
        "policy": "exact_oracle",
        "identity_mode": "future_known_upper_bound",
        "status": "SOLVED_EXACT_SIMULATION_ONLY",
        "exact": True,
        "evaluation_type": "simulation_only",
        "scientific_result_eligible": False,
        "flow_us": result.flow_us,
        "total_tardiness_us": result.total_tardiness_us,
        "deadline_misses": result.deadline_misses,
        "request_completion_us": dict(result.request_completion_us),
        "launches": result.launches,
        "total_service_us": result.total_service_us,
        "action_tokens": list(result.actions),
        "actions": action_ledger,
        "states_evaluated": evaluated,
    }


def oracle_capture(
    immediate_flow_us: float,
    candidate_flow_us: float,
    oracle_flow_us: float,
) -> float | None:
    headroom = immediate_flow_us - oracle_flow_us
    if headroom <= EPS:
        return None
    return (immediate_flow_us - candidate_flow_us) / headroom


def make_service_catalog() -> ServiceCatalog:
    points = [
        CurvePoint(1, 10.0, 10.0),
        CurvePoint(2, 14.0, 14.0),
        CurvePoint(4, 20.0, 20.0),
    ]
    return ServiceCatalog({(MODEL, 0): points, (MODEL, 1): points})


def _contribution(
    request: RequestSpec,
    *,
    sample_id: int,
    step: int,
    layer: int,
    slot: int,
    expert_id: int,
    target_replica: int,
) -> Contribution:
    # These ready timestamps are schema-only placeholders.  The pilot never
    # consumes them as counterfactual time; successor readiness is generated by
    # the event engine from predecessor completion.
    schema_ready = request.arrival_us + step * 100.0 + layer * 10.0
    return Contribution(
        model=MODEL,
        phase="decode",
        request_id=request.request_id,
        sample_id=sample_id,
        arrival_us=request.arrival_us,
        deadline_us=request.deadline_us,
        layer=layer,
        token_position=step,
        rank=slot + 1,
        expert_id=expert_id,
        gate_weight=1.0 / 2.0,
        src_replica=target_replica,
        input_event_id=f"{request.request_id}:decode:{step}",
        token_id=step,
        decode_step=step,
        layer_id=layer,
        topk_slot=slot,
        source_rank=target_replica,
        target_replica=target_replica,
        document_id=f"doc-{request.request_id}",
        request_arrival_us=request.arrival_us,
        layer_ready_us=schema_ready,
        route_end_us=schema_ready,
        legal_replica_set=(target_replica,),
    )


def build_episode(*, overlap: str, arrival: str, deadline: str) -> Episode:
    if overlap not in {"aligned", "crossed"}:
        raise ValueError("unknown overlap factor")
    if arrival not in {"aligned", "staggered"}:
        raise ValueError("unknown arrival factor")
    if deadline not in {"loose", "tight"}:
        raise ValueError("unknown deadline factor")

    b_arrival = 0.0 if arrival == "aligned" else 3.0
    if deadline == "tight":
        requests = (
            RequestSpec("r0", 0.0, 48.0),
            RequestSpec("r1", b_arrival, 80.0),
        )
    else:
        requests = (
            RequestSpec("r0", 0.0, 80.0),
            RequestSpec("r1", b_arrival, 80.0),
        )

    contributions: list[Contribution] = []
    for sample_id, request in enumerate(requests):
        for step in range(2):
            for layer in range(2):
                flip = (step + layer) % 2
                for slot in range(2):
                    if overlap == "aligned":
                        # Requests share the same per-slot queue, while the two
                        # top-k slots still select distinct logical experts.
                        expert = flip if slot == 0 else 2 + flip
                    elif request.request_id == "r0":
                        expert = flip if slot == 0 else 1 - flip
                    else:
                        expert = 1 - flip if slot == 0 else flip
                    contributions.append(
                        _contribution(
                            request,
                            sample_id=sample_id,
                            step=step,
                            layer=layer,
                            slot=slot,
                            expert_id=expert,
                            target_replica=slot,
                        )
                    )
    contributions_tuple = tuple(sorted(contributions, key=lambda row: row.contribution_id))
    validate_causal_route_v3(contributions_tuple, require_observed_stages=False)
    nodes = tuple(
        NodeSpec(
            node_id=row.contribution_id,
            request_id=row.request_id,
            decode_step=row.decode_step,
            layer_id=row.layer_id,
            topk_slot=row.topk_slot,
            expert_id=row.expert_id,
            target_replica=row.target_replica,
            deadline_us=row.deadline_us,
        )
        for row in contributions_tuple
    )
    return Episode(
        episode_id=f"overlap={overlap}__arrival={arrival}__deadline={deadline}",
        requests=requests,
        contributions=contributions_tuple,
        nodes=nodes,
    )


def generate_eight_cells() -> tuple[Episode, ...]:
    return tuple(
        build_episode(overlap=overlap, arrival=arrival, deadline=deadline)
        for overlap in ("aligned", "crossed")
        for arrival in ("aligned", "staggered")
        for deadline in ("loose", "tight")
    )


def _policy_objects() -> tuple[QueuePolicy, ...]:
    return (
        ImmediatePolicy(),
        EDFPolicy(),
        MaxRowsPolicy(),
        QueueLocalCreditPolicy(),
        FrontierCreditPolicy(),
    )


def _cell_summary(episode: Episode, results: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    immediate = float(results["immediate"]["flow_us"])
    oracle = float(results["exact_oracle"]["flow_us"])
    headroom = immediate - oracle
    captures = {
        name: oracle_capture(immediate, float(row["flow_us"]), oracle)
        for name, row in results.items()
        if name != "exact_oracle"
    }
    cellwise_best = min(
        SIMPLE_POLICY_NAMES,
        key=lambda name: (float(results[name]["flow_us"]), SIMPLE_POLICY_NAMES.index(name)),
    )
    return {
        "episode_id": episode.episode_id,
        "oracle_headroom_us": headroom,
        "headroom_status": "NONZERO" if headroom > EPS else "HEADROOM_ABSENT",
        "capture": captures,
        "flow_us": {name: float(row["flow_us"]) for name, row in results.items()},
        "deadline_misses": {
            name: int(row["deadline_misses"]) for name, row in results.items()
        },
        # Diagnostic only: this outcome-selected envelope is never used by the
        # decision rule and is not represented as a deployable baseline.
        "cellwise_best_simple_oracle_envelope": cellwise_best,
        "cellwise_best_simple_oracle_envelope_capture": captures[cellwise_best],
        "identity_sham_applicable": (
            int(results["frontier_identity_sham"]["sham_permuted_members"]) > 0
        ),
        "identity_sham_permuted_decisions": int(
            results["frontier_identity_sham"]["sham_permuted_decisions"]
        ),
        "identity_sham_permuted_members": int(
            results["frontier_identity_sham"]["sham_permuted_members"]
        ),
        "frontier_identity_gap": (
            None
            if captures["frontier_credit"] is None
            else float(captures["frontier_credit"])
            - float(captures["frontier_identity_sham"])
        ),
    }


def _attach_fixed_reference(
    cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for raw_cell in cells:
        cell = dict(raw_cell)
        captures = dict(cell["capture"])
        misses = dict(cell["deadline_misses"])
        reference_capture = captures[REFERENCE_SIMPLE_POLICY]
        frontier_capture = captures["frontier_credit"]
        cell.update(
            {
                "fixed_simple_reference": REFERENCE_SIMPLE_POLICY,
                "fixed_simple_reference_capture": reference_capture,
                "frontier_increment_vs_fixed_reference": (
                    None
                    if frontier_capture is None
                    else float(frontier_capture) - float(reference_capture)
                ),
                "frontier_miss_delta_vs_fixed_reference": (
                    int(misses["frontier_credit"])
                    - int(misses[REFERENCE_SIMPLE_POLICY])
                ),
            }
        )
        annotated.append(cell)
    return annotated


def _decision(cells: Sequence[Mapping[str, object]]) -> dict[str, object]:
    eligible = [cell for cell in cells if cell["headroom_status"] == "NONZERO"]
    if not eligible:
        return {
            "verdict": "ACTION_HEADROOM_ABSENT",
            "supports_current_mechanism": False,
            "reason": "No frozen cell has nonzero whole-ready-queue flush/hold Oracle headroom.",
        }
    simple_statistics: dict[str, dict[str, float]] = {}
    for name in SIMPLE_POLICY_NAMES:
        policy_captures = [float(dict(cell["capture"])[name]) for cell in eligible]
        policy_flows = [float(dict(cell["flow_us"])[name]) for cell in eligible]
        simple_statistics[name] = {
            "median_capture": median(policy_captures),
            "worst_cell_capture": min(policy_captures),
            "aggregate_flow_us": sum(policy_flows),
        }
    best_fixed_simple = max(
        SIMPLE_POLICY_NAMES,
        key=lambda name: (
            simple_statistics[name]["median_capture"],
            -simple_statistics[name]["aggregate_flow_us"],
            -SIMPLE_POLICY_NAMES.index(name),
        ),
    )
    suite_max_capture = simple_statistics[best_fixed_simple]["median_capture"]
    reference_capture = median(
        float(cell["fixed_simple_reference_capture"]) for cell in eligible
    )
    frontier_increment = median(
        float(cell["frontier_increment_vs_fixed_reference"]) for cell in eligible
    )
    identity_gap = median(float(cell["frontier_identity_gap"]) for cell in eligible)
    sham_applicable_cells = sum(bool(cell["identity_sham_applicable"]) for cell in eligible)
    misses_safe = all(
        int(cell["frontier_miss_delta_vs_fixed_reference"]) <= 0 for cell in eligible
    )
    envelope_capture = median(
        float(cell["cellwise_best_simple_oracle_envelope_capture"])
        for cell in eligible
    )
    positive = (
        len(eligible) >= int(DECISION_THRESHOLDS["minimum_eligible_cells"])
        and sham_applicable_cells
        >= int(DECISION_THRESHOLDS["minimum_identity_sham_applicable_eligible_cells"])
        and suite_max_capture
        < float(DECISION_THRESHOLDS["maximum_fixed_simple_median_capture"])
        and frontier_increment
        >= float(DECISION_THRESHOLDS["minimum_frontier_increment_vs_reference"])
        and identity_gap >= float(DECISION_THRESHOLDS["minimum_frontier_identity_gap"])
        and misses_safe
    )
    if positive:
        verdict = "PILOT_SIGNAL_CONTINUE"
        reason = "All corrected descriptive support conditions are satisfied; unseen-cell confirmation remains required."
    elif suite_max_capture >= float(
        DECISION_THRESHOLDS["maximum_fixed_simple_median_capture"]
    ):
        verdict = "SIMPLE_BASELINE_SUFFICIENT"
        reason = "At least one fixed simple policy captures at least 90% of median Oracle headroom."
    else:
        verdict = "FRONTIER_SIGNAL_NOT_SUPPORTED"
        reason = (
            "Nonzero action headroom exists, but FrontierCredit does not clear the frozen "
            "incremental/identity conditions against the fixed queue-local reference."
        )
    return {
        "verdict": verdict,
        "supports_current_mechanism": positive,
        "reason": reason,
        "eligible_cells": len(eligible),
        "identity_sham_applicable_eligible_cells": sham_applicable_cells,
        "fixed_simple_reference": REFERENCE_SIMPLE_POLICY,
        "fixed_simple_reference_selection": (
            "mechanism-matched queue-local comparator selected for the audit-corrected descriptive "
            "rerun after v1 inspection; no per-cell switching; not preregistered"
        ),
        "median_fixed_simple_reference_capture": reference_capture,
        "best_fixed_simple_policy_by_median": best_fixed_simple,
        "maximum_fixed_simple_median_capture": suite_max_capture,
        "simple_policy_statistics": simple_statistics,
        "cellwise_best_simple_oracle_envelope_median_capture_diagnostic_only": envelope_capture,
        "median_frontier_increment_vs_fixed_reference": frontier_increment,
        "median_frontier_identity_gap": identity_gap,
        "frontier_deadline_misses_not_worse_than_fixed_reference_in_all_eligible_cells": misses_safe,
        "thresholds": dict(DECISION_THRESHOLDS),
    }


def run_pilot(*, max_oracle_states: int = 500_000) -> dict[str, object]:
    catalog = make_service_catalog()
    cell_payloads: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for episode in generate_eight_cells():
        results: dict[str, dict[str, object]] = {}
        for policy in _policy_objects():
            results[policy.name] = simulate_policy(episode, catalog, policy)
        sham_policy = FrontierCreditPolicy(name="frontier_identity_sham")
        results[sham_policy.name] = simulate_policy(episode, catalog, sham_policy, sham=True)
        results["exact_oracle"] = solve_exact_oracle(
            episode,
            catalog,
            max_states=max_oracle_states,
        )
        oracle_flow = float(results["exact_oracle"]["flow_us"])
        for name, row in results.items():
            if float(row["flow_us"]) < oracle_flow - EPS:
                raise ProtocolError(f"policy {name} beat the exact Oracle in {episode.episode_id}")
        summary = _cell_summary(episode, results)
        summaries.append(summary)
        cell_payloads.append(
            {
                "episode_id": episode.episode_id,
                "episode": {
                    "requests": [asdict(request) for request in episode.requests],
                    "replicas": episode.replicas,
                    "decode_steps": episode.decode_steps,
                    "layers": episode.layers,
                    "top_k": episode.top_k,
                    "tick_us": episode.tick_us,
                    "max_hold_us": episode.max_hold_us,
                    "combine_us": episode.combine_us,
                    "nodes": [asdict(node) for node in episode.nodes],
                },
                "results": results,
                "summary": summary,
            }
        )
    annotated_summaries = _attach_fixed_reference(summaries)
    for cell_payload, summary in zip(cell_payloads, annotated_summaries, strict=True):
        cell_payload["summary"] = summary
    decision = _decision(annotated_summaries)
    return {
        "schema": "frontiercredit-full-dag-pilot-v2",
        "status": "COMPLETE_SIMULATION_ONLY",
        "analysis_mode": "AUDIT_CORRECTED_DESCRIPTIVE_REANALYSIS",
        "evaluation_type": "simulation_only",
        "scientific_result_eligible": False,
        "formal_gate2_unchanged": True,
        "protocol": {
            "cells": "overlap(2) x arrival(2) x deadline(2) = 8",
            "requests_per_cell": 2,
            "decode_steps": 2,
            "layers": 2,
            "top_k": 2,
            "fixed_executors": 2,
            "action_space": (
                "flush all currently ready rows in one eligible queue on any idle executor, "
                "or bounded hold; no replica-order pruning"
            ),
            "online_visibility": "revealed prefix only; future arrivals/routes hidden",
            "identity_sham": (
                "rotate slot-1 membership only among request groups already revealed at the same "
                "decode-step/layer; no-op when fewer than two groups are visible"
            ),
            "oracle_visibility": "future-known exact upper bound on the identical state transition",
            "service_curve_us": {"rows_1": 10.0, "rows_2": 14.0, "rows_4": 20.0},
            "tick_us": 1.0,
            "max_hold_us": 3.0,
            "combine_us": 1.0,
            "launch_cost_us": 0.0,
            "simple_policy_suite": list(SIMPLE_POLICY_NAMES),
            "fixed_simple_reference": REFERENCE_SIMPLE_POLICY,
            "fixed_simple_reference_selection": (
                "mechanism-matched queue-local comparator selected for the audit-corrected descriptive "
                "rerun after v1 inspection; no per-cell switching; not preregistered"
            ),
            "decision_thresholds": dict(DECISION_THRESHOLDS),
            "cellwise_best_simple_oracle_envelope": "diagnostic only; never used for verdict",
        },
        "cells": cell_payloads,
        "decision": decision,
        "claim_ceiling": (
            "This development-only deterministic simulation may support or weaken the specified "
            "queue-release signal/action formulation. It cannot establish natural-workload prevalence, "
            "real-model performance, GPU serving latency, EP/NCCL/RDMA behavior, or production SLO gains."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"


def write_outputs(output_dir: Path, payload: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = dict(payload["protocol"])
    protocol.update(
        {
            "schema": payload["schema"],
            "evaluation_type": payload["evaluation_type"],
            "scientific_result_eligible": payload["scientific_result_eligible"],
            "analysis_mode": payload["analysis_mode"],
        }
    )
    protocol_path = output_dir / "pilot_protocol.json"
    results_path = output_dir / "pilot_results.json"
    _write_json(protocol_path, protocol)
    _write_json(results_path, payload)

    decision = dict(payload["decision"])
    lines = [
        "# FrontierCredit Full Request-DAG Pilot",
        "",
        f"- Status: `{payload['status']}`",
        f"- Analysis mode: `{payload['analysis_mode']}`",
        f"- Evaluation type: `{payload['evaluation_type']}`",
        f"- Scientific-result eligible: `{str(payload['scientific_result_eligible']).lower()}`",
        f"- Verdict: `{decision['verdict']}`",
        f"- Supports current mechanism: `{str(decision['supports_current_mechanism']).lower()}`",
        f"- Reason: {decision['reason']}",
        f"- Fixed simple reference: `{decision.get('fixed_simple_reference', 'NA')}`",
        f"- Best fixed simple policy by median: `{decision.get('best_fixed_simple_policy_by_median', 'NA')}`",
        f"- Maximum fixed-simple median capture: `{decision.get('maximum_fixed_simple_median_capture', 'NA')}`",
        "- Cell-wise best-simple envelope: diagnostic only; not used by the verdict.",
        "",
        "## Fixed simple-policy aggregate statistics",
        "",
        "| Policy | Median capture | Worst-cell capture | Aggregate flow (eligible cells) |",
        "|---|---:|---:|---:|",
    ]
    for name, stats in dict(decision.get("simple_policy_statistics", {})).items():
        lines.append(
            "| {name} | {median:.6f} | {worst:.6f} | {flow:.6f} |".format(
                name=name,
                median=float(stats["median_capture"]),
                worst=float(stats["worst_cell_capture"]),
                flow=float(stats["aggregate_flow_us"]),
            )
        )
    lines.extend(
        [
        "",
        "## Per-cell summary",
        "",
        "| Cell | Headroom | Fixed reference | Reference capture | Frontier Δ | Identity gap | Miss Δ | Sham decisions/members |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in payload["cells"]:
        summary = cell["summary"]
        def fmt(value: object) -> str:
            return "NA" if value is None else f"{float(value):.6f}"
        lines.append(
            "| {episode_id} | {headroom:.6f} | {simple} | {simple_cap} | {increment} | {gap} | {miss} | {sham_decisions}/{sham_members} |".format(
                episode_id=summary["episode_id"],
                headroom=float(summary["oracle_headroom_us"]),
                simple=summary["fixed_simple_reference"],
                simple_cap=fmt(summary["fixed_simple_reference_capture"]),
                increment=fmt(summary["frontier_increment_vs_fixed_reference"]),
                gap=fmt(summary["frontier_identity_gap"]),
                miss=int(summary["frontier_miss_delta_vs_fixed_reference"]),
                sham_decisions=int(summary["identity_sham_permuted_decisions"]),
                sham_members=int(summary["identity_sham_permuted_members"]),
            )
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            str(payload["claim_ceiling"]),
            "",
        ]
    )
    summary_path = output_dir / "pilot_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    run_status_path = output_dir / "RUN_STATUS.json"
    _write_json(
        run_status_path,
        {
            "status": payload["status"],
            "analysis_mode": payload["analysis_mode"],
            "evaluation_type": payload["evaluation_type"],
            "scientific_result_eligible": payload["scientific_result_eligible"],
            "verdict": decision["verdict"],
        },
    )
    manifest_path = output_dir / "MANIFEST.json"
    runner_path = Path(__file__).resolve()
    core_path = runner_path.with_name("core.py")
    test_path = runner_path.with_name("test_frontiercredit_full_dag_pilot.py")
    _write_json(
        manifest_path,
        {
            "schema": "frontiercredit-pilot-manifest-v2",
            "files": {
                path.name: _sha256(path)
                for path in (protocol_path, results_path, summary_path, run_status_path)
            },
            "source_files": {
                str(path): _sha256(path)
                for path in (runner_path, core_path, test_path)
            },
            "git_commit": _git_commit(runner_path.parent),
            "git_commit_note": (
                "Context only; source_files hashes bind uncommitted or untracked evaluation code."
            ),
            "runtime": {
                "python_executable": sys.executable,
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-oracle-states", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_oracle_states <= 0:
        raise SystemExit("--max-oracle-states must be positive")
    payload = run_pilot(max_oracle_states=args.max_oracle_states)
    output_dir = Path(args.output_dir)
    write_outputs(output_dir, payload)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(f"wrote FrontierCredit pilot artifacts to {output_dir}")


if __name__ == "__main__":
    main()
