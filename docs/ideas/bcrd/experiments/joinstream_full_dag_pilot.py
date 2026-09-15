from __future__ import annotations

"""Exploratory CPU exact-Oracle pilot for JoinStream.

JoinStream leaves the ready-queue membership and launch count unchanged.  Its
only additional physical action is a whole-queue launch whose rows expose
fixed completion milestones while the executor remains occupied until the
last milestone.  This file is intentionally independent of the sealed
CriticalSplit experiment and never emits a formal completion authority.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

try:
    from . import frontiercredit_full_dag_pilot as frontier
except ImportError:
    import frontiercredit_full_dag_pilot as frontier  # type: ignore


EPS = frontier.EPS
MODEL = frontier.MODEL
FROZEN_MAX_STATES = 500_000
CURVES = ("tail", "uniform")
TAXES_US = (0.0, 2.0)
BATCH_ROWS = (2, 4)
ACTION_KINDS = ("atomic", "hold", "stream")
CLAIM_CEILING = (
    "Exploratory deterministic CPU action-space evidence only. It is not a "
    "formal experiment, paper result, GPU result, natural-workload result, or "
    "evidence about EP, NCCL, RDMA, kernels, serving, or production SLOs."
)

NodeId = frontier.NodeId
GroupKey = frontier.GroupKey
QueueKey = frontier.QueueKey
Episode = frontier.Episode


@dataclass(frozen=True, order=True)
class NodeMilestone:
    at_us: float
    node_id: NodeId


@dataclass(frozen=True)
class MilestoneBatch:
    queue_key: QueueKey
    node_ids: tuple[NodeId, ...]
    row_order: tuple[NodeId, ...]
    kind: str
    start_us: float
    finish_us: float
    service_us: float
    pending: tuple[NodeMilestone, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"atomic", "stream"}:
            raise ValueError("batch kind must be atomic or stream")
        if not self.node_ids or self.node_ids != tuple(sorted(self.node_ids)):
            raise ValueError("batch membership must be non-empty and canonical")
        if set(self.row_order) != set(self.node_ids) or len(self.row_order) != len(
            self.node_ids
        ):
            raise ValueError("row order must be a permutation of batch membership")
        if not self.pending:
            raise ValueError("a newly launched batch needs pending milestones")
        if abs(max(item.at_us for item in self.pending) - self.finish_us) > EPS:
            raise ValueError("final milestone must equal executor finish")


@dataclass(frozen=True)
class SimulationState:
    now_us: float
    arrived_requests: tuple[str, ...] = ()
    revealed_at: tuple[tuple[NodeId, float], ...] = ()
    ready: tuple[frontier.ReadyNode, ...] = ()
    running: tuple[MilestoneBatch | None, ...] = ()
    completed_at: tuple[tuple[NodeId, float], ...] = ()
    joined_groups: tuple[GroupKey, ...] = ()
    pending_releases: tuple[frontier.PendingRelease, ...] = ()
    request_completed_at: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class JoinStreamCell:
    episode: Episode
    batch_rows: int
    curve: str
    tax_us: float

    def __post_init__(self) -> None:
        if self.batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if self.curve not in CURVES:
            raise ValueError(f"unknown milestone curve: {self.curve}")
        if self.tax_us < 0 or not math.isfinite(self.tax_us):
            raise ValueError("tax_us must be finite and non-negative")
        if len(self.episode.requests) != self.batch_rows:
            raise ValueError("a frozen cell must contain M requests")


@dataclass(frozen=True, order=True)
class Action:
    kind: str
    queue_key: QueueKey | None = None
    node_ids: tuple[NodeId, ...] = ()
    row_order: tuple[NodeId, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unsupported action kind: {self.kind}")
        if self.kind == "hold":
            if self.queue_key is not None or self.node_ids or self.row_order:
                raise ValueError("hold cannot bind queue rows")
            return
        if self.queue_key is None:
            raise ValueError("launch action requires a queue key")
        if not self.node_ids or self.node_ids != tuple(sorted(self.node_ids)):
            raise ValueError("launch membership must be non-empty and canonical")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("launch membership contains duplicates")
        if self.kind == "atomic":
            if self.row_order:
                raise ValueError("atomic launch has no observable row order")
        elif (
            len(self.node_ids) < 2
            or len(self.row_order) != len(self.node_ids)
            or set(self.row_order) != set(self.node_ids)
        ):
            raise ValueError("stream row order must permute at least two launch rows")

    def token(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "node_ids": list(self.node_ids),
                "queue_key": None if self.queue_key is None else list(self.queue_key),
                "row_order": list(self.row_order),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_token(cls, token: str) -> "Action":
        try:
            raw = json.loads(token)
        except json.JSONDecodeError as error:
            raise frontier.ProtocolError(f"invalid action token JSON: {error}") from error
        if not isinstance(raw, dict) or set(raw) != {
            "kind",
            "node_ids",
            "queue_key",
            "row_order",
        }:
            raise frontier.ProtocolError("action token fields are not canonical")
        queue_raw = raw["queue_key"]
        if queue_raw is None:
            queue_key = None
        elif (
            isinstance(queue_raw, list)
            and len(queue_raw) == 3
            and all(type(value) is int for value in queue_raw)
        ):
            queue_key = tuple(queue_raw)
        else:
            raise frontier.ProtocolError("invalid action queue key")
        nodes = raw["node_ids"]
        order = raw["row_order"]
        if not isinstance(nodes, list) or not all(isinstance(value, str) for value in nodes):
            raise frontier.ProtocolError("invalid action node ids")
        if not isinstance(order, list) or not all(isinstance(value, str) for value in order):
            raise frontier.ProtocolError("invalid action row order")
        if not isinstance(raw["kind"], str):
            raise frontier.ProtocolError("invalid action kind")
        try:
            action = cls(raw["kind"], queue_key, tuple(nodes), tuple(order))
        except ValueError as error:
            raise frontier.ProtocolError(str(error)) from error
        if action.token() != token:
            raise frontier.ProtocolError("action token is not canonical JSON")
        return action


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


def build_cell(*, batch_rows: int, curve: str, tax_us: float) -> JoinStreamCell:
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    requests = tuple(
        frontier.RequestSpec(f"r{index:02d}", 0.0, 1_000.0)
        for index in range(batch_rows)
    )
    contributions: list[frontier.Contribution] = []
    for sample_id, request in enumerate(requests):
        for layer in range(2):
            for slot in range(2):
                expert = layer * 2 + slot
                contributions.append(
                    frontier._contribution(
                        request,
                        sample_id=sample_id,
                        step=0,
                        layer=layer,
                        slot=slot,
                        expert_id=expert,
                        target_replica=slot,
                    )
                )
    frozen_contributions = tuple(
        sorted(contributions, key=lambda row: row.contribution_id)
    )
    nodes = tuple(
        frontier.NodeSpec(
            node_id=row.contribution_id,
            request_id=row.request_id,
            decode_step=row.decode_step,
            layer_id=row.layer_id,
            topk_slot=row.topk_slot,
            expert_id=row.expert_id,
            target_replica=row.target_replica,
            deadline_us=row.deadline_us,
        )
        for row in frozen_contributions
    )
    episode = frontier.Episode(
        episode_id=(
            f"joinstream__M={batch_rows}__curve={curve}__tax={tax_us:g}"
        ),
        requests=requests,
        contributions=frozen_contributions,
        nodes=nodes,
        replicas=2,
        decode_steps=1,
        layers=2,
        top_k=2,
        tick_us=1.0,
        max_hold_us=3.0,
        combine_us=1.0,
        launch_cost_us=0.0,
    )
    return JoinStreamCell(episode, batch_rows, curve, float(tax_us))


def generate_eight_cells() -> tuple[JoinStreamCell, ...]:
    return tuple(
        build_cell(batch_rows=rows, curve=curve, tax_us=tax)
        for rows in BATCH_ROWS
        for curve in ("uniform", "tail")
        for tax in TAXES_US
    )


def make_service_catalog() -> frontier.ServiceCatalog:
    return frontier.make_service_catalog()


def _initial_state(episode: Episode) -> SimulationState:
    first = min(request.arrival_us for request in episode.requests)
    return SimulationState(
        now_us=first,
        running=tuple(None for _ in range(episode.replicas)),
    )


def _reveal_group(
    episode: Episode,
    group_key: GroupKey,
    now_us: float,
    revealed: dict[NodeId, float],
    ready: dict[NodeId, float],
) -> None:
    for node_id in episode.group_map[group_key]:
        if node_id in revealed:
            raise frontier.ProtocolError(f"node {node_id} was revealed twice")
        revealed[node_id] = now_us
        ready[node_id] = now_us


def _settle(episode: Episode, state: SimulationState) -> SimulationState:
    """Process milestones, releases, arrivals, then newly closed joins."""

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

        for replica, batch in enumerate(running):
            if batch is None:
                continue
            due = tuple(item for item in batch.pending if item.at_us <= now + EPS)
            if not due:
                continue
            for milestone in due:
                if milestone.node_id in completed:
                    raise frontier.ProtocolError(
                        f"node {milestone.node_id} completed more than once"
                    )
                completed[milestone.node_id] = milestone.at_us
            remaining = tuple(item for item in batch.pending if item not in due)
            if remaining:
                running[replica] = replace(batch, pending=remaining)
            else:
                if batch.finish_us > now + EPS:
                    raise frontier.ProtocolError("executor released before final milestone")
                running[replica] = None
            changed = True

        due_releases = [item for item in releases if item.release_us <= now + EPS]
        if due_releases:
            releases = [item for item in releases if item not in due_releases]
            for release in sorted(due_releases):
                request_id, step, layer = release.group_key
                if layer + 1 < episode.layers:
                    _reveal_group(
                        episode,
                        (request_id, step, layer + 1),
                        now,
                        revealed,
                        ready,
                    )
                elif step + 1 < episode.decode_steps:
                    _reveal_group(
                        episode,
                        (request_id, step + 1, 0),
                        now,
                        revealed,
                        ready,
                    )
                else:
                    if request_id in request_completed:
                        raise frontier.ProtocolError(
                            f"request {request_id} completed more than once"
                        )
                    request_completed[request_id] = now
                changed = True

        for request in sorted(episode.requests):
            if request.request_id in arrived or request.arrival_us > now + EPS:
                continue
            arrived.add(request.request_id)
            _reveal_group(
                episode,
                (request.request_id, 0, 0),
                now,
                revealed,
                ready,
            )
            changed = True

        pending_groups = {item.group_key for item in releases}
        for group_key, node_ids in episode.group_map.items():
            if group_key in joined or group_key in pending_groups:
                continue
            if all(node_id in completed for node_id in node_ids):
                joined.add(group_key)
                releases.append(frontier.PendingRelease(now + episode.combine_us, group_key))
                changed = True

        next_state = SimulationState(
            now_us=now,
            arrived_requests=tuple(sorted(arrived)),
            revealed_at=_sorted_pairs(revealed),
            ready=tuple(
                frontier.ReadyNode(node_id, ready[node_id]) for node_id in sorted(ready)
            ),
            running=tuple(running),
            completed_at=_sorted_pairs(completed),
            joined_groups=tuple(sorted(joined)),
            pending_releases=tuple(sorted(releases)),
            request_completed_at=_sorted_pairs(request_completed),
        )
        if not changed:
            return next_state
        state = next_state


def _ready_queues(
    episode: Episode, state: SimulationState
) -> dict[QueueKey, tuple[frontier.ReadyNode, ...]]:
    grouped: dict[QueueKey, list[frontier.ReadyNode]] = {}
    node_map = episode.node_map
    for item in state.ready:
        grouped.setdefault(node_map[item.node_id].queue_key, []).append(item)
    return {key: tuple(sorted(items)) for key, items in grouped.items()}


def _eligible_queues(
    episode: Episode, state: SimulationState
) -> dict[QueueKey, tuple[frontier.ReadyNode, ...]]:
    return {
        key: items
        for key, items in _ready_queues(episode, state).items()
        if state.running[key[0]] is None
    }


def _next_external_event(episode: Episode, state: SimulationState) -> float | None:
    values: list[float] = []
    arrived = set(state.arrived_requests)
    values.extend(
        request.arrival_us
        for request in episode.requests
        if request.request_id not in arrived and request.arrival_us > state.now_us + EPS
    )
    values.extend(
        milestone.at_us
        for batch in state.running
        if batch is not None
        for milestone in batch.pending
        if milestone.at_us > state.now_us + EPS
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
        state.now_us - min(item.ready_since_us for item in items)
        < episode.max_hold_us - EPS
        for items in eligible.values()
    )


def _advance_without_action(episode: Episode, state: SimulationState) -> SimulationState:
    next_time = _next_external_event(episode, state)
    if next_time is None:
        raise frontier.ProtocolError("INVALID_DAG_DEADLOCK")
    return _settle(episode, replace(state, now_us=next_time))


def _hold_once(episode: Episode, state: SimulationState) -> SimulationState:
    if not _hold_allowed(episode, state):
        raise frontier.ProtocolError("bounded hold is illegal")
    candidates: list[float] = []
    external = _next_external_event(episode, state)
    if external is not None:
        candidates.append(external)
    for items in _eligible_queues(episode, state).values():
        candidates.append(min(item.ready_since_us for item in items) + episode.max_hold_us)
    future = [value for value in candidates if value > state.now_us + EPS]
    if not future:
        raise frontier.ProtocolError("hold failed to advance time")
    return _settle(episode, replace(state, now_us=min(future)))


def _frozen_row_order(
    episode: Episode, queue_key: QueueKey, node_ids: Sequence[NodeId]
) -> tuple[NodeId, ...]:
    node_map = episode.node_map
    return tuple(
        sorted(
            node_ids,
            key=lambda node_id: (
                node_map[node_id].request_id,
                node_map[node_id].decode_step,
                node_map[node_id].layer_id,
                node_map[node_id].topk_slot,
                node_id,
            ),
        )
    )


def _milestone_offsets(
    *, rows: int, service_us: float, curve: str, tax_us: float
) -> tuple[float, ...]:
    if rows <= 0 or service_us <= 0:
        raise ValueError("milestones need positive rows and service")
    if curve not in CURVES:
        raise ValueError(f"unknown milestone curve: {curve}")
    values = []
    for index in range(1, rows + 1):
        fraction = index / rows
        scaled = fraction if curve == "uniform" else math.sqrt(fraction)
        values.append(tax_us + service_us * scaled)
    if abs(values[-1] - (tax_us + service_us)) > EPS:
        raise AssertionError("stream final milestone is not frozen service plus tax")
    return tuple(values)


def canonical_actions(
    cell: JoinStreamCell, state: SimulationState, *, expanded: bool
) -> tuple[Action, ...]:
    episode = cell.episode
    actions: set[Action] = set()
    for queue_key, items in _eligible_queues(episode, state).items():
        node_ids = tuple(sorted(item.node_id for item in items))
        actions.add(Action("atomic", queue_key, node_ids))
        if expanded and len(node_ids) >= 2:
            actions.add(
                Action(
                    "stream",
                    queue_key,
                    node_ids,
                    _frozen_row_order(episode, queue_key, node_ids),
                )
            )
    if _hold_allowed(episode, state):
        actions.add(Action("hold"))
    return tuple(sorted(actions, key=lambda action: action.token()))


def _launch(
    cell: JoinStreamCell,
    state: SimulationState,
    catalog: frontier.ServiceCatalog,
    action: Action,
    *,
    expanded: bool,
) -> tuple[SimulationState, float, tuple[NodeId, ...], tuple[NodeMilestone, ...]]:
    if action not in canonical_actions(cell, state, expanded=expanded):
        raise frontier.ProtocolError("action is stale, ineligible, or noncanonical")
    if action.kind == "hold":
        return _hold_once(cell.episode, state), 0.0, (), ()
    assert action.queue_key is not None
    episode = cell.episode
    replica, layer, _expert = action.queue_key
    if state.running[replica] is not None:
        raise frontier.ProtocolError("launch selected a busy executor")
    base_service = catalog.estimate_us(MODEL, layer, len(action.node_ids))
    start = state.now_us + episode.launch_cost_us
    if action.kind == "atomic":
        service = base_service
        row_order = action.node_ids
        finish = start + service
        milestones = tuple(
            NodeMilestone(finish, node_id) for node_id in action.node_ids
        )
    else:
        service = base_service + cell.tax_us
        row_order = action.row_order
        offsets = _milestone_offsets(
            rows=len(row_order),
            service_us=base_service,
            curve=cell.curve,
            tax_us=cell.tax_us,
        )
        milestones = tuple(
            NodeMilestone(start + offset, node_id)
            for node_id, offset in zip(row_order, offsets)
        )
        finish = start + service
    selected = set(action.node_ids)
    ready = tuple(item for item in state.ready if item.node_id not in selected)
    running = list(state.running)
    running[replica] = MilestoneBatch(
        queue_key=action.queue_key,
        node_ids=action.node_ids,
        row_order=row_order,
        kind=action.kind,
        start_us=start,
        finish_us=finish,
        service_us=service,
        pending=tuple(sorted(milestones)),
    )
    return (
        replace(state, ready=ready, running=tuple(running)),
        service,
        action.node_ids,
        milestones,
    )


def _terminal(episode: Episode, state: SimulationState) -> bool:
    return len(state.request_completed_at) == len(episode.requests)


def _metrics(episode: Episode, state: SimulationState) -> dict[str, object]:
    if not _terminal(episode, state):
        raise frontier.ProtocolError("metrics require terminal state")
    completion = dict(state.request_completed_at)
    request_map = episode.request_map
    flow = sum(completion[key] - request_map[key].arrival_us for key in completion)
    tardiness = sum(
        max(0.0, completion[key] - request_map[key].deadline_us)
        for key in completion
    )
    misses = sum(
        completion[key] > request_map[key].deadline_us + EPS for key in completion
    )
    expected = set(episode.node_map)
    revealed = dict(state.revealed_at)
    completed = dict(state.completed_at)
    if set(revealed) != expected or set(completed) != expected:
        raise frontier.ProtocolError("full DAG did not reveal and complete every node")
    if (
        state.ready
        or any(batch is not None for batch in state.running)
        or state.pending_releases
        or set(state.joined_groups) != set(episode.group_map)
    ):
        raise frontier.ProtocolError("terminal state is not physically closed")
    return {
        "flow_us": flow,
        "total_tardiness_us": tardiness,
        "deadline_misses": misses,
        "request_completion_us": completion,
        "node_completion_us": completed,
        "node_revealed_us": revealed,
    }


def _replay_actions(
    cell: JoinStreamCell,
    catalog: frontier.ServiceCatalog,
    action_tokens: Sequence[str],
    *,
    expanded: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    episode = cell.episode
    state = _settle(episode, _initial_state(episode))
    ledger: list[dict[str, object]] = []
    launched: list[NodeId] = []
    total_service = 0.0
    launches = 0
    for token in action_tokens:
        while not _terminal(episode, state) and not canonical_actions(
            cell, state, expanded=expanded
        ):
            state = _advance_without_action(episode, state)
        if _terminal(episode, state):
            raise frontier.ProtocolError("action trace continues after terminal state")
        action = Action.from_token(token)
        before = state.now_us
        state, service, selected, milestones = _launch(
            cell, state, catalog, action, expanded=expanded
        )
        if action.kind == "hold":
            ledger.append(
                {
                    "kind": "hold",
                    "start_us": before,
                    "end_us": state.now_us,
                    "token": token,
                }
            )
            continue
        launches += 1
        total_service += service
        launched.extend(selected)
        ledger.append(
            {
                "kind": action.kind,
                "time_us": before,
                "queue_key": list(action.queue_key or ()),
                "node_ids": list(selected),
                "row_order": list(action.row_order),
                "rows": len(selected),
                "service_us": service,
                "milestones": [asdict(item) for item in milestones],
                "token": token,
            }
        )
        state = _settle(episode, state)
    while not _terminal(episode, state):
        if canonical_actions(cell, state, expanded=expanded):
            raise frontier.ProtocolError("action trace ended with a legal action pending")
        state = _advance_without_action(episode, state)
    counts = Counter(launched)
    expected = set(episode.node_map)
    if set(counts) != expected or any(value != 1 for value in counts.values()):
        raise frontier.ProtocolError("replay did not launch every node exactly once")
    metrics = _metrics(episode, state)
    return ledger, {
        **metrics,
        "launches": launches,
        "total_service_us": total_service,
        "node_conservation": {
            "expected_nodes": len(expected),
            "launched_nodes": len(launched),
            "unique_launched_nodes": len(counts),
            "exactly_once": True,
        },
    }


def solve_exact_oracle(
    cell: JoinStreamCell,
    catalog: frontier.ServiceCatalog,
    *,
    expanded: bool,
    max_states: int = FROZEN_MAX_STATES,
) -> dict[str, object]:
    if max_states <= 0:
        raise ValueError("max_states must be positive")
    episode = cell.episode
    evaluated = 0

    @lru_cache(maxsize=None)
    def solve(raw_state: SimulationState) -> OracleTail:
        nonlocal evaluated
        state = _settle(episode, raw_state)
        if state != raw_state:
            return solve(state)
        evaluated += 1
        if evaluated > max_states:
            raise frontier.ProtocolError("UNSOLVED_EXACT_STATE_LIMIT")
        if _terminal(episode, state):
            metrics = _metrics(episode, state)
            return OracleTail(
                flow_us=float(metrics["flow_us"]),
                total_tardiness_us=float(metrics["total_tardiness_us"]),
                deadline_misses=int(metrics["deadline_misses"]),
                launches=0,
                total_service_us=0.0,
                request_completion_us=tuple(
                    sorted(dict(metrics["request_completion_us"]).items())
                ),
                actions=(),
            )
        actions = canonical_actions(cell, state, expanded=expanded)
        if not actions:
            return solve(_advance_without_action(episode, state))
        candidates: list[OracleTail] = []
        for action in actions:
            next_state, service, _selected, _milestones = _launch(
                cell, state, catalog, action, expanded=expanded
            )
            tail = solve(next_state)
            candidates.append(
                replace(
                    tail,
                    launches=tail.launches + int(action.kind != "hold"),
                    total_service_us=tail.total_service_us + service,
                    actions=(action.token(),) + tail.actions,
                )
            )
        return min(candidates, key=lambda result: result.objective)

    initial = _settle(episode, _initial_state(episode))
    solved = solve(initial)
    ledger, replay = _replay_actions(
        cell, catalog, solved.actions, expanded=expanded
    )
    if not (
        math.isclose(float(replay["flow_us"]), solved.flow_us, abs_tol=EPS)
        and math.isclose(
            float(replay["total_tardiness_us"]),
            solved.total_tardiness_us,
            abs_tol=EPS,
        )
        and int(replay["deadline_misses"]) == solved.deadline_misses
        and int(replay["launches"]) == solved.launches
        and math.isclose(
            float(replay["total_service_us"]),
            solved.total_service_us,
            abs_tol=EPS,
        )
        and dict(replay["request_completion_us"])
        == dict(solved.request_completion_us)
    ):
        raise frontier.ProtocolError("Oracle replay does not reproduce solved result")
    return {
        "policy": "joinstream_expanded_exact" if expanded else "atomic_exact",
        "status": "EXPLORATORY_EXACT_SIMULATION_ONLY",
        "exact": True,
        "expanded": expanded,
        "evaluation_type": "simulation_only",
        "scientific_result_eligible": False,
        "paper_result": False,
        "flow_us": solved.flow_us,
        "total_tardiness_us": solved.total_tardiness_us,
        "deadline_misses": solved.deadline_misses,
        "request_completion_us": dict(solved.request_completion_us),
        "launches": solved.launches,
        "total_service_us": solved.total_service_us,
        "action_tokens": list(solved.actions),
        "actions": ledger,
        "atomic_launches": sum(row["kind"] == "atomic" for row in ledger),
        "stream_launches": sum(row["kind"] == "stream" for row in ledger),
        "states_evaluated": evaluated,
        "node_conservation": dict(replay["node_conservation"]),
    }


def _objective_prefix(result: Mapping[str, object]) -> tuple[object, ...]:
    return (
        float(result["flow_us"]),
        float(result["total_tardiness_us"]),
        int(result["deadline_misses"]),
        int(result["launches"]),
        float(result["total_service_us"]),
    )


def run_pilot(*, max_states: int = FROZEN_MAX_STATES) -> dict[str, object]:
    catalog = make_service_catalog()
    cells: list[dict[str, object]] = []
    for cell in generate_eight_cells():
        baseline = solve_exact_oracle(
            cell, catalog, expanded=False, max_states=max_states
        )
        expanded = solve_exact_oracle(
            cell, catalog, expanded=True, max_states=max_states
        )
        if _objective_prefix(expanded) > _objective_prefix(baseline):
            raise frontier.ProtocolError("expanded Oracle is worse than its atomic subset")
        cells.append(
            {
                "episode_id": cell.episode.episode_id,
                "factors": {
                    "batch_rows": cell.batch_rows,
                    "curve": cell.curve,
                    "tax_us": cell.tax_us,
                },
                "baseline_atomic_exact": baseline,
                "expanded_joinstream_exact": expanded,
                "diagnostic": {
                    "flow_delta_us": float(baseline["flow_us"])
                    - float(expanded["flow_us"]),
                    "deadline_miss_delta": int(expanded["deadline_misses"])
                    - int(baseline["deadline_misses"]),
                    "stream_launches": int(expanded["stream_launches"]),
                },
            }
        )
    return {
        "schema": "joinstream-exploratory-pilot-v1",
        "status": "EXPLORATORY_COMPUTED_NOT_SEALED",
        "evaluation_type": "simulation_only",
        "scientific_result_eligible": False,
        "paper_result": False,
        "formal_authority": False,
        "interpretation_authorized": False,
        "protocol": {
            "cells": "M(2) x curve(2) x tax(2) = 8",
            "requests_per_cell": "M",
            "replicas": 2,
            "top_k": 2,
            "layers": 2,
            "decode_steps": 1,
            "arrivals": "aligned at 0 us",
            "deadline": "loose at 1000 us",
            "service_curve_us": {"rows_1": 10.0, "rows_2": 14.0, "rows_4": 20.0},
            "atomic_action": "whole-ready membership, one common finish",
            "stream_action": (
                "same whole-ready membership and one launch; fixed physical row "
                "order; row milestones; executor busy through final milestone"
            ),
            "curves": {
                "uniform": "tax + S*i/M",
                "tail": "tax + S*sqrt(i/M)",
            },
            "tax_us": list(TAXES_US),
            "max_states": max_states,
            "objective": [
                "flow_us",
                "total_tardiness_us",
                "deadline_misses",
                "launches",
                "total_service_us",
                "deterministic_action_tokens",
            ],
        },
        "cells": cells,
        "claim_ceiling": CLAIM_CEILING,
    }


def _summary_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# JoinStream Exploratory CPU Exact-Oracle Output",
        "",
        f"- Status: `{payload['status']}`",
        "- Formal authority: `false`",
        "- Interpretation authorized: `false`",
        "- Paper result: `false`",
        "",
        "| Cell | Atomic flow | Expanded flow | Flow delta | Stream launches | Miss delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["cells"]:
        baseline = row["baseline_atomic_exact"]
        expanded = row["expanded_joinstream_exact"]
        diagnostic = row["diagnostic"]
        lines.append(
            "| {cell} | {base:.6f} | {expanded:.6f} | {delta:.6f} | {stream} | {miss} |".format(
                cell=row["episode_id"],
                base=float(baseline["flow_us"]),
                expanded=float(expanded["flow_us"]),
                delta=float(diagnostic["flow_delta_us"]),
                stream=int(diagnostic["stream_launches"]),
                miss=int(diagnostic["deadline_miss_delta"]),
            )
        )
    lines.extend(["", "## Evidence boundary", "", str(payload["claim_ceiling"]), ""])
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: Mapping[str, object]) -> None:
    """Write ordinary exploratory files; deliberately no seal or COMPLETE."""

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "joinstream_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "joinstream_summary.md").write_text(
        _summary_markdown(payload), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-states", type=int, default=FROZEN_MAX_STATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_pilot(max_states=args.max_states)
    write_outputs(Path(args.output_dir), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "formal_authority": False,
                "interpretation_authorized": False,
                "output_dir": args.output_dir,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
