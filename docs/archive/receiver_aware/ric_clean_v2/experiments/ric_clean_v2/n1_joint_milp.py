#!/usr/bin/env python3
"""Pure core for the RIC-Clean-v2 N1 joint two-world MILP.

This module deliberately does not read route, sealed, or result artifacts.  A
future producer must translate frozen calibration inputs into :class:`PairSpec`
instances and bind those inputs separately.  The small static-ready-set model
implemented here is both an executable MILP (via scipy/HiGHS) and an independent
exact-enumeration oracle suitable for Phase-4 fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import time
from typing import Any, Iterable, Mapping, Sequence


STAGES = ("pack", "shared_cut", "unpack", "receiver_apply")
STAGE_INDEX = {stage: index for index, stage in enumerate(STAGES)}
ALPHA = 0.99
MAX_GAP = 1e-6
TOLERANCE = 1e-9


class N1CoreError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ContributionIdentity:
    request_id: str
    layer_id: int
    token_id: int
    token_block_id: int
    topk_slot: int
    expert_id: int
    sender_rank: int
    receiver_rank: int
    epoch: int


@dataclass(frozen=True)
class ServiceTuple:
    pack_us: float
    shared_cut_us: float
    unpack_us: float
    receiver_apply_us: float

    def total(self) -> float:
        return self.pack_us + self.shared_cut_us + self.unpack_us + self.receiver_apply_us

    def for_stage(self, stage: str) -> float:
        return {
            "pack": self.pack_us,
            "shared_cut": self.shared_cut_us,
            "unpack": self.unpack_us,
            "receiver_apply": self.receiver_apply_us,
        }[stage]


@dataclass(frozen=True)
class Task:
    task_id: str
    identity: ContributionIdentity
    join_key: str
    top_k: int
    release_us: float
    payload_bytes: int
    service: ServiceTuple
    join_arrival_us: float = 0.0
    closure_budget_us: float = math.inf

    def immutable_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "identity": self.identity.__dict__,
            "join_key": self.join_key,
            "top_k": self.top_k,
            "release_us": self.release_us,
            "payload_bytes": self.payload_bytes,
            "service": self.service.__dict__,
            "join_arrival_us": self.join_arrival_us,
            "closure_budget_us": self.closure_budget_us,
        }


@dataclass(frozen=True)
class PriorEvent:
    task_id: str
    stage: str
    start_us: float
    end_us: float


@dataclass(frozen=True)
class WorldSpec:
    world_id: str
    prior_history: tuple[PriorEvent, ...]
    expected_join_masks: Mapping[str, int]
    # Receiver-apply completions already queued at the decision point.  Keys are
    # real task IDs; values are their fixed future completion times.
    future_fixed_apply_us: Mapping[str, float]


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    tasks: tuple[Task, ...]
    worlds: tuple[WorldSpec, WorldSpec]
    current_ready_task_ids: tuple[str, ...]
    flip_candidate_task_ids: tuple[str, str]
    target_sender_rank: int
    decision_time_us: float


@dataclass(frozen=True)
class ReplayState:
    join_masks: Mapping[str, int]
    applied_task_ids: tuple[str, ...]
    history_sha256: str


@dataclass(frozen=True)
class Metrics:
    violation_count: int
    cvar99_us: float
    mean_closure_us: float
    closure_latencies_us: tuple[float, ...]
    action_sequence: tuple[str, ...]

    @property
    def objective(self) -> tuple[float, float, float]:
        return (float(self.violation_count), self.cvar99_us, self.mean_closure_us)


@dataclass(frozen=True)
class SolverStage:
    name: str
    status: str
    primal: float
    bound: float
    absolute_gap: float
    relative_gap: float
    solver_name: str
    solver_version: str
    parameters: tuple[tuple[str, Any], ...]
    seed: int | None
    threads: int | None
    solve_time_s: float


@dataclass(frozen=True)
class JointSolution:
    information_level: str
    world_actions: tuple[tuple[str, ...], tuple[str, ...]]
    objective: tuple[float, float, float]
    world_metrics: tuple[Metrics, Metrics]
    first_action_sets: tuple[frozenset[str], frozenset[str]]
    solver_stages: tuple[SolverStage, ...]
    nonanticipativity_matrix_sha256: str
    independent_replay_sha256: str


def _tasks(pair: PairSpec) -> dict[str, Task]:
    result = {task.task_id: task for task in pair.tasks}
    if len(result) != len(pair.tasks):
        raise N1CoreError("BLOCKED_DUPLICATE_TASK_ID")
    identities = [task.identity for task in pair.tasks]
    if len(set(identities)) != len(identities):
        raise N1CoreError("BLOCKED_DUPLICATE_CONTRIBUTION_IDENTITY")
    return result


def _initial_masks(tasks: Iterable[Task]) -> dict[str, int]:
    masks: dict[str, int] = {}
    topks: dict[str, int] = {}
    slots: set[tuple[str, int]] = set()
    join_contracts: dict[str, tuple[float, float]] = {}
    for task in tasks:
        if task.identity.topk_slot < 0 or task.identity.topk_slot >= task.top_k:
            raise N1CoreError("BLOCKED_INVALID_TOPK_SLOT")
        if task.join_key in topks and topks[task.join_key] != task.top_k:
            raise N1CoreError("BLOCKED_INCONSISTENT_TOPK")
        slot_key = (task.join_key, task.identity.topk_slot)
        if slot_key in slots:
            raise N1CoreError("BLOCKED_DUPLICATE_JOIN_SLOT")
        slots.add(slot_key)
        contract = (task.join_arrival_us, task.closure_budget_us)
        if task.join_key in join_contracts and join_contracts[task.join_key] != contract:
            raise N1CoreError("BLOCKED_INCONSISTENT_JOIN_CONTRACT")
        join_contracts[task.join_key] = contract
        topks[task.join_key] = task.top_k
        masks[task.join_key] = (1 << task.top_k) - 1
    return masks


def _resource(task: Task, stage: str) -> str:
    if stage == "pack":
        return f"sender:{task.identity.sender_rank}:pack"
    if stage == "shared_cut":
        return f"cut:{task.identity.sender_rank}->{task.identity.receiver_rank}"
    return f"receiver:{task.identity.receiver_rank}:{stage}"


def replay_prior_history(pair: PairSpec, world: WorldSpec) -> ReplayState:
    tasks = _tasks(pair)
    masks = _initial_masks(tasks.values())
    seen: dict[str, list[str]] = {}
    last_end: dict[str, float] = {}
    intervals: dict[str, list[tuple[float, float, str]]] = {}
    applied: list[str] = []
    previous_key: tuple[float, int, str] | None = None
    combine_enqueued: set[str] = set()

    for event in world.prior_history:
        if event.task_id not in tasks or event.stage not in STAGE_INDEX:
            raise N1CoreError("BLOCKED_INVALID_HISTORY_EVENT")
        task = tasks[event.task_id]
        if event.start_us + TOLERANCE < task.release_us or event.end_us <= event.start_us:
            raise N1CoreError("BLOCKED_UNREACHABLE_HISTORY")
        duration = event.end_us - event.start_us
        if not math.isclose(duration, task.service.for_stage(event.stage), abs_tol=TOLERANCE):
            raise N1CoreError("BLOCKED_HISTORY_SERVICE_DRIFT")
        order_key = (event.end_us, STAGE_INDEX[event.stage], event.task_id)
        if previous_key is not None and order_key < previous_key:
            raise N1CoreError("BLOCKED_NONCANONICAL_HISTORY_ORDER")
        previous_key = order_key
        stages = seen.setdefault(event.task_id, [])
        expected_stage = STAGES[len(stages)] if len(stages) < len(STAGES) else None
        if event.stage != expected_stage:
            raise N1CoreError("BLOCKED_HISTORY_STAGE_PRECEDENCE")
        if stages:
            if event.start_us + TOLERANCE < last_end[event.task_id]:
                raise N1CoreError("BLOCKED_HISTORY_STAGE_PRECEDENCE")
        stages.append(event.stage)
        last_end[event.task_id] = event.end_us
        resource = _resource(task, event.stage)
        for start, end, _ in intervals.setdefault(resource, []):
            if event.start_us < end - TOLERANCE and start < event.end_us - TOLERANCE:
                raise N1CoreError("BLOCKED_HISTORY_RESOURCE_CAPACITY")
        intervals[resource].append((event.start_us, event.end_us, event.task_id))
        if event.stage == "receiver_apply":
            bit = 1 << task.identity.topk_slot
            if not masks[task.join_key] & bit:
                raise N1CoreError("BLOCKED_DUPLICATE_RECEIVER_APPLY")
            masks[task.join_key] &= ~bit
            applied.append(task.task_id)
            if masks[task.join_key] == 0:
                if task.join_key in combine_enqueued:
                    raise N1CoreError("BLOCKED_DUPLICATE_COMBINE")
                combine_enqueued.add(task.join_key)

    if dict(world.expected_join_masks) != masks:
        raise N1CoreError("BLOCKED_REPLAY_SNAPSHOT_MISMATCH")
    return ReplayState(
        join_masks=dict(sorted(masks.items())),
        applied_task_ids=tuple(applied),
        history_sha256=object_sha256([event.__dict__ for event in world.prior_history]),
    )


def _public_task(task: Task) -> dict[str, Any]:
    # No join key, request/token identity, missing mask, or keyed-state lookup.
    return {
        "task_id": task.task_id,
        "sender_rank": task.identity.sender_rank,
        "receiver_rank": task.identity.receiver_rank,
        "payload_bytes": task.payload_bytes,
        "service": task.service.__dict__,
        "release_us": task.release_us,
    }


def observation_fingerprint(pair: PairSpec, world: WorldSpec, level: str) -> str:
    if level not in {"S", "B", "R0"}:
        raise N1CoreError("BLOCKED_INVALID_INFORMATION_LEVEL")
    tasks = _tasks(pair)
    ready = [_public_task(tasks[task_id]) for task_id in pair.current_ready_task_ids]
    sender_events = []
    aggregate_events = []
    for event in world.prior_history:
        task = tasks[event.task_id]
        public = {
            "stage": event.stage,
            "start_us": event.start_us,
            "end_us": event.end_us,
            "sender_rank": task.identity.sender_rank,
            "receiver_rank": task.identity.receiver_rank,
            "payload_bytes": task.payload_bytes,
            "service_us": task.service.for_stage(event.stage),
        }
        if task.identity.sender_rank == pair.target_sender_rank:
            sender_events.append(public)
        aggregate_events.append(public)
    payload: dict[str, Any] = {"ready": ready, "sender_history": sender_events}
    if level in {"B", "R0"}:
        payload["aggregate_history"] = aggregate_events
        pending = [
            {
                "receiver_rank": tasks[task_id].identity.receiver_rank,
                "payload_bytes": tasks[task_id].payload_bytes,
                "service_us": tasks[task_id].service.receiver_apply_us,
                "apply_us": float(apply_us),
            }
            for task_id, apply_us in world.future_fixed_apply_us.items()
        ]
        payload["receiver_pending"] = sorted(
            pending,
            key=lambda row: (
                row["apply_us"], row["receiver_rank"], row["payload_bytes"], row["service_us"]
            ),
        )
    if level == "R0":
        payload["keyed_join_masks"] = replay_prior_history(pair, world).join_masks
    return object_sha256(payload)


def validate_matched_pair(pair: PairSpec) -> tuple[ReplayState, ReplayState]:
    if len(pair.worlds) != 2 or pair.worlds[0].world_id == pair.worlds[1].world_id:
        raise N1CoreError("BLOCKED_INVALID_WORLD_SET")
    tasks = _tasks(pair)
    if not pair.current_ready_task_ids or len(set(pair.current_ready_task_ids)) != len(pair.current_ready_task_ids):
        raise N1CoreError("BLOCKED_INVALID_ACTION_DOMAIN")
    if any(task_id not in tasks for task_id in pair.current_ready_task_ids):
        raise N1CoreError("BLOCKED_INVALID_ACTION_DOMAIN")
    if any(tasks[task_id].identity.sender_rank != pair.target_sender_rank for task_id in pair.current_ready_task_ids):
        raise N1CoreError("BLOCKED_ACTION_DOMAIN_SENDER_DRIFT")
    if (
        len(set(pair.flip_candidate_task_ids)) != 2
        or any(task_id not in pair.current_ready_task_ids for task_id in pair.flip_candidate_task_ids)
    ):
        raise N1CoreError("BLOCKED_INVALID_FLIP_CANDIDATES")
    states = tuple(replay_prior_history(pair, world) for world in pair.worlds)
    if any(mask == 0 for state in states for mask in state.join_masks.values()):
        raise N1CoreError("BLOCKED_CURRENT_STATE_CONTAINS_CLOSED_JOIN")
    for world, state in zip(pair.worlds, states):
        if set(pair.current_ready_task_ids) & set(state.applied_task_ids):
            raise N1CoreError("BLOCKED_READY_TASK_ALREADY_APPLIED")
        if set(pair.current_ready_task_ids) & set(world.future_fixed_apply_us):
            raise N1CoreError("BLOCKED_READY_TASK_HAS_FIXED_APPLY")
    if states[0].join_masks == states[1].join_masks:
        raise N1CoreError("BLOCKED_KEYED_STATE_NOT_DIFFERENT")
    if observation_fingerprint(pair, pair.worlds[0], "S") != observation_fingerprint(pair, pair.worlds[1], "S"):
        raise N1CoreError("BLOCKED_S_VIEW_MISMATCH")
    if observation_fingerprint(pair, pair.worlds[0], "B") != observation_fingerprint(pair, pair.worlds[1], "B"):
        raise N1CoreError("BLOCKED_B_VIEW_MISMATCH")
    if observation_fingerprint(pair, pair.worlds[0], "R0") == observation_fingerprint(pair, pair.worlds[1], "R0"):
        raise N1CoreError("BLOCKED_R0_VIEW_NOT_DISTINCT")
    candidate_tasks = tuple(tasks[task_id] for task_id in pair.flip_candidate_task_ids)
    if candidate_tasks[0].join_key == candidate_tasks[1].join_key:
        raise N1CoreError("BLOCKED_INVALID_FLIP_CANDIDATES")
    status = tuple(
        tuple(
            state.join_masks[task.join_key] == (1 << task.identity.topk_slot)
            for task in candidate_tasks
        )
        for state in states
    )
    if status not in (((True, False), (False, True)), ((False, True), (True, False))):
        raise N1CoreError("BLOCKED_LAST_MISSING_STATUS_NOT_SWAPPED")
    candidate_joins = {task.join_key for task in candidate_tasks}
    if any(
        states[0].join_masks[join] != states[1].join_masks[join]
        for join in states[0].join_masks
        if join not in candidate_joins
    ):
        raise N1CoreError("BLOCKED_NONCANDIDATE_KEYED_STATE_DRIFT")
    for world in pair.worlds:
        history_by_task: dict[str, list[str]] = {}
        for event in world.prior_history:
            history_by_task.setdefault(event.task_id, []).append(event.stage)
        for task_id, apply_us in world.future_fixed_apply_us.items():
            if task_id not in tasks or history_by_task.get(task_id) != list(STAGES[:3]):
                raise N1CoreError("BLOCKED_UNREACHABLE_FUTURE_APPLY")
            if apply_us + TOLERANCE < pair.decision_time_us:
                raise N1CoreError("BLOCKED_UNREACHABLE_FUTURE_APPLY")
    return states  # type: ignore[return-value]


def empirical_cvar(samples: Sequence[float], alpha: float = ALPHA) -> float:
    if not samples or not 0.0 < alpha < 1.0:
        raise N1CoreError("BLOCKED_INVALID_CVAR_INPUT")
    values = tuple(float(value) for value in samples)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise N1CoreError("BLOCKED_INVALID_CVAR_SAMPLE")
    return min(
        z + sum(max(0.0, value - z) for value in values) / ((1.0 - alpha) * len(values))
        for z in sorted(set(values))
    )


def replay_action_sequence(pair: PairSpec, world: WorldSpec, actions: Sequence[str]) -> Metrics:
    tasks = _tasks(pair)
    if set(actions) != set(pair.current_ready_task_ids) or len(actions) != len(pair.current_ready_task_ids):
        raise N1CoreError("BLOCKED_ACTION_SEQUENCE_NOT_FULL_DRAIN")
    state = replay_prior_history(pair, world)
    masks = dict(state.join_masks)
    events: list[tuple[float, str]] = []
    now = pair.decision_time_us
    for task_id in actions:
        task = tasks[task_id]
        now = max(now, task.release_us) + task.service.total()
        events.append((now, task_id))
    events.extend((float(time_us), task_id) for task_id, time_us in world.future_fixed_apply_us.items())
    closures: dict[str, float] = {}
    for time_us, task_id in sorted(events, key=lambda row: (row[0], row[1])):
        task = tasks[task_id]
        bit = 1 << task.identity.topk_slot
        if masks[task.join_key] & bit:
            masks[task.join_key] &= ~bit
            if masks[task.join_key] == 0:
                closures[task.join_key] = time_us
    if any(mask != 0 for mask in masks.values()) or set(closures) != set(masks):
        raise N1CoreError("BLOCKED_ACTION_REPLAY_NOT_FULL_DRAIN")
    join_task = {task.join_key: task for task in tasks.values()}
    latencies = tuple(
        closures[join] - join_task[join].join_arrival_us for join in sorted(closures)
    )
    violations = sum(
        closure > join_task[join].closure_budget_us + TOLERANCE
        for join, closure in closures.items()
    )
    return Metrics(
        violation_count=violations,
        cvar99_us=empirical_cvar(latencies),
        mean_closure_us=sum(latencies) / len(latencies),
        closure_latencies_us=latencies,
        action_sequence=tuple(actions),
    )


def _policy_rows(pair: PairSpec, level: str) -> list[tuple[tuple[str, ...], tuple[str, ...], Metrics, Metrics]]:
    validate_matched_pair(pair)
    permutations = tuple(itertools.permutations(pair.current_ready_task_ids))
    if level in {"S", "B"}:
        policy_actions = ((actions, actions) for actions in permutations)
    elif level == "R0":
        policy_actions = itertools.product(permutations, permutations)
    elif level == "C":
        raise N1CoreError("BLOCKED_C_NOT_IMPLEMENTED_IN_PURE_CORE")
    else:
        raise N1CoreError("BLOCKED_INVALID_INFORMATION_LEVEL")
    return [
        (
            actions0,
            actions1,
            replay_action_sequence(pair, pair.worlds[0], actions0),
            replay_action_sequence(pair, pair.worlds[1], actions1),
        )
        for actions0, actions1 in policy_actions
    ]


def _joint_objective(metrics0: Metrics, metrics1: Metrics) -> tuple[float, float, float]:
    samples = metrics0.closure_latencies_us + metrics1.closure_latencies_us
    return (
        float(metrics0.violation_count + metrics1.violation_count),
        empirical_cvar(samples),
        (sum(metrics0.closure_latencies_us) + sum(metrics1.closure_latencies_us)) / len(samples),
    )


def solve_independent_enumeration(pair: PairSpec, level: str) -> JointSolution:
    rows = _policy_rows(pair, level)
    objectives = [_joint_objective(row[2], row[3]) for row in rows]
    optimum = min(objectives)
    optimal_indexes = [index for index, value in enumerate(objectives) if _same_objective(value, optimum)]
    chosen = rows[optimal_indexes[0]]
    first_sets = tuple(
        frozenset(rows[index][world][0] for index in optimal_indexes) for world in (0, 1)
    )
    matrix = {
        "level": level,
        "joint_world": True,
        "equal_full_schedule": level in {"S", "B"},
        "S": [observation_fingerprint(pair, world, "S") for world in pair.worlds],
        "B": [observation_fingerprint(pair, world, "B") for world in pair.worlds],
    }
    replay_payload = [
        {"actions": row[:2], "objective": objective}
        for row, objective in zip(rows, objectives)
    ]
    return JointSolution(
        information_level=level,
        world_actions=(chosen[0], chosen[1]),
        objective=optimum,
        world_metrics=(chosen[2], chosen[3]),
        first_action_sets=first_sets,  # type: ignore[arg-type]
        solver_stages=tuple(
            SolverStage(
                name,
                "OPTIMAL",
                value,
                value,
                0.0,
                0.0,
                "independent_exact_enumeration",
                "builtin-v1",
                tuple(),
                0,
                1,
                0.0,
            )
            for name, value in zip(("violations", "cvar99", "mean"), optimum)
        ),
        nonanticipativity_matrix_sha256=object_sha256(matrix),
        independent_replay_sha256=object_sha256(replay_payload),
    )


def _same_objective(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(math.isclose(a, b, rel_tol=0.0, abs_tol=TOLERANCE) for a, b in zip(left, right))


def _scipy_solve_rows(
    rows: Sequence[tuple[tuple[str, ...], tuple[str, ...], Metrics, Metrics]],
    *,
    fixed_first: tuple[int, str] | None = None,
) -> tuple[int, tuple[float, float, float], tuple[SolverStage, ...]]:
    try:
        import numpy as np
        from scipy import __version__ as scipy_version
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:
        raise N1CoreError("BLOCKED_SOLVER_UNAVAILABLE: scipy.optimize.milp required") from exc

    eligible = list(range(len(rows)))
    if fixed_first is not None:
        world, task_id = fixed_first
        eligible = [index for index in eligible if rows[index][world][0] == task_id]
    if not eligible:
        raise N1CoreError("BLOCKED_FIXED_ACTION_INFEASIBLE")
    selected_rows = [rows[index] for index in eligible]
    sample_count = len(selected_rows[0][2].closure_latencies_us + selected_rows[0][3].closure_latencies_us)
    policy_count = len(selected_rows)
    variable_count = policy_count + 1 + sample_count
    z_index = policy_count
    u_start = z_index + 1
    lower = np.zeros(variable_count)
    lower[z_index] = 0.0
    upper = np.full(variable_count, np.inf)
    upper[:policy_count] = 1.0
    integrality = np.zeros(variable_count)
    integrality[:policy_count] = 1
    base_constraints: list[Any] = [
        LinearConstraint(np.r_[np.ones(policy_count), np.zeros(1 + sample_count)], 1.0, 1.0)
    ]
    for sample_index in range(sample_count):
        coefficients = np.zeros(variable_count)
        for policy_index, row in enumerate(selected_rows):
            samples = row[2].closure_latencies_us + row[3].closure_latencies_us
            coefficients[policy_index] = samples[sample_index]
        coefficients[z_index] = -1.0
        coefficients[u_start + sample_index] = -1.0
        base_constraints.append(LinearConstraint(coefficients, -np.inf, 0.0))
    violation_coeff = np.zeros(variable_count)
    mean_coeff = np.zeros(variable_count)
    for index, row in enumerate(selected_rows):
        objective = _joint_objective(row[2], row[3])
        violation_coeff[index] = objective[0]
        mean_coeff[index] = objective[2]
    cvar_coeff = np.zeros(variable_count)
    cvar_coeff[z_index] = 1.0
    cvar_coeff[u_start:] = 1.0 / ((1.0 - ALPHA) * sample_count)

    stages: list[SolverStage] = []
    constraints = list(base_constraints)
    optimum_values: list[float] = []
    result = None
    for name, objective_coeff in (
        ("violations", violation_coeff),
        ("cvar99", cvar_coeff),
        ("mean", mean_coeff),
    ):
        started = time.perf_counter()
        result = milp(
            c=objective_coeff,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={"mip_rel_gap": MAX_GAP, "presolve": True},
        )
        elapsed = time.perf_counter() - started
        if not result.success or result.status != 0 or result.fun is None:
            raise N1CoreError(f"BLOCKED_SOLVER_NOT_OPTIMAL:{name}")
        primal = float(result.fun)
        bound = float(getattr(result, "mip_dual_bound", primal))
        absolute_gap = abs(primal - bound)
        raw_relative_gap = getattr(result, "mip_gap", None)
        relative_gap = (
            absolute_gap / max(1.0, abs(primal))
            if raw_relative_gap is None
            else float(raw_relative_gap)
        )
        if relative_gap > MAX_GAP + TOLERANCE or (abs(primal) <= TOLERANCE and absolute_gap > MAX_GAP):
            raise N1CoreError(f"BLOCKED_SOLVER_GAP:{name}")
        stages.append(
            SolverStage(
                name,
                "OPTIMAL",
                primal,
                bound,
                absolute_gap,
                relative_gap,
                "scipy.optimize.milp/HiGHS",
                str(scipy_version),
                (("mip_rel_gap", MAX_GAP), ("presolve", True)),
                None,
                None,
                elapsed,
            )
        )
        optimum_values.append(primal)
        constraints.append(LinearConstraint(objective_coeff, primal - TOLERANCE, primal + TOLERANCE))
    assert result is not None and result.x is not None
    local_index = int(np.argmax(result.x[:policy_count]))
    original_index = eligible[local_index]
    chosen = rows[original_index]
    exact = _joint_objective(chosen[2], chosen[3])
    if not _same_objective(exact, optimum_values):
        raise N1CoreError("BLOCKED_SOLVER_REPLAY_MISMATCH")
    return original_index, exact, tuple(stages)


def solve_joint_milp(pair: PairSpec, level: str) -> JointSolution:
    rows = _policy_rows(pair, level)
    chosen_index, objective, stages = _scipy_solve_rows(rows)
    reference = solve_independent_enumeration(pair, level)
    if not _same_objective(objective, reference.objective):
        raise N1CoreError("BLOCKED_MILP_ENUMERATION_MISMATCH")
    first_sets: list[frozenset[str]] = []
    for world in (0, 1):
        optimal: set[str] = set()
        for task_id in pair.current_ready_task_ids:
            try:
                _, fixed_objective, _ = _scipy_solve_rows(rows, fixed_first=(world, task_id))
            except N1CoreError as exc:
                if "FIXED_ACTION_INFEASIBLE" in str(exc):
                    continue
                raise
            if _same_objective(fixed_objective, objective):
                optimal.add(task_id)
        first_sets.append(frozenset(optimal))
    chosen = rows[chosen_index]
    result = JointSolution(
        information_level=level,
        world_actions=(chosen[0], chosen[1]),
        objective=objective,
        world_metrics=(chosen[2], chosen[3]),
        first_action_sets=(first_sets[0], first_sets[1]),
        solver_stages=stages,
        nonanticipativity_matrix_sha256=reference.nonanticipativity_matrix_sha256,
        independent_replay_sha256=reference.independent_replay_sha256,
    )
    if level == "R0" and (len(result.first_action_sets[0]) != 1 or len(result.first_action_sets[1]) != 1):
        raise N1CoreError("BLOCKED_AMBIGUOUS_FIRST_ACTION_OPTIMUM")
    return result


def r0_flip(solution: JointSolution) -> bool:
    if solution.information_level != "R0":
        raise N1CoreError("BLOCKED_FLIP_REQUIRES_R0")
    if len(solution.first_action_sets[0]) != 1 or len(solution.first_action_sets[1]) != 1:
        raise N1CoreError("BLOCKED_AMBIGUOUS_FIRST_ACTION_OPTIMUM")
    return solution.first_action_sets[0] != solution.first_action_sets[1]
