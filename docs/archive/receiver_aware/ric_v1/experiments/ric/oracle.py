"""Exact fixed-route matched-world information oracle for RIC-v1.

This is a controlled two-world necessity fixture, not a workload-tail model.
Both worlds expose the same two sender-local actions on one frozen
sender-egress/shared-cut/receiver-ingress path.  They differ only in which
application join is last-missing.  A once-only receiver combine follows the
closing contribution's unpack. S/B therefore share one permutation;
R0/C may select a world-specific permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


class OracleError(RuntimeError):
    """Fail-closed exact-oracle error."""


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    join_fingerprint: str
    release_us: float
    sender_rank: int
    receiver_rank: int
    sender_egress_resource: str
    shared_cut_resource: str
    receiver_ingress_resource: str
    receiver_combine_resource: str
    sender_egress_us: float
    shared_cut_us: float
    receiver_ingress_us: float
    join_combine_us: float
    payload_bytes: int
    deadline_us: float

    def __post_init__(self) -> None:
        if not self.task_id or not self.join_fingerprint or self.payload_bytes <= 0:
            raise OracleError("invalid public task identity/bytes")
        if self.sender_rank < 0 or self.receiver_rank < 0:
            raise OracleError("invalid public task rank")
        if not all(
            isinstance(value, str) and value
            for value in (*self.stage_resources, self.receiver_combine_resource)
        ):
            raise OracleError("public task has an incomplete fixed resource path")
        numeric = (
            self.release_us,
            self.sender_egress_us,
            self.shared_cut_us,
            self.receiver_ingress_us,
            self.join_combine_us,
            self.deadline_us,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise OracleError("public task timing is non-finite")
        if self.release_us < 0 or any(
            value <= 0 for value in (*self.stage_service_us, self.join_combine_us)
        ):
            raise OracleError("public task release/service is invalid")
        if self.deadline_us <= self.release_us:
            raise OracleError("public task deadline precedes release")

    @property
    def stage_resources(self) -> tuple[str, str, str]:
        return (
            self.sender_egress_resource,
            self.shared_cut_resource,
            self.receiver_ingress_resource,
        )

    @property
    def stage_service_us(self) -> tuple[float, float, float]:
        return (
            self.sender_egress_us,
            self.shared_cut_us,
            self.receiver_ingress_us,
        )

    @property
    def service_us(self) -> float:
        return math.fsum(self.stage_service_us)

    @property
    def closure_path_us(self) -> float:
        return self.service_us + self.join_combine_us


@dataclass(frozen=True)
class MatchedWorld:
    world_name: str
    closing_task_id: str
    hidden_join_fingerprint: str

    def __post_init__(self) -> None:
        if not self.world_name or not self.closing_task_id or not self.hidden_join_fingerprint:
            raise OracleError("matched-world identity is incomplete")


@dataclass(frozen=True)
class MatchedPair:
    pair_id: str
    model_key: str
    tasks: tuple[PublicTask, ...]
    worlds: tuple[MatchedWorld, ...]
    closure_budget_us: float
    starvation_us: float
    aggregate_receiver_qdepth: int = 0
    aggregate_shared_cut_backlog_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.pair_id or not self.model_key:
            raise OracleError("matched pair identity is incomplete")
        if len(self.tasks) != 2 or len(self.worlds) != 2:
            raise OracleError("matched pair requires exactly two tasks and two worlds")
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(set(task_ids)) != 2:
            raise OracleError("duplicate task id")
        closing_ids = tuple(world.closing_task_id for world in self.worlds)
        if set(closing_ids) != set(task_ids) or len(set(closing_ids)) != 2:
            raise OracleError("two worlds must swap the two distinct last-missing joins")
        if len({world.world_name for world in self.worlds}) != 2:
            raise OracleError("duplicate world name")
        if len({world.hidden_join_fingerprint for world in self.worlds}) != 2:
            raise OracleError("worlds must reference distinct application joins")
        join_by_task = {task.task_id: task.join_fingerprint for task in self.tasks}
        if any(
            world.hidden_join_fingerprint != join_by_task[world.closing_task_id]
            for world in self.worlds
        ):
            raise OracleError("last-missing state is bound to the wrong application join")
        if self.closure_budget_us <= 0 or not math.isfinite(self.closure_budget_us):
            raise OracleError("invalid closure budget")
        if self.starvation_us < 0 or not math.isfinite(self.starvation_us):
            raise OracleError("invalid starvation bound")
        if {task.release_us for task in self.tasks} != {0.0}:
            raise OracleError("matched-world counterfactual requires common release_us=0")
        if len({task.sender_rank for task in self.tasks}) != 1:
            raise OracleError("matched tasks do not share the exact sender")
        if len({task.receiver_rank for task in self.tasks}) != 1:
            raise OracleError("matched tasks do not share the exact receiver")
        if len({task.stage_resources for task in self.tasks}) != 1 or len(
            {task.receiver_combine_resource for task in self.tasks}
        ) != 1:
            raise OracleError("matched tasks do not share the exact four-resource path")
        if len({task.stage_service_us for task in self.tasks}) != 1 or len(
            {task.join_combine_us for task in self.tasks}
        ) != 1:
            raise OracleError("matched tasks do not share contribution/combine service")
        if len({task.payload_bytes for task in self.tasks}) != 1:
            raise OracleError("matched tasks do not have equal payload bytes")
        if len({self.public_observation_signature(world) for world in self.worlds}) != 1:
            raise OracleError("matched worlds are not public-information equivalent")

    def public_observation_signature(self, _world: MatchedWorld) -> str:
        """Canonical B-level observation, excluding keyed application join state."""

        value = {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "join_fingerprint": task.join_fingerprint,
                    "release_us": task.release_us,
                    "sender_rank": task.sender_rank,
                    "receiver_rank": task.receiver_rank,
                    "stage_resources": task.stage_resources,
                    "stage_service_us": task.stage_service_us,
                    "receiver_combine_resource": task.receiver_combine_resource,
                    "join_combine_us": task.join_combine_us,
                    "payload_bytes": task.payload_bytes,
                    "deadline_us": task.deadline_us,
                }
                for task in sorted(self.tasks, key=lambda item: item.task_id)
            ],
            "receiver_qdepth": self.aggregate_receiver_qdepth,
            "shared_cut_backlog_bytes": self.aggregate_shared_cut_backlog_bytes,
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def observation_history_signature(
    pair: MatchedPair, world: MatchedWorld, information_level: str
) -> str:
    """Build one sender-local observation-history signature without outcomes.

    S contains sender-local ready/action facts.  B adds aggregate public
    receiver/cut state.  R0/C additionally expose the keyed last-missing join
    state used by this controlled necessity fixture.  The grouping code below
    is deliberately independent of the optimizer.
    """

    if information_level not in {"S", "B", "R0", "C"}:
        raise OracleError(f"unknown information level {information_level}")
    if world not in pair.worlds:
        raise OracleError("observation-history world is not in matched pair")
    sender_value: dict[str, object] = {
        "sender_rank": pair.tasks[0].sender_rank,
        "tasks": [
            {
                "task_id": task.task_id,
                "release_us": task.release_us,
                "sender_egress_resource": task.sender_egress_resource,
                "sender_egress_us": task.sender_egress_us,
                "payload_bytes": task.payload_bytes,
                "deadline_us": task.deadline_us,
            }
            for task in sorted(pair.tasks, key=lambda item: item.task_id)
        ],
    }
    if information_level == "S":
        value: object = sender_value
    elif information_level == "B":
        value = {
            "sender": sender_value,
            "public_receiver_cut": pair.public_observation_signature(world),
        }
    else:
        value = {
            "sender": sender_value,
            "public_receiver_cut": pair.public_observation_signature(world),
            "keyed_receiver_state": {
                "closing_task_id": world.closing_task_id,
                "hidden_join_fingerprint": world.hidden_join_fingerprint,
            },
        }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_observation_history_nodes(
    pair: MatchedPair, information_level: str
) -> dict[str, int]:
    """Group worlds into deterministic per-sender nonanticipativity nodes."""

    signatures = {
        world.world_name: observation_history_signature(pair, world, information_level)
        for world in pair.worlds
    }
    signature_to_node = {
        signature: index for index, signature in enumerate(sorted(set(signatures.values())))
    }
    return {
        world_name: signature_to_node[signature]
        for world_name, signature in signatures.items()
    }


def sender_ready_wait_by_position_us(
    sender_egress_us: float, task_count: int
) -> tuple[float, ...]:
    """Ready wait before sender-egress service for the fixed common-release fixture."""

    if not math.isfinite(sender_egress_us) or sender_egress_us <= 0:
        raise OracleError("sender service must be positive and finite")
    if type(task_count) is not int or task_count <= 0:
        raise OracleError("task_count must be positive")
    return tuple(position * sender_egress_us for position in range(task_count))


def _validate_starvation_support(pair: MatchedPair) -> tuple[float, ...]:
    waits = sender_ready_wait_by_position_us(
        pair.tasks[0].sender_egress_us, len(pair.tasks)
    )
    if max(waits) > pair.starvation_us + 1e-9:
        raise OracleError(
            "BLOCKED_STARVATION_INFEASIBLE: frozen sender ready-wait bound "
            "cannot schedule every fixed task"
        )
    return waits


def _candidate_orders_by_world(
    pair: MatchedPair, information_level: str
) -> Iterable[tuple[tuple[str, ...], ...]]:
    """Enumerate actions per independently built observation-history node."""

    task_ids = tuple(sorted(task.task_id for task in pair.tasks))
    nodes = build_observation_history_nodes(pair, information_level)
    node_ids = tuple(sorted(set(nodes.values())))
    node_slot = {node_id: index for index, node_id in enumerate(node_ids)}
    orders = tuple(permutations(task_ids))
    for orders_by_node in product(orders, repeat=len(node_ids)):
        yield tuple(
            orders_by_node[node_slot[nodes[world.world_name]]]
            for world in pair.worlds
        )


@dataclass(frozen=True)
class OracleResult:
    information_level: str
    orders_by_world: Mapping[str, tuple[str, ...]]
    closure_by_world_us: Mapping[str, float]
    violation_count: int
    empirical_cvar99_us: float
    mean_closure_us: float
    first_action_flip_rate: float
    optimal_first_actions_by_world: Mapping[str, tuple[str, ...]]
    unique_optimal_first_action: bool
    position_completion_us: tuple[float, ...]
    solver: str
    solver_status: str
    mip_gap: float
    nonanticipativity_nodes: int


def fixed_route_flowshop_position_completions(
    stage_service_us: Sequence[float],
    task_count: int,
    *,
    common_release_us: float = 0.0,
) -> tuple[float, ...]:
    """Exact three-resource completion at each permutation position.

    Every task has the same route and service tuple.  The recurrence enforces
    release, precedence, and one non-preemptive task at a time on each of the
    sender-egress, shared-cut, and receiver-ingress resources.
    """

    stages = tuple(float(value) for value in stage_service_us)
    if len(stages) != 3 or any(not math.isfinite(value) or value <= 0 for value in stages):
        raise OracleError("flow-shop requires three positive finite stage services")
    if type(task_count) is not int or task_count <= 0:
        raise OracleError("flow-shop task_count must be positive")
    if not math.isfinite(common_release_us) or common_release_us < 0:
        raise OracleError("flow-shop common release is invalid")
    resource_available = [float(common_release_us)] * 3
    completions: list[float] = []
    for _position in range(task_count):
        predecessor_completion = float(common_release_us)
        for stage, service in enumerate(stages):
            start = max(predecessor_completion, resource_available[stage])
            predecessor_completion = start + service
            resource_available[stage] = predecessor_completion
        completions.append(predecessor_completion)
    return tuple(completions)


def literal_fcfs_capacity_position_completions(
    stage_service_us: Sequence[float],
    task_count: int,
    *,
    common_release_us: float = 0.0,
) -> tuple[float, ...]:
    """Independent LP for the frozen work-conserving FCFS/no-overtake path.

    Variables are literal task-stage start times.  Precedence and same-order
    resource-capacity inequalities are separate from the recurrence used by
    the optimizer.  Minimizing all starts gives the work-conserving schedule.
    """

    stages = tuple(float(value) for value in stage_service_us)
    if len(stages) != 3 or any(not math.isfinite(value) or value <= 0 for value in stages):
        raise OracleError("literal capacity model requires three positive services")
    if type(task_count) is not int or task_count <= 0:
        raise OracleError("literal capacity task_count must be positive")
    if not math.isfinite(common_release_us) or common_release_us < 0:
        raise OracleError("literal capacity release is invalid")
    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - environment capability
        raise OracleError("BLOCKED_SOLVER_UNAVAILABLE: scipy.optimize.linprog is required") from exc

    stage_count = len(stages)
    variable_count = task_count * stage_count

    def index(task: int, stage: int) -> int:
        return task * stage_count + stage

    a_ub: list[Any] = []
    b_ub: list[float] = []
    for task in range(task_count):
        release_row = np.zeros(variable_count)
        release_row[index(task, 0)] = -1.0
        a_ub.append(release_row)
        b_ub.append(-common_release_us)
        for stage in range(1, stage_count):
            precedence = np.zeros(variable_count)
            precedence[index(task, stage - 1)] = 1.0
            precedence[index(task, stage)] = -1.0
            a_ub.append(precedence)
            b_ub.append(-stages[stage - 1])
    for task in range(task_count - 1):
        for stage in range(stage_count):
            capacity = np.zeros(variable_count)
            capacity[index(task, stage)] = 1.0
            capacity[index(task + 1, stage)] = -1.0
            a_ub.append(capacity)
            b_ub.append(-stages[stage])
    objective = np.ones(variable_count)
    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=[(0.0, None)] * variable_count,
        method="highs",
    )
    if not result.success or result.x is None:
        raise OracleError("BLOCKED_LITERAL_CAPACITY_MODEL_INFEASIBLE")
    return tuple(
        float(result.x[index(task, stage_count - 1)] + stages[-1])
        for task in range(task_count)
    )


def empirical_cvar99_two_world(closures_us: Sequence[float]) -> float:
    """Exact empirical CVaR99 for two equiprobable matched worlds."""

    values = tuple(float(value) for value in closures_us)
    if len(values) != 2 or any(not math.isfinite(value) or value < 0 for value in values):
        raise OracleError("empirical CVaR99 requires two finite non-negative closures")
    # Each observation has mass 0.5, while the upper tail has mass 0.01.
    # Therefore the entire empirical upper 1% lies within the maximum atom.
    return max(values)


def _optimal_first_action_sets(
    pair: MatchedPair,
    information_level: str,
    objective: tuple[int, float, float],
    completion_by_position: Sequence[float],
) -> dict[str, tuple[str, ...]]:
    """Enumerate objective-equivalent first actions; solver ties are not evidence."""

    candidates = _candidate_orders_by_world(pair, information_level)
    actions = {world.world_name: set() for world in pair.worlds}
    for orders in candidates:
        closures = tuple(
            completion_by_position[orders[index].index(world.closing_task_id)]
            for index, world in enumerate(pair.worlds)
        )
        observed = (
            sum(value > pair.closure_budget_us + 1e-9 for value in closures),
            empirical_cvar99_two_world(closures),
            sum(closures) / len(closures),
        )
        if observed == objective:
            for index, world in enumerate(pair.worlds):
                actions[world.world_name].add(orders[index][0])
    if any(not values for values in actions.values()):
        raise OracleError("optimal first-action set is empty")
    return {key: tuple(sorted(values)) for key, values in actions.items()}


def _set_flip_rate(action_sets: Mapping[str, tuple[str, ...]]) -> tuple[float, bool]:
    values = tuple(action_sets.values())
    unique = all(len(value) == 1 for value in values)
    if not unique:
        return 0.0, False
    return (
        sum(value != values[0] for value in values[1:])
        / max(1, len(values) - 1),
        True,
    )


def synthetic_matched_pair(
    model_key: str,
    pair_index: int,
    *,
    service_us: float,
    payload_bytes: int,
) -> MatchedPair:
    """Create a non-scientific three-stage-contribution plus combine fixture."""

    if service_us <= 0:
        raise OracleError("service_us must be measured and positive")
    prefix = hashlib.sha256(f"ric-v1:{model_key}:{pair_index}".encode()).hexdigest()[:12]
    task_ids = (f"{prefix}-x", f"{prefix}-y")
    stages = (service_us, service_us, service_us)
    contribution_completions = fixed_route_flowshop_position_completions(stages, 2)
    combine_us = service_us
    first_completion, second_completion = tuple(
        value + combine_us for value in contribution_completions
    )
    tasks = tuple(
        PublicTask(
            task_id=task_id,
            join_fingerprint=f"private-{prefix}-{'a' if task_id == task_ids[0] else 'b'}",
            release_us=0.0,
            sender_rank=0,
            receiver_rank=0,
            sender_egress_resource="sender:0:egress",
            shared_cut_resource="cut:node0->node0",
            receiver_ingress_resource="receiver:0:ingress",
            receiver_combine_resource="receiver:0:combine",
            sender_egress_us=stages[0],
            shared_cut_us=stages[1],
            receiver_ingress_us=stages[2],
            join_combine_us=combine_us,
            payload_bytes=payload_bytes,
            deadline_us=second_completion,
        )
        for task_id in task_ids
    )
    worlds = (
        MatchedWorld("world-a", task_ids[0], f"private-{prefix}-a"),
        MatchedWorld("world-b", task_ids[1], f"private-{prefix}-b"),
    )
    return MatchedPair(
        pair_id=f"{model_key}-{pair_index:04d}",
        model_key=model_key,
        tasks=tasks,
        worlds=worlds,
        closure_budget_us=(first_completion + second_completion) / 2.0,
        starvation_us=4.0 * service_us,
    )


def _scipy_modules() -> tuple[Any, str, Any, Any, Any]:
    try:
        import numpy as np
        from scipy import __version__ as scipy_version
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:  # pragma: no cover - environment capability
        raise OracleError("BLOCKED_SOLVER_UNAVAILABLE: scipy.optimize.milp is required") from exc
    return np, scipy_version, Bounds, LinearConstraint, milp


def _solve_once(
    c: Any,
    *,
    np: Any,
    Bounds: Any,
    milp: Any,
    integrality: Any,
    lower: Any,
    upper: Any,
    constraints: Sequence[Any],
) -> object:
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=list(constraints),
        options={"mip_rel_gap": 1e-9, "presolve": True},
    )
    gap = float(getattr(result, "mip_gap", math.inf))
    if not bool(result.success) or result.x is None or gap > 0.01:
        raise OracleError(
            f"BLOCKED_SOLVER_INEXACT status={result.status} message={result.message!s} gap={gap}"
        )
    return result


def solve_matched_pair(pair: MatchedPair, information_level: str) -> OracleResult:
    """Lexicographically solve S/B or R0/C on the exact fixed-route fixture."""

    if information_level not in {"S", "B", "R0", "C"}:
        raise OracleError(f"unknown information level {information_level}")
    np, scipy_version, Bounds, LinearConstraint, milp = _scipy_modules()
    tasks = tuple(sorted(pair.tasks, key=lambda item: item.task_id))
    worlds = pair.worlds
    n_task = len(tasks)
    n_world = len(worlds)
    contribution_completion_by_position = fixed_route_flowshop_position_completions(
        tasks[0].stage_service_us,
        n_task,
        common_release_us=tasks[0].release_us,
    )
    literal_completion_by_position = literal_fcfs_capacity_position_completions(
        tasks[0].stage_service_us,
        n_task,
        common_release_us=tasks[0].release_us,
    )
    if any(
        not math.isclose(observed, literal, rel_tol=1e-10, abs_tol=1e-9)
        for observed, literal in zip(
            contribution_completion_by_position, literal_completion_by_position
        )
    ):
        raise OracleError("fixed-route recurrence/literal capacity model mismatch")
    completion_by_position = tuple(
        value + tasks[0].join_combine_us
        for value in contribution_completion_by_position
    )
    task_index = {task.task_id: index for index, task in enumerate(tasks)}
    node_by_world = build_observation_history_nodes(pair, information_level)
    node_ids = tuple(sorted(set(node_by_world.values())))
    node_slot = {node_id: index for index, node_id in enumerate(node_ids)}
    world_scope = {
        index: node_slot[node_by_world[world.world_name]]
        for index, world in enumerate(worlds)
    }
    x_worlds = len(node_ids)
    n_x = x_worlds * n_task * n_task
    violation_offset = n_x
    cvar_index = violation_offset + n_world
    n_var = cvar_index + 1

    def x_index(world_index: int, task: int, position: int) -> int:
        scope = world_scope[world_index]
        return (scope * n_task + task) * n_task + position

    integrality = np.zeros(n_var, dtype=int)
    integrality[:n_x] = 1
    integrality[violation_offset:cvar_index] = 1
    lower = np.zeros(n_var, dtype=float)
    upper = np.ones(n_var, dtype=float)
    upper[cvar_index] = completion_by_position[-1]
    constraints: list[Any] = []

    # Each permutation executes exactly the same public sender-local actions:
    # one task per position and each task exactly once.  The fixed-route
    # recurrence above supplies exact three-resource position completions.
    representative_by_scope = {
        scope: next(index for index, value in world_scope.items() if value == scope)
        for scope in range(x_worlds)
    }
    waits_by_position = _validate_starvation_support(pair)
    for scope in range(x_worlds):
        representative_world = representative_by_scope[scope]
        for task in range(n_task):
            row = np.zeros(n_var)
            for position in range(n_task):
                row[x_index(representative_world, task, position)] = 1
            constraints.append(LinearConstraint(row, 1, 1))
            starvation_row = np.zeros(n_var)
            for position, wait_us in enumerate(waits_by_position):
                starvation_row[x_index(representative_world, task, position)] = wait_us
            constraints.append(
                LinearConstraint(starvation_row, -np.inf, pair.starvation_us)
            )
        for position in range(n_task):
            row = np.zeros(n_var)
            for task in range(n_task):
                row[x_index(representative_world, task, position)] = 1
            constraints.append(LinearConstraint(row, 1, 1))

    big_m = completion_by_position[-1]
    for world_index, world in enumerate(worlds):
        closing = task_index[world.closing_task_id]
        completion = np.zeros(n_var)
        for position, completion_us in enumerate(completion_by_position):
            completion[x_index(world_index, closing, position)] = completion_us
        violation = completion.copy()
        violation[violation_offset + world_index] = -big_m
        constraints.append(LinearConstraint(violation, -np.inf, pair.closure_budget_us))
        cvar_bound = completion.copy()
        cvar_bound[cvar_index] = -1
        constraints.append(LinearConstraint(cvar_bound, -np.inf, 0))

    objective = np.zeros(n_var)
    objective[violation_offset:cvar_index] = 1
    first = _solve_once(
        objective,
        np=np,
        Bounds=Bounds,
        milp=milp,
        integrality=integrality,
        lower=lower,
        upper=upper,
        constraints=constraints,
    )
    violation_optimum = round(float(np.sum(first.x[violation_offset:cvar_index])))
    violation_row = np.zeros(n_var)
    violation_row[violation_offset:cvar_index] = 1
    constraints.append(LinearConstraint(violation_row, -np.inf, violation_optimum + 1e-7))

    # With exactly two equiprobable worlds, this epigraph is exact empirical
    # CVaR99, not a workload CVaR or a tail-latency proxy.
    objective = np.zeros(n_var)
    objective[cvar_index] = 1
    second = _solve_once(
        objective,
        np=np,
        Bounds=Bounds,
        milp=milp,
        integrality=integrality,
        lower=lower,
        upper=upper,
        constraints=constraints,
    )
    cvar_optimum = float(second.x[cvar_index])
    cvar_row = np.zeros(n_var)
    cvar_row[cvar_index] = 1
    constraints.append(LinearConstraint(cvar_row, -np.inf, cvar_optimum + 1e-7))

    mean_objective = np.zeros(n_var)
    for world_index, world in enumerate(worlds):
        closing = task_index[world.closing_task_id]
        for position, completion_us in enumerate(completion_by_position):
            mean_objective[x_index(world_index, closing, position)] += completion_us / n_world
    third = _solve_once(
        mean_objective,
        np=np,
        Bounds=Bounds,
        milp=milp,
        integrality=integrality,
        lower=lower,
        upper=upper,
        constraints=constraints,
    )
    mean_optimum = float(mean_objective @ third.x)
    constraints.append(LinearConstraint(mean_objective, -np.inf, mean_optimum + 1e-7))

    stable = np.zeros(n_var)
    for world_index in range(n_world):
        for task in range(n_task):
            for position in range(n_task):
                stable[x_index(world_index, task, position)] += (task + 1) * position
    final = _solve_once(
        stable,
        np=np,
        Bounds=Bounds,
        milp=milp,
        integrality=integrality,
        lower=lower,
        upper=upper,
        constraints=constraints,
    )

    orders: dict[str, tuple[str, ...]] = {}
    closures: dict[str, float] = {}
    for world_index, world in enumerate(worlds):
        positioned: list[tuple[int, str]] = []
        for task_index_value, task in enumerate(tasks):
            values = [
                final.x[x_index(world_index, task_index_value, position)]
                for position in range(n_task)
            ]
            position = int(np.argmax(values))
            if values[position] < 0.5:
                raise OracleError("non-integral assignment returned by exact MILP")
            positioned.append((position, task.task_id))
        order = tuple(task_id for _, task_id in sorted(positioned))
        orders[world.world_name] = order
        closures[world.world_name] = completion_by_position[order.index(world.closing_task_id)]

    objective_tuple = (
        sum(value > pair.closure_budget_us + 1e-9 for value in closures.values()),
        empirical_cvar99_two_world(tuple(closures.values())),
        sum(closures.values()) / len(closures),
    )
    action_sets = _optimal_first_action_sets(
        pair, information_level, objective_tuple, completion_by_position
    )
    flip_rate, unique_action = _set_flip_rate(action_sets)
    gaps = [float(getattr(result, "mip_gap", 0.0)) for result in (first, second, third, final)]
    return OracleResult(
        information_level=information_level,
        orders_by_world=orders,
        closure_by_world_us=closures,
        violation_count=sum(
            value > pair.closure_budget_us + 1e-9 for value in closures.values()
        ),
        empirical_cvar99_us=empirical_cvar99_two_world(tuple(closures.values())),
        mean_closure_us=sum(closures.values()) / len(closures),
        first_action_flip_rate=flip_rate,
        optimal_first_actions_by_world=action_sets,
        unique_optimal_first_action=unique_action,
        position_completion_us=completion_by_position,
        solver=f"scipy.optimize.milp/HiGHS scipy={scipy_version}",
        solver_status="OPTIMAL",
        mip_gap=max(gaps),
        nonanticipativity_nodes=x_worlds,
    )


def brute_force_matched_pair(pair: MatchedPair, information_level: str) -> OracleResult:
    """Independent permutation enumerator for the tiny MILP fixture."""

    task_ids = tuple(sorted(task.task_id for task in pair.tasks))
    contribution_completion_by_position = fixed_route_flowshop_position_completions(
        pair.tasks[0].stage_service_us,
        len(pair.tasks),
        common_release_us=pair.tasks[0].release_us,
    )
    completion_by_position = tuple(
        value + pair.tasks[0].join_combine_us
        for value in contribution_completion_by_position
    )
    _validate_starvation_support(pair)
    candidates = _candidate_orders_by_world(pair, information_level)

    best_key: tuple[object, ...] | None = None
    best_orders: tuple[tuple[str, ...], ...] | None = None
    for orders in candidates:
        closures = tuple(
            completion_by_position[orders[index].index(world.closing_task_id)]
            for index, world in enumerate(pair.worlds)
        )
        violations = sum(value > pair.closure_budget_us + 1e-9 for value in closures)
        stable_vector = tuple(item for order in orders for item in order)
        key = (
            violations,
            empirical_cvar99_two_world(closures),
            sum(closures) / len(closures),
            stable_vector,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_orders = orders
    assert best_key is not None and best_orders is not None
    order_map = {
        world.world_name: best_orders[index]
        for index, world in enumerate(pair.worlds)
    }
    closure_map = {
        world.world_name: completion_by_position[
            best_orders[index].index(world.closing_task_id)
        ]
        for index, world in enumerate(pair.worlds)
    }
    objective_tuple = (int(best_key[0]), float(best_key[1]), float(best_key[2]))
    action_sets = _optimal_first_action_sets(
        pair, information_level, objective_tuple, completion_by_position
    )
    flip_rate, unique_action = _set_flip_rate(action_sets)
    return OracleResult(
        information_level=information_level,
        orders_by_world=order_map,
        closure_by_world_us=closure_map,
        violation_count=int(best_key[0]),
        empirical_cvar99_us=float(best_key[1]),
        mean_closure_us=float(best_key[2]),
        first_action_flip_rate=flip_rate,
        optimal_first_actions_by_world=action_sets,
        unique_optimal_first_action=unique_action,
        position_completion_us=completion_by_position,
        solver="itertools.permutations",
        solver_status="OPTIMAL_ENUMERATION",
        mip_gap=0.0,
        nonanticipativity_nodes=len(
            set(build_observation_history_nodes(pair, information_level).values())
        ),
    )


def assert_information_monotonicity(results: Mapping[str, OracleResult]) -> None:
    required = {"S", "B", "R0", "C"}
    if set(results) != required:
        raise OracleError(f"missing information levels: {sorted(required - set(results))}")

    def objective(name: str) -> tuple[int, float, float]:
        result = results[name]
        return (
            result.violation_count,
            result.empirical_cvar99_us,
            result.mean_closure_us,
        )

    if not (objective("C") <= objective("R0") <= objective("B") <= objective("S")):
        raise OracleError("oracle information monotonicity violated")
