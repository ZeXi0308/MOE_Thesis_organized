#!/usr/bin/env python3
"""Independent SciPy-MILP cross-check for PhaseMap information arms.

The core enumerator remains the source of replay semantics.  This module builds
an independent binary assignment MILP over observation-class/action choices,
solves the frozen three numeric lexicographic objectives in sequence, enumerates
every tolerance-banded optimal deterministic policy with MILP no-good cuts,
and verifies the core B0/Q/J/R minima, optimal policy set, canonical tie break,
and the internally exposed replay ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

try:
    from . import phasemap_oracle_core as core
except ImportError:  # pragma: no cover
    import phasemap_oracle_core as core  # type: ignore


GAP_TOLERANCE = 1e-7


class PhaseMapMILPError(RuntimeError):
    """The cross-check model, solver, or enumerator comparison failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _policy_sha256(policy: core.Policy) -> str:
    return hashlib.sha256(_canonical(policy)).hexdigest()


def _require_scipy() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:  # pragma: no cover - exercised on GPU environment
        raise PhaseMapMILPError("SciPy MILP is unavailable") from exc
    return np, Bounds, LinearConstraint, milp, np.ndarray


def _within_lexicographic_bands(
    objective: Sequence[float], minima: Sequence[float]
) -> bool:
    return len(objective) == len(minima) and all(
        float(value) <= float(minimum) + core.lexicographic_tolerance(float(minimum))
        for value, minimum in zip(objective, minima)
    )


def _validate_replay_ledger(result: Mapping[str, Any]) -> str:
    """Independently recompute stage ordering/count invariants from raw events.

    This is an independent ledger validator, not an independent simulator: the
    event times still originate in ``core.simulate``.  The report states that
    boundary explicitly.
    """
    sender_events = result.get("sender_events")
    receiver_events = result.get("receiver_events")
    combine_events = result.get("combine_events")
    requests = result.get("requests")
    accounting = result.get("accounting")
    if not all(isinstance(value, list) for value in (sender_events, receiver_events, combine_events, requests)):
        raise PhaseMapMILPError("replay ledger event surface is malformed")
    if not isinstance(accounting, Mapping):
        raise PhaseMapMILPError("replay accounting is missing")

    sender_tasks: set[str] = set()
    by_sender: dict[int, list[Mapping[str, Any]]] = {}
    for row in sender_events:
        task_id = str(row["task_id"])
        if task_id in sender_tasks:
            raise PhaseMapMILPError("sender ledger executes a decision task twice")
        sender_tasks.add(task_id)
        by_sender.setdefault(int(row["sender_rank"]), []).append(row)
        start, pack_end, cut_end = (
            float(row["pack_start_us"]), float(row["pack_end_us"]), float(row["cut_end_us"])
        )
        if start < 0 or not start < pack_end < cut_end:
            raise PhaseMapMILPError("sender pack/cut ledger is not positive and ordered")
    for rows in by_sender.values():
        available = 0.0
        for row in rows:
            if not math.isclose(float(row["pack_start_us"]), available, rel_tol=1e-10, abs_tol=1e-10):
                raise PhaseMapMILPError("sender ledger has a gap or overlap")
            available = float(row["cut_end_us"])

    receiver_tasks: set[str] = set()
    by_receiver: dict[int, list[Mapping[str, Any]]] = {}
    for row in receiver_events:
        job_id = str(row["job_id"])
        if any(str(existing["job_id"]) == job_id for rows in by_receiver.values() for existing in rows):
            raise PhaseMapMILPError("receiver ledger duplicates a job")
        by_receiver.setdefault(int(row["receiver_rank"]), []).append(row)
        start, end, service, arrival = (
            float(row["start_us"]), float(row["end_us"]),
            float(row["service_us"]), float(row["arrival_us"]),
        )
        if service <= 0 or start < arrival or not math.isclose(end - start, service, rel_tol=1e-10, abs_tol=1e-10):
            raise PhaseMapMILPError("receiver service ledger is inconsistent")
        if row.get("task_id") is not None:
            task_id = str(row["task_id"])
            if task_id in receiver_tasks:
                raise PhaseMapMILPError("receiver ledger executes a foreground task twice")
            receiver_tasks.add(task_id)
    for rows in by_receiver.values():
        available = -math.inf
        for row in sorted(rows, key=lambda value: (float(value["arrival_us"]), str(value["job_id"]))):
            expected = max(float(row["arrival_us"]), available)
            if not math.isclose(float(row["start_us"]), expected, rel_tol=1e-10, abs_tol=1e-10):
                raise PhaseMapMILPError("receiver FIFO replay is inconsistent")
            available = float(row["end_us"])

    close_by_join: dict[str, float] = {}
    by_combine_receiver: dict[int, list[Mapping[str, Any]]] = {}
    for row in combine_events:
        join_id = str(row["join_id"])
        if join_id in close_by_join:
            raise PhaseMapMILPError("combine ledger executes a join twice")
        close_by_join[join_id] = float(row["join_close_us"])
        by_combine_receiver.setdefault(int(row["receiver_rank"]), []).append(row)
    for rows in by_combine_receiver.values():
        available = -math.inf
        for row in sorted(rows, key=lambda value: (float(value["join_ready_us"]), str(value["join_id"]))):
            expected = max(float(row["join_ready_us"]), available)
            if not math.isclose(float(row["combine_start_us"]), expected, rel_tol=1e-10, abs_tol=1e-10):
                raise PhaseMapMILPError("combine serialization ledger is inconsistent")
            if float(row["join_close_us"]) <= float(row["combine_start_us"]):
                raise PhaseMapMILPError("combine service is not positive")
            available = float(row["join_close_us"])
    for row in requests:
        join_id = str(row["join_id"])
        if join_id not in close_by_join or not math.isclose(
            float(row["join_close_us"]), close_by_join[join_id], rel_tol=1e-10, abs_tol=1e-10
        ):
            raise PhaseMapMILPError("request close disagrees with combine ledger")

    expected_counts = {
        "native_siblings": len(receiver_tasks),
        "decision_pack_count": len(sender_events),
        "decision_cut_count": len(sender_events),
        "post_t0_foreground_unpack_count": sum(
            row.get("task_id") is not None and float(row["end_us"]) > 0 for row in receiver_events
        ),
        "combine_count": len(combine_events),
    }
    if any(int(accounting.get(key, -1)) != value for key, value in expected_counts.items()):
        raise PhaseMapMILPError("stage accounting disagrees with raw replay ledger")
    return hashlib.sha256(_canonical({
        "sender_events": sender_events,
        "receiver_events": receiver_events,
        "combine_events": combine_events,
        "requests": requests,
        "accounting": dict(accounting),
    })).hexdigest()


def _world_objective(result: Mapping[str, Any]) -> tuple[float, float, float]:
    _validate_replay_ledger(result)
    requests = result.get("requests")
    if not isinstance(requests, list) or len(requests) != 2:
        raise PhaseMapMILPError("world replay lacks exactly two native requests")
    try:
        miss = sum(bool(row["miss"]) for row in requests) / 4.0
        tardiness = sum(float(row["normalized_tardiness"]) for row in requests) / 4.0
        close = sum(float(row["join_close_us"]) for row in requests) / 4.0
    except (KeyError, TypeError, ValueError) as exc:
        raise PhaseMapMILPError("world replay objective is malformed") from exc
    values = (miss, tardiness, close)
    if any(not math.isfinite(value) for value in values) or miss < 0 or tardiness < 0:
        raise PhaseMapMILPError("world replay objective is non-finite or negative")
    return values


def _build_assignment_model(
    scenario: core.Scenario,
    arm: core.Arm,
) -> dict[str, Any]:
    np, _Bounds, _LinearConstraint, _milp, _ndarray = _require_scipy()
    classes = core.observation_partitions(scenario, arm)
    actions = core.enumerate_actions(scenario)
    if not classes or not actions:
        raise PhaseMapMILPError("empty observation or action space")
    variable_keys = tuple((class_key, action) for class_key, _ in classes for action in actions)
    index = {key: ordinal for ordinal, key in enumerate(variable_keys)}
    if len(index) != len(variable_keys):
        raise PhaseMapMILPError("duplicate MILP variable identity")

    costs = np.zeros((3, len(variable_keys)), dtype=float)
    for class_key, world_indices in classes:
        for action in actions:
            column = index[(class_key, action)]
            for world_index in world_indices:
                result = core.simulate(scenario, world_index, action)
                costs[:, column] += np.asarray(_world_objective(result), dtype=float)

    assignment_rows = []
    for class_key, _world_indices in classes:
        row = np.zeros(len(variable_keys), dtype=float)
        for action in actions:
            row[index[(class_key, action)]] = 1.0
        assignment_rows.append(row)
    return {
        "classes": classes,
        "actions": actions,
        "variable_keys": variable_keys,
        "index": index,
        "costs": costs,
        "assignment_rows": assignment_rows,
    }


def _solve(
    model: Mapping[str, Any],
    objective: Any,
    extra_rows: Sequence[Any],
    extra_lower: Sequence[float],
    extra_upper: Sequence[float],
    *,
    allow_infeasible: bool = False,
) -> tuple[Any | None, float]:
    np, Bounds, LinearConstraint, milp, _ndarray = _require_scipy()
    variable_count = len(model["variable_keys"])
    rows = list(model["assignment_rows"]) + list(extra_rows)
    lower = [1.0] * len(model["assignment_rows"]) + list(extra_lower)
    upper = [1.0] * len(model["assignment_rows"]) + list(extra_upper)
    constraints = LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper))
    result = milp(
        c=np.asarray(objective, dtype=float),
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=constraints,
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not result.success:
        if allow_infeasible and int(getattr(result, "status", -1)) == 2:
            return None, 0.0
        raise PhaseMapMILPError(
            f"SciPy MILP failed with status={getattr(result, 'status', None)}: {result.message}"
        )
    gap_raw = getattr(result, "mip_gap", 0.0)
    gap = 0.0 if gap_raw is None else float(gap_raw)
    if not math.isfinite(gap) or gap > GAP_TOLERANCE:
        raise PhaseMapMILPError(f"MILP gap {gap} exceeds {GAP_TOLERANCE}")
    values = np.asarray(result.x, dtype=float)
    rounded = np.rint(values)
    if values.shape != (variable_count,) or float(np.max(np.abs(values - rounded))) > GAP_TOLERANCE:
        raise PhaseMapMILPError("MILP returned a non-integral policy")
    return rounded.astype(int), gap


def _policy_from_solution(model: Mapping[str, Any], solution: Any) -> core.Policy:
    policy = []
    for class_key, _world_indices in model["classes"]:
        selected = [
            action
            for action in model["actions"]
            if int(solution[model["index"][(class_key, action)]]) == 1
        ]
        if len(selected) != 1:
            raise PhaseMapMILPError("MILP solution does not choose one action per observation class")
        policy.append((class_key, selected[0]))
    return tuple(policy)


def _policy_objective(model: Mapping[str, Any], solution: Any) -> tuple[float, float, float]:
    costs = model["costs"]
    return tuple(float(costs[index] @ solution) for index in range(3))


def _policy_objective_from_identity(
    model: Mapping[str, Any], policy: core.Policy
) -> tuple[float, float, float]:
    policy_map = dict(policy)
    return tuple(
        sum(
            float(model["costs"][objective_index, model["index"][(class_key, policy_map[class_key])]])
            for class_key, _world_indices in model["classes"]
        )
        for objective_index in range(3)
    )


def _enumerate_optimal_policies(
    model: Mapping[str, Any],
    minima: tuple[float, float, float],
) -> tuple[tuple[core.Policy, ...], float]:
    np, _Bounds, _LinearConstraint, _milp, _ndarray = _require_scipy()
    extra_rows = [model["costs"][index] for index in range(3)]
    extra_lower = [-math.inf] * len(minima)
    extra_upper = [
        value + core.lexicographic_tolerance(value) for value in minima
    ]
    policies: dict[bytes, core.Policy] = {}
    maximum = len(model["actions"]) ** len(model["classes"])
    maximum_gap = 0.0

    for _iteration in range(maximum + 1):
        solution, gap = _solve(
            model,
            np.zeros(len(model["variable_keys"]), dtype=float),
            extra_rows,
            extra_lower,
            extra_upper,
            allow_infeasible=True,
        )
        maximum_gap = max(maximum_gap, gap)
        if solution is None:
            break
        policy = _policy_from_solution(model, solution)
        objective = _policy_objective(model, solution)
        if _within_lexicographic_bands(objective, minima):
            policies[_canonical(policy)] = policy
        selected = np.asarray(solution, dtype=float)
        extra_rows.append(selected)
        extra_lower.append(-math.inf)
        extra_upper.append(float(len(model["classes"]) - 1))
    else:  # pragma: no cover - defensive guard
        raise PhaseMapMILPError("MILP optimal-policy enumeration did not terminate")

    if not policies:
        raise PhaseMapMILPError("MILP found no policy at its own lexicographic optimum")
    return tuple(policies[key] for key in sorted(policies)), maximum_gap


def _solve_canonical_policy(
    model: Mapping[str, Any],
    minima: tuple[float, float, float],
) -> tuple[core.Policy, float]:
    """MILP-implement the protocol's final serialized-policy tie break.

    Observation classes already have canonical key order.  Minimizing the
    canonical action rank at each class in sequence is exactly lexicographic
    minimization of the serialized policy without unsafe large positional
    weights.
    """

    np, _Bounds, _LinearConstraint, _milp, _ndarray = _require_scipy()
    extra_rows = [model["costs"][index] for index in range(3)]
    extra_lower = [-math.inf] * len(minima)
    extra_upper = [
        value + core.lexicographic_tolerance(value) for value in minima
    ]
    maximum_gap = 0.0
    solution = None
    canonical_actions = tuple(sorted(model["actions"], key=_canonical))
    for class_key, _world_indices in model["classes"]:
        rank_objective = np.zeros(len(model["variable_keys"]), dtype=float)
        for rank, action in enumerate(canonical_actions):
            rank_objective[model["index"][(class_key, action)]] = float(rank)
        solution, gap = _solve(
            model,
            rank_objective,
            extra_rows,
            extra_lower,
            extra_upper,
        )
        if solution is None:  # pragma: no cover
            raise PhaseMapMILPError("canonical MILP unexpectedly became infeasible")
        maximum_gap = max(maximum_gap, gap)
        selected_rank = float(rank_objective @ solution)
        extra_rows.append(rank_objective)
        extra_lower.append(selected_rank)
        extra_upper.append(selected_rank)
    if solution is None:  # pragma: no cover - every arm has at least one class
        raise PhaseMapMILPError("canonical MILP did not select a policy")
    return _policy_from_solution(model, solution), maximum_gap


def crosscheck_arm(
    scenario: core.Scenario,
    arm: core.Arm,
    core_arm_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    np, _Bounds, _LinearConstraint, _milp, _ndarray = _require_scipy()
    model = _build_assignment_model(scenario, arm)
    extra_rows: list[Any] = []
    extra_lower: list[float] = []
    extra_upper: list[float] = []
    optimum: list[float] = []
    maximum_gap = 0.0

    for objective_index in range(3):
        solution, gap = _solve(
            model,
            model["costs"][objective_index],
            extra_rows,
            extra_lower,
            extra_upper,
        )
        if solution is None:  # pragma: no cover - non-infeasible path cannot return None
            raise PhaseMapMILPError("lexicographic MILP unexpectedly became infeasible")
        maximum_gap = max(maximum_gap, gap)
        value = float(model["costs"][objective_index] @ solution)
        optimum.append(value)
        extra_rows.append(model["costs"][objective_index])
        extra_lower.append(-math.inf)
        extra_upper.append(value + core.lexicographic_tolerance(value))

    optimum_tuple = tuple(optimum)
    policies, enumeration_gap = _enumerate_optimal_policies(model, optimum_tuple)
    selected, canonical_gap = _solve_canonical_policy(model, optimum_tuple)
    maximum_gap = max(maximum_gap, enumeration_gap, canonical_gap)
    if selected != min(policies, key=_canonical):
        raise PhaseMapMILPError("MILP canonical tie break disagrees with serialized policy order")
    action_sets = tuple(
        (
            class_key,
            tuple(
                sorted(
                    {dict(policy)[class_key] for policy in policies},
                    key=_canonical,
                )
            ),
        )
        for class_key, _world_indices in model["classes"]
    )

    if core_arm_report is None:
        core_arm_report = core.optimize_arm(scenario, arm)
    metrics = core_arm_report.get("metrics")
    if not isinstance(metrics, core.PairMetrics):
        raise PhaseMapMILPError("core arm report lacks PairMetrics")
    core_minima_raw = core_arm_report.get("lexicographic_minima")
    if not isinstance(core_minima_raw, tuple) or len(core_minima_raw) != 3:
        raise PhaseMapMILPError("core arm report lacks frozen lexicographic minima")
    core_minima = tuple(float(value) for value in core_minima_raw)
    core_objective = tuple(float(value) for value in metrics.objective)
    minima_gaps = tuple(abs(left - right) for left, right in zip(optimum_tuple, core_minima))
    selected_objective = _policy_objective_from_identity(model, selected)
    selected_gaps = tuple(
        abs(left - right) for left, right in zip(selected_objective, core_objective)
    )
    if max(minima_gaps) > GAP_TOLERANCE:
        raise PhaseMapMILPError(
            f"{arm} core/MILP minima mismatch: core={core_minima}, milp={optimum_tuple}"
        )
    if max(selected_gaps) > GAP_TOLERANCE:
        raise PhaseMapMILPError(
            f"{arm} core/MILP objective mismatch: core={core_objective}, milp={selected_objective}"
        )

    core_policies = tuple(sorted(core_arm_report.get("optimal_policies", ()), key=_canonical))
    if tuple(_canonical(policy) for policy in core_policies) != tuple(
        _canonical(policy) for policy in policies
    ):
        raise PhaseMapMILPError(f"{arm} core/MILP optimal policy set mismatch")
    if core_arm_report.get("selected_canonical_policy") != selected:
        raise PhaseMapMILPError(f"{arm} core/MILP canonical policy mismatch")
    if maximum_gap > GAP_TOLERANCE:
        raise PhaseMapMILPError(f"{arm} maximum MILP gap exceeds frozen tolerance")

    return {
        "arm": arm,
        "observation_class_count": len(model["classes"]),
        "action_count": len(model["actions"]),
        "variable_count": len(model["variable_keys"]),
        "lexicographic_objective": optimum_tuple,
        "core_lexicographic_minima": core_minima,
        "core_objective": core_objective,
        "selected_policy_objective": selected_objective,
        "minima_absolute_gaps": minima_gaps,
        "selected_objective_absolute_gaps": selected_gaps,
        "maximum_objective_absolute_gap": max(max(minima_gaps), max(selected_gaps)),
        "solver_mip_gap": maximum_gap,
        "optimal_policy_count": len(policies),
        "optimal_policies": policies,
        "optimal_policy_sha256": tuple(_policy_sha256(policy) for policy in policies),
        "selected_canonical_policy": selected,
        "optimal_action_sets": action_sets,
        "lexicographic_tolerance": {
            "relative": core.LEX_REL_TOLERANCE,
            "absolute": core.LEX_ABS_TOLERANCE,
            "constraint": "objective_i <= stage_min_i + max(abs_tol, rel_tol*abs(stage_min_i))",
        },
        "ledger_crosscheck_scope": "independent_event_ledger_validation_not_independent_simulator",
        "passed": True,
    }


def crosscheck_information_lattice(
    scenario: core.Scenario,
    core_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if core_report is None:
        core_report = core.optimize_information_lattice(scenario)
    arms = core_report.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"B0", "Q", "J", "R"}:
        raise PhaseMapMILPError("core report lacks the complete B0/Q/J/R lattice")
    reports = {
        arm: crosscheck_arm(scenario, arm, arms[arm])
        for arm in ("B0", "Q", "J", "R")
    }
    maximum_gap = max(float(report["solver_mip_gap"]) for report in reports.values())
    maximum_objective_gap = max(
        float(report["maximum_objective_absolute_gap"]) for report in reports.values()
    )
    if maximum_gap > GAP_TOLERANCE or maximum_objective_gap > GAP_TOLERANCE:
        raise PhaseMapMILPError("information-lattice cross-check exceeded the frozen gap")
    return {
        "schema_version": "phasemap-scipy-milp-crosscheck-v1",
        "scenario_id": scenario.scenario_id,
        "solver": "scipy.optimize.milp/HiGHS",
        "gap_tolerance": GAP_TOLERANCE,
        "lexicographic_tolerance": {
            "relative": core.LEX_REL_TOLERANCE,
            "absolute": core.LEX_ABS_TOLERANCE,
        },
        "crosscheck_scope": "independent_milp_optimizer_plus_event_ledger_validator_not_independent_simulator",
        "maximum_solver_mip_gap": maximum_gap,
        "maximum_core_objective_gap": maximum_objective_gap,
        "arms": reports,
        "passed": True,
    }
