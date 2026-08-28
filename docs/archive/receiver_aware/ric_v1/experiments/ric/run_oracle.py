#!/usr/bin/env python3
"""Build and solve the route-derived RIC-v1 matched-world oracle.

SciPy is imported only when the exact solver is invoked.  Consequently local
fail-closed runner tests can inspect partition/provenance logic on machines
without SciPy, while an actual oracle run still refuses to fall back to a
heuristic solver.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from .build_scenarios import (
        DEFAULT_CONFIG,
        DEFAULT_CONSUMER_AMENDMENT,
        DEFAULT_PROTOCOL,
        ScenarioBuildError,
        _read_self_hashed_json,
        _source_sha256 as _scenario_source_sha256,
        atomic_output_directory,
        canonical_json_bytes,
        load_worlds,
        object_sha256,
        validate_frozen_formal_paths,
        validate_formal_output_path,
    )
    from .prepare_data import add_self_hash, sha256_file
    from .prepare_data import (
        RUN_EXPERIMENT_SOURCE_PATHS,
        _run_experiment_source_sha256,
    )
    from .formal_provenance import (
        EMBEDDED_PRODUCER_SIGNOFF,
        FormalProvenanceError,
        canonical_reviewed_scope_paths,
        load_json_mapping_strict,
        materialize_verified_signoff,
        validate_calibration_lock_fields,
        validate_self_hash,
        verify_phase4_signoff,
    )
    from .scenario import ReplayTask, ReplayWorld
except ImportError:
    from build_scenarios import (  # type: ignore
        DEFAULT_CONFIG,
        DEFAULT_CONSUMER_AMENDMENT,
        DEFAULT_PROTOCOL,
        ScenarioBuildError,
        _read_self_hashed_json,
        _source_sha256 as _scenario_source_sha256,
        atomic_output_directory,
        canonical_json_bytes,
        load_worlds,
        object_sha256,
        validate_frozen_formal_paths,
        validate_formal_output_path,
    )
    from prepare_data import add_self_hash, sha256_file  # type: ignore
    from prepare_data import (  # type: ignore
        RUN_EXPERIMENT_SOURCE_PATHS,
        _run_experiment_source_sha256,
    )
    from formal_provenance import (  # type: ignore
        EMBEDDED_PRODUCER_SIGNOFF,
        FormalProvenanceError,
        canonical_reviewed_scope_paths,
        load_json_mapping_strict,
        materialize_verified_signoff,
        validate_calibration_lock_fields,
        validate_self_hash,
        verify_phase4_signoff,
    )
    from scenario import ReplayTask, ReplayWorld  # type: ignore


REPO_ROOT = next(candidate for candidate in HERE.parents if (candidate / "experiments/shared").is_dir())
ORACLE_SOURCE_PATHS = (
    HERE / "__init__.py",
    Path(__file__),
    HERE / "oracle.py",
    HERE / "build_scenarios.py",
    HERE / "scenario.py",
    HERE / "schema.py",
    HERE / "capture_routes_gpu.py",
    HERE / "prepare_data.py",
    HERE / "formal_provenance.py",
    HERE / "run_experiment.py",
    HERE / "replay.py",
    HERE / "accounting.py",
    HERE / "wire.py",
    HERE / "policy_views.py",
    HERE / "capability_contract.py",
    HERE / "measure_capability_gpu.py",
    HERE / "measure_service_lut_gpu.py",
)


class OracleRunnerError(RuntimeError):
    """Matched-world construction, exactness or gate validation failed."""


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = load_json_mapping_strict(path, label="RIC config")
    except FormalProvenanceError as exc:
        raise OracleRunnerError(str(exc)) from exc
    if value.get("schema_version") != "ric-config-v1":
        raise OracleRunnerError("RIC config schema mismatch")
    return value


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for path in ORACLE_SOURCE_PATHS:
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_oracle_types() -> Mapping[str, Any]:
    try:
        try:
            from .oracle import (  # type: ignore
                MatchedPair,
                MatchedWorld,
                PublicTask,
                assert_information_monotonicity,
                brute_force_matched_pair,
                build_observation_history_nodes,
                solve_matched_pair,
            )
        except ImportError:
            from oracle import (  # type: ignore
                MatchedPair,
                MatchedWorld,
                PublicTask,
                assert_information_monotonicity,
                brute_force_matched_pair,
                build_observation_history_nodes,
                solve_matched_pair,
            )
    except ImportError as exc:
        raise OracleRunnerError(
            "BLOCKED_SOLVER_UNAVAILABLE: scipy.optimize.milp is required"
        ) from exc
    return {
        "MatchedPair": MatchedPair,
        "MatchedWorld": MatchedWorld,
        "PublicTask": PublicTask,
        "assert_information_monotonicity": assert_information_monotonicity,
        "brute_force_matched_pair": brute_force_matched_pair,
        "build_observation_history_nodes": build_observation_history_nodes,
        "solve_matched_pair": solve_matched_pair,
    }


@dataclass(frozen=True)
class RoutePairRecord:
    pair: Any
    source_trace_id: str
    workload_seed: int
    source_task_ids: tuple[str, str]
    source_join_fingerprints: tuple[str, str]
    sender_rank: int
    receiver_rank: int
    stage_resources: tuple[str, str, str, str]
    stage_service_us: tuple[float, float, float, float]
    counterfactual_release_us: float


def _join_fingerprint(task: ReplayTask) -> str:
    return hashlib.sha256(task.join_identity.canonical_bytes()).hexdigest()


def _pair_candidate_key(
    left: ReplayTask, right: ReplayTask, *, trace_id: str
) -> tuple[str, str, str]:
    ordered = tuple(sorted((left.task_id, right.task_id)))
    return object_sha256(("ric-v1-route-pair", trace_id, ordered)), *ordered


def route_pair_from_world(
    world: ReplayWorld,
    *,
    pair_index: int,
    closure_budget_us: float,
    starvation_us: float,
) -> RoutePairRecord:
    """Select one pair without consulting any replay outcome."""

    if closure_budget_us <= 0 or not math.isfinite(closure_budget_us):
        raise OracleRunnerError("matched-world closure budget must be positive")
    if starvation_us < 0 or not math.isfinite(starvation_us):
        raise OracleRunnerError("matched-world starvation bound must be finite/non-negative")
    by_matched_path: dict[tuple[Any, ...], list[ReplayTask]] = {}
    for task in world.tasks:
        key = (
            int(task.identity.sender_rank),
            int(task.identity.receiver_rank),
            task.stage_resources,
            task.stage_service_us,
            task.combine_resource,
            task.stage_service.join_combine_us,
            int(task.contribution.payload_bytes),
        )
        by_matched_path.setdefault(key, []).append(task)
    candidates: list[tuple[tuple[str, str, str], ReplayTask, ReplayTask]] = []
    for bucket in by_matched_path.values():
        ranked = sorted(
            bucket,
            key=lambda task: object_sha256(
                ("ric-v1-route-task", world.trace_id, task.task_id)
            ),
        )
        # The first distinct-join partner in the outcome-blind hash order is
        # sufficient and avoids an artificial prefix that could falsely report
        # missing support.
        if ranked:
            left = ranked[0]
            right = next(
                (task for task in ranked[1:] if task.join_identity != left.join_identity),
                None,
            )
            if right is not None:
                candidates.append(
                    (_pair_candidate_key(left, right, trace_id=world.trace_id), left, right)
                )
    if not candidates:
        raise OracleRunnerError(
            "BLOCKED_MATCHED_WORLD_SUPPORT: no distinct joins share exact "
            "sender/receiver/path/service/bytes"
        )
    _key, left, right = min(candidates, key=lambda row: row[0])
    oracle = _load_oracle_types()
    PublicTask = oracle["PublicTask"]
    MatchedWorld = oracle["MatchedWorld"]
    MatchedPair = oracle["MatchedPair"]
    left_fingerprint = _join_fingerprint(left)
    right_fingerprint = _join_fingerprint(right)
    tasks = (
        PublicTask(
            task_id=left.task_id,
            join_fingerprint=left_fingerprint,
            release_us=0.0,
            sender_rank=int(left.identity.sender_rank),
            receiver_rank=int(left.identity.receiver_rank),
            sender_egress_resource=left.sender_egress_resource,
            shared_cut_resource=left.shared_cut_resource,
            receiver_ingress_resource=left.receiver_ingress_resource,
            receiver_combine_resource=left.combine_resource,
            sender_egress_us=float(left.stage_service.sender_egress_us),
            shared_cut_us=float(left.stage_service.shared_cut_us),
            receiver_ingress_us=float(left.stage_service.receiver_ingress_us),
            join_combine_us=float(left.stage_service.join_combine_us),
            payload_bytes=int(left.contribution.payload_bytes),
            deadline_us=float(left.contribution.deadline_us),
        ),
        PublicTask(
            task_id=right.task_id,
            join_fingerprint=right_fingerprint,
            release_us=0.0,
            sender_rank=int(right.identity.sender_rank),
            receiver_rank=int(right.identity.receiver_rank),
            sender_egress_resource=right.sender_egress_resource,
            shared_cut_resource=right.shared_cut_resource,
            receiver_ingress_resource=right.receiver_ingress_resource,
            receiver_combine_resource=right.combine_resource,
            sender_egress_us=float(right.stage_service.sender_egress_us),
            shared_cut_us=float(right.stage_service.shared_cut_us),
            receiver_ingress_us=float(right.stage_service.receiver_ingress_us),
            join_combine_us=float(right.stage_service.join_combine_us),
            payload_bytes=int(right.contribution.payload_bytes),
            deadline_us=float(right.contribution.deadline_us),
        ),
    )
    worlds = (
        MatchedWorld(
            world_name="world-a",
            closing_task_id=left.task_id,
            hidden_join_fingerprint=left_fingerprint,
        ),
        MatchedWorld(
            world_name="world-b",
            closing_task_id=right.task_id,
            hidden_join_fingerprint=right_fingerprint,
        ),
    )
    pair = MatchedPair(
        pair_id=f"{world.model_key}-route-pair-{pair_index:02d}",
        model_key=world.model_key,
        tasks=tasks,
        worlds=worlds,
        closure_budget_us=closure_budget_us,
        starvation_us=starvation_us,
        aggregate_receiver_qdepth=0,
        aggregate_shared_cut_backlog_bytes=0,
    )
    return RoutePairRecord(
        pair=pair,
        source_trace_id=world.trace_id,
        workload_seed=world.workload_seed,
        source_task_ids=(left.task_id, right.task_id),
        source_join_fingerprints=(left_fingerprint, right_fingerprint),
        sender_rank=left.identity.sender_rank,
        receiver_rank=left.identity.receiver_rank,
        stage_resources=(*left.stage_resources, left.combine_resource),
        stage_service_us=(
            *left.stage_service_us,
            left.stage_service.join_combine_us,
        ),
        counterfactual_release_us=0.0,
    )


def build_route_pairs(
    worlds: Sequence[ReplayWorld],
    *,
    model_key: str,
    primary_cell: str,
    pair_count: int,
    closure_budget_us: float,
    starvation_us: float,
) -> tuple[RoutePairRecord, ...]:
    selected = sorted(
        [world for world in worlds if world.model_key == model_key and world.cell == primary_cell],
        key=lambda world: (world.workload_seed, world.trace_id),
    )
    if len(selected) != pair_count:
        raise OracleRunnerError(
            f"expected {pair_count} disjoint calibration worlds for "
            f"{model_key}, got {len(selected)}"
        )
    if len({world.workload_seed for world in selected}) != pair_count:
        raise OracleRunnerError("matched-pair source workload seeds are not unique")
    records = tuple(
        route_pair_from_world(
            world,
            pair_index=index,
            closure_budget_us=closure_budget_us,
            starvation_us=starvation_us,
        )
        for index, world in enumerate(selected)
    )
    if len({record.source_trace_id for record in records}) != pair_count:
        raise OracleRunnerError("matched pairs reuse a calibration trace")
    return records


def _result_dict(result: Any) -> dict[str, Any]:
    return {
        "information_level": result.information_level,
        "orders_by_world": {
            key: list(value) for key, value in result.orders_by_world.items()
        },
        "closure_by_world_us": dict(result.closure_by_world_us),
        "violation_count": result.violation_count,
        "empirical_cvar99_us": result.empirical_cvar99_us,
        "mean_closure_us": result.mean_closure_us,
        "first_action_flip_rate": result.first_action_flip_rate,
        "optimal_first_actions_by_world": {
            key: list(value)
            for key, value in result.optimal_first_actions_by_world.items()
        },
        "unique_optimal_first_action": result.unique_optimal_first_action,
        "position_completion_us": list(result.position_completion_us),
        "solver": result.solver,
        "solver_status": result.solver_status,
        "mip_gap": result.mip_gap,
        "nonanticipativity_nodes": result.nonanticipativity_nodes,
    }


def solve_route_pairs(
    records: Sequence[RoutePairRecord], *, max_gap: float
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    oracle = _load_oracle_types()
    solve = oracle["solve_matched_pair"]
    brute_force = oracle["brute_force_matched_pair"]
    monotonicity = oracle["assert_information_monotonicity"]
    rows: list[dict[str, Any]] = []
    normalized_gaps: list[float] = []
    flip_rates: list[float] = []
    for record in records:
        results = {level: solve(record.pair, level) for level in ("S", "B", "R0", "C")}
        enumerated = {
            level: brute_force(record.pair, level)
            for level in ("S", "B", "R0", "C")
        }
        for level, result in results.items():
            exact = enumerated[level]
            milp_objective = (
                result.violation_count,
                result.empirical_cvar99_us,
                result.mean_closure_us,
            )
            enumerated_objective = (
                exact.violation_count,
                exact.empirical_cvar99_us,
                exact.mean_closure_us,
            )
            if any(
                not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-7)
                for left, right in zip(milp_objective, enumerated_objective)
            ):
                raise OracleRunnerError(
                    f"BLOCKED_ORACLE_CROSSCHECK: {record.pair.pair_id}/{level}"
                )
        monotonicity(results)
        if len(set(results["S"].orders_by_world.values())) != 1:
            raise OracleRunnerError("S nonanticipativity violated")
        if len(set(results["B"].orders_by_world.values())) != 1:
            raise OracleRunnerError("B nonanticipativity violated")
        if any(result.mip_gap > max_gap for result in results.values()):
            raise OracleRunnerError("BLOCKED_SOLVER_INEXACT")
        if not results["R0"].unique_optimal_first_action:
            raise OracleRunnerError(
                "BLOCKED_AMBIGUOUS_FIRST_ACTION_OPTIMUM: solver tie is not a flip"
            )
        blind = float(results["B"].empirical_cvar99_us)
        aware = float(results["R0"].empirical_cvar99_us)
        if blind <= 0:
            raise OracleRunnerError(
                "BLOCKED_NORMALIZED_GAP_DENOMINATOR: B empirical CVaR99 <= 0"
            )
        normalized_gap = (blind - aware) / blind
        normalized_gaps.append(normalized_gap)
        flip_rates.append(float(results["R0"].first_action_flip_rate))
        for level, result in results.items():
            rows.append(
                {
                    "schema_version": "ric-milp-solution-v2",
                    "pair_id": record.pair.pair_id,
                    "model_key": record.pair.model_key,
                    "source_trace_id": record.source_trace_id,
                    "workload_seed": record.workload_seed,
                    "normalized_b_to_r0_empirical_cvar99_gap": normalized_gap,
                    **_result_dict(result),
                }
            )
    return rows, {
        "pair_count": float(len(records)),
        "median_normalized_empirical_cvar99_gap": float(
            statistics.median(normalized_gaps)
        ),
        "mean_r0_first_action_flip_rate": sum(flip_rates) / len(flip_rates),
        "max_solver_gap": max(float(row["mip_gap"]) for row in rows),
    }


def _instance_dict(record: RoutePairRecord) -> dict[str, Any]:
    pair = record.pair
    return {
        "schema_version": "ric-milp-instance-v2",
        "pair_id": pair.pair_id,
        "model_key": pair.model_key,
        "source_trace_id": record.source_trace_id,
        "workload_seed": record.workload_seed,
        "source_task_ids": list(record.source_task_ids),
        "source_join_fingerprints": list(record.source_join_fingerprints),
        "sender_rank": record.sender_rank,
        "receiver_rank": record.receiver_rank,
        "stage_resources": list(record.stage_resources),
        "stage_service_us": list(record.stage_service_us),
        "counterfactual_release_us": record.counterfactual_release_us,
        "tasks": [asdict(task) for task in pair.tasks],
        "worlds": [asdict(world) for world in pair.worlds],
        "closure_budget_us": pair.closure_budget_us,
        "starvation_us": pair.starvation_us,
        "downstream_service_discipline": "work_conserving_fcfs_no_overtake",
        "aggregate_receiver_qdepth": pair.aggregate_receiver_qdepth,
        "aggregate_shared_cut_backlog_bytes": pair.aggregate_shared_cut_backlog_bytes,
        "public_observation_signature": pair.public_observation_signature(pair.worlds[0]),
        "observation_history_nodes": {
            level: oracle_nodes
            for level, oracle_nodes in (
                (
                    level,
                    _load_oracle_types()["build_observation_history_nodes"](
                        pair, level
                    ),
                )
                for level in ("S", "B", "R0", "C")
            )
        },
    }


def _model_budget(
    lock: Mapping[str, Any], *, model_key: str, main_cells: Sequence[str]
) -> float:
    model = lock.get("models", {}).get(model_key)
    if not isinstance(model, Mapping):
        raise OracleRunnerError(f"calibration lock missing model {model_key}")
    values = []
    for cell in main_cells:
        row = model.get("cells", {}).get(cell)
        if not isinstance(row, Mapping):
            raise OracleRunnerError(f"calibration lock missing {model_key}/{cell}")
        values.append(float(row["closure_budget_us"]))
    if any(value <= 0 or not math.isfinite(value) for value in values):
        raise OracleRunnerError("invalid calibration closure budget")
    # The exact fixture is model-level; use the stricter pre-existing main-cell
    # budget.  This choice is bound before any pair is selected or solved.
    return min(values)


def run_oracle_pipeline(
    *,
    scenario_dirs: Sequence[Path],
    calibration_lock_path: Path,
    output_dir: Path,
    mode: str,
    config_path: Path,
    protocol_path: Path,
    signoff_path: Path | None = None,
) -> Mapping[str, Any]:
    if output_dir.exists():
        raise OracleRunnerError("refusing to overwrite oracle output directory")
    try:
        validate_formal_output_path(output_dir, mode=mode)
        validate_frozen_formal_paths(
            config_path=config_path, protocol_path=protocol_path, mode=mode
        )
    except ScenarioBuildError as exc:
        raise OracleRunnerError(str(exc)) from exc
    if calibration_lock_path.is_symlink() or not calibration_lock_path.is_file():
        raise OracleRunnerError("calibration lock must be an existing regular file")
    config = _load_config(config_path)
    config_sha = sha256_file(config_path)
    protocol_sha = sha256_file(protocol_path)
    try:
        lock = load_json_mapping_strict(
            calibration_lock_path, label="calibration lock"
        )
        if mode == "formal":
            validate_calibration_lock_fields(
                lock,
                config=config,
                protocol_sha256=protocol_sha,
                config_sha256=config_sha,
                expected_run_experiment_source_sha256=(
                    _run_experiment_source_sha256()
                ),
            )
            lock_signoff = (
                calibration_lock_path.parent / EMBEDDED_PRODUCER_SIGNOFF
            )
            if (
                not lock_signoff.is_file()
                or sha256_file(lock_signoff) != lock.get("signoff_sha256")
            ):
                raise FormalProvenanceError(
                    "calibration lock embedded producer signoff mismatch"
                )
            verify_phase4_signoff(
                lock_signoff,
                repo_root=REPO_ROOT,
                expected_fields={
                    "stage": "calibration",
                    "config_sha256": lock.get("config_sha256"),
                    "protocol_sha256": lock.get("protocol_sha256"),
                    "run_experiment_source_sha256": lock.get(
                        "run_experiment_source_sha256"
                    ),
                    "scenario_tree_sha256": lock.get("scenario_tree_sha256"),
                    "scenario_producer_signoff_sha256": lock.get(
                        "scenario_producer_signoff_sha256"
                    ),
                    "capability_probe_sha256": lock.get(
                        "capability_probe_sha256"
                    ),
                    "capability_producer_signoff_sha256": lock.get(
                        "capability_producer_signoff_sha256"
                    ),
                    "consumer_amendment_sha256": lock.get(
                        "consumer_amendment_sha256"
                    ),
                    "historical_reviewed_source_snapshot_sha256": lock.get(
                        "historical_reviewed_source_snapshot_sha256"
                    ),
                    "pre_outcome_attestation_sha256": lock.get(
                        "pre_outcome_attestation_sha256"
                    ),
                    "authoritative_bundle_root": lock.get(
                        "authoritative_bundle_root"
                    ),
                },
                required_source_paths=RUN_EXPERIMENT_SOURCE_PATHS,
                required_reviewed_scope_paths=(
                    *canonical_reviewed_scope_paths(
                        REPO_ROOT, RUN_EXPERIMENT_SOURCE_PATHS
                    ),
                    DEFAULT_CONSUMER_AMENDMENT,
                ),
            )
        elif mode == "dev":
            if lock.get("schema_version") != "ric-calibration-lock-v1":
                raise FormalProvenanceError("calibration lock schema mismatch")
            validate_self_hash(lock)
            for field, wanted in (
                ("status", "NOT_TESTED"),
                ("mode", "dev"),
                ("role", "calibration"),
                ("config_sha256", config_sha),
                ("protocol_sha256", protocol_sha),
                (
                    "run_experiment_source_sha256",
                    _run_experiment_source_sha256(),
                ),
            ):
                if lock.get(field) != wanted:
                    raise FormalProvenanceError(
                        f"development calibration lock mismatch: {field}"
                    )
        else:
            raise FormalProvenanceError("invalid oracle mode")
    except FormalProvenanceError as exc:
        raise OracleRunnerError(str(exc)) from exc
    trees = []
    worlds: list[ReplayWorld] = []
    for directory in scenario_dirs:
        if not directory.is_dir():
            raise OracleRunnerError(f"scenario directory is missing: {directory}")
        tree, loaded = load_worlds(directory, expected_role="calibration", mode=mode)
        trees.append(tree)
        worlds.extend(loaded)
    required_models = tuple(config["matched_world_milp"]["required_models"])
    tree_models = [str(tree["model_key"]) for tree in trees]
    if len(tree_models) != len(required_models) or set(tree_models) != set(required_models):
        raise OracleRunnerError("oracle requires exactly both frozen models")
    if any(
        int(tree["link_gbps"])
        != int(config["topology_proxy"]["primary_link_gbps"])
        for tree in trees
    ):
        raise OracleRunnerError("matched oracle consumes primary-link scenarios only")
    scenario_tree_hashes = {
        str(tree["model_key"]): str(tree["manifest_sha256"]) for tree in trees
    }
    if scenario_tree_hashes != lock.get("scenario_tree_sha256"):
        raise OracleRunnerError(
            "oracle scenario trees do not match the G1 calibration lock"
        )
    migration_fields = (
        "consumer_amendment_sha256",
        "historical_reviewed_source_snapshot_sha256",
        "pre_outcome_attestation_sha256",
        "pre_outcome_producer_signoff_file_sha256",
        "pre_outcome_producer_signoff_self_hash",
        "authoritative_bundle_root",
    )
    if mode == "formal":
        for tree in trees:
            for field in migration_fields:
                if tree.get(field) != lock.get(field) or tree.get(field) in {
                    None,
                    "",
                }:
                    raise OracleRunnerError(
                        f"oracle migration binding mismatch: {tree.get('model_key')}/{field}"
                    )
    scenario_tree_file_hashes = {
        str(tree["model_key"]): sha256_file(directory / "scenario_tree.json")
        for directory, tree in zip(scenario_dirs, trees)
    }
    scenario_producer_signoff_sha256 = {
        str(tree["model_key"]): str(tree["scenario_producer_signoff_sha256"])
        for tree in trees
    }
    if mode == "formal":
        _require_formal_signoff(
            signoff_path,
            config_path=config_path,
            protocol_path=protocol_path,
            calibration_lock_path=calibration_lock_path,
            calibration_lock=lock,
            scenario_tree_hashes=scenario_tree_hashes,
            scenario_tree_file_hashes=scenario_tree_file_hashes,
            scenario_producer_signoff_sha256=(
                scenario_producer_signoff_sha256
            ),
        )
    main_cells = tuple(config["go_no_go"]["required_main_cells"])
    primary_cell = main_cells[0]
    pair_count = int(config["matched_world_milp"]["pairs_per_model"])
    all_records: list[RoutePairRecord] = []
    for model_key in required_models:
        all_records.extend(
            build_route_pairs(
                worlds,
                model_key=model_key,
                primary_cell=primary_cell,
                pair_count=pair_count,
                closure_budget_us=_model_budget(
                    lock, model_key=model_key, main_cells=main_cells
                ),
                starvation_us=float(lock["models"][model_key]["starvation_us"]),
            )
        )
    solution_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Mapping[str, Any]] = {}
    for model_key in required_models:
        records = [record for record in all_records if record.pair.model_key == model_key]
        rows, summary = solve_route_pairs(
            records,
            max_gap=float(config["matched_world_milp"]["max_solver_optimality_gap"]),
        )
        solution_rows.extend(rows)
        gap_pass = float(summary["median_normalized_empirical_cvar99_gap"]) >= float(
            config["matched_world_milp"]["min_exact_oracle_median_normalized_cvar_gap"]
        )
        flip_pass = float(summary["mean_r0_first_action_flip_rate"]) >= float(
            config["matched_world_milp"]["min_r0_optimal_first_action_flip_rate"]
        )
        model_summaries[model_key] = {
            **summary,
            "gap_gate_pass": gap_pass,
            "flip_gate_pass": flip_pass,
            "model_gate_pass": gap_pass and flip_pass,
        }
    gate_pass = all(bool(row["model_gate_pass"]) for row in model_summaries.values())
    with atomic_output_directory(output_dir) as temporary:
        oracle_signoff_sha256 = None
        if mode == "formal":
            try:
                oracle_signoff_sha256 = materialize_verified_signoff(
                    signoff_path, temporary
                )
            except FormalProvenanceError as exc:
                raise OracleRunnerError(str(exc)) from exc
        instance_path = temporary / "milp_instances.jsonl"
        solution_path = temporary / "milp_solutions.jsonl"
        with instance_path.open("x", encoding="utf-8") as handle:
            for record in sorted(all_records, key=lambda item: item.pair.pair_id):
                handle.write(json.dumps(_instance_dict(record), sort_keys=True) + "\n")
        with solution_path.open("x", encoding="utf-8") as handle:
            for row in sorted(
                solution_rows, key=lambda item: (item["pair_id"], item["information_level"])
            ):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        status = add_self_hash(
            {
                "schema_version": "ric-oracle-status-v1",
                "status": "CALIBRATION_ORACLE_COMPLETE" if mode == "formal" else "NOT_TESTED",
                "scientific_result": False,
                "mode": mode,
                "role": "calibration",
                "config_sha256": config_sha,
                "protocol_sha256": protocol_sha,
                "run_oracle_source_sha256": _source_sha256(),
                "signoff_sha256": oracle_signoff_sha256,
                "build_scenarios_source_sha256": _scenario_source_sha256(),
                "calibration_lock_sha256": lock["manifest_sha256"],
                "calibration_lock_file_sha256": sha256_file(
                    calibration_lock_path
                ),
                "scenario_tree_sha256": scenario_tree_hashes,
                "scenario_tree_file_sha256": scenario_tree_file_hashes,
                "scenario_producer_signoff_sha256": (
                    scenario_producer_signoff_sha256
                ),
                **{field: lock.get(field) for field in migration_fields},
                "milp_instances_sha256": sha256_file(instance_path),
                "milp_solutions_sha256": sha256_file(solution_path),
                "model_summaries": model_summaries,
                "oracle_gate_pass": gate_pass,
            }
        )
        (temporary / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return status


def _require_formal_signoff(
    path: Path | None,
    *,
    config_path: Path,
    protocol_path: Path,
    calibration_lock_path: Path,
    calibration_lock: Mapping[str, Any],
    scenario_tree_hashes: Mapping[str, str],
    scenario_tree_file_hashes: Mapping[str, str],
    scenario_producer_signoff_sha256: Mapping[str, str],
) -> Mapping[str, Any]:
    expected = {
        "stage": "oracle",
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(protocol_path),
        "run_oracle_source_sha256": _source_sha256(),
        "run_experiment_source_sha256": _run_experiment_source_sha256(),
        "build_scenarios_source_sha256": _scenario_source_sha256(),
        "calibration_lock_sha256": calibration_lock.get("manifest_sha256"),
        "calibration_lock_file_sha256": sha256_file(calibration_lock_path),
        "scenario_tree_sha256": dict(scenario_tree_hashes),
        "scenario_tree_file_sha256": dict(scenario_tree_file_hashes),
        "scenario_producer_signoff_sha256": dict(
            scenario_producer_signoff_sha256
        ),
        "consumer_amendment_sha256": calibration_lock.get(
            "consumer_amendment_sha256"
        ),
        "historical_reviewed_source_snapshot_sha256": calibration_lock.get(
            "historical_reviewed_source_snapshot_sha256"
        ),
        "pre_outcome_attestation_sha256": calibration_lock.get(
            "pre_outcome_attestation_sha256"
        ),
        "pre_outcome_producer_signoff_file_sha256": calibration_lock.get(
            "pre_outcome_producer_signoff_file_sha256"
        ),
        "pre_outcome_producer_signoff_self_hash": calibration_lock.get(
            "pre_outcome_producer_signoff_self_hash"
        ),
        "authoritative_bundle_root": calibration_lock.get(
            "authoritative_bundle_root"
        ),
    }
    try:
        return verify_phase4_signoff(
            path,
            repo_root=REPO_ROOT,
            expected_fields=expected,
            required_source_paths=ORACLE_SOURCE_PATHS,
            required_reviewed_scope_paths=(
                *canonical_reviewed_scope_paths(REPO_ROOT, ORACLE_SOURCE_PATHS),
                DEFAULT_CONSUMER_AMENDMENT,
            ),
        )
    except FormalProvenanceError as exc:
        raise OracleRunnerError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-dir", type=Path, action="append", required=True)
    parser.add_argument("--calibration-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--signoff", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = run_oracle_pipeline(
        scenario_dirs=args.scenario_dir,
        calibration_lock_path=args.calibration_lock,
        output_dir=args.output_dir,
        mode=args.mode,
        config_path=args.config,
        protocol_path=args.protocol,
        signoff_path=args.signoff,
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
