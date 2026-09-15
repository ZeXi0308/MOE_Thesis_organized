from __future__ import annotations

"""Gate 2: exact future-known fixed-replica assignment/hold Oracle for bounded instances."""

import argparse
from collections import Counter
from itertools import product
import math
from pathlib import Path

try:
    from .core import (
        Contribution,
        ProtocolError,
        ReplayConfig,
        ServiceCatalog,
        clustered_bootstrap_mean_ci,
        objective_key,
        read_instances,
        relative_latency_gain,
        sha256_file,
        simulate_assignment,
        write_json,
        write_jsonl,
    )
    from .policies import HashPolicy, LeastLoadPolicy, assign_online
    from .build_fixed_replica_instances import validate_instance_contracts, validate_instance_metadata, validate_instance_split_disjointness
except ImportError:
    from core import Contribution, ProtocolError, ReplayConfig, ServiceCatalog, clustered_bootstrap_mean_ci, objective_key, read_instances, relative_latency_gain, sha256_file, simulate_assignment, write_json, write_jsonl
    from policies import HashPolicy, LeastLoadPolicy, assign_online
    from build_fixed_replica_instances import validate_instance_contracts, validate_instance_metadata, validate_instance_split_disjointness


FORMAL_HOLD_GRID_US = (0.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--service-curve", required=True)
    parser.add_argument("--holds-us", type=float, nargs="+", default=[0, 5, 10, 20, 50, 100])
    parser.add_argument("--remote-latency-us", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--required-remote-latency-us",
        type=float,
        help="formal Gate cell; defaults to the largest supplied remote latency",
    )
    parser.add_argument("--remote-bytes-per-row", type=int, default=0)
    parser.add_argument("--controller-latency-us", type=float, default=0.0)
    parser.add_argument("--seal-cost-us", type=float, default=0.0)
    parser.add_argument("--launch-cost-us", type=float, default=0.0)
    parser.add_argument("--max-batch-rows", type=int)
    parser.add_argument("--max-exact-states", type=int, default=2_000_000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--min-evaluation-clusters-per-cell", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _replay_costs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "remote_bytes_per_row": int(getattr(args, "remote_bytes_per_row", 0)),
        "controller_latency_us": float(getattr(args, "controller_latency_us", 0.0)),
        "seal_cost_us": float(getattr(args, "seal_cost_us", 0.0)),
        "launch_cost_us": float(getattr(args, "launch_cost_us", 0.0)),
        "max_batch_rows": getattr(args, "max_batch_rows", None),
    }


def _contributions(instance: dict[str, object]) -> list[Contribution]:
    raw = instance.get("contributions")
    if (
        not isinstance(raw, list)
        or len(raw) != int(instance.get("contribution_count", -1))
        or any(not isinstance(item, dict) for item in raw)
    ):
        raise ProtocolError("instance contribution count mismatch")
    return [Contribution.from_mapping(item) for item in raw]


def _weak_compositions(total: int, parts: int):
    """Yield all ordered non-negative ``parts``-tuples summing to total."""

    if parts <= 0:
        return
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for suffix in _weak_compositions(total - first, parts - 1):
            yield (first, *suffix)


def _structured_assignment_count(
    contributions: list[Contribution], replicas: int, *, serialize_decisions: bool = False
) -> tuple[int, list[int], list[tuple[tuple[object, ...], list[int], tuple[int, ...]]]]:
    """Count a symmetry-reduced exact assignment space.

    Contributions whose expert appears once in the instance are exchangeable
    when request, timing, source and legal targets match: the event engine sees
    identical singleton service jobs and request completion depends only on the
    per-replica count. Repeated experts remain explicit because their placement
    controls batching. This converts the common top-k=16/no-repeat case from
    ``R**N`` label permutations into weak compositions without approximation.
    """

    counts = Counter(item.expert_id for item in contributions)
    explicit = [index for index, item in enumerate(contributions) if counts[item.expert_id] > 1]
    grouped: dict[tuple[object, ...], list[int]] = {}
    for index, item in enumerate(contributions):
        if counts[item.expert_id] > 1:
            continue
        legal = item.legal_replicas(replicas)
        key = (
            item.request_id,
            item.input_event_id,
            item.topk_slot if serialize_decisions else None,
            item.dispatch_ready_us,
            item.deadline_us,
            item.source_rank,
            legal,
        )
        grouped.setdefault(key, []).append(index)
    groups = [
        (key, sorted(indices, key=lambda index: contributions[index].contribution_id), key[-1])
        for key, indices in sorted(grouped.items(), key=lambda value: repr(value[0]))
    ]
    count = 1
    for index in explicit:
        count *= len(contributions[index].legal_replicas(replicas))
    for _key, indices, legal in groups:
        count *= math.comb(len(indices) + len(legal) - 1, len(legal) - 1)
    return count, explicit, groups


def _structured_assignments(
    contributions: list[Contribution], replicas: int, *, serialize_decisions: bool = False
):
    _count, explicit, groups = _structured_assignment_count(
        contributions, replicas, serialize_decisions=serialize_decisions
    )
    explicit_options = [contributions[index].legal_replicas(replicas) for index in explicit]
    group_options = [
        tuple(_weak_compositions(len(indices), len(legal)))
        for _key, indices, legal in groups
    ]
    for explicit_values in product(*explicit_options) if explicit_options else [()]:
        for compositions in product(*group_options) if group_options else [()]:
            assignment = [-1] * len(contributions)
            for index, replica in zip(explicit, explicit_values):
                assignment[index] = int(replica)
            for (_key, indices, legal), composition in zip(groups, compositions):
                cursor = 0
                for replica, amount in zip(legal, composition):
                    for index in indices[cursor : cursor + amount]:
                        assignment[index] = int(replica)
                    cursor += amount
            if any(value < 0 for value in assignment):
                raise AssertionError("structured assignment left an action unset")
            yield assignment


def _singleton_action_factor(
    groups: list[tuple[tuple[object, ...], list[int], tuple[int, ...]]],
    holds: tuple[float, ...],
) -> int:
    """Count exact exchangeability classes over joint (replica, hold) actions."""

    factor = 1
    for _key, indices, legal in groups:
        categories = len(legal) * len(holds)
        factor *= math.comb(len(indices) + categories - 1, categories - 1)
    return factor


def _structured_action_count(
    contributions: list[Contribution],
    replicas: int,
    holds: tuple[float, ...],
    *,
    stop_after: int,
    serialize_decisions: bool,
) -> tuple[int, bool]:
    """Count assignment plus every active per-queue hold, stopping above a cap."""

    _assignments, explicit, groups = _structured_assignment_count(
        contributions, replicas, serialize_decisions=serialize_decisions
    )
    singleton_factor = _singleton_action_factor(groups, holds)
    if singleton_factor > stop_after:
        return singleton_factor, False
    explicit_options = [contributions[index].legal_replicas(replicas) for index in explicit]
    count = 0
    for explicit_values in product(*explicit_options) if explicit_options else [()]:
        active = {
            (int(replica), contributions[index].expert_id)
            for index, replica in zip(explicit, explicit_values)
        }
        count += (len(holds) ** len(active)) * singleton_factor
        if count > stop_after:
            return count, False
    return count, True


def _structured_actions(
    contributions: list[Contribution], replicas: int, holds: tuple[float, ...],
    *, serialize_decisions: bool,
):
    """Yield one representative of every exact joint assignment/hold class.

    Repeated experts remain contribution-explicit. For globally singleton
    experts, contributions with identical request/timing/source/legal state are
    exchangeable, so weak compositions over joint ``(replica, hold)``
    categories preserve every distinct event-engine outcome. Positive singleton
    holds are intentionally retained: they can change EDF release order even
    when they cannot create a larger expert batch.
    """

    _count, explicit, groups = _structured_assignment_count(
        contributions, replicas, serialize_decisions=serialize_decisions
    )
    explicit_options = [contributions[index].legal_replicas(replicas) for index in explicit]
    group_categories = [
        tuple((int(replica), float(hold)) for replica in legal for hold in holds)
        for _key, _indices, legal in groups
    ]
    group_options = [
        tuple(_weak_compositions(len(indices), len(categories)))
        for (_key, indices, _legal), categories in zip(groups, group_categories)
    ]
    for explicit_values in product(*explicit_options) if explicit_options else [()]:
        assignment = [-1] * len(contributions)
        for index, replica in zip(explicit, explicit_values):
            assignment[index] = int(replica)
        active_explicit = tuple(
            sorted(
                {
                    (int(replica), contributions[index].expert_id)
                    for index, replica in zip(explicit, explicit_values)
                }
            )
        )
        explicit_hold_options = (
            product(holds, repeat=len(active_explicit)) if active_explicit else [()]
        )
        for explicit_hold_values in explicit_hold_options:
            explicit_holds = dict(zip(active_explicit, explicit_hold_values))
            for compositions in product(*group_options) if group_options else [()]:
                action_assignment = list(assignment)
                hold_map = dict(explicit_holds)
                for ((_key, indices, _legal), categories, composition) in zip(
                    groups, group_categories, compositions
                ):
                    cursor = 0
                    for (replica, hold), amount in zip(categories, composition):
                        for index in indices[cursor : cursor + amount]:
                            action_assignment[index] = replica
                            hold_map[(replica, contributions[index].expert_id)] = hold
                        cursor += amount
                if any(value < 0 for value in action_assignment):
                    raise AssertionError("structured action left an assignment unset")
                active_keys = {
                    (replica, item.expert_id)
                    for item, replica in zip(contributions, action_assignment)
                }
                if set(hold_map) != active_keys:
                    raise AssertionError("structured action left a queue hold unset")
                yield action_assignment, hold_map


def _minimum_orbit_flip_rate(
    current: list[int],
    candidate: list[int],
    explicit: list[int],
    groups: list[tuple[tuple[object, ...], list[int], tuple[int, ...]]],
) -> float:
    """Minimum raw Hamming distance over an exchangeability-class orbit."""

    matches = sum(current[index] == candidate[index] for index in explicit)
    for _key, indices, _legal in groups:
        current_counts = Counter(current[index] for index in indices)
        candidate_counts = Counter(candidate[index] for index in indices)
        matches += sum(
            min(current_counts[replica], candidate_counts[replica])
            for replica in set(current_counts) | set(candidate_counts)
        )
    return 1.0 - matches / len(current)


def solve_instance(instance, catalog, args, remote_latency):
    contributions = _contributions(instance)
    replicas = int(instance["replica_count"])
    holds = tuple(sorted(set(float(value) for value in args.holds_us)))
    if not holds or any(value < 0 for value in holds):
        raise ProtocolError("Oracle hold grid must be non-empty and non-negative")
    replay_costs = _replay_costs(args)
    serialize_decisions = float(replay_costs["controller_latency_us"]) > 0
    assignment_states, explicit, groups = _structured_assignment_count(
        contributions, replicas, serialize_decisions=serialize_decisions
    )
    raw_assignment_states = math.prod(
        len(item.legal_replicas(replicas)) for item in contributions
    )
    if assignment_states > args.max_exact_states:
        return {
            "instance_id": instance["instance_id"],
            "model": instance["model"],
            "phase": instance["phase"],
            "layer": instance["layer"],
            "split": instance["split"],
            "cluster_id": instance["cluster_id"],
            "independent_cluster_id": instance.get("independent_cluster_id", instance["cluster_id"]),
            "remote_latency_us": remote_latency,
            "status": "UNSOLVED_EXACT_STATE_LIMIT",
            "exact": False,
            "solver": "SYMMETRY_REDUCED_EXACT_ENUMERATION",
            "raw_assignment_states": raw_assignment_states,
            "structured_assignment_states": assignment_states,
            "states_required_at_least": assignment_states,
            "max_exact_states": args.max_exact_states,
        }
    # Count the complete structured assignment + every active per-queue hold
    # before evaluating it. Exceeding the cap is INVALID, never heuristic exact.
    state_count, count_complete = _structured_action_count(
        contributions,
        replicas,
        holds,
        stop_after=args.max_exact_states,
        serialize_decisions=serialize_decisions,
    )
    if not count_complete:
        return {
            "instance_id": instance["instance_id"],
            "model": instance["model"],
            "phase": instance["phase"],
            "layer": instance["layer"],
            "split": instance["split"],
            "cluster_id": instance["cluster_id"],
            "independent_cluster_id": instance.get("independent_cluster_id", instance["cluster_id"]),
            "remote_latency_us": remote_latency,
            "status": "UNSOLVED_EXACT_STATE_LIMIT",
            "exact": False,
            "solver": "SYMMETRY_REDUCED_EXACT_ENUMERATION",
            "raw_assignment_states": raw_assignment_states,
            "structured_assignment_states": assignment_states,
            "structured_action_states_at_least": state_count,
            "states_required_at_least": state_count,
            "max_exact_states": args.max_exact_states,
        }
    minimum_hold = min(holds)
    current_candidates = []
    # Hash is a production run salt, not a window/instance salt.  Re-keying it
    # after chunking would change the declared current baseline itself.
    instance_seed = int(getattr(args, "seed", 20260725))
    for current_policy in (
        HashPolicy(instance_seed),
        LeastLoadPolicy(remote_latency),
    ):
        assignment = assign_online(
            contributions,
            current_policy,
            catalog,
            replicas,
            hold_us=minimum_hold,
            remote_bytes_per_row=int(replay_costs["remote_bytes_per_row"]),
            max_batch_rows=replay_costs["max_batch_rows"],
            controller_latency_us=float(replay_costs["controller_latency_us"]),
            seal_cost_us=float(replay_costs["seal_cost_us"]),
            launch_cost_us=float(replay_costs["launch_cost_us"]),
            remote_latency_us=remote_latency,
        )
        metrics = simulate_assignment(
            contributions,
            assignment,
            catalog,
            ReplayConfig(
                replicas,
                hold_us=minimum_hold,
                remote_latency_us=remote_latency,
                **replay_costs,
            ),
        )
        current_candidates.append((current_policy.name, assignment, metrics))
    current_name, current_assignment, baseline = max(
        current_candidates,
        key=lambda value: (objective_key(value[2]), value[0]),
    )
    best = None
    best_assignment = None
    best_holds = None
    optimal_count = 0
    minimum_optimal_flip = math.inf
    objective_values = set()
    states = 0
    for assignment, hold_map in _structured_actions(
        contributions, replicas, holds, serialize_decisions=serialize_decisions
    ):
        states += 1
        result = simulate_assignment(
            contributions,
            assignment,
            catalog,
            ReplayConfig(
                replicas,
                hold_us=minimum_hold,
                hold_by_queue=hold_map,
                remote_latency_us=remote_latency,
                **replay_costs,
            ),
        )
        objective_values.add(tuple(round(value, 9) for value in objective_key(result)))
        if best is None or objective_key(result) > objective_key(best):
            best = result
            best_assignment = list(assignment)
            best_holds = dict(hold_map)
            optimal_count = 1
            minimum_optimal_flip = _minimum_orbit_flip_rate(
                current_assignment, assignment, explicit, groups
            )
        elif objective_key(result) == objective_key(best):
            optimal_count += 1
            candidate_flip = _minimum_orbit_flip_rate(
                current_assignment, assignment, explicit, groups
            )
            if candidate_flip < minimum_optimal_flip:
                best = result
                best_assignment = list(assignment)
                best_holds = dict(hold_map)
                minimum_optimal_flip = candidate_flip
    assert best is not None and best_assignment is not None and best_holds is not None
    return {
        "instance_id": instance["instance_id"],
        "model": instance["model"],
        "phase": instance["phase"],
        "layer": instance["layer"],
        "split": instance["split"],
        "cluster_id": instance["cluster_id"],
        "independent_cluster_id": instance.get("independent_cluster_id", instance["cluster_id"]),
        "remote_latency_us": remote_latency,
        "replay_costs": replay_costs,
        "status": "SOLVED_EXACT",
        "exact": True,
        "solver": "SYMMETRY_REDUCED_EXACT_ENUMERATION",
        "raw_assignment_states": raw_assignment_states,
        "structured_assignment_states": assignment_states,
        "structured_action_states": state_count,
        "states_evaluated": states,
        "optimal_equivalence_class_count": optimal_count,
        "objective_value_count": len(objective_values),
        "actionable": len(objective_values) > 1,
        "action_flip_rate": minimum_optimal_flip,
        "action_flip_rate_semantics": "minimum raw Hamming distance over optimal symmetry orbits",
        "current_assignment": current_assignment,
        "current_policy": current_name,
        "current_candidates": {
            name: {"assignment": assignment, "metrics": metrics}
            for name, assignment, metrics in current_candidates
        },
        "oracle_assignment": best_assignment,
        "oracle_assignment_is_symmetry_representative": True,
        "oracle_holds_us": {
            f"replica={replica},expert={expert}": hold
            for (replica, expert), hold in sorted(best_holds.items())
        },
        "default_hold_us": minimum_hold,
        "current_metrics": baseline,
        "oracle_metrics": best,
        "completion_gain": relative_latency_gain(
            float(baseline["mean_completion_us"]), float(best["mean_completion_us"])
        ),
        "service_gain": relative_latency_gain(
            float(baseline["total_service_us"]), float(best["total_service_us"])
        ),
        "slo_attainment_delta": float(best["slo_attainment"]) - float(baseline["slo_attainment"]),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    holds = tuple(sorted(set(float(value) for value in args.holds_us)))
    remotes = tuple(sorted(set(float(value) for value in args.remote_latency_us)))
    if not args.smoke:
        if holds != FORMAL_HOLD_GRID_US or len(args.holds_us) != len(FORMAL_HOLD_GRID_US):
            raise ProtocolError(f"formal Gate 2 hold grid must be {FORMAL_HOLD_GRID_US}")
        if (
            len(remotes) < 3
            or remotes[0] != 0.0
            or args.required_remote_latency_us is None
            or float(args.required_remote_latency_us) != remotes[-1]
            or remotes[-1] <= 0
        ):
            raise ProtocolError(
                "formal Gate 2 requires zero plus two calibrated remote cells and an explicit conservative maximum"
            )
        if (
            args.min_evaluation_clusters_per_cell != 5
            or args.bootstrap != 2000
            or args.seed != 20260725
        ):
            raise ProtocolError("formal Gate 2 statistics must use 5 clusters, 2000 bootstraps and seed 20260725")
    if args.min_evaluation_clusters_per_cell < 2:
        raise ProtocolError("Gate 2 requires at least two independent evaluation clusters per cell")
    instances = read_instances(args.instances)
    validate_instance_contracts(instances, require_formal_v3=not args.smoke)
    validate_instance_split_disjointness(instances)
    instance_meta = validate_instance_metadata(
        args.instances,
        service_curve_path=args.service_curve,
        expected_smoke=bool(args.smoke),
    )
    catalog = ServiceCatalog.from_csv(args.service_curve)
    results = [
        solve_instance(instance, catalog, args, remote)
        for instance in instances
        for remote in remotes
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_path = output_dir / "oracle_results.jsonl"
    write_jsonl(oracle_path, results)
    unsolved = [row for row in results if not row.get("exact")]
    evaluation = [row for row in results if row.get("exact") and row.get("split") == "evaluation"]
    summaries = []
    grouped = {}
    for row in evaluation:
        key = (row["model"], row["phase"], row["remote_latency_us"])
        grouped.setdefault(key, []).append(row)
    for key, values in sorted(grouped.items()):
        point, low, high = clustered_bootstrap_mean_ci(
            [float(row["completion_gain"]) for row in values],
            [str(row["independent_cluster_id"]) for row in values],
            replicates=args.bootstrap,
            seed=args.seed,
        )
        summaries.append(
            {
                "model": key[0],
                "phase": key[1],
                "remote_latency_us": key[2],
                "completion_gain": point,
                "ci_low": low,
                "ci_high": high,
                "action_flip_rate": sum(float(row["action_flip_rate"]) for row in values) / len(values),
                "actionable_rate": sum(bool(row["actionable"]) for row in values) / len(values),
                "instances": len(values),
                "request_clusters": len({str(row["independent_cluster_id"]) for row in values}),
            }
        )
    models = sorted({str(instance["model"]) for instance in instances})
    phases = sorted({str(instance["phase"]) for instance in instances})
    required_remote = (
        max(remotes)
        if args.required_remote_latency_us is None
        else args.required_remote_latency_us
    )
    if required_remote not in set(remotes):
        raise ProtocolError("required remote latency must be one of --remote-latency-us")
    common = {}
    for row in summaries:
        common.setdefault((row["phase"], row["remote_latency_us"]), {})[row["model"]] = row
    preregistered_keys = tuple((phase, remote) for phase in phases for remote in remotes)
    complete = {
        key: common[key]
        for key in preregistered_keys
        if key in common and set(common[key]) == set(models) and len(models) >= 2
    }
    missing = tuple(key for key in preregistered_keys if key not in complete)
    undersized_cells = tuple(
        (key, model, int(row["request_clusters"]))
        for key, values in complete.items()
        for model, row in values.items()
        if int(row["request_clusters"]) < args.min_evaluation_clusters_per_cell
    )
    passing = [
        key for key, values in complete.items()
        if key[1] == required_remote
        and all(
            float(row["completion_gain"]) >= 0.15
            and float(row["ci_low"]) > 0.10
            and float(row["action_flip_rate"]) >= 0.20
            and float(row["actionable_rate"]) >= 0.20
            for row in values.values()
        )
    ]
    if args.smoke:
        status = "SMOKE_ONLY"
    elif unsolved:
        status = "INVALID_ORACLE_NOT_EXACT"
    elif len(models) < 2:
        status = "INVALID_NEED_TWO_MODELS"
    elif missing:
        status = "INVALID_INCOMPLETE_PREREGISTERED_COMMON_CELLS"
    elif undersized_cells:
        status = "INVALID_INSUFFICIENT_INDEPENDENT_UNITS"
    else:
        # Current v2 instances intentionally contain one input event/request
        # at one layer. They are exact for that local queue action space, but
        # cannot propagate counterfactual delays into later layers/steps and
        # therefore cannot issue request-SLO PASS/KILL decisions.
        status = "INVALID_REQUEST_DAG_REPLAY_NOT_IMPLEMENTED"
    summary = {
        "schema": "bcrd-gate2-v2",
        "status": status,
        "smoke": bool(args.smoke),
        "models": models,
        "required_remote_latency_us": required_remote,
        "remote_latency_grid_us": list(remotes),
        "hold_grid_us": list(holds),
        "replay_costs": _replay_costs(args),
        "unsolved_instances": len(unsolved),
        "preregistered_common_cells": [list(key) for key in preregistered_keys],
        "missing_preregistered_common_cells": [list(key) for key in missing],
        "minimum_evaluation_clusters_per_cell": args.min_evaluation_clusters_per_cell,
        "undersized_independent_cells": [
            [*key, model, count] for key, model, count in undersized_cells
        ],
        "passing_common_cells": [list(key) for key in passing],
        "cells": summaries,
        "instances_sha256": sha256_file(args.instances),
        "instances_meta_sha256": sha256_file(Path(args.instances).with_suffix(".meta.json")),
        "gate1_summary_sha256": instance_meta.get("gate1_summary_sha256"),
        "service_curve_sha256": sha256_file(args.service_curve),
        "oracle_results_sha256": sha256_file(oracle_path),
        "counterfactual_request_dag_propagation": False,
        "evidence_boundary": (
            "SMOKE_ONLY exact-enumeration code-path validation" if args.smoke else
            "single-layer future-known queue replay only; request-DAG/SLO and physical EP claims are forbidden"
        ),
    }
    write_json(output_dir / "gate2_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    result = run(args)
    print(result["status"])


if __name__ == "__main__":
    main()
