from __future__ import annotations

"""Gate 2: exact future-known fixed-replica assignment/hold Oracle for bounded instances."""

import argparse
from itertools import product
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
    from .policies import LeastLoadPolicy, assign_online
except ImportError:
    from core import Contribution, ProtocolError, ReplayConfig, ServiceCatalog, clustered_bootstrap_mean_ci, objective_key, read_instances, relative_latency_gain, sha256_file, simulate_assignment, write_json, write_jsonl
    from policies import LeastLoadPolicy, assign_online


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
    parser.add_argument("--max-exact-states", type=int, default=2_000_000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _contributions(instance: dict[str, object]) -> list[Contribution]:
    raw = instance.get("contributions")
    if not isinstance(raw, list) or len(raw) != int(instance.get("contribution_count", -1)):
        raise ProtocolError("instance contribution count mismatch")
    return [Contribution.from_mapping(item) for item in raw if isinstance(item, dict)]


def solve_instance(instance, catalog, args, remote_latency):
    contributions = _contributions(instance)
    replicas = int(instance["replica_count"])
    state_count = replicas ** len(contributions) * len(args.holds_us)
    if state_count > args.max_exact_states:
        return {
            "instance_id": instance["instance_id"],
            "model": instance["model"],
            "phase": instance["phase"],
            "layer": instance["layer"],
            "split": instance["split"],
            "cluster_id": instance["cluster_id"],
            "remote_latency_us": remote_latency,
            "status": "UNSOLVED_EXACT_STATE_LIMIT",
            "exact": False,
            "states_required": state_count,
        }
    current_assignment = assign_online(
        contributions, LeastLoadPolicy(remote_latency), catalog, replicas
    )
    baseline = simulate_assignment(
        contributions,
        current_assignment,
        catalog,
        ReplayConfig(replicas, hold_us=0.0, remote_latency_us=remote_latency, remote_bytes_per_row=args.remote_bytes_per_row),
    )
    best = None
    best_assignment = None
    best_hold = None
    optimal_count = 0
    objective_values = set()
    states = 0
    for hold in sorted(set(args.holds_us)):
        config = ReplayConfig(
            replicas,
            hold_us=hold,
            remote_latency_us=remote_latency,
            remote_bytes_per_row=args.remote_bytes_per_row,
        )
        for assignment_tuple in product(range(replicas), repeat=len(contributions)):
            states += 1
            result = simulate_assignment(contributions, assignment_tuple, catalog, config)
            objective_values.add(tuple(round(value, 9) for value in objective_key(result)))
            if best is None or objective_key(result) > objective_key(best):
                best = result
                best_assignment = list(assignment_tuple)
                best_hold = hold
                optimal_count = 1
            elif objective_key(result) == objective_key(best):
                optimal_count += 1
    assert best is not None and best_assignment is not None and best_hold is not None
    return {
        "instance_id": instance["instance_id"],
        "model": instance["model"],
        "phase": instance["phase"],
        "layer": instance["layer"],
        "split": instance["split"],
        "cluster_id": instance["cluster_id"],
        "remote_latency_us": remote_latency,
        "status": "SOLVED_EXACT",
        "exact": True,
        "states_evaluated": states,
        "optimal_action_count": optimal_count,
        "objective_value_count": len(objective_values),
        "actionable": len(objective_values) > 1,
        "action_flip_rate": sum(a != b for a, b in zip(current_assignment, best_assignment)) / len(contributions),
        "current_assignment": current_assignment,
        "oracle_assignment": best_assignment,
        "oracle_hold_us": best_hold,
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
    instances = read_instances(args.instances)
    catalog = ServiceCatalog.from_csv(args.service_curve)
    results = [
        solve_instance(instance, catalog, args, remote)
        for instance in instances
        for remote in sorted(set(args.remote_latency_us))
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "oracle_results.jsonl", results)
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
            [str(row["cluster_id"]) for row in values],
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
                "request_clusters": len({str(row["cluster_id"]) for row in values}),
            }
        )
    models = sorted({str(row["model"]) for row in summaries})
    required_remote = (
        max(args.remote_latency_us)
        if args.required_remote_latency_us is None
        else args.required_remote_latency_us
    )
    if required_remote not in set(args.remote_latency_us):
        raise ProtocolError("required remote latency must be one of --remote-latency-us")
    common = {}
    for row in summaries:
        common.setdefault((row["phase"], row["remote_latency_us"]), {})[row["model"]] = row
    passing = [
        key for key, values in common.items()
        if key[1] == required_remote
        and len(values) == len(models) >= 2
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
    elif passing:
        status = "PASS_GATE2"
    elif summaries and all(float(row["completion_gain"]) < 0.10 for row in summaries):
        status = "KILL_BCRD"
    else:
        status = "NO_GO_GATE2"
    summary = {
        "schema": "bcrd-gate2-v1",
        "status": status,
        "smoke": bool(args.smoke),
        "models": models,
        "required_remote_latency_us": required_remote,
        "unsolved_instances": len(unsolved),
        "passing_common_cells": [list(key) for key in passing],
        "cells": summaries,
        "instances_sha256": sha256_file(args.instances),
        "service_curve_sha256": sha256_file(args.service_curve),
        "evidence_boundary": (
            "SMOKE_ONLY exact-enumeration code-path validation" if args.smoke else
            "future-known logical replay upper bound; no deployable policy or physical EP claim"
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
