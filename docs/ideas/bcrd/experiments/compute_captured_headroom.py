from __future__ import annotations

"""Aggregate Gate 3, compute captured headroom, confidence intervals and decision."""

import argparse
import json
from pathlib import Path

try:
    from .core import ProtocolError, clustered_bootstrap_mean_ci, objective_key, read_json, relative_latency_gain, sha256_file, write_json
except ImportError:
    from core import ProtocolError, clustered_bootstrap_mean_ci, objective_key, read_json, relative_latency_gain, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-results", required=True)
    parser.add_argument("--oracle-results", required=True)
    parser.add_argument("--resolved-plan", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: argparse.Namespace) -> dict[str, object]:
    plan = read_json(args.resolved_plan)
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != "bcrd-gate3-plan-v2"
        or bool(plan.get("smoke")) != bool(args.smoke)
    ):
        raise ProtocolError("resolved plan smoke/formal mode mismatch")
    if plan.get("policy_results_sha256") != sha256_file(args.policy_results):
        raise ProtocolError("policy results differ from the resolved plan")
    if plan.get("oracle_results_sha256") != sha256_file(args.oracle_results):
        raise ProtocolError("Oracle results differ from the resolved plan")
    replay_costs = plan.get("replay_costs")
    if not isinstance(replay_costs, dict):
        raise ProtocolError("resolved plan replay costs are missing")
    selected_remote = float(replay_costs.get("remote_latency_us", -1.0))
    policy_rows = _jsonl(args.policy_results)
    oracle_rows = [
        row for row in _jsonl(args.oracle_results)
        if row.get("exact")
        and row.get("split") == "evaluation"
        and float(row.get("remote_latency_us", -1.0)) == selected_remote
    ]
    if not policy_rows or not oracle_rows:
        raise ProtocolError("policy and exact Oracle results are required")
    oracle = {}
    for row in oracle_rows:
        key = (row["instance_id"], float(row["remote_latency_us"]))
        if key in oracle:
            raise ProtocolError(f"{row['instance_id']}: duplicate exact Oracle row")
        oracle[key] = row
    by_instance = {}
    for row in policy_rows:
        if float(row.get("remote_latency_us", -1.0)) != selected_remote:
            raise ProtocolError("policy result remote cell differs from the resolved plan")
        policies = by_instance.setdefault(row["instance_id"], {})
        if row["policy"] in policies:
            raise ProtocolError(f"{row['instance_id']}: duplicate policy result")
        policies[row["policy"]] = row
    analyses = []
    for instance_id, policies in sorted(by_instance.items()):
        required = {"current_hash", "current_least_load", "random", "threshold", "greedy", "bcrd"}
        if set(policies) != required:
            raise ProtocolError(f"{instance_id}: missing policies {sorted(required - set(policies))}")
        current_name = max(
            ("current_hash", "current_least_load"),
            key=lambda name: (objective_key(policies[name]["metrics"]), name),
        )
        simple_name = max(
            ("threshold", "greedy"),
            key=lambda name: (objective_key(policies[name]["metrics"]), name),
        )
        current = float(policies[current_name]["metrics"]["mean_completion_us"])
        simple = float(policies[simple_name]["metrics"]["mean_completion_us"])
        proposed = float(policies["bcrd"]["metrics"]["mean_completion_us"])
        matches = [value for (key_id, _), value in oracle.items() if key_id == instance_id]
        if len(matches) != 1:
            raise ProtocolError(f"{instance_id}: need exactly one Oracle remote-cost row")
        oracle_latency = float(matches[0]["oracle_metrics"]["mean_completion_us"])
        denominator = current - oracle_latency
        captured = (current - simple) / denominator if denominator > 1e-12 else 1.0
        proposed_captured = (current - proposed) / denominator if denominator > 1e-12 else 1.0
        proposed_metrics = policies["bcrd"]["metrics"]
        controller_tax_per_request = (
            float(policies["bcrd"].get("modeled_controller_latency_us_per_action", 0.0))
            * float(proposed_metrics["contributions"])
            / max(float(proposed_metrics["requests"]), 1.0)
        )
        gross_gain_before_controller = max(
            current - (proposed - controller_tax_per_request), 0.0
        )
        analyses.append(
            {
                "instance_id": instance_id,
                "model": policies["bcrd"]["model"],
                "phase": policies["bcrd"]["phase"],
                "cluster_id": policies["bcrd"]["cluster_id"],
                "independent_cluster_id": policies["bcrd"]["independent_cluster_id"],
                "current_policy": current_name,
                "strongest_simple": simple_name,
                "current_latency_us": current,
                "simple_latency_us": simple,
                "proposed_latency_us": proposed,
                "oracle_latency_us": oracle_latency,
                "simple_captured_headroom": captured,
                "proposed_captured_headroom": proposed_captured,
                "proposed_net_gain": relative_latency_gain(current, proposed),
                "proposed_incremental_vs_simple": relative_latency_gain(simple, proposed),
                "controller_tax_us_per_request": controller_tax_per_request,
                "decision_tax_fraction": (
                    controller_tax_per_request / max(gross_gain_before_controller, 1e-9)
                    if controller_tax_per_request > 0
                    else 0.0
                ),
                "p99_regression": (
                    float(policies["bcrd"]["metrics"]["p99_completion_us"])
                    - float(policies[current_name]["metrics"]["p99_completion_us"])
                ) / max(float(policies[current_name]["metrics"]["p99_completion_us"]), 1e-9),
            }
        )
    grouped = {}
    for row in analyses:
        grouped.setdefault((row["model"], row["phase"]), []).append(row)
    cells = []
    for (model, phase), rows in sorted(grouped.items()):
        point, low, high = clustered_bootstrap_mean_ci(
            [float(row["proposed_net_gain"]) for row in rows],
            [str(row["independent_cluster_id"]) for row in rows],
            replicates=args.bootstrap,
            seed=args.seed,
        )
        cells.append(
            {
                "model": model,
                "phase": phase,
                "instances": len(rows),
                "request_clusters": len({str(row["independent_cluster_id"]) for row in rows}),
                "proposed_net_gain": point,
                "net_gain_ci_low": low,
                "net_gain_ci_high": high,
                "simple_captured_headroom": sum(float(row["simple_captured_headroom"]) for row in rows) / len(rows),
                "proposed_captured_headroom": sum(float(row["proposed_captured_headroom"]) for row in rows) / len(rows),
                "incremental_vs_simple": sum(float(row["proposed_incremental_vs_simple"]) for row in rows) / len(rows),
                "decision_tax_fraction": sum(float(row["decision_tax_fraction"]) for row in rows) / len(rows),
                "p99_regression": sum(float(row["p99_regression"]) for row in rows) / len(rows),
            }
        )
    models = sorted({row["model"] for row in cells})
    if args.smoke:
        status = "SMOKE_ONLY"
    elif any(float(row["simple_captured_headroom"]) >= 0.95 for row in cells):
        status = "SIMPLE_WINS"
    elif any(0.90 <= float(row["simple_captured_headroom"]) < 0.95 for row in cells):
        status = "KEEP_SIMPLE_CANCEL_CONTROLLER"
    elif len(models) < 2:
        status = "INVALID_NEED_TWO_MODELS"
    elif all(
        float(row["proposed_net_gain"]) >= 0.10
        and float(row["net_gain_ci_low"]) > 0.0
        and float(row["simple_captured_headroom"]) < 0.90
        and float(row["incremental_vs_simple"]) >= 0.03
        and float(row["decision_tax_fraction"]) <= 0.20
        and float(row["p99_regression"]) <= 0.02
        for row in cells
    ):
        status = "A100_CANDIDATE"
    else:
        status = "NO_GO_GATE3"
    decision = {
        "schema": "bcrd-gate3-decision-v1",
        "status": status,
        "smoke": bool(args.smoke),
        "models": models,
        "cells": cells,
        "per_instance": analyses,
        "inputs": {
            "policy_results_sha256": sha256_file(args.policy_results),
            "oracle_results_sha256": sha256_file(args.oracle_results),
            "resolved_plan_sha256": sha256_file(args.resolved_plan),
        },
        "evidence_boundary": (
            "SMOKE_ONLY; validates accounting and decision branches only" if args.smoke else
            "single-GPU logical replay candidate decision; 8xA100 physical Gate remains mandatory"
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis.json", decision)
    write_json(output_dir / "decision.json", decision)
    return decision


def main() -> None:
    args = parse_args()
    decision = run(args)
    print(decision["status"])


if __name__ == "__main__":
    main()
