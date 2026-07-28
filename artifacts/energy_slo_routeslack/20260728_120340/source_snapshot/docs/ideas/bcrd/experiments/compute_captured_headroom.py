from __future__ import annotations

"""Aggregate Gate 3, compute captured headroom, confidence intervals and decision."""

import argparse
import json
from pathlib import Path

try:
    from .core import ProtocolError, clustered_bootstrap_mean_ci, read_json, relative_latency_gain, sha256_file, write_json
except ImportError:
    from core import ProtocolError, clustered_bootstrap_mean_ci, read_json, relative_latency_gain, sha256_file, write_json


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
    if not isinstance(plan, dict) or bool(plan.get("smoke")) != bool(args.smoke):
        raise ProtocolError("resolved plan smoke/formal mode mismatch")
    policy_rows = _jsonl(args.policy_results)
    oracle_rows = [
        row for row in _jsonl(args.oracle_results)
        if row.get("exact") and row.get("split") == "evaluation"
    ]
    if not policy_rows or not oracle_rows:
        raise ProtocolError("policy and exact Oracle results are required")
    oracle = {}
    for row in oracle_rows:
        key = (row["instance_id"], float(row["remote_latency_us"]))
        oracle[key] = row
    by_instance = {}
    for row in policy_rows:
        by_instance.setdefault(row["instance_id"], {})[row["policy"]] = row
    analyses = []
    for instance_id, policies in sorted(by_instance.items()):
        required = {"current_hash", "current_least_load", "random", "threshold", "greedy", "bcrd"}
        if set(policies) != required:
            raise ProtocolError(f"{instance_id}: missing policies {sorted(required - set(policies))}")
        current_name = min(
            ("current_hash", "current_least_load"),
            key=lambda name: float(policies[name]["metrics"]["modeled_net_mean_completion_us"]),
        )
        simple_name = min(
            ("threshold", "greedy"),
            key=lambda name: float(policies[name]["metrics"]["modeled_net_mean_completion_us"]),
        )
        current = float(policies[current_name]["metrics"]["modeled_net_mean_completion_us"])
        simple = float(policies[simple_name]["metrics"]["modeled_net_mean_completion_us"])
        proposed = float(policies["bcrd"]["metrics"]["modeled_net_mean_completion_us"])
        matches = [value for (key_id, _), value in oracle.items() if key_id == instance_id]
        if len(matches) != 1:
            raise ProtocolError(f"{instance_id}: need exactly one Oracle remote-cost row")
        oracle_latency = float(matches[0]["oracle_metrics"]["mean_completion_us"])
        denominator = current - oracle_latency
        captured = (current - simple) / denominator if denominator > 1e-12 else 1.0
        proposed_captured = (current - proposed) / denominator if denominator > 1e-12 else 1.0
        analyses.append(
            {
                "instance_id": instance_id,
                "model": policies["bcrd"]["model"],
                "phase": policies["bcrd"]["phase"],
                "cluster_id": policies["bcrd"]["cluster_id"],
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
                "decision_tax_fraction": float(policies["bcrd"]["decision_overhead_us"]) / max(float(policies["bcrd"]["metrics"]["total_service_us"]), 1e-9),
                "p99_regression": (
                    float(policies["bcrd"]["metrics"]["modeled_net_p99_completion_us"])
                    - float(policies[current_name]["metrics"]["modeled_net_p99_completion_us"])
                ) / max(float(policies[current_name]["metrics"]["modeled_net_p99_completion_us"]), 1e-9),
            }
        )
    grouped = {}
    for row in analyses:
        grouped.setdefault((row["model"], row["phase"]), []).append(row)
    cells = []
    for (model, phase), rows in sorted(grouped.items()):
        point, low, high = clustered_bootstrap_mean_ci(
            [float(row["proposed_net_gain"]) for row in rows],
            [str(row["cluster_id"]) for row in rows],
            replicates=args.bootstrap,
            seed=args.seed,
        )
        cells.append(
            {
                "model": model,
                "phase": phase,
                "instances": len(rows),
                "request_clusters": len({str(row["cluster_id"]) for row in rows}),
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
