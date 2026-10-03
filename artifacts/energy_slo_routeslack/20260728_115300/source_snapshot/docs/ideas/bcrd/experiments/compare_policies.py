from __future__ import annotations

"""Gate 3 replay: current/random/threshold/greedy/BCRD on frozen instances."""

import argparse
import time
from pathlib import Path

try:
    from .core import Contribution, ProtocolError, ReplayConfig, ServiceCatalog, read_instances, read_json, sha256_file, simulate_assignment, write_json, write_jsonl
    from .policies import assign_online, make_policy
except ImportError:
    from core import Contribution, ProtocolError, ReplayConfig, ServiceCatalog, read_instances, read_json, sha256_file, simulate_assignment, write_json, write_jsonl
    from policies import assign_online, make_policy


POLICY_NAMES = ("current_hash", "current_least_load", "random", "threshold", "greedy", "bcrd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--service-curve", required=True)
    parser.add_argument("--gate2-summary", required=True)
    parser.add_argument("--oracle-results", required=True)
    parser.add_argument("--threshold-candidates", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--credit-candidates", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--hold-candidates-us", type=float, nargs="+", default=[0, 5, 10, 20, 50])
    parser.add_argument("--deadline-risk-weight", type=float, default=4.0)
    parser.add_argument("--remote-latency-us", type=float, default=0.0)
    parser.add_argument("--remote-bytes-per-row", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _contributions(instance):
    raw = instance.get("contributions")
    if not isinstance(raw, list):
        raise ProtocolError("instance contributions must be a list")
    return [Contribution.from_mapping(item) for item in raw if isinstance(item, dict)]


def _run_configuration(instance, catalog, args, name, threshold, credit, hold):
    contributions = _contributions(instance)
    policy = make_policy(
        name,
        seed=args.seed + sum(ord(char) for char in str(instance["instance_id"])),
        remote_latency_us=args.remote_latency_us,
        row_threshold=threshold,
        batching_credit_weight=credit,
        deadline_risk_weight=args.deadline_risk_weight,
    )
    started = time.perf_counter_ns()
    assignment = assign_online(contributions, policy, catalog, int(instance["replica_count"]))
    decision_overhead_us = (time.perf_counter_ns() - started) / 1000.0
    metrics = simulate_assignment(
        contributions,
        assignment,
        catalog,
        ReplayConfig(
            int(instance["replica_count"]),
            hold_us=hold,
            remote_latency_us=args.remote_latency_us,
            remote_bytes_per_row=args.remote_bytes_per_row,
        ),
    )
    metrics["modeled_net_mean_completion_us"] = float(metrics["mean_completion_us"]) + decision_overhead_us / int(metrics["requests"])
    metrics["modeled_net_p99_completion_us"] = float(metrics["p99_completion_us"]) + decision_overhead_us / int(metrics["requests"])
    return {
        "instance_id": instance["instance_id"],
        "model": instance["model"],
        "phase": instance["phase"],
        "layer": instance["layer"],
        "split": instance["split"],
        "cluster_id": instance["cluster_id"],
        "policy": name,
        "row_threshold": threshold if name == "threshold" else None,
        "batching_credit_weight": credit if name == "bcrd" else None,
        "hold_us": hold,
        "decision_overhead_us": decision_overhead_us,
        "assignment": assignment,
        "metrics": metrics,
    }


def _select(calibration, name):
    candidates = {}
    for row in calibration:
        key = (row["row_threshold"], row["batching_credit_weight"], row["hold_us"])
        candidates.setdefault(key, []).append(row)
    if not candidates:
        raise ProtocolError(f"no calibration candidates for {name}")
    def key_score(item):
        config, rows = item
        on_time = sum(float(row["metrics"]["slo_attainment"]) for row in rows) / len(rows)
        mean = sum(float(row["metrics"]["modeled_net_mean_completion_us"]) for row in rows) / len(rows)
        service = sum(float(row["metrics"]["total_service_us"]) for row in rows) / len(rows)
        return on_time, -mean, -service, tuple(-float(value or 0) for value in config)
    return max(candidates.items(), key=key_score)[0]


def run(args: argparse.Namespace) -> dict[str, object]:
    gate2 = read_json(args.gate2_summary)
    if not isinstance(gate2, dict):
        raise ProtocolError("Gate 2 summary must be an object")
    accepted = {"PASS_GATE2"} | ({"SMOKE_ONLY"} if args.smoke else set())
    if gate2.get("status") not in accepted:
        raise ProtocolError(f"Gate 2 status {gate2.get('status')!r} does not authorize Gate 3")
    instances = read_instances(args.instances)
    catalog = ServiceCatalog.from_csv(args.service_curve)
    oracle_rows = []
    with Path(args.oracle_results).open(encoding="utf-8") as handle:
        import json

        oracle_rows = [json.loads(line) for line in handle if line.strip()]
    exact_evaluation = {
        row["instance_id"]
        for row in oracle_rows
        if row.get("exact")
        and row.get("split") == "evaluation"
        and float(row.get("remote_latency_us", -1.0)) == args.remote_latency_us
    }
    expected_evaluation = {row["instance_id"] for row in instances if row.get("split") == "evaluation"}
    if exact_evaluation != expected_evaluation:
        raise ProtocolError("Gate 3 requires one exact matching-remote Oracle for every evaluation instance")

    calibration = []
    for instance in instances:
        if instance["split"] != "calibration":
            continue
        for threshold in args.threshold_candidates:
            for hold in args.hold_candidates_us:
                calibration.append(_run_configuration(instance, catalog, args, "threshold", threshold, 1.0, hold))
        for credit in args.credit_candidates:
            for hold in args.hold_candidates_us:
                calibration.append(_run_configuration(instance, catalog, args, "bcrd", 1, credit, hold))
    threshold_config = _select([row for row in calibration if row["policy"] == "threshold"], "threshold")
    bcrd_config = _select([row for row in calibration if row["policy"] == "bcrd"], "bcrd")

    results = []
    for instance in instances:
        if instance["split"] != "evaluation":
            continue
        for name in POLICY_NAMES:
            if name == "threshold":
                threshold, credit, hold = threshold_config
            elif name == "bcrd":
                threshold, credit, hold = bcrd_config
            else:
                threshold, credit, hold = 1, 1.0, 0.0
            results.append(_run_configuration(instance, catalog, args, name, int(threshold or 1), float(credit or 1.0), float(hold)))
    if not results:
        raise ProtocolError("no evaluation instances")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "policy_results.jsonl", results)
    plan = {
        "schema": "bcrd-gate3-plan-v1",
        "smoke": bool(args.smoke),
        "calibration_instances": len({row["instance_id"] for row in calibration}),
        "evaluation_instances": len({row["instance_id"] for row in results}),
        "selected_threshold": {
            "row_threshold": threshold_config[0], "hold_us": threshold_config[2]
        },
        "selected_bcrd": {
            "batching_credit_weight": bcrd_config[1],
            "hold_us": bcrd_config[2],
            "deadline_risk_weight": args.deadline_risk_weight,
        },
        "instances_sha256": sha256_file(args.instances),
        "service_curve_sha256": sha256_file(args.service_curve),
        "gate2_summary_sha256": sha256_file(args.gate2_summary),
        "oracle_results_sha256": sha256_file(args.oracle_results),
        "future_information_exposed_to_online_policy": False,
        "evidence_boundary": "SMOKE_ONLY replay" if args.smoke else "logical causal replay; no physical EP execution",
    }
    write_json(output_dir / "resolved_plan.json", plan)
    return plan


def main() -> None:
    args = parse_args()
    plan = run(args)
    print(f"evaluated {plan['evaluation_instances']} instances")


if __name__ == "__main__":
    main()
