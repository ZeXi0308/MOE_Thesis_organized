from __future__ import annotations

"""Run the entire BCRD pipeline with synthetic fixtures; never emits a scientific PASS."""

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def call(script: Path, *args: str) -> None:
    command = [sys.executable, str(script), *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    base = Path(args.output_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    traces = []
    curves = []
    for model in ("smoke-olmoe", "smoke-llmjp"):
        trace = base / f"{model}_routes.csv"
        curve = base / f"{model}_curve.csv"
        call(here / "capture_native_routes.py", "--smoke", "--model-key", model, "--samples", "8", "--output", str(trace))
        call(
            here / "benchmark_expert_service_curve.py",
            "--smoke", "--model-key", model, "--rows", "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "--output", str(curve),
        )
        traces.append(trace)
        curves.append(curve)

    merged_curve = base / "service_curve.csv"
    merge_args = [value for curve in curves for value in ("--input", str(curve))]
    call(here / "merge_service_curves.py", *merge_args, "--output", str(merged_curve))

    gate1 = base / "gate1"
    trace_args = [value for trace in traces for value in ("--trace", str(trace))]
    call(
        here / "census_fragmentation.py",
        *trace_args,
        "--service-curve", str(merged_curve),
        "--replicas", "2",
        "--concurrency", "2", "4",
        "--bootstrap", "100",
        "--smoke",
        "--output-dir", str(gate1),
    )
    instances = base / "instances.jsonl"
    call(
        here / "build_fixed_replica_instances.py",
        *trace_args,
        "--gate1-summary", str(gate1 / "gate1_summary.json"),
        "--replicas", "2",
        "--tokens-per-instance", "1",
        "--smoke",
        "--output", str(instances),
    )
    gate2 = base / "gate2"
    call(
        here / "solve_assignment_oracle.py",
        "--instances", str(instances),
        "--service-curve", str(merged_curve),
        "--holds-us", "0", "10",
        "--max-exact-states", "200000",
        "--bootstrap", "100",
        "--smoke",
        "--output-dir", str(gate2),
    )
    gate3 = base / "gate3"
    call(
        here / "compare_policies.py",
        "--instances", str(instances),
        "--service-curve", str(merged_curve),
        "--gate2-summary", str(gate2 / "gate2_summary.json"),
        "--oracle-results", str(gate2 / "oracle_results.jsonl"),
        "--hold-candidates-us", "0", "10",
        "--smoke",
        "--output-dir", str(gate3),
    )
    call(
        here / "compute_captured_headroom.py",
        "--policy-results", str(gate3 / "policy_results.jsonl"),
        "--oracle-results", str(gate2 / "oracle_results.jsonl"),
        "--resolved-plan", str(gate3 / "resolved_plan.json"),
        "--bootstrap", "100",
        "--smoke",
        "--output-dir", str(gate3),
    )
    print(f"SMOKE_ONLY pipeline complete: {gate3 / 'decision.json'}")


if __name__ == "__main__":
    main()
