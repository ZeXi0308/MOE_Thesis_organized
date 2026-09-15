#!/usr/bin/env python3
"""One targeted raw-data check and action/exposure summary for paired ladders."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


policy = module("policy_diagnostics", ROOT.parent / "20260906_native_knee_r01/policy_probe/analyze_policy.py")
cost = module("step_diagnostics", ROOT.parent / "20260908_step_cost_surface_r01/analyze_step_cost.py")


def stats(values):
    ordered = sorted(values)
    return dict(n=len(ordered), min=ordered[0], p50=median(ordered), max=ordered[-1]) if ordered else None


def inspect(path, config):
    raw = read(path)
    causal = policy.causal_check(raw, config)
    aligned = cost.load_cell(path)
    if not aligned["aligned"]:
        raise ValueError(f"step/receipt mismatch: {path}")
    steps = raw["scheduler_steps"]
    first_scheduled = {}
    for step in steps:
        for row in step["scheduled"]:
            first_scheduled.setdefault(row["request_id"], step["start_s"])
    requests = raw["requests"]
    if len(first_scheduled) != len(requests):
        raise ValueError(f"request scheduling population mismatch: {path}")
    bindings = [a["first_binding_opportunity_s"] for a in causal["actions"]
                if a["first_binding_opportunity_s"] is not None]
    applied = [a for a in causal["actions"] if a["applied_change"]]
    return dict(path=str(path), plan=raw["plan"], checks="PASS", causal_and_action=causal,
        step_receipt_alignment=dict(aligned=True, steps=aligned["n_steps"], receipts=aligned["n_receipts"]),
        exposure=dict(actual_active_histogram=dict(sorted(Counter(s["actual_active"] for s in steps).items())),
            decode_width_histogram=dict(sorted(Counter(s["decode_requests"] for s in steps).items())),
            waiting_histogram=dict(sorted(Counter(s["waiting_requests"] for s in steps).items())),
            max_actual_active=max(s["actual_active"] for s in steps),
            max_decode_width=max(s["decode_requests"] for s in steps),
            max_waiting=max(s["waiting_requests"] for s in steps),
            waiting_step_fraction=sum(s["waiting_requests"] > 0 for s in steps) / len(steps),
            waiting_step_fraction_semantics="scheduler snapshots, not wall-time fraction",
            first_scheduling_delay_s=stats(first_scheduled[r["request_id"]] - r["arrival_s"] for r in requests),
            host_submission_delay_s=stats(r["admission_s"] - r["arrival_s"] for r in requests)),
        first_applied_action=applied[0] if applied else None,
        first_binding_opportunity_s=min(bindings) if bindings else None,
        observed_first_schedule_s=first_scheduled,
        step_execution_sum_s=sum(r["exec_ms"] for r in aligned["rows"]) / 1000,
        episode_duration_s=raw["observation_end_s"],
        timing_semantics="step execution intervals lie inside episode time; never add them again")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "gpu_results")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis")
    args = parser.parse_args()
    output = args.analysis_dir / "diagnostics.json"
    if output.exists():
        raise ValueError("diagnostics output must be new")
    analysis = read(args.analysis_dir / "analysis.json")
    if len(analysis["cells"]) != 32 or not all(c["eligible"] for c in analysis["cells"]):
        raise ValueError("final paired analysis requires all 32 eligible cells")
    cells, warmups = [], []
    for block in ("forward", "reverse"):
        folder = args.results_dir / block
        config = read(folder / "config.json")
        for index, plan in enumerate(config["plans"]):
            row = inspect(folder / f"cell-{index:03d}.json", config)
            if row["plan"] != plan:
                raise ValueError("raw plan differs from run config")
            cells.append(dict(row, engine=block, index=index))
        paths = sorted(folder.glob("warmup-*-raw.json"))
        if len(paths) != 15:
            raise ValueError("expected all 15 retained warmup raw files per engine")
        for path in paths:
            raw = read(path)
            if raw["status"] != "COMPLETE" or len(raw["requests"]) != 32:
                raise ValueError(f"incomplete warmup: {path}")
            warmups.append(dict(engine=block, file=path.name, status=raw["status"],
                completed=sum(r["status"] == "completed" for r in raw["requests"]),
                steps=len(raw["scheduler_steps"]), output_events=len(raw["output_events"])))
    comparisons = []
    for pair in analysis["comparisons"]:
        old, new = (pair["arms"][f"feedback:32:{name}"] for name in ("legacy", "aligned"))
        selected = [c for c in cells if c["engine"] == pair["engine"]
                    and c["plan"]["regime"] == pair["regime"] and c["plan"]["policy"] == "feedback"]
        by_name = {c["plan"]["ladder_name"]: c for c in selected}
        old_d, new_d = by_name["legacy"], by_name["aligned"]
        comparisons.append(dict(engine=pair["engine"], regime=pair["regime"],
            legacy=old, aligned=new, best_static_caps=pair["best_static_caps"],
            best_static_goodput_rps=pair["best_static_goodput_rps"],
            aligned_relative_legacy_goodput=pair["aligned_relative_legacy_goodput"],
            aligned_relative_best_static_goodput=new["goodput_rps"] / pair["best_static_goodput_rps"] - 1,
            observed_width_histograms_differ=old_d["exposure"]["decode_width_histogram"] != new_d["exposure"]["decode_width_histogram"],
            observed_first_scheduling_delta_s=stats(new_d["observed_first_schedule_s"][rid] - value
                for rid, value in old_d["observed_first_schedule_s"].items()),
            comparison_semantics="separate full policy reruns; timing differences do not isolate a per-action causal effect"))
    repeat_signals = []
    for regime in ("steady", "bursty"):
        rows = [c for c in comparisons if c["regime"] == regime]
        keys = ("aligned_relative_legacy_goodput", "aligned_relative_best_static_goodput")
        flips = [key for key in keys if rows[0][key] * rows[1][key] < 0]
        small = [key for key in keys if any(abs(row[key]) < 0.03 for row in rows)]
        repeat_signals.append(dict(regime=regime, sign_flips=flips, effects_below_three_percent=small,
            frozen_repeat_condition_met=bool(flips or small),
            both_repeats_clear_gain_threshold=all(row[key] >= 0.03 for row in rows for key in keys),
            latency_threshold_noise_assessment="continuous request latency retained in comparisons; qualitative interpretation required"))
    result = dict(status="TARGETED_RAW_CHECKS_PASS", cells_checked=len(cells),
        request_executions=sum(c["metrics"]["n_completed"] for c in analysis["cells"]),
        total_existing_decode_advances_checked=sum(c["causal_and_action"]["existing_decode_advances_checked"] for c in cells),
        warmups=warmups, cells=cells, comparisons=comparisons, frozen_repeat_signals=repeat_signals,
        limits=["All history, existing decode and step/receipt checks reuse the existing analyzers once.",
            "Binding opportunity is exposure, not proof of saved request time.",
            "No padding, kernel, expert-specific, dynamic Oracle or quality inference.",
            "Threshold-distance/sign diagnostics retain both orders and cannot select a favorable run."])
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], cells=len(cells), warmup_raw=len(warmups),
        comparisons=[{k: v for k, v in c.items() if k in ("engine", "regime",
            "aligned_relative_legacy_goodput", "aligned_relative_best_static_goodput")} for c in comparisons],
        frozen_repeat_signals=repeat_signals)))


if __name__ == "__main__":
    main()
