#!/usr/bin/env python3
"""Recompute every retained native capacity episode; no controller or SLO tuning."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "refine-logs/expert_saturation/experiments/admission_capacity"))
from metrics import summarize_episode_requests

LEGACY = Path(__file__).resolve().parents[1] / "20260906_native_transfer_r01/analyze_native.py"
spec = importlib.util.spec_from_file_location("native_episode_summary", LEGACY)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

KEYS = ("throughput_rps", "goodput_rps", "attainment", "ttft_p50_s", "tpot_p50_s",
        "max_active", "max_decode_requests", "max_native_waiting_after")
GRID = tuple(itertools.product((0.2, 0.5, 1.0), (0.009, 0.012, 0.016)))
FAIR_CONFIG = ("model", "requests", "output_tokens", "prompt_tokens", "workload_sha256",
               "ttft_slo_s", "tpot_slo_s", "reference_slo", "seed", "engine_max_num_seqs",
               "enforce_eager", "queue_policy", "warmup_condition_rounds")


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def inspect_cell(raw, config, workload, plan, saved):
    require(raw["plan"] == plan, "raw/config plan mismatch")
    cap = plan["cap"]
    require(raw["target_cap"] == cap, "raw target cap does not match plan")
    require(raw["admission_cap_state"]["admission_cap"] == cap, "actual scheduler cap mismatch")
    require(raw["admission_cap_state"]["engine_max_num_seqs"] == config["engine_max_num_seqs"],
            "actual engine max differs from config")
    sources = workload["source_requests"]
    require(len(raw["requests"]) == config["requests"] == len(sources), "request cohort size mismatch")
    indexed = {r["request_id"]: r for r in raw["requests"]}
    require(len(indexed) == len(sources), "duplicate request identity")
    arrivals = workload["arrival_traces_s"][plan["regime"]]
    for source, arrival in zip(sources, arrivals):
        row = indexed[source["request_id"]]
        require(row["document_id"] == source["document_id"] and
                row["prompt_token_ids_sha256"] == source["prompt_token_ids_sha256"], "input identity mismatch")
        require(same(row["arrival_s"], arrival * plan["arrival_scale"]), "arrival alignment mismatch")
        if row["status"] == "completed":
            require(len(row["output_token_ids"]) == config["output_tokens"], "fixed output length mismatch")
    result = legacy.summarize_cell(raw, config)
    metrics = result["metrics"]
    result["saved_metrics_equal"] = saved is not None and same(metrics, {k: saved.get(k) for k in metrics})
    require(saved is None or result["saved_metrics_equal"], "saved main metrics differ from raw recomputation")
    reference = summarize_episode_requests(raw["requests"], observation_end_s=raw["observation_end_s"],
                                          **config["reference_slo"])
    result["reference_slo_metrics"] = reference
    result["saved_reference_equal"] = saved is not None and same(reference, saved.get("reference_slo_metrics"))
    require(saved is None or result["saved_reference_equal"], "saved reference metrics differ from raw recomputation")
    result["slo_grid_descriptive_only"] = []
    for ttft, tpot in GRID:
        m = summarize_episode_requests(raw["requests"], observation_end_s=raw["observation_end_s"],
                                       ttft_slo_s=ttft, tpot_slo_s=tpot)
        result["slo_grid_descriptive_only"].append(dict(ttft_slo_s=ttft, tpot_slo_s=tpot,
            n_slo_pass=m["n_slo_pass"], attainment=m["slo_attainment"], goodput_rps=m["goodput_rps"]))
    steps = raw["scheduler_steps"]
    require(all(s["step"] == i and s["target_cap"] == cap for i, s in enumerate(steps)),
            "scheduler step/target alignment mismatch")
    require(all(0 <= s["actual_active"] <= cap for s in steps), "actual active exceeds static cap")
    require(result["preemption_events"] == 0, "preemption violates experiment action scope")
    result.update(n_slo_pass=metrics["n_slo_pass"],
        cap_reached=result["max_active"] == cap,
        decode_cap_reached=result["max_decode_requests"] == cap,
        decode_width_step_counts=dict(sorted(Counter(s["decode_requests"] for s in steps).items())),
        active_step_counts=dict(sorted(Counter(s["actual_active"] for s in steps).items())),
        waiting_after_positive_steps=sum(s["waiting_requests"] > 0 for s in steps),
        mixed_prefill_decode_steps=sum(any(r["prefill_tokens"] > 0 for r in s["scheduled"]) and
                                      any(r["decode_tokens"] > 0 for r in s["scheduled"]) for s in steps),
        violation_counts=dict(ttft_only=sum(not r["ttft_pass"] and r["tpot_pass"] for r in metrics["per_request"]),
            tpot_only=sum(r["ttft_pass"] and not r["tpot_pass"] for r in metrics["per_request"]),
            both=sum(not r["ttft_pass"] and not r["tpot_pass"] for r in metrics["per_request"])),
        output_trajectory_sha256=hashlib.sha256(json.dumps(
            [(r["request_id"], r["output_token_ids"]) for r in raw["requests"]], sort_keys=True).encode()).hexdigest())
    return result


def delta(low, high):
    result = {}
    for key in KEYS:
        a, b = low[key], high[key]
        result[key] = dict(low=a, high=b, high_minus_low=None if a is None or b is None else b - a,
                          relative_to_low=None if a in (None, 0) or b is None else (b - a) / a)
    return result


def analyze(root, campaign):
    cells, configs, engines, issues = [], {}, {}, []
    for trial in campaign["trials"]:
        group, block = trial["group"], trial["block"]
        directory = root / group
        if not (directory / "config.json").exists():
            issues.append(f"UNRUN trial {group}: missing config")
            continue
        config = configs[group] = read(directory / "config.json")
        workload = read(directory / "workload.json")
        workload_ok = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() == config["workload_sha256"]
        engines[group] = read(directory / "engine_args.json") if (directory / "engine_args.json").exists() else None
        for index, plan in enumerate(config["plans"]):
            cell = dict(id=f"{group}/cell-{index:03d}", group=group, block=block, cell=index, plan=plan,
                        cohort=config["workload_sha256"], status="MISSING")
            cells.append(cell)
            path = directory / f"cell-{index:03d}.json"
            if not path.exists():
                continue
            try:
                require(workload_ok, "workload hash changed")
                raw = read(path)
                saved_path = directory / f"metrics-{index:03d}.json"
                saved = read(saved_path) if saved_path.exists() else None
                cell.update(inspect_cell(raw, config, workload, plan, saved), status=raw["status"], error=raw.get("error"))
                if cell["metrics"]["n_completed"] != config["requests"]:
                    cell["status"] = "INCOMPLETE"
                check = directory / f"checks-{index:03d}.json"
                cell["gpu_boundary_check"] = read(check).get("status", "UNKNOWN") if check.exists() else "MISSING"
                cell["qualified"] = (cell["status"] == "COMPLETE" and cell["gpu_boundary_check"] == "PASS" and
                    cell["token_level_timing_resolved"] and cell["saved_metrics_equal"] and cell["saved_reference_equal"])
            except (KeyError, TypeError, ValueError) as exc:
                cell.update(status="INVALID", qualified=False, error=str(exc))
    conditions, paired = [], []
    for cohort, scale, regime in sorted({(c["cohort"], c["plan"]["arrival_scale"], c["plan"]["regime"]) for c in cells}):
        all_rows = [c for c in cells if (c["cohort"], c["plan"]["arrival_scale"], c["plan"]["regime"]) == (cohort, scale, regime)]
        rows = [c for c in all_rows if c.get("qualified")]
        caps = sorted({c["plan"]["cap"] for c in all_rows})
        ranges = {str(cap): {k: legacy.interval([c[k] for c in rows if c["plan"]["cap"] == cap])
                  for k in (*KEYS, "n_slo_pass")} for cap in caps}
        interpretation = "UNRESOLVED_OR_PARTIAL"
        if rows and len(rows) == len(all_rows):
            interpretation = ("ALL_PASS" if all(c["attainment"] == 1 for c in rows) else
                              "ALL_FAIL" if all(c["attainment"] == 0 for c in rows) else "SOME_SLO_SEPARATION")
        conditions.append(dict(cohort=cohort, arrival_scale=scale, regime=regime, expected_cells=len(all_rows),
            qualified_cells=len(rows), cap_ranges=ranges, slo_interpretation=interpretation))
        for block, repeat in sorted({(c["block"], c["plan"]["repeat"]) for c in all_rows}):
            selected = [c for c in all_rows if (c["block"], c["plan"]["repeat"]) == (block, repeat)]
            for low_cap, high_cap in itertools.combinations(caps, 2):
                a = [c for c in selected if c["plan"]["cap"] == low_cap]
                b = [c for c in selected if c["plan"]["cap"] == high_cap]
                pair = dict(block=block, repeat=repeat, cohort=cohort, arrival_scale=scale, regime=regime,
                            low_cap=low_cap, high_cap=high_cap, status="MISSING_OR_UNQUALIFIED")
                if len(a) == len(b) == 1:
                    low, high = a[0], b[0]
                    ga, gb = low["group"], high["group"]
                    compatible = all(configs[ga].get(k) == configs[gb].get(k) for k in FAIR_CONFIG)
                    compatible = compatible and engines[ga] is not None and engines[ga] == engines[gb]
                    pair.update(low_id=low["id"], high_id=high["id"], fair_config_equal=compatible)
                    if compatible and low.get("qualified") and high.get("qualified"):
                        pair.update(status="DESCRIPTIVE_PAIRED_RERUN", differences=delta(low, high),
                            output_trajectory_equal=low["output_trajectory_sha256"] == high["output_trajectory_sha256"])
                elif len(a) > 1 or len(b) > 1:
                    pair["status"] = "AMBIGUOUS_DUPLICATE_PAIR_KEY"
                paired.append(pair)
    expected = campaign.get("expected_episodes", len(cells))
    complete = (bool(cells) and len(cells) == expected and len(configs) == len(campaign["trials"]) and
                all(c.get("qualified") for c in cells))
    return dict(status="MEASUREMENT_ONLY" if complete else "PARTIAL_OR_UNRUN", scientific_go=False,
        complete_expected_capture=complete, expected_episodes=expected, discovered_planned_episodes=len(cells),
        completed_episodes=sum(c["status"] == "COMPLETE" for c in cells),
        qualified_episodes=sum(bool(c.get("qualified")) for c in cells),
        request_executions=sum(c.get("metrics", {}).get("n_completed", 0) for c in cells),
        engine_args_equal=bool(engines) and all(e is not None and e == next(iter(engines.values())) for e in engines.values()),
        cells=cells, conditions=conditions, paired_cap_comparisons=paired, issues=issues,
        limits=["Finite episodes and two counterbalanced trials are not a steady-state capacity estimate or independent workloads.",
            "Comparisons pair the same input, arrival, trial block and within-trial repeat; every cap executes its own future state.",
            "The nine-cell SLO grid is descriptive and cannot select the canonical policy or replace the frozen main SLO.",
            "Step histograms describe exposure; adjacent decode steps are not independent statistical samples.",
            "queue_s is host submission lag. Native queue exposure is separately measured after scheduler admission.",
            "Host token receipt includes capture cost. Core timestamps are used only within the core clock domain.",
            "No dynamic policy, expert signal, Oracle, hidden-state equality or semantic quality is tested.",
            "GPU isolation is checked at episode boundaries, not continuously; small-sample tail quantiles are descriptive."])


def report(result):
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"

    lines = [f"# Native capacity scan: {result['status']}", "",
        f"Completed {result['completed_episodes']}/{result['expected_episodes']} episodes; "
        f"qualified {result['qualified_episodes']}; completed request executions {result['request_executions']}.", "",
        "| Cell | Scale / regime / repeat | Cap | Status | Pass | Throughput | Goodput | TTFT / TPOT p50 s | Max active / decode / waiting after |", "|---|---|---|---|---|---|---|---|---|"]
    for cell in result["cells"]:
        p = cell["plan"]
        lead = f"| {cell['id']} | {p['arrival_scale']} / {p['regime']} / {p['repeat']} | {p['cap']} | {cell['status']}"
        if "metrics" not in cell:
            lines.append(lead + " | — | — | — | — | — |")
        else:
            lines.append(lead + f" | {cell['n_slo_pass']}/{cell['metrics']['n_planned']} | {fmt(cell['throughput_rps'])} | "
                f"{fmt(cell['goodput_rps'])} | {fmt(cell['ttft_p50_s'])} / {fmt(cell['tpot_p50_s'])} | "
                f"{cell['max_active']} / {cell['max_decode_requests']} / {cell['max_native_waiting_after']} |")
    lines += ["", "All cap comparisons (high relative to low); descriptive paired reruns:", "",
              "| Block / scale / regime | Low → high | Status | Throughput Δ% | Goodput Δ% | TPOT p50 Δ% |", "|---|---|---|---|---|---|"]
    for p in result["paired_cap_comparisons"]:
        changes = p.get("differences", {})
        values = [changes.get(k, {}).get("relative_to_low") for k in ("throughput_rps", "goodput_rps", "tpot_p50_s")]
        lines.append(f"| {p['block']} / {p['arrival_scale']} / {p['regime']} | {p['low_cap']} → {p['high_cap']} | {p['status']} | " +
                     " | ".join(fmt(None if v is None else 100 * v) for v in values) + " |")
    lines += ["", *result["limits"], "", *result["issues"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--campaign-plan", type=Path, default=Path(__file__).with_name("campaign_plan.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.results_dir, read(args.campaign_plan))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(report(result))


if __name__ == "__main__":
    main()
