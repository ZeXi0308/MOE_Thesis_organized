#!/usr/bin/env python3
"""Recompute native ABBA episode measurements; no causal or mechanism verdict."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "refine-logs/expert_saturation/experiments/admission_capacity"))
from metrics import summarize_episode_requests

GROUPS = ("a0_cap6", "b0_cap8", "b1_cap8", "a1_cap6")


def read(path):
    return json.loads(path.read_text())


def interval(values):
    available = [v for v in values if v is not None]
    return dict(n=len(available), minimum=min(available) if available else None,
                maximum=max(available) if available else None)


def summarize_cell(raw, config):
    metrics = summarize_episode_requests(raw["requests"], observation_end_s=raw["observation_end_s"],
        ttft_slo_s=config["ttft_slo_s"], tpot_slo_s=config["tpot_slo_s"])
    steps, events = raw["scheduler_steps"], raw["output_events"]
    scheduled = [row for step in steps for row in step["scheduled"]]
    core_wait = []
    for row in raw["requests"]:
        native = row.get("native_metrics", {})
        queued, started = native.get("queued_ts"), native.get("scheduled_ts")
        if all(type(v) in (int, float) and math.isfinite(v) for v in (queued, started)) and started >= queued:
            core_wait.append(started - queued)
    lag = [r["queue_s"] for r in metrics["per_request"] if r["queue_s"] is not None]
    chunks = [e["chunk_size"] for e in events]
    resolved = bool(chunks) and all(n <= 1 for n in chunks)
    return dict(metrics=metrics, goodput_rps=metrics["goodput_rps"], throughput_rps=metrics["throughput_rps"],
        attainment=metrics["slo_attainment"], ttft_p50_s=metrics["latency_s"]["ttft"]["p50"],
        tpot_p50_s=metrics["latency_s"]["tpot"]["p50"], token_level_timing_resolved=resolved,
        scheduler_steps=len(steps), max_active=max((s["actual_active"] for s in steps), default=0),
        max_decode_requests=max((s["decode_requests"] for s in steps), default=0),
        max_native_waiting=max((max(s["waiting_before"], s["waiting_requests"]) for s in steps), default=0),
        max_native_waiting_after=max((s["waiting_requests"] for s in steps), default=0),
        scheduled_prefill_tokens=sum(r["prefill_tokens"] for r in scheduled),
        scheduled_decode_tokens=sum(r["decode_tokens"] for r in scheduled),
        scheduled_total_tokens=sum(s["total_scheduled_tokens"] for s in steps),
        preemption_events=sum(len(s["preempted_request_ids"]) for s in steps),
        unique_preempted_requests=len({rid for s in steps for rid in s["preempted_request_ids"]}),
        computed_adjustment_events=sum(r["computed_adjustment"] != 0 for r in scheduled),
        computed_adjustment_abs_tokens=sum(abs(r["computed_adjustment"]) for r in scheduled),
        multi_token_chunks=sum(n > 1 for n in chunks), max_chunk_size=max(chunks, default=0),
        client_submission_lag_s=dict(p50=statistics.median(lag) if lag else None, **interval(lag)),
        native_core_wait_s=dict(p50=statistics.median(core_wait) if core_wait else None, **interval(core_wait)))


def analyze(root):
    cells, issues, configs = [], [], {}
    for group in GROUPS:
        directory = root / group
        if not (directory / "config.json").exists():
            issues.append(f"missing group config: {group}")
            continue
        config = read(directory / "config.json")
        configs[group] = config
        for index, plan in enumerate(config["plans"]):
            cell = dict(group=group, cell=index, plan=plan, status="MISSING")
            cells.append(cell)
            path = directory / f"cell-{index:03d}.json"
            if not path.exists():
                continue
            try:
                raw = read(path)
                if raw["plan"] != plan:
                    raise ValueError("raw/config plan mismatch")
                cell.update(summarize_cell(raw, config), status=raw["status"], error=raw.get("error"))
                if cell["metrics"]["n_completed"] != config["requests"]:
                    cell["status"] = "INCOMPLETE"
                check = directory / f"checks-{index:03d}.json"
                cell["gpu_boundary_check"] = read(check).get("status", "UNKNOWN") if check.exists() else "MISSING"
            except (KeyError, TypeError, ValueError) as exc:
                cell.update(status="INVALID", error=str(exc))
    valid = [c for c in cells if c["status"] == "COMPLETE" and "metrics" in c]
    conditions = []
    keys = ("goodput_rps", "attainment", "ttft_p50_s", "tpot_p50_s", "max_active", "max_decode_requests", "max_native_waiting", "max_native_waiting_after")
    for scale, regime in sorted({(c["plan"]["arrival_scale"], c["plan"]["regime"]) for c in cells}):
        rows = [c for c in valid if (c["plan"]["arrival_scale"], c["plan"]["regime"]) == (scale, regime)]
        ranges = {str(cap): {k: interval([c[k] for c in rows if c["plan"]["cap"] == cap]) for k in keys} for cap in (6, 8)}
        pairs = []
        for left, right in (("a0_cap6", "b0_cap8"), ("a1_cap6", "b1_cap8")):
            for repeat in (0, 1):
                a = next((c for c in rows if c["group"] == left and c["plan"]["repeat"] == repeat), None)
                b = next((c for c in rows if c["group"] == right and c["plan"]["repeat"] == repeat), None)
                paired = dict(cap6_group=left, cap8_group=right, repeat=repeat, status="MISSING_OR_INCOMPLETE")
                if a and b:
                    compatible = all(configs[left].get(k) == configs[right].get(k) for k in
                        ("workload_sha256", "model", "output_tokens", "prompt_tokens", "ttft_slo_s", "tpot_slo_s", "enforce_eager", "engine_max_num_seqs"))
                    eligible = compatible and all(c["token_level_timing_resolved"] and
                        c["gpu_boundary_check"] == "PASS" for c in (a, b))
                    paired.update(status="DESCRIPTIVE_PAIRED_RERUN" if eligible else "TIMING_OR_CONFIG_NOT_QUALIFIED",
                        cap6_minus_cap8={k: a[k] - b[k] if a[k] is not None and b[k] is not None else None for k in keys[:4]})
                    if eligible and a["goodput_rps"] is not None and b["goodput_rps"] is not None:
                        paired["observed_goodput_winner"] = 6 if a["goodput_rps"] > b["goodput_rps"] else 8 if b["goodput_rps"] > a["goodput_rps"] else "TIE"
                pairs.append(paired)
        interpretation = "UNRESOLVED_OR_PARTIAL"
        if len(rows) == 8 and all(c["token_level_timing_resolved"] for c in rows):
            interpretation = ("ALL_PASS: goodput equals completion throughput; these SLOs do not separate outcomes"
                if all(c["attainment"] == 1 for c in rows) else "ALL_FAIL: zero passing-request separation"
                if all(c["attainment"] == 0 for c in rows) else "Some violations observed; this alone does not establish a calibrated capacity boundary")
        conditions.append(dict(arrival_scale=scale, regime=regime, complete_cells=len(rows),
            cap_ranges=ranges, abba_same_repeat_pairs=pairs, slo_interpretation=interpretation))
    expected = sum(len(config["plans"]) for config in configs.values())
    complete = len(configs) == len(GROUPS) and expected > 0 and len(valid) == expected and not issues
    return dict(status="MEASUREMENT_ONLY" if complete else "PARTIAL_OR_UNRUN",
        expected_episodes=expected, complete_expected_episode_capture=complete,
        complete_32_episode_capture=complete and expected == 32,
        scientific_go=False, cells=cells, conditions=conditions, issues=issues,
        limits=["Four observed episodes per cap/condition reuse one workload; they are not four independent workloads.",
            "Small-sample TTFT/TPOT/ITL tail quantiles are descriptive. ABBA rankings do not identify a mechanism.",
            "queue_s is client submission lag; native waiting is scheduler queue count and core scheduled_ts minus queued_ts.",
            "max_native_waiting_after counts requests still waiting after scheduling; the legacy maximum also includes newly submitted requests before scheduling.",
            "Core and host clocks are never subtracted; multi-token host chunks do not resolve generation ITL.",
            "GPU isolation checks cover episode boundaries; this report does not certify continuous isolation."],
        one_next_question="At a separately frozen native load/SLO setting, does cap 6 versus 8 change violation risk reproducibly while actual scheduling reaches both limits?")


def report(result):
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"

    lines = [f"# Native cap transfer: {result['status']}", "", *result["limits"], "",
        "| Group/cell | Scale/regime/repeat | Cap | Status | Goodput | TTFT p50 s | TPOT p50 s | Max active/decode/wait after schedule | Prefill/decode tokens | Preempt/adjust/chunks>1 |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for c in result["cells"]:
        p = c["plan"]
        ids = f"{c['group']}/{c['cell']} | {p['arrival_scale']}/{p['regime']}/{p['repeat']} | {p['cap']} | {c['status']}"
        if "metrics" not in c:
            lines.append(f"| {ids} | — | — | — | — | — | — |")
            continue
        lines.append(f"| {ids} | {fmt(c['goodput_rps'])} | {fmt(c['ttft_p50_s'])} | {fmt(c['tpot_p50_s'])} | "
            f"{c['max_active']}/{c['max_decode_requests']}/{c['max_native_waiting_after']} | {c['scheduled_prefill_tokens']}/{c['scheduled_decode_tokens']} | "
            f"{c['preemption_events']}/{c['computed_adjustment_events']}/{c['multi_token_chunks']} |")
    for condition in result["conditions"]:
        lines.extend(["", f"Scale {condition['arrival_scale']}, {condition['regime']}: {condition['slo_interpretation']}.",
            f"Per-cap four-episode ranges: {json.dumps(condition['cap_ranges'])}", ""])
        lines.extend(json.dumps(pair) for pair in condition["abba_same_repeat_pairs"])
    lines.extend(["", *result["issues"], "", "One next question: " + result["one_next_question"]])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(report(result))


if __name__ == "__main__":
    main()
