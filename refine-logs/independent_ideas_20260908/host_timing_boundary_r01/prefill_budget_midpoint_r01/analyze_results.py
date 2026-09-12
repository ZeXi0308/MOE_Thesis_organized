#!/usr/bin/env python3
"""Describe retained GPU raw only; compare each engine separately, no Oracle."""
import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))
from metrics import _distribution, summarize_episode_requests

SLO = dict(ttft_slo_s=5.0, tpot_slo_s=0.2)


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    return dict(_distribution(values), mean=statistics.mean(values) if values else None, max=max(values) if values else None)


def analyze_cell(path):
    raw = read(path)
    rows, steps, budget = raw["requests"], raw["scheduler_steps"], raw["plan"]["budget"]
    issues, groups = [], {}
    expected_lengths = [128 if raw["plan"]["cohort"] == "all_short" or i % 2 == 0 else 2048 for i in range(16)]
    if [r["prompt_tokens"] for r in rows] != expected_lengths or any(r["status"] == "completed" and len(r["output_token_ids"]) != 128 for r in rows):
        issues.append("frozen_request_shape_mismatch")
    for name, selected in (("all", rows), ("short", [r for r in rows if r["prompt_tokens"] == 128]),
                           ("long", [r for r in rows if r["prompt_tokens"] == 2048])):
        summary = summarize_episode_requests(selected, observation_end_s=raw["observation_end_s"], **SLO)
        for metric, key in (("ttft", "ttft_s"), ("tpot", "tpot_s"), ("itl", "itl_s")):
            values = [v for r in summary["per_request"] for v in (r[key] if metric == "itl" else [r[key]]) if v is not None]
            summary["latency_s"][metric] = distribution(values)
        summary["whole_episode_completion_rps"] = summary["n_completed"] / raw["observation_end_s"] if raw["observation_end_s"] > 0 else None
        groups[name] = summary
    derived = {r["request_id"]: r for r in groups["all"]["per_request"]}
    requests = [dict(derived.get(r["request_id"], {}), request_id=r["request_id"], status=r["status"],
                    prompt_tokens=r["prompt_tokens"], document_id=r["document_id"], arrival_s=r["arrival_s"],
                    prompt_token_ids_sha256=r["prompt_token_ids_sha256"],
                    arrived_by_end=r["arrival_s"] <= raw["observation_end_s"]) for r in rows]
    returns = raw["engine_step_returns"]
    receipts = {}
    for ret in returns:
        index = ret["scheduler_begin"]
        if ret["scheduler_end"] != index + 1 or index in receipts or index >= len(steps):
            issues.append("invalid_or_duplicate_step_receipt")
            continue
        receipts[index] = ret["received_s"]
        if not ret["call_start_s"] <= steps[index]["start_s"] <= steps[index]["end_s"] <= ret["received_s"]:
            issues.append("step_receipt_clock_order")
    missing_receipts = sorted(set(range(len(steps))) - receipts.keys())
    if raw["status"] == "COMPLETE" and missing_receipts:
        issues.append("complete_cell_missing_receipt")
    exposure = {r["request_id"]: [] for r in rows}
    prefills, totals, mixed, preemptions, skipped, adjustments = [], [], [], [], [], []
    prefill_by_request = Counter()
    for index, step in enumerate(steps):
        scheduled = step["scheduled"]
        prefill = sum(r["prefill_tokens"] for r in scheduled)
        total = sum(r["scheduled_tokens"] for r in scheduled)
        decoded = {r["request_id"] for r in scheduled if r["decode_tokens"] > 0}
        missing = sorted(set(step["existing_decode_request_ids"]) - decoded)
        if step["step"] != index or step["runtime_token_budget"] != budget or total != step["total_scheduled_tokens"] or total > budget:
            issues.append("step_budget_or_identity_mismatch")
        if step["target_cap"] != 8 or step["actual_active"] > 8:
            issues.append("fixed_cap8_mismatch")
        if step["existing_decode_all_scheduled"] != (not missing) or sorted(step["missing_decode_request_ids"]) != missing:
            issues.append("decode_guard_record_mismatch")
        skipped.extend(dict(step=index, request_id=rid) for rid in missing)
        preemptions.extend(dict(step=index, request_id=rid) for rid in step["preempted_request_ids"])
        for row in scheduled:
            prefill_by_request[row["request_id"]] += row["prefill_tokens"]
            if row["scheduled_tokens"] != row["prefill_tokens"] + row["decode_tokens"]:
                issues.append("scheduled_token_partition_mismatch")
            if row["computed_adjustment"]:
                adjustments.append(dict(step=index, request_id=row["request_id"], value=row["computed_adjustment"]))
        prefills.append(prefill)
        totals.append(total)
        if prefill and decoded:
            mixed.append(index)
            for rid in decoded:
                exposure[rid].append(dict(step=index, start_s=step["start_s"], received_s=receipts.get(index)))
    if skipped or preemptions or adjustments:
        issues.append("decode_skip_preemption_or_kv_adjustment")
    if raw["status"] == "COMPLETE" and any(prefill_by_request[r["request_id"]] != r["prompt_tokens"] for r in rows):
        issues.append("complete_cell_prefill_token_conservation")
    for request in requests:
        events = exposure[request["request_id"]]
        request.update(mixed_steps_while_scheduled_decode=events, n_mixed_steps=len(events),
                       mixed_step_start_intervals_s=[b["start_s"] - a["start_s"] for a, b in zip(events, events[1:])])
    if raw["budget_state"] != dict(previous_budget=raw["budget_state"]["previous_budget"], runtime_budget=budget, compiled_token_capacity=1024, cap=8):
        issues.append("budget_state_mismatch")
    eligible = raw["status"] == "COMPLETE" and groups["all"]["n_completed"] == 16 and not issues
    return dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), plan=raw["plan"],
        block=path.parent.name, phase=raw["phase"], status=raw["status"], error=raw["error"], eligible_for_comparison=eligible,
        episode_wall_s=raw["observation_end_s"], groups=groups, requests=requests, issues=sorted(set(issues)),
        diagnostics=dict(steps=len(steps), returns=len(returns), empty_output_returns=sum(r["output_count"] == 0 for r in returns),
            missing_receipt_steps=missing_receipts, total_scheduled_tokens=distribution(totals), prefill_tokens_per_step=distribution(prefills),
            total_prefill_tokens=sum(prefills), ceiling_hit_steps=sum(t == budget for t in totals),
            above_512_steps=sum(t > 512 for t in totals), mixed_steps=len(mixed), mixed_step_indices=mixed,
            existing_decode_checks=sum(len(s["existing_decode_request_ids"]) for s in steps),
            skipped=skipped, preemptions=preemptions, kv_adjustments=adjustments,
            all_short_control=("NO_OBSERVED_BINDING" if eligible and totals and max(totals) < 512 else "NOT_ESTABLISHED")
                if raw["plan"]["cohort"] == "all_short" else "NOT_APPLICABLE"))


def compare(base, alternative):
    identity = lambda c: [(r["request_id"], r["document_id"], r["prompt_token_ids_sha256"], r["arrival_s"]) for r in c["requests"]]
    valid = base["eligible_for_comparison"] and alternative["eligible_for_comparison"] and identity(base) == identity(alternative)
    delta = lambda a, b: dict(baseline1024=a, alternative512=b, absolute_delta=b - a,
                             relative_delta_pct=100 * (b / a - 1) if a else None) if a is not None and b is not None else None
    comparison = dict(block=base["block"], cohort=base["plan"]["cohort"], valid=valid,
        action_exposure_established=valid and base["diagnostics"]["above_512_steps"] > 0,
        identity_match=identity(base) == identity(alternative), metrics={})
    if valid:
        comparison["episode_wall_s"] = delta(base["episode_wall_s"], alternative["episode_wall_s"])
        for group in ("all", "short", "long"):
            comparison["metrics"][group] = {f"{metric}_{stat}_s": delta(base["groups"][group]["latency_s"][metric][stat],
                alternative["groups"][group]["latency_s"][metric][stat]) for metric in ("ttft", "tpot", "itl") for stat in ("mean", "p50", "p95", "p99")}
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cells, warmups, blocks, issues = [], [], {}, []
    for block in ("forward", "reverse"):
        folder = args.results_dir / block
        if not folder.exists():
            blocks[block] = {"status": "NOT_AVAILABLE"}
            continue
        config = read(folder / "config.json")
        if config["reference_slo"] != SLO:
            raise ValueError("reference SLO differs from frozen values")
        blocks[block] = read(folder / "status.json")
        for path in sorted(folder.glob("*-raw.json")):
            result = analyze_cell(path)
            index = int(path.name.split("-")[1])
            if result["plan"] != config["plans" if result["phase"] == "cell" else "warmups"][index]:
                result["issues"].append("frozen_plan_mismatch")
                result["eligible_for_comparison"] = False
            (cells if result["phase"] == "cell" else warmups).append(result)
    if not cells and not warmups:
        raise ValueError("no actual raw episodes available")
    comparisons = []
    for block in ("forward", "reverse"):
        for cohort in ("mixed",):
            selected = [c for c in cells if c["block"] == block and c["plan"]["cohort"] == cohort]
            by_budget = {c["plan"]["budget"]: c for c in selected}
            if len(by_budget) != len(selected):
                issues.append(f"duplicate measured condition: {block}/{cohort}")
            elif set(by_budget) == {512, 1024}:
                comparisons.append(compare(by_budget[1024], by_budget[512]))
    summary = dict(status="MEASUREMENT_ONLY" if len(cells) == 4 and all(c["eligible_for_comparison"] for c in cells) else "PARTIAL_OR_INVALID_MEASUREMENT", blocks=blocks, cells=cells, warmups=warmups, comparisons=comparisons,
        issues=issues, reference_slo=SLO, reference_slo_only=True,
        semantics="Host receipt times; pooled ITL quantiles descriptive, not independent step samples. All observed latency retained including failures; comparisons require both complete cells. Group legacy rates start at that group's first arrival; whole_episode_completion_rps uses the shared episode wall. No Oracle, no fitted counterfactual, no between-engine averaging.")
    lines = ["# Prefill budget: retained request measurements", "", summary["semantics"], "", "All times below are ms; wall is seconds. Warmups excluded.", "",
             "| Block | Cohort | Budget | Status | Group | Complete/planned | Wall s | TTFT mean | TPOT mean | ITL p50 / p95 / p99 |", "|---|---|---:|---|---|---:|---:|---:|---:|---|"]
    fmt = lambda v, scale=1000: "NA" if v is None else f"{v * scale:.4f}"
    for c in cells:
        for group, g in c["groups"].items():
            lat = g["latency_s"]
            lines.append(f"| {c['block']} | {c['plan']['cohort']} | {c['plan']['budget']} | {c['status']} | {group} | {g['n_completed']}/{g['n_planned']} | {fmt(c['episode_wall_s'], 1)} | {fmt(lat['ttft']['mean'])} | {fmt(lat['tpot']['mean'])} | " + " / ".join(fmt(lat['itl'][q]) for q in ('p50', 'p95', 'p99')) + " |")
    lines += ["", "| Block / cohort / budget | Steps | Token max | Ceiling hits | Prefill max | Mixed steps | Empty returns | Skips / preemptions / KV adjustments | Short negative control |", "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for c in cells:
        d = c["diagnostics"]
        lines.append(f"| {c['block']} / {c['plan']['cohort']} / {c['plan']['budget']} | {d['steps']} | {d['total_scheduled_tokens']['max']} | {d['ceiling_hit_steps']} | {d['prefill_tokens_per_step']['max']} | {d['mixed_steps']} | {d['empty_output_returns']} | {len(d['skipped'])} / {len(d['preemptions'])} / {len(d['kv_adjustments'])} | {d['all_short_control']} |")
    lines += ["", "Budget comparisons: delta = 512 minus 1024; negative latency delta is lower. Each engine reported separately.", "",
              "| Block / cohort | Valid | Actual >512 exposure at 1024 | Group / metric | 1024 ms | 512 ms | Delta ms | Delta % |", "|---|---|---|---|---:|---:|---:|---:|"]
    for comp in comparisons:
        if not comp["valid"]:
            lines.append(f"| {comp['block']} / {comp['cohort']} | False | False | Incomplete or invalid pair: no benefit comparison | NA | NA | NA | NA |")
            continue
        d = comp["episode_wall_s"]
        lines.append(f"| {comp['block']} / {comp['cohort']} | True | {comp['action_exposure_established']} | all / episode wall | {fmt(d['baseline1024'])} | {fmt(d['alternative512'])} | {fmt(d['absolute_delta'])} | {fmt(d['relative_delta_pct'], 1)} |")
        for group, metrics in comp["metrics"].items():
            for metric in ("ttft_mean_s", "tpot_mean_s", "itl_p95_s", "itl_p99_s"):
                d = metrics[metric]
                if d:
                    lines.append(f"| {comp['block']} / {comp['cohort']} | {comp['valid']} | {comp['action_exposure_established']} | {group} / {metric} | {fmt(d['baseline1024'])} | {fmt(d['alternative512'])} | {fmt(d['absolute_delta'])} | {fmt(d['relative_delta_pct'], 1)} |")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "TABLE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(measured=len(cells), warmups=len(warmups), comparisons=len(comparisons), issues=issues)))


if __name__ == "__main__":
    main()
