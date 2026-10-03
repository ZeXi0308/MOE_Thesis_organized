#!/usr/bin/env python3
"""Describe retained waiting-order raw; failed cells stay visible and excluded."""
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
MODES = ("fcfs", "short_prompt_first", "bounded_bypass_once")
read = lambda p: json.loads(p.read_text())
digest = lambda x: hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()
identity = lambda r: (r["request_id"], r["document_id"], r["prompt_token_ids_sha256"], r["prompt_tokens"], r["arrival_s"])
distribution = lambda x: dict(_distribution(x), mean=statistics.mean(x) if x else None, max=max(x) if x else None)


def analyze_cell(path):
    raw = read(path)
    rows, steps, plan = raw["requests"], raw["scheduler_steps"], raw["plan"]
    issues, groups, diagnostics = [], {}, Counter()
    check = lambda condition, issue: issues.append(issue) if not condition else None
    expected = read(ROOT / "prepared" / plan["cohort"] / "workload.json")
    expected_ids = [(r["request_id"], r["document_id"], r["prompt_token_ids_sha256"], r["prompt_token_count"], a)
                    for r, a in zip(expected["source_requests"], expected["arrival_traces_s"]["steady"])]
    check([identity(r) for r in rows] == expected_ids, "frozen_identity_or_arrival_mismatch")
    check(plan["budget"] == 1024 and raw["target_cap"] == 8 and raw["waiting_order"] == plan["waiting_order"], "frozen_action_mismatch")
    check(all(len(r["output_token_ids"]) == 128 for r in rows if r["status"] == "completed"), "completed_output_length_mismatch")
    for group, selected in (("all", rows), ("short128", [r for r in rows if r["prompt_tokens"] == 128]), ("long2048", [r for r in rows if r["prompt_tokens"] == 2048])):
        summary = summarize_episode_requests(selected, observation_end_s=raw["observation_end_s"], **SLO)
        derived = summary["per_request"]
        for row in derived:
            row["max_itl_s"] = max(row["itl_s"], default=None)
        summary["latency_s"] = {metric: distribution([v for r in derived for v in (r[key] if metric == "itl" else [r[key]]) if v is not None])
                                for metric, key in (("ttft", "ttft_s"), ("tpot", "tpot_s"), ("itl", "itl_s"), ("max_itl", "max_itl_s"), ("request", "request_latency_s"))}
        summary["whole_episode_completion_rps"] = summary["n_completed"] / raw["observation_end_s"] if raw["observation_end_s"] > 0 else None
        groups[group] = summary
    returns, receipts = raw["engine_step_returns"], {}
    for ret in returns:
        i = ret["scheduler_begin"]
        valid = 0 <= i < len(steps) and ret["scheduler_end"] == i + 1 and i not in receipts
        check(valid, "invalid_or_duplicate_step_receipt")
        if valid:
            receipts[i] = ret["received_s"]
            check(ret["call_start_s"] <= steps[i]["start_s"] <= steps[i]["end_s"] <= ret["received_s"] <= raw["observation_end_s"], "receipt_clock_order")
    check(set(receipts) == set(range(len(steps))), "missing_step_receipt")
    event_tokens, event_times, event_counts = {}, {}, Counter()
    for event in raw["output_events"]:
        rid = event["request_id"]
        previous = event_tokens.setdefault(rid, [])
        check(event["prefix_valid"] and event["cumulative_token_ids"] == previous + event["new_token_ids"], "output_prefix_mismatch")
        previous.extend(event["new_token_ids"])
        event_times.setdefault(rid, []).extend([event["received_s"]] * len(event["new_token_ids"]))
        event_counts[event["received_s"]] += 1
    check(all(event_tokens.get(r["request_id"], []) == r["output_token_ids"] and event_times.get(r["request_id"], []) == r["token_times_s"] for r in rows), "output_receipt_completeness")
    check(event_counts == Counter({r["received_s"]: r["output_count"] for r in returns if r["output_count"]}), "return_event_count_mismatch")
    actions, first, prefills = raw["waiting_order_actions"], [], Counter()
    rows_by_id = {r["request_id"]: r for r in rows}
    ranks = {r["request_id"]: i for i, r in enumerate(rows)}
    source_id = raw["internal_to_source"].__getitem__
    bypasses = {rid: 0 for rid in ranks}
    first_schedule_times = {}
    check(raw["arrival_ranks"] == {internal: ranks[source] for internal, source in raw["internal_to_source"].items()}, "arrival_rank_mismatch")
    check(len(actions) == len(steps), "action_step_count_mismatch")
    for i, step in enumerate(steps):
        scheduled = step["scheduled"]
        total = sum(r["scheduled_tokens"] for r in scheduled)
        check(step["step"] == i and step["runtime_token_budget"] == 1024 and total == step["total_scheduled_tokens"] and total <= 1024, "step_budget_or_identity")
        check(step["target_cap"] == 8 and step["actual_active"] <= 8, "fixed_cap8_mismatch")
        missing = set(step["existing_decode_request_ids"]) - {r["request_id"] for r in scheduled if r["decode_tokens"] > 0}
        check(step["existing_decode_all_scheduled"] == (not missing) and set(step["missing_decode_request_ids"]) == missing, "decode_guard_mismatch")
        diagnostics.update(decode_checks=len(step["existing_decode_request_ids"]), decode_skips=len(missing), preemptions=len(step["preempted_request_ids"]))
        a = actions[step["waiting_order_action_index"]]
        check(a["step"] == i and a["mode"] == plan["waiting_order"] and a["error"] is None and a["action_applied"], "queue_action_mismatch")
        check(sorted(a["before"], key=lambda r: r["request_id"]) == sorted(a["after"], key=lambda r: r["request_id"]), "queue_action_not_permutation")
        if plan["waiting_order"] != "bounded_bypass_once":
            expected_order = sorted(a["before"], key=lambda r: r["prompt_tokens"]) if plan["waiting_order"] == "short_prompt_first" else a["before"]
            check(a["after"] == expected_order, "baseline_queue_order_mismatch")
        check(a["running_unchanged"] and a["running_before"] == a["running_after"], "running_state_changed_by_action")
        check(all(r["computed_tokens"] == 0 and r["output_tokens"] == 0 and r["status"] == "WAITING" and raw["internal_to_source"][r["request_id"]] not in first for r in a["before"]), "action_touched_started_request")
        check(all(rows_by_id[raw["internal_to_source"][r["request_id"]]]["prompt_tokens"] == r["prompt_tokens"] and rows_by_id[raw["internal_to_source"][r["request_id"]]]["arrival_s"] <= a["start_s"] for r in a["before"]), "queue_prompt_identity_or_future_arrival")
        changed = a["before"] != a["after"]
        check(a["order_changed"] == changed, "queue_change_record_mismatch")
        diagnostics.update(actual_reorders=int(changed), action_calls=1, queue_multiple_steps=int(len(a["before"]) > 1))
        new = [r["request_id"] for r in scheduled if r["request_id"] not in first]
        check(new == [raw["internal_to_source"][r["request_id"]] for r in a["after"]][:len(new)], "first_schedule_not_queue_prefix")
        check([source_id(rid) for rid in a["first_admitted"]] == new, "first_admission_event_mismatch")
        pending = [r["request_id"] for r in a["before"]]
        check(a["bypasses_before"] == {rid: bypasses[source_id(rid)] for rid in pending}, "bypass_ledger_before_mismatch")
        charges = []
        for internal in a["first_admitted"]:
            pending.remove(internal)
            admitted = source_id(internal)
            first_schedule_times[admitted] = step["start_s"]
            for older in sorted(pending, key=lambda rid: ranks[source_id(rid)]):
                if ranks[source_id(older)] < ranks[admitted]:
                    bypasses[source_id(older)] += 1
                    charges.append(dict(older=older, admitted=internal))
        check(a["bypass_charges"] == charges, "actual_bypass_charges_mismatch")
        check(a["bypasses_after"] == {r["request_id"]: bypasses[source_id(r["request_id"])] for r in a["before"]}, "bypass_ledger_after_mismatch")
        first.extend(new)
        for r in scheduled:
            prefills[r["request_id"]] += r["prefill_tokens"]
            diagnostics["kv_adjustments"] += int(bool(r["computed_adjustment"]))
            check(r["scheduled_tokens"] == r["prefill_tokens"] + r["decode_tokens"], "scheduled_token_partition")
    check(not any(diagnostics[k] for k in ("decode_skips", "preemptions", "kv_adjustments")), "nonpreemptive_invariant_failed")
    if raw["status"] == "COMPLETE":
        check(all(prefills[r["request_id"]] == r["prompt_tokens"] for r in rows), "prefill_conservation")
    if plan["cohort"] == "all_short":
        check(diagnostics["actual_reorders"] == 0, "all_short_negative_control_changed_order")
    check(raw["first_admission_bypasses"] == {internal: bypasses[source] for internal, source in raw["internal_to_source"].items()}, "final_bypass_ledger_mismatch")
    # Independent order-statistic reconstruction, including requests never admitted.
    recomputed = {rid: sum(ranks[earlier] > ranks[rid] for earlier in first[:first.index(rid) if rid in first else len(first)]) for rid in ranks}
    check(recomputed == bypasses, "first_admission_order_bypass_mismatch")
    if plan["waiting_order"] == "bounded_bypass_once":
        check(max(bypasses.values(), default=0) <= 1, "bounded_bypass_limit_exceeded")
    if plan["waiting_order"] == "fcfs" or plan["cohort"] == "all_short":
        check(not any(bypasses.values()), "fcfs_or_negative_control_bypassed")
    per_request = []
    derived_by_id = {r["request_id"]: r for r in groups["all"]["per_request"]}
    for row in rows:
        rid = row["request_id"]
        per_request.append(dict(derived_by_id.get(rid, {}), request_id=rid, document_id=row["document_id"],
            prompt_tokens=row["prompt_tokens"], arrival_s=row["arrival_s"], status=row["status"],
            arrived=row["arrival_s"] <= raw["observation_end_s"], first_schedule_s=first_schedule_times.get(rid),
            arrival_to_first_schedule_s=first_schedule_times[rid]-row["arrival_s"] if rid in first_schedule_times else None,
            actual_first_admission_bypasses=bypasses[rid]))
    fairness = dict(per_request_bypasses=bypasses, distribution=distribution(list(bypasses.values())),
        bypassed_requests=sum(n > 0 for n in bypasses.values()), total_bypasses=sum(bypasses.values()),
        more_than_one=sum(n > 1 for n in bypasses.values()), fixed_limit=1,
        semantics="Later-arrival requests actually first scheduled before this never-started request; same-step prefix order counts. Not a waiting-time/SLO bound.")
    eligible = raw["status"] == "COMPLETE" and groups["all"]["n_completed"] == 16 and not issues
    return dict(source=str(path), plan=plan, block=path.parent.name, phase=raw["phase"], status=raw["status"], error=raw["error"],
                eligible_for_comparison=eligible, issues=sorted(set(issues)), episode_wall_s=raw["observation_end_s"], groups=groups,
                identity=[identity(r) for r in rows], output_hashes={r["request_id"]: digest(r["output_token_ids"]) for r in rows},
                first_schedule_order=first, per_request_measurements=per_request, fairness=fairness,
                queue_reorder_duration_s=distribution([a["end_s"]-a["start_s"] for a in actions]),
                diagnostics=dict(diagnostics), all_short_noop=(eligible and not diagnostics["actual_reorders"]) if plan["cohort"] == "all_short" else None)


def compare(base, spt):
    valid = base["eligible_for_comparison"] and spt["eligible_for_comparison"] and base["identity"] == spt["identity"]
    delta = lambda a, b: None if a is None or b is None else dict(baseline=a, intervention=b, delta=b-a, delta_pct=100*(b/a-1) if a else None)
    result = dict(block=base["block"], cohort=base["plan"]["cohort"], baseline=base["plan"]["waiting_order"], intervention=spt["plan"]["waiting_order"], valid=valid, exclusion=None if valid else "Incomplete/invalid cells or identity mismatch", metrics={})
    if valid:
        result.update(episode_wall_s=delta(base["episode_wall_s"], spt["episode_wall_s"]), first_schedule_order_changed=base["first_schedule_order"] != spt["first_schedule_order"],
                      equal_output_sequences=sum(h == spt["output_hashes"].get(rid) for rid, h in base["output_hashes"].items()), n_sequences=len(base["output_hashes"]))
        result["fairness"] = {metric: delta(base["fairness"][metric], spt["fairness"][metric]) for metric in ("bypassed_requests", "total_bypasses", "more_than_one")}
        for group in base["groups"]:
            result["metrics"][group] = {f"{metric}_{stat}_s": delta(base["groups"][group]["latency_s"][metric][stat], spt["groups"][group]["latency_s"][metric][stat])
                                       for metric, stats in (("ttft", ("mean", "p95", "p99")), ("tpot", ("mean", "p99")), ("itl", ("p95", "p99")), ("max_itl", ("mean", "p99", "max")), ("request", ("mean", "p95", "max"))) for stat in stats}
            result["metrics"][group].update({metric: delta(base["groups"][group][metric], spt["groups"][group][metric]) for metric in ("goodput_rps", "whole_episode_completion_rps", "n_slo_pass")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    cells, warmups, issues, blocks = [], [], [], {}
    for block in ("forward", "reverse"):
        folder = args.results_dir / block
        if not folder.exists():
            issues.append(f"missing_block:{block}")
            continue
        config = read(folder / "config.json")
        blocks[block] = read(folder / "status.json") if (folder / "status.json").exists() else {"status": "UNKNOWN"}
        if config["reference_slo"] != SLO:
            issues.append(f"frozen_SLO_mismatch:{block}")
        for path in sorted(folder.glob("*-raw.json")):
            phase = "warmup" if path.name.startswith("warmup") else "cell"
            try:
                result = analyze_cell(path)
                if result["plan"] != config["warmups" if phase == "warmup" else "plans"][int(path.name.split("-")[1])] or result["phase"] != phase:
                    result["issues"].append("frozen_plan_or_phase_mismatch")
                result["eligible_for_comparison"] &= not result["issues"] and config["reference_slo"] == SLO
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                raw = read(path)
                result = dict(source=str(path), block=block, phase=phase, plan=raw.get("plan", {}), status=raw.get("status"), error=raw.get("error"), issues=[f"invalid_raw:{exc}"], eligible_for_comparison=False)
            (warmups if phase == "warmup" else cells).append(result)
    if not cells and not warmups:
        raise ValueError("no raw episodes; GPU remains UNRUN")
    comparisons = []
    for block in ("forward", "reverse"):
        for cohort in ("all_short", "mixed"):
            pair = [c for c in cells if c["block"] == block and c["plan"].get("cohort") == cohort]
            by_mode = {c["plan"].get("waiting_order"): c for c in pair}
            if len(pair) == 3 and set(by_mode) == set(MODES):
                comparisons.extend(compare(by_mode[baseline], by_mode[intervention]) for baseline, intervention in
                    (("fcfs", "short_prompt_first"), ("fcfs", "bounded_bypass_once"), ("short_prompt_first", "bounded_bypass_once")))
            else:
                issues.append(f"missing_or_duplicate_three_arms:{block}/{cohort}")
    complete = len(cells) == len(warmups) == 12 and all(c["eligible_for_comparison"] for c in cells + warmups) and not issues
    summary = dict(status="DESCRIPTIVE_MEASUREMENT_ONLY" if complete else "PARTIAL_OR_INVALID_MEASUREMENT", cells=cells, warmups=warmups, comparisons=comparisons, blocks=blocks, issues=issues, reference_slo=SLO,
                   semantics="Only current-campaign same-engine arm comparisons; no cross-GPU or old missing-block substitution. Host receipt timing; pooled ITL is descriptive. Every raw and per-request measurement retained, warmups excluded from comparisons. Completion/SLO require completed status; group rates use group first arrival, wall uses shared episode origin. queue_s is host submission lag, not native queue wait. arrival_to_first_schedule_s includes both. Queue reorder timing uses action end-start; ledger_end also includes native schedule and is not pure policy tax. All ledger overhead is inside request/wall metrics. Bypass counts use actual first-admission prefix order, including same-step order, not a wall-clock waiting guarantee. No Oracle, route counterfactual, significance claim or quality guarantee from output equality.")
    fmt = lambda v, scale=1000: "NA" if v is None else f"{v*scale:.3f}"
    lines = ["# Waiting order with one-bypass limit: retained measurements", "", summary["status"], "", summary["semantics"], "", "Latencies: ms. SLO: TTFT 5 s / mean TPOT 0.2 s. Warmups excluded.", "",
             "| Block/cohort/order | Group | Complete/arrived/planned; SLO pass | Wall s | TTFT mean/p95 | TPOT mean | ITL p95/p99 | Request mean/p95/max |", "|---|---|---|---:|---|---:|---|---|"]
    for c in cells:
        label = f"{c['block']}/{c['plan'].get('cohort')}/{c['plan'].get('waiting_order')}"
        for group, g in c.get("groups", {}).items():
            lat = g["latency_s"]
            metrics = [" / ".join(fmt(lat[m][s]) for s in stats) for m, stats in (("ttft", ("mean", "p95")), ("tpot", ("mean",)), ("itl", ("p95", "p99")), ("request", ("mean", "p95", "max")))]
            lines.append(f"| {label} | {group} | {g['n_completed']}/{g['n_arrived']}/{g['n_planned']}; {g['n_slo_pass']} | {fmt(c['episode_wall_s'], 1)} | " + " | ".join(metrics) + " |")
    for c in cells:
        lines.append(f"\n{c['block']}/{c['plan'].get('cohort')}/{c['plan'].get('waiting_order')}: comparison eligible={c['eligible_for_comparison']}; issues={c['issues']}; error={c.get('error')}; diagnostics={c.get('diagnostics')}; all-short no-op={c.get('all_short_noop')}.\n")
        if "fairness" in c:
            fair = c["fairness"]
            lines.append(f"Actual first-admission bypasses: total={fair['total_bypasses']}, affected requests={fair['bypassed_requests']}, max={fair['distribution']['max']}, requests bypassed >1={fair['more_than_one']}. Queue reorder duration: {c['queue_reorder_duration_s']}.\n")
    lines += ["", "Intervention minus baseline, each block separately; negative latency delta is lower.", "", "| Block/cohort/intervention vs baseline | Group | Request mean Δ% | Request p95 Δ% | Request max Δ% | Wall Δ% |", "|---|---|---:|---:|---:|---:|"]
    for comp in comparisons:
        for group, metrics in comp["metrics"].items():
            values = [fmt(metrics[k]["delta_pct"], 1) if metrics[k] else "NA" for k in ("request_mean_s", "request_p95_s", "request_max_s")]
            lines.append(f"| {comp['block']}/{comp['cohort']}/{comp['intervention']} vs {comp['baseline']} | {group} | " + " | ".join(values) + f" | {fmt(comp['episode_wall_s']['delta_pct'], 1)} |")
    for comp in comparisons:
        lines.append(f"\n{comp['block']}/{comp['cohort']}/{comp['intervention']} vs {comp['baseline']}: valid={comp['valid']}; first-schedule order changed={comp.get('first_schedule_order_changed')}; equal outputs={comp.get('equal_output_sequences')}/{comp.get('n_sequences')}; exclusion={comp['exclusion']}.\n")
    lines += ["", f"Campaign issues: {issues}. Retained warmups: {len(warmups)}; details and all metric deltas in summary.json."]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "TABLE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status=summary["status"], cells=len(cells), warmups=len(warmups), issues=issues)))


if __name__ == "__main__":
    main()
