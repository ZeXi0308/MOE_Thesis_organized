#!/usr/bin/env python3
"""Describe natural-length native1024 repeats and their actual prefill tails."""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))
from metrics import _distribution, summarize_episode_requests

SLO = dict(ttft_slo_s=5.0, tpot_slo_s=0.2)
read = lambda path: json.loads(path.read_text())
identity = lambda row: (row["request_id"], row["document_id"], row["prompt_token_ids_sha256"],
                        row["prompt_tokens"], row["arrival_s"])


def distribution(values):
    values = [value for value in values if value is not None]
    return dict(_distribution(values), mean=statistics.mean(values) if values else None,
                max=max(values) if values else None)


def analyze_cell(path):
    raw = read(path)
    rows, steps, plan = raw["requests"], raw["scheduler_steps"], raw["plan"]
    issues = []
    check = lambda condition, issue: issues.append(issue) if not condition else None
    expected = read(ROOT / "prepared/natural/workload.json")
    expected_identity = [(row["request_id"], row["document_id"], row["prompt_token_ids_sha256"],
                          row["prompt_token_count"], arrival)
                         for row, arrival in zip(expected["source_requests"],
                                                 expected["arrival_traces_s"]["steady"])]
    check([identity(row) for row in rows] == expected_identity, "frozen_identity_or_arrival_mismatch")
    check(all(row["original_document_token_count"] == row["prompt_token_count"]
              for row in expected["source_requests"]), "natural_prompt_is_truncated")
    frozen = (plan.get("cohort"), plan.get("prefill_policy"), plan.get("budget"),
              plan.get("long_prefill_token_threshold"), plan.get("waiting_order"))
    check(frozen == ("natural", "native1024", 1024, 0, "fcfs")
          and raw["target_cap"] == 8 and raw["runtime_token_budget"] == 1024
          and raw["long_prefill_token_threshold"] == 0 and raw["waiting_order"] == "fcfs",
          "frozen_native_action_mismatch")
    state = raw["budget_state"]
    check((state["runtime_budget"], state["long_prefill_token_threshold"],
           state["compiled_token_capacity"], state["cap"]) == (1024, 0, 1024, 8),
          "drained_native_state_mismatch")
    check(len(rows) == 16, "planned_request_count_mismatch")

    receipts = {}
    for ret in raw["engine_step_returns"]:
        index = ret["scheduler_begin"]
        valid = 0 <= index < len(steps) and ret["scheduler_end"] == index + 1 and index not in receipts
        check(valid, "invalid_or_duplicate_step_receipt")
        if valid:
            receipts[index] = ret
            check(ret["call_start_s"] <= steps[index]["start_s"] <= steps[index]["end_s"]
                  <= ret["received_s"] <= raw["observation_end_s"], "receipt_clock_order")
    check(set(receipts) == set(range(len(steps))), "missing_step_receipt")

    event_tokens, event_times = defaultdict(list), defaultdict(list)
    for event in raw["output_events"]:
        previous = event_tokens[event["request_id"]]
        check(event["prefix_valid"] and event["cumulative_token_ids"] == previous + event["new_token_ids"],
              "output_prefix_mismatch")
        previous.extend(event["new_token_ids"])
        event_times[event["request_id"]].extend([event["received_s"]] * len(event["new_token_ids"]))
    check(all(event_tokens[row["request_id"]] == row["output_token_ids"]
              and event_times[row["request_id"]] == row["token_times_s"] for row in rows),
          "output_receipt_completeness")

    actions, chunks, first_schedule, first_seen = raw["waiting_order_actions"], defaultdict(list), {}, set()
    rows_by_id = {row["request_id"]: row for row in rows}
    source_id = raw["internal_to_source"].__getitem__
    check(len(actions) == len(steps), "action_step_count_mismatch")
    decode_checks = decode_skips = preemptions = adjustments = 0
    step_context = {}
    for index, step in enumerate(steps):
        scheduled = step["scheduled"]
        total = sum(row["scheduled_tokens"] for row in scheduled)
        check(step["step"] == index and step["runtime_token_budget"] == 1024
              and step["long_prefill_token_threshold"] == 0
              and total == step["total_scheduled_tokens"] and total <= 1024,
              "step_budget_or_identity")
        check(step["target_cap"] == 8 and step["actual_active"] <= 8, "fixed_cap8_mismatch")
        decoded = {row["request_id"] for row in scheduled if row["decode_tokens"] > 0}
        missing = set(step["existing_decode_request_ids"]) - decoded
        check(step["existing_decode_all_scheduled"] == (not missing)
              and set(step["missing_decode_request_ids"]) == missing, "decode_guard_mismatch")
        decode_checks += len(step["existing_decode_request_ids"])
        decode_skips += len(missing)
        preemptions += len(step["preempted_request_ids"])
        action = actions[step["waiting_order_action_index"]]
        check(action["step"] == index and action["mode"] == "fcfs" and action["error"] is None
              and action["action_applied"] and action["before"] == action["after"]
              and action["running_unchanged"], "fcfs_action_mismatch")
        check(action["runtime_token_budget"] == 1024
              and action["long_prefill_token_threshold"] == 0, "action_live_config_mismatch")
        check(all(rows_by_id[source_id(item["request_id"])]["arrival_s"] <= action["start_s"]
                  for item in action["before"]), "future_arrival_in_waiting_queue")
        newly_scheduled = [row["request_id"] for row in scheduled if row["request_id"] not in first_seen]
        check(newly_scheduled == [source_id(item["request_id"]) for item in action["after"]][:len(newly_scheduled)]
              and [source_id(rid) for rid in action["first_admitted"]] == newly_scheduled,
              "first_schedule_not_fcfs_prefix")
        for rid in newly_scheduled:
            first_schedule[rid] = step["start_s"]
        first_seen.update(row["request_id"] for row in scheduled)
        for row in scheduled:
            check(row["scheduled_tokens"] == row["prefill_tokens"] + row["decode_tokens"],
                  "scheduled_token_partition")
            adjustments += int(bool(row["computed_adjustment"]))
            if row["prefill_tokens"] > 0:
                chunks[row["request_id"]].append(dict(step_id=index, prefill_tokens=row["prefill_tokens"],
                    scheduled_start_computed=row["scheduled_start_computed"],
                    computed_before=row["computed_before"]))
        ret = receipts.get(index)
        step_context[index] = dict(step_id=index,
            decode_tokens=sum(row["decode_tokens"] for row in scheduled),
            prefill_tokens=sum(row["prefill_tokens"] for row in scheduled),
            total_scheduled_tokens=total, decode_requests=step["decode_requests"],
            prefill_requests=step["prefill_requests"],
            host_engine_step_duration_s=None if ret is None else ret["received_s"] - ret["call_start_s"],
            semantics="Observed whole engine.step duration for all step work; not removable tail cost.")
    check(not decode_skips and not preemptions and not adjustments, "nonpreemptive_invariant_failed")
    check(not any(raw["first_admission_bypasses"].values()), "fcfs_first_admission_bypass")

    summary = summarize_episode_requests(rows, observation_end_s=raw["observation_end_s"], **SLO)
    derived = {row["request_id"]: row for row in summary["per_request"]}
    per_request = []
    for row in rows:
        rid = row["request_id"]
        request_chunks = chunks[rid]
        total_prefill = sum(chunk["prefill_tokens"] for chunk in request_chunks)
        if raw["status"] == "COMPLETE":
            check(total_prefill == row["prompt_tokens"], "prefill_conservation")
        last_tokens = request_chunks[-1]["prefill_tokens"] if request_chunks else None
        multi = len(request_chunks) >= 2
        tiny = multi and last_tokens <= 32
        first = first_schedule.get(rid)
        first_token = row["token_times_s"][0] if row["token_times_s"] else None
        completion = row["completion_s"]
        parts = dict(
            host_submission_lag_s=None if row["admission_s"] is None else row["admission_s"] - row["arrival_s"],
            engine_add_call_s=None if row["admission_s"] is None or row["engine_add_return_s"] is None
                else row["engine_add_return_s"] - row["admission_s"],
            post_add_to_first_schedule_s=None if first is None or row["engine_add_return_s"] is None
                else first - row["engine_add_return_s"],
            arrival_to_first_schedule_s=None if first is None else first - row["arrival_s"],
            first_schedule_to_first_token_s=None if first is None or first_token is None else first_token - first,
            first_token_to_completion_s=None if first_token is None or completion is None else completion - first_token)
        if completion is not None and first is not None and first_token is not None:
            check(abs((parts["arrival_to_first_schedule_s"] + parts["first_schedule_to_first_token_s"]
                       + parts["first_token_to_completion_s"])
                      - (completion - row["arrival_s"])) < 1e-9, "flow_partition_mismatch")
        metric = derived[rid]
        per_request.append(dict(metric, document_id=row["document_id"], prompt_tokens=row["prompt_tokens"],
            flow_s=metric["request_latency_s"], max_itl_s=max(metric["itl_s"], default=None),
            first_schedule_s=first, queue_decomposition_s=parts,
            prefill_chunks=request_chunks, prefill_chunk_step_ids=[chunk["step_id"] for chunk in request_chunks],
            prefill_chunk_tokens=[chunk["prefill_tokens"] for chunk in request_chunks],
            prefill_chunk_count=len(request_chunks), sum_prefill_tokens=total_prefill,
            last_prefill_tokens=last_tokens, multi_chunk_prefill=multi, tiny_tail=tiny,
            last_prefill_step=None if not request_chunks else step_context[request_chunks[-1]["step_id"]]))
    multi_count = sum(row["multi_chunk_prefill"] for row in per_request)
    tiny_count = sum(row["tiny_tail"] for row in per_request)
    summary["per_request"] = per_request
    summary["latency_s"] = {
        key: distribution([value for row in per_request for value in
            (row[source] if key == "itl" else [row[source]])])
        for key, source in (("ttft", "ttft_s"), ("tpot", "tpot_s"), ("itl", "itl_s"),
                            ("max_itl", "max_itl_s"), ("flow", "flow_s"))}
    queue_keys = tuple(per_request[0]["queue_decomposition_s"]) if per_request else ()
    summary["queue_decomposition_s"] = {key: distribution(
        [row["queue_decomposition_s"][key] for row in per_request]) for key in queue_keys}
    tail = dict(definition="prefill_chunk_count >= 2 and last_prefill_tokens <= 32",
        single_chunk_requests_excluded=True, all_request_denominator=len(per_request),
        multi_chunk_denominator=multi_count, tiny_tail_requests=tiny_count,
        tiny_tail_fraction_all_requests=tiny_count / len(per_request) if per_request else None,
        tiny_tail_fraction_multi_chunk=tiny_count / multi_count if multi_count else None,
        chunk_count=distribution([row["prefill_chunk_count"] for row in per_request]),
        last_prefill_tokens_multi_chunk=distribution(
            [row["last_prefill_tokens"] for row in per_request if row["multi_chunk_prefill"]]))
    eligible = raw["status"] == "COMPLETE" and summary["n_completed"] == 16 and not issues
    return dict(source=str(path), block=path.parent.name, phase=raw["phase"], plan=plan,
        status=raw["status"], error=raw["error"], eligible=eligible, issues=sorted(set(issues)),
        episode_wall_s=raw["observation_end_s"], metrics=summary, prefill_tail=tail,
        diagnostics=dict(scheduler_steps=len(steps), decode_checks=decode_checks,
                         decode_skips=decode_skips, preemptions=preemptions,
                         computed_adjustments=adjustments),
        semantics="Host receipt timing. Full flow is arrival-to-completion. Last-prefill engine.step duration includes all decode/prefill work and is observational, not removable cost.")


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
        if config.get("reference_slo") != SLO:
            issues.append(f"frozen_SLO_mismatch:{block}")
        expected_files = {"warmup-000-raw.json", "cell-000-raw.json"}
        actual_files = {path.name for path in folder.glob("*-raw.json")}
        if actual_files != expected_files:
            issues.append(f"unexpected_episode_set:{block}")
        for phase, key in (("warmup", "warmups"), ("cell", "plans")):
            path = folder / f"{phase}-000-raw.json"
            if not path.exists():
                result = dict(source=str(path), block=block, phase=phase, status="MISSING",
                              error="missing raw", eligible=False, issues=["missing_raw"])
            else:
                try:
                    result = analyze_cell(path)
                    plans = config.get(key, [])
                    if len(plans) != 1 or result["plan"] != plans[0] or result["phase"] != phase:
                        result["issues"].append("frozen_plan_or_phase_mismatch")
                    result["issues"] = sorted(set(result["issues"]))
                    result["eligible"] &= not result["issues"]
                except (KeyError, ValueError, TypeError, IndexError) as exc:
                    raw = read(path)
                    result = dict(source=str(path), block=block, phase=phase,
                        plan=raw.get("plan", {}), status=raw.get("status"), error=raw.get("error"),
                        eligible=False, issues=[f"invalid_raw:{type(exc).__name__}:{exc}"])
            (warmups if phase == "warmup" else cells).append(result)
    if not cells and not warmups:
        raise ValueError("no raw episodes; GPU remains UNRUN")
    complete = len(cells) == len(warmups) == 2 and all(
        row.get("eligible") for row in cells + warmups) and not issues
    semantics = ("Two separate fresh-engine native1024/FCFS/cap8/threshold0 repeats are retained "
        "without pooling or cross-campaign causal ratios. Tiny tail means >=2 actual prefill chunks "
        "and final chunk <=32 tokens; single-chunk requests are excluded. Queue components and whole "
        "engine.step durations use host clocks; the latter includes all same-step work and is not removable cost.")
    summary = dict(status="DESCRIPTIVE_NATURAL_PREFILL_MEASUREMENT" if complete
        else "PARTIAL_OR_INVALID_MEASUREMENT", cells=cells, warmups=warmups, blocks=blocks,
        issues=issues, reference_slo=SLO, comparisons=[], semantics=semantics)

    fmt = lambda value, scale=1000: "NA" if value is None else f"{value * scale:.3f}"
    lines = ["# Natural prefill tails: retained native baseline repeats", "", summary["status"],
        "", semantics, "", "Latencies: ms. Warmups and formal repeats are shown separately.", "",
        "| Phase/block | Status | Complete/arrived/planned | Wall s | TTFT mean/p95 | TPOT mean/p95 | maxITL mean/p95/max | Flow mean/p95/max | Arrival→first schedule mean/p95 | Multi-chunk | Tiny/all | Tiny/multi |",
        "|---|---|---|---:|---|---|---|---|---|---:|---:|---:|"]
    for row in warmups + cells:
        if "metrics" not in row:
            lines.append(f"| {row['phase']}/{row['block']} | {row['status']} | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
            continue
        metric, lat, tail = row["metrics"], row["metrics"]["latency_s"], row["prefill_tail"]
        shown = lambda name, stats: " / ".join(fmt(lat[name][stat]) for stat in stats)
        queue = metric["queue_decomposition_s"]["arrival_to_first_schedule_s"]
        lines.append(f"| {row['phase']}/{row['block']} | {row['status']} | "
            f"{metric['n_completed']}/{metric['n_arrived']}/{metric['n_planned']} | "
            f"{fmt(row['episode_wall_s'], 1)} | {shown('ttft', ('mean','p95'))} | "
            f"{shown('tpot', ('mean','p95'))} | {shown('max_itl', ('mean','p95','max'))} | "
            f"{shown('flow', ('mean','p95','max'))} | {fmt(queue['mean'])} / {fmt(queue['p95'])} | "
            f"{tail['multi_chunk_denominator']} | {tail['tiny_tail_requests']}/{tail['all_request_denominator']} | "
            f"{tail['tiny_tail_requests']}/{tail['multi_chunk_denominator']} |")
    lines += ["", "Formal per-request prefill accounting:",
        "", "| Block/request | Prompt | Chunk tokens | Step IDs | Sum | Last | Tiny tail | Last-step decode/prefill/total | engine.step ms |",
        "|---|---:|---|---|---:|---:|---|---|---:|"]
    for cell in cells:
        for row in cell.get("metrics", {}).get("per_request", []):
            context = row["last_prefill_step"] or {}
            work = "/".join(str(context.get(key, "NA")) for key in
                            ("decode_tokens", "prefill_tokens", "total_scheduled_tokens"))
            lines.append(f"| {cell['block']}/{row['request_id']} | {row['prompt_tokens']} | "
                f"{row['prefill_chunk_tokens']} | {row['prefill_chunk_step_ids']} | "
                f"{row['sum_prefill_tokens']} | {row['last_prefill_tokens']} | {row['tiny_tail']} | "
                f"{work} | {fmt(context.get('host_engine_step_duration_s'))} |")
    lines += ["", f"Campaign issues: {issues}. Cell-local issues and every warmup/formal request are in summary.json."]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "TABLE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status=summary["status"], cells=len(cells),
                         warmups=len(warmups), issues=issues)))


if __name__ == "__main__":
    main()
