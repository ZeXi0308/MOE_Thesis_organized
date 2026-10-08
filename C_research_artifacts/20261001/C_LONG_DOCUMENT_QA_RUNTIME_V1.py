#!/usr/bin/env python3
"""Describe the completed fixed LongBench first-16 native run from original host records.

This is a post-observation runtime summary, not a policy comparison or GPU benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


N = 16
CAP = 64
FILES = ("status.json", "measured-outputs.json", "measured-steps.json",
         "timing.json", "resolved-scheduler.json", "native-drain.json",
         "engine_args.json")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summary(values: list[float | int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty measurement set")
    return dict(count=len(ordered), minimum=ordered[0], mean=statistics.mean(ordered),
                median=statistics.median(ordered),
                p95_nearest_rank=ordered[math.ceil(.95 * len(ordered)) - 1],
                maximum=ordered[-1])


def analyze(run: Path, quality_path: Path) -> dict:
    raw = {name: (run / name).read_bytes() for name in FILES}
    data = {name: json.loads(content) for name, content in raw.items()}
    quality_raw = quality_path.read_bytes()
    quality = json.loads(quality_raw)
    status, outputs = data["status.json"], data["measured-outputs.json"]
    trace, timing = data["measured-steps.json"], data["timing.json"]
    resolved, drain = data["resolved-scheduler.json"], data["native-drain.json"]
    args = data["engine_args.json"]
    if (status.get("status") != "COMPLETE" or status.get("request_count") != N
            or not isinstance(outputs, list) or len(outputs) != N
            or quality.get("status") != "COMPLETE" or quality.get("requests_completed") != N
            or quality.get("requests_planned") != N or quality.get("issues")
            or drain.get("status") != "QUALIFIED"
            or status.get("eos_qualification_status") != "QUALIFIED"):
        raise ValueError("requires the completed, qualified LongBench first-16 run")
    for name in FILES:
        pinned = quality.get("original_run_sha256", {}).get(name)
        if pinned is not None and pinned != sha(raw[name]):
            raise ValueError("quality analysis and original runtime file differ: " + name)
    if (resolved.get("usable_kv_blocks") != 4096
            or resolved.get("allocated_kv_blocks") != 4097
            or resolved.get("max_num_seqs") != args.get("max_num_seqs")
            or resolved.get("max_num_batched_tokens") != args.get("max_num_batched_tokens")
            or resolved.get("prefix_caching") != args.get("enable_prefix_caching")
            or not resolved.get("prefix_caching")):
        raise ValueError("resolved native resource receipt differs")
    if (drain.get("free_blocks") != 4096 or drain.get("running") != 0
            or drain.get("waiting") != 0 or drain.get("scheduler_request_count") != 0):
        raise ValueError("native request/KV state did not drain")

    quality_rows = {r["request_id"]: r for r in quality["per_request"]}
    ids = [r["request_id"] for r in outputs]
    if (len(set(ids)) != N or set(ids) != set(quality_rows)
            or any(quality_rows[rid]["status"] != "COMPLETE" for rid in ids)):
        raise ValueError("runtime and quality request inventories differ")
    per_request = []
    for row in outputs:
        rid, tokens, times = (row["request_id"], row["output_token_ids"],
                              row["token_times_s"])
        arrival, finish = row["arrival_s"], row["host_elapsed_s"]
        reason = row["finish_reason"]
        if (not row["finished"] or reason not in ("stop", "length")
                or not isinstance(tokens, list) or not isinstance(times, list)
                or not tokens or len(tokens) != len(times) or len(tokens) > CAP
                or any(type(t) is not int or t < 0 for t in tokens)
                or any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
                or any(b < a for a, b in zip(times, times[1:]))
                or not all(isinstance(t, (int, float)) and math.isfinite(t)
                           for t in (arrival, finish))
                or arrival != 0 or not arrival <= times[0] <= times[-1] <= finish
                or finish > status["observation_end_s"]
                or (reason == "length" and len(tokens) != CAP)
                or quality_rows[rid]["output_tokens"] != len(tokens)
                or quality_rows[rid]["finish_reason"] != reason):
            raise ValueError("original request completion/timing differs: " + rid)
        distinct = sorted(set(times))
        gap = max((b - a for a, b in zip(distinct, distinct[1:])), default=0.0)
        per_request.append(dict(request_id=rid, source_index=row["source_index"],
            prompt_tokens=len(row["prompt_token_ids"]), output_tokens=len(tokens),
            finish_reason=reason, arrival_s=arrival,
            host_ttft_s=times[0] - arrival,
            host_arrival_to_completion_s=finish - arrival,
            maximum_distinct_host_return_gap_s=gap,
            distinct_host_return_times=len(distinct)))
    per_request.sort(key=lambda r: r["source_index"])
    if ([r["source_index"] for r in per_request] != list(range(N))
            or sum(r["output_tokens"] for r in per_request) != status["output_tokens"]
            or dict(Counter(r["finish_reason"] for r in per_request))
               != status["finish_reason_counts"]
            or sum(r["finish_reason"] == "stop" for r in per_request)
               != quality["natural_eos_count"]
            or sum(r["finish_reason"] == "length" for r in per_request)
               != quality["length_cap_count"]):
        raise ValueError("original output counts differ from qualified receipts")

    calls, steps = trace["scheduler_calls"], trace["steps"]
    batch = resolved["max_num_batched_tokens"]
    seqs, blocks = resolved["max_num_seqs"], resolved["usable_kv_blocks"]
    if (not calls or len(calls) != len(steps) or len(calls) != status["schedule_calls"]
            or any(c["call"] != i or not 0 <= c["scheduled_tokens_total"] <= batch
                   or not 0 <= c["scheduled_request_count"] <= seqs
                   for i, c in enumerate(calls))):
        raise ValueError("scheduler-call inventory or budget differs")
    states = [c[phase] for c in calls for phase in ("before", "after")]
    if any(not (0 <= s["used_blocks"] <= blocks
                and 0 <= s["running"] <= seqs and 0 <= s["waiting"] <= N)
           for s in states):
        raise ValueError("observed schedule state outside native bounds")
    if any(not (0 <= step["start_s"] <= step["return_s"]
                <= status["observation_end_s"]
                and 0 <= step["used_blocks_after_step"] <= blocks)
           for step in steps):
        raise ValueError("observed step timestamps or block counts differ")
    preempted = [rid for c in calls for rid in c["preempted_request_ids"]]
    if (preempted != trace["preempted_request_ids"]
            or len(preempted) != status["preemptions"]
            or max(c["after"]["used_blocks"] for c in calls)
               != status["peak_used_blocks_after_schedule"]
            or max(c["after"]["running"] for c in calls)
               != status["peak_running_after_schedule"]
            or max(c["after"]["waiting"] for c in calls)
               != status["peak_waiting_after_schedule"]):
        raise ValueError("schedule observation and status receipt differ")
    first_empty = next((c for c in calls if c["after"]["waiting"] == 0), None)
    if first_empty is None:
        raise ValueError("waiting queue never cleared in completed trace")

    spans = {label: timing[end] - timing[start] for label, start, end in (
        ("model_prepare_s", "model_prepare_start_perf_s", "model_prepare_end_perf_s"),
        ("engine_init_s", "engine_init_start_perf_s", "engine_init_end_perf_s"),
        ("warmup_s", "warmup_start_perf_s", "warmup_end_perf_s"),
        ("measurement_wrapper_s", "measurement_start_perf_s", "measurement_return_perf_s"),
        ("shutdown_s", "shutdown_start_perf_s", "process_end_perf_s"),
        ("whole_child_s", "process_start_perf_s", "process_end_perf_s"))}
    if any(not math.isfinite(value) or value < 0 for value in spans.values()):
        raise ValueError("stage timing differs")
    return dict(schema="c-longbench-multifieldqa-en-native-runtime-v1", status="COMPLETE",
        scope="Post-observation fixed first-16 native host/runtime description; no policy comparison",
        requests_completed=N, resolved_scheduler=resolved,
        prompt_tokens=summary([r["prompt_tokens"] for r in per_request]),
        output_tokens=summary([r["output_tokens"] for r in per_request]),
        host_ttft_s=summary([r["host_ttft_s"] for r in per_request]),
        host_arrival_to_completion_s=summary(
            [r["host_arrival_to_completion_s"] for r in per_request]),
        request_equal_weight_maximum_distinct_host_return_gap_s=summary(
            [r["maximum_distinct_host_return_gap_s"] for r in per_request]),
        per_request=per_request, schedule_calls=len(calls),
        peak_running_before_schedule=max(c["before"]["running"] for c in calls),
        peak_running_after_schedule=max(c["after"]["running"] for c in calls),
        peak_waiting_before_schedule=max(c["before"]["waiting"] for c in calls),
        peak_waiting_after_schedule=max(c["after"]["waiting"] for c in calls),
        peak_used_blocks_before_schedule=max(c["before"]["used_blocks"] for c in calls),
        peak_used_blocks_after_schedule=max(c["after"]["used_blocks"] for c in calls),
        peak_used_blocks_after_step=max(s["used_blocks_after_step"] for s in steps),
        minimum_observed_free_blocks=blocks - max(
            [s["used_blocks"] for s in states]
            + [s["used_blocks_after_step"] for s in steps]),
        peak_scheduled_tokens_per_call=max(c["scheduled_tokens_total"] for c in calls),
        total_scheduled_tokens=sum(c["scheduled_tokens_total"] for c in calls),
        calls_scheduling_full_batch=sum(c["scheduled_tokens_total"] == batch for c in calls),
        preemption_events=len(preempted), unique_preempted_requests=len(set(preempted)),
        first_waiting_zero_after_schedule=dict(call=first_empty["call"],
            host_s=first_empty["host_s"], after=first_empty["after"]),
        waiting_reappeared_after_first_zero=any(
            c["after"]["waiting"] > 0 for c in calls[first_empty["call"]:]),
        first_request_completion_s=min(r["host_arrival_to_completion_s"] for r in per_request),
        generation_observation_s=status["observation_end_s"],
        stage_durations_s=spans,
        limitations=[
            "TTFT, completion, and distinct-return gaps are host observations; instrumentation and batched multi-token returns are included, and within-return token timing is unavailable.",
            "Waiting includes scheduling/token-budget effects and is not an exclusive KV-capacity stall measurement.",
            "Before/after-schedule and after-step block snapshots do not expose every internal allocator transient.",
            "Initialization, warmup, measurement, and shutdown durations describe this instrumented child only; no cross-run speed or policy claim.",
        ],
        original_run_sha256={name: sha(content) for name, content in raw.items()},
        quality_analysis_sha256=sha(quality_raw))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.quality)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in (
        "status", "schedule_calls", "peak_used_blocks_after_schedule",
        "peak_waiting_after_schedule", "preemption_events")}))


if __name__ == "__main__":
    main()
