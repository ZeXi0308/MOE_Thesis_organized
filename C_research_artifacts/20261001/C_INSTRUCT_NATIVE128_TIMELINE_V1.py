#!/usr/bin/env python3
"""Post-observation queue witness and equal-request host latency description."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(values):
    values = sorted(values)
    return dict(count=len(values), minimum=min(values), mean=statistics.mean(values),
                median=statistics.median(values),
                p95_nearest_rank=values[math.ceil(.95 * len(values)) - 1],
                maximum=max(values))


def analyze(run):
    names = ("measured-steps.json", "measured-outputs.json", "timing.json", "status.json")
    raw = {n: (run / n).read_bytes() for n in names}
    d = {n: json.loads(b) for n, b in raw.items()}
    calls = d["measured-steps.json"]["scheduler_calls"]
    rows = d["measured-outputs.json"]
    status = d["status.json"]
    if status["status"] != "COMPLETE" or len(rows) != 128:
        raise ValueError("requires the complete native128 observation")
    first_all = next(c for c in calls if c["after"]["running"] == 128)
    first_empty = next(c for c in calls if c["after"]["waiting"] == 0)
    initial = calls[:first_empty["call"]]
    first_finish = min(r["host_elapsed_s"] for r in rows)
    ttft, flow, gaps, per_request = [], [], [], []
    for r in rows:
        times = r["token_times_s"]
        if not r["finished"] or not times or len(times) != len(r["output_token_ids"]):
            raise ValueError("incomplete request/token timing")
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("non-monotonic token return timing")
        unique = sorted(set(times))
        gap = max((b - a for a, b in zip(unique, unique[1:])), default=0)
        a = times[0] - r["arrival_s"]
        f = r["host_elapsed_s"] - r["arrival_s"]
        ttft.append(a); flow.append(f); gaps.append(gap)
        per_request.append(dict(request_id=r["request_id"], ttft_s=a, flow_s=f,
                                maximum_distinct_host_return_gap_s=gap))
    t = d["timing.json"]
    durations = {name: t[end] - t[start] for name, start, end in (
        ("model_prepare_s", "model_prepare_start_perf_s", "model_prepare_end_perf_s"),
        ("engine_init_s", "engine_init_start_perf_s", "engine_init_end_perf_s"),
        ("warmup_s", "warmup_start_perf_s", "warmup_end_perf_s"),
        ("measurement_wrapper_s", "measurement_start_perf_s", "measurement_return_perf_s"),
        ("shutdown_s", "shutdown_start_perf_s", "process_end_perf_s"),
        ("whole_child_s", "process_start_perf_s", "process_end_perf_s"))}
    episode = status["observation_end_s"]
    return dict(schema="c-instruct-native128-post-observation-timeline-v1",
        first_all_128_running=first_all, first_zero_waiting=first_empty,
        first_request_completion_s=first_finish,
        all_128_running_before_any_completion=first_all["host_s"] < first_finish,
        initial_calls_with_waiting_after_schedule=len(initial),
        all_initial_waiting_calls_exhausted_1024_token_budget=all(
            c["after"]["waiting"] > 0 and c["scheduled_tokens_total"] == 1024 for c in initial),
        waiting_reappeared_after_first_zero=any(c["after"]["waiting"] for c in calls[first_empty["call"]:]),
        peak_after_schedule_call=max(calls, key=lambda c: c["after"]["used_blocks"]),
        host_ttft_s=summarize(ttft), host_flow_s=summarize(flow),
        request_equal_weight_maximum_distinct_host_return_gap_s=summarize(gaps),
        descriptive_request_rate_per_s=len(rows) / episode,
        descriptive_output_token_rate_per_s=sum(len(r["output_token_ids"]) for r in rows) / episode,
        generation_observation_s=episode, separate_durations=durations,
        per_request=per_request,
        limitations=["Post-observation description, not a new preregistered service target or policy comparison.",
            "Host return timings include instrumentation and do not expose token timing within a batched return.",
            "Token-budget exhaustion is recorded; no exclusive causal decomposition of queue delay is claimed.",
            "Before/after schedule states share the recorded post-call timestamp; not continuous physical peaks."],
        input_sha256={n: hashlib.sha256(b).hexdigest() for n, b in raw.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = analyze(a.run_dir)
    with a.output.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_request"}, indent=2))
