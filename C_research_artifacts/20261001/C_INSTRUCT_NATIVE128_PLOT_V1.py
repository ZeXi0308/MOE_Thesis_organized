#!/usr/bin/env python3
"""Plot observed native128 scheduler boundary states and all-request quality.

The paired before/after values share the host timestamp recorded immediately
after each scheduler call. Markers are snapshots, not continuous KV peaks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAPACITY = 4096
REQUESTS = 128
STEM = "c_instruct_native128_schedule_snapshots_v1"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def count(value, limit: int, name: str) -> int:
    require(type(value) is int and 0 <= value <= limit, f"invalid {name}")
    return value


def checked_data(run_dir: Path, quality_path: Path, capacity_path: Path):
    steps_path = run_dir / "measured-steps.json"
    raw = steps_path.read_bytes()
    trace = json.loads(raw)
    quality = load(quality_path)
    capacity = load(capacity_path)
    digest = hashlib.sha256(raw).hexdigest()
    require(quality.get("schema") == "c-instruct-native128-analysis-v1"
            and quality.get("status") == "PILOT_COMPLETE"
            and quality.get("requests_planned") == REQUESTS
            and quality.get("requests_completed") == REQUESTS,
            "quality report is not the complete native128 cohort")
    require(capacity.get("schema") == "c-instruct-native128-capacity-analysis-v1"
            and capacity.get("status") == "COMPLETE"
            and capacity.get("requests_planned") == REQUESTS
            and capacity.get("requests_completed") == REQUESTS,
            "capacity report is not the complete native128 cohort")
    require(quality.get("original_run_sha256", {}).get("measured-steps.json") == digest
            and capacity.get("input_sha256", {}).get("measured-steps.json") == digest,
            "quality/capacity reports do not match the raw scheduler trace")
    resolved = capacity.get("resolved_scheduler", {})
    require(resolved.get("max_num_seqs") == REQUESTS
            and resolved.get("max_num_batched_tokens") == 1024
            and resolved.get("usable_kv_blocks") == CAPACITY,
            "resolved native scheduler/capacity contract differs")
    correct = count(quality.get("answer_correct_count"), REQUESTS, "correct count")
    eos = count(quality.get("natural_eos_count"), REQUESTS, "EOS count")
    capped = count(quality.get("length_cap_count"), REQUESTS, "length-cap count")
    correct_eos = count(quality.get("answer_correct_and_eos_count"), REQUESTS,
                        "correct-and-EOS count")
    require(eos + capped == REQUESTS and correct_eos <= min(correct, eos),
            "quality counts disagree with the complete 128-request denominator")
    calls = trace.get("scheduler_calls")
    require(isinstance(calls, list) and bool(calls)
            and len(calls) == capacity.get("schedule_calls"),
            "scheduler call inventory differs from capacity report")
    series = {name: [] for name in (
        "host_s", "before_blocks", "after_blocks", "before_running",
        "after_running", "before_waiting", "after_waiting")}
    for index, call in enumerate(calls):
        require(isinstance(call, dict) and call.get("call") == index,
                "scheduler call IDs are not source ordered")
        host_s = call.get("host_s")
        require(type(host_s) in (int, float) and math.isfinite(host_s)
                and host_s >= 0 and (not series["host_s"] or host_s >= series["host_s"][-1]),
                "invalid or decreasing host_s")
        series["host_s"].append(host_s)
        for phase in ("before", "after"):
            state = call.get(phase)
            require(isinstance(state, dict), f"missing {phase} scheduler state")
            series[f"{phase}_blocks"].append(
                count(state.get("used_blocks"), CAPACITY, f"{phase} used blocks"))
            series[f"{phase}_running"].append(
                count(state.get("running"), REQUESTS, f"{phase} running"))
            series[f"{phase}_waiting"].append(
                count(state.get("waiting"), REQUESTS, f"{phase} waiting"))
    require(series["host_s"][-1] > 0, "observation has no positive host time")
    require(max(series["before_blocks"]) == capacity.get("peak_used_blocks_before_schedule")
            and max(series["after_blocks"]) == capacity.get("peak_used_blocks_after_schedule")
            and max(series["before_running"] + series["after_running"])
            == capacity.get("peak_running")
            and max(series["before_waiting"] + series["after_waiting"])
            == capacity.get("peak_waiting"),
            "recorded boundary peaks disagree with capacity report")
    preemptions = count(capacity.get("preemption_events"), len(calls) * REQUESTS,
                        "preemption event count")
    return series, dict(correct=correct, eos=eos, capped=capped,
                        correct_eos=correct_eos, preemptions=preemptions)


def plot(series: dict, counts: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png, svg = (output_dir / f"{STEM}.{suffix}" for suffix in ("png", "svg"))
    if any(path.exists() or path.is_symlink() for path in (png, svg)):
        raise FileExistsError("plot output already exists; source figures are immutable")

    times = series["host_s"]
    fig, (kv, queue) = plt.subplots(2, 1, sharex=True, figsize=(11, 7.1))
    fig.subplots_adjust(left=.09, right=.97, top=.84, bottom=.14, hspace=.18)
    fig.suptitle("Native128 GSM8K: observed scheduler boundary states", y=.97,
                 fontsize=15, fontweight="semibold")
    fig.text(.5, .91,
             f"All 128 requests: correct {counts['correct']}/128  |  "
             f"natural EOS {counts['eos']}/128  |  "
             f"length cap {counts['capped']}/128  |  "
             f"correct + EOS {counts['correct_eos']}/128",
             ha="center", va="center", fontsize=10)

    kv.scatter(times, series["before_blocks"], s=7, marker="o",
               color="#235b86", alpha=.62, label="Before schedule", rasterized=False)
    kv.scatter(times, series["after_blocks"], s=7, marker="D",
               color="#bf6534", alpha=.62, label="After schedule", rasterized=False)
    kv.axhline(CAPACITY, color="#9d313a", ls="--", lw=1.2,
               clip_on=False, label="4,096 usable blocks")
    kv.set(ylabel="Used GPU KV blocks", ylim=(0, CAPACITY),
           title="Physical KV allocation at each native scheduler call")
    kv.set_yticks((0, 1024, 2048, 3072, CAPACITY))
    kv.legend(loc="upper right", ncol=3, fontsize=9, frameon=False)

    queue.scatter(times, series["before_running"], s=8, marker="o",
                  facecolors="none", edgecolors="#235b86", linewidths=.55,
                  alpha=.6, label="Running before", rasterized=False)
    queue.scatter(times, series["after_running"], s=8, marker="o",
                  color="#235b86", alpha=.6, label="Running after", rasterized=False)
    queue.scatter(times, series["before_waiting"], s=8, marker="s",
                  facecolors="none", edgecolors="#bf6534", linewidths=.55,
                  alpha=.6, label="Waiting before", rasterized=False)
    queue.scatter(times, series["after_waiting"], s=8, marker="s",
                  color="#bf6534", alpha=.6, label="Waiting after", rasterized=False)
    queue.set(xlabel="Recorded host_s since measurement origin (seconds)",
              ylabel="Requests", ylim=(0, REQUESTS),
              title="Running and waiting requests at the same call boundaries")
    queue.set_yticks((0, 32, 64, 96, REQUESTS))
    queue.legend(loc="upper right", ncol=2, fontsize=9, frameon=False)
    queue.set_xlim(0, times[-1])
    for axis in (kv, queue):
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    fig.text(.5, .035,
             f"Native preemption events: {counts['preemptions']}. "
             "Paired before/after values share a post-call host timestamp; "
             "markers are boundary snapshots, not continuous allocation peaks.",
             ha="center", va="center", fontsize=8.5, color="#444444")
    try:
        fig.savefig(png, dpi=180)
        fig.savefig(svg)
    finally:
        plt.close(fig)
    print(json.dumps({"png": str(png), "svg": str(svg), "scheduler_calls": len(times)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="native128 raw/native directory")
    parser.add_argument("--quality", required=True, type=Path,
                        help="C_INSTRUCT_NATIVE128_ANALYZE_V1 JSON")
    parser.add_argument("--capacity", required=True, type=Path,
                        help="C_INSTRUCT_NATIVE128_CAPACITY_V1 JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    series, counts = checked_data(args.run_dir, args.quality, args.capacity)
    plot(series, counts, args.output_dir)


if __name__ == "__main__":
    main()
