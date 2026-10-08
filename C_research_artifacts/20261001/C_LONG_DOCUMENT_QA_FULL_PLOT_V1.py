#!/usr/bin/env python3
"""Plot completed LongBench150 native after-schedule snapshots and logged events."""
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
STEM = "long_document_qa_full_native_schedule_v1"


def read(path: Path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def plot(run: Path, quality_path: Path, runtime_path: Path, output_dir: Path) -> None:
    trace, digest = read(run / "measured-steps.json")
    quality, _ = read(quality_path)
    runtime, _ = read(runtime_path)
    calls = trace["scheduler_calls"]
    failures = trace["allocation_failures"]
    if (quality.get("status") != "COMPLETE" or quality.get("requests_completed") != 150
            or runtime.get("status") != "COMPLETE" or runtime.get("requests_completed") != 150
            or quality.get("original_run_sha256", {}).get("measured-steps.json") != digest
            or runtime.get("original_run_sha256", {}).get("measured-steps.json") != digest
            or len(calls) != runtime.get("schedule_calls")
            or len(failures) != runtime.get("allocation_failures_returning_none")):
        raise ValueError("full-run quality/runtime and original schedule trace differ")
    if (not calls or [c["call"] for c in calls] != list(range(len(calls)))
            or any(c["host_s"] < 0 or not math.isfinite(c["host_s"]) for c in calls)):
        raise ValueError("native schedule call times differ")
    times = [c["host_s"] for c in calls]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("native schedule host time moved backward")
    blocks = [c["after"]["used_blocks"] for c in calls]
    running = [c["after"]["running"] for c in calls]
    waiting = [c["after"]["waiting"] for c in calls]
    if (not all(0 <= n <= CAPACITY for n in blocks)
            or max(blocks) != runtime["peak_used_blocks_after_schedule"]
            or max(running) != runtime["peak_running_after_schedule"]
            or max(waiting) != runtime["peak_waiting_after_schedule"]):
        raise ValueError("after-schedule peaks differ from runtime summary")
    preempt_times = [c["host_s"] for c in calls for _ in c["preempted_request_ids"]]
    if len(preempt_times) != runtime["preemption_events"]:
        raise ValueError("preemption events differ from runtime summary")
    failure_times = [event["host_s"] for event in failures]
    if any(not 0 <= t <= runtime["generation_observation_s"] for t in failure_times):
        raise ValueError("allocation failure event time outside observation")
    png, svg = (output_dir / f"{STEM}.{ext}" for ext in ("png", "svg"))
    if any(p.exists() or p.is_symlink() for p in (png, svg)):
        raise FileExistsError("figure output already exists")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8.5),
                             gridspec_kw={"height_ratios": [1.25, 1.1, .65]})
    fig.subplots_adjust(left=.1, right=.97, top=.87, bottom=.12, hspace=.24)
    fig.suptitle("LongBench MultiFieldQA-en: native source-order 150", y=.97,
                 fontsize=15, fontweight="semibold")
    fig.text(.5, .92, "Complete output/EOS observation · after-schedule host snapshots · no policy comparison",
             ha="center", fontsize=9)
    ax = axes[0]
    ax.plot(times, blocks, color="#236389", lw=.75, alpha=.65)
    ax.scatter(times, blocks, color="#236389", s=5, label="After-schedule snapshots")
    ax.axhline(CAPACITY, color="#aa4146", ls="--", lw=1, label="4,096 usable blocks")
    ax.set(ylabel="Used GPU KV blocks", ylim=(0, CAPACITY * 1.03))
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax = axes[1]
    ax.plot(times, running, color="#285f9e", lw=.8, marker=".", ms=2, label="Running")
    ax.plot(times, waiting, color="#c27a36", lw=.8, marker=".", ms=2, label="Waiting")
    ax.set(ylabel="Requests", ylim=(0, max(150, max(running + waiting)) * 1.03))
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax = axes[2]
    if failure_times:
        ax.scatter(failure_times, [1] * len(failure_times), marker="|", s=85,
                   color="#aa4146", label=f"allocate_slots returned None: {len(failure_times)}")
    if preempt_times:
        ax.scatter(preempt_times, [0] * len(preempt_times), marker="|", s=85,
                   color="#6a4c93", label=f"Native preemption: {len(preempt_times)}")
    if not failure_times and not preempt_times:
        ax.text(.5, .5, "No logged allocation None or preemption events",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
    ax.set(yticks=[0, 1], yticklabels=["Preempt", "Allocate None"], ylim=(-.6, 1.6),
           xlabel="Host seconds since measured episode origin")
    if failure_times or preempt_times:
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=.17)
        ax.set_xlim(0, runtime["generation_observation_s"])
    fig.text(.5, .035, "Markers are instantaneous host observations; connecting lines guide the eye. "
             "Waiting is not attributed to KV capacity.", ha="center", fontsize=8.5)
    try:
        fig.savefig(png, dpi=180)
        fig.savefig(svg)
    finally:
        plt.close(fig)
    print(json.dumps(dict(png=str(png), svg=str(svg), schedule_calls=len(calls))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--quality", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    plot(a.run_dir, a.quality, a.runtime, a.output_dir)


if __name__ == "__main__":
    main()
