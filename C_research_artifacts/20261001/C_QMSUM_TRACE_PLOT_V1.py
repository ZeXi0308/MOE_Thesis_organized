#!/usr/bin/env python3
"""Plot actual QMSum baseline pressure, step costs and request gap distribution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(native, analysis_path, output):
    trace = json.loads((native / "measured-steps.json").read_text())
    mix = json.loads((native / "measured-service-mix.json").read_text())
    report = json.loads(analysis_path.read_text())
    calls, steps = trace["scheduler_calls"], trace["steps"]
    assert len(calls) == len(steps) == len(mix["scheduler_calls"]) == 927
    times = [r["return_s"] for r in steps]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), layout="constrained")
    axes[0].plot(times, [r["after"]["used_blocks"] for r in calls], lw=1.2,
                 label="Active KV blocks after schedule")
    axes[0].axhline(4096, color="black", ls="--", lw=.8, label="4096-block capacity")
    failed = sorted({r["schedule_call"] for r in trace["allocation_failures"]
                     if r["request_status"] == "WAITING"})
    axes[0].scatter([times[i] for i in failed], [calls[i]["after"]["used_blocks"]
                    for i in failed], s=6, c="#d99b19", label="Waiting-head allocation failure")
    labeled_preemption = False
    for r in calls:
        if r["preempted_request_ids"]:
            axes[0].axvline(times[r["call"]], color="#c44e52", alpha=.7, lw=.8,
                           label=None if labeled_preemption else "Two preemptions per marked step")
            labeled_preemption = True
    axes[0].set(ylabel="Active KV blocks", xlabel="Seconds from batch release")
    axes[0].legend(fontsize=8, loc="lower center")
    kinds = {"Mixed prefill/decode": ([], [], "#4c72b0"),
             "Decode only": ([], [], "#55a868"), "Prefill only": ([], [], "#c44e52")}
    for step, row in zip(steps, mix["scheduler_calls"]):
        key = ("Mixed prefill/decode" if row["prefill_tokens"] and row["decode_tokens"]
               else "Prefill only" if row["prefill_tokens"] else "Decode only")
        kinds[key][0].append(step["return_s"])
        kinds[key][1].append(1000*(step["return_s"]-step["start_s"]))
    for label, (x, y, color) in kinds.items():
        axes[1].scatter(x, y, s=6, alpha=.6, color=color, label=label)
    axes[1].set(ylabel="Host engine.step wall (ms)", xlabel="Seconds from batch release")
    axes[1].legend(fontsize=8, loc="upper right")
    gaps = sorted(1000*r["max_distinct_host_return_gap_s"] for r in report["per_request"])
    axes[2].step(gaps, [(i+1)/len(gaps) for i in range(len(gaps))], where="post")
    axes[2].scatter([gaps[-1]], [1], s=16, c="#c44e52")
    axes[2].annotate("Two preempted requests: 211 ms", (gaps[-1], 1),
                     xytext=(100, .75), arrowprops=dict(arrowstyle="->"), fontsize=9)
    axes[2].set(xlabel="Maximum distinct host token-return gap per request (ms)",
                ylabel="Request-weighted CDF", ylim=(0, 1.06), xlim=(0, 230))
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("QMSum200: one density baseline, all requests retained", fontsize=12)
    output.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / ("qmsum_native_pressure_v1."+suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "analysis", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    a = parser.parse_args()
    plot(a.native, a.analysis, a.output)
