#!/usr/bin/env python3
"""Plot every execution in the frozen fresh native/FIFO/retirement mirrored block."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--analysis", required=True, type=Path)
parser.add_argument("--output-prefix", required=True, type=Path)
args = parser.parse_args()
x = json.loads(args.analysis.read_text())
assert x["status"] == "COMPLETE_FROZEN_FRESH_COHORT_BLOCK"
cells = x["cells"]
labels = ["Native 1", "FIFO 1", "Retirement 1", "Retirement 2", "FIFO 2", "Native 2"]
colors = ["#bc772e", "#777d86", "#126e89", "#126e89", "#777d86", "#bc772e"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), layout="constrained",
                         gridspec_kw={"width_ratios": [1, 1, 1.65]})
for ax, key, title, ylabel in zip(axes[:2],
    ["goodput_20_4_requests_s", "mean_completed_flow_s"],
    ["Fixed 20 s / 4 s target", "Complete-request cost"],
    ["Qualified requests / episode second", "Mean arrival-to-completion time (s)"]):
    values = [c["metrics"][key] for c in cells]
    bars = ax.bar(range(6), values, color=colors, width=.68)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=9)
    ax.set_xticks(range(6), labels, rotation=32, ha="right")
    ax.set(title=title, ylabel=ylabel, ylim=(0, max(values)*1.16))
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
ax = axes[2]
for i, (c, label, color) in enumerate(zip(cells, labels, colors)):
    deadlines = sorted(r["ttft_s"] for r in c["all_request_metrics"]
                       if r["ttft_s"] is not None and (r["max_gap_s"] is None or r["max_gap_s"] <= 4))
    duration = c["metrics"]["makespan_s"]
    upper = max(65, max(deadlines, default=0)) + 2
    ax.step([0]+deadlines+[upper], [0]+[(j+1)/duration for j in range(len(deadlines))]+[len(deadlines)/duration],
        where="post", color=color, linestyle="-" if i < 3 else "--", lw=1.5, label=label)
    ax.scatter([20], [c["metrics"]["goodput_20_4_requests_s"]], color=color, s=24, zorder=4)
ax.axvline(20, color="#555555", linestyle=":", lw=1)
ax.set(title="Descriptive TTFT deadline curves (gap ≤ 4 s)", xlabel="TTFT deadline (s)",
       ylabel="Qualified requests / episode second", xlim=(0, 70), ylim=(0, None))
ax.grid(alpha=.18)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.suptitle("Fresh cohort: six consecutive executions; frozen policies and input", fontsize=14)
fig.supxlabel("All requests retained through drain. Primary remains 20 s / 4 s; curves are descriptive.\nOne fresh cohort; two runs per arm on the same replacement GPU; outputs may differ. No equal-work claim.", fontsize=9)
for suffix in (".png", ".svg"):
    fig.savefig(args.output_prefix.with_suffix(suffix), dpi=170)
