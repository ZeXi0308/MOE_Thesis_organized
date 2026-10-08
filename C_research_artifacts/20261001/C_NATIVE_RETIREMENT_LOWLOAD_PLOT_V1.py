#!/usr/bin/env python3
"""Plot all four executions of the fixed fivefold slower-arrival control."""
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
assert x["status"] == "COMPLETE_FIXED_LOWER_RATE_CONTROL"
order = ["native_1", "retirement_1", "retirement_2", "native_2"]
assert x["order"] == order
cells = x["cells"]
assert [c["label"] for c in cells] == order
labels = ["Native 1", "Retirement 1", "Retirement 2", "Native 2"]
colors = ["#bc772e", "#126e89", "#126e89", "#bc772e"]
metrics = [c["metrics"] for c in cells]
assert all(m["makespan_s"] > 0 and m["native_preemptions"] is not None for m in metrics)
fig, axes = plt.subplot_mosaic(
    [["goodput", "flow", "curve"], ["output", "preempt", "curve"]],
    figsize=(14, 7.5), layout="constrained",
    gridspec_kw={"width_ratios": [1, 1, 1.65]})
panels = [
    ("goodput", [m["goodput_20_4_requests_s"] for m in metrics],
     "Fixed 20 s / 4 s target", "Qualified requests / episode second", ".3f"),
    ("flow", [m["mean_completed_flow_s"] for m in metrics],
     "Complete-request cost", "Mean arrival-to-completion time (s)", ".3f"),
    ("output", [m["output_tokens"] / m["makespan_s"] for m in metrics],
     "Actual output rate", "Output tokens / episode second", ".1f"),
    ("preempt", [m["native_preemptions"] for m in metrics],
     "Native preemptions", "Recorded preemption count", ".0f"),
]
for name, values, title, ylabel, fmt in panels:
    ax = axes[name]
    bars = ax.bar(range(4), values, color=colors, width=.68)
    ax.bar_label(bars, labels=[format(v, fmt) for v in values], padding=4, fontsize=9)
    ax.set_xticks(range(4), labels, rotation=32, ha="right")
    ax.set(title=title, ylabel=ylabel, ylim=(0, max(1e-6, max(values)) * 1.18))
    if name == "preempt":
        ax.set_ylim(0, max(1, max(values) * 1.18))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
ax = axes["curve"]
deadlines_by_cell = [sorted(r["ttft_s"] for r in c["all_request_metrics"]
    if r["ttft_s"] is not None and (r["max_gap_s"] is None or r["max_gap_s"] <= 4))
    for c in cells]
upper = max(22, max((max(ds, default=0) for ds in deadlines_by_cell), default=0) + 2)
for i, (c, deadlines, label, color) in enumerate(zip(cells, deadlines_by_cell, labels, colors)):
    duration = c["metrics"]["makespan_s"]
    ax.step([0] + deadlines + [upper],
        [0] + [(j + 1) / duration for j in range(len(deadlines))] + [len(deadlines) / duration],
        where="post", color=color, linestyle="-" if i < 2 else "--", lw=1.5, label=label)
    ax.scatter([20], [c["metrics"]["goodput_20_4_requests_s"]], color=color, s=24, zorder=4)
ax.axvline(20, color="#555555", linestyle=":", lw=1)
ax.set(title="Descriptive TTFT deadline curves (gap ≤ 4 s)", xlabel="TTFT deadline (s)",
       ylabel="Qualified requests / episode second", xlim=(0, upper), ylim=(0, None))
ax.grid(alpha=.18)
ax.legend(frameon=False, fontsize=9, loc="lower right")
zoom_upper = max((max(ds, default=0) for ds in deadlines_by_cell), default=0) * 1.05
if 0 < zoom_upper < 1:
    zoom = ax.inset_axes([.16, .34, .75, .43])
    for i, (c, deadlines, color) in enumerate(zip(cells, deadlines_by_cell, colors)):
        duration = c["metrics"]["makespan_s"]
        zoom.step([0] + deadlines + [zoom_upper],
            [0] + [(j + 1) / duration for j in range(len(deadlines))] + [len(deadlines) / duration],
            where="post", color=color, linestyle="-" if i < 2 else "--", lw=1.3)
    zoom.set(title="Observed TTFT range (descriptive zoom)", xlabel="TTFT deadline (s)",
             ylabel="Qualified requests / episode s", xlim=(0, zoom_upper), ylim=(0, None))
    zoom.tick_params(labelsize=8)
    zoom.title.set_fontsize(9)
    zoom.xaxis.label.set_fontsize(8)
    zoom.yaxis.label.set_fontsize(8)
    zoom.grid(alpha=.18)
fig.suptitle("Fixed 5× slower arrivals: four executions on the same viewed cohort", fontsize=14)
fig.supxlabel("All requests retained through drain; rates include the full episode. Primary remains 20 s / 4 s.\n"
              "Two runs per arm; actual pressure must be checked. Outputs may differ; no holdout or equal-work claim.", fontsize=9)
for suffix in (".png", ".svg"):
    fig.savefig(args.output_prefix.with_suffix(suffix), dpi=170)
