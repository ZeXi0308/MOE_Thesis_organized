#!/usr/bin/env python3
"""Plot the four deterministic-cap / sampled-AE peak ablation executions."""
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
assert x["status"].startswith("COMPLETE")
order = x["order"]
assert len(order) == 4
cells = x["cells"]
assert [c["label"] for c in cells] == order
assert [c["arm"] for c in cells] == ["native_past_future_cap", "native_past_future_ae", "native_past_future_ae", "native_past_future_cap"]
labels = ["Full-cap 1", "Past-Future 1", "Past-Future 2", "Full-cap 2"]
colors = ["#bc772e", "#126e89", "#126e89", "#bc772e"]
metrics = [c["metrics"] for c in cells]
assert all(m["makespan_s"] > 0 and m["native_preemptions"] is not None for m in metrics)
fig, axes = plt.subplot_mosaic(
    [["goodput", "flow", "curve"], ["output", "delta", "curve"]],
    figsize=(14, 7.5), layout="constrained",
    gridspec_kw={"width_ratios": [1, 1, 1.65]})
panels = [
    ("goodput", [m["goodput_20_4_requests_s"] for m in metrics],
     "Fixed 20 s / 4 s target", "Qualified requests / episode second", ".3f"),
    ("flow", [m["mean_completed_flow_s"] for m in metrics],
     "Complete-request cost", "Mean arrival-to-completion time (s)", ".3f"),
    ("output", [m["output_tokens"] / m["makespan_s"] for m in metrics],
     "Actual output rate", "Output tokens / episode second", ".1f"),
]
for name, values, title, ylabel, fmt in panels:
    ax = axes[name]
    bars = ax.bar(range(4), values, color=colors, width=.68)
    ax.bar_label(bars, labels=[format(v, fmt) for v in values], padding=4, fontsize=9)
    ax.set_xticks(range(4), labels, rotation=32, ha="right")
    ax.set(title=title, ylabel=ylabel, ylim=(0, max(1e-6, max(values)) * 1.18))
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
ax = axes["delta"]
for i, pair in enumerate(x["cap_to_past_future_adjacent_pairs"]):
    deltas = sorted(r["flow_difference_s"] for r in pair["per_request_deltas"])
    higher = sum(d > 0 for d in deltas)
    ax.plot(range(1, len(deltas) + 1), deltas,
            color=["#126e89", "#934d91"][i], linestyle=["-", "--"][i],
            label=f"Pair {i + 1}: {higher}/128 slower")
ax.axhline(0, color="#555555", linewidth=.8)
ax.set(title="Most requests finish later", xlabel="Request rank (sorted within each pair)",
       ylabel="Flow difference: PF − full-cap (s)", xlim=(1, 128))
ax.grid(alpha=.18)
ax.legend(frameon=False, fontsize=8, loc="lower right")
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
fig.suptitle("AE peak admission: full-cap vs history sampling (same 5% reserve)", fontsize=14)
fig.supxlabel("All requests retained through drain; primary remains 20 s / 4 s; zero preemptions in all four runs.\n"
              "Two runs per arm on one viewed cohort. Outputs differ; no full LightLLM or equal-work claim.", fontsize=9)
for suffix in (".png", ".svg"):
    fig.savefig(args.output_prefix.with_suffix(suffix), dpi=170)
