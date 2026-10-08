#!/usr/bin/env python3
"""Four-panel descriptive plot of completed LongBench fixed-grain cells."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {"dfs": "DFS", "whole": "whole", "1": "q1", "2": "q2"}
COLORS = {"dfs": "#49677e", "whole": "#c08437", "1": "#569486", "2": "#8964a8"}
PANELS = (
    ("Mean completion", "mean_completion_s", 1, "s", 3),
    ("p95 completion", "p95_completion_s", 1, "s", 3),
    ("Worst per-request host gap", "max_distinct_host_return_gap_s", 1000, "ms", 1),
    ("Deduced prompt work", "deduced_prompt_scheduled_tokens", .001, "k tokens", 1),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    names = data.get("execution_order")
    if (data.get("schema") != "c-longqa-grain-analysis-v1"
            or data.get("status") != "COMPLETE"
            or not isinstance(names, list) or not 2 <= len(names) <= 4
            or len(names) != len(set(names)) or any(name not in LABELS for name in names)
            or set(data.get("arms", {})) != set(names)):
        raise ValueError("requires one complete actual grain-analysis summary")
    arms = data["arms"]
    for name in names:
        for _, field, _, _, _ in PANELS:
            value = arms[name].get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"missing actual {field} for {name}")
        for field in ("output_tokens", "extra_shape_warmup_wall_s"):
            value = arms[name].get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"missing actual {field} for {name}")

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), constrained_layout=False)
    fig.suptitle("LongBench150  |  enqueue order development cells", fontsize=15, weight="bold", y=.965)
    xs = range(len(names))
    for ax, (title, field, scale, unit, places) in zip(axes.flat, PANELS):
        values = [arms[name][field] * scale for name in names]
        bars = ax.bar(xs, values, width=.58, color=[COLORS[name] for name in names])
        ax.set_title(title, loc="left", fontsize=11, weight="semibold")
        ax.set_ylabel(unit)
        ax.set_xticks(list(xs), [LABELS[name] for name in names])
        ax.set_ylim(0, max(values) * 1.20 if max(values) else 1)
        ax.grid(axis="y", color="#dce3e8", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:,.{places}f}",
                        (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    outputs = "  ·  ".join(f"{LABELS[name]} {arms[name]['output_tokens']:,}"
                           for name in names)
    warmup = "  ·  ".join(f"{LABELS[name]} {arms[name]['extra_shape_warmup_wall_s']:.3f}s"
                          for name in names)
    fig.text(.08, .087, "One completed viewed development cell per arm; no uncertainty bars or service SLO.",
             fontsize=9, color="#49545d")
    fig.text(.08, .063, "Actual output tokens: " + outputs, fontsize=9, color="#49545d")
    fig.text(.08, .039, "Extra shape warmup outside measured service: " + warmup,
             fontsize=9, color="#49545d")
    fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.18, wspace=.25, hspace=.34)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = prefix.with_suffix("." + ext)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180 if ext == "png" else None, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
