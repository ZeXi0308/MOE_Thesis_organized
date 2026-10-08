#!/usr/bin/env python3
"""Plot one QMSum200 native 512/1024 development pair from actual analyses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"512": "#2364AA", "1024": "#DB6A2B"}


def read(path: Path):
    return json.loads(path.read_text())


def ecdf(ax, values, *, label, color):
    ordered = np.sort(np.asarray(values, dtype=float))
    ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
            where="post", label=label, color=color, linewidth=2.2)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.grid(alpha=.20, linewidth=.7)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small-analysis", type=Path, required=True)
    ap.add_argument("--large-analysis", type=Path, required=True)
    ap.add_argument("--actions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    data = {"512": read(args.small_analysis), "1024": read(args.large_analysis)}
    actions = read(args.actions)
    if actions.get("schema") != "c-qmsum-native-chunk-actions-v1":
        raise ValueError("chunk action analysis schema differs")
    for label in ("512", "1024"):
        report = data[label]
        rows = report.get("per_request", [])
        if (report.get("status") != "COMPLETE" or len(rows) != 200
                or {r["source_index"] for r in rows} != set(range(200))
                or report["original_run_sha256"]["measured-outputs.json"]
                   != actions["raw_sha256"][label]["measured-outputs.json"]
                or report["service_mix"]["schedule_call_types"]
                   != actions["arms"][label]["schedule_type_counts"]):
            raise ValueError(f"{label} analysis/action pair differs")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.55),
                             gridspec_kw={"width_ratios": [1.12, 1.12, .90]})
    fig.subplots_adjust(left=.065, right=.985, top=.78, bottom=.22, wspace=.34)
    for label in ("512", "1024"):
        report = data[label]
        completion = [r["flow_s"] for r in report["per_request"]]
        gap_ms = [1000 * r["max_distinct_host_return_gap_s"]
                  for r in report["per_request"]]
        ecdf(axes[0], completion,
             label=f"{label} · mean {report['flow_s']['mean']:.2f} s",
             color=COLORS[label])
        ecdf(axes[1], gap_ms,
             label=f"{label} · p95 {1000*report['max_distinct_host_return_gap_s']['p95_nearest_rank']:.1f} ms",
             color=COLORS[label])
    axes[0].set(title="A  Request completion", xlabel="Arrival to completion (s)",
                ylabel="Fraction of all 200 requests")
    axes[1].set(title="B  Largest host return gap per request",
                xlabel="Gap between distinct token returns (ms, log scale)")
    axes[1].set_xscale("log")
    axes[1].set_xlim(15, 210)
    axes[1].set_xticks([20, 30, 50, 100, 200], labels=["20", "30", "50", "100", "200"])
    for ax in axes[:2]:
        ax.legend(loc="lower right", frameon=False, fontsize=8.8)

    kinds = ["Prefill only", "Mixed", "Decode only"]
    keys = ["prefill_only", "mixed", "decode_only"]
    x = np.arange(len(kinds))
    for offset, label in ((-.19, "512"), (.19, "1024")):
        counts = actions["arms"][label]["schedule_type_counts"]
        heights = [counts.get(key, 0) for key in keys]
        bars = axes[2].bar(x + offset, heights, width=.34, color=COLORS[label],
                           label=label)
        axes[2].bar_label(bars, padding=2, fontsize=8)
    axes[2].set(title="C  Actual service steps", ylabel="Schedule calls",
                xticks=x, xticklabels=["Prefill\nonly", "Mixed", "Decode\nonly"])
    axes[2].grid(axis="y", alpha=.20, linewidth=.7)
    axes[2].set_axisbelow(True)
    axes[2].set_ylim(0, max(actions["arms"][label]["schedule_type_counts"].get(key, 0)
                            for label in ("512", "1024") for key in keys) * 1.16)
    axes[2].legend(frameon=False, loc="upper right", fontsize=8.8)

    fig.suptitle("QMSum native batch budget: 512 vs 1024 tokens", x=.065, ha="left",
                 fontsize=15, fontweight="bold", y=.965)
    fig.text(.065, .875,
             "Single same-host development pair · fixed 200 prompts, density order, 128 sequences, 4096 KV blocks",
             fontsize=9.5, color="#404040")
    fig.text(.065, .055,
             f"Generated token IDs differ for {actions['output_ids_changed']}/200 requests. "
             "Host timings include scheduler and observer work; curves describe this pair only.",
             fontsize=8.7, color="#555555")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = args.output_dir / "qmsum_chunk_development_v1"
    fig.savefig(stem.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(json.dumps({"png": str(stem.with_suffix(".png")),
                      "svg": str(stem.with_suffix(".svg"))}))


if __name__ == "__main__":
    main()
