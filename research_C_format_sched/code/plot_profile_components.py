#!/usr/bin/env python3
"""Plot existing profiler component sums; never reprofile or infer uncertainty.

The sole numerical input is runs/profile_component_analysis_r01/summary.json.
Each bar is one instrumented CUDA-graph replay, not an unprofiled timing median.
Component groups preserve whole-kernel accounting, including different fused
work in the FP8/Triton and NVFP4/CUTLASS projections. Summed durations are not
wall-clock latency or a causal estimate of a primitive/fusion's cost.

  /private/tmp/format_sched_plot_venv/bin/python code/plot_profile_components.py
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
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


ROOT = Path(__file__).resolve().parents[1]
BATCHES = (1, 8, 2048, 8192)
ROUTING = ("uniform", "zipf")
FORMATS = ("fp8", "nvfp4")
GROUPS = (
    ("Projection", ("projection_kernels",), "#426B9A"),
    ("Activation + quant.", ("activation_and_quantization",), "#E5A34A"),
    ("Row shuffle", ("row_shuffles",), "#8A78A8"),
    ("Output", ("output_operations",), "#4B9B8F"),
    ("Setup + other", ("routing_metadata", "grouped_gemm_setup",
                       "unattributed_copy_kernel", "unclassified"), "#ADB2B8"),
)


def read_components(path):
    data = json.loads(path.read_text())
    if not data["duration_unit"].startswith("microseconds"):
        raise ValueError("expected explicit microsecond trace durations")
    selected = {}
    expected_keys = {key for _, keys, _ in GROUPS for key in keys}
    for trace in data["traces"]:
        key = trace["routing"], trace["tokens"], trace["format"]
        if key[0] not in ROUTING or key[1] not in BATCHES or key[2] not in FORMATS:
            continue
        if key in selected or trace["status"] != "ok" or trace["timing_mode"] != "cuda_graph_replay":
            raise ValueError(f"duplicate, failed, or unexpected timing-mode trace: {key}")
        if set(trace["components"]) != expected_keys:
            raise ValueError(f"component schema changed; cannot silently omit kernel groups: {key}")
        values = [sum(trace["components"][name]["duration_sum_us"] for name in keys)
                  for _, keys, _ in GROUPS]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"invalid component time: {key}")
        total = trace["profiled_kernel_duration_sum_us"]
        if not math.isclose(sum(values), total, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError(f"component stack does not sum to the recorded kernel total: {key}")
        selected[key] = {"values_us": values, "total_us": total,
                         "weight_source": trace["weight_source"], "trace_sha256": trace["trace_sha256"]}
    expected = {(routing, batch, fmt) for routing in ROUTING for batch in BATCHES for fmt in FORMATS}
    if set(selected) != expected:
        raise ValueError(f"missing required comparisons: {sorted(expected - set(selected))}")
    return selected


def plot(source, output):
    selected = read_components(source)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .6, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 4.6), sharey=False)
    fig.subplots_adjust(left=.085, right=.985, bottom=.235, top=.82, wspace=.40, hspace=.55)
    for row, routing in enumerate(ROUTING):
        for col, batch in enumerate(BATCHES):
            ax = axes[row, col]
            records = [selected[(routing, batch, fmt)] for fmt in FORMATS]
            upper = max(record["total_us"] for record in records)
            for x, record in enumerate(records):
                bottom = 0.
                for value, (_, _, color) in zip(record["values_us"], GROUPS):
                    ax.bar(x, value, bottom=bottom, width=.60, color=color,
                           edgecolor="white", linewidth=.35, zorder=3)
                    bottom += value
                ax.text(x, bottom + upper * .035, f"{record['total_us']:.1f}",
                        ha="center", va="bottom", fontsize=7.8, color="#20252B")
            ax.set_ylim(0, upper * 1.24)
            ax.set_xlim(-.6, 1.6)
            ax.set_xticks((0, 1), ("FP8", "NVFP4"))
            ax.set_title(f"$B={batch}$", pad=7)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax.grid(axis="y", color="#DDE1E5", linewidth=.5, zorder=0)
            ax.tick_params(axis="both", length=2.5, width=.5, pad=2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#80878E")
            if col == 0:
                ax.set_ylabel(("Uniform" if routing == "uniform" else "Zipf") + "\nKernel sum ($\\mu$s)", labelpad=5)
    fig.suptitle("Qwen3-shape components: one instrumented graph replay", y=.985, fontsize=10.4)
    fig.legend(handles=[Patch(facecolor=color, label=label) for label, _, color in GROUPS],
               loc="upper center", bbox_to_anchor=(.52, .94), ncol=5, frameon=False,
               fontsize=7.5, handlelength=1.3, handletextpad=.4, columnspacing=1.05)
    notes = (
        "Independent linear y axes; numbers are summed kernel durations in microseconds. No error bars (one replay).",
        "FP8: synthetic weights / Triton. NVFP4: calibrated layer 0 / CUTLASS. Fused-work boundaries differ.",
        "Setup + other includes routing metadata, grouped-GEMM setup, and remaining copy kernels.",
        "Source: runs/profile_component_analysis_r01/summary.json",
    )
    for y, note in zip((.157, .117, .077, .037), notes):
        fig.text(.085, y, note, fontsize=6.8, ha="left", va="center", color="#3C434A")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    fig.savefig(output.with_suffix(".pdf"), metadata={
        "Title": "Qwen3 instrumented MoE component durations",
        "Subject": "Single instrumented replay; synthetic FP8 vs calibrated NVFP4; source SHA256 " + source_hash,
        "Creator": "plot_profile_components.py"})
    fig.savefig(output.with_suffix(".png"), dpi=250)
    plt.close(fig)
    print(json.dumps({"source": str(source), "source_sha256": source_hash,
                      "bars": len(selected), "component_sums_verified": True,
                      "outputs": [str(output.with_suffix(ext)) for ext in (".pdf", ".png")]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/profile_component_analysis_r01/summary.json")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "figures/F1b_qwen_components")
    args = parser.parse_args()
    plot(args.source.resolve(), args.output_prefix.resolve())


if __name__ == "__main__":
    main()
