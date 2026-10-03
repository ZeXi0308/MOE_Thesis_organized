#!/usr/bin/env python3
"""Plot two actual lease blocks separately and save their compact comparison.

No pooled metric, confidence interval or within-run independent replication.
"""
import argparse
import hashlib
import json
from pathlib import Path

from plot_newhost_results import ROOT, LEASE_ARMS, LABELS, COLORS, METRICS, read_group
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


def load_block(path, name):
    rows, identity = read_group(path, LEASE_ARMS)
    source = json.loads(path.read_text())
    by_arm = dict(zip(LEASE_ARMS, rows))
    fixed, adaptive = by_arm["fixed4"], by_arm["adaptive"]
    deltas = {}
    for key, *_ in METRICS:
        ratio = adaptive[key] / fixed[key]
        deltas[key] = {"adaptive_minus_fixed4": adaptive[key] - fixed[key],
                       "adaptive_over_fixed4": ratio, "change_pct": (ratio - 1) * 100}
    a = {r["request_id"]: r for r in adaptive["requests"]}
    f = {r["request_id"]: r for r in fixed["requests"]}
    pairs = {"max_gap_s": {}, "actual_completion_flow_s": {}, "ttft_s": {}}
    for field in pairs:
        values = [a[rid][field] - f[rid][field] for rid in f]
        pairs[field] = {"adaptive_higher": sum(v > 0 for v in values),
                        "adaptive_lower": sum(v < 0 for v in values),
                        "equal": sum(v == 0 for v in values)}
    comparison = source["comparisons"]["adaptive_vs_fixed4"]
    changed_sequences = set(comparison["sequence_differences"])
    worst = {}
    for field in ("actual_completion_flow_s", "ttft_s"):
        worst[field] = [{"request_id": rid, "adaptive_minus_fixed4_s": a[rid][field] - f[rid][field],
                         "adaptive_s": a[rid][field], "fixed4_s": f[rid][field],
                         "adaptive_output_tokens": a[rid]["outputs"], "fixed4_output_tokens": f[rid]["outputs"],
                         "output_sequences_differ": rid in changed_sequences}
                        for rid in sorted(f, key=lambda rid: a[rid][field] - f[rid][field], reverse=True)[:3]]
    frontier = comparison["frontier"]
    grid = {"threshold_count": len(frontier),
            "ttft_deadlines_s": sorted({r["ttft_deadline_s"] for r in frontier}),
            "gap_deadlines_s": sorted({r["gap_deadline_s"] for r in frontier}),
            "scope": "Correlated threshold views of the same runs, not independent replicates."}
    for field in ("goodput_difference_requests_s", "qualifying_difference"):
        values = [r[field] for r in frontier]
        grid[field] = {"adaptive_higher_thresholds": sum(v > 0 for v in values),
                       "adaptive_lower_thresholds": sum(v < 0 for v in values),
                       "equal_thresholds": sum(v == 0 for v in values),
                       "minimum_difference": min(values), "maximum_difference": max(values)}
    compact = {"source_file": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "execution_order": list(source["arms"]),
               "arms": {arm: {key: values[key] for key in (
                   "expected_requests", "completed", "failed", "unfinished", "total_output_tokens",
                   "max_gap_request_p95_s", "actual_output_tokens_s", "mean_flow_with_incomplete_penalty_s")}
                   for arm, values in by_arm.items()},
               "adaptive_vs_fixed4": {"metrics": deltas, "paired_request_direction_counts": pairs,
                   "paired_scope": "All128 request IDs and inputs match; output sequences may differ. Counts describe these trajectories, not equal-work causal wins/losses.",
                   "largest_three_adaptive_regressions_by_metric": worst,
                   "goodput_grid_summary": grid,
                   "sequence_difference_requests": len(comparison["sequence_differences"]),
                   "stop_difference_requests": len(comparison["stop_differences"]),
                   "service_criteria": comparison["exploratory_tradeoff"]}}
    return by_arm, compact, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r01", type=Path, default=ROOT / "output/LEASE_SERVICE_VS_FIXED4_R01_20261002.json")
    parser.add_argument("--r02", type=Path, default=ROOT / "output/LEASE_SERVICE_VS_FIXED4_R02_20261002.json")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "output/lease_repeat_comparison")
    args = parser.parse_args()
    rows = {}; blocks = {}; identity = None
    for name, path in (("R01", args.r01), ("R02", args.r02)):
        rows[name], blocks[name], current = load_block(path, name)
        if identity is not None and current != identity:
            raise ValueError("Input cohort differs between repeat blocks")
        identity = current
    changes = {block: [blocks[block]["adaptive_vs_fixed4"]["metrics"][key]["change_pct"]
                       for key, *_ in METRICS] for block in blocks}
    summary = {"schema": "lease-reverse-order-comparison-v1", "blocks": blocks,
        "across_block_direction_checks": {
            "adaptive_vs_fixed4_p95_gap_sign_flips": changes["R01"][0] * changes["R02"][0] < 0,
            "adaptive_vs_fixed4_output_rate_sign_flips": changes["R01"][1] * changes["R02"][1] < 0,
            "adaptive_mean_flow_higher_in_both": all(v[2] > 0 for v in changes.values())},
        "scope": "Two separate same-host blocks, each with one run per arm. Descriptive ID-paired request directions; no pooling or confidence intervals. Requests/events are not independent run replicates. Different output trajectories preclude equal-generated-work interpretation. A reversed ordering does not by itself identify an order effect versus run noise."}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#B5BDC3",
        "text.color": "#25343D", "axes.labelcolor": "#25343D", "xtick.color": "#34434D",
        "ytick.color": "#34434D", "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.3))
    fig.subplots_adjust(left=.065, right=.992, top=.68, bottom=.28, wspace=.36)
    fig.suptitle("Lease ranking changes across the two run blocks", x=.065, y=.973, ha="left", fontsize=13, fontweight="semibold")
    fig.text(.065, .903, f"Same base-model article inputs · {len(identity)}/{len(identity)} complete in every arm · one run per arm per block · RTX 5090",
             fontsize=9, color="#5F6D76")
    fig.legend(handles=[Patch(facecolor=color, label=LABELS[arm]) for arm, color in zip(LEASE_ARMS, COLORS)],
               loc="upper left", bbox_to_anchor=(.056, .865), ncol=3, frameon=False, fontsize=9, handlelength=1.3, columnspacing=2)
    for column, (key, title, ylabel, direction, number_format) in enumerate(METRICS):
        ax = axes[column]
        all_values = [rows[block][arm][key] for block in ("R01", "R02") for arm in LEASE_ARMS]
        peak = max(all_values)
        for block_index, block in enumerate(("R01", "R02")):
            for arm_index, (arm, color) in enumerate(zip(LEASE_ARMS, COLORS)):
                x = block_index + (arm_index - 1) * .28
                value = rows[block][arm][key]
                ax.bar(x, value, width=.215, color=color, zorder=3)
                display = format(value, ".0f" if column == 1 else number_format)
                ax.text(x, value + peak * .035, display, ha="center", va="bottom", fontsize=7.2)
        ax.set(xlim=(-.55, 1.55), ylim=(0, peak * 1.22), ylabel=ylabel)
        ax.set_xticks([0, 1], ["R01\nOriginal order", "R02\nReversed order"])
        ax.tick_params(axis="x", length=0, pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}" if column == 1 else "{x:g}"))
        ax.grid(axis="y", color="#E6EAED", linewidth=.65)
        ax.set_axisbelow(True)
        ax.set_title(f"({chr(ord('a') + column)}) {title}", loc="left", pad=22, fontweight="semibold")
        ax.text(0, 1.04, direction, transform=ax.transAxes, color="#67757D", fontsize=8)
    a, b = changes["R01"], changes["R02"]
    footer = (f"Adaptive vs fixed Q4: P95 gap {a[0]:+.1f}% → {b[0]:+.1f}%; "
              f"output rate {a[1]:+.1f}% → {b[1]:+.1f}%; mean flow {a[2]:+.2f}% → {b[2]:+.2f}%.\n"
              "R01 order: Q1 → fixed Q4 → adaptive. R02 order: adaptive → fixed Q4 → Q1. Each block is shown separately.\n"
              "No pooled estimate or confidence intervals. Output trajectories differ; two blocks do not distinguish ordering effects from run noise.")
    fig.text(.065, .036, footer, fontsize=7.8, color="#5F6D76", linespacing=1.6)
    metadata = "; ".join(f"{name}: {block['source_sha256']}" for name, block in blocks.items())
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=240)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), metadata={"Title": "Lease reverse-order comparison", "Subject": metadata})
    plt.close(fig)
    print(json.dumps({"adaptive_vs_fixed4_change_pct": changes, **summary["across_block_direction_checks"]}))
    print(str(args.output_prefix) + ".{png,pdf,json}")


if __name__ == "__main__":
    main()
