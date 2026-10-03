#!/usr/bin/env python3
"""Actual request distributions and descriptive ID-paired completion deltas.

All requests are retained. No resampling, confidence intervals, causal effects,
or independence of within-run requests/events are assumed.
"""
import argparse
import hashlib
from pathlib import Path

from plot_newhost_results import ROOT, STRONG_ARMS, LABELS, COLORS, read_group
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "output/STRONG_NEWHOST_RESULT_R01_20261002.json")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "output/strong_full_cohort_distribution")
    args = parser.parse_args()
    groups, identity = read_group(args.result, STRONG_ARMS)
    requests = {arm: {r["request_id"]: r for r in metrics["requests"]}
                for arm, metrics in zip(STRONG_ARMS, groups)}
    ordinary, fund = requests["ordinary"], requests["queue_fund"]
    n = len(identity)
    delta = np.sort([fund[rid]["actual_completion_flow_s"] - ordinary[rid]["actual_completion_flow_s"]
                     for rid in ordinary])
    gaps = {arm: np.sort([r["max_gap_s"] for r in rows.values()]) for arm, rows in requests.items()}
    if not all(len(values) == n and np.all(np.isfinite(values)) and np.all(values > 0) for values in gaps.values()):
        raise ValueError("Log-gap ECDF requires a finite positive gap for every request; no exclusions are allowed")
    worse_gap = sum(fund[rid]["max_gap_s"] > ordinary[rid]["max_gap_s"] for rid in ordinary)
    above = {arm: [int(np.sum(gaps[arm] > threshold)) for threshold in (1, 2)] for arm in STRONG_ARMS}
    earlier, later = int(np.sum(delta < 0)), int(np.sum(delta > 0))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B5BDC3", "text.color": "#25343D", "axes.labelcolor": "#25343D",
        "xtick.color": "#34434D", "ytick.color": "#34434D", "pdf.fonttype": 42,
        "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.25))
    fig.subplots_adjust(left=.075, right=.985, top=.75, bottom=.285, wspace=.28)
    fig.suptitle("Tail gains and request-level costs", x=.075, y=.97, ha="left", fontsize=13, fontweight="semibold")
    fig.text(.075, .9, f"Base-model article workload · all {n} requests per arm · one exploratory run per arm · RTX 5090",
             fontsize=9, color="#5F6D76")

    ax = axes[0]
    low = min(v[0] for v in gaps.values()) * .8
    high = max(v[-1] for v in gaps.values()) * 1.25
    for arm, color in zip(STRONG_ARMS, COLORS):
        values = gaps[arm]
        ax.step(np.r_[low, values, high], np.r_[0, np.arange(1, n + 1) / n, 1],
                where="post", label=LABELS[arm].replace("\n", " "), color=color, linewidth=1.6, zorder=3)
    for threshold in (1, 2):
        ax.axvline(threshold, color="#BDC5CA", linestyle=(0, (3, 3)), linewidth=.8, zorder=1)
    ax.set(xscale="log", xlim=(low, high), ylim=(0, 1.02),
           xlabel="Per-request maximum output gap (s; log scale)", ylabel="Fraction of requests")
    ax.set_xticks([.02, .1, .5, 1, 2, 10])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title("(a) Full-cohort maximum-gap ECDF", loc="left", pad=14, fontweight="semibold")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelspacing=.65)
    ax.grid(axis="y", color="#E6EAED", linewidth=.65)

    ax = axes[1]
    ranks = np.arange(1, n + 1)
    for selected, color in ((delta < 0, "#16796E"), (delta >= 0, "#B96F43")):
        ax.scatter(ranks[selected], delta[selected], s=10, c=color, edgecolors="none", zorder=3)
    ax.axhline(0, color="#596973", linewidth=1.1, zorder=2)
    ax.set(xlim=(-1, n + 2), ylim=(min(delta) - .7, max(delta) + .7),
           xlabel="Requests ordered by completion-time change", ylabel="Funded Q1 − free-fit completion time (s)")
    ax.set_xticks([1, 32, 64, 96, 128] if n == 128 else np.linspace(1, n, 5, dtype=int))
    ax.set_title("(b) Paired completion-time changes", loc="left", pad=14, fontweight="semibold")
    ax.text(.025, .94, f"{earlier} earlier · {later} later", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(.025, .84, f"Mean change: {delta.mean():+.2f} s", transform=ax.transAxes, va="top", fontsize=8, color="#5F6D76")
    ax.grid(axis="y", color="#E6EAED", linewidth=.65)
    ax.set_axisbelow(True)

    caption = (f"Funded Q1 vs free-fit: maximum gap worsens for {worse_gap}/{n}; "
               f"gap >1 s: {above['ordinary'][0]} → {above['queue_fund'][0]}; "
               f"gap >2 s: {above['ordinary'][1]} → {above['queue_fund'][1]}.\n"
               "Paired by request ID with identical arrivals; output trajectories differ. These are descriptive changes, not equal-work causal effects.\n"
               "No confidence intervals. Requests and action events are not independent run replicates; no pooling across machines.")
    fig.text(.075, .043, caption, fontsize=7.8, color="#5F6D76", linespacing=1.55)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    provenance = f"{args.result.name} SHA256 {hashlib.sha256(args.result.read_bytes()).hexdigest()}"
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=240)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), metadata={"Title": "Strong comparison full-cohort distributions", "Subject": provenance})
    plt.close(fig)
    print({"requests_per_arm": n, "gap_worse_vs_ordinary": worse_gap, "gap_counts_above_1s_2s": above,
           "completion_earlier": earlier, "completion_later": later, "mean_completion_delta_s": float(delta.mean())})
    print(args.output_prefix.with_suffix(".png"))
    print(args.output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
