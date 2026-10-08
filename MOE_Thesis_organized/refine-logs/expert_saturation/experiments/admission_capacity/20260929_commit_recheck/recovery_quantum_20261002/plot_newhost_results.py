#!/usr/bin/env python3
"""Plot actual new-host, full-cohort outcomes; each arm is one independent run.

The optional lease report is an analyze_service_session.py result for the same
article inputs. It receives its own row, never a pooled or averaged observation.
Healthy Instruct outcomes are intentionally excluded from these base-model axes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/moe-a-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parent
STRONG_ARMS = ("native", "ordinary", "queue_fund")
LEASE_ARMS = ("q1", "fixed4", "adaptive")
LABELS = {"native": "Native", "ordinary": "Free-fit\nbackfill", "queue_fund": "Funded Q1",
          "q1": "Q1", "fixed4": "Fixed Q4", "adaptive": "Adaptive"}
COLORS = ("#677889", "#589AB5", "#16796E")
METRICS = (
    ("max_gap_request_p95_s", "Maximum-gap tail", "Per-request max gap, P95 (s)", "Lower is better", ".2f"),
    ("actual_output_tokens_s", "Output throughput", "Actual output tokens / s", "Higher is better", ",.0f"),
    ("mean_flow_with_incomplete_penalty_s", "Mean request flow", "Completion − arrival (s)", "Lower is better", ".2f"),
)


def read_group(path, order):
    result = json.loads(path.read_text())
    rows = []
    identity = None
    for arm in order:
        cell = result["arms"][arm]
        metrics = cell["metrics"]
        if metrics["status"] != "COMPLETE" or metrics["completed"] != metrics["expected_requests"] or metrics["failed"] or metrics["unfinished"]:
            raise ValueError(f"{path}: {arm} is not a complete cohort")
        for key, *_ in METRICS:
            if not isinstance(metrics[key], (int, float)) or not math.isfinite(metrics[key]) or metrics[key] < 0:
                raise ValueError(f"Invalid plotted metric: {arm}/{key}")
        current = sorted((r["request_id"], r["prompt_sha256"], r["arrival_s"], r["max_output"])
                         for r in metrics["requests"])
        if identity is not None and current != identity:
            raise ValueError("Input cohort differs between plotted arms")
        identity = current
        rows.append(metrics)
    return rows, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-result", type=Path, default=ROOT / "output/STRONG_NEWHOST_RESULT_R01_20261002.json")
    parser.add_argument("--lease-result", type=Path, help="Optional actual article lease service report; drawn in a separate row")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "output/newhost_results")
    args = parser.parse_args()
    strong, identity = read_group(args.strong_result, STRONG_ARMS)
    groups = [("Strong baselines", STRONG_ARMS, strong)]
    sources = [args.strong_result]
    if args.lease_result:
        lease, lease_identity = read_group(args.lease_result, LEASE_ARMS)
        if lease_identity != identity:
            raise ValueError("Lease report is not the same article input cohort")
        groups.append(("Lease comparison (separate run group)", LEASE_ARMS, lease))
        sources.append(args.lease_result)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B5BDC3", "xtick.color": "#34434D", "ytick.color": "#34434D",
        "text.color": "#25343D", "axes.labelcolor": "#25343D", "pdf.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white"})
    nrows = len(groups)
    fig, axes = plt.subplots(nrows, 3, figsize=(9.8, 3.7 + (nrows - 1) * 2.9), squeeze=False)
    fig.subplots_adjust(left=.075, right=.99, bottom=.235 if nrows == 1 else .14,
                        top=.73 if nrows == 1 else .80, wspace=.38, hspace=.95)
    fig.suptitle("Single-run strong baseline comparison" if nrows == 1 else "New-host service outcomes · separate run groups",
                 x=.075, y=.97, ha="left", fontsize=13, fontweight="semibold")
    fig.text(.075, .897 if nrows == 1 else .922,
             f"OLMoE-1B-7B (base) · article workload · {len(identity)}/{len(identity)} requests complete in every arm · RTX 5090",
             fontsize=9, color="#5F6D76")
    for row_index, (group_label, order, rows) in enumerate(groups):
        for column, (key, title, ylabel, direction, number_format) in enumerate(METRICS):
            ax = axes[row_index, column]
            values = [r[key] for r in rows]
            ax.bar(range(len(order)), values, width=.60, color=COLORS, zorder=3)
            ax.set_xticks(range(len(order)), [LABELS[a] for a in order])
            ax.tick_params(axis="x", length=0, pad=7)
            ax.tick_params(axis="y", length=3)
            ax.set_ylim(0, max(values) * 1.24)
            ax.set_ylabel(ylabel, labelpad=7)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}" if column == 1 else "{x:g}"))
            ax.grid(axis="y", color="#E6EAED", linewidth=.65, zorder=0)
            ax.set_axisbelow(True)
            letter = chr(ord("a") + row_index * 3 + column)
            ax.set_title(f"({letter}) {title}", loc="left", pad=24, fontweight="semibold")
            ax.text(0, 1.04, direction, transform=ax.transAxes, color="#67757D", fontsize=8)
            for x, value in enumerate(values):
                ax.text(x, value + max(values) * .04, format(value, number_format), ha="center", va="bottom", fontsize=9)
        if nrows > 1:
            axes[row_index, 0].text(0, 1.43, group_label, transform=axes[row_index, 0].transAxes, fontsize=10, fontweight="semibold")
    footer = "Exploratory: one run per arm; no confidence intervals or pooling across machines."
    if nrows == 1:
        footer += "\nSerial run order: free-fit, funded Q1, native. Independent trajectories; actual output counts differ."
    else:
        footer += "\nRows are separate execution groups. Independent trajectories; actual output counts differ."
        fixed, adaptive = groups[1][2][1:]
        if (adaptive[METRICS[0][0]] > fixed[METRICS[0][0]]
                and adaptive[METRICS[1][0]] < fixed[METRICS[1][0]]
                and adaptive[METRICS[2][0]] > fixed[METRICS[2][0]]):
            footer = "Lease block: adaptive is worse than fixed Q4 on all three displayed metrics.\n" + footer
    fig.text(.075, .055 if nrows == 1 else .035, footer, fontsize=8, color="#5F6D76", linespacing=1.55)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    provenance = "; ".join(f"{p.name} SHA256 {hashlib.sha256(p.read_bytes()).hexdigest()}" for p in sources)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=240)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), metadata={"Title": "New-host single-run service outcomes", "Subject": provenance})
    plt.close(fig)
    for label, order, rows in groups:
        print(label, {arm: {key: m[key] for key, *_ in METRICS} for arm, m in zip(order, rows)})
    print(args.output_prefix.with_suffix(".png"))
    print(args.output_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
