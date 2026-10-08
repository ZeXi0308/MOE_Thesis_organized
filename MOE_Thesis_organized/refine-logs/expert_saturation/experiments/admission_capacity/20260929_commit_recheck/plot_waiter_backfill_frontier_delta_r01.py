#!/usr/bin/env python3
"""Plot all fixed TTFT/gap frontier differences: ordinary minus primary-first."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "A_WAITER_BACKFILL_TRIPLET_RESULT_R01_20261001.json"
SOURCE_SHA256 = "c76107abf47fef8410010a197bd6cdaf975f9a46442570f721c1ad93081a2973"
STEM = HERE / "A_WAITER_BACKFILL_FRONTIER_DELTA_R01_20261001"
TTFT = (10, 20, 30, 40)
GAP = (1, 2, 4, 8, 12)


def cells(metrics):
    rows = {(row["ttft_deadline_s"], row["gap_deadline_s"]): row
            for row in metrics["frontier"]}
    expected = {(ttft, gap) for ttft in TTFT for gap in GAP}
    if len(rows) != 20 or set(rows) != expected:
        raise ValueError("Fixed 4 by 5 frontier differs")
    duration = metrics["duration_s"]
    for row in rows.values():
        if abs(row["goodput_requests_s"] - row["qualifying_requests"] / duration) > 1e-10:
            raise ValueError("Goodput denominator differs from arm duration")
    return rows


def main():
    actual_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if actual_sha != SOURCE_SHA256:
        raise ValueError("Frozen canonical JSON changed")
    canonical = json.loads(SOURCE.read_text())
    if canonical["status"] != "COMPLETE_TRIPLET":
        raise ValueError("Canonical triplet is incomplete")
    ordinary = canonical["arms"]["ordinary"]["metrics"]
    primary = canonical["arms"]["primary_first"]["metrics"]
    if ordinary["completed"] != 128 or primary["completed"] != 128:
        raise ValueError("Both arms must have 128 complete requests")
    old, new = cells(ordinary), cells(primary)
    count = np.array([[old[(t, g)]["qualifying_requests"]
                       - new[(t, g)]["qualifying_requests"] for g in GAP] for t in TTFT])
    rate = np.array([[old[(t, g)]["goodput_requests_s"]
                      - new[(t, g)]["goodput_requests_s"] for g in GAP] for t in TTFT])
    count_wins, rate_wins = int((count > 0).sum()), int((rate > 0).sum())

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "figure.titlesize": 13, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))
    fig.suptitle("Ordinary waiter backfill minus primary-first follow-up\n"
                 "Fixed TTFT / maximum inter-token gap frontier", y=.98)
    count_image = axes[0].imshow(
        count, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-46, vcenter=0, vmax=46),
        aspect="auto")
    rate_image = axes[1].imshow(rate, cmap="YlGn", vmin=0, vmax=.60, aspect="auto")
    axes[0].set_title(f"Qualified requests: {count_wins}/20 cells higher\n"
                      "Difference in count; each arm has 128 requests")
    axes[1].set_title(f"Goodput: {rate_wins}/20 cells higher\n"
                      "Difference in requests/s; each arm uses its own duration")
    for ax in axes:
        ax.set_xticks(range(len(GAP)), GAP)
        ax.set_yticks(range(len(TTFT)), TTFT)
        ax.set_xlabel("Maximum inter-token gap deadline (s)")
        ax.set_ylabel("TTFT deadline (s)")
        ax.set_xticks(np.arange(-.5, len(GAP), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(TTFT), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(TTFT)):
        for j in range(len(GAP)):
            axes[0].text(j, i, f"{count[i,j]:+d}", ha="center", va="center",
                         color="white" if count[i,j] > 25 else "black", fontsize=10)
            axes[1].text(j, i, f"{rate[i,j]:+.3f}", ha="center", va="center",
                         color="white" if rate[i,j] > .35 else "black", fontsize=9)
    fig.colorbar(count_image, ax=axes[0], shrink=.82,
                 ticks=[-46, -20, 0, 20, 46],
                 label="Ordinary − primary (requests)")
    fig.colorbar(rate_image, ax=axes[1], shrink=.82, label="Ordinary − primary (requests/s)")
    fig.text(.5, .04,
             f"Goodput = qualifying requests / arm duration: ordinary {ordinary['duration_s']:.3f} s; "
             f"primary-first {primary['duration_s']:.3f} s. One seen-input pair; no all-metric dominance claim.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.07, right=.95, top=.78, bottom=.20, wspace=.28)
    fig.savefig(STEM.with_suffix(".png"), dpi=220)
    fig.savefig(STEM.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps({"canonical_sha256": actual_sha, "count_positive": count_wins,
                      "rate_positive": rate_wins,
                      "count_min_max": [int(count.min()), int(count.max())],
                      "rate_min_max": [float(rate.min()), float(rate.max())]}))


if __name__ == "__main__":
    main()
