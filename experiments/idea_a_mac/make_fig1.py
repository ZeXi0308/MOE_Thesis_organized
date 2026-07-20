"""Generate fig1: Rank contribution longtail + cross-model comparison (aligned with redraw_figures).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
BASE = Path("outputs/thesis_evidence")
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "PingFang SC", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,
    "legend.fontsize": 9.5,
    "legend.frameon": True,
    "legend.edgecolor": "#dddddd",
    "legend.fancybox": True,
    "legend.framealpha": 0.9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.unicode_minus": False,
    "axes.titlepad": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_RANK1   = "#E74C3C"   # Red — rank-1 (danger, high importance)
C_RANKK   = "#2980B9"   # Blue — tail-rank (safe to compress)
C_UNIFORM = "#95A5A6"   # Gray — uniform INT4 (reference)
C_MID     = "#AAB7B8"   # Light gray — medium ranks

def main() -> None:
    rrs = pd.read_csv(BASE / "01_main_experiment/olmoe_receiver_rank_share.csv")
    olmoe = rrs.groupby("rank")["median_share"].median().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    # --- Panel (a): OLMoE per-rank bar ---
    ranks = olmoe["rank"].values
    shares = olmoe["median_share"].values * 100
    colors = [C_RANK1 if r == 1 else (C_RANKK if r == ranks.max() else C_MID) for r in ranks]
    bars = ax1.bar(ranks, shares, color=colors, edgecolor="white", linewidth=0.6, width=0.72, zorder=3)

    for b, s in zip(bars, shares):
        ax1.text(b.get_x() + b.get_width() / 2, s + 0.35, f"{s:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Rank (1 = highest gate  →  8 = lowest gate)")
    ax1.set_ylabel("Median contribution share (%)")
    ax1.set_title("(a) OLMoE top-8: contribution share by rank")
    ax1.set_xticks(ranks)
    ax1.set_ylim(0, max(shares) * 1.18)
    ax1.axhline(10, color=C_UNIFORM, ls="--", lw=0.8, alpha=0.5, zorder=1)
    ax1.text(8.2, 10.3, "10%", color=C_UNIFORM, fontsize=8, ha="left")

    # --- Panel (b): cross-model tail-rank share + ratio ---
    models = ["OLMoE\n(top-8)", "Mixtral\n(top-2)", "LLM-jp\n(top-16)"]
    tail_shares = [4.91, 0.014, 2.05]
    ratios = [5.43, 14656, 9.39]
    x = np.arange(len(models))
    w = 0.38
    ax2b = ax2.twinx()
    b1 = ax2.bar(x - w / 2, tail_shares, w, color=C_RANKK, edgecolor="white",
                 linewidth=0.6, zorder=3, label="Tail-rank share (%)")
    b2 = ax2b.bar(x + w / 2, ratios, w, color=C_RANK1, edgecolor="white",
                  linewidth=0.6, zorder=3, label="rank1 / tail-rank ratio (x)")

    ax2.set_ylabel("Tail-rank median share (%)", color=C_RANKK, fontweight="bold")
    ax2b.set_ylabel("rank1 / tail-rank ratio (×)", color=C_RANK1, fontweight="bold")
    ax2b.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.set_title("(b) Cross-model: all below 10% threshold")

    for b, v in zip(b1, tail_shares):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_RANKK)
    for b, v in zip(b2, ratios):
        ax2b.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:g}×",
                  ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_RANK1)
    ax2.set_ylim(0, 6)
    ax2b.set_ylim(0.8, 40000)
    ax2.axhline(10, color=C_UNIFORM, ls="--", lw=1, alpha=0.6)
    ax2.text(2.5, 10.5, "10% threshold", color=C_UNIFORM, fontsize=7.5, ha="center", va="bottom")

    # Unified legend outside on the right of panel (b) for 100% safety from overlap
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.15, 1.0), fontsize=9)

    fig.suptitle("C1: top-k contribution is long-tailed — tail rank is safe to compress first",
                 fontsize=13, fontweight="bold", y=1.02)
    out = FIG_DIR / "fig1_rank_contribution_longtail.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✅ Aligned make_fig1.py and saved to → {out}")

if __name__ == "__main__":
    main()
