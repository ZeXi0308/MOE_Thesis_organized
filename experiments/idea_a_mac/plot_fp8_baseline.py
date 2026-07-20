"""Plot FP8 baseline vs rank-aware INT4 Pareto frontier (responds to MegaScale-MoE).

Reads ``olmoe_fp8_baseline/approx_results.csv`` and overlays the existing MILP
end-to-end point to show where uniform FP8 sits relative to the rank-aware
mixed-precision strategies.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("outputs/main_experiments")
CSV = BASE / "olmoe_fp8_baseline" / "approx_results.csv"
FIG_DIR = BASE / "olmoe_fp8_baseline" / "figures"

# Existing MILP end-to-end point (from thesis_evidence run, eps=0.1).
MILP_POINT = {"strategy": "MILP (existing)", "byte_saving": 0.558, "mean_kl_vs_full": 9.414}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    df = df[df["strategy"] != "full"].copy()

    # Groups for the Pareto plot.
    fp8_mix = ["rank8_fp8", "keep4_bf16_rest_fp8", "keep2_bf16_rest_fp8", "keep1_bf16_rest_fp8", "uniform_fp8"]
    int4_ref = ["rank8_int4", "rank1_int4", "uniform_int4"]
    int8_ref = ["uniform_int8"]

    color_fp8 = "#1f77b4"
    color_int4 = "#d62728"
    color_int8 = "#ff7f0e"
    color_milp = "#9467bd"

    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    label_box = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85)
    fp8_offsets = {
        "rank8_fp8": (-4, 10),
        "keep4_bf16_rest_fp8": (-18, 10),
        "keep2_bf16_rest_fp8": (8, 10),
        "keep1_bf16_rest_fp8": (8, -14),
        "uniform_fp8": (8, 8),
    }
    int4_offsets = {
        "rank8_int4": (8, 8),
        "rank1_int4": (8, 8),
        "uniform_int4": (-42, 8),
    }

    # FP8 frontier (connected).
    sub = df[df["strategy"].isin(fp8_mix)].set_index("strategy").loc[fp8_mix]
    ax.plot(sub["byte_saving"] * 100, sub["mean_kl_vs_full"], "-o", color=color_fp8,
            label="FP8 frontier (rank-aware + uniform)", markersize=8, linewidth=2)
    for s, row in sub.iterrows():
        ax.annotate(s, (row["byte_saving"] * 100, row["mean_kl_vs_full"]),
                    textcoords="offset points", xytext=fp8_offsets.get(s, (6, 6)),
                    fontsize=7.5, color=color_fp8, bbox=label_box, zorder=10)

    # INT4 reference points.
    sub4 = df[df["strategy"].isin(int4_ref)].set_index("strategy").loc[int4_ref]
    ax.scatter(sub4["byte_saving"] * 100, sub4["mean_kl_vs_full"], color=color_int4, marker="X", s=110,
               label="INT4 reference (existing)", zorder=5)
    for s, row in sub4.iterrows():
        ax.annotate(s, (row["byte_saving"] * 100, row["mean_kl_vs_full"]),
                    textcoords="offset points", xytext=int4_offsets.get(s, (6, 6)),
                    fontsize=7.5, color=color_int4, bbox=label_box, zorder=10)

    # INT8 reference.
    sub8 = df[df["strategy"].isin(int8_ref)]
    ax.scatter(sub8["byte_saving"] * 100, sub8["mean_kl_vs_full"], color=color_int8, marker="s", s=90,
               label="uniform INT8 (same 1-B cost as FP8)", zorder=5)
    for _, row in sub8.iterrows():
        ax.annotate(row["strategy"], (row["byte_saving"] * 100, row["mean_kl_vs_full"]),
                    textcoords="offset points", xytext=(8, -14),
                    fontsize=7.5, color=color_int8, bbox=label_box, zorder=10)

    # MILP point.
    ax.scatter([MILP_POINT["byte_saving"] * 100], [MILP_POINT["mean_kl_vs_full"]], color=color_milp, marker="D", s=110,
               label="MILP receiver-aware (existing, eps=0.1)", zorder=5)
    ax.annotate("MILP", (MILP_POINT["byte_saving"] * 100, MILP_POINT["mean_kl_vs_full"]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=7.5, color=color_milp, bbox=label_box, zorder=10)

    ax.set_yscale("log")
    ax.set_xlabel("byte saving (%)", fontsize=11)
    ax.set_ylabel("next-token KL vs full (log scale)", fontsize=11)
    ax.set_title("OLMoE top-8: FP8 baseline vs rank-aware INT4 (32 samples, WikiText-2)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout(rect=(0, 0, 0.83, 1))
    out = FIG_DIR / "fig11_fp8_vs_int4_pareto.png"
    plt.savefig(out, dpi=180)
    plt.close()
    
    import shutil
    shutil.copy(out, Path("outputs/thesis_evidence/figures/fig11_fp8_vs_int4_pareto.png"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
