"""Combined Pareto plot: FP8+tail-INT4 sweep + rank-selection control.

The centerpiece figure for the reframed contribution: shows the tail-INT4
upgrade curve (50% -> 62.5% saving, PPL flat) and the dramatic gap between
tail/head/odd INT4 placement at fixed 62.5% saving.
"""
from __future__ import annotations


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("outputs/main_experiments")
SWEEP = BASE / "olmoe_fp8_tail_int4" / "fp8_tail_int4_results.csv"
CONTROL = BASE / "olmoe_fp8_rank_control" / "rank_control_results.csv"
FIG_DIR = BASE / "olmoe_fp8_tail_int4" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sweep = pd.read_csv(SWEEP)
    ctrl = pd.read_csv(CONTROL)
    sweep = sweep[sweep["strategy"] != "full"]
    ctrl = ctrl[ctrl["strategy"] != "full"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- (a) FP8 + tail-INT4 upgrade sweep: PPL delta vs byte saving ---
    sweep_sorted = sweep.sort_values("byte_saving")
    ax1.plot(sweep_sorted["byte_saving"] * 100, sweep_sorted["ppl_delta"], "-o",
             color="#1f77b4", markersize=8, linewidth=2, label="FP8 + tail-INT4 upgrade")
    for _, r in sweep_sorted.iterrows():
        ax1.annotate(r["strategy"], (r["byte_saving"] * 100, r["ppl_delta"]),
                     textcoords="offset points", xytext=(5, 7), fontsize=6.5, color="#1f77b4")
    # uniform_fp8 reference line
    uf = sweep[sweep["strategy"] == "uniform_fp8"]
    if not uf.empty:
        ax1.axhline(float(uf["ppl_delta"].iloc[0]), color="green", linestyle="--", alpha=0.5,
                    label="uniform_fp8 PPL level")
    ax1.set_xlabel("byte saving (%)", fontsize=11)
    ax1.set_ylabel("PPL delta vs full", fontsize=11)
    ax1.set_title("(a) FP8 + tail-INT4 upgrade: PPL stays flat to 62.5%")
    ax1.legend(loc="upper left", fontsize=9.5)
    ax1.grid(True, alpha=0.3)

    # --- (b) Rank-selection control at fixed 62.5% saving ---
    labels = {"fp8_r5678int4_tail": "tail (rank5-8)\nrank-aware",
              "fp8_r1357int4_odd": "odd (random)\nno rank signal",
              "fp8_r1234int4_head": "head (rank1-4)\nanti-rank-aware"}
    colors = {"fp8_r5678int4_tail": "#2ca02c", "fp8_r1357int4_odd": "#ff7f0e",
              "fp8_r1234int4_head": "#d62728"}
    for _, r in ctrl.iterrows():
        ax2.bar(labels[r["strategy"]], r["ppl_delta"], color=colors[r["strategy"]], alpha=0.8)
        ax2.text(labels[r["strategy"]], r["ppl_delta"], f"KL={r['mean_kl_vs_full']:.1f}\nPPL={r['ppl_delta']:+.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("PPL delta vs full", fontsize=11)
    ax2.set_title("(b) Same 62.5% saving: rank selection = 600x PPL gap")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, 7.2)

    plt.tight_layout()
    out = FIG_DIR / "fig12_fp8_tail_int4_pareto.png"
    plt.savefig(out, dpi=180)
    plt.close()
    # also copy to thesis_evidence figures
    import shutil
    shutil.copy(out, Path("outputs/thesis_evidence/figures/fig12_fp8_tail_int4_pareto.png"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
