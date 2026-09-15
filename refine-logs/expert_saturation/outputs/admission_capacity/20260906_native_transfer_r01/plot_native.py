#!/usr/bin/env python3
"""Show every retained fast-arrival native episode; no CIs or pooled estimates."""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="native-capacity-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent


def main():
    destination = HERE / "native_capacity.png"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    result = json.loads((HERE / "analysis/analysis.json").read_text())
    rows = [c for c in result["cells"] if c["plan"]["arrival_scale"] == 0.02]
    assert len(rows) == 16 and all(c["status"] == "COMPLETE" for c in rows)
    assert all(c["metrics"]["n_arrived"] == c["metrics"]["n_slo_pass"] == 16 for c in rows)
    colors = {"steady": "#1D4ED8", "bursty": "#C2410C"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.2))
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.25, top=0.74, wspace=0.28)
    panels = [("goodput_rps", 1, "Finite-episode goodput", "Completed SLO-passing requests / s", (30, 65)),
              ("ttft_p50_s", 1000, "Median TTFT", "Host arrival-to-first-token latency (ms)", (50, 220)),
              ("tpot_p50_s", 1000, "Median per-request mean TPOT", "Host token-delivery interval (ms)", (7, 9))]
    for ax, (metric, scale, title, ylabel, ylim) in zip(axes, panels):
        for regime, color in colors.items():
            for pair, groups in enumerate((("a0_cap6", "b0_cap8"), ("a1_cap6", "b1_cap8"))):
                for repeat in (0, 1):
                    matched = [next(c for c in rows if c["group"] == group and c["plan"]["regime"] == regime
                                    and c["plan"]["repeat"] == repeat) for group in groups]
                    shift = (-0.10 if regime == "steady" else 0.10) + (-0.04 if pair == 0 else 0.04) + (-0.02 if repeat == 0 else 0.02)
                    ax.plot([c["plan"]["cap"] + shift for c in matched], [c[metric] * scale for c in matched],
                        color=color, linewidth=0.9, linestyle="-" if repeat == 0 else "--",
                        marker="o" if pair == 0 else "D", markersize=7,
                        markerfacecolor=color if repeat == 0 else "white", markeredgewidth=1.4)
        ax.set(title=title, xlabel="Native max_num_seqs", ylabel=ylabel, ylim=ylim, xlim=(5.65, 8.35))
        ax.set_xticks([6, 8])
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Native vLLM: cap 6 vs cap 8 at fast arrivals", x=0.055, y=0.965, ha="left", fontsize=21, weight="bold")
    fig.text(0.055, 0.902, "arrival_scale = 0.02  |  0–30 ms arrival horizon  |  All 16 episodes retained  |  Actual decode width reaches each cap", color="#475569")
    handles = [Line2D([], [], color=c, label=name.capitalize()) for name, c in colors.items()]
    handles += [Line2D([], [], color="#334155", marker=m, linestyle="none", markersize=7, label=name)
                for m, name in (("o", "ABBA pair 1"), ("D", "ABBA pair 2"))]
    handles += [Line2D([], [], color="#334155", marker="o", markerfacecolor="#334155" if r == 0 else "white",
                       linestyle="-" if r == 0 else "--", label=f"Repeat {r + 1}") for r in (0, 1)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.86), ncol=6, frameon=False)
    caption = ("16 requests/episode; one RTX 5090; pretrained OLMoE BF16; vLLM 0.26.0 synchronous in-process capture.\n"
               "All 256 plotted requests pass TTFT <= 5 s and mean TPOT <= 200 ms; goodput therefore equals completion throughput.\n"
               "One workload is reused across ABBA processes/repeats. Every point is shown; no confidence intervals or independent-workload claim.\n"
               "Lines link the declared same-repeat comparisons. Horizontal offsets only separate points; finite episodes do not establish steady-state capacity.")
    fig.text(0.055, 0.15, caption, ha="left", va="top", fontsize=10, linespacing=1.45, color="#475569")
    fig.savefig(destination, dpi=190, facecolor="white")
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
