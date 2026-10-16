"""Publication layout of existing r07/r09b CSVs; no new measurements.

Run from any directory with matplotlib installed. All nine arms, seven
batches and both formats are retained. Legends are outside the data axes.
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
BATCHES = [1, 2, 4, 8, 16, 32, 64]
FORMATS = [("BF16", "results_r07"), ("FP8 W8A8", "results_fp8_r09b")]
ARMS = ["global_j1", "global_j2", "global_j4", "gate_low", "gate_high",
        "marginal_low", "marginal_mid", "marginal_high"]
LABELS = ["Native top-7", "Native top-6", "Native top-4", "Gate-low", "Gate-high",
          "Marginal-low", "Marginal-mid", "Marginal-high"]
COLORS = ["#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#1f77b4",
          "#e377c2", "#7f7f7f", "#8c564b"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "h"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def read_rows(directory):
    with (FIGURES / directory / "bench_points.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 63
    points = {(int(r["batch"]), r["arm"]): r for r in rows}
    assert len(points) == 63
    for batch in BATCHES:
        for arm in ["native"] + ARMS:
            row = points[batch, arm]
            assert row["run_status"] == "COMPLETE"
            assert row["diagnostic"] == "False"
            assert float(row["energy_window_s"]) >= 3
            assert float(row["decode_output_tokens"]) == 2047 * batch
    return points


def axis_style(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid(alpha=.23)
    ax.set_xlabel("Decode batch, B")


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(HERE / (name + "." + ext), dpi=220)
    plt.close(fig)


def main():
    data = [read_rows(directory) for _, directory in FORMATS]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), sharex=True, sharey="col")
    for row_i, ((name, _), points) in enumerate(zip(FORMATS, data)):
        for col, field in enumerate(("tpot_saving_fraction", "logical_footprint_saving_fraction")):
            ax = axes[row_i, col]
            for arm, label, color, marker in zip(ARMS, LABELS, COLORS, MARKERS):
                values = [100 * float(points[batch, arm][field]) for batch in BATCHES]
                ax.plot(BATCHES, values, label=label, color=color, marker=marker,
                        markersize=3.4, linewidth=1.25)
            ax.axhline(0, color=".45", linewidth=.7)
            axis_style(ax)
            ax.set_title(name, fontsize=9, loc="left")
            ax.set_ylabel(["TPOT reduction (%)", "Logical-footprint reduction (%)"][col])
            ax.tick_params(labelbottom=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.51, 1), fontsize=7.5, columnspacing=1.4)
    fig.subplots_adjust(left=.09, right=.985, bottom=.10, top=.84, hspace=.48, wspace=.28)
    save(fig, "F2_formats")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, (name, _), points in zip(axes, FORMATS, data):
        for arm, label, color, marker in zip(["native"] + ARMS,
                ["Native top-8"] + LABELS, ["#111111"] + COLORS, ["."] + MARKERS):
            values = [float(points[batch, arm]["j_per_decode_token"]) for batch in BATCHES]
            ax.plot(BATCHES, values, label=label, color=color, marker=marker,
                    markersize=3.4, linewidth=1.25)
        axis_style(ax)
        ax.set_title(name, fontsize=9, loc="left")
    axes[0].set_ylabel("Whole-board energy / decode token (J)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.51, 1), fontsize=7.5, columnspacing=1.9)
    fig.subplots_adjust(left=.09, right=.985, bottom=.18, top=.70, wspace=.18)
    save(fig, "F4_formats")


if __name__ == "__main__":
    main()
