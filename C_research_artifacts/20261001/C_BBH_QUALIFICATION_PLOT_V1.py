#!/usr/bin/env python3
"""Plot archived 27-task BBH development qualification output lengths."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ANALYSIS_SHA = "53d04b3dcb12beaf924ee0e1dd93d0a7428c6934f7bdf61022fda174690e9974"
COLORS = {
    "correct_blankline": "#218A78",
    "incorrect_blankline": "#D48A2F",
    "length_limit": "#B94B55",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.analysis) != ANALYSIS_SHA:
        raise ValueError("archived BBH qualification analysis SHA changed")
    if args.png.exists() or args.svg.exists():
        raise FileExistsError("plot outputs must be new")
    report = json.loads(args.analysis.read_text())
    tasks = report["tasks"]
    if (len(tasks) != 27 or [row["task"] for row in tasks] !=
            sorted(row["task"] for row in tasks)
            or len({row["task"] for row in tasks}) != 27):
        raise ValueError("expected 27 unique alphabetically ordered tasks")
    groups = []
    for row in tasks:
        count = row["output_token_count"]
        if type(count) is not int or not 0 <= count <= 512:
            raise ValueError("output token count outside the measured cap")
        if row["finish_reason"] == "length" and count == 512:
            groups.append("length_limit")
        elif row["finish_reason"] == "stop" and row["stop_reason"] == "\n\n":
            groups.append("correct_blankline" if row["correct"] else "incorrect_blankline")
        else:
            raise ValueError("unclassified finish reason")
    counts = Counter(groups)
    if not (counts == {"correct_blankline": 6, "incorrect_blankline": 17,
                       "length_limit": 4}
            and report["denominator"]["correct"] == 6
            and report["repetition_diagnostic"]["flagged_count"] == 1):
        raise ValueError("archived result totals changed")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#AAB3BD",
        "axes.labelcolor": "#243447",
        "xtick.color": "#455463",
        "ytick.color": "#243447",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(12.3, 10.5))
    fig.subplots_adjust(left=.39, right=.95, bottom=.09, top=.85)
    y = list(range(27))
    lengths = [row["output_token_count"] for row in tasks]
    ax.barh(y, lengths, height=.69, color=[COLORS[group] for group in groups])
    ax.set_yticks(y, [row["task"].replace("_", " ") for row in tasks])
    ax.invert_yaxis()
    ax.set_xlim(0, 560)
    ax.set_xticks([0, 128, 256, 384, 512])
    ax.set_xlabel("Generated tokens  ·  limit 512", labelpad=9)
    ax.xaxis.grid(True, color="#E7EAEE", linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.axvline(512, color="#81444A", linestyle=(0, (4, 3)), linewidth=1)
    for yi, length in zip(y, lengths):
        ax.text(length + 5, yi, str(length), va="center", ha="left",
                fontsize=8, color="#374656")
    fig.text(.06, .965, "BBH first-example workload qualification",
             fontsize=17, fontweight="bold", color="#162635", va="top")
    fig.text(.06, .931,
             "6/27 exact  ·  23 blank-line stops  ·  4 length-limit stops  ·  "
             "0 EOS  ·  suffix diagnostic: 1 task",
             fontsize=10.2, color="#465768", va="top")
    fig.legend(handles=[
        Patch(facecolor=COLORS["correct_blankline"], label="Correct · blank-line stop (6)"),
        Patch(facecolor=COLORS["incorrect_blankline"], label="Incorrect · blank-line stop (17)"),
        Patch(facecolor=COLORS["length_limit"], label="Length limit (4)"),
    ], loc="upper left", bbox_to_anchor=(.06, .905), frameon=False, ncol=3,
       fontsize=9.4, handlelength=1.3, columnspacing=1.8)
    fig.text(.06, .026,
             "Development workload qualification: one fixed example per task, "
             "one base-model run. These results are not an official BBH score "
             "or a policy-performance comparison.",
             fontsize=8.8, color="#647484", va="bottom")
    fig.savefig(args.png, dpi=180)
    fig.savefig(args.svg)
    plt.close(fig)
    print(json.dumps({"png": str(args.png), "svg": str(args.svg),
                      "tasks": len(tasks), "categories": dict(counts)}))


if __name__ == "__main__":
    main()
