#!/usr/bin/env python3
"""Plot the pinned four-point G64 development calibration, without inference bars."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "moe_g64_plot_mplcache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


EXPECTED_SELECTION_SHA256 = "b0f4761c3b6d7820c022c3c21fbd45952a7d3a8d48e41c482f2c01fd84e7c986"
SELECTION_NAME = "A_G64_GUARDED_FOUR_POINT_SELECTION_20260930.json"
OUTPUT_DIR_NAME = "A_G64_GUARDED_FOUR_POINT_PLOTS_20260930"
ARMS = ("ltr_t30_q1", "ltr_t30_q10", "ltr_t200_q1", "ltr_t200_q10")
SOURCE_NAMES = (
    "A_G64_GUARDED_T30_PERFORMANCE_AUDIT_R02_20260930.json",
    "A_G64_GUARDED_T200_PERFORMANCE_AUDIT_R01_20260930.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_selection(path: Path, expected_sha256: str) -> tuple[dict, list[dict]]:
    if path.name != SELECTION_NAME or sha256(path) != expected_sha256:
        raise ValueError("four-point selection identity or SHA-256 differs")
    selection = json.loads(path.read_text())
    if (selection.get("schema_version") != 1 or
            selection.get("status") != "FOUR_POINTS_COMPLETE_DEVELOPMENT_SELECTION" or
            selection.get("selected_arm") not in ARMS or
            len(selection.get("points", [])) != 4 or
            tuple(point.get("arm") for point in selection["points"]) != ARMS):
        raise ValueError("four-point selection status or point identity differs")
    if set(selection.get("eligible_arms", [])) != {
            p["arm"] for p in selection["points"] if p.get("eligible") is True}:
        raise ValueError("eligible arm list differs from point flags")
    if selection["selected_arm"] not in selection["eligible_arms"]:
        raise ValueError("selected arm is not eligible")

    sources = selection.get("audit_sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("two pinned audit sources required")
    verified = []
    for source, expected_name in zip(sources, SOURCE_NAMES):
        source_path = Path(source["path"])
        if source_path.name != expected_name or source_path.is_absolute():
            raise ValueError("audit source path identity differs")
        actual_path = path.parent / source_path
        if sha256(actual_path) != source["sha256"]:
            raise ValueError(f"audit source drift: {source_path}")
        verified.append({"path": source_path.as_posix(), "sha256": source["sha256"]})

    for point in selection["points"]:
        arm = point["arm"]
        if point.get("block") != ("T30" if "t30" in arm else "T200"):
            raise ValueError(f"{arm}: block differs")
        for key in ("output_rate_ratio", "mean_flow_ratio", "gap_ratio", "max_gap_s"):
            value = point.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{arm}: invalid {key}")
        eligible = point["output_rate_ratio"] >= .97 and point["mean_flow_ratio"] <= 1.05
        if point["eligible"] is not eligible:
            raise ValueError(f"{arm}: budget flag differs")
    return selection, verified


def make_figure(selection: dict, selection_sha256: str):
    points = selection["points"]
    selected = selection["selected_arm"]
    labels = ["T30 / Q1", "T30 / Q10", "T200 / Q1", "T200 / Q10"]
    axes_data = (
        ("output_rate_ratio", "Output token rate / eager (%)", 97, "Rate floor: 97%"),
        ("mean_flow_ratio", "Mean completed flow / eager (%)", 105, "Flow ceiling: 105%"),
        ("gap_ratio", "Maximum generation gap / eager (%)", None, None),
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": selection_sha256})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.1), sharex=True)
    fig.suptitle("G64 guarded LTR calibration · four development points", fontsize=14,
                 fontweight="semibold", y=.965)
    colors = ["#0b7a75" if p["arm"] == selected else "#718093" for p in points]
    for ax, (metric, ylabel, threshold, threshold_label) in zip(axes, axes_data):
        values = [100 * p[metric] for p in points]
        ax.axhline(100, color="#9aa5b1", linewidth=1, zorder=0)
        if threshold is not None:
            ax.axhline(threshold, color="#b45638", linewidth=1.2, linestyle="--", zorder=0)
        for index, (point, value, color) in enumerate(zip(points, values, colors)):
            marker = "D" if point["arm"] == selected else "o"
            ax.scatter(index, value, s=84 if marker == "D" else 68,
                       color=color, marker=marker, edgecolor="white", linewidth=.9, zorder=3)
            annotation = f"{value:.1f}%"
            if metric == "gap_ratio":
                annotation += f"\n({point['max_gap_s']:.2f} s)"
            ax.annotate(annotation, (index, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9,
                        color="#263542")
        ax.axvline(1.5, color="#d8dfe5", linewidth=.8, zorder=0)
        ax.set_xlim(-.55, 3.55)
        low = min(values + [100] + ([threshold] if threshold is not None else []))
        high = max(values + [100] + ([threshold] if threshold is not None else []))
        margin = max(4.5, .12 * (high - low))
        ax.set_ylim(max(0, low - margin), high + 1.8 * margin)
        ax.set_xticks(range(4), labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        if threshold_label:
            ax.text(.98, .94, threshold_label, transform=ax.transAxes, ha="right",
                    va="top", fontsize=8.5, color="#a2442c")
    legend = [
        Line2D([], [], color="#0b7a75", marker="D", linestyle="None", markersize=8,
               label="Selected eligible point"),
        Line2D([], [], color="#718093", marker="o", linestyle="None", markersize=8,
               label="Other point"),
        Line2D([], [], color="#9aa5b1", linestyle="-", label="Same-block eager = 100%"),
        Line2D([], [], color="#b45638", linestyle="--", label="Service budget"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .12),
               ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .057,
             "Single ordered development block per point. Natural output lengths and sequences may differ; no confidence intervals or blind-test claim.",
             ha="center", va="center", fontsize=9, color="#45515d")
    fig.text(.5, .025, f"Selection SHA-256: {selection_sha256}",
             ha="center", va="center", fontsize=7.5, color="#68727d")
    fig.subplots_adjust(left=.065, right=.985, top=.86, bottom=.275, wspace=.38)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path(SELECTION_NAME))
    parser.add_argument("--selection-sha256", default=EXPECTED_SELECTION_SHA256)
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR_NAME))
    args = parser.parse_args()
    selection, audit_sources = read_selection(args.selection, args.selection_sha256)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fig = make_figure(selection, args.selection_sha256)
    png = args.output_dir / "g64_guarded_four_point_relative.png"
    svg = args.output_dir / "g64_guarded_four_point_relative.svg"
    fig.savefig(png, dpi=240, facecolor="white", metadata={"Software": "matplotlib"})
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    plt.close(fig)
    hashes = {
        "schema_version": 1,
        "selection": {"path": args.selection.as_posix(), "sha256": args.selection_sha256},
        "audit_sources": audit_sources,
        "plot_script": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "outputs": [
            {"path": png.name, "sha256": sha256(png)},
            {"path": svg.name, "sha256": sha256(svg)},
        ],
        "interpretation": "Single ordered development blocks; no confidence intervals or blind-test claim.",
    }
    (args.output_dir / "source_hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "selected_arm": selection["selected_arm"],
                      "source_hashes": str(args.output_dir / "source_hashes.json")}))


if __name__ == "__main__":
    main()
