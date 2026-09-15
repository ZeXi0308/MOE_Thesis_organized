#!/usr/bin/env python3
"""Plot retained real GPU calibration cells; raw request metrics are recomputed."""
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="moe-calibration-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "refine-logs/expert_saturation/experiments/admission_capacity"))
from metrics import summarize_episode_requests


def main():
    output = HERE / "calibration_capacity.png"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    base = HERE.parent / "20260906_gpu_pilot_r01"
    records = []
    for campaign in ("scan", "bracket"):
        config = json.loads((base / campaign / "config.json").read_text())
        assert (config["requests"], config["prompt_tokens"], config["output_tokens"]) == (16, 128, 16)
        assert (config["ttft_slo_s"], config["tpot_slo_s"]) == (5, 0.2)
        for index, plan in enumerate(config["run_order"]):
            if plan["telemetry"]:
                continue
            raw = json.loads((base / campaign / f"cell-{index:03d}.json").read_text())
            assert raw["plan"] == plan and raw["status"] == "COMPLETE"
            metrics = summarize_episode_requests(raw["requests"],
                observation_end_s=raw["observation_end_s"], ttft_slo_s=5, tpot_slo_s=0.2)
            assert metrics["n_arrived"] == metrics["n_completed"] == 16
            records.append(dict(campaign=campaign, **plan, goodput=metrics["goodput_rps"],
                ttft=metrics["latency_s"]["ttft"]["p50"], tpot=metrics["latency_s"]["tpot"]["p50"]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
                         "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 6.3))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.27, top=0.74, wspace=0.26)
    colors = {"steady": "#1D4ED8", "bursty": "#C2410C"}
    markers = {"scan": "o", "bracket": "D"}
    metrics = [("goodput", "Finite-episode goodput", "SLO-passing requests / s", (0, 2.2)),
               ("ttft", "Median TTFT", "Seconds, arrival to first token", (0, 7)),
               ("tpot", "Median per-request mean TPOT", "Seconds per output-token interval", (0.10, 0.22))]
    for ax, (metric, title, label, limits) in zip(axes, metrics):
        for campaign in ("scan", "bracket"):
            for regime in ("steady", "bursty"):
                for repeat in (0, 1):
                    group = sorted([r for r in records if (r["campaign"], r["regime"], r["repeat"])
                                    == (campaign, regime, repeat)], key=lambda r: r["cap"])
                    jitter = (-0.10 if regime == "steady" else 0.10) + (-0.045 if campaign == "scan" else 0.045) + (-0.02 if repeat == 0 else 0.02)
                    color = colors[regime]
                    ax.plot([r["cap"] + jitter for r in group], [r[metric] for r in group],
                            color=color, linestyle="-" if repeat == 0 else "--", linewidth=0.9,
                            marker=markers[campaign], markersize=6.5, markeredgewidth=1.4,
                            markerfacecolor=color if repeat == 0 else "white", alpha=0.90)
        ax.set(title=title, xlabel="Admission cap (requests)", ylabel=label, ylim=limits, xlim=(1.5, 8.5))
        ax.set_xticks([2, 4, 6, 8])
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        if metric in ("ttft", "tpot"):
            threshold = 5 if metric == "ttft" else 0.2
            ax.axhspan(threshold, limits[1], color="#DC2626", alpha=0.05)
            ax.axhline(threshold, color="#991B1B", linestyle=(0, (5, 3, 1, 3)), linewidth=1.2)
            ax.text(0.98, threshold + (limits[1] - limits[0]) * 0.022, f"SLO {threshold:g} s",
                    transform=ax.get_yaxis_transform(), ha="right", color="#991B1B", fontsize=10)
    fig.suptitle("MoE admission capacity: real GPU calibration", x=0.055, y=0.965, ha="left", fontsize=21, weight="bold")
    fig.text(0.055, 0.902, "Original cap scan (2 / 4 / 8) and follow-up bracket (4 / 6 / 8)  |  Telemetry OFF  |  2026-09-06", color="#475569", fontsize=11)
    handles = [Line2D([], [], color=c, linewidth=2, label=r.capitalize()) for r, c in colors.items()]
    handles += [Line2D([], [], color="#334155", linestyle="none", marker=m, markersize=7, label=c.capitalize()) for c, m in markers.items()]
    handles += [Line2D([], [], color="#334155", marker="o", markerfacecolor="#334155" if r == 0 else "white",
                       linestyle="-" if r == 0 else "--", label=f"Repeat {r + 1}") for r in (0, 1)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.865), ncol=6, frameon=False, columnspacing=1.8)
    caption = ("16 requests/run; OLMoE BF16; one GPU; custom eager runtime; 128 prompt + 16 output tokens.\n"
               "Goodput counts completed requests satisfying both TTFT <= 5 s and mean TPOT <= 0.2 s, divided by episode duration.\n"
               "The same source cohort is reused: repeats are not independent workloads. Each point is a run; no confidence intervals.\n"
               "Small horizontal offsets separate overlapping points; lines guide the eye. This is finite-cohort calibration, not steady-state capacity.")
    fig.text(0.055, 0.165, caption, ha="left", va="top", fontsize=10, linespacing=1.5, color="#475569")
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)
    print(output)
    for cap in (2, 4, 6, 8):
        rows = [r for r in records if r["cap"] == cap]
        print(cap, {key: [min(r[key] for r in rows), max(r[key] for r in rows)] for key in ("goodput", "ttft", "tpot")})


if __name__ == "__main__":
    main()
