#!/usr/bin/env python3
"""Plot the actual fixed-512 QMSum Horn / whole development pair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"whole": "#2364AA", "horn": "#D06B35"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--output-prefix", type=Path)
    args = ap.parse_args()
    data = args.data_dir
    paths = {name: data / f"qmsum_classical_{name}_analysis_v1.json"
             for name in ("whole", "horn")}
    report = {name: read(path) for name, path in paths.items()}
    comparison = read(data / "qmsum_classical_pair_comparison_v1.json")
    actions = read(data / "qmsum_horn_actions_v1.json")
    proxy = read(data / "qmsum_horn_reference_v1.json")
    if (comparison.get("status") != "COMPLETE" or actions.get("output_ids_changed") != 85
            or proxy.get("request_count") != 200):
        raise ValueError("actual pair or classical reference differs")
    for name, role in (("whole", "control"), ("horn", "treatment")):
        r = report[name]
        rows = r.get("per_request", [])
        if (r.get("status") != "COMPLETE" or len(rows) != 200
                or {row["source_index"] for row in rows} != set(range(200))
                or sha(paths[name]) != comparison["analysis_sha256"][role]
                or r["original_run_sha256"]["measured-outputs.json"]
                   != actions["raw_sha256"][name]["measured-outputs.json"]
                or r["service_mix"]["classified_scheduled_prefill_tokens"]
                   != actions["arms"][name]["initial_prompt_compute_tokens"]
                      + actions["arms"][name]["prefill_recompute_tokens_relative_to_first_cache_hit"]):
            raise ValueError(f"{name} input analyses disagree")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 4.8),
                             gridspec_kw={"width_ratios": [1.18, 1.02, 1.18]})
    fig.subplots_adjust(left=.055, right=.985, top=.78, bottom=.27, wspace=.34)

    # These are distinct models/units; only each Horn:whole ratio is plotted.
    whole_proxy = proxy["whole"]["mean_prompt_completion_work"]
    horn_proxy = proxy["horn"]["mean_prompt_completion_work"]
    whole_flow = report["whole"]["flow_s"]["mean"]
    horn_flow = report["horn"]["flow_s"]["mean"]
    x = np.arange(2)
    for shift, name, heights in ((-.18, "whole", [1, 1]),
                                 (.18, "horn", [horn_proxy / whole_proxy,
                                                horn_flow / whole_flow])):
        bars = axes[0].bar(x + shift, heights, width=.34, color=COLORS[name],
                           label="Whole density" if name == "whole" else "Classical Horn")
        axes[0].bar_label(bars, labels=[f"{v:.3f}×" for v in heights],
                          padding=3, fontsize=8.5)
    axes[0].set(title="A  Proxy and actual completion", ylabel="Horn / whole ratio",
                xticks=x, xticklabels=["Serial prompt-work\ncompletion (proxy)",
                                       "Mean request\ncompletion (actual)"])
    axes[0].set_ylim(0, 1.16)
    axes[0].axhline(1, color="#555555", lw=.8, alpha=.7)
    axes[0].grid(axis="y", alpha=.18)

    prefill = [report[name]["service_mix"]["classified_scheduled_prefill_tokens"]
               for name in ("whole", "horn")]
    bars = axes[1].bar([0, 1], np.asarray(prefill) / 1000,
                       color=[COLORS["whole"], COLORS["horn"]], width=.58)
    axes[1].bar_label(bars, labels=[f"{v:,}" for v in prefill],
                      padding=3, fontsize=8.8)
    axes[1].set(title="B  Actual scheduled prefill", ylabel="Thousand tokens",
                xticks=[0, 1], xticklabels=["Whole\ndensity", "Classical\nHorn"])
    axes[1].set_ylim(0, max(prefill) / 1000 * 1.18)
    axes[1].grid(axis="y", alpha=.18)
    fig.text(.51, .133,
             "Horn: 44,176 fewer cached prefix tokens at first allocation\n"
             "+4 extra prefill recovery tokens",
             ha="center", va="top", fontsize=8.2, color="#4A4A4A")

    for name in ("whole", "horn"):
        r = report[name]
        ordered = np.sort([row["flow_s"] for row in r["per_request"]])
        axes[2].step(ordered, np.arange(1, 201) / 200, where="post",
                     lw=2.1, color=COLORS[name],
                     label=("Whole density" if name == "whole" else "Classical Horn")
                           + f" · mean {r['flow_s']['mean']:.2f} s")
    axes[2].set(title="C  All 200 request completions",
                xlabel="Arrival to completion (s)", ylabel="Fraction completed",
                ylim=(0, 1.02), yticks=[0, .25, .5, .75, 1])
    axes[2].grid(alpha=.18)
    axes[2].legend(loc="lower right", frameon=False, fontsize=8.3)

    fig.suptitle("QMSum: classical Horn order vs whole-subtree density",
                 x=.055, ha="left", y=.97, fontsize=14.8, fontweight="bold")
    fig.text(.055, .868,
             "Single viewed spare-host pair · fixed native batch 512 · same 200 prompts, model, 128 slots and 4096 KV blocks",
             fontsize=9.0, color="#3D3D3D")
    fig.text(.055, .047,
             "Prompt-only serial proxy has separate units and is not a GPU bound. "
             "Output IDs differ for 85/200 requests; host timings include the observer. "
             "No exact eviction or causal attribution is implied.",
             fontsize=8.3, color="#555555")
    stem = args.output_prefix or data / "qmsum_horn_plot_v1"
    for path in (stem.with_suffix(".png"), stem.with_suffix(".svg")):
        if path.exists():
            raise FileExistsError(path)
    fig.savefig(stem.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(json.dumps({"png": str(stem.with_suffix(".png")),
                      "svg": str(stem.with_suffix(".svg"))}))


if __name__ == "__main__":
    main()
