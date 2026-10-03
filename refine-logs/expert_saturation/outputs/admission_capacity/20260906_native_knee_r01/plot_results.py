#!/usr/bin/env python3
"""Plot every main-load repeat and every feedback trajectory; no fitted intervals."""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="moe-knee-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    stage1 = read(ROOT / "analysis/analysis.json")
    stage2 = read(ROOT / "policy_probe/analysis/analysis.json")
    if not stage1["complete_expected_capture"] or not stage2["complete_expected_capture"]:
        raise ValueError("Plotting requires complete retained analyses for both stages")
    cells1 = [c for c in stage1["cells"] if c["plan"]["arrival_scale"] == 1]
    cells2 = [c for c in stage2["cells"] if c["plan"]["arrival_scale"] == 1]
    feedback = sorted((c for c in cells2 if c["plan"]["policy"] == "feedback"),
                      key=lambda c: (c["plan"]["regime"] != "steady", c["block"], c["id"]))
    if len(feedback) != 4 or not all(c["qualified"] for c in cells1 + cells2):
        raise ValueError("All main-load cells and all four feedback episodes must be qualified")
    raw = [read(ROOT / "policy_probe/gpu_results" / (c["id"] + ".json")) for c in feedback]
    config = read(ROOT / "policy_probe/gpu_results" / feedback[0]["group"] / "config.json")
    slo = f"TTFT <= {config['ttft_slo_s'] * 1000:g} ms; request mean TPOT <= {config['tpot_slo_s'] * 1000:g} ms"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 220})
    fig = plt.figure(figsize=(13.5, 10), layout="constrained")
    grid = fig.add_gridspec(4, 2, width_ratios=(1.18, 1))
    left = [fig.add_subplot(grid[:2, 0]), fig.add_subplot(grid[2:, 0])]
    categories = [("Stage 1", "static", c) for c in (4, 8, 16, 32)]
    categories += [("Stage 2", "static", c) for c in (8, 12, 16, 32)]
    categories += [("Stage 2", "shadow", 32), ("Stage 2", "feedback", 32)]
    labels = [str(cap) if policy == "static" else {"shadow": "Shadow\n32", "feedback": "Feedback\n32"}[policy]
              for _, policy, cap in categories]
    colors = {"Stage 1": "#77838c", "static": "#22669d", "shadow": "#cb8b2e", "feedback": "#19836b"}
    ymax = max(c["goodput_rps"] for c in cells1 + cells2) * 1.14
    for ax, regime in zip(left, ("steady", "bursty")):
        for x, (stage, policy, cap) in enumerate(categories):
            selected = sorted((c for c in (cells1 if stage == "Stage 1" else cells2)
                if c["plan"]["regime"] == regime and c["plan"]["cap"] == cap
                and c["plan"].get("policy", "static") == policy), key=lambda c: c["block"])
            if len(selected) != 2:
                raise ValueError(f"Expected exactly two repeats: {stage}/{policy}/{cap}/{regime}")
            color = colors["Stage 1" if stage == "Stage 1" else policy]
            for repeat, c in enumerate(selected):
                ax.scatter(x + (-0.10, 0.10)[repeat], c["goodput_rps"], marker=("o", "D")[repeat],
                           color=color, s=38, edgecolors="white", linewidths=0.45, zorder=3)
        ax.set(title=f"{regime.capitalize()} arrivals: request SLO-goodput", ylabel="Goodput (requests/s)",
               xticks=range(len(labels)), xticklabels=labels, ylim=(0, ymax), xlim=(-0.6, 9.6))
        ax.axvspan(-0.6, 3.5, color="#77838c", alpha=0.06)
        ax.axvline(3.5, color="#b8b8b8", linewidth=0.8)
        ax.text(1.5, 0.96, "Stage 1: static cap", ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.text(6.5, 0.96, "Stage 2: static / shadow / feedback", ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.set_xlabel("Admission cap or policy (feedback starts at 32)")
        ax.grid(axis="y", alpha=0.2)
    left[0].scatter([], [], color="#555555", marker="o", label="Forward repeat")
    left[0].scatter([], [], color="#555555", marker="D", label="Reverse repeat")
    left[0].legend(loc="upper left", bbox_to_anchor=(0, 0.90), frameon=False, fontsize=8)
    xmax = max(r["observation_end_s"] for r in raw) * 1.02
    for index, (c, r) in enumerate(zip(feedback, raw)):
        ax = fig.add_subplot(grid[index, 1])
        steps, decisions = r["scheduler_steps"], r["feedback_decisions"]
        finished = max(q["completion_s"] for q in r["requests"])
        target_x = [0] + [d["applied_s"] for d in decisions] + [finished]
        target_y = [r["target_cap"]] + [d["target_cap"] for d in decisions] + [decisions[-1]["target_cap"]]
        ax.step(target_x, target_y, where="post", color="#333333", linestyle="--", linewidth=1.35, label="Target B")
        times = [0] + [s["end_s"] for s in steps] + [finished]
        for field, label, color, style in (("actual_active", "Active", "#22669d", "-"),
                ("decode_requests", "Decode batch", "#d47828", ":"),
                ("waiting_requests", "Waiting", "#19836b", "-.")):
            ax.step(times, [0] + [s[field] for s in steps] + [0], where="post", color=color,
                    linestyle=style, linewidth=1.1, label=label)
        repeat = "forward" if c["block"] == 0 else "reverse"
        ax.set(title=f"Feedback: {c['plan']['regime']} / {repeat} ({c['id']})",
               ylabel="Requests", xlim=(0, xmax), ylim=(-1, 34), yticks=(0, 8, 16, 24, 32))
        ax.grid(alpha=0.15)
        if index == 0:
            ax.legend(ncol=4, loc="upper right", fontsize=7, frameon=False)
        if index == 3:
            ax.set_xlabel("Episode host time (s); target at application, counts after scheduling")
    fig.suptitle("Native nonpreemptive admission: all main-load repeats and feedback trajectories\n" + slo, fontsize=13)
    fig.supxlabel("32 real texts, 128 prompt / 128 output tokens. Points are whole episodes; no confidence intervals. "
                  "All four feedback episodes shown.", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"figures.{suffix}")
    print("Wrote figures.png and figures.pdf from all retained main-load cells and four feedback episodes.")


if __name__ == "__main__":
    main()
