#!/usr/bin/env python3
"""Three completed LongQA150 development orderings: prompt work and host flow."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEM = "long_document_qa_order_development_v1"
LABELS = ("Source order", "Author DFS", "Subtree density")
COLORS = ("#587894", "#28827e", "#b36d39")


def read(path: Path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def flow_rows(run: Path):
    rows, digest = read(run / "measured-outputs.json")
    if (len(rows) != 150 or len({r["request_id"] for r in rows}) != 150
            or any(r.get("finished") is not True for r in rows)):
        raise ValueError("requires a complete 150-request original run")
    flow = {r["request_id"]: r["host_elapsed_s"] - r["arrival_s"] for r in rows}
    if any(not math.isfinite(value) or value < 0 for value in flow.values()):
        raise ValueError("invalid original host flow")
    return flow, digest


def values(native: Path, dfs: Path, dfs_compare: Path, candidate_compare: Path):
    original, original_sha = flow_rows(native)
    author, author_sha = flow_rows(dfs)
    a, _ = read(dfs_compare)
    b, _ = read(candidate_compare)  # Absent candidate is an error; no placeholder.
    if (a.get("schema") != "c-longbench-multifieldqa-en-author-dfs-compare-v1"
            or b.get("schema") != "c-longbench-multifieldqa-en-policy-compare-v1"
            or not (a.get("requests") == b.get("requests") == 150)
            or a.get("common_workload_sha256") != b.get("common_workload_sha256")
            or a["original_sha256"]["baseline"]["measured-outputs.json"] != original_sha
            or a["original_sha256"]["dfs"]["measured-outputs.json"] != author_sha
            or b["original_sha256"]["baseline"]["measured-outputs.json"] != author_sha
            or b["baseline"]["deduced_prompt_scheduled_tokens"]
               != a["dfs"]["deduced_prompt_scheduled_tokens"]):
        raise ValueError("source, DFS, and candidate comparison provenance differs")
    deltas = {r["request_id"]: r["flow_delta_s"] for r in b["per_request"]}
    if not (len(deltas) == 150 and set(original) == set(author) == set(deltas)):
        raise ValueError("per-request inventories differ")
    candidate = {rid: author[rid] + deltas[rid] for rid in author}
    arms = (list(original.values()), list(author.values()), list(candidate.values()))
    if any(not math.isfinite(x) or x < 0 for arm in arms for x in arm):
        raise ValueError("candidate host flow is invalid")
    work = (a["baseline"]["deduced_prompt_scheduled_tokens"],
            a["dfs"]["deduced_prompt_scheduled_tokens"],
            b["candidate"]["deduced_prompt_scheduled_tokens"])
    if any(type(x) is not int or x < 0 for x in work):
        raise ValueError("prompt work deduction unavailable, e.g. due to preemption")
    for arm, report in zip(arms, (a["host_flow_s"]["baseline"],
                                  a["host_flow_s"]["dfs"],
                                  b["host_flow_s"]["candidate"])):
        ordered = sorted(arm)
        if (not math.isclose(statistics.mean(arm), report["mean"], abs_tol=1e-9)
                or not math.isclose(ordered[math.ceil(.95 * 150) - 1],
                                    report["p95_nearest_rank"], abs_tol=1e-9)):
            raise ValueError("reconstructed host flow differs from qualified comparison")
    return work, arms


def draw(work, flows, output_dir: Path) -> None:
    png, svg = (output_dir / f"{STEM}.{ext}" for ext in ("png", "svg"))
    if any(p.exists() or p.is_symlink() for p in (png, svg)):
        raise FileExistsError("scientific figure already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.1, 1.45]})
    fig.subplots_adjust(left=.065, right=.98, top=.77, bottom=.22, wspace=.34)
    fig.suptitle("LongQA150 development orderings", y=.96, fontsize=15)
    fig.text(.5, .87, "Three single cells · host-observed flow · descriptive, not a matched speedup",
             ha="center", fontsize=9)
    x = range(3)
    axes[0].bar(x, work, color=COLORS, width=.7)
    axes[0].set(title="Deduced prompt work", ylabel="Scheduled tokens",
                xticks=list(x), xticklabels=("Native", "DFS", "Density"))
    axes[0].tick_params(axis="x", labelrotation=25)
    mean = [statistics.mean(arm) for arm in flows]
    p95 = [sorted(arm)[math.ceil(.95 * 150) - 1] for arm in flows]
    axes[1].bar([i - .17 for i in x], mean, width=.34, color="#4b7796", label="Mean")
    axes[1].bar([i + .17 for i in x], p95, width=.34, color="#bd8249", label="p95")
    axes[1].set(title="Arrival-to-completion", ylabel="Host seconds",
                xticks=list(x), xticklabels=("Native", "DFS", "Density"))
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].legend(frameon=False, fontsize=8)
    for label, color, arm in zip(LABELS, COLORS, flows):
        ordered = sorted(arm)
        axes[2].step(ordered, [(i + 1) / 150 for i in range(150)], where="post",
                     color=color, lw=1.6, label=label)
    axes[2].set(title="Per-request flow CDF", xlabel="Host seconds", ylabel="Fraction of requests",
                ylim=(0, 1.02))
    axes[2].legend(loc="lower right", frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.text(.5, .055, "Prompt work uses zero-preemption P+O−1 deduction; outputs may differ. "
             "No causal speed or equal-work claim.", ha="center", fontsize=8)
    try:
        fig.savefig(png, dpi=180)
        fig.savefig(svg)
    finally:
        plt.close(fig)
    print(json.dumps(dict(png=str(png), svg=str(svg))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-run", type=Path, required=True)
    p.add_argument("--dfs-run", type=Path, required=True)
    p.add_argument("--dfs-comparison", type=Path, required=True)
    p.add_argument("--candidate-comparison", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    draw(*values(a.native_run, a.dfs_run, a.dfs_comparison,
                 a.candidate_comparison), a.output_dir)


if __name__ == "__main__":
    main()
