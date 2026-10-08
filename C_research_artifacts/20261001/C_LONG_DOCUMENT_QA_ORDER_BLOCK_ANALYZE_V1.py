#!/usr/bin/env python3
"""Describe four same-primary full150 order cells as two adjacent DFS/density pairs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import C_LONG_DOCUMENT_QA_FULL_ANALYZE_V1 as quality
import C_LONG_DOCUMENT_QA_POLICY_COMPARE_V1 as policy

ORDER = ("dfs1", "density1", "density2", "dfs2")
GPU = "GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36"
REORDER = {"dfs": "author_default_dfs_reorder_s",
           "density": "subtree_density_reorder_s"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        stream.write("\n")


def check_receipts(root, expected_gpu):
    start = quality.load(root / "start.json")
    pair = quality.load(root / "pair-receipt.json")
    require(start.get("gpu_uuid") == expected_gpu and start.get("run_order") == list(ORDER)
            and pair.get("status") == "COMPLETE" and not pair.get("errors")
            and pair.get("run_order") == list(ORDER)
            and pair.get("freeze_sha256") == start.get("freeze_sha256")
            and [row.get("name") for row in pair.get("cells", [])] == list(ORDER),
            "primary pair start/order/completion receipt differs")
    receipts = {}
    for name, item in zip(ORDER, pair["cells"]):
        row = quality.load(root / name / "launcher-receipt.json")
        before, after = row.get("gpu_before", {}), row.get("gpu_after", {})
        cleanup, life = row.get("stage_cleanup", {}), row.get("child_lifecycle", {})
        require(row.get("cell") == name and row.get("status") == "COMPLETE"
                and not row.get("errors") and row.get("freeze_sha256") == pair["freeze_sha256"]
                and before.get("gpu_uuid") == after.get("gpu_uuid") == expected_gpu
                and before.get("compute_processes") == after.get("compute_processes") == []
                and cleanup.get("status") == item.get("stage_cleanup") == "REMOVED"
                and life.get("child_reaped") is True and life.get("exit_code") == 0
                and row.get("output_qualification", {}).get("status") == "QUALIFIED"
                and item.get("status") == "COMPLETE"
                and item.get("output_qualification", {}).get("status") == "QUALIFIED",
                "primary cell source GPU/lifecycle receipt differs: " + name)
        receipts[name] = dict(launcher_receipt_sha256=quality.sha(
            root / name / "launcher-receipt.json"),
            measured_outputs_sha256=row["output_qualification"]["measured_outputs_sha256"])
    return receipts


def arm_summary(name, cell, q):
    key = REORDER["dfs" if name.startswith("dfs") else "density"]
    reorder = cell["status"].get(key)
    require(type(reorder) in (int, float) and math.isfinite(reorder)
            and reorder >= 0 and reorder == cell["trace"].get(key),
            "reorder overhead receipt differs: " + name)
    rows = list(cell["rows"].values())
    stats = lambda field: policy.summary([row[field] for row in rows])
    return dict(quality_f1_percent=q["mean_f1_percent"],
        natural_eos=q["natural_eos_count"], length_cap=q["length_cap_count"],
        output_tokens=cell["status"]["output_tokens"],
        episode_s=cell["status"]["observation_end_s"],
        reorder_s_included_in_host_origin=reorder,
        ttft_s=stats("ttft_s"), flow_s=stats("flow_s"),
        max_distinct_host_return_gap_s=stats("max_distinct_return_gap_s"),
        scheduled_tokens=cell["scheduled_tokens"],
        deduced_prompt_scheduled_tokens=cell["deduced_prompt_scheduled_tokens"],
        prompt_deduction_scope=cell["prompt_deduction_scope"],
        preemption_events=cell["status"]["preemptions"])


def paired_summary(dfs_name, density_name, arms, comparison):
    a, b = arms[dfs_name], arms[density_name]
    delta = lambda key: b[key] - a[key]
    pct = lambda key: 100 * delta(key) / a[key] if a[key] else None
    return dict(dfs=dfs_name, density=density_name,
        chronological_order=[name for name in ORDER if name in (dfs_name, density_name)],
        mean_ttft_delta_s=b["ttft_s"]["mean"] - a["ttft_s"]["mean"],
        mean_ttft_delta_percent=100 * (
            b["ttft_s"]["mean"] / a["ttft_s"]["mean"] - 1),
        p95_ttft_delta_s=b["ttft_s"]["p95_nearest_rank"] - a["ttft_s"]["p95_nearest_rank"],
        mean_flow_delta_s=b["flow_s"]["mean"] - a["flow_s"]["mean"],
        mean_flow_delta_percent=100 * (
            b["flow_s"]["mean"] / a["flow_s"]["mean"] - 1),
        p95_flow_delta_s=b["flow_s"]["p95_nearest_rank"] - a["flow_s"]["p95_nearest_rank"],
        episode_delta_s=delta("episode_s"), episode_delta_percent=pct("episode_s"),
        worst_host_return_gap_delta_s=(
            b["max_distinct_host_return_gap_s"]["maximum"]
            - a["max_distinct_host_return_gap_s"]["maximum"]),
        p95_host_return_gap_delta_s=(
            b["max_distinct_host_return_gap_s"]["p95_nearest_rank"]
            - a["max_distinct_host_return_gap_s"]["p95_nearest_rank"]),
        reorder_delta_s=delta("reorder_s_included_in_host_origin"),
        prompt_scheduled_work_delta=(
            b["deduced_prompt_scheduled_tokens"] - a["deduced_prompt_scheduled_tokens"]
            if a["deduced_prompt_scheduled_tokens"] is not None
            and b["deduced_prompt_scheduled_tokens"] is not None else None),
        f1_delta_percentage_points=delta("quality_f1_percent"),
        output_change_counts=comparison["output_change_counts"],
        per_request_direction=dict(ttft=comparison["host_ttft_s"]["direction"],
            flow=comparison["host_flow_s"]["direction"],
            maxgap=comparison["maximum_distinct_host_return_gap_s"]["direction"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("input-dir", "metadata-dir", "block-root", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--expected-gpu", default=GPU, choices=(GPU, "GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e"))
    args = parser.parse_args()
    receipts = check_receipts(args.block_root, args.expected_gpu)
    args.output_dir.mkdir(exist_ok=False)
    cells, arms = {}, {}
    for name in ORDER:
        run = args.block_root / name / "native"
        q = quality.analyze(args.input_dir, run, args.metadata_dir)
        require(q["status"] == "COMPLETE", "quality incomplete: " + name)
        qpath = args.output_dir / (name + "-quality.json")
        write(qpath, q)
        cells[name] = policy.read_cell(run, qpath)
        arms[name] = arm_summary(name, cells[name], q)
        raw = run / "measured-outputs.json"
        copied = args.output_dir / (name + "-measured-outputs.json")
        shutil.copyfile(raw, copied)
        require(quality.sha(copied) == quality.sha(raw) == receipts[name]["measured_outputs_sha256"],
                "copied original outputs differ: " + name)
    pairs = []
    for index, (dfs, density) in enumerate((("dfs1", "density1"), ("dfs2", "density2")), 1):
        result = policy.compare(cells[dfs], cells[density], REORDER["density"])
        write(args.output_dir / f"pair{index}-dfs-vs-density.json", result)
        pairs.append(paired_summary(dfs, density, arms, result))
    summary = dict(schema="c-longqa-order-block-analysis-v1",
        status="COMPLETE", gpu_uuid=args.expected_gpu, execution_order=list(ORDER),
        requests_per_cell=150, full150_input_sha256=policy.WORKLOAD_SHA,
        cells=arms, pairs=pairs, source_receipts=receipts,
        interpretation="Two adjacent reversed development pairs; each density-minus-DFS delta is calculated within its pair. No pooling of tokens, confidence interval, equal-output speedup, full PEEK, novelty, or held-out claim.")
    write(args.output_dir / "summary.json", summary)
    print(json.dumps(dict(status="COMPLETE", pairs=pairs), ensure_ascii=False))


if __name__ == "__main__":
    main()
