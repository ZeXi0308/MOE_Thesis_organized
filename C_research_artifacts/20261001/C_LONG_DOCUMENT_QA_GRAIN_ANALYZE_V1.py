#!/usr/bin/env python3
"""Summarize completed shape-warmed LongBench order cells without copying raw prompts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import C_LONG_DOCUMENT_QA_FULL_ANALYZE_V1 as quality
import C_LONG_DOCUMENT_QA_POLICY_COMPARE_V1 as policy


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        stream.write("\n")


def check_block(root, arms):
    start = quality.load(root / "start.json")
    receipt = quality.load(root / "pair-receipt.json")
    gpu = start.get("gpu_uuid")
    require(isinstance(gpu, str) and gpu.startswith("GPU-")
            and start.get("run_order") == arms
            and receipt.get("status") == "COMPLETE" and not receipt.get("errors")
            and receipt.get("run_order") == arms
            and receipt.get("freeze_sha256") == start.get("freeze_sha256")
            and [row.get("name") for row in receipt.get("cells", [])] == arms,
            "block completion, GPU, order, or freeze differs")
    result = {}
    for name, item in zip(arms, receipt["cells"]):
        cell_root = root / name
        row = quality.load(cell_root / "launcher-receipt.json")
        before, after = row.get("gpu_before", {}), row.get("gpu_after", {})
        life = row.get("child_lifecycle", {})
        require(row.get("cell") == name and row.get("status") == item.get("status") == "COMPLETE"
                and not row.get("errors") and row.get("freeze_sha256") == receipt["freeze_sha256"]
                and before.get("gpu_uuid") == after.get("gpu_uuid") == gpu
                and before.get("compute_processes") == after.get("compute_processes") == []
                and row.get("stage_cleanup", {}).get("status") == item.get("stage_cleanup") == "REMOVED"
                and life.get("child_reaped") is True and life.get("exit_code") == 0
                and row.get("output_qualification", {}).get("status") == "QUALIFIED"
                and item.get("output_qualification", {}).get("status") == "QUALIFIED",
                "cell GPU/lifecycle/qualification differs: " + name)
        native = cell_root / "native"
        require(row["output_qualification"].get("measured_outputs_sha256")
                == quality.sha(native / "measured-outputs.json"),
                "launcher/raw output SHA differs: " + name)
        source = quality.load(native / "shape-warmup-source.json")
        warm = quality.load(native / "warmup-large-odd-steps.json")
        pre = quality.load(native / "warmup-large-odd-pre-reset.json")
        reset = quality.load(native / "prefix-cache-reset.json")
        status = quality.load(native / "status.json")
        def clean_reset(doc):
            return (doc.get("reset_succeeded") is True
                    and doc.get("cached_hash_keys_after") == 0
                    and doc.get("drained_after", {}).get("status") == "QUALIFIED"
                    and doc["drained_after"].get("free_blocks") == 4096)
        require(status.get("status") == "COMPLETE" and status.get("request_count") == 150
                and status.get("shape_warmup_policy") == name
                and source.get("policy") == name
                and source.get("source_index") == 0
                and source.get("prompt_tokens") == 1023
                and source.get("max_tokens") == 1
                and source.get("input_workload_sha256") == policy.WORKLOAD_SHA
                and source.get("measurement_prefix_reset_file") == "prefix-cache-reset.json"
                and source.get("scheduled_tokens_per_call") == [1023]
                and len(warm.get("scheduler_calls", [])) == 1
                and warm["scheduler_calls"][0].get("scheduled_tokens_total") == 1023
                and clean_reset(pre) and clean_reset(reset),
                "shared 1023-token warmup or measurement cache reset differs: " + name)
        extra = source.get("total_shape_control_wall_s")
        require(type(extra) in (int, float) and math.isfinite(extra) and extra >= 0,
                "extra warmup timing differs: " + name)
        result[name] = dict(native=native, launcher_sha256=quality.sha(cell_root / "launcher-receipt.json"),
                            shape_source_sha256=quality.sha(native / "shape-warmup-source.json"),
                            shape_steps_sha256=quality.sha(native / "warmup-large-odd-steps.json"),
                            measurement_reset_sha256=quality.sha(native / "prefix-cache-reset.json"),
                            extra_warmup_s=extra,
                            cell_source_sha256=quality.load(cell_root / "start.json")["cell_source_sha256"])
    return gpu, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("block-root", "input-dir", "metadata-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline", choices=("dfs", "whole"), required=True)
    parser.add_argument("--arms", nargs="+", required=True,
                        help="ordered cells, e.g. dfs whole or whole 1 2")
    args = parser.parse_args()
    arms = args.arms
    require(len(arms) == len(set(arms)) and args.baseline in arms
            and set(arms) <= {"dfs", "whole", "1", "2"}
            and all(name != "dfs" or name == args.baseline for name in arms),
            "arms must be unique, ordered, and have the chosen baseline")
    gpu, receipts = check_block(args.block_root, arms)
    reports = {}
    for name in arms:
        q = quality.analyze(args.input_dir, receipts[name]["native"], args.metadata_dir)
        require(q["status"] == "COMPLETE" and q["requests_completed"] == 150,
                "quality incomplete: " + name)
        reports[name] = q
    args.output_dir.mkdir(exist_ok=False)
    cells, summaries = {}, {}
    for name in arms:
        path = args.output_dir / (name + "-quality.json")
        write(path, reports[name])
        cell = policy.read_cell(receipts[name]["native"], path)
        cells[name] = cell
        steps = cell["trace"]["steps"]
        require(len(steps) == cell["status"]["schedule_calls"],
                "step/schedule count differs: " + name)
        index, maximum = max(enumerate(steps), key=lambda pair: pair[1]["return_s"]-pair[1]["start_s"])
        rows = list(cell["rows"].values())
        summaries[name] = dict(
            quality_f1_percent=reports[name]["mean_f1_percent"],
            output_tokens=cell["status"]["output_tokens"],
            mean_completion_s=policy.summary([r["flow_s"] for r in rows])["mean"],
            p95_completion_s=policy.summary([r["flow_s"] for r in rows])["p95_nearest_rank"],
            max_distinct_host_return_gap_s=max(r["max_distinct_return_gap_s"] for r in rows),
            deduced_prompt_scheduled_tokens=cell["deduced_prompt_scheduled_tokens"],
            prompt_deduction_scope=cell["prompt_deduction_scope"],
            peak_used_blocks_after_schedule=cell["status"]["peak_used_blocks_after_schedule"],
            extra_shape_warmup_wall_s=receipts[name]["extra_warmup_s"],
            max_step=dict(call=index, wall_s=maximum["return_s"]-maximum["start_s"],
                          scheduled_tokens=cell["trace"]["scheduler_calls"][index]["scheduled_tokens_total"]),
            source_sha256=dict(cell=receipts[name]["cell_source_sha256"],
                               launcher=receipts[name]["launcher_sha256"],
                               shape_warmup=receipts[name]["shape_source_sha256"],
                               shape_steps=receipts[name]["shape_steps_sha256"],
                               measurement_reset=receipts[name]["measurement_reset_sha256"],
                               measured_steps=cell["hashes"]["measured-steps.json"],
                               measured_outputs=cell["hashes"]["measured-outputs.json"]))
    comparisons = {}
    for name in arms:
        if name == args.baseline:
            continue
        result = policy.compare(cells[args.baseline], cells[name], "subtree_density_reorder_s")
        filename = args.baseline + "-vs-" + name + ".json"
        write(args.output_dir / filename, result)
        comparisons[name] = dict(path=filename,
            output_change_counts=result["output_change_counts"],
            mean_completion_delta_s=result["host_flow_s"]["deltas"]["mean"],
            p95_completion_delta_s=(result["host_flow_s"]["candidate"]["p95_nearest_rank"]
                                    - result["host_flow_s"]["baseline"]["p95_nearest_rank"]))
    write(args.output_dir / "summary.json", dict(
        schema="c-longqa-grain-analysis-v1", status="COMPLETE", block_root=str(args.block_root),
        gpu_uuid=gpu, execution_order=arms, baseline=args.baseline,
        input_workload_sha256=quality.sha(args.input_dir / "workload.json"),
        arms=summaries, comparisons=comparisons,
        scope="Completed viewed development cells; full output differences preserved, no raw prompt copy, no new SLO or equal-output speedup claim"))
    print(json.dumps(dict(status="COMPLETE", arms=arms, comparisons=list(comparisons))))


if __name__ == "__main__":
    main()
