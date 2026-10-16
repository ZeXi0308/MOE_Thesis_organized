#!/usr/bin/env python3
"""Build the twelve code/summary domain plans from actual common populations.

Writes new files only. Existing ShareGPT and 80-record auxiliary plans remain
unchanged. This prepares commands; it does not execute GPU work or claim results.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from prepare_reduced_evaluation import check_config, describe, domain_signature

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("olmoe_ngram", "olmoe", "olmoe_ngram_compiled.json"),
    ("olmoe_dense_draft", "olmoe", "olmoe_dense_draft_reduced.json"),
    ("qwen3_awq_ngram", "qwen3_awq", "qwen3_awq_ngram_reduced.json"),
    ("qwen3_awq_eagle3", "qwen3_awq", "qwen3_awq_eagle3_reduced.json"),
    ("granite3b_ngram", "granite3b", "granite3b_ngram_reduced.json"),
    ("granite3b_neural_draft", "granite3b", "granite3b_neural_draft_reduced.json"),
)
TASKS = {"lcc": 3072, "multi_news": 3584}
BUDGETS_GIB = {"olmoe": 48, "qwen3_awq": 24, "granite3b": 24}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    resource_path = ROOT / "runs/auxiliary_resource_plan_r01/resource_plan.json"
    population_path = ROOT / "runs/auxiliary_population_preparation_r01/manifest.json"
    resources = read(resource_path)
    if resources["population_manifest_sha256"] != sha(population_path):
        raise ValueError("resource plan refers to a different common population")
    if resources["models"]["olmoe_target"]["kv_bytes_per_resident_token"] != 128 * 1024:
        raise ValueError("OLMoE target KV geometry has not been identified")
    for family, gib in BUDGETS_GIB.items():
        proposal = resources["recommendations"][family]
        if proposal["candidate_fixed_kv_bytes"] != gib * 1024**3 or not proposal["same_budget_for_ngram_and_neural"]:
            raise ValueError("fixed KV proposal differs from the input resource estimate")

    entries, files, input_hashes, source_hashes = [], {}, {}, {}
    identity_by_task = {}
    for task, limit in TASKS.items():
        for base, family, source_name in SOURCES:
            source_path = ROOT / "configs" / source_name
            original = read(source_path)
            source_hashes[str(source_path.relative_to(ROOT))] = sha(source_path)
            workload_name = f"configs/{task}_{family}_common128.json"
            workload_path = ROOT / workload_name
            workload = read(workload_path)
            records = workload["records"]
            ids = [r["source_index"] for r in records]
            if len(records) != 128 or ids != sorted(set(ids)) or max(len(r["prompt_token_ids"]) for r in records) > limit:
                raise ValueError("invalid complete common 128-record population")
            if task in identity_by_task and ids != identity_by_task[task]:
                raise ValueError("model families do not share the same original task population")
            identity_by_task[task] = ids
            input_hashes[workload_name] = sha(workload_path)
            domain = f"{base}_{task}_common"
            plan = copy.deepcopy(original)
            plan.update(workload=workload_name, workload_sha256=sha(workload_path),
                batches=[1, 4, 16, 64], gammas=[0, 3, 7], repeats=2,
                measured_requests=64, output_tokens=128, warmup_tokens=32,
                input_offset=0, warmup_offset=0, shape_warmup_offset=0,
                warmup_each_batch=True, max_model_len=4096, max_num_seqs=64,
                max_num_batched_tokens=16384, enforce_eager=False,
                kv_cache_memory_bytes=BUDGETS_GIB[family] * 1024**3,
                admission_mode="wave", skip_trace=True, native_route_return=False,
                candidate_gammas=[0, 3, 7], trace_requests=64,
                group_gpu_budget_seconds=10800, total_gpu_budget_seconds=72000,
                preparation_domain=domain, preparation_role="native_development",
                measurement="Common-source full prompts0-63; native gamma0/3/7 development matrix; two complete finite-wave repeats, 128 forced output tokens. Not a natural-task quality score.",
                execution_domain_note="Compiled V1, complete LongBench-derived prompts. Each task shares original source IDs across targets, with native chat wrappers for OLMoE/Qwen and explicit raw continuation for Granite. Use an independent calibration and held-out evaluation for this workload/domain.",
                warmup_population="All general and shape warmup use development entries0-63; evaluation entries64-127 are separate. Prefix caching is disabled; EWMA is continuous within each engine.",
                kv_budget_scope="Same fixed worker KV budget for both draft sources of this target family; target/draft attention layers share it. This is an input-based capacity estimate, not verified allocation, peak memory, or fixed total resident VRAM.",
                resource_plan="runs/auxiliary_resource_plan_r01/resource_plan.json",
                resource_plan_sha256=sha(resource_path))
            native_name = f"configs/{domain}_reduced.json"
            calibration = copy.deepcopy(plan)
            calibration.update(repeats=1, output_tokens=64,
                preparation_role="budget7_complete_cost",
                measurement="Common-source development prompts0-63; complete host-step costs with production budget7 and post-proposal static caps0/3/7. One calibration run per cell; all drafting remains charged.")
            evaluation = copy.deepcopy(plan)
            evaluation.update(input_offset=64, preparation_role="heldout_policy_evaluation",
                measurement="Common-source held-out prompts64-127; five separately executed policies chosen from development prompts0-63. Candidate caps0/3/7, independent full-budget7 cost table, continuous development-warmed EWMA; two repeats.")
            check_config(calibration, calibration=True)
            check_config(evaluation, calibration=False)
            if not (domain_signature(plan) == domain_signature(calibration) == domain_signature(evaluation)):
                raise ValueError("native/calibration/evaluation execution domains differ")
            names = (native_name, f"configs/{domain}_reduced_calibration.json",
                     f"configs/{domain}_reduced_evaluation.json")
            for name, obj in zip(names, (plan, calibration, evaluation)):
                files[name] = obj
            entries.append({"domain": domain, "task": task, "family": family,
                "draft_source": plan["draft_source"], "source_plan": str(source_path.relative_to(ROOT)),
                "native_config": names[0], "calibration_config": names[1],
                "evaluation_config": names[2], "workload": workload_name,
                "kv_cache_memory_bytes": plan["kv_cache_memory_bytes"]})

    manifest_path = ROOT / "configs/auxiliary_campaigns.json"
    if manifest_path.exists() or any((ROOT / name).exists() for name in files):
        raise FileExistsError("auxiliary outputs already exist; preserve prior plans")
    for name, obj in files.items():
        write_new(ROOT / name, obj)
        input_hashes[name] = sha(ROOT / name)
    for path in (resource_path, population_path, ROOT / "configs/reduced_protocol.json"):
        input_hashes[str(path.relative_to(ROOT))] = sha(path)
    for entry in entries:
        describe(entry["domain"])
    write_new(manifest_path, {"schema_version": 1, "status": "prepared_pending_gpu",
        "total_gpu_budget_seconds": 72000, "domains": entries,
        "input_sha256": input_hashes, "source_plan_sha256": source_hashes,
        "preparer_sha256": sha(Path(__file__)),
        "boundary": "Twelve new workload domains, independent of existing ShareGPT and 80-record auxiliary campaigns. Capacity estimates require actual engine allocation and B64/G7 behavior checks. No new performance evidence or GPU execution is claimed.",
        "execution": "run_auxiliary_sequence.py waits for the original evaluation sequence, then uses the same lock/budgeted run_campaign for native matrices, own G7 calibration and five actual policies. Failures stop continuation; original data are never overwritten."})
    print(json.dumps({"status": "prepared_pending_gpu", "domains": len(entries),
                      "configs": len(files), "workloads": len({x['workload'] for x in entries}),
                      "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
