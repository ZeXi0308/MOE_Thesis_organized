#!/usr/bin/env python3
"""Recompute fixed-width native companion contrasts from retained CPU tensors."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch


ARMS = ("original", "permuted", "replaced")
FIELDS = ("input", "logits", "weights", "ids", "output")


def read(path):
    return json.loads(path.read_text())


def compare(a, b):
    same_layout = a.shape == b.shape and a.dtype == b.dtype
    if not same_layout:
        return dict(bitwise_equal=False, max_abs=None, layout_equal=False)
    return dict(bitwise_equal=bool(torch.equal(a.contiguous().view(torch.uint8),
                                              b.contiguous().view(torch.uint8))),
                max_abs=float((a.double() - b.double()).abs().max()) if a.numel() else 0.0,
                layout_equal=True)


def validate_record(record, tensors, fixture, config):
    pos, width = config["target_position"], config["width"]
    indices, target = record["source_indices"], record["target"]
    assert len(indices) == width and len(set(indices)) == width
    assert record["target_position"] == pos and indices[pos] == target
    assert all(0 <= i < fixture["hidden"].shape[0] for i in indices)
    ids = tensors["ids"]
    assert ids.ndim == 2 and ids.shape[0] == width
    n_experts = tensors["logits"].shape[1]
    counts = torch.bincount(ids.flatten().long(), minlength=n_experts)
    assert counts.tolist() == record["expert_counts"] and len(counts) == n_experts
    assert abs(record["U"] - float((counts > 0).float().mean())) < 1e-7
    assert abs(record["C"] - float(counts.max().float() / counts.float().mean())) < 1e-7
    assert tensors["weights"].shape == ids.shape
    assert tensors["output"].shape == (width, fixture["hidden"].shape[1])
    assert tensors["logits"].shape[0] == width
    for name in ("logits", "weights", "output"):
        assert torch.isfinite(tensors[name]).all(), name
    sorted_ids, experts = tensors["sorted_token_ids"], tensors["expert_ids"]
    assert sorted_ids.numel() == record["padded_count"]
    block = record["actual_kernel_config"]["BLOCK_SIZE_M"]
    actual_locations, internal_locations = {}, {}
    for slot in range(ids.shape[1]):
        assignment = pos * ids.shape[1] + slot
        locations = (sorted_ids == assignment).nonzero().flatten().tolist()
        assert len(locations) == 1
        assert int(experts[locations[0] // block]) == int(ids[pos, slot])
        actual_locations[str(assignment)] = locations
        internal_locations[str(assignment)] = dict(absolute_offset=locations[0],
            block_index=locations[0] // block, row_in_block=locations[0] % block,
            expert_id=int(experts[locations[0] // block]), selected_slot=slot)
    assert actual_locations == record["target_sorted_locations"]
    record["derived_target_dispatch"] = internal_locations
    return dict(input=fixture["hidden"][indices[pos]],
                **{name: tensors[name][pos] for name in FIELDS if name != "input"})


def contrast(kind, left, right):
    a, b = left["record"], right["record"]
    output = dict(kind=kind, target=a["target"],
                  left_process=left["process"], right_process=right["process"],
                  left_key=a["key"], right_key=b["key"],
                  fields={name: compare(left["target_tensors"][name], right["target_tensors"][name])
                          for name in FIELDS})
    output.update(expert_counts_equal=a["expert_counts"] == b["expert_counts"],
                  expert_count_l1=sum(abs(x - y) for x, y in zip(a["expert_counts"], b["expert_counts"])),
                  target_sorted_locations_equal=a["target_sorted_locations"] == b["target_sorted_locations"],
                  target_internal_row_equal=all(a["derived_target_dispatch"][key]["row_in_block"] ==
                      b["derived_target_dispatch"][key]["row_in_block"] for key in a["derived_target_dispatch"]),
                  target_block_equal=all(a["derived_target_dispatch"][key]["block_index"] ==
                      b["derived_target_dispatch"][key]["block_index"] for key in a["derived_target_dispatch"]),
                  target_dispatch_expert_equal=all(a["derived_target_dispatch"][key]["expert_id"] ==
                      b["derived_target_dispatch"][key]["expert_id"] for key in a["derived_target_dispatch"]),
                  actual_kernel_config_equal=a["actual_kernel_config"] == b["actual_kernel_config"],
                  left_padded_count=a["padded_count"], right_padded_count=b["padded_count"])
    return output


def summarize(rows):
    return dict(comparisons=len(rows),
                different={name: sum(not row["fields"][name]["bitwise_equal"] for row in rows)
                           for name in FIELDS},
                max_abs={name: max((row["fields"][name]["max_abs"] or 0.0 for row in rows), default=0.0)
                         for name in FIELDS},
                expert_counts_changed=sum(not row["expert_counts_equal"] for row in rows),
                target_sorted_locations_changed=sum(not row["target_sorted_locations_equal"] for row in rows),
                target_internal_row_changed=sum(not row["target_internal_row_equal"] for row in rows),
                target_block_changed=sum(not row["target_block_equal"] for row in rows),
                target_dispatch_expert_changed=sum(not row["target_dispatch_expert_equal"] for row in rows),
                actual_kernel_config_changed=sum(not row["actual_kernel_config_equal"] for row in rows))


def analyze(root):
    config = read(root / "config.json")
    fixture_path = root / "fixture.pt"
    if not fixture_path.exists():
        fixture_path = root / "gpu_results" / "fixture.pt"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    fixture_digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    assert fixture["layer"] == config["layer"]
    assert fixture["hidden"].shape[0] == config["fixture_requests"]
    assert fixture["hidden"].dtype == torch.bfloat16 and torch.isfinite(fixture["hidden"]).all()
    inputs = read(root / "inputs.json")
    assert fixture["input_sha256"] == hashlib.sha256((root / "inputs.json").read_bytes()).hexdigest()
    assert fixture["source_requests"] == inputs["source_requests"]
    pool, engine_args, extraction_checks, environments = {}, [], [], []
    expected = {f"r{r}/target{t}/{arm}" for r in range(config["repeats"])
                for t in config["target_source_indices"] for arm in ARMS}
    for process in range(config["processes"]):
        directory = root / "gpu_results" / f"process-{process}"
        status = read(directory / "status.json")
        assert status["status"] == "COMPLETE" and status["measured_calls"] == len(expected)
        assert status["fixture_sha256"] == fixture_digest
        assert read(directory / "config.json") == config
        engine_args.append(read(directory / "engine_args.json"))
        environments.append(read(directory / "environment.json"))
        checks = read(directory / "extraction_checks.json")
        assert len(checks) == len(config["target_source_indices"]) * len(ARMS)
        assert {(c["target"], c["arm"]) for c in checks} == set(itertools.product(config["target_source_indices"], ARMS))
        assert all(c["native_full_equals_split"] and c["max_abs"] == 0 for c in checks)
        extraction_checks.append(dict(process=process, checks=checks))
        records = read(directory / "records.json")
        tensors = torch.load(directory / "tensors.pt", map_location="cpu", weights_only=True)
        assert len(records) == len(expected) and {r["key"] for r in records} == expected
        assert set(tensors) == expected
        for record in records:
            key = record["key"]
            assert key == f"r{record['repeat']}/target{record['target']}/{record['arm']}"
            target_tensors = validate_record(record, tensors[key], fixture, config)
            pool[process, key] = dict(process=process, record=record, target_tensors=target_tensors)
    assert all(args == engine_args[0] for args in engine_args)
    common_environment = ("python", "torch", "cuda", "vllm", "cpu_threads", "use_layername",
                          "model_layer_type", "expert_type", "quant_method_type", "source_sha256")
    environment_equal = {field: all(env[field] == environments[0][field] for env in environments)
                         for field in common_environment}
    assert all(environment_equal.values()), environment_equal
    comparisons = []
    for process in range(config["processes"]):
        for target in config["target_source_indices"]:
            for arm in ARMS:
                for ra, rb in itertools.combinations(range(config["repeats"]), 2):
                    comparisons.append(contrast("same_arm_repeat", pool[process, f"r{ra}/target{target}/{arm}"],
                                                pool[process, f"r{rb}/target{target}/{arm}"]))
            for repeat in range(config["repeats"]):
                for arm in ARMS[1:]:
                    comparisons.append(contrast(f"original_vs_{arm}",
                        pool[process, f"r{repeat}/target{target}/original"],
                        pool[process, f"r{repeat}/target{target}/{arm}"]))
    for pa, pb in itertools.combinations(range(config["processes"]), 2):
        for key in sorted(expected):
            comparisons.append(contrast("cross_process", pool[pa, key], pool[pb, key]))
    groups = {kind: summarize([row for row in comparisons if row["kind"] == kind])
              for kind in ("same_arm_repeat", "original_vs_permuted", "original_vs_replaced", "cross_process")}
    targets = [{"source_index": target,
                **{key: fixture["source_requests"][target][key]
                   for key in ("request_id", "dataset_row_index", "document_id")}}
               for target in config["target_source_indices"]]
    records = [entry["record"] | {"process": entry["process"]} for entry in pool.values()]
    within_unstable = any(groups["same_arm_repeat"]["different"].values())
    across_unstable = any(groups["cross_process"]["different"].values())
    intervention_changes = any(groups[k]["different"]["output"] for k in ("original_vs_permuted", "original_vs_replaced"))
    if within_unstable or across_unstable:
        verdict = "REPEAT_INSTABILITY_LOCALIZATION_REQUIRED"
    elif intervention_changes:
        verdict = "STABLE_COMPANION_DEPENDENT_TARGET_OUTPUT_OBSERVED"
    else:
        verdict = "NO_TARGET_OUTPUT_DIFFERENCE_IN_TESTED_FIXED_WIDTH_OPERATOR"
    return dict(verdict=verdict, evidence_ceiling=config["evidence_ceiling"],
                measured_calls=len(records), distinct_targets=len(targets), targets=targets,
                fixture_sha256=fixture_digest, global_config=config, global_engine_args=engine_args[0],
                global_engine_args_equal=True, common_environment_equal=environment_equal,
                extraction_checks=extraction_checks,
                actual_kernel_configs=list({json.dumps(r["actual_kernel_config"], sort_keys=True):
                                            r["actual_kernel_config"] for r in records}.values()),
                summary=groups, comparisons=comparisons, dispatch_records=records,
                boundaries=["Input comparisons reconstruct the frozen fixture row using retained source indices; per-call input tensors were not separately exported.",
                            "Repeat comparisons and tensor elements are not independent text samples; only two target rows were tested.",
                            "GPU-selected IDs and weights are observed; no CPU top-k reconstruction is used.",
                            "Expert counts and sorted locations are dispatch metadata, not measured HBM traffic, congestion or request benefit.",
                            "Prefill-end activations and isolated layer calls do not establish decode trajectory, task semantics, native graph-serving or performance effects."])


def metrics_text(result):
    lines = [f"Verdict: `{result['verdict']}`", "",
             f"Measured calls: {result['measured_calls']}; distinct target rows: {result['distinct_targets']}.", "",
             "| Contrast | Comparisons | Input differs | Router differs | Weights differ | IDs differ | Output differs | Max output abs | Counts changed | Absolute locations changed | Internal rows changed | Kernel config changed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind, row in result["summary"].items():
        differences = " | ".join(str(row["different"][name]) for name in FIELDS)
        lines.append(f"| {kind} | {row['comparisons']} | {differences} | {row['max_abs']['output']:.8g} | {row['expert_counts_changed']} | {row['target_sorted_locations_changed']} | {row['target_internal_row_changed']} | {row['actual_kernel_config_changed']} |")
    lines += ["", "Counts are comparisons, not independent samples. All per-repeat contrasts and exact dispatch counts/locations are in analysis.json.", "",
              "| Process | Target | Comparison | Output differing repeats | Max output abs |",
              "|---:|---:|---|---:|---:|"]
    for process in range(result["global_config"]["processes"]):
        for target in result["global_config"]["target_source_indices"]:
            for kind in ("original_vs_permuted", "original_vs_replaced"):
                rows = [row for row in result["comparisons"] if row["kind"] == kind and row["left_process"] == process and row["target"] == target]
                summary = summarize(rows)
                lines.append(f"| {process} | {target} | {kind} | {summary['different']['output']}/{len(rows)} | {summary['max_abs']['output']:.8g} |")
    lines += ["", *[f"- {boundary}" for boundary in result["boundaries"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.root / "analysis"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    result = analyze(args.root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "metrics.md").write_text(metrics_text(result))
    print(json.dumps(dict(verdict=result["verdict"], summary=result["summary"]), indent=2))


if __name__ == "__main__":
    main()
