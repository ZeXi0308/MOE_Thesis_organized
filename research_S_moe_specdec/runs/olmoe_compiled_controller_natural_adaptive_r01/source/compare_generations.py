#!/usr/bin/env python3
"""CPU-only, source-index-aligned generation accounting and fixed-sample inspection.

python code/compare_generations.py --workload configs/sharegpt_olmoe.json \
    --run natural=runs/olmoe_natural_smoke_r01 --output runs/generation_check_r01 \
    --tokenizer /root/autodl-tmp/moe-research-20261002/model

--run LABEL=PATH is repeatable and accepts a campaign, gamma directory, or one
perf-rN-bB.json. All selected cells enter summaries. Human inspection always
uses up to eight consecutive workload records starting at --sample-workload-offset
(default zero), minimum selected batch (or --sample-batch), and repeat zero
(or --sample-repeat), independently of generated outcomes. The offset is a
zero-based workload position, not a source_index value.
Without a local tokenizer, outputs retain complete token IDs. No model is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
import statistics
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PERF = re.compile(r"perf-r(\d+)-b(\d+)\.json$")


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def stats(values):
    return {"count": len(values), "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "max": max(values) if values else None}


def termination(row, natural, eos_ids):
    reason, stop = row.get("finish_reason"), row.get("stop_reason")
    ids = row.get("output_token_ids", [])
    if row.get("error") or reason in ("abort", "error", "failed", "cancelled", "canceled"):
        return "failed", "explicit error/abort/cancellation in request record"
    if row.get("finished") is not True:
        return "unfinished", "request is not marked finished"
    if reason == "length":
        return "length_limit", "engine finish_reason=length; output cap and context limit may both cause this"
    if reason == "stop" and natural:
        if (isinstance(stop, int) and stop in eos_ids) or (ids and ids[-1] in eos_ids):
            return "natural_eos_confirmed", "EOS ID appears in stop_reason or final returned token"
        if stop is None:
            return "natural_eos_inferred", "run_matrix configures no custom stops; stop with null stop_reason is inferred EOS"
    if reason == "stop":
        return "other_stop", "preserved engine stop; not classified as confirmed EOS"
    return "finished_unspecified", "finished without a recognized termination reason"


def discover(label, root, batch_filter):
    """Include expected but absent cells; absence is not a request failure."""
    if root.is_file():
        paths, fallback_plan, fallback_args = [root], {}, {}
    elif (root / "configuration.json").is_file():
        cfg, _ = read_json(root / "configuration.json")
        fallback_plan, fallback_args = cfg["plan"], cfg.get("args", {})
        batches = fallback_args.get("batches") or fallback_plan["batches"]
        repeats = fallback_args.get("repeats") or fallback_plan["repeats"]
        paths = [root / f"perf-r{r}-b{b}.json" for r in range(repeats) for b in batches]
    elif (root / "plan.json").is_file():
        fallback_plan, _ = read_json(root / "plan.json")
        state = read_json(root / "status.json")[0] if (root / "status.json").is_file() else {}
        fallback_args = state.get("args", {})
        batches = fallback_args.get("batches") or fallback_plan["batches"]
        repeats = fallback_args.get("repeats") or fallback_plan["repeats"]
        gammas = fallback_args.get("gammas") or fallback_plan["gammas"]
        paths = [root / f"gamma{g}" / f"perf-r{r}-b{b}.json" for g in gammas for r in range(repeats) for b in batches]
    else:
        paths, fallback_plan, fallback_args = sorted(root.rglob("perf-r*-b*.json")), {}, {}
    jobs = []
    for path in paths:
        match = PERF.fullmatch(path.name)
        if not match or (batch_filter and int(match[2]) not in batch_filter):
            continue
        config_path = path.parent / "configuration.json"
        if config_path.is_file():
            cfg, config_hash = read_json(config_path)
        else:
            cfg, config_hash = {"plan": fallback_plan, "args": fallback_args}, None
        jobs.append({"run": label, "path": path, "repeat": int(match[1]), "batch": int(match[2]),
                     "configuration": cfg, "configuration_sha256": config_hash,
                     "variant": label + ":" + (str(path.parent.relative_to(root)) if root.is_dir() and path.parent != root else path.parent.name)})
    if not jobs:
        raise ValueError(f"no matching performance cells found or declared under {root}")
    failures = []
    if root.is_dir():
        for path in sorted(root.rglob("failure.json")):
            data, digest = read_json(path)
            failures.append({"run": label, "path": str(path), "sha256": digest, "error": data.get("error")})
    return jobs, failures


def load_cell(job, workload, workload_hash, eos_ids):
    cfg, path = job["configuration"], job["path"]
    if cfg.get("workload_sha256") and cfg["workload_sha256"] != workload_hash:
        raise ValueError(f"fixed workload hash differs from recorded configuration: {path}")
    plan, args = cfg.get("plan", {}), cfg.get("args", {})
    payload, artifact_hash = read_json(path) if path.is_file() else ({}, None)
    population = plan.get("measured_requests", payload.get("population"))
    offset = plan.get("input_offset", 0)
    if population is None:
        raise ValueError(f"cannot establish expected population for {path}")
    expected_records = workload["records"][offset:offset + population]
    if len(expected_records) != population:
        raise ValueError(f"workload does not cover declared population in {path}")
    expected = [r["source_index"] for r in expected_records]
    kind = payload.get("measurement_kind", "natural_generation" if args.get("natural_stop") else "fixed_length_generation")
    natural = args.get("natural_stop", kind == "natural_generation")
    if "natural_stop" in args and ((kind == "natural_generation") != bool(natural)) and kind != "route_diagnostic":
        raise ValueError(f"measurement kind and natural_stop disagree: {path}")
    gamma_match = re.fullmatch(r"gamma(\d+)", path.parent.name)
    metadata = {"model": payload.get("model", plan.get("model")), "model_revision": plan.get("model_revision"),
                "draft_source": payload.get("draft_source", plan.get("draft_source")),
                "gamma": payload.get("gamma", args.get("gamma", int(gamma_match[1]) if gamma_match else None)),
                "mode": payload.get("mode", args.get("mode", "native")),
                "proposal_budget": payload.get("proposal_budget", args.get("proposal_budget")),
                "measurement_kind": kind, "natural_stop": natural,
                "execution_mode": payload.get("execution_mode", "eager" if plan.get("enforce_eager", True) else "compiled"),
                "output_cap": args.get("output_tokens") or plan.get("output_tokens"),
                "batch": job["batch"], "repeat": job["repeat"]}
    source_map = {r["source_index"]: r for r in expected_records}
    rows, counts, lengths = {}, Counter(), []
    for row in payload.get("requests", []):
        index, ids = row.get("source_index"), row.get("output_token_ids")
        if index not in source_map or index in rows:
            raise ValueError(f"unknown or duplicate source_index={index!r}: {path}")
        if not isinstance(ids, list) or not all(type(x) is int and x >= 0 for x in ids):
            raise ValueError(f"invalid token IDs for source_index={index}: {path}")
        if row.get("prompt_tokens") != len(source_map[index]["prompt_token_ids"]):
            raise ValueError(f"prompt length differs from fixed workload for source_index={index}: {path}")
        state, evidence = termination(row, natural, eos_ids)
        counts[state] += 1
        lengths.append(len(ids))
        rows[index] = {"source_index": index, "request_id": row.get("request_id"),
                       "output_token_ids": ids, "output_tokens": len(ids), "finished": row.get("finished"),
                       "finish_reason": row.get("finish_reason"), "stop_reason": row.get("stop_reason"),
                       "termination": state, "termination_evidence": evidence, "error": row.get("error"),
                       "at_configured_output_cap": len(ids) == metadata["output_cap"] if metadata["output_cap"] else None}
    missing = [i for i in expected if i not in rows]
    for state in ("natural_eos_confirmed", "natural_eos_inferred", "length_limit", "failed", "unfinished", "other_stop", "finished_unspecified"):
        counts.setdefault(state, 0)
    return {"id": f"{job['variant']}/r{job['repeat']}/b{job['batch']}", "run": job["run"],
            "variant": job["variant"], "path": str(path), "artifact_sha256": artifact_hash,
            "configuration_sha256": job["configuration_sha256"], "metadata": metadata,
            "artifact_status": "recorded" if artifact_hash else "not_recorded",
            "expected_population": population, "expected_source_indices": expected,
            "observed_requests": len(rows), "missing_source_indices": missing,
            "unobserved_requests": len(missing),
            "termination_counts": dict(counts), "output_length_tokens": stats(lengths),
            "reported_all_finished": payload.get("all_finished"), "rows": rows}


def compare_pair(left, right, kind):
    details, prefixes, exact = [], [], 0
    for index in left["expected_source_indices"]:
        a, b = left["rows"].get(index), right["rows"].get(index)
        item = {"source_index": index, "left_present": a is not None, "right_present": b is not None}
        if a is not None and b is not None:
            ai, bi = a["output_token_ids"], b["output_token_ids"]
            prefix = 0
            for x, y in zip(ai, bi):
                if x != y:
                    break
                prefix += 1
            complete = a["finished"] is True and b["finished"] is True and a["termination"] != "failed" and b["termination"] != "failed"
            item.update(token_sequence_equal=ai == bi, common_prefix_tokens=prefix,
                        shorter_sequence_is_prefix=prefix == min(len(ai), len(bi)),
                        left_length=len(ai), right_length=len(bi), both_successfully_finished=complete,
                        termination_equal=(a["finish_reason"], a["stop_reason"]) == (b["finish_reason"], b["stop_reason"]))
            if complete:
                prefixes.append(prefix)
                exact += ai == bi
        details.append(item)
    return {"kind": kind, "left": left["id"], "right": right["id"],
            "expected_population": left["expected_population"], "both_successfully_finished": len(prefixes),
            "exact_token_sequences": exact, "exact_fraction_among_completed_pairs": exact / len(prefixes) if prefixes else None,
            "common_prefix_tokens_among_completed_pairs": stats(prefixes), "per_source": details}


def comparisons(cells):
    result, exclusions = [], Counter()
    for left, right in combinations(cells, 2):
        a, b = left["metadata"], right["metadata"]
        if a["batch"] != b["batch"]:
            continue
        same_config = left["variant"] == right["variant"] and left["configuration_sha256"] == right["configuration_sha256"]
        if same_config and a["repeat"] != b["repeat"]:
            kind = "same_configuration_repeats"
        elif left["variant"] != right["variant"] and a["repeat"] == b["repeat"]:
            kind = "cross_strategy_or_configuration"
        else:
            continue
        if left["artifact_status"] != "recorded" or right["artifact_status"] != "recorded":
            exclusions["one_or_both_artifacts_not_recorded"] += 1
            continue
        compatible = (a["model_revision"] or a["model"]) == (b["model_revision"] or b["model"])
        compatible &= all(a[key] == b[key] for key in ("measurement_kind", "natural_stop", "output_cap"))
        compatible &= left["expected_source_indices"] == right["expected_source_indices"]
        if not compatible:
            exclusions["different_model_population_or_generation_contract"] += 1
            continue
        result.append(compare_pair(left, right, kind))
    return result, dict(exclusions)


def fixed_samples(workload, cells, sample_batch, sample_repeat, tokenizer, sample_workload_offset=0):
    if (type(sample_workload_offset) is not int
            or not 0 <= sample_workload_offset < len(workload["records"])):
        raise ValueError("sample workload offset must identify an existing zero-based workload record")
    variants = sorted({c["variant"] for c in cells})
    selected = {(c["variant"], c["metadata"]["batch"], c["metadata"]["repeat"]): c for c in cells}
    samples = []
    for record in workload["records"][sample_workload_offset:sample_workload_offset + 8]:
        sample = {"source_index": record["source_index"], "source_id": record.get("source_id"),
                  "prompt": record.get("prompt"), "prompt_token_ids": record["prompt_token_ids"],
                  "reference": record.get("reference"), "references": record.get("references"), "outputs": []}
        for variant in variants:
            cell = selected.get((variant, sample_batch, sample_repeat))
            row = cell["rows"].get(record["source_index"]) if cell else None
            out = {"variant": variant, "batch": sample_batch, "repeat": sample_repeat,
                   "cell_id": cell["id"] if cell else None,
                   "status": "recorded" if row else "not_in_population" if cell and record["source_index"] not in cell["expected_source_indices"] else "not_recorded"}
            if row:
                out.update(row)
                if tokenizer:
                    try:
                        out["text"] = tokenizer.decode(row["output_token_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    except Exception as error:
                        out["decode_error"] = f"{type(error).__name__}: {error}"
            sample["outputs"].append(out)
        samples.append(sample)
    return samples


def fenced(value):
    value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    fence = "`" * max(3, 1 + max((len(x) for x in re.findall(r"`+", value)), default=0))
    return fence + "text\n" + value + "\n" + fence


def markdown(report, samples):
    lines = ["# Generation semantics and fixed-sample inspection", "", report["interpretation"], "",
             "Token differences are descriptive. They do not by themselves establish lower quality, a sampler bug, or loss of distributional correctness.", "",
             "Missing artifacts/requests are reported as unobserved, not automatically failed. Run-level errors appear separately.", "",
             "| Cell | Artifact | Observed/expected | Termination counts | Output length min/median/max |", "|---|---|---:|---|---|"]
    for cell in report["cells"]:
        length = cell["output_length_tokens"]
        lines.append(f"| {cell['id']} | {cell['artifact_status']} | {cell['observed_requests']}/{cell['expected_population']} | {json.dumps(cell['termination_counts'])} | {length['min']}/{length['median']}/{length['max']} |")
    lines += ["", "## Paired token comparison", "", "Exact-match denominators include only pairs recorded as finished without an explicit failure. Missing and unfinished cases remain in the JSON per-source records.", "",
              "| Kind | Left | Right | Exact/completed pairs | Mean shared-prefix tokens |", "|---|---|---|---:|---:|"]
    for pair in report["comparisons"]:
        lines.append(f"| {pair['kind']} | {pair['left']} | {pair['right']} | {pair['exact_token_sequences']}/{pair['both_successfully_finished']} | {pair['common_prefix_tokens_among_completed_pairs']['mean']} |")
    offset = report["sample_workload_offset"]
    lines += ["", "## Fixed source samples", "", f"Original workload positions [{offset}, {offset + len(samples)}) (zero-based); batch {report['sample_batch']}, repeat {report['sample_repeat']}; selection does not depend on generated outputs. No automatic quality grade is assigned."]
    for sample in samples:
        lines += ["", f"### Source index {sample['source_index']}", "", "Prompt:", "", fenced(sample["prompt"])]
        reference = sample["references"] if sample["references"] is not None else sample["reference"]
        lines += ["", "Reference (if supplied):", "", fenced(reference)]
        for out in sample["outputs"]:
            lines += ["", f"**{out['variant']}** — {out['status']}", ""]
            if out["status"] == "recorded":
                lines += [f"Length: {out['output_tokens']}; termination: {out['termination']}; finish_reason={out['finish_reason']!r}; stop_reason={out['stop_reason']!r}.", "",
                          out["termination_evidence"], "", fenced(out.get("text", out["output_token_ids"]))]
    if report["run_failures"]:
        lines += ["", "## Recorded run-level errors", "", fenced(report["run_failures"])]
    if report["input_errors"]:
        lines += ["", "## Invalid or unreadable inputs", "", fenced(report["input_errors"])]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing files are never overwritten")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--eos-token-id", action="append", type=int, default=[])
    parser.add_argument("--batches", nargs="+", type=int)
    parser.add_argument("--sample-batch", type=int)
    parser.add_argument("--sample-repeat", type=int, default=0)
    parser.add_argument("--sample-workload-offset", type=int, default=0,
                        help="zero-based workload position starting the fixed window of up to eight records; not a source_index")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    workload, workload_hash = read_json(args.workload)
    indices = [record["source_index"] for record in workload["records"]]
    if len(indices) != len(set(indices)) or not indices:
        raise ValueError("workload must contain unique source_index values")
    tokenizer, eos_ids = None, set(args.eos_token_id)
    if args.tokenizer:
        if not args.tokenizer.is_dir():
            raise ValueError("--tokenizer must be a local directory")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
        token_eos = tokenizer.eos_token_id
        eos_ids.update(token_eos if isinstance(token_eos, list) else [token_eos] if token_eos is not None else [])
    cells, failures, errors, seen_labels = [], [], [], set()
    for value in args.run:
        label, sep, raw_path = value.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or label in seen_labels:
            raise ValueError("each --run needs a unique safe LABEL=PATH")
        seen_labels.add(label)
        jobs, run_failures = discover(label, Path(raw_path).resolve(), args.batches)
        failures.extend(run_failures)
        for job in jobs:
            try:
                cells.append(load_cell(job, workload, workload_hash, eos_ids))
            except (ValueError, KeyError, OSError) as error:
                errors.append({"path": str(job["path"]), "error": f"{type(error).__name__}: {error}"})
    if not cells:
        raise ValueError(f"no valid cells: {errors}")
    pairs, exclusions = comparisons(cells)
    sample_batch = args.sample_batch if args.sample_batch is not None else min(c["metadata"]["batch"] for c in cells)
    samples = fixed_samples(workload, cells, sample_batch, args.sample_repeat, tokenizer,
                            args.sample_workload_offset)
    report = {"schema_version": 1, "workload": str(args.workload.resolve()), "workload_sha256": workload_hash,
              "sample_selection": "up to eight consecutive workload records from the explicit zero-based offset, without outcome filtering",
              "sample_workload_offset": args.sample_workload_offset,
              "sample_source_indices": [sample["source_index"] for sample in samples],
              "sample_count": len(samples),
              "sample_batch": sample_batch, "sample_repeat": args.sample_repeat,
              "decoded_with": str(args.tokenizer.resolve()) if args.tokenizer else None, "eos_token_ids": sorted(eos_ids),
              "interpretation": "This is a termination/workload and paired-token inspection, not a benchmark quality score or a distributional-equivalence proof. Fixed-length generation is not natural task completion. Repeated requests within one run are not independent system repetitions.",
              "eos_inference_rule": "Natural stop with null stop_reason is inferred EOS only under run_matrix's no-custom-stop configuration; confirmed token-ID evidence is counted separately.",
              "cells": [{k: v for k, v in c.items() if k != "rows"} for c in cells],
              "comparisons": pairs, "comparison_exclusions": exclusions, "run_failures": failures, "input_errors": errors}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("summary.json", report), ("samples.json", {"selection": report["sample_selection"], "sample_workload_offset": args.sample_workload_offset, "sample_count": len(samples), "sample_batch": sample_batch, "sample_repeat": args.sample_repeat, "samples": samples})):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "inspection.md").write_text(markdown(report, samples))
    print(json.dumps({"output": str(args.output), "cells": len(cells), "pairs": len(pairs), "fixed_samples": len(samples), "input_errors": len(errors)}))
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
