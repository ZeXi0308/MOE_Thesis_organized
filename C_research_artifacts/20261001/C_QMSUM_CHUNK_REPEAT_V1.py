#!/usr/bin/env python3
"""Describe forward and reverse QMSum native-budget development pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from C_QMSUM_ANALYZE_V1 import load, sha
from C_QMSUM_CHUNK_DECOMPOSE_V1 import cell as raw_cell


def arm(root: Path, budget: str, analysis_path: Path):
    native = root / budget / "native"
    raw = raw_cell(native)
    analysis = load(analysis_path)
    if (analysis.get("status") != "COMPLETE" or analysis.get("requests_completed") != 200
            or raw["resolved"]["max_num_batched_tokens"] != int(budget)
            or analysis["original_run_sha256"]["measured-outputs.json"]
               != raw["raw_sha256"]["measured-outputs.json"]
            or analysis["original_run_sha256"]["measured-service-mix.json"]
               != raw["raw_sha256"]["measured-service-mix.json"]):
        raise ValueError(f"analysis/raw/budget differs: {analysis_path}")
    receipt = load(root / budget / "launcher-receipt.json")
    if (receipt.get("status") != "COMPLETE" or receipt.get("cell") != budget
            or receipt["output_qualification"]["status"] != "QUALIFIED"):
        raise ValueError(f"cell launcher incomplete: {root / budget}")
    metrics = dict(batch_budget=int(budget),
        mean_completion_s=analysis["flow_s"]["mean"],
        p95_completion_s=analysis["flow_s"]["p95_nearest_rank"],
        mean_ttft_s=analysis["ttft_s"]["mean"],
        p95_ttft_s=analysis["ttft_s"]["p95_nearest_rank"],
        mean_max_host_gap_s=analysis["max_distinct_host_return_gap_s"]["mean"],
        p95_max_host_gap_s=analysis["max_distinct_host_return_gap_s"]["p95_nearest_rank"],
        largest_host_gap_s=analysis["max_distinct_host_return_gap_s"]["maximum"],
        episode_s=raw["status"]["observation_end_s"],
        output_tokens=raw["status"]["output_tokens"],
        finish_reasons=raw["status"]["finish_reason_counts"],
        rouge_l_f1_percent=analysis["rouge_l_f1_percent"],
        schedule_calls=raw["status"]["schedule_calls"],
        scheduled_prefill_tokens=analysis["service_mix"]["classified_scheduled_prefill_tokens"],
        scheduled_decode_tokens=analysis["service_mix"]["classified_scheduled_decode_tokens"],
        first_allocation_cached_prefix_tokens=analysis["service_mix"]["first_allocation_prefix_cached_tokens"],
        preemptions=raw["status"]["preemptions"],
        peak_running=raw["status"]["peak_running_after_schedule"],
        peak_used_kv_blocks=raw["status"]["peak_used_blocks_after_schedule"],
        allocation_failures=raw["status"]["allocation_failure_count"])
    return dict(raw=raw, analysis=analysis, metrics=metrics,
                analysis_sha256=sha(analysis_path),
                launcher_sha256=sha(root / budget / "launcher-receipt.json"),
                gpu_uuid=receipt["gpu_before"]["gpu_uuid"])


def repeat_comparison(a, b):
    left, right = a["raw"], b["raw"]
    moved = sum(x != y for x, y in zip(left["first"], right["first"]))
    ids_changed = sum(left["outputs"][i]["output_token_ids"]
                      != right["outputs"][i]["output_token_ids"] for i in range(200))
    length_changed = sum(len(left["outputs"][i]["output_token_ids"])
                         != len(right["outputs"][i]["output_token_ids"])
                         for i in range(200))
    finish_changed = sum(left["outputs"][i]["finish_reason"]
                         != right["outputs"][i]["finish_reason"] for i in range(200))
    return dict(output_ids_changed=ids_changed, output_lengths_changed=length_changed,
        finish_reasons_changed=finish_changed,
        first_admission_order_identical=left["first"] == right["first"],
        first_admission_position_changed=moved,
        scheduled_batch_signature_identical=left["signature"] == right["signature"],
        first_admission_sha256={"forward": left["first_sha256"],
                                "reverse": right["first_sha256"]},
        scheduled_batch_signature_sha256={"forward": left["schedule_sha256"],
                                          "reverse": right["schedule_sha256"]},
        output_ids_sha256={"forward": left["output_ids_sha256"],
                           "reverse": right["output_ids_sha256"]})


def pair_delta(arms):
    small, large = arms["512"]["metrics"], arms["1024"]["metrics"]
    keys = ("mean_completion_s", "p95_completion_s", "mean_ttft_s", "p95_ttft_s",
            "mean_max_host_gap_s", "p95_max_host_gap_s", "largest_host_gap_s",
            "episode_s", "output_tokens", "rouge_l_f1_percent", "schedule_calls",
            "scheduled_prefill_tokens", "scheduled_decode_tokens", "preemptions",
            "peak_running", "peak_used_kv_blocks", "allocation_failures")
    return {key: small[key] - large[key] for key in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward-dir", type=Path, required=True)
    ap.add_argument("--reverse-dir", type=Path, required=True)
    for pair in ("forward", "reverse"):
        for budget in ("512", "1024"):
            ap.add_argument(f"--{pair}-{budget}-analysis", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    roots = {"forward": args.forward_dir, "reverse": args.reverse_dir}
    expected_order = {"forward": ["512", "1024"], "reverse": ["1024", "512"]}
    arms, pair_receipts = {}, {}
    for pair, root in roots.items():
        receipt_path = root / "pair-receipt.json"
        receipt = load(receipt_path)
        if (receipt.get("status") != "COMPLETE" or receipt.get("run_order") != expected_order[pair]
                or len(receipt.get("cells", [])) != 2
                or any(c.get("status") != "COMPLETE" for c in receipt["cells"])):
            raise ValueError(f"{pair} pair receipt incomplete or wrong order")
        pair_receipts[pair] = sha(receipt_path)
        arms[pair] = {budget: arm(root, budget,
            getattr(args, f"{pair}_{budget}_analysis")) for budget in ("512", "1024")}
    all_arms = [arms[p][b] for p in ("forward", "reverse") for b in ("512", "1024")]
    if (len({x["gpu_uuid"] for x in all_arms}) != 1
            or len({tuple(x["raw"]["submission"]) for x in all_arms}) != 1
            or len({json.dumps(x["analysis"]["frozen_input_sha256"], sort_keys=True)
                    for x in all_arms}) != 1):
        raise ValueError("GPU, frozen source or input submission differs across pairs")
    for i in range(200):
        prompts = {tuple(x["raw"]["outputs"][i]["prompt_token_ids"]) for x in all_arms}
        if len(prompts) != 1:
            raise ValueError(f"source prompt differs at index {i}")
    result = dict(schema="c-qmsum-native-chunk-repeat-v1",
        scope="Two whole same-host development pairs, forward 512→1024 then reverse 1024→512; no pooled request-level replication or causal speedup claim",
        gpu_uuid=all_arms[0]["gpu_uuid"], pair_receipt_sha256=pair_receipts,
        arms={pair: {budget: dict(metrics=arms[pair][budget]["metrics"],
            raw_sha256=arms[pair][budget]["raw"]["raw_sha256"],
            analysis_sha256=arms[pair][budget]["analysis_sha256"],
            launcher_sha256=arms[pair][budget]["launcher_sha256"])
            for budget in ("512", "1024")} for pair in ("forward", "reverse")},
        within_pair_512_minus_1024={pair: pair_delta(arms[pair])
                                    for pair in ("forward", "reverse")},
        same_budget_forward_vs_reverse={budget: repeat_comparison(
            arms["forward"][budget], arms["reverse"][budget])
            for budget in ("512", "1024")},
        interpretation_limits=[
            "Each pair contributes one whole-episode contrast on the same 200 requests; request rows are not independent pair replicates or a confidence interval.",
            "Scheduled-batch signature is the ordered per-call external request ID, scheduled/prefill/decode tokens, previous computed tokens and newly cached prefix tokens; internal randomized IDs and host timestamps are excluded.",
            "Outputs may differ, changing decode work; budget effects cannot be read as equal-output-length latency effects."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(within_pair_512_minus_1024=result["within_pair_512_minus_1024"],
                          same_budget_forward_vs_reverse=result["same_budget_forward_vs_reverse"]),
                     sort_keys=True))


if __name__ == "__main__":
    main()
