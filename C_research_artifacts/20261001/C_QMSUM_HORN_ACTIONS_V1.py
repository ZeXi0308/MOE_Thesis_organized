#!/usr/bin/env python3
"""Actual first-admission, cache and service actions for QMSum Horn/whole."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from C_QMSUM_ANALYZE_V1 import load, sha
from C_QMSUM_CHUNK_ACTIONS_V1 import cell

HERE = Path(__file__).resolve().parent
RECEIPTS = {"horn": HERE / "qmsum_horn_compact_v1.json",
            "whole": HERE / "qmsum_density_order_v1.json"}


def common_prefix(a, b):
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def load_arm(directory: Path, mode: str):
    data = cell(directory, 512)
    receipt = load(RECEIPTS[mode])
    config = load(directory / "config.json")
    if (receipt["request_count"] != 200
            or receipt["workload_sha256"] != config["workload_sha256"]
            or data["frozen_order"] != receipt["source_indices_in_submission_order"]):
        raise ValueError(f"{mode}: frozen submission receipt differs")
    observed = [data["outputs"][rid]["source_index"] for rid in data["first_order"]]
    if observed != receipt["source_indices_in_submission_order"]:
        raise ValueError(f"{mode}: actual first-successful-allocation order differs")
    mix = load(directory / "measured-service-mix.json")
    steps = load(directory / "measured-steps.json")
    by_index = {data["outputs"][rid]["source_index"]: data["outputs"][rid]
                for rid in data["first_order"]}
    source_by_external = {row["external_request_id"]: index
                          for index, row in by_index.items()}
    first = {source_by_external[row["external_request_id"]]: row
             for row in mix["first_successful_allocation"]}
    if len(first) != 200 or set(first) != set(range(200)):
        raise ValueError(f"{mode}: first allocation request inventory differs")
    return dict(data=data, receipt=receipt, config=config, by_index=by_index,
                first=first, steps=steps, receipt_sha256=sha(RECEIPTS[mode]),
                config_sha256=sha(directory / "config.json"))


def historical_prefix_witnesses(arm, other_first, *, limit=5):
    """Prior prompt computation before a target call, never live cache residency."""
    rows, first, calls = arm["by_index"], arm["first"], arm["steps"]["scheduler_calls"]
    candidates = []
    for index, target in first.items():
        actual = target["new_prefix_cached_tokens"]
        other_hit = other_first[index]["new_prefix_cached_tokens"]
        if actual >= other_hit:
            continue
        call = target["schedule_call"]
        host_s = calls[call]["host_s"]
        best = None
        for prior_index, prior in rows.items():
            if prior_index == index or prior["token_times_s"][0] >= host_s:
                continue
            shared = 16 * (min(common_prefix(rows[index]["prompt_token_ids"],
                                             prior["prompt_token_ids"]),
                               len(rows[index]["prompt_token_ids"]) - 1) // 16)
            if shared <= actual:
                continue
            candidate = dict(target_source_index=index, prior_source_index=prior_index,
                target_first_allocation_call=call,
                target_schedule_host_return_s=host_s,
                earlier_request_first_token_host_return_s=prior["token_times_s"][0],
                block_aligned_shared_prompt_tokens=shared,
                target_observed_first_cache_hit_tokens=actual,
                target_other_arm_first_cache_hit_tokens=other_hit,
                historical_common_prefix_minus_observed_hit_tokens=shared-actual)
            if best is None or candidate["historical_common_prefix_minus_observed_hit_tokens"] > best["historical_common_prefix_minus_observed_hit_tokens"]:
                best = candidate
        if best is not None:
            candidates.append(best)
    return sorted(candidates, key=lambda x: (-x["historical_common_prefix_minus_observed_hit_tokens"],
                                              x["target_source_index"]))[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horn-dir", type=Path, required=True)
    ap.add_argument("--whole-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    arms = {"horn": load_arm(args.horn_dir, "horn"),
            "whole": load_arm(args.whole_dir, "whole")}
    horn, whole = arms["horn"], arms["whole"]
    if (horn["config_sha256"] != whole["config_sha256"]
            or horn["data"]["environment"]["gpu_before"]["gpu"].split(",")[0]
               != whole["data"]["environment"]["gpu_before"]["gpu"].split(",")[0]):
        raise ValueError("pair input config or GPU differs")
    positions = {mode: {source: position for position, source in enumerate(
        arm["receipt"]["source_indices_in_submission_order"])}
        for mode, arm in arms.items()}
    per_request = []
    for index in range(200):
        a, b = horn["by_index"][index], whole["by_index"][index]
        if a["prompt_token_ids"] != b["prompt_token_ids"] or a["request_id"] != b["request_id"]:
            raise ValueError(f"input prompt differs at source index {index}")
        per_request.append(dict(source_index=index,
            horn_order_position=positions["horn"][index],
            whole_order_position=positions["whole"][index],
            position_delta_horn_minus_whole=positions["horn"][index]-positions["whole"][index],
            horn_first_allocation_call=horn["first"][index]["schedule_call"],
            whole_first_allocation_call=whole["first"][index]["schedule_call"],
            horn_first_cache_hit_tokens=horn["first"][index]["new_prefix_cached_tokens"],
            whole_first_cache_hit_tokens=whole["first"][index]["new_prefix_cached_tokens"],
            output_ids_equal=a["output_token_ids"] == b["output_token_ids"],
            horn_output_tokens=len(a["output_token_ids"]),
            whole_output_tokens=len(b["output_token_ids"]),
            horn_finish_reason=a["finish_reason"], whole_finish_reason=b["finish_reason"]))
    metrics = {mode: arm["data"]["metrics"] for mode, arm in arms.items()}
    selected = ("schedule_calls", "scheduled_tokens", "initial_prompt_compute_tokens",
                "prefill_recompute_tokens_relative_to_first_cache_hit",
                "first_allocation_cached_prefix_tokens", "peak_running", "peak_used_kv_blocks",
                "preemptions", "allocation_failures", "output_tokens", "observation_end_s")
    numeric_delta = {key: metrics["horn"][key]-metrics["whole"][key]
                     for key in selected if isinstance(metrics["horn"][key], (int, float))}
    for name in ("mixed", "decode_only", "prefill_only"):
        numeric_delta[name + "_steps"] = (metrics["horn"]["schedule_type_counts"].get(name, 0)
                                         - metrics["whole"]["schedule_type_counts"].get(name, 0))
    for name in ("prefill_tokens", "decode_tokens", "scheduled_tokens"):
        numeric_delta[name] = (metrics["horn"]["scheduled_tokens"].get(name, 0)
                               - metrics["whole"]["scheduled_tokens"].get(name, 0))
    witnesses = {mode: historical_prefix_witnesses(arms[mode],
        arms["whole" if mode == "horn" else "horn"]["first"])
        for mode in ("horn", "whole")}
    result = dict(schema="c-qmsum-horn-whole-native512-actions-v1",
        scope="One same-host classical Horn then density-whole development pair, fixed native512; actual scheduler work and outputs only",
        order_receipt_sha256={mode: arm["receipt_sha256"] for mode, arm in arms.items()},
        raw_sha256={mode: arm["data"]["raw_sha256"] for mode, arm in arms.items()},
        actual_first_allocation_order={mode: arm["receipt"]["source_indices_in_submission_order"]
                                       for mode, arm in arms.items()},
        arms=metrics, horn_minus_whole=numeric_delta,
        request_order_positions_changed=sum(x["position_delta_horn_minus_whole"] != 0
                                            for x in per_request),
        output_ids_changed=sum(not x["output_ids_equal"] for x in per_request),
        output_lengths_changed=sum(x["horn_output_tokens"] != x["whole_output_tokens"]
                                   for x in per_request),
        finish_reasons_changed=sum(x["horn_finish_reason"] != x["whole_finish_reason"]
                                   for x in per_request),
        per_request=per_request, historical_computed_prefix_witnesses=witnesses,
        interpretation_limits=[
            "A prior request's first host token return shows its prompt was computed before a later allocation call; shared prompt tokens do not prove cache residency or an exact eviction event.",
            "First cache hits, scheduled prefill and recomputation are actual host-observed scheduler fields. They are not GPU-kernel time.",
            "Different output token IDs or lengths change decode work; whole-episode comparisons are not equal-work causal estimates."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(horn_minus_whole=numeric_delta,
        request_order_positions_changed=result["request_order_positions_changed"],
        output_ids_changed=result["output_ids_changed"],
        historical_prefix_witnesses={k: len(v) for k, v in witnesses.items()}),
        sort_keys=True))


if __name__ == "__main__":
    main()
