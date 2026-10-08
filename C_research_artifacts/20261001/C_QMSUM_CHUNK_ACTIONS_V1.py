#!/usr/bin/env python3
"""Compare actual service and warmup geometry of one QMSum 512/1024 pair."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from C_QMSUM_ANALYZE_V1 import distribution, load, sha

N = 200


def cell(directory: Path, budget: int):
    names = ("status.json", "engine_args.json", "resolved-scheduler.json",
             "environment.json", "measured-steps.json", "measured-service-mix.json",
             "measured-outputs.json", "warmup-summary.json", "shape-warmup-source.json",
             "timing.json", "prefix-cache-reset.json")
    docs = {name: load(directory / name) for name in names}
    status, args, resolved, environment, trace, mix, outputs, warmup, shape, timing, reset = (
        docs[name] for name in names)
    if (status.get("status") != "COMPLETE" or status.get("request_count") != N
            or args.get("max_num_batched_tokens") != budget
            or resolved.get("max_num_batched_tokens") != budget
            or resolved.get("max_num_seqs") != 128 or resolved.get("usable_kv_blocks") != 4096
            or reset.get("reset_succeeded") is not True
            or shape.get("native_batch_budget") != budget
            or [r.get("prompt_tokens") for r in shape.get("shapes", [])] != [511, 1023]
            or len(outputs) != N):
        raise ValueError(f"budget {budget}: native cell or common warmup differs")
    steps, schedules, service = (trace["steps"], trace["scheduler_calls"],
                                 mix["scheduler_calls"])
    if not len(steps) == len(schedules) == len(service) == status["schedule_calls"]:
        raise ValueError(f"budget {budget}: step/schedule/service counts differ")
    per_step, by_type = [], defaultdict(list)
    type_counts, token_totals = Counter(), Counter()
    for i, (step, scheduled, classified) in enumerate(zip(steps, schedules, service)):
        if (scheduled.get("call") != i or classified.get("call") != i
                or not step["start_s"] <= scheduled["host_s"] <= step["return_s"]
                or scheduled["scheduled_tokens_total"] != classified["scheduled_tokens"]
                or classified["scheduled_tokens"] > budget
                or classified["prefill_tokens"] + classified["decode_tokens"]
                   != classified["scheduled_tokens"]):
            raise ValueError(f"budget {budget}: call {i} service alignment differs")
        prefill, decode = classified["prefill_tokens"], classified["decode_tokens"]
        kind = ("mixed" if prefill and decode else "prefill_only" if prefill else
                "decode_only" if decode else "idle")
        wall = step["return_s"] - step["start_s"]
        type_counts[kind] += 1
        token_totals.update(prefill_tokens=prefill, decode_tokens=decode,
                            scheduled_tokens=prefill + decode)
        by_type[kind].append(wall)
        per_step.append(dict(call=i, start_s=step["start_s"], return_s=step["return_s"],
            wall_s=wall, service_type=kind, prefill_tokens=prefill, decode_tokens=decode,
            scheduled_tokens=prefill + decode,
            running_after=scheduled["after"]["running"],
            waiting_after=scheduled["after"]["waiting"],
            used_kv_blocks_after=scheduled["after"]["used_blocks"],
            preemptions=len(scheduled["preempted_request_ids"])))
    first = mix["first_successful_allocation"]
    order = [row["external_request_id"] for row in first]
    output_by_id = {row["external_request_id"]: row for row in outputs}
    if (len(first) != N or len(set(order)) != N or len(output_by_id) != N
            or set(order) != set(output_by_id)
            or not all(row.get("finished") and row.get("arrival_s") == 0.0
                       and 1 <= len(row["output_token_ids"]) <= 512 for row in outputs)):
        raise ValueError(f"budget {budget}: request inventory or completion differs")
    initial_prompt = sum(row["prompt_length_tokens"] - row["new_prefix_cached_tokens"]
                         for row in first)
    recomputed = token_totals["prefill_tokens"] - initial_prompt
    if recomputed < 0:
        raise ValueError(f"budget {budget}: scheduled prefill below initial prompt need")
    warmup_shapes = [dict(prompt_tokens=s["prompt_tokens"],
        schedule_calls=s["schedule_calls"], scheduled_tokens_per_call=s["scheduled_tokens_per_call"],
        warmup_and_reset_wall_s=s["warmup_and_reset_wall_s"])
        for s in shape["shapes"]]
    metrics = dict(batch_budget=budget, schedule_calls=len(steps),
        schedule_type_counts=dict(type_counts), scheduled_tokens=dict(token_totals),
        mixed_step_prefill_tokens=sum(x["prefill_tokens"] for x in per_step
                                       if x["service_type"] == "mixed"),
        mixed_step_decode_tokens=sum(x["decode_tokens"] for x in per_step
                                      if x["service_type"] == "mixed"),
        step_wall_s=distribution([x["wall_s"] for x in per_step]),
        step_wall_s_by_type={k: distribution(v) for k, v in sorted(by_type.items())},
        step_wall_sum_s_by_type={k: sum(v) for k, v in sorted(by_type.items())},
        initial_prompt_compute_tokens=initial_prompt,
        prefill_recompute_tokens_relative_to_first_cache_hit=recomputed,
        first_allocation_cached_prefix_tokens=sum(row["new_prefix_cached_tokens"] for row in first),
        peak_running=status["peak_running_after_schedule"],
        peak_waiting=status["peak_waiting_after_schedule"],
        peak_used_kv_blocks=status["peak_used_blocks_after_schedule"],
        preemptions=status["preemptions"],
        allocation_failures=status["allocation_failure_count"],
        output_tokens=status["output_tokens"],
        finish_reasons=status["finish_reason_counts"],
        mean_arrival_to_completion_s=sum(r["host_elapsed_s"] - r["arrival_s"]
                                         for r in outputs) / N,
        observation_end_s=status["observation_end_s"],
        warmup=dict(first_observation_s=warmup["first"]["observation_end_s"],
            last_observation_s=warmup["last"]["observation_end_s"],
            shape_total_wall_s=shape["total_shape_control_wall_s"],
            shapes=warmup_shapes,
            total_warmup_phase_s=timing["warmup_end_perf_s"]-timing["warmup_start_perf_s"]))
    return dict(metrics=metrics, per_step=per_step, first_order=order,
        first_call={r["external_request_id"]: r["schedule_call"] for r in first},
        frozen_order=trace["source_indices_in_submission_order"],
        outputs=output_by_id, environment=environment,
        raw_sha256={name: sha(directory / name) for name in names})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small-dir", type=Path, required=True)
    ap.add_argument("--large-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    small, large = cell(args.small_dir, 512), cell(args.large_dir, 1024)
    if (small["frozen_order"] != large["frozen_order"]
            or small["environment"]["gpu_before"]["gpu"].split(",")[0]
               != large["environment"]["gpu_before"]["gpu"].split(",")[0]
            or set(small["first_order"]) != set(large["first_order"])):
        raise ValueError("pair host, frozen order or request set differs")
    position = {"512": {rid: i for i, rid in enumerate(small["first_order"])},
                "1024": {rid: i for i, rid in enumerate(large["first_order"])}}
    movements = []
    changed_outputs = 0
    for rid in large["first_order"]:
        a, b = small["outputs"][rid], large["outputs"][rid]
        if (a["source_index"] != b["source_index"] or a["request_id"] != b["request_id"]
                or a["prompt_token_ids"] != b["prompt_token_ids"]):
            raise ValueError("pair source request or prompt differs: " + rid)
        changed_outputs += a["output_token_ids"] != b["output_token_ids"]
        movements.append(dict(request_id=rid, source_index=a["source_index"],
            position_512=position["512"][rid], position_1024=position["1024"][rid],
            position_delta_512_minus_1024=position["512"][rid]-position["1024"][rid],
            first_call_512=small["first_call"][rid], first_call_1024=large["first_call"][rid],
            output_tokens_512=len(a["output_token_ids"]),
            output_tokens_1024=len(b["output_token_ids"]),
            completion_s_512=a["host_elapsed_s"]-a["arrival_s"],
            completion_s_1024=b["host_elapsed_s"]-b["arrival_s"]))
    a, b = small["metrics"], large["metrics"]
    keys = ("schedule_calls", "mixed_step_prefill_tokens", "mixed_step_decode_tokens",
            "initial_prompt_compute_tokens", "prefill_recompute_tokens_relative_to_first_cache_hit",
            "first_allocation_cached_prefix_tokens", "peak_running", "peak_used_kv_blocks",
            "preemptions", "allocation_failures", "output_tokens", "observation_end_s",
            "mean_arrival_to_completion_s")
    delta = {key: a[key]-b[key] for key in keys}
    delta["mixed_steps"] = a["schedule_type_counts"].get("mixed", 0)-b["schedule_type_counts"].get("mixed", 0)
    delta["mixed_step_wall_sum_s"] = a["step_wall_sum_s_by_type"].get("mixed", 0)-b["step_wall_sum_s_by_type"].get("mixed", 0)
    result = dict(schema="c-qmsum-native-chunk-actions-v1",
        scope="One viewed same-host 512 then 1024 native-budget pair; outputs and work may differ, no equal-work or method claim",
        raw_sha256={"512": small["raw_sha256"], "1024": large["raw_sha256"]},
        arms={"512": a, "1024": b}, small_minus_large=delta,
        first_allocation_order={"512": small["first_order"], "1024": large["first_order"]},
        first_allocation_position_changes=sum(x["position_delta_512_minus_1024"] != 0
                                              for x in movements),
        output_ids_changed=changed_outputs, per_request=movements,
        per_step={"512": small["per_step"], "1024": large["per_step"]},
        interpretation_limits=[
            "Step wall is host engine.step start-to-return, including observer and scheduling work; it is not GPU kernel time.",
            "Prefill/decode totals classify scheduled tokens at the prompt boundary and include recomputation after preemption.",
            "The arms may produce different output token counts and admission orders; wall differences are policy-level observations, not equal-work chunk effects.",
            "First-allocation cache hits are observed when the allocator succeeds; no cache-eviction event is inferred."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(small_minus_large=delta,
                          first_allocation_position_changes=result["first_allocation_position_changes"],
                          output_ids_changed=changed_outputs), sort_keys=True))


if __name__ == "__main__":
    main()
