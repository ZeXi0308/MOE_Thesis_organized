#!/usr/bin/env python3
"""Read-only native QMSum pressure diagnostic; no policy counterfactual."""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_index(external_id):
    return int(external_id.rsplit("-", 1)[1])


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    wp = args.input_dir / "workload.json"
    sp = args.run_dir / "measured-steps.json"
    mp = args.run_dir / "measured-service-mix.json"
    op = args.run_dir / "measured-outputs.json"
    tp = args.run_dir / "status.json"
    workload, steps, mix, outputs, status = map(read, (wp, sp, mp, op, tp))
    rows = {r["source_index"]: r for r in workload["requests"]}
    assert len(rows) == len(outputs) == len(mix["first_successful_allocation"]) == 200
    calls, service_calls, engine_steps = (steps["scheduler_calls"],
                                         mix["scheduler_calls"], steps["steps"])
    assert len(calls) == len(service_calls) == len(engine_steps) == status["schedule_calls"]
    assert all(c["call"] == i and s["call"] == i and e["start_s"] <= c["host_s"] <= e["return_s"]
               for i, (c, s, e) in enumerate(zip(calls, service_calls, engine_steps)))
    first = {source_index(x["external_request_id"]): x
             for x in mix["first_successful_allocation"]}
    internal = {x["internal_request_id"]: source_index(x["external_request_id"])
                for x in mix["first_successful_allocation"]}
    assert set(first) == set(rows) and len(internal) == 200
    order = steps["source_indices_in_submission_order"]
    assert sorted(order) == sorted(rows)
    rank = {index: pos for pos, index in enumerate(order)}
    prompt = {i: r["prompt_token_ids"] for i, r in rows.items()}
    lcp = {(i, j): common_prefix(prompt[i], prompt[j])
           for i in rows for j in rows if i != j}

    failures = collections.defaultdict(list)
    by_status = collections.Counter()
    for event in steps["allocation_failures"]:
        failures[event["schedule_call"]].append(event)
        by_status[event["request_status"]] += 1
    assert sum(by_status.values()) == status["allocation_failure_count"]
    computed_depth = {i: 0 for i in rows}  # Only earlier *completed* scheduler calls.
    waiting_events = []
    service_types = collections.Counter()
    waiting_service_types = collections.Counter()
    scheduled_prefill = collections.Counter()
    for call_index, service in enumerate(service_calls):
        prefill, decode = service["prefill_tokens"], service["decode_tokens"]
        service_type = ("mixed" if prefill and decode else "prefill_only" if prefill
                        else "decode_only" if decode else "idle")
        service_types[service_type] += 1
        waiting_here = [e for e in failures[call_index] if e["request_status"] == "WAITING"]
        if waiting_here:
            waiting_service_types[service_type] += 1
        assert len(waiting_here) <= 1  # Native FCFS breaks on first failed waiter.
        for event in waiting_here:
            head = internal[event["request_id"]]
            assert event["full_sequence_must_fit"] is True
            assert event["current_num_tokens"] == len(prompt[head])
            cached = event["num_new_computed_tokens"]
            assert cached % 16 == 0
            # New blocks only; actual gate can also charge evictable cached blocks
            # and a nonnegative watermark. Thus this is a necessary fit condition.
            head_min_blocks = math.ceil(len(prompt[head]) / 16) - cached // 16
            later_optimistic_fit = []
            later_deeper_history = []
            for candidate in order[rank[head] + 1:]:
                if first[candidate]["schedule_call"] <= call_index:
                    continue  # Already allocated; not a later unadmitted waiter.
                max_historical_prefix = max(
                    (min(lcp[candidate, prior], depth)
                     for prior, depth in computed_depth.items() if prior != candidate),
                    default=0,
                )
                historical_blocks = max_historical_prefix // 16
                min_blocks = math.ceil(len(prompt[candidate]) / 16) - historical_blocks
                if max_historical_prefix // 16 > cached // 16:
                    later_deeper_history.append(candidate)
                if min_blocks <= event["free_blocks_before"]:
                    later_optimistic_fit.append({
                        "source_index": candidate,
                        "context_sha256": rows[candidate]["context_sha256"],
                        "historical_prefix_blocks_upper_bound": historical_blocks,
                        "new_blocks_lower_bound": min_blocks,
                    })
            waiting_events.append({
                "schedule_call": call_index,
                "host_s": event["host_s"],
                "head_source_index": head,
                "head_context_sha256": rows[head]["context_sha256"],
                "head_internal_request_id": event["request_id"],
                "head_cached_tokens_observed": cached,
                "head_new_blocks_lower_bound": head_min_blocks,
                "free_blocks_before": event["free_blocks_before"],
                "block_deficit_lower_bound": head_min_blocks - event["free_blocks_before"],
                "later_deeper_historical_prefix_indices": later_deeper_history,
                "later_optimistic_fit": later_optimistic_fit,
                "step_prefill_tokens": prefill,
                "step_decode_tokens": decode,
            })
        for request in service["requests"]:
            idx = source_index(request["external_request_id"])
            assert internal[request["internal_request_id"]] == idx
            scheduled_prefill[idx] += request["scheduled_prefill_tokens"]
            depth = min(len(prompt[idx]), request["computed_before_execution"]
                        + request["scheduled_prefill_tokens"])
            computed_depth[idx] = max(computed_depth[idx], depth)

    first_work = {i: len(prompt[i]) - row["new_prefix_cached_tokens"]
                  for i, row in first.items()}
    extra_prefill = {str(i): scheduled_prefill[i] - first_work[i]
                     for i in rows if scheduled_prefill[i] != first_work[i]}
    assert sum(extra_prefill.values()) == sum(scheduled_prefill.values()) - sum(first_work.values())
    assert len(waiting_events) == by_status["WAITING"]
    max_gaps = []
    for row in outputs:
        times = row["token_times_s"]
        if len(times) < 2:
            continue
        gap, pos = max((times[i + 1] - times[i], i) for i in range(len(times) - 1))
        lo, hi = times[pos:pos + 2]
        covered = [i for i, step in enumerate(engine_steps)
                   if lo < step["return_s"] <= hi + 1e-8]
        max_gaps.append({
            "source_index": row["source_index"], "gap_s": gap,
            "from_s": lo, "to_s": hi,
            "schedule_calls": [min(covered), max(covered)] if covered else [],
            "scheduled_prefill_tokens": sum(service_calls[i]["prefill_tokens"] for i in covered),
            "scheduled_decode_tokens": sum(service_calls[i]["decode_tokens"] for i in covered),
            "waiting_failures": sum(e["request_status"] == "WAITING"
                                    for i in covered for e in failures[i]),
            "preemptions": sum(len(calls[i]["preempted_request_ids"]) for i in covered),
        })
    max_gaps.sort(key=lambda x: (-x["gap_s"], x["source_index"]))
    optimistic_events = [e for e in waiting_events if e["later_optimistic_fit"]]
    result = {
        "schema": "c-qmsum-native-pressure-diagnostic-v1",
        "scope": "Single completed 200-request native run; host scheduler observations, no policy outcome estimate",
        "sources_sha256": {name: digest(path) for name, path in
                           (("workload.json", wp), ("measured-steps.json", sp),
                            ("measured-service-mix.json", mp), ("measured-outputs.json", op),
                            ("status.json", tp))},
        "summary": {
            "requests": len(rows), "schedule_calls": len(calls),
            "failure_status_counts": dict(by_status),
            "waiting_failure_distinct_calls": len({e["schedule_call"] for e in waiting_events}),
            "waiting_head_new_blocks_lower_bound_exceeds_free_count": sum(
                e["block_deficit_lower_bound"] > 0 for e in waiting_events),
            "waiting_min_block_deficit_range": [min(e["block_deficit_lower_bound"] for e in waiting_events),
                                                max(e["block_deficit_lower_bound"] for e in waiting_events)],
            "waiting_head_observed_cached_tokens": dict(collections.Counter(
                e["head_cached_tokens_observed"] for e in waiting_events)),
            "waiting_with_later_deeper_historical_prefix": sum(
                bool(e["later_deeper_historical_prefix_indices"]) for e in waiting_events),
            "waiting_with_optimistic_later_fit": len(optimistic_events),
            "optimistic_later_candidate_indices": sorted({x["source_index"]
                                                            for e in optimistic_events
                                                            for x in e["later_optimistic_fit"]}),
            "service_step_types": dict(service_types),
            "waiting_failure_step_types": dict(waiting_service_types),
            "classified_prefill_tokens": sum(scheduled_prefill.values()),
            "initial_necessary_prompt_compute_tokens": sum(first_work.values()),
            "extra_scheduled_prefill_after_first_allocation_by_index": extra_prefill,
            "classified_decode_tokens": sum(c["decode_tokens"] for c in service_calls),
        },
        "waiting_failures": waiting_events,
        "max_host_token_gap_top_12": max_gaps[:12],
        "interpretation_limits": [
            "Native scheduler breaks the WAITING loop on an allocation failure; no later waiter is probed in that call.",
            "Block fit uses new-block lower bounds only; actual admission also accounts for evictable cached blocks and any watermark.",
            "Historical prefix depth is an optimistic prior-computation bound, not a live KV residency or actual cache-hit observation for an unadmitted waiter.",
            "Per-call prefill/decode fields classify scheduled tokens, not GPU kernel time; token gaps are host return timestamps.",
            "Extra prefill is measured scheduled recomputation beyond each request's initial prompt minus its first admission cache hit, not a causal latency estimate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
