#!/usr/bin/env python3
"""Descriptive joins of R02 recoveries, service, and the next real eviction.

No event is an independent repeat; no alternative schedule is replayed.
Reads existing immutable raw/result files and writes one new JSON diagnostic.
"""
import argparse
import bisect
import collections
import json
import math
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def quantile(p):
        x = (len(values) - 1) * p
        lo = math.floor(x)
        hi = math.ceil(x)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return {"n": len(values), "min": values[0], "median": quantile(.5),
            "p90": quantile(.9), "p95": quantile(.95), "max": values[-1]}


def analyze(experiment_dir):
    result_name = "A_NATIVE_OLDEST_REPEAT_TRIPLET_RESULT_R02_20261002.json"
    result = json.loads((experiment_dir / result_name).read_text())
    episodes = result["repeated_episode_chains"]["queue_fund"]["episodes"]
    transfer_name = "A_NATIVE_OLDEST_REPEAT_TRANSFER_FOLLOWUP_R02_20261002.json"
    transfers = json.loads((experiment_dir / transfer_name).read_text())
    transfer_by_id = {e["episode_id"]: e for e in transfers["episodes_detail"]}
    forced_by_key = {(e["actual_forced_preemption_record"]["engine_call_index"],
                      e["victim_source_request"]): e for e in episodes
                     if e["actual_forced_preemption"]}
    native_by_step = {v["step"]: v for v in result["victim_actions"]["queue_fund"]["rows"]}
    archive = experiment_dir / result["arms"]["queue_fund"]["archive"]
    raw = json.loads((archive / "raw.json").read_text())
    raw_requests = {r["request_id"]: r for r in raw["requests"]}
    raw_preemptions = {(p["engine_call_index"], p["request_id"]): p
                       for p in raw["preemption_events"]
                       if p["original_preemption_returned"]}
    rows = []
    for episode in episodes:
        anchor = episode["anchor"]
        target = episode["target"]
        request = raw_requests[episode["target_source_request"]]
        first_output = target["first_output_after_reference_s"]
        next_preempt = next((p for p in target["all_actual_preemptions"]
                             if p["time_s"] > first_output), None)
        release, = episode["protection_releases"]
        assert release["new_output_tokens"] == 1
        assert release["future_compute_tokens"] == request["prompt_tokens"] + release["output_count"] - 1
        end_output = next_preempt["output_count_at_preemption"] if next_preempt else target["output_count"]
        first_output_count = anchor["target_output_count"] + 1
        obtained = end_output - anchor["target_output_count"]
        # Cross-check against original host token times, not the joined result alone.
        assert request["token_times_s"][first_output_count - 1] == first_output
        if next_preempt:
            independent = raw_preemptions[(next_preempt["step"], episode["target_source_request"])]
            assert independent["native_output_count_before"] == end_output
            assert independent["method_entered_s"] == next_preempt["time_s"]
        forced = forced_by_key.get((next_preempt["step"], episode["target_source_request"])) if next_preempt else None
        native = native_by_step.get(next_preempt["step"]) if next_preempt and not forced else None
        selected = (next(c for c in native["candidates"] if c["request"] == native["selected"])
                    if native else None)
        selected_computed = selected["bidkv"]["computed_tokens"] if selected else None
        row = {
            "episode_id": episode["episode_id"], "source_request_id": episode["target_source_request"],
            "anchor_s": episode["anchor_time_s"], "first_output_s": first_output,
            "anchor_output_count": anchor["target_output_count"],
            "end_output_count": end_output, "new_outputs_until_next_eviction_or_completion": obtained,
            "ended_by_repreemption": next_preempt is not None,
            "next_preemption_s": next_preempt["time_s"] if next_preempt else None,
            "next_preemption_step": next_preempt["step"] if next_preempt else None,
            "next_eviction_kind": "FUNDS_OTHER_RECOVERY" if forced else "NATIVE_TAIL" if native else "COMPLETED",
            "next_funded_source_id": forced["target_source_request"] if forced else None,
            "next_eviction_last_to_next_output_gap_s": next_preempt["output_gap_s"] if next_preempt else None,
            "first_output_to_next_eviction_s": next_preempt["time_s"] - first_output if next_preempt else None,
            "anchor_to_first_output_s": episode["anchor_to_first_new_output_s"],
            "host_native_admission_to_output_observation_s": transfer_by_id[episode["episode_id"]]["host_intervals"]["native_admission_to_first_output_observed_s"],
            "preceding_host_output_gap_s": episode["target_last_to_next_output_gap_s"],
            "anchor_free_pages": anchor["free_blocks"], "anchor_full_history_need_pages": anchor["target_full_history_need_blocks"],
            "anchor_deficit_pages": anchor["deficit_blocks"], "planned_victim_pages": anchor["victim_held_blocks"],
            "free_pages_at_q1_release": release["free_blocks"], "held_pages_at_q1_release": release["held_blocks"],
            "computed_tokens_at_q1_release": release["future_compute_tokens"],
            "owned_unused_slots_at_q1_release": 16 * release["held_blocks"] - release["future_compute_tokens"],
            "native_next_eviction_failed_request": native["failed_request"] if native else None,
            "free_pages_at_native_next_eviction": native["free_blocks"] if native else None,
            "target_owned_unused_slots_at_native_next_eviction": 16 * selected["held_blocks"] - selected_computed if selected else None,
            "target_held_pages_at_native_next_eviction": selected["held_blocks"] if selected else None,
        }
        rows.append(row)
    repeated = [r for r in rows if r["ended_by_repreemption"]]
    short = [r for r in repeated if r["new_outputs_until_next_eviction_or_completion"] <= 9]
    return {
        "status": "OBSERVED_RESIDENCY_DIAGNOSTIC",
        "evidence_tier": "NATIVE_SERVING_DESCRIPTIVE_ONE_SEEN_INPUT_BLOCK",
        "sources": [result_name, transfer_name, str(archive.relative_to(experiment_dir) / "raw.json")],
        "scientific_scope": "One newly answered question: after an already qualified funded recovery, how much real output service occurs before the next real preemption, and what owned capacity exists then?",
        "limitations": [
            "These episodes share requests and one policy trajectory, so are not independent experiments.",
            "The alternative policy has not been run; observed later timing and eviction are outcomes, not online inputs.",
            "Host admission-to-output elapsed time includes queue, transfer and compute overlap; it is neither device-copy duration nor additive exposed overhead.",
            "Owned unused slots imply a target-only zero-growth opportunity, not a full-system free service guarantee.",
            "A short residency can be followed by a short gap; residency length is not the service objective.",
        ],
        "summary": {
            "episodes": len(rows), "unique_sources": len({r["source_request_id"] for r in rows}),
            "repreempted": len(repeated), "complete_without_repreemption": len(rows) - len(repeated),
            "reeviction_kind_counts": dict(collections.Counter(r["next_eviction_kind"] for r in repeated)),
            "new_outputs_before_reeviction": distribution([r["new_outputs_until_next_eviction_or_completion"] for r in repeated]),
            "new_outputs_all_residencies_including_terminal": distribution([r["new_outputs_until_next_eviction_or_completion"] for r in rows]),
            "one_output_then_eviction": sum(r["new_outputs_until_next_eviction_or_completion"] == 1 for r in repeated),
            "at_most_nine_outputs_then_eviction": len(short),
            "short_reeviction_kind_counts": dict(collections.Counter(r["next_eviction_kind"] for r in short)),
            "short_native_reevictions_with_unused_owned_slots": sum(r["next_eviction_kind"] == "NATIVE_TAIL" and r["target_owned_unused_slots_at_native_next_eviction"] > 0 for r in short),
            "owned_unused_slots_at_q1_release": distribution([r["owned_unused_slots_at_q1_release"] for r in rows]),
            "host_native_admission_to_output_observation_s": distribution([r["host_native_admission_to_output_observation_s"] for r in rows]),
            "observed_next_gaps_after_short_residency_s": distribution([r["next_eviction_last_to_next_output_gap_s"] for r in short]),
        },
        "representative_episode_ids": ["oldest-000016", "oldest-000022", "oldest-000026", "oldest-000052"],
        "episodes": rows,
        "model_candidate_not_implemented_or_evaluated": {
            "state": "Current output ages; per-request computed tokens, owned private pages, service since recovery; native in-flight page reservations; waiting/recovering/running phase; past-only recovery and decode scheduling times.",
            "decision": "Jointly choose one recovery target, legal victims, a finite set of requests allowed to grow, and a bounded useful-service quantum. Reconsider at a page-growth boundary or a competing waiter's urgency boundary.",
            "capacity_condition": "F + released_private_pages(V) - inflight_reservations >= target_missing_pages(q) + sum(extra_pages_j(q) for co-served j). Decode q is clipped at known max_tokens; no future EOS is used.",
            "execution_condition": "Reserve one actual decode-token scheduling slot per promised output and keep the target eligible. A request whose growth is unfunded is temporarily deferred while zero-growth promised outputs execute; do not simply increase an output-count protection flag.",
            "quantum_rule": "Enumerate q only to the next target page frontier (at most16 outputs in the zero-growth variant), truncate when a competing output-bearing waiter's current age plus an online duration estimate reaches the existing urgency threshold. Among feasible plans prefer enough service to amortize observed prior recovery rounds; if no such q exists take q1 and record the conflict.",
            "resource_release": "Release unused commitment immediately at EOS or failure; stop the quantum at its frontier and re-evaluate using new actual state.",
            "conditional_property": "With private pages, truthful reservation accounting, available token slots, target scheduling and eventually completed transfers, committed q outputs or earlier terminal completion cannot be prevented by a KV-allocation failure for the promised set. No wall-clock guarantee without transfer/step bounds.",
            "difference_from_failed_q10": "Couples admission to actual permitted running-set growth and execution credits; releases at resource and urgency boundaries; does not blanket-block waiting until ten outputs. Prior fixed Q10 and yieldQ10 failures remain strong ablations.",
            "next_discriminating_run": "Preserve pending ordinary/fund/native triplet. Then compare frozen Q1 funding, target-only longer protection, and joint growth/execution quantum on the same complete requests; primary per-request maximum host-gap p95 plus full rate/flow/peer costs.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("recovery_residencies.json"))
    args = parser.parse_args()
    result = analyze(args.experiment_dir)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
