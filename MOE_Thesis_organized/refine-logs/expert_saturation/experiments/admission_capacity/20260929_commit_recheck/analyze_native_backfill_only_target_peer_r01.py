#!/usr/bin/env python3
"""Describe observed actor and gap trajectories in the native backfill pair."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MAIN = HERE / "A_NATIVE_BACKFILL_ONLY_PAIR_RESULT_R01_20261001.json"
OUTPUT = HERE / "A_NATIVE_BACKFILL_ONLY_TARGET_PEER_R01_20261001.json"
ROLES = ("target", "queue_head", "oldest_waiter", "displaced_oldest")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actual_preemptions(raw, internal_id, start=None, end=None):
    return [
        {"time_s": event["method_entered_s"], "step": event["engine_call_index"],
         "output_count": event.get("last_returned_output_count")}
        for event in raw["preemption_events"]
        if event.get("internal_request_id") == internal_id
        and event.get("original_preemption_called") is True
        and event.get("original_preemption_returned") is True
        and (start is None or event["method_entered_s"] >= start)
        and (end is None or event["method_entered_s"] <= end)
    ]


def next_output(request, after):
    return next((time for time in request["token_times_s"] if time > after), None)


def action_rows(actions, raw, metrics, reference):
    by_internal = {request["internal_request_id"]: request for request in raw["requests"]}
    rows = []
    for action in actions:
        choice = action["choice"]
        target, head = by_internal[choice["target"]], by_internal[choice["queue_head"]]
        first = action["first_new_output_s"]
        head_next = next_output(head, action["choice_s"])
        target_id, head_id = target["request_id"], head["request_id"]
        rows.append({
            "step": choice["step"], "choice_s": action["choice_s"],
            "target": target_id, "queue_head": head_id,
            "oldest_waiter": choice.get("oldest_waiter"),
            "displaced_oldest": choice.get("displaced_oldest"),
            "free_blocks": choice["free_blocks"],
            "target_required_blocks": choice["required_blocks"],
            "head_required_blocks": choice["head_required_blocks"],
            "native_admission": action["native_receipt"]["native_admission"],
            "same_step_actual_preemptions": action["actual_preemptions_same_step"],
            "target_first_new_output_s": first,
            "admission_to_target_output_s": action["admission_to_output_s"],
            "target_completed_s": target["completion_s"],
            "target_preemptions_after_first_new_output":
                actual_preemptions(raw, choice["target"], first, target["completion_s"]),
            "target_max_gap_s": metrics[target_id]["max_gap_s"],
            "target_max_gap_delta_vs_native_s":
                metrics[target_id]["max_gap_s"] - reference[target_id]["max_gap_s"],
            "head_next_output_s": head_next,
            "head_choice_to_next_output_s":
                head_next - action["choice_s"] if head_next is not None else None,
            "head_completed_s": head["completion_s"],
            "head_max_gap_s": metrics[head_id]["max_gap_s"],
            "head_max_gap_delta_vs_native_s":
                metrics[head_id]["max_gap_s"] - reference[head_id]["max_gap_s"],
            "head_flow_delta_vs_native_s":
                metrics[head_id]["flow_s"] - reference[head_id]["flow_s"],
        })
    return rows


def actor_groups(actions, raw, metrics, reference):
    by_internal = {request["internal_request_id"]: request for request in raw["requests"]}
    result = {}
    for role in ROLES:
        grouped = defaultdict(list)
        for action in actions:
            internal = action["choice"].get(role)
            if internal:
                grouped[internal].append(action)
        rows = []
        for internal, appearances in sorted(grouped.items()):
            request = by_internal[internal]
            rid = request["request_id"]
            current, baseline = metrics[rid], reference[rid]
            rows.append({
                "request_id": rid, "internal_id": internal,
                "occurrences": len(appearances),
                "choice_times_s": [a["choice_s"] for a in appearances],
                "choice_to_next_output_s": [
                    next_output(request, a["choice_s"]) - a["choice_s"]
                    if next_output(request, a["choice_s"]) is not None else None
                    for a in appearances],
                "completed_s": request["completion_s"],
                "actual_preemptions_after_first_choice": actual_preemptions(
                    raw, internal, appearances[0]["choice_s"], request["completion_s"]),
                "max_gap_s": current["max_gap_s"],
                "max_gap_delta_vs_native_s": current["max_gap_s"] - baseline["max_gap_s"],
                "flow_s": current["flow_s"],
                "flow_delta_vs_native_s": current["flow_s"] - baseline["flow_s"],
                "ttft_delta_vs_native_s": current["ttft_s"] - baseline["ttft_s"],
            })
        result[role] = {
            "action_occurrences": sum(row["occurrences"] for row in rows),
            "distinct_requests": len(rows),
            "repeated_requests": sum(row["occurrences"] > 1 for row in rows),
            "requests_with_worse_max_gap_vs_native": sum(
                row["max_gap_delta_vs_native_s"] > 0 for row in rows),
            "requests": rows,
        }
    return result


def worst_gaps(raw, store, actions, limit=5):
    origin = raw["measurement_origin_perf_counter_s"]
    ranked = []
    for request in raw["requests"]:
        times = request["token_times_s"]
        if len(times) > 1:
            gap, start, end, index = max(
                (end - start, start, end, i + 1)
                for i, (start, end) in enumerate(zip(times, times[1:])))
            ranked.append((gap, start, end, index, request))
    ranked.sort(key=lambda row: (row[0], row[4]["request_id"]), reverse=True)
    result = []
    for gap, start, end, index, request in ranked[:limit]:
        internal = request["internal_request_id"]
        preemptions = actual_preemptions(raw, internal, start, end)
        receipts = [
            {"step": event["step"], "time_s": event["host_perf_counter_s"] - origin,
             "native_admission": event["native_admission"]}
            for event in store["events"]
            if event.get("event") == "ordinary_backfill_admission"
            and event.get("target") == internal
            and start <= event["host_perf_counter_s"] - origin <= end]
        roles = [
            {"step": action["choice"]["step"], "choice_s": action["choice_s"], "role": role}
            for action in actions if start <= action["choice_s"] <= end
            for role in ROLES if action["choice"].get(role) == internal]
        result.append({
            "request_id": request["request_id"], "gap_s": gap,
            "gap_start_s": start, "gap_end_s": end, "token_index_after_gap": index,
            "actual_preemptions_in_gap": preemptions,
            "preemption_to_next_output_s":
                end - preemptions[-1]["time_s"] if preemptions else None,
            "recorded_ordinary_native_admissions_in_gap": receipts,
            "admission_to_next_output_s":
                end - receipts[-1]["time_s"] if receipts else None,
            "backfill_choice_roles_in_gap": roles,
            "ordinary_admission_observability":
                "RECORDED" if receipts else "UNKNOWN_NATIVE_RESUMPTION_TIME",
        })
    return result


def main():
    if OUTPUT.exists():
        raise SystemExit("Preserve existing native backfill target-peer result")
    canonical = json.loads(MAIN.read_text())
    if canonical["status"] != "COMPLETE_PAIR":
        raise SystemExit("Canonical pair incomplete")
    source = {"canonical": str(MAIN), "canonical_sha256": sha(MAIN), "arms": {}}
    raws, stores, metrics = {}, {}, {}
    for arm in ("native_full", "ordinary"):
        archive = HERE / canonical["arms"][arm]["archive"]
        raw_path, store_path = archive / "raw.json", archive / "selective-store.json"
        raws[arm], stores[arm] = json.loads(raw_path.read_text()), json.loads(store_path.read_text())
        metrics[arm] = {row["request_id"]: row
                        for row in canonical["arms"][arm]["metrics"]["requests"]}
        source["arms"][arm] = {"raw": str(raw_path), "raw_sha256": sha(raw_path),
                               "selective_store": str(store_path),
                               "selective_store_sha256": sha(store_path)}
    if len(metrics["native_full"]) != 128 or set(metrics["native_full"]) != set(metrics["ordinary"]):
        raise SystemExit("Cohort identities incomplete or mismatched")
    actions = canonical["ordinary_actions"]["actions"]
    rows = action_rows(actions, raws["ordinary"], metrics["ordinary"], metrics["native_full"])
    groups = actor_groups(actions, raws["ordinary"], metrics["ordinary"], metrics["native_full"])
    gaps = {arm: worst_gaps(raws[arm], stores[arm], actions if arm == "ordinary" else [])
            for arm in raws}
    report = {
        "status": "DESCRIPTIVE_SOURCE_LOCALIZATION_COMPLETE",
        "sources": source,
        "full_cohort": {arm: {
            "completed": canonical["arms"][arm]["metrics"]["completed"],
            "duration_s": canonical["arms"][arm]["metrics"]["duration_s"],
            "actual_output_tokens_s": canonical["arms"][arm]["metrics"]["actual_output_tokens_s"],
            "mean_flow_s": canonical["arms"][arm]["metrics"]["mean_flow_with_incomplete_penalty_s"],
            "max_gap_s": canonical["arms"][arm]["metrics"]["max_gap_request_max_s"],
            "actual_preemptions": canonical["arms"][arm]["actual_preemption_count"],
            "forced_rotations": canonical["arms"][arm]["forced_rotations"],
        } for arm in raws},
        "all_request_deltas_vs_native": [{
            "request_id": rid,
            "max_gap_delta_s": metrics["ordinary"][rid]["max_gap_s"] - metrics["native_full"][rid]["max_gap_s"],
            "flow_delta_s": metrics["ordinary"][rid]["flow_s"] - metrics["native_full"][rid]["flow_s"],
            "ttft_delta_s": metrics["ordinary"][rid]["ttft_s"] - metrics["native_full"][rid]["ttft_s"],
        } for rid in sorted(metrics["native_full"])],
        "ordinary_actions": {
            "count": len(rows),
            "native_admission_kinds": dict(Counter(row["native_admission"] for row in rows)),
            "actions_with_later_target_preemption": sum(
                bool(row["target_preemptions_after_first_new_output"]) for row in rows),
            "distinct_targets_with_later_preemption": len({
                row["target"] for row in rows if row["target_preemptions_after_first_new_output"]}),
            "actions": rows,
            "actors": groups,
        },
        "worst_gaps": gaps,
        "cpu_cost_observability": {
            "source_fact": "staged_store_rotation.begin constructs view(r) for every running request each step; view validates each owned block and materializes a block-ID tuple. The ordinary gate can then inspect running and waiting requests. The native reference has no staged adapter.",
            "ordinary_schedule_calls": stores["ordinary"].get("schedule_calls"),
            "ordinary_gate_counts": stores["ordinary"].get("ordinary_backfill_gate_counts"),
            "per_begin_hold_wrapper_wall_or_thread_cpu_recorded": False,
            "interpretation": "The pair's 4.048 s measurement-duration difference cannot be assigned to adapter CPU work from these logs; scheduling and execution trajectories differ.",
        },
        "next_step_recommendation": "If the CPU-cost question remains decision-relevant, run one policy-unchanged lightweight diagnostic on this fixed input that accumulates disjoint begin, hold, and wrapper pre/post wall and thread-CPU times; do not treat that diagnostic as another throughput rank or change a threshold.",
        "limits": [
            "Same fixed requests run on separate trajectories; per-request differences are descriptive, not action-level counterfactuals.",
            "Sparse ordinary receipts identify selected async admissions only. Native waiting-queue resume times for other requests are unrecorded; a preemption-to-output gap is not entirely queue waiting.",
            "A non-fitting head is observed at choice; the log does not show how a different native schedule would have served it at that step.",
        ],
    }
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": report["status"], "actions": len(rows),
        "later_target_preemption_actions": report["ordinary_actions"]["actions_with_later_target_preemption"],
        "heads_repeated": groups["queue_head"]["repeated_requests"],
        "ordinary_worst_gap": gaps["ordinary"][0],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
