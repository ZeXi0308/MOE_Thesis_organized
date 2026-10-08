#!/usr/bin/env python3
"""Describe actor and gap trajectories in the completed waiter-backfill triplet."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MAIN = HERE / "A_WAITER_BACKFILL_TRIPLET_RESULT_R01_20261001.json"
OUTPUT = HERE / "A_WAITER_BACKFILL_TARGET_PEER_R01_20261001.json"
ADMISSION_EVENTS = {"recovery_commit_admitted", "ordinary_backfill_admission",
                    "spare_followup_admission", "direct_commit", "fit_first_admitted"}
ROLES = {
    "ordinary": ("target", "queue_head", "oldest_waiter", "displaced_oldest"),
    "primary_first": ("target", "primary_target", "primary_victim",
                      "queue_head", "displaced_oldest"),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gap_rows(raw, store, actions, limit=5):
    origin = raw["measurement_origin_perf_counter_s"]
    preemptions = raw.get("preemption_events", [])
    events = store["events"]
    ranked = []
    for request in raw["requests"]:
        times = request["token_times_s"]
        if len(times) < 2:
            continue
        gap, start, end, index = max((b - a, a, b, i + 1)
                                     for i, (a, b) in enumerate(zip(times, times[1:])))
        ranked.append((gap, start, end, index, request))
    ranked.sort(key=lambda item: (item[0], item[4]["request_id"]), reverse=True)
    result = []
    for gap, start, end, index, request in ranked[:limit]:
        internal = request["internal_request_id"]
        actual = [{"step": p["engine_call_index"], "time_s": p["method_entered_s"],
                   "output_count": p.get("last_returned_output_count")}
                  for p in preemptions if p.get("internal_request_id") == internal
                  and p.get("original_preemption_called") is True
                  and p.get("original_preemption_returned") is True
                  and start <= p["method_entered_s"] <= end]
        admissions = [{"event": e["event"], "step": e["step"],
                       "time_s": e["host_perf_counter_s"] - origin,
                       "kind": e.get("native_admission")}
                      for e in events if e.get("event") in ADMISSION_EVENTS
                      and e.get("target") == internal
                      and isinstance(e.get("host_perf_counter_s"), (int, float))
                      and start <= e["host_perf_counter_s"] - origin <= end]
        admissions.sort(key=lambda item: item["time_s"])
        pre = actual[0] if len(actual) == 1 else None
        admission = next((item for item in admissions if pre is None
                          or item["time_s"] >= pre["time_s"]), None)
        related = [{"role": role, "step": action["choice"]["step"],
                    "choice_s": action["choice_s"],
                    "within_gap": start <= action["choice_s"] <= end}
                   for action in actions for role in dict.fromkeys(
                       ROLES["ordinary"] + ROLES["primary_first"])
                   if action["choice"].get(role) == internal]
        result.append({
            "request_id": request["request_id"], "internal_id": internal,
            "gap_s": gap, "gap_start_s": start, "gap_end_s": end,
            "token_index_after_gap": index,
            "actual_preemptions_in_gap": actual,
            "recorded_native_admissions_in_gap": admissions,
            "preemption_to_first_recorded_admission_s":
                admission["time_s"] - pre["time_s"] if pre and admission else None,
            "first_recorded_admission_to_output_s":
                end - admission["time_s"] if admission else None,
            "choice_roles": related,
        })
    return result


def actor_rows(arm, actions, raw, metrics, q1_metrics):
    internal = {request["internal_request_id"]: request for request in raw["requests"]}
    result = {}
    for role in ROLES[arm]:
        grouped = defaultdict(list)
        for action in actions:
            choice = action["choice"]
            actor = choice.get(role)
            if actor is None:
                continue
            request = internal.get(actor)
            if request is None:
                grouped[actor].append({"step": choice["step"],
                                       "request_id": None, "status": "UNKNOWN_ACTOR"})
                continue
            choice_s = action["choice_s"]
            later = next((time for time in request["token_times_s"] if time > choice_s), None)
            grouped[actor].append({
                "step": choice["step"], "request_id": request["request_id"],
                "choice_s": choice_s,
                "next_output_after_choice_s": later - choice_s if later is not None else None,
                "head_required_blocks": choice.get("head_required_blocks") if role == "queue_head" else None,
                "free_blocks_at_choice": choice.get("free_blocks"),
                "head_output_count_at_choice": choice.get("head_output_count") if role == "queue_head" else None,
            })
        rows = []
        for actor, occurrences in sorted(grouped.items()):
            rid = next((item["request_id"] for item in occurrences if item["request_id"]), None)
            now, ref = metrics.get(rid), q1_metrics.get(rid)
            rows.append({
                "internal_id": actor, "request_id": rid,
                "occurrences": len(occurrences), "choices": occurrences,
                "completed": bool(now and now["completed"]),
                "max_gap_s": now["max_gap_s"] if now else None,
                "max_gap_difference_vs_q1_s": now["max_gap_s"] - ref["max_gap_s"]
                    if now and ref else None,
                "flow_s": now["flow_s"] if now else None,
                "flow_difference_vs_q1_s": now["flow_s"] - ref["flow_s"]
                    if now and ref else None,
            })
        result[role] = {
            "action_occurrences": sum(row["occurrences"] for row in rows),
            "distinct_requests": len(rows),
            "repeated_requests": sum(row["occurrences"] > 1 for row in rows),
            "requests_worse_gap_than_q1": sum(row["max_gap_difference_vs_q1_s"] is not None
                                              and row["max_gap_difference_vs_q1_s"] > 0
                                              for row in rows),
            "requests": rows,
        }
    return result


def main():
    if OUTPUT.exists():
        raise SystemExit("Preserve existing target-peer result")
    canonical = json.loads(MAIN.read_text())
    if canonical["status"] != "COMPLETE_TRIPLET":
        raise SystemExit("Canonical triplet is incomplete")
    raw, store, metrics = {}, {}, {}
    sources = {"canonical": str(MAIN), "canonical_sha256": sha(MAIN), "arms": {}}
    for arm in ("q1", "ordinary", "primary_first"):
        archive = HERE / canonical["arms"][arm]["archive"]
        raw_path, store_path = archive / "raw.json", archive / "selective-store.json"
        raw[arm], store[arm] = json.loads(raw_path.read_text()), json.loads(store_path.read_text())
        metrics[arm] = {row["request_id"]: row
                        for row in canonical["arms"][arm]["metrics"]["requests"]}
        sources["arms"][arm] = {"raw": str(raw_path), "raw_sha256": sha(raw_path),
                                "selective_store": str(store_path),
                                "selective_store_sha256": sha(store_path)}
    ids = set(metrics["q1"])
    if len(ids) != 128 or any(set(metrics[arm]) != ids for arm in metrics):
        raise SystemExit("Complete request identities differ")
    individual = [{
        "request_id": rid,
        "arms": {arm: {key: metrics[arm][rid][key]
                       for key in ("status", "stop_reason", "outputs", "ttft_s", "max_gap_s", "flow_s")}
                 for arm in metrics},
        "ordinary_minus_q1_max_gap_s": metrics["ordinary"][rid]["max_gap_s"]
                                          - metrics["q1"][rid]["max_gap_s"],
        "primary_first_minus_q1_max_gap_s": metrics["primary_first"][rid]["max_gap_s"]
                                               - metrics["q1"][rid]["max_gap_s"],
    } for rid in sorted(ids)]
    actions = {arm: canonical["actions"][arm]["actions"]
               for arm in ("ordinary", "primary_first")}
    actors = {arm: actor_rows(arm, actions[arm], raw[arm], metrics[arm], metrics["q1"])
              for arm in actions}
    gaps = {arm: gap_rows(raw[arm], store[arm], actions.get(arm, []))
            for arm in raw}
    ordinary_events = store["ordinary"]["events"]
    regular_releases = {event["step"] for event in ordinary_events
                        if event.get("event") == "protection_release"
                        and event.get("protection_origin") == "REGULAR_COMMIT"
                        and event.get("reason") == "OUTPUT_GOAL_REACHED"}
    ordinary_steps = [action["choice"]["step"] for action in actions["ordinary"]]
    same_step_regular = sum(step in regular_releases for step in ordinary_steps)
    same_step_prepare = sum(any(event.get("step") == step and event.get("event") in
                                ("prepare", "commit_check", "recovery_commit_admitted")
                                for event in ordinary_events) for step in ordinary_steps)
    ordinary_head = actors["ordinary"]["queue_head"]["requests"]
    primary_head = actors["primary_first"]["queue_head"]["requests"]
    report = {
        "status": "DESCRIPTIVE_SOURCE_LOCALIZATION_COMPLETE",
        "sources": sources,
        "full_cohort": {arm: {"completed": canonical["arms"][arm]["metrics"]["completed"],
                              "rate_tokens_s": canonical["arms"][arm]["metrics"]["actual_output_tokens_s"],
                              "mean_flow_s": canonical["arms"][arm]["metrics"]["mean_flow_with_incomplete_penalty_s"],
                              "max_gap_s": canonical["arms"][arm]["metrics"]["max_gap_request_max_s"],
                              "actual_preemptions": canonical["arms"][arm]["actual_preemption_count"],
                              "forced_rotations": canonical["arms"][arm]["forced_rotations"]}
                        for arm in raw},
        "all_individual_requests": individual,
        "actions": {arm: {"choices": len(actions[arm]),
                          "native_admissions": canonical["actions"][arm]["native_admissions"],
                          "actual_output_completion_chains": canonical["actions"][arm]["actual_output_completion_chains"],
                          "actors": actors[arm]}
                    for arm in actions},
        "ordinary_trigger_scope": {
            "same_step_regular_commit_first_output_release": same_step_regular,
            "outside_same_step_regular_commit_first_output_release":
                len(ordinary_steps) - same_step_regular,
            "same_step_regular_prepare_commit_or_admission": same_step_prepare,
            "all_choices_used_ordinary_origin": canonical["actions"]["ordinary"]["ordinary_origin_starts"]
                == len(ordinary_steps),
            "interpretation": "Successful ordinary action branches before the regular selector in this implementation; zero same-step regular prepare/admission is observed. Which normal recovery would have occurred without the branch is unobserved.",
        },
        "repeated_bypass": {
            "ordinary_blocked_queue_head_occurrences": sum(row["occurrences"] for row in ordinary_head),
            "ordinary_blocked_queue_head_distinct": len(ordinary_head),
            "ordinary_heads_bypassed_more_than_once": [row for row in ordinary_head if row["occurrences"] > 1],
            "ordinary_displaced_oldest": actors["ordinary"]["displaced_oldest"],
            "primary_first_queue_head_occurrences": sum(row["occurrences"] for row in primary_head),
            "primary_first_queue_head_distinct": len(primary_head),
            "primary_first_heads_bypassed_more_than_once": [row for row in primary_head if row["occurrences"] > 1],
            "primary_first_displaced_oldest": actors["primary_first"]["displaced_oldest"],
        },
        "worst_gaps": gaps,
        "limits": [
            "Every per-request comparison uses the same fixed articles but distinct execution trajectories; no action-level counterfactual is identified.",
            "Preemption-to-recorded-admission intervals are observed; their full duration cannot be assigned to queue waiting, and missing sparse admission receipts remain unknown.",
            "Numerical head non-fit and source branch order do not prove what a different selector would have admitted or completed at that step.",
        ],
    }
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "ordinary_actions": len(actions["ordinary"]),
                      "primary_first_actions": len(actions["primary_first"]),
                      "same_step_regular": same_step_regular,
                      "ordinary_max_gap": gaps["ordinary"][0],
                      "primary_first_max_gap": gaps["primary_first"][0]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
