#!/usr/bin/env python3
"""Observed self-preempt continuation and complete-request three-arm comparison."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import read_cell
from analyze_native_residency_victim_triplet_r01 import score, victim_receipts
from analyze_protection_yield_triplet_r01 import compare, require
from analyze_waiter_backfill_triplet_r01 import ordinary_actions


ARMS = {("tail", False): "tail_break",
        ("service_density", False): "density_break",
        ("service_density", True): "density_continue"}
PAIRS = {"density_break_vs_tail_break": ("tail_break", "density_break"),
         "density_continue_vs_density_break": ("density_break", "density_continue"),
         "density_continue_vs_tail_break": ("tail_break", "density_continue")}


def discover(session):
    dirs = sorted(session.glob("cell-[0-9][0-9]-*"))
    require(len(dirs) == 3, "Expected exactly three serial cells")
    cells = {}
    for directory in dirs:
        index = int(directory.name[5:7])
        cell = read_cell(session, index, directory.name[8:], ordinary=True)
        store = cell["store"]
        key = (store.get("native_victim_rule"),
               store.get("self_preempt_continue_enabled"))
        require(key in ARMS and ARMS[key] not in cells,
                "Missing, duplicate, or unexpected victim/continuation arm")
        cells[ARMS[key]] = cell
    require(set(cells) == set(ARMS.values()), "Three requested arms incomplete")
    require(len({cell["config"].get("workload_sha256") for cell in cells.values()}) == 1,
            "Arm input identities differ")
    return cells


def continuation_receipts(cell, victims, enabled):
    events = cell["store"].get("self_preempt_continuations")
    require(isinstance(events, list), "Continuation event list missing")
    raw = cell["raw"]
    origin = raw["measurement_origin_perf_counter_s"]
    by_internal = {request["internal_request_id"]: request
                   for request in raw["requests"]}
    queries = {(row["step"], by_internal[rid]["request_id"])
               for row in events if row.get("applied") is True
               for rid, tokens in row.get("suffix_scheduled_tokens", {}).items()
               if tokens > 0}
    same_call_output = {}
    for event in raw.get("output_events", []):
        key = (event.get("engine_call_index"), event.get("request_id"))
        if key in queries and event.get("new_token_ids"):
            same_call_output.setdefault(key, event["received_s"])
    actual = {(row["step"], row["selected"])
              for row in victims["rows"]
              if row["actual_native_preemption"] and row["selected"] == row["failed_request"]}
    per_event = []
    unique_tokens, unique_preempted, later_outputs, same_call_outputs = {}, set(), set(), set()
    for row in events:
        step, source = row["step"], row["request"]
        suffix = row["suffix_ids"]
        scheduled = row.get("suffix_scheduled_tokens")
        require((step, source) in actual and row["enabled"] is enabled
                and isinstance(suffix, list) and suffix
                and isinstance(scheduled, dict) and set(scheduled) == set(suffix)
                and all(type(value) is int and value >= 0 for value in scheduled.values())
                and row.get("source_scheduled_tokens") == 0
                and set(row.get("suffix_preempted", [])) <= set(suffix),
                "Continuation receipt does not match native self-preemption/output")
        require(row["applied"] is False or (enabled and row["eligible"] is True
                and row["next_running_visited"] == row["expected_next_request"]
                == suffix[0]), "Applied continuation did not visit original next running request")
        require(row["applied"] is True or row["next_running_visited"] is None,
                "Unapplied continuation visited a successor")
        decorated = dict(row)
        decorated["positive_suffix_output_chains"] = []
        if row["applied"]:
            for request, tokens in scheduled.items():
                key = (step, request)
                require(key not in unique_tokens or unique_tokens[key] == tokens,
                        "Overlapping continuation receipts disagree on scheduled tokens")
                unique_tokens[key] = tokens
                if tokens > 0:
                    source = by_internal[request]["request_id"]
                    event_time = row["host_perf_counter_s"] - origin
                    later = next((time for time in by_internal[request]["token_times_s"]
                                  if time > event_time), None)
                    same_call = same_call_output.get((step, source))
                    if same_call is not None and same_call <= event_time:
                        same_call = None
                    if later is not None:
                        later_outputs.add(key)
                    if same_call is not None:
                        same_call_outputs.add(key)
                    decorated["positive_suffix_output_chains"].append(dict(
                        request=request, source_request_id=source,
                        scheduled_tokens=tokens, first_later_output_s=later,
                        same_engine_call_new_output_s=same_call,
                        join_status=("SAME_ENGINE_CALL_NEW_OUTPUT" if same_call is not None
                                     else "LATER_OUTPUT_ONLY" if later is not None
                                     else "UNKNOWN_NO_RECORDED_LATER_OUTPUT")))
            unique_preempted.update((step, request) for request in row["suffix_preempted"])
        per_event.append(decorated)
    applied = [row for row in per_event if row["applied"]]
    return dict(observed_self_preemptions_with_suffix=len(per_event),
                eligible=sum(row["eligible"] is True for row in per_event),
                applied=len(applied),
                next_running_visited=sum(row["next_running_visited"] is not None
                                         for row in per_event),
                applied_with_positive_suffix_tokens=sum(any(row["suffix_scheduled_tokens"].values())
                                                        for row in applied),
                applied_unique_step_request_suffix_scheduled_tokens=sum(unique_tokens.values()),
                applied_unique_step_request_suffix_preempted=len(unique_preempted),
                applied_positive_suffix_step_request_later_outputs=len(later_outputs),
                applied_positive_suffix_step_request_same_call_outputs=len(same_call_outputs),
                source_scheduled_tokens_total=sum(row["source_scheduled_tokens"]
                                                  for row in per_event),
                rows=per_event)


def analyze(session):
    cells = discover(session)
    victims = {arm: victim_receipts(cell) for arm, cell in cells.items()}
    continuations = {arm: continuation_receipts(cell, victims[arm],
                     arm == "density_continue") for arm, cell in cells.items()}
    ordinary = {arm: dict(ordinary_actions(cell),
                         gate_counts=cell["store"].get("ordinary_backfill_gate_counts"))
                for arm, cell in cells.items()}
    comparisons = {name: compare(cells[old], cells[new])
                   for name, (old, new) in PAIRS.items()}
    scores = {name: score(cells[old], cells[new])
              for name, (old, new) in PAIRS.items()}
    complete = all(cell["complete"] for cell in cells.values())
    require(all(len(result["full_cohort_pair"]["frontier"]) == 20
                for result in comparisons.values()), "Fixed 20-point frontier incomplete")
    require(all(cell["status"].get("forced_rotations") == 0
                and cell["store"].get("applied_rotations") == 0
                for cell in cells.values()), "Unexpected forced rotation")
    action = continuations["density_continue"]
    output_chain = action["applied_positive_suffix_step_request_later_outputs"] > 0
    mechanism = ("NO_ACTION" if action["applied"] == 0 else
                 "CONTINUED_WITH_POSITIVE_SUFFIX_TOKENS_AND_LATER_OUTPUT"
                 if action["applied_with_positive_suffix_tokens"] and output_chain else
                 "POSITIVE_SUFFIX_TOKENS_WITHOUT_CONFIRMED_LATER_OUTPUT"
                 if action["applied_with_positive_suffix_tokens"] else
                 "NEXT_RUNNING_VISITED_WITHOUT_POSITIVE_SUFFIX_TOKENS")
    service_budget = all(scores[name]["rate_at_least_97pct"]
                         and scores[name]["mean_flow_at_most_105pct"]
                         and scores[name]["lower_max_request_gap"]
                         for name in ("density_continue_vs_density_break",
                                      "density_continue_vs_tail_break"))
    validity = dict(
        all_128_complete=complete,
        native_full_saving_all=all(cell["store"].get("store_scope") == "native_full"
                               and cell["store"].get("native_calc_overridden") is False
                               for cell in cells.values()),
        no_forced_rotation_all=all(cell["status"].get("forced_rotations") == 0
                                   and cell["store"].get("applied_rotations") == 0
                                   for cell in cells.values()),
        native_victim_decisions_match_raw_all=all(
            v["unmatched_decisions"] == 0
            and v["changed"] == v["changed_with_actual_preemption"]
            for v in victims.values()),
    )
    mechanism_criteria = dict(
        continuation_applied=action["applied"] > 0,
        original_successor_visited=(action["applied"] > 0 and
                                    action["next_running_visited"] == action["applied"]),
        source_scheduled_zero=action["source_scheduled_tokens_total"] == 0,
        positive_suffix_tokens_and_later_output=(
            action["applied_with_positive_suffix_tokens"] > 0 and output_chain),
    )
    service_criteria = {}
    for control in ("density_break", "tail_break"):
        values = scores[f"density_continue_vs_{control}"]
        service_criteria[f"vs_{control}_rate_at_least_97pct"] = values["rate_at_least_97pct"]
        service_criteria[f"vs_{control}_mean_flow_at_most_105pct"] = values["mean_flow_at_most_105pct"]
        service_criteria[f"vs_{control}_lower_max_gap"] = values["lower_max_request_gap"]
    criteria = dict(**validity, **mechanism_criteria, **service_criteria)
    return dict(
        status="INCOMPLETE_TRIPLET" if not complete else "NO_ACTION" if not action["applied"]
               else "COMPLETE_TRIPLET",
        session=str(session),
        arms={arm: dict(archive=cell["archive"], source_sha256=cell["source_sha256"],
                       metrics=cell["arm"]["metrics"],
                       actual_preemption_count=cell["raw"].get("actual_preemption_count"),
                       forced_rotations=cell["status"].get("forced_rotations"),
                       stop_reason_counts=dict(Counter(request.get("stop_reason")
                                                       for request in cell["raw"]["requests"])))
              for arm, cell in cells.items()},
        pairwise_comparisons=comparisons, pairwise_criterion_values=scores,
        victim_actions=victims, ordinary_backfill_actions=ordinary,
        self_preempt_continuations=continuations,
        validity_criteria=validity,
        mechanism_criteria=mechanism_criteria,
        service_criteria=service_criteria,
        predeclared_criteria=criteria,
        all_criteria_met=all(criteria.values()),
        pilot_all_criteria_met=all(criteria.values()),
        mechanism_verdict=mechanism,
        service_verdict=("INCOMPLETE" if not complete else
                         "NO_ACTION" if not action["applied"] else
                         "OLD_97_105_LOWER_GAP_BUDGET_MET_VS_BOTH_IN_ONE_BLOCK" if service_budget else
                         "OLD_97_105_LOWER_GAP_BUDGET_NOT_MET_VS_BOTH_IN_ONE_BLOCK"),
        limitations=["A visited successor is not necessarily allocated; positive suffix tokens are same-step service, not whole-request benefit.",
                     "Applied suffix receipts are deduplicated by (step, request) for aggregate token/preemption counts; every event keeps its exact per-suffix maps.",
                     "Output event step joins use the recorded engine_call_index; when an indexed output match is absent, later token times are labeled separately rather than assigned to that scheduler call.",
                     "All three arms have different later trajectories; pairwise request/frontier results are not same-state causal effects or a new confirmation."],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be a new file")
    result = analyze(args.session)
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    print(json.dumps(dict(status=result["status"], mechanism=result["mechanism_verdict"],
                          service=result["service_verdict"])))


if __name__ == "__main__":
    main()
