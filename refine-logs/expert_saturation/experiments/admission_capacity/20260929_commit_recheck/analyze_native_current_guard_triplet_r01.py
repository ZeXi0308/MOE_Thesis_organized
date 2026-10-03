#!/usr/bin/env python3
"""Three-arm BidKV current-victim guard: real action, actor cost, full requests."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import read_cell
from analyze_native_residency_victim_triplet_r01 import score, victim_receipts
from analyze_native_self_preempt_continue_triplet_r01 import continuation_receipts
from analyze_protection_yield_triplet_r01 import compare, require
from analyze_waiter_backfill_triplet_r01 import ordinary_actions


ARMS = {(False, False): "bidkv_break", (True, False): "bidkv_continue",
        (False, True): "bidkv_current_guard"}
PAIRS = {"bidkv_continue_vs_break": ("bidkv_break", "bidkv_continue"),
         "current_guard_vs_break": ("bidkv_break", "bidkv_current_guard"),
         "current_guard_vs_continue": ("bidkv_continue", "bidkv_current_guard")}


def discover(session):
    dirs = sorted(session.glob("cell-[0-9][0-9]-*"))
    require(len(dirs) == 3, "Expected exactly three serial cells")
    cells = {}
    for directory in dirs:
        cell = read_cell(session, int(directory.name[5:7]), directory.name[8:],
                         ordinary=True)
        store = cell["store"]
        key = (store.get("self_preempt_continue_enabled"),
               store.get("current_victim_guard_enabled"))
        require(store.get("native_victim_rule") == "bidkv_score"
                and key in ARMS and ARMS[key] not in cells,
                "Missing, duplicate, or unexpected BidKV guard arm")
        cells[ARMS[key]] = cell
    require(set(cells) == set(ARMS.values()), "Requested arms incomplete")
    require(len({cell["config"].get("workload_sha256") for cell in cells.values()}) == 1,
            "Arm input identities differ")
    return cells


def guard_actions(cell, reference_metrics):
    raw, store = cell["raw"], cell["store"]
    origin = raw["measurement_origin_perf_counter_s"]
    requests = {request["internal_request_id"]: request for request in raw["requests"]}
    metrics = {request["request_id"]: request for request in cell["arm"]["metrics"]["requests"]}
    preemptions = {(event["engine_call_index"], event["internal_request_id"]): event
                   for event in raw["preemption_events"]
                   if event.get("original_preemption_called") is True
                   and event.get("original_preemption_returned") is True}
    output_calls = {}
    for event in raw.get("output_events", []):
        if event.get("new_token_ids"):
            output_calls.setdefault((event["engine_call_index"], event["request_id"]),
                                    event["received_s"])
    rows = []
    for decision in store["victim_decisions"]:
        if decision.get("current_guard_applied") is not True:
            continue
        step, current_id, victim_id = (decision[key] for key in
                                       ("step", "failed_request", "selected"))
        require(decision.get("current_guard_eligible") is True
                and decision.get("unrestricted_selected") == current_id
                and current_id != victim_id,
                "Guard action did not replace an unrestricted self-victim")
        require(type(decision.get("failed_request_scheduled_tokens")) is int
                and type(decision.get("selected_scheduled_tokens")) is int
                and type(decision.get("failed_request_preempted_after_guard")) is bool,
                "Guard postnative receipt missing")
        preempt = preemptions.get((step, victim_id))
        current, victim = requests[current_id], requests[victim_id]
        current_source, victim_source = current["request_id"], victim["request_id"]
        decision_time = decision["host_perf_counter_s"] - origin
        same_call = output_calls.get((step, current_source))
        if same_call is not None and same_call <= decision_time:
            same_call = None
        preempt_time = preempt["method_entered_s"] if preempt else None
        next_output = (next((time for time in victim["token_times_s"]
                             if time > preempt_time), None)
                       if preempt_time is not None else None)
        admissions = [event for event in store["residency_admissions"]
                      if event["request"] == victim_id
                      and preempt_time is not None
                      and event["host_perf_counter_s"] - origin > preempt_time]
        first_admission = min((event["host_perf_counter_s"] - origin
                               for event in admissions), default=None)
        rows.append(dict(
            step=step, decision_s=decision_time, current=current_source,
            selected_victim=victim_source, unrestricted_was_current=True,
            selected_matches_raw_native_preemption=preempt is not None,
            current_scheduled_tokens=decision["failed_request_scheduled_tokens"],
            selected_scheduled_tokens=decision["selected_scheduled_tokens"],
            current_preempted_after_guard=decision["failed_request_preempted_after_guard"],
            current_same_engine_call_new_output_s=same_call,
            victim_actual_preemption_s=preempt_time,
            victim_output_count_at_preemption=(preempt.get("last_returned_output_count")
                                               if preempt else None),
            victim_first_recorded_running_readmission_s=first_admission,
            victim_next_new_output_s=next_output,
            victim_preemption_to_next_output_s=(next_output - preempt_time
                                                if next_output is not None else None),
            victim_completed=victim["status"] == "completed",
            victim_completion_s=victim.get("completion_s"),
            victim_max_gap_s=metrics[victim_source]["max_gap_s"],
            victim_max_gap_in_controls_s={arm: values[victim_source]["max_gap_s"]
                                          for arm, values in reference_metrics.items()},
            victim_flow_s=metrics[victim_source]["flow_s"],
            victim_flow_in_controls_s={arm: values[victim_source]["flow_s"]
                                       for arm, values in reference_metrics.items()},
        ))
    joint = [row for row in rows
             if row["selected_matches_raw_native_preemption"]
             and row["current_scheduled_tokens"] > 0
             and row["current_same_engine_call_new_output_s"] is not None
             and not row["current_preempted_after_guard"]
             and row["selected_scheduled_tokens"] == 0]
    return dict(applied=len(rows),
                changed_victim_actually_preempted=sum(
                    row["selected_matches_raw_native_preemption"] for row in rows),
                current_positive_tokens_and_same_call_output=sum(
                    row["current_scheduled_tokens"] > 0
                    and row["current_same_engine_call_new_output_s"] is not None
                    for row in rows),
                joint_changed_victim_current_output_actions=len(joint),
                current_preempted_after_guard=sum(row["current_preempted_after_guard"]
                                                for row in rows),
                selected_scheduled_tokens_total=sum(row["selected_scheduled_tokens"]
                                                    for row in rows),
                distinct_displaced_victims=len({row["selected_victim"] for row in rows}),
                displaced_victims_with_later_output=sum(
                    row["victim_next_new_output_s"] is not None for row in rows),
                displaced_victims_completed=sum(row["victim_completed"] for row in rows),
                rows=rows)


def analyze(session):
    cells = discover(session)
    victims = {arm: victim_receipts(cell) for arm, cell in cells.items()}
    continuations = {arm: continuation_receipts(cell, victims[arm],
                     arm == "bidkv_continue") for arm, cell in cells.items()}
    ordinary = {arm: dict(ordinary_actions(cell),
                         gate_counts=cell["store"].get("ordinary_backfill_gate_counts"))
                for arm, cell in cells.items()}
    comparisons = {name: compare(cells[old], cells[new])
                   for name, (old, new) in PAIRS.items()}
    scores = {name: score(cells[old], cells[new])
              for name, (old, new) in PAIRS.items()}
    require(all(len(pair["full_cohort_pair"]["frontier"]) == 20
                for pair in comparisons.values()), "Fixed 20-point frontier incomplete")
    references = {arm: {row["request_id"]: row
                        for row in cells[arm]["arm"]["metrics"]["requests"]}
                  for arm in ("bidkv_break", "bidkv_continue")}
    actions = guard_actions(cells["bidkv_current_guard"], references)
    validity = dict(
        all_128_complete=all(cell["complete"] for cell in cells.values()),
        native_full_saving_all=all(cell["store"].get("store_scope") == "native_full"
                               and cell["store"].get("native_calc_overridden") is False
                               for cell in cells.values()),
        no_forced_rotation_all=all(cell["status"].get("forced_rotations") == 0
                                   and cell["store"].get("applied_rotations") == 0
                                   for cell in cells.values()),
        native_victim_decisions_match_raw_all=all(
            result["unmatched_decisions"] == 0
            and result["changed"] == result["changed_with_actual_preemption"]
            for result in victims.values()),
    )
    mechanism = dict(
        guard_changed_victim=actions["applied"] > 0,
        all_guard_selections_actually_preempted=(actions["applied"] > 0 and
            actions["changed_victim_actually_preempted"] == actions["applied"]),
        same_action_changed_victim_and_current_output=(
            actions["joint_changed_victim_current_output_actions"] > 0),
        no_current_preempted_after_guard=actions["current_preempted_after_guard"] == 0,
        selected_source_scheduled_zero=actions["selected_scheduled_tokens_total"] == 0,
    )
    service = {}
    for control in ("bidkv_break", "bidkv_continue"):
        values = scores[f"current_guard_vs_{control.replace('bidkv_', '')}"]
        service[f"vs_{control}_rate_at_least_97pct"] = values["rate_at_least_97pct"]
        service[f"vs_{control}_mean_flow_at_most_105pct"] = values["mean_flow_at_most_105pct"]
        service[f"vs_{control}_lower_max_gap"] = values["lower_max_request_gap"]
    criteria = dict(**validity, **mechanism, **service)
    return dict(
        status="INCOMPLETE_TRIPLET" if not validity["all_128_complete"] else
               "NO_ACTION" if actions["applied"] == 0 else "COMPLETE_TRIPLET",
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
        control_continuations=continuations, current_guard_actions=actions,
        validity_criteria=validity, mechanism_criteria=mechanism,
        service_criteria=service, predeclared_criteria=criteria,
        all_criteria_met=all(criteria.values()),
        mechanism_verdict=("NO_ACTION" if actions["applied"] == 0 else
                           "CURRENT_OUTPUT_AFTER_REAL_CHANGED_VICTIM" if all(mechanism.values())
                           else "GUARD_ACTION_WITH_UNCONFIRMED_OUTPUT_OR_VICTIM"),
        service_verdict=("INCOMPLETE" if not validity["all_128_complete"] else
                         "NO_ACTION" if actions["applied"] == 0 else
                         "OLD_97_105_LOWER_GAP_BUDGET_MET_VS_BOTH_IN_ONE_BLOCK"
                         if all(service.values()) else
                         "OLD_97_105_LOWER_GAP_BUDGET_NOT_MET_VS_BOTH_IN_ONE_BLOCK"),
        limitations=[
            "A current request output in the guard step proves local progress, not a whole-request benefit.",
            "A displaced victim's later pause and completion are observed costs, not a same-state causal estimate of the guard.",
            "A missing running readmission or later token time leaves that interval unknown; no whole gap is automatically called queue wait.",
            "One seen-input serial triplet is not statistical or new-input confirmation."],
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
    print(json.dumps(dict(status=result["status"], actions=result["current_guard_actions"]["applied"],
                          all_criteria_met=result["all_criteria_met"])))


if __name__ == "__main__":
    main()
