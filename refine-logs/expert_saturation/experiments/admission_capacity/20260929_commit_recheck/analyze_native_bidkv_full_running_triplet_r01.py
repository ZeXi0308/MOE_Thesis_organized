#!/usr/bin/env python3
"""Native-tail, suffix BidKV, and full-running BidKV qualification triplet."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import read_cell
from analyze_native_residency_victim_triplet_r01 import score, victim_receipts
from analyze_protection_yield_triplet_r01 import compare, require


ARMS = {
    ("tail", False): "tail_break",
    ("bidkv_score", False): "bidkv_suffix_break",
    ("bidkv_score", True): "bidkv_full_break",
}
PAIRS = {
    "suffix_vs_tail": ("tail_break", "bidkv_suffix_break"),
    "full_vs_tail": ("tail_break", "bidkv_full_break"),
    "full_vs_suffix": ("bidkv_suffix_break", "bidkv_full_break"),
}


def discover(session):
    directories = sorted(session.glob("cell-[0-9][0-9]-*"))
    require(len(directories) == 3, "Expected exactly three serial cells")
    cells = {}
    for directory in directories:
        cell = read_cell(session, int(directory.name[5:7]), directory.name[8:],
                         ordinary=True)
        store, config = cell["store"], cell["config"]
        key = (store.get("native_victim_rule"),
               store.get("native_victim_full_running_enabled", False))
        require(key in ARMS and ARMS[key] not in cells,
                "Missing, duplicate, or unexpected victim arm")
        require(config.get("native_victim_full_running", False) is key[1],
                "Configured full-running flag differs from applied flag")
        require(store.get("current_victim_guard_enabled") is False
                and store.get("self_preempt_continue_enabled") is False,
                "Guard or continuation changed the baseline contrast")
        cells[ARMS[key]] = cell
    require(set(cells) == set(ARMS.values()), "Three requested arms incomplete")
    require(len({cell["config"].get("workload_sha256") for cell in cells.values()}) == 1,
            "Arm input identities differ")
    return cells


def prefix_actions(cell):
    raw, store = cell["raw"], cell["store"]
    origin = raw["measurement_origin_perf_counter_s"]
    requests = {row["internal_request_id"]: row for row in raw["requests"]}
    metrics = {row["request_id"]: row for row in cell["arm"]["metrics"]["requests"]}
    actual = defaultdict(list)
    for event in raw.get("preemption_events", []):
        if (event.get("original_preemption_called") is True
                and event.get("original_preemption_returned") is True):
            actual[(event.get("engine_call_index"),
                    event.get("internal_request_id"))].append(event)
    same_call_outputs = defaultdict(int)
    for event in raw.get("output_events", []):
        if event.get("new_token_ids"):
            same_call_outputs[(event.get("engine_call_index"),
                               event.get("request_id"))] += len(event["new_token_ids"])
    admissions = defaultdict(list)
    for event in store.get("residency_admissions", []):
        admissions[event.get("request")].append(event)
    rows = []
    for decision in store["victim_decisions"]:
        if decision.get("selected_was_scheduled_prefix") is not True:
            continue
        step, selected = decision["step"], decision["selected"]
        preemptions = actual[(step, selected)]
        preempt = preemptions[0] if len(preemptions) == 1 else None
        preempt_s = preempt.get("method_entered_s") if preempt else None
        victim = requests.get(selected)
        source = victim.get("request_id") if victim else None
        cancelled_same_call_tokens = (same_call_outputs[(step, source)]
                                      if source is not None else None)
        next_output_s = (next((time for time in victim["token_times_s"]
                               if time > preempt_s), None)
                         if victim and preempt_s is not None else None)
        next_admission_s = min((event["host_perf_counter_s"] - origin
                                for event in admissions[selected]
                                if preempt_s is not None
                                and event["host_perf_counter_s"] - origin > preempt_s),
                               default=None)
        rollback = decision.get("selected_prefix_rollback") or {}
        before = decision.get("unprocessed_suffix_start")
        candidate = next((row for row in decision.get("candidates", [])
                          if row.get("request") == selected), None)
        chosen_prefix_valid = (candidate is not None
                               and candidate.get("scheduled_prefix") is True
                               and candidate.get("qualified") is True
                               and type(candidate.get("index")) is int
                               and type(before) is int
                               and candidate["index"] < before
                               and decision.get("full_running_candidate_set") is True)
        refunded = rollback.get("refunded_tokens")
        rollback_valid = (rollback.get("scheduled_running_removed") is True
                          and rollback.get("num_scheduled_tokens_removed") is True
                          and rollback.get("req_to_new_blocks_removed") is True
                          and type(refunded) is int and refunded > 0
                          and rollback.get("req_index_before") == before
                          and rollback.get("req_index_after") == before - 1)
        plan_absent = (decision.get("selected_prefix_absent_from_output_plan") is True
                       and decision.get("selected_prefix_in_native_preempted_ids") is True)
        failures = []
        if not chosen_prefix_valid: failures.append("PREFIX_CANDIDATE_NOT_QUALIFIED_OR_NOT_EARLIER")
        if not rollback_valid: failures.append("PREFIX_ROLLBACK_NOT_VERIFIED")
        if not plan_absent: failures.append("SELECTED_PREFIX_REMAINS_IN_OUTPUT_OR_MISSING_NATIVE_ID")
        if len(preemptions) != 1: failures.append("SAME_STEP_RAW_NATIVE_PREEMPTION_NOT_UNIQUE")
        if cancelled_same_call_tokens != 0: failures.append("CANCELLED_PREFIX_HAS_SAME_CALL_OUTPUT")
        if victim is None: failures.append("VICTIM_REQUEST_MISSING")
        rows.append(dict(
            step=step, failed_current=decision["failed_request"],
            selected_victim=selected, selected_source_request=source,
            candidate_index=candidate.get("index") if candidate else None,
            unprocessed_suffix_start=before,
            chosen_prefix_qualified=chosen_prefix_valid,
            native_same_step_actual_preemption_count=len(preemptions),
            selected_victim_same_call_new_output_tokens=cancelled_same_call_tokens,
            native_actual_preemption_s=preempt_s,
            output_count_at_preemption=(preempt.get("last_returned_output_count")
                                        if preempt else None),
            rollback=decision.get("selected_prefix_rollback"),
            rollback_verified=rollback_valid,
            absent_from_same_step_output_plan=plan_absent,
            refunded_tokens=refunded,
            victim_first_recorded_running_readmission_s=next_admission_s,
            victim_next_new_output_s=next_output_s,
            victim_preemption_to_next_output_s=(next_output_s - preempt_s
                                                if next_output_s is not None else None),
            victim_status=victim.get("status") if victim else None,
            victim_completed=victim.get("status") == "completed" if victim else False,
            victim_completion_s=victim.get("completion_s") if victim else None,
            victim_max_gap_s=metrics[source]["max_gap_s"] if source in metrics else None,
            failures=failures,
        ))
    return dict(
        selected_prefix_actions=len(rows),
        prefix_actions_matching_raw_native_preemption=sum(
            row["native_same_step_actual_preemption_count"] == 1 for row in rows),
        prefix_actions_with_verified_rollback=sum(row["rollback_verified"] for row in rows),
        prefix_actions_absent_from_output_plan=sum(
            row["absent_from_same_step_output_plan"] for row in rows),
        prefix_actions_without_same_call_raw_output=sum(
            row["selected_victim_same_call_new_output_tokens"] == 0 for row in rows),
        total_refunded_scheduled_tokens=sum(
            row["refunded_tokens"] for row in rows
            if type(row["refunded_tokens"]) is int and row["refunded_tokens"] > 0),
        prefix_victims_with_later_output=sum(
            row["victim_next_new_output_s"] is not None for row in rows),
        prefix_victims_completed=sum(row["victim_completed"] for row in rows),
        prefix_actions_with_failures=sum(bool(row["failures"]) for row in rows),
        rows=rows,
    )


def analyze(session):
    cells = discover(session)
    victims = {arm: victim_receipts(cell) for arm, cell in cells.items()}
    comparisons = {name: compare(cells[old], cells[new])
                   for name, (old, new) in PAIRS.items()}
    scores = {name: score(cells[old], cells[new])
              for name, (old, new) in PAIRS.items()}
    require(all(len(pair["full_cohort_pair"]["frontier"]) == 20
                and len(pair["full_cohort_pair"]["per_request"]) == 128
                for pair in comparisons.values()),
            "Fixed 128-request or 20-point frontier incomplete")
    actions = prefix_actions(cells["bidkv_full_break"])
    validity = dict(
        all_128_complete=all(cell["complete"] for cell in cells.values()),
        native_full_saving_all=all(cell["store"].get("store_scope") == "native_full"
                                   and cell["store"].get("native_calc_overridden") is False
                                   for cell in cells.values()),
        no_forced_rotation_all=all(cell["status"].get("forced_rotations") == 0
                                   and cell["store"].get("applied_rotations") == 0
                                   for cell in cells.values()),
        native_victim_decisions_match_raw_all=all(
            receipt["unmatched_decisions"] == 0
            and receipt["changed"] == receipt["changed_with_actual_preemption"]
            for receipt in victims.values()),
    )
    mechanism = dict(
        at_least_one_prefix_action=actions["selected_prefix_actions"] > 0,
        all_prefix_choices_qualified_and_earlier=all(
            row["chosen_prefix_qualified"] for row in actions["rows"]),
        all_prefix_choices_same_step_native_preempted=(
            actions["prefix_actions_matching_raw_native_preemption"]
            == actions["selected_prefix_actions"]),
        all_prefix_rollbacks_verified=(
            actions["prefix_actions_with_verified_rollback"]
            == actions["selected_prefix_actions"]),
        all_prefix_victims_absent_from_output_plan=(
            actions["prefix_actions_absent_from_output_plan"]
            == actions["selected_prefix_actions"]),
        no_cancelled_victim_same_call_raw_output=(
            actions["prefix_actions_without_same_call_raw_output"]
            == actions["selected_prefix_actions"]),
    )
    service = {}
    for control in ("tail", "suffix"):
        values = scores[f"full_vs_{control}"]
        service[f"vs_{control}_rate_at_least_97pct"] = values["rate_at_least_97pct"]
        service[f"vs_{control}_mean_flow_at_most_105pct"] = values["mean_flow_at_most_105pct"]
        service[f"vs_{control}_lower_max_gap"] = values["lower_max_request_gap"]
    qualification = dict(**validity, **mechanism)
    return dict(
        status="INCOMPLETE_TRIPLET" if not validity["all_128_complete"] else
               "NO_PREFIX_ACTION" if actions["selected_prefix_actions"] == 0 else
               "COMPLETE_TRIPLET",
        session=str(session),
        arms={arm: dict(archive=cell["archive"], source_sha256=cell["source_sha256"],
                       metrics=cell["arm"]["metrics"],
                       actual_preemption_count=cell["raw"].get("actual_preemption_count"),
                       forced_rotations=cell["status"].get("forced_rotations"),
                       stop_reason_counts=dict(Counter(request.get("stop_reason")
                                                       for request in cell["raw"]["requests"])))
              for arm, cell in cells.items()},
        pairwise_comparisons=comparisons, pairwise_criterion_values=scores,
        victim_actions=victims, prefix_actions=actions,
        validity_criteria=validity, mechanism_criteria=mechanism,
        qualification_criteria=qualification,
        qualification_met=all(qualification.values()),
        service_criteria=service,
        descriptive_service_budget_met=all(service.values()),
        limitations=[
            "A CPU AST rollback test is not proof of native allocation success; this triplet requires actual same-step native preemption receipts.",
            "Victim later output and completion report observed cost, not a same-state causal effect of prefix choice.",
            "A failed 97/105/lower-gap service budget does not invalidate an otherwise verified prefix implementation.",
            "Each pair uses its own full 128-request cohort and 20 fixed goodput points; no action-conditioned denominator.",
        ],
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
    print(json.dumps(dict(status=result["status"],
                          prefix_actions=result["prefix_actions"]["selected_prefix_actions"],
                          qualification_met=result["qualification_met"],
                          descriptive_service_budget_met=result["descriptive_service_budget_met"])))


if __name__ == "__main__":
    main()
