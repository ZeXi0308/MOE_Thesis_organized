#!/usr/bin/env python3
"""Compare native tail with smallest/largest sufficient suffix victim rules."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import read_cell
from analyze_native_residency_victim_triplet_r01 import score, victim_receipts
from analyze_protection_yield_triplet_r01 import compare, require


RULES = {"tail", "min_held_other", "max_held_other"}
PAIRS = {"min_vs_tail": ("tail", "min_held_other"),
         "max_vs_tail": ("tail", "max_held_other"),
         "min_vs_max": ("max_held_other", "min_held_other")}


def discover(session):
    directories = sorted(session.glob("cell-[0-9][0-9]-*"))
    require(len(directories) == 3, "Expected exactly three serial cells")
    cells = {}
    for directory in directories:
        cell = read_cell(session, int(directory.name[5:7]), directory.name[8:],
                         ordinary=True)
        store, config = cell["store"], cell["config"]
        rule = store.get("native_victim_rule")
        require(rule in RULES and rule not in cells,
                "Missing, duplicate, or unexpected size-victim arm")
        require(config.get("native_victim_rule") == rule,
                "Requested/applied native victim rule differs")
        require(store.get("native_victim_full_running_enabled") is False
                and config.get("native_victim_full_running") is False
                and store.get("current_victim_guard_enabled") is False
                and store.get("self_preempt_continue_enabled") is False
                and store.get("capacity_deferral_mode") == "off"
                and config.get("capacity_deferral_mode") == "off",
                "Another victim or deferral policy changed the contrast")
        cells[rule] = cell
    require(set(cells) == RULES, "Three requested arms incomplete")
    require(len({cell["config"].get("workload_sha256") for cell in cells.values()}) == 1,
            "Arm input identities differ")
    return cells


def size_actions(cell, peer_metrics):
    raw, store = cell["raw"], cell["store"]
    rule = store["native_victim_rule"]
    if rule == "tail":
        return dict(rule=rule, decisions=0, qualified_choices=0,
                    changed_qualified_actual_actions=0, rows=[])
    qualification = json.loads((Path(cell["archive"]) /
                                "safe-cap-qualification.json").read_text())
    block_size = qualification.get("block_size")
    require(block_size == qualification.get("scheduler_block_size") == 16,
            "KV block size differs from fixed 16")
    origin = raw["measurement_origin_perf_counter_s"]
    requests = {request["internal_request_id"]: request
                for request in raw["requests"]}
    metrics = {request["request_id"]: request
               for request in cell["arm"]["metrics"]["requests"]}
    preemptions = defaultdict(list)
    for event in raw.get("preemption_events", []):
        if (event.get("original_preemption_called") is True
                and event.get("original_preemption_returned") is True):
            preemptions[(event.get("engine_call_index"),
                         event.get("internal_request_id"))].append(event)
    outputs = defaultdict(int)
    for event in raw.get("output_events", []):
        if event.get("new_token_ids"):
            outputs[(event.get("engine_call_index"),
                     event.get("request_id"))] += len(event["new_token_ids"])
    admissions = defaultdict(list)
    for event in store.get("residency_admissions", []):
        admissions[event.get("request")].append(event)
    rows = []
    for decision in store["victim_decisions"]:
        require(decision.get("rule") == rule, "Decision rule differs from arm")
        step, selected_id, current_id = (decision[key] for key in
                                         ("step", "selected", "failed_request"))
        selected, current = requests.get(selected_id), requests.get(current_id)
        require(selected is not None and current is not None,
                "Decision actor missing from fixed 128 requests")
        selected_source, current_source = (selected["request_id"], current["request_id"])
        candidate_rows = decision["candidates"]
        selected_row = next((row for row in candidate_rows
                             if row.get("request") == selected_id), None)
        tail_row = next((row for row in candidate_rows
                         if row.get("request") == decision["native_tail"]), None)
        require(selected_row is not None and tail_row is not None,
                "Selected or native-tail candidate missing")
        index = decision["unprocessed_suffix_start"]
        deficit = decision.get("size_rule_deficit_blocks")
        free = decision.get("free_blocks")
        held = decision.get("size_rule_current_held_blocks")
        computed = decision.get("size_rule_current_computed_tokens")
        expected_deficit = ((computed + 1 + block_size - 1) // block_size
                            - held - free if all(type(value) is int
                                                 for value in (computed, held, free))
                            else None)
        sufficient = [row for row in candidate_rows
                      if row.get("qualified") is True
                      and type(row.get("index")) is int and row["index"] > index
                      and row.get("already_scheduled") is False
                      and type(row.get("held_blocks")) is int
                      and type(deficit) is int and row["held_blocks"] >= deficit]
        expected = (min(sufficient, key=lambda row: (row["held_blocks"], -row["index"]))
                    if sufficient and rule == "min_held_other" else
                    max(sufficient, key=lambda row: (row["held_blocks"], row["index"]))
                    if sufficient else None)
        fallback = decision.get("size_rule_fallback_reason")
        qualified_choice = fallback is None
        choice_consistent = (deficit == expected_deficit
                             and decision.get("size_rule_sufficient_other_count")
                             == len(sufficient)
                             and (expected is not None
                                  and expected["request"] == selected_id
                                  and selected_row["qualified"] is True
                                  and selected_row["already_scheduled"] is False
                                  and selected_row["held_blocks"] >= deficit
                                  if qualified_choice else
                                  selected_id == decision["native_tail"]))
        native = preemptions[(step, selected_id)]
        actual = native[0] if len(native) == 1 else None
        decision_s = decision["host_perf_counter_s"] - origin
        preempt_s = actual.get("method_entered_s") if actual else None
        selected_next_output_s = (next((time for time in selected["token_times_s"]
                                        if time > preempt_s), None)
                                  if preempt_s is not None else None)
        selected_admission_s = (min((event["host_perf_counter_s"] - origin
                                     for event in admissions[selected_id]
                                     if preempt_s is not None
                                     and event["host_perf_counter_s"] - origin > preempt_s),
                                    default=None))
        current_next_output_s = next((time for time in current["token_times_s"]
                                      if time > decision_s), None)
        selected_cost = metrics[selected_source]
        current_cost = metrics[current_source]
        rows.append(dict(
            step=step, decision_s=decision_s, rule=rule,
            failed_current=current_source, selected_victim=selected_source,
            native_tail=decision["native_tail"],
            changed_from_native_tail=selected_id != decision["native_tail"],
            selected_index=selected_row["index"], native_tail_index=tail_row["index"],
            selected_held_blocks=selected_row["held_blocks"],
            native_tail_held_blocks=tail_row["held_blocks"],
            current_held_blocks=held, current_computed_tokens=computed,
            free_blocks_before=free, needed_blocks=deficit,
            sufficient_other_count=len(sufficient),
            fallback_reason=fallback, qualified_choice=qualified_choice,
            choice_consistent=choice_consistent,
            actual_native_preemption_count=len(native),
            selected_preemption_s=preempt_s,
            selected_output_count_at_preemption=(actual.get("last_returned_output_count")
                                                 if actual else None),
            failed_current_same_call_new_output_tokens=outputs[(step, current_source)],
            failed_current_next_output_s=current_next_output_s,
            failed_current_decision_to_next_output_s=(current_next_output_s-decision_s
                                                      if current_next_output_s is not None
                                                      else None),
            failed_current_completed=current["status"] == "completed",
            selected_first_recorded_running_readmission_s=selected_admission_s,
            selected_next_new_output_s=selected_next_output_s,
            selected_preemption_to_next_output_s=(selected_next_output_s-preempt_s
                                                  if selected_next_output_s is not None
                                                  else None),
            selected_completed=selected["status"] == "completed",
            selected_completion_s=selected.get("completion_s"),
            selected_complete_request_costs=dict(
                max_gap_s=selected_cost["max_gap_s"], flow_s=selected_cost["flow_s"],
                other_arms={arm: dict(max_gap_s=values[selected_source]["max_gap_s"],
                                      flow_s=values[selected_source]["flow_s"])
                            for arm, values in peer_metrics.items()}),
            failed_current_complete_request_costs=dict(
                max_gap_s=current_cost["max_gap_s"], flow_s=current_cost["flow_s"],
                other_arms={arm: dict(max_gap_s=values[current_source]["max_gap_s"],
                                      flow_s=values[current_source]["flow_s"])
                            for arm, values in peer_metrics.items()}),
        ))
    return dict(
        rule=rule, decisions=len(rows),
        qualified_choices=sum(row["qualified_choice"] for row in rows),
        fallback_reasons=dict(Counter(row["fallback_reason"] for row in rows
                                      if row["fallback_reason"] is not None)),
        changed_from_native_tail=sum(row["changed_from_native_tail"] for row in rows),
        changed_qualified_actual_actions=sum(
            row["changed_from_native_tail"] and row["qualified_choice"]
            and row["choice_consistent"] and row["actual_native_preemption_count"] == 1
            and row["failed_current_same_call_new_output_tokens"] > 0
            and row["failed_current_next_output_s"] is not None for row in rows),
        all_choices_consistent=all(row["choice_consistent"] for row in rows),
        all_selected_actually_preempted=all(row["actual_native_preemption_count"] == 1
                                            for row in rows),
        qualified_choices_with_same_call_current_output=sum(
            row["qualified_choice"]
            and row["failed_current_same_call_new_output_tokens"] > 0 for row in rows),
        qualified_choices_with_later_current_output=sum(
            row["qualified_choice"] and row["failed_current_next_output_s"] is not None
            for row in rows),
        selected_victims_with_later_output=sum(
            row["selected_next_new_output_s"] is not None for row in rows),
        selected_victims_completed=sum(row["selected_completed"] for row in rows),
        rows=rows,
    )


def analyze(session):
    cells = discover(session)
    victims = {rule: victim_receipts(cell) for rule, cell in cells.items()}
    metrics = {rule: {row["request_id"]: row for row in cell["arm"]["metrics"]["requests"]}
               for rule, cell in cells.items()}
    actions = {rule: size_actions(cell, {other: values for other, values in metrics.items()
                                        if other != rule})
               for rule, cell in cells.items()}
    pairs = {name: compare(cells[reference], cells[candidate])
             for name, (reference, candidate) in PAIRS.items()}
    require(all(len(pair["full_cohort_pair"]["per_request"]) == 128
                and len(pair["full_cohort_pair"]["frontier"]) == 20
                for pair in pairs.values()), "Fixed cohort/frontier differs")
    scores = {name: score(cells[reference], cells[candidate])
              for name, (reference, candidate) in PAIRS.items()}
    validity = dict(
        all_128_complete=all(cell["complete"] for cell in cells.values()),
        native_full_saving_all=all(cell["store"].get("store_scope") == "native_full"
                                   and cell["store"].get("native_calc_overridden") is False
                                   for cell in cells.values()),
        no_forced_rotations_all=all(cell["status"].get("forced_rotations") == 0
                                    and cell["store"].get("applied_rotations") == 0
                                    for cell in cells.values()),
        native_victim_decisions_match_raw_all=all(
            result["unmatched_decisions"] == 0
            and result["changed"] == result["changed_with_actual_preemption"]
            for result in victims.values()),
        size_choices_consistent_both=all(actions[rule]["all_choices_consistent"]
                                         for rule in ("min_held_other", "max_held_other")),
        size_selected_preemptions_real_both=all(
            actions[rule]["all_selected_actually_preempted"]
            for rule in ("min_held_other", "max_held_other")),
    )
    mechanism = dict(
        min_has_changed_qualified_actual_action=(
            actions["min_held_other"]["changed_qualified_actual_actions"] > 0),
        max_has_changed_qualified_actual_action=(
            actions["max_held_other"]["changed_qualified_actual_actions"] > 0),
    )
    service = {}
    for control in ("tail", "max"):
        values = scores[f"min_vs_{control}"]
        service[f"vs_{control}_rate_at_least_97pct"] = values["rate_at_least_97pct"]
        service[f"vs_{control}_mean_flow_at_most_105pct"] = values["mean_flow_at_most_105pct"]
        service[f"vs_{control}_lower_max_gap"] = values["lower_max_request_gap"]
    qualification = dict(**validity, **mechanism)
    return dict(
        status="INCOMPLETE_TRIPLET" if not validity["all_128_complete"] else
               "NO_MIN_ACTION" if not mechanism["min_has_changed_qualified_actual_action"]
               else "COMPLETE_TRIPLET",
        session=str(session),
        arms={rule: dict(archive=cell["archive"], source_sha256=cell["source_sha256"],
                         metrics=cell["arm"]["metrics"],
                         actual_preemption_count=cell["raw"].get("actual_preemption_count"),
                         forced_rotations=cell["status"].get("forced_rotations"),
                         stop_reason_counts=dict(Counter(request.get("stop_reason")
                                                         for request in cell["raw"]["requests"])))
              for rule, cell in cells.items()},
        pairwise_comparisons=pairs, pairwise_criterion_values=scores,
        victim_actions=victims, size_actions=actions,
        validity_criteria=validity, mechanism_criteria=mechanism,
        qualification_criteria=qualification, qualification_met=all(qualification.values()),
        service_criteria=service, descriptive_service_budget_met=all(service.values()),
        limitations=[
            "All three arms are observed serial trajectories; each victim and current cost is descriptive, not a same-state action effect.",
            "A qualified candidate and one actual native preemption do not prove a cheaper host restore; transfer and host coverage are not measured here.",
            "The size rules retain native allocation retry and fallback; a changed victim choice is distinct from the request's later observed output.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be a new file")
    result = analyze(args.session)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"],
                          changed_actual={rule: row["changed_qualified_actual_actions"]
                                          for rule, row in result["size_actions"].items()},
                          qualification_met=result["qualification_met"],
                          descriptive_service_budget_met=result["descriptive_service_budget_met"])))


if __name__ == "__main__":
    main()
