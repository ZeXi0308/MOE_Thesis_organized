#!/usr/bin/env python3
"""Native tail versus ordinary prefix work and completion-funded deferral."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import read_cell
from analyze_native_residency_victim_triplet_r01 import score, victim_receipts
from analyze_protection_yield_triplet_r01 import compare, require


MODES = {"off": "tail", "prefix_work": "prefix_work",
         "prefix_finish": "prefix_finish"}
PAIRS = {"work_vs_tail": ("tail", "prefix_work"),
         "finish_vs_tail": ("tail", "prefix_finish"),
         "finish_vs_work": ("prefix_work", "prefix_finish")}


def discover(session):
    directories = sorted(session.glob("cell-[0-9][0-9]-*"))
    require(len(directories) == 3, "Expected exactly three serial cells")
    cells = {}
    for directory in directories:
        cell = read_cell(session, int(directory.name[5:7]), directory.name[8:],
                         ordinary=True)
        store, config = cell["store"], cell["config"]
        mode = store.get("capacity_deferral_mode")
        require(mode in MODES and MODES[mode] not in cells,
                "Missing, duplicate, or unexpected capacity-deferral arm")
        require(config.get("capacity_deferral_mode") == mode,
                "Requested/applied capacity-deferral mode differs")
        require(store.get("native_victim_rule") == "tail"
                and store.get("native_victim_full_running_enabled") is False
                and store.get("current_victim_guard_enabled") is False
                and store.get("self_preempt_continue_enabled") is False,
                "Non-deferral native victim policy differs")
        require(isinstance(store.get("capacity_deferrals"), list),
                "Capacity-deferral receipt list missing")
        cells[MODES[mode]] = cell
    require(set(cells) == set(MODES.values()), "Three requested arms incomplete")
    require(len({cell["config"].get("workload_sha256") for cell in cells.values()}) == 1,
            "Arm input identities differ")
    return cells


def deferral_actions(cell, mode, peer_metrics):
    raw, store, config = cell["raw"], cell["store"], cell["config"]
    archive = Path(cell["archive"])
    qualification = json.loads((archive / "safe-cap-qualification.json").read_text())
    block_size = qualification.get("block_size")
    require(block_size == qualification.get("scheduler_block_size")
            == qualification.get("single_type_block_size") == 16,
            "Runtime block size differs from fixed 16-token KV page")
    require(qualification.get("num_lookahead_tokens") == 0
            and qualification.get("num_spec_tokens") == 0
            and qualification.get("prefix_caching") is False,
            "Lookahead, speculation, or prefix caching changes capacity interpretation")
    require(config.get("max_output_tokens") == 1024,
            "Static hard output cap differs from fixed input")
    origin = raw["measurement_origin_perf_counter_s"]
    requests = {request["internal_request_id"]: request for request in raw["requests"]}
    require(len(requests) == 128
            and all(request.get("max_output_tokens") == 1024
                    for request in requests.values()),
            "Static raw request cap or cohort differs")
    metrics = {request["request_id"]: request
               for request in cell["arm"]["metrics"]["requests"]}
    preemptions = defaultdict(list)
    for event in raw.get("preemption_events", []):
        if (event.get("original_preemption_called") is True
                and event.get("original_preemption_returned") is True):
            preemptions[event["internal_request_id"]].append(event)
    output_tokens = defaultdict(int)
    for event in raw.get("output_events", []):
        if event.get("new_token_ids"):
            output_tokens[(event.get("engine_call_index"),
                           event.get("request_id"))] += len(event["new_token_ids"])
    rows = []
    for action in store["capacity_deferrals"]:
        step = action["step"]
        current_id, witness_id = action["failed_request"], action["witness_request"]
        current, witness = requests.get(current_id), requests.get(witness_id)
        require(current is not None and witness is not None,
                "Deferral actor missing from 128 requests")
        current_source, witness_source = current["request_id"], witness["request_id"]
        decision_s = action["host_perf_counter_s"] - origin
        current_same_call = output_tokens[(step, current_source)]
        witness_same_call = output_tokens[(step, witness_source)]
        current_same_call_preempts = sum(event.get("engine_call_index") == step
                                         for event in preemptions[current_id])
        witness_same_call_preempts = sum(event.get("engine_call_index") == step
                                         for event in preemptions[witness_id])
        current_next_output = next((value for value in current["token_times_s"]
                                    if value > decision_s), None)
        witness_next_output = next((value for value in witness["token_times_s"]
                                    if value > decision_s), None)
        witness_later_preemptions = [event for event in preemptions[witness_id]
                                     if event.get("method_entered_s") is not None
                                     and event["method_entered_s"] > decision_s
                                     and (witness.get("completion_s") is None
                                          or event["method_entered_s"]
                                          < witness["completion_s"])]
        recorded_remaining = action.get("witness_remaining_outputs")
        witness_cap_at_choice = action.get("witness_max_tokens")
        witness_outputs_at_choice = action.get("witness_output_tokens")
        held = action.get("witness_held_blocks")
        computed = action.get("witness_computed_tokens")
        remaining = (witness_cap_at_choice - witness_outputs_at_choice
                     if type(witness_cap_at_choice) is int
                     and type(witness_outputs_at_choice) is int else None)
        recorded_slack = action.get("witness_capacity_slack_tokens")
        expected_slack = (held * block_size - computed - remaining
                          if type(held) is int and type(computed) is int
                          and type(remaining) is int else None)
        predecision_cap_consistent = (
            type(witness_cap_at_choice) is int and witness_cap_at_choice == 1024
            and type(witness_outputs_at_choice) is int and witness_outputs_at_choice >= 0
            and type(remaining) is int and remaining > 0
            and type(held) is int and held > 0
            and type(computed) is int and computed >= 0
            and action.get("witness_capacity_tokens") == held * block_size
            and ((recorded_remaining == remaining and recorded_slack == expected_slack)
                 if mode == "prefix_finish" else
                 (recorded_remaining is None and recorded_slack is None)))
        completion_funded_at_choice = (
            predecision_cap_consistent
            and computed + remaining <= held * block_size)
        prefix_plan_consistent = (
            action.get("mode") == mode and current_id != witness_id
            and type(action.get("witness_index")) is int
            and type(action.get("failed_current_index")) is int
            and action["witness_index"] < action["failed_current_index"]
            and type(action.get("witness_scheduled_tokens_before")) is int
            and action["witness_scheduled_tokens_before"] > 0
            and type(action.get("failed_current_num_new_tokens")) is int
            and action["failed_current_num_new_tokens"] > 0)
        native_plan_consistent = (
            action.get("native_plan_verified") is True
            and action.get("failed_current_scheduled_tokens_final") == 0
            and action.get("failed_current_preempted_final") is False
            and type(action.get("witness_scheduled_tokens_final")) is int
            and action["witness_scheduled_tokens_final"] > 0
            and action.get("witness_preempted_final") is False)
        raw_action_consistent = (
            current_same_call_preempts == 0 and current_same_call == 0
            and witness_same_call_preempts == 0 and witness_same_call > 0)
        failures = []
        if not predecision_cap_consistent: failures.append("PREDECISION_CAP_ARITHMETIC_MISMATCH")
        if mode == "prefix_finish" and not completion_funded_at_choice:
            failures.append("FINISH_WITNESS_NOT_FUNDED_BY_HELD_PAGES")
        if not prefix_plan_consistent: failures.append("PREFIX_PLAN_NOT_VERIFIED")
        if not native_plan_consistent: failures.append("NATIVE_PLAN_NOT_VERIFIED")
        if not raw_action_consistent: failures.append("RAW_DEFER_OR_WITNESS_OUTPUT_MISMATCH")
        rows.append(dict(
            step=step, decision_s=decision_s, mode=mode,
            failed_current=current_source, witness=witness_source,
            failed_current_internal_id=current_id, witness_internal_id=witness_id,
            failed_current_index=action.get("failed_current_index"),
            witness_index=action.get("witness_index"),
            current_output_count_at_choice=action.get("failed_current_output_tokens"),
            witness_output_count_at_choice=witness_outputs_at_choice,
            witness_hard_cap_at_choice=witness_cap_at_choice,
            witness_remaining_output_cap_at_choice=remaining,
            witness_remaining_output_cap_logged=recorded_remaining,
            witness_held_blocks_at_choice=held,
            witness_capacity_slack_tokens_at_choice=expected_slack,
            witness_capacity_slack_tokens_logged=recorded_slack,
            predecision_cap_consistent=predecision_cap_consistent,
            completion_funded_at_choice=completion_funded_at_choice,
            prefix_plan_consistent=prefix_plan_consistent,
            native_plan_consistent=native_plan_consistent,
            raw_current_same_call_preemptions=current_same_call_preempts,
            raw_current_same_call_new_output_tokens=current_same_call,
            raw_witness_same_call_preemptions=witness_same_call_preempts,
            raw_witness_same_call_new_output_tokens=witness_same_call,
            raw_action_consistent=raw_action_consistent,
            current_next_new_output_s=current_next_output,
            current_completion_s=current.get("completion_s"),
            current_status=current.get("status"),
            witness_next_new_output_s=witness_next_output,
            witness_completion_s=witness.get("completion_s"),
            witness_status=witness.get("status"),
            witness_stop_reason=witness.get("stop_reason"),
            witness_final_output_count=len(witness["output_token_ids"]),
            witness_completed_at_hard_cap=(witness.get("status") == "completed"
                                           and witness.get("stop_reason") == "length"
                                           and len(witness["output_token_ids"]) == 1024),
            witness_completed_by_eos=(witness.get("status") == "completed"
                                      and witness.get("stop_reason") == "stop"),
            witness_repreempted_before_completion=bool(witness_later_preemptions),
            witness_later_preemption_steps=[event.get("engine_call_index")
                                              for event in witness_later_preemptions],
            actor_costs={role: dict(
                max_gap_s=metrics[source]["max_gap_s"],
                flow_s=metrics[source]["flow_s"],
                other_arms={arm: dict(max_gap_s=values[source]["max_gap_s"],
                                      flow_s=values[source]["flow_s"])
                            for arm, values in peer_metrics.items()})
                for role, source in (("current", current_source),
                                     ("witness", witness_source))},
            failures=failures,
        ))
    return dict(
        mode=mode, actions=len(rows),
        native_and_raw_action_verified=sum(row["native_plan_consistent"]
                                           and row["raw_action_consistent"]
                                           for row in rows),
        prefix_and_cap_fields_verified=sum(row["prefix_plan_consistent"]
                                           and row["predecision_cap_consistent"]
                                           for row in rows),
        completion_funded_at_choice=sum(row["completion_funded_at_choice"] for row in rows),
        witnesses_repreempted_before_completion=sum(
            row["witness_repreempted_before_completion"] for row in rows),
        witnesses_completed=sum(row["witness_status"] == "completed" for row in rows),
        witnesses_completed_at_hard_cap=sum(
            row["witness_completed_at_hard_cap"] for row in rows),
        witnesses_completed_by_eos=sum(
            row["witness_completed_by_eos"] for row in rows),
        currents_with_later_output=sum(row["current_next_new_output_s"] is not None
                                       for row in rows),
        currents_completed=sum(row["current_status"] == "completed" for row in rows),
        action_rows_with_failures=sum(bool(row["failures"]) for row in rows),
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
    peer_metrics = {arm: {row["request_id"]: row
                          for row in cell["arm"]["metrics"]["requests"]}
                    for arm, cell in cells.items()}
    actions = {arm: deferral_actions(cell, arm, {
        other: metrics for other, metrics in peer_metrics.items() if other != arm})
        for arm, cell in cells.items()}
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
        tail_has_no_deferral=actions["tail"]["actions"] == 0,
        prefix_work_has_actual_action=actions["prefix_work"]["actions"] > 0,
        prefix_finish_has_actual_action=actions["prefix_finish"]["actions"] > 0,
        all_work_actions_native_and_raw_verified=(
            actions["prefix_work"]["actions"] == actions["prefix_work"]["native_and_raw_action_verified"]),
        all_finish_actions_native_and_raw_verified=(
            actions["prefix_finish"]["actions"] == actions["prefix_finish"]["native_and_raw_action_verified"]),
        all_prefix_and_cap_fields_verified=all(
            actions[arm]["actions"] == actions[arm]["prefix_and_cap_fields_verified"]
            for arm in ("prefix_work", "prefix_finish")),
        all_finish_witnesses_funded_at_choice=(
            actions["prefix_finish"]["actions"] ==
            actions["prefix_finish"]["completion_funded_at_choice"]),
    )
    service = {}
    for control in ("tail", "work"):
        values = scores[f"finish_vs_{control}"]
        service[f"vs_{control}_rate_at_least_97pct"] = values["rate_at_least_97pct"]
        service[f"vs_{control}_mean_flow_at_most_105pct"] = values["mean_flow_at_most_105pct"]
        service[f"vs_{control}_lower_max_gap"] = values["lower_max_request_gap"]
    qualification = dict(**validity, **mechanism)
    return dict(
        status="INCOMPLETE_TRIPLET" if not validity["all_128_complete"] else
               "NO_FINISH_ACTION" if not actions["prefix_finish"]["actions"] else
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
        victim_actions=victims, deferral_actions=actions,
        validity_criteria=validity, mechanism_criteria=mechanism,
        qualification_criteria=qualification,
        qualification_met=all(qualification.values()),
        service_criteria=service,
        descriptive_service_budget_met=all(service.values()),
        limitations=[
            "Every event's remaining output cap uses predecision recorded output tokens and the static per-request cap; final outputs are used only for observed terminal reporting.",
            "A numerical fit does not imply the witness stayed resident until completion; later native preemptions are reported for each witness.",
            "Deferral changes the path of current and peer requests; complete-request comparisons are observed serial trajectories, not same-state causal effects.",
            "A failed service budget does not erase verified native/raw deferral actions.",
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
                          actions={arm: row["actions"]
                                   for arm, row in result["deferral_actions"].items()},
                          qualification_met=result["qualification_met"],
                          descriptive_service_budget_met=result["descriptive_service_budget_met"])))


if __name__ == "__main__":
    main()
