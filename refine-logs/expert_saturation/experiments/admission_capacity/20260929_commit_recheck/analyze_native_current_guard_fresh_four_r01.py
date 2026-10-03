#!/usr/bin/env python3
"""Four-arm native victim comparison, with independent fresh-input blocks."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import read_cell
from analyze_native_current_guard_triplet_r01 import guard_actions
from analyze_native_residency_victim_triplet_r01 import score, victim_receipts
from analyze_native_self_preempt_continue_triplet_r01 import continuation_receipts
from analyze_protection_yield_triplet_r01 import compare, require
from analyze_waiter_backfill_triplet_r01 import ordinary_actions


ARMS = {
    ("tail", False, False): "tail_break",
    ("bidkv_score", False, False): "bidkv_break",
    ("bidkv_score", True, False): "bidkv_continue",
    ("bidkv_score", False, True): "bidkv_current_guard",
}
PAIRS = {
    "bidkv_break_vs_tail_break": ("tail_break", "bidkv_break"),
    "bidkv_continue_vs_tail_break": ("tail_break", "bidkv_continue"),
    "bidkv_continue_vs_bidkv_break": ("bidkv_break", "bidkv_continue"),
    "current_guard_vs_tail_break": ("tail_break", "bidkv_current_guard"),
    "current_guard_vs_bidkv_break": ("bidkv_break", "bidkv_current_guard"),
    "current_guard_vs_bidkv_continue": ("bidkv_continue", "bidkv_current_guard"),
}
CONTROLS = ("tail_break", "bidkv_break", "bidkv_continue")


def discover(session):
    dirs = sorted(session.glob("cell-[0-9][0-9]-*"))
    require(len(dirs) == 4, "Expected exactly four serial cells")
    cells = {}
    for directory in dirs:
        cell = read_cell(session, int(directory.name[5:7]), directory.name[8:],
                         ordinary=True)
        store = cell["store"]
        key = (store.get("native_victim_rule"),
               store.get("self_preempt_continue_enabled"),
               store.get("current_victim_guard_enabled"))
        require(key in ARMS and ARMS[key] not in cells,
                "Missing, duplicate, or unexpected four-arm policy")
        cells[ARMS[key]] = cell
    require(set(cells) == set(ARMS.values()), "Four requested arms incomplete")
    require(len({cell["config"].get("workload_sha256") for cell in cells.values()}) == 1,
            "Arm input identities differ within block")
    return cells


def analyze_block(session):
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
                and len(pair["full_cohort_pair"]["per_request"]) == 128
                for pair in comparisons.values()),
            "Fixed 128-request or 20-point comparison incomplete")
    references = {arm: {row["request_id"]: row
                        for row in cells[arm]["arm"]["metrics"]["requests"]}
                  for arm in CONTROLS}
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
    for control in CONTROLS:
        values = scores[f"current_guard_vs_{control}"]
        service[f"vs_{control}_rate_at_least_97pct"] = values["rate_at_least_97pct"]
        service[f"vs_{control}_mean_flow_at_most_105pct"] = values["mean_flow_at_most_105pct"]
        service[f"vs_{control}_lower_max_gap"] = values["lower_max_request_gap"]
    criteria = dict(**validity, **mechanism, **service)
    require(len(criteria) == 18, "Four-arm criterion set changed")
    complete = validity["all_128_complete"]
    return dict(
        status="INCOMPLETE_FOUR" if not complete else
               "NO_ACTION" if actions["applied"] == 0 else "COMPLETE_FOUR",
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
                           "CURRENT_OUTPUT_AFTER_REAL_CHANGED_VICTIM"
                           if all(mechanism.values()) else
                           "GUARD_ACTION_WITH_UNCONFIRMED_OUTPUT_OR_VICTIM"),
        service_verdict=("INCOMPLETE" if not complete else
                         "NO_ACTION" if actions["applied"] == 0 else
                         "OLD_97_105_LOWER_GAP_BUDGET_MET_VS_ALL_THREE_IN_ONE_BLOCK"
                         if all(service.values()) else
                         "OLD_97_105_LOWER_GAP_BUDGET_NOT_MET_VS_ALL_THREE_IN_ONE_BLOCK"),
        limitations=[
            "Displaced victim pause, later output, and completion are observed costs, not same-state causal effects.",
            "Missing recorded running readmission leaves that interval unknown; a whole gap is not labeled queue wait.",
            "Each block keeps its own 128-request denominator and 20-point frontier; no cross-block pooling or best-block selection.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="Analyze one four-arm block")
    parser.add_argument("--first", type=Path, help="First independent four-arm block")
    parser.add_argument("--second", type=Path, help="Second independent four-arm block")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require((args.session is not None) != (args.first is not None or args.second is not None),
            "Choose one --session or both --first and --second")
    require(args.session is not None or (args.first is not None and args.second is not None),
            "Both independent block paths are required")
    require(not args.output.exists(), "Output must be a new file")
    if args.session is not None:
        result = analyze_block(args.session)
        summary = dict(status=result["status"], all_criteria_met=result["all_criteria_met"],
                       guard_actions=result["current_guard_actions"]["applied"])
    else:
        first, second = analyze_block(args.first), analyze_block(args.second)
        result = dict(status="COMPLETE_TWO_BLOCKS" if first["status"] == second["status"] ==
                      "COMPLETE_FOUR" else "TWO_BLOCKS_WITH_INCOMPLETE_OR_NO_ACTION",
                      blocks=dict(first=first, second=second),
                      block_criteria_met=dict(first=first["all_criteria_met"],
                                              second=second["all_criteria_met"]),
                      both_blocks_all_criteria_met=(first["all_criteria_met"]
                                                    and second["all_criteria_met"]),
                      denominator_rule="Each block has separate requests, duration, frontier, and criteria; no pooling or best-block selection.")
        summary = dict(status=result["status"],
                       block_criteria_met=result["block_criteria_met"],
                       both_blocks_all_criteria_met=result["both_blocks_all_criteria_met"])
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
