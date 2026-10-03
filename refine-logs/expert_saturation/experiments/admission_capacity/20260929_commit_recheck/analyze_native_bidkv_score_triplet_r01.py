#!/usr/bin/env python3
"""Adapted BidKV score and native self-preempt continuation, three serial arms."""

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import read_cell
from analyze_native_residency_victim_triplet_r01 import score, victim_receipts
from analyze_native_self_preempt_continue_triplet_r01 import continuation_receipts
from analyze_protection_yield_triplet_r01 import compare, require
from analyze_waiter_backfill_triplet_r01 import ordinary_actions


ARMS = {("tail", False): "tail_break",
        ("bidkv_score", False): "bidkv_break",
        ("bidkv_score", True): "bidkv_continue"}
PAIRS = {"bidkv_break_vs_tail_break": ("tail_break", "bidkv_break"),
         "bidkv_continue_vs_bidkv_break": ("bidkv_break", "bidkv_continue"),
         "bidkv_continue_vs_tail_break": ("tail_break", "bidkv_continue")}


def discover(session):
    dirs = sorted(session.glob("cell-[0-9][0-9]-*"))
    require(len(dirs) == 3, "Expected exactly three serial cells")
    cells = {}
    for directory in dirs:
        cell = read_cell(session, int(directory.name[5:7]), directory.name[8:],
                         ordinary=True)
        store = cell["store"]
        key = (store.get("native_victim_rule"),
               store.get("self_preempt_continue_enabled"))
        require(key in ARMS and ARMS[key] not in cells,
                "Missing, duplicate, or unexpected BidKV/continuation arm")
        cells[ARMS[key]] = cell
    require(set(cells) == set(ARMS.values()), "Requested arms incomplete")
    require(len({cell["config"].get("workload_sha256") for cell in cells.values()}) == 1,
            "Arm input identities differ")
    return cells


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
    complete = all(cell["complete"] for cell in cells.values())
    action = continuations["bidkv_continue"]
    output_chain = action["applied_positive_suffix_step_request_later_outputs"] > 0
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
    mechanism = dict(
        continuation_applied=action["applied"] > 0,
        original_successor_visited=(action["applied"] > 0 and
                                    action["next_running_visited"] == action["applied"]),
        source_scheduled_zero=action["source_scheduled_tokens_total"] == 0,
        positive_suffix_tokens_and_later_output=(
            action["applied_with_positive_suffix_tokens"] > 0 and output_chain),
    )
    service = {}
    for control in ("bidkv_break", "tail_break"):
        values = scores[f"bidkv_continue_vs_{control}"]
        service[f"vs_{control}_rate_at_least_97pct"] = values["rate_at_least_97pct"]
        service[f"vs_{control}_mean_flow_at_most_105pct"] = values["mean_flow_at_most_105pct"]
        service[f"vs_{control}_lower_max_gap"] = values["lower_max_request_gap"]
    criteria = dict(**validity, **mechanism, **service)
    require(len(criteria) == 14, "Frozen continuation criterion set changed")
    score_cells = (cells["tail_break"], cells["bidkv_break"])
    score_actions = (victims["tail_break"], victims["bidkv_break"])
    score_values = scores["bidkv_break_vs_tail_break"]
    score_baseline_criteria = dict(
        both_128_complete=all(cell["complete"] for cell in score_cells),
        native_full_saving_both=all(cell["store"].get("store_scope") == "native_full"
                                    and cell["store"].get("native_calc_overridden") is False
                                    for cell in score_cells),
        no_forced_rotation_both=all(cell["status"].get("forced_rotations") == 0
                                    and cell["store"].get("applied_rotations") == 0
                                    for cell in score_cells),
        native_victim_decisions_match_raw_both=all(
            value["unmatched_decisions"] == 0
            and value["changed"] == value["changed_with_actual_preemption"]
            for value in score_actions),
        bidkv_changed_victim_actually_preempted=(
            victims["bidkv_break"]["changed"] > 0
            and victims["bidkv_break"]["changed"] ==
            victims["bidkv_break"]["changed_with_actual_preemption"]),
        bidkv_break_vs_tail_rate_at_least_97pct=score_values["rate_at_least_97pct"],
        bidkv_break_vs_tail_mean_flow_at_most_105pct=score_values["mean_flow_at_most_105pct"],
        bidkv_break_vs_tail_lower_max_gap=score_values["lower_max_request_gap"],
    )
    if action["applied"] == 0:
        mechanism_verdict = "NO_ACTION"
    elif action["applied_with_positive_suffix_tokens"] and output_chain:
        mechanism_verdict = "CONTINUED_WITH_POSITIVE_SUFFIX_TOKENS_AND_LATER_OUTPUT"
    elif action["applied_with_positive_suffix_tokens"]:
        mechanism_verdict = "POSITIVE_SUFFIX_TOKENS_WITHOUT_CONFIRMED_LATER_OUTPUT"
    else:
        mechanism_verdict = "NEXT_RUNNING_VISITED_WITHOUT_POSITIVE_SUFFIX_TOKENS"
    break_pair = comparisons["bidkv_break_vs_tail_break"]["full_cohort_pair"]
    goodput = [point["goodput_difference_requests_s"]
               for point in break_pair["frontier"]]
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
        adapted_bidkv_break_vs_tail=dict(
            score=scores["bidkv_break_vs_tail_break"],
            changed_native_victim_choices=victims["bidkv_break"]["changed"],
            changed_choices_with_actual_native_preemption=(
                victims["bidkv_break"]["changed_with_actual_preemption"]),
            goodput_20_points=dict(higher=sum(value > 0 for value in goodput),
                                   equal=sum(value == 0 for value in goodput),
                                   lower=sum(value < 0 for value in goodput),
                                   direction="adapted BidKV break minus native-tail break, requests/s"),
            scope="Official default scoring/tie order adapted to this qualified running suffix; not a full BidKV reproduction."),
        score_baseline_criteria=score_baseline_criteria,
        score_baseline_all_criteria_met=all(score_baseline_criteria.values()),
        score_baseline_verdict=(
            "INCOMPLETE" if not score_baseline_criteria["both_128_complete"] else
            "NO_CHANGED_NATIVE_VICTIM_ACTION" if not victims["bidkv_break"]["changed"] else
            "ADAPTED_SCORE_BUDGET_MET_IN_ONE_BLOCK" if all(score_baseline_criteria.values()) else
            "ADAPTED_SCORE_BUDGET_NOT_MET_IN_ONE_BLOCK"),
        validity_criteria=validity, mechanism_criteria=mechanism,
        service_criteria=service, predeclared_criteria=criteria,
        all_criteria_met=all(criteria.values()),
        pilot_all_criteria_met=all(criteria.values()),
        mechanism_verdict=mechanism_verdict,
        service_verdict=("INCOMPLETE" if not complete else "NO_ACTION" if not action["applied"]
                         else "OLD_97_105_LOWER_GAP_BUDGET_MET_VS_BOTH_IN_ONE_BLOCK"
                         if all(service.values()) else
                         "OLD_97_105_LOWER_GAP_BUDGET_NOT_MET_VS_BOTH_IN_ONE_BLOCK"),
        limitations=[
            "BidKV score is adapted to the existing native-full qualified running suffix; it is not the complete upstream scheduler.",
            "The two-arm adapted score baseline criteria are separate from the inherited three-arm continuation criteria; the continuation arm cannot rescue a failed score baseline.",
            "Continuation's positive same-step suffix tokens and later output are separate from whole-request service; all arms evolve different trajectories.",
            "Same-engine-call output uses the recorded engine_call_index. Otherwise the linked later token time is labeled separately.",
            "One seen-input triplet is neither same-state causal evidence nor new-input confirmation."],
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
                          service=result["service_verdict"],
                          pilot_all_criteria_met=result["pilot_all_criteria_met"],
                          score_baseline_all_criteria_met=result["score_baseline_all_criteria_met"])))


if __name__ == "__main__":
    main()
