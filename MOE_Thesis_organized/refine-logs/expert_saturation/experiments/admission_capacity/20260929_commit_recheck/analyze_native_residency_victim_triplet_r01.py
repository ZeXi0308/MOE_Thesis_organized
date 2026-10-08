#!/usr/bin/env python3
"""Complete-request comparison of three native-full victim rules."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_native_backfill_only_pair_r01 import read_cell, ratio
from analyze_protection_yield_triplet_r01 import compare, require
from analyze_waiter_backfill_triplet_r01 import ordinary_actions


RULES = ("tail", "arrival", "service_density")


def victim_receipts(cell):
    decisions = cell["store"].get("victim_decisions")
    require(isinstance(decisions, list), "Native victim decision log missing")
    preemptions = cell["raw"].get("preemption_events")
    require(isinstance(preemptions, list), "Raw actual preemption log missing")
    actual = defaultdict(list)
    for event in preemptions:
        if event.get("original_preemption_called") is True and event.get("original_preemption_returned") is True:
            actual[(event.get("engine_call_index"), event.get("internal_request_id"))].append(event)
    rows = []
    for decision in decisions:
        step, selected = decision["step"], decision["selected"]
        matches = actual[(step, selected)]
        preemption = matches.pop(0) if matches else None
        changed = selected != decision["native_tail"]
        require(decision.get("changed") is changed, "Victim changed flag differs from selected/tail")
        rows.append(dict(
            step=step, rule=decision["rule"], selected=selected,
            native_tail=decision["native_tail"], changed=changed,
            fallback_unknown=decision.get("fallback_unknown"),
            failed_request=decision.get("failed_request"),
            unprocessed_suffix_start=decision.get("unprocessed_suffix_start"),
            free_blocks=decision.get("free_blocks"),
            candidates=decision.get("candidates"),
            actual_native_preemption=preemption is not None,
            actual_preemption_s=preemption.get("method_entered_s") if preemption else None,
            actual_preemption_output_count=(preemption.get("last_returned_output_count")
                                            if preemption else None),
        ))
    unmatched_actual = sum(len(events) for events in actual.values())
    return dict(
        decisions=len(rows), changed=sum(row["changed"] for row in rows),
        matched_actual_preemptions=sum(row["actual_native_preemption"] for row in rows),
        changed_with_actual_preemption=sum(row["changed"] and row["actual_native_preemption"]
                                           for row in rows),
        unmatched_decisions=sum(not row["actual_native_preemption"] for row in rows),
        raw_actual_preemptions_not_joined=unmatched_actual,
        fallback_unknown=sum(bool(row["fallback_unknown"]) for row in rows),
        selected_minus_tail=Counter(row["selected"] for row in rows if row["changed"]),
        rows=rows,
    )


def discover_cells(session):
    directories = sorted(session.glob("cell-[0-9][0-9]-*"))
    require(len(directories) == 3, "Expected exactly three serial cell directories")
    cells = {}
    for directory in directories:
        index = int(directory.name[5:7])
        name = directory.name[8:]
        cell = read_cell(session, index, name, ordinary=True)
        rule = cell["store"].get("native_victim_rule")
        require(rule in RULES and rule not in cells, "Missing or duplicate native victim rule")
        cells[rule] = cell
    require(set(cells) == set(RULES), "Three victim rules incomplete")
    require(len({cell["config"].get("workload_sha256") for cell in cells.values()}) == 1,
            "Victim-rule cells use different input identities")
    return cells


def score(reference, candidate):
    old, new = reference["arm"]["metrics"], candidate["arm"]["metrics"]
    return dict(
        rate_ratio=ratio(new["actual_output_tokens_s"], old["actual_output_tokens_s"]),
        mean_flow_ratio=ratio(new["mean_flow_with_incomplete_penalty_s"],
                              old["mean_flow_with_incomplete_penalty_s"]),
        max_gap_reference_s=old["max_gap_request_max_s"],
        max_gap_candidate_s=new["max_gap_request_max_s"],
        rate_at_least_97pct=(old["actual_output_tokens_s"] > 0
                             and new["actual_output_tokens_s"] >= .97 * old["actual_output_tokens_s"]),
        mean_flow_at_most_105pct=(new["mean_flow_with_incomplete_penalty_s"]
                                  <= 1.05 * old["mean_flow_with_incomplete_penalty_s"]),
        lower_max_request_gap=(new["max_gap_request_max_s"] is not None
                               and old["max_gap_request_max_s"] is not None
                               and new["max_gap_request_max_s"] < old["max_gap_request_max_s"]),
    )


def analyze(session):
    cells = discover_cells(session)
    victims = {rule: victim_receipts(cell) for rule, cell in cells.items()}
    ordinary = {rule: dict(ordinary_actions(cell),
                           gate_counts=cell["store"].get("ordinary_backfill_gate_counts"))
                for rule, cell in cells.items()}
    pairs = {"arrival_vs_tail": ("tail", "arrival"),
             "density_vs_tail": ("tail", "service_density"),
             "density_vs_arrival": ("arrival", "service_density")}
    comparisons = {name: compare(cells[reference], cells[candidate])
                   for name, (reference, candidate) in pairs.items()}
    scores = {name: score(cells[reference], cells[candidate])
              for name, (reference, candidate) in pairs.items()}
    complete = all(cell["complete"] for cell in cells.values())
    native_full = all(cell["store"].get("store_scope") == "native_full"
                      and cell["store"].get("native_calc_overridden") is False
                      for cell in cells.values())
    forced_zero = all(cell["status"].get("forced_rotations") == 0
                      and cell["store"].get("applied_rotations") == 0
                      for cell in cells.values())
    matched = all(result["unmatched_decisions"] == 0 for result in victims.values())
    density_changed = victims["service_density"]["changed"] > 0
    density_changed_admitted = (victims["service_density"]["changed_with_actual_preemption"]
                               == victims["service_density"]["changed"])
    criteria = dict(
        all_128_complete=complete,
        native_full_saving_all=native_full,
        no_adapter_forced_rotation_all=forced_zero,
        victim_decisions_match_raw_actual_preemptions=matched,
        density_changed_native_victim_action=density_changed,
        density_changed_choices_actually_preempted=density_changed_admitted,
        density_vs_tail_rate_at_least_97pct=scores["density_vs_tail"]["rate_at_least_97pct"],
        density_vs_tail_mean_flow_at_most_105pct=scores["density_vs_tail"]["mean_flow_at_most_105pct"],
        density_vs_tail_lower_max_gap=scores["density_vs_tail"]["lower_max_request_gap"],
        density_vs_arrival_rate_at_least_97pct=scores["density_vs_arrival"]["rate_at_least_97pct"],
        density_vs_arrival_mean_flow_at_most_105pct=scores["density_vs_arrival"]["mean_flow_at_most_105pct"],
        density_vs_arrival_lower_max_gap=scores["density_vs_arrival"]["lower_max_request_gap"],
    )
    status = ("INCOMPLETE_TRIPLET" if not complete else
              "NO_ACTION" if not density_changed else "COMPLETE_TRIPLET")
    return dict(
        status=status, session=str(session),
        arms={rule: dict(
            archive=cell["archive"], source_sha256=cell["source_sha256"],
            metrics=cell["arm"]["metrics"],
            output_tokens=cell["arm"]["metrics"]["total_output_tokens"],
            stop_reason_counts=dict(Counter(request.get("stop_reason")
                                            for request in cell["raw"]["requests"])),
            actual_preemption_count=cell["raw"].get("actual_preemption_count"),
            forced_rotations=cell["status"].get("forced_rotations"),
            store_scope=cell["store"].get("store_scope"),
            native_calc_overridden=cell["store"].get("native_calc_overridden"),
        ) for rule, cell in cells.items()},
        pairwise_comparisons=comparisons, pairwise_criterion_values=scores,
        victim_actions=victims, ordinary_backfill_actions=ordinary,
        predeclared_criteria=criteria, all_criteria_met=all(criteria.values()),
        interpretation=("NO_ACTION: density made no changed native victim choice; ordinary backfill is a separate action and cannot satisfy this rule's action requirement."
                        if not density_changed else
                        "Victim decisions are joined to same-step actual native preemptions; ordinary recovery chains are reported separately."),
        limitations=[
            "All three cells use the same fixed input but evolve different output, allocation, and preemption trajectories; pairwise differences are not same-state action effects.",
            "A victim decision at an allocation failure is not itself an observed preemption; the selected request must appear in the raw same-step native preemption log.",
            "The density rule uses output since this native running admission divided by current physical blocks; it is a simple live ratio, not a host-residency or peer-QoE model.",
            "A zero changed-victim count is NO_ACTION even if ordinary backfill occurred or whole-cohort metrics differ.",
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
                          victim_changed={rule: value["changed"]
                                          for rule, value in result["victim_actions"].items()},
                          all_criteria_met=result["all_criteria_met"])))


if __name__ == "__main__":
    main()
