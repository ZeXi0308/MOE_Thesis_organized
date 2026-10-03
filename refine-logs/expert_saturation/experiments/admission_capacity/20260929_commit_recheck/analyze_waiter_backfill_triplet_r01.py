#!/usr/bin/env python3
"""Analyze one fixed Q1 / ordinary backfill / primary-first follow-up triplet.

Run only on an archived real three-cell session. Missing action evidence remains a
failed evidence gate; output rates and delays always use all 128 requests per cell.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_capacity_protection_pair_r01 import distribution
from analyze_protection_yield_triplet_r01 import (
    compare, read_cell, request_result, require,
)
from analyze_spare_followup_pair_r01 import followups


ARMS = (
    ("q1", "waiter_reference_q1"),
    ("ordinary", "ordinary_backfill"),
    ("primary_first", "primary_first_followup"),
)
def actual_same_step_preemptions(raw: dict, step: int) -> list[dict] | None:
    records = raw.get("preemption_events")
    if not isinstance(records, list):
        return None
    return [dict(internal_id=event.get("internal_request_id"),
                 entered_s=event.get("method_entered_s"))
            for event in records
            if event.get("engine_call_index") == step
            and event.get("original_preemption_called") is True
            and event.get("original_preemption_returned") is True]


def ordinary_actions(cell: dict) -> dict:
    raw, store = cell["raw"], cell["store"]
    origin = raw["measurement_origin_perf_counter_s"]
    requests = {r["internal_request_id"]: r for r in raw["requests"]}
    events = store.get("events", [])
    require(isinstance(events, list), "Ordinary events missing")
    choices = [e for e in events if e.get("event") == "ordinary_backfill_choice"]
    admissions = [e for e in events if e.get("event") == "ordinary_backfill_admission"]
    require(len({(e["step"], e["target"]) for e in choices}) == len(choices),
            "Duplicate ordinary choice key")
    require(len({e["step"] for e in choices}) == len(choices),
            "More than one ordinary action in one schedule step")
    receipts = {(e["step"], e["target"]): e for e in admissions}
    require(len(receipts) == len(admissions), "Duplicate ordinary admission receipt")
    starts = {(e["step"], e["request"]): e for e in events
              if e.get("event") == "protection_start"
              and e.get("protection_origin") == "ORDINARY_BACKFILL"}
    require(len(starts) == sum(e.get("event") == "protection_start"
                              and e.get("protection_origin") == "ORDINARY_BACKFILL"
                              for e in events), "Duplicate ordinary protection start")
    regular_outputs = [e for e in events if e.get("event") == "protection_release"
                       and e.get("reason") == "OUTPUT_GOAL_REACHED"
                       and e.get("protection_origin") == "REGULAR_COMMIT"]
    ordinary_outputs = [e for e in events if e.get("event") == "protection_release"
                        and e.get("reason") == "OUTPUT_GOAL_REACHED"
                        and e.get("protection_origin") == "ORDINARY_BACKFILL"]
    rows = []
    for choice in choices:
        step, target = choice["step"], choice["target"]
        require(target in requests, "Ordinary target absent from full cohort")
        request = requests[target]
        when = choice["host_perf_counter_s"] - origin
        receipt = receipts.get((step, target))
        start = starts.get((step, target))
        count = choice.get("output_tokens_at_choice")
        times = request["token_times_s"]
        first_output = (times[count] if type(count) is int and 0 <= count < len(times)
                        and times[count] > when else None)
        kind = receipt.get("native_admission") if receipt else None
        admitted = bool(receipt and (
            (kind == "SCHEDULED_TOKENS" and receipt.get("scheduled_tokens", 0) > 0)
            or (kind == "ASYNC_LOAD_ADMITTED" and receipt.get("load_job_ids"))))
        admission_s = receipt["host_perf_counter_s"] - origin if receipt else None
        preemptions = actual_same_step_preemptions(raw, step)
        reported_preemptions = (receipt.get("forced_preemptions_same_step")
                                if receipt else None)
        native_preempted_ids = (receipt.get("actual_preempted_req_ids")
                                if receipt else None)
        prior_regular = [e for e in regular_outputs
                         if e["host_perf_counter_s"] <= choice["host_perf_counter_s"]]
        prior_ordinary = [e for e in ordinary_outputs
                          if e["host_perf_counter_s"] <= choice["host_perf_counter_s"]]
        nearest_regular = max(prior_regular,
                              key=lambda e: e["host_perf_counter_s"], default=None)
        nearest_ordinary = max(prior_ordinary,
                               key=lambda e: e["host_perf_counter_s"], default=None)
        same_regular_step = any(e["step"] == step for e in regular_outputs)
        final = request_result(request)
        chain = bool(start and start.get("protection_origin") == "ORDINARY_BACKFILL"
                     and admitted and admission_s is not None
                     and first_output is not None and first_output > admission_s
                     and final["status"] == "completed")
        rows.append(dict(choice=choice, native_receipt=receipt,
                         ordinary_protection_start=start,
                         native_admitted=admitted,
                         actual_admission_output_completion_chain=chain,
                         choice_s=when, admission_s=admission_s,
                         first_new_output_s=first_output,
                         admission_to_output_s=(first_output - admission_s if chain else None),
                         final=final,
                         actual_preemptions_same_step=preemptions,
                         reported_forced_preemptions_same_step=reported_preemptions,
                         native_actual_preempted_req_ids=native_preempted_ids,
                         same_step_regular_output_release=same_regular_step,
                         preceding_regular_output_release_step=(nearest_regular["step"]
                                                                if nearest_regular else None),
                         steps_since_regular_output_release=(step - nearest_regular["step"]
                                                              if nearest_regular else None),
                         preceding_ordinary_output_release_step=(nearest_ordinary["step"]
                                                                 if nearest_ordinary else None),
                         steps_since_ordinary_output_release=(step - nearest_ordinary["step"]
                                                               if nearest_ordinary else None)))
    kinds = Counter(r["native_receipt"].get("native_admission")
                    if r["native_receipt"] else "MISSING" for r in rows)
    return dict(choices=len(choices), receipts=len(admissions),
                native_admissions=sum(r["native_admitted"] for r in rows),
                actual_output_completion_chains=sum(
                    r["actual_admission_output_completion_chain"] for r in rows),
                ordinary_origin_starts=sum(r["ordinary_protection_start"] is not None
                                           for r in rows),
                same_step_regular_output_release=sum(
                    r["same_step_regular_output_release"] for r in rows),
                outside_same_step_regular_output_release=sum(
                    not r["same_step_regular_output_release"] for r in rows),
                with_prior_ordinary_output_release=sum(
                    r["preceding_ordinary_output_release_step"] is not None for r in rows),
                all_without_same_step_preemption=all(
                    r["actual_preemptions_same_step"] == []
                    and r["reported_forced_preemptions_same_step"] == 0
                    and r["native_actual_preempted_req_ids"] == []
                    for r in rows),
                raw_preemption_evidence_complete=all(
                    r["actual_preemptions_same_step"] is not None for r in rows),
                postnative_preemption_evidence_complete=all(
                    type(r["reported_forced_preemptions_same_step"]) is int
                    and isinstance(r["native_actual_preempted_req_ids"], list)
                    for r in rows),
                admission_kinds=dict(kinds),
                admission_to_output_s=distribution(
                    [r["admission_to_output_s"] for r in rows]),
                steps_since_regular_output_release=distribution(
                    [r["steps_since_regular_output_release"] for r in rows]),
                actions=rows,
                scope=("Ordinary choices have no primary-output trigger field. "
                       "Same-step and previous REGULAR_COMMIT output releases are observational "
                       "relations, not causal ancestry. Later ordinary actions after its own "
                       "release are separate Q1 opportunities, not spare-follow-up recursion. "
                       "An async load receipt is admission, not transfer completion."))


def performance_criteria(reference: dict, candidate: dict, actions: dict,
                         *, primary_first: bool) -> dict:
    old = reference["arm"]["metrics"]
    new = candidate["arm"]["metrics"]
    criteria = dict(full_cohort_complete=reference["complete"] and candidate["complete"],
                    actual_admission_output_completion=(
                        actions["actual_output_completion_chains"] > 0),
                    no_extra_preemption_at_action=(
                        actions["all_without_same_step_preemption"]),
                    rate_at_least_97pct_q1=(new["actual_output_tokens_s"] >=
                                            0.97 * old["actual_output_tokens_s"]),
                    mean_flow_at_most_105pct_q1=(
                        new["mean_flow_with_incomplete_penalty_s"] <=
                        1.05 * old["mean_flow_with_incomplete_penalty_s"]),
                    lower_max_request_gap=(
                        new["max_gap_request_max_s"] is not None
                        and old["max_gap_request_max_s"] is not None
                        and new["max_gap_request_max_s"] < old["max_gap_request_max_s"]))
    if primary_first:
        criteria.update(original_target_output_precedes_followup=
                        actions["all_after_primary_output"],
                        raw_primary_output_precedes_followup=
                        actions["all_after_actual_primary_output"],
                        no_recursive_followups=actions["all_origins_regular"])
        criteria["raw_preemption_evidence_complete"] = (
            actions["raw_preemption_evidence_complete"])
    else:
        criteria["ordinary_origin_recorded"] = (
            actions["ordinary_origin_starts"] == actions["choices"])
        criteria["raw_preemption_evidence_complete"] = (
            actions["raw_preemption_evidence_complete"])
        criteria["postnative_preemption_evidence_complete"] = (
            actions["postnative_preemption_evidence_complete"])
    return criteria


def analyze(session: Path) -> dict:
    cells = {label: read_cell(session, index, name, 1)
             for index, (label, name) in enumerate(ARMS)}
    for label, cell in cells.items():
        expected_ordinary = label == "ordinary"
        expected_spare = label == "primary_first"
        require(cell["config"].get("ordinary_backfill") is expected_ordinary
                and cell["store"].get("ordinary_backfill") is expected_ordinary,
                f"{label}: ordinary backfill flag differs from frozen cell")
        require(cell["config"].get("spare_followup") is expected_spare
                and cell["store"].get("spare_followup") is expected_spare,
                f"{label}: primary-first flag differs from frozen cell")
        present = {e.get("event") for e in cell["store"].get("events", [])}
        forbidden = ({"ordinary_backfill_choice", "ordinary_backfill_admission"}
                     if not expected_ordinary else
                     {"spare_followup_choice", "spare_followup_admission"})
        require(not (present & forbidden),
                f"{label}: another arm's action appears in selective-store")
    ordinary = ordinary_actions(cells["ordinary"])
    primary_first = followups(cells["primary_first"])
    primary_first["raw_preemption_evidence_complete"] = isinstance(
        cells["primary_first"]["raw"].get("preemption_events"), list)
    comparisons = dict(ordinary_vs_q1=compare(cells["q1"], cells["ordinary"]),
                       primary_first_vs_q1=compare(cells["q1"], cells["primary_first"]),
                       primary_first_vs_ordinary=compare(cells["ordinary"],
                                                         cells["primary_first"]))
    criteria = dict(ordinary_vs_q1=performance_criteria(
                        cells["q1"], cells["ordinary"], ordinary,
                        primary_first=False),
                    primary_first_vs_q1=performance_criteria(
                        cells["q1"], cells["primary_first"], primary_first,
                        primary_first=True))
    complete = all(cell["complete"] for cell in cells.values())
    return dict(status="COMPLETE_TRIPLET" if complete else "INCOMPLETE_TRIPLET",
                session=str(session),
                arms={label: dict(archive=cell["archive"],
                                  source_sha256=cell["source_sha256"],
                                  metrics=cell["arm"]["metrics"],
                                  preemption_summary=cell["arm"]["preemption_summary"],
                                  actual_preemption_count=cell["raw"].get(
                                      "actual_preemption_count"),
                                  forced_rotations=cell["status"].get(
                                      "forced_rotations"),
                                  stop_reason_counts=dict(Counter(
                                      r.get("stop_reason")
                                      for r in cell["raw"]["requests"])))
                      for label, cell in cells.items()},
                comparisons=comparisons,
                actions=dict(ordinary=ordinary, primary_first=primary_first),
                predeclared_criteria=criteria,
                criteria_met_by_arm={label: all(values.values())
                                     for label, values in criteria.items()},
                all_criteria_met_for_both=all(all(values.values())
                                               for values in criteria.values()),
                limitations=[
                    "Two candidates compare separately against the same full 128-request Q1 arm; never pool or replace that denominator.",
                    "Ordinary versus primary-first is a direct full-cohort comparison. Action groups overlap and are not causal cost estimates.",
                    "The three cells follow independent natural output trajectories. This is one seen-input development comparison, not equal-work or statistical confirmation.",
                    "If either action has zero chains or unknown postnative/preemption evidence, retain it as a failed evidence gate; no rescue run is implied.",
                    "Eligible backfilling is a strong simple control, not a new scheduling primitive or independent novelty claim.",
                ])


def main() -> None:
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
                          criteria=result["criteria_met_by_arm"],
                          ordinary_choices=result["actions"]["ordinary"]["choices"],
                          primary_first_choices=result["actions"]["primary_first"]["choices"])))


if __name__ == "__main__":
    main()
