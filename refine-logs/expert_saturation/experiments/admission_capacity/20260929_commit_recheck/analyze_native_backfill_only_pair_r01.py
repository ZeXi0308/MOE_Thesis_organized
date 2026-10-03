#!/usr/bin/env python3
"""Compare native full-save with and without ordinary waiter backfill.

The two complete 128-request cells use one fixed input. Actions require native
admission, a later output, and final completion; an empty action set is NO_ACTION.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from analyze_protection_yield_triplet_r01 import compare, require
from analyze_waiter_backfill_triplet_r01 import ordinary_actions
from evaluate_goodput import summarize


ARMS = (("native_full", "native_full_reference"),
        ("ordinary", "native_full_ordinary_only"))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cell(session, index, name, *, ordinary):
    archive = session / f"cell-{index:02d}-{name}" / "archive"
    paths = {key: archive / f"{key}.json" for key in
             ("raw", "config", "status", "selective-store")}
    raw, config, status, store = (json.loads(paths[key].read_text()) for key in paths)
    metrics = summarize(raw, 128, 180)
    requests = raw.get("requests", [])
    complete = (raw.get("status") == "COMPLETE" and raw.get("error") is None
                and status.get("status") == "COMPLETE"
                and status.get("capture_status") == "COMPLETE"
                and status.get("error") is None
                and len(requests) == metrics["completed"]
                == status.get("requests_completed") == 128
                and metrics["failed"] == metrics["unfinished"] == 0)
    require(len({request["request_id"] for request in requests}) == len(requests),
            f"{name}: duplicate request identity")
    require(config.get("store_scope") == store.get("store_scope") == "native_full",
            f"{name}: native full-save scope differs")
    require(store.get("native_calc_overridden") is False,
            f"{name}: native saving calculation changed")
    require(config.get("ordinary_backfill") is ordinary,
            f"{name}: ordinary flag differs from requested arm")
    if ordinary:
        require(store.get("status") == "DRAINED"
                and store.get("ordinary_backfill") is True,
                "Ordinary adapter not installed and drained")
        require(config.get("spare_followup") is False
                and store.get("spare_followup") is False,
                "Ordinary arm includes primary-first follow-up")
        require(store.get("applied_rotations") == status.get("forced_rotations") == 0,
                "Ordinary arm applied a forced rotation")
        require(store.get("allow_forced_rotations") is False,
                "Ordinary arm did not disable forced rotations")
        require(not any(event.get("event") in
                        ("spare_followup_choice", "spare_followup_admission")
                        for event in store.get("events", [])),
                "Ordinary arm contains primary-first action")
    else:
        require(store.get("status") == "NOT_APPLICABLE"
                and store.get("scheduler_schedule_overridden") is False,
                "Native reference unexpectedly installed a scheduler adapter")
        require(status.get("forced_rotations") is None
                and store.get("applied_rotations") is None,
                "Native reference has an adapter forced-rotation counter")
        require(not store.get("events"), "Native reference contains policy events")
    return dict(archive=str(archive), raw=raw, config=config, status=status,
                store=store, arm=dict(metrics=metrics), complete=complete,
                source_sha256={key: sha256(path) for key, path in paths.items()})


def analyze(session):
    cells = {label: read_cell(session, index, name, ordinary=(label == "ordinary"))
             for index, (label, name) in enumerate(ARMS)}
    native, candidate = cells["native_full"], cells["ordinary"]
    action = dict(ordinary_actions(candidate),
                  gate_counts=candidate["store"].get("ordinary_backfill_gate_counts"),
                  allow_forced_rotations=candidate["store"].get("allow_forced_rotations"))
    comparison = compare(native, candidate)
    reference_metrics, candidate_metrics = (cell["arm"]["metrics"]
                                             for cell in (native, candidate))
    full_cohort_complete = native["complete"] and candidate["complete"]
    criteria = dict(
        full_cohort_complete=full_cohort_complete,
        actual_ordinary_action=action["choices"] > 0,
        actual_native_admission_output_completion=action["actual_output_completion_chains"] > 0,
        no_forced_rotation=(candidate["status"].get("forced_rotations") == 0
                            and candidate["store"].get("applied_rotations") == 0),
        no_extra_preemption_at_action=(action["all_without_same_step_preemption"]
                                       and action["raw_preemption_evidence_complete"]
                                       and action["postnative_preemption_evidence_complete"]),
        native_full_saving_both=(native["store"].get("store_scope") == "native_full"
                                 and candidate["store"].get("store_scope") == "native_full"
                                 and native["store"].get("native_calc_overridden") is False
                                 and candidate["store"].get("native_calc_overridden") is False),
        rate_at_least_97pct_native=(reference_metrics["actual_output_tokens_s"] > 0
                                    and candidate_metrics["actual_output_tokens_s"]
                                    >= .97 * reference_metrics["actual_output_tokens_s"]),
        mean_flow_at_most_105pct_native=(
            candidate_metrics["mean_flow_with_incomplete_penalty_s"]
            <= 1.05 * reference_metrics["mean_flow_with_incomplete_penalty_s"]),
        lower_max_request_gap=(candidate_metrics["max_gap_request_max_s"] is not None
                               and reference_metrics["max_gap_request_max_s"] is not None
                               and candidate_metrics["max_gap_request_max_s"]
                               < reference_metrics["max_gap_request_max_s"]),
    )
    state = ("INCOMPLETE_PAIR" if not full_cohort_complete else
             "NO_ACTION" if action["choices"] == 0 else "COMPLETE_PAIR")
    return dict(
        status=state, session=str(session),
        arms={label: dict(archive=cell["archive"], source_sha256=cell["source_sha256"],
                          metrics=cell["arm"]["metrics"],
                          actual_preemption_count=cell["raw"].get("actual_preemption_count"),
                          forced_rotations=cell["status"].get("forced_rotations"),
                          store_scope=cell["store"].get("store_scope"),
                          native_calc_overridden=cell["store"].get("native_calc_overridden"),
                          allow_forced_rotations=cell["store"].get("allow_forced_rotations"),
                          stop_reason_counts=dict(Counter(
                              request.get("stop_reason") for request in cell["raw"]["requests"])))
              for label, cell in cells.items()},
        comparison=comparison,
        ordinary_actions=action,
        action_interpretation=(
            "NO_ACTION: no ordinary choice was recorded; retain gate_counts and do not assign metric differences to backfill."
            if action["choices"] == 0 else
            "Recorded ordinary choices require separate native admission, later output, and completion evidence."),
        criterion_values=dict(rate_ratio_native=ratio(
            candidate_metrics["actual_output_tokens_s"],
            reference_metrics["actual_output_tokens_s"]),
            mean_flow_ratio_native=ratio(
                candidate_metrics["mean_flow_with_incomplete_penalty_s"],
                reference_metrics["mean_flow_with_incomplete_penalty_s"]),
            max_gap_native_s=reference_metrics["max_gap_request_max_s"],
            max_gap_ordinary_s=candidate_metrics["max_gap_request_max_s"]),
        predeclared_criteria=criteria,
        all_criteria_met=all(criteria.values()),
        limitations=[
            "Both arms use native full saving and all 128 requests; the reference has no scheduler adapter and its forced-rotation field is NOT_APPLICABLE, not zero.",
            "An async-load receipt records native admission, not transfer completion; each observed action chain also requires later output and request completion.",
            "If no ordinary action occurs, status is NO_ACTION and no negative or positive action utility is assigned from metric differences.",
            "The two cells have independent output and preemption trajectories. One pair does not identify a same-state action effect or statistical repeatability.",
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
                          choices=result["ordinary_actions"]["choices"],
                          criteria=result["predeclared_criteria"])))


if __name__ == "__main__":
    main()
