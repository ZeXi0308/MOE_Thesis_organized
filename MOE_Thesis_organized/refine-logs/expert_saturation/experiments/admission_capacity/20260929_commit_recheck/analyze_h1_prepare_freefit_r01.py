"""Count observed numeric free-fit waiters at H1 prepare and commit steps.

This is a state description of the existing diagnostic, not an alternate policy
or proof that an unchosen waiter could have been admitted by native scheduling.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import ijson

from analyze_h1_resource_transition_r02 import ARCHIVE, BASE
from audit_h128_guarded_transfer_r02 import sha_file


TRANSITIONS = BASE / "A_H1_READY_COMMIT_TRANSITIONS_R01_20261001.json"
OUTPUT = BASE / "A_H1_PREPARE_FREEFIT_BOUND_R01_20261001.json"


def fit_waiters(snapshot: dict, selected: str) -> set[str]:
    free = snapshot["free_blocks"]
    return {rid for rid in snapshot["waiting_ids"] if rid != selected and
            snapshot["requests"][rid]["status"] == "PREEMPTED" and
            snapshot["requests"][rid]["held_blocks"] == 0 and
            (snapshot["requests"][rid]["prompt"] +
             snapshot["requests"][rid]["output"] + 15) // 16 <= free}


def analyze() -> dict:
    transition_rows = [r for r in json.loads(TRANSITIONS.read_text())["rows"]
                       if r["commit_reason"] == "KEEP_INSUFFICIENT_FREE_BLOCKS"]
    installed = json.loads((ARCHIVE / "selective-store.json").read_text())
    wanted = {step for r in transition_rows for step in
              (r["prepare_step"], r["commit_step"])}
    snapshots = {s["step"]: s for s in installed["eligibility_snapshots"]
                 if s["step"] in wanted}
    assert len(transition_rows) == 267 and set(snapshots) == wanted
    del installed
    preparatory_steps = {r["prepare_step"] for r in transition_rows}
    scheduler_steps = {}
    # scheduler_steps precedes the 1+ GB cumulative output_events array.
    with (ARCHIVE / "raw.json").open("rb") as stream:
        for step in ijson.items(stream, "scheduler_steps.item", use_float=True):
            if step["step"] in preparatory_steps:
                scheduler_steps[step["step"]] = step
            if step["step"] >= max(preparatory_steps):
                break
    assert set(scheduler_steps) == preparatory_steps

    count = Counter()
    commit_fit_steps = []
    prepare_only_steps = []
    for row in transition_rows:
        p, c = snapshots[row["prepare_step"]], snapshots[row["commit_step"]]
        selected = c["plan_target"]
        prepare_fit, commit_fit = fit_waiters(p, selected), fit_waiters(c, selected)
        count["prepare_fit_steps"] += bool(prepare_fit)
        count["prepare_fit_request_steps"] += len(prepare_fit)
        count["commit_fit_steps"] += bool(commit_fit)
        count["commit_fit_request_steps"] += len(commit_fit)
        age30 = {rid for rid in prepare_fit if rid in p["tracker"]["absent_since"] and
                 p["step"] - p["tracker"]["absent_since"][rid] >= 30}
        count["prepare_fit_age30_steps"] += bool(age30)
        count["prepare_fit_age30_request_steps"] += len(age30)
        tracked = [rid for rid in p["waiting_ids"] if
                   p["requests"][rid]["status"] == "PREEMPTED" and
                   rid in p["tracker"]["absent_since"]]
        oldest = max(tracked, key=lambda rid: (p["step"] -
                    p["tracker"]["absent_since"][rid], rid))
        count["selected_is_longest_absent_at_prepare"] += selected == oldest
        if prepare_fit and not commit_fit:
            prepare_only_steps.append(p["step"])
        if not commit_fit:
            continue
        commit_fit_steps.append(c["step"])
        count["commit_fit_already_present_at_prepare_steps"] += bool(prepare_fit & commit_fit)
        count["commit_fit_already_present_at_prepare_request_steps"] += len(prepare_fit & commit_fit)
        head = p["waiting_ids"][0]
        head_state = p["requests"][head]
        head_need = (head_state["prompt"] + head_state["output"] + 15) // 16
        count["prepare_head_preempted_and_gross_need_exceeds_free"] += (
            head_state["status"] == "PREEMPTED" and head_need > p["free_blocks"])
        native = scheduler_steps[p["step"]]
        count["prepare_no_native_preemption"] += not native["preempted_request_ids"]
        count["prepare_token_budget_not_exhausted"] += native["total_scheduled_tokens"] < 1024
        count["prepare_no_waiting_request_scheduled"] += not any(
            scheduled["internal_request_id"] in p["waiting_ids"]
            for scheduled in native["scheduled"])
    return {
        "status": "OBSERVED_H1_NUMERIC_FREEFIT_BOUND",
        "source_transition_sha256": sha_file(TRANSITIONS),
        "fallback_ready_commits": len(transition_rows),
        "counts": dict(count),
        "commit_fit_steps": commit_fit_steps,
        "prepare_only_fit_steps": prepare_only_steps,
        "source_paths": {"snapshots": str(ARCHIVE / "selective-store.json"),
                         "scheduler_steps": str(ARCHIVE / "raw.json")},
        "interpretation": "The frozen selector chooses only the longest-absent waiter, then rotates a victim if it cannot fit; it does not search smaller waiters. Pinned native FCFS tries the queue head and breaks when allocation fails. In the recorded prepare steps no waiting request was admitted, but the exact native refusal branch was not logged. Numeric gross full-history fit is an upper bound, not proof of host readiness, priority legality, native admission, or benefit.",
    }


if __name__ == "__main__":
    assert not OUTPUT.exists(), "output must be new"
    result = analyze()
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"]},
                     ensure_ascii=False))
