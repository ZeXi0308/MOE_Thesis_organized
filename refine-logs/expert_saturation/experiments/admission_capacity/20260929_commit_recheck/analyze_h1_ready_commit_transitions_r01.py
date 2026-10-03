"""Describe preceding native preemptions for every observed H1 READY commit.

This reads the existing diagnostic trajectory. It does not replay another policy.
The raw parser stops after the scheduler steps, before cumulative output events.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import median

import ijson

from analyze_h1_resource_transition_r02 import ARCHIVE, BASE, QUAL


OUTPUT = BASE / "A_H1_READY_COMMIT_TRANSITIONS_R01_20261001.json"


def describe(values: list[int]) -> dict:
    return {"n": len(values), "min": min(values) if values else None,
            "median": median(values) if values else None,
            "max": max(values) if values else None}


def analyze() -> dict:
    qualification = json.loads(QUAL.read_text())
    installed = json.loads((ARCHIVE / "selective-store.json").read_text())
    events = installed["events"]
    prepares = {e["step"]: e for e in events if e.get("event") == "prepare"}
    checks = {e["step"]: e for e in events if e.get("event") == "commit_check" and e.get("reason") == "READY"}
    rechecks = {e["step"]: e for e in events if e.get("event") == "commit_recheck"}
    assert len(checks) == len(rechecks) == 269 and set(checks) == set(rechecks)
    wanted = {s for commit in checks for s in (commit - 1, commit)}
    snapshots = {x["step"]: x for x in installed["eligibility_snapshots"] if x["step"] in wanted}
    assert set(snapshots) == wanted
    del installed

    internal_to_source = {}
    with (ARCHIVE / "raw.json").open("rb") as stream:
        for row in ijson.items(stream, "requests.item", use_float=True):
            internal_to_source[row["internal_request_id"]] = row["request_id"]
            if len(internal_to_source) == 128:
                break
    assert len(internal_to_source) == 128
    steps = {}
    with (ARCHIVE / "raw.json").open("rb") as stream:
        for row in ijson.items(stream, "scheduler_steps.item", use_float=True):
            if row["step"] in wanted:
                steps[row["step"]] = row
            if row["step"] >= max(wanted):
                break
    assert set(steps) == wanted

    rows = []
    for s in sorted(checks):
        p = s - 1
        check, decision = checks[s], rechecks[s]
        prepare = prepares[p]
        assert (prepare["target"], prepare["victim"]) == (check["target"], check["victim"])
        assert (decision["target"], decision["planned_victim"]) == (check["target"], check["victim"])
        before, after = snapshots[p], snapshots[s]
        target, victim = check["target"], check["victim"]
        assert target in before["requests"] and target in after["requests"]
        need = (after["requests"][target]["prompt"] +
                after["requests"][target]["output"] + 15) // 16
        assert need == (before["requests"][target]["prompt"] +
                        before["requests"][target]["output"] + 15) // 16
        # Adapter-forced victims share native_capture's preempted list. Remove
        # actual commit_recheck victims before labeling a preceding preemption native.
        forced = [e for e in events if e.get("event") == "commit_recheck" and
                  e.get("step") == p and e.get("reason") != "DIRECT_READY"]
        forced_source = {internal_to_source[e["planned_victim"]] for e in forced}
        recorded = steps[p]["preempted_request_ids"]
        assert forced_source <= set(recorded)
        native = [rid for rid in recorded if rid not in forced_source]
        # A prepare is the mutually exclusive noncommit branch at this step.
        assert not forced
        victim_source = internal_to_source[victim]
        commit_preempted = steps[s]["preempted_request_ids"]
        direct = decision["reason"] == "DIRECT_READY"
        assert (victim_source in commit_preempted) is not direct
        rows.append({
            "commit_step": s, "prepare_step": p,
            "target_request_id": internal_to_source[target],
            "planned_victim_request_id": victim_source,
            "preceding_native_preempted_source_ids": native,
            "free_before_prepare": before["free_blocks"],
            "free_before_commit": after["free_blocks"],
            "target_full_need_blocks": need,
            "free_minus_need_before_prepare": before["free_blocks"] - need,
            "free_minus_need_before_commit": after["free_blocks"] - need,
            "commit_reason": decision["reason"],
            "planned_victim_preempted_at_commit": victim_source in commit_preempted,
        })
    native_rows = [r for r in rows if r["preceding_native_preempted_source_ids"]]
    no_native_rows = [r for r in rows if not r["preceding_native_preempted_source_ids"]]
    direct_rows = [r for r in rows if r["commit_reason"] == "DIRECT_READY"]
    assert {r["commit_step"] for r in direct_rows} == {
        chain["step"] for chain in qualification["direct_action"]["chains"]}
    return {
        "status": "OBSERVED_H1_READY_COMMIT_TRANSITIONS",
        "source_qualification_audit": QUAL.name,
        "source_raw_sha256": qualification["raw_sha256"],
        "ready_commits": len(rows),
        "preceded_by_native_preemption": len(native_rows),
        "preceded_by_no_native_preemption": len(no_native_rows),
        "native_preemption_request_events": sum(len(r["preceding_native_preempted_source_ids"])
                                                for r in native_rows),
        "native_preemption_then_planned_victim_preempted": sum(
            r["planned_victim_preempted_at_commit"] for r in native_rows),
        "native_preemption_then_direct": sum(r["commit_reason"] == "DIRECT_READY"
                                              for r in native_rows),
        "all_commit_reasons": dict(Counter(r["commit_reason"] for r in rows)),
        "free_minus_full_need_blocks": {
            "all_before_prepare": describe([r["free_minus_need_before_prepare"] for r in rows]),
            "all_before_commit": describe([r["free_minus_need_before_commit"] for r in rows]),
            "native_preemption_before_commit": describe([
                r["free_minus_need_before_commit"] for r in native_rows]),
            "no_native_preemption_before_commit": describe([
                r["free_minus_need_before_commit"] for r in no_native_rows]),
        },
        "crossed_from_negative_to_full_fit": sum(
            r["free_minus_need_before_prepare"] < 0 <= r["free_minus_need_before_commit"]
            for r in rows),
        "rows": rows,
        "scope": "Observed diagnostic trajectory only. A prior native preemption and a free-block margin do not identify a counterfactual policy effect or saved transfer cost. Scheduler steps were streamed before cumulative output events; raw original was previously hash-qualified in the H1 qualification audit.",
    }


if __name__ == "__main__":
    assert not OUTPUT.exists(), "output must be new"
    result = analyze()
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "status", "ready_commits", "preceded_by_native_preemption",
        "native_preemption_then_planned_victim_preempted",
        "native_preemption_then_direct", "all_commit_reasons")}, ensure_ascii=False))
