#!/usr/bin/env python3
"""Recount numerical second-waiter headroom in the completed H1 diagnostic."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "moe-a-h1-guard-qual-session-r02-20260930" /
          "cell-00-eager_diagnostic_on/archive/selective-store.json")
OUTPUT = HERE / "A_SPARE_FOLLOWUP_OPPORTUNITY_R01_20261001.json"


def need(row):
    # The snapshot logs current output count and owned GPU blocks at begin().
    return max(0, (row["prompt"] + row["output"] + 15) // 16 - row["held_blocks"])


def preempted_waiter(snapshot, rid):
    row = snapshot["requests"].get(rid)
    return row is not None and row["status"] == "PREEMPTED" and rid in snapshot["waiting_ids"]


def state(snapshot, rid):
    row = snapshot["requests"].get(rid)
    return None if row is None else {
        "status": row["status"], "output_tokens": row["output"],
        "held_blocks": row["held_blocks"], "full_history_need_blocks": need(row),
    }


def main():
    digest = hashlib.sha256()
    with SOURCE.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    with SOURCE.open() as stream:
        data = json.load(stream)
    snapshots = {row["step"]: row for row in data["eligibility_snapshots"]}
    events = defaultdict(list)
    first_outputs = defaultdict(list)
    for event in data["events"]:
        events[event["step"]].append(event)
        if event["event"] == "target_new_output":
            first_outputs[event["request"]].append(event["step"])

    ready = [e for e in data["events"] if e["event"] == "commit_check" and e["reason"] == "READY"]
    direct_steps = {e["step"] for e in data["events"] if e["event"] == "direct_commit"}
    forced = [e for e in ready if e["step"] not in direct_steps]
    assert len(ready) == 269 and len(direct_steps) == 2 and len(forced) == 267

    cofit = []
    for commit in forced:
        step = commit["step"]
        before = snapshots[step]
        victim_id, target_id = commit["victim"], commit["target"]
        victim = before["requests"][victim_id]
        target = before["requests"][target_id]
        free_after_release_numerical = before["free_blocks"] + victim["held_blocks"]
        target_need = need(target)
        alternatives = []
        for rid in before["waiting_ids"]:
            if rid == target_id or not preempted_waiter(before, rid):
                continue
            candidate_need = need(before["requests"][rid])
            if free_after_release_numerical >= target_need + candidate_need:
                alternatives.append({"request": rid, "need_blocks": candidate_need})
        if not alternatives:
            continue

        after = snapshots[step + 1]
        target_first_step = next((s for s in first_outputs[target_id] if s > step), None)
        assert target_first_step is not None
        first = snapshots[target_first_step]
        after_fit = [a for a in alternatives if preempted_waiter(after, a["request"])
                     and after["free_blocks"] >= need(after["requests"][a["request"]])]
        first_fit = [a for a in alternatives if preempted_waiter(first, a["request"])
                     and first["free_blocks"] >= need(first["requests"][a["request"]])]
        first_aged_fit = [a for a in first_fit if a["request"] in first["tracker"]["absent_since"]
                          and target_first_step - first["tracker"]["absent_since"][a["request"]] >= 30]
        commit_aged_fit = [a for a in alternatives if a["request"] in before["tracker"]["absent_since"]
                           and step - before["tracker"]["absent_since"][a["request"]] >= 30]
        head_id = after["waiting_ids"][0] if after["waiting_ids"] else None
        head = after["requests"].get(head_id) if head_id else None
        cofit.append({
            "step": step, "victim": victim_id, "target": target_id,
            "alternatives": alternatives, "after_fit": after_fit,
            "first_output_fit": first_fit, "first_output_aged_fit": first_aged_fit,
            "commit_aged_fit": commit_aged_fit,
            "first_output_step": target_first_step,
            "after_waiting_head": head_id,
            "after_waiting_head_need_blocks": need(head) if head else None,
            "after_free_blocks": after["free_blocks"],
            "after_running_count": len(after["running_ids"]),
            "after_skipped_count": len(after["skipped_ids"]),
            "target_async_after_commit": after["requests"][target_id]["status"] == "WAITING_FOR_REMOTE_KVS",
            "commit_metadata_load": any(e["event"] == "metadata" and bool(e["loads"]) for e in events[step]),
            "commit_metadata_flush": any(e["event"] == "metadata" and bool(e["flush"]) for e in events[step]),
            "first_output_free_blocks": first["free_blocks"],
        })

    counts = {
        "ready_commits": len(ready),
        "direct_commits_excluded": len(direct_steps),
        "forced_victim_commits": len(forced),
        "numerically_cofit_commit_steps": len(cofit),
        "numerically_cofit_alternative_instances": sum(len(x["alternatives"]) for x in cofit),
        "cofit_steps_with_age_at_least_30_at_commit": sum(bool(x["commit_aged_fit"]) for x in cofit),
        "cofit_steps_still_fit_at_next_snapshot": sum(bool(x["after_fit"]) for x in cofit),
        "cofit_steps_still_fit_at_target_first_output": sum(bool(x["first_output_fit"]) for x in cofit),
        "cofit_steps_with_age_at_least_30_and_fit_at_first_output": sum(bool(x["first_output_aged_fit"]) for x in cofit),
        "cofit_steps_after_waiting_head_is_new_victim": sum(x["after_waiting_head"] == x["victim"] for x in cofit),
        "cofit_steps_after_waiting_head_numerically_unfunded": sum(x["after_waiting_head_need_blocks"] is not None
            and x["after_waiting_head_need_blocks"] > x["after_free_blocks"] for x in cofit),
        "cofit_steps_next_snapshot_running_count_le_30": sum(x["after_running_count"] <= 30 for x in cofit),
        "cofit_steps_target_async_after_commit": sum(x["target_async_after_commit"] for x in cofit),
        "cofit_steps_commit_metadata_load": sum(x["commit_metadata_load"] for x in cofit),
        "cofit_steps_commit_metadata_flush": sum(x["commit_metadata_flush"] for x in cofit),
        "distinct_original_targets": len({x["target"] for x in cofit}),
        "distinct_alternatives": len({a["request"] for x in cofit for a in x["alternatives"]}),
    }
    assert (counts["forced_victim_commits"], counts["numerically_cofit_commit_steps"],
            counts["cofit_steps_still_fit_at_next_snapshot"],
            counts["cofit_steps_still_fit_at_target_first_output"]) == (267, 93, 92, 85)

    example = next(x for x in cofit if x["step"] == 515)
    alt_id = example["alternatives"][0]["request"]
    assert all(snapshots[s]["requests"][alt_id]["status"] == "PREEMPTED" for s in range(516, 527))
    assert snapshots[527]["requests"][alt_id]["status"] == "WAITING_FOR_REMOTE_KVS"
    trace = []
    for step in (515, 516, 518, 527):
        snapshot = snapshots[step]
        trace.append({
            "step": step, "host_perf_counter_s": snapshot["host_perf_counter_s"],
            "free_blocks": snapshot["free_blocks"],
            "running_count": len(snapshot["running_ids"]),
            "waiting_head": snapshot["waiting_ids"][0] if snapshot["waiting_ids"] else None,
            "skipped_count": len(snapshot["skipped_ids"]),
            "victim": state(snapshot, example["victim"]),
            "target": state(snapshot, example["target"]),
            "alternative": state(snapshot, alt_id),
            "target_first_output_event": any(e["event"] == "target_new_output" and e.get("request") == example["target"]
                                             for e in events[step]),
        })

    result = {
        "status": "NUMERICAL_EXISTENCE_OBSERVED_NATIVE_ADMISSION_UNKNOWN",
        "source": str(SOURCE.relative_to(HERE)),
        "source_sha256": digest.hexdigest(),
        "definition": "At a READY forced-victim commit, F + victim held blocks covers full-history incremental blocks of the original target and one other PREEMPTED waiting request. This is a necessary numerical test only.",
        "counts": counts,
        "typical_trace": {
            "original_target": example["target"], "new_victim": example["victim"],
            "other_preempted_request": alt_id,
            "commit_full_history_need_blocks": {"target": need(snapshots[515]["requests"][example["target"]]),
                                                 "other": need(snapshots[515]["requests"][alt_id])},
            "commit_numerical_margin_blocks": (snapshots[515]["free_blocks"] +
                snapshots[515]["requests"][example["victim"]]["held_blocks"] -
                need(snapshots[515]["requests"][example["target"]]) -
                need(snapshots[515]["requests"][alt_id])),
            "snapshots": trace,
            "alternative_first_nonpreempted_step": 527,
            "alternative_first_nonpreempted_status": snapshots[527]["requests"][alt_id]["status"],
            "commit_to_alternative_status_change_s": (snapshots[527]["host_perf_counter_s"] -
                                                      snapshots[515]["host_perf_counter_s"]),
        },
        "limits": [
            "The post-commit FCFS waiting head is the newly preempted victim in every numerical co-fit step; another fit waiter is behind it. In the step-515 trace this head needs 218 blocks while free is 215.",
            "In 92/93 co-fit steps the original target is in skipped waiting during an asynchronous load at the next snapshot and commit metadata reports a load; all 93 report a store flush. The current target latch and queue order prevent treating numerical fit as native admission.",
            "Snapshots do not record num_waiting_for_streaming_input, a direct native reservation R at these commits, complete transfer-job state, connector lookup, or allocation outcome for an additional waiter. Full slot/jobs/R/admission gates remain UNKNOWN.",
            "Repeated steps and alternative instances are correlated observations from one seen-input diagnostic, not independent opportunities or a service-benefit estimate.",
        ],
        "bounded_next_action_proposal_only": (
            "After the Q1 original target's first output, choose at most one additional waiter with >=30 steps absence only if the actual _direct_resume_reason, known R=0, and no-jobs gates pass. Preserve original-target priority; keep Q10 protection and concurrent loading disabled. Native admission and full-cohort service require separate observation."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(OUTPUT)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
