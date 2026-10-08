#!/usr/bin/env python3
"""Describe one same-host QMSum backfill/control pair from complete raw files."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from C_QMSUM_ANALYZE_V1 import load, sha


def cell(directory: Path, mode: str):
    files = {name: directory / name for name in (
        "measured-backfill.json", "measured-service-mix.json", "measured-steps.json",
        "measured-outputs.json", "status.json")}
    docs = {name: load(path) for name, path in files.items()}
    action, mix, steps, outputs, status = (docs[name] for name in files)
    if (status.get("status") != "COMPLETE" or status.get("request_count") != 200
            or action.get("mode") != mode or len(outputs) != 200
            or len(mix.get("first_successful_allocation", [])) != 200
            or len(steps.get("scheduler_calls", [])) != status.get("schedule_calls")):
        raise ValueError(f"{mode}: incomplete or mismatched native cell")
    by_id = {r["external_request_id"]: r for r in outputs}
    order = [r["external_request_id"] for r in mix["first_successful_allocation"]]
    if (len(by_id) != 200 or len(set(order)) != 200 or set(by_id) != set(order)
            or not all(r.get("finished") and r.get("arrival_s") == 0.0
                       and 1 <= len(r["output_token_ids"]) <= 512 for r in outputs)):
        raise ValueError(f"{mode}: request inventory, completion or cap differs")
    frozen = steps["source_indices_in_submission_order"]
    if sorted(frozen) != list(range(200)):
        raise ValueError(f"{mode}: frozen submission order differs")
    return dict(action=action, mix=mix, steps=steps, outputs=by_id,
                order=order, first={r["external_request_id"]: r
                                    for r in mix["first_successful_allocation"]},
                frozen=frozen, hashes={name: sha(path) for name, path in files.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backfill-native", type=Path, required=True)
    ap.add_argument("--control-native", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    back = cell(args.backfill_native, "backfill")
    control = cell(args.control_native, "control")
    if back["frozen"] != control["frozen"]:
        raise ValueError("pair submitted different frozen orders")
    ids = set(back["order"])
    if ids != set(control["order"]):
        raise ValueError("pair request IDs differ")
    for rid in ids:
        a, b = back["outputs"][rid], control["outputs"][rid]
        if (a["request_id"] != b["request_id"] or a["source_index"] != b["source_index"]
                or a["prompt_token_ids"] != b["prompt_token_ids"]):
            raise ValueError("pair request identity or prompt differs: " + rid)
    frozen_position = {source_index: i for i, source_index in enumerate(back["frozen"])}
    position = {mode: {rid: i for i, rid in enumerate(data["order"])}
                for mode, data in (("backfill", back), ("control", control))}
    rows = []
    for rid in control["order"]:
        a, b = back["outputs"][rid], control["outputs"][rid]
        a_done = a["host_elapsed_s"] - a["arrival_s"]
        b_done = b["host_elapsed_s"] - b["arrival_s"]
        rows.append(dict(request_id=rid, source_index=a["source_index"],
            frozen_submission_position=frozen_position[a["source_index"]],
            first_allocation_position_backfill=position["backfill"][rid],
            first_allocation_position_control=position["control"][rid],
            first_allocation_position_delta=position["backfill"][rid]-position["control"][rid],
            first_allocation_call_backfill=back["first"][rid]["schedule_call"],
            first_allocation_call_control=control["first"][rid]["schedule_call"],
            arrival_to_completion_s_backfill=a_done,
            arrival_to_completion_s_control=b_done,
            completion_delta_s=a_done-b_done,
            output_tokens_backfill=len(a["output_token_ids"]),
            output_tokens_control=len(b["output_token_ids"]),
            output_tokens_delta=len(a["output_token_ids"])-len(b["output_token_ids"]),
            output_ids_equal=a["output_token_ids"] == b["output_token_ids"],
            finish_reason_backfill=a["finish_reason"],
            finish_reason_control=b["finish_reason"]))
    by_id = {r["request_id"]: r for r in rows}
    scans = back["action"].get("scans", [])
    restores = back["action"].get("restored", [])
    if control["action"].get("scans") or control["action"].get("restored"):
        raise ValueError("control unexpectedly changed waiting order")
    skipped_by_call = defaultdict(list)
    action_rows = []
    for scan in scans:
        call = scan["call"]
        skipped = scan["skipped_external_request_ids"]
        accepted = scan["accepted_external_request_id"]
        if (not 0 <= call < len(back["steps"]["scheduler_calls"])
                or not skipped or not set(skipped) <= ids
                or accepted is not None and accepted not in ids):
            raise ValueError("invalid scan inventory")
        skipped_by_call[call].extend(skipped)
        if accepted is not None and back["first"][accepted]["schedule_call"] != call:
            raise ValueError("accepted scan request has no first allocation in that call")
        action_rows.append(dict(schedule_call=call,
            schedule_host_return_s=back["steps"]["scheduler_calls"][call]["host_s"],
            skipped_heads=[dict(request_id=rid, frozen_submission_position=by_id[rid]["frozen_submission_position"],
                                completion_delta_s=by_id[rid]["completion_delta_s"],
                                output_tokens_delta=by_id[rid]["output_tokens_delta"]) for rid in skipped],
            candidate_allocation_attempts=scan["candidate_allocation_attempts"],
            accepted_after_skip=None if accepted is None else dict(by_id[accepted]),
            interval_host_s=scan["scan_host_s"],
            interval_process_cpu_s=scan["scan_process_cpu_s"]))
    restored_by_call = {r["call"]: r["external_request_ids"] for r in restores}
    if (len(restored_by_call) != len(restores) or set(restored_by_call) != set(skipped_by_call)
            or any(restored_by_call[c] != skipped_by_call[c] for c in skipped_by_call)):
        raise ValueError("skipped-head restoration order differs")
    restored_rows = [dict(schedule_call=c, restored_heads=[dict(
        restore_position_in_call=i, **by_id[rid]) for i, rid in enumerate(restored_by_call[c])])
        for c in sorted(restored_by_call)]
    if (sum(len(x["skipped_external_request_ids"]) for x in scans)
            != back["action"]["skipped_request_count"]
            or sum(x["accepted_external_request_id"] is not None for x in scans)
               != back["action"]["accepted_after_skip_count"]):
        raise ValueError("action aggregate counters differ")
    movement = Counter("earlier" if r["first_allocation_position_delta"] < 0 else
                       "later" if r["first_allocation_position_delta"] > 0 else "same"
                       for r in rows)
    result = dict(schema="c-qmsum-backfill-actions-v1",
        scope="One same-host backfill then control development pair; descriptive request movements, no causal per-request attribution",
        raw_sha256={"backfill": back["hashes"], "control": control["hashes"]},
        summary=dict(requests=200, scans=len(scans), skipped_head_events=back["action"]["skipped_request_count"],
            successful_after_skip=back["action"]["accepted_after_skip_count"],
            scan_intervals_host_s=back["action"]["scan_host_s"],
            scan_intervals_process_cpu_s=back["action"]["scan_process_cpu_s"],
            first_allocation_position_movement=dict(movement),
            output_ids_changed=sum(not r["output_ids_equal"] for r in rows)),
        first_allocation_order={"backfill": back["order"], "control": control["order"]},
        per_request=rows, scans=action_rows, restored_heads=restored_rows,
        interpretation_limits=[
            "Scan interval timers omit scheduling work outside each open scan and are not total policy CPU cost.",
            "Failed candidate allocation attempts are not successful admissions or performance benefit.",
            "Schedule host return is an observer timestamp, not a GPU start time.",
            "Per-request completion changes are descriptive; output lengths may differ across arms."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
