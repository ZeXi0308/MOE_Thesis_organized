#!/usr/bin/env python3
"""Enumerate later observed opportunities missed by the v9 first-target probe.

python reservation_followup_diagnostic.py --output analysis/NEW.json
Only reads the two frozen v9 groups; no policy simulation or threshold tuning.
"""
import argparse
import hashlib
import json
from pathlib import Path

from opportunity_summary import target_preemptions


GROUPS = ("reservation-probe-r01", "reservation-probe-r02")
ARMS = ("probe-00-baseline", "probe-01-reservation", "probe-02-reservation", "probe-03-baseline")


def read_hashed(path):
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def summarize(path):
    admission, admission_sha = read_hashed(path/"admission.json")
    raw, raw_sha = read_hashed(path/"raw.json")
    probe = admission["reservation_probe"]
    assert probe.get("signal_kind", "known_prefill") == "known_prefill"
    event, release = probe["first_opportunity"], probe["first_release"]
    assert event is not None and probe["max_extra_s"] == .25
    origin = raw["measurement_origin_perf_counter_s"]
    offset = admission["origin_perf_s"]-origin
    identity = {request[key]: request["request_id"] for request in raw["requests"]
                for key in ("request_id", "internal_request_id", "external_request_id")}
    requests = {request["request_id"]: request for request in raw["requests"]}
    starts = {row["request_id"]: row["first_prefill_perf_s"]-origin for row in admission["starts"]}
    canonical = lambda rid: identity[rid]
    full = [(i, row) for i, row in enumerate(admission["decisions"])
            if row.get("full_commit_opportunity") is True]
    assert len(full) == admission["reservation_observation"]["full_commit_opportunity_evaluations"]
    successful = []
    evaluations = []
    for index, row in full:
        rid = canonical(row["request_id"])
        same_target = row["request_id"] == event["target_request_id"]
        phase = ("baseline_no_action_release" if not probe["probe_enabled"] else
                 "after_release" if release and row["reservation_observed_t"] >= release["t"] else
                 "before_release")
        actual_local = (row.get("native_fit") is True and row.get("base_allowed") is True and
            row.get("native_allocation_result") is True and row.get("num_new_tokens", 0) > 0 and
            row.get("load_kv_async") is False)
        evaluations.append(dict(decision_index=index, request_id=rid, t=row["t"],
            phase=phase, is_first_target=same_target, reason=row["reason"],
            actual_successful_local_allocation=actual_local))
        if not actual_local:
            continue
        assert not row["denied"] and row["reason"] == "allow"
        prefill = starts[row["request_id"]]
        assert prefill >= offset+row["t"]
        request = requests[rid]
        token = request["token_times_s"][0] if request["token_times_s"] else None
        preemptions = target_preemptions(raw, rid, prefill, token, canonical)
        before = [r for r in preemptions["events"] if r["original_preemption_returned"] and
                  r["after_reservation_event"] and r["native_output_zero_before"] and
                  r["before_first_host_token"] is True]
        successful.append(dict(decision_index=index, request_id=rid,
            native_request_id=row["request_id"], is_first_target=same_target, phase=phase,
            decision_admission_s=row["t"], opportunity_admission_s=row["reservation_observed_t"],
            opportunity_external_s=offset+row["reservation_observed_t"],
            free_blocks=row["free_blocks"], full_required_blocks=row["full_required_blocks"],
            known_prefill_unallocated_blocks=row["known_prefill_unallocated_blocks"],
            full_plus_existing_unallocated_blocks=row["full_plus_existing_unallocated_blocks"],
            chunk_plus_existing_unallocated_blocks=row["chunk_plus_existing_unallocated_blocks"],
            chunk_commit_opportunity=row["chunk_commit_opportunity"],
            first_prefill_schedule_return_external_s=prefill, first_token_host_external_s=token,
            completion_external_s=request["completion_s"], output_tokens=len(request["output_token_ids"]),
            actual_preemptions_whole_lifecycle=preemptions["actual_preemptions"],
            actual_preemptions_after_prefill_before_first_output=len(before) if token is not None else None,
            preoutput_preemptions=[{key:r[key] for key in ("method_entered_s", "method_returned_s",
                "native_output_count_before", "original_preemption_called", "original_preemption_returned")}
                for r in before]))
    return dict(cell=str(path), admission_sha256=admission_sha, raw_sha256=raw_sha,
        probe_enabled=probe["probe_enabled"], first_target=canonical(event["target_request_id"]),
        first_opportunity_admission_s=event["t"],
        first_release=(dict(release, external_s=offset+release["t"]) if release else None),
        full_opportunity_evaluations=len(full),
        full_opportunity_unique_requests=len({canonical(row["request_id"]) for _, row in full}),
        full_opportunity_rows=evaluations,
        successful_local_opportunities=len(successful),
        successful_local_unique_requests=len({row["request_id"] for row in successful}),
        after_release_nonfirst_target_successes=sum(row["phase"] == "after_release" and
            not row["is_first_target"] for row in successful),
        baseline_nonfirst_target_successes=sum(not probe["probe_enabled"] and
            not row["is_first_target"] for row in successful),
        successful_local_rows=successful)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/westb-20261008"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(schema_version=1,
        question="Did the single first-target action cover all later distinct legal full-commit opportunities?",
        semantics="Separate observed trajectories, not policy replay or a service-benefit claim. "
            "Distinct requests within an arm are distinct opportunities, not independent experiments. "
            "Candidate successful opportunities are split by actual first_release; baseline release is absent. "
            "The success filter requires full_commit_opportunity, native_fit, base_allowed, actual allocation "
            "true, num_new_tokens>0 and local load. Exact-ID starts must follow that allocation decision. "
            "FIFO/unvisited suffixes have no assumed fit. Full commitment excess is not an immediate "
            "chunk deficit. Pre-output preemption requires successful original native preempt return, "
            "entry after first-prefill schedule return and before first host token, and native output0. "
            "All these boundaries are host observations, not GPU completion. No future output is used "
            "to construct an opportunity; outputs are used only to describe subsequent outcomes.",
        clock="Decision/admission times use admission.origin_perf_s; external times use "
            "raw.measurement_origin_perf_counter_s. Preemption event *_s uses the external clock.",
        groups={group:{arm:summarize(args.runs_root/group/arm) for arm in ARMS} for group in GROUPS})
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    for group, arms in result["groups"].items():
        for arm, data in arms.items():
            print(group, arm, "full", data["full_opportunity_evaluations"],
                  "successful", data["successful_local_opportunities"],
                  "later_candidate", data["after_release_nonfirst_target_successes"])


if __name__ == "__main__":
    main()
