#!/usr/bin/env python3
"""Bounded actor and gap localization for the uninstrumented lazy/native pair."""

from collections import Counter
import json
from pathlib import Path
import statistics

import analyze_native_backfill_only_target_peer_r01 as common


HERE = Path(__file__).resolve().parent
MAIN = HERE / "A_NATIVE_BACKFILL_LAZY_PERF_PAIR_RESULT_R01_20261002.json"
OUTPUT = HERE / "A_NATIVE_BACKFILL_LAZY_PERF_TARGET_PEER_R01_20261002.json"


def main():
    if OUTPUT.exists():
        raise SystemExit("Preserve existing lazy performance target-peer result")
    canonical = json.loads(MAIN.read_text())
    if canonical["status"] != "COMPLETE_PAIR":
        raise SystemExit("Canonical pair incomplete")
    source = {"canonical": str(MAIN), "canonical_sha256": common.sha(MAIN), "arms": {}}
    raw, store, metric = {}, {}, {}
    for arm in ("ordinary", "native_full"):
        archive = HERE / canonical["arms"][arm]["archive"]
        raw_path, store_path = archive / "raw.json", archive / "selective-store.json"
        raw[arm], store[arm] = json.loads(raw_path.read_text()), json.loads(store_path.read_text())
        metric[arm] = {row["request_id"]: row
                       for row in canonical["arms"][arm]["metrics"]["requests"]}
        source["arms"][arm] = {
            "raw": str(raw_path), "raw_sha256": common.sha(raw_path),
            "selective_store": str(store_path), "selective_store_sha256": common.sha(store_path),
        }
    if len(metric["ordinary"]) != 128 or set(metric["ordinary"]) != set(metric["native_full"]):
        raise SystemExit("Cohort identities incomplete or mismatched")
    actions = canonical["ordinary_actions"]["actions"]
    rows = common.action_rows(actions, raw["ordinary"], metric["ordinary"], metric["native_full"])
    actors = common.actor_groups(actions, raw["ordinary"], metric["ordinary"], metric["native_full"])
    gaps = {arm: common.worst_gaps(raw[arm], store[arm], actions if arm == "ordinary" else [])
            for arm in raw}
    role_ids = {role: {row["request_id"] for row in group["requests"]}
                for role, group in actors.items()}
    comparisons = []
    for rid in sorted(metric["ordinary"]):
        ordinary, native = metric["ordinary"][rid], metric["native_full"][rid]
        comparisons.append({
            "request_id": rid,
            "ordinary": {k: ordinary[k] for k in ("max_gap_s", "flow_s", "ttft_s", "outputs", "stop_reason")},
            "native_full": {k: native[k] for k in ("max_gap_s", "flow_s", "ttft_s", "outputs", "stop_reason")},
            "max_gap_delta_s": ordinary["max_gap_s"] - native["max_gap_s"],
            "flow_delta_s": ordinary["flow_s"] - native["flow_s"],
            "ttft_delta_s": ordinary["ttft_s"] - native["ttft_s"],
            "ordinary_action_roles": [role for role, ids in role_ids.items() if rid in ids],
        })
    worse_gap = sorted((row for row in comparisons if row["max_gap_delta_s"] > 0),
                       key=lambda row: row["max_gap_delta_s"], reverse=True)
    worse_flow = sorted((row for row in comparisons if row["flow_delta_s"] > 0),
                        key=lambda row: row["flow_delta_s"], reverse=True)
    report = {
        "status": "DESCRIPTIVE_SOURCE_LOCALIZATION_COMPLETE",
        "sources": source,
        "scope": "One ordered ordinary-lazy then native-full uninstrumented pair; all 128 fixed requests included; separate trajectories, no action counterfactual.",
        "complete_cohort": {arm: {
            "completed": canonical["arms"][arm]["metrics"]["completed"],
            "duration_s": canonical["arms"][arm]["metrics"]["duration_s"],
            "actual_output_tokens_s": canonical["arms"][arm]["metrics"]["actual_output_tokens_s"],
            "mean_flow_s": canonical["arms"][arm]["metrics"]["mean_flow_with_incomplete_penalty_s"],
            "max_gap_s": canonical["arms"][arm]["metrics"]["max_gap_request_max_s"],
            "actual_preemptions": canonical["arms"][arm]["actual_preemption_count"],
            "forced_rotations": canonical["arms"][arm]["forced_rotations"],
        } for arm in raw},
        "ordinary_actions": {
            "count": len(rows),
            "native_admissions": dict(Counter(row["native_admission"] for row in rows)),
            "actual_output_completion_chains": canonical["ordinary_actions"]["actual_output_completion_chains"],
            "distinct_targets": actors["target"]["distinct_requests"],
            "actions_with_target_later_preemption": sum(bool(row["target_preemptions_after_first_new_output"]) for row in rows),
            "distinct_targets_later_preempted": len({row["target"] for row in rows
                if row["target_preemptions_after_first_new_output"]}),
            "admission_to_output_median_s": statistics.median(row["admission_to_target_output_s"] for row in rows),
            "admission_to_output_max_s": max(row["admission_to_target_output_s"] for row in rows),
            "actions": rows, "actors": actors,
        },
        "request_regressions": {
            "worse_max_gap_count": len(worse_gap), "worse_max_gap_requests": worse_gap,
            "worse_flow_count": len(worse_flow), "worse_flow_requests": worse_flow,
        },
        "worst_gaps": gaps,
        "limits": [
            "Each arm has different natural output, preemption, and admission trajectories; request differences cannot be assigned to one action.",
            "Sparse ordinary receipts identify chosen backfill admissions only. Unchosen native waiting requests have no recorded native admission time, so a preemption-to-next-output interval is not wholly queue waiting.",
            "The ordinary longest gap can overlap bypass choices without proving those choices caused its duration.",
        ],
    }
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": report["status"], "actions": len(rows),
        "target_later_preempted_distinct": report["ordinary_actions"]["distinct_targets_later_preempted"],
        "worse_gap": len(worse_gap), "worse_flow": len(worse_flow),
        "ordinary_worst_gap": gaps["ordinary"][0],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
