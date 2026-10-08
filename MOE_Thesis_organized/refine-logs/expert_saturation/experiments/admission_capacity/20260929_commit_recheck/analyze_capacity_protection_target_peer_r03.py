#!/usr/bin/env python3
"""Describe complete-request Q1/Q10 differences by actual Q10 extension exposure."""

import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "A_CAPACITY_PROTECTION_PAIR_RESULT_R03_20261001.json"
OUTPUT = HERE / "A_CAPACITY_PROTECTION_TARGET_PEER_R03_20261001.json"
INTERNAL_ID = re.compile(r"^measured/(memory-train-article-\d+)-[0-9a-f]+$")


def paired_summary(rows):
    answer = {
        "requests": len(rows),
        "q10_extended_episodes": sum(r["q10_extended_episodes"] for r in rows),
        "q10_extended_planned_victim_episodes": sum(
            r["q10_extended_planned_victim_episodes"] for r in rows
        ),
        "output_sequence_changed_requests": sum(r["output_sequence_changed"] for r in rows),
        "stop_reason_changed_requests": sum(r["stop_reason_changed"] for r in rows),
        "output_count_changed_requests": sum(r["delta_q10_minus_q1"]["outputs"] != 0 for r in rows),
        "q1_output_tokens": sum(r["q1"]["outputs"] for r in rows),
        "q10_output_tokens": sum(r["q10"]["outputs"] for r in rows),
    }
    for metric in ("flow_s", "max_gap_s", "ttft_s"):
        values = [r["delta_q10_minus_q1"][metric] for r in rows]
        answer[metric] = {
            "mean_delta_s": statistics.mean(values),
            "median_delta_s": statistics.median(values),
            "sum_delta_s": sum(values),
            "q10_better_requests": sum(x < -1e-12 for x in values),
            "q10_worse_requests": sum(x > 1e-12 for x in values),
            "tied_requests": sum(abs(x) <= 1e-12 for x in values),
        }
    return answer


def main():
    raw = SOURCE.read_bytes()
    data = json.loads(raw)
    assert data["status"] == "COMPLETE_PAIR"
    arms = {}
    for arm in ("q1", "q10"):
        metrics = data["arms"][arm]["metrics"]
        assert metrics["completed"] == metrics["expected_requests"] == 128
        requests = metrics["requests"]
        assert len(requests) == 128
        arms[arm] = {r["request_id"]: r for r in requests}
        assert len(arms[arm]) == 128
    assert set(arms["q1"]) == set(arms["q10"])

    extended_counts = Counter()
    extended_planned_victim_counts = Counter()
    for episode in data["arms"]["q10"]["episodes"]:
        if episode["extend"] is None:
            continue
        extended_counts[episode["request_id"]] += 1
        victim = episode["admission"]["planned_victim"]
        if victim is not None:
            match = INTERNAL_ID.fullmatch(victim)
            assert match, victim
            extended_planned_victim_counts[match.group(1)] += 1
    assert sum(extended_counts.values()) == 206

    sequence_changed = set(data["output_sequence_difference_requests"])
    stop_changed = set(data["stop_reason_difference_requests"])
    assert sequence_changed <= set(arms["q1"])
    assert stop_changed <= set(arms["q1"])
    rows = []
    for request_id in sorted(arms["q1"]):
        first, second = arms["q1"][request_id], arms["q10"][request_id]
        assert first["prompt_sha256"] == second["prompt_sha256"]
        assert first["arrival_s"] == second["arrival_s"]
        assert first["completed"] and second["completed"]
        fields = ("flow_s", "max_gap_s", "ttft_s", "outputs", "stop_reason")
        row = {
            "request_id": request_id,
            "q10_actual_extended_target": extended_counts[request_id] > 0,
            "q10_extended_episodes": extended_counts[request_id],
            "q10_extended_planned_victim_episodes": extended_planned_victim_counts[request_id],
            "q1": {field: first[field] for field in fields},
            "q10": {field: second[field] for field in fields},
            "delta_q10_minus_q1": {
                field: second[field] - first[field]
                for field in ("flow_s", "max_gap_s", "ttft_s", "outputs")
            },
            "output_sequence_changed": request_id in sequence_changed,
            "stop_reason_changed": request_id in stop_changed,
        }
        assert row["stop_reason_changed"] == (first["stop_reason"] != second["stop_reason"])
        rows.append(row)
    assert sum(r["output_sequence_changed"] for r in rows) == len(sequence_changed)

    groups = {
        "all_128": paired_summary(rows),
        "actual_q10_extended_targets": paired_summary(
            [r for r in rows if r["q10_actual_extended_target"]]
        ),
        "other_requests_without_q10_extension": paired_summary(
            [r for r in rows if not r["q10_actual_extended_target"]]
        ),
    }
    assert groups["actual_q10_extended_targets"]["requests"] == 95
    assert groups["other_requests_without_q10_extension"]["requests"] == 33
    assert groups["all_128"]["q1_output_tokens"] == data["arms"]["q1"]["metrics"]["total_output_tokens"]
    assert groups["all_128"]["q10_output_tokens"] == data["arms"]["q10"]["metrics"]["total_output_tokens"]

    worst = {}
    for metric in ("flow_s", "max_gap_s", "ttft_s"):
        worst[metric] = sorted(
            rows, key=lambda r: (-r["delta_q10_minus_q1"][metric], r["request_id"])
        )[:8]

    result = {
        "status": "DESCRIPTIVE_POST_TREATMENT_SPLIT",
        "source": SOURCE.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "group_definition": "Targets had at least one actual Q10 protection_extend event; all others are peer/nonextended requests. Groups are defined after treatment and are not causal comparisons.",
        "scope": [
            "All 128 requests completed in both independently executed cells; each request has equal weight in flow, gap, and TTFT differences.",
            "Q10-minus-Q1 paired differences compare two natural trajectories; output sequences and stop reasons can differ.",
            "Planned-victim counts record labels in Q10 extended admissions, not proof of a subsequent native preemption or a quantified peer cost.",
            "Extension windows are disjoint in the separate exposure reconstruction; outputs of other requests during them do not imply blocked opportunities, exclusive GPU time, or causal cost.",
        ],
        "q10_extended_admission_planned_victims": {
            "distinct_request_ids": len(extended_planned_victim_counts),
            "also_extended_targets": len(set(extended_planned_victim_counts) & set(extended_counts)),
            "never_extended": len(set(extended_planned_victim_counts) - set(extended_counts)),
            "extended_targets_never_planned_victim": len(set(extended_counts) - set(extended_planned_victim_counts)),
        },
        "groups": groups,
        "worst_q10_losses": worst,
        "per_request": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(OUTPUT)
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
