#!/usr/bin/env python3
"""Describe completed LongBench150 native host records and allocation returns."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import C_LONG_DOCUMENT_QA_RUNTIME_V1 as base

FAILURE_FIELDS = ("schedule_call", "host_s", "request_id", "request_status",
    "num_computed_tokens", "current_num_tokens", "prompt_tokens", "num_new_tokens",
    "num_new_computed_tokens", "full_sequence_must_fit", "free_blocks_before", "free_blocks_after")
INPUT_FIELDS = ("request_id", "task", "source_index", "example_index", "source_id",
    "source_line_sha256", "context_sha256", "question", "answers", "raw_prompt",
    "chat_user_content", "prompt", "prompt_token_ids", "prompt_token_ids_sha256",
    "truncation", "answer_surface_diagnostic")


def read(path: Path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def compare_first16(full_rows: list, full_quality: dict, old_run: Path) -> dict:
    old_rows, old_sha = read(old_run / "measured-outputs.json")
    old_quality, old_quality_sha = read(
        Path(__file__).with_name("long_document_qa_qualification_v1.json"))
    if (len(old_rows) != 16 or old_quality.get("status") != "COMPLETE"
            or old_quality.get("requests_completed") != 16
            or old_quality.get("original_run_sha256", {}).get("measured-outputs.json") != old_sha):
        raise ValueError("first-16 original outputs/quality differ")
    old = {r["source_index"]: r for r in old_rows}
    old_f1 = {r["source_index"]: r for r in old_quality["per_request"]}
    new_f1 = {r["source_index"]: r for r in full_quality["per_request"]}
    first = sorted((r for r in full_rows if r["source_index"] < 16),
                   key=lambda r: r["source_index"])
    if not (set(old) == set(old_f1) == set(range(16))
            and [r["source_index"] for r in first] == list(range(16))):
        raise ValueError("first-16 source inventory differs")
    rows = []
    for new in first:
        i = new["source_index"]
        before, a, b = old[i], old_f1[i], new_f1[i]
        if (any(new.get(key) != before.get(key) for key in INPUT_FIELDS)
                or len(before["output_token_ids"]) != a["output_tokens"]
                or len(new["output_token_ids"]) != b["output_tokens"]
                or before["finish_reason"] != a["finish_reason"]
                or new["finish_reason"] != b["finish_reason"]):
            raise ValueError(f"first-16 frozen input/output receipt differs at {i}")
        rows.append(dict(source_index=i, request_id=new["request_id"], input_equal=True,
            first16_output_token_ids=before["output_token_ids"],
            full150_output_token_ids=new["output_token_ids"],
            output_token_ids_equal=before["output_token_ids"] == new["output_token_ids"],
            output_text_equal=before["output_text"] == new["output_text"],
            first16_output_tokens=len(before["output_token_ids"]),
            full150_output_tokens=len(new["output_token_ids"]),
            first16_finish_reason=before["finish_reason"],
            full150_finish_reason=new["finish_reason"],
            first16_f1=a["f1"], full150_f1=b["f1"],
            f1_delta_full_minus_first16=b["f1"] - a["f1"]))
    return dict(scope="Same frozen inputs; output differences only, no causal comparison",
        identical_inputs=16,
        identical_output_token_id_sequences=sum(r["output_token_ids_equal"] for r in rows),
        differing_output_lengths=sum(r["first16_output_tokens"] != r["full150_output_tokens"] for r in rows),
        differing_finish_reasons=sum(r["first16_finish_reason"] != r["full150_finish_reason"] for r in rows),
        differing_f1=sum(r["f1_delta_full_minus_first16"] != 0 for r in rows),
        first16_f1_mean=sum(r["first16_f1"] for r in rows) / 16,
        full150_first16_f1_mean=sum(r["full150_f1"] for r in rows) / 16,
        per_request=rows, first16_outputs_sha256=old_sha,
        first16_quality_sha256=old_quality_sha)


def analyze(run: Path, quality_path: Path, first16_run: Path | None) -> dict:
    old_n = base.N
    try:
        base.N = 150
        report = base.analyze(run, quality_path)
    finally:
        base.N = old_n
    trace, _ = read(run / "measured-steps.json")
    status, _ = read(run / "status.json")
    outputs, _ = read(run / "measured-outputs.json")
    quality, _ = read(quality_path)
    calls, failures = trace["scheduler_calls"], trace["allocation_failures"]
    allocation_calls = trace["allocation_calls"]
    if (type(allocation_calls) is not int or allocation_calls < len(failures)
            or allocation_calls != status["allocation_calls"]
            or len(failures) != status["allocation_failure_count"]):
        raise ValueError("allocation-call/failure receipt differs")
    external_ids = {r["external_request_id"] for r in outputs}
    for event in failures:
        if (not set(FAILURE_FIELDS) <= set(event)
                or type(event["schedule_call"]) is not int
                or not 0 <= event["schedule_call"] < len(calls)
                or event["request_id"] not in external_ids
                or not 0 <= event["host_s"] <= status["observation_end_s"]
                or not 0 <= event["free_blocks_before"] <= 4096
                or not 0 <= event["free_blocks_after"] <= 4096):
            raise ValueError("allocation failure event differs")
    failed_calls = {e["schedule_call"] for e in failures}
    preempt_calls = {c["call"] for c in calls if c["preempted_request_ids"]}
    report.update(schema="c-longbench-multifieldqa-en-full-native-runtime-v1",
        scope="Post-observation source-order 150 native runtime; no capacity-policy comparison",
        allocation_calls=allocation_calls, allocation_failures_returning_none=len(failures),
        allocation_failure_request_status_counts=dict(Counter(e["request_status"] for e in failures)),
        allocation_failure_schedule_calls=len(failed_calls), allocation_failures=failures,
        calls_with_waiting_after_schedule=sum(c["after"]["waiting"] > 0 for c in calls),
        calls_with_waiting_after_and_full_token_budget=sum(
            c["after"]["waiting"] > 0 and c["scheduled_tokens_total"] ==
            report["resolved_scheduler"]["max_num_batched_tokens"] for c in calls),
        calls_with_waiting_after_without_logged_failure_or_preemption=sum(
            c["after"]["waiting"] > 0 and c["call"] not in failed_calls | preempt_calls for c in calls),
        calls_with_preemption=len(preempt_calls),
        base_runtime_source_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest())
    report["limitations"].append(
        "allocate_slots returning None, preemption, and waiting snapshots are distinct, possibly overlapping observations; unrecorded causes are not inferred.")
    if first16_run is not None:
        report["first16_output_comparison"] = compare_first16(outputs, quality, first16_run)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--quality", type=Path, required=True)
    p.add_argument("--first16-run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = analyze(a.run_dir, a.quality, a.first16_run)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "requests_completed",
        "allocation_calls", "allocation_failures_returning_none", "preemption_events")}))


if __name__ == "__main__":
    main()
