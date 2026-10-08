#!/usr/bin/env python3
"""Score one completed MultiFieldQA-en native cell using pinned LongBench English F1.

Every source-first request remains in the denominator. This is a 16-row local
OLMoE/LongBench adaptation, not an official full-dataset LongBench score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from C_LONG_DOCUMENT_QA_METRIC_V1 import multi_reference_f1, normalize_answer


ROWS = 16
OUTPUT_CAP = 64
TASK = "multifieldqa_en"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def optional(path: Path):
    return load(path) if path.is_file() else None


def validated_inputs(input_dir: Path) -> tuple[dict, dict, dict]:
    config_path = input_dir / "config.json"
    workload_path = input_dir / "workload.json"
    geometry_path = input_dir / "geometry.json"
    config, workload = load(config_path), load(workload_path)
    receipt = load(input_dir / "SOURCE_RECEIPT.json")
    if (config.get("schema") != "c-longbench-multifieldqa-en-config-v1"
            or workload.get("schema") != "c-longbench-multifieldqa-en-workload-v1"
            or receipt.get("schema") != "c-longbench-multifieldqa-en-source-receipt-v1"
            or config.get("workload_sha256") != sha(workload_path)
            or receipt.get("workload_sha256") != sha(workload_path)
            or receipt.get("config_sha256") != sha(config_path)
            or receipt.get("geometry_sha256") != sha(geometry_path)):
        raise ValueError("frozen LongBench input schema or SHA-256 differs")
    requests = workload.get("requests")
    if (not isinstance(requests, list) or len(requests) != ROWS
            or config.get("requests") != ROWS
            or not (config.get("task") == workload.get("task") == TASK)
            or config.get("output_tokens") != OUTPUT_CAP
            or config.get("max_model_len") != 4096
            or workload.get("arrival_traces_s") != [0.0] * ROWS
            or [row.get("source_index") for row in requests] != list(range(ROWS))
            or [row.get("example_index") for row in requests] != list(range(ROWS))
            or len({row.get("request_id") for row in requests}) != ROWS
            or workload.get("sampling") != dict(temperature=0.0, max_tokens=OUTPUT_CAP,
                                                min_tokens=0, ignore_eos=False, stop=[])):
        raise ValueError("source-order request, arrival, context, or sampling contract differs")
    for row in requests:
        ids, answers = row.get("prompt_token_ids"), row.get("answers")
        if (not isinstance(ids, list) or not ids
                or any(type(token) is not int or token < 0 for token in ids)
                or len(ids) + OUTPUT_CAP > 4096
                or not isinstance(answers, list) or not answers
                or any(not isinstance(answer, str) for answer in answers)
                or not isinstance(row.get("truncation"), dict)
                or not isinstance(row.get("answer_surface_diagnostic"), list)):
            raise ValueError("frozen prompt, references, or input diagnostics differ")
    return config, workload, receipt


def output_issues(row: dict, source: dict) -> list[str]:
    issues = []
    for field in source:
        if row.get(field) != source.get(field):
            issues.append("output_" + field + "_differs")
    ids, times = row.get("output_token_ids"), row.get("token_times_s")
    if (not isinstance(ids, list) or not isinstance(times, list)
            or len(ids) != len(times)
            or any(type(token) is not int or token < 0 for token in ids)
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0
                   for t in times)
            or any(a > b for a, b in zip(times, times[1:]))):
        issues.append("invalid_output_tokens_or_host_times")
    if isinstance(ids, list) and len(ids) > OUTPUT_CAP:
        issues.append("output_exceeds_64")
    reason = row.get("finish_reason")
    if row.get("finished") is True:
        if reason not in ("stop", "length"):
            issues.append("unexpected_finish_reason")
        if reason == "length" and isinstance(ids, list) and len(ids) != OUTPUT_CAP:
            issues.append("length_finish_without_64_tokens")
    elif reason is not None:
        issues.append("unfinished_with_finish_reason")
    if reason == "stop" and row.get("stop_reason") is not None:
        issues.append("stop_reason_not_eos_only")
    if not isinstance(row.get("output_text"), str):
        issues.append("output_text_missing")
    return issues


def analyze(input_dir: Path, run_dir: Path, metadata_dir: Path) -> dict:
    from tokenizers import Tokenizer

    config, workload, receipt = validated_inputs(input_dir)
    tokenizer_path = metadata_dir / "tokenizer.json"
    if (not tokenizer_path.is_file()
            or config.get("metadata_sha256", {}).get("tokenizer.json") != sha(tokenizer_path)
            or receipt.get("metadata_sha256", {}).get("tokenizer.json") != sha(tokenizer_path)):
        raise ValueError("analysis tokenizer differs from frozen input tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    names = ("status.json", "measured-outputs.json", "measured-steps.json",
             "input-tokenizer-check.json", "resolved-eos.json", "native-drain.json",
             "timing.json", "environment.json")
    run_sha = {name: sha(run_dir / name) if (run_dir / name).is_file() else None
               for name in names}
    status, outputs = optional(run_dir / "status.json"), optional(run_dir / "measured-outputs.json")
    tokenizer_check = optional(run_dir / "input-tokenizer-check.json")
    eos_receipt = optional(run_dir / "resolved-eos.json")
    drain, environment = optional(run_dir / "native-drain.json"), optional(run_dir / "environment.json")
    eos_ids = eos_receipt.get("qualified_eos_token_ids") if isinstance(eos_receipt, dict) else None
    eos_qualified = (isinstance(eos_receipt, dict)
                     and eos_receipt.get("qualification_status") == "QUALIFIED"
                     and isinstance(eos_ids, list) and bool(eos_ids)
                     and all(type(token) is int and token >= 0 for token in eos_ids))
    issues = []
    if status is None or status.get("status") != "COMPLETE":
        issues.append("cell_status_not_complete")
    if tokenizer_check is None or tokenizer_check.get("status") != "QUALIFIED":
        issues.append("runtime_tokenizer_check_not_qualified")
    if not eos_qualified:
        issues.append("runtime_eos_not_qualified")
    if drain is None or drain.get("status") != "QUALIFIED":
        issues.append("native_drain_not_qualified")
    input_sha = {name: sha(input_dir / name) for name in
                 ("config.json", "workload.json", "geometry.json", "SOURCE_RECEIPT.json")}
    if (environment is None
            or any(environment.get("input_sha256", {}).get(name) != input_sha[name]
                   for name in ("config.json", "workload.json", "SOURCE_RECEIPT.json"))):
        issues.append("environment_input_identity_not_verified")
    if outputs is None:
        outputs = []
        issues.append("measured_outputs_missing")
    if not isinstance(outputs, list):
        outputs = []
        issues.append("measured_outputs_not_list")
    if len(outputs) != ROWS:
        issues.append("measured_request_inventory_differs")
    by_id = {row.get("request_id"): row for row in outputs if isinstance(row, dict)}
    if len(by_id) != len(outputs):
        issues.append("duplicate_or_invalid_measured_request_id")

    per_request = []
    for source in workload["requests"]:
        rid = source["request_id"]
        row = by_id.get(rid)
        common = dict(request_id=rid, source_index=source["source_index"],
                      source_id=source["source_id"], question=source["question"],
                      answers=source["answers"],
                      truncation_applied=source["truncation"]["applied"],
                      answer_surface_diagnostic=source["answer_surface_diagnostic"])
        if row is None:
            per_request.append(dict(common, status="MISSING", f1=0.0,
                normalized_exact_match=False, output_tokens=0, output_text=None,
                finish_reason=None, natural_eos=False, issues=["original_request_output_missing"]))
            continue
        row_issues = output_issues(row, source)
        ids = row.get("output_token_ids")
        ids = ids if isinstance(ids, list) and all(type(token) is int and token >= 0 for token in ids) else []
        text = row.get("output_text") if isinstance(row.get("output_text"), str) else ""
        decoded_match = tokenizer.decode(ids, skip_special_tokens=True) == text
        if not decoded_match:
            row_issues.append("decoded_output_text_differs")
        completed = row.get("finished") is True and row.get("finish_reason") in ("stop", "length")
        valid_complete = completed and not row_issues
        normalized = normalize_answer(text)
        per_request.append(dict(common,
            status="COMPLETE" if valid_complete else "INCOMPLETE",
            f1=multi_reference_f1(text, source["answers"]) if valid_complete else 0.0,
            normalized_exact_match=(bool(normalized) and any(
                normalized == normalize_answer(answer) for answer in source["answers"]))
                if valid_complete else False,
            output_tokens=len(ids), output_text=text,
            empty_output=not bool(text.strip()),
            finish_reason=row.get("finish_reason"), stop_reason=row.get("stop_reason"),
            natural_eos=(valid_complete and eos_qualified
                         and row.get("finish_reason") == "stop" and row.get("stop_reason") is None),
            decoded_token_text_matches_native_text=decoded_match,
            host_completion_s=row.get("host_elapsed_s"), issues=row_issues))
        issues.extend(rid + ":" + issue for issue in row_issues)
    if set(by_id) != {source["request_id"] for source in workload["requests"]}:
        issues.append("extra_or_missing_measured_request_id")
    completed = [row for row in per_request if row["status"] == "COMPLETE"]
    return dict(
        schema="c-longbench-multifieldqa-en-native-analysis-v1",
        status="COMPLETE" if not issues and len(completed) == ROWS else "INCOMPLETE",
        issues=issues, task=TASK, model=config["model"],
        requests_planned=ROWS, requests_with_original_output=sum(
            row["status"] != "MISSING" for row in per_request),
        requests_completed=len(completed),
        metric="official LongBench English qa_f1_score, max over all reference answers, full output",
        mean_f1=sum(row["f1"] for row in per_request) / ROWS,
        mean_f1_percent=100.0 * sum(row["f1"] for row in per_request) / ROWS,
        normalized_exact_match_count=sum(row["normalized_exact_match"] for row in per_request),
        natural_eos_count=sum(row["natural_eos"] for row in per_request),
        runtime_eos_qualified=eos_qualified,
        length_cap_count=sum(row["status"] == "COMPLETE" and row["finish_reason"] == "length"
                             for row in per_request),
        empty_output_count=sum(row.get("empty_output", False) for row in per_request),
        output_tokens_total=sum(row["output_tokens"] for row in per_request),
        finish_reason_counts=dict(Counter(row["finish_reason"] or "missing_or_unfinished"
                                          for row in per_request)),
        truncated_input_count=sum(row["truncation_applied"] for row in per_request),
        answer_surface_scope="Copied weak input attribute only; not used for scoring, filtering, or evidence-retention claims",
        per_request=per_request, frozen_input_sha256=input_sha,
        original_run_sha256=run_sha,
        analysis_tokenizer_sha256=sha(tokenizer_path),
        metric_source_sha256=sha(Path(__file__).with_name("C_LONG_DOCUMENT_QA_METRIC_V1.py")),
        limitations=[
            "Source-first 16/150 rows and local OLMoE chat/truncation adaptation; not the official full LongBench model score.",
            "F1 scores the entire completed output against every reference and retains all 16 requests in the denominator; missing or incomplete outputs contribute zero.",
            "A stop counts as natural EOS only with a qualified runtime EOS receipt and null text stop reason.",
            "Truncation and answer-surface fields are weak frozen-input diagnostics, never scoring filters or proof that evidence survived.",
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input_dir, args.run_dir, args.metadata_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "requests_completed", "natural_eos_count",
                       "length_cap_count", "mean_f1_percent")}))
    if report["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
