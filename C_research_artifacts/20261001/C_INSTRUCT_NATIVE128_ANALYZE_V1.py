#!/usr/bin/env python3
"""Score an original native128 GSM8K observation cell without changing it.

The first 128 official test rows are a source-order systems observation, not the
authors' 200-example GSM8K benchmark. All planned requests remain in the
denominator, including failed or unfinished outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


EXPECTED_REQUESTS = 128
CHECKPOINTS = (64, 128, 256, 512, 1024)
NUMBERS = re.compile(r"[-+]?\d*\.\d+|\d+")
GROUP_COMMA = re.compile(r"(\d),(\d)")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def optional(path: Path):
    return load(path) if path.is_file() else None


def normalized_last_number(text: str) -> str | None:
    # Exact historical eval/gsm/run_eval.py extraction: remove each digit
    # grouping comma, then take the last regex number from model output.
    normalized = GROUP_COMMA.sub(r"\1\2", text)
    numbers = NUMBERS.findall(normalized)
    return numbers[-1] if numbers else None


def periodic_suffix_period(ids: list[int], *, span: int = 256,
                           max_period: int = 16) -> int | None:
    """Shortest exact token period in the final fixed-width suffix, if any."""
    if len(ids) < span:
        return None
    suffix = ids[-span:]
    return next((period for period in range(1, max_period + 1)
                 if all(suffix[index] == suffix[index - period]
                        for index in range(period, span))), None)


def validated_inputs(inputs: Path) -> tuple[dict, dict, dict]:
    config_path, workload_path = inputs / "config.json", inputs / "workload.json"
    config, workload = load(config_path), load(workload_path)
    receipt = load(inputs / "SOURCE_RECEIPT.json")
    if (config.get("schema") != "c-instruct-native128-config-v1"
            or workload.get("schema") != "c-instruct-native128-workload-v1"
            or receipt.get("schema") != "c-instruct-native128-source-receipt-v1"
            or config.get("workload_sha256") != sha(workload_path)
            or receipt.get("workload_sha256") != sha(workload_path)
            or receipt.get("config_sha256") != sha(config_path)):
        raise ValueError("input schema or frozen file hashes differ")
    rows = workload.get("requests")
    if (not isinstance(rows, list) or len(rows) != EXPECTED_REQUESTS
            or workload.get("arrival_traces_s") != [0.0] * EXPECTED_REQUESTS
            or [row.get("example_index") for row in rows] != list(range(EXPECTED_REQUESTS))
            or len({row.get("request_id") for row in rows}) != EXPECTED_REQUESTS):
        raise ValueError("source-order request or arrival contract differs")
    sampling = workload.get("sampling")
    if sampling != dict(temperature=0.0, max_tokens=1024, min_tokens=0,
                        ignore_eos=False, stop=[]):
        raise ValueError("sampling contract differs")
    if (config.get("requests") != EXPECTED_REQUESTS
            or config.get("output_tokens") != 1024
            or config.get("max_model_len") != 4096):
        raise ValueError("config request, output, or context bound differs")
    if any(len(row["prompt_token_ids"]) + 1024 > 4096 for row in rows):
        raise ValueError("frozen prompt exceeds the context limit")
    return config, workload, receipt


def check_output_identity(row: dict, source: dict) -> list[str]:
    issues = []
    for field in ("request_id", "example_index", "question", "gold", "prompt",
                  "chat_user_content", "prompt_token_ids"):
        if row.get(field) != source.get(field):
            issues.append("output_" + field + "_differs")
    ids = row.get("output_token_ids")
    times = row.get("token_times_s")
    if (not isinstance(ids, list) or not isinstance(times, list)
            or len(ids) != len(times)
            or any(type(token) is not int or token < 0 for token in ids)
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0
                   for t in times)
            or any(a > b for a, b in zip(times, times[1:]))):
        issues.append("invalid_output_tokens_or_host_times")
    if isinstance(ids, list) and len(ids) > 1024:
        issues.append("output_exceeds_1024")
    reason = row.get("finish_reason")
    if row.get("finished") is True:
        if reason not in ("stop", "length"):
            issues.append("unexpected_finish_reason")
        if reason == "length" and isinstance(ids, list) and len(ids) != 1024:
            issues.append("length_finish_without_1024_tokens")
    elif reason is not None:
        issues.append("unfinished_with_finish_reason")
    if reason == "stop" and row.get("stop_reason") is not None:
        issues.append("stop_reason_not_eos_only")
    return issues


def analyze(inputs: Path, run: Path, metadata_dir: Path) -> dict:
    from tokenizers import Tokenizer

    config, workload, source_receipt = validated_inputs(inputs)
    tokenizer_path = metadata_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError("local pinned tokenizer.json required for token checkpoint decode")
    if (source_receipt.get("local_metadata_sha256", {}).get("tokenizer.json")
            != sha(tokenizer_path)):
        raise ValueError("analysis tokenizer differs from frozen input tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    names = ("status.json", "measured-outputs.json", "measured-steps.json",
             "input-tokenizer-check.json", "resolved-eos.json", "native-drain.json",
             "timing.json", "environment.json")
    original_hashes = {name: sha(run / name) if (run / name).is_file() else None
                       for name in names}
    status = optional(run / "status.json")
    outputs = optional(run / "measured-outputs.json")
    tokenizer_check = optional(run / "input-tokenizer-check.json")
    eos_receipt = optional(run / "resolved-eos.json")
    drain = optional(run / "native-drain.json")
    environment = optional(run / "environment.json")
    issues = []
    if status is None or status.get("status") != "COMPLETE":
        issues.append("cell_status_not_complete")
    if outputs is None:
        issues.append("measured_outputs_missing")
        outputs = []
    if tokenizer_check is None or tokenizer_check.get("status") != "QUALIFIED":
        issues.append("runtime_tokenizer_check_not_qualified")
    if eos_receipt is None or eos_receipt.get("qualification_status") != "QUALIFIED":
        issues.append("resolved_eos_not_qualified")
    if drain is None or drain.get("status") != "QUALIFIED":
        issues.append("native_drain_not_qualified")
    if (environment is None
            or environment.get("input_sha256", {}).get("workload.json")
            != sha(inputs / "workload.json")
            or environment.get("input_sha256", {}).get("config.json")
            != sha(inputs / "config.json")
            or environment.get("input_sha256", {}).get("SOURCE_RECEIPT.json")
            != sha(inputs / "SOURCE_RECEIPT.json")):
        issues.append("environment_input_identity_not_verified")
    if (not isinstance(outputs, list)
            or len(outputs) != EXPECTED_REQUESTS
            or len({row.get("request_id") for row in outputs
                    if isinstance(row, dict)}) != len(outputs)):
        issues.append("measured_request_inventory_differs")
        outputs = outputs if isinstance(outputs, list) else []
    by_id = {row.get("request_id"): row for row in outputs if isinstance(row, dict)}
    per_request = []
    for source in workload["requests"]:
        rid = source["request_id"]
        row = by_id.get(rid)
        if row is None:
            per_request.append(dict(request_id=rid,
                                    example_index=source["example_index"],
                                    gold=source["gold"], status="MISSING",
                                    correct=False, correct_and_eos=False,
                                    finish_reason=None,
                                    output_tokens=0, prediction=None,
                                    periodic_suffix_period=None,
                                    checkpoint_diagnostics={},
                                    issues=["original_request_output_missing"]))
            continue
        row_issues = check_output_identity(row, source)
        if row_issues:
            issues.extend(rid + ":" + issue for issue in row_issues)
        ids = row.get("output_token_ids")
        ids = ids if isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids) else []
        period = periodic_suffix_period(ids)
        text = row.get("output_text")
        if not isinstance(text, str):
            text = ""
            row_issues.append("output_text_missing")
            issues.append(rid + ":output_text_missing")
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        prediction = normalized_last_number(text)
        completed = row.get("finished") is True and row.get("finish_reason") in ("stop", "length")
        checkpoints = {}
        times = row.get("token_times_s")
        times = times if isinstance(times, list) else []
        for mark in CHECKPOINTS:
            if len(ids) < mark:
                continue
            provisional = normalized_last_number(
                tokenizer.decode(ids[:mark], skip_special_tokens=True))
            checkpoints[str(mark)] = dict(
                provisional_last_number=provisional,
                provisional_gold_match=provisional == source["gold"],
                host_return_s=times[mark - 1] if len(times) >= mark else None,
                semantics="prefix-only provisional; the model may continue or revise its answer",
            )
        per_request.append(dict(
            request_id=rid, example_index=source["example_index"],
            gold=source["gold"],
            status="COMPLETE" if completed else "INCOMPLETE",
            finish_reason=row.get("finish_reason"),
            eos_only_stop_inferred=completed and row.get("finish_reason") == "stop"
                                   and row.get("stop_reason") is None,
            output_tokens=len(ids), output_text=text,
            decoded_token_text_matches_native_text=(decoded == text),
            prediction=prediction, correct=completed and prediction == source["gold"],
            correct_and_eos=(completed and prediction == source["gold"]
                             and row.get("finish_reason") == "stop"),
            periodic_suffix_period=period,
            host_completion_s=row.get("host_elapsed_s"),
            checkpoint_diagnostics=checkpoints,
            issues=row_issues,
        ))
    if set(by_id) != {source["request_id"] for source in workload["requests"]}:
        issues.append("extra_or_missing_measured_request_id")
    completed = [r for r in per_request if r["status"] == "COMPLETE"]
    checkpoints_summary = {}
    for mark in CHECKPOINTS:
        key = str(mark)
        reached = [r["checkpoint_diagnostics"][key] for r in per_request
                   if key in r.get("checkpoint_diagnostics", {})]
        checkpoints_summary[key] = dict(
            requests_reaching_checkpoint=len(reached),
            provisional_last_number_present=sum(
                r["provisional_last_number"] is not None for r in reached),
            provisional_gold_matches=sum(r["provisional_gold_match"] for r in reached),
            semantics="descriptive prefix status at a fixed generated-token count; not final accuracy",
        )
    return dict(
        schema="c-instruct-native128-analysis-v1",
        status=("PILOT_COMPLETE" if not issues and len(completed) == EXPECTED_REQUESTS
                else "INCOMPLETE_OR_INVALID"),
        issues=issues,
        model=config["model"],
        task="gsm8k", requests_planned=EXPECTED_REQUESTS,
        requests_with_original_output=sum(r["status"] != "MISSING" for r in per_request),
        requests_completed=len(completed),
        natural_eos_count=sum(r["finish_reason"] == "stop" for r in completed),
        length_cap_count=sum(r["finish_reason"] == "length" for r in completed),
        numeric_prediction_count=sum(r["prediction"] is not None for r in completed),
        answer_correct_count=sum(r["correct"] for r in per_request),
        answer_correct_and_eos_count=sum(r["correct_and_eos"] for r in per_request),
        answer_correct_and_length_capped_count=sum(
            r["correct"] and r["finish_reason"] == "length" for r in per_request),
        output_tokens_total=sum(r["output_tokens"] for r in per_request),
        periodic_suffix_count=sum(r["periodic_suffix_period"] is not None
                                  for r in per_request),
        periodic_suffix_length_capped_count=sum(
            r["periodic_suffix_period"] is not None
            and r["finish_reason"] == "length" for r in per_request),
        periodic_suffix_definition="Exact repetition of the last 256 generated token IDs with shortest period 1..16; descriptive output-loop flag",
        finish_reason_counts=dict(Counter(
            r["finish_reason"] or "missing_or_unfinished" for r in per_request)),
        checkpoint_diagnostics=checkpoints_summary,
        per_request=per_request,
        frozen_input_sha256={name: sha(inputs / name)
                             for name in ("config.json", "workload.json", "SOURCE_RECEIPT.json")},
        original_run_sha256=original_hashes,
        analysis_tokenizer_sha256=sha(tokenizer_path),
        limitations=[
            "First 128 official test rows in source order; not the OLMoE authors' random 200-example benchmark.",
            "Eight historical CoT exemplars and OLMoE chat prompt, but 1024 generated-token cap rather than historical 512.",
            "Greedy EOS-only native generation with no text stop strings; a stop finish is attributed to EOS only when the runtime EOS receipt is qualified and stop_reason is null.",
            "Numeric extraction follows historical comma removal and last-number regex; this first-128 cohort has positive integer gold, for which string equality matches the historical exact-match result.",
            "Prefix checkpoints are provisional diagnostics; no answer-based task selection or early-exit correctness claim.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path,
                        default=Path("/private/tmp/c-olmoe-instruct-metadata-20261001"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"immutable analysis already exists: {args.output}")
    report = analyze(args.inputs_dir, args.run_dir, args.metadata_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in
                      ("status", "requests_completed", "natural_eos_count",
                       "length_cap_count", "answer_correct_count")}))
    if report["status"] != "PILOT_COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
