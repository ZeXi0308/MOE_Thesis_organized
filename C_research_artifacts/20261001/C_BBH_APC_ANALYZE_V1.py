#!/usr/bin/env python3
"""Offline paired BBH APC output diagnosis; no equal-work speed claim."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
FROZEN_SCORER = HERE / "C_BBH_QUALIFICATION_ANALYZE_V1.py"
FROZEN_SCORER_SHA = "9a56605e1daeb038b34b36fd631457d980c7cd48c44dc73695f2ddad550bfdad"
TASK, COUNT, CAP = "geometric_shapes", 32, 512


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def frozen_scoring_functions():
    require(sha(FROZEN_SCORER) == FROZEN_SCORER_SHA,
            "frozen BBH extraction/normalization/suffix source changed")
    spec = importlib.util.spec_from_file_location("frozen_bbh_qualification_scorer",
                                                   FROZEN_SCORER)
    require(spec is not None and spec.loader is not None, "cannot load frozen BBH scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_difference(left: str, right: str) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def load_rows(path: Path, workload: dict, scorer) -> tuple[dict, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(rows, list) and len(rows) == COUNT,
            f"{path}: expected exactly 32 measured output rows")
    expected = {row["request_id"]: row for row in workload["requests"]}
    found, scored = {}, {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("request_id"), str),
                f"{path}: row lacks request ID")
        rid = row["request_id"]
        require(rid in expected and rid not in found, f"{path}: unexpected or duplicate {rid}")
        source = expected[rid]
        for key in ("task", "example_index", "prompt", "prompt_token_ids", "gold"):
            require(row.get(key) == source[key], f"{path}: {rid} {key} differs from input")
        require(row.get("finished") is True, f"{path}: {rid} not finished")
        text = row.get("output_text")
        ids = row.get("output_token_ids")
        require(isinstance(text, str) and isinstance(ids, list)
                and all(type(token) is int and token >= 0 for token in ids)
                and len(ids) <= CAP, f"{path}: {rid} invalid output")
        finish = row.get("finish_reason")
        require(finish in ("stop", "length"), f"{path}: {rid} invalid finish")
        elapsed = row.get("host_elapsed_s")
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                f"{path}: {rid} invalid host elapsed")
        prediction, method = scorer.extract_answer(text)
        normalized_prediction = scorer.normalize_exact_match(prediction)
        normalized_gold = scorer.normalize_exact_match(source["gold"])
        suffix = scorer.periodic_suffix(ids)
        found[rid] = row
        scored[rid] = dict(correct=normalized_prediction == normalized_gold,
            gold=source["gold"], prediction=prediction,
            extraction_method=method, normalized_prediction=normalized_prediction,
            normalized_gold=normalized_gold, output_token_count=len(ids),
            output_text_chars=len(text), output_text_sha256=sha_bytes(text.encode("utf-8")),
            finish_reason=finish, stop_reason=row.get("stop_reason"),
            host_elapsed_s=elapsed,
            longest_periodic_suffix=suffix,
            periodic_suffix_ge_256=suffix["token_count"] >= scorer.REPETITION_THRESHOLD,
            trailing_identical_token_run=scorer.trailing_identical_run(ids))
    require(set(found) == set(expected), f"{path}: missing measured IDs")
    return found, scored


def arm_summary(scored: dict) -> dict:
    rows = list(scored.values())
    times = [row["host_elapsed_s"] for row in rows]
    return dict(requests=len(rows), exact_correct=sum(row["correct"] for row in rows),
        exact_fraction=sum(row["correct"] for row in rows) / len(rows),
        finish_reason_counts=dict(sorted(Counter(row["finish_reason"] for row in rows).items())),
        stop_reason_counts=dict(sorted(Counter(
            json.dumps(row["stop_reason"], ensure_ascii=False, sort_keys=True)
            for row in rows).items())),
        total_output_tokens=sum(row["output_token_count"] for row in rows),
        mean_host_elapsed_s=sum(times) / len(times),
        median_host_elapsed_s=statistics.median(times),
        periodic_suffix_ge_256_count=sum(row["periodic_suffix_ge_256"] for row in rows))


def analyze(off_path: Path, on_path: Path, workload_path: Path) -> dict:
    scorer = frozen_scoring_functions()
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    requests = workload["requests"]
    require(workload["request_count"] == len(requests) == COUNT
            and [row["request_id"] for row in requests] ==
                [f"bbh/{TASK}/{index}" for index in range(COUNT)]
            and [row["example_index"] for row in requests] == list(range(COUNT))
            and all(row["task"] == TASK and isinstance(row["gold"], str)
                    and len(row["prompt_token_ids"]) + CAP <= 4096 for row in requests),
            "APC workload differs from fixed 32 source-ordered examples")
    off_raw, off_scored = load_rows(off_path, workload, scorer)
    on_raw, on_scored = load_rows(on_path, workload, scorer)
    per_request = []
    for source in requests:
        rid = source["request_id"]
        off, on = off_scored[rid], on_scored[rid]
        left_text, right_text = off_raw[rid]["output_text"], on_raw[rid]["output_text"]
        per_request.append(dict(request_id=rid, example_index=source["example_index"],
            prompt_token_count=len(source["prompt_token_ids"]),
            apc_off=off, apc_on=on,
            delta=dict(output_tokens_on_minus_off=
                on["output_token_count"] - off["output_token_count"],
                host_elapsed_s_on_minus_off=on["host_elapsed_s"] - off["host_elapsed_s"],
                raw_text_equal=left_text == right_text,
                raw_text_first_different_char=first_difference(left_text, right_text),
                raw_text_chars_on_minus_off=len(right_text) - len(left_text),
                output_token_ids_equal=(off_raw[rid]["output_token_ids"] ==
                                        on_raw[rid]["output_token_ids"]),
                normalized_prediction_equal=(off["normalized_prediction"] ==
                                             on["normalized_prediction"]),
                correctness_changed=off["correct"] != on["correct"],
                finish_reason_changed=off["finish_reason"] != on["finish_reason"],
                stop_reason_changed=off["stop_reason"] != on["stop_reason"])))
    deltas = [row["delta"] for row in per_request]
    return dict(schema="c-bbh-apc-analysis-v1",
        sources=dict(workload_sha256=sha(workload_path),
            apc_off_outputs_sha256=sha(off_path), apc_on_outputs_sha256=sha(on_path),
            frozen_scorer_sha256=FROZEN_SCORER_SHA),
        scope="32 source-first geometric_shapes examples, development APC on/off "
              "workload diagnostic. Raw outputs and host times are descriptive; "
              "no equal-work speed claim is made. No official BBH score or "
              "new admission-policy claim.",
        apc_off=arm_summary(off_scored), apc_on=arm_summary(on_scored),
        paired=dict(requests=COUNT,
            raw_text_equal=sum(row["raw_text_equal"] for row in deltas),
            output_token_ids_equal=sum(row["output_token_ids_equal"] for row in deltas),
            normalized_prediction_equal=sum(row["normalized_prediction_equal"] for row in deltas),
            correctness_changed=sum(row["correctness_changed"] for row in deltas),
            finish_reason_changed=sum(row["finish_reason_changed"] for row in deltas),
            stop_reason_changed=sum(row["stop_reason_changed"] for row in deltas),
            total_output_tokens_on_minus_off=sum(row["output_tokens_on_minus_off"] for row in deltas)),
        requests=per_request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apc-off-outputs", type=Path, required=True)
    parser.add_argument("--apc-on-outputs", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.apc_off_outputs, args.apc_on_outputs, args.workload)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), apc_off=value["apc_off"],
                          apc_on=value["apc_on"], paired=value["paired"]),
                     ensure_ascii=False))
