#!/usr/bin/env python3
"""CPU-only analysis of the frozen 27-task BBH qualification cell.

Scoring follows the historical Open-Instruct BBH evaluator, not a forgiving
final-answer parser:
https://github.com/allenai/open-instruct/blob/d05effeb4df018dd82c15a956a4a58da82547eb5/eval/bbh/run_eval.py

Its evaluate/exact_match implementation lowercases and deletes ASCII
punctuation. It does not remove articles or normalize whitespace:
https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py

The periodic-suffix measure is a predeclared repetition diagnostic only.
It never changes the 27-task denominator or the exact-match score.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import string
import subprocess
from typing import Any


BBH_COMMIT = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
BBH_SOURCE_DEFAULT = Path("/private/tmp/c-bbh-source-20261001")
EXPECTED_TASKS = frozenset(
    """boolean_expressions causal_judgement date_understanding disambiguation_qa
    dyck_languages formal_fallacies geometric_shapes hyperbaton
    logical_deduction_five_objects logical_deduction_seven_objects
    logical_deduction_three_objects movie_recommendation
    multistep_arithmetic_two navigate object_counting penguins_in_a_table
    reasoning_about_colored_objects ruin_names salient_translation_error_detection
    snarks sports_understanding temporal_sequences
    tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects
    tracking_shuffled_objects_three_objects web_of_lies word_sorting""".split()
)
ANSWER_RE = re.compile(r"[t|T]he answer is (.*?)\.")
PUNCTUATION_DELETE = str.maketrans("", "", string.punctuation)
REQUIRED_KEYS = frozenset(
    {
        "task",
        "request_id",
        "gold",
        "prompt_token_ids",
        "output_text",
        "output_token_ids",
        "finish_reason",
        "stop_reason",
        "host_elapsed_s",
    }
)
MAX_PERIOD = 16
REPETITION_THRESHOLD = 256


def extract_answer(output: str) -> tuple[str, str]:
    match = ANSWER_RE.search(output)
    if match:
        return match.group(1).strip(), "first_the_answer_is_period"
    return output.strip(), "whole_output_fallback"


def normalize_exact_match(value: str) -> str:
    return value.lower().translate(PUNCTUATION_DELETE)


def periodic_suffix(token_ids: list[int]) -> dict[str, int | None]:
    """Longest suffix with at least two full repeats, for period 1..16.

    Periodicity means each token equals the token one period earlier within
    the suffix. Ties use the shortest period. Complexity is O(16 * tokens).
    """
    n = len(token_ids)
    best_length = 0
    best_period: int | None = None
    for period in range(1, min(MAX_PERIOD, n // 2) + 1):
        i = n - period - 1
        while i >= 0 and token_ids[i] == token_ids[i + period]:
            i -= 1
        length = n - i - 1
        if length >= 2 * period and (
            length > best_length or (length == best_length and (best_period is None or period < best_period))
        ):
            best_length = length
            best_period = period
    return {"period": best_period, "token_count": best_length}


def trailing_identical_run(token_ids: list[int]) -> int:
    if not token_ids:
        return 0
    i = len(token_ids) - 1
    while i > 0 and token_ids[i - 1] == token_ids[-1]:
        i -= 1
    return len(token_ids) - i


def _nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _validate_token_ids(value: Any, name: str, task: str) -> list[int]:
    if not isinstance(value, list) or any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError(f"{task}: {name} must be a list of nonnegative integer token IDs")
    return value


def load_official_first_gold(source: Path) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if commit != BBH_COMMIT:
        raise ValueError(f"BBH source commit {commit} differs from frozen {BBH_COMMIT}")
    task_files = sorted((source / "bbh").glob("*.json"))
    prompt_files = sorted((source / "cot-prompts").glob("*.txt"))
    if {path.stem for path in task_files} != EXPECTED_TASKS:
        raise ValueError("BBH source does not contain the frozen 27 task JSON files")
    if {path.stem for path in prompt_files} != EXPECTED_TASKS:
        raise ValueError("BBH source does not contain the matching 27 CoT prompts")
    first_gold: dict[str, str] = {}
    for path in task_files:
        examples = json.loads(path.read_text(encoding="utf-8"))["examples"]
        if not examples or not isinstance(examples[0].get("target"), str):
            raise ValueError(f"invalid first example in {path}")
        first_gold[path.stem] = examples[0]["target"]
    return first_gold


def analyze(rows: list[dict[str, Any]], first_gold: dict[str, str]) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError("outputs.json must be a JSON list")
    if set(first_gold) != EXPECTED_TASKS:
        raise ValueError("official gold mapping must contain the frozen 27 tasks")
    if len(rows) != len(EXPECTED_TASKS):
        raise ValueError(f"expected exactly 27 output rows, found {len(rows)}")
    seen_tasks: set[str] = set()
    seen_ids: set[str] = set()
    tasks = []
    for row in rows:
        if not isinstance(row, dict) or not REQUIRED_KEYS <= row.keys():
            raise ValueError(f"output row lacks required fields: {row!r}")
        task = row["task"]
        if not isinstance(task, str) or task not in EXPECTED_TASKS or task in seen_tasks:
            raise ValueError(f"unexpected or duplicate task {task!r}")
        seen_tasks.add(task)
        request_id = row["request_id"]
        if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
            raise ValueError(f"{task}: invalid request_id")
        id_key = str(request_id)
        if id_key in seen_ids:
            raise ValueError(f"duplicate request_id {request_id!r}")
        seen_ids.add(id_key)
        gold = row["gold"]
        if not isinstance(gold, str) or gold != first_gold[task]:
            raise ValueError(f"{task}: gold does not match official first example")
        output = row["output_text"]
        if not isinstance(output, str):
            raise ValueError(f"{task}: output_text must be a string, possibly empty")
        prompt_ids = _validate_token_ids(row["prompt_token_ids"], "prompt_token_ids", task)
        output_ids = _validate_token_ids(row["output_token_ids"], "output_token_ids", task)
        if row["finish_reason"] is not None and not isinstance(row["finish_reason"], str):
            raise ValueError(f"{task}: invalid finish_reason")
        elapsed = row["host_elapsed_s"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"{task}: invalid host_elapsed_s")
        prediction, extraction_method = extract_answer(output)
        normalized_prediction = normalize_exact_match(prediction)
        normalized_gold = normalize_exact_match(gold)
        suffix = periodic_suffix(output_ids)
        tasks.append(
            {
                "task": task,
                "request_id": request_id,
                "gold": gold,
                "prediction": prediction,
                "extraction_method": extraction_method,
                "correct": normalized_prediction == normalized_gold,
                "normalized_prediction": normalized_prediction,
                "normalized_gold": normalized_gold,
                "prompt_token_count": len(prompt_ids),
                "output_token_count": len(output_ids),
                "finish_reason": row["finish_reason"],
                "stop_reason": row["stop_reason"],
                "host_elapsed_s": elapsed,
                "last_16_output_token_ids": output_ids[-16:],
                "adjacent_equal_token_pairs": sum(a == b for a, b in zip(output_ids, output_ids[1:])),
                "trailing_identical_token_run": trailing_identical_run(output_ids),
                "longest_periodic_suffix": suffix,
                "periodic_suffix_ge_256": suffix["token_count"] >= REPETITION_THRESHOLD,
                "raw_record": row,
            }
        )
    if seen_tasks != EXPECTED_TASKS:
        raise ValueError(f"missing tasks: {sorted(EXPECTED_TASKS - seen_tasks)}")
    tasks.sort(key=lambda row: row["task"])
    lengths = [row["output_token_count"] for row in tasks]
    correct = sum(row["correct"] for row in tasks)
    bins = {
        "0": sum(n == 0 for n in lengths),
        "1-31": sum(1 <= n <= 31 for n in lengths),
        "32-63": sum(32 <= n <= 63 for n in lengths),
        "64-127": sum(64 <= n <= 127 for n in lengths),
        "128-255": sum(128 <= n <= 255 for n in lengths),
        "256-511": sum(256 <= n <= 511 for n in lengths),
        "512+": sum(n >= 512 for n in lengths),
    }
    return {
        "schema_version": 1,
        "sources": {
            "bbh_commit": BBH_COMMIT,
            "bbh_repo": "https://github.com/suzgunmirac/BIG-Bench-Hard",
            "scoring_script": "https://github.com/allenai/open-instruct/blob/d05effeb4df018dd82c15a956a4a58da82547eb5/eval/bbh/run_eval.py",
            "exact_match_metric": "https://github.com/huggingface/evaluate/blob/main/metrics/exact_match/exact_match.py",
        },
        "denominator": {"expected_tasks": 27, "present_tasks": len(tasks), "scored_tasks": len(tasks), "correct": correct, "accuracy": correct / len(tasks)},
        "finish_reason_counts": dict(sorted(Counter("<null>" if row["finish_reason"] is None else row["finish_reason"] for row in tasks).items())),
        "stop_reason_counts": dict(sorted(Counter(json.dumps(row["stop_reason"], ensure_ascii=False, sort_keys=True) for row in tasks).items())),
        "output_token_lengths": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "p50_nearest_rank": _nearest_rank(lengths, 0.5),
            "p90_nearest_rank": _nearest_rank(lengths, 0.9),
            "p95_nearest_rank": _nearest_rank(lengths, 0.95),
            "bins": bins,
        },
        "repetition_diagnostic": {
            "max_period_tokens": MAX_PERIOD,
            "minimum_periodic_suffix_tokens": REPETITION_THRESHOLD,
            "flagged_count": sum(row["periodic_suffix_ge_256"] for row in tasks),
            "flagged_tasks": [row["task"] for row in tasks if row["periodic_suffix_ge_256"]],
            "affects_scoring": False,
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path, help="cell outputs.json: exactly one first-example record per BBH task")
    parser.add_argument("--out", type=Path, required=True, help="analysis JSON destination")
    parser.add_argument("--bbh-source", type=Path, default=BBH_SOURCE_DEFAULT)
    args = parser.parse_args()
    if args.outputs.resolve() == args.out.resolve():
        parser.error("--out must differ from outputs input")
    first_gold = load_official_first_gold(args.bbh_source)
    rows = json.loads(args.outputs.read_text(encoding="utf-8"))
    report = analyze(rows, first_gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), **report["denominator"], "repetition_flags": report["repetition_diagnostic"]["flagged_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
