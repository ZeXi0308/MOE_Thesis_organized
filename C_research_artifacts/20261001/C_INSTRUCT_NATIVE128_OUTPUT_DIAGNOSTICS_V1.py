#!/usr/bin/env python3
"""Read-only output diagnostics for the completed native128 and overlapping fixed16 cells."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
STABLE = Path("/Users/zhaozhenyu/Desktop/毕业设计/C_research_artifacts/20261001")
EXPECTED_SHA256 = {
    "old_raw": "40f20006f55e76a3c58388adfac690063c25c76070c3d2e217044cf6974d7856",
    "new_raw": "a08e13948903e6ccc39215cf5fe3af874c82433bf197f9cb5c01a1ea9c791937",
    "old_quality": "cf4c5625736e90c96710030a8e030f6c0e713d77a7acafd8f1a0d3064a7be629",
    "new_quality": "f17ca9a15fd7abe2cdde9f6cf5914d7c74f632570e1cfb78108916b2bb56da06",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_by_id(rows: list[dict]) -> dict[str, dict]:
    indexed = {row["request_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate request ID")
    return indexed


def build(paths: dict[str, Path]) -> dict:
    hashes = {name: sha(path) for name, path in paths.items()}
    if hashes != EXPECTED_SHA256:
        raise ValueError("completed run or frozen quality input SHA-256 differs")
    old_raw, new_raw = load(paths["old_raw"]), load(paths["new_raw"])
    old_quality, new_quality = load(paths["old_quality"]), load(paths["new_quality"])
    if (len(old_raw) != 16 or len(new_raw) != 128
            or any(q.get("status") != "PILOT_COMPLETE" for q in (old_quality, new_quality))
            or len(old_quality["per_request"]) != 16
            or len(new_quality["per_request"]) != 128):
        raise ValueError("completed request inventory differs")
    old, new = index_by_id(old_raw), index_by_id(new_raw)
    old_score = index_by_id(old_quality["per_request"])
    new_score = index_by_id(new_quality["per_request"])
    if not set(old).issubset(new) or not set(old_score).issubset(new_score):
        raise ValueError("fixed16 requests are not contained in native128")

    overlap = []
    for index in range(16):
        rid = f"gsm8k-test-{index:04d}"
        a, b = old[rid], new[rid]
        ao, bo = old_score[rid], new_score[rid]
        if (not a["example_index"] == b["example_index"] == index
                or not ao["example_index"] == bo["example_index"] == index
                or not a["gold"] == b["gold"] == ao["gold"] == bo["gold"]):
            raise ValueError("overlap source identity differs")
        overlap.append(dict(
            example_index=index, request_id=rid,
            prompt_text_equal=a["prompt"] == b["prompt"],
            prompt_token_ids_equal=a["prompt_token_ids"] == b["prompt_token_ids"],
            output_text_equal=a["output_text"] == b["output_text"],
            output_token_ids_equal=a["output_token_ids"] == b["output_token_ids"],
            finish_reason_equal=a["finish_reason"] == b["finish_reason"],
            stop_reason_equal=a["stop_reason"] == b["stop_reason"],
            old_output_tokens=len(a["output_token_ids"]),
            new_output_tokens=len(b["output_token_ids"]),
            old_finish_reason=a["finish_reason"], new_finish_reason=b["finish_reason"],
            old_correct=ao["correct"], new_correct=bo["correct"],
            old_prediction=ao["prediction"], new_prediction=bo["prediction"],
        ))

    cap_cases = []
    for row in new_raw:
        if row["finish_reason"] != "length":
            continue
        score = new_score[row["request_id"]]
        cap_cases.append(dict(example_index=row["example_index"],
            request_id=row["request_id"], gold=row["gold"],
            prediction=score["prediction"], correct=score["correct"],
            output_tokens=len(row["output_token_ids"]),
            finish_reason=row["finish_reason"],
            exact_final_256_token_period_1_to_16=score["periodic_suffix_period"]))
    cap_cases.sort(key=lambda row: row["example_index"])
    if [case["example_index"] for case in cap_cases] != [98, 114]:
        raise ValueError("two known capped outputs differ")

    eligible = [row for row in new_score.values() if row["output_tokens"] >= 256]
    flagged = [row for row in eligible if row["periodic_suffix_period"] is not None]
    # These annotations are manually read from the pinned raw output. They are
    # separate from the scorer fields and are never used to compute correctness.
    manual = {
        "kind": "manual_output_reading_not_automatic_scoring",
        "gsm8k-test-0098": {
            "observation": "Sets the first-15-minute count to x, then repeats an unsolved expression to the cap.",
            "evidence_excerpts": ["let's call it x", "30 - x - 15 = 30 - x - 15"],
        },
        "gsm8k-test-0114": {
            "observation": "Gives 15000 to the original question, then continues with unrelated synthetic Question/Answer exercises; the final 2 is extracted from an unfinished extra question.",
            "evidence_excerpts": ["So the answer is 15000.",
                                  "Question: A pizza has 8 slices.",
                                  "Question: There are 12 apples in a basket. If 2 are taken, how many are left?"],
        },
    }
    for rid in ("gsm8k-test-0098", "gsm8k-test-0114"):
        if any(excerpt not in new[rid]["output_text"]
               for excerpt in manual[rid]["evidence_excerpts"]):
            raise ValueError("manual excerpt no longer appears in pinned output")

    counts = dict(
        requests=16,
        prompt_text_equal=sum(row["prompt_text_equal"] for row in overlap),
        prompt_token_ids_equal=sum(row["prompt_token_ids_equal"] for row in overlap),
        output_text_equal=sum(row["output_text_equal"] for row in overlap),
        output_token_ids_equal=sum(row["output_token_ids_equal"] for row in overlap),
        finish_reason_equal=sum(row["finish_reason_equal"] for row in overlap),
        stop_reason_equal=sum(row["stop_reason_equal"] for row in overlap),
        output_token_length_changed=sum(row["old_output_tokens"] != row["new_output_tokens"]
                                        for row in overlap),
        output_changed_indices=[row["example_index"] for row in overlap
                                if not row["output_text_equal"] or not row["output_token_ids_equal"]],
    )
    return dict(
        schema="c-instruct-native128-output-diagnostics-v1",
        scope="Pinned completed-output comparison only; no concurrency effect, performance, or quality-causality claim",
        inputs={name: dict(path=str(paths[name]), sha256=hashes[name]) for name in paths},
        capped_outputs=cap_cases,
        exact_suffix_diagnostic=dict(
            definition=new_quality["periodic_suffix_definition"],
            minimum_output_tokens=256, eligible_requests=len(eligible),
            eligible_eos=sum(row["finish_reason"] == "stop" for row in eligible),
            eligible_length_capped=sum(row["finish_reason"] == "length" for row in eligible),
            flagged_requests=len(flagged),
            flagged_indices=sorted(row["example_index"] for row in flagged),
            limitation="No exact token period flag does not rule out semantic or variable-template repetition"),
        fixed16_overlap_counts=counts,
        fixed16_overlap_per_request=overlap,
        fixed16_correctness=dict(
            old_correct=sum(row["old_correct"] for row in overlap),
            new_correct=sum(row["new_correct"] for row in overlap),
            incorrect_to_correct_indices=[row["example_index"] for row in overlap
                                          if not row["old_correct"] and row["new_correct"]],
            correct_to_incorrect_indices=[row["example_index"] for row in overlap
                                          if row["old_correct"] and not row["new_correct"]]),
        manual_annotations=manual,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-raw", type=Path, default=STABLE / "c-instruct-gsm8k-qualification-spare-dev-v1/native/measured-outputs.json")
    parser.add_argument("--new-raw", type=Path, default=STABLE / "c-instruct-gsm8k-native128-dev-v1/native/measured-outputs.json")
    parser.add_argument("--old-quality", type=Path, default=BASE / "instruct_gsm8k_spare_qualification_v1.json")
    parser.add_argument("--new-quality", type=Path, default=BASE / "instruct_native128_quality_v1.json")
    parser.add_argument("--output", type=Path, default=BASE / "instruct_native128_output_diagnostics_v1.json")
    args = parser.parse_args()
    paths = dict(old_raw=args.old_raw, new_raw=args.new_raw,
                 old_quality=args.old_quality, new_quality=args.new_quality)
    result = build(paths)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(capped=[r["example_index"] for r in result["capped_outputs"]],
        suffix_eligible=result["exact_suffix_diagnostic"]["eligible_requests"],
        suffix_flagged=result["exact_suffix_diagnostic"]["flagged_requests"],
        overlap_equal=result["fixed16_overlap_counts"]["output_text_equal"],
        old_correct=result["fixed16_correctness"]["old_correct"],
        new_correct=result["fixed16_correctness"]["new_correct"])))


if __name__ == "__main__":
    main()
