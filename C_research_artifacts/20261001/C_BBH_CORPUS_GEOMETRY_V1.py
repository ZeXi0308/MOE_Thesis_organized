#!/usr/bin/env python3
"""Token/cap geometry of every official pinned BBH example; no generation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
TOKENIZER_SHA = "a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40"
QUAL_WORKLOAD_SHA = "7f429bb80ba24b444300bf72a4f69625a320748a65ac793f58fab0dafbafeac5"
OUTPUT_CAP, CONTEXT, BLOCK, SLOTS, USABLE = 512, 4096, 16, 32, 4096


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def cap_blocks(prompt_tokens: int) -> int:
    return (prompt_tokens + OUTPUT_CAP + BLOCK - 1) // BLOCK


def distribution(values: list[int]) -> dict:
    require(bool(values), "empty distribution")
    ordered = sorted(values)
    def rank(percent: int) -> int:
        return ordered[math.ceil(percent * len(ordered) / 100) - 1]
    return dict(count=len(values), min=ordered[0], p10=rank(10), p25=rank(25),
                median=statistics.median(ordered), p75=rank(75), p90=rank(90),
                p95=rank(95), p99=rank(99), max=ordered[-1],
                mean=sum(values) / len(values))


def extreme32(values: list[int]) -> dict:
    if len(values) < SLOTS:
        return dict(count=len(values), smallest32_sum_blocks=None,
                    largest32_sum_blocks=None)
    ordered = sorted(values)
    return dict(count=len(values), smallest32_sum_blocks=sum(ordered[:SLOTS]),
                largest32_sum_blocks=sum(ordered[-SLOTS:]))


def prompt_hist(values: list[int]) -> dict:
    labels = ("P<512", "512<=P<1024", "1024<=P<1536", "1536<=P<2048",
              "2048<=P<2560", "2560<=P<3072", "3072<=P<=3584", "P>3584")
    counts = Counter()
    for value in values:
        index = next((i for i, upper in enumerate((512, 1024, 1536, 2048,
                                                      2560, 3072))
                      if value < upper), None)
        if index is None:
            index = 6 if value <= CONTEXT - OUTPUT_CAP else 7
        counts[labels[index]] += 1
    return {label: counts[label] for label in labels}


def common_prefix(current: list[int] | None, ids: list[int]) -> list[int]:
    if current is None:
        return ids.copy()
    matched = 0
    for left, right in zip(current, ids):
        if left != right:
            break
        matched += 1
    return current[:matched]


def analyze(source: Path, tokenizer_file: Path, qualification_workload: Path) -> dict:
    require(sha(tokenizer_file) == TOKENIZER_SHA, "tokenizer pin changed")
    require(sha(qualification_workload) == QUAL_WORKLOAD_SHA,
            "fixed first-example qualification workload changed")
    from tokenizers import Tokenizer, __version__ as tokenizers_version
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    qualification = json.loads(qualification_workload.read_text())["requests"]
    tasks = sorted(path.stem for path in (source / "bbh").glob("*.json"))
    prompts = sorted(path.stem for path in (source / "cot-prompts").glob("*.txt"))
    require(len(tasks) == 27 and tasks == prompts and len(qualification) == 27,
            "expected 27 matching BBH tasks and 27 fixed qualification requests")
    all_lengths, all_blocks, eligible_blocks = [], [], []
    first27_blocks, first27_lengths, by_task, manifest = [], [], {}, []
    for task_index, task in enumerate(tasks):
        data_file = source / "bbh" / f"{task}.json"
        prompt_file = source / "cot-prompts" / f"{task}.txt"
        data = json.loads(data_file.read_text(encoding="utf-8"))
        examples = data.get("examples")
        require(isinstance(examples, list) and examples, f"{task}: no examples")
        with prompt_file.open(encoding="utf-8") as stream:
            prefix = "".join(stream.readlines()[2:]).strip()
        require(bool(prefix), f"{task}: empty official CoT prefix")
        lengths, blocks, valid_blocks = [], [], []
        lcp = None
        for index, example in enumerate(examples):
            require(isinstance(example, dict) and isinstance(example.get("input"), str)
                    and isinstance(example.get("target"), str),
                    f"{task}: invalid example {index}")
            text = prefix + "\n\nQ: " + example["input"] + "\nA:"
            ids = tokenizer.encode(text, add_special_tokens=True).ids
            length, bound = len(ids), cap_blocks(len(ids))
            lcp = common_prefix(lcp, ids)
            lengths.append(length)
            blocks.append(bound)
            all_lengths.append(length)
            all_blocks.append(bound)
            if length + OUTPUT_CAP <= CONTEXT:
                valid_blocks.append(bound)
                eligible_blocks.append(bound)
            if index == 0:
                fixed = qualification[task_index]
                require(fixed["task"] == task and fixed["example_index"] == 0
                        and fixed["prompt"] == text and fixed["prompt_token_ids"] == ids,
                        f"{task}: fixed qualification first example differs")
                first27_blocks.append(bound)
                first27_lengths.append(length)
        lcp_len = len(lcp)
        by_task[task] = dict(examples=len(examples),
            prompt_tokens=distribution(lengths),
            context_violations=sum(length + OUTPUT_CAP > CONTEXT for length in lengths),
            all_fullcap_blocks=extreme32(blocks),
            context_eligible_fullcap_blocks=extreme32(valid_blocks),
            source_first32_fullcap_sum_blocks=sum(blocks[:SLOTS])
                if len(blocks) >= SLOTS else None,
            same_task_largest32_eligible_exceeds_usable=
                len(valid_blocks) >= SLOTS and sum(sorted(valid_blocks)[-SLOTS:]) > USABLE,
            structural_shared_prefix_tokens=lcp_len,
            structural_whole_common_blocks_floor=lcp_len // BLOCK,
            structural_prefix_touched_blocks_ceil=math.ceil(lcp_len / BLOCK))
        for path in (data_file, prompt_file):
            manifest.append(dict(path=str(path.relative_to(source)),
                                 bytes=path.stat().st_size, sha256=sha(path)))
    manifest.sort(key=lambda row: row["path"])
    violations = sum(length + OUTPUT_CAP > CONTEXT for length in all_lengths)
    return dict(schema="c-bbh-corpus-geometry-v1",
        source=dict(repository="suzgunmirac/BIG-Bench-Hard", revision=REVISION,
                    files=manifest, tokenizer_sha256=TOKENIZER_SHA,
                    tokenizers_version=tokenizers_version,
                    qualification_workload_sha256=QUAL_WORKLOAD_SHA),
        method=dict(prompt_rule="''.join(prompt_file.readlines()[2:]).strip() + "
                         "'\\n\\nQ: ' + input + '\\nA:'",
                    tokenization="tokenizers.Tokenizer.encode(add_special_tokens=True)",
                    output_cap=OUTPUT_CAP, context_tokens=CONTEXT, block_tokens=BLOCK,
                    max_num_seqs=SLOTS, usable_blocks=USABLE,
                    quantiles="nearest-rank; median is conventional midpoint"),
        overall=dict(tasks=len(tasks), examples=len(all_lengths),
            prompt_tokens=distribution(all_lengths),
            prompt_token_hist=prompt_hist(all_lengths),
            context_violations=violations,
            context_eligible_examples=len(eligible_blocks),
            all_fullcap_blocks=extreme32(all_blocks),
            context_eligible_fullcap_blocks=extreme32(eligible_blocks),
            fixed_first27_fullcap_sum_blocks=sum(first27_blocks),
            fixed_first27_fullcap_sum_tokens=sum(length + OUTPUT_CAP
                                                 for length in first27_lengths)),
        by_task=by_task,
        interpretation="ceil((P+512)/16) is a full-cap reservation bound, not actual KV "
                       "occupancy. Common-prefix tokens and floor/ceil blocks describe "
                       "structure, not physical APC savings or a measured service result.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tokenizer-file", type=Path, required=True)
    parser.add_argument("--qualification-workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.source_root, args.tokenizer_file, args.qualification_workload)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        stream.write("\n")
