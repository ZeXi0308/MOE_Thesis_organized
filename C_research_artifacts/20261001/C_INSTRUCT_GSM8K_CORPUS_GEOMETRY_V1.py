#!/usr/bin/env python3
"""CPU-only geometry of the pinned official GSM8K test set under the fixed Instruct prompt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from C_INSTRUCT_GSM8K_INPUTS_V1 import (
    CONTEXT_LIMIT, MODEL_ID, MODEL_REVISION, OUTPUT_CAP, render_prompt,
    sha, source_data, tokenizer_data,
)

BLOCK_TOKENS = 16
EXPECTED_ROWS = 1319
COHORTS = (32, 64, 128)


def nearest_rank(sorted_values: list[int], percent: int) -> int:
    return sorted_values[(percent * len(sorted_values) + 99) // 100 - 1]


def cohort_geometry(prompts: list[list[int]], lengths: list[int], count: int) -> dict:
    # A node is one exact 16-token prompt block plus its preceding prefix node.
    # This counts only full prompt blocks as shared. Each request keeps its own
    # partial prompt tail and all 1024 possible output tokens.
    nodes: dict[tuple[int, tuple[int, ...]], int] = {}
    full_prompt_blocks = 0
    independent_cap_blocks = 0
    for ids, prompt_len in zip(prompts[:count], lengths[:count], strict=True):
        parent = 0
        blocks = prompt_len // BLOCK_TOKENS
        full_prompt_blocks += blocks
        independent_cap_blocks += (prompt_len + OUTPUT_CAP + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        for k in range(blocks):
            chunk = tuple(ids[k * BLOCK_TOKENS:(k + 1) * BLOCK_TOKENS])
            key = (parent, chunk)
            if key not in nodes:
                nodes[key] = len(nodes) + 1
            parent = nodes[key]
    unique_prompt_blocks = len(nodes)
    shared_full_prompt_blocks = full_prompt_blocks - unique_prompt_blocks
    private_tail_and_output_blocks = independent_cap_blocks - full_prompt_blocks
    return dict(
        requests=count,
        independent_full_cap_blocks=independent_cap_blocks,
        full_prompt_blocks_without_sharing=full_prompt_blocks,
        unique_full_prompt_prefix_blocks=unique_prompt_blocks,
        shared_full_prompt_blocks=shared_full_prompt_blocks,
        private_partial_tail_and_output_blocks=private_tail_and_output_blocks,
        conservative_unique_full_cap_blocks=unique_prompt_blocks + private_tail_and_output_blocks,
        fits_4096_blocks_if_all_resident=unique_prompt_blocks + private_tail_and_output_blocks <= 4096,
        resident_sequence_limit=32,
        same_time_interpretation="eligible" if count <= 32 else "aggregate only; exceeds 32 resident sequences",
    )


def build(test_jsonl: Path, source_dir: Path, metadata_dir: Path, inputs_dir: Path) -> dict:
    first16, prefix, manifest = source_data(source_dir)
    expected = manifest["official_test"]
    if (test_jsonl.stat().st_size != expected["full_source_bytes"]
            or sha(test_jsonl) != expected["full_source_sha256"]):
        raise ValueError("complete official test bytes differ from pinned source manifest")
    rows = [json.loads(line) for line in test_jsonl.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != EXPECTED_ROWS or len(rows) != expected["full_source_rows"]
            or rows[:16] != first16 or any(set(row) != {"question", "answer"} for row in rows)):
        raise ValueError("official source order, first 16, or row schema differs")
    tokenizer, metadata_hashes, tokenizer_version = tokenizer_data(metadata_dir)
    fixed_config = json.loads((inputs_dir / "config.json").read_text(encoding="utf-8"))
    fixed_workload = json.loads((inputs_dir / "workload.json").read_text(encoding="utf-8"))
    if (fixed_config["workload_sha256"] != sha(inputs_dir / "workload.json")
            or fixed_config["metadata_sha256"] != metadata_hashes
            or fixed_config["model"] != dict(id=MODEL_ID, revision=MODEL_REVISION,
                                               tokenizer_revision=MODEL_REVISION)
            or len(fixed_workload["requests"]) != 16):
        raise ValueError("frozen first-16 input or tokenizer provenance differs")

    lengths, first_prompts = [], []
    for index, row in enumerate(rows):
        content = prefix + "Question: " + row["question"].strip()
        prompt = render_prompt(content)
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if not ids:
            raise ValueError(f"empty prompt at official row {index}")
        if index < 16:
            fixed = fixed_workload["requests"][index]
            if (fixed["example_index"] != index or fixed["prompt"] != prompt
                    or fixed["prompt_token_ids"] != ids):
                raise ValueError(f"frozen prompt/token IDs differ at official row {index}")
        lengths.append(len(ids))
        if index < max(COHORTS):
            first_prompts.append(ids)

    ordered = sorted(lengths)
    per_request_cap_blocks = [(length + OUTPUT_CAP + BLOCK_TOKENS - 1) // BLOCK_TOKENS
                              for length in lengths]
    largest32_cap_blocks = sum(sorted(per_request_cap_blocks, reverse=True)[:32])
    context_violations = [i for i, length in enumerate(lengths)
                          if length + OUTPUT_CAP > CONTEXT_LIMIT]
    return dict(
        schema="c-instruct-gsm8k-corpus-geometry-v1",
        scope="CPU-only structural capacity geometry; no model outputs, arrivals, cache runtime, or GPU result",
        source=dict(url=expected["url"], commit=expected["commit"],
                    sha256=expected["full_source_sha256"], bytes=expected["full_source_bytes"],
                    rows=len(rows), source_manifest_sha256=sha(source_dir / "SOURCE_MANIFEST.json")),
        model=dict(id=MODEL_ID, revision=MODEL_REVISION, metadata_sha256=metadata_hashes,
                   tokenizers_version=tokenizer_version, fixed_first16_workload_sha256=fixed_config["workload_sha256"]),
        prompt_protocol="historical eight-shot CoT + one user chat message + assistant Answer: cue; add_special_tokens=False",
        output_cap=OUTPUT_CAP, context_limit=CONTEXT_LIMIT, block_tokens=BLOCK_TOKENS,
        prompt_tokens_source_order=lengths,
        prompt_tokens_summary=dict(min=ordered[0], median=median(ordered),
                                   p90=nearest_rank(ordered, 90), p95=nearest_rank(ordered, 95),
                                   p99=nearest_rank(ordered, 99), max=ordered[-1]),
        prompt_plus_cap_context_violations=dict(count=len(context_violations),
                                                source_indices=context_violations),
        any_32_request_capacity_bound=dict(
            largest_32_independent_full_cap_blocks=largest32_cap_blocks,
            simple_32_times_longest_request_bound=32 * max(per_request_cap_blocks),
            usable_blocks=4096,
            all_32_request_subsets_fit_without_prefix_sharing=largest32_cap_blocks <= 4096,
            scope="Known cap and prompt lengths only; all choices of at most 32 source requests, no arrival or EOS assumptions"),
        cohorts={f"first_{count}": cohort_geometry(first_prompts, lengths, count)
                 for count in COHORTS},
        accounting="Unique full-cap blocks = exact unique 16-token full prompt-prefix blocks + each request's private conservative partial-tail/output blocks; no generated-token or partial-block sharing assumed.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("/private/tmp/c-gsm8k-source-20261001"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("/private/tmp/c-olmoe-instruct-metadata-20261001"))
    parser.add_argument("--inputs-dir", type=Path, default=Path(__file__).with_name("20261001_c_instruct_gsm8k_inputs_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.test_jsonl, args.source_dir, args.metadata_dir, args.inputs_dir)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(dict(rows=result["source"]["rows"],
                          context_violations=result["prompt_plus_cap_context_violations"]["count"],
                          cohorts={key: (value["independent_full_cap_blocks"],
                                         value["conservative_unique_full_cap_blocks"])
                                   for key, value in result["cohorts"].items()})))


if __name__ == "__main__":
    main()
