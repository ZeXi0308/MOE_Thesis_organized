#!/usr/bin/env python3
"""Freeze the first 32 official geometric_shapes examples for APC qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile

REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
TOKENIZER_SHA = "a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40"
MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
PRIOR_WORKLOAD_SHA = "7f429bb80ba24b444300bf72a4f69625a320748a65ac793f58fab0dafbafeac5"
DATA_SHA = "07be68a124420caa2ebc0b2c858ad79fc55c5c8806170e496e1e17ebfb11f31c"
PROMPT_SHA = "f36d9893507eb518d84f388fb677dc0226c74a98873700de0b556548d65d3f76"
TASK, COUNT, OUTPUT_CAP, CONTEXT, BLOCK = "geometric_shapes", 32, 512, 4096, 16
EXPECTED_FULLCAP_BLOCKS = 4993


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_sha(value: object) -> str:
    return sha_bytes(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tokenizer-file", type=Path, required=True)
    parser.add_argument("--prior-workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    if not out.parent.is_dir() or out.exists() or out.is_symlink():
        raise FileExistsError("output directory must be new under an existing parent")
    data_file = args.source_root / "bbh" / f"{TASK}.json"
    prompt_file = args.source_root / "cot-prompts" / f"{TASK}.txt"
    if (sha_file(data_file) != DATA_SHA or sha_file(prompt_file) != PROMPT_SHA
            or sha_file(args.tokenizer_file) != TOKENIZER_SHA
            or sha_file(args.prior_workload) != PRIOR_WORKLOAD_SHA):
        raise ValueError("frozen official BBH, tokenizer, or first27 workload bytes differ")
    from tokenizers import Tokenizer, __version__ as tokenizers_version
    tokenizer = Tokenizer.from_file(str(args.tokenizer_file))
    examples = json.loads(data_file.read_text(encoding="utf-8"))["examples"]
    if len(examples) < COUNT:
        raise ValueError("fewer than 32 official source-ordered examples")
    with prompt_file.open(encoding="utf-8") as stream:
        prefix = "".join(stream.readlines()[2:]).strip()
    if not prefix:
        raise ValueError("empty official three-shot base prompt")
    rows = []
    for index, example in enumerate(examples[:COUNT]):
        source_input, gold = example["input"], example["target"]
        if not isinstance(source_input, str) or not isinstance(gold, str):
            raise ValueError(f"invalid official input/target at source index {index}")
        prompt = prefix + "\n\nQ: " + source_input + "\nA:"
        ids = tokenizer.encode(prompt, add_special_tokens=True).ids
        if len(ids) + OUTPUT_CAP > CONTEXT:
            raise ValueError(f"entire 32-request build rejected at index {index}: "
                             f"P={len(ids)}, P+512={len(ids) + OUTPUT_CAP}")
        rid = f"bbh/{TASK}/{index}"
        rows.append(dict(task=TASK, id=rid, request_id=rid,
            index=index, example_index=index, input=source_input, gold=gold,
            prompt=prompt, prompt_token_ids=ids, prompt_token_count=len(ids),
            prompt_sha256=sha_bytes(prompt.encode("utf-8")),
            prompt_token_ids_sha256=compact_sha(ids),
            source_input_sha256=sha_bytes(source_input.encode("utf-8")),
            source_gold_sha256=sha_bytes(gold.encode("utf-8"))))
    old = json.loads(args.prior_workload.read_text(encoding="utf-8"))
    fixed = next(row for row in old["requests"] if row["task"] == TASK)
    if not (rows[0]["prompt"] == fixed["prompt"]
            and rows[0]["prompt_token_ids"] == fixed["prompt_token_ids"]
            and rows[0]["gold"] == fixed["gold"]):
        raise ValueError("first source example differs from fixed BBH qualification input")
    fullcap_blocks = sum((len(row["prompt_token_ids"]) + OUTPUT_CAP + BLOCK - 1) // BLOCK
                         for row in rows)
    if fullcap_blocks != EXPECTED_FULLCAP_BLOCKS:
        raise ValueError(f"source-first32 fullcap blocks changed: {fullcap_blocks}")
    lengths = [row["prompt_token_count"] for row in rows]
    sampling = dict(strategy="greedy", temperature=0.0, top_p=1.0,
                    max_tokens=OUTPUT_CAP, stop=["\n\n"],
                    ignore_eos=False, min_tokens=0)
    workload = dict(schema="c-bbh-apc-workload-v1", request_count=COUNT,
        requests=rows, arrival_traces_s=[0.0] * COUNT, sampling=sampling)
    source = dict(repository="suzgunmirac/BIG-Bench-Hard", revision=REVISION,
        selection="geometric_shapes examples 0..31 in official source order",
        data_file=str(data_file.relative_to(args.source_root)), data_file_sha256=DATA_SHA,
        prompt_file=str(prompt_file.relative_to(args.source_root)),
        prompt_file_sha256=PROMPT_SHA, prior_workload_sha256=PRIOR_WORKLOAD_SHA,
        tokenizer_json_sha256=TOKENIZER_SHA)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        write_json(stage / "workload.json", workload)
        config = dict(schema="c-bbh-apc-config-v1", status="INPUTS_ONLY_UNRUN",
            purpose="development APC on/off workload diagnostic, not a new policy",
            request_count=COUNT, model=dict(id="allenai/OLMoE-1B-7B-0924",
                revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION,
                tokenizer_json_sha256=TOKENIZER_SHA, tokenizer_add_special_tokens=True),
            source=source, max_model_len=CONTEXT, max_num_batched_tokens=1024,
            max_num_seqs=COUNT, max_output_tokens=OUTPUT_CAP, sampling=sampling,
            arrival_process="all 32 external arrivals at t=0",
            prompt_tokens=dict(min=min(lengths), median=statistics.median(lengths),
                               max=max(lengths), sum=sum(lengths)),
            fullcap_reservation_blocks=fullcap_blocks,
            workload_sha256=sha_file(stage / "workload.json"),
            tokenizers_version=tokenizers_version, runtime_status="UNRUN")
        write_json(stage / "config.json", config)
        receipt = dict(schema="c-bbh-apc-source-receipt-v1", source=source,
            per_request=[dict(request_id=row["request_id"], example_index=row["example_index"],
                prompt_sha256=row["prompt_sha256"],
                prompt_token_ids_sha256=row["prompt_token_ids_sha256"],
                input_sha256=row["source_input_sha256"],
                gold_sha256=row["source_gold_sha256"]) for row in rows],
            output_files_sha256={name: sha_file(stage / name)
                                 for name in ("workload.json", "config.json")})
        write_json(stage / "SOURCE_RECEIPT.json", receipt)
        os.replace(stage, out)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(json.dumps(dict(output_dir=str(out), requests=COUNT,
        prompt_tokens_min=min(lengths), prompt_tokens_max=max(lengths),
        fullcap_reservation_blocks=fullcap_blocks,
        files_sha256={name: sha_file(out / name)
                      for name in ("workload.json", "config.json", "SOURCE_RECEIPT.json")}),
        indent=2))


if __name__ == "__main__":
    main()
