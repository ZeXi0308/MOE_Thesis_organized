#!/usr/bin/env python3
"""Build 27 fixed BBH CoT inputs for a native workload qualification cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile

BBH_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
TOKENIZER_SHA = "a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40"
MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
MAX_MODEL_LEN = 4096
MAX_TOKENS = 512
REQUESTS = 27


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--tokenizer-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.source_root, args.output_dir
    require(root.is_dir() and (root / "bbh").is_dir() and (root / "cot-prompts").is_dir(),
            "BBH source root requires bbh/ and cot-prompts/")
    require(out.parent.is_dir() and not out.exists() and not out.is_symlink(),
            "output directory must be new under an existing parent")
    require(sha_file(args.tokenizer_file) == TOKENIZER_SHA, "pinned tokenizer.json SHA differs")
    from tokenizers import Tokenizer, __version__ as tokenizers_version

    tokenizer = Tokenizer.from_file(str(args.tokenizer_file))
    tasks = sorted(path.stem for path in (root / "bbh").glob("*.json"))
    prompt_tasks = sorted(path.stem for path in (root / "cot-prompts").glob("*.txt"))
    require(len(tasks) == REQUESTS and tasks == prompt_tasks and len(set(tasks)) == REQUESTS,
            "expected exactly 27 matching, alphabetically ordered BBH task/prompt files")
    rows, source_files, per_task, invalid = [], [], [], []
    for task in tasks:
        data_path = root / "bbh" / f"{task}.json"
        prompt_path = root / "cot-prompts" / f"{task}.txt"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        require(isinstance(data, dict) and isinstance(data.get("examples"), list)
                and data["examples"] and isinstance(data["examples"][0], dict),
                f"{task}: missing first example")
        first = data["examples"][0]
        source_input, gold = first.get("input"), first.get("target")
        require(isinstance(source_input, str) and isinstance(gold, str),
                f"{task}: first example input/target must be strings")
        with prompt_path.open(encoding="utf-8") as stream:
            prefix = "".join(stream.readlines()[2:]).strip()
        require(bool(prefix), f"{task}: empty CoT prompt after the first two lines")
        prompt = prefix + "\n\nQ: " + source_input + "\nA:"
        ids = tokenizer.encode(prompt, add_special_tokens=True).ids
        length = len(ids)
        if length + MAX_TOKENS > MAX_MODEL_LEN:
            invalid.append((task, length, length + MAX_TOKENS))
        request_id = f"bbh/{task}/0"
        rows.append(dict(task=task, id=request_id, request_id=request_id,
            index=0, example_index=0, input=source_input, gold=gold,
            prompt=prompt, prompt_token_ids=ids, prompt_token_count=length,
            prompt_sha256=sha_bytes(prompt.encode("utf-8")),
            prompt_token_ids_sha256=compact_sha(ids),
            source_input_sha256=sha_bytes(source_input.encode("utf-8")),
            source_gold_sha256=sha_bytes(gold.encode("utf-8"))))
        for path in (data_path, prompt_path):
            source_files.append(dict(path=str(path.relative_to(root)),
                bytes=path.stat().st_size, sha256=sha_file(path)))
        per_task.append(dict(task=task, example_index=0,
            data_file=str(data_path.relative_to(root)), data_file_sha256=sha_file(data_path),
            prompt_file=str(prompt_path.relative_to(root)),
            prompt_file_sha256=sha_file(prompt_path),
            input_sha256=rows[-1]["source_input_sha256"],
            gold_sha256=rows[-1]["source_gold_sha256"],
            assembled_prompt_sha256=rows[-1]["prompt_sha256"],
            prompt_token_ids_sha256=rows[-1]["prompt_token_ids_sha256"],
            prompt_tokens=length))
    require(not invalid, f"entire 27-task build rejected: P+512 exceeds 4096: {invalid}")
    source_files.sort(key=lambda row: row["path"])
    lengths = [row["prompt_token_count"] for row in rows]
    sampling = dict(strategy="greedy", temperature=0.0, top_p=1.0,
                    max_tokens=MAX_TOKENS, stop=["\n\n"],
                    ignore_eos=False, min_tokens=0)
    workload = dict(schema="c-bbh-qualification-workload-v1",
        request_count=REQUESTS, requests=rows, arrival_traces_s=[0.0] * REQUESTS,
        sampling=sampling)
    source = dict(repository="suzgunmirac/BIG-Bench-Hard", revision=BBH_REVISION,
        selected_rule="27 task names sorted alphabetically; first example index 0 in each",
        prompt_rule="''.join(prompt_file.readlines()[2:]).strip() + '\\n\\nQ: ' + input + '\\nA:'",
        files=source_files, files_sha256=compact_sha(source_files))
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        write_json(stage / "workload.json", workload)
        config = dict(schema="c-bbh-qualification-config-v1", status="INPUTS_ONLY_UNRUN",
            purpose="workload qualification, not a policy comparison",
            request_count=REQUESTS, model=dict(id="allenai/OLMoE-1B-7B-0924",
                revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION,
                tokenizer_json_sha256=TOKENIZER_SHA,
                tokenizer_add_special_tokens=True),
            source=dict(repository=source["repository"], revision=BBH_REVISION,
                        files_sha256=source["files_sha256"]),
            max_model_len=MAX_MODEL_LEN, max_num_batched_tokens=1024,
            prompt_tokens=dict(min=min(lengths), max=max(lengths),
                median=statistics.median(lengths), sum=sum(lengths)),
            max_prompt_plus_output=max(lengths) + MAX_TOKENS,
            sampling=sampling, arrival_process="all 27 external arrivals at t=0",
            workload_sha256=sha_file(stage / "workload.json"),
            tokenizers_version=tokenizers_version, runtime_status="UNRUN")
        write_json(stage / "config.json", config)
        receipt = dict(schema="c-bbh-qualification-source-receipt-v1",
            source=source, per_task=per_task,
            source_files_count=len(source_files), request_count=REQUESTS,
            prompt_sha256_by_task={row["task"]: row["prompt_sha256"] for row in rows},
            output_files_sha256={name: sha_file(stage / name)
                                 for name in ("workload.json", "config.json")})
        write_json(stage / "SOURCE_RECEIPT.json", receipt)
        os.replace(stage, out)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(json.dumps(dict(output_dir=str(out), requests=REQUESTS,
        prompt_tokens_min=min(lengths), prompt_tokens_max=max(lengths),
        max_prompt_plus_output=max(lengths) + MAX_TOKENS,
        files_sha256={name: sha_file(out / name)
                      for name in ("workload.json", "config.json", "SOURCE_RECEIPT.json")}),
        indent=2))


if __name__ == "__main__":
    main()
