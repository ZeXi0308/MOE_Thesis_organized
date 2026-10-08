#!/usr/bin/env python3
"""Build all 150 source-order LongBench MultiFieldQA-en inputs; CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import string


BASE = Path(__file__).resolve().parent
CODE_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
DATA_COMMIT = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
DATA_ZIP_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
MODEL_REVISION = "7f1c97f440f06ce36705e4f2b843edb5925f4498"
TASK = "multifieldqa_en"
ROWS = 150
SOURCE_ROWS = 150
RAW_PROMPT_BUDGET = 3500
OUTPUT_CAP = 64
CONTEXT_LIMIT = 4096
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}\n"
    "{% if message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] }}\n"
    "{% elif message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return digest(path.read_bytes())


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def normalize_answer(value: str) -> str:
    """Official LongBench English QA normalization from metrics.py."""
    text = "".join(char for char in value.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def present(answer: str, text: str, *, normalized: bool) -> bool:
    needle = normalize_answer(answer) if normalized else answer
    haystack = normalize_answer(text) if normalized else text
    return bool(needle) and needle in haystack


def official_source(code_dir: Path) -> tuple[str, dict]:
    receipt_path = code_dir / "SOURCE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("repository") != "THUDM/LongBench"
            or receipt.get("commit") != CODE_COMMIT
            or len(receipt.get("files", [])) != 6):
        raise ValueError("official LongBench code receipt differs")
    for record in receipt["files"]:
        path = code_dir / record["path"]
        if not path.is_file() or sha(path) != record["sha256"]:
            raise ValueError(f"official source differs: {record['path']}")
    config = code_dir / "LongBench/config"
    prompt = json.loads((config / "dataset2prompt.json").read_text(encoding="utf-8"))[TASK]
    max_gen = json.loads((config / "dataset2maxlen.json").read_text(encoding="utf-8"))[TASK]
    four_k_budget = json.loads((config / "model2maxlen.json").read_text(encoding="utf-8"))[
        "llama2-7b-chat-4k"]
    expected = ("Read the following text and answer briefly.\n\n{context}\n\n"
                "Now, answer the following question based on the above text, "
                "only give me the answer and do not output any other words.\n\n"
                "Question: {input}\nAnswer:")
    if prompt != expected or max_gen != OUTPUT_CAP or four_k_budget != RAW_PROMPT_BUDGET:
        raise ValueError("official prompt/output/4k budget contract differs")
    return prompt, dict(commit=CODE_COMMIT, receipt_sha256=sha(receipt_path),
                        files_sha256={record["path"]: record["sha256"]
                                      for record in receipt["files"]})


def model_tokenizer(metadata_dir: Path):
    from tokenizers import Tokenizer, __version__ as tokenizers_version

    receipt = json.loads((metadata_dir / "metadata-receipt.json").read_text(encoding="utf-8"))
    if receipt.get("revision") != MODEL_REVISION:
        raise ValueError("OLMoE metadata revision differs")
    hashes = {record["filename"]: record["sha256"] for record in receipt["files"]}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if sha(metadata_dir / name) != hashes.get(name):
            raise ValueError(f"OLMoE metadata differs: {name}")
    model = json.loads((metadata_dir / "config.json").read_text(encoding="utf-8"))
    config = json.loads((metadata_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    if (model.get("max_position_embeddings") != CONTEXT_LIMIT
            or model.get("eos_token_id") != 50279
            or config.get("chat_template") != CHAT_TEMPLATE
            or config.get("bos_token") != "<|endoftext|>"
            or config.get("eos_token") != "<|endoftext|>"
            or config.get("add_bos_token") is not False
            or config.get("add_eos_token") is not False):
        raise ValueError("OLMoE context/chat/special-token config differs")
    tokenizer_path = metadata_dir / "tokenizer.json"
    json_tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    post = json_tokenizer.get("post_processor", {})
    if (post.get("type") != "TemplateProcessing"
            or post.get("single") != [{"Sequence": {"id": "A", "type_id": 0}}]
            or post.get("special_tokens") != {}):
        raise ValueError("OLMoE default add_special_tokens behavior differs")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.token_to_id("<|endoftext|>") != 50279:
        raise ValueError("OLMoE BOS/EOS tokenizer ID differs")
    return tokenizer, {name: hashes[name] for name in
                       ("config.json", "tokenizer_config.json", "tokenizer.json")}, tokenizers_version


def render_chat(user: str) -> str:
    # Exact one-user/generation-prefix render of the pinned OLMoE template.
    return "<|endoftext|><|user|>\n" + user + "\n<|assistant|>\n"


def build(source_jsonl: Path, code_dir: Path, metadata_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"immutable output directory exists: {output_dir}")
    if source_jsonl.name != "multifieldqa_en.jsonl":
        raise ValueError("expected official multifieldqa_en.jsonl filename")
    source_bytes = source_jsonl.read_bytes()
    lines = source_bytes.splitlines(keepends=True)
    if len(lines) != SOURCE_ROWS:
        raise ValueError(f"expected {SOURCE_ROWS} original source lines")
    prompt_format, official = official_source(code_dir)
    tokenizer, metadata_hashes, tokenizer_version = model_tokenizer(metadata_dir)
    requests = []
    for index, line in enumerate(lines[:ROWS]):
        row = json.loads(line)
        context, question, answers = row.get("context"), row.get("input"), row.get("answers")
        if (row.get("dataset") != TASK or not isinstance(context, str) or not context
                or not isinstance(question, str) or not question
                or not isinstance(answers, list) or not answers
                or any(not isinstance(answer, str) for answer in answers)
                or not isinstance(row.get("_id"), str)):
            raise ValueError(f"invalid official source row {index}")
        raw_prompt = prompt_format.format(context=context, input=question)
        raw_ids = tokenizer.encode(raw_prompt, add_special_tokens=True).ids
        if raw_ids != tokenizer.encode(raw_prompt, add_special_tokens=False).ids:
            raise ValueError(f"default raw-prompt special tokens differ at row {index}")
        truncated = len(raw_ids) > RAW_PROMPT_BUDGET
        user = (tokenizer.decode(raw_ids[:RAW_PROMPT_BUDGET // 2], skip_special_tokens=True)
                + tokenizer.decode(raw_ids[-RAW_PROMPT_BUDGET // 2:], skip_special_tokens=True)
                if truncated else raw_prompt)
        if not user.endswith("Answer:"):
            raise ValueError(f"LongBench answer cue lost at row {index}")
        prompt = render_chat(user)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if not prompt_ids or len(prompt_ids) + OUTPUT_CAP > CONTEXT_LIMIT:
            raise ValueError(f"final chat prompt exceeds context at row {index}")
        diagnostic = [dict(answer=answer,
            literal_in_original_context=present(answer, context, normalized=False),
            literal_in_truncated_user=present(answer, user, normalized=False),
            normalized_in_original_context=present(answer, context, normalized=True),
            normalized_in_truncated_user=present(answer, user, normalized=True))
            for answer in answers]
        requests.append(dict(
            request_id=f"longbench-multifieldqa-en-test-{index:04d}",
            task=TASK, source_index=index, example_index=index, source_id=row["_id"],
            source_line_sha256=digest(line), context_sha256=digest(context.encode("utf-8")),
            question=question, answers=answers, raw_prompt=raw_prompt,
            chat_user_content=user, prompt=prompt, prompt_token_ids=prompt_ids,
            prompt_token_ids_sha256=digest(json.dumps(prompt_ids, separators=(",", ":")).encode()),
            truncation=dict(applied=truncated, raw_prompt_tokens=len(raw_ids),
                            retained_raw_token_budget=RAW_PROMPT_BUDGET if truncated else len(raw_ids),
                            decoded_user_tokens=len(tokenizer.encode(user, add_special_tokens=True).ids),
                            final_chat_prompt_tokens=len(prompt_ids)),
            answer_surface_diagnostic=diagnostic,
        ))
    workload = dict(schema="c-longbench-multifieldqa-en-full-workload-v1", task=TASK,
        requests=requests, arrival_traces_s=[0.0] * ROWS,
        sampling=dict(temperature=0.0, max_tokens=OUTPUT_CAP, min_tokens=0,
                      ignore_eos=False, stop=[]),
        selection="all 150 original JSONL source indices 0..149; no row filtering",
        answer_surface_semantics="Weak literal/normalized substring diagnostic only; neither presence nor absence proves evidence retention; never used for selection")
    block_bounds = [(len(row["prompt_token_ids"]) + OUTPUT_CAP + 15) // 16 for row in requests]
    geometry = dict(schema="c-longbench-multifieldqa-en-full-geometry-v1",
        source_rows_total=SOURCE_ROWS, selected_source_indices=list(range(ROWS)),
        max_model_len=CONTEXT_LIMIT, max_generated_tokens=OUTPUT_CAP,
        raw_prompt_budget=RAW_PROMPT_BUDGET, block_tokens=16,
        prompt_tokens=[len(row["prompt_token_ids"]) for row in requests],
        raw_prompt_tokens=[row["truncation"]["raw_prompt_tokens"] for row in requests],
        truncated_count=sum(row["truncation"]["applied"] for row in requests),
        max_prompt_plus_cap=max(len(row["prompt_token_ids"]) + OUTPUT_CAP for row in requests),
        independent_full_cap_blocks=sum(block_bounds),
        scope="CPU-only independent request upper bound; not actual shared KV allocation, pressure, or outcome")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    write_json(output_dir / "workload.json", workload)
    config = dict(schema="c-longbench-multifieldqa-en-full-config-v1",
        workload_sha256=sha(output_dir / "workload.json"), task=TASK,
        model=dict(id=MODEL_ID, revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION),
        requests=ROWS, output_tokens=OUTPUT_CAP, max_model_len=CONTEXT_LIMIT,
        raw_prompt_max_length=RAW_PROMPT_BUDGET, metadata_sha256=metadata_hashes,
        proposed_runtime=dict(status="NOT_FROZEN_OR_EXECUTED", seed=20260905,
                              max_num_seqs=128, max_num_batched_tokens=1024,
                              usable_kv_blocks=4096, prefix_caching=True),
        scope="all 150 source-order task rows, CPU input construction; no GPU cell or LongBench score")
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "geometry.json", geometry)
    receipt = dict(schema="c-longbench-multifieldqa-en-full-source-receipt-v1",
        status="INPUTS_BUILT_CPU_ONLY", builder_sha256=sha(Path(__file__)),
        official_code=official, official_data_revision=DATA_COMMIT,
        official_data_zip_sha256=DATA_ZIP_SHA256,
        official_zip_verification="NOT_CHECKED_BY_THIS_BUILDER; verify source preparation separately",
        source_jsonl_path=str(source_jsonl.resolve()), source_jsonl_sha256=digest(source_bytes),
        source_rows_total=SOURCE_ROWS, selected_indices=list(range(ROWS)),
        model_revision=MODEL_REVISION, metadata_sha256=metadata_hashes,
        tokenizers_version=tokenizer_version,
        raw_prompt_default_add_special_tokens="True; identical IDs to False under pinned OLMoE post_processor",
        runtime_tokenizer_check="Before inference, compare AutoTokenizer raw-token IDs and apply_chat_template/tokenize IDs against every frozen row",
        workload_sha256=sha(output_dir / "workload.json"),
        config_sha256=sha(output_dir / "config.json"),
        geometry_sha256=sha(output_dir / "geometry.json"))
    write_json(output_dir / "SOURCE_RECEIPT.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--official-code-dir", type=Path,
                        default=BASE / "20261001_c_longbench_official_code_v1")
    parser.add_argument("--metadata-dir", type=Path,
                        default=BASE / "20261001_c_instruct_model_metadata_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.source_jsonl, args.official_code_dir,
                    args.metadata_dir, args.output_dir)
    print(json.dumps(dict(status=receipt["status"], rows=ROWS,
                          source_jsonl_sha256=receipt["source_jsonl_sha256"],
                          workload_sha256=receipt["workload_sha256"])))


if __name__ == "__main__":
    main()
