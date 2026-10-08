#!/usr/bin/env python3
"""Freeze 16 official GSM8K test prompts for a C-line native Instruct pilot.

This is an input builder. It does not download a model, initialize CUDA, run
inference, select questions by gold answers, or claim a GSM8K benchmark score.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


SOURCE_MANIFEST_SHA256 = "a50577e10038fa60518820eb63bbf03926765443f447b2c9eaedb5399fbd6e23"
MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
MODEL_REVISION = "7f1c97f440f06ce36705e4f2b843edb5925f4498"
REQUESTS = 16
OUTPUT_CAP = 1024
CONTEXT_LIMIT = 4096
EXPECTED_CHAT_TEMPLATE = (
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def source_data(source_dir: Path) -> tuple[list[dict], str, dict]:
    manifest_path = source_dir / "SOURCE_MANIFEST.json"
    if sha(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("GSM8K source manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "c-gsm8k-source-v1":
        raise ValueError("GSM8K source schema changed")
    test = manifest["official_test"]
    history = manifest["historical_open_instruct"]
    if (test["repository"] != "openai/grade-school-math"
            or test["commit"] != "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc"
            or history["commit"] != "4c0e9e803616661f1188372acaceecf0f03870dd"):
        raise ValueError("GSM8K source commit changed")
    if sha(source_dir / test["local_slice"]) != test["local_slice_sha256"]:
        raise ValueError("official test slice changed")
    for record in history["files"].values():
        if sha(source_dir / record["local"]) != record["sha256"]:
            raise ValueError("historical evaluator source changed")
    if sha(source_dir / history["eight_shot_prefix"]["local"]) != history["eight_shot_prefix"]["sha256"]:
        raise ValueError("eight-shot prefix changed")
    rows = [json.loads(line) for line in
            (source_dir / test["local_slice"]).read_text(encoding="utf-8").splitlines()]
    if len(rows) != REQUESTS or any(set(row) != {"question", "answer"} for row in rows):
        raise ValueError("expected the first 16 unmodified official test rows")
    gold = [re.sub(r"(\d),(\d)", r"\1\2",
                   row["answer"].split("####")[1].strip()) for row in rows]
    if gold != test["first16_numeric_gold_after_comma_normalization"]:
        raise ValueError("source-order gold values changed")
    exemplar_source = (source_dir / "examplars.py").read_text(encoding="utf-8")
    parsed = ast.parse(exemplar_source)
    exemplars = next(ast.literal_eval(node.value) for node in parsed.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "EXAMPLARS"
                             for target in node.targets))
    if len(exemplars) != 8:
        raise ValueError("historical CoT exemplar count changed")
    rebuilt = ("Answer the following questions.\n\n"
               + "\n\n".join("Question: " + item["question"] + "\nAnswer: "
                              + item["cot_answer"] for item in exemplars) + "\n\n")
    original = (source_dir / "eight_shot_prefix.txt").read_text(encoding="utf-8")
    if rebuilt != original:
        raise ValueError("rendered eight-shot prefix differs from historical source")
    return rows, original, manifest


def tokenizer_data(metadata_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers import __version__ as tokenizers_version

    needed = ("config.json", "tokenizer_config.json", "tokenizer.json")
    if any(not (metadata_dir / name).is_file() for name in needed):
        raise FileNotFoundError("pinned config.json, tokenizer_config.json, tokenizer.json required")
    model_config = json.loads((metadata_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads(
        (metadata_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    if (model_config.get("architectures") != ["OlmoeForCausalLM"]
            or model_config.get("max_position_embeddings") != CONTEXT_LIMIT
            or model_config.get("eos_token_id") != 50279
            or tokenizer_config.get("bos_token") != "<|endoftext|>"
            or tokenizer_config.get("eos_token") != "<|endoftext|>"
            or tokenizer_config.get("chat_template") != EXPECTED_CHAT_TEMPLATE
            or tokenizer_config.get("add_bos_token") is not False
            or tokenizer_config.get("add_eos_token") is not False):
        raise ValueError("pinned OLMoE tokenizer/config contract changed")
    tokenizer = Tokenizer.from_file(str(metadata_dir / "tokenizer.json"))
    if tokenizer.token_to_id("<|endoftext|>") != 50279:
        raise ValueError("tokenizer BOS/EOS ID differs from model config")
    hashes = {name: sha(metadata_dir / name) for name in needed}
    special = metadata_dir / "special_tokens_map.json"
    if special.is_file():
        mapping = json.loads(special.read_text(encoding="utf-8"))
        bos = mapping.get("bos_token")
        eos = mapping.get("eos_token")
        if (not isinstance(bos, dict) or bos.get("content") != "<|endoftext|>"
                or not isinstance(eos, dict) or eos.get("content") != "<|endoftext|>"):
            raise ValueError("special token map differs")
        hashes[special.name] = sha(special)
    return tokenizer, hashes, tokenizers_version


def render_prompt(user_content: str) -> str:
    # For this exact asserted Jinja template and its one-user-message input,
    # Transformers' trim_blocks/lstrip_blocks render to these literal bytes.
    # The historical GSM evaluator then appends "Answer:" because the rendered
    # text ends in a newline.
    rendered = "<|endoftext|><|user|>\n" + user_content + "\n<|assistant|>\n"
    if not rendered.endswith("\n"):
        raise AssertionError("chat render lost the assistant newline")
    return rendered + "Answer:"


def build(source_dir: Path, metadata_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"immutable input directory already exists: {output_dir}")
    rows, prefix, manifest = source_data(source_dir)
    tokenizer, metadata_hashes, tokenizer_version = tokenizer_data(metadata_dir)
    requests = []
    for index, row in enumerate(rows):
        user_content = prefix + "Question: " + row["question"].strip()
        prompt = render_prompt(user_content)
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if (not ids or any(type(token) is not int or token < 0 for token in ids)
                or len(ids) + OUTPUT_CAP > CONTEXT_LIMIT):
            raise ValueError(f"test index {index} exceeds context or has invalid token IDs")
        gold = re.sub(r"(\d),(\d)", r"\1\2",
                      row["answer"].split("####")[1].strip())
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", gold):
            raise ValueError(f"test index {index} has nonnumeric final gold")
        requests.append(dict(
            request_id=f"gsm8k-test-{index:04d}", task="gsm8k",
            example_index=index, question=row["question"],
            chat_user_content=user_content, prompt=prompt,
            prompt_token_ids=ids, prompt_token_ids_sha256=compact_hash(ids),
            gold=gold,
        ))
    workload = dict(
        schema="c-instruct-gsm8k-qualification-workload-v1",
        task="gsm8k", requests=requests, arrival_traces_s=[0.0] * REQUESTS,
        sampling=dict(temperature=0.0, max_tokens=OUTPUT_CAP, min_tokens=0,
                      ignore_eos=False, stop=[]),
        selection="official test.jsonl indices 0..15 in file order; no answer-based selection",
        prompt_source="historical Open-Instruct eight-shot CoT + model chat template",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".building-",
                                    dir=output_dir.parent))
    try:
        write_json(staging / "workload.json", workload)
        config = dict(
            schema="c-instruct-gsm8k-qualification-config-v1",
            workload_sha256=sha(staging / "workload.json"),
            model=dict(id=MODEL_ID, revision=MODEL_REVISION,
                       tokenizer_revision=MODEL_REVISION),
            requests=REQUESTS, output_tokens=OUTPUT_CAP, min_tokens=0,
            ignore_eos=False, max_model_len=CONTEXT_LIMIT,
            metadata_sha256=metadata_hashes,
            scope="single native same-architecture useful-output qualification, not GSM8K benchmark",
        )
        write_json(staging / "config.json", config)
        receipt = dict(
            schema="c-instruct-gsm8k-source-receipt-v1",
            status="INPUTS_FROZEN_CPU_ONLY",
            model=config["model"],
            builder_sha256=sha(Path(__file__)),
            source_manifest_sha256=SOURCE_MANIFEST_SHA256,
            full_official_test_sha256=manifest["official_test"]["full_source_sha256"],
            first16_source_sha256=sha(source_dir / "test-first16.jsonl"),
            historical_exemplar_prefix_sha256=sha(source_dir / "eight_shot_prefix.txt"),
            historical_open_instruct_commit=manifest["historical_open_instruct"]["commit"],
            historical_olmoe_driver_commit=manifest["olmoe_driver"]["commit"],
            local_metadata_sha256=metadata_hashes,
            local_tokenizers_version=tokenizer_version,
            workload_sha256=sha(staging / "workload.json"),
            config_sha256=sha(staging / "config.json"),
            prompt_token_counts=[len(row["prompt_token_ids"]) for row in requests],
            max_prompt_plus_cap=max(len(row["prompt_token_ids"]) + OUTPUT_CAP
                                    for row in requests),
            runtime_tokenizer_check="REQUIRED_BEFORE_INFERENCE: AutoTokenizer.apply_chat_template text and encode(add_special_tokens=False) IDs must equal every frozen row",
            evidence_scope="16 first source-order GSM8K test rows; eight fixed CoT exemplars; EOS-only 1024 cap; not the authors' random 200 / 512-token evaluation",
        )
        write_json(staging / "SOURCE_RECEIPT.json", receipt)
        staging.rename(output_dir)
        return receipt
    except BaseException:
        shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=Path("/private/tmp/c-gsm8k-source-20261001"))
    parser.add_argument("--metadata-dir", type=Path,
                        default=Path("/private/tmp/c-olmoe-instruct-metadata-20261001"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_dir, args.metadata_dir, args.output_dir)
    print(json.dumps(dict(status=result["status"],
                          requests=len(result["prompt_token_counts"]),
                          min_prompt_tokens=min(result["prompt_token_counts"]),
                          max_prompt_tokens=max(result["prompt_token_counts"]),
                          max_prompt_plus_cap=result["max_prompt_plus_cap"],
                          workload_sha256=result["workload_sha256"])))


if __name__ == "__main__":
    main()
