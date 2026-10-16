#!/usr/bin/env python3
"""Prepare pinned ShareGPT first-human text as Granite base-model continuation.

Only existing local files are read. No model weights are instantiated, no GPU
is initialized and no downloads are allowed. The base checkpoint has no chat
template; raw human text is encoded without chat delimiters or generation tags.
This is a systems continuation workload, not an instruction-following evaluation.

  python code/prepare_granite_workload.py --output configs/sharegpt_granite1b.json
  python code/prepare_granite_workload.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import unittest

SOURCE = "/root/autodl-tmp/research_C_format_sched/datasets/serving_sources/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
SOURCE_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
SOURCE_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "granite1b_ngram_reduced.json"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_model(config: dict) -> dict:
    manifest_path = Path(config["model_source_manifest"])
    if sha256(manifest_path) != config["model_source_manifest_sha256"]:
        raise ValueError("existing model-source manifest differs from the checked complete manifest")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("repo") != config["model_repo"] or manifest.get("revision") != config["model_revision"]
            or manifest.get("status") != "COMPLETE"):
        raise ValueError("wrong repository/revision or incomplete local model")
    model = Path(config["model"])
    checked = {}
    for name, expected in config["model_artifacts"].items():
        path = model / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"missing/incomplete existing model file: {path}")
        digest = sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"SHA256 mismatch in existing model file: {path}")
        checked[name] = {"bytes": path.stat().st_size, "sha256": digest}
    architecture = json.loads((model / "config.json").read_text())
    expected = config["architecture"]
    if architecture["architectures"] != [expected["name"]]:
        raise ValueError("unexpected architecture")
    for key, field in (("num_local_experts", "experts"), ("num_experts_per_tok", "top_k"),
                       ("hidden_size", "hidden_size"), ("intermediate_size", "intermediate_size"),
                       ("num_hidden_layers", "layers"), ("vocab_size", "vocab_size")):
        if architecture[key] != expected[field]:
            raise ValueError(f"unexpected {key}")
    return checked


def select_raw_records(data: list, tokenizer, count: int, minimum: int, maximum: int) -> list[dict]:
    if count <= 0 or not 0 < minimum <= maximum:
        raise ValueError("invalid count or prompt token range")
    if getattr(tokenizer, "chat_template", None):
        raise ValueError("pinned base tokenizer unexpectedly gained a chat template")
    records = []
    for index, item in enumerate(data):
        turns = item.get("conversations", [])
        if len(turns) < 2 or turns[0].get("from") not in ("human", "user"):
            continue
        text = turns[0].get("value", "")
        if not isinstance(text, str) or not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not minimum <= len(ids) <= maximum:
            continue
        records.append({"source_index": index, "source_id": item.get("id"),
                        "prompt": text, "prompt_token_ids": ids,
                        "rendered_sha256": hashlib.sha256(text.encode()).hexdigest()})
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError(f"only {len(records)} eligible prompts, required {count}")
    return records


def self_test() -> None:
    class FakeTokenizer:
        chat_template = None
        def encode(self, text, add_special_tokens=True):
            if add_special_tokens:
                raise AssertionError("raw continuation must not insert special/chat tokens")
            return [ord(letter) for letter in text]
    class PreparationTests(unittest.TestCase):
        def test_raw_order_and_no_truncation(self):
            data = [{"id": str(i), "conversations": [{"from": "human", "value": text},
                     {"from": "gpt", "value": "unused"}]} for i, text in enumerate(["a", "abc", "abcd", "abcde"])]
            rows = select_raw_records(data, FakeTokenizer(), 2, 3, 4)
            self.assertEqual([r["source_index"] for r in rows], [1, 2])
            self.assertEqual(rows[0]["prompt_token_ids"], [97, 98, 99])
        def test_chat_template_is_not_silently_applied(self):
            tokenizer = FakeTokenizer()
            tokenizer.chat_template = "new template"
            with self.assertRaises(ValueError):
                select_raw_records([], tokenizer, 1, 1, 2)
        def test_incomplete_population_fails(self):
            with self.assertRaises(ValueError):
                select_raw_records([], FakeTokenizer(), 160, 32, 256)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreparationTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output")
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--min-prompt", type=int, default=32)
    parser.add_argument("--max-prompt", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output:
        parser.error("--output is required")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(Path(args.config).read_text())
    verified = verify_model(config)
    source = Path(args.source)
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("ShareGPT source differs from the pinned revision/content SHA256")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True, trust_remote_code=False)
    with source.open() as stream:
        records = select_raw_records(json.load(stream), tokenizer, args.count, args.min_prompt, args.max_prompt)
    payload = {"dataset": "ShareGPT_V3_unfiltered_cleaned_split", "source": str(source),
        "source_sha256": SOURCE_SHA256, "source_revision": SOURCE_REVISION,
        "model": config["model"], "model_repo": config["model_repo"], "model_revision": config["model_revision"],
        "verified_model_files": verified, "model_source_manifest_sha256": config["model_source_manifest_sha256"],
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_chat_template": None,
        "prompt_format": "raw_continuation_no_chat_template", "add_special_tokens": False,
        "selection": "first eligible first-human turn in original order; model-tokenizer length filter; no truncation",
        "prompt_range": [args.min_prompt, args.max_prompt],
        "scope": "base-model continuation systems workload; not an instruction-following quality evaluation; tokenization may select a different prompt subset than other models",
        "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "requests": len(records), "sha256": sha256(output),
                      "prompt_tokens_min": min(len(r["prompt_token_ids"]) for r in records),
                      "prompt_tokens_max": max(len(r["prompt_token_ids"]) for r in records),
                      "chat_template": None, "model_revision": config["model_revision"]}))


if __name__ == "__main__":
    main()
