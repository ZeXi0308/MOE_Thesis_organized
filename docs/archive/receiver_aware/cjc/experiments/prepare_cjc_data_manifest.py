#!/usr/bin/env python3
"""Freeze the CJC/JouleQueue WikiText prompt manifest before route capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


MODEL_REVISIONS = {
    "olmoe": (
        "allenai/OLMoE-1B-7B-0924",
        "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    ),
    "llm_jp": (
        "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M",
        "1d5983076dfc67aee4a77ec06a27027f5bab6055",
    ),
}

WINDOWS = {
    "calibration": (20_000, 22_000, 64),
    "sealed": (40_000, 44_000, 128),
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(WINDOWS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def _load_dataset_rows(args: argparse.Namespace) -> list[str]:
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:  # pragma: no cover - environment capability
        raise RuntimeError("datasets is required to freeze the prompt manifest") from exc
    start, stop, _ = WINDOWS[args.split]
    download = DownloadConfig(local_files_only=not args.allow_download)
    dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split=f"train[{start}:{stop}]",
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        download_config=download,
    )
    return [str(row["text"]) for row in dataset]


def _load_tokenizers(args: argparse.Namespace) -> Mapping[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment capability
        raise RuntimeError("transformers is required to validate prompt lengths") from exc
    return {
        key: AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=not args.allow_download,
        )
        for key, (model_id, revision) in MODEL_REVISIONS.items()
    }


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    start, stop, count = WINDOWS[args.split]
    rows = _load_dataset_rows(args)
    if len(rows) != stop - start:
        raise RuntimeError("dataset slice length drifted from the frozen candidate window")
    tokenizers = _load_tokenizers(args)
    candidates: list[dict[str, object]] = []
    for offset, text in enumerate(rows):
        text_sha = sha256_bytes(text.encode("utf-8"))
        lengths = {
            key: len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for key, tokenizer in tokenizers.items()
        }
        if min(lengths.values()) < 129:
            continue
        rank = sha256_bytes(f"{args.seed}:{text_sha}".encode("utf-8"))
        candidates.append(
            {
                "rank_sha256": rank,
                "source_row": start + offset,
                "text_sha256": text_sha,
                "token_lengths": lengths,
                "text": text,
            }
        )
    selected = sorted(candidates, key=lambda row: str(row["rank_sha256"]))[:count]
    if len(selected) != count:
        raise RuntimeError(
            f"frozen window has only {len(selected)} dual-tokenizer-valid rows; "
            f"required {count}; offsets may not be changed"
        )
    requests = []
    for index, row in enumerate(selected):
        requests.append(
            {
                **row,
                "request_id": f"{args.split}-{index:04d}-{str(row['text_sha256'])[:12]}",
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": "wikitext/wikitext-103-raw-v1",
        "dataset_split": "train",
        "protocol_split": args.split,
        "candidate_window": [start, stop],
        "selection": "smallest_sha256(seed:text_sha256)",
        "seed": args.seed,
        "sequence_tokens": 128,
        "model_revisions": {
            key: f"{model_id}@{revision}"
            for key, (model_id, revision) in MODEL_REVISIONS.items()
        },
        "requests": requests,
    }
    payload["manifest_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    main()

