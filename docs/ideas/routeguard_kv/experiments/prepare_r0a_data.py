#!/usr/bin/env python3
"""Prepare disjoint smoke/calibration/sealed RouteGuard-KV R0-A manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from r0a_artifacts import (
    ArtifactError,
    load_config,
    ordered_hash_of_hashes,
    sha256_file,
    write_json_no_overwrite,
    write_jsonl_no_overwrite,
)


class DataPreparationError(RuntimeError):
    pass


HASH_FIELDS = ("text_sha256", "document_sha256", "canonical_text_sha256")
HEX = set("0123456789abcdef")


def canonical_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def text_sha256(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def parse_wikitext_articles(rows: Iterable[str], *, min_chars: int) -> list[str]:
    documents: list[str] = []
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal title, body
        if title is not None:
            value = canonical_text(" ".join([title, *body]).strip())
            if len(value) >= min_chars:
                documents.append(value)
        title = None
        body = []

    for raw in rows:
        line = " ".join(str(raw).split())
        is_title = line.startswith("= ") and line.endswith(" =") and not line.startswith("= =")
        if is_title:
            flush()
            title = line.strip("= ")
        elif line and title is not None:
            body.append(line)
    flush()
    return documents


def _valid_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    digest = value.lower()
    return digest if len(digest) == 64 and set(digest) <= HEX else None


def _walk_hash_fields(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in HASH_FIELDS:
                digest = _valid_digest(child)
                if digest:
                    yield digest
            yield from _walk_hash_fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_hash_fields(child)


def hashes_from_artifact(path: Path) -> set[str]:
    result: set[str] = set()
    try:
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            result.update(_walk_hash_fields(value))
            if path.name == "historical_hash_registry.json" and isinstance(value, Mapping):
                for candidate in value.get("hashes", []):
                    digest = _valid_digest(candidate)
                    if digest:
                        result.add(digest)
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    result.update(_walk_hash_fields(json.loads(line)))
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    for field in HASH_FIELDS:
                        digest = _valid_digest(row.get(field))
                        if digest:
                            result.add(digest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
        raise DataPreparationError(f"cannot audit historical artifact {path}: {exc}") from exc
    return result


def scan_historical_hashes(
    repo_root: Path, *, excluded_roots: Iterable[Path] = ()
) -> tuple[set[str], list[dict[str, Any]]]:
    hashes: set[str] = set()
    sources: list[dict[str, Any]] = []
    excluded = [path.resolve() for path in excluded_roots]
    for suffix in ("*.json", "*.jsonl", "*.csv"):
        for path in sorted(repo_root.rglob(suffix)):
            if ".git" in path.parts or ".venv" in path.parts:
                continue
            skip = False
            for excluded_root in excluded:
                try:
                    path.resolve().relative_to(excluded_root)
                    skip = True
                    break
                except ValueError:
                    pass
            if skip:
                continue
            found = hashes_from_artifact(path)
            if found:
                hashes.update(found)
                sources.append(
                    {
                        "path": str(path.relative_to(repo_root)),
                        "sha256": sha256_file(path),
                        "recognized_hash_count": len(found),
                    }
                )
    return hashes, sources


def historical_prefix_hashes(documents: list[str], *, seed: int, count: int) -> set[str]:
    if count <= 0 or count > len(documents):
        raise DataPreparationError("historical article prefix exceeds the article pool")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    return {text_sha256(text) for text in shuffled[:count]}


def selection_sha256(salt: str, digest: str) -> str:
    return hashlib.sha256(f"{salt}|{digest}".encode("utf-8")).hexdigest()


def eligible_documents(
    documents: list[str], tokenizer: Any, *, required_tokens: int, excluded_hashes: set[str]
) -> tuple[dict[str, str], dict[str, int]]:
    candidates: dict[str, str] = {}
    for text in documents:
        digest = text_sha256(text)
        if digest not in excluded_hashes:
            candidates.setdefault(digest, text)
    accepted: dict[str, str] = {}
    lengths: dict[str, int] = {}
    items = list(candidates.items())
    for start in range(0, len(items), 128):
        batch = items[start : start + 128]
        encoded = tokenizer(
            [text for _, text in batch],
            add_special_tokens=True,
            truncation=True,
            max_length=required_tokens,
            padding=False,
        )["input_ids"]
        if len(encoded) != len(batch):
            raise DataPreparationError("tokenizer batch cardinality mismatch")
        for (digest, text), token_ids in zip(batch, encoded):
            length = len(token_ids)
            if length >= required_tokens:
                accepted[digest] = text
                lengths[digest] = length
    return accepted, lengths


def _records(
    digests: list[str],
    *,
    split: str,
    texts: Mapping[str, str],
    lengths: Mapping[str, int],
    salt: str,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "routeguard-kv-document-v1",
            "split": split,
            "document_index": index,
            "text_sha256": digest,
            "selection_sha256": selection_sha256(salt, digest),
            "token_length_at_least": int(lengths[digest]),
            "text": texts[digest],
        }
        for index, digest in enumerate(digests)
    ]


def _load_cached_pinned_dataset(config: Mapping[str, Any], cache_dir: Path | None):
    from datasets import Dataset, concatenate_datasets

    data_config = config["dataset"]
    roots: list[Path] = []
    if cache_dir is not None:
        roots.append(cache_dir / "datasets")
        roots.append(cache_dir)
    if os.environ.get("HF_DATASETS_CACHE"):
        roots.append(Path(os.environ["HF_DATASETS_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "datasets")
    roots.append(Path.home() / ".cache/huggingface/datasets")
    pattern = (
        f"{data_config['repo_id']}/{data_config['config']}/**/{data_config['revision']}/"
        f"{data_config['repo_id']}-{data_config['split']}*.arrow"
    )
    for root in dict.fromkeys(roots):
        paths = sorted(root.glob(pattern))
        if paths:
            parts = [Dataset.from_file(str(path)) for path in paths]
            dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
            return dataset, paths
    raise DataPreparationError(
        "pinned dataset is unavailable offline; use --allow-download or populate the exact revision cache"
    )


def load_dataset_and_tokenizer(config: Mapping[str, Any], args: argparse.Namespace):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    local_only = not args.allow_download
    dataset_config = config["dataset"]
    model_config = config["model"]
    if local_only:
        dataset, dataset_files = _load_cached_pinned_dataset(config, args.cache_dir)
    else:
        dataset = load_dataset(
            dataset_config["repo_id"],
            dataset_config["config"],
            revision=dataset_config["revision"],
            split=dataset_config["split"],
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        dataset_files = []
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["repo_id"],
        revision=model_config["revision"],
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        local_files_only=local_only,
    )
    return dataset, tokenizer, dataset_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise DataPreparationError(f"output directory already exists: {args.output_dir}")
    config = load_config(args.config)
    dataset, tokenizer, dataset_files = load_dataset_and_tokenizer(config, args)
    data_config = config["dataset"]
    articles = parse_wikitext_articles(
        (str(row["text"]) for row in dataset), min_chars=int(data_config["minimum_characters"])
    )
    # A rerun of this exact v1 preparation must reproduce its split instead of
    # treating its own prior generated manifests as historical experiments.
    own_data_root = args.repo_root / "docs/ideas/routeguard_kv/experiments/data"
    scanned, sources = scan_historical_hashes(
        args.repo_root, excluded_roots=(args.output_dir, own_data_root)
    )
    prefix = historical_prefix_hashes(
        articles,
        seed=int(data_config["historical_article_shuffle_seed"]),
        count=int(data_config["historical_article_prefix_excluded"]),
    )
    historical = scanned | prefix
    eligible, token_lengths = eligible_documents(
        articles,
        tokenizer,
        required_tokens=int(data_config["required_tokens"]),
        excluded_hashes=historical,
    )
    counts = {
        "calibration": int(data_config["calibration_documents"]),
        "sealed": int(data_config["sealed_documents"]),
        "smoke": int(data_config["smoke_documents"]),
    }
    needed = sum(counts.values())
    if len(eligible) < needed:
        raise DataPreparationError(f"only {len(eligible)} eligible documents for {needed} slots")
    salt = str(data_config["selection_salt"])
    ordered = sorted(eligible, key=lambda digest: selection_sha256(salt, digest))[:needed]
    calibration_hashes = ordered[: counts["calibration"]]
    sealed_start = counts["calibration"]
    sealed_hashes = ordered[sealed_start : sealed_start + counts["sealed"]]
    smoke_hashes = ordered[sealed_start + counts["sealed"] :]
    if len(set(ordered)) != needed or set(ordered) & historical:
        raise AssertionError("selected split overlap or historical reuse")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifests = {
        "calibration": _records(
            calibration_hashes,
            split="calibration",
            texts=eligible,
            lengths=token_lengths,
            salt=salt,
        ),
        "sealed": _records(
            sealed_hashes, split="sealed", texts=eligible, lengths=token_lengths, salt=salt
        ),
        "smoke": _records(
            smoke_hashes, split="smoke", texts=eligible, lengths=token_lengths, salt=salt
        ),
    }
    write_jsonl_no_overwrite(args.output_dir / "calibration_manifest.jsonl", manifests["calibration"])
    write_jsonl_no_overwrite(
        args.output_dir / "sealed_manifest.jsonl", manifests["sealed"], mode=0o600
    )
    write_jsonl_no_overwrite(args.output_dir / "smoke_manifest.jsonl", manifests["smoke"])
    registry_path = args.output_dir / "historical_hash_registry.json"
    write_json_no_overwrite(
        registry_path,
        {
            "schema_version": "routeguard-kv-historical-hashes-v1",
            "recognized_fields": list(HASH_FIELDS),
            "scanned_sources": sources,
            "scanned_unique_hash_count": len(scanned),
            "shuffle_seed": data_config["historical_article_shuffle_seed"],
            "excluded_prefix_count": data_config["historical_article_prefix_excluded"],
            "prefix_hashes": sorted(prefix),
            "hashes": sorted(historical),
        },
    )
    write_json_no_overwrite(
        args.output_dir / "provenance.json",
        {
            "schema_version": "routeguard-kv-data-provenance-v1",
            "config_sha256": sha256_file(args.config),
            "dataset_repo_id": data_config["repo_id"],
            "dataset_config": data_config["config"],
            "dataset_revision": data_config["revision"],
            "dataset_split": data_config["split"],
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "dataset_cache_files": [
                {"path": str(path), "sha256": sha256_file(path)} for path in dataset_files
            ],
            "historical_hash_registry_sha256": sha256_file(registry_path),
            "tokenizer": f"{config['model']['repo_id']}@{config['model']['revision']}",
            "article_count": len(articles),
            "historical_unique_count": len(historical),
            "eligible_count": len(eligible),
            "required_tokens": data_config["required_tokens"],
            "selection_key_format": data_config["selection_key_format"],
            "ordered_hash_of_hashes": {
                split: ordered_hash_of_hashes([str(row["text_sha256"]) for row in rows])
                for split, rows in manifests.items()
            },
            "selected_token_lengths": {
                digest: token_lengths[digest] for digest in ordered
            },
        },
    )
    print(
        "DATA_PREPARATION_OK "
        f"articles={len(articles)} historical={len(historical)} eligible={len(eligible)} "
        f"calibration={len(calibration_hashes)} sealed={len(sealed_hashes)} smoke={len(smoke_hashes)}"
    )


if __name__ == "__main__":
    try:
        main()
    except (ArtifactError, DataPreparationError) as exc:
        raise SystemExit(f"DATA_PREPARATION_FAILED: {exc}") from exc
