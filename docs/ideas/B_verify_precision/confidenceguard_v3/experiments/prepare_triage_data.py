#!/usr/bin/env python3
"""Freeze dual-tokenizer-valid TriageAudit article manifests.

The historical registry conservatively excludes the first 1,000 articles in
the exact ``random.Random(20260720).shuffle`` ordering used by earlier
WikiText-103 article experiments. Known prior ranges end at 728; the larger
prefix prevents accidental reuse without pretending that the corpus was new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Mapping

from triage_artifacts import write_json_no_overwrite
from triage_manifest import select_documents, write_jsonl_no_overwrite


class DataPreparationError(RuntimeError):
    pass


KNOWN_HISTORICAL_RANGES = (
    "quality_isolation_seed20260720_train_articles_0_160",
    "receiver_progressive_seed20260720_train_articles_160_184",
    "decode_fragility_candidate_window_seed20260720_train_articles_184_328",
    "expert_persistence_candidate_window_seed20260720_train_articles_500_596",
    "idea_a_seed20260720_train_articles_600_728",
)


def parse_wikitext_articles(rows: Iterable[str], *, min_chars: int = 500) -> list[str]:
    documents: list[str] = []
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal title, body
        if title is not None:
            text = " ".join([title, *body]).strip()
            if len(text) >= min_chars:
                documents.append(text)
        title = None
        body = []

    for raw in rows:
        text = " ".join(str(raw).split())
        is_top_level_title = text.startswith("= ") and text.endswith(" =") and not text.startswith("= =")
        if is_top_level_title:
            flush()
            title = text.strip("= ")
        elif text and title is not None:
            body.append(text)
    flush()
    return documents


def historical_prefix_hashes(documents: list[str], *, seed: int, count: int) -> list[str]:
    if count <= 0 or len(documents) < count:
        raise DataPreparationError("historical prefix exceeds the article pool")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    return sorted(hashlib.sha256(text.encode("utf-8")).hexdigest() for text in shuffled[:count])


def dual_tokenizer_filter(
    documents: Iterable[str],
    tokenizers: Mapping[str, object],
    *,
    required_tokens: int,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    texts = list(documents)
    digests = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    lengths: dict[str, dict[str, int]] = {digest: {} for digest in digests}
    batch_size = 256
    for key, tokenizer in tokenizers.items():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                truncation=True,
                max_length=required_tokens,
                padding=False,
            )["input_ids"]
            if len(encoded) != len(batch):
                raise DataPreparationError("tokenizer batch cardinality mismatch")
            for digest, token_ids in zip(digests[start : start + batch_size], encoded):
                lengths[digest][key] = len(token_ids)
    accepted = [
        text for text, digest in zip(texts, digests)
        if len(lengths[digest]) == len(tokenizers) and min(lengths[digest].values()) >= required_tokens
    ]
    lengths = {
        digest: per_model for digest, per_model in lengths.items()
        if len(per_model) == len(tokenizers) and min(per_model.values()) >= required_tokens
    }
    return accepted, lengths


def _load_dataset_and_tokenizers(config: Mapping[str, object], args: argparse.Namespace):
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoTokenizer

    local_only = not args.allow_download
    dataset = load_dataset(
        "wikitext",
        str(config["dataset"]["config"]),
        revision=str(config["dataset"]["revision"]),
        split=str(config["dataset"]["split"]),
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        download_config=DownloadConfig(local_files_only=local_only),
    )
    tokenizers = {
        key: AutoTokenizer.from_pretrained(
            value["repo_id"],
            revision=value["revision"],
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=local_only,
        )
        for key, value in config["models"].items()
    }
    return dataset, tokenizers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise DataPreparationError("output directory already exists")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset, tokenizers = _load_dataset_and_tokenizers(config, args)
    articles = parse_wikitext_articles((str(row["text"]) for row in dataset))
    historical = historical_prefix_hashes(
        articles,
        seed=int(config["dataset"]["historical_article_shuffle_seed"]),
        count=int(config["dataset"]["historical_article_prefix_excluded"]),
    )
    historical_set = set(historical)
    unused_articles = [
        text for text in articles
        if hashlib.sha256(text.encode("utf-8")).hexdigest() not in historical_set
    ]
    required_tokens = int(config["dataset"]["prompt_len"]) + int(config["dataset"]["decode_steps"])
    eligible, token_lengths = dual_tokenizer_filter(
        unused_articles,
        tokenizers,
        required_tokens=required_tokens,
    )
    calibration, sealed = select_documents(
        eligible,
        seed=int(config["seed"]),
        calibration_count=int(config["dataset"]["calibration_documents"]),
        sealed_count=int(config["dataset"]["sealed_documents"]),
        excluded_hashes=historical,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl_no_overwrite(args.output_dir / "calibration_manifest.jsonl", calibration)
    write_jsonl_no_overwrite(args.output_dir / "sealed_manifest.jsonl", sealed)
    write_json_no_overwrite(args.output_dir / "historical_hash_registry.json", {
        "schema_version": "triage-historical-hashes-v2",
        "shuffle_seed": config["dataset"]["historical_article_shuffle_seed"],
        "excluded_prefix_count": config["dataset"]["historical_article_prefix_excluded"],
        "known_prior_ranges": list(KNOWN_HISTORICAL_RANGES),
        "hashes": historical,
    })
    selected = calibration + sealed
    write_json_no_overwrite(args.output_dir / "provenance.json", {
        "schema_version": "triage-data-provenance-v2",
        "dataset_repo_id": "wikitext",
        "dataset_config": config["dataset"]["config"],
        "dataset_revision": config["dataset"]["revision"],
        "dataset_split": config["dataset"]["split"],
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "article_count": len(articles),
        "historically_excluded_count": len(historical),
        "dual_tokenizer_eligible_count": len(eligible),
        "required_tokens_with_special_tokens": required_tokens,
        "model_revisions": {
            key: f"{value['repo_id']}@{value['revision']}" for key, value in config["models"].items()
        },
        "selected_token_lengths": {
            str(row["text_sha256"]): token_lengths[str(row["text_sha256"])] for row in selected
        },
        "calibration_hashes": [row["text_sha256"] for row in calibration],
        "sealed_hashes": [row["text_sha256"] for row in sealed],
    })
    (args.output_dir / "sealed_manifest.jsonl").chmod(0o600)


if __name__ == "__main__":
    main()
