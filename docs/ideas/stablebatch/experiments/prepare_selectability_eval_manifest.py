#!/usr/bin/env python3
"""Prepare the globally untouched StableBatch Selectability eval-16 split.

This is an offline, pre-outcome data preparation step.  It reuses the pinned
WikiText/OLMoE cache and the historical exclusion contract, then additionally
excludes the already executed SemanticFence fresh-32 manifest.  Selection is
deterministic and independent of all StableBatch action outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from docs.ideas.semanticfence.experiments import prepare_eval_manifest as source


SCHEMA_VERSION = "stablebatch-selectability-fresh-eval-v1"
SPLIT = "stablebatch_selectability_eval_fresh"
DOCUMENT_COUNT = 16
TOKEN_OFFSET = 512
WINDOW_TOKENS = 16
SEMANTICFENCE_MANIFEST = Path(
    "docs/ideas/semanticfence/experiments/data/fresh_eval_20260809_v1/"
    "eval_manifest.jsonl"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def window_sha256(token_ids: Sequence[int]) -> str:
    return sha256_bytes(
        json.dumps(
            list(map(int, token_ids)),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def prepare(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise source.PreparationError(f"refusing to overwrite {output_dir}")

    documents, token_lengths, dataset_provenance = source.load_pinned_candidate_stream()
    historical, historical_report = source.load_exclusion_hashes(
        source.default_exclusion_sources(repo_root)
    )
    semantic_path = repo_root / SEMANTICFENCE_MANIFEST
    semantic_hashes, semantic_rows = source._hashes_from_document_manifest(
        semantic_path
    )
    excluded = historical | semantic_hashes
    if historical & semantic_hashes:
        raise source.PreparationError(
            "SemanticFence fresh manifest unexpectedly overlaps historical exclusions"
        )

    candidates: dict[str, tuple[int, str]] = {}
    for article_index, raw_text in enumerate(documents):
        text = source.canonical_text(raw_text)
        digest = source.text_sha256(text)
        candidates.setdefault(digest, (article_index, text))

    eligible = [
        (digest, article_index, text)
        for digest, (article_index, text) in sorted(candidates.items())
        if digest not in excluded
    ]
    qualifying: dict[str, tuple[int, str, int]] = {}
    for start in range(0, len(eligible), source.TOKENIZER_BATCH_SIZE):
        batch = eligible[start : start + source.TOKENIZER_BATCH_SIZE]
        lengths = list(token_lengths([text for _, _, text in batch]))
        if len(lengths) != len(batch):
            raise source.PreparationError("token length batch cardinality mismatch")
        for (digest, article_index, text), length in zip(batch, lengths):
            if int(length) >= source.REQUIRED_TOKENS:
                qualifying[digest] = (article_index, text, int(length))

    ordered = sorted(
        qualifying,
        key=lambda digest: (source.selection_sha256(digest), digest),
    )
    selected = ordered[:DOCUMENT_COUNT]
    if len(selected) != DOCUMENT_COUNT:
        raise source.PreparationError("fewer than 16 eligible untouched documents")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source.MODEL_REPO_ID,
        revision=source.MODEL_REVISION,
        local_files_only=True,
    )
    rows: list[dict[str, Any]] = []
    window_hashes: list[str] = []
    for document_index, digest in enumerate(selected):
        article_index, text, token_length = qualifying[digest]
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        window = list(map(int, token_ids[TOKEN_OFFSET : TOKEN_OFFSET + WINDOW_TOKENS]))
        if len(window) != WINDOW_TOKENS:
            raise source.PreparationError("selected document has incomplete offset-512 window")
        digest_window = window_sha256(window)
        window_hashes.append(digest_window)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "split": SPLIT,
                "document_index": document_index,
                "source_article_index": article_index,
                "text_sha256": digest,
                "selection_sha256": source.selection_sha256(digest),
                "token_length_at_least": token_length,
                "offset512_window_token_ids_sha256": digest_window,
                "text": text,
            }
        )

    ordered_window_digest = sha256_bytes("".join(window_hashes).encode("utf-8"))
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "eval_manifest.jsonl"
    source._write_jsonl_exclusive(manifest_path, rows)
    provenance = {
        "schema_version": "stablebatch-selectability-fresh-eval-provenance-v1",
        "status": "PREPARED_NOT_EXECUTED",
        "selection_rule": "first_16_by_semanticfence_fixed_salted_hash_after_union_exclusion",
        "selection_salt": source.FIXED_SELECTION_SALT,
        "document_count": DOCUMENT_COUNT,
        "required_tokens": source.REQUIRED_TOKENS,
        "window": {
            "token_offset": TOKEN_OFFSET,
            "window_tokens": WINDOW_TOKENS,
            "add_special_tokens": False,
            "ordered_window_hash_digest_method": (
                "sha256_of_concatenated_window_sha256_hex_in_document_order"
            ),
            "ordered_window_hash_digest": ordered_window_digest,
            "window_token_ids_sha256": window_hashes,
        },
        "dataset": {
            "repo_id": source.DATASET_REPO_ID,
            "config": source.DATASET_CONFIG,
            "revision": source.DATASET_REVISION,
            "split": source.DATASET_SPLIT,
            **dataset_provenance,
        },
        "tokenizer": {
            "repo_id": source.MODEL_REPO_ID,
            "revision": source.MODEL_REVISION,
        },
        "exclusions": {
            "historical_unique_hashes": len(historical),
            "historical_sources": historical_report,
            "semanticfence_manifest": str(SEMANTICFENCE_MANIFEST),
            "semanticfence_manifest_sha256": source.sha256_file(semantic_path),
            "semanticfence_manifest_rows": semantic_rows,
            "semanticfence_unique_hashes": len(semantic_hashes),
            "union_unique_hashes": len(excluded),
            "selected_overlap_count": len(set(selected) & excluded),
        },
        "selected_text_sha256": selected,
        "manifest_sha256": source.sha256_file(manifest_path),
    }
    source._write_json_exclusive(output_dir / "provenance.json", provenance)
    artifact_hashes = {
        "schema_version": "stablebatch-selectability-fresh-eval-artifacts-v1",
        "files": {
            "eval_manifest.jsonl": source.sha256_file(manifest_path),
            "provenance.json": source.sha256_file(output_dir / "provenance.json"),
        },
    }
    source._write_json_exclusive(output_dir / "artifact_hashes.json", artifact_hashes)
    return {
        "output_dir": str(output_dir),
        "manifest_sha256": source.sha256_file(manifest_path),
        "ordered_window_hash_digest": ordered_window_digest,
        "selected_document_count": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.repo_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
