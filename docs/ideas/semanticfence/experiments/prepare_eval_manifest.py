#!/usr/bin/env python3
"""Prepare a fresh, deterministic SemanticFence 32-document eval manifest.

The selection API accepts an injected document stream and token-length function so
its integrity rules can be tested without Hugging Face, a network, or a GPU.  The
CLI is deliberately offline-only and reads the already pinned WikiText and OLMoE
caches used by the workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "semanticfence-fresh-eval-v1"
FIXED_SELECTION_SALT = "semanticfence-p0-eval-20260809-v1"
EVAL_DOCUMENT_COUNT = 32
REQUIRED_TOKENS = 2081
MINIMUM_CHARACTERS = 500
TOKENIZER_BATCH_SIZE = 128

DATASET_REPO_ID = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_SPLIT = "train"
MODEL_REPO_ID = "allenai/OLMoE-1B-7B-0924"
MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"

DEFAULT_EXCLUSION_PATHS = {
    "historical_registry": Path(
        "docs/ideas/routeguard_kv/experiments/data/r0a_5090_v1/"
        "historical_hash_registry.json"
    ),
    "calibration_manifest": Path(
        "docs/ideas/routeguard_kv/experiments/data/r0a_5090_v1/"
        "calibration_manifest.jsonl"
    ),
    "sealed_manifest": Path(
        "docs/ideas/routeguard_kv/experiments/data/r0a_5090_v1/"
        "sealed_manifest.jsonl"
    ),
    "smoke_manifest": Path(
        "docs/ideas/routeguard_kv/experiments/data/r0a_5090_v1/"
        "smoke_manifest.jsonl"
    ),
}
_REQUIRED_EXCLUSION_NAMES = frozenset(DEFAULT_EXCLUSION_PATHS)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

TokenLengthBatch = Callable[[Sequence[str]], Sequence[int]]


class PreparationError(RuntimeError):
    """The fresh eval split cannot be prepared without weakening its contract."""


def canonical_text(text: str) -> str:
    """Use the workspace's full-text canonicalization rule."""

    if not isinstance(text, str):
        raise PreparationError("candidate document text must be a string")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def text_sha256(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def selection_sha256(text_digest: str) -> str:
    digest = _require_sha256(text_digest, field="selection text_sha256")
    return hashlib.sha256(
        f"{FIXED_SELECTION_SALT}|{digest}".encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PreparationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def ordered_hash_of_hashes(values: Sequence[str]) -> str:
    for value in values:
        _require_sha256(value, field="ordered hash entry")
    return hashlib.sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


def parse_wikitext_articles(
    rows: Iterable[str], *, minimum_characters: int = MINIMUM_CHARACTERS
) -> list[str]:
    """Convert raw WikiText lines into article-level independent documents."""

    if minimum_characters < 1:
        raise PreparationError("minimum_characters must be positive")
    documents: list[str] = []
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal title, body
        if title is not None:
            value = canonical_text(" ".join([title, *body]).strip())
            if len(value) >= minimum_characters:
                documents.append(value)
        title = None
        body = []

    for raw in rows:
        line = " ".join(str(raw).split())
        is_title = line.startswith("= ") and line.endswith(" =") and not line.startswith(
            "= ="
        )
        if is_title:
            flush()
            title = line.strip("= ")
        elif line and title is not None:
            body.append(line)
    flush()
    return documents


def default_exclusion_sources(repo_root: Path) -> dict[str, Path]:
    return {name: repo_root / relative for name, relative in DEFAULT_EXCLUSION_PATHS.items()}


def load_exclusion_hashes(
    sources: Mapping[str, Path],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Load every required historical/current full-document exclusion source."""

    if set(sources) != _REQUIRED_EXCLUSION_NAMES:
        raise PreparationError(
            "exclusion sources must be exactly "
            f"{sorted(_REQUIRED_EXCLUSION_NAMES)}, got {sorted(sources)}"
        )

    union: set[str] = set()
    report: dict[str, dict[str, Any]] = {}
    for name in sorted(sources):
        path = Path(sources[name])
        if name == "historical_registry":
            values, record_count = _hashes_from_historical_registry(path)
        else:
            values, record_count = _hashes_from_document_manifest(path)
        overlap = len(union & values)
        union.update(values)
        report[name] = {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "record_count": record_count,
            "unique_full_text_hash_count": len(values),
            "overlap_with_prior_sources": overlap,
        }
    return union, report


def select_fresh_documents(
    candidate_documents: Iterable[str],
    *,
    excluded_hashes: set[str],
    token_lengths: TokenLengthBatch,
    document_count: int = EVAL_DOCUMENT_COUNT,
    required_tokens: int = REQUIRED_TOKENS,
) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
    """Select a deterministic exact-size split independent of candidate order."""

    if document_count != EVAL_DOCUMENT_COUNT:
        raise PreparationError(
            f"fresh eval document_count is frozen at {EVAL_DOCUMENT_COUNT}"
        )
    if required_tokens != REQUIRED_TOKENS:
        raise PreparationError(f"required_tokens is frozen at {REQUIRED_TOKENS}")

    candidates: dict[str, str] = {}
    candidate_stream_hashes: list[str] = []
    duplicate_count = 0
    excluded_count = 0
    for raw_text in candidate_documents:
        text = canonical_text(raw_text)
        digest = text_sha256(text)
        candidate_stream_hashes.append(digest)
        if digest in candidates:
            duplicate_count += 1
            continue
        candidates[digest] = text

    eligible_pool = [
        (digest, text)
        for digest, text in sorted(candidates.items())
        if digest not in excluded_hashes
    ]
    excluded_count = len(candidates) - len(eligible_pool)
    lengths: dict[str, int] = {}
    short_count = 0
    for start in range(0, len(eligible_pool), TOKENIZER_BATCH_SIZE):
        batch = eligible_pool[start : start + TOKENIZER_BATCH_SIZE]
        observed_lengths = list(token_lengths([text for _, text in batch]))
        if len(observed_lengths) != len(batch):
            raise PreparationError(
                "token length function returned a different number of rows"
            )
        for (digest, _), raw_length in zip(batch, observed_lengths):
            if isinstance(raw_length, bool) or not isinstance(raw_length, int):
                raise PreparationError("token lengths must be integers")
            if raw_length < 0:
                raise PreparationError("token lengths must be non-negative")
            if raw_length >= required_tokens:
                lengths[digest] = raw_length
            else:
                short_count += 1

    ordered = sorted(lengths, key=lambda digest: (selection_sha256(digest), digest))
    selected = ordered[:document_count]
    if len(selected) != document_count:
        raise PreparationError(
            f"only {len(selected)} eligible fresh documents for exact-{document_count} eval"
        )
    if len(set(selected)) != document_count or set(selected) & excluded_hashes:
        raise PreparationError("selected documents are duplicate or excluded")

    rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "split": "semanticfence_eval_fresh",
            "document_index": index,
            "text_sha256": digest,
            "selection_sha256": selection_sha256(digest),
            "token_length_at_least": lengths[digest],
            "text": candidates[digest],
        }
        for index, digest in enumerate(selected)
    ]
    statistics: dict[str, int | str] = {
        "candidate_stream_record_count": len(candidate_stream_hashes),
        "candidate_stream_unique_count": len(candidates),
        "candidate_stream_duplicate_count": duplicate_count,
        "candidate_stream_ordered_hash_of_hashes": ordered_hash_of_hashes(
            candidate_stream_hashes
        ),
        "excluded_candidate_count": excluded_count,
        "short_candidate_count": short_count,
        "eligible_candidate_count": len(lengths),
        "selected_document_count": len(rows),
    }
    return rows, statistics


def prepare_fresh_eval_manifest(
    *,
    candidate_documents: Iterable[str],
    token_lengths: TokenLengthBatch,
    exclusion_sources: Mapping[str, Path],
    output_dir: Path,
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create all fresh-eval artifacts once and refuse an existing output root."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise PreparationError(f"refusing to overwrite output directory: {output_dir}")

    excluded_hashes, source_report = load_exclusion_hashes(exclusion_sources)
    rows, statistics = select_fresh_documents(
        candidate_documents,
        excluded_hashes=excluded_hashes,
        token_lengths=token_lengths,
    )
    selected_hashes = [str(row["text_sha256"]) for row in rows]

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "eval_manifest.jsonl"
    _write_jsonl_exclusive(manifest_path, rows)
    manifest_sha256 = sha256_file(manifest_path)

    exclusion_report = {
        "schema_version": "semanticfence-fresh-eval-exclusions-v1",
        "required_source_names": sorted(_REQUIRED_EXCLUSION_NAMES),
        "sources": source_report,
        "excluded_full_text_hash_count": len(excluded_hashes),
        "selected_overlap_count": len(set(selected_hashes) & excluded_hashes),
        **statistics,
    }
    exclusion_path = output_dir / "exclusion_report.json"
    _write_json_exclusive(exclusion_path, exclusion_report)

    provenance = {
        "schema_version": "semanticfence-fresh-eval-provenance-v1",
        "status": "FRESH_EVAL_PREPARED_NOT_EXECUTED",
        "selection_salt": FIXED_SELECTION_SALT,
        "selection_key_format": "sha256(utf8(salt)+ascii_pipe+ascii(text_sha256_hex))",
        "document_count": EVAL_DOCUMENT_COUNT,
        "required_tokens": REQUIRED_TOKENS,
        "canonical_text_rule": "normalize_CRLF_and_CR_to_LF",
        "dataset": {
            "repo_id": DATASET_REPO_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "sampling_unit": "article_document",
        },
        "tokenizer": {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "add_special_tokens": True,
            "truncation": True,
            "max_length": REQUIRED_TOKENS,
        },
        "source_provenance": dict(source_provenance or {}),
        "exclusion_source_file_sha256": {
            name: str(details["file_sha256"])
            for name, details in sorted(source_report.items())
        },
        "exclusion_report_sha256": sha256_file(exclusion_path),
        "eval_manifest_sha256": manifest_sha256,
        "selected_ordered_hash_of_hashes": ordered_hash_of_hashes(selected_hashes),
        "selected_text_sha256": selected_hashes,
        "statistics": statistics,
    }
    provenance_path = output_dir / "provenance.json"
    _write_json_exclusive(provenance_path, provenance)

    artifact_hashes = {
        "schema_version": "semanticfence-fresh-eval-artifact-hashes-v1",
        "files": {
            "eval_manifest.jsonl": manifest_sha256,
            "exclusion_report.json": sha256_file(exclusion_path),
            "provenance.json": sha256_file(provenance_path),
        },
    }
    artifact_hashes_path = output_dir / "artifact_hashes.json"
    _write_json_exclusive(artifact_hashes_path, artifact_hashes)
    return {
        "output_dir": str(output_dir),
        "eval_manifest_sha256": manifest_sha256,
        "artifact_hashes_sha256": sha256_file(artifact_hashes_path),
        "selected_document_count": len(rows),
    }


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PreparationError(f"{field} must be one lowercase SHA-256")
    return value


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot load JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PreparationError(f"{path}:{line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot load JSONL {path}: {exc}") from exc
    return rows


def _hashes_from_historical_registry(path: Path) -> tuple[set[str], int]:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise PreparationError(f"historical registry is not an object: {path}")
    hashes: set[str] = set()
    declared_count = 0
    for field in ("hashes", "prefix_hashes"):
        entries = value.get(field, [])
        if not isinstance(entries, list):
            raise PreparationError(f"historical registry field {field} is not a list")
        declared_count += len(entries)
        for entry in entries:
            hashes.add(_require_sha256(entry, field=f"historical {field} entry"))
    return hashes, declared_count


def _hashes_from_document_manifest(path: Path) -> tuple[set[str], int]:
    rows = _load_jsonl(path)
    hashes: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row.get("text"), str):
            raise PreparationError(f"{path}: row {index} has no full text")
        declared = _require_sha256(
            row.get("text_sha256"), field=f"{path}: row {index} text_sha256"
        )
        observed = text_sha256(str(row["text"]))
        if observed != declared:
            raise PreparationError(
                f"{path}: row {index} full-text hash mismatch {observed} != {declared}"
            )
        hashes.add(observed)
    return hashes, len(rows)


def _write_json_exclusive(path: Path, value: Any) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PreparationError(f"refusing to overwrite artifact: {path}") from exc


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    try:
        with path.open("xb") as handle:
            for row in rows:
                handle.write(canonical_json_bytes(dict(row)))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PreparationError(f"refusing to overwrite artifact: {path}") from exc


def _offline_cache_roots(cache_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    if cache_dir is not None:
        roots.extend((cache_dir / "datasets", cache_dir))
    if os.environ.get("HF_DATASETS_CACHE"):
        roots.append(Path(os.environ["HF_DATASETS_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "datasets")
    roots.append(Path.home() / ".cache/huggingface/datasets")
    return list(dict.fromkeys(roots))


def load_pinned_candidate_stream(
    *, cache_dir: Path | None = None
) -> tuple[list[str], TokenLengthBatch, dict[str, Any]]:
    """Load the exact WikiText/OLMoE inputs from local caches only."""

    try:
        from datasets import Dataset, concatenate_datasets
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise PreparationError(
            "CLI requires locally installed datasets and transformers packages"
        ) from exc

    pattern = (
        f"{DATASET_REPO_ID}/{DATASET_CONFIG}/**/{DATASET_REVISION}/"
        f"{DATASET_REPO_ID}-{DATASET_SPLIT}*.arrow"
    )
    dataset_files: list[Path] = []
    for root in _offline_cache_roots(cache_dir):
        dataset_files = sorted(root.glob(pattern))
        if dataset_files:
            break
    if not dataset_files:
        raise PreparationError(
            "pinned WikiText cache is unavailable offline; no download is permitted"
        )
    parts = [Dataset.from_file(str(path)) for path in dataset_files]
    dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    raw_rows = [str(row["text"]) for row in dataset]
    documents = parse_wikitext_articles(raw_rows)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO_ID,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=True,
    )

    def token_lengths(texts: Sequence[str]) -> Sequence[int]:
        encoded = tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=True,
            max_length=REQUIRED_TOKENS,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        return [len(token_ids) for token_ids in encoded]

    provenance = {
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "parsed_article_count": len(documents),
        "dataset_cache_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in dataset_files
        ],
    }
    return documents, token_lengths, provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, but do not execute, the fresh SemanticFence eval-32 split."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    documents, token_lengths, source_provenance = load_pinned_candidate_stream(
        cache_dir=args.cache_dir
    )
    result = prepare_fresh_eval_manifest(
        candidate_documents=documents,
        token_lengths=token_lengths,
        exclusion_sources=default_exclusion_sources(repo_root),
        output_dir=args.output_dir,
        source_provenance=source_provenance,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreparationError as error:
        print(f"PREPARATION_FAILED: {error}", file=os.sys.stderr)
        raise SystemExit(2)
