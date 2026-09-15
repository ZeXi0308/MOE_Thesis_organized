"""Deterministic document-manifest construction for TriageAudit v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class ManifestError(RuntimeError):
    pass


def canonical_text(text: str) -> str:
    if not isinstance(text, str):
        raise ManifestError("document text must be a string")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def text_sha256(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def selection_digest(seed: int, digest: str) -> str:
    return hashlib.sha256(f"{seed}|{digest}".encode("utf-8")).hexdigest()


def select_documents(
    documents: Sequence[str],
    *,
    seed: int,
    calibration_count: int,
    sealed_count: int,
    excluded_hashes: Iterable[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if calibration_count <= 0 or sealed_count <= 0:
        raise ManifestError("split counts must be positive")
    excluded = set(excluded_hashes)
    candidates: dict[str, str] = {}
    for text in documents:
        canonical = canonical_text(text)
        digest = text_sha256(canonical)
        if digest not in excluded:
            candidates.setdefault(digest, canonical)
    needed = calibration_count + sealed_count
    if len(candidates) < needed:
        raise ManifestError(f"only {len(candidates)} eligible documents for {needed} slots")
    ordered = sorted(candidates, key=lambda digest: selection_digest(seed, digest))[:needed]

    def records(digests: Sequence[str], split: str) -> list[dict[str, object]]:
        return [
            {
                "schema_version": "triage-document-v2",
                "split": split,
                "document_index": index,
                "text_sha256": digest,
                "selection_sha256": selection_digest(seed, digest),
                "text": candidates[digest],
            }
            for index, digest in enumerate(digests)
        ]

    calibration = records(ordered[:calibration_count], "calibration")
    sealed = records(ordered[calibration_count:], "sealed")
    if {row["text_sha256"] for row in calibration} & {row["text_sha256"] for row in sealed}:
        raise AssertionError("calibration/sealed overlap")
    return calibration, sealed


def load_excluded_hashes(paths: Sequence[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise ManifestError(f"historical hash source does not exist: {path}")
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            rows = value if isinstance(value, list) else value.get("documents", [])
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        else:
            rows = [{"text_sha256": line.strip()} for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            if not isinstance(row, Mapping):
                raise ManifestError(f"invalid historical row in {path}")
            digest = row.get("text_sha256") or row.get("document_sha256") or row.get("canonical_text_sha256")
            if digest is None:
                continue
            digest = str(digest)
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ManifestError(f"invalid historical hash in {path}: {digest}")
            result.add(digest)
    return result


def write_jsonl_no_overwrite(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if path.exists():
        raise ManifestError(f"refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-jsonl", type=Path, required=True)
    parser.add_argument("--historical-hash-file", type=Path, action="append", default=[])
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--sealed-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026072302)
    parser.add_argument("--calibration-count", type=int, default=32)
    parser.add_argument("--sealed-count", type=int, default=64)
    args = parser.parse_args()
    documents = []
    for line in args.documents_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, Mapping) or not isinstance(row.get("text"), str):
                raise ManifestError("documents JSONL requires a string text field")
            documents.append(row["text"])
    excluded = load_excluded_hashes(args.historical_hash_file)
    calibration, sealed = select_documents(
        documents,
        seed=args.seed,
        calibration_count=args.calibration_count,
        sealed_count=args.sealed_count,
        excluded_hashes=excluded,
    )
    write_jsonl_no_overwrite(args.calibration_output, calibration)
    write_jsonl_no_overwrite(args.sealed_output, sealed)


if __name__ == "__main__":
    main()
