#!/usr/bin/env python3
"""Bind frozen calibration/sealed prompt manifests and historical exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--historical-hashes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, expected_split: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol_split") != expected_split:
        raise RuntimeError(f"expected {expected_split} prompt manifest")
    supplied = value.get("manifest_sha256")
    unhashed = dict(value)
    unhashed.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(
            unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if supplied != actual:
        raise RuntimeError(f"{expected_split} prompt manifest self-hash mismatch")
    return value


def main() -> None:
    args = parse_args()
    calibration = _load(args.calibration, "calibration")
    sealed = _load(args.sealed, "sealed")
    if calibration.get("model_revisions") != sealed.get("model_revisions"):
        raise RuntimeError("calibration/sealed model revisions differ")
    calibration_hashes = [str(row["text_sha256"]) for row in calibration["requests"]]
    sealed_hashes = [str(row["text_sha256"]) for row in sealed["requests"]]
    historical_hashes: list[str] = []
    if args.historical_hashes is not None:
        raw = json.loads(args.historical_hashes.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise RuntimeError("historical hash registry must be a JSON list")
        historical_hashes = [str(value) for value in raw]
    calibration_set = set(calibration_hashes)
    sealed_set = set(sealed_hashes)
    historical_set = set(historical_hashes)
    if calibration_set & sealed_set or calibration_set & historical_set or sealed_set & historical_set:
        raise RuntimeError("calibration/sealed/historical prompt hashes overlap")
    registry = {
        "schema_version": 1,
        "dataset": "wikitext/wikitext-103-raw-v1",
        "dataset_split": "train",
        "sealed": True,
        "selection_seed": 20260722,
        "calibration_window": [20000, 22000],
        "sealed_window": [40000, 44000],
        "calibration_selected_count": 64,
        "sealed_selected_count": 128,
        "tokens_per_request": 128,
        "tokenizer_min_tokens": 129,
        "calibration_manifest_sha256": calibration["manifest_sha256"],
        "sealed_manifest_sha256": sealed["manifest_sha256"],
        "calibration_hashes": calibration_hashes,
        "sealed_hashes": sealed_hashes,
        "historical_hashes": historical_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

