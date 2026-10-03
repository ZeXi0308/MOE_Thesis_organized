#!/usr/bin/env python3
"""Fail-closed merge of the two reviewed per-model cjc-v1 LUT artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(candidate for candidate in HERE.parents if (candidate / "experiments/shared").is_dir())
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cjc_policy import (  # noqa: E402
    CJCValidationError,
    LUT_COMPONENT_PROVENANCE,
    LUT_EXPERT_SOURCE,
    LUT_HOST_STAGING_SOURCE,
    LUT_LAUNCH_SOURCE,
    LUT_PACK_SOURCE,
    LUT_REDUCTION_SOURCE,
)
from run_cjc_oracle import load_json, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "docs/archive/receiver_aware/cjc/configs/cjc_v1.json",
    )
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise CJCValidationError(f"empty per-model LUT: {path}")
    return rows


def validate_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    revision: str,
    expected_layers: Sequence[int],
    expected_row_grid: Sequence[int],
) -> None:
    expected_sources = {
        "source": LUT_COMPONENT_PROVENANCE,
        "expert_source": LUT_EXPERT_SOURCE,
        "pack_source": LUT_PACK_SOURCE,
        "launch_source": LUT_LAUNCH_SOURCE,
        "host_staging_source": LUT_HOST_STAGING_SOURCE,
        "reduction_source": LUT_REDUCTION_SOURCE,
    }
    identities: set[tuple[int, int]] = set()
    for row in rows:
        if row.get("model_revision") != revision:
            raise CJCValidationError("per-model LUT revision mismatch")
        for name, expected in expected_sources.items():
            if row.get(name) != expected:
                raise CJCValidationError(f"LUT component provenance mismatch: {name}")
        if float(row.get("launch_us", "nan")) != 0.0:
            raise CJCValidationError("separate launch charge would double count component events")
        identity = int(row["layer_id"]), int(row["rows"])
        if identity in identities:
            raise CJCValidationError("duplicate model/layer/row LUT point")
        identities.add(identity)
    expected = {
        (int(layer), int(row_count))
        for layer in expected_layers
        for row_count in expected_row_grid
    }
    if identities != expected:
        raise CJCValidationError("per-model LUT layer/row grid is incomplete or unexpected")


def write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    models = config.get("required_models")
    lut_cfg = config.get("lut")
    if not isinstance(models, dict) or not isinstance(lut_cfg, dict):
        raise CJCValidationError("merge config lacks model/LUT definitions")
    if len(args.input_dir) != len(models):
        raise CJCValidationError("merge requires exactly one LUT directory per frozen model")
    required_revisions = {
        str(value["revision"]) for value in models.values() if isinstance(value, dict)
    }
    seen_revisions: set[str] = set()
    gpu_names: set[str] = set()
    merged: list[dict[str, str]] = []
    input_hashes: dict[str, object] = {}
    for directory in args.input_dir:
        metadata_path = directory / "lut_metadata.json"
        lut_path = directory / "lut.csv"
        metadata = load_json(metadata_path)
        revision = str(metadata.get("model_revision", ""))
        if revision not in required_revisions or revision in seen_revisions:
            raise CJCValidationError("duplicate or unexpected per-model LUT revision")
        if args.mode == "formal" and (
            metadata.get("status") != "LUT_ONLY" or metadata.get("mode") != "formal"
        ):
            raise CJCValidationError("formal merge requires formal reviewed LUT_ONLY inputs")
        selected_layers = metadata.get("selected_layers")
        if not isinstance(selected_layers, list):
            raise CJCValidationError("LUT metadata lacks selected layers")
        rows = load_rows(lut_path)
        validate_rows(
            rows,
            revision=revision,
            expected_layers=[int(value) for value in selected_layers],
            expected_row_grid=[int(value) for value in lut_cfg["rows"]],
        )
        seen_revisions.add(revision)
        gpu_names.add(str(metadata.get("gpu_name", "")))
        merged.extend(rows)
        input_hashes[revision] = {
            "lut_sha256": sha256_file(lut_path),
            "metadata_sha256": sha256_file(metadata_path),
        }
    if seen_revisions != required_revisions:
        raise CJCValidationError("merged LUT does not cover both frozen model revisions")
    if len(gpu_names) != 1:
        raise CJCValidationError("per-model LUTs were not measured on one GPU identity")
    gpu_name = next(iter(gpu_names))
    if args.mode == "formal" and gpu_name != "NVIDIA GeForce RTX 5090":
        raise CJCValidationError("formal CJC LUT must use the frozen RTX 5090")
    if args.output_dir.exists():
        raise CJCValidationError("refusing to overwrite merged LUT output")
    args.output_dir.mkdir(parents=True)
    merged.sort(key=lambda row: (row["model_revision"], int(row["layer_id"]), int(row["rows"])))
    write_rows(args.output_dir / "lut.csv", merged)
    metadata = {
        "schema_version": "cjc-lut-merged-v1",
        "status": "LUT_ONLY" if args.mode == "formal" else "NOT_TESTED",
        "scientific_result": False,
        "mode": args.mode,
        "gpu_name": gpu_name,
        "model_revisions": sorted(seen_revisions),
        "component_provenance": "preserved_verbatim_per_point",
        "input_hashes": input_hashes,
        "config_sha256": sha256_file(args.config),
        "merge_source_sha256": sha256_file(Path(__file__)),
    }
    (args.output_dir / "lut_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": metadata["status"], "rows": len(merged)}, indent=2))


if __name__ == "__main__":
    main()
