#!/usr/bin/env python3
"""Run the calibration-only SemanticFence M=2 partner-permutation probe.

The protocol has three public stages:

* ``plan`` derives a deterministic, outcome-stratified schedule from the
  already-frozen run03 calibration artifact.  It performs no GPU work.
* ``seal`` binds that schedule, every source artifact, the implementation and
  a fresh live-stack acceptance artifact before any scientific call.
* ``run`` replays the same focal rows with alternate partners.  It writes
  ``COMPLETE.json`` last; an incomplete directory has no authority.

This is deliberately a calibration mechanism audit.  It tests partner
invariance while holding the focal row's original M=2 slot fixed.  It is not a
fresh-evaluation, full-layer, serving, EP, latency, or paper result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "semanticfence-partner-permutation-config-v1"
PLAN_SCHEMA = "semanticfence-partner-permutation-plan-v1"
PLAN_COMPLETE_SCHEMA = "semanticfence-partner-permutation-plan-complete-v1"
SCHEDULE_SCHEMA = "semanticfence-partner-permutation-schedule-v1"
LOCK_SCHEMA = "semanticfence-partner-permutation-lock-v1"
NUMERIC_SCHEMA = "semanticfence-partner-permutation-numeric-v1"
RESULT_SCHEMA = "semanticfence-partner-permutation-result-v1"
COMPLETE_SCHEMA = "semanticfence-partner-permutation-complete-v1"
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = EXPERIMENT_DIR.parents[3]
BASE_RUNNER_PATH = EXPERIMENT_DIR / "run_pilot_5090.py"
GPU_EXECUTION_PATH = EXPERIMENT_DIR / "gpu_execution.py"
CONTRACT_PATH = EXPERIMENT_DIR / "executor_contract.py"
TEST_PATH = EXPERIMENT_DIR / "test_run_partner_permutation.py"
HEX = set("0123456789abcdef")


class ProtocolError(RuntimeError):
    """The partner-permutation result cannot be interpreted."""


def _load_module(name: str, path: Path) -> Any:
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PILOT = _load_module("semanticfence_partner_base_pilot", BASE_RUNNER_PATH)
CONTRACT = PILOT.CONTRACT


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolError(f"{path}:{line_number} is not an object")
            yield value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json_no_overwrite(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolError(f"refusing to overwrite {path}") from exc


def write_jsonl_no_overwrite(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolError(f"refusing to overwrite {path}") from exc


def _require_exact_fields(
    value: Mapping[str, Any], required: set[str], *, label: str
) -> None:
    if set(value) != required:
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required)
        raise ProtocolError(f"{label} fields differ: missing={missing}, unknown={unknown}")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "evidence_boundary",
        "source",
        "selection",
        "execution",
        "decision",
    }
    _require_exact_fields(config, required, label="config")
    if config["schema_version"] != SCHEMA or config["status"] != "FROZEN_PRE_RUN":
        raise ProtocolError("partner config schema/status is not frozen")
    if config["evidence_boundary"] != (
        "single_rtx5090_reused_run03_calibration_rows_expert_stage_only_"
        "row_plus_original_slot_plus_m2_partner_invariance_not_fresh_evaluation_"
        "not_full_layer_not_serving_not_ep_not_latency"
    ):
        raise ProtocolError("evidence boundary changed")

    source = config["source"]
    required_source = {
        "source_run",
        "complete_sha256",
        "calibration_captures_sha256",
        "calibration_capture_manifest_sha256",
        "calibration_numeric_sha256",
        "calibration_reference_rows_sha256",
        "frozen_pilot_config_sha256",
        "stack_digest",
        "expected_reference_rows",
        "expected_m2_packs",
        "expected_m2_rows",
        "expected_safe_rows",
        "expected_unsafe_rows",
    }
    if not isinstance(source, Mapping):
        raise ProtocolError("source config is not an object")
    _require_exact_fields(source, required_source, label="source config")
    for key in (
        "complete_sha256",
        "calibration_captures_sha256",
        "calibration_capture_manifest_sha256",
        "calibration_numeric_sha256",
        "calibration_reference_rows_sha256",
        "frozen_pilot_config_sha256",
        "stack_digest",
    ):
        if not is_sha256(source[key]):
            raise ProtocolError(f"source {key} is not SHA-256")
    source_constants = {
        "source_run": "semanticfence_pilot_20260810_run03",
        "expected_reference_rows": 32768,
        "expected_m2_packs": 16117,
        "expected_m2_rows": 32234,
        "expected_safe_rows": 2768,
        "expected_unsafe_rows": 29466,
    }
    for key, expected in source_constants.items():
        if source[key] != expected:
            raise ProtocolError(f"source {key} changed")

    selection = config["selection"]
    required_selection = {
        "seed",
        "layer_count",
        "cells_per_layer",
        "safe_focals_per_cell",
        "unsafe_focals_per_cell",
        "safe_partners_per_focal",
        "unsafe_partners_per_focal",
        "require_different_document",
        "require_different_hidden",
        "exclude_original_partner",
        "preserve_focal_original_slot",
        "expected_selected_cells",
        "expected_focals",
        "expected_pair_calls",
    }
    if not isinstance(selection, Mapping):
        raise ProtocolError("selection config is not an object")
    _require_exact_fields(selection, required_selection, label="selection config")
    selection_constants = {
        "seed": "semanticfence-partner-permutation-v1|"
        + str(source["calibration_numeric_sha256"]),
        "layer_count": 16,
        "cells_per_layer": 16,
        "safe_focals_per_cell": 1,
        "unsafe_focals_per_cell": 1,
        "safe_partners_per_focal": 2,
        "unsafe_partners_per_focal": 2,
        "require_different_document": True,
        "require_different_hidden": True,
        "exclude_original_partner": True,
        "preserve_focal_original_slot": True,
        "expected_selected_cells": 256,
        "expected_focals": 512,
        "expected_pair_calls": 2048,
    }
    for key, expected in selection_constants.items():
        if selection[key] != expected:
            raise ProtocolError(f"selection {key} changed")

    execution = config["execution"]
    required_execution = {
        "m",
        "dtype",
        "warmups",
        "repeats",
        "max_gpu_seconds",
        "require_old_m1_reference_match",
        "require_clean_gpu",
    }
    if not isinstance(execution, Mapping):
        raise ProtocolError("execution config is not an object")
    _require_exact_fields(execution, required_execution, label="execution config")
    execution_constants = {
        "m": 2,
        "dtype": "bfloat16",
        "warmups": 3,
        "repeats": 10,
        "max_gpu_seconds": 600,
        "require_old_m1_reference_match": True,
        "require_clean_gpu": True,
    }
    for key, expected in execution_constants.items():
        if execution[key] != expected:
            raise ProtocolError(f"execution {key} changed")

    decision = config["decision"]
    required_decision = {
        "support_requires_stable_flip_count",
        "support_requires_mixed_pair_count",
        "stable_opposite_outcome_falsifies",
        "mixed_outcome_weakens_stability",
        "claim",
    }
    if not isinstance(decision, Mapping):
        raise ProtocolError("decision config is not an object")
    _require_exact_fields(decision, required_decision, label="decision config")
    decision_constants = {
        "support_requires_stable_flip_count": 0,
        "support_requires_mixed_pair_count": 0,
        "stable_opposite_outcome_falsifies": True,
        "mixed_outcome_weakens_stability": True,
        "claim": "calibration_only_row_plus_original_slot_plus_m2_partner_invariance",
    }
    for key, expected in decision_constants.items():
        if decision[key] != expected:
            raise ProtocolError(f"decision {key} changed")
    return dict(config)


SOURCE_FILES = {
    "complete_sha256": "COMPLETE.json",
    "calibration_captures_sha256": "calibration_captures.pt",
    "calibration_capture_manifest_sha256": "calibration_capture_manifest.jsonl",
    "calibration_numeric_sha256": "calibration_numeric.jsonl",
    "calibration_reference_rows_sha256": "calibration_reference_rows.jsonl",
    "frozen_pilot_config_sha256": "frozen_inputs/config.json",
}


def verify_source_artifacts(
    config: Mapping[str, Any], source_dir: Path
) -> dict[str, str]:
    source_dir = Path(source_dir).resolve()
    source = config["source"]
    observed: dict[str, str] = {}
    for key, relative in SOURCE_FILES.items():
        path = source_dir / relative
        if not path.is_file():
            raise ProtocolError(f"source artifact is absent: {path}")
        digest = sha256_file(path)
        if digest != source[key]:
            raise ProtocolError(f"source artifact hash mismatch: {relative}")
        observed[relative] = digest

    complete = load_json(source_dir / "COMPLETE.json")
    if complete.get("status") != "SUCCESS_COMPLETE":
        raise ProtocolError("source COMPLETE status is not successful")
    declared = complete.get("artifact_sha256")
    if not isinstance(declared, Mapping):
        raise ProtocolError("source COMPLETE lacks artifact hashes")
    for relative in (
        "calibration_captures.pt",
        "calibration_capture_manifest.jsonl",
        "calibration_numeric.jsonl",
        "calibration_reference_rows.jsonl",
    ):
        if declared.get(relative) != observed[relative]:
            raise ProtocolError(f"source COMPLETE does not bind {relative}")

    worker_status = load_json(source_dir / "calibration_worker_status.json")
    if worker_status.get("status") != "COMPLETE":
        raise ProtocolError("source calibration worker is incomplete")
    if worker_status.get("stack_digest") != source["stack_digest"]:
        raise ProtocolError("source calibration stack digest mismatch")
    if int(worker_status.get("calibration_row_count", -1)) != int(
        source["expected_reference_rows"]
    ):
        raise ProtocolError("source calibration row denominator mismatch")
    observed["calibration_worker_status.json"] = sha256_file(
        source_dir / "calibration_worker_status.json"
    )
    return dict(sorted(observed.items()))


@dataclass(frozen=True, slots=True)
class RowEvidence:
    record: Any
    baseline_label: str
    original_partner_row_id: str
    original_slot: int
    original_pack_id: str
    row_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", self.record.row_id)


def load_reference_hashes(
    config: Mapping[str, Any], source_dir: Path
) -> dict[str, str]:
    references: dict[str, str] = {}
    required = {
        "schema_version",
        "row_id",
        "reference_sha256",
        "repeat_sha256",
        "repeat_mismatch_counts",
        "bitwise_stable",
        "all_exact_to_reference",
    }
    path = Path(source_dir) / "calibration_reference_rows.jsonl"
    for value in iter_jsonl(path):
        _require_exact_fields(value, required, label="source M1 reference")
        row_id = value["row_id"]
        hashes = value["repeat_sha256"]
        mismatches = value["repeat_mismatch_counts"]
        if (
            not is_sha256(row_id)
            or row_id in references
            or not isinstance(hashes, list)
            or len(hashes) != 10
            or any(not is_sha256(item) for item in hashes)
            or not isinstance(mismatches, list)
            or len(mismatches) != 10
        ):
            raise ProtocolError("source M1 reference identity/repeats are invalid")
        if (
            value["schema_version"] != "semanticfence-row-execution-v1"
            or value["reference_sha256"] != hashes[0]
            or len(set(hashes)) != 1
            or any(int(item) != 0 for item in mismatches)
            or value["bitwise_stable"] is not True
            or value["all_exact_to_reference"] is not True
        ):
            raise ProtocolError("source M1 reference is not 10/10 stable")
        references[row_id] = hashes[0]
    if len(references) != int(config["source"]["expected_reference_rows"]):
        raise ProtocolError("source M1 reference denominator changed")
    return references


def load_m2_evidence(
    config: Mapping[str, Any], source_dir: Path
) -> dict[str, RowEvidence]:
    path = Path(source_dir) / "calibration_numeric.jsonl"
    evidence: dict[str, RowEvidence] = {}
    pack_count = 0
    labels: Counter[str] = Counter()
    required = {
        "schema_version",
        "call_index",
        "arm",
        "pack_id",
        "layer",
        "expert_id",
        "m",
        "row_ids",
        "row_records",
        "repeat_row_exact",
        "repeat_row_sha256",
        "representative_full_output_sha256",
    }
    for value in iter_jsonl(path):
        _require_exact_fields(value, required, label="source calibration numeric")
        if int(value["m"]) != 2:
            continue
        pack_count += 1
        row_ids = value["row_ids"]
        row_mappings = value["row_records"]
        repeat_exact = value["repeat_row_exact"]
        repeat_sha = value["repeat_row_sha256"]
        if (
            value["schema_version"] != "semanticfence-calibration-numeric-v1"
            or not isinstance(row_ids, list)
            or len(row_ids) != 2
            or len(set(row_ids)) != 2
            or not isinstance(row_mappings, list)
            or len(row_mappings) != 2
            or not isinstance(repeat_exact, list)
            or len(repeat_exact) != 10
            or any(
                not isinstance(repeat, list)
                or len(repeat) != 2
                or any(type(flag) is not bool for flag in repeat)
                for repeat in repeat_exact
            )
            or not isinstance(repeat_sha, list)
            or len(repeat_sha) != 10
            or any(
                not isinstance(repeat, list)
                or len(repeat) != 2
                or any(not is_sha256(item) for item in repeat)
                for repeat in repeat_sha
            )
        ):
            raise ProtocolError("source M=2 observation has invalid repeats")
        records = tuple(
            PILOT.row_record_from_mapping(mapping) for mapping in row_mappings
        )
        pack = CONTRACT.Pack(
            layer=int(value["layer"]),
            expert_id=int(value["expert_id"]),
            rows=records,
        )
        if pack.m != 2 or pack.pack_id != value["pack_id"]:
            raise ProtocolError("source M=2 pack identity mismatch")
        if tuple(row.row_id for row in records) != tuple(row_ids):
            raise ProtocolError("source M=2 row identity mismatch")
        for slot, record in enumerate(records):
            flags = tuple(bool(repeat[slot]) for repeat in repeat_exact)
            if all(flags):
                label = "safe"
            elif not any(flags):
                label = "unsafe"
            else:
                raise ProtocolError("source M=2 row has mixed-repeat label")
            if record.row_id in evidence:
                raise ProtocolError("source row occurs in more than one M=2 pack")
            evidence[record.row_id] = RowEvidence(
                record=record,
                baseline_label=label,
                original_partner_row_id=records[1 - slot].row_id,
                original_slot=slot,
                original_pack_id=pack.pack_id,
            )
            labels[label] += 1
    source = config["source"]
    expected = {
        "packs": int(source["expected_m2_packs"]),
        "rows": int(source["expected_m2_rows"]),
        "safe": int(source["expected_safe_rows"]),
        "unsafe": int(source["expected_unsafe_rows"]),
    }
    observed = {
        "packs": pack_count,
        "rows": len(evidence),
        "safe": labels["safe"],
        "unsafe": labels["unsafe"],
    }
    if observed != expected:
        raise ProtocolError(f"source M=2 denominators changed: {observed} != {expected}")
    return evidence


def _selection_digest(config: Mapping[str, Any], *parts: Any) -> str:
    seed = str(config["selection"]["seed"])
    payload = seed + "\x1f" + "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _partner_is_valid(
    focal: RowEvidence, partner: RowEvidence, other_focal_row_id: str
) -> bool:
    return (
        partner.row_id != focal.row_id
        and partner.row_id != other_focal_row_id
        and partner.row_id != focal.original_partner_row_id
        and partner.record.document_sha256 != focal.record.document_sha256
        and partner.record.hidden_sha256 != focal.record.hidden_sha256
        and partner.record.layer == focal.record.layer
        and partner.record.expert_id == focal.record.expert_id
    )


def _best_focal_pair(
    config: Mapping[str, Any], group: Sequence[RowEvidence]
) -> tuple[RowEvidence, RowEvidence, dict[str, tuple[RowEvidence, ...]]] | None:
    safe = tuple(row for row in group if row.baseline_label == "safe")
    unsafe = tuple(row for row in group if row.baseline_label == "unsafe")
    eligible: list[
        tuple[RowEvidence, RowEvidence, dict[str, tuple[RowEvidence, ...]]]
    ] = []
    for safe_focal in safe:
        for unsafe_focal in unsafe:
            pools = {
                "safe_for_safe": tuple(
                    row
                    for row in safe
                    if _partner_is_valid(
                        safe_focal, row, unsafe_focal.row_id
                    )
                ),
                "unsafe_for_safe": tuple(
                    row
                    for row in unsafe
                    if _partner_is_valid(
                        safe_focal, row, unsafe_focal.row_id
                    )
                ),
                "safe_for_unsafe": tuple(
                    row
                    for row in safe
                    if _partner_is_valid(
                        unsafe_focal, row, safe_focal.row_id
                    )
                ),
                "unsafe_for_unsafe": tuple(
                    row
                    for row in unsafe
                    if _partner_is_valid(
                        unsafe_focal, row, safe_focal.row_id
                    )
                ),
            }
            required = int(config["selection"]["safe_partners_per_focal"])
            if all(len(pool) >= required for pool in pools.values()):
                eligible.append((safe_focal, unsafe_focal, pools))
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda value: _selection_digest(
            config, "focals", value[0].row_id, value[1].row_id
        ),
    )


def select_partner_schedule(
    config: Mapping[str, Any],
    evidence: Mapping[str, RowEvidence],
    reference_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[int, int], list[RowEvidence]] = defaultdict(list)
    for row in evidence.values():
        if row.row_id not in reference_hashes:
            raise ProtocolError("M=2 row lacks a frozen M=1 reference")
        grouped[(int(row.record.layer), int(row.record.expert_id))].append(row)

    best_by_cell: dict[
        tuple[int, int],
        tuple[RowEvidence, RowEvidence, dict[str, tuple[RowEvidence, ...]]],
    ] = {}
    eligible_by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for cell, group in sorted(grouped.items()):
        best = _best_focal_pair(config, tuple(group))
        if best is not None:
            best_by_cell[cell] = best
            eligible_by_layer[cell[0]].append(cell)

    selected_cells: list[tuple[int, int]] = []
    layer_count = int(config["selection"]["layer_count"])
    cells_per_layer = int(config["selection"]["cells_per_layer"])
    for layer in range(layer_count):
        eligible = eligible_by_layer.get(layer, [])
        if len(eligible) < cells_per_layer:
            raise ProtocolError(
                f"layer {layer} has only {len(eligible)} eligible cells"
            )
        selected_cells.extend(
            sorted(
                eligible,
                key=lambda cell: _selection_digest(
                    config, "cell", cell[0], cell[1]
                ),
            )[:cells_per_layer]
        )

    schedule: list[dict[str, Any]] = []
    focal_ids: set[str] = set()
    for layer, expert_id in selected_cells:
        safe_focal, unsafe_focal, pools = best_by_cell[(layer, expert_id)]
        for focal, other_focal in (
            (safe_focal, unsafe_focal),
            (unsafe_focal, safe_focal),
        ):
            focal_ids.add(focal.row_id)
            for partner_label in ("safe", "unsafe"):
                pool_key = f"{partner_label}_for_{focal.baseline_label}"
                pool = sorted(
                    pools[pool_key],
                    key=lambda partner: _selection_digest(
                        config,
                        "partner",
                        focal.row_id,
                        partner_label,
                        partner.row_id,
                    ),
                )
                count = int(
                    config["selection"][f"{partner_label}_partners_per_focal"]
                )
                for partner_rank, partner in enumerate(pool[:count]):
                    ordered = (
                        (focal, partner)
                        if focal.original_slot == 0
                        else (partner, focal)
                    )
                    schedule.append(
                        {
                            "schema_version": SCHEDULE_SCHEMA,
                            "call_index": len(schedule),
                            "layer": layer,
                            "expert_id": expert_id,
                            "m": 2,
                            "row_ids": [row.row_id for row in ordered],
                            "row_records": [
                                row.record.identity_payload() for row in ordered
                            ],
                            "focal_row_id": focal.row_id,
                            "focal_original_slot": int(focal.original_slot),
                            "focal_baseline_label": focal.baseline_label,
                            "focal_old_m1_reference_sha256": reference_hashes[
                                focal.row_id
                            ],
                            "original_partner_row_id": focal.original_partner_row_id,
                            "original_pack_id": focal.original_pack_id,
                            "partner_row_id": partner.row_id,
                            "partner_baseline_label": partner.baseline_label,
                            "partner_rank_within_label": partner_rank,
                            "partner_old_m1_reference_sha256": reference_hashes[
                                partner.row_id
                            ],
                            "different_document": True,
                            "different_hidden": True,
                            "selection_sha256": _selection_digest(
                                config,
                                "scheduled-pair",
                                focal.row_id,
                                focal.original_slot,
                                partner_label,
                                partner_rank,
                                partner.row_id,
                            ),
                        }
                    )

    validation = validate_schedule(config, schedule, evidence=evidence)
    unique_rows = {row_id for call in schedule for row_id in call["row_ids"]}
    summary = {
        "eligible_cell_count": len(best_by_cell),
        "eligible_cells_by_layer": {
            str(layer): len(eligible_by_layer.get(layer, []))
            for layer in range(layer_count)
        },
        "selected_cell_count": len(selected_cells),
        "selected_cells": [
            {"layer": layer, "expert_id": expert_id}
            for layer, expert_id in selected_cells
        ],
        "focal_count": len(focal_ids),
        "pair_call_count": len(schedule),
        "unique_scheduled_row_count": len(unique_rows),
        "scheduled_document_count": len(
            {
                evidence[row_id].record.document_sha256
                for row_id in unique_rows
            }
        ),
        "validation": validation,
    }
    return schedule, summary


def validate_schedule(
    config: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    *,
    evidence: Mapping[str, RowEvidence] | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "call_index",
        "layer",
        "expert_id",
        "m",
        "row_ids",
        "row_records",
        "focal_row_id",
        "focal_original_slot",
        "focal_baseline_label",
        "focal_old_m1_reference_sha256",
        "original_partner_row_id",
        "original_pack_id",
        "partner_row_id",
        "partner_baseline_label",
        "partner_rank_within_label",
        "partner_old_m1_reference_sha256",
        "different_document",
        "different_hidden",
        "selection_sha256",
    }
    expected_calls = int(config["selection"]["expected_pair_calls"])
    if len(schedule) != expected_calls:
        raise ProtocolError(
            f"schedule call count {len(schedule)} != {expected_calls}"
        )
    focal_counts: Counter[str] = Counter()
    focal_partner_label_counts: Counter[tuple[str, str]] = Counter()
    focal_label_by_id: dict[str, str] = {}
    cell_focals: dict[tuple[int, int], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    cells_by_layer: dict[int, set[tuple[int, int]]] = defaultdict(set)
    unordered_pairs: set[tuple[str, str]] = set()
    for index, call in enumerate(schedule):
        _require_exact_fields(call, required, label="schedule call")
        if (
            call["schema_version"] != SCHEDULE_SCHEMA
            or int(call["call_index"]) != index
            or int(call["m"]) != 2
            or call["focal_baseline_label"] not in {"safe", "unsafe"}
            or call["partner_baseline_label"] not in {"safe", "unsafe"}
            or int(call["partner_rank_within_label"]) not in {0, 1}
            or call["different_document"] is not True
            or call["different_hidden"] is not True
            or not is_sha256(call["selection_sha256"])
            or not is_sha256(call["focal_old_m1_reference_sha256"])
            or not is_sha256(call["partner_old_m1_reference_sha256"])
        ):
            raise ProtocolError("schedule call scalar fields are invalid")
        row_ids = call["row_ids"]
        row_mappings = call["row_records"]
        slot = int(call["focal_original_slot"])
        if (
            not isinstance(row_ids, list)
            or len(row_ids) != 2
            or len(set(row_ids)) != 2
            or not isinstance(row_mappings, list)
            or len(row_mappings) != 2
            or slot not in {0, 1}
            or row_ids[slot] != call["focal_row_id"]
            or row_ids[1 - slot] != call["partner_row_id"]
            or call["partner_row_id"] == call["original_partner_row_id"]
        ):
            raise ProtocolError("schedule does not preserve focal slot/partner exclusion")
        records = tuple(
            PILOT.row_record_from_mapping(mapping) for mapping in row_mappings
        )
        if tuple(row.row_id for row in records) != tuple(row_ids):
            raise ProtocolError("schedule row record identity mismatch")
        if any(
            int(row.layer) != int(call["layer"])
            or int(row.expert_id) != int(call["expert_id"])
            for row in records
        ):
            raise ProtocolError("schedule crosses layer/expert cell")
        focal_record = records[slot]
        partner_record = records[1 - slot]
        if (
            focal_record.document_sha256 == partner_record.document_sha256
            or focal_record.hidden_sha256 == partner_record.hidden_sha256
        ):
            raise ProtocolError("schedule partner is not document/hidden-distinct")
        pair = tuple(sorted(row_ids))
        if pair in unordered_pairs:
            raise ProtocolError("schedule repeats an unordered focal/partner pair")
        unordered_pairs.add(pair)

        focal_id = str(call["focal_row_id"])
        focal_label = str(call["focal_baseline_label"])
        partner_label = str(call["partner_baseline_label"])
        prior_label = focal_label_by_id.setdefault(focal_id, focal_label)
        if prior_label != focal_label:
            raise ProtocolError("one focal has inconsistent baseline labels")
        focal_counts[focal_id] += 1
        focal_partner_label_counts[(focal_id, partner_label)] += 1
        cell = (int(call["layer"]), int(call["expert_id"]))
        cell_focals[cell][focal_label].add(focal_id)
        cells_by_layer[cell[0]].add(cell)

        if evidence is not None:
            focal = evidence.get(focal_id)
            partner = evidence.get(str(call["partner_row_id"]))
            if focal is None or partner is None:
                raise ProtocolError("schedule row is absent from source evidence")
            if (
                focal.baseline_label != focal_label
                or partner.baseline_label != partner_label
                or focal.original_slot != slot
                or focal.original_partner_row_id != call["original_partner_row_id"]
                or focal.original_pack_id != call["original_pack_id"]
                or focal.record.identity_payload() != row_mappings[slot]
                or partner.record.identity_payload() != row_mappings[1 - slot]
                or not _partner_is_valid(
                    focal,
                    partner,
                    next(
                        iter(
                            cell_focals[cell][
                                "unsafe" if focal_label == "safe" else "safe"
                            ]
                        ),
                        "",
                    ),
                )
            ):
                # The other-focal exclusion is independently guaranteed by the
                # final per-cell focal sets below; use the direct invariants here.
                if (
                    focal.baseline_label != focal_label
                    or partner.baseline_label != partner_label
                    or focal.original_slot != slot
                    or focal.original_partner_row_id
                    != call["original_partner_row_id"]
                    or focal.original_pack_id != call["original_pack_id"]
                    or focal.record.identity_payload() != row_mappings[slot]
                    or partner.record.identity_payload()
                    != row_mappings[1 - slot]
                    or partner.row_id == focal.original_partner_row_id
                    or partner.record.document_sha256
                    == focal.record.document_sha256
                    or partner.record.hidden_sha256 == focal.record.hidden_sha256
                ):
                    raise ProtocolError("schedule metadata differs from source evidence")

    calls_per_focal = 4
    expected_focals = int(config["selection"]["expected_focals"])
    if len(focal_counts) != expected_focals or set(focal_counts.values()) != {
        calls_per_focal
    }:
        raise ProtocolError("schedule focal multiplicity changed")
    for focal_id in focal_counts:
        for label in ("safe", "unsafe"):
            if focal_partner_label_counts[(focal_id, label)] != 2:
                raise ProtocolError("schedule partner-label multiplicity changed")
    expected_cells = int(config["selection"]["expected_selected_cells"])
    if len(cell_focals) != expected_cells:
        raise ProtocolError("schedule selected-cell count changed")
    for cell, labels in cell_focals.items():
        if len(labels["safe"]) != 1 or len(labels["unsafe"]) != 1:
            raise ProtocolError(f"cell {cell} does not have one focal per label")
        if labels["safe"] & labels["unsafe"]:
            raise ProtocolError("one row is both safe and unsafe focal")
        if evidence is not None:
            focal_pair = labels["safe"] | labels["unsafe"]
            for call in schedule:
                if (
                    (int(call["layer"]), int(call["expert_id"])) == cell
                    and call["partner_row_id"] in focal_pair
                ):
                    raise ProtocolError("one cell focal is reused as the other's partner")
    for layer in range(int(config["selection"]["layer_count"])):
        if len(cells_by_layer.get(layer, set())) != int(
            config["selection"]["cells_per_layer"]
        ):
            raise ProtocolError(f"layer {layer} selected-cell count changed")
    labels = Counter(focal_label_by_id.values())
    if labels != Counter({"safe": 256, "unsafe": 256}):
        raise ProtocolError("schedule focal-label balance changed")
    return {
        "call_count": len(schedule),
        "focal_count": len(focal_counts),
        "selected_cell_count": len(cell_focals),
        "focal_label_counts": dict(sorted(labels.items())),
        "partner_label_call_counts": dict(
            sorted(Counter(call["partner_baseline_label"] for call in schedule).items())
        ),
        "unordered_pair_count": len(unordered_pairs),
    }


def _plan_artifact_hashes(plan_dir: Path) -> dict[str, str]:
    return {
        name: sha256_file(Path(plan_dir) / name)
        for name in ("config.json", "schedule.jsonl", "PLAN.json")
    }


def finalize_plan(
    *,
    config_path: Path,
    source_dir: Path,
    plan_dir: Path,
    source_hashes: Mapping[str, str],
    schedule: Sequence[Mapping[str, Any]],
    selection_summary: Mapping[str, Any],
) -> None:
    plan_dir = Path(plan_dir)
    if plan_dir.exists():
        raise ProtocolError(f"plan directory exists: {plan_dir}")
    plan_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, plan_dir / "config.json")
    (plan_dir / "config.json").chmod(0o444)
    write_jsonl_no_overwrite(plan_dir / "schedule.jsonl", schedule)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "PLANNED_NOT_EXECUTED",
        "evidence_boundary": validate_config(load_json(config_path))[
            "evidence_boundary"
        ],
        "config_sha256": sha256_file(config_path),
        "source_dir_name": Path(source_dir).name,
        "source_artifact_sha256": dict(sorted(source_hashes.items())),
        "schedule_sha256": sha256_file(plan_dir / "schedule.jsonl"),
        "selection_summary": dict(selection_summary),
        "gpu_executed": False,
    }
    write_json_no_overwrite(plan_dir / "PLAN.json", plan)
    write_json_no_overwrite(
        plan_dir / "PLAN_COMPLETE.json",
        {
            "schema_version": PLAN_COMPLETE_SCHEMA,
            "status": "SUCCESS_COMPLETE",
            "artifact_sha256": _plan_artifact_hashes(plan_dir),
            "authority_rule": "absence_of_this_file_means_invalid_or_incomplete_plan",
        },
    )


def load_and_verify_plan(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    plan_dir: Path,
    evidence: Mapping[str, RowEvidence] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_dir = Path(plan_dir).resolve()
    complete = load_json(plan_dir / "PLAN_COMPLETE.json")
    if (
        complete.get("schema_version") != PLAN_COMPLETE_SCHEMA
        or complete.get("status") != "SUCCESS_COMPLETE"
        or complete.get("artifact_sha256") != _plan_artifact_hashes(plan_dir)
    ):
        raise ProtocolError("plan completion/hash authority is invalid")
    if sha256_file(plan_dir / "config.json") != sha256_file(config_path):
        raise ProtocolError("plan config differs from requested config")
    plan = load_json(plan_dir / "PLAN.json")
    schedule = load_jsonl(plan_dir / "schedule.jsonl")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "PLANNED_NOT_EXECUTED"
        or plan.get("config_sha256") != sha256_file(config_path)
        or plan.get("schedule_sha256")
        != sha256_file(plan_dir / "schedule.jsonl")
        or plan.get("evidence_boundary") != config["evidence_boundary"]
        or plan.get("gpu_executed") is not False
    ):
        raise ProtocolError("plan metadata is invalid")
    validate_schedule(config, schedule, evidence=evidence)
    return schedule, plan


def source_bindings(repo_root: Path, config_path: Path) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    paths = (
        Path(__file__).resolve(),
        TEST_PATH.resolve(),
        BASE_RUNNER_PATH.resolve(),
        GPU_EXECUTION_PATH.resolve(),
        CONTRACT_PATH.resolve(),
        Path(config_path).resolve(),
    )
    bindings: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise ProtocolError(f"bound source is absent: {path}")
        try:
            relative = str(path.relative_to(repo_root))
        except ValueError as exc:
            raise ProtocolError(f"bound source escapes repository: {path}") from exc
        bindings[relative] = sha256_file(path)
    return dict(sorted(bindings.items()))


def _verify_acceptance(
    config: Mapping[str, Any], acceptance_path: Path
) -> dict[str, Any]:
    acceptance = PILOT.load_acceptance(Path(acceptance_path))
    if acceptance.get("config_sha256") != config["source"][
        "frozen_pilot_config_sha256"
    ]:
        raise ProtocolError("acceptance is not bound to the source pilot config")
    if acceptance.get("stack", {}).get("stack_digest") != config["source"][
        "stack_digest"
    ]:
        raise ProtocolError("acceptance stack differs from the source calibration stack")
    return acceptance


def create_lock(
    *,
    config_path: Path,
    repo_root: Path,
    source_dir: Path,
    plan_dir: Path,
    acceptance_path: Path,
) -> dict[str, Any]:
    config = validate_config(load_json(config_path))
    source_hashes = verify_source_artifacts(config, source_dir)
    references = load_reference_hashes(config, source_dir)
    evidence = load_m2_evidence(config, source_dir)
    schedule, plan = load_and_verify_plan(
        config=config,
        config_path=config_path,
        plan_dir=plan_dir,
        evidence=evidence,
    )
    if any(row_id not in references for call in schedule for row_id in call["row_ids"]):
        raise ProtocolError("sealed schedule lacks source M1 references")
    acceptance = _verify_acceptance(config, acceptance_path)
    plan_hashes = _plan_artifact_hashes(plan_dir) | {
        "PLAN_COMPLETE.json": sha256_file(Path(plan_dir) / "PLAN_COMPLETE.json")
    }
    payload = {
        "schema_version": LOCK_SCHEMA,
        "status": "SEALED_BEFORE_GPU_EXECUTION",
        "config_sha256": sha256_file(config_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "acceptance_complete_sha256": sha256_file(
            Path(acceptance_path).parent / "ACCEPTANCE_COMPLETE.json"
        ),
        "stack_digest": acceptance["stack"]["stack_digest"],
        "source_artifact_sha256": source_hashes,
        "plan_artifact_sha256": dict(sorted(plan_hashes.items())),
        "plan_schedule_sha256": plan["schedule_sha256"],
        "source_bindings": source_bindings(repo_root, config_path),
        "frozen_constants": {
            "m": 2,
            "warmups": 3,
            "repeats": 10,
            "selected_cells": 256,
            "focals": 512,
            "pair_calls": 2048,
            "max_gpu_seconds": 600,
            "preserve_focal_original_slot": True,
        },
    }
    return payload | {"lock_sha256": canonical_sha256(payload)}


def verify_lock(
    lock: Mapping[str, Any],
    *,
    config_path: Path,
    repo_root: Path,
    source_dir: Path,
    plan_dir: Path,
    acceptance_path: Path,
) -> dict[str, Any]:
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != "SEALED_BEFORE_GPU_EXECUTION"
    ):
        raise ProtocolError("partner lock schema/status mismatch")
    payload = dict(lock)
    declared_digest = payload.pop("lock_sha256", None)
    if not is_sha256(declared_digest) or canonical_sha256(payload) != declared_digest:
        raise ProtocolError("partner lock content hash mismatch")
    config = validate_config(load_json(config_path))
    acceptance = _verify_acceptance(config, acceptance_path)
    expected_plan_hashes = _plan_artifact_hashes(plan_dir) | {
        "PLAN_COMPLETE.json": sha256_file(Path(plan_dir) / "PLAN_COMPLETE.json")
    }
    checks = {
        "config_sha256": sha256_file(config_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "acceptance_complete_sha256": sha256_file(
            Path(acceptance_path).parent / "ACCEPTANCE_COMPLETE.json"
        ),
        "stack_digest": acceptance["stack"]["stack_digest"],
        "source_artifact_sha256": verify_source_artifacts(config, source_dir),
        "plan_artifact_sha256": dict(sorted(expected_plan_hashes.items())),
        "source_bindings": source_bindings(repo_root, config_path),
    }
    for key, expected in checks.items():
        if lock.get(key) != expected:
            raise ProtocolError(f"partner lock binding changed: {key}")
    evidence = load_m2_evidence(config, source_dir)
    schedule, plan = load_and_verify_plan(
        config=config,
        config_path=config_path,
        plan_dir=plan_dir,
        evidence=evidence,
    )
    if lock.get("plan_schedule_sha256") != plan["schedule_sha256"]:
        raise ProtocolError("partner lock schedule binding changed")
    if len(schedule) != int(config["selection"]["expected_pair_calls"]):
        raise ProtocolError("partner lock schedule denominator changed")
    return dict(lock)


def _copy_read_only(source: Path, target: Path) -> str:
    shutil.copyfile(source, target)
    target.chmod(0o444)
    return sha256_file(target)


def snapshot_inputs(
    *,
    output_dir: Path,
    config_path: Path,
    repo_root: Path,
    source_dir: Path,
    plan_dir: Path,
    acceptance_path: Path,
    lock_path: Path,
) -> dict[str, str]:
    snapshot = Path(output_dir) / "frozen_inputs"
    snapshot.mkdir()
    sources = {
        "config.json": Path(config_path),
        "ACCEPTANCE.json": Path(acceptance_path),
        "ACCEPTANCE_COMPLETE.json": Path(acceptance_path).parent
        / "ACCEPTANCE_COMPLETE.json",
        "FROZEN_RUN_LOCK.json": Path(lock_path),
        "schedule.jsonl": Path(plan_dir) / "schedule.jsonl",
        "PLAN.json": Path(plan_dir) / "PLAN.json",
        "PLAN_COMPLETE.json": Path(plan_dir) / "PLAN_COMPLETE.json",
        "SOURCE_COMPLETE.json": Path(source_dir) / "COMPLETE.json",
        "SOURCE_CALIBRATION_WORKER_STATUS.json": Path(source_dir)
        / "calibration_worker_status.json",
    }
    for relative in source_bindings(repo_root, config_path):
        path = Path(repo_root) / relative
        sources[relative.replace("/", "__")] = path
    result: dict[str, str] = {}
    for name, source in sorted(sources.items()):
        result[name] = _copy_read_only(source.resolve(), snapshot / name)
    return result


def _worker_command(
    args: argparse.Namespace, *, deadline_epoch: float
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--config",
        str(Path(args.config).resolve()),
        "--repo-root",
        str(Path(args.repo_root).resolve()),
        "--source-dir",
        str(Path(args.source_dir).resolve()),
        "--plan-dir",
        str(Path(args.plan_dir).resolve()),
        "--acceptance-artifact",
        str(Path(args.acceptance_artifact).resolve()),
        "--frozen-lock",
        str(Path(args.frozen_lock).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--deadline-epoch",
        str(deadline_epoch),
    ]
    if args.model_path:
        command.extend(("--model-path", str(Path(args.model_path).resolve())))
    return command


def _load_live_model(
    *,
    config: Mapping[str, Any],
    source_dir: Path,
    acceptance_path: Path,
    model_path_override: str | None,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    acceptance = _verify_acceptance(config, acceptance_path)
    original_config_path = Path(source_dir) / "frozen_inputs" / "config.json"
    original_config = PILOT.validate_config(PILOT.load_json(original_config_path))
    identity = PILOT._nvidia_identity()
    if identity != acceptance["stack"]["gpu"]:
        raise ProtocolError("live GPU identity differs from acceptance")
    PILOT.assert_clean_gpu(identity["uuid"], allowed_pids=set())
    resolved_model = PILOT.resolve_model_path(original_config, model_path_override)
    PILOT._set_math_state()
    model, _tokenizer = PILOT.load_model(original_config, resolved_model)
    observed_stack = PILOT.observe_stack(model)
    if observed_stack != acceptance["stack"]:
        raise ProtocolError("live stack differs from acceptance")
    PILOT.assert_clean_gpu(identity["uuid"], allowed_pids={os.getpid()})
    return original_config, acceptance, model


def worker_run(args: argparse.Namespace) -> int:
    PILOT.assert_numeric_logging_disabled()
    import torch

    config_path = Path(args.config).resolve()
    repo_root = Path(args.repo_root).resolve()
    source_dir = Path(args.source_dir).resolve()
    plan_dir = Path(args.plan_dir).resolve()
    acceptance_path = Path(args.acceptance_artifact).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = validate_config(load_json(config_path))
    verify_lock(
        load_json(Path(args.frozen_lock).resolve()),
        config_path=config_path,
        repo_root=repo_root,
        source_dir=source_dir,
        plan_dir=plan_dir,
        acceptance_path=acceptance_path,
    )
    if time.time() >= float(args.deadline_epoch):
        raise TimeoutError("partner-permutation deadline reached before worker start")
    evidence = load_m2_evidence(config, source_dir)
    old_references = load_reference_hashes(config, source_dir)
    schedule, _plan = load_and_verify_plan(
        config=config,
        config_path=config_path,
        plan_dir=plan_dir,
        evidence=evidence,
    )
    original_config, acceptance, model = _load_live_model(
        config=config,
        source_dir=source_dir,
        acceptance_path=acceptance_path,
        model_path_override=args.model_path,
    )
    gpu = PILOT._gpu()
    capture_path = source_dir / "calibration_captures.pt"
    if sha256_file(capture_path) != config["source"]["calibration_captures_sha256"]:
        raise ProtocolError("calibration capture changed before torch.load")
    captures = torch.load(capture_path, map_location="cpu", weights_only=False)
    rows = gpu.materialize_routed_rows(captures)
    if len(rows) != int(config["source"]["expected_reference_rows"]):
        raise ProtocolError("materialized capture denominator changed")
    materialized = {row.row_id: row for row in rows}
    scheduled_ids = sorted(
        {row_id for call in schedule for row_id in call["row_ids"]}
    )
    if any(row_id not in materialized for row_id in scheduled_ids):
        raise ProtocolError("scheduled row is absent from materialized captures")
    selected_rows = tuple(materialized[row_id] for row_id in scheduled_ids)
    packs: list[Any] = []
    for call in schedule:
        ordered = tuple(materialized[row_id].record for row_id in call["row_ids"])
        if [row.identity_payload() for row in ordered] != call["row_records"]:
            raise ProtocolError("materialized row differs from sealed schedule")
        packs.append(
            CONTRACT.Pack(
                layer=int(call["layer"]),
                expert_id=int(call["expert_id"]),
                rows=ordered,
            )
        )

    binding_context = PILOT.descriptor_binding_context(original_config, acceptance)
    call_plan = gpu.calibration_call_plan(packs)
    calls = [
        PILOT.planned_call_record(
            call, call_index=index, binding_context=binding_context
        )
        for index, call in enumerate(call_plan.calls)
    ]
    for index, call in enumerate(calls):
        PILOT.validate_call_record(
            call, expected_index=index, binding_context=binding_context
        )
        if call["row_ids"] != schedule[index]["row_ids"] or int(call["m"]) != 2:
            raise ProtocolError("pre-call ledger differs from sealed schedule")
    write_jsonl_no_overwrite(output_dir / "partner_calls.jsonl", calls)
    write_json_no_overwrite(
        output_dir / "partner_pre_call_seal.json",
        {
            "schema_version": "semanticfence-partner-pre-call-seal-v1",
            "status": "SEALED_BEFORE_EXECUTION",
            "call_count": len(calls),
            "calls_sha256": sha256_file(output_dir / "partner_calls.jsonl"),
            "schedule_sha256": sha256_file(plan_dir / "schedule.jsonl"),
            "binding_context_sha256": canonical_sha256(binding_context),
        },
    )
    execution = gpu.execute_calibration(
        model=model,
        packs=tuple(packs),
        rows=selected_rows,
        repeats=int(config["execution"]["repeats"]),
    )
    raw_path = output_dir / "partner_raw_outputs.pt"
    if raw_path.exists():
        raise ProtocolError("partner raw-output path already exists")
    torch.save(execution, raw_path)

    new_reference_rows = {
        row.row_id: row for row in execution.reference.rows
    }
    if set(new_reference_rows) != set(scheduled_ids):
        raise ProtocolError("new M1 reference coverage differs from schedule")
    mismatched_reference_ids: list[str] = []
    for row_id in scheduled_ids:
        row = new_reference_rows[row_id]
        if not row.bitwise_stable or not row.all_exact_to_reference:
            raise ProtocolError("new M1 reference is not 10/10 stable")
        if row.reference_sha256 != old_references[row_id]:
            mismatched_reference_ids.append(row_id)
    if mismatched_reference_ids:
        raise ProtocolError(
            f"new M1 reference differs from run03 for {len(mismatched_reference_ids)} rows"
        )
    write_jsonl_no_overwrite(
        output_dir / "partner_reference_rows.jsonl",
        [PILOT.row_execution_record(new_reference_rows[row_id]) for row_id in scheduled_ids],
    )
    write_json_no_overwrite(
        output_dir / "reference_verification.json",
        {
            "schema_version": "semanticfence-partner-reference-verification-v1",
            "status": "ALL_MATCH",
            "scheduled_unique_row_count": len(scheduled_ids),
            "new_reference_all_stable": True,
            "old_reference_hash_match_count": len(scheduled_ids),
            "old_reference_mismatch_count": 0,
            "source_reference_sha256": config["source"][
                "calibration_reference_rows_sha256"
            ],
        },
    )

    numeric: list[dict[str, Any]] = []
    if len(execution.packs) != len(schedule):
        raise ProtocolError("execution pack denominator differs from schedule")
    for index, (call, observed) in enumerate(zip(schedule, execution.packs)):
        if observed.pack.pack_id != packs[index].pack_id:
            raise ProtocolError("execution pack identity differs from schedule")
        slot = int(call["focal_original_slot"])
        numeric.append(
            {
                "schema_version": NUMERIC_SCHEMA,
                "call_index": index,
                "pack_id": observed.pack.pack_id,
                "layer": int(call["layer"]),
                "expert_id": int(call["expert_id"]),
                "m": 2,
                "row_ids": list(call["row_ids"]),
                "focal_row_id": call["focal_row_id"],
                "focal_original_slot": slot,
                "focal_baseline_label": call["focal_baseline_label"],
                "partner_row_id": call["partner_row_id"],
                "partner_baseline_label": call["partner_baseline_label"],
                "focal_repeat_exact": [
                    bool(repeat[slot]) for repeat in observed.repeat_row_exact
                ],
                "partner_repeat_exact": [
                    bool(repeat[1 - slot]) for repeat in observed.repeat_row_exact
                ],
                "focal_repeat_sha256": [
                    repeat[slot] for repeat in observed.repeat_row_sha256
                ],
                "partner_repeat_sha256": [
                    repeat[1 - slot] for repeat in observed.repeat_row_sha256
                ],
                "representative_full_output_sha256": observed.representative_full_output_sha256,
            }
        )
    write_jsonl_no_overwrite(output_dir / "partner_numeric.jsonl", numeric)
    write_json_no_overwrite(
        output_dir / "worker_status.json",
        {
            "schema_version": "semanticfence-partner-worker-status-v1",
            "status": "COMPLETE",
            "stack_digest": acceptance["stack"]["stack_digest"],
            "pair_call_count": len(numeric),
            "scheduled_unique_row_count": len(scheduled_ids),
            "partner_calls_sha256": sha256_file(output_dir / "partner_calls.jsonl"),
            "partner_numeric_sha256": sha256_file(output_dir / "partner_numeric.jsonl"),
            "partner_reference_rows_sha256": sha256_file(
                output_dir / "partner_reference_rows.jsonl"
            ),
            "reference_verification_sha256": sha256_file(
                output_dir / "reference_verification.json"
            ),
            "partner_raw_outputs_sha256": sha256_file(raw_path),
            "pre_call_seal_sha256": sha256_file(
                output_dir / "partner_pre_call_seal.json"
            ),
        },
    )
    PILOT.assert_clean_gpu(
        acceptance["stack"]["gpu"]["uuid"], allowed_pids={os.getpid()}
    )
    return 0


def _status_from_flags(flags: Sequence[bool]) -> str:
    if len(flags) != 10 or any(type(value) is not bool for value in flags):
        raise ProtocolError("partner outcome does not contain 10 boolean repeats")
    if all(flags):
        return "safe"
    if not any(flags):
        return "unsafe"
    return "mixed"


def summarize_partner_outcomes(
    config: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    numeric: Sequence[Mapping[str, Any]],
    reference_verification: Mapping[str, Any],
) -> dict[str, Any]:
    validate_schedule(config, schedule)
    if len(numeric) != len(schedule):
        raise ProtocolError("numeric/schedule denominators differ")
    required = {
        "schema_version",
        "call_index",
        "pack_id",
        "layer",
        "expert_id",
        "m",
        "row_ids",
        "focal_row_id",
        "focal_original_slot",
        "focal_baseline_label",
        "partner_row_id",
        "partner_baseline_label",
        "focal_repeat_exact",
        "partner_repeat_exact",
        "focal_repeat_sha256",
        "partner_repeat_sha256",
        "representative_full_output_sha256",
    }
    if (
        reference_verification.get("status") != "ALL_MATCH"
        or reference_verification.get("new_reference_all_stable") is not True
        or int(reference_verification.get("old_reference_mismatch_count", -1)) != 0
    ):
        raise ProtocolError("M1 reference verification is not closed")

    stable_flip_calls: list[int] = []
    mixed_calls: list[int] = []
    stable_flip_focals: set[str] = set()
    mixed_focals: set[str] = set()
    pair_status_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    focal_any_flip: dict[str, bool] = defaultdict(bool)
    focal_any_mixed: dict[str, bool] = defaultdict(bool)
    for index, (call, value) in enumerate(zip(schedule, numeric)):
        _require_exact_fields(value, required, label="partner numeric")
        comparable = (
            value["schema_version"] == NUMERIC_SCHEMA
            and int(value["call_index"]) == index
            and int(value["layer"]) == int(call["layer"])
            and int(value["expert_id"]) == int(call["expert_id"])
            and int(value["m"]) == 2
            and value["row_ids"] == call["row_ids"]
            and value["focal_row_id"] == call["focal_row_id"]
            and int(value["focal_original_slot"])
            == int(call["focal_original_slot"])
            and value["focal_baseline_label"] == call["focal_baseline_label"]
            and value["partner_row_id"] == call["partner_row_id"]
            and value["partner_baseline_label"]
            == call["partner_baseline_label"]
            and is_sha256(value["pack_id"])
            and is_sha256(value["representative_full_output_sha256"])
            and isinstance(value["focal_repeat_sha256"], list)
            and len(value["focal_repeat_sha256"]) == 10
            and all(is_sha256(item) for item in value["focal_repeat_sha256"])
            and isinstance(value["partner_repeat_sha256"], list)
            and len(value["partner_repeat_sha256"]) == 10
            and all(is_sha256(item) for item in value["partner_repeat_sha256"])
        )
        if not comparable:
            raise ProtocolError("numeric record differs from sealed schedule")
        status = _status_from_flags(value["focal_repeat_exact"])
        _status_from_flags(value["partner_repeat_exact"])
        focal_id = str(call["focal_row_id"])
        baseline = str(call["focal_baseline_label"])
        stratum = f"focal_{baseline}__partner_{call['partner_baseline_label']}"
        stratum_counts[stratum] += 1
        pair_status_counts[f"baseline_{baseline}__new_{status}"] += 1
        if status == "mixed":
            mixed_calls.append(index)
            mixed_focals.add(focal_id)
            focal_any_mixed[focal_id] = True
        elif status != baseline:
            stable_flip_calls.append(index)
            stable_flip_focals.add(focal_id)
            focal_any_flip[focal_id] = True

    if stable_flip_calls:
        decision = "FALSIFY_PARTNER_INVARIANCE"
    elif mixed_calls:
        decision = "WEAKEN_ROW_STABILITY"
    else:
        decision = "SUPPORT_CALIBRATION_ONLY"
    focal_labels = {
        str(call["focal_row_id"]): str(call["focal_baseline_label"])
        for call in schedule
    }
    per_label: dict[str, Any] = {}
    for label in ("safe", "unsafe"):
        ids = sorted(row_id for row_id, value in focal_labels.items() if value == label)
        flips = sum(bool(focal_any_flip[row_id]) for row_id in ids)
        mixed = sum(bool(focal_any_mixed[row_id]) for row_id in ids)
        upper = None
        if flips == 0 and ids:
            upper = 1.0 - math.pow(0.05, 1.0 / len(ids))
        per_label[label] = {
            "focal_count": len(ids),
            "stable_flip_focal_count": flips,
            "mixed_focal_count": mixed,
            "one_sided_95pct_zero_flip_upper_bound": upper,
        }
    return {
        "schema_version": RESULT_SCHEMA,
        "decision": decision,
        "paper_result": False,
        "evidence_complete": True,
        "claim": config["decision"]["claim"],
        "evidence_boundary": config["evidence_boundary"],
        "pair_call_count": len(schedule),
        "focal_count": len(focal_labels),
        "stable_flip_call_count": len(stable_flip_calls),
        "stable_flip_focal_count": len(stable_flip_focals),
        "mixed_call_count": len(mixed_calls),
        "mixed_focal_count": len(mixed_focals),
        "stable_flip_call_indices": stable_flip_calls,
        "mixed_call_indices": mixed_calls,
        "pair_status_counts": dict(sorted(pair_status_counts.items())),
        "stratum_call_counts": dict(sorted(stratum_counts.items())),
        "per_focal_label": per_label,
        "reference_verification": dict(reference_verification),
        "interpretation": (
            "This result tests reused calibration rows at fixed layer, expert, "
            "BF16 M=2 shape and original focal slot. It is not fresh evaluation, "
            "a pure row-only claim, latency, full-layer, serving, EP, or paper evidence."
        ),
    }


def finalize_complete(output_dir: Path, result: Mapping[str, Any]) -> None:
    output_dir = Path(output_dir)
    if (output_dir / "COMPLETE.json").exists():
        raise ProtocolError("partner completion sentinel already exists")
    write_json_no_overwrite(output_dir / "PARTNER_RESULT.json", result)
    artifacts = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "COMPLETE.json"
    }
    write_json_no_overwrite(
        output_dir / "COMPLETE.json",
        {
            "schema_version": COMPLETE_SCHEMA,
            "status": "SUCCESS_COMPLETE",
            "partner_result_sha256": sha256_file(
                output_dir / "PARTNER_RESULT.json"
            ),
            "artifact_sha256": artifacts,
            "authority_rule": "absence_of_this_file_means_invalid_or_incomplete",
        },
    )


def run_plan(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    source_dir = Path(args.source_dir).resolve()
    plan_dir = Path(args.plan_dir).resolve()
    config = validate_config(load_json(config_path))
    source_hashes = verify_source_artifacts(config, source_dir)
    references = load_reference_hashes(config, source_dir)
    evidence = load_m2_evidence(config, source_dir)
    schedule, summary = select_partner_schedule(config, evidence, references)
    finalize_plan(
        config_path=config_path,
        source_dir=source_dir,
        plan_dir=plan_dir,
        source_hashes=source_hashes,
        schedule=schedule,
        selection_summary=summary,
    )
    print(
        json.dumps(
            {
                "plan_dir": str(plan_dir),
                "schedule_sha256": sha256_file(plan_dir / "schedule.jsonl"),
                "pair_call_count": len(schedule),
                "focal_count": summary["focal_count"],
                "unique_scheduled_row_count": summary[
                    "unique_scheduled_row_count"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_seal(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    if output.exists():
        raise ProtocolError(f"lock output exists: {output}")
    lock = create_lock(
        config_path=Path(args.config).resolve(),
        repo_root=Path(args.repo_root).resolve(),
        source_dir=Path(args.source_dir).resolve(),
        plan_dir=Path(args.plan_dir).resolve(),
        acceptance_path=Path(args.acceptance_artifact).resolve(),
    )
    write_json_no_overwrite(output, lock)
    return 0


def run_parent(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    repo_root = Path(args.repo_root).resolve()
    source_dir = Path(args.source_dir).resolve()
    plan_dir = Path(args.plan_dir).resolve()
    acceptance_path = Path(args.acceptance_artifact).resolve()
    lock_path = Path(args.frozen_lock).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ProtocolError(f"run output directory exists: {output_dir}")
    config = validate_config(load_json(config_path))
    lock = verify_lock(
        load_json(lock_path),
        config_path=config_path,
        repo_root=repo_root,
        source_dir=source_dir,
        plan_dir=plan_dir,
        acceptance_path=acceptance_path,
    )
    evidence = load_m2_evidence(config, source_dir)
    schedule, _plan = load_and_verify_plan(
        config=config,
        config_path=config_path,
        plan_dir=plan_dir,
        evidence=evidence,
    )
    acceptance = _verify_acceptance(config, acceptance_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = snapshot_inputs(
        output_dir=output_dir,
        config_path=config_path,
        repo_root=repo_root,
        source_dir=source_dir,
        plan_dir=plan_dir,
        acceptance_path=acceptance_path,
        lock_path=lock_path,
    )
    deadline_epoch = time.time() + int(config["execution"]["max_gpu_seconds"])
    write_json_no_overwrite(
        output_dir / "run_request.json",
        {
            "schema_version": "semanticfence-partner-run-request-v1",
            "started_at_epoch": time.time(),
            "deadline_epoch": deadline_epoch,
            "evidence_boundary": config["evidence_boundary"],
            "config_sha256": sha256_file(config_path),
            "lock_sha256": lock["lock_sha256"],
            "schedule_sha256": sha256_file(plan_dir / "schedule.jsonl"),
            "source_artifact_sha256": lock["source_artifact_sha256"],
            "snapshot_sha256": snapshot,
        },
    )
    command = _worker_command(args, deadline_epoch=deadline_epoch)
    PILOT.run_worker_monitored(
        command=command,
        log_path=output_dir / "worker.log",
        expected_gpu_uuid=acceptance["stack"]["gpu"]["uuid"],
        deadline_epoch=deadline_epoch,
    )
    worker_status = load_json(output_dir / "worker_status.json")
    if worker_status.get("status") != "COMPLETE":
        raise ProtocolError("partner worker did not complete")
    declared_hashes = {
        "partner_calls_sha256": "partner_calls.jsonl",
        "partner_numeric_sha256": "partner_numeric.jsonl",
        "partner_reference_rows_sha256": "partner_reference_rows.jsonl",
        "reference_verification_sha256": "reference_verification.json",
        "partner_raw_outputs_sha256": "partner_raw_outputs.pt",
        "pre_call_seal_sha256": "partner_pre_call_seal.json",
    }
    for key, name in declared_hashes.items():
        if worker_status.get(key) != sha256_file(output_dir / name):
            raise ProtocolError(f"worker artifact hash mismatch: {name}")
    numeric = load_jsonl(output_dir / "partner_numeric.jsonl")
    reference_verification = load_json(output_dir / "reference_verification.json")
    result = summarize_partner_outcomes(
        config, schedule, numeric, reference_verification
    ) | {
        "stack_digest": acceptance["stack"]["stack_digest"],
        "schedule_sha256": sha256_file(plan_dir / "schedule.jsonl"),
        "source_calibration_numeric_sha256": config["source"][
            "calibration_numeric_sha256"
        ],
        "worker_status_sha256": sha256_file(output_dir / "worker_status.json"),
        "within_gpu_budget": time.time() <= deadline_epoch,
    }
    if result["within_gpu_budget"] is not True:
        raise ProtocolError("partner run exceeded the frozen GPU deadline")
    finalize_complete(output_dir, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--source-dir", required=True)
    plan.add_argument("--plan-dir", required=True)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--config", required=True)
    seal.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    seal.add_argument("--source-dir", required=True)
    seal.add_argument("--plan-dir", required=True)
    seal.add_argument("--acceptance-artifact", required=True)
    seal.add_argument("--output", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    run.add_argument("--source-dir", required=True)
    run.add_argument("--plan-dir", required=True)
    run.add_argument("--acceptance-artifact", required=True)
    run.add_argument("--frozen-lock", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--model-path")

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True)
    worker.add_argument("--repo-root", required=True)
    worker.add_argument("--source-dir", required=True)
    worker.add_argument("--plan-dir", required=True)
    worker.add_argument("--acceptance-artifact", required=True)
    worker.add_argument("--frozen-lock", required=True)
    worker.add_argument("--output-dir", required=True)
    worker.add_argument("--deadline-epoch", required=True, type=float)
    worker.add_argument("--model-path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "plan":
        return run_plan(args)
    if args.command == "seal":
        return run_seal(args)
    if args.command == "_worker":
        return worker_run(args)
    return run_parent(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProtocolError, PILOT.ProtocolError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        raise SystemExit(2)
