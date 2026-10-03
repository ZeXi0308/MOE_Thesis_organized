#!/usr/bin/env python3
"""Compare OLMoE pre-top-k router logits at serial and fixed batch width.

This is a development-only execution-conformance diagnostic.  It replays the
frozen serial-audit requests from an existing continuous-decode capture, binds
the replay to the capture's workload snapshot and token ledger, and records
native router logits before top-k selection.  It does not measure safe
capacity, execute a controller action, or authorize either kind of claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from capture_dev_continuous_decode import PRODUCER


EXPECTED_MODEL = {
    "id": "allenai/OLMoE-1B-7B-0924",
    "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    "tokenizer_revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    "dtype": "bfloat16",
}
ALLCLOSE_ATOL = 1e-6
ALLCLOSE_RTOL = 1e-5
NEAR_TIE_MARGIN = 1e-2
GPU_POLL_INTERVAL_SECONDS = 0.20
CAPTURE_FILES = {
    "routes.csv",
    "decode_batches.jsonl",
    "request_ledger.jsonl",
    "workload_manifest.json",
    "preregistration.json",
    "environment.json",
    "serial_audit.json",
}


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PRODUCER.ProtocolError(f"{name} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PRODUCER.ProtocolError(f"cannot read {path}: {exc}") from exc
    return dict(_require_mapping(value, str(path)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PRODUCER.ProtocolError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_captured_file(
    capture_dir: Path,
    complete: Mapping[str, Any],
    name: str,
) -> Path:
    path = capture_dir / name
    files = _require_mapping(complete.get("files"), "CAPTURE_COMPLETE.files")
    expected = str(files.get(name, ""))
    if len(expected) != 64:
        raise PRODUCER.ProtocolError(f"capture sentinel does not bind {name}")
    if _sha256_file(path) != expected:
        raise PRODUCER.ProtocolError(f"captured {name} SHA-256 mismatch")
    return path


def load_capture_contract(
    capture_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Validate the completed capture and return its exact workload snapshot."""

    if not capture_dir.is_dir():
        raise PRODUCER.ProtocolError(f"capture directory does not exist: {capture_dir}")
    run_status = _read_json(capture_dir / "RUN_STATUS.json")
    if run_status != {
        "required_sentinel": "CAPTURE_COMPLETE.json",
        "status": "COMPLETE",
    }:
        raise PRODUCER.ProtocolError("capture RUN_STATUS is not the closed COMPLETE marker")
    complete = _read_json(capture_dir / "CAPTURE_COMPLETE.json")
    if (
        complete.get("schema") != "bcrd-continuous-capture-complete-v1"
        or complete.get("status") != "CAPTURE_COMPLETE"
        or complete.get("run_class") != "development"
    ):
        raise PRODUCER.ProtocolError("capture is not a completed development capture")
    files = _require_mapping(complete.get("files"), "CAPTURE_COMPLETE.files")
    if set(files) != CAPTURE_FILES:
        raise PRODUCER.ProtocolError("capture does not satisfy the seven-file contract")
    verified_paths = {
        name: _require_captured_file(capture_dir, complete, name)
        for name in sorted(CAPTURE_FILES)
    }
    serial_audit = _read_json(verified_paths["serial_audit.json"])
    if complete.get("serial_audit") != serial_audit:
        raise PRODUCER.ProtocolError(
            "capture sentinel serial audit differs from the sealed payload"
        )

    manifest_path = verified_paths["workload_manifest.json"]
    manifest_hash = _sha256_file(manifest_path)
    if str(complete.get("workload_manifest_sha256", "")) != manifest_hash:
        raise PRODUCER.ProtocolError("capture sentinel workload hash does not close")
    manifest = PRODUCER.load_workload_manifest(manifest_path)
    if manifest.get("run_class") != "development":
        raise PRODUCER.ProtocolError("router-logit diagnostic is development-only")
    marker = _require_mapping(
        manifest.get("route_capacity_envelope"),
        "workload.route_capacity_envelope",
    )
    if marker.get("serial_route_identity_semantics") != (
        "per_layer_expert_assignment_multiset"
    ):
        raise PRODUCER.ProtocolError("workload does not freeze the v2 route audit")
    if (
        marker.get("episode_id") != "olmoe-dev-steady"
        or marker.get("arrival_regime") != "steady"
    ):
        raise PRODUCER.ProtocolError("diagnostic requires the frozen steady episode")
    return manifest, complete, manifest_path


def load_captured_environment(
    capture_dir: Path, complete: Mapping[str, Any]
) -> dict[str, Any]:
    return _read_json(_require_captured_file(capture_dir, complete, "environment.json"))


def select_request_ids(manifest: Mapping[str, Any], request_count: int) -> list[str]:
    frozen = manifest.get("serial_audit_request_ids")
    if not isinstance(frozen, list) or request_count > len(frozen):
        raise PRODUCER.ProtocolError(
            "--requests exceeds frozen serial_audit_request_ids"
        )
    selected = [str(value) for value in frozen[:request_count]]
    if len(selected) != request_count or len(set(selected)) != request_count:
        raise PRODUCER.ProtocolError("selected serial audit request IDs are not unique")
    known = {str(value["request_id"]) for value in manifest["requests"]}
    if not set(selected).issubset(known):
        raise PRODUCER.ProtocolError("selected serial audit request is absent from workload")
    return selected


def load_reference_tokens(
    capture_dir: Path,
    complete: Mapping[str, Any],
    request_ids: Sequence[str],
    decode_steps: int,
) -> list[dict[str, Any]]:
    """Load token state from the hash-bound original batched capture."""

    ledger_path = _require_captured_file(capture_dir, complete, "request_ledger.jsonl")
    rows: dict[str, Mapping[str, Any]] = {}
    try:
        with ledger_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                row = _require_mapping(value, f"request ledger line {line_number}")
                request_id = str(row.get("request_id", ""))
                if not request_id or request_id in rows:
                    raise PRODUCER.ProtocolError(
                        "request ledger IDs are empty or duplicated"
                    )
                rows[request_id] = row
    except (OSError, json.JSONDecodeError) as exc:
        raise PRODUCER.ProtocolError(f"cannot parse request ledger: {exc}") from exc

    tokens: list[dict[str, Any]] = []
    for request_id in request_ids:
        row = rows.get(request_id)
        if row is None:
            raise PRODUCER.ProtocolError(
                f"request ledger is missing selected request {request_id}"
            )
        steps = row.get("steps")
        if not isinstance(steps, list) or len(steps) < decode_steps:
            raise PRODUCER.ProtocolError(
                f"captured request {request_id} has fewer than {decode_steps} steps"
            )
        for expected_step, raw in enumerate(steps[:decode_steps]):
            step = _require_mapping(raw, f"ledger {request_id} step {expected_step}")
            if int(step.get("decode_step", -1)) != expected_step:
                raise PRODUCER.ProtocolError("request ledger decode-step identity drifted")
            tokens.append(
                {
                    "request_id": request_id,
                    "decode_step": expected_step,
                    "input_token_id": int(step["input_token_id"]),
                    "predicted_next_token_id": int(step["predicted_next_token_id"]),
                }
            )
    return tokens


def _row_index(
    rows: Iterable[Mapping[str, Any]], fields: Sequence[str], label: str
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise PRODUCER.ProtocolError(f"{label} contains duplicate key {key!r}")
        result[key] = row
    return result


def _close(left: float, right: float, *, atol: float, rtol: float) -> bool:
    return abs(left - right) <= atol + rtol * abs(right)


def _float_metrics(
    pairs: Iterable[tuple[float, float]], *, atol: float, rtol: float
) -> dict[str, Any]:
    count = 0
    exact = 0
    close = 0
    delta_sum = 0.0
    max_delta = 0.0
    for left, right in pairs:
        left_value = float(left)
        right_value = float(right)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            raise PRODUCER.ProtocolError("router diagnostic encountered non-finite value")
        delta = abs(left_value - right_value)
        count += 1
        exact += int(left_value == right_value)
        close += int(_close(left_value, right_value, atol=atol, rtol=rtol))
        delta_sum += delta
        max_delta = max(max_delta, delta)
    return {
        "count": count,
        "exact_match_fraction": exact / max(1, count),
        "allclose_match_fraction": close / max(1, count),
        "max_abs_delta": max_delta,
        "mean_abs_delta": delta_sum / max(1, count),
    }


def _swapped_expert_crossing(
    left_logits: Sequence[float],
    right_logits: Sequence[float],
    left_experts: Sequence[int],
    right_experts: Sequence[int],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Bind an assignment change to the logits of the experts that swapped.

    A change elsewhere in the expert vector is not explanatory.  For a valid
    top-k membership change, at least one expert selected only on the left must
    cross one expert selected only on the right between the two arms.
    """

    dropped = sorted(set(int(value) for value in left_experts) - set(right_experts))
    gained = sorted(set(int(value) for value in right_experts) - set(left_experts))
    if len(dropped) != len(gained):
        raise PRODUCER.ProtocolError("top-k assignment cardinality changed")
    crossings: list[dict[str, Any]] = []
    for dropped_expert in dropped:
        for gained_expert in gained:
            left_gap = float(left_logits[dropped_expert]) - float(
                left_logits[gained_expert]
            )
            right_gap = float(right_logits[dropped_expert]) - float(
                right_logits[gained_expert]
            )
            if left_gap >= 0.0 and right_gap <= 0.0:
                crossings.append(
                    {
                        "dropped_expert": dropped_expert,
                        "gained_expert": gained_expert,
                        "left_dropped_minus_gained": left_gap,
                        "right_dropped_minus_gained": right_gap,
                        "gap_change_abs": abs(left_gap - right_gap),
                        "material_gap_change": not _close(
                            left_gap, right_gap, atol=atol, rtol=rtol
                        ),
                    }
                )
    return {
        "dropped_experts": dropped,
        "gained_experts": gained,
        "has_order_crossing": bool(crossings),
        "has_material_gap_change": any(
            bool(row["material_gap_change"]) for row in crossings
        ),
        "crossings": crossings,
    }


def summarize_token_pair(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = ("request_id", "decode_step")
    left = _row_index(left_rows, fields, "left token trace")
    right = _row_index(right_rows, fields, "right token trace")
    if left.keys() != right.keys():
        raise PRODUCER.ProtocolError("token trace identities do not align")
    input_matches = 0
    predicted_matches = 0
    for key in sorted(left):
        input_matches += int(
            int(left[key]["input_token_id"]) == int(right[key]["input_token_id"])
        )
        predicted_matches += int(
            int(left[key]["predicted_next_token_id"])
            == int(right[key]["predicted_next_token_id"])
        )
    count = len(left)
    return {
        "records": count,
        "input_token_match_fraction": input_matches / max(1, count),
        "predicted_token_match_fraction": predicted_matches / max(1, count),
        "full_token_parity": input_matches == count and predicted_matches == count,
    }


def summarize_trace_pair(
    left_trace: Mapping[str, Any],
    right_trace: Mapping[str, Any],
    *,
    atol: float = ALLCLOSE_ATOL,
    rtol: float = ALLCLOSE_RTOL,
    near_tie_margin: float = NEAR_TIE_MARGIN,
) -> dict[str, Any]:
    """Summarize two aligned traces without importing CUDA libraries."""

    left_tokens = list(left_trace.get("tokens", []))
    right_tokens = list(right_trace.get("tokens", []))
    token_summary = summarize_token_pair(left_tokens, right_tokens)

    fields = ("request_id", "decode_step", "layer")
    left = _row_index(left_trace.get("router", []), fields, "left router trace")
    right = _row_index(right_trace.get("router", []), fields, "right router trace")
    if left.keys() != right.keys():
        raise PRODUCER.ProtocolError("router trace identities do not align")

    scalar_pairs: list[tuple[float, float]] = []
    boundary_pairs: list[tuple[float, float]] = []
    top1_pairs: list[tuple[float, float]] = []
    internal_margin_pairs: list[tuple[float, float]] = []
    exact_records = 0
    close_records = 0
    ordered_matches = 0
    multiset_matches = 0
    assignment_differences = 0
    assignment_differences_with_any_logit_change = 0
    assignment_differences_with_swapped_order_crossing = 0
    assignment_differences_with_material_swapped_gap_change = 0
    assignment_differences_near_boundary = 0
    assignment_differences_crossing_and_near_boundary = 0
    for key in sorted(left):
        left_row = left[key]
        right_row = right[key]
        left_logits = [float(value) for value in left_row["router_logits"]]
        right_logits = [float(value) for value in right_row["router_logits"]]
        if len(left_logits) != len(right_logits) or not left_logits:
            raise PRODUCER.ProtocolError("router-logit vector width does not align")
        row_pairs = list(zip(left_logits, right_logits))
        scalar_pairs.extend(row_pairs)
        row_exact = all(a == b for a, b in row_pairs)
        row_close = all(_close(a, b, atol=atol, rtol=rtol) for a, b in row_pairs)
        exact_records += int(row_exact)
        close_records += int(row_close)

        left_experts = [int(value) for value in left_row["selected_experts"]]
        right_experts = [int(value) for value in right_row["selected_experts"]]
        ordered_match = left_experts == right_experts
        multiset_match = sorted(left_experts) == sorted(right_experts)
        ordered_matches += int(ordered_match)
        multiset_matches += int(multiset_match)
        if not multiset_match:
            assignment_differences += 1
            assignment_differences_with_any_logit_change += int(not row_exact)
        crossing = _swapped_expert_crossing(
            left_logits,
            right_logits,
            left_experts,
            right_experts,
            atol=atol,
            rtol=rtol,
        )

        left_margins = _require_mapping(left_row["topk_margins"], "left margins")
        right_margins = _require_mapping(right_row["topk_margins"], "right margins")
        top1_pairs.append(
            (float(left_margins["top1_minus_top2"]), float(right_margins["top1_minus_top2"]))
        )
        left_boundary = left_margins.get("selection_boundary")
        right_boundary = right_margins.get("selection_boundary")
        if (left_boundary is None) != (right_boundary is None):
            raise PRODUCER.ProtocolError("top-k boundary margin availability differs")
        if left_boundary is not None:
            boundary_pairs.append((float(left_boundary), float(right_boundary)))
        near_boundary = bool(
            left_boundary is not None
            and right_boundary is not None
            and min(abs(float(left_boundary)), abs(float(right_boundary)))
            <= near_tie_margin
        )
        if not multiset_match:
            assignment_differences_with_swapped_order_crossing += int(
                crossing["has_order_crossing"]
            )
            assignment_differences_with_material_swapped_gap_change += int(
                crossing["has_material_gap_change"]
            )
            assignment_differences_near_boundary += int(near_boundary)
            assignment_differences_crossing_and_near_boundary += int(
                crossing["has_order_crossing"] and near_boundary
            )
        left_internal = [float(value) for value in left_margins["within_selected"]]
        right_internal = [float(value) for value in right_margins["within_selected"]]
        if len(left_internal) != len(right_internal):
            raise PRODUCER.ProtocolError("within-top-k margin width differs")
        internal_margin_pairs.extend(zip(left_internal, right_internal))

    record_count = len(left)
    logits = _float_metrics(scalar_pairs, atol=atol, rtol=rtol)
    logits.update(
        {
            "records": record_count,
            "record_exact_match_fraction": exact_records / max(1, record_count),
            "record_allclose_match_fraction": close_records / max(1, record_count),
        }
    )
    return {
        "tokens": token_summary,
        "router_logits": logits,
        "expert_assignment": {
            "records": record_count,
            "ordered_match_fraction": ordered_matches / max(1, record_count),
            "multiset_match_fraction": multiset_matches / max(1, record_count),
            "different_multiset_records": assignment_differences,
            "different_multiset_records_with_any_logit_change": (
                assignment_differences_with_any_logit_change
            ),
            "different_multiset_records_with_swapped_expert_order_crossing": (
                assignment_differences_with_swapped_order_crossing
            ),
            "different_multiset_records_with_material_swapped_expert_gap_change": (
                assignment_differences_with_material_swapped_gap_change
            ),
            "different_multiset_records_near_selection_boundary": (
                assignment_differences_near_boundary
            ),
            "different_multiset_records_crossing_and_near_boundary": (
                assignment_differences_crossing_and_near_boundary
            ),
            "swapped_expert_order_crossing_coverage": (
                assignment_differences_with_swapped_order_crossing
                / max(1, assignment_differences)
            ),
            "material_swapped_expert_gap_change_coverage": (
                assignment_differences_with_material_swapped_gap_change
                / max(1, assignment_differences)
            ),
            "near_selection_boundary_coverage": (
                assignment_differences_near_boundary
                / max(1, assignment_differences)
            ),
            "near_tie_margin": near_tie_margin,
        },
        "topk_margins": {
            "top1_minus_top2": _float_metrics(top1_pairs, atol=atol, rtol=rtol),
            "selection_boundary": _float_metrics(
                boundary_pairs, atol=atol, rtol=rtol
            ),
            "within_selected": _float_metrics(
                internal_margin_pairs, atol=atol, rtol=rtol
            ),
        },
        "allclose": {"atol": atol, "rtol": rtol},
    }


def compare_records(
    serial_rows: Sequence[Mapping[str, Any]],
    batched_rows: Sequence[Mapping[str, Any]],
    *,
    logit_atol: float,
    near_tie_margin: float,
    include_logits: bool,
) -> dict[str, Any]:
    """Small record-level API used by CPU tests and offline result readers.

    Runtime traces carry richer ``topk_margins`` objects.  This helper also
    accepts the compact ``selection_boundary_logit_margin`` representation so
    a synthetic CPU test need not reproduce a model output object.
    """

    fields = ("request_id", "decode_step", "layer")
    serial = _row_index(serial_rows, fields, "serial records")
    batched = _row_index(batched_rows, fields, "batched records")
    if serial.keys() != batched.keys():
        raise PRODUCER.ProtocolError("serial and batched record identities differ")
    scalar_pairs: list[tuple[float, float]] = []
    exact_rows = 0
    allclose_rows = 0
    logit_difference_rows = 0
    material_logit_difference_rows = 0
    assignment_difference_rows = 0
    swapped_order_crossing_rows = 0
    material_swapped_gap_change_rows = 0
    near_tie_assignment_difference_rows = 0
    token_matches = 0
    examples: list[dict[str, Any]] = []
    for key in sorted(serial):
        left = serial[key]
        right = batched[key]
        token_matches += int(
            int(left["input_token_id"]) == int(right["input_token_id"])
            and int(left["predicted_next_token_id"])
            == int(right["predicted_next_token_id"])
        )
        left_logits = [float(value) for value in left["router_logits"]]
        right_logits = [float(value) for value in right["router_logits"]]
        if len(left_logits) != len(right_logits) or not left_logits:
            raise PRODUCER.ProtocolError("router-logit vector width does not align")
        pairs = list(zip(left_logits, right_logits))
        scalar_pairs.extend(pairs)
        exact = all(a == b for a, b in pairs)
        allclose = all(abs(a - b) <= logit_atol for a, b in pairs)
        exact_rows += int(exact)
        allclose_rows += int(allclose)
        logit_difference_rows += int(not exact)
        material_logit_difference_rows += int(not allclose)
        left_experts = sorted(int(value) for value in left["selected_experts"])
        right_experts = sorted(int(value) for value in right["selected_experts"])
        assignment_changed = left_experts != right_experts
        assignment_difference_rows += int(assignment_changed)
        crossing = _swapped_expert_crossing(
            left_logits,
            right_logits,
            left_experts,
            right_experts,
            atol=logit_atol,
            rtol=0.0,
        )
        swapped_order_crossing_rows += int(
            assignment_changed and crossing["has_order_crossing"]
        )
        material_swapped_gap_change_rows += int(
            assignment_changed and crossing["has_material_gap_change"]
        )

        def boundary(row: Mapping[str, Any]) -> float | None:
            if "selection_boundary_logit_margin" in row:
                return float(row["selection_boundary_logit_margin"])
            margins = row.get("topk_margins")
            if isinstance(margins, Mapping) and margins.get("selection_boundary") is not None:
                return float(margins["selection_boundary"])
            return None

        left_boundary = boundary(left)
        right_boundary = boundary(right)
        near_tie = bool(
            left_boundary is not None
            and right_boundary is not None
            and min(abs(left_boundary), abs(right_boundary)) <= near_tie_margin
        )
        near_tie_assignment_difference_rows += int(assignment_changed and near_tie)
        if assignment_changed and len(examples) < 16:
            example = {
                "request_id": key[0],
                "decode_step": key[1],
                "layer": key[2],
                "serial_experts": left_experts,
                "batched_experts": right_experts,
                "serial_selection_boundary_logit_margin": left_boundary,
                "batched_selection_boundary_logit_margin": right_boundary,
                "logits_exact": exact,
                "logits_allclose": allclose,
                "swapped_expert_crossing": crossing,
            }
            if include_logits:
                example["serial_router_logits"] = left_logits
                example["batched_router_logits"] = right_logits
            examples.append(example)
    count = len(serial)
    return {
        "rows": count,
        "token_parity": token_matches == count,
        "router_logit_row_exact_match_fraction": exact_rows / max(1, count),
        "router_logit_row_allclose_match_fraction": allclose_rows / max(1, count),
        "logit_difference_rows": logit_difference_rows,
        "material_logit_difference_rows": material_logit_difference_rows,
        "expert_assignment_difference_rows": assignment_difference_rows,
        "swapped_expert_order_crossing_rows": swapped_order_crossing_rows,
        "material_swapped_expert_gap_change_rows": (
            material_swapped_gap_change_rows
        ),
        "near_tie_assignment_difference_rows": near_tie_assignment_difference_rows,
        "swapped_expert_order_crossing_coverage": (
            swapped_order_crossing_rows / max(1, assignment_difference_rows)
        ),
        "material_swapped_expert_gap_change_coverage": (
            material_swapped_gap_change_rows / max(1, assignment_difference_rows)
        ),
        "near_tie_coverage": (
            near_tie_assignment_difference_rows / max(1, assignment_difference_rows)
        ),
        "router_logit_delta": _float_metrics(
            scalar_pairs, atol=logit_atol, rtol=0.0
        ),
        "difference_examples": examples,
        "thresholds": {
            "logit_atol": logit_atol,
            "near_tie_margin": near_tie_margin,
        },
    }


def classify(comparison: Mapping[str, Any], *, stable: bool) -> str:
    """Classify only the conformance pivot; never a capacity/action claim."""

    if not stable:
        return "STOP_WITHIN_ARM_UNSTABLE"
    if comparison.get("token_parity") is not True:
        return "STOP_TOKEN_PARITY_FAILED"
    assignment_changes = int(comparison.get("expert_assignment_difference_rows", 0))
    if assignment_changes > 0:
        crossing = float(
            comparison.get("swapped_expert_order_crossing_coverage", 0.0)
        )
        material = float(
            comparison.get("material_swapped_expert_gap_change_coverage", 0.0)
        )
        near_tie = float(comparison.get("near_tie_coverage", 0.0))
        if crossing != 1.0:
            return "INCONCLUSIVE_ASSIGNMENT_NOT_EXPLAINED_BY_SWAPPED_LOGITS"
        if material == 1.0 and near_tie == 1.0:
            return "PROMISING_PRE_TOPK_NEAR_TIE_AMPLIFICATION"
        if material == 1.0:
            return "PROMISING_PRE_TOPK_BATCH_CONTEXT_EFFECT"
        if material == 0.0 and near_tie == 1.0:
            return "PROMISING_TOPK_BOUNDARY_TIE_EFFECT"
        return "INCONCLUSIVE_MIXED_BATCH_CONTEXT_EFFECT"
    return "STOP_NO_REPRODUCED_ASSIGNMENT_DIVERGENCE"


def _prefixed_trace(trace: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    def rows(name: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for raw in trace.get(name, []):
            row = dict(raw)
            row["request_id"] = f"{prefix}:{row['request_id']}"
            output.append(row)
        return output

    return {"tokens": rows("tokens"), "router": rows("router")}


def _combine_traces(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    combined: dict[str, list[dict[str, Any]]] = {"tokens": [], "router": []}
    for index, trace in enumerate(traces):
        prefixed = _prefixed_trace(trace, f"repeat-{index}")
        combined["tokens"].extend(prefixed["tokens"])
        combined["router"].extend(prefixed["router"])
    return combined


def summarize_repeat_stability(
    traces: Sequence[Mapping[str, Any]], *, atol: float, rtol: float
) -> dict[str, Any]:
    if len(traces) < 2:
        raise PRODUCER.ProtocolError("at least two repeats are required for stability")
    left: list[Mapping[str, Any]] = []
    right: list[Mapping[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for repeat_index in range(1, len(traces)):
        comparisons.append(
            summarize_trace_pair(traces[0], traces[repeat_index], atol=atol, rtol=rtol)
        )
        left.append(_prefixed_trace(traces[0], f"repeat-{repeat_index}"))
        right.append(_prefixed_trace(traces[repeat_index], f"repeat-{repeat_index}"))
    aggregate = summarize_trace_pair(
        _combine_traces(left), _combine_traces(right), atol=atol, rtol=rtol
    )
    stable = bool(
        aggregate["tokens"]["full_token_parity"]
        and aggregate["router_logits"]["record_allclose_match_fraction"] == 1.0
        and aggregate["expert_assignment"]["ordered_match_fraction"] == 1.0
    )
    exact_stable = bool(
        stable
        and aggregate["router_logits"]["record_exact_match_fraction"] == 1.0
    )
    return {
        "stable_within_allclose": stable,
        "stable_exact": exact_stable,
        "reference_repeat": 0,
        "comparisons": comparisons,
        "aggregate": aggregate,
    }


def _assignment_difference_keys(
    left_trace: Mapping[str, Any], right_trace: Mapping[str, Any]
) -> list[str]:
    fields = ("request_id", "decode_step", "layer")
    left = _row_index(left_trace.get("router", []), fields, "left router trace")
    right = _row_index(right_trace.get("router", []), fields, "right router trace")
    if left.keys() != right.keys():
        raise PRODUCER.ProtocolError("router trace identities do not align")
    result = []
    for key in sorted(left):
        left_experts = sorted(int(value) for value in left[key]["selected_experts"])
        right_experts = sorted(int(value) for value in right[key]["selected_experts"])
        if left_experts != right_experts:
            result.append("/".join(str(value) for value in key))
    return result


def _reset_rng(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prefill_states(
    torch: Any, model: Any, requests: Sequence[Any]
) -> list[SimpleNamespace]:
    states: list[SimpleNamespace] = []
    for spec in requests:
        with torch.inference_mode():
            output, _ = PRODUCER._timed_call(
                model,
                "router_logit_diagnostic_prefill",
                1,
                None,
                input_ids=spec.input_ids,
                attention_mask=spec.attention_mask,
                use_cache=True,
                output_router_logits=True,
                return_dict=True,
            )
        cache = getattr(output, "past_key_values", None)
        logits = getattr(output, "logits", None)
        prompt_length = int(spec.input_ids.shape[1])
        if (
            cache is None
            or logits is None
            or PRODUCER._cache_length(cache) != prompt_length
        ):
            raise PRODUCER.ProtocolError(
                f"prefill cache/logit closure failed for {spec.request_id}"
            )
        states.append(
            SimpleNamespace(
                spec=spec,
                cache=cache,
                attention_mask=spec.attention_mask,
                next_token=torch.argmax(logits[:, -1, :], dim=-1, keepdim=True),
                prompt_length=prompt_length,
                decode_step=0,
            )
        )
    return states


def _router_rows(
    torch: Any,
    output: Any,
    *,
    model: Any,
    request_ids: Sequence[str],
    decode_step: int,
    input_tokens: Sequence[int],
    predicted_tokens: Sequence[int],
) -> list[dict[str, Any]]:
    expected_rows = len(request_ids)
    batches = PRODUCER._native_route_batches(
        output, expected_rows=expected_rows, config=getattr(model, "config")
    )
    router_logits = getattr(output, "router_logits", None)
    if not isinstance(router_logits, (tuple, list)) or len(router_logits) != len(batches):
        raise PRODUCER.ProtocolError("native router-logit layer closure failed")
    result: list[dict[str, Any]] = []
    for batch, raw_logits in zip(batches, router_logits):
        if raw_logits.ndim != 2 or int(raw_logits.shape[0]) != expected_rows:
            raise PRODUCER.ProtocolError("native router-logit row identity failed")
        logits = raw_logits.detach().float().cpu()
        selected = batch["selected_experts"]
        top_k = int(selected.shape[1])
        expert_count = int(logits.shape[1])
        if top_k < 2 or expert_count < top_k:
            raise PRODUCER.ProtocolError("invalid native top-k/expert dimensions")
        for row_index, request_id in enumerate(request_ids):
            vector = logits[row_index]
            if not bool(torch.isfinite(vector).all().item()):
                raise PRODUCER.ProtocolError("native router logits are non-finite")
            experts = [int(value) for value in selected[row_index].tolist()]
            selected_values = [float(vector[index].item()) for index in experts]
            within = [
                selected_values[index] - selected_values[index + 1]
                for index in range(len(selected_values) - 1)
            ]
            boundary = None
            if expert_count > top_k:
                ordered = torch.topk(vector, k=top_k + 1, dim=-1).values.tolist()
                boundary = float(ordered[top_k - 1] - ordered[top_k])
            result.append(
                {
                    "request_id": request_id,
                    "decode_step": decode_step,
                    "layer": int(batch["layer"]),
                    "input_token_id": int(input_tokens[row_index]),
                    "predicted_next_token_id": int(predicted_tokens[row_index]),
                    "router_logits": [float(value) for value in vector.tolist()],
                    "router_logits_dtype_before_float32_copy": str(raw_logits.dtype),
                    "selected_experts": experts,
                    "selected_logits": selected_values,
                    "topk_margins": {
                        "top1_minus_top2": selected_values[0] - selected_values[1],
                        "within_selected": within,
                        "selection_boundary": boundary,
                    },
                }
            )
    return result


def run_serial_arm(
    torch: Any,
    model: Any,
    requests: Sequence[Any],
    decode_steps: int,
) -> dict[str, Any]:
    states = _prefill_states(torch, model, requests)
    trace: dict[str, list[dict[str, Any]]] = {"tokens": [], "router": []}
    for decode_step in range(decode_steps):
        for state in states:
            input_token = int(state.next_token.item())
            prior_length = PRODUCER._cache_length(state.cache)
            state.attention_mask = torch.cat(
                (
                    state.attention_mask,
                    state.attention_mask.new_ones((1, 1)),
                ),
                dim=1,
            )
            position_ids = state.attention_mask.long().cumsum(-1)[:, -1:] - 1
            with torch.inference_mode():
                output, _ = PRODUCER._timed_call(
                    model,
                    "router_logit_diagnostic_serial_decode",
                    1,
                    None,
                    input_ids=state.next_token,
                    attention_mask=state.attention_mask,
                    position_ids=position_ids,
                    cache_position=torch.tensor(
                        [prior_length], dtype=torch.long, device=state.next_token.device
                    ),
                    past_key_values=state.cache,
                    use_cache=True,
                    output_router_logits=True,
                    return_dict=True,
                )
            state.cache = getattr(output, "past_key_values", None)
            logits = getattr(output, "logits", None)
            if (
                state.cache is None
                or logits is None
                or PRODUCER._cache_length(state.cache) != prior_length + 1
            ):
                raise PRODUCER.ProtocolError("serial decode cache/logit closure failed")
            predicted = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            predicted_token = int(predicted.item())
            token_row = {
                "request_id": state.spec.request_id,
                "decode_step": decode_step,
                "input_token_id": input_token,
                "predicted_next_token_id": predicted_token,
            }
            trace["tokens"].append(token_row)
            trace["router"].extend(
                _router_rows(
                    torch,
                    output,
                    model=model,
                    request_ids=[state.spec.request_id],
                    decode_step=decode_step,
                    input_tokens=[input_token],
                    predicted_tokens=[predicted_token],
                )
            )
            state.next_token = predicted
            state.decode_step += 1
    return trace


def run_batched_arm(
    torch: Any,
    model: Any,
    requests: Sequence[Any],
    decode_steps: int,
) -> dict[str, Any]:
    states = _prefill_states(torch, model, requests)
    trace: dict[str, list[dict[str, Any]]] = {"tokens": [], "router": []}
    for decode_step in range(decode_steps):
        (
            input_ids,
            attention_mask,
            position_ids,
            cache,
            prior_lengths,
            prior_max,
        ) = PRODUCER._pad_decode_inputs(states)
        if int(input_ids.shape[0]) != len(requests):
            raise PRODUCER.ProtocolError("batched arm did not preserve the frozen width")
        with torch.inference_mode():
            output, _ = PRODUCER._timed_call(
                model,
                "router_logit_diagnostic_batched_decode",
                len(states),
                None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=torch.tensor(
                    [prior_max], dtype=torch.long, device=input_ids.device
                ),
                past_key_values=cache,
                use_cache=True,
                output_router_logits=True,
                return_dict=True,
            )
        logits = getattr(output, "logits", None)
        output_cache = getattr(output, "past_key_values", None)
        if logits is None or output_cache is None:
            raise PRODUCER.ProtocolError("batched decode returned no cache/logits")
        split_caches = PRODUCER.split_left_padded_cache(
            output_cache,
            prior_lengths=prior_lengths,
            prior_max_length=prior_max,
        )
        predicted = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        request_ids = [state.spec.request_id for state in states]
        input_tokens = [int(input_ids[index].item()) for index in range(len(states))]
        predicted_tokens = [
            int(predicted[index].item()) for index in range(len(states))
        ]
        trace["router"].extend(
            _router_rows(
                torch,
                output,
                model=model,
                request_ids=request_ids,
                decode_step=decode_step,
                input_tokens=input_tokens,
                predicted_tokens=predicted_tokens,
            )
        )
        for index, state in enumerate(states):
            trace["tokens"].append(
                {
                    "request_id": state.spec.request_id,
                    "decode_step": decode_step,
                    "input_token_id": input_tokens[index],
                    "predicted_next_token_id": predicted_tokens[index],
                }
            )
            state.cache = split_caches[index]
            state.attention_mask = torch.cat(
                (
                    state.attention_mask,
                    state.attention_mask.new_ones((1, 1)),
                ),
                dim=1,
            )
            state.next_token = predicted[index : index + 1]
            state.decode_step += 1
    return trace


def _repo_root() -> Path:
    return next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())


def validate_tokenizer_and_prompt_identity(
    manifest: Mapping[str, Any],
    tokenizer: Any,
    prepared: Sequence[Any],
    request_ids: Sequence[str],
) -> dict[str, Any]:
    """Apply the formal token-identity checks skipped by development producer code."""

    frozen_tokenizer = _require_mapping(manifest.get("tokenizer"), "workload.tokenizer")
    observed_tokenizer = {
        "revision": str(
            _require_mapping(manifest.get("model"), "workload.model").get(
                "tokenizer_revision", ""
            )
        ),
        "class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "length": int(len(tokenizer)),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "")),
    }
    for key, value in observed_tokenizer.items():
        if frozen_tokenizer.get(key) != value:
            raise PRODUCER.ProtocolError(
                f"loaded tokenizer.{key} differs from frozen workload"
            )

    raw_by_id = {
        str(_require_mapping(row, "workload request").get("request_id", "")): row
        for row in manifest["requests"]
    }
    prepared_by_id = {request.request_id: request for request in prepared}
    checks: list[dict[str, Any]] = []
    for request_id in request_ids:
        raw = _require_mapping(raw_by_id.get(request_id), f"workload request {request_id}")
        request = prepared_by_id.get(request_id)
        if request is None:
            raise PRODUCER.ProtocolError(f"prepared request is missing: {request_id}")
        expected_count = int(raw.get("prompt_token_count", -1))
        expected_hash = str(raw.get("prompt_token_ids_sha256", ""))
        observed_count = int(request.input_ids.shape[1])
        observed_hash = PRODUCER._prompt_token_ids_sha256(request.input_ids)
        if expected_count <= 0 or len(expected_hash) != 64:
            raise PRODUCER.ProtocolError(
                f"frozen prompt-token identity is unresolved: {request_id}"
            )
        if observed_count != expected_count or observed_hash != expected_hash:
            raise PRODUCER.ProtocolError(
                f"prompt-token identity drifted for {request_id}"
            )
        checks.append(
            {
                "request_id": request_id,
                "prompt_token_count": observed_count,
                "prompt_token_ids_sha256": observed_hash,
            }
        )
    return {
        "tokenizer": observed_tokenizer,
        "selected_request_prompt_tokens": checks,
    }


def load_exact_model(manifest: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, float]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise PRODUCER.ProtocolError("diagnostic requires PyTorch and Transformers") from exc
    if not torch.cuda.is_available() or int(torch.cuda.device_count()) != 1:
        raise PRODUCER.ProtocolError("diagnostic requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)

    model_spec = _require_mapping(manifest.get("model"), "workload.model")
    for key, expected in EXPECTED_MODEL.items():
        if str(model_spec.get(key, "")) != expected:
            raise PRODUCER.ProtocolError(
                f"workload model.{key} differs from the frozen OLMoE identity"
            )
    generation = _require_mapping(manifest.get("generation"), "workload.generation")
    if generation.get("mode") != "greedy" or bool(generation.get("do_sample", False)):
        raise PRODUCER.ProtocolError("diagnostic requires frozen greedy generation")

    shared = _repo_root() / "experiments" / "shared"
    sys.path.insert(0, str(shared))
    from modeling import load_model, load_tokenizer

    tokenizer = load_tokenizer(
        EXPECTED_MODEL["id"],
        local_files_only=True,
        revision=EXPECTED_MODEL["tokenizer_revision"],
    )
    model, load_seconds = load_model(
        EXPECTED_MODEL["id"],
        dtype_name=EXPECTED_MODEL["dtype"],
        local_files_only=True,
        revision=EXPECTED_MODEL["revision"],
    )
    model.eval()
    if not str(getattr(model, "device", "")).startswith("cuda"):
        raise PRODUCER.ProtocolError("loaded model is not resident on CUDA")
    parameter = next(model.parameters())
    if parameter.dtype != torch.bfloat16:
        raise PRODUCER.ProtocolError("loaded model dtype differs from frozen bfloat16")
    observed_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    if observed_commit not in (None, EXPECTED_MODEL["revision"]):
        raise PRODUCER.ProtocolError("loaded model config commit differs from revision")
    return torch, transformers, tokenizer, model, float(load_seconds)


def _environment(torch: Any, transformers: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "cuda_version": getattr(torch.version, "cuda", None),
        "visible_gpu_count": int(torch.cuda.device_count()),
        "gpu": {
            "index": 0,
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
        },
    }


def validate_runtime_environment(
    captured: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed on runtime drift from the hash-bound pilot environment."""

    captured_gpus = captured.get("gpus")
    if not isinstance(captured_gpus, list) or len(captured_gpus) != 1:
        raise PRODUCER.ProtocolError("captured environment is not one-GPU closed")
    captured_gpu = _require_mapping(captured_gpus[0], "captured GPU")
    current_gpu = _require_mapping(current.get("gpu"), "current GPU")
    comparisons = {
        "python": (captured.get("python"), current.get("python")),
        "torch": (captured.get("torch"), current.get("torch")),
        "transformers": (
            captured.get("transformers"),
            current.get("transformers"),
        ),
        "cuda_version": (captured.get("cuda_version"), current.get("cuda_version")),
        "gpu_count": (captured.get("gpu_count"), current.get("visible_gpu_count")),
        "gpu_name": (captured_gpu.get("name"), current_gpu.get("name")),
        "gpu_capability": (
            list(captured_gpu.get("capability", [])),
            list(current_gpu.get("capability", [])),
        ),
    }
    drift = {
        key: {"captured": left, "current": right}
        for key, (left, right) in comparisons.items()
        if left != right
    }
    if drift:
        raise PRODUCER.ProtocolError(f"runtime differs from captured pilot: {drift}")

    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    current_git_sha = git_result.stdout.strip()
    if git_result.returncode != 0 or current_git_sha != str(captured.get("git_sha", "")):
        raise PRODUCER.ProtocolError("repository HEAD differs from captured pilot")
    return {
        "status": "MATCH_CAPTURED_RUNTIME",
        "checked_fields": sorted(comparisons),
        "git_sha": current_git_sha,
    }


def query_gpu_compute_processes() -> tuple[tuple[str, str, str], ...]:
    """Return opaque GPU process identities; PIDs need not match container PIDs."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise PRODUCER.ProtocolError(
            f"cannot query GPU process isolation: {result.stderr.strip()}"
        )
    rows: list[tuple[str, str, str]] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or text.upper() in {"N/A", "[N/A]"}:
            continue
        fields = tuple(value.strip() for value in text.split(",", 2))
        if len(fields) != 3 or not fields[1].isdigit():
            raise PRODUCER.ProtocolError(f"unexpected nvidia-smi process row: {text!r}")
        rows.append(fields)
    return tuple(sorted(set(rows)))


class GpuIsolationMonitor:
    """Sample GPU process identity during every arm and fail on any overlap."""

    def __init__(
        self,
        expected: tuple[tuple[str, str, str], ...],
        interval_seconds: float = GPU_POLL_INTERVAL_SECONDS,
    ) -> None:
        if len(expected) != 1:
            raise PRODUCER.ProtocolError(
                "model-load isolation requires exactly one stable compute process"
            )
        self.expected = expected
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0
        self.violations: list[dict[str, Any]] = []

    def check(self, label: str) -> None:
        try:
            observed = query_gpu_compute_processes()
        except Exception as exc:  # preserve failure for the main thread
            self.violations.append({"label": label, "error": str(exc)})
            return
        self.samples += 1
        if observed != self.expected:
            self.violations.append(
                {
                    "label": label,
                    "expected": [list(row) for row in self.expected],
                    "observed": [list(row) for row in observed],
                }
            )

    def start(self) -> None:
        self.check("monitor_start")

        def poll() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.check("periodic_poll")

        self._thread = threading.Thread(
            target=poll, name="rce-gpu-isolation-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
            if self._thread.is_alive():
                self.violations.append({"label": "monitor_join", "error": "timeout"})
        self.check("monitor_stop")

    def require_clean(self) -> None:
        if self.violations:
            raise PRODUCER.ProtocolError(
                f"GPU isolation was not preserved: {self.violations[:4]}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "status": "PASS_SAMPLED_PROCESS_ISOLATION" if not self.violations else "FAIL",
            "poll_interval_seconds": self.interval_seconds,
            "samples": self.samples,
            "expected_compute_process": [list(row) for row in self.expected],
            "violations": self.violations,
        }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish complete JSON atomically without clobbering an existing result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise PRODUCER.ProtocolError(f"refusing to overwrite output: {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _reference_summary(
    traces: Sequence[Mapping[str, Any]], reference_tokens: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    comparisons = [
        summarize_token_pair(trace["tokens"], reference_tokens) for trace in traces
    ]
    return {
        "all_repeats_match": all(value["full_token_parity"] for value in comparisons),
        "comparisons": comparisons,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output).resolve()
    try:
        output_path.relative_to(Path("/tmp"))
    except ValueError as exc:
        raise PRODUCER.ProtocolError("diagnostic output must remain under /tmp") from exc
    if output_path.suffix != ".json":
        raise PRODUCER.ProtocolError("diagnostic output must be one JSON file")
    if output_path.exists():
        raise PRODUCER.ProtocolError(f"refusing to overwrite output: {output_path}")
    frozen_scale = (4, 8, 4, 3)
    observed_scale = (
        args.requests,
        args.decode_steps,
        args.batch_size,
        args.repeats,
    )
    if observed_scale != frozen_scale:
        raise PRODUCER.ProtocolError(
            "diagnostic scale is frozen as requests=4, decode_steps=8, "
            "batch_size=4, repeats=3"
        )
    if not args.offline:
        raise PRODUCER.ProtocolError(
            "diagnostic is fail-closed: pass --offline to forbid model downloads"
        )

    capture_dir = Path(args.capture_dir).resolve()
    manifest, complete, manifest_path = load_capture_contract(capture_dir)
    captured_environment = load_captured_environment(capture_dir, complete)
    max_steps = int(_require_mapping(manifest["generation"], "generation")["max_decode_steps"])
    if args.decode_steps > max_steps:
        raise PRODUCER.ProtocolError("--decode-steps exceeds frozen workload maximum")
    request_ids = select_request_ids(manifest, args.requests)
    reference_tokens = load_reference_tokens(
        capture_dir, complete, request_ids, args.decode_steps
    )

    processes_before_model_load = query_gpu_compute_processes()
    if processes_before_model_load:
        raise PRODUCER.ProtocolError(
            "GPU is not idle before model load: "
            f"{[list(row) for row in processes_before_model_load]}"
        )
    torch, transformers, tokenizer, model, load_seconds = load_exact_model(manifest)
    current_environment = _environment(torch, transformers)
    runtime_validation = validate_runtime_environment(
        captured_environment, current_environment
    )
    own_gpu_process = query_gpu_compute_processes()
    if len(own_gpu_process) != 1:
        raise PRODUCER.ProtocolError(
            "model load did not produce exactly one isolated GPU compute process: "
            f"{[list(row) for row in own_gpu_process]}"
        )
    prepared = PRODUCER._prepare_requests(manifest, tokenizer, model.device)
    by_id = {request.request_id: request for request in prepared}
    if not set(request_ids).issubset(by_id):
        raise PRODUCER.ProtocolError("prepared requests do not close the audit IDs")
    requests = [by_id[request_id] for request_id in request_ids]
    token_identity = validate_tokenizer_and_prompt_identity(
        manifest, tokenizer, prepared, request_ids
    )
    seed = int(manifest["seed"])
    serial_traces: list[dict[str, Any]] = []
    batched_traces: list[dict[str, Any]] = []
    arm_orders: list[list[str]] = []
    monitor = GpuIsolationMonitor(own_gpu_process)

    def run_checked_arm(arm: str) -> dict[str, Any]:
        monitor.check(f"before_{arm}")
        monitor.require_clean()
        _reset_rng(torch, seed)
        trace = (
            run_serial_arm(torch, model, requests, args.decode_steps)
            if arm == "serial_width_1"
            else run_batched_arm(torch, model, requests, args.decode_steps)
        )
        monitor.check(f"after_{arm}")
        monitor.require_clean()
        return trace

    monitor.start()
    try:
        # Discard one full warmup per arm so compilation/cache initialization is
        # not uniquely assigned to either measured width.
        run_checked_arm("serial_width_1")
        run_checked_arm("batch_width_n")
        torch.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        for repeat_index in range(args.repeats):
            order = (
                ["serial_width_1", "batch_width_n"]
                if repeat_index % 2 == 0
                else ["batch_width_n", "serial_width_1"]
            )
            arm_orders.append(order)
            for arm in order:
                trace = run_checked_arm(arm)
                if arm == "serial_width_1":
                    serial_traces.append(trace)
                else:
                    batched_traces.append(trace)
        torch.cuda.synchronize(0)
        elapsed_seconds = time.monotonic() - started
    finally:
        monitor.stop()
    monitor.require_clean()

    serial_stability = summarize_repeat_stability(
        serial_traces, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
    )
    batched_stability = summarize_repeat_stability(
        batched_traces, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
    )
    cross_by_repeat = [
        summarize_trace_pair(serial_trace, batched_trace)
        for serial_trace, batched_trace in zip(serial_traces, batched_traces)
    ]
    cross_aggregate = summarize_trace_pair(
        _combine_traces(serial_traces), _combine_traces(batched_traces)
    )
    difference_keys = [
        _assignment_difference_keys(serial_trace, batched_trace)
        for serial_trace, batched_trace in zip(serial_traces, batched_traces)
    ]
    effect_consistent = bool(
        difference_keys
        and difference_keys[0]
        and all(keys == difference_keys[0] for keys in difference_keys[1:])
    )
    serial_reference = _reference_summary(serial_traces, reference_tokens)
    batched_reference = _reference_summary(batched_traces, reference_tokens)
    token_parity = bool(
        cross_aggregate["tokens"]["full_token_parity"]
        and serial_reference["all_repeats_match"]
        and batched_reference["all_repeats_match"]
    )
    repeats_stable = bool(
        serial_stability["stable_within_allclose"]
        and batched_stability["stable_within_allclose"]
    )
    assignment = cross_aggregate["expert_assignment"]
    assignment_changed = assignment["different_multiset_records"] > 0
    if not token_parity:
        status = "STOP_TOKEN_PARITY_FAILED"
    elif not repeats_stable:
        status = "STOP_WITHIN_ARM_UNSTABLE"
    elif not assignment_changed:
        status = "STOP_NO_REPRODUCED_ASSIGNMENT_DIVERGENCE"
    elif not effect_consistent:
        status = "STOP_CROSS_WIDTH_EFFECT_NOT_REPEAT_CONSISTENT"
    else:
        status = classify(
            {
                "token_parity": token_parity,
                "expert_assignment_difference_rows": assignment[
                    "different_multiset_records"
                ],
                "swapped_expert_order_crossing_coverage": assignment[
                    "swapped_expert_order_crossing_coverage"
                ],
                "material_swapped_expert_gap_change_coverage": assignment[
                    "material_swapped_expert_gap_change_coverage"
                ],
                "near_tie_coverage": assignment[
                    "near_selection_boundary_coverage"
                ],
            },
            stable=True,
        )

    marker = _require_mapping(
        manifest.get("route_capacity_envelope"), "route_capacity_envelope"
    )
    payload: dict[str, Any] = {
        "schema": "route-capacity-envelope-router-logit-conformance-v1",
        "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": {
            "evidence_class": "development_execution_conformance_diagnostic",
            "comparison": "serial_width_1_vs_fixed_batched_width_n",
            "router_signal": "native_pre_top_k_router_logits",
            "capacity_claim_authorized": False,
            "action_claim_authorized": False,
            "safe_capacity_measured": False,
            "controller_action_executed": False,
        },
        "source_capture": {
            "capture_dir": str(capture_dir),
            "workload_manifest": str(manifest_path),
            "workload_manifest_sha256": _sha256_file(manifest_path),
            "capture_complete_sha256": _sha256_file(
                capture_dir / "CAPTURE_COMPLETE.json"
            ),
            "capture_files_sha256": dict(
                _require_mapping(complete.get("files"), "CAPTURE_COMPLETE.files")
            ),
            "episode_id": str(marker.get("episode_id", "")),
            "arrival_regime": str(marker.get("arrival_regime", "")),
            "reference_tokens": reference_tokens,
            "captured_environment": captured_environment,
        },
        "model": {**EXPECTED_MODEL, "offline": True, "load_seconds": load_seconds},
        "execution": {
            "seed_reset_identically_before_each_arm": seed,
            "requests": args.requests,
            "request_ids": request_ids,
            "decode_steps": args.decode_steps,
            "serial_batch_size": 1,
            "batched_batch_size": args.batch_size,
            "repeats": args.repeats,
            "warmup_arms_discarded": ["serial_width_1", "batch_width_n"],
            "measured_arm_order_by_repeat": arm_orders,
            "allclose": {"atol": ALLCLOSE_ATOL, "rtol": ALLCLOSE_RTOL},
            "near_tie_margin": NEAR_TIE_MARGIN,
            "elapsed_seconds_excluding_model_load": elapsed_seconds,
            "peak_cuda_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(0)
            ),
            "environment": current_environment,
            "runtime_match": runtime_validation,
            "token_identity": token_identity,
            "gpu_isolation": monitor.summary(),
        },
        "token_parity": {
            "enforced": True,
            "passed": token_parity,
            "serial_vs_batched": cross_aggregate["tokens"],
            "serial_vs_source_capture": serial_reference,
            "batched_vs_source_capture": batched_reference,
        },
        "within_arm_repeat_stability": {
            "passed": repeats_stable,
            "serial_width_1": serial_stability,
            "batched_width_n": batched_stability,
        },
        "serial_vs_batched": {
            "aggregate": cross_aggregate,
            "by_repeat": cross_by_repeat,
            "assignment_difference_keys_by_repeat": difference_keys,
            "batch_width_effect_repeat_consistent": effect_consistent,
        },
        "traces": {
            "serial_width_1": [
                {"repeat": index, **trace}
                for index, trace in enumerate(serial_traces)
            ],
            "batched_width_n": [
                {"repeat": index, **trace}
                for index, trace in enumerate(batched_traces)
            ],
        },
    }
    _write_json_exclusive(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="require the exact frozen model and tokenizer from the local cache",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(Path(args.output).resolve()),
                "capacity_claim_authorized": False,
                "action_claim_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
