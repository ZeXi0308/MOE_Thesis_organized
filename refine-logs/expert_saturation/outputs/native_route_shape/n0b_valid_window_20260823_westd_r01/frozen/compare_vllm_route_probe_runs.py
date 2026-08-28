#!/usr/bin/env python3
"""Fail-closed telemetry OFF/ON parity and overhead comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np


MATCHED_CONFIG_FIELDS = (
    "model",
    "revision",
    "dtype",
    "batch_sizes",
    "prompt_lengths",
    "output_tokens",
    "groups",
    "within_process_repeats",
    "process_repeat",
    "seed",
    "order_seed",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "runtime_patch_id",
    "enforce_eager",
    "require_exclusive_gpu",
    "workload_manifest_sha256",
    "probe_script_sha256",
    "producer_source_artifact",
    "producer_source_artifact_sha256",
    "runtime_identity",
    "model_shape",
)

FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT = 5.0
APPROVED_PRODUCER_SOURCE = Path(__file__).with_name(
    "run_vllm_route_shape_probe.py"
)
APPROVED_PRODUCER_SHA256 = hashlib.sha256(
    APPROVED_PRODUCER_SOURCE.read_bytes()
).hexdigest()

_REQUIRED_FILES = (
    "config.json",
    "environment.json",
    "model_shape.json",
    "workload_manifest.json",
    "batches.jsonl",
    "summary.json",
    "ARTIFACT_HASHES.json",
    "RUN_COMPLETE.json",
)
_MANIFEST_FIXED_ARTIFACTS = (
    "environment.json",
    "model_shape.json",
    "workload_manifest.json",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _quantile(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _json_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = _strict_json_loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        payload = _strict_json_loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {path.name}:{line_number}")
        rows.append(payload)
    return rows


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _safe_artifact_path(run_dir: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"unsafe artifact path: {relative!r}")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"unsafe artifact path: {relative!r}")
    root = run_dir.resolve()
    candidate = (run_dir / Path(*pure.parts)).resolve(strict=False)
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"artifact escapes bundle: {relative!r}")
    return candidate


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _inspect_npz(
    path: Path,
    expected_key: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {expected_key}:
                raise ValueError(
                    f"NPZ keys for {path.name}: expected [{expected_key!r}], "
                    f"got {sorted(archive.files)!r}"
                )
            array = np.asarray(archive[expected_key])
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid NPZ {path.name}: {type(exc).__name__}: {exc}") from exc
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"NPZ dtype for {path.name} is not integer: {array.dtype}")
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"NPZ shape for {path.name}: expected {expected_shape}, got {array.shape}"
        )
    return array


def _close(actual: Any, expected: float, label: str) -> None:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        raise ValueError(f"missing/non-numeric timing summary:{label}")
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(
            f"timing summary mismatch:{label}:reported={actual}:recomputed={expected}"
        )


def _recompute_row_timing(
    row: dict[str, Any], expected_output_tokens: int, expected_batch_size: int
) -> dict[str, float]:
    """Rebuild every producer timing aggregate from per-request evidence."""

    requests = row.get("request_metrics")
    if not isinstance(requests, list) or len(requests) != expected_batch_size:
        raise ValueError("request_metrics count mismatch")
    tpots: list[float] = []
    decode_spans: list[float] = []
    ttfts: list[float] = []
    queues: list[float] = []
    total_tokens = 0
    request_ids: set[str] = set()
    for request_number, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"request_metrics entry is not an object:{request_number}")
        generated = request.get("generated_tokens")
        if not _positive_int(generated) or generated != expected_output_tokens:
            raise ValueError(f"generated_tokens denominator mismatch:{request_number}")
        tokens = request.get("token_ids")
        if not isinstance(tokens, list) or len(tokens) != generated:
            raise ValueError(f"generated token count mismatch:{request_number}")
        if request.get("finish_reason") != "length":
            raise ValueError(f"unexpected finish_reason:{request_number}")
        request_id = request.get("request_id")
        if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
            raise ValueError(f"missing/invalid request_id:{request_number}")
        request_id_text = str(request_id)
        if request_id_text in request_ids:
            raise ValueError(f"duplicate request_id:{request_id_text}")
        request_ids.add(request_id_text)

        decode_span = request.get("decode_span_ms")
        tpot = request.get("tpot_ms")
        ttft = request.get("ttft_ms")
        queue = request.get("queue_ms")
        if not _positive_finite(decode_span):
            raise ValueError(f"invalid decode_span_ms:{request_number}")
        if not _positive_finite(tpot):
            raise ValueError(f"invalid tpot_ms:{request_number}")
        if not _nonnegative_finite(ttft):
            raise ValueError(f"invalid ttft_ms:{request_number}")
        if not _nonnegative_finite(queue):
            raise ValueError(f"invalid queue_ms:{request_number}")
        expected_tpot = float(decode_span) / (generated - 1)
        if not math.isclose(float(tpot), expected_tpot, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"request TPOT/decode-span mismatch:{request_number}:"
                f"reported={tpot}:recomputed={expected_tpot}"
            )
        tpots.append(float(tpot))
        decode_spans.append(float(decode_span))
        ttfts.append(float(ttft))
        queues.append(float(queue))
        total_tokens += generated

    timing = row.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("missing timing summary")
    wall_ms = timing.get("wall_ms")
    if not _positive_finite(wall_ms):
        raise ValueError("nonpositive/nonfinite wall_ms")
    recomputed = {
        "wall_ms": float(wall_ms),
        "throughput_tokens_per_s": total_tokens / (float(wall_ms) / 1000.0),
        "request_tpot_p50_ms": float(np.quantile(tpots, 0.50)),
        "request_tpot_p95_ms": float(np.quantile(tpots, 0.95)),
        "request_tpot_max_ms": max(tpots),
        "request_ttft_p50_ms": float(np.quantile(ttfts, 0.50)),
        "request_ttft_p95_ms": float(np.quantile(ttfts, 0.95)),
        "request_queue_p95_ms": float(np.quantile(queues, 0.95)),
    }
    # decode_spans is intentionally retained in the raw evidence. Its exact
    # per-request denominator relation was checked above; the v1 producer did
    # not emit a separate decode-span aggregate.
    if len(decode_spans) != expected_batch_size:  # pragma: no cover - invariant
        raise ValueError("decode-span coverage mismatch")
    for field, expected in recomputed.items():
        _close(timing.get(field), expected, field)
    return recomputed


def _append_error(errors: list[str], label: str, exc: Exception) -> None:
    errors.append(f"{label}:{type(exc).__name__}:{exc}")


def _inspect_bundle(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Verify a sealed producer bundle and return parsed data only if valid."""

    errors: list[str] = []
    source_status = "PRODUCER_SOURCE_UNKNOWN"
    source_verified = False
    source_semantics_approved = False
    exclusive_gpu_verified = False
    timing_evidence_verified = True
    validation_warnings: list[str] = []
    if not run_dir.is_dir():
        return {
            "valid": False,
            "errors": ["run_dir_missing"],
            "producer_source_status": source_status,
            "producer_source_verified": source_verified,
            "producer_source_semantics_approved": source_semantics_approved,
            "exclusive_gpu_verified": exclusive_gpu_verified,
            "timing_evidence_verified": False,
            "validation_warnings": validation_warnings,
        }, None
    for relative in _REQUIRED_FILES:
        path = _safe_artifact_path(run_dir, relative)
        if not path.is_file():
            errors.append(f"missing:{relative}")
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "producer_source_status": source_status,
            "producer_source_verified": source_verified,
            "producer_source_semantics_approved": source_semantics_approved,
            "exclusive_gpu_verified": exclusive_gpu_verified,
            "timing_evidence_verified": False,
            "validation_warnings": validation_warnings,
        }, None

    try:
        config = _load_json(run_dir / "config.json")
        environment = _load_json(run_dir / "environment.json")
        model_shape = _load_json(run_dir / "model_shape.json")
        summary = _load_json(run_dir / "summary.json")
        artifact_hashes = _load_json(run_dir / "ARTIFACT_HASHES.json")
        seal = _load_json(run_dir / "RUN_COMPLETE.json")
        rows = _load_jsonl(run_dir / "batches.jsonl")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _append_error(errors, "parse", exc)
        return {
            "valid": False,
            "errors": errors,
            "producer_source_status": source_status,
            "producer_source_verified": source_verified,
            "producer_source_semantics_approved": source_semantics_approved,
            "exclusive_gpu_verified": exclusive_gpu_verified,
            "timing_evidence_verified": False,
            "validation_warnings": validation_warnings,
        }, None

    sealed = {
        "config_sha256": run_dir / "config.json",
        "raw_sha256": run_dir / "batches.jsonl",
        "summary_sha256": run_dir / "summary.json",
        "artifact_hashes_sha256": run_dir / "ARTIFACT_HASHES.json",
    }
    if seal.get("status") != "RUN_COMPLETE":
        errors.append("seal_status")
    for field, path in sealed.items():
        expected = seal.get(field)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if not _is_sha256(expected) or expected != actual:
            errors.append(f"seal_hash:{field}")

    manifest_paths: dict[str, Path] = {}
    for relative, expected in artifact_hashes.items():
        if not _is_sha256(expected):
            errors.append(f"artifact_hash_format:{relative!r}")
            continue
        try:
            path = _safe_artifact_path(run_dir, relative)
        except (TypeError, ValueError) as exc:
            _append_error(errors, "artifact_path", exc)
            continue
        manifest_paths[relative] = path
        if not path.is_file():
            errors.append(f"artifact_missing:{relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"artifact_hash:{relative}")
    for relative in _MANIFEST_FIXED_ARTIFACTS:
        if relative not in artifact_hashes:
            errors.append(f"artifact_unlisted:{relative}")

    producer_signals = (
        "producer_source_artifact" in config,
        "producer_source_artifact_sha256" in config,
        "producer_source.py" in artifact_hashes,
        (run_dir / "producer_source.py").exists(),
    )
    if any(producer_signals):
        producer_errors: list[str] = []
        relative = config.get("producer_source_artifact")
        expected = config.get("producer_source_artifact_sha256")
        if relative != "producer_source.py":
            producer_errors.append("producer_source_path")
        if not _is_sha256(expected):
            producer_errors.append("producer_source_config_hash")
        if config.get("probe_script_sha256") != expected:
            producer_errors.append("producer_source_probe_hash")
        if relative != "producer_source.py" or relative not in artifact_hashes:
            producer_errors.append("producer_source_manifest_entry")
        elif artifact_hashes.get(relative) != expected:
            producer_errors.append("producer_source_manifest_hash")
        try:
            producer_path = _safe_artifact_path(run_dir, relative)
            if not producer_path.is_file():
                producer_errors.append("producer_source_missing")
            elif _is_sha256(expected) and hashlib.sha256(
                producer_path.read_bytes()
            ).hexdigest() != expected:
                producer_errors.append("producer_source_file_hash")
        except (OSError, TypeError, ValueError) as exc:
            _append_error(producer_errors, "producer_source", exc)
        if producer_errors:
            source_status = "PRODUCER_SOURCE_INVALID"
            errors.extend(producer_errors)
        else:
            source_verified = True
            source_semantics_approved = expected == APPROVED_PRODUCER_SHA256
            source_status = (
                "PRODUCER_SOURCE_VERIFIED_APPROVED"
                if source_semantics_approved
                else "PRODUCER_SOURCE_INTEGRITY_VERIFIED_SEMANTICS_UNAPPROVED"
            )
    else:
        # Historical sealed bundles predate embedded producer bytes. They stay
        # usable for structural measurement, but may never authorize an action.
        source_status = "PRODUCER_SOURCE_UNVERIFIED_HISTORICAL"

    if summary.get("schema") != "vllm-native-route-shape-probe-v1":
        errors.append("summary_schema")
    if summary.get("status") != "COMPLETE":
        errors.append("summary_status")
    capture_routes = config.get("capture_routes")
    if not isinstance(capture_routes, bool):
        errors.append("config_capture_routes")
    if summary.get("capture_routes") is not capture_routes:
        errors.append("summary_capture_routes")
    if summary.get("record_count") != len(rows):
        errors.append("summary_record_count")
    if config.get("claim_ceiling") != summary.get("claim_ceiling"):
        errors.append("summary_claim_ceiling")
    if config.get("model_shape") != model_shape:
        errors.append("config_model_shape")

    try:
        num_experts = model_shape["num_experts"]
        num_layers = model_shape["num_layers"]
        top_k = model_shape["top_k"]
        output_tokens = config["output_tokens"]
        if not all(
            _positive_int(value)
            for value in (num_experts, num_layers, top_k, output_tokens)
        ):
            raise ValueError("model/output dimensions must be positive integers")
        if top_k > num_experts:
            raise ValueError("top_k exceeds num_experts")
        if output_tokens < 2:
            raise ValueError("output_tokens must be >= 2")
    except (KeyError, TypeError, ValueError) as exc:
        _append_error(errors, "model_shape", exc)
        num_experts = num_layers = top_k = output_tokens = None

    runtime_identity = config.get("runtime_identity")
    if not isinstance(runtime_identity, dict) or not runtime_identity:
        errors.append("config_runtime_identity")
    else:
        for key, value in runtime_identity.items():
            if key not in environment or environment[key] != value:
                errors.append(f"runtime_environment:{key}")
    require_exclusive_gpu = config.get("require_exclusive_gpu")
    if not isinstance(require_exclusive_gpu, bool):
        errors.append("config_require_exclusive_gpu")
    elif require_exclusive_gpu and environment.get(
        "compute_processes_before_engine_init"
    ) != []:
        errors.append("exclusive_gpu_environment")
    exclusive_gpu_verified = bool(
        require_exclusive_gpu is True
        and environment.get("compute_processes_before_engine_init") == []
    )

    workload_path = run_dir / "workload_manifest.json"
    workload_hash = hashlib.sha256(workload_path.read_bytes()).hexdigest()
    if config.get("workload_manifest_sha256") != workload_hash:
        errors.append("workload_manifest_sha256")

    seen_batch_ids: set[str] = set()
    seen_input_artifacts: set[str] = set()
    seen_route_artifacts: set[str] = set()
    execution_orders: list[int] = []
    actual_cell_counts: Counter[tuple[int, int]] = Counter()
    for row_number, row in enumerate(rows, 1):
        prefix = f"row:{row_number}"
        try:
            batch_id = row["batch_id"]
            prompt_length = row["prompt_length"]
            batch_size = row["batch_size"]
            if not isinstance(batch_id, str) or not batch_id:
                raise ValueError("invalid batch_id")
            if batch_id in seen_batch_ids:
                raise ValueError(f"duplicate batch_id:{batch_id}")
            seen_batch_ids.add(batch_id)
            if not _positive_int(prompt_length) or not _positive_int(batch_size):
                raise ValueError("invalid prompt_length/batch_size")
            actual_cell_counts[(prompt_length, batch_size)] += 1
            execution_order = row["execution_order"]
            if not isinstance(execution_order, int) or isinstance(execution_order, bool):
                raise ValueError("invalid execution_order")
            execution_orders.append(execution_order)
            if row.get("process_repeat") != config.get("process_repeat"):
                raise ValueError("process_repeat drift")

            input_relative = row["input_artifact"]
            input_expected = row["input_artifact_sha256"]
            if input_relative in seen_input_artifacts:
                raise ValueError(f"reused input artifact:{input_relative}")
            seen_input_artifacts.add(input_relative)
            input_path = _safe_artifact_path(run_dir, input_relative)
            if input_relative not in artifact_hashes:
                raise ValueError(f"unlisted input artifact:{input_relative}")
            if not _is_sha256(input_expected):
                raise ValueError("invalid input row hash")
            if artifact_hashes[input_relative] != input_expected:
                raise ValueError("input row/manifest hash mismatch")
            if not input_path.is_file():
                raise ValueError(f"missing input artifact:{input_relative}")
            if hashlib.sha256(input_path.read_bytes()).hexdigest() != input_expected:
                raise ValueError("input row/file hash mismatch")
            prompts = _inspect_npz(
                input_path,
                "prompt_token_ids",
                (batch_size, prompt_length),
            )
            if _json_hash(prompts.tolist()) != row.get("prompt_token_ids_sha256"):
                raise ValueError("input contents/prompt digest mismatch")

            try:
                _recompute_row_timing(row, output_tokens, batch_size)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                timing_evidence_verified = False
                if source_verified:
                    raise
                # Pre-embedding historical bundles may lack the request-level
                # timing fields needed for a modern timing claim. Preserve
                # their structural route evidence, but explicitly withhold
                # timing qualification and action eligibility.
                timing_evidence_verified = False
                validation_warnings.append(
                    f"{prefix}:historical_timing_unverified:"
                    f"{type(exc).__name__}:{exc}"
                )

            if capture_routes:
                route_relative = row["route_artifact"]
                route_expected = row["route_artifact_sha256"]
                route_summary = row["route"]
                if route_relative in seen_route_artifacts:
                    raise ValueError(f"reused route artifact:{route_relative}")
                seen_route_artifacts.add(route_relative)
                route_path = _safe_artifact_path(run_dir, route_relative)
                if route_relative not in artifact_hashes:
                    raise ValueError(f"unlisted route artifact:{route_relative}")
                if not _is_sha256(route_expected):
                    raise ValueError("invalid route row hash")
                if artifact_hashes[route_relative] != route_expected:
                    raise ValueError("route row/manifest hash mismatch")
                if not route_path.is_file():
                    raise ValueError(f"missing route artifact:{route_relative}")
                if hashlib.sha256(route_path.read_bytes()).hexdigest() != route_expected:
                    raise ValueError("route row/file hash mismatch")
                routes = _inspect_npz(
                    route_path,
                    "routes",
                    (batch_size, output_tokens - 1, num_layers, top_k),
                )
                if np.any(routes < 0) or np.any(routes >= num_experts):
                    raise ValueError("route expert outside configured range")
                if top_k > 1 and np.any(np.diff(np.sort(routes, axis=-1), axis=-1) == 0):
                    raise ValueError("duplicate expert within per-token top-k")
                if not isinstance(route_summary, dict):
                    raise ValueError("route summary missing")
                expected_route_summary = {
                    "batch_size": batch_size,
                    "decode_route_steps": output_tokens - 1,
                    "num_layers": num_layers,
                    "top_k": top_k,
                    "num_experts": num_experts,
                    "total_assignments": int(routes.size),
                }
                for key, expected_value in expected_route_summary.items():
                    if route_summary.get(key) != expected_value:
                        raise ValueError(f"route summary drift:{key}")
            elif any(
                key in row
                for key in ("route_artifact", "route_artifact_sha256", "route")
            ):
                raise ValueError("route-OFF row contains route artifact references")
        except (
            EOFError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ) as exc:
            _append_error(errors, prefix, exc)

    if execution_orders and sorted(execution_orders) != list(range(len(rows))):
        errors.append("execution_order_coverage")

    summary_cells = summary.get("cell_summaries")
    if not isinstance(summary_cells, list):
        errors.append("summary_cell_summaries")
    else:
        declared_cell_counts: Counter[tuple[int, int]] = Counter()
        for item in summary_cells:
            if not isinstance(item, dict):
                errors.append("summary_cell_entry")
                continue
            try:
                cell = (int(item["prompt_length"]), int(item["batch_size"]))
                samples = item["samples"]
                if not _positive_int(samples):
                    raise ValueError("invalid cell sample count")
                if cell in declared_cell_counts:
                    raise ValueError(f"duplicate summary cell:{cell}")
                declared_cell_counts[cell] = samples
            except (KeyError, TypeError, ValueError) as exc:
                _append_error(errors, "summary_cell", exc)
        if declared_cell_counts != actual_cell_counts:
            errors.append("summary_cell_counts")

    if not capture_routes and any(
        relative.startswith("routes/") for relative in manifest_paths
    ):
        errors.append("route_off_manifest_contains_routes")

    integrity = {
        "valid": not errors,
        "errors": errors,
        "producer_source_status": source_status,
        "producer_source_verified": source_verified,
        "producer_source_semantics_approved": source_semantics_approved,
        "exclusive_gpu_verified": exclusive_gpu_verified,
        "timing_evidence_verified": timing_evidence_verified,
        "validation_warnings": validation_warnings,
    }
    if errors:
        return integrity, None
    return integrity, {
        "config": config,
        "environment": environment,
        "rows": rows,
    }


def verify_bundle(run_dir: Path) -> dict[str, Any]:
    integrity, _ = _inspect_bundle(run_dir)
    return integrity


def _key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["prompt_length"]),
        int(row["batch_size"]),
        int(row["group"]),
        int(row["within_process_repeat"]),
    )


def _tokens(row: dict[str, Any]) -> list[list[int]]:
    return [list(request["token_ids"]) for request in row["request_metrics"]]


def compare_runs(
    on_config: dict[str, Any],
    off_config: dict[str, Any],
    on_rows: Sequence[dict[str, Any]],
    off_rows: Sequence[dict[str, Any]],
    max_p95_overhead_pct: float,
) -> dict[str, Any]:
    threshold_valid = (
        isinstance(max_p95_overhead_pct, (int, float))
        and not isinstance(max_p95_overhead_pct, bool)
        and math.isfinite(float(max_p95_overhead_pct))
        and float(max_p95_overhead_pct)
        == FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT
    )
    serialized_threshold = (
        float(max_p95_overhead_pct)
        if isinstance(max_p95_overhead_pct, (int, float))
        and not isinstance(max_p95_overhead_pct, bool)
        and math.isfinite(float(max_p95_overhead_pct))
        else None
    )
    config_drift = {
        field: [off_config.get(field), on_config.get(field)]
        for field in MATCHED_CONFIG_FIELDS
        if off_config.get(field) != on_config.get(field)
    }
    row_schema_errors: list[str] = []
    try:
        on_keys = [_key(row) for row in on_rows]
        off_keys = [_key(row) for row in off_rows]
    except (KeyError, TypeError, ValueError) as exc:
        on_keys, off_keys = [], []
        _append_error(row_schema_errors, "row_key", exc)
    on_map = dict(zip(on_keys, on_rows))
    off_map = dict(zip(off_keys, off_rows))
    duplicate_keys = len(on_map) != len(on_rows) or len(off_map) != len(off_rows)
    try:
        expected_keys = set(
            product(
                [int(value) for value in on_config.get("prompt_lengths", [])],
                [int(value) for value in on_config.get("batch_sizes", [])],
                range(int(on_config.get("groups", 0))),
                range(int(on_config.get("within_process_repeats", 0))),
            )
        )
    except (TypeError, ValueError) as exc:
        expected_keys = set()
        _append_error(row_schema_errors, "config_grid", exc)
    missing_on = sorted(set(off_map) - set(on_map))
    missing_off = sorted(set(on_map) - set(off_map))
    incomplete_on = sorted(expected_keys - set(on_map))
    incomplete_off = sorted(expected_keys - set(off_map))
    unexpected_on = sorted(set(on_map) - expected_keys)
    unexpected_off = sorted(set(off_map) - expected_keys)
    common = sorted(set(on_map) & set(off_map))

    comparisons: list[dict[str, Any]] = []
    token_mismatches: list[list[int]] = []
    digest_mismatches: list[list[int]] = []
    timing_errors: list[list[int]] = []
    timing_validation_errors: list[dict[str, Any]] = []
    for key in common:
        on = on_map[key]
        off = off_map[key]
        try:
            if on["prompt_token_ids_sha256"] != off["prompt_token_ids_sha256"]:
                digest_mismatches.append(list(key))
            if _tokens(on) != _tokens(off):
                token_mismatches.append(list(key))
        except (KeyError, TypeError, ValueError) as exc:
            _append_error(row_schema_errors, f"row_payload:{key}", exc)
        try:
            on_timing = _recompute_row_timing(
                on, int(on_config["output_tokens"]), int(on["batch_size"])
            )
            off_timing = _recompute_row_timing(
                off, int(off_config["output_tokens"]), int(off["batch_size"])
            )
            on_wall = on_timing["wall_ms"]
            off_wall = off_timing["wall_ms"]
            on_tpot = on_timing["request_tpot_p95_ms"]
            off_tpot = off_timing["request_tpot_p95_ms"]
            wall_delta = 100.0 * (float(on_wall) / float(off_wall) - 1.0)
            tpot_delta = 100.0 * (float(on_tpot) / float(off_tpot) - 1.0)
            comparisons.append(
                {
                    "key": list(key),
                    "wall_overhead_pct": wall_delta,
                    "wall_absolute_deviation_pct": abs(wall_delta),
                    "tpot_p95_overhead_pct": tpot_delta,
                    "tpot_p95_absolute_deviation_pct": abs(tpot_delta),
                }
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            timing_errors.append(list(key))
            timing_validation_errors.append(
                {"key": list(key), "error": f"{type(exc).__name__}:{exc}"}
            )

    invalid = bool(
        not threshold_valid
        or config_drift
        or duplicate_keys
        or missing_on
        or missing_off
        or incomplete_on
        or incomplete_off
        or unexpected_on
        or unexpected_off
        or not expected_keys
        or digest_mismatches
        or token_mismatches
        or timing_errors
        or row_schema_errors
        or len(comparisons) != len(common)
        or not comparisons
        or on_config.get("capture_routes") is not True
        or off_config.get("capture_routes") is not False
    )
    wall = [row["wall_overhead_pct"] for row in comparisons]
    tpot = [row["tpot_p95_overhead_pct"] for row in comparisons]
    wall_absolute = [row["wall_absolute_deviation_pct"] for row in comparisons]
    tpot_absolute = [
        row["tpot_p95_absolute_deviation_pct"] for row in comparisons
    ]
    wall_p95 = _quantile(wall, 0.95)
    tpot_p95 = _quantile(tpot, 0.95)
    wall_absolute_p95 = _quantile(wall_absolute, 0.95)
    tpot_absolute_p95 = _quantile(tpot_absolute, 0.95)
    if invalid:
        status = "INVALID_TELEMETRY_PAIR"
    elif max(wall_absolute_p95, tpot_absolute_p95) <= (
        FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT
    ):
        status = "TELEMETRY_OVERHEAD_QUALIFIED"
    else:
        status = "ROUTE_EXPORT_TOO_EXPENSIVE_FOR_TIMING_CLAIM"

    per_cell: list[dict[str, Any]] = []
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        cells[(row["key"][0], row["key"][1])].append(row)
    for (prompt_length, batch_size), rows in sorted(cells.items()):
        per_cell.append(
            {
                "prompt_length": prompt_length,
                "batch_size": batch_size,
                "pairs": len(rows),
                "wall_overhead_p50_pct": _quantile(
                    [row["wall_overhead_pct"] for row in rows], 0.50
                ),
                "wall_overhead_p95_pct": _quantile(
                    [row["wall_overhead_pct"] for row in rows], 0.95
                ),
                "wall_absolute_deviation_p95_pct": _quantile(
                    [row["wall_absolute_deviation_pct"] for row in rows], 0.95
                ),
                "tpot_p95_overhead_p50_pct": _quantile(
                    [row["tpot_p95_overhead_pct"] for row in rows], 0.50
                ),
                "tpot_p95_overhead_p95_pct": _quantile(
                    [row["tpot_p95_overhead_pct"] for row in rows], 0.95
                ),
                "tpot_p95_absolute_deviation_p95_pct": _quantile(
                    [row["tpot_p95_absolute_deviation_pct"] for row in rows],
                    0.95,
                ),
            }
        )
    return {
        "schema": "vllm-route-telemetry-parity-v2",
        "status": status,
        "claim_ceiling": "NATIVE_OFFLINE_FIXED_BATCH_TELEMETRY_QUALIFICATION",
        "thresholds": {"max_p95_absolute_deviation_pct": serialized_threshold},
        "frozen_max_p95_absolute_deviation_pct": (
            FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT
        ),
        "gate_metric": "max(wall_abs_pairwise_delta_p95,tpot_abs_pairwise_delta_p95)",
        "pair_count": len(comparisons),
        "token_parity": not token_mismatches and not row_schema_errors,
        "threshold_valid": threshold_valid,
        "config_drift": config_drift,
        "duplicate_keys": duplicate_keys,
        "missing_on": [list(key) for key in missing_on],
        "missing_off": [list(key) for key in missing_off],
        "incomplete_on": [list(key) for key in incomplete_on],
        "incomplete_off": [list(key) for key in incomplete_off],
        "unexpected_on": [list(key) for key in unexpected_on],
        "unexpected_off": [list(key) for key in unexpected_off],
        "prompt_digest_mismatches": digest_mismatches,
        "token_mismatches": token_mismatches,
        "timing_errors": timing_errors,
        "timing_validation_errors": timing_validation_errors,
        "row_schema_errors": row_schema_errors,
        "wall_overhead_p50_pct": _quantile(wall, 0.50),
        "wall_overhead_p95_pct": wall_p95,
        "wall_absolute_deviation_p95_pct": wall_absolute_p95,
        "tpot_p95_overhead_p50_pct": _quantile(tpot, 0.50),
        "tpot_p95_overhead_p95_pct": tpot_p95,
        "tpot_p95_absolute_deviation_p95_pct": tpot_absolute_p95,
        "per_cell": per_cell,
        "pairs": comparisons,
    }


def _invalid_report(max_p95_overhead_pct: float) -> dict[str, Any]:
    serialized_threshold = (
        float(max_p95_overhead_pct)
        if isinstance(max_p95_overhead_pct, (int, float))
        and not isinstance(max_p95_overhead_pct, bool)
        and math.isfinite(float(max_p95_overhead_pct))
        else None
    )
    return {
        "schema": "vllm-route-telemetry-parity-v2",
        "status": "INVALID_TELEMETRY_PAIR",
        "claim_ceiling": "NATIVE_OFFLINE_FIXED_BATCH_TELEMETRY_QUALIFICATION",
        "thresholds": {"max_p95_absolute_deviation_pct": serialized_threshold},
        "pair_count": 0,
        "token_parity": False,
        "pairs": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-on", type=Path, required=True)
    parser.add_argument("--route-off", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-p95-overhead-pct", type=float, default=5.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; comparison artifacts are write-once")

    on_integrity, on_payload = _inspect_bundle(args.route_on)
    off_integrity, off_payload = _inspect_bundle(args.route_off)
    if on_payload is not None and off_payload is not None:
        report = compare_runs(
            on_payload["config"],
            off_payload["config"],
            on_payload["rows"],
            off_payload["rows"],
            args.max_p95_overhead_pct,
        )
    else:
        report = _invalid_report(args.max_p95_overhead_pct)
    report["bundle_integrity"] = {
        "route_on": on_integrity,
        "route_off": off_integrity,
    }
    if not on_integrity["valid"] or not off_integrity["valid"]:
        report["status"] = "INVALID_TELEMETRY_PAIR"
    qualification_prerequisites = {
        "producer_source_verified": bool(
            on_integrity.get("producer_source_verified")
            and off_integrity.get("producer_source_verified")
        ),
        "producer_source_semantics_approved": bool(
            on_integrity.get("producer_source_semantics_approved")
            and off_integrity.get("producer_source_semantics_approved")
        ),
        "exclusive_gpu_verified": bool(
            on_integrity.get("exclusive_gpu_verified")
            and off_integrity.get("exclusive_gpu_verified")
        ),
        "request_timing_evidence_verified": bool(
            on_integrity.get("timing_evidence_verified")
            and off_integrity.get("timing_evidence_verified")
        ),
    }
    report["qualification_prerequisites"] = qualification_prerequisites
    if (
        report["status"] != "INVALID_TELEMETRY_PAIR"
        and not all(qualification_prerequisites.values())
    ):
        # Legacy bundles remain usable for structural measurement, but a
        # standalone comparator must never turn missing provenance, timing
        # evidence, or GPU-isolation evidence into a telemetry success.
        report["status"] = "PROVENANCE_OR_ENVIRONMENT_UNQUALIFIED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    )
    print(json.dumps({k: report[k] for k in ("status", "pair_count", "token_parity")}))
    if report["status"] == "TELEMETRY_OVERHEAD_QUALIFIED":
        raise SystemExit(0)
    if report["status"] == "ROUTE_EXPORT_TOO_EXPENSIVE_FOR_TIMING_CLAIM":
        raise SystemExit(1)
    if report["status"] == "PROVENANCE_OR_ENVIRONMENT_UNQUALIFIED":
        raise SystemExit(1)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
