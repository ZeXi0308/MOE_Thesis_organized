#!/usr/bin/env python3
"""Fail-closed comparison of low/mid/high native vLLM cap branches.

Decision timing is reconstructed from the complete request ledger. Route
pressure is reconstructed from the sealed route tensor. ``summary.json`` is
checked against those reconstructions but is never the decision source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "vllm-native-decode-cap-branch-analysis-v1"
BRANCH_SCHEMA = "vllm-native-initial-decode-cap-branch-v1"
CLAIM_CEILING = "REQUEST_LEVEL_INITIAL_SAME_START_BRANCH_EXPLORATORY"
HELPER_SOURCE_ARTIFACT = "run_vllm_route_shape_probe.py"
EXPECTED_VLLM_VERSION = "0.26.0"
EXPECTED_RUNTIME_PATCH_ID = "valid-window-clear-v1"
APPROVED_BRANCH_RUNNER_SOURCE = Path(__file__).with_name(
    "run_vllm_decode_cap_branch.py"
)
APPROVED_HELPER_SOURCE = Path(__file__).with_name(HELPER_SOURCE_ARTIFACT)
APPROVED_BRANCH_RUNNER_SHA256 = hashlib.sha256(
    APPROVED_BRANCH_RUNNER_SOURCE.read_bytes()
).hexdigest()
APPROVED_HELPER_SHA256 = hashlib.sha256(
    APPROVED_HELPER_SOURCE.read_bytes()
).hexdigest()
FROZEN_MAX_TELEMETRY_DEVIATION_PCT = 5.0
FROZEN_MIN_HEADROOM_PCT = 3.0
EXPECTED_ACTUATOR_SOURCE_SHA256 = {
    "entrypoints/offline_utils.py": "688fbad0af9c2180b83aa77dcd0dbda85ca076a6c72bffa61840896d950cf458",
    "v1/core/sched/scheduler.py": "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941",
    "v1/core/sched/request_queue.py": "4b8d938e5fb8152fe61030b8aa991f983fcac9a405c37ed8908045e05e82ae5e",
    "config/scheduler.py": "a816cf79a3e74ffc0984f9bebb274275b26f46be8b28cb77a29388a0996263c8",
}
EXPECTED_TELEMETRY_SOURCE_SHA256 = {
    "model_executor/layers/fused_moe/routed_experts_capturer.py": "690de10ebd1ccb4ce156f8432ec351513d59702a1eacd9d1bf86dabb2b54226e",
    "v1/worker/gpu_model_runner.py": "1253bec5fafbc8e4ad0ea2735b50d2b6e6f4f97f02300fd489f4d29b2a2ee8ac",
}
ARMS = ("low", "mid", "high")
ROW_TIMING_FIELDS = (
    "ttft_ms",
    "queue_ms",
    "prefill_ms",
    "decode_span_ms",
    "tpot_ms",
    "e2e_ms",
)
TIMING_METRICS = (
    "wall_ms",
    "throughput_output_tokens_per_s",
    "request_ttft_p95_ms",
    "request_queue_p95_ms",
    "request_tpot_p95_ms",
    "request_e2e_p95_ms",
)
BASE_ARTIFACTS = {
    "config.json",
    "environment.json",
    "workload_manifest.json",
    "input_cohort.npz",
    "requests.jsonl",
    "summary.json",
    "producer_source.py",
    HELPER_SOURCE_ARTIFACT,
}
POSITIVE_STATUS = "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_POSITIVE"
VALID_NONPOSITIVE_STATUSES = {
    "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_BELOW_GATE",
    "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_TELEMETRY_INVALID",
    "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_SLO_UNSET",
    "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_UNDEFINED_RELATIVE_BASELINE",
}
INVALID_STATUS = "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_INVALID_INPUT"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("requests.jsonl must contain JSON objects only")
    return rows


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_bundle(run_dir: Path) -> dict[str, Any]:
    """Verify an exact, self-contained bundle and every seal identity."""
    errors: list[str] = []
    required = BASE_ARTIFACTS | {"ARTIFACT_HASHES.json", "RUN_COMPLETE.json"}
    for relative in sorted(required):
        if not (run_dir / relative).is_file():
            errors.append(f"missing:{relative}")
    if errors:
        return {"valid": False, "errors": errors}

    try:
        config = _load_json(run_dir / "config.json")
        artifact_hashes = _load_json(run_dir / "ARTIFACT_HASHES.json")
        seal = _load_json(run_dir / "RUN_COMPLETE.json")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {"valid": False, "errors": [f"unreadable_metadata:{exc}"]}

    expected_artifacts = BASE_ARTIFACTS | (
        {"routes.npz"} if bool(config.get("capture_routes")) else set()
    )
    if set(artifact_hashes) != expected_artifacts:
        missing = sorted(expected_artifacts - set(artifact_hashes))
        extra = sorted(set(artifact_hashes) - expected_artifacts)
        if missing:
            errors.append(f"artifact_set_missing:{missing}")
        if extra:
            errors.append(f"artifact_set_extra:{extra}")

    present_payloads = {
        path.name
        for path in run_dir.iterdir()
        if path.is_file()
        and path.name not in {"ARTIFACT_HASHES.json", "RUN_COMPLETE.json"}
    }
    if present_payloads != expected_artifacts:
        errors.append("bundle_file_set_mismatch")

    for relative, expected in artifact_hashes.items():
        if not _valid_sha256(expected):
            errors.append(f"invalid_artifact_digest:{relative}")
            continue
        path = run_dir / relative
        try:
            path.resolve().relative_to(run_dir.resolve())
        except ValueError:
            errors.append(f"unsafe_artifact_path:{relative}")
            continue
        if not path.is_file():
            errors.append(f"artifact_missing:{relative}")
        elif _sha256(path) != expected:
            errors.append(f"artifact_hash:{relative}")

    if seal.get("status") != "RUN_COMPLETE" or seal.get("schema") != BRANCH_SCHEMA:
        errors.append("seal_status_or_schema")
    if seal.get("artifact_hashes") != artifact_hashes:
        errors.append("seal_artifact_map_mismatch")
    if seal.get("artifact_hashes_sha256") != _sha256(
        run_dir / "ARTIFACT_HASHES.json"
    ):
        errors.append("seal_hash:artifact_hashes_sha256")
    sealed = {
        "config_sha256": "config.json",
        "requests_sha256": "requests.jsonl",
        "summary_sha256": "summary.json",
        "producer_source_sha256": "producer_source.py",
        "helper_source_sha256": HELPER_SOURCE_ARTIFACT,
    }
    for field, relative in sealed.items():
        if seal.get(field) != artifact_hashes.get(relative):
            errors.append(f"seal_hash:{field}")
    return {"valid": not errors, "errors": errors}


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite timing: {name}")
    if positive and number <= 0:
        raise ValueError(f"non-positive timing: {name}")
    return number


def _validate_actuator_identity(runtime_identity: Mapping[str, Any]) -> None:
    if runtime_identity.get("vllm") != EXPECTED_VLLM_VERSION:
        raise ValueError("decode-cap Gate requires exact vLLM 0.26.0")
    sources = runtime_identity.get("vllm_actuator_sources")
    if not isinstance(sources, Mapping) or set(sources) != set(
        EXPECTED_ACTUATOR_SOURCE_SHA256
    ):
        raise ValueError("vLLM actuator source map mismatch")
    for relative, expected_sha in EXPECTED_ACTUATOR_SOURCE_SHA256.items():
        entry = sources.get(relative)
        if not isinstance(entry, Mapping) or entry.get("sha256") != expected_sha:
            raise ValueError(f"vLLM actuator source hash mismatch: {relative}")
        size = entry.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"vLLM actuator source size invalid: {relative}")
    if runtime_identity.get("compute_processes_before_engine_init") != []:
        raise ValueError("decode-cap Gate requires an isolated GPU")
    telemetry_sources = runtime_identity.get("vllm_runtime_sources")
    if not isinstance(telemetry_sources, Mapping) or set(telemetry_sources) != set(
        EXPECTED_TELEMETRY_SOURCE_SHA256
    ):
        raise ValueError("valid-window telemetry source map mismatch")
    for relative, expected_sha in EXPECTED_TELEMETRY_SOURCE_SHA256.items():
        entry = telemetry_sources.get(relative)
        if not isinstance(entry, Mapping) or entry.get("sha256") != expected_sha:
            raise ValueError(f"valid-window telemetry source hash mismatch: {relative}")
        size = entry.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"valid-window telemetry source size invalid: {relative}")


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7)


def _assert_equivalent(expected: Any, observed: Any, name: str) -> None:
    """Compare JSON-shaped evidence, tolerating only float serialization noise."""
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        if set(expected) != set(observed):
            raise ValueError(f"{name} field set mismatch")
        for key in expected:
            _assert_equivalent(expected[key], observed[key], f"{name}.{key}")
        return
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            raise ValueError(f"{name} length mismatch")
        for index, (left, right) in enumerate(zip(expected, observed)):
            _assert_equivalent(left, right, f"{name}[{index}]")
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(observed, (int, float))
        and not isinstance(observed, bool)
    ):
        if not _close(float(expected), float(observed)):
            raise ValueError(f"{name} numeric mismatch")
        return
    if expected != observed:
        raise ValueError(f"{name} mismatch")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty timing denominator")
    return float(np.quantile(values, q))


def _validate_denominator(
    run_dir: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    _validate_rows_denominator(config, rows)
    cohort = int(config["cohort_size"])
    output_tokens = int(config["output_tokens"])
    prompt_length = int(config["prompt_length"])
    if prompt_length < 1:
        raise ValueError("invalid frozen generation denominator")

    timing = summary["timing"]
    if (
        int(timing["request_count"]) != cohort
        or int(timing["completed_request_count"]) != cohort
        or int(timing["total_generated_tokens"]) != cohort * output_tokens
        or int(timing["total_decode_intervals"]) != cohort * (output_tokens - 1)
    ):
        raise ValueError("summary generation denominator is incomplete")

    with np.load(run_dir / "input_cohort.npz", allow_pickle=False) as payload:
        if payload.files != ["prompt_token_ids"]:
            raise ValueError("input cohort archive member set drift")
        prompts = np.asarray(payload["prompt_token_ids"])
    if not np.issubdtype(prompts.dtype, np.integer):
        raise ValueError("input cohort tensor dtype is not integral")
    if tuple(prompts.shape) != (cohort, prompt_length):
        raise ValueError("input cohort tensor shape drift")
    semantic = _json_hash(prompts.tolist())
    identity = config["experiment_identity"]
    if semantic != identity["prompt_token_ids_sha256"]:
        raise ValueError("input cohort semantic hash drift")
    ordered = sorted(rows, key=lambda row: int(row["cohort_index"]))
    for prompt, row in zip(prompts.tolist(), ordered):
        if row.get("prompt_token_ids_sha256") != _json_hash(prompt):
            raise ValueError("per-request prompt identity drift")
    if identity["workload_manifest_sha256"] != _sha256(
        run_dir / "workload_manifest.json"
    ):
        raise ValueError("workload manifest identity drift")


def _validate_rows_denominator(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    cohort = int(config["cohort_size"])
    output_tokens = int(config["output_tokens"])
    if cohort < 1 or output_tokens < 2:
        raise ValueError("invalid frozen generation denominator")
    indexes = [int(row["cohort_index"]) for row in rows]
    request_ids = [str(row["request_id"]) for row in rows]
    if len(rows) != cohort or sorted(indexes) != list(range(cohort)):
        raise ValueError("request cohort is incomplete or duplicated")
    if len(set(request_ids)) != cohort:
        raise ValueError("request IDs are not unique")
    for row in rows:
        token_ids = row["token_ids"]
        if (
            int(row["generated_tokens"]) != output_tokens
            or int(row["decode_intervals"]) != output_tokens - 1
            or not isinstance(token_ids, list)
            or len(token_ids) != output_tokens
            or any(
                not isinstance(token, int) or isinstance(token, bool)
                for token in token_ids
            )
            or row["finish_reason"] != "length"
        ):
            raise ValueError("request generation denominator is incomplete")


def _recompute_timing(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Reconstruct every decision timing metric from complete request rows."""
    if not rows:
        raise ValueError("request timing denominator is empty")
    output_tokens = int(config["output_tokens"])
    normalized: list[dict[str, float]] = []
    wall_intervals: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        raw = row.get("raw_timing_s")
        if not isinstance(raw, Mapping):
            raise ValueError(f"request {index} has no raw timing evidence")
        started = _finite(
            raw.get("branch_started_perf_counter"),
            f"request[{index}].branch_started_perf_counter",
        )
        finished = _finite(
            raw.get("branch_finished_perf_counter"),
            f"request[{index}].branch_finished_perf_counter",
        )
        if finished <= started:
            raise ValueError("non-positive branch wall timing")
        wall_intervals.append((started, finished))
        queued = _finite(raw.get("queued_ts"), f"request[{index}].queued_ts")
        scheduled = _finite(
            raw.get("scheduled_ts"), f"request[{index}].scheduled_ts"
        )
        first = _finite(
            raw.get("first_token_ts"), f"request[{index}].first_token_ts"
        )
        last = _finite(raw.get("last_token_ts"), f"request[{index}].last_token_ts")
        ttft_s = _finite(
            raw.get("first_token_latency"),
            f"request[{index}].first_token_latency",
            positive=True,
        )
        if not queued <= scheduled < first < last:
            raise ValueError(f"request {index} timing timestamps are not ordered")
        queue_s = scheduled - queued
        prefill_s = first - scheduled
        decode_s = last - first
        intervals = int(row["decode_intervals"])
        if intervals != output_tokens - 1 or intervals <= 0:
            raise ValueError("request decode interval denominator drift")
        calculated = {
            "ttft_ms": ttft_s * 1000.0,
            "queue_ms": queue_s * 1000.0,
            "prefill_ms": prefill_s * 1000.0,
            "decode_span_ms": decode_s * 1000.0,
            "tpot_ms": decode_s * 1000.0 / intervals,
            "e2e_ms": (ttft_s + decode_s) * 1000.0,
        }
        for field, value in calculated.items():
            observed = _finite(row.get(field), f"request[{index}].{field}")
            if field != "queue_ms" and value <= 0:
                raise ValueError(f"non-positive timing: request[{index}].{field}")
            if observed < 0 or (field != "queue_ms" and observed <= 0):
                raise ValueError(f"non-positive timing: request[{index}].{field}")
            if not _close(observed, value):
                raise ValueError(f"request {index} derived timing mismatch: {field}")
        normalized.append(calculated)

    first_wall = wall_intervals[0]
    if any(interval != first_wall for interval in wall_intervals[1:]):
        raise ValueError("branch wall timing differs across request rows")
    wall_s = first_wall[1] - first_wall[0]
    result: dict[str, Any] = {
        "request_count": len(rows),
        "completed_request_count": len(rows),
        "expected_output_tokens_per_request": output_tokens,
        "total_generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
        "total_decode_intervals": sum(int(row["decode_intervals"]) for row in rows),
        "wall_ms": wall_s * 1000.0,
        "throughput_output_tokens_per_s": sum(
            int(row["generated_tokens"]) for row in rows
        )
        / wall_s,
    }
    for field in ROW_TIMING_FIELDS:
        values = [row[field] for row in normalized]
        stem = field.removesuffix("_ms")
        result[f"request_{stem}_p50_ms"] = _quantile(values, 0.50)
        result[f"request_{stem}_p95_ms"] = _quantile(values, 0.95)
        result[f"request_{stem}_max_ms"] = max(values)
    for metric in TIMING_METRICS:
        value = _finite(
            result[metric],
            metric,
            positive=metric != "request_queue_p95_ms",
        )
        if metric == "request_queue_p95_ms" and value < 0:
            raise ValueError("negative timing: request_queue_p95_ms")
    return result


def _recompute_route_pressure(
    config: Mapping[str, Any], routes: np.ndarray | None
) -> dict[str, Any] | None:
    capture = bool(config["capture_routes"])
    if not capture:
        if routes is not None:
            raise ValueError("route-OFF branch unexpectedly contains routes")
        return None
    if routes is None:
        raise ValueError("route-ON branch has no route tensor")
    shape = config.get("route_shape")
    if not isinstance(shape, Mapping):
        raise ValueError("frozen route shape is missing")
    num_experts = int(shape["num_experts"])
    num_layers = int(shape["num_layers"])
    top_k = int(shape["top_k"])
    decode_steps = int(shape["decode_route_steps"])
    cohort = int(config["cohort_size"])
    cap = int(config["decode_cap"])
    if (
        num_experts < 2
        or num_layers < 1
        or top_k < 1
        or top_k > num_experts
        or decode_steps != int(config["output_tokens"]) - 1
    ):
        raise ValueError("invalid frozen route configuration")
    if cap < 1 or cohort % cap:
        raise ValueError("invalid FCFS cap denominator")
    if not np.issubdtype(routes.dtype, np.integer) or np.issubdtype(
        routes.dtype, np.bool_
    ):
        raise ValueError(f"route tensor dtype is not integral: {routes.dtype}")
    expected_shape = (cohort, decode_steps, num_layers, top_k)
    if tuple(routes.shape) != expected_shape:
        raise ValueError(
            f"route tensor shape drift: expected {expected_shape}, got {routes.shape}"
        )
    if np.any(routes < 0) or np.any(routes >= num_experts):
        raise ValueError("expert ID outside frozen range")
    if top_k > 1 and np.any(
        np.diff(np.sort(routes, axis=-1), axis=-1) == 0
    ):
        raise ValueError("route tensor contains duplicate top-k expert IDs")

    waves: list[dict[str, Any]] = []
    for wave_index, start in enumerate(range(0, cohort, cap)):
        wave = routes[start : start + cap]
        max_loads: list[float] = []
        concentrations: list[float] = []
        active_counts: list[float] = []
        load_cvs: list[float] = []
        for step in range(decode_steps):
            for layer in range(num_layers):
                ids = wave[:, step, layer, :].reshape(-1).astype(
                    np.int64, copy=False
                )
                counts = np.bincount(ids, minlength=num_experts).astype(np.float64)
                max_loads.append(float(counts.max()))
                concentrations.append(float(counts.max() / counts.sum()))
                active_counts.append(float(np.count_nonzero(counts)))
                load_cvs.append(float(counts.std() / counts.mean()))
        working_sets = [
            float(np.unique(wave[:, :, layer, :]).size)
            for layer in range(num_layers)
        ]
        waves.append(
            {
                "wave_index": wave_index,
                "cohort_index_start": start,
                "cohort_index_end_exclusive": start + cap,
                "max_layer_step_load": max(max_loads),
                "p95_layer_step_max_load": _quantile(max_loads, 0.95),
                "max_layer_step_concentration": max(concentrations),
                "mean_active_expert_fraction": float(np.mean(active_counts))
                / num_experts,
                "mean_load_cv": float(np.mean(load_cvs)),
                "mean_layer_working_set_fraction": float(np.mean(working_sets))
                / num_experts,
            }
        )
    loads = [float(wave["max_layer_step_load"]) for wave in waves]
    return {
        "scope": "INFERRED_EQUAL_LENGTH_FCFS_CAP_WAVES",
        "scheduler_trace_captured": False,
        "wave_count": len(waves),
        "max_expert_load_across_waves": max(loads),
        "p95_wave_max_expert_load": _quantile(loads, 0.95),
        "waves": waves,
    }


def _validate_branch_evidence(branch: Mapping[str, Any]) -> dict[str, Any]:
    config = branch["config"]
    summary = branch["summary"]
    rows = branch["rows"]
    if not isinstance(config, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("branch config and summary must be objects")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("branch rows must be a sequence")
    if not isinstance(config.get("capture_routes"), bool):
        raise ValueError("capture_routes must be boolean")
    if config.get("schema") != BRANCH_SCHEMA or summary.get("schema") != BRANCH_SCHEMA:
        raise ValueError("branch schema mismatch")
    if summary.get("status") != "COMPLETE":
        raise ValueError("branch summary is incomplete")
    if config.get("claim_ceiling") != CLAIM_CEILING or summary.get(
        "claim_ceiling"
    ) != CLAIM_CEILING:
        raise ValueError("branch claim ceiling mismatch")
    if config.get("runtime_patch_id") != EXPECTED_RUNTIME_PATCH_ID:
        raise ValueError("decode-cap Gate requires the valid-window runtime patch")
    experiment_identity = config.get("experiment_identity")
    runtime_identity = config.get("runtime_identity")
    if not isinstance(experiment_identity, Mapping):
        raise ValueError("experiment identity must be an object")
    if not isinstance(runtime_identity, Mapping):
        raise ValueError("runtime identity must be an object")
    if config.get("experiment_identity_sha256") != _json_hash(experiment_identity):
        raise ValueError("experiment identity hash is inconsistent")
    if config.get("runtime_identity_sha256") != _json_hash(runtime_identity):
        raise ValueError("runtime identity hash is inconsistent")
    _validate_actuator_identity(runtime_identity)
    if config.get("require_exclusive_gpu") is not True:
        raise ValueError("decode-cap Gate requires require_exclusive_gpu=true")
    if experiment_identity.get("require_exclusive_gpu") is not True:
        raise ValueError("experiment identity does not bind GPU isolation")
    if experiment_identity.get("compute_processes_before_engine_init") != []:
        raise ValueError("experiment identity does not prove an isolated GPU")
    producer_sha = config.get("producer_source_sha256")
    if not _valid_sha256(producer_sha):
        raise ValueError("producer source identity is invalid")
    if config.get("probe_script_sha256") != producer_sha:
        raise ValueError("probe/producer source identity mismatch")
    if experiment_identity.get("runner_sha256") != producer_sha:
        raise ValueError("experiment/producer source identity mismatch")
    if producer_sha != APPROVED_BRANCH_RUNNER_SHA256:
        raise ValueError("branch producer semantics are not approved")
    helper_sha = config.get("helper_source_sha256")
    if not _valid_sha256(helper_sha):
        raise ValueError("helper source identity is invalid")
    if config.get("helper_source_artifact") != HELPER_SOURCE_ARTIFACT:
        raise ValueError("helper source artifact path mismatch")
    if experiment_identity.get("helper_source_sha256") != helper_sha:
        raise ValueError("experiment/helper source identity mismatch")
    if helper_sha != APPROVED_HELPER_SHA256:
        raise ValueError("branch helper semantics are not approved")

    _validate_rows_denominator(config, rows)
    timing = _recompute_timing(config, rows)
    _assert_equivalent(timing, summary.get("timing"), "summary.timing")
    pressure = _recompute_route_pressure(config, branch.get("routes"))
    if pressure is None:
        if (
            summary.get("route_pressure") is not None
            or summary.get("route_shape") is not None
        ):
            raise ValueError("route-OFF summary contains route evidence")
    else:
        _assert_equivalent(
            config["route_shape"], summary.get("route_shape"), "summary.route_shape"
        )
        _assert_equivalent(
            pressure, summary.get("route_pressure"), "summary.route_pressure"
        )
    return {"timing": timing, "route_pressure": pressure}


def load_branch(run_dir: Path) -> dict[str, Any]:
    integrity = verify_bundle(run_dir)
    if not integrity["valid"]:
        raise ValueError(f"invalid sealed bundle {run_dir}: {integrity['errors']}")
    config = _load_json(run_dir / "config.json")
    summary = _load_json(run_dir / "summary.json")
    rows = _load_jsonl(run_dir / "requests.jsonl")
    environment = _load_json(run_dir / "environment.json")
    runtime_identity = config.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping):
        raise ValueError("runtime identity must be an object")
    if runtime_identity != {
        key: environment.get(key) for key in runtime_identity
    }:
        raise ValueError("runtime identity does not match sealed environment")
    producer_path = run_dir / str(config.get("producer_source_artifact"))
    if producer_path.name != "producer_source.py" or producer_path.parent != run_dir:
        raise ValueError("producer source artifact path mismatch")
    producer_sha = _sha256(producer_path)
    if producer_sha != config.get("producer_source_sha256"):
        raise ValueError("embedded producer source hash mismatch")
    if producer_sha != config.get("probe_script_sha256"):
        raise ValueError("embedded producer/probe source hash mismatch")
    helper_path = run_dir / str(config.get("helper_source_artifact"))
    if (
        helper_path.name != HELPER_SOURCE_ARTIFACT
        or helper_path.parent != run_dir
    ):
        raise ValueError("helper source artifact path mismatch")
    helper_sha = _sha256(helper_path)
    if helper_sha != config.get("helper_source_sha256"):
        raise ValueError("embedded helper source hash mismatch")
    experiment_identity = config.get("experiment_identity")
    if not isinstance(experiment_identity, Mapping) or experiment_identity.get(
        "helper_source_sha256"
    ) != helper_sha:
        raise ValueError("embedded experiment/helper source hash mismatch")
    _validate_denominator(run_dir, config, summary, rows)
    routes: np.ndarray | None = None
    if bool(config.get("capture_routes")):
        with np.load(run_dir / "routes.npz", allow_pickle=False) as payload:
            if payload.files != ["routes"]:
                raise ValueError("route archive member set drift")
            routes = np.asarray(payload["routes"])
    branch = {
        "path": str(run_dir),
        "integrity": integrity,
        "config": config,
        "summary": summary,
        "rows": rows,
        "routes": routes,
    }
    branch["recomputed"] = _validate_branch_evidence(branch)
    return branch


def _tokens(branch: Mapping[str, Any]) -> list[list[int]]:
    rows = sorted(branch["rows"], key=lambda row: int(row["cohort_index"]))
    return [list(row["token_ids"]) for row in rows]


def token_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or len(left_tokens) != len(right_tokens):
        raise ValueError("token comparison cohort size mismatch")
    mismatched_requests: list[int] = []
    matched, total = 0, 0
    for index, (lhs, rhs) in enumerate(zip(left_tokens, right_tokens)):
        if lhs != rhs:
            mismatched_requests.append(index)
        if len(lhs) != len(rhs):
            raise ValueError("token comparison denominator mismatch")
        matched += sum(a == b for a, b in zip(lhs, rhs))
        total += len(lhs)
    if total <= 0:
        raise ValueError("token comparison denominator is empty")
    return {
        "exact": not mismatched_requests,
        "exact_request_fraction": 1.0 - len(mismatched_requests) / len(left_tokens),
        "token_position_match_fraction": matched / total,
        "mismatched_cohort_indexes": mismatched_requests,
    }


def _pct(new: float, old: float) -> float:
    new = _finite(new, "percentage numerator")
    old = _finite(old, "percentage denominator", positive=True)
    if new < 0:
        raise ValueError("negative percentage numerator")
    return 100.0 * (new / old - 1.0)


def _slo_metrics(
    branch: Mapping[str, Any],
    timing: Mapping[str, Any],
    tpot_slo_ms: float,
    ttft_slo_ms: float,
) -> dict[str, Any]:
    rows = branch["rows"]
    wall_s = float(timing["wall_ms"]) / 1000.0
    output_tokens = int(branch["config"]["output_tokens"])
    tpot_pass = sum(float(row["tpot_ms"]) <= tpot_slo_ms for row in rows)
    ttft_pass = sum(float(row["ttft_ms"]) <= ttft_slo_ms for row in rows)
    joint_pass = sum(
        float(row["tpot_ms"]) <= tpot_slo_ms
        and float(row["ttft_ms"]) <= ttft_slo_ms
        for row in rows
    )
    return {
        "request_count": len(rows),
        "tpot_attainment": tpot_pass / len(rows),
        "ttft_attainment": ttft_pass / len(rows),
        "joint_attainment": joint_pass / len(rows),
        "joint_passed_requests": joint_pass,
        "joint_slo_goodput_requests_per_s": joint_pass / wall_s,
        "joint_slo_goodput_output_tokens_per_s": joint_pass * output_tokens / wall_s,
    }


def analyze(
    branches: Mapping[str, Mapping[str, Any]],
    *,
    tpot_slo_ms: float | None,
    ttft_slo_ms: float | None,
    max_telemetry_overhead_pct: float = 5.0,
    min_headroom_pct: float = 3.0,
) -> dict[str, Any]:
    if (tpot_slo_ms is None) != (ttft_slo_ms is None):
        raise ValueError("TPOT and TTFT SLO thresholds must be set together")
    if tpot_slo_ms is not None:
        _finite(tpot_slo_ms, "TPOT SLO", positive=True)
        _finite(ttft_slo_ms, "TTFT SLO", positive=True)
    if (
        not math.isfinite(max_telemetry_overhead_pct)
        or max_telemetry_overhead_pct != FROZEN_MAX_TELEMETRY_DEVIATION_PCT
    ):
        raise ValueError(
            "telemetry deviation threshold must equal the frozen 5.0 percent Gate"
        )
    if (
        not math.isfinite(min_headroom_pct)
        or min_headroom_pct != FROZEN_MIN_HEADROOM_PCT
    ):
        raise ValueError("headroom threshold must equal the frozen 3.0 percent Gate")
    expected = {f"{arm}_{mode}" for arm in ARMS for mode in ("off", "on")}
    if set(branches) != expected:
        raise ValueError(f"expected exactly six branches: {sorted(expected)}")

    evidence = {
        key: _validate_branch_evidence(branch) for key, branch in branches.items()
    }
    identities = {
        branch["config"]["experiment_identity_sha256"]
        for branch in branches.values()
    }
    runtimes = {
        branch["config"]["runtime_identity_sha256"] for branch in branches.values()
    }
    patch_ids = {branch["config"]["runtime_patch_id"] for branch in branches.values()}
    producer_sources = {
        branch["config"]["producer_source_sha256"] for branch in branches.values()
    }
    helper_sources = {
        branch["config"]["helper_source_sha256"] for branch in branches.values()
    }
    route_shapes = {
        _json_hash(branch["config"]["route_shape"]) for branch in branches.values()
    }
    if len(identities) != 1:
        raise ValueError("same-start input identity mismatch")
    if len(runtimes) != 1 or len(patch_ids) != 1:
        raise ValueError("runtime source or runtime patch mismatch")
    if len(producer_sources) != 1:
        raise ValueError("branch runner source mismatch")
    if len(helper_sources) != 1:
        raise ValueError("branch helper source mismatch")
    if len(route_shapes) != 1:
        raise ValueError("frozen route configuration mismatch")

    budgets: dict[str, int] = {}
    telemetry_pairs: dict[str, Any] = {}
    measurements: dict[str, Any] = {}
    for arm in ARMS:
        off, on = branches[f"{arm}_off"], branches[f"{arm}_on"]
        for mode, branch in (("off", off), ("on", on)):
            config, summary = branch["config"], branch["summary"]
            if config["budget_arm"] != arm or bool(config["capture_routes"]) != (
                mode == "on"
            ):
                raise ValueError(f"arm or telemetry label mismatch for {arm}_{mode}")
            if (
                summary.get("budget_arm") != arm
                or bool(summary.get("capture_routes")) != (mode == "on")
                or int(config["decode_cap"]) != int(summary["decode_cap"])
            ):
                raise ValueError(f"summary/config label mismatch for {arm}_{mode}")
        if int(off["config"]["decode_cap"]) != int(on["config"]["decode_cap"]):
            raise ValueError(f"OFF/ON cap mismatch for {arm}")
        budgets[arm] = int(off["config"]["decode_cap"])
        telemetry_pairs[arm] = token_comparison(off, on)
        off_timing = evidence[f"{arm}_off"]["timing"]
        on_timing = evidence[f"{arm}_on"]["timing"]
        signed_deviation = {
            metric: _pct(float(on_timing[metric]), float(off_timing[metric]))
            for metric in ("wall_ms", "request_tpot_p95_ms")
        }
        absolute_deviation = {
            metric: abs(value) for metric, value in signed_deviation.items()
        }
        telemetry_pairs[arm]["signed_deviation_pct"] = signed_deviation
        telemetry_pairs[arm]["absolute_deviation_pct"] = absolute_deviation
        telemetry_pairs[arm]["overhead_qualified"] = all(
            value <= max_telemetry_overhead_pct
            for value in absolute_deviation.values()
        )
        pressure = evidence[f"{arm}_on"]["route_pressure"]
        if pressure is None:
            raise ValueError(f"route-ON branch has no recomputed pressure for {arm}")
        measurements[arm] = {
            "route_off_timing": {
                metric: off_timing[metric] for metric in TIMING_METRICS
            },
            "route_on_signed_deviation_pct": signed_deviation,
            "route_on_absolute_deviation_pct": absolute_deviation,
            "route_pressure_raw_recomputed": pressure,
        }
    if not budgets["low"] < budgets["mid"] < budgets["high"]:
        raise ValueError(f"budgets are not strictly increasing: {budgets}")

    telemetry_transparent = all(pair["exact"] for pair in telemetry_pairs.values())
    telemetry_overhead_qualified = all(
        pair["overhead_qualified"] for pair in telemetry_pairs.values()
    )
    telemetry_join_qualified = telemetry_transparent and telemetry_overhead_qualified
    for arm in ARMS:
        measurements[arm]["route_pressure_joined"] = (
            measurements[arm]["route_pressure_raw_recomputed"]
            if telemetry_join_qualified
            else None
        )
    cross_budget = {
        "mid_vs_low": token_comparison(branches["low_off"], branches["mid_off"]),
        "high_vs_mid": token_comparison(branches["mid_off"], branches["high_off"]),
        "high_vs_low": token_comparison(branches["low_off"], branches["high_off"]),
    }
    slo_configured = tpot_slo_ms is not None and ttft_slo_ms is not None
    if slo_configured:
        for arm in ARMS:
            measurements[arm]["route_off_slo"] = _slo_metrics(
                branches[f"{arm}_off"],
                evidence[f"{arm}_off"]["timing"],
                float(tpot_slo_ms),
                float(ttft_slo_ms),
            )
    timing_eligible = telemetry_join_qualified and slo_configured
    timing_comparison: dict[str, Any] = {
        "status": "ELIGIBLE" if timing_eligible else "FAIL_CLOSED",
        "canonical_source": "requests.jsonl route-OFF recomputation",
        "comparisons": [],
    }
    if timing_eligible:
        for newer, older in (("mid", "low"), ("high", "mid"), ("high", "low")):
            current = measurements[newer]["route_off_timing"]
            baseline = measurements[older]["route_off_timing"]
            older_goodput = float(
                measurements[older]["route_off_slo"][
                    "joint_slo_goodput_output_tokens_per_s"
                ]
            )
            timing_comparison["comparisons"].append(
                {
                    "comparison": f"{newer}_vs_{older}",
                    "delta_pct": {
                        metric: _pct(float(current[metric]), float(baseline[metric]))
                        for metric in TIMING_METRICS
                    },
                    "joint_slo_goodput_delta_pct": _pct(
                        float(
                            measurements[newer]["route_off_slo"][
                                "joint_slo_goodput_output_tokens_per_s"
                            ]
                        ),
                        older_goodput,
                    )
                    if older_goodput > 0
                    else None,
                }
            )
    else:
        failures = []
        if not telemetry_transparent:
            failures.append("ROUTE_ON_OFF_TOKEN_PARITY_FAILED")
        if not telemetry_overhead_qualified:
            failures.append("ROUTE_ON_TIMING_ABSOLUTE_DEVIATION_ABOVE_GATE")
        if not slo_configured:
            failures.append("SLO_THRESHOLDS_UNSET_OR_INVALID")
        timing_comparison["failure_reasons"] = failures

    if timing_eligible:
        goodputs = {
            arm: float(
                measurements[arm]["route_off_slo"][
                    "joint_slo_goodput_output_tokens_per_s"
                ]
            )
            for arm in ARMS
        }
        best_arm = max(ARMS, key=lambda arm: goodputs[arm])
        low_goodput = goodputs["low"]
        improvement = (
            _pct(goodputs[best_arm], low_goodput) if low_goodput > 0 else None
        )
        positive = bool(
            low_goodput > 0
            and best_arm != "low"
            and improvement is not None
            and improvement >= min_headroom_pct
        )
        if low_goodput == 0:
            headroom_status = (
                "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_UNDEFINED_RELATIVE_BASELINE"
            )
        elif positive:
            headroom_status = POSITIVE_STATUS
        else:
            headroom_status = "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_BELOW_GATE"
        headroom = {
            "status": headroom_status,
            "best_arm": best_arm,
            "best_vs_low_goodput_pct": improvement,
            "low_arm_goodput_output_tokens_per_s": low_goodput,
            "minimum_headroom_pct": min_headroom_pct,
        }
    elif not slo_configured:
        headroom = {"status": "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_SLO_UNSET"}
    else:
        headroom = {
            "status": "INITIAL_ACTIVE_SEQUENCE_CAP_HEADROOM_TELEMETRY_INVALID"
        }

    return {
        "schema": SCHEMA,
        "status": headroom["status"],
        "claim_ceiling": CLAIM_CEILING,
        "experiment_identity_sha256": next(iter(identities)),
        "runtime_identity_sha256": next(iter(runtimes)),
        "runtime_patch_id": next(iter(patch_ids)),
        "branch_runner_sha256": next(iter(producer_sources)),
        "branch_helper_sha256": next(iter(helper_sources)),
        "budgets": budgets,
        "replication_scope": "SINGLE_FROZEN_SEXTET_EXPLORATORY",
        "complete_generation_denominator": True,
        "decision_evidence": {
            "timing": "recomputed from complete requests.jsonl raw timing",
            "pressure": "recomputed from sealed routes.npz",
            "summary_role": "consistency check only",
        },
        "gates": {
            "route_on_off_token_parity": telemetry_transparent,
            "telemetry_overhead_qualified": telemetry_overhead_qualified,
            "telemetry_join_qualified": telemetry_join_qualified,
            "max_absolute_telemetry_deviation_pct": max_telemetry_overhead_pct,
            "slo_thresholds_configured": slo_configured,
            "exclusive_gpu_bound_in_all_branches": True,
        },
        "telemetry_pairs": telemetry_pairs,
        "cross_budget_token_drift": cross_budget,
        "arm_measurements": measurements,
        "timing_comparison": timing_comparison,
        "slo_definition": {
            "tpot_slo_ms": tpot_slo_ms,
            "ttft_slo_ms": ttft_slo_ms,
            "denominator": "all frozen-cohort requests and fixed output tokens",
        },
        "headroom": headroom,
        "interpretation_boundary": [
            "cross-budget token drift is an action-conditioned trajectory outcome, not telemetry interference",
            "when cross-budget tokens drift, timing cannot be called a same-token execution-only effect",
            "max_num_seqs also changes initial prefill admission and queueing",
            "this is an initial same-start branch, not later-epoch snapshot replay",
            "no online controller or safe-capacity claim is authorized",
        ],
    }


def exit_code_for_status(status: str) -> int:
    if status == POSITIVE_STATUS:
        return 0
    if status in VALID_NONPOSITIVE_STATUSES:
        return 1
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm}-off", type=Path, required=True)
        parser.add_argument(f"--{arm}-on", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--max-telemetry-overhead-pct", type=float, default=5.0)
    parser.add_argument("--min-headroom-pct", type=float, default=3.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; analysis artifacts are write-once")
    paths = {
        f"{arm}_{mode}": getattr(args, f"{arm}_{mode}")
        for arm in ARMS
        for mode in ("off", "on")
    }
    try:
        report = analyze(
            {key: load_branch(path) for key, path in paths.items()},
            tpot_slo_ms=args.tpot_slo_ms,
            ttft_slo_ms=args.ttft_slo_ms,
            max_telemetry_overhead_pct=args.max_telemetry_overhead_pct,
            min_headroom_pct=args.min_headroom_pct,
        )
        report["input_bundles"] = {key: str(path) for key, path in paths.items()}
    except (EOFError, KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        report = {
            "schema": SCHEMA,
            "status": INVALID_STATUS,
            "claim_ceiling": CLAIM_CEILING,
            "error": str(exc),
            "input_bundles": {key: str(path) for key, path in paths.items()},
        }
    report["analysis_script_sha256"] = _sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return exit_code_for_status(report["status"])


if __name__ == "__main__":
    raise SystemExit(main())
