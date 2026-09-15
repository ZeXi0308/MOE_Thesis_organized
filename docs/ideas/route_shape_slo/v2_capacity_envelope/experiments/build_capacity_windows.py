#!/usr/bin/env python3
"""Build v2 capacity-envelope windows from qualified development captures.

The script deliberately reuses the identity-closed RouteShape-SLO v1 window
builder.  Before doing so it runs the sibling lightweight P0 qualifier and
materializes the descriptive, unhashed sidecar that the v1 builder consumes.
Only arrived unfinished requests are used for the custom waiting/running state;
the producer's future-unarrived counter is never read.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


class ProtocolError(RuntimeError):
    """Raised when capture identity, timing, or metadata does not close."""


HERE = Path(__file__).resolve().parent
BASE_BUILDER_PATH = HERE.parents[1] / "experiments" / "build_route_windows.py"
QUALIFIER_PATH = HERE / "qualify_dev_capture.py"
SIDECAR_NAME = "route_shape_slo_capture.json"
ALLOWED_QUALIFICATION_STATUSES = {
    "P0_DEV_READY",
    "P0_READY_WITH_PROXY_RUNTIME",
}


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module("route_shape_slo_v1_window_builder", BASE_BUILDER_PATH)
QUALIFIER = _load_module("route_capacity_envelope_p0_qualifier", QUALIFIER_PATH)


WINDOW_FIELDS = [
    "model",
    "model_key",
    "model_revision",
    "arrival_episode_id",
    "arrival_episode_independent",
    "episode_id",
    "arrival_regime",
    "split",
    "window_id",
    "batch_index",
    "window_start_us",
    "window_end_us",
    "feature_available_at_us",
    "request_ids",
    "request_ids_json",
    "document_ids",
    "document_ids_json",
    "decode_step",
    "decode_stage",
    "arrival_count",
    "arrival_rate_per_s",
    "custom_waiting_count",
    "custom_running_set",
    "custom_running_set_size",
    "running_sequences",
    "arrived_active_sequences",
    "active_tokens",
    "batch_tokens",
    "mean_logical_kv",
    "max_logical_kv",
    "mean_physical_kv",
    "max_physical_kv",
    "left_padding_ratio",
    "step_service_ms",
    "recent_step_ms",
    "recent_tokens_per_second",
    "tokens_completed",
    "route_max_expert_load",
    "route_max_mean",
    "route_cv",
    "route_hhi",
    "active_experts",
    "top1_share",
    "top1_expert_share",
    "max_expert_tokens",
    "cross_layer_max_pressure",
    "cross_layer_mean_pressure",
    "hotspot_persistence",
    "route_shape_ewma",
    "route_shape_delta",
    "expert_identity_turnover",
    "top1_share_persistence",
    "running_identity_turnover",
    "route_layer_count",
    "top_k",
    "per_layer_features_json",
    "timing_boundary",
    "queue_semantics",
    "kv_semantics",
    "evidence_type",
    "runtime_kind",
    "runtime_status",
    "serial_route_conformance",
    "serial_route_identity_match_fraction",
    "batch_dependent_route_observed",
    "runtime_representative",
    "native_serving_runtime",
    "instrumentation_overhead_measured",
    "fresh_holdout_sealed",
    "gate_weight_available",
    "scientific_result_eligible",
    "source_capture",
]


@dataclass(frozen=True)
class CaptureContract:
    capture_dir: Path
    episode_id: str
    arrival_regime: str
    model_id: str
    model_key: str
    model_revision: str
    num_experts: int
    runtime_kind: str
    request_ids: frozenset[str]
    document_ids: frozenset[str]
    qualification: Mapping[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{path} must contain one JSON object")
    return value


def read_optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProtocolError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ProtocolError(
                        f"{path}:{line_number} must contain one JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSONL rows: {path}") from exc
    if not rows:
        raise ProtocolError(f"empty JSONL file: {path}")
    return rows


def _unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ProtocolError(f"{label} must be a list")
    output = [str(item) for item in value]
    if any(not item for item in output) or len(output) != len(set(output)):
        raise ProtocolError(f"{label} must contain unique non-empty identities")
    return output


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"{label} must be finite")
    return result


def _turnover(current: set[str], previous: set[str] | None) -> float:
    if previous is None:
        return 0.0
    union = current | previous
    return 0.0 if not union else 1.0 - len(current & previous) / len(union)


def _qualify(
    capture_dir: Path, overhead_path: Path | None = None
) -> dict[str, Any]:
    try:
        result = QUALIFIER.qualify(capture_dir, overhead_path)
    except Exception as exc:
        raise ProtocolError(f"P0 qualification failed for {capture_dir}: {exc}") from exc
    if not isinstance(result, dict):
        raise ProtocolError("P0 qualifier did not return an object")
    status = str(result.get("status", ""))
    if status not in ALLOWED_QUALIFICATION_STATUSES:
        raise ProtocolError(f"capture is not P0-qualified: {status or 'missing status'}")
    checks = result.get("checks")
    if not isinstance(checks, Mapping):
        raise ProtocolError("P0 qualification lacks lightweight checks")
    if checks.get("route_request_step_latency_alignment") != "PASS":
        raise ProtocolError("P0 route/request/step/latency alignment did not pass")
    if not str(checks.get("causal_future_information", "")).startswith("PASS"):
        raise ProtocolError("P0 causal cutoff did not pass")
    hook_check = str(checks.get("hook_no_hook_distortion", ""))
    if hook_check not in {
        "PASS",
        "PASS_FIXED_BATCH_PROXY",
        "NOT_MEASURED",
    }:
        raise ProtocolError("P0 hook distortion check has an unsupported status")
    return result


def inspect_capture(
    capture_dir: Path,
    *,
    num_experts: int,
    overhead_path: Path | None = None,
) -> CaptureContract:
    capture_dir = capture_dir.resolve()
    if not capture_dir.is_dir():
        raise ProtocolError(f"capture directory does not exist: {capture_dir}")
    if num_experts <= 1:
        raise ProtocolError("num_experts must exceed one")

    manifest = read_json(capture_dir / "workload_manifest.json")
    if manifest.get("run_class") != "development":
        raise ProtocolError("v2 P0 accepts only development captures")
    marker = manifest.get("route_capacity_envelope")
    if not isinstance(marker, Mapping):
        raise ProtocolError("capture workload lacks route_capacity_envelope metadata")
    episode_id = str(marker.get("episode_id", ""))
    arrival_regime = str(marker.get("arrival_regime", ""))
    if not episode_id or arrival_regime not in {"steady", "bursty"}:
        raise ProtocolError("capture episode identity/regime is invalid")
    if marker.get("request_document_disjoint_across_episodes") is not True:
        raise ProtocolError("workload marker does not assert cross-episode disjointness")
    runtime_kind = str(marker.get("evidence_scope", ""))
    if runtime_kind != "development_custom_continuous_runtime":
        raise ProtocolError("capture runtime kind is not the frozen custom runtime")

    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ProtocolError("capture workload lacks model identity")
    model_id = str(model.get("id", ""))
    model_key = str(model.get("key") or model_id)
    model_revision = str(model.get("revision", ""))
    if not model_id or not model_key or not model_revision:
        raise ProtocolError("capture model id/key/revision is unresolved")
    declared_experts = model.get("num_experts")
    if declared_experts is not None and int(declared_experts) != num_experts:
        raise ProtocolError("CLI num_experts differs from workload model metadata")
    if overhead_path is not None:
        overhead = read_json(overhead_path)
        overhead_model = overhead.get("model")
        if (
            not isinstance(overhead_model, Mapping)
            or str(overhead_model.get("id", "")) != model_id
            or str(overhead_model.get("revision", "")) != model_revision
        ):
            raise ProtocolError("telemetry check model differs from capture model")

    manifest_requests = manifest.get("requests")
    if not isinstance(manifest_requests, list) or not manifest_requests:
        raise ProtocolError("capture workload requests are missing")
    manifest_request_ids: set[str] = set()
    manifest_documents: set[str] = set()
    for row in manifest_requests:
        if not isinstance(row, Mapping):
            raise ProtocolError("workload request must be an object")
        request_id = str(row.get("request_id", ""))
        document_id = str(row.get("document_id", ""))
        if not request_id or not document_id or request_id in manifest_request_ids:
            raise ProtocolError("workload request/document identity is invalid")
        manifest_request_ids.add(request_id)
        manifest_documents.add(document_id)

    ledger = read_jsonl(capture_dir / "request_ledger.jsonl")
    ledger_request_ids: set[str] = set()
    ledger_documents: set[str] = set()
    for row in ledger:
        request_id = str(row.get("request_id", ""))
        document_id = str(row.get("document_id", ""))
        if not request_id or not document_id or request_id in ledger_request_ids:
            raise ProtocolError("request-ledger identity is invalid or duplicated")
        ledger_request_ids.add(request_id)
        ledger_documents.add(document_id)
    if ledger_request_ids != manifest_request_ids or ledger_documents != manifest_documents:
        raise ProtocolError("workload and request-ledger identities do not close")

    qualification = _qualify(capture_dir, overhead_path)
    if Path(str(qualification.get("capture_dir", ""))).resolve() != capture_dir:
        raise ProtocolError("qualifier returned a different capture directory")
    if str(qualification.get("episode_id", "")) != episode_id:
        raise ProtocolError("qualifier/workload episode identity differs")
    if str(qualification.get("arrival_regime", "")) != arrival_regime:
        raise ProtocolError("qualifier/workload arrival regime differs")
    if int(qualification.get("request_count", -1)) != len(ledger_request_ids):
        raise ProtocolError("qualifier/request-ledger request counts differ")

    return CaptureContract(
        capture_dir=capture_dir,
        episode_id=episode_id,
        arrival_regime=arrival_regime,
        model_id=model_id,
        model_key=model_key,
        model_revision=model_revision,
        num_experts=num_experts,
        runtime_kind=runtime_kind,
        request_ids=frozenset(ledger_request_ids),
        document_ids=frozenset(ledger_documents),
        qualification=qualification,
    )


def _validate_contracts(contracts: Sequence[CaptureContract]) -> None:
    if len(contracts) != 2:
        raise ProtocolError("P0 requires exactly two captures: steady and bursty")
    paths = {contract.capture_dir for contract in contracts}
    episodes = {contract.episode_id for contract in contracts}
    regimes = {contract.arrival_regime for contract in contracts}
    identities = {
        (contract.model_id, contract.model_key, contract.model_revision)
        for contract in contracts
    }
    if len(paths) != len(contracts) or len(episodes) != len(contracts):
        raise ProtocolError("capture path or episode identity is duplicated")
    if regimes != {"steady", "bursty"}:
        raise ProtocolError("P0 captures must contain one steady and one bursty episode")
    if len(identities) != 1:
        raise ProtocolError("P0 captures do not share one exact model/revision")
    left, right = contracts
    if left.request_ids & right.request_ids:
        raise ProtocolError("request identities overlap across episodes")
    if left.document_ids & right.document_ids:
        raise ProtocolError("document identities overlap across episodes")


def _write_unbound_sidecar(contract: CaptureContract) -> None:
    sentinel = read_json(contract.capture_dir / "CAPTURE_COMPLETE.json")
    bound_files = sentinel.get("files")
    if isinstance(bound_files, Mapping) and SIDECAR_NAME in bound_files:
        raise ProtocolError("refusing to update a capture-hash-bound sidecar")

    path = contract.capture_dir / SIDECAR_NAME
    if path.is_symlink():
        raise ProtocolError("refusing to update a symlinked capture sidecar")
    current = read_optional_json(path)
    expected_identities = {
        "episode_id": contract.episode_id,
        "arrival_episode_id": contract.episode_id,
        "arrival_regime": contract.arrival_regime,
        "model_id": contract.model_id,
        "model_revision": contract.model_revision,
    }
    for key, expected in expected_identities.items():
        if key in current and str(current[key]) != expected:
            raise ProtocolError(f"existing sidecar {key} conflicts with workload marker")

    sidecar = {
        **current,
        "schema": "route-capacity-envelope-v2-unbound-sidecar-v1",
        "binding_status": "UNBOUND_DESCRIPTIVE_METADATA",
        **expected_identities,
        "arrival_episode_independent": True,
        "split": "unassigned",
        "num_experts": {contract.model_key: contract.num_experts},
        "runtime_kind": contract.runtime_kind,
        "qualification_status": str(contract.qualification["status"]),
        "runtime_representative": False,
        "instrumentation_overhead_measured": (
            str(contract.qualification["status"]) == "P0_DEV_READY"
        ),
        "fresh_holdout_sealed": False,
        "scientific_result_eligible": False,
    }
    payload = json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ProtocolError(f"stale temporary sidecar exists: {temporary}")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def augment_capture(contract: CaptureContract) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _write_unbound_sidecar(contract)
    try:
        base_rows, base_diagnostic = BASE.build_capture(
            contract.capture_dir,
            {contract.model_key: contract.num_experts},
        )
    except Exception as exc:
        raise ProtocolError(f"v1 window construction failed: {exc}") from exc

    batches = sorted(
        read_jsonl(contract.capture_dir / "decode_batches.jsonl"),
        key=lambda row: int(row["batch_index"]),
    )
    request_rows = read_jsonl(contract.capture_dir / "request_ledger.jsonl")
    request_index = {str(row["request_id"]): row for row in request_rows}
    if len(request_index) != len(request_rows):
        raise ProtocolError("duplicate request identity in request ledger")
    if len(base_rows) != len(batches):
        raise ProtocolError("v1 windows and decode batches differ in length")
    if int(contract.qualification.get("decode_windows", -1)) != len(batches):
        raise ProtocolError("qualifier and window builder disagree on decode-window count")

    arrivals: dict[str, float] = {}
    for request_id, row in request_index.items():
        arrivals[request_id] = _finite_float(
            row.get("arrival_us"), f"request {request_id} arrival_us"
        )
        prefill_start = _finite_float(
            row.get("prefill_start_us"), f"request {request_id} prefill_start_us"
        )
        if prefill_start < arrivals[request_id]:
            raise ProtocolError(f"request {request_id} was prefetched before arrival")

    output: list[dict[str, Any]] = []
    previously_arrived: set[str] = set()
    prior_arrival_observation_us = min(arrivals.values())
    prior_running: set[str] | None = None
    prior_top1_share: float | None = None
    for position, (base, batch) in enumerate(zip(base_rows, batches)):
        batch_index = int(batch.get("batch_index", -1))
        if batch_index != position or int(base.get("batch_index", -1)) != batch_index:
            raise ProtocolError("batch/window identity is not contiguous")
        start_us = _finite_float(base.get("window_start_us"), "window_start_us")
        end_us = _finite_float(base.get("window_end_us"), "window_end_us")
        if end_us <= start_us:
            raise ProtocolError("window end must be after window start")
        batch_start = _finite_float(batch.get("start_us"), "batch.start_us")
        batch_end = _finite_float(batch.get("end_us"), "batch.end_us")
        if abs(start_us - batch_start) > 1e-6 or abs(end_us - batch_end) > 1e-6:
            raise ProtocolError("v1 window and producer batch timing differ")
        if _finite_float(base.get("feature_available_at_us"), "feature timestamp") > end_us:
            raise ProtocolError("window feature timestamp leaks beyond window end")
        if str(base.get("episode_id")) != contract.episode_id:
            raise ProtocolError("v1 sidecar episode identity was not applied")
        if str(base.get("arrival_regime")) != contract.arrival_regime:
            raise ProtocolError("v1 sidecar arrival regime was not applied")
        if str(base.get("model")) != contract.model_key:
            raise ProtocolError("v1 route model differs from workload model key")
        if str(base.get("model_revision")) != contract.model_revision:
            raise ProtocolError("v1 model revision differs from workload revision")

        request_ids = _unique_strings(batch.get("request_ids"), "request_ids")
        active_ids = _unique_strings(
            batch.get("active_request_ids"), "active_request_ids"
        )
        decode_steps = batch.get("decode_steps")
        logical_raw = batch.get("prior_cache_lengths")
        padding_raw = batch.get("left_padding")
        if not isinstance(decode_steps, list):
            raise ProtocolError("decode_steps must be a list")
        if not isinstance(logical_raw, list) or not isinstance(padding_raw, list):
            raise ProtocolError("logical KV and left-padding vectors must be lists")
        batch_size = int(batch.get("batch_size", -1))
        if not (
            batch_size
            == len(request_ids)
            == len(decode_steps)
            == len(logical_raw)
            == len(padding_raw)
        ):
            raise ProtocolError("batch identity/decode/KV vectors are misaligned")
        running = set(request_ids)
        active = set(active_ids)
        if not running <= active:
            raise ProtocolError("custom running set is not a subset of active set")
        if not active <= set(request_index):
            raise ProtocolError("active set contains an unknown request")

        arrived_at_start = {
            request_id for request_id, arrival in arrivals.items() if arrival <= start_us
        }
        if not previously_arrived <= arrived_at_start:
            raise ProtocolError("arrived-request set regressed across windows")
        if not active <= arrived_at_start:
            raise ProtocolError("active request appears before its recorded arrival")
        arrival_count = len(arrived_at_start - previously_arrived)
        previously_arrived = arrived_at_start
        arrival_observation_span_us = start_us - prior_arrival_observation_us
        if arrival_observation_span_us < 0:
            raise ProtocolError("arrival observation time regressed")
        arrival_rate_per_s = (
            arrival_count * 1_000_000.0 / arrival_observation_span_us
            if arrival_observation_span_us > 0
            else 0.0
        )
        prior_arrival_observation_us = start_us

        logical_lengths = [int(value) for value in logical_raw]
        left_padding = [int(value) for value in padding_raw]
        if any(value < 0 for value in logical_lengths + left_padding):
            raise ProtocolError("logical KV and left padding must be non-negative")
        physical_lengths = [
            logical + padding
            for logical, padding in zip(logical_lengths, left_padding)
        ]
        if len(set(physical_lengths)) != 1:
            raise ProtocolError("left-padded batch lacks one common physical extent")
        physical_total = sum(physical_lengths)
        left_padding_ratio = (
            sum(left_padding) / physical_total if physical_total else 0.0
        )

        documents = [
            str(request_index[request_id].get("document_id", ""))
            for request_id in request_ids
        ]
        if any(not document for document in documents):
            raise ProtocolError("scheduled request lacks document identity")
        request_ids_json = json.dumps(request_ids, separators=(",", ":"))
        document_ids_json = json.dumps(documents, separators=(",", ":"))
        if str(base.get("request_ids_json")) != request_ids_json:
            raise ProtocolError("v1 window and producer request order differ")
        if str(base.get("document_ids_json")) != document_ids_json:
            raise ProtocolError("v1 window and producer document order differ")

        decode_step = fmean(int(value) for value in decode_steps)
        if abs(float(base.get("decode_stage")) - decode_step) > 1e-12:
            raise ProtocolError("v1 window and producer decode stages differ")
        step_service_ms = (end_us - start_us) / 1000.0
        if abs(float(base.get("step_service_ms")) - step_service_ms) > 1e-9:
            raise ProtocolError("v1 window and producer step latency differ")
        tokens_per_second = batch_size / (step_service_ms / 1000.0)

        max_expert_tokens = int(base["max_expert_tokens"])
        top1_share = float(base["top1_expert_share"])
        hotspot_persistence = float(base["hotspot_persistence"])
        top1_share_persistence = (
            0.0
            if prior_top1_share is None
            else 1.0
            - min(
                abs(top1_share - prior_top1_share)
                / max(abs(prior_top1_share), 1e-12),
                1.0,
            )
        )
        runtime_status = str(contract.qualification["status"])
        serial_route_conformance = str(
            contract.qualification["checks"]["serial_route_conformance"]
        )
        serial_route_match = float(
            contract.qualification["serial_route_identity_match_fraction"]
        )
        batch_dependent_route = bool(
            contract.qualification["batch_dependent_route_observed"]
        )
        output.append(
            {
                "model": contract.model_id,
                "model_key": contract.model_key,
                "model_revision": contract.model_revision,
                "arrival_episode_id": contract.episode_id,
                "arrival_episode_independent": "true",
                "episode_id": contract.episode_id,
                "arrival_regime": contract.arrival_regime,
                "split": "unassigned",
                "window_id": f"{contract.episode_id}:{batch_index:08d}",
                "batch_index": batch_index,
                "window_start_us": start_us,
                "window_end_us": end_us,
                "feature_available_at_us": end_us,
                "request_ids": request_ids_json,
                "request_ids_json": request_ids_json,
                "document_ids": document_ids_json,
                "document_ids_json": document_ids_json,
                "decode_step": decode_step,
                "decode_stage": decode_step,
                "arrival_count": arrival_count,
                "arrival_rate_per_s": arrival_rate_per_s,
                "custom_waiting_count": len(active - running),
                "custom_running_set": request_ids_json,
                "custom_running_set_size": len(running),
                "running_sequences": len(running),
                "arrived_active_sequences": len(active),
                "active_tokens": len(running),
                "batch_tokens": len(running),
                "mean_logical_kv": fmean(logical_lengths),
                "max_logical_kv": max(logical_lengths),
                "mean_physical_kv": fmean(physical_lengths),
                "max_physical_kv": max(physical_lengths),
                "left_padding_ratio": left_padding_ratio,
                "step_service_ms": step_service_ms,
                "recent_step_ms": step_service_ms,
                "recent_tokens_per_second": tokens_per_second,
                "tokens_completed": len(running),
                "route_max_expert_load": max_expert_tokens,
                "route_max_mean": float(base["route_max_mean"]),
                "route_cv": float(base["route_cv"]),
                "route_hhi": float(base["route_hhi"]),
                "active_experts": float(base["active_experts"]),
                "top1_share": top1_share,
                "top1_expert_share": top1_share,
                "max_expert_tokens": max_expert_tokens,
                "cross_layer_max_pressure": float(
                    base["cross_layer_max_pressure"]
                ),
                "cross_layer_mean_pressure": float(
                    base["cross_layer_mean_pressure"]
                ),
                "hotspot_persistence": hotspot_persistence,
                "route_shape_ewma": float(base["route_shape_ewma"]),
                "route_shape_delta": float(base["route_shape_delta"]),
                "expert_identity_turnover": (
                    0.0 if position == 0 else 1.0 - hotspot_persistence
                ),
                "top1_share_persistence": top1_share_persistence,
                "running_identity_turnover": _turnover(running, prior_running),
                "route_layer_count": int(base["route_layer_count"]),
                "top_k": int(base["top_k"]),
                "per_layer_features_json": str(base["per_layer_features_json"]),
                "timing_boundary": str(base["timing_boundary"]),
                "queue_semantics": (
                    "custom_waiting_count=arrived_unfinished_minus_custom_running_set"
                ),
                "kv_semantics": (
                    "logical=prior_cache_tokens; physical=left_padded_token_extent,"
                    " not allocated KV bytes"
                ),
                "evidence_type": str(base["evidence_type"]),
                "runtime_kind": contract.runtime_kind,
                "runtime_status": runtime_status,
                "serial_route_conformance": serial_route_conformance,
                "serial_route_identity_match_fraction": serial_route_match,
                "batch_dependent_route_observed": str(
                    batch_dependent_route
                ).lower(),
                "runtime_representative": "false",
                "native_serving_runtime": "false",
                "instrumentation_overhead_measured": str(
                    runtime_status == "P0_DEV_READY"
                ).lower(),
                "fresh_holdout_sealed": "false",
                "gate_weight_available": "true",
                "scientific_result_eligible": "false",
                "source_capture": str(contract.capture_dir),
            }
        )
        prior_running = running
        prior_top1_share = top1_share

    diagnostic = {
        **base_diagnostic,
        "episode_id": contract.episode_id,
        "arrival_regime": contract.arrival_regime,
        "model": contract.model_id,
        "model_key": contract.model_key,
        "model_revision": contract.model_revision,
        "runtime_kind": contract.runtime_kind,
        "runtime_status": str(contract.qualification["status"]),
        "arrival_episode_independent": True,
        "custom_waiting_field": "custom_waiting_count",
        "custom_running_field": "custom_running_set",
        "future_unarrived_counter_consumed": False,
        "physical_kv_bytes_available": False,
        "future_fields_consumed": [],
        "qualification": dict(contract.qualification),
    }
    return output, diagnostic


def build_all(
    capture_dirs: Sequence[Path],
    *,
    num_experts: int,
    overhead_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contracts = [
        inspect_capture(
            path,
            num_experts=num_experts,
            overhead_path=overhead_path,
        )
        for path in capture_dirs
    ]
    _validate_contracts(contracts)

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for contract in sorted(contracts, key=lambda value: value.episode_id):
        built, diagnostic = augment_capture(contract)
        rows.extend(built)
        diagnostics.append(diagnostic)
    rows.sort(key=lambda row: (str(row["episode_id"]), int(row["batch_index"])))
    return rows, {
        "schema": "route-capacity-envelope-windows-diagnostic-v1",
        "status": "P0_WINDOWS_BUILT",
        "windows": len(rows),
        "episodes": sorted(contract.episode_id for contract in contracts),
        "arrival_regimes": sorted(contract.arrival_regime for contract in contracts),
        "model_identities": [
            [contracts[0].model_id, contracts[0].model_revision]
        ],
        "request_overlap_across_episodes": 0,
        "document_overlap_across_episodes": 0,
        "arrival_episodes_independent": True,
        "future_fields_consumed": [],
        "captures": diagnostics,
    }


def write_windows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ProtocolError("refusing to write an empty window table")
    if path.suffix.lower() != ".csv":
        raise ProtocolError("output path must end in .csv")
    if path.exists() or path.is_symlink():
        raise ProtocolError(f"refusing to overwrite windows: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=WINDOW_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", action="append", required=True)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--overhead")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, diagnostic = build_all(
        [Path(value) for value in args.capture_dir],
        num_experts=args.num_experts,
        overhead_path=Path(args.overhead).resolve() if args.overhead else None,
    )
    write_windows(Path(args.output).resolve(), rows)
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
