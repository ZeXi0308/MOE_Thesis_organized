#!/usr/bin/env python3
"""Build causal per-window workload and MoE route-shape features.

The primary input is a same-directory output from the existing continuous-decode
producer: routes.csv, decode_batches.jsonl, and request_ledger.jsonl. A narrowly
scoped ``--stablebatch-dir`` adapter can also reuse the frozen StableBatch
teacher-forced single-GPU artifact. The latter is always labelled an observed
isolated GPU primitive, never a representative serving/runtime/SLO trace.

The builder never shifts targets; every emitted feature is available by the end
of the same window. The analyzer is responsible for the explicit t -> t+1 join.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence


class ProtocolError(RuntimeError):
    pass


REQUIRED_ROUTE_COLUMNS = {
    "model",
    "phase",
    "request_id",
    "input_event_id",
    "decode_step",
    "layer_id",
    "topk_slot",
    "expert_id",
    "gate_weight",
    "document_id",
    "layer_ready_us",
    "route_end_us",
}

REQUIRED_BATCH_FIELDS = {
    "batch_index",
    "start_us",
    "end_us",
    "batch_size",
    "active_request_ids",
    "request_ids",
    "decode_steps",
    "prior_cache_lengths",
}

REQUIRED_REQUEST_FIELDS = {
    "request_id",
    "document_id",
    "arrival_us",
    "deadline_us",
    "prompt_tokens",
    "steps",
}

OUTPUT_FIELDS = [
    "model",
    "model_revision",
    "arrival_episode_id",
    "arrival_episode_independent",
    "episode_id",
    "split",
    "window_id",
    "batch_index",
    "window_start_us",
    "window_end_us",
    "feature_available_at_us",
    "decode_stage",
    "arrival_regime",
    "active_tokens",
    "running_sequences",
    "queue_depth",
    "mean_kv_length",
    "max_kv_length",
    "prompt_tokens",
    "decode_tokens",
    "batch_size",
    "batch_tokens",
    "step_service_ms",
    "tokens_completed",
    "recent_step_ms",
    "recent_tokens_per_second",
    "route_max_mean",
    "route_cv",
    "route_hhi",
    "active_experts",
    "top1_expert_share",
    "cross_layer_max_pressure",
    "cross_layer_mean_pressure",
    "hotspot_persistence",
    "route_shape_ewma",
    "route_shape_delta",
    "max_expert_tokens",
    "route_layer_count",
    "top_k",
    "per_layer_features_json",
    "request_ids_json",
    "document_ids_json",
    "evidence_type",
    "runtime_representative",
    "instrumentation_overhead_measured",
    "fresh_holdout_sealed",
    "gate_weight_available",
    "timing_boundary",
    "queue_depth_semantics",
    "source_capture",
]


STABLEBATCH_ARM = "native_variable_m"
STABLEBATCH_PHASE = "measured"
STABLEBATCH_REPEAT = 0
STABLEBATCH_ROSTER_SCHEMA = "stablebatch-shape-lane-native-roster-row-v1"
STABLEBATCH_STEP_SCHEMA = "stablebatch-shape-lane-decode-step-v1"
STABLEBATCH_CALL_SCHEMA = "stablebatch-shape-lane-expert-call-v1"
STABLEBATCH_CONFIG_SCHEMA = "stablebatch-shape-lane-continuous-cost-gate-v1"
STABLEBATCH_WORKLOAD_SCHEMA = "bcrd-continuous-workload-v1"
STABLEBATCH_STATUS_SCHEMA = "stablebatch-shape-lane-cost-run-status-v1"
STABLEBATCH_COMPLETE_SCHEMA = "stablebatch-shape-lane-cost-complete-v1"
BCRD_COMPLETE_SCHEMA = "bcrd-continuous-capture-complete-v1"


@dataclass(frozen=True)
class RouteSummary:
    """Permutation-invariant route-shape summary including zero-count experts."""

    top_k: int
    layer_count: int
    route_max_mean: float
    route_cv: float
    route_hhi: float
    active_experts: float
    top1_expert_share: float
    cross_layer_max_pressure: float
    cross_layer_mean_pressure: float
    max_expert_tokens: int
    hotspots: dict[int, int]
    per_layer: dict[int, dict[str, Any]]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"{path} must contain one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProtocolError(f"missing required ledger: {path}")
    output: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolError(f"{path}:{number} must be a JSON object")
            output.append(value)
    if not output:
        raise ProtocolError(f"empty required ledger: {path}")
    return output


def read_routes(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ProtocolError(f"missing required route trace: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_ROUTE_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ProtocolError(
                f"legacy/partial route trace lacks P0 fields: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ProtocolError(f"route trace is empty: {path}")
    for number, row in enumerate(rows, 2):
        try:
            weight = float(row["gate_weight"])
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"{path}:{number} gate_weight is invalid") from exc
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ProtocolError(f"{path}:{number} gate_weight is outside [0,1]")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_fields(row: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(row)
    if missing:
        raise ProtocolError(f"{label} lacks fields: {sorted(missing)}")


def parse_num_experts(values: Iterable[str]) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ProtocolError("--num-experts must use MODEL=COUNT")
        model, count = value.split("=", 1)
        parsed = int(count)
        if not model or parsed <= 1:
            raise ProtocolError("--num-experts requires a model and COUNT > 1")
        output[model] = parsed
    return output


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _capture_metadata(capture_dir: Path) -> dict[str, Any]:
    metadata = read_json(capture_dir / "route_shape_slo_capture.json")
    workload = read_json(capture_dir / "workload_manifest.json")
    sentinel = read_json(capture_dir / "CAPTURE_COMPLETE.json")
    environment = read_json(capture_dir / "environment.json")
    model_spec = workload.get("model") if isinstance(workload.get("model"), dict) else {}
    metadata.setdefault("episode_id", capture_dir.name)
    metadata.setdefault("arrival_episode_id", metadata["episode_id"])
    metadata["arrival_episode_independent"] = False
    metadata.setdefault("split", "unassigned")
    metadata.setdefault("arrival_regime", "unknown")
    metadata.setdefault("model_revision", model_spec.get("revision", "unknown"))
    if sentinel.get("scientific_result_eligible") is True:
        raise ProtocolError(
            "the P0 builder cannot validate a self-asserted scientific-result sentinel"
        )
    # This builder validates identity/timing alignment only. Descriptive sidecar
    # metadata must never upgrade P0 into a representative P1 result. A later
    # qualifier needs an independently verified schema and paired overhead run.
    metadata["evidence_type"] = (
        "[Observed real runtime]"
        if sentinel.get("schema") == BCRD_COMPLETE_SCHEMA
        and sentinel.get("status") == "CAPTURE_COMPLETE"
        and environment.get("cuda_available") is True
        else "[Synthetic fixture]"
    )
    metadata["runtime_representative"] = False
    metadata["instrumentation_overhead_measured"] = False
    metadata["fresh_holdout_sealed"] = False
    metadata.setdefault("timing_boundary", sentinel.get("timing_boundary", "unknown"))
    return metadata


def _layer_features(rows: Sequence[Mapping[str, str]], num_experts: int) -> dict[str, Any]:
    counts = [0] * num_experts
    for row in rows:
        expert = int(row["expert_id"])
        if expert < 0 or expert >= num_experts:
            raise ProtocolError(
                f"expert_id={expert} outside declared [0,{num_experts})"
            )
        counts[expert] += 1
    return _layer_features_from_counts(counts)


def _layer_features_from_counts(counts: Sequence[int]) -> dict[str, Any]:
    """Compute zero-inclusive, permutation-invariant expert-load features."""

    if len(counts) <= 1 or any(value < 0 for value in counts):
        raise ProtocolError("expert counts must be non-negative with width > 1")
    total = sum(counts)
    if total <= 0:
        raise ProtocolError("a route layer contains no contributions")
    mean = total / len(counts)
    maximum = max(counts)
    active = sum(value > 0 for value in counts)
    pressure = maximum / mean
    cv = pstdev(counts) / mean
    hhi = sum((value / total) ** 2 for value in counts)
    hotspot = min(index for index, value in enumerate(counts) if value == maximum)
    return {
        "max_expert_tokens": maximum,
        "max_over_mean": pressure,
        "cv": cv,
        "hhi": hhi,
        "active_experts": active,
        "top1_expert_share": maximum / total,
        "hotspot_expert": hotspot,
    }


def _route_summary(
    counts_by_layer: Mapping[int, Mapping[int, int]],
    *,
    num_experts: int,
    batch_size: int,
) -> RouteSummary:
    """Summarize one window and verify top-k conservation in every layer."""

    if not counts_by_layer:
        raise ProtocolError("route window contains no layers")
    if num_experts <= 1 or batch_size <= 0:
        raise ProtocolError("num_experts and batch_size must be positive")
    dense_by_layer: dict[int, list[int]] = {}
    layer_totals: dict[int, int] = {}
    for raw_layer, sparse in counts_by_layer.items():
        layer = int(raw_layer)
        dense = [0] * num_experts
        for raw_expert, raw_count in sparse.items():
            expert = int(raw_expert)
            count = int(raw_count)
            if expert < 0 or expert >= num_experts:
                raise ProtocolError(
                    f"expert_id={expert} outside declared [0,{num_experts})"
                )
            if count <= 0 or dense[expert]:
                raise ProtocolError("occupied expert counts must be unique and positive")
            dense[expert] = count
        dense_by_layer[layer] = dense
        layer_totals[layer] = sum(dense)
    if any(total <= 0 or total % batch_size for total in layer_totals.values()):
        raise ProtocolError("route layer does not conserve an integer top-k width")
    topks = {total // batch_size for total in layer_totals.values()}
    if len(topks) != 1:
        raise ProtocolError("top-k width changes across route layers")
    top_k = next(iter(topks))
    if top_k <= 0 or top_k > num_experts:
        raise ProtocolError("route top-k width is outside the expert domain")

    per_layer = {
        layer: _layer_features_from_counts(dense)
        for layer, dense in sorted(dense_by_layer.items())
    }
    pressures = [value["max_over_mean"] for value in per_layer.values()]
    return RouteSummary(
        top_k=top_k,
        layer_count=len(per_layer),
        route_max_mean=max(pressures),
        route_cv=fmean(value["cv"] for value in per_layer.values()),
        route_hhi=fmean(value["hhi"] for value in per_layer.values()),
        active_experts=fmean(
            value["active_experts"] for value in per_layer.values()
        ),
        top1_expert_share=fmean(
            value["top1_expert_share"] for value in per_layer.values()
        ),
        cross_layer_max_pressure=max(pressures),
        cross_layer_mean_pressure=fmean(pressures),
        max_expert_tokens=max(
            int(value["max_expert_tokens"]) for value in per_layer.values()
        ),
        hotspots={
            layer: int(value["hotspot_expert"])
            for layer, value in per_layer.items()
        },
        per_layer=per_layer,
    )


def _event_rows(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, int], list[Mapping[str, str]]]:
    output: dict[tuple[str, int], list[Mapping[str, str]]] = {}
    for row in rows:
        if row["phase"] != "decode":
            raise ProtocolError("prefill rows must not enter decode-window features")
        key = (row["request_id"], int(row["decode_step"]))
        output.setdefault(key, []).append(row)
    return output


def _validate_event(event: Sequence[Mapping[str, str]]) -> tuple[set[int], int]:
    layers: dict[int, set[int]] = {}
    experts_by_layer: dict[int, set[int]] = {}
    identities: set[tuple[int, int]] = set()
    input_events: set[str] = set()
    for row in event:
        layer = int(row["layer_id"])
        slot = int(row["topk_slot"])
        gate_weight = float(row["gate_weight"])
        if not math.isfinite(gate_weight) or gate_weight < 0:
            raise ProtocolError("gate weights must be finite and non-negative")
        input_event = str(row["input_event_id"])
        if not input_event:
            raise ProtocolError("route row has an empty input_event_id")
        input_events.add(input_event)
        identity = (layer, slot)
        if identity in identities:
            raise ProtocolError(f"duplicate layer/top-k slot in event: {identity}")
        identities.add(identity)
        layers.setdefault(layer, set()).add(slot)
        expert = int(row["expert_id"])
        experts = experts_by_layer.setdefault(layer, set())
        if expert in experts:
            raise ProtocolError(
                "one token/layer routes multiple top-k slots to the same expert"
            )
        experts.add(expert)
    topks = {len(slots) for slots in layers.values()}
    if len(topks) != 1:
        raise ProtocolError("top-k width changes across layers in one event")
    observed_layers = set(layers)
    if observed_layers != set(range(len(observed_layers))):
        raise ProtocolError("route layers must be contiguous from zero")
    top_k = next(iter(topks))
    expected_slots = set(range(top_k))
    if any(slots != expected_slots for slots in layers.values()):
        raise ProtocolError("top-k slots are not contiguous from zero")
    if len(input_events) != 1:
        raise ProtocolError("one request/decode event has multiple input_event_id values")
    return set(layers), top_k


def _request_index(requests: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in requests:
        require_fields(row, REQUIRED_REQUEST_FIELDS, "request ledger row")
        request_id = str(row["request_id"])
        if request_id in output:
            raise ProtocolError(f"duplicate request ledger row: {request_id}")
        output[request_id] = row
    return output


def _step_is_in_batch(row: Mapping[str, Any], decode_step: int, batch_index: int) -> bool:
    for step in row["steps"]:
        if (
            int(step.get("decode_step", -1)) == decode_step
            and int(step.get("batch_index", -1)) == batch_index
        ):
            return True
    return False


def build_capture(
    capture_dir: Path,
    explicit_num_experts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = _capture_metadata(capture_dir)
    routes = read_routes(capture_dir / "routes.csv")
    batches = read_jsonl(capture_dir / "decode_batches.jsonl")
    requests = read_jsonl(capture_dir / "request_ledger.jsonl")
    sentinel = read_json(capture_dir / "CAPTURE_COMPLETE.json")
    if sentinel.get("status") != "CAPTURE_COMPLETE":
        raise ProtocolError(f"capture lacks CAPTURE_COMPLETE status: {capture_dir}")
    if sentinel.get("schema") == BCRD_COMPLETE_SCHEMA:
        status = read_json(capture_dir / "RUN_STATUS.json")
        if (
            status.get("status") != "COMPLETE"
            or status.get("required_sentinel") != "CAPTURE_COMPLETE.json"
            or sentinel.get("producer_formal_eligible") is not False
            or sentinel.get("scientific_result_eligible") is not False
        ):
            raise ProtocolError("BCRD producer sentinel boundary is invalid")
        files = sentinel.get("files")
        if not isinstance(files, dict):
            raise ProtocolError("BCRD producer sentinel lacks bound file hashes")
        for name in (
            "routes.csv",
            "decode_batches.jsonl",
            "request_ledger.jsonl",
            "workload_manifest.json",
            "environment.json",
        ):
            expected = files.get(name)
            path = capture_dir / name
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or not path.is_file()
                or _sha256_file(path) != expected
            ):
                raise ProtocolError(f"BCRD producer hash mismatch for {name}")
    elif metadata["evidence_type"] != "[Synthetic fixture]":
        raise ProtocolError("unverified capture cannot claim observed runtime")
    event_rows = _event_rows(routes)
    request_index = _request_index(requests)
    used_events: set[tuple[str, int]] = set()
    output: list[dict[str, Any]] = []
    prior_hotspots: dict[int, int] = {}
    prior_pressure: float | None = None
    ewma: float | None = None
    alpha = 0.3
    seen_batch_indices: set[int] = set()
    prior_window_end_us: float | None = None

    for raw_batch in sorted(batches, key=lambda row: int(row["batch_index"])):
        require_fields(raw_batch, REQUIRED_BATCH_FIELDS, "decode batch row")
        batch_index = int(raw_batch["batch_index"])
        if batch_index in seen_batch_indices:
            raise ProtocolError(f"duplicate batch_index={batch_index}")
        if batch_index != len(seen_batch_indices):
            raise ProtocolError("decode batch indices must be contiguous from zero")
        seen_batch_indices.add(batch_index)
        request_ids = [str(value) for value in raw_batch["request_ids"]]
        decode_steps = [int(value) for value in raw_batch["decode_steps"]]
        active_ids = [str(value) for value in raw_batch["active_request_ids"]]
        kv_lengths = [int(value) for value in raw_batch["prior_cache_lengths"]]
        batch_size = int(raw_batch["batch_size"])
        if not (
            batch_size
            == len(request_ids)
            == len(decode_steps)
            == len(kv_lengths)
        ):
            raise ProtocolError("batch identity/KV vectors do not match batch_size")
        if len(set(request_ids)) != batch_size or not set(request_ids) <= set(active_ids):
            raise ProtocolError("scheduled requests are not a unique subset of active requests")
        if any(value < 0 for value in kv_lengths):
            raise ProtocolError("KV lengths must be non-negative")
        start_us = float(raw_batch["start_us"])
        end_us = float(raw_batch["end_us"])
        if not (math.isfinite(start_us) and math.isfinite(end_us) and end_us > start_us):
            raise ProtocolError("batch timestamps must be finite and strictly increasing")
        if prior_window_end_us is not None and start_us < prior_window_end_us:
            raise ProtocolError(
                "decode batch time regresses; route-history features would leak future state"
            )

        window_routes: list[Mapping[str, str]] = []
        event_layer_sets: list[set[int]] = []
        event_topks: list[int] = []
        documents: list[str] = []
        prompts: list[int] = []
        for request_id, decode_step in zip(request_ids, decode_steps):
            key = (request_id, decode_step)
            event = event_rows.get(key)
            if not event:
                raise ProtocolError(f"batch event has no route rows: {key}")
            if key in used_events:
                raise ProtocolError(f"route event appears in more than one batch: {key}")
            used_events.add(key)
            layers, top_k = _validate_event(event)
            event_layer_sets.append(layers)
            event_topks.append(top_k)
            ledger = request_index.get(request_id)
            if ledger is None:
                raise ProtocolError(f"missing request ledger identity: {request_id}")
            if not _step_is_in_batch(ledger, decode_step, batch_index):
                raise ProtocolError(
                    f"request/decode step does not point back to batch {batch_index}"
                )
            document = str(ledger["document_id"])
            prompts.append(int(ledger["prompt_tokens"]))
            documents.append(document)
            for row in event:
                if row["document_id"] != document:
                    raise ProtocolError("route and request document identities disagree")
                if abs(float(row["layer_ready_us"]) - start_us) > 1e-6:
                    raise ProtocolError("route layer_ready_us is not aligned to its window")
                if abs(float(row["route_end_us"]) - end_us) > 1e-6:
                    raise ProtocolError("route route_end_us is not aligned to its window")
            window_routes.extend(event)
        if len({frozenset(value) for value in event_layer_sets}) != 1:
            raise ProtocolError("route layer closure changes within one batch")
        if len(set(event_topks)) != 1:
            raise ProtocolError("top-k width changes within one batch")
        layers = sorted(event_layer_sets[0])
        top_k = event_topks[0]
        models = {row["model"] for row in window_routes}
        if len(models) != 1:
            raise ProtocolError("one decode window contains multiple models")
        model = next(iter(models))
        declared = metadata.get("num_experts")
        num_experts = int(
            declared.get(model) if isinstance(declared, dict) and model in declared
            else declared if isinstance(declared, int)
            else explicit_num_experts.get(model, 0)
        )
        if num_experts <= 1:
            raise ProtocolError(
                f"exact num_experts is required for zero-inclusive pressure/CV: {model}"
            )
        by_layer = {
            layer: [row for row in window_routes if int(row["layer_id"]) == layer]
            for layer in layers
        }
        layer_features = {
            layer: _layer_features(layer_rows, num_experts)
            for layer, layer_rows in by_layer.items()
        }
        pressures = [value["max_over_mean"] for value in layer_features.values()]
        cvs = [value["cv"] for value in layer_features.values()]
        hhis = [value["hhi"] for value in layer_features.values()]
        active_experts = [value["active_experts"] for value in layer_features.values()]
        top1 = [value["top1_expert_share"] for value in layer_features.values()]
        hotspots = {
            layer: int(value["hotspot_expert"])
            for layer, value in layer_features.items()
        }
        persistence = (
            fmean(float(prior_hotspots.get(layer) == expert) for layer, expert in hotspots.items())
            if prior_hotspots
            else 0.0
        )
        mean_pressure = fmean(pressures)
        delta = 0.0 if prior_pressure is None else mean_pressure - prior_pressure
        ewma = mean_pressure if ewma is None else alpha * mean_pressure + (1 - alpha) * ewma
        elapsed_ms = (end_us - start_us) / 1000.0
        tokens_per_second = batch_size / (elapsed_ms / 1000.0)
        row = {
            "model": model,
            "model_revision": str(metadata.get("model_revision", "unknown")),
            "arrival_episode_id": str(metadata["arrival_episode_id"]),
            "arrival_episode_independent": str(
                _bool(metadata["arrival_episode_independent"])
            ).lower(),
            "episode_id": str(metadata["episode_id"]),
            "split": str(metadata.get("split", "unassigned")),
            "window_id": f"{metadata['episode_id']}:{batch_index:08d}",
            "batch_index": batch_index,
            "window_start_us": start_us,
            "window_end_us": end_us,
            "feature_available_at_us": end_us,
            "decode_stage": fmean(decode_steps),
            "arrival_regime": str(metadata.get("arrival_regime", "unknown")),
            "active_tokens": batch_size,
            "running_sequences": len(active_ids),
            "queue_depth": len(active_ids) - batch_size,
            "mean_kv_length": fmean(kv_lengths),
            "max_kv_length": max(kv_lengths),
            "prompt_tokens": sum(prompts),
            "decode_tokens": batch_size,
            "batch_size": batch_size,
            "batch_tokens": batch_size,
            "step_service_ms": elapsed_ms,
            "tokens_completed": batch_size,
            "recent_step_ms": elapsed_ms,
            "recent_tokens_per_second": tokens_per_second,
            "route_max_mean": max(pressures),
            "route_cv": fmean(cvs),
            "route_hhi": fmean(hhis),
            "active_experts": fmean(active_experts),
            "top1_expert_share": fmean(top1),
            "cross_layer_max_pressure": max(pressures),
            "cross_layer_mean_pressure": mean_pressure,
            "hotspot_persistence": persistence,
            "route_shape_ewma": ewma,
            "route_shape_delta": delta,
            "max_expert_tokens": max(
                value["max_expert_tokens"] for value in layer_features.values()
            ),
            "route_layer_count": len(layers),
            "top_k": top_k,
            "per_layer_features_json": json.dumps(
                {str(key): value for key, value in layer_features.items()},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "request_ids_json": json.dumps(request_ids, separators=(",", ":")),
            "document_ids_json": json.dumps(documents, separators=(",", ":")),
            "evidence_type": str(metadata["evidence_type"]),
            "runtime_representative": str(_bool(metadata["runtime_representative"])).lower(),
            "instrumentation_overhead_measured": str(
                _bool(metadata["instrumentation_overhead_measured"])
            ).lower(),
            "fresh_holdout_sealed": str(_bool(metadata["fresh_holdout_sealed"])).lower(),
            "gate_weight_available": "true",
            "timing_boundary": str(metadata.get("timing_boundary", "unknown")),
            "queue_depth_semantics": "active_decode_sequences_not_selected_in_window",
            "source_capture": str(capture_dir),
        }
        output.append(row)
        prior_hotspots = hotspots
        prior_pressure = mean_pressure
        prior_window_end_us = end_us

    unused = set(event_rows) - used_events
    if unused:
        raise ProtocolError(
            f"{len(unused)} route events are not owned by any decode window"
        )
    diagnostic = {
        "capture_dir": str(capture_dir),
        "episode_id": str(metadata["episode_id"]),
        "windows": len(output),
        "route_events": len(used_events),
        "route_rows": len(routes),
        "models": sorted({row["model"] for row in output}),
        "evidence_type": str(metadata["evidence_type"]),
        "runtime_representative": _bool(metadata["runtime_representative"]),
        "instrumentation_overhead_measured": _bool(
            metadata["instrumentation_overhead_measured"]
        ),
        "qualification_boundary": (
            "P0 identity/timing alignment only; representative runtime, overhead, "
            "and fresh-holdout qualification are forced false"
        ),
        "future_fields_consumed": [],
    }
    return output, diagnostic


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Stream a large JSONL ledger without duplicating its raw payload."""

    if not path.is_file():
        raise ProtocolError(f"missing required ledger: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolError(f"{path}:{line_number} must be a JSON object")
            yield value


def _stablebatch_episode(request_ids: Sequence[str]) -> str:
    import hashlib

    payload = json.dumps(sorted(request_ids), separators=(",", ":")).encode("utf-8")
    return f"stablebatch-request-set-{hashlib.sha256(payload).hexdigest()[:16]}"


def build_stablebatch_capture(
    capture_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt one frozen StableBatch RTX 5090 trace for a smoke-only P1 run.

    The adapter intentionally consumes only ``native_variable_m``,
    ``phase=measured``, ``repeat=0``.  Repeat 1 is the same frozen request/route
    replay and must not masquerade as a fresh holdout.
    """

    required = [
        "RUN_STATUS.json",
        "COMPLETE.json",
        "config_snapshot.json",
        "workload_snapshot.json",
        "native_roster.jsonl",
        "decode_step_ledger.jsonl",
        "expert_call_ledger.jsonl",
    ]
    missing = [name for name in required if not (capture_dir / name).is_file()]
    if missing:
        raise ProtocolError(f"StableBatch capture lacks files: {missing}")
    status = read_json(capture_dir / "RUN_STATUS.json")
    complete = read_json(capture_dir / "COMPLETE.json")
    config = read_json(capture_dir / "config_snapshot.json")
    workload = read_json(capture_dir / "workload_snapshot.json")
    if (
        status.get("schema_version") != STABLEBATCH_STATUS_SCHEMA
        or complete.get("schema_version") != STABLEBATCH_COMPLETE_SCHEMA
        or config.get("schema_version") != STABLEBATCH_CONFIG_SCHEMA
        or workload.get("schema") != STABLEBATCH_WORKLOAD_SCHEMA
    ):
        raise ProtocolError("StableBatch sentinel/config/workload schema drifted")
    if status.get("status") != "COMPLETE" or complete.get("status") != "COMPLETE":
        raise ProtocolError("StableBatch capture is incomplete")
    if status.get("serving_result") is not False:
        raise ProtocolError("StableBatch serving-result boundary is not explicit")
    execution_config = config.get("execution")
    measured_orders = (
        execution_config.get("measured_arm_orders")
        if isinstance(execution_config, dict)
        else None
    )
    if (
        not isinstance(measured_orders, list)
        or not measured_orders
        or not isinstance(measured_orders[0], list)
        or not measured_orders[0]
        or measured_orders[0][0] != STABLEBATCH_ARM
    ):
        raise ProtocolError(
            "StableBatch config does not place the selected repeat-0 arm first"
        )
    model_config = config.get("model")
    workload_model = workload.get("model")
    if not isinstance(model_config, dict) or not isinstance(workload_model, dict):
        raise ProtocolError("StableBatch model identity is missing")
    model = str(model_config.get("repo_id") or "")
    revision = str(model_config.get("revision") or "")
    num_experts = int(model_config.get("num_experts", 0))
    num_layers = int(model_config.get("num_hidden_layers", 0))
    configured_top_k = int(model_config.get("num_experts_per_tok", 0))
    workload_config = config.get("workload")
    if not isinstance(workload_config, dict):
        raise ProtocolError("StableBatch frozen workload configuration is missing")
    expected_requests = int(workload_config.get("expected_requests", 0))
    expected_steps = int(
        workload_config.get("expected_decode_steps_per_request", 0)
    )
    max_batch_size = int(workload_config.get("max_batch_size", 0))
    expected_request_steps = int(
        workload_config.get("expected_request_steps", 0)
    )
    if (
        not model
        or not revision
        or num_experts <= 1
        or num_layers <= 0
        or not 0 < configured_top_k <= num_experts
        or model != str(workload_model.get("id") or "")
        or revision != str(workload_model.get("revision") or "")
    ):
        raise ProtocolError("StableBatch frozen model identity does not close")
    if (
        expected_requests <= 0
        or expected_steps <= 1
        or max_batch_size <= 0
        or expected_request_steps != expected_requests * expected_steps
        or expected_requests % max_batch_size
    ):
        raise ProtocolError("StableBatch frozen workload dimensions do not close")
    expected_cohorts = expected_requests // max_batch_size
    expected_batches = expected_cohorts * expected_steps

    request_specs: dict[str, Mapping[str, Any]] = {}
    raw_requests = workload.get("requests")
    if not isinstance(raw_requests, list):
        raise ProtocolError("StableBatch workload requests are missing")
    for request in raw_requests:
        if not isinstance(request, dict):
            raise ProtocolError("StableBatch request row must be an object")
        request_id = str(request.get("request_id") or "")
        if not request_id or request_id in request_specs:
            raise ProtocolError("StableBatch request identity is empty or duplicated")
        request_specs[request_id] = request

    roster: dict[int, Mapping[str, Any]] = {}
    for row in _iter_jsonl(capture_dir / "native_roster.jsonl"):
        if row.get("schema_version") != STABLEBATCH_ROSTER_SCHEMA:
            raise ProtocolError("StableBatch native roster schema drifted")
        batch_index = int(row.get("batch_index", -1))
        if batch_index < 0 or batch_index in roster:
            raise ProtocolError("StableBatch native roster batch is invalid/duplicated")
        roster[batch_index] = row

    decode: dict[int, Mapping[str, Any]] = {}
    for row in _iter_jsonl(capture_dir / "decode_step_ledger.jsonl"):
        if (
            row.get("arm") != STABLEBATCH_ARM
            or row.get("phase") != STABLEBATCH_PHASE
            or int(row.get("repeat", -1)) != STABLEBATCH_REPEAT
        ):
            continue
        if row.get("schema_version") != STABLEBATCH_STEP_SCHEMA:
            raise ProtocolError("StableBatch measured decode schema drifted")
        batch_index = int(row.get("batch_index", -1))
        if batch_index < 0 or batch_index in decode:
            raise ProtocolError("StableBatch measured decode batch is invalid/duplicated")
        decode[batch_index] = row
    if not decode:
        raise ProtocolError("StableBatch has no measured native repeat-0 windows")

    # key -> layer -> expert -> logical routed rows
    route_counts: dict[int, dict[int, dict[int, int]]] = {}
    seen_calls: set[tuple[int, int, int]] = set()
    seen_slots: dict[tuple[int, int], set[str]] = {}
    seen_rows: dict[tuple[int, int], set[str]] = {}
    target_started = False
    target_closed = False
    for row in _iter_jsonl(capture_dir / "expert_call_ledger.jsonl"):
        is_target = (
            row.get("arm") == STABLEBATCH_ARM
            and row.get("phase") == STABLEBATCH_PHASE
            and int(row.get("repeat", -1)) == STABLEBATCH_REPEAT
        )
        # The frozen StableBatch config places this arm first in measured
        # repeat 0, and the ledger is emitted as contiguous arm blocks.  Once
        # the selected block closes, stop before parsing the remaining ~1 GiB;
        # exact batch/layer coverage below still fails closed on truncation.
        if target_started and not is_target:
            target_closed = True
            break
        if not is_target:
            continue
        target_started = True
        if row.get("schema_version") != STABLEBATCH_CALL_SCHEMA:
            raise ProtocolError("StableBatch measured expert-call schema drifted")
        batch_index = int(row.get("batch_index", -1))
        layer = int(row.get("layer", -1))
        expert = int(row.get("expert_id", -1))
        logical_m = int(row.get("logical_m", 0))
        row_ids = row.get("row_ids")
        slot_ids = row.get("slot_ids")
        identity = (batch_index, layer, expert)
        normalized_row_ids = [str(value) for value in row_ids] if isinstance(row_ids, list) else []
        normalized_slot_ids = [str(value) for value in slot_ids] if isinstance(slot_ids, list) else []
        if (
            batch_index < 0
            or layer < 0
            or not 0 <= expert < num_experts
            or logical_m <= 0
            or not isinstance(row_ids, list)
            or not isinstance(slot_ids, list)
            or len(row_ids) != logical_m
            or len(slot_ids) != logical_m
            or len(set(normalized_row_ids)) != logical_m
            or len(set(normalized_slot_ids)) != logical_m
            or identity in seen_calls
        ):
            raise ProtocolError(f"invalid/duplicate StableBatch expert call {identity}")
        layer_identity = (batch_index, layer)
        if (
            seen_rows.setdefault(layer_identity, set()) & set(normalized_row_ids)
            or seen_slots.setdefault(layer_identity, set()) & set(normalized_slot_ids)
        ):
            raise ProtocolError(
                f"StableBatch route row/slot is duplicated across experts at {layer_identity}"
            )
        seen_rows[layer_identity].update(normalized_row_ids)
        seen_slots[layer_identity].update(normalized_slot_ids)
        seen_calls.add(identity)
        route_counts.setdefault(batch_index, {}).setdefault(layer, {})[
            expert
        ] = logical_m
    if not target_started or not target_closed:
        raise ProtocolError(
            "StableBatch selected expert-call block was not found and closed"
        )
    if set(decode) != set(route_counts):
        raise ProtocolError("StableBatch decode/route window coverage is not exact")
    expected_indices = set(range(expected_batches))
    if set(roster) != expected_indices or set(decode) != expected_indices:
        raise ProtocolError("StableBatch roster/decode batch denominator is incomplete")
    for batch_index in expected_indices:
        for layer in range(num_layers):
            observed_slots = seen_slots.get((batch_index, layer), set())
            expected_slots = max_batch_size * configured_top_k
            if len(observed_slots) != expected_slots:
                raise ProtocolError(
                    f"StableBatch route-slot denominator fails at {(batch_index, layer)}"
                )

    output: list[dict[str, Any]] = []
    episode_state: dict[str, dict[str, Any]] = {}
    episode_clock: dict[str, float] = {}
    cohort_requests: dict[int, tuple[str, ...]] = {}
    request_owner: dict[str, int] = {}
    for batch_index in sorted(decode):
        step = decode[batch_index]
        native = roster.get(batch_index)
        if native is None:
            raise ProtocolError(f"StableBatch roster misses batch {batch_index}")
        request_ids = [str(value) for value in step.get("request_ids", [])]
        roster_request_ids = [str(value) for value in native.get("request_ids", [])]
        decode_steps = [int(value) for value in step.get("decode_steps", [])]
        roster_steps = [int(value) for value in native.get("decode_steps", [])]
        kv_lengths = [int(value) for value in native.get("prior_cache_lengths", [])]
        active_ids = [str(value) for value in native.get("active_request_ids", [])]
        batch_size = len(request_ids)
        cohort_index, cohort_step = divmod(batch_index, expected_steps)
        if (
            request_ids != roster_request_ids
            or decode_steps != roster_steps
            or batch_size <= 0
            or batch_size != max_batch_size
            or len(kv_lengths) != batch_size
            or len(set(request_ids)) != batch_size
            or not set(request_ids) <= set(active_ids)
        ):
            raise ProtocolError(f"StableBatch window identity fails at {batch_index}")
        if set(decode_steps) != {cohort_step}:
            raise ProtocolError(
                f"StableBatch cohort decode-step closure fails at {batch_index}"
            )
        request_tuple = tuple(request_ids)
        previous_requests = cohort_requests.setdefault(cohort_index, request_tuple)
        if previous_requests != request_tuple:
            raise ProtocolError(
                f"StableBatch cohort request set changes at batch {batch_index}"
            )
        for request_id in request_ids:
            previous_owner = request_owner.setdefault(request_id, cohort_index)
            if previous_owner != cohort_index:
                raise ProtocolError(
                    f"StableBatch request {request_id} crosses proxy split units"
                )
        if step.get("route_membership_sha256") != native.get(
            "native_route_membership_sha256"
        ):
            raise ProtocolError(
                f"StableBatch measured/native route identity drifts at {batch_index}"
            )
        documents: list[str] = []
        prompts: list[int] = []
        for request_id, decode_step, kv_length in zip(
            request_ids, decode_steps, kv_lengths
        ):
            spec = request_specs.get(request_id)
            if spec is None:
                raise ProtocolError(f"StableBatch workload misses {request_id}")
            prompt_count = int(spec.get("prompt_token_count", 0))
            if prompt_count <= 0 or kv_length - decode_step != prompt_count:
                raise ProtocolError(f"StableBatch prompt/KV identity fails for {request_id}")
            prompts.append(prompt_count)
            documents.append(str(spec.get("document_id") or request_id))

        if set(route_counts[batch_index]) != set(range(num_layers)):
            raise ProtocolError(
                f"StableBatch route layer denominator fails at {batch_index}"
            )
        summary = _route_summary(
            route_counts[batch_index],
            num_experts=num_experts,
            batch_size=batch_size,
        )
        if summary.top_k != configured_top_k:
            raise ProtocolError("StableBatch route top-k differs from frozen model")
        layer_features = summary.per_layer
        pressures = [value["max_over_mean"] for value in layer_features.values()]
        cvs = [value["cv"] for value in layer_features.values()]
        hhis = [value["hhi"] for value in layer_features.values()]
        active_experts = [value["active_experts"] for value in layer_features.values()]
        top1 = [value["top1_expert_share"] for value in layer_features.values()]
        hotspots = {
            layer: int(value["hotspot_expert"])
            for layer, value in layer_features.items()
        }
        episode_id = _stablebatch_episode(request_ids)
        prior = episode_state.get(episode_id)
        mean_pressure = fmean(pressures)
        if prior is None:
            persistence = 0.0
            delta = 0.0
            ewma = mean_pressure
        else:
            persistence = fmean(
                float(prior["hotspots"].get(layer) == expert)
                for layer, expert in hotspots.items()
            )
            delta = mean_pressure - float(prior["pressure"])
            ewma = 0.3 * mean_pressure + 0.7 * float(prior["ewma"])
        elapsed_ms = float(step.get("whole_step_wall_ms", 0.0))
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise ProtocolError("StableBatch whole-step duration is invalid")
        start_us = episode_clock.get(episode_id, 0.0)
        end_us = start_us + elapsed_ms * 1000.0
        tokens_per_second = batch_size / (elapsed_ms / 1000.0)
        output.append(
            {
                "model": model,
                "model_revision": revision,
                "arrival_episode_id": "stablebatch-single-capture-arrival-episode",
                "arrival_episode_independent": "false",
                "episode_id": episode_id,
                "split": "unassigned",
                "window_id": f"{episode_id}:{decode_steps[0]:08d}",
                "batch_index": batch_index,
                "window_start_us": start_us,
                "window_end_us": end_us,
                "feature_available_at_us": end_us,
                "decode_stage": fmean(decode_steps),
                "arrival_regime": "frozen_burstgpt_single_episode",
                # Only scheduled/executed width is available. Native roster
                # active/pending counts are not serving-engine counters.
                "active_tokens": batch_size,
                "running_sequences": batch_size,
                "queue_depth": 0,
                "mean_kv_length": fmean(kv_lengths),
                "max_kv_length": max(kv_lengths),
                "prompt_tokens": sum(prompts),
                "decode_tokens": batch_size,
                "batch_size": batch_size,
                "batch_tokens": batch_size,
                "step_service_ms": elapsed_ms,
                "tokens_completed": batch_size,
                "recent_step_ms": elapsed_ms,
                "recent_tokens_per_second": tokens_per_second,
                "route_max_mean": summary.route_max_mean,
                "route_cv": summary.route_cv,
                "route_hhi": summary.route_hhi,
                "active_experts": summary.active_experts,
                "top1_expert_share": summary.top1_expert_share,
                "cross_layer_max_pressure": summary.cross_layer_max_pressure,
                "cross_layer_mean_pressure": summary.cross_layer_mean_pressure,
                "hotspot_persistence": persistence,
                "route_shape_ewma": ewma,
                "route_shape_delta": delta,
                "max_expert_tokens": summary.max_expert_tokens,
                "route_layer_count": summary.layer_count,
                "top_k": summary.top_k,
                "per_layer_features_json": json.dumps(
                    {str(key): value for key, value in layer_features.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "request_ids_json": json.dumps(request_ids, separators=(",", ":")),
                "document_ids_json": json.dumps(documents, separators=(",", ":")),
                "evidence_type": "[Observed isolated GPU primitive]",
                "runtime_representative": "false",
                "instrumentation_overhead_measured": "false",
                "fresh_holdout_sealed": "false",
                "gate_weight_available": "false",
                "timing_boundary": (
                    "CUDA-synchronized whole model step in teacher-forced "
                    "StableBatch replay; excludes serving queue"
                ),
                "queue_depth_semantics": (
                    "NOT_OBSERVED_SENTINEL_ZERO_EXCLUDED_FROM_CLAIMS"
                ),
                "source_capture": str(capture_dir),
            }
        )
        episode_clock[episode_id] = end_us
        episode_state[episode_id] = {
            "hotspots": hotspots,
            "pressure": mean_pressure,
            "ewma": ewma,
        }
    if len(output) != expected_batches:
        raise ProtocolError(
            "StableBatch frozen smoke expects "
            f"{expected_batches} windows, observed {len(output)}"
        )
    episodes = {str(row["episode_id"]) for row in output}
    if len(episodes) != expected_cohorts:
        raise ProtocolError(
            "StableBatch frozen smoke expects "
            f"{expected_cohorts} request sets, observed {len(episodes)}"
        )
    return output, {
        "capture_dir": str(capture_dir),
        "episode_id": (
            f"{expected_cohorts} request-disjoint proxy split units from one replay"
        ),
        "windows": len(output),
        "route_events": expected_request_steps,
        "route_rows": sum(
            logical_m
            for layers in route_counts.values()
            for experts in layers.values()
            for logical_m in experts.values()
        ),
        "models": [model],
        "evidence_type": "[Observed isolated GPU primitive]",
        "runtime_representative": False,
        "instrumentation_overhead_measured": False,
        "future_fields_consumed": [],
        "adapter_boundary": (
            "DEVELOPMENT_REUSE_NOT_SERVING / NOT_P1_ELIGIBLE; repeat 1 excluded; "
            "request-disjoint cohorts are split units, not independent runtime episodes"
        ),
        "split_limit": (
            "ONE_ARRIVAL_EPISODE; the request-set split is development-only and "
            "cannot qualify an arrival-episode holdout"
        ),
        "field_surrogates": {
            "active_tokens": "executed decode tokens in fixed batch",
            "running_sequences": (
                "scheduled batch width, not engine running-sequence telemetry"
            ),
            "queue_depth": (
                "NOT_OBSERVED sentinel zero; pending_request_count is deliberately "
                "never consumed"
            ),
            "tokens_completed": "executed decode rows, not request completions",
        },
        "serving_queue_observed": False,
        "slo_observed": False,
        "gate_weight_observed": False,
        "source_replay_filter": {
            "arm": STABLEBATCH_ARM,
            "phase": STABLEBATCH_PHASE,
            "repeat": STABLEBATCH_REPEAT,
        },
    }


def build_all(
    capture_dirs: Sequence[Path],
    explicit_num_experts: Mapping[str, int],
    stablebatch_dirs: Sequence[Path] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    for capture_dir in capture_dirs:
        built, diagnostic = build_capture(capture_dir.resolve(), explicit_num_experts)
        episode = str(diagnostic["episode_id"])
        if episode in episode_ids:
            raise ProtocolError(f"duplicate episode_id across captures: {episode}")
        episode_ids.add(episode)
        rows.extend(built)
        captures.append(diagnostic)
    for capture_dir in stablebatch_dirs:
        built, diagnostic = build_stablebatch_capture(capture_dir.resolve())
        built_episodes = {str(row["episode_id"]) for row in built}
        overlap = built_episodes & episode_ids
        if overlap:
            raise ProtocolError(f"duplicate episode_id across captures: {sorted(overlap)}")
        episode_ids.update(built_episodes)
        rows.extend(built)
        captures.append(diagnostic)
    if not rows:
        raise ProtocolError("no windows were built")
    representative = all(item["runtime_representative"] for item in captures)
    overhead = all(item["instrumentation_overhead_measured"] for item in captures)
    evidence = {item["evidence_type"] for item in captures}
    status = (
        "READY_FOR_SIGNAL_TEST"
        if representative and overhead and evidence == {"[Observed real runtime]"}
        else "BLOCKED_RUNTIME_NOT_REPRESENTATIVE"
    )
    return rows, {
        "schema": "route-shape-window-features-v1",
        "status": status,
        "windows": len(rows),
        "episodes": len(episode_ids),
        "models": sorted({str(row["model"]) for row in rows}),
        "captures": captures,
        "information_boundary": "all features use workload/route/latency state at or before window t end",
        "target_alignment": "not performed by builder; analyzer shifts within episode",
        "queue_depth_semantics": (
            "capture-specific; StableBatch proxy uses NOT_OBSERVED sentinel zero; "
            "inspect captures[].field_surrogates"
            if stablebatch_dirs
            else "active decode sequences not selected in the current model call"
        ),
        "active_token_semantics": "decode tokens executed in the current model call",
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", action="append", default=[])
    parser.add_argument("--stablebatch-dir", action="append", default=[])
    parser.add_argument("--num-experts", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.capture_dir and not args.stablebatch_dir:
        raise SystemExit("at least one --capture-dir or --stablebatch-dir is required")
    output = Path(args.output)
    rows, metadata = build_all(
        [Path(value) for value in args.capture_dir],
        parse_num_experts(args.num_experts),
        [Path(value) for value in args.stablebatch_dir],
    )
    write_csv(output, rows)
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output.with_suffix(output.suffix + ".meta.json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
