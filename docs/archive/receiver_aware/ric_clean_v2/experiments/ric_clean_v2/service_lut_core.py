#!/usr/bin/env python3
"""Pure validators and accounting for the RIC clean-v2 service LUT."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

try:
    from . import native_route_core as native
    from . import prepare_clean_v2_data as data
except ImportError:  # pragma: no cover
    import native_route_core as native  # type: ignore
    import prepare_clean_v2_data as data  # type: ignore


SERVICE_ADDENDUM_SHA256 = "66830ee80c34ad386b35cd03600b6527df4c15d61a781d5f5bb5980b539732dd"
N1_ADDENDUM_SHA256 = "360f7e9d1476acc39e19618bd65a95658d0f6b0fe3e3c6aa6b72a066b6a25c64"
WARMUPS = 10
TRIALS = 30
COMBINE_JOINS = 32
PRIMARY_COMPONENTS = (
    "expert_execution_route_specific_row1",
    "sender_pack_route_specific_row1",
    "receiver_unpack_route_specific_row1",
)
ALL_COMPONENTS = PRIMARY_COMPONENTS + ("canonical_combine_once_per_join",)
ROUTE_IDENTITY_FIELDS = (
    "request_id", "forward_id", "batch_id", "phase", "decode_step", "layer_id",
    "token_id", "token_block_id", "topk_slot", "expert_id", "sender_rank",
    "receiver_rank", "epoch",
)
FULL_JOIN_FIELDS = (
    "model_key", "model_revision", "data_manifest_sha256", "placement_manifest_sha256",
    "request_id", "forward_id", "batch_id", "phase", "decode_step", "layer_id",
    "token_id", "token_block_id", "receiver_rank", "epoch",
)
RAW_COLUMNS = (
    "model_key", "model_revision", "layer_id", "expert_id", "component",
    "route_identity_sha256", "join_identity_sha256", "route_tuple_sha256",
    "input_tensor_sha256", "input_shape", "input_dtype",
    "output_tensor_descriptor_sha256", "tensor_numel", "tensor_element_size_bytes",
    "payload_bytes", "descriptor_bytes", "alignment_boundary_bytes",
    "alignment_padding_bytes", "transport_bytes", "rows", "phase", "trial_index",
    "execution_ordinal", "cuda_event_us", "wall_sync_us", "energy_status", "source",
    "evidence_boundary", "gpu_uuid", "stream_id", "producer_source_sha256",
)
SUMMARY_COLUMNS = (
    "model_key", "model_revision", "layer_id", "expert_id", "component",
    "route_identity_sha256", "join_identity_sha256", "rows", "measured_count",
    "median_cuda_event_us", "p95_cuda_event_us", "max_cuda_event_us",
    "median_wall_sync_us", "stability_ratio", "payload_bytes", "descriptor_bytes",
    "alignment_boundary_bytes", "alignment_padding_bytes", "transport_bytes",
    "output_tensor_descriptor_sha256", "source", "evidence_boundary",
    "producer_source_sha256",
)
RAW_INT_FIELDS = {
    "layer_id", "expert_id", "tensor_numel", "tensor_element_size_bytes", "payload_bytes",
    "descriptor_bytes", "alignment_boundary_bytes", "alignment_padding_bytes",
    "transport_bytes", "rows", "trial_index", "execution_ordinal", "stream_id",
}
RAW_FLOAT_FIELDS = {"cuda_event_us", "wall_sync_us"}
SUMMARY_INT_FIELDS = {
    "layer_id", "expert_id", "rows", "measured_count", "payload_bytes",
    "descriptor_bytes", "alignment_boundary_bytes", "alignment_padding_bytes",
    "transport_bytes",
}
SUMMARY_FLOAT_FIELDS = {
    "median_cuda_event_us", "p95_cuda_event_us", "max_cuda_event_us",
    "median_wall_sync_us", "stability_ratio",
}


class ServiceLutError(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_identity(row: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    if any(field not in row for field in fields):
        raise ServiceLutError("identity field is missing")
    value = {field: row[field] for field in fields}
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ServiceLutError("identity is not strict canonical JSON") from exc
    return value


def route_identity_sha256(row: Mapping[str, Any]) -> str:
    return object_sha256(strict_identity(row, ROUTE_IDENTITY_FIELDS))


def full_join_identity_sha256(row: Mapping[str, Any]) -> str:
    return object_sha256(strict_identity(row, FULL_JOIN_FIELDS))


def tensor_descriptor(tensor: Any) -> dict[str, int]:
    numel = int(tensor.numel())
    element = int(tensor.element_size())
    payload = numel * element
    descriptor = 16
    boundary = 16
    padding = (boundary - ((payload + descriptor) % boundary)) % boundary
    return {
        "tensor_numel": numel,
        "tensor_element_size_bytes": element,
        "payload_bytes": payload,
        "descriptor_bytes": descriptor,
        "alignment_boundary_bytes": boundary,
        "alignment_padding_bytes": padding,
        "transport_bytes": payload + descriptor + padding,
    }


def descriptor_sha256(descriptor: Mapping[str, Any]) -> str:
    fields = (
        "tensor_numel", "tensor_element_size_bytes", "payload_bytes", "descriptor_bytes",
        "alignment_boundary_bytes", "alignment_padding_bytes", "transport_bytes",
    )
    value = {field: descriptor.get(field) for field in fields}
    if any(type(value[field]) is not int for field in fields):
        raise ServiceLutError("descriptor fields must be integers")
    if (
        value["descriptor_bytes"] != 16
        or value["alignment_boundary_bytes"] != 16
        or value["payload_bytes"] != value["tensor_numel"] * value["tensor_element_size_bytes"]
        or value["alignment_padding_bytes"]
        != (16 - ((value["payload_bytes"] + 16) % 16)) % 16
        or value["transport_bytes"]
        != value["payload_bytes"] + 16 + value["alignment_padding_bytes"]
    ):
        raise ServiceLutError("transport descriptor accounting mismatch")
    return object_sha256(value)


def _expected_token_identity(request_id: str, token_position: int) -> str:
    return f"{request_id}:token:{token_position:03d}"


def validate_route_rows(
    rows: Sequence[Mapping[str, Any]], *, model_key: str, model_revision: str,
    requests: Sequence[Mapping[str, Any]], all_layers: Sequence[int], selected_layers: Sequence[int],
    num_experts: int, top_k: int, ep_size: int, placement: Mapping[str, Any],
    data_manifest_sha256: str,
) -> tuple[dict[tuple[int, int], Mapping[str, Any]], dict[int, list[list[Mapping[str, Any]]]]]:
    request_ids = {str(row["request_id"]) for row in requests}
    if len(request_ids) != len(requests) or len(set(selected_layers)) != 4:
        raise ServiceLutError("request/selected layer census mismatch")
    receivers = placement.get("request_to_receiver")
    experts_to_sender = placement.get("expert_to_sender")
    if not isinstance(receivers, Mapping) or not isinstance(experts_to_sender, Mapping):
        raise ServiceLutError("placement mappings are missing")
    expected_count = len(request_ids) * len(all_layers) * 128 * top_k
    if len(rows) != expected_count:
        raise ServiceLutError("route row count mismatch")
    seen: set[tuple[Any, ...]] = set()
    selected: list[Mapping[str, Any]] = []
    joins: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    tuple_by_request_layer: dict[tuple[str, int], str] = {}
    for row in rows:
        identity = strict_identity(row, ROUTE_IDENTITY_FIELDS)
        request_id = identity["request_id"]
        layer = identity["layer_id"]
        position = row.get("token_position")
        slot = identity["topk_slot"]
        expert = identity["expert_id"]
        if (
            row.get("schema_version") != "ric-clean-v2-route-row-v1"
            or row.get("model_key") != model_key or row.get("model_revision") != model_revision
            or row.get("data_manifest_sha256") != data_manifest_sha256
            or request_id not in request_ids or layer not in set(all_layers)
            or type(position) is not int or not 0 <= position < 128
            or type(slot) is not int or not 0 <= slot < top_k
            or type(expert) is not int or not 0 <= expert < num_experts
            or identity["phase"] != "prefill" or identity["decode_step"] != 0
            or identity["forward_id"] != f"{request_id}:prefill:0"
            or identity["batch_id"] != f"batch:{request_id}:prefill:0"
            or identity["token_id"] != _expected_token_identity(request_id, position)
            or identity["token_block_id"] != identity["token_id"]
            or identity["epoch"] != 1 or row.get("valid") is not True
            or identity["sender_rank"] != native.expert_sender(expert, num_experts, ep_size)
            or experts_to_sender.get(str(expert)) != identity["sender_rank"]
            or receivers.get(request_id) != identity["receiver_rank"]
            or row.get("placement_manifest_sha256") != placement.get("manifest_sha256")
            or row.get("route_source") != "native_aten_topk_plus_raw_logit_and_output_parity"
        ):
            raise ServiceLutError("consumer-side route identity/placement mismatch")
        key = tuple(identity[field] for field in ROUTE_IDENTITY_FIELDS)
        if key in seen:
            raise ServiceLutError("duplicate route identity")
        seen.add(key)
        route_tuple = row.get("native_route_tuple_sha256")
        if not isinstance(route_tuple, str) or len(route_tuple) != 64:
            raise ServiceLutError("route tuple hash is missing")
        tuple_key = (request_id, layer)
        prior = tuple_by_request_layer.setdefault(tuple_key, route_tuple)
        if prior != route_tuple:
            raise ServiceLutError("route tuple hash drift within request/layer")
        if layer in set(selected_layers):
            selected.append(row)
            joins.setdefault((layer, full_join_identity_sha256(row)), []).append(row)
    expected_keys = {
        (request_id, layer, position, slot)
        for request_id in request_ids for layer in all_layers for position in range(128)
        for slot in range(top_k)
    }
    observed_keys = {(r["request_id"], r["layer_id"], r["token_position"], r["topk_slot"]) for r in rows}
    if observed_keys != expected_keys:
        raise ServiceLutError("route Cartesian identity mismatch")
    by_expert: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in selected:
        by_expert.setdefault((int(row["layer_id"]), int(row["expert_id"])), []).append(row)
    expected_cells = {(layer, expert) for layer in selected_layers for expert in range(num_experts)}
    if set(by_expert) != expected_cells:
        raise ServiceLutError("BLOCKED_ROUTE_SPECIFIC_SERVICE_COVERAGE")
    chosen = {key: min(values, key=route_identity_sha256) for key, values in by_expert.items()}
    selected_joins: dict[int, list[list[Mapping[str, Any]]]] = {}
    for layer in selected_layers:
        candidates = []
        for (join_layer, join_hash), siblings in joins.items():
            if join_layer != layer:
                continue
            ordered = sorted(siblings, key=lambda row: int(row["topk_slot"]))
            if (
                len(ordered) != top_k
                or [row["topk_slot"] for row in ordered] != list(range(top_k))
                or len({row["expert_id"] for row in ordered}) != top_k
                or len({row["receiver_rank"] for row in ordered}) != 1
            ):
                raise ServiceLutError("full join sibling census mismatch")
            candidates.append((join_hash, ordered))
        candidates.sort(key=lambda item: item[0])
        if len(candidates) < COMBINE_JOINS:
            raise ServiceLutError("fewer than 32 complete joins")
        selected_joins[layer] = [siblings for _, siblings in candidates[:COMBINE_JOINS]]
    return chosen, selected_joins


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def p95(values: Sequence[float]) -> float:
    if not values:
        raise ServiceLutError("empty statistic")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        raise ServiceLutError("boolean CSV cell is forbidden")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ServiceLutError("non-finite CSV cell")
        return format(value, ".17g")
    if isinstance(value, (list, dict, tuple)):
        return canonical_json_bytes(value).decode("utf-8")
    return str(value)


def csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, dialect="excel", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        if set(row) != set(columns):
            raise ServiceLutError("CSV row schema differs from frozen columns")
        writer.writerow([_format_cell(row[column]) for column in columns])
    return output.getvalue().encode("utf-8")


def parse_csv_bytes(raw: bytes, columns: Sequence[str], *, int_fields: set[str], float_fields: set[str]) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ServiceLutError("CSV is not UTF-8") from exc
    if "\r" in text:
        raise ServiceLutError("CSV is not LF-only")
    reader = csv.reader(io.StringIO(text, newline=""), dialect="excel")
    materialized = list(reader)
    if not materialized or tuple(materialized[0]) != tuple(columns):
        raise ServiceLutError("CSV header/order mismatch")
    result = []
    for cells in materialized[1:]:
        if len(cells) != len(columns):
            raise ServiceLutError("CSV row width mismatch")
        row: dict[str, Any] = {}
        for field, cell in zip(columns, cells):
            try:
                if field in int_fields:
                    row[field] = int(cell)
                    if str(row[field]) != cell:
                        raise ValueError
                elif field in float_fields:
                    row[field] = float(cell)
                    if not math.isfinite(row[field]) or format(row[field], ".17g") != cell:
                        raise ValueError
                elif field == "input_shape":
                    value = json.loads(cell)
                    if not isinstance(value, list) or any(type(item) is not int for item in value):
                        raise ValueError
                    row[field] = value
                else:
                    row[field] = cell
            except (ValueError, json.JSONDecodeError) as exc:
                raise ServiceLutError(f"non-canonical CSV cell: {field}") from exc
        result.append(row)
    if csv_bytes(result, columns) != raw:
        raise ServiceLutError("CSV does not round-trip canonically")
    return result


def summarize_raw(
    raw_rows: Sequence[Mapping[str, Any]], *, model_key: str, model_revision: str,
    selected_layers: Sequence[int], num_experts: int, producer_source_sha256: str,
) -> list[dict[str, Any]]:
    expected_total = 3 * len(selected_layers) * num_experts * 40 + len(selected_layers) * COMBINE_JOINS * 40
    if len(raw_rows) != expected_total:
        raise ServiceLutError("primary raw census mismatch")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    ordinals = []
    for row in raw_rows:
        if row.get("model_key") != model_key or row.get("model_revision") != model_revision:
            raise ServiceLutError("raw model identity mismatch")
        if row.get("component") not in ALL_COMPONENTS or row.get("source") != "measured_5090_cuda":
            raise ServiceLutError("raw component/source mismatch")
        if row.get("evidence_boundary") != "REAL_5090_CUDA_NOT_NETWORK":
            raise ServiceLutError("raw evidence boundary mismatch")
        if row.get("energy_status") != "AUXILIARY_SEPARATE":
            raise ServiceLutError("energy leaked into primary raw")
        for field in ("cuda_event_us", "wall_sync_us"):
            if not isinstance(row.get(field), (int, float)) or isinstance(row.get(field), bool) or not math.isfinite(float(row[field])) or float(row[field]) <= 0:
                raise ServiceLutError("non-positive/non-finite timing")
        if float(row["cuda_event_us"]) > float(row["wall_sync_us"]):
            raise ServiceLutError("CUDA event exceeds wall time")
        descriptor = {field: row[field] for field in (
            "tensor_numel", "tensor_element_size_bytes", "payload_bytes", "descriptor_bytes",
            "alignment_boundary_bytes", "alignment_padding_bytes", "transport_bytes",
        )}
        if descriptor_sha256(descriptor) != row.get("output_tensor_descriptor_sha256"):
            raise ServiceLutError("descriptor hash mismatch")
        component = row["component"]
        join_hash = row.get("join_identity_sha256")
        if component == "canonical_combine_once_per_join":
            if row.get("expert_id") != -1 or not isinstance(join_hash, str) or len(join_hash) != 64 or row.get("route_identity_sha256") != "":
                raise ServiceLutError("combine identity schema mismatch")
            group = (component, row["layer_id"], -1, join_hash)
        else:
            if type(row.get("expert_id")) is not int or row["expert_id"] < 0 or join_hash != "" or len(str(row.get("route_identity_sha256"))) != 64:
                raise ServiceLutError("expert point identity schema mismatch")
            group = (component, row["layer_id"], row["expert_id"], "")
        groups.setdefault(group, []).append(row)
        ordinals.append(row.get("execution_ordinal"))
    if ordinals != list(range(expected_total)):
        raise ServiceLutError("execution ordinal is not globally contiguous")
    expected_expert = {(component, layer, expert, "") for component in PRIMARY_COMPONENTS for layer in selected_layers for expert in range(num_experts)}
    actual_expert = {key for key in groups if key[0] in PRIMARY_COMPONENTS}
    if actual_expert != expected_expert:
        raise ServiceLutError("raw expert Cartesian surface mismatch")

    summary: list[dict[str, Any]] = []
    combine_descriptors: dict[int, set[tuple[Any, ...]]] = {}
    def summarize_group(key: tuple[Any, ...], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        warmups = [row for row in rows if row["phase"] == "warmup"]
        measured = [row for row in rows if row["phase"] == "measured"]
        if len(warmups) != WARMUPS or len(measured) != TRIALS or {row["trial_index"] for row in warmups} != set(range(WARMUPS)) or {row["trial_index"] for row in measured} != set(range(TRIALS)):
            raise ServiceLutError("10+30 point census mismatch")
        cuda = [float(row["cuda_event_us"]) for row in measured]
        wall = [float(row["wall_sync_us"]) for row in measured]
        ratio = p95(cuda) / median(cuda)
        if ratio > 2.0:
            raise ServiceLutError("BLOCKED_UNSTABLE_SERVICE_POINT")
        frozen = (
            "route_identity_sha256", "join_identity_sha256", "rows", "payload_bytes",
            "descriptor_bytes", "alignment_boundary_bytes", "alignment_padding_bytes",
            "transport_bytes", "output_tensor_descriptor_sha256", "source", "evidence_boundary",
            "producer_source_sha256",
        )
        values = {field: {row[field] for row in rows} for field in frozen}
        if any(len(items) != 1 for items in values.values()):
            raise ServiceLutError("summary group immutable field drift")
        return {
            "model_key": model_key, "model_revision": model_revision, "layer_id": key[1],
            "expert_id": key[2], "component": key[0],
            **{field: next(iter(values[field])) for field in frozen if field not in {"producer_source_sha256"}},
            "measured_count": TRIALS, "median_cuda_event_us": median(cuda),
            "p95_cuda_event_us": p95(cuda), "max_cuda_event_us": max(cuda),
            "median_wall_sync_us": median(wall), "stability_ratio": ratio,
            "producer_source_sha256": producer_source_sha256,
        }
    for key in sorted(groups):
        row = summarize_group(key, groups[key])
        summary.append(row)
        if key[0] == "canonical_combine_once_per_join":
            combine_descriptors.setdefault(int(key[1]), set()).add(tuple(row[field] for field in (
                "payload_bytes", "descriptor_bytes", "alignment_boundary_bytes",
                "alignment_padding_bytes", "transport_bytes", "output_tensor_descriptor_sha256",
            )))
    for layer in selected_layers:
        join_rows = [row for row in summary if row["component"] == "canonical_combine_once_per_join" and row["layer_id"] == layer]
        if len(join_rows) != COMBINE_JOINS or len(combine_descriptors.get(layer, set())) != 1:
            raise ServiceLutError("combine 32-join/descriptor census mismatch")
        measured = [raw for raw in raw_rows if raw["component"] == "canonical_combine_once_per_join" and raw["layer_id"] == layer and raw["phase"] == "measured"]
        cuda = [float(row["cuda_event_us"]) for row in measured]
        wall = [float(row["wall_sync_us"]) for row in measured]
        descriptor = join_rows[0]
        summary.append({
            "model_key": model_key, "model_revision": model_revision, "layer_id": layer,
            "expert_id": -1, "component": "canonical_combine_pooled_layer",
            "route_identity_sha256": "", "join_identity_sha256": "", "rows": 1,
            "measured_count": COMBINE_JOINS * TRIALS,
            "median_cuda_event_us": median(cuda), "p95_cuda_event_us": p95(cuda),
            "max_cuda_event_us": max(cuda), "median_wall_sync_us": median(wall),
            "stability_ratio": p95(cuda) / median(cuda),
            **{field: descriptor[field] for field in (
                "payload_bytes", "descriptor_bytes", "alignment_boundary_bytes",
                "alignment_padding_bytes", "transport_bytes", "output_tensor_descriptor_sha256",
                "source", "evidence_boundary",
            )},
            "producer_source_sha256": producer_source_sha256,
        })
    return sorted(summary, key=lambda row: (int(row["layer_id"]), str(row["component"]), int(row["expert_id"]), str(row["join_identity_sha256"])))


def analytic_cut_summary(timing_summary: Sequence[Mapping[str, Any]], *, link_gbps: float = 200.0) -> list[dict[str, Any]]:
    result = []
    for row in timing_summary:
        if row["component"] != "sender_pack_route_specific_row1":
            continue
        result.append({
            "model_key": row["model_key"], "model_revision": row["model_revision"],
            "layer_id": row["layer_id"], "expert_id": row["expert_id"],
            "component": "shared_cut_analytic_l2_proxy", "transport_bytes": row["transport_bytes"],
            "link_gbps": link_gbps,
            "service_us": int(row["transport_bytes"]) * 8.0 / (link_gbps * 1000.0),
            "source": "analytic_network", "evidence_boundary": "ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA",
        })
    return result

