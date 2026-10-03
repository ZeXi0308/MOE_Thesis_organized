"""Reference policy and replay core for Critical-Join Credits (CJC).

This module implements the frozen ``cjc-v1`` Phase-2 protocol.  It is a
route-real, timing-calibrated *proxy* replay; it is not an RDMA or serving
runtime.  The API deliberately separates identity-complete route records,
join-blind observations, and receiver join state so a policy cannot obtain
future or sibling information accidentally.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, fields
import hashlib
import math
import random
import struct
from typing import Iterable, Mapping, Sequence


PROTOCOL_VERSION = "cjc-v1"
ROUTE_SCHEMA_VERSION = "cjc-route-v1"
ACK_HEADER_BYTES = 16
ACK_RECORD_BYTES = 16
ACK_ALIGNMENT_BYTES = 16
ACK_HEADER = struct.Struct("<4sHHII")
ACK_RECORD = struct.Struct("<QHHI")
ACK_MAGIC = b"CJC1"

JOIN_BLIND_ARMS = (
    "fifo",
    "srpt",
    "edf",
    "receiver_qdepth",
    "largest_flow_first",
    "sync_token_order",
    "topology_join_blind",
    "calib_best_static",
)

ALLOWED_TIME_SOURCES = {
    "measured_same_gpu",
    "derived_from_measured_lut",
    "analytic_link",
    "synthetic_stress",
}

LUT_COMPONENT_PROVENANCE = "component_provenance_v1"
LUT_EXPERT_SOURCE = "measured_cuda_event_real_expert_same_gpu"
LUT_PACK_SOURCE = "measured_cuda_event_index_select_same_gpu"
LUT_LAUNCH_SOURCE = "included_in_gpu_component_measurements"
LUT_HOST_STAGING_SOURCE = "measured_pinned_h2d_same_run_host_not_rdma"
LUT_REDUCTION_SOURCE = "measured_cuda_event_canonical_reduce_amortized_same_gpu"


class CJCValidationError(ValueError):
    """Fail-closed validation error for protocol inputs."""


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def align_up(value: int, alignment: int = ACK_ALIGNMENT_BYTES) -> int:
    if value < 0 or alignment <= 0:
        raise CJCValidationError("alignment inputs must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def ack_message_bytes(record_count: int) -> int:
    if record_count < 0:
        raise CJCValidationError("negative ACK record count")
    if record_count == 0:
        return 0
    return align_up(ACK_HEADER_BYTES + ACK_RECORD_BYTES * record_count)


@dataclass(frozen=True)
class AckWireRecord:
    token_hash64: int
    layer_id: int
    missing_count: int
    epoch: int


def encode_ack_message(records: Sequence[AckWireRecord]) -> bytes:
    if not records:
        return b""
    if len(records) > 0xFFFF:
        raise CJCValidationError("too many ACK records")
    payload = bytearray(ACK_HEADER.pack(ACK_MAGIC, 1, len(records), 0, 0))
    for record in records:
        if not 0 <= record.token_hash64 <= 0xFFFFFFFFFFFFFFFF:
            raise CJCValidationError("ACK token hash overflow")
        if not 0 <= record.layer_id <= 0xFFFF:
            raise CJCValidationError("ACK layer overflow")
        if not 0 <= record.missing_count <= 0xFFFF:
            raise CJCValidationError("ACK missing-count overflow")
        if not 0 < record.epoch <= 0xFFFFFFFF:
            raise CJCValidationError("ACK epoch overflow")
        payload.extend(
            ACK_RECORD.pack(
                record.token_hash64,
                record.layer_id,
                record.missing_count,
                record.epoch,
            )
        )
    payload.extend(b"\0" * (align_up(len(payload)) - len(payload)))
    if len(payload) != ack_message_bytes(len(records)):
        raise AssertionError("ACK byte-accounting implementation drift")
    return bytes(payload)


def decode_ack_message(payload: bytes) -> tuple[AckWireRecord, ...]:
    if not payload:
        return ()
    if len(payload) % ACK_ALIGNMENT_BYTES or len(payload) < ACK_HEADER_BYTES:
        raise CJCValidationError("malformed ACK alignment/length")
    magic, version, count, reserved0, reserved1 = ACK_HEADER.unpack_from(payload, 0)
    if magic != ACK_MAGIC or version != 1 or reserved0 != 0 or reserved1 != 0:
        raise CJCValidationError("malformed ACK header")
    expected = ack_message_bytes(count)
    if len(payload) != expected:
        raise CJCValidationError("ACK count/length mismatch")
    records: list[AckWireRecord] = []
    offset = ACK_HEADER_BYTES
    for _ in range(count):
        token_hash64, layer_id, missing_count, epoch = ACK_RECORD.unpack_from(payload, offset)
        if epoch == 0:
            raise CJCValidationError("ACK epoch zero is invalid")
        records.append(AckWireRecord(token_hash64, layer_id, missing_count, epoch))
        offset += ACK_RECORD_BYTES
    if any(payload[offset:]):
        raise CJCValidationError("non-zero ACK alignment padding")
    return tuple(records)


@dataclass(frozen=True)
class RouteContribution:
    schema_version: str
    model_revision: str
    data_manifest_sha256: str
    request_id: str
    forward_id: str
    batch_id: str
    phase: str
    decode_step: int
    layer_id: int
    token_id: str
    token_position: int
    topk_slot: int
    expert_id: int
    sender_rank: int
    receiver_rank: int
    valid: bool
    route_weight: float
    route_source: str
    placement_manifest_sha256: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "RouteContribution":
        required = {f.name for f in fields(cls)}
        missing = required - set(row)
        if missing:
            raise CJCValidationError(
                "identity-incomplete route row; missing=" + ",".join(sorted(missing))
            )
        if type(row["valid"]) is not bool:
            raise CJCValidationError("route valid must be a JSON boolean")
        try:
            return cls(
                schema_version=str(row["schema_version"]),
                model_revision=str(row["model_revision"]),
                data_manifest_sha256=str(row["data_manifest_sha256"]),
                request_id=str(row["request_id"]),
                forward_id=str(row["forward_id"]),
                batch_id=str(row["batch_id"]),
                phase=str(row["phase"]),
                decode_step=int(row["decode_step"]),
                layer_id=int(row["layer_id"]),
                token_id=str(row["token_id"]),
                token_position=int(row["token_position"]),
                topk_slot=int(row["topk_slot"]),
                expert_id=int(row["expert_id"]),
                sender_rank=int(row["sender_rank"]),
                receiver_rank=int(row["receiver_rank"]),
                valid=row["valid"],
                route_weight=float(row["route_weight"]),
                route_source=str(row["route_source"]),
                placement_manifest_sha256=str(row["placement_manifest_sha256"]),
            )
        except (TypeError, ValueError) as exc:
            raise CJCValidationError(f"malformed route row: {exc}") from exc

    @property
    def route_key(self) -> tuple[object, ...]:
        return (
            self.request_id,
            self.forward_id,
            self.layer_id,
            self.token_id,
            self.topk_slot,
        )

    @property
    def join_key(self) -> tuple[object, ...]:
        return (self.request_id, self.forward_id, self.layer_id, self.token_id)

    @property
    def stable_task_id(self) -> str:
        return stable_hash(self.model_revision, *self.route_key)


@dataclass(frozen=True)
class PlacementManifest:
    sha256: str
    ep_size: int
    gpus_per_node: int
    expert_to_sender: Mapping[tuple[str, int], int]
    request_to_receiver: Mapping[str, int]

    def receiver_resource(self, receiver_rank: int) -> str:
        if not 0 <= receiver_rank < self.ep_size:
            raise CJCValidationError(f"receiver rank out of range: {receiver_rank}")
        return f"node{receiver_rank // self.gpus_per_node}:combine_ingress"


def validate_route_contributions(
    rows: Sequence[RouteContribution],
    *,
    expected_model_revision: str,
    top_k: int,
    num_experts: int,
    placement: PlacementManifest,
    expected_data_manifest_sha256: str | None = None,
    formal: bool = True,
) -> None:
    """Validate G0 identity, physical ownership, and exact top-k closure."""
    if not rows:
        raise CJCValidationError("empty route trace")
    if top_k <= 0 or num_experts <= 0:
        raise CJCValidationError("invalid model dimensions")
    if placement.ep_size <= 0 or placement.gpus_per_node <= 0:
        raise CJCValidationError("invalid topology")

    seen: set[tuple[object, ...]] = set()
    by_join: dict[tuple[object, ...], list[RouteContribution]] = defaultdict(list)
    forward_owner: dict[str, str] = {}
    forward_batch: dict[str, str] = {}
    position_to_token: dict[tuple[object, ...], str] = {}
    token_to_position: dict[tuple[object, ...], int] = {}
    for row in rows:
        if row.schema_version != ROUTE_SCHEMA_VERSION:
            raise CJCValidationError(
                f"legacy/unknown route schema {row.schema_version!r}; formal CJC requires {ROUTE_SCHEMA_VERSION}"
            )
        if row.model_revision != expected_model_revision:
            raise CJCValidationError("model revision mismatch")
        if expected_data_manifest_sha256 is not None and (
            row.data_manifest_sha256 != expected_data_manifest_sha256
        ):
            raise CJCValidationError("route/data manifest hash mismatch")
        if row.placement_manifest_sha256 != placement.sha256:
            raise CJCValidationError("route/placement manifest hash mismatch")
        if formal and row.route_source != "native_model_forward":
            raise CJCValidationError("formal route source must be native_model_forward")
        if not row.valid:
            raise CJCValidationError("formal batch=1 trace must not contain padding/drop rows")
        if not row.batch_id:
            raise CJCValidationError("batch_id must be non-empty")
        if row.phase not in {"prefill", "decode"}:
            raise CJCValidationError(f"unknown phase {row.phase!r}")
        if row.decode_step < 0 or row.layer_id < 0 or row.token_position < 0:
            raise CJCValidationError("negative causal identity field")
        if not 0 <= row.topk_slot < top_k:
            raise CJCValidationError("topk_slot outside frozen model top-k")
        if not 0 <= row.expert_id < num_experts:
            raise CJCValidationError("expert_id outside frozen model expert count")
        if not math.isfinite(row.route_weight):
            raise CJCValidationError("non-finite route weight")
        if row.route_key in seen:
            raise CJCValidationError(f"duplicate route identity: {row.route_key}")
        seen.add(row.route_key)

        previous_request = forward_owner.setdefault(row.forward_id, row.request_id)
        if previous_request != row.request_id:
            raise CJCValidationError("forward_id reused by multiple requests")
        previous_batch = forward_batch.setdefault(row.forward_id, row.batch_id)
        if previous_batch != row.batch_id:
            raise CJCValidationError("one forward_id spans multiple batch identities")
        if placement.expert_to_sender.get((row.model_revision, row.expert_id)) != row.sender_rank:
            raise CJCValidationError("sender rank is not the placement-manifest owner")
        if placement.request_to_receiver.get(row.request_id) != row.receiver_rank:
            raise CJCValidationError("receiver rank is not the origin-manifest owner")
        if not 0 <= row.sender_rank < placement.ep_size:
            raise CJCValidationError("sender rank outside EP topology")
        position_key = (row.request_id, row.forward_id, row.layer_id, row.token_position)
        token_key = (row.request_id, row.forward_id, row.layer_id, row.token_id)
        previous_token = position_to_token.setdefault(position_key, row.token_id)
        previous_position = token_to_position.setdefault(token_key, row.token_position)
        if previous_token != row.token_id or previous_position != row.token_position:
            raise CJCValidationError("token_id/token_position mapping is not one-to-one")
        by_join[row.join_key].append(row)

    for join_key, siblings in by_join.items():
        if len(siblings) != top_k:
            raise CJCValidationError(
                f"join closure failed for {join_key}: {len(siblings)} != top_k {top_k}"
            )
        slots = {row.topk_slot for row in siblings}
        experts = {row.expert_id for row in siblings}
        if slots != set(range(top_k)):
            raise CJCValidationError(f"missing/duplicate logical slot for {join_key}")
        if len(experts) != top_k:
            raise CJCValidationError(f"duplicate selected expert for {join_key}")
        positions = {row.token_position for row in siblings}
        receivers = {row.receiver_rank for row in siblings}
        if len(positions) != 1 or len(receivers) != 1:
            raise CJCValidationError(f"inconsistent token/receiver identity for {join_key}")


@dataclass(frozen=True)
class LUTPoint:
    model_revision: str
    layer_id: int
    rows: int
    expert_us: float
    pack_us: float
    launch_us: float
    host_staging_us: float
    reduction_us: float
    source: str
    expert_source: str | None = None
    pack_source: str | None = None
    launch_source: str | None = None
    host_staging_source: str | None = None
    reduction_source: str | None = None

    def validate(self, *, formal: bool = False) -> None:
        if self.rows <= 0 or self.layer_id < -1:
            raise CJCValidationError("invalid LUT key")
        values = (
            self.expert_us,
            self.pack_us,
            self.launch_us,
            self.host_staging_us,
            self.reduction_us,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise CJCValidationError("invalid LUT timing")
        component_sources = (
            self.expert_source,
            self.pack_source,
            self.launch_source,
            self.host_staging_source,
            self.reduction_source,
        )
        if self.source == "measured_same_gpu" and all(value is None for value in component_sources):
            if formal:
                raise CJCValidationError(
                    "formal LUT requires per-component provenance; legacy measured_same_gpu is ambiguous"
                )
            return
        expected = (
            LUT_EXPERT_SOURCE,
            LUT_PACK_SOURCE,
            LUT_LAUNCH_SOURCE,
            LUT_HOST_STAGING_SOURCE,
            LUT_REDUCTION_SOURCE,
        )
        if self.source != LUT_COMPONENT_PROVENANCE or component_sources != expected:
            raise CJCValidationError("invalid or incomplete LUT component provenance")
        if self.launch_us != 0.0:
            raise CJCValidationError(
                "launch is included in measured GPU components and must not be double-counted"
            )


class ServiceLUT:
    """Piecewise-linear lookup; formal lookups never extrapolate."""

    def __init__(self, points: Iterable[LUTPoint], *, formal: bool = False) -> None:
        self._points: dict[tuple[str, int], list[LUTPoint]] = defaultdict(list)
        for point in points:
            point.validate(formal=formal)
            self._points[(point.model_revision, point.layer_id)].append(point)
        for key in self._points:
            self._points[key].sort(key=lambda point: point.rows)
            rows = [point.rows for point in self._points[key]]
            if len(rows) != len(set(rows)):
                raise CJCValidationError(f"duplicate LUT row count for {key}")
            provenance = {
                (
                    point.source,
                    point.expert_source,
                    point.pack_source,
                    point.launch_source,
                    point.host_staging_source,
                    point.reduction_source,
                )
                for point in self._points[key]
            }
            if len(provenance) != 1:
                raise CJCValidationError(f"mixed LUT provenance within model/layer {key}")
        if not self._points:
            raise CJCValidationError("empty service LUT")

    def lookup(self, model_revision: str, layer_id: int, rows: int) -> LUTPoint:
        points = self._points.get((model_revision, layer_id))
        if points is None:
            points = self._points.get((model_revision, -1))
        if not points:
            raise CJCValidationError(f"missing LUT for model/layer {model_revision}/{layer_id}")
        exact = next((point for point in points if point.rows == rows), None)
        if exact is not None:
            return exact
        lower = [point for point in points if point.rows < rows]
        upper = [point for point in points if point.rows > rows]
        if not lower or not upper:
            raise CJCValidationError(
                f"LUT extrapolation forbidden for model/layer/rows={model_revision}/{layer_id}/{rows}"
            )
        lo, hi = lower[-1], upper[0]
        alpha = (rows - lo.rows) / (hi.rows - lo.rows)

        def lerp(name: str) -> float:
            return getattr(lo, name) + alpha * (getattr(hi, name) - getattr(lo, name))

        return LUTPoint(
            model_revision=model_revision,
            layer_id=layer_id,
            rows=rows,
            expert_us=lerp("expert_us"),
            pack_us=lerp("pack_us"),
            launch_us=lerp("launch_us"),
            host_staging_us=lerp("host_staging_us"),
            reduction_us=lerp("reduction_us"),
            source=lo.source,
            expert_source=lo.expert_source,
            pack_source=lo.pack_source,
            launch_source=lo.launch_source,
            host_staging_source=lo.host_staging_source,
            reduction_source=lo.reduction_source,
        )


@dataclass(frozen=True)
class WorkloadSpec:
    cell: str
    arrival_rate_per_us: float
    layer_period_us: float
    slo_us: float
    mmpp_low_multiplier: float = 0.5
    mmpp_high_multiplier: float = 1.5
    mmpp_switch_probability: float = 0.10

    def validate(self) -> None:
        if self.cell not in {"steady_rho50", "bursty_rho80"}:
            raise CJCValidationError(f"unknown frozen cell {self.cell!r}")
        if self.arrival_rate_per_us <= 0 or self.layer_period_us <= 0 or self.slo_us <= 0:
            raise CJCValidationError("calibration-derived workload constants must be positive")
        if not 0 < self.mmpp_switch_probability < 1:
            raise CJCValidationError("invalid MMPP transition probability")


@dataclass(frozen=True)
class Task:
    task_id: str
    route_key: tuple[object, ...]
    join_key: tuple[object, ...]
    episode_id: str
    model_revision: str
    cell: str
    seed: int
    layer_id: int
    token_position: int
    topk_slot: int
    expert_id: int
    sender_rank: int
    receiver_rank: int
    resource_id: str
    release_us: float
    ready_us: float
    service_us: float
    deadline_us: float
    payload_bytes: int
    descriptor_bytes: int
    alignment_bytes: int
    time_source: str = "derived_from_measured_lut"

    @property
    def wire_bytes(self) -> int:
        return self.payload_bytes + self.descriptor_bytes + self.alignment_bytes

    @property
    def sync_order_key(self) -> tuple[object, ...]:
        request_id, _forward_id, layer_id, token_id, slot = self.route_key
        return (request_id, layer_id, token_id, slot, self.task_id)


def _request_arrivals(
    request_ids: Sequence[str], spec: WorkloadSpec, seed: int
) -> dict[str, float]:
    spec.validate()
    rng = random.Random(seed)
    ordered = sorted(request_ids, key=lambda request_id: stable_hash(seed, request_id))
    now = 0.0
    high = False
    out: dict[str, float] = {}
    for request_id in ordered:
        if spec.cell == "steady_rho50":
            rate = spec.arrival_rate_per_us
        else:
            if rng.random() < spec.mmpp_switch_probability:
                high = not high
            multiplier = spec.mmpp_high_multiplier if high else spec.mmpp_low_multiplier
            rate = spec.arrival_rate_per_us * multiplier
        now += rng.expovariate(rate)
        out[request_id] = now
    return out


def build_tasks_from_routes(
    routes: Sequence[RouteContribution],
    *,
    lut: ServiceLUT,
    placement: PlacementManifest,
    workload: WorkloadSpec,
    seed: int,
    hidden_size: int,
    dtype_bytes: int,
    descriptor_bytes: int,
    alignment_bytes: int,
    link_gbps: float,
) -> list[Task]:
    """Derive ready/service times from LUT + workload, never route timestamps."""
    if not routes:
        raise CJCValidationError("cannot build tasks from empty route trace")
    if hidden_size <= 0 or dtype_bytes <= 0 or link_gbps <= 0:
        raise CJCValidationError("invalid byte/link configuration")
    if descriptor_bytes < 0 or alignment_bytes < 0:
        raise CJCValidationError("negative descriptor/alignment bytes")
    workload.validate()

    model_revision = routes[0].model_revision
    if any(route.model_revision != model_revision for route in routes):
        raise CJCValidationError("one task build may contain only one model revision")
    arrivals = _request_arrivals(sorted({route.request_id for route in routes}), workload, seed)

    compute_groups: dict[tuple[object, ...], list[RouteContribution]] = defaultdict(list)
    for route in routes:
        compute_groups[
            (
                route.request_id,
                route.forward_id,
                route.layer_id,
                route.sender_rank,
                route.expert_id,
            )
        ].append(route)

    sender_jobs: dict[int, list[tuple[float, tuple[object, ...], list[RouteContribution]]]] = defaultdict(list)
    for key, group in compute_groups.items():
        request_id, _forward_id, layer_id, sender_rank, _expert_id = key
        release = arrivals[str(request_id)] + int(layer_id) * workload.layer_period_us
        sender_jobs[int(sender_rank)].append((release, key, group))

    ready_by_route: dict[tuple[object, ...], float] = {}
    for sender_rank, jobs in sender_jobs.items():
        del sender_rank
        cursor = 0.0
        for release, key, group in sorted(jobs, key=lambda item: (item[0], item[1])):
            point = lut.lookup(model_revision, int(key[2]), len(group))
            cursor = max(cursor, release) + point.expert_us
            for route in group:
                ready_by_route[route.route_key] = cursor

    payload_bytes = hidden_size * dtype_bytes
    bytes_per_us = link_gbps * 1e9 / 8.0 / 1e6
    tasks: list[Task] = []
    for route in routes:
        point = lut.lookup(model_revision, route.layer_id, 1)
        wire_bytes = payload_bytes + descriptor_bytes + alignment_bytes
        service_us = (
            point.pack_us
            + point.launch_us
            + point.host_staging_us
            + point.reduction_us
            + wire_bytes / bytes_per_us
        )
        release = arrivals[route.request_id] + route.layer_id * workload.layer_period_us
        tasks.append(
            Task(
                task_id=stable_hash(workload.cell, seed, route.stable_task_id),
                route_key=route.route_key,
                join_key=route.join_key,
                episode_id=route.request_id,
                model_revision=model_revision,
                cell=workload.cell,
                seed=seed,
                layer_id=route.layer_id,
                token_position=route.token_position,
                topk_slot=route.topk_slot,
                expert_id=route.expert_id,
                sender_rank=route.sender_rank,
                receiver_rank=route.receiver_rank,
                resource_id=placement.receiver_resource(route.receiver_rank),
                release_us=release,
                ready_us=ready_by_route[route.route_key],
                service_us=service_us,
                deadline_us=release + workload.slo_us,
                payload_bytes=payload_bytes,
                descriptor_bytes=descriptor_bytes,
                alignment_bytes=alignment_bytes,
            )
        )
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: Sequence[Task]) -> None:
    if not tasks:
        raise CJCValidationError("empty task trace")
    seen_ids: set[str] = set()
    seen_routes: set[tuple[object, ...]] = set()
    by_join: dict[tuple[object, ...], list[Task]] = defaultdict(list)
    scenario = (tasks[0].model_revision, tasks[0].cell, tasks[0].seed)
    for task in tasks:
        if (task.model_revision, task.cell, task.seed) != scenario:
            raise CJCValidationError("task validation requires one model/cell/seed")
        if task.task_id in seen_ids or task.route_key in seen_routes:
            raise CJCValidationError("duplicate task/route identity")
        seen_ids.add(task.task_id)
        seen_routes.add(task.route_key)
        if task.time_source not in ALLOWED_TIME_SOURCES:
            raise CJCValidationError("unknown timing source")
        values = (task.release_us, task.ready_us, task.service_us, task.deadline_us)
        if any(not math.isfinite(value) for value in values):
            raise CJCValidationError("non-finite task time")
        if task.ready_us < task.release_us or task.service_us < 0:
            raise CJCValidationError("invalid release/ready/service order")
        if task.deadline_us <= task.release_us:
            raise CJCValidationError("deadline must follow release")
        if min(task.payload_bytes, task.descriptor_bytes, task.alignment_bytes) < 0:
            raise CJCValidationError("negative byte component")
        by_join[task.join_key].append(task)
    for join_key, siblings in by_join.items():
        slots = {task.topk_slot for task in siblings}
        if len(slots) != len(siblings) or slots != set(range(len(siblings))):
            raise CJCValidationError(f"task join closure failed for {join_key}")
        if len({task.resource_id for task in siblings}) != 1:
            raise CJCValidationError("siblings must return to one receiver resource")


@dataclass(frozen=True)
class JoinBlindTask:
    """Join-blind observation.  It intentionally has no join/missing field."""

    opaque_id: str
    ready_us: float
    service_us: float
    wire_bytes: int
    receiver_rank: int
    resource_id: str
    deadline_us: float
    sync_order_key: tuple[object, ...]


def to_join_blind(task: Task) -> JoinBlindTask:
    return JoinBlindTask(
        opaque_id=task.task_id,
        ready_us=task.ready_us,
        service_us=task.service_us,
        wire_bytes=task.wire_bytes,
        receiver_rank=task.receiver_rank,
        resource_id=task.resource_id,
        deadline_us=task.deadline_us,
        sync_order_key=task.sync_order_key,
    )


def choose_join_blind(
    arm: str,
    ready: Sequence[JoinBlindTask],
    *,
    now_us: float,
    receiver_queue_depth: Mapping[int, int],
    resource_backlog_us: Mapping[str, float],
    calib_static_arm: str = "fifo",
) -> str:
    if not ready:
        raise CJCValidationError("policy called with empty ready set")
    if arm == "calib_best_static":
        if calib_static_arm not in JOIN_BLIND_ARMS[:-1]:
            raise CJCValidationError("invalid calibration-selected static baseline")
        arm = calib_static_arm

    def key(task: JoinBlindTask) -> tuple[object, ...]:
        if arm == "fifo":
            return (task.ready_us, task.opaque_id)
        if arm == "srpt":
            return (task.service_us, task.ready_us, task.opaque_id)
        if arm == "edf":
            return (task.deadline_us - now_us, task.ready_us, task.opaque_id)
        if arm == "receiver_qdepth":
            return (-receiver_queue_depth.get(task.receiver_rank, 0), task.ready_us, task.opaque_id)
        if arm == "largest_flow_first":
            return (-task.wire_bytes, task.ready_us, task.opaque_id)
        if arm == "sync_token_order":
            return (*task.sync_order_key, task.opaque_id)
        if arm == "topology_join_blind":
            projected = resource_backlog_us.get(task.resource_id, 0.0) + task.service_us
            return (projected, task.deadline_us - now_us, task.ready_us, task.opaque_id)
        raise CJCValidationError(f"unknown join-blind arm {arm!r}")

    return min(ready, key=key).opaque_id


def choose_global_causal_join(
    ready: Sequence[Task],
    *,
    now_us: float,
    visible_missing: Mapping[tuple[object, ...], int],
    fairness_debt: Mapping[str, float],
) -> str:
    """Frozen causal score.  No future trace/service argument exists."""
    if not ready:
        raise CJCValidationError("join policy called with empty ready set")

    def key(task: Task) -> tuple[object, ...]:
        missing = visible_missing.get(task.join_key)
        if missing is None or missing <= 0:
            raise CJCValidationError("missing/invalid current join state")
        closes_frontier = missing == 1
        slack = task.deadline_us - now_us
        debt = fairness_debt.get(task.task_id, max(0.0, now_us - task.ready_us))
        return (
            0 if closes_frontier else 1,
            missing,
            slack,
            -debt,
            task.ready_us,
            task.task_id,
        )

    return min(ready, key=key).task_id


@dataclass(frozen=True)
class AckConfig:
    enabled: bool = True
    staleness_us: float = 5.0
    build_us: float = 0.0
    serialize_us: float = 0.0
    wire_us: float = 0.0
    parse_us: float = 0.0
    policy_lookup_us: float = 0.0
    malformed_task_ids: frozenset[str] = frozenset()

    @property
    def charged_us(self) -> float:
        return self.build_us + self.serialize_us + self.wire_us + self.parse_us

    def validate(self) -> None:
        values = (
            self.staleness_us,
            self.build_us,
            self.serialize_us,
            self.wire_us,
            self.parse_us,
            self.policy_lookup_us,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise CJCValidationError("invalid ACK/policy timing")


@dataclass(frozen=True)
class AckEvent:
    visible_us: float
    join_key: tuple[object, ...]
    missing_count: int
    epoch: int
    valid: bool


@dataclass(frozen=True)
class ActionRecord:
    arm: str
    task_id: str
    resource_id: str
    decision_us: float
    start_us: float
    completion_us: float
    visible_missing: int | None
    fallback: bool
    starvation_override: bool


@dataclass
class SimulationResult:
    arm: str
    completion_by_task: dict[str, float]
    action_trace: list[ActionRecord]
    ack_bytes: int
    ack_messages: int
    coordination_charged_us: float
    policy_charged_us: float
    fallback_count: int
    stale_decisions: int
    starvation_overrides: int
    task_fingerprint: str
    data_bytes: int


def task_set_fingerprint(tasks: Sequence[Task]) -> str:
    rows = [
        (
            task.task_id,
            task.route_key,
            task.ready_us,
            task.service_us,
            task.wire_bytes,
            task.resource_id,
        )
        for task in sorted(tasks, key=lambda task: task.task_id)
    ]
    return stable_hash(repr(rows))


def simulate(
    tasks: Sequence[Task],
    *,
    arm: str,
    ack: AckConfig | None = None,
    calib_static_arm: str = "fifo",
    fallback_arm: str = "topology_join_blind",
    starvation_us: float = math.inf,
) -> SimulationResult:
    """Run a deterministic, event-driven, non-preemptive ingress replay."""
    validate_tasks(tasks)
    if arm != "global_causal_join" and arm not in JOIN_BLIND_ARMS:
        raise CJCValidationError(f"unsupported formal arm {arm!r}")
    if fallback_arm not in JOIN_BLIND_ARMS:
        raise CJCValidationError("fallback must be join-blind")
    ack = ack or AckConfig(enabled=False)
    ack.validate()
    if starvation_us <= 0:
        raise CJCValidationError("starvation bound must be positive")

    completion_by_task: dict[str, float] = {}
    actions: list[ActionRecord] = []
    total_ack_bytes = 0
    ack_messages = 0
    coordination_charged_us = 0.0
    policy_charged_us = 0.0
    fallback_count = 0
    stale_decisions = 0
    starvation_overrides = 0

    by_resource: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_resource[task.resource_id].append(task)

    for resource_id, resource_tasks in sorted(by_resource.items()):
        expected: dict[tuple[object, ...], int] = defaultdict(int)
        for task in resource_tasks:
            expected[task.join_key] += 1
        pending = sorted(resource_tasks, key=lambda task: (task.ready_us, task.task_id))
        ready: list[Task] = []
        now = pending[0].ready_us
        actual_missing = dict(expected)
        visible_missing = dict(expected)
        ack_events: list[AckEvent] = []
        last_visible_epoch: dict[tuple[object, ...], int] = defaultdict(int)
        epoch = 0
        force_fallback = False

        while pending or ready:
            # ACK delivery is processed before releases at the same timestamp.
            delivered = [event for event in ack_events if event.visible_us <= now]
            ack_events = [event for event in ack_events if event.visible_us > now]
            for event in sorted(delivered, key=lambda item: item.visible_us):
                if event.valid and event.epoch > last_visible_epoch[event.join_key]:
                    visible_missing[event.join_key] = event.missing_count
                    last_visible_epoch[event.join_key] = event.epoch
                else:
                    force_fallback = True

            while pending and pending[0].ready_us <= now:
                ready.append(pending.pop(0))
            if not ready:
                next_ready = pending[0].ready_us if pending else math.inf
                next_ack = min((event.visible_us for event in ack_events), default=math.inf)
                now = min(next_ready, next_ack)
                continue

            receiver_qdepth: dict[int, int] = defaultdict(int)
            for task in ready:
                receiver_qdepth[task.receiver_rank] += 1
            ready_joins = {task.join_key for task in ready}
            late_for_ready = any(
                event.join_key in ready_joins and event.visible_us > now
                for event in ack_events
            )
            overdue = [task for task in ready if now - task.ready_us >= starvation_us]
            fallback = False
            if overdue:
                chosen_id = min(overdue, key=lambda task: (task.ready_us, task.task_id)).task_id
                starvation_overrides += 1
                starvation = True
            elif arm == "global_causal_join" and not force_fallback and not late_for_ready:
                fairness = {task.task_id: max(0.0, now - task.ready_us) for task in ready}
                chosen_id = choose_global_causal_join(
                    ready,
                    now_us=now,
                    visible_missing=visible_missing,
                    fairness_debt=fairness,
                )
                starvation = False
                if any(visible_missing[t.join_key] != actual_missing[t.join_key] for t in ready):
                    stale_decisions += 1
            else:
                fallback = arm == "global_causal_join"
                if fallback:
                    fallback_count += 1
                    force_fallback = False
                    if late_for_ready:
                        stale_decisions += 1
                blind = [to_join_blind(task) for task in ready]
                chosen_id = choose_join_blind(
                    fallback_arm if fallback else arm,
                    blind,
                    now_us=now,
                    receiver_queue_depth=receiver_qdepth,
                    resource_backlog_us={resource_id: sum(task.service_us for task in ready)},
                    calib_static_arm=calib_static_arm,
                )
                starvation = False

            chosen = next(task for task in ready if task.task_id == chosen_id)
            ready.remove(chosen)
            decision_us = now
            lookup_us = ack.policy_lookup_us if arm == "global_causal_join" else 0.0
            start_us = now + lookup_us
            completion_us = start_us + chosen.service_us
            policy_charged_us += lookup_us
            completion_by_task[chosen.task_id] = completion_us
            actions.append(
                ActionRecord(
                    arm=arm,
                    task_id=chosen.task_id,
                    resource_id=resource_id,
                    decision_us=decision_us,
                    start_us=start_us,
                    completion_us=completion_us,
                    visible_missing=(
                        visible_missing[chosen.join_key] if arm == "global_causal_join" else None
                    ),
                    fallback=fallback,
                    starvation_override=starvation,
                )
            )

            actual_missing[chosen.join_key] -= 1
            if actual_missing[chosen.join_key] < 0:
                raise CJCValidationError("exactly-once violation")
            if arm == "global_causal_join":
                epoch += 1
                if ack.enabled:
                    bytes_this = ack_message_bytes(1)
                    total_ack_bytes += bytes_this
                    ack_messages += 1
                    coordination_charged_us += ack.charged_us
                    ack_events.append(
                        AckEvent(
                            # With no timestamped overlap proof, build/serialize/
                            # transfer/parse is additive to the configured state
                            # staleness.  Charging CPU time while exposing the ACK
                            # at completion+staleness would grant free overlap.
                            visible_us=completion_us + ack.charged_us + ack.staleness_us,
                            join_key=chosen.join_key,
                            missing_count=actual_missing[chosen.join_key],
                            epoch=epoch,
                            valid=chosen.task_id not in ack.malformed_task_ids,
                        )
                    )
                    now = completion_us + ack.charged_us
                else:
                    visible_missing[chosen.join_key] = actual_missing[chosen.join_key]
                    now = completion_us
            else:
                now = completion_us

    if len(completion_by_task) != len(tasks):
        raise CJCValidationError("full-drain failure")
    if any(value != 0 for value in actual_missing.values()):
        raise CJCValidationError("join set not fully drained")
    return SimulationResult(
        arm=arm,
        completion_by_task=completion_by_task,
        action_trace=actions,
        ack_bytes=total_ack_bytes,
        ack_messages=ack_messages,
        coordination_charged_us=coordination_charged_us,
        policy_charged_us=policy_charged_us,
        fallback_count=fallback_count,
        stale_decisions=stale_decisions,
        starvation_overrides=starvation_overrides,
        task_fingerprint=task_set_fingerprint(tasks),
        data_bytes=sum(task.wire_bytes for task in tasks),
    )


def assert_arm_equivalence(results: Sequence[SimulationResult], task_count: int) -> None:
    if not results:
        raise CJCValidationError("no arm results")
    fingerprints = {result.task_fingerprint for result in results}
    data_bytes = {result.data_bytes for result in results}
    completed = {len(result.completion_by_task) for result in results}
    if len(fingerprints) != 1 or len(data_bytes) != 1 or completed != {task_count}:
        raise CJCValidationError("task-set/data/service/full-drain equivalence failed")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise CJCValidationError("quantile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    alpha = position - lo
    return float(ordered[lo] * (1 - alpha) + ordered[hi] * alpha)


@dataclass(frozen=True)
class EpisodeMetrics:
    model_revision: str
    cell: str
    seed: int
    episode_id: str
    arm: str
    token_latencies_us: tuple[float, ...]
    slo_us: float

    @property
    def p99_us(self) -> float:
        return _quantile(self.token_latencies_us, 0.99)

    @property
    def violation_fraction(self) -> float:
        return sum(value > self.slo_us for value in self.token_latencies_us) / len(
            self.token_latencies_us
        )


def episode_metrics(
    tasks: Sequence[Task], result: SimulationResult, *, slo_us: float
) -> list[EpisodeMetrics]:
    task_by_id = {task.task_id: task for task in tasks}
    completion_by_join: dict[tuple[object, ...], float] = defaultdict(float)
    release_by_join: dict[tuple[object, ...], float] = {}
    for task_id, completion in result.completion_by_task.items():
        task = task_by_id[task_id]
        completion_by_join[task.join_key] = max(completion_by_join[task.join_key], completion)
        release_by_join.setdefault(task.join_key, task.release_us)
        if release_by_join[task.join_key] != task.release_us:
            raise CJCValidationError("siblings disagree on token release")
    by_episode: dict[str, list[float]] = defaultdict(list)
    for join_key, completion in completion_by_join.items():
        by_episode[str(join_key[0])].append(completion - release_by_join[join_key])
    first = tasks[0]
    return [
        EpisodeMetrics(
            model_revision=first.model_revision,
            cell=first.cell,
            seed=first.seed,
            episode_id=episode_id,
            arm=result.arm,
            token_latencies_us=tuple(sorted(latencies)),
            slo_us=slo_us,
        )
        for episode_id, latencies in sorted(by_episode.items())
    ]


@dataclass(frozen=True)
class BootstrapSummary:
    candidate_arm: str
    baseline_arms: tuple[str, ...]
    p99_gain: float
    p99_gain_ci_low: float
    p99_gain_ci_high: float
    violation_reduction: float
    violation_reduction_ci_low: float
    violation_reduction_ci_high: float
    n_episodes: int
    n_seeds: int
    n_bootstrap: int


def paired_hierarchical_bootstrap(
    rows: Sequence[EpisodeMetrics],
    *,
    candidate_arm: str,
    baseline_arms: Sequence[str],
    n_bootstrap: int = 2000,
    seed: int = 20260722,
) -> BootstrapSummary:
    """Paired document-then-seed bootstrap; never resamples contributions."""
    if n_bootstrap <= 0:
        raise CJCValidationError("n_bootstrap must be positive")
    arms = {candidate_arm, *baseline_arms}
    if not baseline_arms:
        raise CJCValidationError("strong baseline set is empty")
    index: dict[tuple[str, int, str], EpisodeMetrics] = {}
    episodes: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row.arm not in arms:
            continue
        key = (row.episode_id, row.seed, row.arm)
        if key in index:
            raise CJCValidationError("duplicate episode/seed/arm metric")
        index[key] = row
        episodes[row.episode_id].add(row.seed)
    if not episodes:
        raise CJCValidationError("no paired episodes")
    for episode_id, seeds in episodes.items():
        for run_seed in seeds:
            missing = [arm for arm in arms if (episode_id, run_seed, arm) not in index]
            if missing:
                raise CJCValidationError(
                    f"unpaired arms for episode/seed {episode_id}/{run_seed}: {missing}"
                )

    episode_ids = sorted(episodes)

    def evaluate(selected: Sequence[tuple[str, int]]) -> tuple[float, float]:
        latencies: dict[str, list[float]] = {arm: [] for arm in arms}
        violations: dict[str, list[bool]] = {arm: [] for arm in arms}
        for episode_id, run_seed in selected:
            for arm in arms:
                row = index[(episode_id, run_seed, arm)]
                latencies[arm].extend(row.token_latencies_us)
                violations[arm].extend(value > row.slo_us for value in row.token_latencies_us)
        p99 = {arm: _quantile(values, 0.99) for arm, values in latencies.items()}
        violation = {
            arm: sum(values) / len(values) for arm, values in violations.items()
        }
        best_p99 = min(p99[arm] for arm in baseline_arms)
        best_violation = min(violation[arm] for arm in baseline_arms)
        p99_gain = (best_p99 - p99[candidate_arm]) / max(best_p99, 1e-12)
        violation_reduction = best_violation - violation[candidate_arm]
        return p99_gain, violation_reduction

    point_pairs = [
        (episode_id, run_seed)
        for episode_id in episode_ids
        for run_seed in sorted(episodes[episode_id])
    ]
    point_gain, point_violation = evaluate(point_pairs)
    rng = random.Random(seed)
    gain_samples: list[float] = []
    violation_samples: list[float] = []
    for _ in range(n_bootstrap):
        sampled: list[tuple[str, int]] = []
        for _episode_draw in episode_ids:
            episode_id = rng.choice(episode_ids)
            seeds = sorted(episodes[episode_id])
            for _seed_draw in seeds:
                sampled.append((episode_id, rng.choice(seeds)))
        gain, violation = evaluate(sampled)
        gain_samples.append(gain)
        violation_samples.append(violation)

    return BootstrapSummary(
        candidate_arm=candidate_arm,
        baseline_arms=tuple(baseline_arms),
        p99_gain=point_gain,
        p99_gain_ci_low=_quantile(gain_samples, 0.025),
        p99_gain_ci_high=_quantile(gain_samples, 0.975),
        violation_reduction=point_violation,
        violation_reduction_ci_low=_quantile(violation_samples, 0.025),
        violation_reduction_ci_high=_quantile(violation_samples, 0.975),
        n_episodes=len(episode_ids),
        n_seeds=len({seed_value for values in episodes.values() for seed_value in values}),
        n_bootstrap=n_bootstrap,
    )


def canonical_reduction_signature(
    contributions: Sequence[tuple[int, int, bytes]],
) -> str:
    """Hash slot/expert-ordered payloads; arrival/action order is irrelevant."""
    ordered = sorted(contributions, key=lambda row: (row[0], row[1]))
    if len({slot for slot, _expert, _payload in ordered}) != len(ordered):
        raise CJCValidationError("duplicate canonical reduction slot")
    digest = hashlib.sha256()
    for slot, expert, payload in ordered:
        digest.update(int(slot).to_bytes(4, "little", signed=False))
        digest.update(int(expert).to_bytes(4, "little", signed=False))
        digest.update(len(payload).to_bytes(8, "little", signed=False))
        digest.update(payload)
    return digest.hexdigest()
