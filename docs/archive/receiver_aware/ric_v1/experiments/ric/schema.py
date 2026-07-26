"""Fail-closed identities and full-background-load validation for RIC v1."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence


PROTOCOL_VERSION = "ric-v1"
SOURCE_TAGS = frozenset(
    {
        "measured_5090_cuda",
        "measured_5090_host",
        "measured_5090_h2d_not_rdma",
        "derived_from_measured_lut",
        "analytic_network",
        "synthetic_delay",
    }
)


class RICValidationError(ValueError):
    """An input violated a frozen RIC invariant."""


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise RICValidationError(f"{name} must be a non-empty string")


def _require_int_range(name: str, value: int, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise RICValidationError(f"{name} must be an integer in [{lower}, {upper}]")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True, order=True)
class JoinIdentity:
    """Full application join identity used by the collision-checked table."""

    request_id: str
    forward_id: str
    batch_id: str
    phase: str
    decode_step: int
    layer_id: int
    token_id: str
    token_block_id: str
    receiver_rank: int
    epoch: int

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "forward_id",
            "batch_id",
            "token_id",
            "token_block_id",
        ):
            _require_nonempty(name, getattr(self, name))
        if self.phase not in {"prefill", "decode"}:
            raise RICValidationError("phase must be 'prefill' or 'decode'")
        _require_int_range("decode_step", self.decode_step, 0, 0xFFFFFFFF)
        _require_int_range("layer_id", self.layer_id, 0, 0xFFFF)
        _require_int_range("receiver_rank", self.receiver_rank, 0, 0xFF)
        _require_int_range("epoch", self.epoch, 1, 0xFFFFFFFF)

    def canonical_tuple(self) -> tuple[object, ...]:
        return tuple(getattr(self, field.name) for field in fields(self))

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.canonical_tuple())


@dataclass(frozen=True, order=True)
class ContributionIdentity:
    """The exact frozen contribution identity from RIC Phase 2 section 2.2."""

    request_id: str
    forward_id: str
    batch_id: str
    phase: str
    decode_step: int
    layer_id: int
    token_id: str
    token_block_id: str
    topk_slot: int
    expert_id: int
    sender_rank: int
    receiver_rank: int
    epoch: int

    def __post_init__(self) -> None:
        # Constructing the JoinIdentity applies all common validation.
        self.join_identity
        _require_int_range("topk_slot", self.topk_slot, 0, 0xFFFF)
        _require_int_range("expert_id", self.expert_id, 0, 0xFFFF)
        _require_int_range("sender_rank", self.sender_rank, 0, 0xFF)

    @property
    def join_identity(self) -> JoinIdentity:
        return JoinIdentity(
            request_id=self.request_id,
            forward_id=self.forward_id,
            batch_id=self.batch_id,
            phase=self.phase,
            decode_step=self.decode_step,
            layer_id=self.layer_id,
            token_id=self.token_id,
            token_block_id=self.token_block_id,
            receiver_rank=self.receiver_rank,
            epoch=self.epoch,
        )

    def canonical_tuple(self) -> tuple[object, ...]:
        return tuple(getattr(self, field.name) for field in fields(self))

    @property
    def stable_full_task_id(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.canonical_tuple())).hexdigest()


@dataclass(frozen=True)
class ContributionRecord:
    """One full-load return contribution and its frozen service realization."""

    identity: ContributionIdentity
    model_revision: str
    valid: bool
    arrival_us: float
    ready_us: float
    service_us: float
    deadline_us: float
    payload_bytes: int
    descriptor_bytes: int
    alignment_bytes: int
    source_tag: str

    def __post_init__(self) -> None:
        if type(self.identity) is not ContributionIdentity:
            raise RICValidationError("identity must be exactly ContributionIdentity")
        _require_nonempty("model_revision", self.model_revision)
        if type(self.valid) is not bool:
            raise RICValidationError("valid must be a bool")
        for name in ("arrival_us", "ready_us", "service_us", "deadline_us"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RICValidationError(f"{name} must be finite")
        if self.arrival_us < 0 or self.ready_us < self.arrival_us:
            raise RICValidationError("ready_us must follow a non-negative arrival_us")
        if self.service_us <= 0:
            raise RICValidationError("service_us must be positive")
        if self.deadline_us <= self.arrival_us:
            raise RICValidationError("deadline_us must follow arrival_us")
        for name in ("payload_bytes", "descriptor_bytes", "alignment_bytes"):
            _require_int_range(name, getattr(self, name), 0, 0x7FFFFFFFFFFFFFFF)
        if self.source_tag not in SOURCE_TAGS:
            raise RICValidationError(f"unknown frozen source tag {self.source_tag!r}")

    @property
    def wire_bytes(self) -> int:
        return self.payload_bytes + self.descriptor_bytes + self.alignment_bytes

    @property
    def stable_full_task_id(self) -> str:
        return self.identity.stable_full_task_id


def _record_fingerprint_row(record: ContributionRecord) -> tuple[object, ...]:
    return (
        record.identity.canonical_tuple(),
        record.model_revision,
        record.valid,
        float(record.arrival_us),
        float(record.ready_us),
        float(record.service_us),
        float(record.deadline_us),
        record.payload_bytes,
        record.descriptor_bytes,
        record.alignment_bytes,
        record.source_tag,
    )


def full_task_fingerprint(records: Sequence[ContributionRecord]) -> str:
    """Hash all tasks and service/byte realizations; never accepts a score mask."""

    if not records:
        raise RICValidationError("cannot fingerprint an empty full workload")
    rows = sorted((_record_fingerprint_row(record) for record in records), key=repr)
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


@dataclass(frozen=True)
class FullLoadAudit:
    record_count: int
    request_count: int
    join_count: int
    scored_join_count: int
    payload_bytes: int
    descriptor_bytes: int
    alignment_bytes: int
    wire_bytes: int
    service_us: float
    full_task_fingerprint: str
    resource_demand_fingerprint: str
    score_mask_fingerprint: str


def validate_full_background(
    records: Sequence[ContributionRecord],
    *,
    top_k: int,
    num_experts: int,
    ep_size: int,
    expected_request_ids: Iterable[str],
    expected_token_blocks_per_request: int = 128,
    expert_to_sender: Mapping[int, int] | None = None,
    request_to_receiver: Mapping[str, int] | None = None,
    expected_layer_by_request: Mapping[str, int] | None = None,
    expected_model_revision: str | None = None,
    score_join_identities: Iterable[JoinIdentity] | None = None,
) -> FullLoadAudit:
    """Validate identity closure without ever filtering the scheduled workload.

    ``score_join_identities`` is deliberately consumed only after all task,
    byte, service, placement, and closure checks.  Consequently it cannot alter
    the full task or resource-demand fingerprints.
    """

    if not records:
        raise RICValidationError("full workload is empty")
    _require_int_range("top_k", top_k, 1, 16)
    _require_int_range("num_experts", num_experts, top_k, 0xFFFF)
    _require_int_range("ep_size", ep_size, 1, 0x100)
    if expected_token_blocks_per_request <= 0:
        raise RICValidationError("expected token-block count must be positive")

    expected_requests = tuple(expected_request_ids)
    if not expected_requests or len(expected_requests) != len(set(expected_requests)):
        raise RICValidationError("expected request ids must be non-empty and unique")
    expected_request_set = set(expected_requests)

    seen_identity: set[ContributionIdentity] = set()
    joins: dict[JoinIdentity, list[ContributionRecord]] = defaultdict(list)
    forward_to_batch: dict[str, str] = {}
    batch_to_forward: dict[str, str] = {}
    forward_to_request: dict[str, str] = {}
    batch_to_request: dict[str, str] = {}
    observed_requests: set[str] = set()

    for record in records:
        if type(record) is not ContributionRecord:
            raise RICValidationError("all workload rows must be ContributionRecord")
        identity = record.identity
        if not record.valid:
            raise RICValidationError("padding/drop/invalid contribution in full workload")
        if identity in seen_identity:
            raise RICValidationError("duplicate full contribution identity")
        seen_identity.add(identity)
        if identity.topk_slot >= top_k:
            raise RICValidationError("topk_slot outside the frozen model top-k")
        if identity.expert_id >= num_experts:
            raise RICValidationError("expert_id outside the frozen model")
        if identity.sender_rank >= ep_size or identity.receiver_rank >= ep_size:
            raise RICValidationError("sender/receiver outside the frozen EP topology")
        if expected_model_revision is not None and record.model_revision != expected_model_revision:
            raise RICValidationError("model revision mismatch")
        if identity.request_id not in expected_request_set:
            raise RICValidationError("unexpected request in full workload")
        observed_requests.add(identity.request_id)

        previous = forward_to_batch.setdefault(identity.forward_id, identity.batch_id)
        if previous != identity.batch_id:
            raise RICValidationError("one forward_id maps to multiple batch_id values")
        previous = batch_to_forward.setdefault(identity.batch_id, identity.forward_id)
        if previous != identity.forward_id:
            raise RICValidationError("one batch_id maps to multiple forward_id values")
        previous = forward_to_request.setdefault(identity.forward_id, identity.request_id)
        if previous != identity.request_id:
            raise RICValidationError("forward_id reused across requests")
        previous = batch_to_request.setdefault(identity.batch_id, identity.request_id)
        if previous != identity.request_id:
            raise RICValidationError("batch_id reused across requests")

        if expert_to_sender is not None:
            if expert_to_sender.get(identity.expert_id) != identity.sender_rank:
                raise RICValidationError("sender_rank is not the frozen expert owner")
        if request_to_receiver is not None:
            if request_to_receiver.get(identity.request_id) != identity.receiver_rank:
                raise RICValidationError("receiver_rank is not the frozen request origin")
        if expected_layer_by_request is not None:
            if expected_layer_by_request.get(identity.request_id) != identity.layer_id:
                raise RICValidationError("request is not bound to its frozen selected layer")
        joins[identity.join_identity].append(record)

    if observed_requests != expected_request_set:
        missing = sorted(expected_request_set - observed_requests)
        raise RICValidationError(f"missing full-background request(s): {missing}")

    joins_by_request: dict[str, set[JoinIdentity]] = defaultdict(set)
    for join_identity, siblings in joins.items():
        if len(siblings) != top_k:
            raise RICValidationError("join set does not contain exactly top_k contributions")
        slots = {row.identity.topk_slot for row in siblings}
        experts = {row.identity.expert_id for row in siblings}
        if slots != set(range(top_k)):
            raise RICValidationError("join set has a missing/duplicate logical top-k slot")
        if len(experts) != top_k:
            raise RICValidationError("join set selects a duplicate expert")
        if len({float(row.arrival_us) for row in siblings}) != 1:
            raise RICValidationError("top-k siblings must share one micro-coflow arrival")
        if len({float(row.deadline_us) for row in siblings}) != 1:
            raise RICValidationError("top-k siblings must share one join-level deadline")
        joins_by_request[join_identity.request_id].add(join_identity)

    for request_id in expected_requests:
        request_joins = joins_by_request.get(request_id, set())
        if len(request_joins) != expected_token_blocks_per_request:
            raise RICValidationError(
                f"request {request_id!r} has {len(request_joins)} token blocks; "
                f"expected {expected_token_blocks_per_request}"
            )
        block_ids = {join.token_block_id for join in request_joins}
        if len(block_ids) != expected_token_blocks_per_request:
            raise RICValidationError("token_block_id is not unique within a request trace")

    all_joins = set(joins)
    scored = all_joins if score_join_identities is None else set(score_join_identities)
    if not scored <= all_joins:
        raise RICValidationError("score mask references a join outside the full workload")

    resource_rows: dict[tuple[int, int], list[ContributionRecord]] = defaultdict(list)
    for record in records:
        resource_rows[(record.identity.sender_rank, record.identity.receiver_rank)].append(record)
    resource_summary = []
    for resource, members in sorted(resource_rows.items()):
        resource_summary.append(
            (
                resource,
                len(members),
                sum(row.payload_bytes for row in members),
                sum(row.descriptor_bytes for row in members),
                sum(row.alignment_bytes for row in members),
                math.fsum(float(row.service_us) for row in members),
            )
        )
    resource_fingerprint = hashlib.sha256(_canonical_json_bytes(resource_summary)).hexdigest()
    score_fingerprint = hashlib.sha256(
        _canonical_json_bytes(sorted((join.canonical_tuple() for join in scored), key=repr))
    ).hexdigest()

    payload = sum(record.payload_bytes for record in records)
    descriptor = sum(record.descriptor_bytes for record in records)
    alignment = sum(record.alignment_bytes for record in records)
    return FullLoadAudit(
        record_count=len(records),
        request_count=len(observed_requests),
        join_count=len(joins),
        scored_join_count=len(scored),
        payload_bytes=payload,
        descriptor_bytes=descriptor,
        alignment_bytes=alignment,
        wire_bytes=payload + descriptor + alignment,
        service_us=math.fsum(float(record.service_us) for record in records),
        full_task_fingerprint=full_task_fingerprint(records),
        resource_demand_fingerprint=resource_fingerprint,
        score_mask_fingerprint=score_fingerprint,
    )


__all__ = [
    "PROTOCOL_VERSION",
    "SOURCE_TAGS",
    "RICValidationError",
    "JoinIdentity",
    "ContributionIdentity",
    "ContributionRecord",
    "FullLoadAudit",
    "full_task_fingerprint",
    "validate_full_background",
]
