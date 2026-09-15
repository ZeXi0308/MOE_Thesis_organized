"""Pure DDRC policy, visibility, serialization, and accounting reference.

This module implements the Phase-2-frozen DDRC semantics without model or
transport dependencies.  The deployable policies deliberately consume narrow
views:

* ``plan_sender_local`` receives one sender's dispatch handle only.
* ``build_receiver_credit_messages`` receives one receive-resource view only.
* ``apply_receiver_credit_messages`` receives one sender view plus serialized
  credit messages.
* ``plan_global_oracle`` is the only entry point that receives the full matrix.

The module is a numerical/accounting reference.  Its link time and H2D fields
remain proxies and must never be reported as NCCL/RDMA measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import copy
import itertools
import math
import struct
from typing import Iterable, Mapping, Sequence


Lane = tuple[int, int]

INFORMATION_SENDER_LOCAL = "deployable_sender_local"
INFORMATION_RECEIVER_CREDIT = "deployable_receiver_credit"
INFORMATION_GLOBAL_ORACLE = "oracle_global_same_layer"
INFORMATION_CAUSAL_HISTORY = "causal_history_strictly_past"

CREDIT_MAGIC = b"DDRC"
CREDIT_VERSION = 1
CREDIT_HEADER = struct.Struct("<4sBBHIHH")  # exactly 16 bytes
CREDIT_RECORD = struct.Struct("<HHBBH")  # exactly 8 bytes
LOW_PRECISION_CODE = 1


def align_up(value: int, alignment: int) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    if alignment < 1:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def deterministic_origin_lpt(
    request_weights: Mapping[str, int], ep_size: int
) -> dict[str, int]:
    """Assign new requests using scheduler-visible weights only.

    Requests are sorted by descending current token weight and stable request
    key.  The least-loaded rank wins, with rank id as the tie-break.  No route
    or expert field is accepted by this API.
    """

    if ep_size < 1:
        raise ValueError("ep_size must be positive")
    if any(weight < 0 for weight in request_weights.values()):
        raise ValueError("request weights cannot be negative")
    loads = [0] * ep_size
    assignment: dict[str, int] = {}
    for request_id, weight in sorted(
        request_weights.items(), key=lambda item: (-item[1], item[0])
    ):
        rank = min(range(ep_size), key=lambda candidate: (loads[candidate], candidate))
        assignment[request_id] = rank
        loads[rank] += int(weight)
    return assignment


@dataclass(frozen=True)
class Topology:
    """Frozen rank to node/receive-resource mapping."""

    ep_size: int
    node_by_rank: tuple[int, ...]
    receive_resource_by_rank: tuple[str, ...]
    link_gbps: float

    def __post_init__(self) -> None:
        if self.ep_size < 1:
            raise ValueError("ep_size must be positive")
        if len(self.node_by_rank) != self.ep_size:
            raise ValueError("node_by_rank length must equal ep_size")
        if len(self.receive_resource_by_rank) != self.ep_size:
            raise ValueError("receive_resource_by_rank length must equal ep_size")
        if self.link_gbps <= 0:
            raise ValueError("link_gbps must be positive")

    def remote(self, sender: int, receiver: int) -> bool:
        self._check_rank(sender)
        self._check_rank(receiver)
        return self.node_by_rank[sender] != self.node_by_rank[receiver]

    def receive_resource(self, receiver: int) -> str:
        self._check_rank(receiver)
        return self.receive_resource_by_rank[receiver]

    def _check_rank(self, rank: int) -> None:
        if not 0 <= rank < self.ep_size:
            raise ValueError(f"rank {rank} outside EP size {self.ep_size}")


@dataclass(frozen=True)
class FormatTiming:
    """Measured unit timing at ``measured_rows`` rows, in microseconds."""

    pack_us: float
    unpack_us: float
    h2d_us: float
    measured_rows: int
    source: str

    def __post_init__(self) -> None:
        if min(self.pack_us, self.unpack_us, self.h2d_us) < 0:
            raise ValueError("format timing cannot be negative")
        if self.measured_rows < 1:
            raise ValueError("measured_rows must be positive")
        if self.source not in {
            "measured_same_run",
            "measured_other_gpu",
            "analytic",
            "assumed",
        }:
            raise ValueError(f"invalid timing source: {self.source}")


@dataclass(frozen=True)
class CreditTiming:
    build_us: float
    aggregate_us: float
    transfer_us: float
    parse_us: float
    pack_deadline_slack_us: float
    overlap_proven: bool
    source: str

    def __post_init__(self) -> None:
        values = (
            self.build_us,
            self.aggregate_us,
            self.transfer_us,
            self.parse_us,
            self.pack_deadline_slack_us,
        )
        if min(values) < 0:
            raise ValueError("credit timing cannot be negative")
        if self.source not in {
            "measured_same_run",
            "measured_other_gpu",
            "analytic",
            "assumed",
        }:
            raise ValueError(f"invalid credit timing source: {self.source}")

    @property
    def total_us(self) -> float:
        return self.build_us + self.aggregate_us + self.transfer_us + self.parse_us

    @property
    def visible_us(self) -> float:
        if not self.overlap_proven:
            return self.total_us
        return max(0.0, self.total_us - self.pack_deadline_slack_us)

    @property
    def on_time(self) -> bool:
        return self.total_us <= self.pack_deadline_slack_us


@dataclass(frozen=True)
class AccountingConfig:
    hidden_dim: int
    high_bits: int = 8
    low_bits: int = 4
    high_scale_bytes_per_row: int = 4
    low_scale_bytes_per_row: int = 4
    lane_descriptor_bytes: int = 16
    lane_alignment_bytes: int = 16
    codec_tile_rows: int = 32
    codec_tax_mode: str = "serialized_tiles"
    high_timing: FormatTiming = field(
        default_factory=lambda: FormatTiming(0.0, 0.0, 0.0, 128, "assumed")
    )
    low_timing: FormatTiming = field(
        default_factory=lambda: FormatTiming(0.0, 0.0, 0.0, 128, "assumed")
    )
    credit_timing: CreditTiming = field(
        default_factory=lambda: CreditTiming(0.0, 0.0, 0.0, 0.0, 0.0, False, "assumed")
    )
    credit_header_bytes: int = CREDIT_HEADER.size
    credit_record_bytes: int = CREDIT_RECORD.size
    credit_alignment_bytes: int = 16
    evidence_boundary: str = "NOT_RDMA / host-staging proxy"

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.high_bits <= self.low_bits or self.low_bits < 1:
            raise ValueError("expected high_bits > low_bits > 0")
        if min(self.high_scale_bytes_per_row, self.low_scale_bytes_per_row) < 0:
            raise ValueError("scale bytes cannot be negative")
        if self.lane_descriptor_bytes < 0:
            raise ValueError("descriptor bytes cannot be negative")
        if self.lane_alignment_bytes < 1 or self.credit_alignment_bytes < 1:
            raise ValueError("alignment must be positive")
        if self.codec_tile_rows < 1:
            raise ValueError("codec_tile_rows must be positive")
        if self.codec_tax_mode not in {
            "serialized_tiles",
            "amortized_once_per_step_proxy",
        }:
            raise ValueError("unsupported codec_tax_mode")
        if self.credit_header_bytes != CREDIT_HEADER.size:
            raise ValueError("DDRC-v0 credit header must be exactly 16 B")
        if self.credit_record_bytes != CREDIT_RECORD.size:
            raise ValueError("DDRC-v0 credit record must be exactly 8 B")
        if "NOT_RDMA" not in self.evidence_boundary:
            raise ValueError("accounting boundary must explicitly contain NOT_RDMA")


@dataclass(frozen=True)
class ByteBreakdown:
    rows: int
    payload_bytes: int
    scale_bytes: int
    descriptor_bytes: int
    padding_bytes: int
    wire_bytes: int

    def __post_init__(self) -> None:
        if min(asdict(self).values()) < 0:
            raise ValueError("byte fields cannot be negative")
        if (
            self.payload_bytes
            + self.scale_bytes
            + self.descriptor_bytes
            + self.padding_bytes
            != self.wire_bytes
        ):
            raise ValueError("wire byte accounting does not close")


def lane_byte_breakdown(rows: int, bits: int, scale_bytes: int, cfg: AccountingConfig) -> ByteBreakdown:
    if rows < 1:
        raise ValueError("lane rows must be positive")
    payload = rows * math.ceil(cfg.hidden_dim * bits / 8)
    scales = rows * scale_bytes
    raw = payload + scales + cfg.lane_descriptor_bytes
    wire = align_up(raw, cfg.lane_alignment_bytes)
    return ByteBreakdown(
        rows=rows,
        payload_bytes=payload,
        scale_bytes=scales,
        descriptor_bytes=cfg.lane_descriptor_bytes,
        padding_bytes=wire - raw,
        wire_bytes=wire,
    )


def high_lane_bytes(rows: int, cfg: AccountingConfig) -> ByteBreakdown:
    return lane_byte_breakdown(rows, cfg.high_bits, cfg.high_scale_bytes_per_row, cfg)


def low_lane_bytes(rows: int, cfg: AccountingConfig) -> ByteBreakdown:
    return lane_byte_breakdown(rows, cfg.low_bits, cfg.low_scale_bytes_per_row, cfg)


def lane_saved_bytes(rows: int, cfg: AccountingConfig) -> int:
    return high_lane_bytes(rows, cfg).wire_bytes - low_lane_bytes(rows, cfg).wire_bytes


def _scaled_phase_us(rows: int, unit_us: float, timing: FormatTiming, cfg: AccountingConfig) -> float:
    if rows <= 0 or unit_us <= 0:
        return 0.0
    if cfg.codec_tax_mode == "amortized_once_per_step_proxy":
        return unit_us
    tiles = math.ceil(rows / cfg.codec_tile_rows)
    return tiles * unit_us * cfg.codec_tile_rows / timing.measured_rows


def lane_codec_phases(rows: int, low: bool, cfg: AccountingConfig) -> tuple[float, float, float]:
    timing = cfg.low_timing if low else cfg.high_timing
    return (
        _scaled_phase_us(rows, timing.pack_us, timing, cfg),
        _scaled_phase_us(rows, timing.h2d_us, timing, cfg),
        _scaled_phase_us(rows, timing.unpack_us, timing, cfg),
    )


@dataclass(frozen=True)
class LaneMatrix:
    """All routed pairs, including local lanes, for one layer/step."""

    lane_counts: Mapping[Lane, int]
    valid_origin_tokens: Mapping[int, int]
    top_k: int
    step: int
    layer: int
    trace_id: str = ""
    stream_id: str = ""
    dropped_pairs_by_receiver: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.step < 0 or self.layer < 0:
            raise ValueError("step/layer must be non-negative")
        for lane, count in self.lane_counts.items():
            if len(lane) != 2 or count < 1:
                raise ValueError("lane counts must contain positive (sender, receiver) entries")
        if any(value < 0 for value in self.valid_origin_tokens.values()):
            raise ValueError("valid_origin_tokens cannot be negative")
        if any(value < 0 for value in self.dropped_pairs_by_receiver.values()):
            raise ValueError("dropped pairs cannot be negative")

    def validate_closure(self, topology: Topology) -> None:
        receiver_rows = {rank: 0 for rank in range(topology.ep_size)}
        for (sender, receiver), count in self.lane_counts.items():
            topology._check_rank(sender)
            topology._check_rank(receiver)
            receiver_rows[receiver] += count
        for receiver in range(topology.ep_size):
            expected = (
                self.top_k * int(self.valid_origin_tokens.get(receiver, 0))
                - int(self.dropped_pairs_by_receiver.get(receiver, 0))
            )
            if receiver_rows[receiver] != expected:
                raise ValueError(
                    f"receiver closure failed for rank {receiver}: "
                    f"rows={receiver_rows[receiver]} expected={expected}"
                )

    def remote_counts(self, topology: Topology) -> dict[Lane, int]:
        self.validate_closure(topology)
        return {
            lane: int(count)
            for lane, count in self.lane_counts.items()
            if topology.remote(*lane)
        }


@dataclass(frozen=True)
class SenderLocalView:
    sender_rank: int
    lane_counts: Mapping[Lane, int]
    step: int
    layer: int

    def __post_init__(self) -> None:
        if any(sender != self.sender_rank for sender, _ in self.lane_counts):
            raise ValueError("sender-local view leaked another sender")


@dataclass(frozen=True)
class ReceiverResourceView:
    resource: str
    receiver_ranks: tuple[int, ...]
    lane_counts: Mapping[Lane, int]
    step: int
    layer: int

    def __post_init__(self) -> None:
        allowed = set(self.receiver_ranks)
        if any(receiver not in allowed for _, receiver in self.lane_counts):
            raise ValueError("receiver-resource view leaked another receive resource")


def make_sender_local_views(matrix: LaneMatrix, topology: Topology) -> dict[int, SenderLocalView]:
    remote = matrix.remote_counts(topology)
    views: dict[int, SenderLocalView] = {}
    for sender in range(topology.ep_size):
        local_counts = {lane: count for lane, count in remote.items() if lane[0] == sender}
        views[sender] = SenderLocalView(sender, local_counts, matrix.step, matrix.layer)
    return views


def make_receiver_resource_views(
    matrix: LaneMatrix, topology: Topology
) -> dict[str, ReceiverResourceView]:
    remote = matrix.remote_counts(topology)
    ranks_by_resource: dict[str, list[int]] = {}
    for rank, resource in enumerate(topology.receive_resource_by_rank):
        ranks_by_resource.setdefault(resource, []).append(rank)
    return {
        resource: ReceiverResourceView(
            resource=resource,
            receiver_ranks=tuple(ranks),
            lane_counts={
                lane: count for lane, count in remote.items() if lane[1] in set(ranks)
            },
            step=matrix.step,
            layer=matrix.layer,
        )
        for resource, ranks in ranks_by_resource.items()
    }


def positive_net_gate(
    wire_saving_us: float,
    incremental_codec_us: float,
    allocated_credit_us: float,
) -> bool:
    """Strict positive gate; equality is rejected."""

    if min(wire_saving_us, incremental_codec_us, allocated_credit_us) < 0:
        raise ValueError("hard-gate terms cannot be negative")
    return wire_saving_us > incremental_codec_us + allocated_credit_us


def _lane_gate(lane: Lane, rows: int, cfg: AccountingConfig, topology: Topology, credit_us: float) -> bool:
    high = high_lane_bytes(rows, cfg)
    low = low_lane_bytes(rows, cfg)
    bytes_per_us = topology.link_gbps * 1e9 / 8.0 / 1e6
    wire_saving_us = (high.wire_bytes - low.wire_bytes) / bytes_per_us
    high_phases = lane_codec_phases(rows, False, cfg)
    low_phases = lane_codec_phases(rows, True, cfg)
    incremental_codec = max(0.0, sum(low_phases) - sum(high_phases))
    return positive_net_gate(wire_saving_us, incremental_codec, credit_us)


def _minimal_lanes_to_threshold(
    lane_counts: Mapping[Lane, int], threshold_bytes: int, cfg: AccountingConfig
) -> tuple[Lane, ...]:
    current = sum(high_lane_bytes(rows, cfg).wire_bytes for rows in lane_counts.values())
    if current <= threshold_bytes:
        return ()
    ordered = sorted(
        lane_counts,
        key=lambda lane: (-lane_saved_bytes(lane_counts[lane], cfg), lane[0], lane[1]),
    )
    selected: list[Lane] = []
    for lane in ordered:
        current -= lane_saved_bytes(lane_counts[lane], cfg)
        selected.append(lane)
        if current <= threshold_bytes:
            break
    return tuple(selected)


@dataclass
class PolicyState:
    committed_low_lanes: tuple[Lane, ...] = ()
    action_epoch: int = 0

    def snapshot(self) -> tuple[tuple[Lane, ...], int]:
        return self.committed_low_lanes, self.action_epoch

    def restore(self, snapshot: tuple[tuple[Lane, ...], int]) -> None:
        self.committed_low_lanes, self.action_epoch = snapshot

    def commit(self, lanes: Iterable[Lane]) -> None:
        selected = tuple(sorted(set(lanes)))
        self.committed_low_lanes = selected
        self.action_epoch += 1


@dataclass(frozen=True)
class CreditRecord:
    sender_rank: int
    receiver_rank: int
    precision_code: int = LOW_PRECISION_CODE
    flags: int = 0

    def encode(self) -> bytes:
        if not 0 <= self.sender_rank <= 0xFFFF or not 0 <= self.receiver_rank <= 0xFFFF:
            raise ValueError("credit rank exceeds uint16")
        if not 0 <= self.precision_code <= 0xFF or not 0 <= self.flags <= 0xFF:
            raise ValueError("credit code/flags exceed uint8")
        return CREDIT_RECORD.pack(
            self.sender_rank,
            self.receiver_rank,
            self.precision_code,
            self.flags,
            0,
        )


@dataclass(frozen=True)
class CreditEnvelope:
    source_resource: str
    target_sender: int
    step: int
    layer: int
    payload: bytes
    arrival_us: float
    deadline_us: float

    @property
    def on_time(self) -> bool:
        return self.arrival_us <= self.deadline_us


def encode_credit_message(records: Sequence[CreditRecord], step: int, layer: int, alignment: int) -> bytes:
    if not records:
        raise ValueError("credit message requires at least one record")
    if len(records) > 0xFFFF or step > 0xFFFFFFFF or layer > 0xFFFF:
        raise ValueError("credit message field overflow")
    header = CREDIT_HEADER.pack(
        CREDIT_MAGIC,
        CREDIT_VERSION,
        0,
        len(records),
        step,
        layer,
        0,
    )
    body = b"".join(record.encode() for record in records)
    raw = header + body
    return raw + bytes(align_up(len(raw), alignment) - len(raw))


def decode_credit_message(payload: bytes, alignment: int) -> tuple[int, int, tuple[CreditRecord, ...]]:
    if len(payload) < CREDIT_HEADER.size or len(payload) % alignment:
        raise ValueError("malformed credit length/alignment")
    magic, version, _flags, count, step, layer, _reserved = CREDIT_HEADER.unpack_from(payload)
    if magic != CREDIT_MAGIC or version != CREDIT_VERSION:
        raise ValueError("malformed credit magic/version")
    required = CREDIT_HEADER.size + count * CREDIT_RECORD.size
    if required > len(payload):
        raise ValueError("truncated credit records")
    if any(payload[required:]):
        raise ValueError("non-zero credit padding")
    records: list[CreditRecord] = []
    offset = CREDIT_HEADER.size
    for _ in range(count):
        sender, receiver, precision, flags, reserved = CREDIT_RECORD.unpack_from(payload, offset)
        if reserved != 0 or precision != LOW_PRECISION_CODE:
            raise ValueError("malformed credit record")
        records.append(CreditRecord(sender, receiver, precision, flags))
        offset += CREDIT_RECORD.size
    return step, layer, tuple(records)


@dataclass(frozen=True)
class ActionPlan:
    arm: str
    information_set: str
    low_lanes: tuple[Lane, ...]
    requested_lanes: tuple[Lane, ...] = ()
    blocked_lanes: tuple[Lane, ...] = ()
    fallback_reason: str = ""
    credit_bytes: int = 0
    credit_total_us: float = 0.0
    credit_visible_us: float = 0.0
    state_committed: bool = False


def plan_sender_local(
    view: SenderLocalView,
    *,
    threshold_bytes: int,
    cfg: AccountingConfig,
    topology: Topology,
) -> ActionPlan:
    """Deployable same-layer baseline.  No global matrix argument exists."""

    requested = _minimal_lanes_to_threshold(view.lane_counts, threshold_bytes, cfg)
    accepted: list[Lane] = []
    blocked: list[Lane] = []
    for lane in requested:
        if _lane_gate(lane, view.lane_counts[lane], cfg, topology, 0.0):
            accepted.append(lane)
        else:
            blocked.append(lane)
    return ActionPlan(
        arm="sender_local_exact_handle",
        information_set=INFORMATION_SENDER_LOCAL,
        low_lanes=tuple(sorted(accepted)),
        requested_lanes=tuple(sorted(requested)),
        blocked_lanes=tuple(sorted(blocked)),
    )


def build_receiver_credit_messages(
    view: ReceiverResourceView,
    *,
    threshold_bytes: int,
    cfg: AccountingConfig,
) -> tuple[tuple[CreditEnvelope, ...], tuple[Lane, ...]]:
    """Build DDRC-v0 messages using only one receive resource's current routes."""

    selected = _minimal_lanes_to_threshold(view.lane_counts, threshold_bytes, cfg)
    by_sender: dict[int, list[CreditRecord]] = {}
    for sender, receiver in selected:
        by_sender.setdefault(sender, []).append(CreditRecord(sender, receiver))
    envelopes: list[CreditEnvelope] = []
    for sender in sorted(by_sender):
        records = tuple(sorted(by_sender[sender], key=lambda record: record.receiver_rank))
        payload = encode_credit_message(
            records,
            view.step,
            view.layer,
            cfg.credit_alignment_bytes,
        )
        envelopes.append(
            CreditEnvelope(
                source_resource=view.resource,
                target_sender=sender,
                step=view.step,
                layer=view.layer,
                payload=payload,
                arrival_us=cfg.credit_timing.total_us,
                deadline_us=cfg.credit_timing.pack_deadline_slack_us,
            )
        )
    return tuple(envelopes), tuple(sorted(selected))


def apply_receiver_credit_messages(
    view: SenderLocalView,
    messages: Sequence[CreditEnvelope],
    *,
    cfg: AccountingConfig,
    topology: Topology,
    state: PolicyState,
) -> ActionPlan:
    """Decode explicit credit at one sender and apply local hard gates.

    A malformed, duplicate, wrong-step, or late message causes a full FP8
    fallback for this sender.  Credit bytes/time remain charged in the plan.
    """

    snapshot = state.snapshot()
    credit_bytes = sum(len(message.payload) for message in messages)
    credit_total = max((message.arrival_us for message in messages), default=0.0)
    credit_visible = (
        max(0.0, credit_total - cfg.credit_timing.pack_deadline_slack_us)
        if cfg.credit_timing.overlap_proven
        else credit_total
    )
    requested: list[Lane] = []
    seen: set[Lane] = set()
    fallback = ""
    try:
        for message in messages:
            if message.target_sender != view.sender_rank:
                raise ValueError("credit delivered to wrong sender")
            if message.step != view.step or message.layer != view.layer:
                raise ValueError("credit envelope step/layer mismatch")
            if not message.on_time:
                raise TimeoutError("late_credit")
            step, layer, records = decode_credit_message(
                message.payload, cfg.credit_alignment_bytes
            )
            if step != view.step or layer != view.layer:
                raise ValueError("credit payload step/layer mismatch")
            for record in records:
                lane = (record.sender_rank, record.receiver_rank)
                if record.sender_rank != view.sender_rank or lane not in view.lane_counts:
                    raise ValueError("credit references inactive/wrong sender lane")
                if lane in seen:
                    raise ValueError("duplicate_credit")
                seen.add(lane)
                requested.append(lane)
    except TimeoutError:
        fallback = "late_credit"
    except (ValueError, struct.error):
        fallback = "malformed_or_duplicate_credit"

    if fallback:
        state.restore(snapshot)
        return ActionPlan(
            arm="DDRC",
            information_set=INFORMATION_RECEIVER_CREDIT,
            low_lanes=(),
            requested_lanes=tuple(sorted(seen)),
            fallback_reason=fallback,
            credit_bytes=credit_bytes,
            credit_total_us=credit_total,
            credit_visible_us=credit_visible,
            state_committed=False,
        )

    credit_share = credit_visible / len(requested) if requested else 0.0
    accepted: list[Lane] = []
    blocked: list[Lane] = []
    for lane in sorted(requested):
        if _lane_gate(lane, view.lane_counts[lane], cfg, topology, credit_share):
            accepted.append(lane)
        else:
            blocked.append(lane)
    if accepted:
        state.commit(accepted)
    else:
        state.restore(snapshot)
    return ActionPlan(
        arm="DDRC",
        information_set=INFORMATION_RECEIVER_CREDIT,
        low_lanes=tuple(accepted),
        requested_lanes=tuple(sorted(requested)),
        blocked_lanes=tuple(blocked),
        fallback_reason="hardgate_reject" if requested and not accepted else "",
        credit_bytes=credit_bytes,
        credit_total_us=credit_total,
        credit_visible_us=credit_visible,
        state_committed=bool(accepted),
    )


def plan_ddrc(
    matrix: LaneMatrix,
    *,
    receiver_threshold_bytes: Mapping[str, int],
    cfg: AccountingConfig,
    topology: Topology,
    states: Mapping[int, PolicyState] | None = None,
) -> ActionPlan:
    """Orchestrate explicit receiver messages without passing global data to senders."""

    sender_views = make_sender_local_views(matrix, topology)
    receiver_views = make_receiver_resource_views(matrix, topology)
    all_messages: list[CreditEnvelope] = []
    all_requested: set[Lane] = set()
    for resource, view in sorted(receiver_views.items()):
        threshold = receiver_threshold_bytes.get(resource)
        if threshold is None:
            threshold = 2**63 - 1
        messages, requested = build_receiver_credit_messages(
            view,
            threshold_bytes=int(threshold),
            cfg=cfg,
        )
        all_messages.extend(messages)
        all_requested.update(requested)

    credit_bytes_by_source: dict[str, int] = {}
    credit_bytes_by_sender: dict[int, int] = {}
    for message in all_messages:
        credit_bytes_by_source[message.source_resource] = (
            credit_bytes_by_source.get(message.source_resource, 0) + len(message.payload)
        )
        credit_bytes_by_sender[message.target_sender] = (
            credit_bytes_by_sender.get(message.target_sender, 0) + len(message.payload)
        )
    critical_credit_bytes = max(
        max(credit_bytes_by_source.values(), default=0),
        max(credit_bytes_by_sender.values(), default=0),
    )
    bytes_per_us = topology.link_gbps * 1e9 / 8.0 / 1e6
    credit_wire_us = critical_credit_bytes / bytes_per_us
    credit_total_us = (
        cfg.credit_timing.build_us
        + cfg.credit_timing.aggregate_us
        + cfg.credit_timing.transfer_us
        + credit_wire_us
        + cfg.credit_timing.parse_us
        if all_messages
        else 0.0
    )
    all_messages = [
        replace(
            message,
            arrival_us=credit_total_us,
            deadline_us=cfg.credit_timing.pack_deadline_slack_us,
        )
        for message in all_messages
    ]
    credit_visible_us = (
        max(0.0, credit_total_us - cfg.credit_timing.pack_deadline_slack_us)
        if cfg.credit_timing.overlap_proven
        else credit_total_us
    )

    states_by_sender = states if states is not None else {
        sender: PolicyState() for sender in range(topology.ep_size)
    }
    low: set[Lane] = set()
    blocked: set[Lane] = set()
    fallbacks: list[str] = []
    committed = False
    for sender, view in sender_views.items():
        messages = [message for message in all_messages if message.target_sender == sender]
        sender_plan = apply_receiver_credit_messages(
            view,
            messages,
            cfg=cfg,
            topology=topology,
            state=states_by_sender[sender],
        )
        low.update(sender_plan.low_lanes)
        blocked.update(sender_plan.blocked_lanes)
        if sender_plan.fallback_reason:
            fallbacks.append(sender_plan.fallback_reason)
        committed = committed or sender_plan.state_committed
    return ActionPlan(
        arm="DDRC",
        information_set=INFORMATION_RECEIVER_CREDIT,
        low_lanes=tuple(sorted(low)),
        requested_lanes=tuple(sorted(all_requested)),
        blocked_lanes=tuple(sorted(blocked)),
        fallback_reason=";".join(sorted(set(fallbacks))),
        credit_bytes=sum(len(message.payload) for message in all_messages),
        credit_total_us=credit_total_us,
        credit_visible_us=credit_visible_us,
        state_committed=committed,
    )


def plan_causal_prev_step(
    current: LaneMatrix,
    previous: LaneMatrix | None,
    *,
    receiver_threshold_bytes: Mapping[str, int],
    cfg: AccountingConfig,
    topology: Topology,
) -> ActionPlan:
    current_remote = current.remote_counts(topology)
    if previous is None:
        return ActionPlan("causal_prev_step", INFORMATION_CAUSAL_HISTORY, ())
    if current.stream_id != previous.stream_id:
        raise ValueError("causal_prev_step cannot cross stream_id")
    if current.layer != previous.layer:
        raise ValueError("causal_prev_step requires the same layer")
    if previous.step >= current.step:
        raise ValueError("causal_prev_step requires a strictly earlier step")
    previous_views = make_receiver_resource_views(previous, topology)
    selected: set[Lane] = set()
    for resource, prior_view in previous_views.items():
        prior_total = sum(high_lane_bytes(rows, cfg).wire_bytes for rows in prior_view.lane_counts.values())
        threshold = receiver_threshold_bytes.get(resource)
        if threshold is None:
            threshold = 2**63 - 1
        if prior_total <= int(threshold):
            continue
        for lane in current_remote:
            if topology.receive_resource(lane[1]) == resource:
                selected.add(lane)
    accepted = tuple(
        sorted(
            lane for lane in selected
            if _lane_gate(lane, current_remote[lane], cfg, topology, 0.0)
        )
    )
    return ActionPlan(
        "causal_prev_step",
        INFORMATION_CAUSAL_HISTORY,
        accepted,
        tuple(sorted(selected)),
    )


def plan_calib_static(
    matrix: LaneMatrix,
    *,
    static_low_lanes: Iterable[Lane],
    cfg: AccountingConfig,
    topology: Topology,
) -> ActionPlan:
    remote = matrix.remote_counts(topology)
    requested = tuple(sorted(set(static_low_lanes) & set(remote)))
    accepted = tuple(
        lane for lane in requested if _lane_gate(lane, remote[lane], cfg, topology, 0.0)
    )
    return ActionPlan(
        "calib_static",
        INFORMATION_SENDER_LOCAL,
        accepted,
        requested,
        tuple(lane for lane in requested if lane not in set(accepted)),
    )


@dataclass(frozen=True)
class StepLedger:
    arm: str
    information_set: str
    low_lanes: tuple[Lane, ...]
    requested_lanes: tuple[Lane, ...]
    blocked_lanes: tuple[Lane, ...]
    fallback_reason: str
    remote_pairs: int
    low_pairs: int
    payload_bytes: int
    scale_bytes: int
    descriptor_bytes: int
    padding_bytes: int
    wire_bytes: int
    critical_wire_bytes: int
    wire_us: float
    pack_us: float
    h2d_us: float
    unpack_us: float
    codec_us: float
    credit_bytes: int
    credit_total_us: float
    credit_visible_us: float
    total_us: float
    baseline_total_us: float
    net_saving_fraction: float
    state_committed: bool
    evidence_boundary: str

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["low_lanes"] = ";".join(f"{s}:{r}" for s, r in self.low_lanes)
        row["requested_lanes"] = ";".join(f"{s}:{r}" for s, r in self.requested_lanes)
        row["blocked_lanes"] = ";".join(f"{s}:{r}" for s, r in self.blocked_lanes)
        return row


def _account_phases(
    remote: Mapping[Lane, int],
    low_lanes: set[Lane],
    cfg: AccountingConfig,
    topology: Topology,
) -> tuple[dict[str, int], dict[int, int], dict[str, int], float, float, float]:
    totals = {"payload": 0, "scale": 0, "descriptor": 0, "padding": 0, "wire": 0}
    sender_wire = {rank: 0 for rank in range(topology.ep_size)}
    receiver_wire: dict[str, int] = {}
    sender_pack = {rank: 0.0 for rank in range(topology.ep_size)}
    receiver_h2d: dict[str, float] = {}
    receiver_unpack: dict[str, float] = {}
    formats_seen: set[bool] = set()
    for lane, rows in remote.items():
        low = lane in low_lanes
        breakdown = low_lane_bytes(rows, cfg) if low else high_lane_bytes(rows, cfg)
        totals["payload"] += breakdown.payload_bytes
        totals["scale"] += breakdown.scale_bytes
        totals["descriptor"] += breakdown.descriptor_bytes
        totals["padding"] += breakdown.padding_bytes
        totals["wire"] += breakdown.wire_bytes
        sender, receiver = lane
        resource = topology.receive_resource(receiver)
        sender_wire[sender] += breakdown.wire_bytes
        receiver_wire[resource] = receiver_wire.get(resource, 0) + breakdown.wire_bytes
        pack, h2d, unpack = lane_codec_phases(rows, low, cfg)
        if cfg.codec_tax_mode == "amortized_once_per_step_proxy" and low in formats_seen:
            pack = h2d = unpack = 0.0
        formats_seen.add(low)
        sender_pack[sender] += pack
        receiver_h2d[resource] = receiver_h2d.get(resource, 0.0) + h2d
        receiver_unpack[resource] = receiver_unpack.get(resource, 0.0) + unpack
    critical_wire = max(
        max(sender_wire.values(), default=0),
        max(receiver_wire.values(), default=0),
    )
    return (
        totals,
        sender_wire,
        receiver_wire,
        max(sender_pack.values(), default=0.0),
        max(receiver_h2d.values(), default=0.0),
        max(receiver_unpack.values(), default=0.0),
    )


def account_step(
    matrix: LaneMatrix,
    plan: ActionPlan,
    *,
    cfg: AccountingConfig,
    topology: Topology,
) -> StepLedger:
    remote = matrix.remote_counts(topology)
    unknown = set(plan.low_lanes) - set(remote)
    if unknown:
        raise ValueError(f"plan selected inactive/non-remote lanes: {sorted(unknown)}")
    low_set = set(plan.low_lanes)
    totals, sender_wire, receiver_wire, pack_us, h2d_us, unpack_us = _account_phases(
        remote, low_set, cfg, topology
    )
    bytes_per_us = topology.link_gbps * 1e9 / 8.0 / 1e6
    critical_wire = max(
        max(sender_wire.values(), default=0),
        max(receiver_wire.values(), default=0),
    )
    wire_us = critical_wire / bytes_per_us
    codec_us = pack_us + h2d_us + unpack_us
    total_us = codec_us + wire_us + plan.credit_visible_us

    baseline_plan = ActionPlan("uniform_full", INFORMATION_SENDER_LOCAL, ())
    if plan.arm == "uniform_full":
        baseline_total = total_us
    else:
        baseline = account_step(matrix, baseline_plan, cfg=cfg, topology=topology)
        baseline_total = baseline.total_us
    saving = (baseline_total - total_us) / baseline_total if baseline_total > 0 else 0.0
    return StepLedger(
        arm=plan.arm,
        information_set=plan.information_set,
        low_lanes=plan.low_lanes,
        requested_lanes=plan.requested_lanes,
        blocked_lanes=plan.blocked_lanes,
        fallback_reason=plan.fallback_reason,
        remote_pairs=sum(remote.values()),
        low_pairs=sum(remote[lane] for lane in low_set),
        payload_bytes=totals["payload"],
        scale_bytes=totals["scale"],
        descriptor_bytes=totals["descriptor"],
        padding_bytes=totals["padding"],
        wire_bytes=totals["wire"],
        critical_wire_bytes=critical_wire,
        wire_us=wire_us,
        pack_us=pack_us,
        h2d_us=h2d_us,
        unpack_us=unpack_us,
        codec_us=codec_us,
        credit_bytes=plan.credit_bytes,
        credit_total_us=plan.credit_total_us,
        credit_visible_us=plan.credit_visible_us,
        total_us=total_us,
        baseline_total_us=baseline_total,
        net_saving_fraction=saving,
        state_committed=plan.state_committed,
        evidence_boundary=cfg.evidence_boundary,
    )


def plan_global_oracle(
    matrix: LaneMatrix,
    *,
    cfg: AccountingConfig,
    topology: Topology,
    deployable_seed_plans: Sequence[ActionPlan] = (),
) -> ActionPlan:
    """Exact full-matrix oracle for the frozen serialized-tile ledger.

    The objective is linearized with four max variables: sender pack,
    receive-resource H2D, receive-resource unpack, and the shared sender/receive
    wire maximum.  Every active remote lane has one binary high/low variable.
    This is deliberately isolated from deployable policy code.
    """

    if cfg.codec_tax_mode != "serialized_tiles":
        raise ValueError("exact global oracle is defined only for serialized_tiles")
    remote = matrix.remote_counts(topology)
    lanes = tuple(sorted(remote))
    if not lanes:
        return ActionPlan(
            arm="global_full_matrix_oracle",
            information_set=INFORMATION_GLOBAL_ORACLE,
            low_lanes=(),
        )
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:  # pragma: no cover - formal environment guard
        raise RuntimeError("exact global oracle requires scipy.optimize.milp") from exc

    n_lane = len(lanes)
    z_pack = n_lane
    z_h2d = n_lane + 1
    z_unpack = n_lane + 2
    z_wire = n_lane + 3
    n_var = n_lane + 4
    objective = np.zeros(n_var, dtype=np.float64)
    objective[[z_pack, z_h2d, z_unpack, z_wire]] = 1.0
    integrality = np.zeros(n_var, dtype=np.int32)
    integrality[:n_lane] = 1
    lower = np.zeros(n_var, dtype=np.float64)
    upper = np.full(n_var, np.inf, dtype=np.float64)
    upper[:n_lane] = 1.0

    high_wire: dict[Lane, float] = {}
    low_wire: dict[Lane, float] = {}
    high_pack: dict[Lane, float] = {}
    low_pack: dict[Lane, float] = {}
    high_h2d: dict[Lane, float] = {}
    low_h2d: dict[Lane, float] = {}
    high_unpack: dict[Lane, float] = {}
    low_unpack: dict[Lane, float] = {}
    bytes_per_us = topology.link_gbps * 1e9 / 8.0 / 1e6
    for lane, rows in remote.items():
        high_wire[lane] = float(high_lane_bytes(rows, cfg).wire_bytes) / bytes_per_us
        low_wire[lane] = float(low_lane_bytes(rows, cfg).wire_bytes) / bytes_per_us
        high_pack[lane], high_h2d[lane], high_unpack[lane] = lane_codec_phases(rows, False, cfg)
        low_pack[lane], low_h2d[lane], low_unpack[lane] = lane_codec_phases(rows, True, cfg)

    constraint_rows: list[np.ndarray] = []
    constraint_upper: list[float] = []

    def add_max_constraint(
        member_lanes: Iterable[Lane],
        high: Mapping[Lane, float],
        low: Mapping[Lane, float],
        z_index: int,
    ) -> None:
        members = tuple(member_lanes)
        row = np.zeros(n_var, dtype=np.float64)
        base = 0.0
        lane_to_index = {lane: index for index, lane in enumerate(lanes)}
        for lane in members:
            index = lane_to_index[lane]
            base += high[lane]
            row[index] = low[lane] - high[lane]
        row[z_index] = -1.0
        constraint_rows.append(row)
        constraint_upper.append(-base)

    for sender in range(topology.ep_size):
        members = (lane for lane in lanes if lane[0] == sender)
        add_max_constraint(members, high_pack, low_pack, z_pack)
        add_max_constraint(
            (lane for lane in lanes if lane[0] == sender), high_wire, low_wire, z_wire
        )
    for resource in sorted(set(topology.receive_resource_by_rank)):
        members = tuple(
            lane for lane in lanes if topology.receive_resource(lane[1]) == resource
        )
        add_max_constraint(members, high_h2d, low_h2d, z_h2d)
        add_max_constraint(members, high_unpack, low_unpack, z_unpack)
        add_max_constraint(members, high_wire, low_wire, z_wire)

    matrix_a = np.vstack(constraint_rows)
    constraints = LinearConstraint(
        matrix_a,
        lb=np.full(len(constraint_rows), -np.inf, dtype=np.float64),
        ub=np.asarray(constraint_upper, dtype=np.float64),
    )
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"global oracle MILP failed: {result.message}")
    selected = tuple(lane for index, lane in enumerate(lanes) if result.x[index] >= 0.5)
    oracle = ActionPlan(
        arm="global_full_matrix_oracle",
        information_set=INFORMATION_GLOBAL_ORACLE,
        low_lanes=selected,
    )
    oracle_value = account_step(matrix, oracle, cfg=cfg, topology=topology).total_us
    for seed in deployable_seed_plans:
        zero_credit_seed = ActionPlan(
            arm="oracle_seed_check",
            information_set=INFORMATION_GLOBAL_ORACLE,
            low_lanes=tuple(sorted(set(seed.low_lanes) & set(lanes))),
        )
        seed_value = account_step(matrix, zero_credit_seed, cfg=cfg, topology=topology).total_us
        if oracle_value > seed_value + 1e-7:
            raise RuntimeError("exact oracle is worse than a deployable zero-credit seed")
    return oracle


def plan_uniform(arm: str, matrix: LaneMatrix, topology: Topology) -> ActionPlan:
    remote = matrix.remote_counts(topology)
    if arm == "uniform_full":
        return ActionPlan(arm, INFORMATION_SENDER_LOCAL, ())
    if arm == "uniform_low":
        return ActionPlan(arm, INFORMATION_SENDER_LOCAL, tuple(sorted(remote)))
    raise ValueError(f"unsupported uniform arm: {arm}")


def combine_sender_local_plans(plans: Iterable[ActionPlan]) -> ActionPlan:
    plans = tuple(plans)
    if any(plan.information_set != INFORMATION_SENDER_LOCAL for plan in plans):
        raise ValueError("cannot combine non-sender-local plan")
    return ActionPlan(
        arm="sender_local_exact_handle",
        information_set=INFORMATION_SENDER_LOCAL,
        low_lanes=tuple(sorted(set(itertools.chain.from_iterable(plan.low_lanes for plan in plans)))),
        requested_lanes=tuple(
            sorted(set(itertools.chain.from_iterable(plan.requested_lanes for plan in plans)))
        ),
        blocked_lanes=tuple(
            sorted(set(itertools.chain.from_iterable(plan.blocked_lanes for plan in plans)))
        ),
    )


def clone_state_map(states: Mapping[int, PolicyState]) -> dict[int, PolicyState]:
    return {rank: copy.deepcopy(state) for rank, state in states.items()}
