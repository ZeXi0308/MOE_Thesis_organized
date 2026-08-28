"""The single frozen RIC-v1 contract codec and fail-closed apply path."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable, Mapping, Sequence

from .schema import JoinIdentity, RICValidationError


MAGIC = b"RIC1"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sBBBBII")
RECORD_STRUCT = struct.Struct("<QHHHBB")
HEADER_BYTES = 16
RECORD_BYTES = 16
ALIGNMENT_BYTES = 16

if HEADER_STRUCT.size != HEADER_BYTES or RECORD_STRUCT.size != RECORD_BYTES:
    raise AssertionError("RIC frozen wire struct size drift")


class WireProtocolError(RICValidationError):
    """Malformed or semantically invalid RIC wire input."""


def _u(name: str, value: int, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise WireProtocolError(f"{name} does not fit its frozen unsigned field")


@dataclass(frozen=True)
class ContractRecord:
    join_key_hash64: int
    layer_id: int
    missing_slot_mask: int
    identity_tag16: int
    slack_bucket: int
    flags: int = 0

    def __post_init__(self) -> None:
        _u("join_key_hash64", self.join_key_hash64, 0xFFFFFFFFFFFFFFFF)
        _u("layer_id", self.layer_id, 0xFFFF)
        _u("missing_slot_mask", self.missing_slot_mask, 0xFFFF)
        if self.missing_slot_mask == 0:
            raise WireProtocolError("contract cannot advertise an already-closed join")
        _u("identity_tag16", self.identity_tag16, 0xFFFF)
        _u("slack_bucket", self.slack_bucket, 3)
        _u("flags", self.flags, 0xFF)
        if self.flags != 0:
            raise WireProtocolError("contract contains unknown flags")

    @property
    def missing_count(self) -> int:
        return bin(self.missing_slot_mask).count("1")


@dataclass(frozen=True)
class ContractMessage:
    sender_rank: int
    receiver_rank: int
    epoch: int
    sequence: int
    records: tuple[ContractRecord, ...]

    def __post_init__(self) -> None:
        _u("sender_rank", self.sender_rank, 0xFF)
        _u("receiver_rank", self.receiver_rank, 0xFF)
        _u("epoch", self.epoch, 0xFFFFFFFF)
        _u("sequence", self.sequence, 0xFFFFFFFF)
        if self.epoch == 0 or self.sequence == 0:
            raise WireProtocolError("epoch and sequence are one-based")
        if type(self.records) is not tuple or not 1 <= len(self.records) <= 0xFF:
            raise WireProtocolError("contract message requires 1..255 immutable records")
        if any(type(record) is not ContractRecord for record in self.records):
            raise WireProtocolError("contract message contains a non-ContractRecord")


def encoded_contract_bytes(record_count: int) -> int:
    if type(record_count) is not int or not 1 <= record_count <= 0xFF:
        raise WireProtocolError("record_count must be in [1, 255]")
    size = HEADER_BYTES + RECORD_BYTES * record_count
    if size % ALIGNMENT_BYTES:
        raise AssertionError("frozen RIC message unexpectedly needs extra alignment")
    return size


def encode_contract(message: ContractMessage) -> bytes:
    """Encode the only formal RIC wire representation."""

    if type(message) is not ContractMessage:
        raise WireProtocolError("encode_contract requires ContractMessage")
    payload = bytearray(
        HEADER_STRUCT.pack(
            MAGIC,
            VERSION,
            message.sender_rank,
            message.receiver_rank,
            len(message.records),
            message.epoch,
            message.sequence,
        )
    )
    for record in message.records:
        payload.extend(
            RECORD_STRUCT.pack(
                record.join_key_hash64,
                record.layer_id,
                record.missing_slot_mask,
                record.identity_tag16,
                record.slack_bucket,
                record.flags,
            )
        )
    if len(payload) != encoded_contract_bytes(len(message.records)):
        raise AssertionError("RIC wire byte accounting drift")
    return bytes(payload)


def decode_contract(payload: bytes) -> ContractMessage:
    """Decode and validate exact length, magic, version, and field widths."""

    if type(payload) is not bytes:
        raise WireProtocolError("wire payload must be immutable bytes")
    if len(payload) < HEADER_BYTES or len(payload) % ALIGNMENT_BYTES:
        raise WireProtocolError("malformed contract length/alignment")
    magic, version, sender, receiver, count, epoch, sequence = HEADER_STRUCT.unpack_from(payload)
    if magic != MAGIC:
        raise WireProtocolError("contract magic mismatch")
    if version != VERSION:
        raise WireProtocolError("unsupported contract version")
    if count == 0 or len(payload) != encoded_contract_bytes(count):
        raise WireProtocolError("contract record_count/length mismatch")
    records = []
    offset = HEADER_BYTES
    for _ in range(count):
        values = RECORD_STRUCT.unpack_from(payload, offset)
        records.append(ContractRecord(*values))
        offset += RECORD_BYTES
    return ContractMessage(
        sender_rank=sender,
        receiver_rank=receiver,
        epoch=epoch,
        sequence=sequence,
        records=tuple(records),
    )


def join_identity_hash_parts(identity: JoinIdentity) -> tuple[int, int]:
    """Return SHA-256 low 64 bits and the next 16-bit identity tag.

    Integers use little-endian interpretation to match the frozen little-endian
    wire layout.  The full identity remains in ``IdentityTable`` for collision
    checking; these truncated values are never treated as identity themselves.
    """

    if type(identity) is not JoinIdentity:
        raise WireProtocolError("hashing requires a full JoinIdentity")
    import hashlib

    digest = hashlib.sha256(identity.canonical_bytes()).digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:10], "little")


@dataclass(frozen=True)
class IdentityBinding:
    join_identity: JoinIdentity
    join_key_hash64: int
    identity_tag16: int

    def __post_init__(self) -> None:
        if type(self.join_identity) is not JoinIdentity:
            raise WireProtocolError("identity binding requires JoinIdentity")
        _u("join_key_hash64", self.join_key_hash64, 0xFFFFFFFFFFFFFFFF)
        _u("identity_tag16", self.identity_tag16, 0xFFFF)

    @classmethod
    def from_join(cls, identity: JoinIdentity) -> "IdentityBinding":
        hash64, tag16 = join_identity_hash_parts(identity)
        return cls(identity, hash64, tag16)


class IdentityTable:
    """Full-identity table; ambiguous truncated keys fail at lookup time."""

    def __init__(self, bindings: Iterable[IdentityBinding], *, top_k: int = 16) -> None:
        if type(top_k) is not int or not 1 <= top_k <= 16:
            raise WireProtocolError("identity table top_k must be in [1, 16]")
        by_key: dict[tuple[int, int, int, int], list[JoinIdentity]] = {}
        known_epochs: set[tuple[int, int]] = set()
        for binding in bindings:
            if type(binding) is not IdentityBinding:
                raise WireProtocolError("identity table contains an invalid binding")
            join = binding.join_identity
            key = (binding.join_key_hash64, binding.identity_tag16, join.layer_id, join.epoch)
            bucket = by_key.setdefault(key, [])
            if join not in bucket:
                bucket.append(join)
            known_epochs.add((join.receiver_rank, join.epoch))
        if not by_key:
            raise WireProtocolError("identity table cannot be empty")
        self._by_key = {key: tuple(values) for key, values in by_key.items()}
        self._known_epochs = frozenset(known_epochs)
        self.top_k = top_k

    @classmethod
    def from_joins(
        cls, identities: Iterable[JoinIdentity], *, top_k: int = 16
    ) -> "IdentityTable":
        unique = tuple(dict.fromkeys(identities))
        return cls(
            (IdentityBinding.from_join(identity) for identity in unique), top_k=top_k
        )

    def knows_epoch(self, receiver_rank: int, epoch: int) -> bool:
        return (receiver_rank, epoch) in self._known_epochs

    def resolve(
        self, record: ContractRecord, *, receiver_rank: int, epoch: int
    ) -> JoinIdentity:
        if record.missing_slot_mask & ~((1 << self.top_k) - 1):
            raise WireProtocolError("contract missing-slot mask exceeds model top_k")
        key = (record.join_key_hash64, record.identity_tag16, record.layer_id, epoch)
        candidates = tuple(
            identity
            for identity in self._by_key.get(key, ())
            if identity.receiver_rank == receiver_rank
        )
        if not candidates:
            raise WireProtocolError("unknown contract identity")
        if len(candidates) != 1:
            raise WireProtocolError("ambiguous contract identity collision")
        identity = candidates[0]
        # A single manually corrupted binding must not silently alias a full
        # identity.  Collision fixtures with multiple candidates fail above.
        if join_identity_hash_parts(identity) != (
            record.join_key_hash64,
            record.identity_tag16,
        ):
            raise WireProtocolError("contract identity binding/hash mismatch")
        if identity.layer_id != record.layer_id or identity.epoch != epoch:
            raise WireProtocolError("contract identity layer/epoch mismatch")
        return identity


@dataclass(frozen=True)
class ContractTax:
    state_build_us: float = 0.0
    hash_us: float = 0.0
    encode_us: float = 0.0
    transfer_us: float = 0.0
    decode_us: float = 0.0
    lookup_us: float = 0.0
    apply_us: float = 0.0
    policy_lookup_us: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise WireProtocolError(f"{name} must be finite and non-negative")

    @property
    def total_us(self) -> float:
        return math.fsum(float(value) for value in self.__dict__.values())


@dataclass(frozen=True)
class ContractCacheEntry:
    join_identity: JoinIdentity
    receiver_rank: int
    epoch: int
    sequence: int
    missing_slot_mask: int
    slack_bucket: int
    flags: int

    def __post_init__(self) -> None:
        if type(self.join_identity) is not JoinIdentity:
            raise WireProtocolError("cache entry requires a full JoinIdentity")
        _u("receiver_rank", self.receiver_rank, 0xFF)
        _u("epoch", self.epoch, 0xFFFFFFFF)
        _u("sequence", self.sequence, 0xFFFFFFFF)
        _u("missing_slot_mask", self.missing_slot_mask, 0xFFFF)
        _u("slack_bucket", self.slack_bucket, 3)
        _u("flags", self.flags, 0xFF)
        if self.epoch == 0 or self.sequence == 0 or self.missing_slot_mask == 0:
            raise WireProtocolError("cache epoch/sequence/mask must be nonzero")
        if (
            self.join_identity.receiver_rank != self.receiver_rank
            or self.join_identity.epoch != self.epoch
        ):
            raise WireProtocolError("cache entry rank/epoch disagrees with identity")
        if self.flags != 0:
            raise WireProtocolError("cache entry contains unknown flags")

    @property
    def missing_count(self) -> int:
        return bin(self.missing_slot_mask).count("1")

    @property
    def is_last_sibling(self) -> bool:
        return self.missing_count == 1


@dataclass(frozen=True)
class ContractApplyResult:
    applied: bool
    fallback: bool
    fault: str | None
    entries: tuple[ContractCacheEntry, ...]
    charged_bytes: int
    received_bytes: int
    charged_us: float


class SenderContractCache:
    """Mutable sender-local cache with atomic epoch/sequence application."""

    def __init__(self, sender_rank: int) -> None:
        _u("sender_rank", sender_rank, 0xFF)
        self.sender_rank = sender_rank
        self._entries: dict[JoinIdentity, ContractCacheEntry] = {}
        self._current_epoch: dict[int, int] = {}
        self._last_sequence: dict[tuple[int, int], int] = {}

    def snapshot(self) -> Mapping[JoinIdentity, ContractCacheEntry]:
        return dict(self._entries)

    def current_epoch(self, receiver_rank: int) -> int | None:
        return self._current_epoch.get(receiver_rank)

    def last_sequence(self, receiver_rank: int, epoch: int) -> int | None:
        return self._last_sequence.get((receiver_rank, epoch))

    def _sequence_fault(self, message: ContractMessage) -> str | None:
        current = self._current_epoch.get(message.receiver_rank)
        if current is not None and message.epoch < current:
            return "stale_epoch"
        if current is None or message.epoch > current:
            if message.sequence != 1:
                return "unknown_or_missing_epoch_start"
            return None
        last = self._last_sequence[(message.receiver_rank, message.epoch)]
        if message.sequence == last:
            return "duplicate_sequence"
        if message.sequence < last:
            return "out_of_order_sequence"
        if message.sequence > last + 1:
            return "missing_sequence"
        return None

    def _commit(
        self, message: ContractMessage, entries: Sequence[ContractCacheEntry]
    ) -> None:
        current = self._current_epoch.get(message.receiver_rank)
        if current is None or message.epoch > current:
            self._entries = {
                identity: entry
                for identity, entry in self._entries.items()
                if identity.receiver_rank != message.receiver_rank
            }
            self._current_epoch[message.receiver_rank] = message.epoch
        for entry in entries:
            self._entries[entry.join_identity] = entry
        self._last_sequence[(message.receiver_rank, message.epoch)] = message.sequence


def _fallback(
    payload: bytes, tax: ContractTax, fault: str, *, produced_bytes: int
) -> ContractApplyResult:
    return ContractApplyResult(
        applied=False,
        fallback=True,
        fault=fault,
        entries=(),
        charged_bytes=produced_bytes,
        received_bytes=len(payload),
        charged_us=tax.total_us,
    )


def apply_wire_contract(
    payload: bytes,
    *,
    cache: SenderContractCache,
    identity_table: IdentityTable,
    expected_sender_rank: int,
    tax: ContractTax,
    produced_bytes: int | None = None,
) -> ContractApplyResult:
    """Decode, resolve, and atomically apply one charged wire message.

    All wire and state faults return a fallback result.  The supplied bytes and
    every frozen tax component remain charged on failure.
    """

    if type(payload) is not bytes:
        raise WireProtocolError("apply path requires immutable wire bytes")
    if type(cache) is not SenderContractCache or type(identity_table) is not IdentityTable:
        raise WireProtocolError("apply path requires canonical cache and identity table")
    if type(tax) is not ContractTax:
        raise WireProtocolError("apply path requires ContractTax")
    if produced_bytes is None:
        produced_bytes = len(payload)
    if (
        type(produced_bytes) is not int
        or produced_bytes < len(payload)
        or produced_bytes < 0
    ):
        raise WireProtocolError("produced_bytes must cover the received payload")
    _u("expected_sender_rank", expected_sender_rank, 0xFF)
    if cache.sender_rank != expected_sender_rank:
        raise WireProtocolError("expected sender does not own the supplied cache")

    try:
        message = decode_contract(payload)
    except RICValidationError as exc:
        return _fallback(
            payload, tax, f"malformed:{exc}", produced_bytes=produced_bytes
        )
    if message.sender_rank != expected_sender_rank:
        return _fallback(payload, tax, "wrong_sender", produced_bytes=produced_bytes)
    if not identity_table.knows_epoch(message.receiver_rank, message.epoch):
        return _fallback(payload, tax, "unknown_epoch", produced_bytes=produced_bytes)
    sequence_fault = cache._sequence_fault(message)
    if sequence_fault is not None:
        return _fallback(payload, tax, sequence_fault, produced_bytes=produced_bytes)

    resolved: list[tuple[JoinIdentity, ContractRecord]] = []
    seen: set[JoinIdentity] = set()
    try:
        for record in message.records:
            identity = identity_table.resolve(
                record, receiver_rank=message.receiver_rank, epoch=message.epoch
            )
            if identity in seen:
                raise WireProtocolError("duplicate identity in one contract message")
            seen.add(identity)
            resolved.append((identity, record))
    except RICValidationError as exc:
        return _fallback(
            payload, tax, f"lookup:{exc}", produced_bytes=produced_bytes
        )

    entries = tuple(
        ContractCacheEntry(
            join_identity=identity,
            receiver_rank=message.receiver_rank,
            epoch=message.epoch,
            sequence=message.sequence,
            missing_slot_mask=record.missing_slot_mask,
            slack_bucket=record.slack_bucket,
            flags=record.flags,
        )
        for identity, record in resolved
    )
    cache._commit(message, entries)
    return ContractApplyResult(
        applied=True,
        fallback=False,
        fault=None,
        entries=entries,
        charged_bytes=produced_bytes,
        received_bytes=len(payload),
        charged_us=tax.total_us,
    )


__all__ = [
    "MAGIC",
    "VERSION",
    "HEADER_STRUCT",
    "RECORD_STRUCT",
    "HEADER_BYTES",
    "RECORD_BYTES",
    "ALIGNMENT_BYTES",
    "WireProtocolError",
    "ContractRecord",
    "ContractMessage",
    "ContractTax",
    "ContractCacheEntry",
    "ContractApplyResult",
    "IdentityBinding",
    "IdentityTable",
    "SenderContractCache",
    "encoded_contract_bytes",
    "join_identity_hash_parts",
    "encode_contract",
    "decode_contract",
    "apply_wire_contract",
]
