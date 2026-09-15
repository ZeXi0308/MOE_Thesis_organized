"""Pure, fail-closed contracts for SemanticFence expert-row packing.

This module deliberately contains no Torch or GPU dependencies.  It defines the
identity, packing, calibration, serialization, and selection rules used by the
first SemanticFence pilot.  Runtime measurement belongs in a separate runner.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping


ROW_ID_SCHEMA = "semanticfence-row-id-v1"
CONTRACT_SCHEMA = "semanticfence-executor-contract-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ContractError(ValueError):
    """The requested packing or contract operation is not trustworthy."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: str, field: str) -> str:
    text = str(value)
    if _SHA256_RE.fullmatch(text) is None:
        raise ContractError(f"{field} must be one lowercase SHA-256")
    return text


def _require_non_negative(value: int, field: str) -> int:
    observed = int(value)
    if observed < 0:
        raise ContractError(f"{field} must be non-negative")
    return observed


@dataclass(frozen=True, slots=True)
class RowRecord:
    """One real routed row with a content-addressed semantic identity."""

    split: str
    document_sha256: str
    document_index: int
    offset: int
    token_position: int
    layer: int
    expert_id: int
    route_rank: int
    hidden_sha256: str

    def __post_init__(self) -> None:
        if not self.split or not re.fullmatch(r"[a-z][a-z0-9_-]*", self.split):
            raise ContractError("split must be a non-empty lowercase identifier")
        _require_sha256(self.document_sha256, "document_sha256")
        _require_sha256(self.hidden_sha256, "hidden_sha256")
        _require_non_negative(self.document_index, "document_index")
        _require_non_negative(self.offset, "offset")
        _require_non_negative(self.token_position, "token_position")
        _require_non_negative(self.layer, "layer")
        _require_non_negative(self.expert_id, "expert_id")
        if int(self.route_rank) <= 0:
            raise ContractError("route_rank must be positive")

    @property
    def row_id(self) -> str:
        return canonical_row_id(self)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ROW_ID_SCHEMA,
            "split": self.split,
            "document_sha256": self.document_sha256,
            "document_index": int(self.document_index),
            "offset": int(self.offset),
            "token_position": int(self.token_position),
            "layer": int(self.layer),
            "expert_id": int(self.expert_id),
            "route_rank": int(self.route_rank),
            "hidden_sha256": self.hidden_sha256,
        }


def canonical_row_id(row: RowRecord) -> str:
    """Return the canonical content-addressed identity of ``row``."""

    if not isinstance(row, RowRecord):
        raise ContractError("canonical_row_id requires a RowRecord")
    return _canonical_sha256(row.identity_payload())


@dataclass(frozen=True, slots=True)
class Pack:
    """An immutable batch of distinct rows for exactly one expert instance."""

    layer: int
    expert_id: int
    rows: tuple[RowRecord, ...]

    def __post_init__(self) -> None:
        _require_non_negative(self.layer, "pack.layer")
        _require_non_negative(self.expert_id, "pack.expert_id")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ContractError("pack rows must be one non-empty tuple")
        row_ids: list[str] = []
        for row in self.rows:
            if not isinstance(row, RowRecord):
                raise ContractError("pack rows must contain only RowRecord values")
            if row.layer != self.layer or row.expert_id != self.expert_id:
                raise ContractError("pack may not cross layer or expert boundaries")
            row_ids.append(row.row_id)
        if len(set(row_ids)) != len(row_ids):
            raise ContractError("pack may not contain duplicate rows")

    @property
    def m(self) -> int:
        return len(self.rows)

    @property
    def pack_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": "semanticfence-pack-id-v1",
                "layer": int(self.layer),
                "expert_id": int(self.expert_id),
                "row_ids": [row.row_id for row in self.rows],
            }
        )


def pack_distinct_rows(
    rows: Iterable[RowRecord], allowed_ms: Iterable[int]
) -> tuple[Pack, ...]:
    """Deterministically pack all rows, using M=1 as the mandatory remainder.

    Rows are sorted by canonical identity, grouped by ``(layer, expert_id)``,
    and greedily assigned the largest allowed M that fits.  Padding, duplicated
    filler rows, cross-expert packing, and dropped remainders are impossible.
    """

    materialized = tuple(rows)
    if any(not isinstance(row, RowRecord) for row in materialized):
        raise ContractError("rows must contain only RowRecord values")
    row_ids = [row.row_id for row in materialized]
    if len(set(row_ids)) != len(row_ids):
        raise ContractError("input contains duplicate canonical row identities")

    normalized_ms = {int(value) for value in allowed_ms}
    if any(value <= 0 for value in normalized_ms):
        raise ContractError("allowed M values must be positive")
    normalized_ms.add(1)
    descending_ms = tuple(sorted(normalized_ms, reverse=True))

    grouped: dict[tuple[int, int], list[RowRecord]] = defaultdict(list)
    for row in materialized:
        grouped[(row.layer, row.expert_id)].append(row)

    packs: list[Pack] = []
    for (layer, expert_id), group in sorted(grouped.items()):
        remaining = sorted(group, key=lambda row: row.row_id)
        cursor = 0
        while cursor < len(remaining):
            available = len(remaining) - cursor
            m_value = next(value for value in descending_ms if value <= available)
            pack_rows = tuple(remaining[cursor : cursor + m_value])
            packs.append(Pack(layer=layer, expert_id=expert_id, rows=pack_rows))
            cursor += m_value

    result = tuple(packs)
    validate_row_coverage(materialized, result)
    return result


def validate_row_coverage(
    rows: Iterable[RowRecord], packs: Iterable[Pack]
) -> None:
    """Require exact, one-to-one coverage of ``rows`` by ``packs``."""

    expected_rows = tuple(rows)
    expected_ids = [row.row_id for row in expected_rows]
    if len(set(expected_ids)) != len(expected_ids):
        raise ContractError("expected row set contains duplicate identities")

    observed_ids: list[str] = []
    for pack in packs:
        if not isinstance(pack, Pack):
            raise ContractError("packs must contain only Pack values")
        observed_ids.extend(row.row_id for row in pack.rows)
    if len(set(observed_ids)) != len(observed_ids):
        raise ContractError("packed coverage contains a duplicate row")

    expected_set = set(expected_ids)
    observed_set = set(observed_ids)
    if observed_set != expected_set:
        missing = sorted(expected_set - observed_set)
        unexpected = sorted(observed_set - expected_set)
        raise ContractError(
            f"packed coverage mismatch: missing={missing}, unexpected={unexpected}"
        )


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    """Raw exactness outcomes for every row in every repeat of one pack."""

    pack: Pack
    signature: str
    repeat_row_exact: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pack, Pack):
            raise ContractError("calibration observation requires a Pack")
        _require_sha256(self.signature, "signature")
        if not isinstance(self.repeat_row_exact, tuple) or not self.repeat_row_exact:
            raise ContractError("calibration observation requires at least one repeat")
        for repeat in self.repeat_row_exact:
            if not isinstance(repeat, tuple) or len(repeat) != self.pack.m:
                raise ContractError(
                    "each repeat must contain one exactness flag per packed row"
                )
            if any(type(value) is not bool for value in repeat):
                raise ContractError("repeat exactness flags must be booleans")

    @property
    def all_repeats_exact(self) -> bool:
        return all(value for repeat in self.repeat_row_exact for value in repeat)

    @property
    def total_checks(self) -> int:
        return sum(len(repeat) for repeat in self.repeat_row_exact)

    @property
    def exact_checks(self) -> int:
        return sum(
            int(value) for repeat in self.repeat_row_exact for value in repeat
        )


@dataclass(frozen=True, slots=True)
class ContractEntry:
    layer: int
    expert_id: int
    m: int
    signature: str
    pack_count: int
    document_count: int
    repeat_count: int
    exact_checks: int
    total_checks: int
    all_repeats_exact: bool
    allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer),
            "expert_id": int(self.expert_id),
            "m": int(self.m),
            "signature": self.signature,
            "pack_count": int(self.pack_count),
            "document_count": int(self.document_count),
            "repeat_count": int(self.repeat_count),
            "exact_checks": int(self.exact_checks),
            "total_checks": int(self.total_checks),
            "all_repeats_exact": bool(self.all_repeats_exact),
            "allowed": bool(self.allowed),
        }


@dataclass(frozen=True, slots=True)
class ExecutorContract:
    schema_version: str
    source_split: str
    stack_digest: str
    min_packs: int
    min_documents: int
    calibration_sha256: str
    entries: tuple[ContractEntry, ...]
    contract_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_split": self.source_split,
            "stack_digest": self.stack_digest,
            "min_packs": int(self.min_packs),
            "min_documents": int(self.min_documents),
            "calibration_sha256": self.calibration_sha256,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.payload() | {"contract_sha256": self.contract_sha256}


def build_contract(
    observations: Iterable[CalibrationObservation],
    *,
    stack_digest: str,
    min_packs: int,
    min_documents: int,
) -> ExecutorContract:
    """Build a calibration-only, content-addressed executor contract."""

    _require_sha256(stack_digest, "stack_digest")
    if int(min_packs) <= 0 or int(min_documents) <= 0:
        raise ContractError("min_packs and min_documents must be positive")
    rows = tuple(observations)
    if not rows:
        raise ContractError("at least one calibration observation is required")

    seen_pack_ids: set[str] = set()
    grouped: dict[
        tuple[int, int, int, str], list[CalibrationObservation]
    ] = defaultdict(list)
    calibration_payload: list[dict[str, Any]] = []
    for observation in rows:
        if not isinstance(observation, CalibrationObservation):
            raise ContractError(
                "observations must contain only CalibrationObservation values"
            )
        if observation.pack.m <= 1:
            raise ContractError("M=1 is canonical fallback, not a contract action")
        if any(row.split != "calibration" for row in observation.pack.rows):
            raise ContractError("executor contracts may use calibration rows only")
        if observation.pack.pack_id in seen_pack_ids:
            raise ContractError("duplicate calibration pack cannot inflate support")
        seen_pack_ids.add(observation.pack.pack_id)
        grouped[
            (
                observation.pack.layer,
                observation.pack.expert_id,
                observation.pack.m,
                observation.signature,
            )
        ].append(observation)
        calibration_payload.append(
            {
                "pack_id": observation.pack.pack_id,
                "layer": observation.pack.layer,
                "expert_id": observation.pack.expert_id,
                "m": observation.pack.m,
                "signature": observation.signature,
                "row_ids": [row.row_id for row in observation.pack.rows],
                "repeat_row_exact": [
                    list(repeat) for repeat in observation.repeat_row_exact
                ],
            }
        )

    entries: list[ContractEntry] = []
    for (layer, expert_id, m_value, signature), group in sorted(grouped.items()):
        document_ids = {
            row.document_sha256
            for observation in group
            for row in observation.pack.rows
        }
        exact_checks = sum(observation.exact_checks for observation in group)
        total_checks = sum(observation.total_checks for observation in group)
        repeat_count = sum(len(observation.repeat_row_exact) for observation in group)
        all_exact = exact_checks == total_checks and total_checks > 0
        allowed = (
            len(group) >= int(min_packs)
            and len(document_ids) >= int(min_documents)
            and all_exact
        )
        entries.append(
            ContractEntry(
                layer=layer,
                expert_id=expert_id,
                m=m_value,
                signature=signature,
                pack_count=len(group),
                document_count=len(document_ids),
                repeat_count=repeat_count,
                exact_checks=exact_checks,
                total_checks=total_checks,
                all_repeats_exact=all_exact,
                allowed=allowed,
            )
        )

    calibration_payload.sort(
        key=lambda item: (
            item["layer"],
            item["expert_id"],
            item["m"],
            item["signature"],
            item["pack_id"],
        )
    )
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "source_split": "calibration",
        "stack_digest": stack_digest,
        "min_packs": int(min_packs),
        "min_documents": int(min_documents),
        "calibration_sha256": _canonical_sha256(calibration_payload),
        "entries": [entry.to_dict() for entry in entries],
    }
    contract = ExecutorContract(
        schema_version=CONTRACT_SCHEMA,
        source_split="calibration",
        stack_digest=stack_digest,
        min_packs=int(min_packs),
        min_documents=int(min_documents),
        calibration_sha256=payload["calibration_sha256"],
        entries=tuple(entries),
        contract_sha256=_canonical_sha256(payload),
    )
    return validate_contract(contract)


def _entry_from_mapping(value: Mapping[str, Any]) -> ContractEntry:
    required = {
        "layer",
        "expert_id",
        "m",
        "signature",
        "pack_count",
        "document_count",
        "repeat_count",
        "exact_checks",
        "total_checks",
        "all_repeats_exact",
        "allowed",
    }
    if set(value) != required:
        raise ContractError("contract entry fields are missing or unknown")
    return ContractEntry(
        layer=int(value["layer"]),
        expert_id=int(value["expert_id"]),
        m=int(value["m"]),
        signature=str(value["signature"]),
        pack_count=int(value["pack_count"]),
        document_count=int(value["document_count"]),
        repeat_count=int(value["repeat_count"]),
        exact_checks=int(value["exact_checks"]),
        total_checks=int(value["total_checks"]),
        all_repeats_exact=value["all_repeats_exact"],
        allowed=value["allowed"],
    )


def contract_from_dict(value: Mapping[str, Any]) -> ExecutorContract:
    """Parse and authenticate a serialized executor contract."""

    required = {
        "schema_version",
        "source_split",
        "stack_digest",
        "min_packs",
        "min_documents",
        "calibration_sha256",
        "entries",
        "contract_sha256",
    }
    if set(value) != required:
        raise ContractError("contract fields are missing or unknown")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list):
        raise ContractError("contract entries must be a list")
    contract = ExecutorContract(
        schema_version=str(value["schema_version"]),
        source_split=str(value["source_split"]),
        stack_digest=str(value["stack_digest"]),
        min_packs=int(value["min_packs"]),
        min_documents=int(value["min_documents"]),
        calibration_sha256=str(value["calibration_sha256"]),
        entries=tuple(_entry_from_mapping(entry) for entry in raw_entries),
        contract_sha256=str(value["contract_sha256"]),
    )
    return validate_contract(contract)


def validate_contract(
    contract: ExecutorContract | Mapping[str, Any],
) -> ExecutorContract:
    """Validate contract semantics and its content hash, or fail closed."""

    if isinstance(contract, Mapping):
        return contract_from_dict(contract)
    if not isinstance(contract, ExecutorContract):
        raise ContractError("contract must be ExecutorContract or mapping")
    if contract.schema_version != CONTRACT_SCHEMA:
        raise ContractError("unknown executor contract schema")
    if contract.source_split != "calibration":
        raise ContractError("executor contract source_split must be calibration")
    _require_sha256(contract.stack_digest, "stack_digest")
    _require_sha256(contract.calibration_sha256, "calibration_sha256")
    _require_sha256(contract.contract_sha256, "contract_sha256")
    if contract.min_packs <= 0 or contract.min_documents <= 0:
        raise ContractError("contract minimums must be positive")

    observed_keys: list[tuple[int, int, int, str]] = []
    for entry in contract.entries:
        if not isinstance(entry, ContractEntry):
            raise ContractError("contract contains a non-ContractEntry value")
        _require_non_negative(entry.layer, "entry.layer")
        _require_non_negative(entry.expert_id, "entry.expert_id")
        if entry.m <= 1:
            raise ContractError("contract entries must describe M>1")
        _require_sha256(entry.signature, "entry.signature")
        for field, value in (
            ("pack_count", entry.pack_count),
            ("document_count", entry.document_count),
            ("repeat_count", entry.repeat_count),
            ("exact_checks", entry.exact_checks),
            ("total_checks", entry.total_checks),
        ):
            _require_non_negative(value, field)
        if entry.exact_checks > entry.total_checks:
            raise ContractError("exact_checks exceeds total_checks")
        recomputed_all_exact = (
            entry.total_checks > 0 and entry.exact_checks == entry.total_checks
        )
        if type(entry.all_repeats_exact) is not bool or (
            entry.all_repeats_exact != recomputed_all_exact
        ):
            raise ContractError("all_repeats_exact is inconsistent")
        recomputed_allowed = (
            entry.pack_count >= contract.min_packs
            and entry.document_count >= contract.min_documents
            and recomputed_all_exact
        )
        if type(entry.allowed) is not bool or entry.allowed != recomputed_allowed:
            raise ContractError("allowed flag is inconsistent with contract rules")
        observed_keys.append(
            (entry.layer, entry.expert_id, entry.m, entry.signature)
        )
    if observed_keys != sorted(set(observed_keys)):
        raise ContractError("contract entries must be unique and canonically ordered")
    if _canonical_sha256(contract.payload()) != contract.contract_sha256:
        raise ContractError("contract content hash mismatch")
    return contract


def choose_pack_size(
    contract: ExecutorContract | Mapping[str, Any],
    *,
    stack_digest: str,
    layer: int,
    expert_id: int,
    requested_m: int,
    signature: str,
) -> int:
    """Return ``requested_m`` only for an authenticated allowed entry.

    An unknown stack, M, or signature is ordinary runtime uncertainty and
    therefore returns the canonical M=1 fallback.  A malformed/tampered
    contract still raises ``ContractError`` rather than being silently used.
    """

    trusted = validate_contract(contract)
    requested = int(requested_m)
    if requested <= 1 or stack_digest != trusted.stack_digest:
        return 1
    for entry in trusted.entries:
        if (
            entry.layer == int(layer)
            and entry.expert_id == int(expert_id)
            and entry.m == requested
            and entry.signature == signature
            and entry.allowed
        ):
            return requested
    return 1


def allowed_entries(
    contract: ExecutorContract | Mapping[str, Any],
) -> Mapping[tuple[int, int, int, str], ContractEntry]:
    """Return an immutable lookup of authenticated allowed entries."""

    trusted = validate_contract(contract)
    return MappingProxyType(
        {
            (entry.layer, entry.expert_id, entry.m, entry.signature): entry
            for entry in trusted.entries
            if entry.allowed
        }
    )


__all__ = [
    "CalibrationObservation",
    "ContractEntry",
    "ContractError",
    "ExecutorContract",
    "Pack",
    "RowRecord",
    "allowed_entries",
    "build_contract",
    "canonical_row_id",
    "choose_pack_size",
    "contract_from_dict",
    "pack_distinct_rows",
    "validate_contract",
    "validate_row_coverage",
]
