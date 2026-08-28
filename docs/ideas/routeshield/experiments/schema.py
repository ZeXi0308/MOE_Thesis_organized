from __future__ import annotations

"""Tenant-qualified route-ledger contracts for RouteShield Gate-0."""

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import AbstractSet, Iterable, Mapping


class ProtocolError(RuntimeError):
    """An input violates the frozen RouteShield contract."""


TRAFFIC_CLASSES = frozenset(
    {"NAT_BENIGN", "NAT_PATHOLOGICAL", "ADV_TEXT", "SYN_ROUTE"}
)
SPLITS = frozenset({"calibration", "evaluation", "smoke"})
ROLES = frozenset({"victim", "cotenant", "attacker"})
PHASES = frozenset({"prefill", "decode"})
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

ROUTE_COLUMNS = (
    "model",
    "model_revision",
    "tenant_id",
    "request_id",
    "document_id",
    "isolation_domain",
    "split",
    "role",
    "traffic_class",
    "prompt_hash",
    "prompt_tokens",
    "phase",
    "chunk_id",
    "decode_step",
    "token_position",
    "token_id",
    "layer_id",
    "topk_slot",
    "expert_id",
    "gate_weight",
    "placement_id",
    "target_rank",
    "rank_binding_stage",
    "replica_instance_id",
    "device_uuid",
    "dispatch_event_id",
    "request_arrival_us",
    "route_observed_us",
)


def _required_text(row: Mapping[str, object], key: str) -> str:
    raw = row.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ProtocolError(f"{key} must be non-empty")
    return raw.strip()


def _integer(row: Mapping[str, object], key: str, *, minimum: int) -> int:
    raw = row.get(key)
    if isinstance(raw, bool):
        raise ProtocolError(f"{key} must be an integer")
    try:
        value = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{key} must be an integer") from exc
    if value < minimum:
        raise ProtocolError(f"{key} must be >= {minimum}")
    return value


def _number(row: Mapping[str, object], key: str, *, minimum: float) -> float:
    try:
        value = float(str(row.get(key)))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{key} must be numeric") from exc
    if not math.isfinite(value) or value < minimum:
        raise ProtocolError(f"{key} must be finite and >= {minimum}")
    return value


@dataclass(frozen=True)
class RouteContribution:
    model: str
    model_revision: str
    tenant_id: str
    request_id: str
    document_id: str
    isolation_domain: str
    split: str
    role: str
    traffic_class: str
    prompt_hash: str
    prompt_tokens: int
    phase: str
    chunk_id: int
    decode_step: int
    token_position: int
    token_id: int
    layer_id: int
    topk_slot: int
    expert_id: int
    gate_weight: float
    placement_id: str
    target_rank: int
    rank_binding_stage: str
    replica_instance_id: str
    device_uuid: str
    dispatch_event_id: str
    request_arrival_us: float
    route_observed_us: float

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, object],
        *,
        require_rank_binding: bool = True,
    ) -> "RouteContribution":
        if set(row) != set(ROUTE_COLUMNS):
            missing = sorted(set(ROUTE_COLUMNS) - set(row))
            extra = sorted(set(row) - set(ROUTE_COLUMNS), key=str)
            raise ProtocolError(
                f"route contribution fields mismatch; missing={missing}, extra={extra}"
            )
        model_revision = _required_text(row, "model_revision")
        prompt_hash = _required_text(row, "prompt_hash")
        if not HEX40.fullmatch(model_revision):
            raise ProtocolError("model_revision must be a lowercase 40-hex commit")
        if not HEX64.fullmatch(prompt_hash):
            raise ProtocolError("prompt_hash must be a lowercase SHA-256")

        split = _required_text(row, "split")
        role = _required_text(row, "role")
        traffic_class = _required_text(row, "traffic_class")
        phase = _required_text(row, "phase")
        if split not in SPLITS:
            raise ProtocolError(f"unknown split: {split}")
        if role not in ROLES:
            raise ProtocolError(f"unknown role: {role}")
        if traffic_class not in TRAFFIC_CLASSES:
            raise ProtocolError(f"unknown traffic_class: {traffic_class}")
        if phase not in PHASES:
            raise ProtocolError(f"unknown phase: {phase}")
        if traffic_class == "ADV_TEXT" and role != "attacker":
            raise ProtocolError("ADV_TEXT rows must belong to the attacker role")
        if role == "victim" and traffic_class not in {"NAT_BENIGN", "NAT_PATHOLOGICAL"}:
            raise ProtocolError("victim rows must use a natural traffic class")

        decode_step = _integer(row, "decode_step", minimum=-1)
        if phase == "prefill" and decode_step != -1:
            raise ProtocolError("prefill rows must use decode_step=-1")
        if phase == "decode" and decode_step < 0:
            raise ProtocolError("decode rows must use decode_step>=0")

        placement_id = _required_text(row, "placement_id")
        target_rank = _integer(row, "target_rank", minimum=-1)
        if require_rank_binding and (
            target_rank < 0 or placement_id.startswith("UNRESOLVED_")
        ):
            raise ProtocolError(
                "formal rank metrics require a frozen placement_id and target_rank>=0"
            )
        rank_binding_stage = _required_text(row, "rank_binding_stage")
        replica_instance_id = _required_text(row, "replica_instance_id")
        device_uuid = _required_text(row, "device_uuid")
        dispatch_event_id = _required_text(row, "dispatch_event_id")
        if require_rank_binding:
            if rank_binding_stage != "EXECUTED_DISPATCH":
                raise ProtocolError(
                    "formal rank metrics require rank_binding_stage=EXECUTED_DISPATCH"
                )
            if replica_instance_id.startswith(("UNBOUND", "UNRESOLVED_")):
                raise ProtocolError("formal rank metrics require a replica instance ID")
            if device_uuid.startswith(("UNBOUND", "UNRESOLVED_")):
                raise ProtocolError("formal rank metrics require a physical device UUID")
            if not HEX64.fullmatch(dispatch_event_id):
                raise ProtocolError("dispatch_event_id must be a lowercase SHA-256")

        request_arrival_us = _number(row, "request_arrival_us", minimum=0.0)
        route_observed_us = _number(row, "route_observed_us", minimum=0.0)
        if route_observed_us < request_arrival_us:
            raise ProtocolError("route_observed_us must be >= request_arrival_us")

        return cls(
            model=_required_text(row, "model"),
            model_revision=model_revision,
            tenant_id=_required_text(row, "tenant_id"),
            request_id=_required_text(row, "request_id"),
            document_id=_required_text(row, "document_id"),
            isolation_domain=_required_text(row, "isolation_domain"),
            split=split,
            role=role,
            traffic_class=traffic_class,
            prompt_hash=prompt_hash,
            prompt_tokens=_integer(row, "prompt_tokens", minimum=1),
            phase=phase,
            chunk_id=_integer(row, "chunk_id", minimum=0),
            decode_step=decode_step,
            token_position=_integer(row, "token_position", minimum=0),
            token_id=_integer(row, "token_id", minimum=0),
            layer_id=_integer(row, "layer_id", minimum=0),
            topk_slot=_integer(row, "topk_slot", minimum=0),
            expert_id=_integer(row, "expert_id", minimum=0),
            gate_weight=_number(row, "gate_weight", minimum=0.0),
            placement_id=placement_id,
            target_rank=target_rank,
            rank_binding_stage=rank_binding_stage,
            replica_instance_id=replica_instance_id,
            device_uuid=device_uuid,
            dispatch_event_id=dispatch_event_id,
            request_arrival_us=request_arrival_us,
            route_observed_us=route_observed_us,
        )

    @property
    def request_key(self) -> tuple[str, str]:
        return (self.tenant_id, self.request_id)

    @property
    def token_event_key(self) -> tuple[object, ...]:
        return (
            self.tenant_id,
            self.request_id,
            self.phase,
            self.chunk_id,
            self.decode_step,
            self.token_position,
            self.token_id,
            self.layer_id,
        )

    @property
    def contribution_id(self) -> tuple[object, ...]:
        return (*self.token_event_key, self.topk_slot)


@dataclass(frozen=True)
class ExpectedRouteEvent:
    model: str
    model_revision: str
    tokenizer_sha256: str
    tenant_id: str
    request_id: str
    prompt_hash: str
    prompt_tokens: int
    phase: str
    chunk_id: int
    decode_step: int
    token_position: int
    token_id: int
    layer_id: int
    expected_top_k: int

    @property
    def event_key(self) -> tuple[object, ...]:
        return (
            self.model,
            self.tenant_id,
            self.request_id,
            self.phase,
            self.chunk_id,
            self.decode_step,
            self.token_position,
            self.token_id,
            self.layer_id,
        )


EXPECTED_EVENT_FIELDS = frozenset(
    {
        "model",
        "model_revision",
        "tokenizer_sha256",
        "tenant_id",
        "request_id",
        "prompt_hash",
        "prompt_tokens",
        "phase",
        "chunk_id",
        "decode_step",
        "token_position",
        "token_id",
        "layer_id",
        "expected_top_k",
    }
)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ProtocolError(f"duplicate expected-event JSON key: {key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ProtocolError(f"non-finite expected-event JSON value is forbidden: {value}")


def load_expected_events_jsonl(path: str | Path) -> list[ExpectedRouteEvent]:
    output: list[ExpectedRouteEvent] = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                raise ProtocolError(f"{path}:{lineno}: blank event-manifest row")
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_json_constant,
                )
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"{path}:{lineno}: invalid JSON") from exc
            if not isinstance(row, Mapping) or set(row) != EXPECTED_EVENT_FIELDS:
                raise ProtocolError(f"{path}:{lineno}: expected-event fields changed")
            revision = _required_text(row, "model_revision")
            tokenizer_sha256 = _required_text(row, "tokenizer_sha256")
            prompt_hash = _required_text(row, "prompt_hash")
            if (
                not HEX40.fullmatch(revision)
                or not HEX64.fullmatch(tokenizer_sha256)
                or not HEX64.fullmatch(prompt_hash)
            ):
                raise ProtocolError("expected event contains an invalid revision/hash")
            event = ExpectedRouteEvent(
                model=_required_text(row, "model"),
                model_revision=revision,
                tokenizer_sha256=tokenizer_sha256,
                tenant_id=_required_text(row, "tenant_id"),
                request_id=_required_text(row, "request_id"),
                prompt_hash=prompt_hash,
                prompt_tokens=_integer(row, "prompt_tokens", minimum=1),
                phase=_required_text(row, "phase"),
                chunk_id=_integer(row, "chunk_id", minimum=0),
                decode_step=_integer(row, "decode_step", minimum=-1),
                token_position=_integer(row, "token_position", minimum=0),
                token_id=_integer(row, "token_id", minimum=0),
                layer_id=_integer(row, "layer_id", minimum=0),
                expected_top_k=_integer(row, "expected_top_k", minimum=1),
            )
            if event.phase not in PHASES:
                raise ProtocolError("expected event phase is invalid")
            if event.phase == "prefill" and event.decode_step != -1:
                raise ProtocolError("expected prefill events must use decode_step=-1")
            if event.phase == "decode" and event.decode_step < 0:
                raise ProtocolError("expected decode events must use decode_step>=0")
            output.append(event)
    if not output:
        raise ProtocolError("expected route-event manifest is empty")
    if len({row.event_key for row in output}) != len(output):
        raise ProtocolError("expected route-event manifest contains duplicate events")
    return output


def load_route_csv(
    path: str | Path,
    *,
    expected_topk: Mapping[str, int],
    expected_revisions: Mapping[str, str] | None = None,
    expected_tokenizers: Mapping[str, str] | None = None,
    num_experts: Mapping[str, int] | None = None,
    expected_dispatch_bindings: Mapping[
        str, Mapping[int, AbstractSet[tuple[int, str, str]]]
    ]
    | None = None,
    expected_events: Iterable[ExpectedRouteEvent] | None = None,
    require_rank_binding: bool = True,
) -> list[RouteContribution]:
    route_path = Path(path)
    with route_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError("route CSV has no header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ProtocolError("route CSV contains duplicate headers")
        missing = sorted(set(ROUTE_COLUMNS) - set(reader.fieldnames))
        extra = sorted(set(reader.fieldnames) - set(ROUTE_COLUMNS))
        if missing or extra:
            raise ProtocolError(
                f"route CSV fields mismatch; missing={missing}, extra={extra}"
            )
        rows = [
            RouteContribution.from_mapping(
                raw, require_rank_binding=require_rank_binding
            )
            for raw in reader
        ]
    validate_route_rows(
        rows,
        expected_topk=expected_topk,
        expected_revisions=expected_revisions,
        expected_tokenizers=expected_tokenizers,
        num_experts=num_experts,
        expected_dispatch_bindings=expected_dispatch_bindings,
        expected_events=expected_events,
        require_rank_binding=require_rank_binding,
    )
    return rows


def validate_route_rows(
    rows: Iterable[RouteContribution],
    *,
    expected_topk: Mapping[str, int],
    expected_revisions: Mapping[str, str] | None = None,
    expected_tokenizers: Mapping[str, str] | None = None,
    num_experts: Mapping[str, int] | None = None,
    expected_dispatch_bindings: Mapping[
        str, Mapping[int, AbstractSet[tuple[int, str, str]]]
    ]
    | None = None,
    expected_events: Iterable[ExpectedRouteEvent] | None = None,
    require_rank_binding: bool = True,
) -> None:
    materialized = list(rows)
    if not materialized:
        raise ProtocolError("route ledger is empty")

    seen: set[tuple[object, ...]] = set()
    dispatch_ids: set[str] = set()
    events: dict[tuple[object, ...], list[RouteContribution]] = {}
    request_metadata: dict[tuple[str, str], tuple[object, ...]] = {}
    prompt_splits: dict[str, set[str]] = {}
    token_identities: dict[tuple[object, ...], int] = {}

    if require_rank_binding and expected_dispatch_bindings is None:
        raise ProtocolError(
            "formal rank metrics require a verified placement/replica dispatch map"
        )
    if require_rank_binding and expected_events is None:
        raise ProtocolError("formal route metrics require an expected event manifest")
    if require_rank_binding and expected_tokenizers is None:
        raise ProtocolError("formal route metrics require frozen tokenizer hashes")

    materialized_expected_events = (
        list(expected_events) if expected_events is not None else None
    )
    if materialized_expected_events is not None and len(
        {event.event_key for event in materialized_expected_events}
    ) != len(materialized_expected_events):
        raise ProtocolError("expected route-event manifest contains duplicate events")
    if materialized_expected_events is not None and expected_tokenizers is not None:
        for event in materialized_expected_events:
            if event.model not in expected_tokenizers:
                raise ProtocolError(
                    f"expected event uses an unregistered tokenizer model: {event.model}"
                )
            if event.tokenizer_sha256 != expected_tokenizers[event.model]:
                raise ProtocolError(
                    f"tokenizer hash mismatch for expected event model {event.model}"
                )

    for row in materialized:
        if row.model not in expected_topk:
            raise ProtocolError(f"unregistered model key: {row.model}")
        if expected_revisions is not None and row.model_revision != expected_revisions.get(
            row.model
        ):
            raise ProtocolError(f"revision mismatch for model {row.model}")
        if num_experts is not None and row.expert_id >= num_experts[row.model]:
            raise ProtocolError(
                f"expert_id={row.expert_id} exceeds model {row.model} expert count"
            )
        if require_rank_binding and row.target_rank < 0:
            raise ProtocolError("formal ledger contains an unbound target rank")
        if require_rank_binding:
            assert expected_dispatch_bindings is not None
            allowed_dispatches = expected_dispatch_bindings.get(row.model, {}).get(
                row.expert_id
            )
            dispatch_binding = (
                row.target_rank,
                row.replica_instance_id,
                row.device_uuid,
            )
            if allowed_dispatches is None or dispatch_binding not in allowed_dispatches:
                raise ProtocolError(
                    "executed dispatch binding is absent from the frozen placement map: "
                    f"model={row.model}, expert={row.expert_id}, binding={dispatch_binding}"
                )
        if row.contribution_id in seen:
            raise ProtocolError(f"duplicate contribution identity: {row.contribution_id}")
        seen.add(row.contribution_id)
        if require_rank_binding:
            if row.dispatch_event_id in dispatch_ids:
                raise ProtocolError(
                    f"duplicate executed dispatch_event_id: {row.dispatch_event_id}"
                )
            dispatch_ids.add(row.dispatch_event_id)
        events.setdefault(row.token_event_key, []).append(row)

        metadata = (
            row.model,
            row.model_revision,
            row.document_id,
            row.isolation_domain,
            row.split,
            row.role,
            row.traffic_class,
            row.prompt_hash,
            row.prompt_tokens,
            row.phase,
            row.placement_id,
            row.request_arrival_us,
        )
        previous = request_metadata.setdefault(row.request_key, metadata)
        if previous != metadata:
            raise ProtocolError(f"request metadata changed within {row.request_key}")
        prompt_splits.setdefault(row.prompt_hash, set()).add(row.split)
        token_key = (
            row.tenant_id,
            row.request_id,
            row.phase,
            row.decode_step,
            row.token_position,
        )
        previous_token_id = token_identities.setdefault(token_key, row.token_id)
        if previous_token_id != row.token_id:
            raise ProtocolError(
                f"token_id changed across slots/layers for {token_key}: "
                f"{previous_token_id} != {row.token_id}"
            )

    leaked = sorted(
        prompt_hash for prompt_hash, splits in prompt_splits.items() if len(splits) > 1
    )
    if leaked:
        raise ProtocolError(
            f"prompt hashes cross calibration/evaluation/smoke splits: {leaked[:3]}"
        )

    for event_key, siblings in events.items():
        expected = expected_topk[siblings[0].model]
        slots = sorted(row.topk_slot for row in siblings)
        if slots != list(range(expected)):
            raise ProtocolError(
                f"top-k closure failed for {event_key}: expected 0..{expected - 1}, got {slots}"
            )
        experts = [row.expert_id for row in siblings]
        if len(experts) != len(set(experts)):
            raise ProtocolError(f"duplicate expert within top-k siblings for {event_key}")
        placements = {row.placement_id for row in siblings}
        observed_times = {row.route_observed_us for row in siblings}
        if len(placements) != 1 or len(observed_times) != 1:
            raise ProtocolError(
                f"top-k siblings must share placement and observation time for {event_key}"
            )

    if materialized_expected_events is not None:
        expected_by_key = {row.event_key: row for row in materialized_expected_events}
        observed_by_key = {
            (row.model, *row.token_event_key): row for row in materialized
        }
        if set(observed_by_key) != set(expected_by_key):
            missing = sorted(set(expected_by_key) - set(observed_by_key), key=str)
            extra = sorted(set(observed_by_key) - set(expected_by_key), key=str)
            raise ProtocolError(
                "route ledger does not close the expected token/chunk/layer events; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        for key, expected_event in expected_by_key.items():
            observed = observed_by_key[key]
            if (
                observed.model_revision != expected_event.model_revision
                or observed.prompt_hash != expected_event.prompt_hash
                or observed.prompt_tokens != expected_event.prompt_tokens
                or expected_topk[observed.model] != expected_event.expected_top_k
            ):
                raise ProtocolError(f"route event metadata differs from manifest: {key}")
