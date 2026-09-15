from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence


class ProtocolError(RuntimeError):
    """An artifact violates the frozen experiment contract."""


def _optional_int(row: Mapping[str, object], key: str, default: int) -> int:
    value = row.get(key)
    return default if value is None or value == "" else int(value)


ROUTE_V2_COLUMNS = (
    "model",
    "phase",
    "request_id",
    "sample_id",
    "arrival_us",
    "deadline_us",
    "layer",
    "token_position",
    "rank",
    "expert_id",
    "gate_weight",
    "src_replica",
    "input_event_id",
    "token_id",
    "decode_step",
    "layer_id",
    "topk_slot",
    "source_rank",
    "target_replica",
)

# route-v3 retains the v2 identity fields and adds the causal timing/legality
# ledger required by a continuous decode replay.  Legacy rows remain readable,
# but they are never formal-v3 eligible merely because defaults can be filled.
ROUTE_COLUMNS = ROUTE_V2_COLUMNS + (
    "document_id",
    "request_arrival_us",
    "layer_ready_us",
    "route_end_us",
    "dispatch_end_us",
    "expert_start_us",
    "expert_end_us",
    "combine_end_us",
    "legal_replica_set",
)

# These are the columns emitted by the original v1 producer. Loading remains
# backwards compatible, but only an explicit, validated v3 trace is eligible
# for the causal Gate path.
ROUTE_V1_REQUIRED_COLUMNS = ROUTE_V2_COLUMNS[:12]


def _optional_float(row: Mapping[str, object], key: str, default: float) -> float:
    value = row.get(key)
    return default if value is None or value == "" else float(value)


def _replica_set(value: object) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("legal_replica_set must be a JSON integer list") from exc
    else:
        decoded = value
    if not isinstance(decoded, (list, tuple)):
        raise ValueError("legal_replica_set must be a list or tuple")
    replicas = tuple(int(item) for item in decoded)
    if any(item < 0 for item in replicas) or len(replicas) != len(set(replicas)):
        raise ValueError("legal_replica_set must contain unique non-negative integers")
    return tuple(sorted(replicas))


@dataclass(frozen=True)
class Contribution:
    model: str
    phase: str
    request_id: str
    sample_id: int
    arrival_us: float
    deadline_us: float
    layer: int
    token_position: int
    rank: int
    expert_id: int
    gate_weight: float
    src_replica: int = 0
    input_event_id: str = ""
    token_id: int = -1
    decode_step: int = -1
    layer_id: int = -1
    topk_slot: int = -1
    source_rank: int = -1
    target_replica: int = -1
    document_id: str = ""
    request_arrival_us: float = -1.0
    layer_ready_us: float = -1.0
    route_end_us: float = -1.0
    dispatch_end_us: float = -1.0
    expert_start_us: float = -1.0
    expert_end_us: float = -1.0
    combine_end_us: float = -1.0
    legal_replica_set: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.model or not self.phase or not self.request_id:
            raise ValueError("model, phase and request_id must be non-empty")
        # Normalize legacy v1 rows into the explicit identity schema.  Frozen
        # dataclasses are retained so identities cannot change after routing.
        if not self.input_event_id:
            object.__setattr__(
                self,
                "input_event_id",
                f"{self.request_id}:{self.phase}:{self.token_position}",
            )
        if self.token_id < 0:
            object.__setattr__(self, "token_id", self.token_position)
        if self.layer_id < 0:
            object.__setattr__(self, "layer_id", self.layer)
        if self.topk_slot < 0:
            object.__setattr__(self, "topk_slot", self.rank - 1)
        if self.source_rank < 0:
            object.__setattr__(self, "source_rank", self.src_replica)
        if not self.document_id:
            object.__setattr__(self, "document_id", self.request_id)
        if self.request_arrival_us < 0:
            object.__setattr__(self, "request_arrival_us", self.arrival_us)
        if self.layer_ready_us < 0:
            object.__setattr__(self, "layer_ready_us", self.arrival_us)
        if self.route_end_us < 0:
            object.__setattr__(self, "route_end_us", self.layer_ready_us)
        object.__setattr__(self, "legal_replica_set", _replica_set(self.legal_replica_set))

        for name in (
            "sample_id",
            "layer",
            "token_position",
            "expert_id",
            "src_replica",
            "token_id",
            "layer_id",
            "topk_slot",
            "source_rank",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.input_event_id:
            raise ValueError("input_event_id must be non-empty")
        if not self.document_id:
            raise ValueError("document_id must be non-empty")
        if isinstance(self.decode_step, bool) or not isinstance(self.decode_step, int) or self.decode_step < -1:
            raise ValueError("decode_step must be an integer >= -1")
        if (
            isinstance(self.target_replica, bool)
            or not isinstance(self.target_replica, int)
            or self.target_replica < -1
        ):
            raise ValueError("target_replica must be an integer >= -1")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if self.layer_id != self.layer:
            raise ValueError("layer_id must equal legacy layer")
        if self.topk_slot != self.rank - 1:
            raise ValueError("topk_slot must equal rank - 1")
        if self.source_rank != self.src_replica:
            raise ValueError("source_rank must equal legacy src_replica")
        if not math.isfinite(self.arrival_us) or self.arrival_us < 0:
            raise ValueError("arrival_us must be finite and non-negative")
        if not math.isfinite(self.deadline_us) or self.deadline_us < self.arrival_us:
            raise ValueError("deadline_us must be finite and >= arrival_us")
        if not (
            math.isfinite(self.request_arrival_us)
            and math.isfinite(self.layer_ready_us)
            and math.isfinite(self.route_end_us)
        ):
            raise ValueError("causal route timestamps must be finite")
        if not (
            0 <= self.request_arrival_us <= self.layer_ready_us <= self.route_end_us
        ):
            raise ValueError(
                "causal route timestamps require request_arrival <= layer_ready <= route_end"
            )
        observed_stage_times = (
            self.dispatch_end_us,
            self.expert_start_us,
            self.expert_end_us,
            self.combine_end_us,
        )
        if any(not math.isfinite(value) for value in observed_stage_times):
            raise ValueError("observed stage timestamps must be finite or -1")
        present = [value >= 0 for value in observed_stage_times]
        if any(present) and not all(present):
            raise ValueError("dispatch/expert/combine timestamps must be all present or all absent")
        if all(present) and not (
            self.route_end_us
            <= self.dispatch_end_us
            <= self.expert_start_us
            <= self.expert_end_us
            <= self.combine_end_us
        ):
            raise ValueError("observed stage timestamps violate causal order")
        if not math.isfinite(self.gate_weight) or self.gate_weight < 0:
            raise ValueError("gate_weight must be finite and non-negative")

    @property
    def contribution_id(self) -> str:
        """Stable mathematical slot identity, independent of route attributes."""
        return (
            f"{self.model}|{self.phase}|{self.request_id}|{self.input_event_id}|"
            f"{self.decode_step}|{self.token_id}|{self.layer_id}|{self.topk_slot}"
        )

    @property
    def route_semantic_id(self) -> str:
        """Slot identity plus the exact router decision that must be conserved."""
        return (
            f"{self.contribution_id}|expert={self.expert_id}|"
            f"weight={format(self.gate_weight, '.17g')}|source={self.source_rank}"
        )

    @property
    def dispatch_ready_us(self) -> float:
        return self.route_end_us

    @property
    def has_observed_stage_ledger(self) -> bool:
        return self.combine_end_us >= 0

    def legal_replicas(self, replica_count: int) -> tuple[int, ...]:
        if replica_count <= 0:
            raise ValueError("replica_count must be positive")
        if not self.legal_replica_set:
            return tuple(range(replica_count))
        if any(replica >= replica_count for replica in self.legal_replica_set):
            raise ProtocolError(
                f"legal replica set {self.legal_replica_set} exceeds replica_count={replica_count}"
            )
        return self.legal_replica_set

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["legal_replica_set"] = list(self.legal_replica_set)
        return payload

    def to_csv(self) -> dict[str, object]:
        payload = self.to_json()
        payload["legal_replica_set"] = json.dumps(
            payload["legal_replica_set"], separators=(",", ":")
        )
        return payload

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "Contribution":
        try:
            return cls(
                model=str(row["model"]),
                phase=str(row["phase"]),
                request_id=str(row["request_id"]),
                sample_id=int(row["sample_id"]),
                arrival_us=float(row["arrival_us"]),
                deadline_us=float(row["deadline_us"]),
                layer=int(row["layer"]),
                token_position=int(row["token_position"]),
                rank=int(row["rank"]),
                expert_id=int(row["expert_id"]),
                gate_weight=float(row["gate_weight"]),
                src_replica=_optional_int(row, "src_replica", 0),
                input_event_id=str(row.get("input_event_id") or ""),
                token_id=_optional_int(row, "token_id", -1),
                decode_step=_optional_int(row, "decode_step", -1),
                layer_id=_optional_int(row, "layer_id", -1),
                topk_slot=_optional_int(row, "topk_slot", -1),
                source_rank=_optional_int(row, "source_rank", -1),
                target_replica=_optional_int(row, "target_replica", -1),
                document_id=str(row.get("document_id") or ""),
                request_arrival_us=_optional_float(row, "request_arrival_us", -1.0),
                layer_ready_us=_optional_float(row, "layer_ready_us", -1.0),
                route_end_us=_optional_float(row, "route_end_us", -1.0),
                dispatch_end_us=_optional_float(row, "dispatch_end_us", -1.0),
                expert_start_us=_optional_float(row, "expert_start_us", -1.0),
                expert_end_us=_optional_float(row, "expert_end_us", -1.0),
                combine_end_us=_optional_float(row, "combine_end_us", -1.0),
                legal_replica_set=_replica_set(row.get("legal_replica_set")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid route row: {exc}") from exc


def load_routes(
    paths: Sequence[str | Path], *, require_explicit_v2: bool = False,
    require_explicit_v3: bool = False,
) -> list[Contribution]:
    rows: list[Contribution] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if require_explicit_v3:
                required = ROUTE_COLUMNS
            elif require_explicit_v2:
                required = ROUTE_V2_COLUMNS
            else:
                required = ROUTE_V1_REQUIRED_COLUMNS
            missing = set(required) - set(reader.fieldnames or ())
            if missing:
                raise ProtocolError(f"{path}: missing route columns {sorted(missing)}")
            rows.extend(Contribution.from_mapping(row) for row in reader)
    validate_identity_conservation(rows)
    if require_explicit_v3:
        validate_causal_route_v3(rows, require_observed_stages=True)
    return rows


def write_routes(path: str | Path, rows: Sequence[Contribution]) -> None:
    validate_identity_conservation(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUTE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.to_csv() for row in rows)


def validate_identity_conservation(
    rows: Sequence[Contribution],
    *,
    expected_top_k: int | None = None,
    expected_layer_ids: Sequence[int] | None = None,
    expected_input_events: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, int]:
    if not rows:
        raise ProtocolError("route trace is empty")
    identities = [row.contribution_id for row in rows]
    if len(identities) != len(set(identities)):
        raise ProtocolError("duplicate routed contribution identity")
    by_token: dict[tuple[str, str, str, str, int], list[Contribution]] = {}
    for row in rows:
        key = (row.model, row.phase, row.request_id, row.input_event_id, row.layer_id)
        by_token.setdefault(key, []).append(row)
    for key, token_rows in by_token.items():
        ranks = sorted(row.rank for row in token_rows)
        if ranks != list(range(1, len(ranks) + 1)):
            raise ProtocolError(f"non-contiguous top-k ranks for token {key}: {ranks}")
        experts = [row.expert_id for row in token_rows]
        if len(experts) != len(set(experts)):
            raise ProtocolError(f"duplicate expert for token {key}")
        token_ids = {row.token_id for row in token_rows}
        decode_steps = {row.decode_step for row in token_rows}
        if len(token_ids) != 1 or len(decode_steps) != 1:
            raise ProtocolError(f"inconsistent event identity for token {key}")
        if expected_top_k is not None and len(token_rows) != expected_top_k:
            raise ProtocolError(
                f"expected top_k={expected_top_k} contributions for token {key}, "
                f"observed={len(token_rows)}"
            )

    # Every input event in a request must close the same router-layer/top-k
    # sibling shape.  This catches a missed layer or a partially recorded step.
    by_request_event: dict[tuple[str, str, str], dict[str, tuple[tuple[int, int], ...]]] = {}
    for (model, phase, request_id, event_id, layer_id), token_rows in by_token.items():
        request_key = (model, phase, request_id)
        event = by_request_event.setdefault(request_key, {})
        signature = list(event.get(event_id, ()))
        signature.append((layer_id, len(token_rows)))
        event[event_id] = tuple(sorted(signature))
    for request_key, events in by_request_event.items():
        signatures = set(events.values())
        if len(signatures) != 1:
            raise ProtocolError(
                f"request sibling closure failed for {request_key}: {events}"
            )
        if expected_layer_ids is not None:
            expected_layers = tuple(sorted(int(value) for value in expected_layer_ids))
            for event_id, signature in events.items():
                observed_layers = tuple(layer_id for layer_id, _ in signature)
                if observed_layers != expected_layers:
                    raise ProtocolError(
                        f"event {event_id} layer closure failed: "
                        f"expected={expected_layers}, observed={observed_layers}"
                    )
    if expected_input_events is not None:
        observed_events: dict[str, set[str]] = {}
        for row in rows:
            observed_events.setdefault(row.request_id, set()).add(row.input_event_id)
        if set(observed_events) != set(expected_input_events):
            raise ProtocolError(
                "request set differs from the frozen input-event manifest: "
                f"expected={sorted(expected_input_events)}, observed={sorted(observed_events)}"
            )
        for request_id, expected in expected_input_events.items():
            expected_set = set(expected)
            if observed_events[request_id] != expected_set:
                raise ProtocolError(
                    f"input-event closure failed for request {request_id}: "
                    f"missing={sorted(expected_set - observed_events[request_id])}, "
                    f"extra={sorted(observed_events[request_id] - expected_set)}"
                )
    event_keys = {
        (row.model, row.phase, row.request_id, row.input_event_id) for row in rows
    }
    return {
        "contributions": len(rows),
        "tokens": len(event_keys),
        "requests": len({(row.model, row.phase, row.request_id) for row in rows}),
    }


def validate_causal_route_v3(
    rows: Sequence[Contribution], *, require_observed_stages: bool = True
) -> dict[str, int]:
    """Validate the route-v3 ledger and autoregressive/layer dependencies.

    A v3 CSV may still be development-only, but a formal consumer must never
    infer missing stage times from row order or synthetic request arrivals.
    Top-k siblings share one event timeline. Within decode, layer ``l+1`` waits
    for layer ``l`` combine and step ``t+1`` waits for the previous step's final
    combine.
    """

    validate_identity_conservation(rows)
    if not rows:
        raise ProtocolError("route-v3 trace is empty")
    request_contracts: dict[tuple[str, str, str], tuple[str, float, float]] = {}
    step_events: dict[tuple[str, str, str, int], str] = {}
    events: dict[
        tuple[str, str, str, str, int], tuple[float, float, float, float, float, float, float]
    ] = {}
    representatives: dict[tuple[str, str, str, int, int], Contribution] = {}
    for row in rows:
        request_key = (row.model, row.phase, row.request_id)
        request_contract = (row.document_id, row.request_arrival_us, row.deadline_us)
        prior_contract = request_contracts.setdefault(request_key, request_contract)
        if prior_contract != request_contract:
            raise ProtocolError(
                f"request {request_key} changes document, arrival, or deadline"
            )
        if abs(row.arrival_us - row.request_arrival_us) > 1e-12:
            raise ProtocolError("legacy arrival_us must equal route-v3 request_arrival_us")
        if not row.legal_replica_set:
            raise ProtocolError("formal route-v3 requires a non-empty legal_replica_set")
        if row.target_replica >= 0 and row.target_replica not in row.legal_replica_set:
            raise ProtocolError("observed target_replica is outside legal_replica_set")
        if require_observed_stages and not row.has_observed_stage_ledger:
            raise ProtocolError("formal route-v3 requires dispatch/expert/combine timestamps")
        if row.phase == "decode":
            step_key = (*request_key, row.decode_step)
            prior_event = step_events.setdefault(step_key, row.input_event_id)
            if prior_event != row.input_event_id:
                raise ProtocolError(
                    f"decode step {row.decode_step} maps to multiple input events for {request_key}"
                )
        event_key = (
            row.model,
            row.phase,
            row.request_id,
            row.input_event_id,
            row.layer_id,
        )
        timeline = (
            row.request_arrival_us,
            row.layer_ready_us,
            row.route_end_us,
            row.dispatch_end_us,
            row.expert_start_us,
            row.expert_end_us,
            row.combine_end_us,
        )
        prior = events.setdefault(event_key, timeline)
        if prior != timeline:
            raise ProtocolError(f"top-k siblings disagree on causal timeline for {event_key}")
        representatives.setdefault(
            (row.model, row.phase, row.request_id, row.decode_step, row.layer_id), row
        )

    by_request_step: dict[tuple[str, str, str], dict[int, list[Contribution]]] = {}
    for (model, phase, request_id, step, _layer), row in representatives.items():
        if phase != "decode":
            continue
        if step < 0:
            raise ProtocolError("decode route-v3 row is missing decode_step")
        by_request_step.setdefault((model, phase, request_id), {}).setdefault(step, []).append(row)
    for request_key, steps in by_request_step.items():
        ordered_steps = sorted(steps)
        if ordered_steps != list(range(ordered_steps[0], ordered_steps[-1] + 1)):
            raise ProtocolError(f"decode steps are not contiguous for {request_key}: {ordered_steps}")
        previous_step_end = -math.inf
        for step in ordered_steps:
            layer_rows = sorted(steps[step], key=lambda item: item.layer_id)
            if layer_rows[0].layer_ready_us < previous_step_end - 1e-12:
                raise ProtocolError(
                    f"decode step {step} becomes ready before prior combine for {request_key}"
                )
            previous_layer_end = -math.inf
            for row in layer_rows:
                if row.layer_ready_us < previous_layer_end - 1e-12:
                    raise ProtocolError(
                        f"layer {row.layer_id} becomes ready before prior layer combine for {request_key}"
                    )
                if row.has_observed_stage_ledger:
                    previous_layer_end = row.combine_end_us
            if layer_rows[-1].has_observed_stage_ledger:
                previous_step_end = layer_rows[-1].combine_end_us
    return {
        "contributions": len(rows),
        "events": len(events),
        "requests": len(request_contracts),
        "documents": len({contract[0] for contract in request_contracts.values()}),
    }


def validate_stage_identity_conservation(
    routed: Sequence[Contribution],
    dispatched: Sequence[Contribution],
    executed: Sequence[Contribution],
    combined: Sequence[Contribution],
    *,
    require_assigned_target: bool = False,
) -> dict[str, int]:
    """Assert exact contribution conservation across the execution stages.

    The target replica is deliberately excluded from ``contribution_id``: it
    is an actuator decision, not part of the routed mathematical contribution.
    Every stage must nevertheless carry the same immutable route identity.
    """

    stages = {
        "routed": routed,
        "dispatched": dispatched,
        "executed": executed,
        "combined": combined,
    }
    identities: dict[str, tuple[str, ...]] = {}
    for name, values in stages.items():
        validate_identity_conservation(values)
        identities[name] = tuple(sorted(item.route_semantic_id for item in values))
    reference = identities["routed"]
    for name in ("dispatched", "executed", "combined"):
        if identities[name] != reference:
            missing = sorted(set(reference) - set(identities[name]))
            extra = sorted(set(identities[name]) - set(reference))
            raise ProtocolError(
                f"identity conservation failed at {name}: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
    target_maps = {
        name: {
            item.contribution_id: item.target_replica
            for item in stages[name]
        }
        for name in ("dispatched", "executed", "combined")
    }
    for contribution_id in target_maps["dispatched"]:
        targets = {
            target_maps[name][contribution_id]
            for name in ("dispatched", "executed", "combined")
        }
        if len(targets) != 1:
            raise ProtocolError(
                "identity conservation failed: target replica changed after dispatch"
            )
        if require_assigned_target and next(iter(targets)) < 0:
            raise ProtocolError(
                "identity conservation failed: formal stage has unassigned target replica"
            )
    return {name: len(values) for name, values in stages.items()}


@dataclass(frozen=True)
class CurvePoint:
    rows: int
    median_us: float
    p95_us: float

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.median_us <= 0 or self.p95_us < self.median_us:
            raise ValueError("invalid service-curve point")


class ServiceCatalog:
    """Fail-closed interpolation of measured expert service curves."""

    def __init__(self, points: Mapping[tuple[str, int], Sequence[CurvePoint]]) -> None:
        self._points: dict[tuple[str, int], tuple[CurvePoint, ...]] = {}
        for key, values in points.items():
            ordered = tuple(sorted(values, key=lambda point: point.rows))
            if not ordered or len({point.rows for point in ordered}) != len(ordered):
                raise ProtocolError(f"invalid or duplicate curve rows for {key}")
            if any(b.median_us < a.median_us for a, b in zip(ordered, ordered[1:])):
                raise ProtocolError(f"non-monotone median curve for {key}")
            self._points[key] = ordered
        if not self._points:
            raise ProtocolError("service catalog is empty")

    @classmethod
    def from_csv(cls, path: str | Path) -> "ServiceCatalog":
        grouped: dict[tuple[str, int], list[CurvePoint]] = {}
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"model", "layer", "rows", "median_us", "p95_us"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ProtocolError(f"service curve missing columns {sorted(missing)}")
            for row in reader:
                key = (str(row["model"]), int(row["layer"]))
                grouped.setdefault(key, []).append(
                    CurvePoint(int(row["rows"]), float(row["median_us"]), float(row["p95_us"]))
                )
        return cls(grouped)

    def keys(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._points))

    def estimate_us(self, model: str, layer: int, rows: int, *, conservative: bool = False) -> float:
        if rows <= 0:
            raise ValueError("rows must be positive")
        points = self._points.get((model, layer)) or self._points.get((model, -1))
        if points is None:
            raise ProtocolError(f"no service curve for model={model!r}, layer={layer}")
        if rows < points[0].rows or rows > points[-1].rows:
            raise ProtocolError(
                f"rows={rows} outside measured curve [{points[0].rows},{points[-1].rows}] "
                f"for model={model!r}, layer={layer}"
            )
        field = "p95_us" if conservative else "median_us"
        for point in points:
            if rows == point.rows:
                return float(getattr(point, field))
        for left, right in zip(points, points[1:]):
            if left.rows < rows < right.rows:
                ratio = (rows - left.rows) / (right.rows - left.rows)
                return float(getattr(left, field)) + ratio * (
                    float(getattr(right, field)) - float(getattr(left, field))
                )
        raise AssertionError("in-range interpolation failed")


def stable_index(material: str, modulo: int, *, seed: int = 0) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    digest = hashlib.sha256(f"{seed}|{material}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def percentile(values: Sequence[float], q: float) -> float:
    if not values or not 0 <= q <= 1:
        raise ValueError("percentile needs non-empty values and q in [0,1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def bootstrap_mean_ci(
    values: Sequence[float], *, replicates: int = 2000, seed: int = 20260725
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap values cannot be empty")
    point = sum(values) / len(values)
    if len(values) == 1 or replicates <= 0:
        return point, point, point
    rng = random.Random(seed)
    means = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    ]
    return point, percentile(means, 0.025), percentile(means, 0.975)


def clustered_bootstrap_mean_ci(
    values: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    replicates: int = 2000,
    seed: int = 20260725,
) -> tuple[float, float, float]:
    if len(values) != len(cluster_ids) or not values:
        raise ValueError("clustered bootstrap needs equal non-empty values and cluster ids")
    grouped: dict[str, list[float]] = {}
    for value, cluster_id in zip(values, cluster_ids):
        grouped.setdefault(cluster_id, []).append(float(value))
    cluster_means = [sum(items) / len(items) for _, items in sorted(grouped.items())]
    return bootstrap_mean_ci(cluster_means, replicates=replicates, seed=seed)


@dataclass(frozen=True)
class ReplayConfig:
    replica_count: int
    hold_us: float = 0.0
    remote_latency_us: float = 0.0
    remote_bytes_per_row: int = 0
    conservative_curve: bool = False
    hold_by_queue: Mapping[tuple[int, int], float] = field(default_factory=dict)
    max_batch_rows: int | None = None
    controller_latency_us: float = 0.0
    seal_cost_us: float = 0.0
    launch_cost_us: float = 0.0

    def __post_init__(self) -> None:
        if self.replica_count < 2:
            raise ValueError("BCRD requires at least two replicas")
        if (
            self.hold_us < 0
            or self.remote_latency_us < 0
            or self.remote_bytes_per_row < 0
            or self.controller_latency_us < 0
            or self.seal_cost_us < 0
            or self.launch_cost_us < 0
        ):
            raise ValueError("replay costs must be non-negative")
        if self.max_batch_rows is not None and self.max_batch_rows <= 0:
            raise ValueError("max_batch_rows must be positive when specified")
        for key, value in self.hold_by_queue.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(isinstance(part, bool) or not isinstance(part, int) or part < 0 for part in key)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError("hold_by_queue must map (replica, expert) to non-negative hold")

    def hold_for(self, replica: int, expert: int) -> float:
        return float(self.hold_by_queue.get((replica, expert), self.hold_us))


class CausalReplayEngine:
    """Discrete-event expert queue used by both online policies and Oracle.

    Dispatch choices may be submitted ahead of time by an offline evaluator,
    but batching itself is strictly event driven: destination arrivals are
    processed before same-time seals, a singleton pays the complete hold, and
    launched rows are removed from the open queue immediately.
    """

    _EPS = 1e-12

    def __init__(
        self,
        catalog: ServiceCatalog,
        config: ReplayConfig,
        *,
        model: str,
        layer: int,
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.model = model
        self.layer = layer
        self.now_us = 0.0
        self.inflight: list[dict[str, object]] = []
        self.open_batches: dict[tuple[int, int], dict[str, object]] = {}
        self.sealed: dict[int, list[dict[str, object]]] = {
            replica: [] for replica in range(config.replica_count)
        }
        self.running: list[dict[str, object] | None] = [None] * config.replica_count
        self.completion_us: dict[int, float] = {}
        self.submitted: dict[int, tuple[Contribution, int]] = {}
        self.batch_records: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []
        self.total_service_us = 0.0
        self.total_launch_cost_us = 0.0
        self.remote_assignments = 0
        self.last_finish_us = [0.0] * config.replica_count

    def clone(self) -> "CausalReplayEngine":
        return copy.deepcopy(self)

    def submit(
        self,
        index: int,
        item: Contribution,
        replica: int,
        *,
        hold_us: float | None = None,
        decision_end_us: float | None = None,
    ) -> None:
        if index in self.submitted:
            raise ProtocolError(f"duplicate replay contribution index {index}")
        if item.model != self.model or item.layer != self.layer:
            raise ProtocolError("one replay engine must contain one model and one layer")
        legal = item.legal_replicas(self.config.replica_count)
        if replica not in legal:
            raise ProtocolError(
                f"illegal replica {replica} for contribution {index}; legal={legal}"
            )
        hold = self.config.hold_for(replica, item.expert_id) if hold_us is None else float(hold_us)
        if not math.isfinite(hold) or hold < 0:
            raise ProtocolError("hold action must be finite and non-negative")
        earliest_dispatch = item.dispatch_ready_us + self.config.controller_latency_us
        dispatch_end = earliest_dispatch if decision_end_us is None else float(decision_end_us)
        if dispatch_end < earliest_dispatch - self._EPS:
            raise ProtocolError("decision_end_us precedes route/controller readiness")
        remote = replica != item.source_rank
        destination_ready = dispatch_end + (self.config.remote_latency_us if remote else 0.0)
        self.submitted[index] = (item, replica)
        self.inflight.append(
            {
                "index": index,
                "item": item,
                "replica": replica,
                "hold_us": hold,
                "dispatch_end_us": dispatch_end,
                "ready_us": destination_ready,
            }
        )
        if remote:
            self.remote_assignments += 1
        self.events.append(
            {
                "event": "DISPATCH_END",
                "time_us": dispatch_end,
                "index": index,
                "replica": replica,
            }
        )

    def _next_event_time(self) -> float | None:
        values: list[float] = []
        values.extend(float(row["ready_us"]) for row in self.inflight)
        values.extend(float(row["seal_at_us"]) for row in self.open_batches.values())
        values.extend(
            float(row["finish_us"]) for row in self.running if row is not None
        )
        for replica, queue in self.sealed.items():
            if self.running[replica] is None:
                values.extend(float(row["ready_us"]) for row in queue)
        return min(values) if values else None

    def _seal(self, key: tuple[int, int], time_us: float, reason: str) -> None:
        batch = self.open_batches.pop(key)
        batch["seal_us"] = time_us
        batch["ready_us"] = time_us + self.config.seal_cost_us
        batch["seal_reason"] = reason
        self.sealed[key[0]].append(batch)
        self.events.append(
            {
                "event": {
                    "timeout": "SEAL_TIMEOUT",
                    "max_batch": "SEAL_MAX_BATCH",
                    "deadline": "SEAL_DEADLINE_CAP",
                }[reason],
                "time_us": time_us,
                "replica": key[0],
                "expert_id": key[1],
                "rows": len(batch["indices"]),
            }
        )

    def _launch_idle(self, time_us: float) -> None:
        for replica in range(self.config.replica_count):
            if self.running[replica] is not None:
                continue
            ready = [
                batch
                for batch in self.sealed[replica]
                if float(batch["ready_us"]) <= time_us + self._EPS
            ]
            if not ready:
                continue
            batch = min(
                ready,
                key=lambda row: (
                    float(row["deadline_us"]),
                    float(row["ready_us"]),
                    tuple(sorted(row["request_ids"])),
                    tuple(row["indices"]),
                ),
            )
            self.sealed[replica].remove(batch)
            service = self.catalog.estimate_us(
                self.model,
                self.layer,
                len(batch["indices"]),
                conservative=self.config.conservative_curve,
            )
            finish = time_us + self.config.launch_cost_us + service
            batch["start_us"] = time_us
            batch["finish_us"] = finish
            batch["service_us"] = service
            self.running[replica] = batch
            self.total_service_us += service
            self.total_launch_cost_us += self.config.launch_cost_us
            self.events.append(
                {
                    "event": "BATCH_LAUNCH",
                    "time_us": time_us,
                    "replica": replica,
                    "expert_id": batch["expert_id"],
                    "rows": len(batch["indices"]),
                }
            )

    def _process_time(self, time_us: float) -> None:
        if time_us < self.now_us - self._EPS:
            raise AssertionError("event time moved backwards")
        self.now_us = time_us

        # Finish first so a batch completing at t releases the executor before
        # arrivals and seals at t are considered for the next launch.
        for replica, running in enumerate(self.running):
            if running is None or float(running["finish_us"]) > time_us + self._EPS:
                continue
            for index in running["indices"]:
                self.completion_us[int(index)] = time_us
            self.batch_records.append(dict(running))
            self.running[replica] = None
            self.last_finish_us[replica] = time_us
            self.events.append(
                {
                    "event": "EXPERT_FINISH",
                    "time_us": time_us,
                    "replica": replica,
                    "expert_id": running["expert_id"],
                    "rows": len(running["indices"]),
                }
            )

        arrivals = [
            row for row in self.inflight if float(row["ready_us"]) <= time_us + self._EPS
        ]
        self.inflight = [row for row in self.inflight if row not in arrivals]
        for row in sorted(arrivals, key=lambda value: int(value["index"])):
            item = row["item"]
            assert isinstance(item, Contribution)
            replica = int(row["replica"])
            key = (replica, item.expert_id)
            batch = self.open_batches.get(key)
            if batch is None:
                batch = {
                    "replica": replica,
                    "expert_id": item.expert_id,
                    "indices": [],
                    "request_ids": set(),
                    "first_ready_us": time_us,
                    "seal_at_us": time_us + float(row["hold_us"]),
                    "requested_seal_at_us": time_us + float(row["hold_us"]),
                    "deadline_us": item.deadline_us,
                    "deadline_capped": False,
                }
                self.open_batches[key] = batch
            batch["indices"].append(int(row["index"]))
            batch["request_ids"].add(item.request_id)
            batch["deadline_us"] = min(float(batch["deadline_us"]), item.deadline_us)
            service_lower_bound = self.catalog.estimate_us(
                self.model,
                self.layer,
                len(batch["indices"]),
                conservative=self.config.conservative_curve,
            )
            deadline_seal_cap = max(
                time_us,
                float(batch["deadline_us"])
                - self.config.seal_cost_us
                - self.config.launch_cost_us
                - service_lower_bound,
            )
            if deadline_seal_cap < float(batch["seal_at_us"]) - self._EPS:
                batch["seal_at_us"] = deadline_seal_cap
                batch["deadline_capped"] = True
            self.events.append(
                {
                    "event": "CONTRIBUTION_ARRIVAL",
                    "time_us": time_us,
                    "index": int(row["index"]),
                    "replica": replica,
                    "expert_id": item.expert_id,
                }
            )
            if (
                self.config.max_batch_rows is not None
                and len(batch["indices"]) == self.config.max_batch_rows
            ):
                # max_batch is an observed causal early-seal action. Seal at
                # the exact bound before another same-time arrival can create
                # an oversized batch; subsequent rows open a fresh batch.
                self._seal(key, time_us, "max_batch")

        # All arrivals at t join before hold=0 timeouts at t fire, except that
        # the explicit max_batch bound above seals at the exact row limit.
        for key, batch in list(self.open_batches.items()):
            timed_out = float(batch["seal_at_us"]) <= time_us + self._EPS
            if timed_out:
                self._seal(
                    key,
                    time_us,
                    "deadline" if bool(batch["deadline_capped"]) else "timeout",
                )
        self._launch_idle(time_us)

    def advance_to(self, target_us: float) -> None:
        if target_us < self.now_us - self._EPS:
            raise ProtocolError("cannot rewind causal replay")
        while True:
            next_time = self._next_event_time()
            if next_time is None or next_time > target_us + self._EPS:
                break
            self._process_time(next_time)
        self.now_us = max(self.now_us, target_us)

    def run_all(self) -> None:
        iterations = 0
        while self._next_event_time() is not None:
            next_time = self._next_event_time()
            assert next_time is not None
            self._process_time(next_time)
            iterations += 1
            if iterations > 10_000_000:
                raise AssertionError("causal replay failed to make progress")
        if len(self.completion_us) != len(self.submitted):
            missing = sorted(set(self.submitted) - set(self.completion_us))
            raise AssertionError(f"replay lost contributions: {missing[:3]}")

    def projected_available_us(self) -> list[float]:
        projected = self.clone()
        projected.run_all()
        return list(projected.last_finish_us)

    def predict_submission(
        self,
        item: Contribution,
        replica: int,
        *,
        hold_us: float,
        decision_end_us: float | None = None,
    ) -> dict[str, float]:
        baseline = self.clone()
        baseline.run_all()
        trial = self.clone()
        prediction_index = min([-1, *(index - 1 for index in trial.submitted if index < 0)])
        trial.submit(
            prediction_index,
            item,
            replica,
            hold_us=hold_us,
            decision_end_us=decision_end_us,
        )
        trial.run_all()
        record = next(
            batch for batch in trial.batch_records if prediction_index in batch["indices"]
        )
        return {
            "completion_us": trial.completion_us[prediction_index],
            "batch_rows": float(len(record["indices"])),
            "marginal_service_us": trial.total_service_us - baseline.total_service_us,
            "projected_replica_finish_us": trial.last_finish_us[replica],
        }

    def metrics(self) -> dict[str, object]:
        if not self.submitted or len(self.completion_us) != len(self.submitted):
            raise ProtocolError("metrics require a complete non-empty replay")
        request_completion: dict[str, float] = {}
        request_arrival: dict[str, float] = {}
        request_deadline: dict[str, float] = {}
        for index, (item, _replica) in self.submitted.items():
            request_completion[item.request_id] = max(
                request_completion.get(item.request_id, -math.inf), self.completion_us[index]
            )
            request_arrival[item.request_id] = min(
                request_arrival.get(item.request_id, math.inf), item.request_arrival_us
            )
            request_deadline[item.request_id] = min(
                request_deadline.get(item.request_id, math.inf), item.deadline_us
            )
        latencies = [request_completion[key] - request_arrival[key] for key in request_completion]
        on_time = sum(request_completion[key] <= request_deadline[key] for key in request_completion)
        return {
            "requests": len(request_completion),
            "contributions": len(self.submitted),
            "on_time": on_time,
            "slo_attainment": on_time / len(request_completion),
            "mean_completion_us": sum(latencies) / len(latencies),
            "p50_completion_us": percentile(latencies, 0.50),
            "p95_completion_us": percentile(latencies, 0.95),
            "p99_completion_us": percentile(latencies, 0.99),
            "makespan_us": max(request_completion.values()) - min(request_arrival.values()),
            "total_service_us": self.total_service_us,
            "launch_cost_us": self.total_launch_cost_us,
            "launches": len(self.batch_records),
            "remote_assignments": self.remote_assignments,
            "remote_bytes": self.remote_assignments * self.config.remote_bytes_per_row,
            "request_completion_us": request_completion,
            "contribution_completion_us": dict(self.completion_us),
            "batch_records": [
                {
                    **{key: value for key, value in row.items() if key != "request_ids"},
                    "request_ids": sorted(row["request_ids"]),
                }
                for row in self.batch_records
            ],
            "event_count": len(self.events),
        }


def simulate_assignment(
    contributions: Sequence[Contribution],
    assignments: Sequence[int],
    catalog: ServiceCatalog,
    config: ReplayConfig,
) -> dict[str, object]:
    """Replay one layer with causal seal/launch/finish events and fork-join."""
    if not contributions or len(contributions) != len(assignments):
        raise ProtocolError("assignment must cover every contribution exactly once")
    model = contributions[0].model
    layer = contributions[0].layer
    if any(item.model != model or item.layer != layer for item in contributions):
        raise ProtocolError("one replay instance must contain one model and one layer")
    engine = CausalReplayEngine(catalog, config, model=model, layer=layer)
    decision_cursor_us = 0.0
    ordered = sorted(
        enumerate(zip(contributions, assignments)),
        key=lambda value: (
            value[1][0].dispatch_ready_us,
            value[1][0].deadline_us,
            value[1][0].contribution_id,
        ),
    )
    for index, (item, replica) in ordered:
        decision_end_us = (
            max(item.dispatch_ready_us, decision_cursor_us) + config.controller_latency_us
        )
        engine.submit(index, item, int(replica), decision_end_us=decision_end_us)
        decision_cursor_us = decision_end_us
    engine.run_all()
    return engine.metrics()


def objective_key(metrics: Mapping[str, object]) -> tuple[float, ...]:
    return (
        float(metrics["on_time"]),
        -float(metrics["p99_completion_us"]),
        -float(metrics["mean_completion_us"]),
        -float(metrics["total_service_us"]),
        -float(metrics["remote_assignments"]),
    )


def relative_latency_gain(baseline: float, candidate: float) -> float:
    if baseline <= 0 or candidate < 0 or not math.isfinite(baseline + candidate):
        raise ValueError("latency gain needs finite non-negative values and positive baseline")
    return (baseline - candidate) / baseline


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_instances(path: str | Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ProtocolError(f"{path}:{lineno}: invalid JSON") from exc
    if not rows:
        raise ProtocolError("instance file is empty")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
