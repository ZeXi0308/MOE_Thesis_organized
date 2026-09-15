from __future__ import annotations

"""Fail-closed protocol primitives for RouteSlack-MoE.

This module is intentionally hardware-agnostic.  Passing its unit tests proves
only schema, accounting, and replay invariants; it never authorizes a formal
GPU result or substitutes for a native continuous-decode backend.
"""

from collections import defaultdict
from dataclasses import dataclass, fields
import math
import random
from typing import Callable, Iterable, Sequence, TypeVar


class ProtocolError(RuntimeError):
    """Raised when evidence would otherwise become scientifically ambiguous."""


STAGES = ("routed", "dispatched", "executed", "combined")


@dataclass(frozen=True, order=True)
class ContributionIdentity:
    request_id: str
    input_event_id: str
    token_id: int
    decode_step: int
    layer_id: int
    expert_id: int
    topk_slot: int
    source_rank: int
    target_replica: int

    def __post_init__(self) -> None:
        if not self.request_id or not self.input_event_id:
            raise ProtocolError("request_id and input_event_id must be non-empty")
        for name in (
            "token_id",
            "decode_step",
            "layer_id",
            "expert_id",
            "topk_slot",
            "source_rank",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.target_replica, bool)
            or not isinstance(self.target_replica, int)
            or self.target_replica < -1
        ):
            raise ProtocolError("target_replica must be an integer >= -1")

    @property
    def contribution_key(self) -> tuple[object, ...]:
        """Immutable contribution identity; target replica is an actuator."""

        return (
            self.request_id,
            self.input_event_id,
            self.token_id,
            self.decode_step,
            self.layer_id,
            self.expert_id,
            self.topk_slot,
            self.source_rank,
        )

    @property
    def token_layer_key(self) -> tuple[object, ...]:
        return (
            self.request_id,
            self.input_event_id,
            self.token_id,
            self.decode_step,
            self.layer_id,
            self.source_rank,
        )

    def as_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


class IdentityLedger:
    """Conserves each exact top-k contribution across the four serving stages."""

    def __init__(self, *, expected_top_k: int) -> None:
        if isinstance(expected_top_k, bool) or expected_top_k <= 0:
            raise ProtocolError("expected_top_k must be a positive integer")
        self.expected_top_k = int(expected_top_k)
        self._stages: dict[str, tuple[ContributionIdentity, ...]] = {}

    def record(
        self, stage: str, contributions: Iterable[ContributionIdentity]
    ) -> None:
        if stage not in STAGES:
            raise ProtocolError(f"unknown identity stage {stage!r}")
        if stage in self._stages:
            raise ProtocolError(f"identity stage {stage!r} was already recorded")
        rows = tuple(contributions)
        keys = [row.contribution_key for row in rows]
        if len(keys) != len(set(keys)):
            raise ProtocolError(f"duplicate contribution identity at stage {stage}")
        if stage != "routed" and any(row.target_replica < 0 for row in rows):
            raise ProtocolError(f"stage {stage} contains an unassigned target replica")

        grouped: dict[tuple[object, ...], list[ContributionIdentity]] = defaultdict(list)
        for row in rows:
            grouped[row.token_layer_key].append(row)
        expected_slots = set(range(self.expected_top_k))
        for key, group in grouped.items():
            slots = [row.topk_slot for row in group]
            if len(slots) != self.expected_top_k or set(slots) != expected_slots:
                raise ProtocolError(
                    f"top-k slots are incomplete or duplicated at stage {stage}: "
                    f"key={key!r}, slots={slots!r}, expected={sorted(expected_slots)!r}"
                )
        self._stages[stage] = rows

    def assert_conserved(self) -> None:
        missing = [stage for stage in STAGES if stage not in self._stages]
        if missing:
            raise ProtocolError(f"identity conservation missing stages {missing!r}")
        reference = {row.contribution_key for row in self._stages[STAGES[0]]}
        for stage in STAGES[1:]:
            observed = {row.contribution_key for row in self._stages[stage]}
            if observed != reference:
                missing_rows = len(reference - observed)
                extra_rows = len(observed - reference)
                raise ProtocolError(
                    "identity conservation failed: "
                    f"stage={stage}, missing={missing_rows}, extra={extra_rows}"
                )
        target_maps = {
            stage: {
                row.contribution_key: row.target_replica
                for row in self._stages[stage]
            }
            for stage in STAGES[1:]
        }
        for key in reference:
            if len({target_maps[stage][key] for stage in STAGES[1:]}) != 1:
                raise ProtocolError(
                    "identity conservation failed: target replica changed after dispatch"
                )


def assert_cached_decode_equivalence(
    cached_logits: Sequence[Sequence[float]],
    recomputed_logits: Sequence[Sequence[float]],
    *,
    atol: float,
    rtol: float,
) -> None:
    """Compare per-step logits without silently truncating either sequence."""

    if atol < 0 or rtol < 0 or not math.isfinite(atol + rtol):
        raise ProtocolError("cache equivalence tolerances must be finite and non-negative")
    if len(cached_logits) != len(recomputed_logits):
        raise ProtocolError(
            "cache equivalence failed: decode-step count differs "
            f"({len(cached_logits)} != {len(recomputed_logits)})"
        )
    for step, (cached, full) in enumerate(zip(cached_logits, recomputed_logits)):
        if len(cached) != len(full):
            raise ProtocolError(f"cache equivalence failed at step {step}: logit width differs")
        for index, (left, right) in enumerate(zip(cached, full)):
            if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol):
                raise ProtocolError(
                    "cache equivalence failed: "
                    f"step={step}, logit={index}, cached={left}, recomputed={right}"
                )


@dataclass(frozen=True)
class DeadlineState:
    deadline_ns: int
    now_ns: int
    predicted_remaining_ns: int

    def __post_init__(self) -> None:
        if min(self.deadline_ns, self.now_ns, self.predicted_remaining_ns) < 0:
            raise ProtocolError("deadline state values must be non-negative")

    @property
    def slack_ns(self) -> int:
        return self.deadline_ns - self.now_ns - self.predicted_remaining_ns

    def advance(self, *, now_ns: int, predicted_remaining_ns: int) -> "DeadlineState":
        if now_ns < self.now_ns:
            raise ProtocolError("deadline clock moved backwards")
        return DeadlineState(self.deadline_ns, now_ns, predicted_remaining_ns)


@dataclass(frozen=True)
class SurfacePoint:
    rows: int
    tier: str
    latency_us: float
    raw_energy_j: float

    def __post_init__(self) -> None:
        if self.rows <= 0 or not self.tier:
            raise ProtocolError("surface rows/tier must be positive/non-empty")
        if (
            not math.isfinite(self.latency_us)
            or self.latency_us <= 0
            or not math.isfinite(self.raw_energy_j)
            or self.raw_energy_j <= 0
        ):
            raise ProtocolError("surface latency and raw energy must be finite and positive")


@dataclass(frozen=True)
class SurfaceLookup:
    point: SurfacePoint
    status: str
    action_eligible: bool


class ServiceEnergySurface:
    """Exact/ceiling lookup with a mandatory default-action fallback."""

    def __init__(self, points: Iterable[SurfacePoint], *, default_tier: str) -> None:
        self.default_tier = default_tier
        self._by_tier: dict[str, list[SurfacePoint]] = defaultdict(list)
        seen: set[tuple[str, int]] = set()
        for point in points:
            key = (point.tier, point.rows)
            if key in seen:
                raise ProtocolError(f"duplicate surface point {key!r}")
            seen.add(key)
            self._by_tier[point.tier].append(point)
        for tier in self._by_tier:
            self._by_tier[tier].sort(key=lambda row: row.rows)
        if default_tier not in self._by_tier:
            raise ProtocolError("surface has no default tier")

    def _default(self, rows: int) -> SurfaceLookup:
        points = self._by_tier[self.default_tier]
        point = next((row for row in points if row.rows >= rows), points[-1])
        return SurfaceLookup(point, "FALLBACK_DEFAULT", False)

    def lookup(self, *, rows: int, tier: str) -> SurfaceLookup:
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ProtocolError("surface lookup rows must be a positive integer")
        points = self._by_tier.get(tier)
        if points is None or rows > points[-1].rows:
            return self._default(rows)
        point = next(row for row in points if row.rows >= rows)
        status = "EXACT" if point.rows == rows else "CONSERVATIVE_CEILING"
        return SurfaceLookup(point, status, True)


@dataclass(frozen=True)
class OnlineObservation:
    now_ns: int
    queue_depth: int
    visible_rows: tuple[int, ...]


@dataclass(frozen=True)
class OracleInput:
    online: OnlineObservation
    future_arrival_ns: tuple[int, ...]


T = TypeVar("T")


def run_online_policy(
    policy: Callable[[OnlineObservation], T], observation: OnlineObservation
) -> T:
    if type(observation) is not OnlineObservation:
        raise ProtocolError("future-known Oracle input cannot enter an online policy")
    return policy(observation)


@dataclass(frozen=True)
class CompletionSet:
    token_keys: frozenset[tuple[str, int]]
    output_sha256: str

    def __post_init__(self) -> None:
        if not self.token_keys:
            raise ProtocolError("completed-token set must be non-empty")
        if len(self.output_sha256) != 64:
            raise ProtocolError("output_sha256 must be a full SHA-256 digest")


def assert_matched_completion(left: CompletionSet, right: CompletionSet) -> None:
    if left.token_keys != right.token_keys:
        raise ProtocolError("completed-token identity sets do not match")
    if left.output_sha256 != right.output_sha256:
        raise ProtocolError("exact output hashes do not match")


def counter_delta(start_j: float, end_j: float, *, modulus_j: float | None = None) -> float:
    if not math.isfinite(start_j) or not math.isfinite(end_j) or min(start_j, end_j) < 0:
        raise ProtocolError("energy counters must be finite and non-negative")
    if end_j >= start_j:
        return end_j - start_j
    if modulus_j is None:
        raise ProtocolError("energy counter wraparound has no declared modulus")
    if not math.isfinite(modulus_j) or modulus_j <= start_j or end_j >= modulus_j:
        raise ProtocolError("invalid energy counter wraparound modulus")
    return modulus_j - start_j + end_j


@dataclass(frozen=True)
class EnergyNormalization:
    raw_energy_j_per_repeat: float
    work_energy_j_per_repeat: float
    meter_overhead_j: float


def normalize_trial(
    raw_board_energy_j: float, repeats: int, meter_overhead_j: float = 0.0
) -> EnergyNormalization:
    if (
        not math.isfinite(raw_board_energy_j)
        or raw_board_energy_j < 0
        or isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats <= 0
        or not math.isfinite(meter_overhead_j)
        or meter_overhead_j < 0
        or meter_overhead_j > raw_board_energy_j
    ):
        raise ProtocolError("invalid energy/repeat/meter-overhead values")
    return EnergyNormalization(
        raw_energy_j_per_repeat=raw_board_energy_j / repeats,
        work_energy_j_per_repeat=(raw_board_energy_j - meter_overhead_j) / repeats,
        meter_overhead_j=meter_overhead_j,
    )


@dataclass(frozen=True)
class ThermalPair:
    pair_id: str
    order: str
    temperature_start_c: float
    temperature_end_c: float


@dataclass(frozen=True)
class EnergyTrial:
    strategy: str
    pair_id: str
    order: str
    raw_board_energy_j: float
    repeats: int
    completed: CompletionSet
    temperature_start_c: float
    temperature_end_c: float


def assert_ab_ba_pairs(
    trials: Sequence[EnergyTrial | ThermalPair], *, max_temperature_delta_c: float
) -> None:
    if not trials:
        raise ProtocolError("AB/BA validation requires trials")
    if max_temperature_delta_c < 0 or not math.isfinite(max_temperature_delta_c):
        raise ProtocolError("thermal limit must be finite and non-negative")
    orders = {trial.order for trial in trials}
    if orders != {"AB", "BA"}:
        raise ProtocolError("AB/BA pairing requires both AB and BA orders")
    for trial in trials:
        if trial.order not in {"AB", "BA"}:
            raise ProtocolError(f"invalid AB/BA order {trial.order!r}")
        delta = abs(trial.temperature_end_c - trial.temperature_start_c)
        if delta > max_temperature_delta_c:
            raise ProtocolError(
                f"thermal gate failed for {trial.pair_id}: delta_c={delta:.3f}"
            )

    energy_trials = [trial for trial in trials if isinstance(trial, EnergyTrial)]
    if not energy_trials:
        return
    grouped: dict[str, list[EnergyTrial]] = defaultdict(list)
    for trial in energy_trials:
        grouped[trial.pair_id].append(trial)
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or len({row.strategy for row in pair}) != 2:
            raise ProtocolError(f"AB/BA pair {pair_id!r} must contain two strategies")
        if pair[0].repeats != pair[1].repeats:
            raise ProtocolError(f"AB/BA pair {pair_id!r} has unequal repeat denominators")
        assert_matched_completion(pair[0].completed, pair[1].completed)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def paired_bootstrap_mean_ci(
    paired_differences: Sequence[float], *, replicates: int, seed: int
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in paired_differences)
    if not values or replicates < 100 or any(not math.isfinite(value) for value in values):
        raise ProtocolError("paired bootstrap requires finite pairs and at least 100 replicates")
    point = sum(values) / len(values)
    rng = random.Random(seed)
    draws = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    ]
    return point, _percentile(draws, 0.025), _percentile(draws, 0.975)


@dataclass(frozen=True)
class Gate0Evidence:
    native_continuous_decode: bool
    kv_advances_one: bool
    route_identity_complete: bool
    latency_window_aligned: bool
    energy_window_aligned: bool
    warmup_excluded: bool
    repeat_denominator_equal: bool
    thermal_state_logged: bool
    matched_completion_set: bool
    output_exactness: bool
    oracle_isolated: bool

    @classmethod
    def all_true(cls) -> "Gate0Evidence":
        return cls(**{field.name: True for field in fields(cls)})


@dataclass(frozen=True)
class GateResult:
    status: str
    open_items: tuple[str, ...]


def evaluate_gate0(evidence: Gate0Evidence) -> GateResult:
    open_items = tuple(
        field.name for field in fields(evidence) if not getattr(evidence, field.name)
    )
    return GateResult("FAIL" if open_items else "PASS", open_items)


def no_op_tax_ratio(*, default_cost: float, no_op_cost: float, proposed_cost: float) -> float:
    values = (default_cost, no_op_cost, proposed_cost)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ProtocolError("no-op tax inputs must be finite and non-negative")
    gross_saving = default_cost - proposed_cost
    if gross_saving <= 0:
        raise ProtocolError("no-op tax is undefined without positive gross saving")
    return max(no_op_cost - default_cost, 0.0) / gross_saving
