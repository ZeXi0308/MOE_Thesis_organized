from __future__ import annotations

"""CPU-testable RouteSlack Gate-0 contracts.

These helpers validate artifact shapes and accounting invariants.  They do not
authorize a formal result: a caller must still supply native GPU artifacts and
the repository's external provenance/sign-off checks.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence


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
            raise ValueError("request_id and input_event_id must be non-empty")
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
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.target_replica, bool)
            or not isinstance(self.target_replica, int)
            or self.target_replica < -1
        ):
            raise ValueError("target_replica must be an integer >= -1")

    @property
    def contribution_key(self) -> tuple[object, ...]:
        """Immutable mathematical identity; assignment is an action, not identity."""

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

    @property
    def output_token_key(self) -> tuple[object, ...]:
        """One logical output token, independent of layer and route siblings."""

        return (
            self.request_id,
            self.input_event_id,
            self.token_id,
            self.decode_step,
        )


@dataclass(frozen=True)
class DecodeStepAudit:
    decode_step: int
    input_token_ids: tuple[int, ...]
    token_id: int
    kv_length_before: int
    kv_length_after: int
    cached_logits: tuple[float, ...]
    full_recompute_logits: tuple[float, ...]
    route_ids: tuple[str, ...]


def validate_cached_decode_audit(
    steps: Sequence[DecodeStepAudit],
    *,
    prompt_length: int,
    contributions_per_step: int,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, int]:
    if not steps or prompt_length <= 0 or contributions_per_step <= 0:
        raise RuntimeError("decode audit parameters must be positive and non-empty")
    all_routes: list[str] = []
    for expected_step, step in enumerate(steps):
        if step.decode_step != expected_step:
            raise RuntimeError("decode steps must be contiguous from zero")
        if len(step.input_token_ids) != 1:
            raise RuntimeError("cached decode must consume one new token per step")
        expected_before = prompt_length + expected_step
        if (
            step.kv_length_before != expected_before
            or step.kv_length_after != expected_before + 1
        ):
            raise RuntimeError("KV cache did not advance by exactly one token")
        if len(step.cached_logits) != len(step.full_recompute_logits):
            raise RuntimeError("cached/full logits mismatch in shape")
        if any(
            not math.isclose(left, right, abs_tol=atol, rel_tol=rtol)
            for left, right in zip(step.cached_logits, step.full_recompute_logits)
        ):
            raise RuntimeError("cached/full logits mismatch")
        if len(step.route_ids) != contributions_per_step:
            raise RuntimeError("decode route contribution count mismatch")
        if len(step.route_ids) != len(set(step.route_ids)):
            raise RuntimeError("duplicate route contribution within decode step")
        all_routes.extend(step.route_ids)
    if len(all_routes) != len(set(all_routes)):
        raise RuntimeError("duplicate route contribution across decode steps")
    return {
        "decode_steps": len(steps),
        "route_contributions": len(all_routes),
    }


def _strict_stage_map(
    name: str, rows: Sequence[ContributionIdentity]
) -> dict[tuple[object, ...], ContributionIdentity]:
    keys = [row.contribution_key for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        raise RuntimeError(f"{name} contains duplicate contribution identity")
    return {row.contribution_key: row for row in rows}


def assert_identity_conservation(
    *,
    routed: Sequence[ContributionIdentity],
    dispatched: Sequence[ContributionIdentity],
    executed: Sequence[ContributionIdentity],
    combined: Sequence[ContributionIdentity],
) -> dict[str, int]:
    stages = {
        "routed": _strict_stage_map("routed", routed),
        "dispatched": _strict_stage_map("dispatched", dispatched),
        "executed": _strict_stage_map("executed", executed),
        "combined": _strict_stage_map("combined", combined),
    }
    reference = set(stages["routed"])
    if not reference:
        raise RuntimeError("routed contribution set is empty")
    for name in ("dispatched", "executed", "combined"):
        observed = set(stages[name])
        if observed != reference:
            raise RuntimeError(
                f"identity conservation failed at {name}: "
                f"missing={len(reference - observed)}, extra={len(observed - reference)}"
            )
        if any(row.target_replica < 0 for row in stages[name].values()):
            raise RuntimeError(f"{name} contains an unassigned target replica")
    for key in reference:
        targets = {
            stages[name][key].target_replica
            for name in ("dispatched", "executed", "combined")
        }
        if len(targets) != 1:
            raise RuntimeError("target replica changed after dispatch")
    return {
        "contributions": len(reference),
        **{name: len(values) for name, values in stages.items()},
    }


def completed_token_count(
    routed: Sequence[ContributionIdentity], completed: Sequence[ContributionIdentity]
) -> int:
    routed_by_token: dict[tuple[object, ...], set[tuple[object, ...]]] = defaultdict(set)
    completed_by_token: dict[tuple[object, ...], set[tuple[object, ...]]] = defaultdict(set)
    for row in routed:
        routed_by_token[row.output_token_key].add(row.contribution_key)
    for row in completed:
        completed_by_token[row.output_token_key].add(row.contribution_key)
    for token_key, completed_keys in completed_by_token.items():
        if completed_keys != routed_by_token.get(token_key, set()):
            raise RuntimeError("completed token has partial top-k sibling closure")
    return len(completed_by_token)


def slack_ns(*, deadline_ns: int, now_ns: int, predicted_remaining_ns: int) -> int:
    if min(deadline_ns, now_ns, predicted_remaining_ns) < 0:
        raise ValueError("slack inputs must be non-negative")
    return deadline_ns - now_ns - predicted_remaining_ns


@dataclass(frozen=True)
class SurfacePoint:
    model: str
    layer_id: int
    expert_id: int
    rows: int
    tier: str
    latency_us: float
    energy_j: float


@dataclass(frozen=True)
class SurfaceFallback:
    action: str
    reason: str


class SurfaceCatalog:
    """Exact measured-cell catalog; no interpolation or extrapolation."""

    def __init__(self, points: Sequence[SurfacePoint]) -> None:
        self._points: dict[tuple[str, int, int, int, str], SurfacePoint] = {}
        for point in points:
            key = (
                point.model,
                point.layer_id,
                point.expert_id,
                point.rows,
                point.tier,
            )
            if key in self._points:
                raise ValueError("duplicate surface cell")
            if point.rows <= 0 or point.latency_us <= 0 or point.energy_j <= 0:
                raise ValueError("surface metrics must be positive")
            self._points[key] = point

    def lookup(
        self, model: str, layer_id: int, expert_id: int, rows: int, tier: str
    ) -> SurfacePoint:
        try:
            return self._points[(model, layer_id, expert_id, rows, tier)]
        except KeyError as exc:
            raise RuntimeError("unmeasured surface cell") from exc

    def lookup_or_fallback(
        self, model: str, layer_id: int, expert_id: int, rows: int, tier: str
    ) -> SurfacePoint | SurfaceFallback:
        try:
            return self.lookup(model, layer_id, expert_id, rows, tier)
        except RuntimeError:
            return SurfaceFallback(
                action="immediate_default",
                reason=(
                    "unmeasured surface cell; fail closed to immediate/default tier "
                    "without estimating energy"
                ),
            )


@dataclass(frozen=True)
class OnlineObservation:
    now_ns: int
    visible_request_ids: tuple[str, ...]
    surface_version: str


@dataclass(frozen=True)
class OracleTrace:
    online_prefix: OnlineObservation
    future_arrivals: tuple[str, ...]
    future_routes: tuple[str, ...]
    future_service_us: tuple[float, ...]


def require_online_observation(value: OnlineObservation) -> OnlineObservation:
    if type(value) is not OnlineObservation:
        raise TypeError(f"online policy rejected future-known {type(value).__name__}")
    return value


def counter_delta_j(
    start_j: float, end_j: float, *, modulus_j: float | None = None
) -> float:
    if not math.isfinite(start_j + end_j) or min(start_j, end_j) < 0:
        raise ValueError("counter values must be finite and non-negative")
    if end_j >= start_j:
        return end_j - start_j
    if modulus_j is None:
        raise RuntimeError("energy counter moved backwards without wrap metadata")
    if not math.isfinite(modulus_j) or modulus_j <= start_j:
        raise ValueError("counter modulus must be finite and greater than start")
    return modulus_j - start_j + end_j


@dataclass(frozen=True)
class EnergyMeasurement:
    arm: str
    pair_id: str
    order_index: int
    raw_board_energy_j: float
    duration_s: float
    idle_power_w: float
    fixed_meter_overhead_j: float
    inner_repeats: int
    logical_work_ids: tuple[str, ...]
    completed_token_ids: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedMeasurement:
    raw_j_per_repeat: float
    work_j_per_repeat: float


def normalize_measurement(value: EnergyMeasurement) -> NormalizedMeasurement:
    if value.inner_repeats <= 0:
        raise RuntimeError("inner repeat denominator must be positive")
    if (
        value.raw_board_energy_j < 0
        or value.fixed_meter_overhead_j < 0
        or value.fixed_meter_overhead_j > value.raw_board_energy_j
    ):
        raise RuntimeError("invalid raw energy or fixed meter overhead")
    return NormalizedMeasurement(
        raw_j_per_repeat=value.raw_board_energy_j / value.inner_repeats,
        work_j_per_repeat=(
            value.raw_board_energy_j - value.fixed_meter_overhead_j
        )
        / value.inner_repeats,
    )


@dataclass(frozen=True)
class PairedEnergyDifference:
    raw_board_delta_j_per_completed_token: float
    work_delta_j_per_repeat: float


def paired_energy_difference(
    left: EnergyMeasurement, right: EnergyMeasurement, *, formal: bool
) -> PairedEnergyDifference:
    if formal:
        if left.inner_repeats != right.inner_repeats:
            raise RuntimeError("formal pair has unequal inner repeat denominators")
        if left.logical_work_ids != right.logical_work_ids:
            raise RuntimeError("formal pair has unequal logical work identity")
        if left.completed_token_ids != right.completed_token_ids:
            raise RuntimeError("formal pair has unequal completed-token identity")
    if not left.completed_token_ids or not right.completed_token_ids:
        raise RuntimeError("completed-token identity is empty")
    left_normalized = normalize_measurement(left)
    right_normalized = normalize_measurement(right)
    return PairedEnergyDifference(
        raw_board_delta_j_per_completed_token=(
            left.raw_board_energy_j - right.raw_board_energy_j
        )
        / (len(left.completed_token_ids) * left.inner_repeats),
        work_delta_j_per_repeat=(
            left_normalized.work_j_per_repeat
            - right_normalized.work_j_per_repeat
        ),
    )


def validate_abba_pairing(order: Sequence[str]) -> None:
    if tuple(order) not in (("A", "B", "B", "A"), ("B", "A", "A", "B")):
        raise RuntimeError("energy trials must use AB/BA crossing order")


@dataclass(frozen=True)
class ThermalState:
    timestamp_ns: int
    temperature_c: float
    graphics_clock_mhz: float
    memory_clock_mhz: float
    power_limit_w: float
    power_draw_w: float
    utilization_pct: float
    throttling_reason: str


def validate_thermal_pair(
    left: ThermalState,
    right: ThermalState,
    *,
    max_temperature_delta_c: float = 2.0,
) -> None:
    if abs(left.temperature_c - right.temperature_c) > max_temperature_delta_c:
        raise RuntimeError("thermal pair temperature mismatch")
    for name in ("graphics_clock_mhz", "memory_clock_mhz", "power_limit_w"):
        if getattr(left, name) != getattr(right, name):
            raise RuntimeError(f"thermal pair {name} mismatch")
    if left.throttling_reason != right.throttling_reason:
        raise RuntimeError("thermal pair throttling reason mismatch")


@dataclass(frozen=True)
class EnergySummary:
    raw_j_per_completed_token: float
    dynamic_j_per_completed_token: float


def energy_summary(
    *,
    raw_board_energy_j: float,
    duration_s: float,
    idle_power_w: float,
    completed_tokens: int,
) -> EnergySummary:
    if completed_tokens <= 0 or duration_s <= 0:
        raise ValueError("energy denominator and duration must be positive")
    dynamic = max(raw_board_energy_j - idle_power_w * duration_s, 0.0)
    return EnergySummary(
        raw_j_per_completed_token=raw_board_energy_j / completed_tokens,
        dynamic_j_per_completed_token=dynamic / completed_tokens,
    )


def formal_gate_status(checks: Mapping[str, bool]) -> str:
    if not checks:
        raise ValueError("formal gate requires named checks")
    return "PASS" if all(value is True for value in checks.values()) else "FAIL"
