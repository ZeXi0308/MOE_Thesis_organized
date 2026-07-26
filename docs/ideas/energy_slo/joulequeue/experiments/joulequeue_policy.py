from __future__ import annotations

"""Pure scheduling, accounting, and audit primitives for JouleQueue v1.

The module intentionally contains no model or GPU surface producer.  It can
exercise the frozen scheduling semantics on CPU fixtures, but it cannot make a
formal result scientifically eligible.  The top-level runner must fail closed
until a reviewed native route producer, RTX 5090 surface producer, and real
board-energy executor exist.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Mapping, Sequence


class ProtocolError(RuntimeError):
    pass


class SurfaceOutOfRange(ProtocolError):
    pass


class InfeasibleSchedule(ProtocolError):
    pass


@dataclass(frozen=True, order=True)
class JobIdentity:
    request_id: str
    forward_id: int
    layer_id: int
    token_id: int
    route_slot: int
    expert_id: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        for name in ("forward_id", "layer_id", "token_id", "route_slot", "expert_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def token_key(self) -> tuple[str, int, int]:
        return (self.request_id, self.forward_id, self.token_id)

    @property
    def queue_key(self) -> tuple[int, int]:
        return (self.layer_id, self.expert_id)

    def stable_id(self) -> str:
        return ":".join(str(part) for part in (
            self.request_id,
            self.forward_id,
            self.layer_id,
            self.token_id,
            self.route_slot,
            self.expert_id,
        ))


@dataclass(frozen=True)
class Job:
    identity: JobIdentity
    arrival_us: float
    rows: int
    deadline_us: float
    activation_sha256: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.arrival_us) or self.arrival_us < 0:
            raise ValueError("arrival_us must be finite and non-negative")
        if isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows <= 0:
            raise ValueError("rows must be a positive integer")
        if not math.isfinite(self.deadline_us) or self.deadline_us < self.arrival_us:
            raise ValueError("deadline_us must be finite and not precede arrival")
        if self.activation_sha256 is not None and not _is_sha256(self.activation_sha256):
            raise ValueError("activation_sha256 must be lowercase SHA-256")

    @property
    def queue_key(self) -> tuple[int, int]:
        return self.identity.queue_key

    @property
    def token_key(self) -> tuple[str, int, int]:
        return self.identity.token_key


@dataclass(frozen=True)
class SurfacePoint:
    rows: int
    energy_j: float
    latency_us: float
    energy_ucb95_j: float | None = None
    latency_ucb95_us: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows <= 0:
            raise ValueError("surface rows must be a positive integer")
        for name in ("energy_j", "latency_us"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.energy_ucb95_j is not None and (
            not math.isfinite(self.energy_ucb95_j)
            or self.energy_ucb95_j < self.energy_j
        ):
            raise ValueError("energy UCB must be finite and >= mean")
        if self.latency_ucb95_us is not None and (
            not math.isfinite(self.latency_ucb95_us)
            or self.latency_ucb95_us < self.latency_us
        ):
            raise ValueError("latency UCB must be finite and >= mean")

    def conservative(self) -> tuple[float, float]:
        return (
            self.energy_ucb95_j
            if self.energy_ucb95_j is not None
            else self.energy_j,
            self.latency_ucb95_us
            if self.latency_ucb95_us is not None
            else self.latency_us,
        )


@dataclass(frozen=True)
class SurfaceEstimate:
    rows: int
    energy_j: float
    latency_us: float
    interpolated: bool


class SurfaceCurve:
    """Conservative, in-range-only interpolation for one layer/expert."""

    def __init__(self, points: Sequence[SurfacePoint]) -> None:
        ordered = tuple(sorted(points, key=lambda point: point.rows))
        if not ordered:
            raise ValueError("surface curve cannot be empty")
        if len({point.rows for point in ordered}) != len(ordered):
            raise ValueError("surface curve has duplicate row points")
        self.points = ordered

    @property
    def min_rows(self) -> int:
        return self.points[0].rows

    @property
    def max_rows(self) -> int:
        return self.points[-1].rows

    def estimate(self, rows: int) -> SurfaceEstimate:
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError("rows must be a positive integer")
        if rows < self.min_rows or rows > self.max_rows:
            raise SurfaceOutOfRange(
                f"row count {rows} outside [{self.min_rows},{self.max_rows}]"
            )
        for point in self.points:
            if point.rows == rows:
                energy, latency = point.conservative()
                return SurfaceEstimate(rows, energy, latency, False)
        for left, right in zip(self.points, self.points[1:]):
            if left.rows < rows < right.rows:
                weight = (rows - left.rows) / (right.rows - left.rows)
                left_energy, left_latency = left.conservative()
                right_energy, right_latency = right.conservative()
                return SurfaceEstimate(
                    rows=rows,
                    energy_j=left_energy + weight * (right_energy - left_energy),
                    latency_us=left_latency + weight * (right_latency - left_latency),
                    interpolated=True,
                )
        raise AssertionError("in-range row count had no interpolation bracket")


class SurfaceCatalog:
    def __init__(
        self,
        curves: Mapping[tuple[int, int], SurfaceCurve],
        *,
        default_curve: SurfaceCurve | None = None,
        energy_basis: str = "dynamic_incremental",
    ) -> None:
        if not curves and default_curve is None:
            raise ValueError("surface catalog cannot be empty")
        self.curves = dict(curves)
        self.default_curve = default_curve
        if energy_basis not in {"dynamic_incremental", "total_during_launch"}:
            raise ValueError("unknown surface energy basis")
        self.energy_basis = energy_basis

    def estimate(self, jobs: Sequence[Job]) -> SurfaceEstimate:
        if not jobs:
            raise ValueError("cannot estimate an empty launch")
        queue_key = jobs[0].queue_key
        if any(job.queue_key != queue_key for job in jobs):
            raise ProtocolError("only jobs for the same layer/expert may be coalesced")
        curve = self.curves.get(queue_key, self.default_curve)
        if curve is None:
            raise SurfaceOutOfRange(f"missing surface for layer/expert {queue_key}")
        return curve.estimate(sum(job.rows for job in jobs))

    def individual_energy(self, jobs: Sequence[Job]) -> float:
        return sum(self.estimate((job,)).energy_j for job in jobs)


@dataclass(frozen=True)
class Decision:
    kind: str
    job_ids: tuple[JobIdentity, ...] = ()
    wake_after_us: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"launch", "defer", "urgent_flush"}:
            raise ValueError(f"unsupported decision kind: {self.kind}")
        if self.kind in {"launch", "urgent_flush"} and not self.job_ids:
            raise ValueError("launch decisions require jobs")
        if self.kind == "defer" and (
            self.wake_after_us is None
            or not math.isfinite(self.wake_after_us)
            or self.wake_after_us <= 0
        ):
            raise ValueError("defer requires a finite positive wake interval")


class ImmediatePolicy:
    def decide(self, now_us: float, ready: Sequence[Job], surfaces: SurfaceCatalog) -> Decision:
        del now_us, surfaces
        job = min(ready, key=lambda item: (item.arrival_us, item.identity))
        return Decision("launch", (job.identity,))


class EDFPolicy:
    def decide(self, now_us: float, ready: Sequence[Job], surfaces: SurfaceCatalog) -> Decision:
        del now_us, surfaces
        first = min(ready, key=lambda item: (item.deadline_us, item.arrival_us, item.identity))
        group = tuple(job.identity for job in ready if job.queue_key == first.queue_key)
        return Decision("launch", tuple(sorted(group)))


class FixedTimeoutPolicy:
    def __init__(self, timeout_us: float, tick_us: float = 1.0) -> None:
        if timeout_us < 0 or tick_us <= 0:
            raise ValueError("timeout must be non-negative and tick positive")
        self.timeout_us = float(timeout_us)
        self.tick_us = float(tick_us)

    def decide(self, now_us: float, ready: Sequence[Job], surfaces: SurfaceCatalog) -> Decision:
        del surfaces
        groups = _groups(ready)
        queue_key, jobs = min(
            groups.items(), key=lambda item: (min(job.arrival_us for job in item[1]), item[0])
        )
        del queue_key
        age = now_us - min(job.arrival_us for job in jobs)
        if age + 1e-12 >= self.timeout_us:
            return Decision("launch", tuple(sorted(job.identity for job in jobs)))
        return Decision("defer", wake_after_us=max(min(self.timeout_us - age, self.tick_us), 1e-9))


class StaticRowsPolicy:
    def __init__(self, threshold_rows: int, max_age_us: float, tick_us: float = 1.0) -> None:
        if threshold_rows <= 0 or max_age_us <= 0 or tick_us <= 0:
            raise ValueError("row threshold, max age, and tick must be positive")
        self.threshold_rows = int(threshold_rows)
        self.max_age_us = float(max_age_us)
        self.tick_us = float(tick_us)

    def decide(self, now_us: float, ready: Sequence[Job], surfaces: SurfaceCatalog) -> Decision:
        del surfaces
        groups = _groups(ready)
        ranked = sorted(
            groups.values(),
            key=lambda jobs: (-sum(job.rows for job in jobs), min(job.deadline_us for job in jobs)),
        )
        jobs = ranked[0]
        rows = sum(job.rows for job in jobs)
        age = now_us - min(job.arrival_us for job in jobs)
        if rows >= self.threshold_rows or age + 1e-12 >= self.max_age_us:
            kind = "urgent_flush" if age + 1e-12 >= self.max_age_us else "launch"
            return Decision(kind, tuple(sorted(job.identity for job in jobs)))
        return Decision("defer", wake_after_us=min(self.tick_us, self.max_age_us - age))


class ThroughputMuQueuePolicy(StaticRowsPolicy):
    """Capability-equivalent occupancy/age queue; it never reads energy."""


class AmoEStylePolicy(StaticRowsPolicy):
    """Per-layer ready-job defrag proxy; not a claim of reproducing AMoE."""

    def decide(self, now_us: float, ready: Sequence[Job], surfaces: SurfaceCatalog) -> Decision:
        del surfaces
        groups = _groups(ready)
        layer_rows: dict[int, int] = {}
        for (layer_id, _expert_id), jobs in groups.items():
            layer_rows[layer_id] = layer_rows.get(layer_id, 0) + sum(job.rows for job in jobs)
        layer = max(layer_rows, key=lambda item: (layer_rows[item], -item))
        eligible = [jobs for (layer_id, _), jobs in groups.items() if layer_id == layer]
        jobs = max(eligible, key=lambda items: (sum(job.rows for job in items), -min(job.deadline_us for job in items)))
        rows = sum(job.rows for job in jobs)
        age = now_us - min(job.arrival_us for job in jobs)
        if rows >= self.threshold_rows or age + 1e-12 >= self.max_age_us:
            return Decision(
                "urgent_flush" if age + 1e-12 >= self.max_age_us else "launch",
                tuple(sorted(job.identity for job in jobs)),
            )
        return Decision("defer", wake_after_us=min(self.tick_us, self.max_age_us - age))


class FestinaLikeProfiledPolicy(StaticRowsPolicy):
    """Calibration-frozen operating point; no future information is exposed."""


class CausalJouleQueuePolicy:
    def __init__(
        self,
        *,
        target_rows: int,
        min_saving_fraction: float,
        max_age_us: float,
        urgent_margin_us: float,
        tick_us: float = 1.0,
    ) -> None:
        if target_rows <= 0 or max_age_us <= 0 or urgent_margin_us < 0 or tick_us <= 0:
            raise ValueError("invalid causal policy timing/row parameter")
        if not 0 <= min_saving_fraction < 1:
            raise ValueError("min_saving_fraction must be in [0,1)")
        self.target_rows = int(target_rows)
        self.min_saving_fraction = float(min_saving_fraction)
        self.max_age_us = float(max_age_us)
        self.urgent_margin_us = float(urgent_margin_us)
        self.tick_us = float(tick_us)

    def decide(self, now_us: float, ready: Sequence[Job], surfaces: SurfaceCatalog) -> Decision:
        # This API intentionally has no future-arrival argument.
        groups = _groups(ready)
        scored: list[tuple[float, int, float, tuple[int, int], list[Job], SurfaceEstimate]] = []
        for queue_key, jobs in groups.items():
            try:
                batch = surfaces.estimate(jobs)
                separate = surfaces.individual_energy(jobs)
            except SurfaceOutOfRange:
                first = min(jobs, key=lambda job: (job.deadline_us, job.identity))
                return Decision("urgent_flush", (first.identity,))
            saving_fraction = max((separate - batch.energy_j) / separate, 0.0)
            rows = sum(job.rows for job in jobs)
            slack = min(job.deadline_us for job in jobs) - now_us - batch.latency_us
            scored.append((saving_fraction, rows, slack, queue_key, jobs, batch))

        urgent = [item for item in scored if (
            now_us - min(job.arrival_us for job in item[4]) >= self.max_age_us
            or item[2] <= self.urgent_margin_us
        )]
        if urgent:
            chosen = min(urgent, key=lambda item: (item[2], item[3]))
            return Decision("urgent_flush", tuple(sorted(job.identity for job in chosen[4])))

        launchable = [item for item in scored if (
            item[1] >= self.target_rows or item[0] >= self.min_saving_fraction
        )]
        if launchable:
            chosen = max(launchable, key=lambda item: (item[0], item[1], -item[2]))
            return Decision("launch", tuple(sorted(job.identity for job in chosen[4])))
        return Decision("defer", wake_after_us=self.tick_us)


@dataclass(frozen=True)
class LaunchRecord:
    arm: str
    kind: str
    start_us: float
    end_us: float
    job_ids: tuple[JobIdentity, ...]
    rows: int
    surface_energy_j: float


@dataclass(frozen=True)
class ScheduleResult:
    arm: str
    actions: tuple[LaunchRecord, ...]
    completion_us: Mapping[JobIdentity, float]
    window_start_us: float
    window_end_us: float
    surface_energy_j: float
    idle_energy_j: float
    total_board_energy_j: float
    energy_basis: str

    @property
    def makespan_us(self) -> float:
        return self.window_end_us - self.window_start_us


def simulate_causal(
    jobs: Sequence[Job],
    surfaces: SurfaceCatalog,
    policy: object,
    *,
    idle_power_w: float,
    arm: str,
) -> ScheduleResult:
    validate_jobs(jobs)
    if not math.isfinite(idle_power_w) or idle_power_w < 0:
        raise ValueError("idle_power_w must be finite and non-negative")
    pending = sorted(jobs, key=lambda job: (job.arrival_us, job.identity))
    ready: list[Job] = []
    completion: dict[JobIdentity, float] = {}
    actions: list[LaunchRecord] = []
    window_start = pending[0].arrival_us
    now = window_start
    iteration_budget = max(10_000, len(jobs) * 10_000)
    iterations = 0
    while pending or ready:
        iterations += 1
        if iterations > iteration_budget:
            raise ProtocolError("scheduler failed to make progress")
        while pending and pending[0].arrival_us <= now + 1e-12:
            ready.append(pending.pop(0))
        if not ready:
            now = pending[0].arrival_us
            continue
        decision = policy.decide(now, tuple(ready), surfaces)
        if decision.kind == "defer":
            assert decision.wake_after_us is not None
            policy_wake = now + decision.wake_after_us
            # The runtime naturally wakes on a new arrival; the policy did not
            # receive that future timestamp when making its decision.
            now = min(policy_wake, pending[0].arrival_us) if pending else policy_wake
            continue
        selected_set = set(decision.job_ids)
        if len(selected_set) != len(decision.job_ids):
            raise ProtocolError("policy returned duplicate job ids")
        selected = [job for job in ready if job.identity in selected_set]
        if len(selected) != len(selected_set):
            raise ProtocolError("policy selected a non-ready or unknown job")
        try:
            estimate = surfaces.estimate(selected)
        except SurfaceOutOfRange:
            # The frozen protocol permits no extrapolation.  An oversized
            # coalesced launch therefore degrades to one immediate job; a
            # missing/out-of-range individual surface remains a hard error.
            fallback = min(selected, key=lambda job: (job.deadline_us, job.identity))
            selected = [fallback]
            selected_set = {fallback.identity}
            estimate = surfaces.estimate(selected)
        end = now + estimate.latency_us
        for job in selected:
            if job.identity in completion:
                raise ProtocolError("job completed more than once")
            completion[job.identity] = end
        actions.append(LaunchRecord(
            arm=arm,
            kind=decision.kind,
            start_us=now,
            end_us=end,
            job_ids=tuple(sorted(selected_set)),
            rows=sum(job.rows for job in selected),
            surface_energy_j=estimate.energy_j,
        ))
        ready = [job for job in ready if job.identity not in selected_set]
        now = end
    window_end = max(completion.values())
    surface_energy = sum(action.surface_energy_j for action in actions)
    busy_us = sum(action.end_us - action.start_us for action in actions)
    idle_charge_us = (
        window_end - window_start
        if surfaces.energy_basis == "dynamic_incremental"
        else max(window_end - window_start - busy_us, 0.0)
    )
    idle = idle_power_w * idle_charge_us / 1_000_000.0
    result = ScheduleResult(
        arm=arm,
        actions=tuple(actions),
        completion_us=completion,
        window_start_us=window_start,
        window_end_us=window_end,
        surface_energy_j=surface_energy,
        idle_energy_j=idle,
        total_board_energy_j=surface_energy + idle,
        energy_basis=surfaces.energy_basis,
    )
    validate_full_drain(jobs, result)
    return result


def exact_clairvoyant_oracle(
    jobs: Sequence[Job],
    surfaces: SurfaceCatalog,
    *,
    idle_power_w: float,
    max_jobs: int = 12,
    enforce_deadlines: bool = True,
    max_age_us: float | None = None,
) -> ScheduleResult:
    """Exact exhaustive dynamic program for a deliberately small episode."""

    validate_jobs(jobs)
    if len(jobs) > max_jobs:
        raise ProtocolError(f"exact oracle supports at most {max_jobs} jobs")
    if max_age_us is not None and (
        not math.isfinite(max_age_us) or max_age_us <= 0
    ):
        raise ValueError("max_age_us must be finite and positive")
    ordered = tuple(sorted(jobs, key=lambda job: job.identity))
    all_mask = (1 << len(ordered)) - 1
    start = min(job.arrival_us for job in ordered)

    @lru_cache(maxsize=None)
    def solve(mask: int, now_key: float) -> tuple[float, tuple[tuple[float, tuple[int, ...], str], ...]]:
        now = float(now_key)
        if mask == all_mask:
            return 0.0, ()
        remaining = [index for index in range(len(ordered)) if not mask & (1 << index)]
        available = [index for index in remaining if ordered[index].arrival_us <= now + 1e-12]
        candidates: list[tuple[float, tuple[tuple[float, tuple[int, ...], str], ...]]] = []

        future_arrivals = sorted({ordered[index].arrival_us for index in remaining if ordered[index].arrival_us > now + 1e-12})
        if future_arrivals:
            wake = future_arrivals[0]
            try:
                tail_cost, tail_plan = solve(mask, round(wake, 9))
            except InfeasibleSchedule:
                pass
            else:
                wait_cost = idle_power_w * (wake - now) / 1_000_000.0
                candidates.append((wait_cost + tail_cost, ((now, (), "defer"),) + tail_plan))

        groups: dict[tuple[int, int], list[int]] = {}
        for index in available:
            groups.setdefault(ordered[index].queue_key, []).append(index)
        for indices in groups.values():
            for count in range(1, len(indices) + 1):
                for subset in combinations(indices, count):
                    selected = tuple(ordered[index] for index in subset)
                    try:
                        estimate = surfaces.estimate(selected)
                    except SurfaceOutOfRange:
                        continue
                    end = now + estimate.latency_us
                    if enforce_deadlines and any(end > job.deadline_us + 1e-12 for job in selected):
                        continue
                    if max_age_us is not None and any(
                        end - job.arrival_us > max_age_us + 1e-12 for job in selected
                    ):
                        continue
                    next_mask = mask
                    for index in subset:
                        next_mask |= 1 << index
                    try:
                        tail_cost, tail_plan = solve(next_mask, round(end, 9))
                    except InfeasibleSchedule:
                        continue
                    interval_energy = (
                        estimate.energy_j
                        + (
                            idle_power_w * estimate.latency_us / 1_000_000.0
                            if surfaces.energy_basis == "dynamic_incremental"
                            else 0.0
                        )
                    )
                    candidates.append((
                        interval_energy + tail_cost,
                        ((now, tuple(subset), "launch"),) + tail_plan,
                    ))
        if not candidates:
            raise InfeasibleSchedule("no deadline-feasible exact schedule")
        return min(candidates, key=lambda item: (item[0], item[1]))

    _cost, plan = solve(0, round(start, 9))
    completion: dict[JobIdentity, float] = {}
    actions: list[LaunchRecord] = []
    for at, indices, kind in plan:
        if kind == "defer":
            continue
        selected = tuple(ordered[index] for index in indices)
        estimate = surfaces.estimate(selected)
        end = at + estimate.latency_us
        for job in selected:
            completion[job.identity] = end
        actions.append(LaunchRecord(
            arm="clairvoyant_energy_oracle",
            kind="launch",
            start_us=at,
            end_us=end,
            job_ids=tuple(job.identity for job in selected),
            rows=sum(job.rows for job in selected),
            surface_energy_j=estimate.energy_j,
        ))
    end = max(completion.values())
    surface_energy = sum(action.surface_energy_j for action in actions)
    busy_us = sum(action.end_us - action.start_us for action in actions)
    idle_charge_us = (
        end - start
        if surfaces.energy_basis == "dynamic_incremental"
        else max(end - start - busy_us, 0.0)
    )
    idle = idle_power_w * idle_charge_us / 1_000_000.0
    result = ScheduleResult(
        arm="clairvoyant_energy_oracle",
        actions=tuple(actions),
        completion_us=completion,
        window_start_us=start,
        window_end_us=end,
        surface_energy_j=surface_energy,
        idle_energy_j=idle,
        total_board_energy_j=surface_energy + idle,
        energy_basis=surfaces.energy_basis,
    )
    validate_full_drain(ordered, result)
    return result


@dataclass(frozen=True)
class ScheduleMetrics:
    arm: str
    completed_jobs: int
    completed_tokens: int
    board_j_per_completed_token: float
    p99_token_completion_us: float
    p99_tpot_proxy_us: float
    slo_violation_rate: float
    makespan_us: float
    launch_count: int
    mean_rows_per_launch: float
    token_latencies_us: tuple[float, ...]
    tpot_values_us: tuple[float, ...]
    violation_flags: tuple[int, ...]


def schedule_metrics(jobs: Sequence[Job], result: ScheduleResult, *, slo_us: float) -> ScheduleMetrics:
    validate_full_drain(jobs, result)
    if not math.isfinite(slo_us) or slo_us <= 0:
        raise ValueError("slo_us must be finite and positive")
    by_token: dict[tuple[str, int, int], list[Job]] = {}
    for job in jobs:
        by_token.setdefault(job.token_key, []).append(job)
    token_records: list[tuple[tuple[str, int, int], float, float]] = []
    for token_key, token_jobs in by_token.items():
        arrival = min(job.arrival_us for job in token_jobs)
        completion = max(result.completion_us[job.identity] for job in token_jobs)
        token_records.append((token_key, arrival, completion))
    token_records.sort(key=lambda item: item[0])
    latencies = tuple(completion - arrival for _, arrival, completion in token_records)
    violations = tuple(int(value > slo_us) for value in latencies)
    request_tokens: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for (request_id, forward_id, token_id), _arrival, completion in token_records:
        request_tokens.setdefault((request_id, forward_id), []).append((token_id, completion))
    tpot: list[float] = []
    for values in request_tokens.values():
        ordered = sorted(values)
        tpot.extend(right[1] - left[1] for left, right in zip(ordered, ordered[1:]))
    completed_tokens = len(token_records)
    if completed_tokens <= 0:
        raise ProtocolError("completed-token denominator is zero")
    launch_rows = [action.rows for action in result.actions]
    return ScheduleMetrics(
        arm=result.arm,
        completed_jobs=len(result.completion_us),
        completed_tokens=completed_tokens,
        board_j_per_completed_token=board_j_per_completed_token(
            result.total_board_energy_j, completed_tokens
        ),
        p99_token_completion_us=_percentile(latencies, 0.99),
        p99_tpot_proxy_us=_percentile(tpot, 0.99) if tpot else 0.0,
        slo_violation_rate=sum(violations) / completed_tokens,
        makespan_us=result.makespan_us,
        launch_count=len(result.actions),
        mean_rows_per_launch=sum(launch_rows) / len(launch_rows),
        token_latencies_us=latencies,
        tpot_values_us=tuple(tpot),
        violation_flags=violations,
    )


@dataclass(frozen=True)
class BootstrapGate:
    energy_improvement_lcb95: float
    energy_improvement_mean: float
    completion_ratio_ucb95: float
    tpot_ratio_ucb95: float
    violation_delta_ucb95: float
    replicates: int


def paired_hierarchical_bootstrap(
    baseline: Sequence[ScheduleMetrics],
    candidate: Sequence[ScheduleMetrics],
    *,
    replicates: int = 2000,
    seed: int = 2026072299,
) -> BootstrapGate:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap requires equal non-empty episode lists")
    if replicates < 100:
        raise ValueError("bootstrap requires at least 100 replicates")
    for left, right in zip(baseline, candidate):
        if left.completed_tokens != right.completed_tokens:
            raise ProtocolError("paired episodes have different completed-token sets")
    rng = random.Random(seed)
    improvements: list[float] = []
    completion_ratios: list[float] = []
    tpot_ratios: list[float] = []
    violation_deltas: list[float] = []
    count = len(baseline)
    for _ in range(replicates):
        indices = [rng.randrange(count) for _ in range(count)]
        b_tokens = sum(baseline[i].completed_tokens for i in indices)
        c_tokens = sum(candidate[i].completed_tokens for i in indices)
        b_energy = sum(
            baseline[i].board_j_per_completed_token * baseline[i].completed_tokens
            for i in indices
        ) / b_tokens
        c_energy = sum(
            candidate[i].board_j_per_completed_token * candidate[i].completed_tokens
            for i in indices
        ) / c_tokens
        b_latency = tuple(value for i in indices for value in baseline[i].token_latencies_us)
        c_latency = tuple(value for i in indices for value in candidate[i].token_latencies_us)
        b_tpot = tuple(value for i in indices for value in baseline[i].tpot_values_us)
        c_tpot = tuple(value for i in indices for value in candidate[i].tpot_values_us)
        b_viol = tuple(value for i in indices for value in baseline[i].violation_flags)
        c_viol = tuple(value for i in indices for value in candidate[i].violation_flags)
        if b_energy <= 0 or _percentile(b_latency, 0.99) <= 0:
            raise ProtocolError("baseline denominator must be positive")
        improvements.append((b_energy - c_energy) / b_energy)
        completion_ratios.append(_percentile(c_latency, 0.99) / _percentile(b_latency, 0.99))
        if b_tpot and c_tpot:
            if _percentile(b_tpot, 0.99) <= 0:
                raise ProtocolError("baseline TPOT proxy denominator must be positive")
            tpot_ratios.append(_percentile(c_tpot, 0.99) / _percentile(b_tpot, 0.99))
        else:
            tpot_ratios.append(1.0)
        violation_deltas.append(sum(c_viol) / len(c_viol) - sum(b_viol) / len(b_viol))
    return BootstrapGate(
        energy_improvement_lcb95=_percentile(improvements, 0.05),
        energy_improvement_mean=sum(improvements) / len(improvements),
        completion_ratio_ucb95=_percentile(completion_ratios, 0.95),
        tpot_ratio_ucb95=_percentile(tpot_ratios, 0.95),
        violation_delta_ucb95=_percentile(violation_deltas, 0.95),
        replicates=replicates,
    )


def oracle_gate_passes(gate: BootstrapGate) -> bool:
    return (
        gate.energy_improvement_lcb95 >= 0.10
        and gate.completion_ratio_ucb95 <= 1.03
        and gate.tpot_ratio_ucb95 <= 1.03
        and gate.violation_delta_ucb95 <= 0.01
    )


def board_j_per_completed_token(total_board_energy_j: float, completed_tokens: int) -> float:
    if not math.isfinite(total_board_energy_j) or total_board_energy_j < 0:
        raise ValueError("total board energy must be finite and non-negative")
    if (
        isinstance(completed_tokens, bool)
        or not isinstance(completed_tokens, int)
        or completed_tokens <= 0
    ):
        raise ValueError("completed-token denominator must be a positive integer")
    return total_board_energy_j / completed_tokens


def milliwatts_to_watts(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("power must be finite and non-negative")
    return value / 1000.0


def millijoules_to_joules(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("energy must be finite and non-negative")
    return value / 1000.0


@dataclass(frozen=True)
class FrozenCalibrationChoice:
    parameter: str
    calibration_manifest_sha256: str


def select_calibration_only(
    candidates: Mapping[str, ScheduleMetrics],
    *,
    split: str,
    calibration_manifest_sha256: str,
    p99_limit_us: float,
    violation_limit: float,
) -> FrozenCalibrationChoice:
    """Choose a baseline operating point without allowing sealed selection."""

    if split != "calibration":
        raise ProtocolError("policy parameter selection is calibration-only")
    if not _is_sha256(calibration_manifest_sha256):
        raise ValueError("calibration manifest must be lowercase SHA-256")
    eligible = {
        name: metric
        for name, metric in candidates.items()
        if metric.p99_token_completion_us <= p99_limit_us
        and metric.slo_violation_rate <= violation_limit
    }
    if not eligible:
        raise ProtocolError("no SLO-qualified calibration operating point")
    winner = min(
        eligible,
        key=lambda name: (eligible[name].board_j_per_completed_token, name),
    )
    return FrozenCalibrationChoice(winner, calibration_manifest_sha256)


def validate_jobs(jobs: Sequence[Job]) -> None:
    if not jobs:
        raise ValueError("job trace cannot be empty")
    identities = [job.identity for job in jobs]
    if len(identities) != len(set(identities)):
        raise ProtocolError("duplicate job identity")


def validate_full_drain(jobs: Sequence[Job], result: ScheduleResult) -> None:
    expected = {job.identity for job in jobs}
    observed = set(result.completion_us)
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ProtocolError(f"full-drain identity mismatch: missing={missing}, extra={extra}")
    action_ids = [identity for action in result.actions for identity in action.job_ids]
    if len(action_ids) != len(set(action_ids)) or set(action_ids) != expected:
        raise ProtocolError("action trace violates exactly-once/full-drain")


def validate_route_closure(records: Sequence[Mapping[str, object]], *, top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {
        "model_revision", "data_manifest_sha256", "request_id", "forward_id",
        "batch_id", "phase", "decode_step", "layer_id", "token_id",
        "token_position", "topk_slot", "expert_id", "sender_rank",
        "receiver_rank", "valid", "route_weight", "placement_sha256",
    }
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    model_hashes: set[object] = set()
    data_hashes: set[object] = set()
    placement_hashes: set[object] = set()
    for record in records:
        missing = required - set(record)
        if missing:
            raise ProtocolError(f"identity-incomplete contribution record: {sorted(missing)}")
        if type(record["valid"]) is not bool:
            raise ProtocolError("valid must be a JSON boolean")
        model_hashes.add(record["model_revision"])
        data_hashes.add(record["data_manifest_sha256"])
        placement_hashes.add(record["placement_sha256"])
        key = tuple(record[name] for name in (
            "request_id", "forward_id", "layer_id", "token_id", "token_position"
        ))
        groups.setdefault(key, []).append(record)
    if len(model_hashes) != 1 or len(data_hashes) != 1 or len(placement_hashes) != 1:
        raise ProtocolError("model/data/placement identity drift within route trace")
    for key, group in groups.items():
        valid = [record for record in group if record["valid"] is True]
        slots = sorted(int(record["topk_slot"]) for record in valid)
        experts = [int(record["expert_id"]) for record in valid]
        if slots != list(range(top_k)):
            raise ProtocolError(f"top-k route closure failed for {key}: {slots}")
        if len(experts) != len(set(experts)):
            raise ProtocolError(f"duplicate expert in one token top-k for {key}")


@dataclass(frozen=True)
class NumericalGateResult:
    passed: bool
    max_abs_error: float
    mean_abs_error: float
    cosine_error: float


def numerical_equivalence_gate(
    separate: Sequence[float],
    coalesced: Sequence[float],
    *,
    max_abs_limit: float = 2e-2,
    mean_abs_limit: float = 2e-3,
    cosine_error_limit: float = 1e-4,
) -> NumericalGateResult:
    if len(separate) != len(coalesced) or not separate:
        raise ProtocolError("numerical comparison requires equal non-empty row-aligned outputs")
    if any(not math.isfinite(value) for value in (*separate, *coalesced)):
        raise ProtocolError("numerical comparison contains non-finite values")
    errors = [abs(left - right) for left, right in zip(separate, coalesced)]
    dot = sum(left * right for left, right in zip(separate, coalesced))
    left_norm = math.sqrt(sum(value * value for value in separate))
    right_norm = math.sqrt(sum(value * value for value in coalesced))
    if left_norm == 0 or right_norm == 0:
        cosine_error = 0.0 if separate == coalesced else math.inf
    else:
        cosine_error = 1.0 - dot / (left_norm * right_norm)
    result = NumericalGateResult(
        passed=(
            max(errors) <= max_abs_limit
            and sum(errors) / len(errors) <= mean_abs_limit
            and cosine_error <= cosine_error_limit
        ),
        max_abs_error=max(errors),
        mean_abs_error=sum(errors) / len(errors),
        cosine_error=cosine_error,
    )
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_review_attestation(
    attestation: Mapping[str, object],
    *,
    protocol_version: str,
    files: Mapping[str, Path],
) -> dict[str, str]:
    if attestation.get("status") != "SIGNED-OFF":
        raise ProtocolError("formal run requires Phase 4 SIGNED-OFF")
    if attestation.get("protocol_version") != protocol_version:
        raise ProtocolError("review attestation protocol version drift")
    expected = {name: sha256_file(path) for name, path in files.items()}
    observed = attestation.get("file_sha256")
    if not isinstance(observed, dict) or observed != expected:
        raise ProtocolError("review attestation file hash drift")
    return expected


def _groups(ready: Sequence[Job]) -> dict[tuple[int, int], list[Job]]:
    groups: dict[tuple[int, int], list[Job]] = {}
    for job in ready:
        groups.setdefault(job.queue_key, []).append(job)
    return groups


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    index = quantile * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
