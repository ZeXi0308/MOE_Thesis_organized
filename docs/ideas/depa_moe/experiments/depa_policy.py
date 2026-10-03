from __future__ import annotations

"""Pure DEPA-MoE replay, policy, accounting, and exact-oracle primitives.

This module deliberately contains no GPU executor.  It can replay measured
expert service curves and exercise the frozen decision semantics on CPU, but a
synthetic trace or development curve is never scientific evidence.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import hashlib
import math
import random
import time
from typing import Mapping, Protocol, Sequence


class ProtocolError(RuntimeError):
    pass


class SurfaceOutOfRange(ProtocolError):
    pass


class InfeasibleSchedule(ProtocolError):
    pass


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    model: str
    cell: str
    arrival_us: float
    deadline_us: float
    expert_rows: tuple[tuple[int, int], ...]
    request_class: str = "default"
    activation_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.model or not self.cell or not self.request_class:
            raise ValueError("request identity fields must be non-empty")
        if not math.isfinite(self.arrival_us) or self.arrival_us < 0:
            raise ValueError("arrival_us must be finite and non-negative")
        if not math.isfinite(self.deadline_us) or self.deadline_us < self.arrival_us:
            raise ValueError("deadline_us must be finite and not precede arrival")
        if not self.expert_rows:
            raise ValueError("expert_rows cannot be empty")
        expert_ids: set[int] = set()
        for expert_id, rows in self.expert_rows:
            if isinstance(expert_id, bool) or not isinstance(expert_id, int) or expert_id < 0:
                raise ValueError("expert_id must be a non-negative integer")
            if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
                raise ValueError("expert rows must be a positive integer")
            if expert_id in expert_ids:
                raise ValueError("duplicate expert_id in one request")
            expert_ids.add(expert_id)
        if tuple(sorted(self.expert_rows)) != self.expert_rows:
            raise ValueError("expert_rows must be sorted by expert_id")
        if self.activation_sha256 is not None and not _is_sha256(self.activation_sha256):
            raise ValueError("activation_sha256 must be lowercase SHA-256")

    @property
    def slack_budget_us(self) -> float:
        return self.deadline_us - self.arrival_us


def validate_requests(requests: Sequence[RequestSpec]) -> tuple[RequestSpec, ...]:
    ordered = tuple(sorted(requests, key=lambda item: (item.arrival_us, item.request_id)))
    if not ordered:
        raise ProtocolError("episode cannot be empty")
    ids = [item.request_id for item in ordered]
    if len(set(ids)) != len(ids):
        raise ProtocolError("duplicate request_id in episode")
    models = {item.model for item in ordered}
    cells = {item.cell for item in ordered}
    if len(models) != 1 or len(cells) != 1:
        raise ProtocolError("one episode must contain exactly one model and one cell")
    return ordered


@dataclass(frozen=True)
class SurfacePoint:
    rows: int
    latency_us: float
    latency_ucb95_us: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows <= 0:
            raise ValueError("surface rows must be a positive integer")
        if not math.isfinite(self.latency_us) or self.latency_us <= 0:
            raise ValueError("latency_us must be finite and positive")
        if self.latency_ucb95_us is not None and (
            not math.isfinite(self.latency_ucb95_us)
            or self.latency_ucb95_us < self.latency_us
        ):
            raise ValueError("latency UCB must be finite and >= mean")

    @property
    def conservative_latency_us(self) -> float:
        return self.latency_ucb95_us or self.latency_us


class ServiceCurve:
    """Conservative, in-range-only interpolation for expert row counts."""

    def __init__(self, points: Sequence[SurfacePoint]) -> None:
        ordered = tuple(sorted(points, key=lambda point: point.rows))
        if not ordered:
            raise ValueError("service curve cannot be empty")
        if len({point.rows for point in ordered}) != len(ordered):
            raise ValueError("service curve has duplicate row points")
        conservative = [point.conservative_latency_us for point in ordered]
        if any(right < left for left, right in zip(conservative, conservative[1:])):
            raise ValueError("conservative service latency must be non-decreasing")
        self.points = ordered

    @property
    def min_rows(self) -> int:
        return self.points[0].rows

    @property
    def max_rows(self) -> int:
        return self.points[-1].rows

    def estimate_us(self, rows: int) -> float:
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError("rows must be a positive integer")
        if rows < self.min_rows or rows > self.max_rows:
            raise SurfaceOutOfRange(
                f"row count {rows} outside [{self.min_rows},{self.max_rows}]"
            )
        for point in self.points:
            if point.rows == rows:
                return point.conservative_latency_us
        for left, right in zip(self.points, self.points[1:]):
            if left.rows < rows < right.rows:
                weight = (rows - left.rows) / (right.rows - left.rows)
                return left.conservative_latency_us + weight * (
                    right.conservative_latency_us - left.conservative_latency_us
                )
        raise AssertionError("in-range row count had no interpolation bracket")


class ServiceCatalog:
    """Measured expert execution model for one model revision.

    The v1 5090 mechanism is explicit: expert kernels are serialized and a
    launch pays one fixed overhead plus the sum of aggregated per-expert curves.
    No network, EP, or multi-rank claim is represented here.
    """

    def __init__(
        self,
        curves: Mapping[int, ServiceCurve],
        *,
        default_curve: ServiceCurve | None = None,
        launch_overhead_us: float = 0.0,
        execution_model: str = "serial_experts",
    ) -> None:
        if not curves and default_curve is None:
            raise ValueError("service catalog cannot be empty")
        if execution_model != "serial_experts":
            raise ValueError("v1 supports only serial_experts")
        if not math.isfinite(launch_overhead_us) or launch_overhead_us < 0:
            raise ValueError("launch_overhead_us must be finite and non-negative")
        self.curves = dict(curves)
        self.default_curve = default_curve
        self.launch_overhead_us = launch_overhead_us
        self.execution_model = execution_model

    def _curve(self, expert_id: int) -> ServiceCurve:
        curve = self.curves.get(expert_id, self.default_curve)
        if curve is None:
            raise SurfaceOutOfRange(f"missing service curve for expert {expert_id}")
        return curve

    def aggregate_rows(self, requests: Sequence[RequestSpec]) -> dict[int, int]:
        if not requests:
            raise ValueError("cannot estimate an empty batch")
        rows: dict[int, int] = {}
        for request in requests:
            for expert_id, count in request.expert_rows:
                rows[expert_id] = rows.get(expert_id, 0) + count
        return rows

    def estimate_batch_us(self, requests: Sequence[RequestSpec]) -> float:
        rows = self.aggregate_rows(requests)
        return self.launch_overhead_us + sum(
            self._curve(expert_id).estimate_us(count)
            for expert_id, count in sorted(rows.items())
        )

    def separate_service_us(self, requests: Sequence[RequestSpec]) -> float:
        return sum(self.estimate_batch_us((request,)) for request in requests)

    def coalescing_saving_fraction(self, requests: Sequence[RequestSpec]) -> float:
        separate = self.separate_service_us(requests)
        if separate <= 0:
            return 0.0
        return max(0.0, (separate - self.estimate_batch_us(requests)) / separate)


@dataclass(frozen=True)
class Decision:
    kind: str
    request_ids: tuple[str, ...] = ()
    wake_us: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"launch", "defer", "reject"}:
            raise ValueError("decision kind must be launch, defer, or reject")
        if self.kind in {"launch", "reject"} and not self.request_ids:
            raise ValueError(f"{self.kind} requires request_ids")
        if self.kind == "defer":
            if self.request_ids:
                raise ValueError("defer cannot name requests")
            if self.wake_us is None or not math.isfinite(self.wake_us):
                raise ValueError("defer requires finite wake_us")
        elif self.wake_us is not None:
            raise ValueError("only defer may set wake_us")


class OnlinePolicy(Protocol):
    name: str

    def decide(
        self,
        now_us: float,
        ready: Sequence[RequestSpec],
        surface: ServiceCatalog,
    ) -> Decision:
        ...


def _ordered_ready(ready: Sequence[RequestSpec]) -> tuple[RequestSpec, ...]:
    return tuple(sorted(ready, key=lambda item: (item.arrival_us, item.request_id)))


class FCFSPolicy:
    name = "current_fcfs"

    def __init__(self, max_batch: int) -> None:
        self.max_batch = _positive_int(max_batch, "max_batch")

    def decide(self, now_us: float, ready: Sequence[RequestSpec], surface: ServiceCatalog) -> Decision:
        del now_us, surface
        chosen = _ordered_ready(ready)[: self.max_batch]
        return Decision("launch", tuple(item.request_id for item in chosen), reason="fcfs")


class EDFPolicy:
    name = "edf"

    def __init__(self, max_batch: int) -> None:
        self.max_batch = _positive_int(max_batch, "max_batch")

    def decide(self, now_us: float, ready: Sequence[RequestSpec], surface: ServiceCatalog) -> Decision:
        del now_us, surface
        chosen = sorted(ready, key=lambda item: (item.deadline_us, item.arrival_us, item.request_id))[
            : self.max_batch
        ]
        return Decision("launch", tuple(item.request_id for item in chosen), reason="earliest-deadline")


class LeastLaxityPolicy:
    name = "least_laxity"

    def __init__(self, max_batch: int) -> None:
        self.max_batch = _positive_int(max_batch, "max_batch")

    def decide(self, now_us: float, ready: Sequence[RequestSpec], surface: ServiceCatalog) -> Decision:
        chosen = sorted(
            ready,
            key=lambda item: (
                item.deadline_us - now_us - surface.estimate_batch_us((item,)),
                item.deadline_us,
                item.request_id,
            ),
        )[: self.max_batch]
        return Decision("launch", tuple(item.request_id for item in chosen), reason="least-laxity")


class DeterministicRandomPolicy:
    name = "random"

    def __init__(self, max_batch: int, seed: int) -> None:
        self.max_batch = _positive_int(max_batch, "max_batch")
        self.seed = int(seed)

    def decide(self, now_us: float, ready: Sequence[RequestSpec], surface: ServiceCatalog) -> Decision:
        del surface
        chosen = list(_ordered_ready(ready))
        material = f"{self.seed}|{now_us:.9f}|" + "|".join(item.request_id for item in chosen)
        stable_seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
        random.Random(stable_seed).shuffle(chosen)
        chosen = chosen[: self.max_batch]
        return Decision("launch", tuple(item.request_id for item in chosen), reason="seeded-random")


class ThresholdPolicy:
    name = "threshold"

    def __init__(self, max_batch: int, target_batch: int, max_wait_us: float) -> None:
        self.max_batch = _positive_int(max_batch, "max_batch")
        self.target_batch = _positive_int(target_batch, "target_batch")
        if self.target_batch > self.max_batch:
            raise ValueError("target_batch cannot exceed max_batch")
        if not math.isfinite(max_wait_us) or max_wait_us < 0:
            raise ValueError("max_wait_us must be finite and non-negative")
        self.max_wait_us = max_wait_us

    def decide(self, now_us: float, ready: Sequence[RequestSpec], surface: ServiceCatalog) -> Decision:
        ordered = _ordered_ready(ready)
        candidate = ordered[: self.max_batch]
        duration = surface.estimate_batch_us(candidate)
        oldest = ordered[0]
        urgent = min(item.deadline_us for item in candidate) <= now_us + duration
        wake = oldest.arrival_us + self.max_wait_us
        if len(ordered) >= self.target_batch or urgent or wake <= now_us:
            return Decision("launch", tuple(item.request_id for item in candidate), reason="threshold")
        return Decision("defer", wake_us=wake, reason="wait-for-batch")


class GreedyPressureSlackPolicy:
    name = "greedy_pressure_slack"

    def __init__(self, max_batch: int) -> None:
        self.max_batch = _positive_int(max_batch, "max_batch")

    def decide(self, now_us: float, ready: Sequence[RequestSpec], surface: ServiceCatalog) -> Decision:
        remaining = list(ready)
        anchor = min(
            remaining,
            key=lambda item: (
                item.deadline_us - now_us - surface.estimate_batch_us((item,)),
                item.request_id,
            ),
        )
        chosen = [anchor]
        remaining.remove(anchor)
        while remaining and len(chosen) < self.max_batch:
            before = surface.estimate_batch_us(chosen)
            best = max(
                remaining,
                key=lambda item: (
                    surface.estimate_batch_us((item,)) + before
                    - surface.estimate_batch_us((*chosen, item)),
                    -(item.deadline_us - now_us),
                    item.request_id,
                ),
            )
            trial = (*chosen, best)
            completion = now_us + surface.estimate_batch_us(trial)
            if completion > min(item.deadline_us for item in trial):
                break
            chosen.append(best)
            remaining.remove(best)
        return Decision("launch", tuple(item.request_id for item in chosen), reason="greedy-pressure-slack")


class DEPARollingPolicy:
    """Causal deadline/expert-pressure policy over currently ready requests only."""

    name = "depa_rolling"

    def __init__(
        self,
        max_batch: int,
        *,
        max_candidates: int = 10,
        min_batch: int = 2,
        max_wait_us: float = 10.0,
        reject_infeasible: bool = True,
    ) -> None:
        self.max_batch = _positive_int(max_batch, "max_batch")
        self.max_candidates = _positive_int(max_candidates, "max_candidates")
        self.min_batch = _positive_int(min_batch, "min_batch")
        if self.min_batch > self.max_batch:
            raise ValueError("min_batch cannot exceed max_batch")
        if not math.isfinite(max_wait_us) or max_wait_us < 0:
            raise ValueError("max_wait_us must be finite and non-negative")
        self.max_wait_us = max_wait_us
        self.reject_infeasible = bool(reject_infeasible)

    def decide(self, now_us: float, ready: Sequence[RequestSpec], surface: ServiceCatalog) -> Decision:
        candidates = sorted(
            ready,
            key=lambda item: (item.deadline_us, item.arrival_us, item.request_id),
        )[: self.max_candidates]
        individually_infeasible = [
            item
            for item in candidates
            if now_us + surface.estimate_batch_us((item,)) > item.deadline_us
        ]
        if individually_infeasible and self.reject_infeasible:
            victim = min(
                individually_infeasible,
                key=lambda item: (item.deadline_us, item.arrival_us, item.request_id),
            )
            return Decision("reject", (victim.request_id,), reason="cannot-meet-deadline")

        best_batch: tuple[RequestSpec, ...] | None = None
        best_score: tuple[float, ...] | None = None
        limit = min(self.max_batch, len(candidates))
        for size in range(1, limit + 1):
            for batch in combinations(candidates, size):
                service = surface.estimate_batch_us(batch)
                completion = now_us + service
                on_time = sum(completion <= item.deadline_us for item in batch)
                min_slack_after = min(item.deadline_us - completion for item in batch)
                saving = surface.coalescing_saving_fraction(batch)
                # Lexicographic: protect SLO count first, then amortize expert
                # pressure, then retain slack and prefer deterministic IDs.
                score = (
                    float(on_time),
                    saving,
                    min_slack_after / max(service, 1e-9),
                    float(size),
                )
                if best_score is None or score > best_score or (
                    score == best_score
                    and tuple(item.request_id for item in batch)
                    < tuple(item.request_id for item in best_batch or ())
                ):
                    best_score = score
                    best_batch = batch
        if best_batch is None:
            raise InfeasibleSchedule("DEPA found no executable ready subset")

        oldest = min(candidates, key=lambda item: (item.arrival_us, item.request_id))
        wake = oldest.arrival_us + self.max_wait_us
        urgent = min(item.deadline_us for item in best_batch) <= (
            now_us + surface.estimate_batch_us(best_batch)
        )
        if len(best_batch) < self.min_batch and wake > now_us and not urgent:
            return Decision("defer", wake_us=wake, reason="bounded-pressure-wait")
        return Decision(
            "launch",
            tuple(item.request_id for item in best_batch),
            reason="rolling-pressure-slack",
        )


@dataclass(frozen=True)
class Action:
    kind: str
    start_us: float
    end_us: float
    request_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class LedgerEntry:
    request_id: str
    disposition: str
    terminal_us: float | None
    deadline_us: float
    reason: str

    @property
    def on_time(self) -> bool:
        return self.disposition == "completed" and self.terminal_us is not None and self.terminal_us <= self.deadline_us


@dataclass(frozen=True)
class ReplayResult:
    arm: str
    actions: tuple[Action, ...]
    ledger: tuple[LedgerEntry, ...]
    decision_overhead_us: float


def simulate_causal(
    requests: Sequence[RequestSpec],
    surface: ServiceCatalog,
    policy: OnlinePolicy,
    *,
    arm: str | None = None,
    observation_end_us: float | None = None,
    max_decisions: int = 100_000,
) -> ReplayResult:
    ordered = validate_requests(requests)
    if observation_end_us is not None and (
        not math.isfinite(observation_end_us) or observation_end_us < ordered[0].arrival_us
    ):
        raise ValueError("observation_end_us must be finite and not precede the episode")
    pending = {item.request_id: item for item in ordered}
    terminal: dict[str, LedgerEntry] = {}
    actions: list[Action] = []
    now_us = ordered[0].arrival_us
    decision_overhead_us = 0.0
    decisions = 0

    while pending:
        if observation_end_us is not None and now_us >= observation_end_us:
            break
        ready = tuple(
            sorted(
                (item for item in pending.values() if item.arrival_us <= now_us + 1e-12),
                key=lambda item: (item.arrival_us, item.request_id),
            )
        )
        if not ready:
            next_arrival = min(item.arrival_us for item in pending.values())
            now_us = next_arrival if observation_end_us is None else min(next_arrival, observation_end_us)
            continue
        decisions += 1
        if decisions > max_decisions:
            raise InfeasibleSchedule("policy exceeded max_decisions")
        started_ns = time.perf_counter_ns()
        decision = policy.decide(now_us, ready, surface)
        decision_overhead_us += (time.perf_counter_ns() - started_ns) / 1000.0
        ready_by_id = {item.request_id: item for item in ready}
        if decision.kind == "defer":
            assert decision.wake_us is not None
            if decision.wake_us <= now_us + 1e-12:
                raise ProtocolError("policy defer must advance time")
            future_arrivals = [item.arrival_us for item in pending.values() if item.arrival_us > now_us]
            next_event = min([decision.wake_us, *future_arrivals]) if future_arrivals else decision.wake_us
            now_us = next_event if observation_end_us is None else min(next_event, observation_end_us)
            continue

        if len(set(decision.request_ids)) != len(decision.request_ids):
            raise ProtocolError("decision contains duplicate request_ids")
        unknown = set(decision.request_ids) - set(ready_by_id)
        if unknown:
            raise ProtocolError(f"policy selected non-ready requests: {sorted(unknown)}")
        chosen = tuple(ready_by_id[item_id] for item_id in decision.request_ids)
        if decision.kind == "reject":
            for item in chosen:
                terminal[item.request_id] = LedgerEntry(
                    item.request_id, "rejected", now_us, item.deadline_us, decision.reason
                )
                del pending[item.request_id]
            actions.append(Action("reject", now_us, now_us, decision.request_ids, decision.reason))
            continue

        duration = surface.estimate_batch_us(chosen)
        end_us = now_us + duration
        for item in chosen:
            terminal[item.request_id] = LedgerEntry(
                item.request_id, "completed", end_us, item.deadline_us, decision.reason
            )
            del pending[item.request_id]
        actions.append(Action("launch", now_us, end_us, decision.request_ids, decision.reason))
        now_us = end_us

    for item in pending.values():
        terminal[item.request_id] = LedgerEntry(
            item.request_id, "pending", None, item.deadline_us, "observation-window-end"
        )
    result = ReplayResult(
        arm=arm or policy.name,
        actions=tuple(actions),
        ledger=tuple(terminal[item.request_id] for item in ordered),
        decision_overhead_us=decision_overhead_us,
    )
    validate_ledger(ordered, result)
    return result


def validate_ledger(requests: Sequence[RequestSpec], result: ReplayResult) -> None:
    expected = {item.request_id for item in requests}
    actual = [entry.request_id for entry in result.ledger]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ProtocolError("ledger must contain every offered request exactly once")
    valid = {"completed", "rejected", "pending"}
    if any(entry.disposition not in valid for entry in result.ledger):
        raise ProtocolError("ledger contains unknown disposition")
    launched = [request_id for action in result.actions if action.kind == "launch" for request_id in action.request_ids]
    completed = [entry.request_id for entry in result.ledger if entry.disposition == "completed"]
    if sorted(launched) != sorted(completed) or len(launched) != len(set(launched)):
        raise ProtocolError("completed requests must have exactly one launch action")


@dataclass(frozen=True)
class ScheduleMetrics:
    offered: int
    completed: int
    on_time: int
    rejected: int
    pending: int
    slo_goodput_per_s: float
    slo_attainment: float
    p99_completion_latency_us: float
    jain_class_fairness: float
    modeled_service_us: float
    decision_overhead_us: float
    decision_overhead_fraction: float
    launches: int


def schedule_metrics(
    requests: Sequence[RequestSpec],
    result: ReplayResult,
    *,
    window_start_us: float,
    window_end_us: float,
) -> ScheduleMetrics:
    if not math.isfinite(window_start_us) or not math.isfinite(window_end_us) or window_end_us <= window_start_us:
        raise ValueError("observation window must be finite and positive")
    by_id = {item.request_id: item for item in requests}
    completed = [entry for entry in result.ledger if entry.disposition == "completed"]
    on_time = sum(entry.on_time for entry in result.ledger)
    latencies = [
        entry.terminal_us - by_id[entry.request_id].arrival_us
        for entry in completed
        if entry.terminal_us is not None
    ]
    class_totals: dict[str, int] = {}
    class_good: dict[str, int] = {}
    for entry in result.ledger:
        request_class = by_id[entry.request_id].request_class
        class_totals[request_class] = class_totals.get(request_class, 0) + 1
        class_good[request_class] = class_good.get(request_class, 0) + int(entry.on_time)
    rates = [class_good[name] / total for name, total in class_totals.items()]
    fairness = _jain(rates)
    launches = [action for action in result.actions if action.kind == "launch"]
    modeled_service = sum(action.end_us - action.start_us for action in launches)
    overhead_fraction = result.decision_overhead_us / modeled_service if modeled_service > 0 else math.inf
    return ScheduleMetrics(
        offered=len(result.ledger),
        completed=len(completed),
        on_time=on_time,
        rejected=sum(entry.disposition == "rejected" for entry in result.ledger),
        pending=sum(entry.disposition == "pending" for entry in result.ledger),
        slo_goodput_per_s=on_time / ((window_end_us - window_start_us) / 1_000_000.0),
        slo_attainment=on_time / len(result.ledger),
        p99_completion_latency_us=_percentile(latencies, 0.99) if latencies else math.inf,
        jain_class_fairness=fairness,
        modeled_service_us=modeled_service,
        decision_overhead_us=result.decision_overhead_us,
        decision_overhead_fraction=overhead_fraction,
        launches=len(launches),
    )


@dataclass(frozen=True)
class _OraclePlan:
    on_time: int
    late_completed: int
    rejected: int
    terminal_us: float
    actions: tuple[Action, ...]

    @property
    def score(self) -> tuple[float, ...]:
        return (
            float(self.on_time),
            float(-self.late_completed),
            float(-self.rejected),
            -self.terminal_us,
        )


def exact_slo_goodput_oracle(
    requests: Sequence[RequestSpec],
    surface: ServiceCatalog,
    *,
    max_batch: int,
    max_exact_requests: int = 12,
) -> ReplayResult:
    """Clairvoyant exact upper bound; never a deployable online policy."""

    ordered = validate_requests(requests)
    max_batch = _positive_int(max_batch, "max_batch")
    if len(ordered) > max_exact_requests:
        raise ProtocolError(
            f"exact oracle supports at most {max_exact_requests} requests, got {len(ordered)}"
        )
    all_mask = (1 << len(ordered)) - 1
    id_to_index = {item.request_id: index for index, item in enumerate(ordered)}

    def candidates(mask: int, now_us: float) -> tuple[int, ...]:
        return tuple(
            index
            for index, item in enumerate(ordered)
            if not (mask & (1 << index)) and item.arrival_us <= now_us + 1e-12
        )

    def better(left: _OraclePlan | None, right: _OraclePlan) -> _OraclePlan:
        if left is None or right.score > left.score:
            return right
        if right.score == left.score:
            left_key = tuple((action.kind, action.request_ids) for action in left.actions)
            right_key = tuple((action.kind, action.request_ids) for action in right.actions)
            if right_key < left_key:
                return right
        return left

    @lru_cache(maxsize=None)
    def solve(mask: int, rounded_now_us: float) -> _OraclePlan:
        now_us = rounded_now_us
        if mask == all_mask:
            return _OraclePlan(0, 0, 0, now_us, ())
        ready_indices = candidates(mask, now_us)
        if not ready_indices:
            next_arrival = min(
                item.arrival_us
                for index, item in enumerate(ordered)
                if not (mask & (1 << index))
            )
            return solve(mask, round(next_arrival, 9))

        best: _OraclePlan | None = None
        for size in range(1, min(max_batch, len(ready_indices)) + 1):
            for selected in combinations(ready_indices, size):
                batch = tuple(ordered[index] for index in selected)
                end_us = round(now_us + surface.estimate_batch_us(batch), 9)
                next_mask = mask
                for index in selected:
                    next_mask |= 1 << index
                child = solve(next_mask, end_us)
                on_time = sum(end_us <= ordered[index].deadline_us for index in selected)
                action = Action(
                    "launch",
                    now_us,
                    end_us,
                    tuple(ordered[index].request_id for index in selected),
                    "offline-exact-oracle",
                )
                plan = _OraclePlan(
                    on_time + child.on_time,
                    size - on_time + child.late_completed,
                    child.rejected,
                    child.terminal_us,
                    (action, *child.actions),
                )
                best = better(best, plan)

        # Explicit finite-admission rejection is part of the action space.
        for index in ready_indices:
            next_mask = mask | (1 << index)
            child = solve(next_mask, rounded_now_us)
            action = Action(
                "reject",
                now_us,
                now_us,
                (ordered[index].request_id,),
                "offline-capacity-reject",
            )
            plan = _OraclePlan(
                child.on_time,
                child.late_completed,
                child.rejected + 1,
                child.terminal_us,
                (action, *child.actions),
            )
            best = better(best, plan)

        future_arrivals = [
            item.arrival_us
            for index, item in enumerate(ordered)
            if not (mask & (1 << index)) and item.arrival_us > now_us
        ]
        if future_arrivals:
            child = solve(mask, round(min(future_arrivals), 9))
            best = better(best, child)
        if best is None:
            raise InfeasibleSchedule("oracle found no terminal plan")
        return best

    plan = solve(0, round(ordered[0].arrival_us, 9))
    terminal: dict[str, LedgerEntry] = {}
    for action in plan.actions:
        for request_id in action.request_ids:
            item = ordered[id_to_index[request_id]]
            terminal[request_id] = LedgerEntry(
                request_id,
                "completed" if action.kind == "launch" else "rejected",
                action.end_us,
                item.deadline_us,
                action.reason,
            )
    result = ReplayResult(
        arm="oracle",
        actions=plan.actions,
        ledger=tuple(terminal[item.request_id] for item in ordered),
        decision_overhead_us=0.0,
    )
    validate_ledger(ordered, result)
    return result


def relative_gain(candidate: float, baseline: float) -> float:
    if baseline < 0 or candidate < 0 or not math.isfinite(baseline) or not math.isfinite(candidate):
        raise ValueError("gain inputs must be finite and non-negative")
    if baseline == 0:
        return math.inf if candidate > 0 else 0.0
    return (candidate - baseline) / baseline


def paired_bootstrap_lcb95(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap requires equal non-empty samples")
    _positive_int(replicates, "replicates")
    gains = [relative_gain(cand, base) for base, cand in zip(baseline, candidate)]
    if any(not math.isfinite(value) for value in gains):
        raise ProtocolError("bootstrap gain is non-finite")
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        sampled = [gains[rng.randrange(len(gains))] for _ in gains]
        means.append(sum(sampled) / len(sampled))
    return sum(gains) / len(gains), _percentile(means, 0.05)


def bootstrap_fraction_lcb95(
    values: Sequence[float], *, replicates: int, seed: int
) -> tuple[float, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be finite and non-empty")
    rng = random.Random(seed)
    means = []
    for _ in range(_positive_int(replicates, "replicates")):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    return sum(values) / len(values), _percentile(means, 0.05)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _jain(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0:
        return 1.0
    return sum(values) ** 2 / denominator
