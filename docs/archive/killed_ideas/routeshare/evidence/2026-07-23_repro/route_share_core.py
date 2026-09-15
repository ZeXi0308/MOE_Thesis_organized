"""Pure helpers for the RouteShare Gate-0 mechanism screen.

This module deliberately has no torch/numpy dependency so its invariants can be
checked on the local CPU-only workspace before the CUDA runner is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RoutePlan:
    plan_id: str
    total_rows: int
    counts: tuple[int, ...]
    active_experts: tuple[int, ...]
    shape: str
    replica: int

    def validate(self, num_experts: int) -> None:
        if self.total_rows <= 0:
            raise ValueError("total_rows must be positive")
        if len(self.counts) != len(self.active_experts) or not self.counts:
            raise ValueError("counts and active_experts must be non-empty and aligned")
        if sum(self.counts) != self.total_rows or any(value <= 0 for value in self.counts):
            raise ValueError("counts must be positive and sum exactly to total_rows")
        if len(set(self.active_experts)) != len(self.active_experts):
            raise ValueError("active experts must be unique")
        if min(self.active_experts) < 0 or max(self.active_experts) >= num_experts:
            raise ValueError("expert id outside model range")

    @property
    def active_count(self) -> int:
        return len(self.counts)

    @property
    def max_fraction(self) -> float:
        return max(self.counts) / self.total_rows

    @property
    def cv(self) -> float:
        mean = self.total_rows / self.active_count
        variance = sum((value - mean) ** 2 for value in self.counts) / self.active_count
        return math.sqrt(variance) / mean


def _positive_partition(total: int, weights: Sequence[float]) -> tuple[int, ...]:
    if total < len(weights) or any(value <= 0 for value in weights):
        raise ValueError("cannot form a positive partition")
    remaining = total - len(weights)
    raw = [remaining * value / sum(weights) for value in weights]
    base = [1 + int(math.floor(value)) for value in raw]
    missing = total - sum(base)
    order = sorted(range(len(weights)), key=lambda index: raw[index] - math.floor(raw[index]), reverse=True)
    for index in order[:missing]:
        base[index] += 1
    result = tuple(base)
    if sum(result) != total or any(value <= 0 for value in result):
        raise AssertionError("partition construction failed")
    return result


def counts_for_shape(total_rows: int, active_count: int, shape: str) -> tuple[int, ...]:
    if active_count < 1 or total_rows < active_count:
        raise ValueError("active_count must be in [1, total_rows]")
    if shape == "uniform":
        weights = [1.0] * active_count
    elif shape == "linear_skew":
        weights = [float(active_count - index) for index in range(active_count)]
    elif shape == "zipf":
        weights = [1.0 / (index + 1) for index in range(active_count)]
    elif shape == "hot50":
        if active_count == 1:
            weights = [1.0]
        else:
            weights = [float(active_count - 1)] + [1.0] * (active_count - 1)
    else:
        raise ValueError(f"unknown shape: {shape}")
    return _positive_partition(total_rows, weights)


def build_plans(
    *,
    num_experts: int,
    row_grid: Iterable[int],
    active_grid: Iterable[int],
    shapes: Sequence[str],
    replicas: int,
    seed: int,
) -> list[RoutePlan]:
    if replicas < 1:
        raise ValueError("replicas must be positive")
    rng = random.Random(seed)
    plans: list[RoutePlan] = []
    for total_rows in row_grid:
        for active_count in active_grid:
            if active_count > num_experts or active_count > total_rows:
                continue
            # Every histogram shape in one matched cell must use the exact same
            # experts. Otherwise weight identity/cache effects masquerade as a
            # route-histogram effect.
            for replica in range(replicas):
                expert_ids = tuple(sorted(rng.sample(range(num_experts), active_count)))
                for shape in shapes:
                    counts = counts_for_shape(total_rows, active_count, shape)
                    plan_id = f"r{total_rows}_a{active_count}_{shape}_rep{replica}"
                    plan = RoutePlan(plan_id, total_rows, counts, expert_ids, shape, replica)
                    plan.validate(num_experts)
                    plans.append(plan)
    if len({plan.plan_id for plan in plans}) != len(plans):
        raise AssertionError("duplicate plan ids")
    return plans


def simple_features(plan: RoutePlan, num_experts: int) -> tuple[float, ...]:
    """Strong separable baseline: rows, activity/skew, and expert identities."""
    identity = [0.0] * num_experts
    for expert_id in plan.active_experts:
        identity[expert_id] = 1.0
    return (
        1.0,
        float(plan.total_rows),
        float(plan.active_count),
        float(plan.max_fraction),
        float(plan.cv),
        *identity,
    )


def paired_shape_differences(rows: Sequence[dict[str, object]]) -> list[float]:
    """Within-block, rows/active/replica matched max-minus-min contrasts."""
    groups: dict[tuple[int, int, int, int], list[float]] = {}
    for row in rows:
        key = (
            int(row["block"]),
            int(row["total_rows"]),
            int(row["active_count"]),
            int(row["replica"]),
        )
        groups.setdefault(key, []).append(float(row["latency_us"]))
    return [max(values) - min(values) for values in groups.values() if len(values) >= 2]
