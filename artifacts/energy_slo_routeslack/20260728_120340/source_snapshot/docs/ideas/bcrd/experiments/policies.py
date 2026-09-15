from __future__ import annotations

"""Causal assignment policies for the BCRD fixed-replica action space."""

from dataclasses import dataclass, field
import math
import random
from typing import Protocol, Sequence

try:
    from .core import Contribution, ServiceCatalog, stable_index
except ImportError:
    from core import Contribution, ServiceCatalog, stable_index


@dataclass
class AssignmentState:
    replica_count: int
    expert_rows: dict[tuple[int, int], int] = field(default_factory=dict)
    replica_available_us: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.replica_available_us = [0.0] * self.replica_count

    def rows(self, replica: int, expert: int) -> int:
        return self.expert_rows.get((replica, expert), 0)

    def add(
        self,
        item: Contribution,
        replica: int,
        catalog: ServiceCatalog,
        *,
        remote_latency_us: float,
    ) -> None:
        old_rows = self.rows(replica, item.expert_id)
        old_service = catalog.estimate_us(item.model, item.layer, old_rows) if old_rows else 0.0
        new_service = catalog.estimate_us(item.model, item.layer, old_rows + 1)
        self.expert_rows[(replica, item.expert_id)] = old_rows + 1
        ready = item.arrival_us + (remote_latency_us if replica != item.src_replica else 0.0)
        self.replica_available_us[replica] = max(self.replica_available_us[replica], ready) + (
            new_service - old_service
        )


class Policy(Protocol):
    name: str

    def choose(self, item: Contribution, state: AssignmentState, catalog: ServiceCatalog) -> int:
        """Choose using only the current item and prefix state; no future trace is exposed."""


@dataclass(frozen=True)
class HashPolicy:
    seed: int = 0
    name: str = "current_hash"

    def choose(self, item: Contribution, state: AssignmentState, catalog: ServiceCatalog) -> int:
        return stable_index(item.contribution_id, state.replica_count, seed=self.seed)


@dataclass
class RandomPolicy:
    seed: int = 0
    name: str = "random"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose(self, item: Contribution, state: AssignmentState, catalog: ServiceCatalog) -> int:
        return self._rng.randrange(state.replica_count)


@dataclass(frozen=True)
class LeastLoadPolicy:
    remote_latency_us: float = 0.0
    name: str = "current_least_load"

    def choose(self, item: Contribution, state: AssignmentState, catalog: ServiceCatalog) -> int:
        return min(
            range(state.replica_count),
            key=lambda replica: (
                state.replica_available_us[replica]
                + (self.remote_latency_us if replica != item.src_replica else 0.0),
                state.rows(replica, item.expert_id),
                replica,
            ),
        )


@dataclass(frozen=True)
class ThresholdPolicy:
    row_threshold: int
    remote_latency_us: float = 0.0
    name: str = "threshold"

    def choose(self, item: Contribution, state: AssignmentState, catalog: ServiceCatalog) -> int:
        fullest = max(
            range(state.replica_count),
            key=lambda replica: (state.rows(replica, item.expert_id), -state.replica_available_us[replica], -replica),
        )
        if state.rows(fullest, item.expert_id) < self.row_threshold:
            return fullest
        return LeastLoadPolicy(self.remote_latency_us).choose(item, state, catalog)


@dataclass(frozen=True)
class GreedyCompletionPolicy:
    remote_latency_us: float = 0.0
    name: str = "greedy"

    def choose(self, item: Contribution, state: AssignmentState, catalog: ServiceCatalog) -> int:
        def score(replica: int) -> tuple[float, int]:
            old_rows = state.rows(replica, item.expert_id)
            old = catalog.estimate_us(item.model, item.layer, old_rows) if old_rows else 0.0
            new = catalog.estimate_us(item.model, item.layer, old_rows + 1)
            remote = self.remote_latency_us if replica != item.src_replica else 0.0
            predicted = max(item.arrival_us + remote, state.replica_available_us[replica]) + (new - old)
            return predicted, replica

        return min(range(state.replica_count), key=score)


@dataclass(frozen=True)
class BCRDPolicy:
    remote_latency_us: float = 0.0
    batching_credit_weight: float = 1.0
    deadline_risk_weight: float = 4.0
    name: str = "bcrd"

    def choose(self, item: Contribution, state: AssignmentState, catalog: ServiceCatalog) -> int:
        singleton = catalog.estimate_us(item.model, item.layer, 1)

        def score(replica: int) -> tuple[float, float, int]:
            old_rows = state.rows(replica, item.expert_id)
            old = catalog.estimate_us(item.model, item.layer, old_rows) if old_rows else 0.0
            new = catalog.estimate_us(item.model, item.layer, old_rows + 1)
            marginal = new - old
            batching_credit = max(0.0, singleton - marginal)
            remote = self.remote_latency_us if replica != item.src_replica else 0.0
            predicted = max(item.arrival_us + remote, state.replica_available_us[replica]) + marginal
            slack = max(item.deadline_us - item.arrival_us, 1e-9)
            risk = max(0.0, predicted - item.deadline_us) / slack
            value = (
                predicted
                + remote
                + self.deadline_risk_weight * risk * slack
                - self.batching_credit_weight * batching_credit
            )
            if not math.isfinite(value):
                raise ValueError("non-finite BCRD score")
            return value, state.replica_available_us[replica], replica

        return min(range(state.replica_count), key=score)


def assign_online(
    contributions: Sequence[Contribution], policy: Policy, catalog: ServiceCatalog, replica_count: int
) -> list[int]:
    state = AssignmentState(replica_count)
    indexed = sorted(
        enumerate(contributions),
        key=lambda value: (value[1].arrival_us, value[1].deadline_us, value[1].contribution_id),
    )
    assignments = [-1] * len(contributions)
    for index, item in indexed:
        replica = policy.choose(item, state, catalog)
        if replica < 0 or replica >= replica_count:
            raise ValueError(f"policy {policy.name} chose illegal replica {replica}")
        assignments[index] = replica
        state.add(
            item,
            replica,
            catalog,
            remote_latency_us=float(getattr(policy, "remote_latency_us", 0.0)),
        )
    return assignments


def make_policy(
    name: str,
    *,
    seed: int,
    remote_latency_us: float,
    row_threshold: int = 4,
    batching_credit_weight: float = 1.0,
    deadline_risk_weight: float = 4.0,
) -> Policy:
    if name == "current_hash":
        return HashPolicy(seed)
    if name == "current_least_load":
        return LeastLoadPolicy(remote_latency_us)
    if name == "random":
        return RandomPolicy(seed)
    if name == "threshold":
        return ThresholdPolicy(row_threshold, remote_latency_us)
    if name == "greedy":
        return GreedyCompletionPolicy(remote_latency_us)
    if name == "bcrd":
        return BCRDPolicy(remote_latency_us, batching_credit_weight, deadline_risk_weight)
    raise ValueError(f"unknown policy {name!r}")
