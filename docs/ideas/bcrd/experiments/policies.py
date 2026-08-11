from __future__ import annotations

"""Causal assignment policies for the BCRD fixed-replica action space."""

from dataclasses import dataclass
from itertools import groupby
import math
import random
from typing import Mapping, Protocol, Sequence

try:
    from .core import CausalReplayEngine, Contribution, ReplayConfig, ServiceCatalog, stable_index
except ImportError:
    from core import CausalReplayEngine, Contribution, ReplayConfig, ServiceCatalog, stable_index


@dataclass(frozen=True)
class OnlineContributionView:
    """Causal policy input with observed execution suffixes intentionally removed."""

    model: str
    phase: str
    request_id: str
    input_event_id: str
    contribution_id: str
    request_arrival_us: float
    dispatch_ready_us: float
    deadline_us: float
    layer: int
    expert_id: int
    gate_weight: float
    source_rank: int
    legal_replica_set: tuple[int, ...]

    @classmethod
    def from_contribution(cls, item: Contribution) -> "OnlineContributionView":
        return cls(
            model=item.model,
            phase=item.phase,
            request_id=item.request_id,
            input_event_id=item.input_event_id,
            contribution_id=item.contribution_id,
            request_arrival_us=item.request_arrival_us,
            dispatch_ready_us=item.dispatch_ready_us,
            deadline_us=item.deadline_us,
            layer=item.layer,
            expert_id=item.expert_id,
            gate_weight=item.gate_weight,
            source_rank=item.source_rank,
            legal_replica_set=item.legal_replica_set,
        )

    def legal_replicas(self, replica_count: int) -> tuple[int, ...]:
        legal = self.legal_replica_set or tuple(range(replica_count))
        if any(replica < 0 or replica >= replica_count for replica in legal):
            raise ValueError("online contribution has an out-of-range legal replica")
        return legal


@dataclass(frozen=True)
class AssignmentState:
    """Read-only causal prefix snapshot with no engine or future trace access."""

    replica_count: int
    _replica_available_us: tuple[float, ...]
    _current_contribution_id: str
    _predictions: Mapping[int, Mapping[str, float]]
    hold_us: float
    decision_end_us: float | None = None

    @property
    def replica_available_us(self) -> tuple[float, ...]:
        return self._replica_available_us

    def predict(self, item: OnlineContributionView, replica: int) -> dict[str, float]:
        if item.contribution_id != self._current_contribution_id:
            raise ValueError("policy prediction requested for a non-current contribution")
        if replica not in self._predictions:
            raise ValueError("policy prediction requested for an illegal replica")
        return dict(self._predictions[replica])

    def joinable_rows(self, item: OnlineContributionView, replica: int) -> int:
        return max(0, int(self.predict(item, replica)["batch_rows"]) - 1)


class Policy(Protocol):
    name: str

    def choose(self, item: OnlineContributionView, state: AssignmentState, catalog: ServiceCatalog) -> int:
        """Choose using only the current item and prefix state; no future trace is exposed."""


@dataclass(frozen=True)
class HashPolicy:
    seed: int = 0
    name: str = "current_hash"

    def choose(self, item: OnlineContributionView, state: AssignmentState, catalog: ServiceCatalog) -> int:
        legal = item.legal_replicas(state.replica_count)
        return legal[stable_index(item.contribution_id, len(legal), seed=self.seed)]


@dataclass
class RandomPolicy:
    seed: int = 0
    name: str = "random"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose(self, item: OnlineContributionView, state: AssignmentState, catalog: ServiceCatalog) -> int:
        legal = item.legal_replicas(state.replica_count)
        return legal[self._rng.randrange(len(legal))]


@dataclass(frozen=True)
class LeastLoadPolicy:
    remote_latency_us: float = 0.0
    name: str = "current_least_load"

    def choose(self, item: OnlineContributionView, state: AssignmentState, catalog: ServiceCatalog) -> int:
        return min(
            item.legal_replicas(state.replica_count),
            key=lambda replica: (
                state.replica_available_us[replica],
                replica,
            ),
        )


@dataclass(frozen=True)
class ThresholdPolicy:
    row_threshold: int
    remote_latency_us: float = 0.0
    name: str = "threshold"

    def choose(self, item: OnlineContributionView, state: AssignmentState, catalog: ServiceCatalog) -> int:
        fullest = max(
            item.legal_replicas(state.replica_count),
            key=lambda replica: (
                state.joinable_rows(item, replica),
                -state.predict(item, replica)["completion_us"],
                -replica,
            ),
        )
        if state.joinable_rows(item, fullest) < self.row_threshold:
            return fullest
        return LeastLoadPolicy(self.remote_latency_us).choose(item, state, catalog)


@dataclass(frozen=True)
class GreedyCompletionPolicy:
    remote_latency_us: float = 0.0
    name: str = "greedy"

    def choose(self, item: OnlineContributionView, state: AssignmentState, catalog: ServiceCatalog) -> int:
        def score(replica: int) -> tuple[float, int]:
            return state.predict(item, replica)["completion_us"], replica

        return min(item.legal_replicas(state.replica_count), key=score)


@dataclass(frozen=True)
class BCRDPolicy:
    remote_latency_us: float = 0.0
    batching_credit_weight: float = 1.0
    deadline_risk_weight: float = 4.0
    name: str = "bcrd"

    def choose(self, item: OnlineContributionView, state: AssignmentState, catalog: ServiceCatalog) -> int:
        singleton = catalog.estimate_us(item.model, item.layer, 1)

        def score(replica: int) -> tuple[float, float, int]:
            prediction = state.predict(item, replica)
            marginal = prediction["marginal_service_us"]
            batching_credit = max(0.0, singleton - marginal)
            predicted = prediction["completion_us"]
            slack = max(item.deadline_us - item.request_arrival_us, 1e-9)
            risk = max(0.0, predicted - item.deadline_us) / slack
            value = (
                predicted
                + self.deadline_risk_weight * risk * slack
                - self.batching_credit_weight * batching_credit
            )
            if not math.isfinite(value):
                raise ValueError("non-finite BCRD score")
            return value, prediction["projected_replica_finish_us"], replica

        return min(item.legal_replicas(state.replica_count), key=score)


def assign_online(
    contributions: Sequence[Contribution],
    policy: Policy,
    catalog: ServiceCatalog,
    replica_count: int,
    *,
    hold_us: float = 0.0,
    remote_bytes_per_row: int = 0,
    max_batch_rows: int | None = None,
    controller_latency_us: float = 0.0,
    seal_cost_us: float = 0.0,
    launch_cost_us: float = 0.0,
    remote_latency_us: float | None = None,
) -> list[int]:
    assignments, _metrics = simulate_online_policy(
        contributions,
        policy,
        catalog,
        ReplayConfig(
            replica_count,
            hold_us=hold_us,
            remote_latency_us=(
                float(getattr(policy, "remote_latency_us", 0.0))
                if remote_latency_us is None
                else float(remote_latency_us)
            ),
            remote_bytes_per_row=remote_bytes_per_row,
            max_batch_rows=max_batch_rows,
            controller_latency_us=controller_latency_us,
            seal_cost_us=seal_cost_us,
            launch_cost_us=launch_cost_us,
        ),
    )
    return assignments


def simulate_online_policy(
    contributions: Sequence[Contribution],
    policy: Policy,
    catalog: ServiceCatalog,
    config: ReplayConfig,
) -> tuple[list[int], dict[str, object]]:
    """Choose actions on a causal prefix and execute them in the same engine."""

    if not contributions:
        raise ValueError("online replay requires contributions")
    model, layer = contributions[0].model, contributions[0].layer
    if any(item.model != model or item.layer != layer for item in contributions):
        raise ValueError("online replay requires one model and layer")
    engine = CausalReplayEngine(catalog, config, model=model, layer=layer)
    indexed = sorted(
        enumerate(contributions),
        key=lambda value: (
            value[1].dispatch_ready_us,
            value[1].deadline_us,
            value[1].contribution_id,
        ),
    )
    assignments = [-1] * len(contributions)
    decision_cursor_us = 0.0
    for ready_us, group in groupby(indexed, key=lambda value: value[1].dispatch_ready_us):
        engine.advance_to(float(ready_us))
        for index, item in group:
            decision_end_us = max(float(ready_us), decision_cursor_us) + config.controller_latency_us
            online_item = OnlineContributionView.from_contribution(item)
            legal = item.legal_replicas(config.replica_count)
            predictions = {
                replica: engine.predict_submission(
                    item,
                    replica,
                    hold_us=config.hold_us,
                    decision_end_us=decision_end_us,
                )
                for replica in legal
            }
            state = AssignmentState(
                config.replica_count,
                tuple(engine.projected_available_us()),
                item.contribution_id,
                predictions,
                config.hold_us,
                decision_end_us,
            )
            replica = policy.choose(online_item, state, catalog)
            if replica not in item.legal_replicas(config.replica_count):
                raise ValueError(f"policy {policy.name} chose illegal replica {replica}")
            assignments[index] = replica
            # Submissions at the same route timestamp remain pending until the
            # group is complete. Thus all same-time arrivals precede hold=0 seal.
            engine.submit(
                index,
                item,
                replica,
                hold_us=config.hold_us,
                decision_end_us=decision_end_us,
            )
            decision_cursor_us = decision_end_us
        engine.advance_to(float(ready_us))
    engine.run_all()
    return assignments, engine.metrics()


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
