"""Complete-load virtual-EP scenario construction for RIC-v1.

The synthetic builder in this module is an implementation/capability fixture,
not a scientific data producer.  Formal route/LUT producers may construct the
same ``ReplayWorld`` type from their sealed artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import random
from typing import Iterable, Mapping, Sequence

try:  # Package import in formal runner.
    from .schema import (
        ContributionIdentity,
        ContributionRecord,
        FullLoadAudit,
        JoinIdentity,
        RICValidationError,
        validate_full_background,
    )
except ImportError:  # Direct test execution from this directory.
    from schema import (  # type: ignore
        ContributionIdentity,
        ContributionRecord,
        FullLoadAudit,
        JoinIdentity,
        RICValidationError,
        validate_full_background,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class StageService:
    """Three contribution stages plus one once-per-join combine service."""

    sender_pack_us: float
    shared_cut_us: float
    receiver_unpack_us: float
    join_combine_us: float
    sender_pack_source: str = "derived_from_measured_lut"
    cut_source: str = "analytic_network"
    receiver_unpack_source: str = "derived_from_measured_lut"
    join_combine_source: str = "measured_5090_cuda"

    def __post_init__(self) -> None:
        values = (
            self.sender_pack_us,
            self.shared_cut_us,
            self.receiver_unpack_us,
            self.join_combine_us,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise RICValidationError("stage service times must be finite and positive")
        allowed = {
            "measured_5090_cuda",
            "measured_5090_host",
            "measured_5090_h2d_not_rdma",
            "derived_from_measured_lut",
            "analytic_network",
            "synthetic_delay",
        }
        if {
            self.sender_pack_source,
            self.cut_source,
            self.receiver_unpack_source,
            self.join_combine_source,
        } - allowed:
            raise RICValidationError("stage service has an unknown source tag")

    @property
    def total_us(self) -> float:
        """Contribution service only; join combine is charged exactly once."""

        return math.fsum(
            (self.sender_pack_us, self.shared_cut_us, self.receiver_unpack_us)
        )

    @property
    def isolated_join_path_us(self) -> float:
        return self.total_us + self.join_combine_us

    @property
    def source_tags(self) -> tuple[str, str, str, str]:
        return (
            self.sender_pack_source,
            self.cut_source,
            self.receiver_unpack_source,
            self.join_combine_source,
        )

    # Read-only compatibility names for code that has not yet migrated its
    # display labels.  They never include the once-only join combine service.
    @property
    def sender_egress_us(self) -> float:
        return self.sender_pack_us

    @property
    def receiver_ingress_us(self) -> float:
        return self.receiver_unpack_us

    @property
    def sender_source(self) -> str:
        return self.sender_pack_source

    @property
    def ingress_source(self) -> str:
        return self.receiver_unpack_source


@dataclass(frozen=True)
class ReplayTask:
    """Canonical contribution plus explicit virtual resource path."""

    contribution: ContributionRecord
    stage_service: StageService
    sender_egress_resource: str
    shared_cut_resource: str
    receiver_ingress_resource: str

    def __post_init__(self) -> None:
        if type(self.contribution) is not ContributionRecord:
            raise RICValidationError("ReplayTask requires a canonical ContributionRecord")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.sender_egress_resource,
                self.shared_cut_resource,
                self.receiver_ingress_resource,
            )
        ):
            raise RICValidationError("ReplayTask resource path is incomplete")
        if not math.isclose(
            float(self.contribution.service_us),
            self.stage_service.total_us,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RICValidationError(
                "ContributionRecord service_us does not equal contribution-stage service"
            )

    @property
    def task_id(self) -> str:
        return self.contribution.stable_full_task_id

    @property
    def identity(self) -> ContributionIdentity:
        return self.contribution.identity

    @property
    def join_identity(self) -> JoinIdentity:
        return self.identity.join_identity

    @property
    def stage_resources(self) -> tuple[str, str, str]:
        return (
            self.sender_egress_resource,
            self.shared_cut_resource,
            self.receiver_ingress_resource,
        )

    @property
    def stage_service_us(self) -> tuple[float, float, float]:
        return (
            self.stage_service.sender_pack_us,
            self.stage_service.shared_cut_us,
            self.stage_service.receiver_unpack_us,
        )

    @property
    def combine_resource(self) -> str:
        return f"receiver:{self.identity.receiver_rank}:combine"


@dataclass(frozen=True)
class ReplayWorld:
    """One complete four-request workload trace and its score mask."""

    trace_id: str
    workload_seed: int
    model_key: str
    model_revision: str
    cell: str
    top_k: int
    num_experts: int
    ep_size: int
    ranks_per_node: int
    tasks: tuple[ReplayTask, ...]
    expected_request_ids: tuple[str, ...]
    expert_to_sender: Mapping[int, int]
    request_to_receiver: Mapping[str, int]
    expected_layer_by_request: Mapping[str, int]
    scored_joins: frozenset[JoinIdentity]
    full_load_audit: FullLoadAudit

    def __post_init__(self) -> None:
        if not self.trace_id or not self.model_key or not self.cell:
            raise RICValidationError("ReplayWorld identity is incomplete")
        if isinstance(self.workload_seed, bool) or not isinstance(
            self.workload_seed, int
        ):
            raise RICValidationError("ReplayWorld workload_seed must be an integer")
        if len(self.expected_request_ids) != 4:
            raise RICValidationError("RIC complete replay requires exactly four requests")
        if self.ranks_per_node <= 0 or self.ep_size % self.ranks_per_node:
            raise RICValidationError("invalid virtual-EP node geometry")
        if not 1 <= self.top_k <= 16 or self.num_experts < self.top_k:
            raise RICValidationError("invalid frozen model dimensions")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise RICValidationError("duplicate ReplayTask identity")
        if self.full_load_audit.record_count != len(self.tasks):
            raise RICValidationError("full-load audit/task count mismatch")
        expected_count = 4 * 128 * self.top_k
        if len(self.tasks) != expected_count:
            raise RICValidationError(
                f"complete trace contains {len(self.tasks)} tasks, expected {expected_count}"
            )
        if not self.scored_joins:
            raise RICValidationError("score mask cannot be empty")
        if set(self.expert_to_sender) != set(range(self.num_experts)):
            raise RICValidationError("expert placement does not cover the frozen model")
        if set(self.request_to_receiver) != set(self.expected_request_ids):
            raise RICValidationError("request-origin map does not cover the trace")
        if set(self.expected_layer_by_request) != set(self.expected_request_ids):
            raise RICValidationError("request-layer map does not cover the trace")
        for task in self.tasks:
            identity = task.identity
            expected_resources = (
                f"sender:{identity.sender_rank}:egress",
                (
                    f"cut:node{identity.sender_rank // self.ranks_per_node}"
                    f"->node{identity.receiver_rank // self.ranks_per_node}"
                ),
                f"receiver:{identity.receiver_rank}:ingress",
            )
            if task.stage_resources != expected_resources:
                raise RICValidationError("ReplayTask resource path disagrees with its identity")
        recomputed = validate_full_background(
            [task.contribution for task in self.tasks],
            top_k=self.top_k,
            num_experts=self.num_experts,
            ep_size=self.ep_size,
            expected_request_ids=self.expected_request_ids,
            expected_token_blocks_per_request=128,
            expert_to_sender=self.expert_to_sender,
            request_to_receiver=self.request_to_receiver,
            expected_layer_by_request=self.expected_layer_by_request,
            expected_model_revision=self.model_revision,
            score_join_identities=self.scored_joins,
        )
        if recomputed != self.full_load_audit:
            raise RICValidationError("ReplayWorld full-load audit is stale or inconsistent")
        for join, siblings in self.joins.items():
            combine = {task.stage_service.join_combine_us for task in siblings}
            sources = {task.stage_service.join_combine_source for task in siblings}
            if len(combine) != 1 or len(sources) != 1:
                raise RICValidationError("one join has inconsistent once-only combine service")

    @property
    def task_fingerprint(self) -> str:
        return self.full_load_audit.full_task_fingerprint

    @property
    def score_mask_fingerprint(self) -> str:
        return self.full_load_audit.score_mask_fingerprint

    @property
    def service_fingerprint(self) -> str:
        contribution_rows = [
            (
                task.task_id,
                task.stage_resources,
                task.stage_service_us,
                task.stage_service.source_tags[:3],
            )
            for task in sorted(self.tasks, key=lambda item: item.task_id)
        ]
        combine_rows = [
            (
                join.canonical_tuple(),
                siblings[0].combine_resource,
                siblings[0].stage_service.join_combine_us,
                siblings[0].stage_service.join_combine_source,
            )
            for join, siblings in sorted(self.joins.items())
        ]
        return _sha256((contribution_rows, combine_rows))

    @property
    def resource_demand_fingerprint(self) -> str:
        demand: dict[str, float] = {}
        for task in self.tasks:
            for resource, service in zip(task.stage_resources, task.stage_service_us):
                demand[resource] = demand.get(resource, 0.0) + service
        for siblings in self.joins.values():
            task = siblings[0]
            demand[task.combine_resource] = (
                demand.get(task.combine_resource, 0.0)
                + task.stage_service.join_combine_us
            )
        return _sha256(tuple(sorted(demand.items())))

    @property
    def task_by_id(self) -> dict[str, ReplayTask]:
        return {task.task_id: task for task in self.tasks}

    @property
    def joins(self) -> dict[JoinIdentity, tuple[ReplayTask, ...]]:
        result: dict[JoinIdentity, list[ReplayTask]] = {}
        for task in self.tasks:
            result.setdefault(task.join_identity, []).append(task)
        return {
            join: tuple(sorted(rows, key=lambda row: row.identity.topk_slot))
            for join, rows in result.items()
        }

    @property
    def source_tags(self) -> tuple[str, ...]:
        result = {task.contribution.source_tag for task in self.tasks}
        for task in self.tasks:
            result.update(task.stage_service.source_tags)
        return tuple(sorted(result))

    def with_score_mask(self, score_joins: Iterable[JoinIdentity]) -> "ReplayWorld":
        """Change metrics only; never rebuild/filter tasks or service demand."""

        scored = frozenset(score_joins)
        audit = validate_full_background(
            [task.contribution for task in self.tasks],
            top_k=self.top_k,
            num_experts=self.num_experts,
            ep_size=self.ep_size,
            expected_request_ids=self.expected_request_ids,
            expected_token_blocks_per_request=128,
            expert_to_sender=self.expert_to_sender,
            request_to_receiver=self.request_to_receiver,
            expected_layer_by_request=self.expected_layer_by_request,
            expected_model_revision=self.model_revision,
            score_join_identities=scored,
        )
        if (
            audit.full_task_fingerprint != self.task_fingerprint
        ):
            raise RICValidationError("score mask changed scheduled workload demand")
        updated = replace(self, scored_joins=scored, full_load_audit=audit)
        if updated.resource_demand_fingerprint != self.resource_demand_fingerprint:
            raise RICValidationError("score mask changed the real resource graph")
        return updated


def contiguous_expert_placement(num_experts: int, ep_size: int) -> dict[int, int]:
    if num_experts <= 0 or ep_size <= 0 or num_experts % ep_size:
        raise RICValidationError("fixture requires evenly divisible contiguous placement")
    per_rank = num_experts // ep_size
    return {expert: expert // per_rank for expert in range(num_experts)}


def _route_blind_equal_load_origins(request_ids: Sequence[str], ep_size: int) -> dict[str, int]:
    """Token-count LPT; all requests have the frozen 128-token weight."""

    loads = [0] * ep_size
    result: dict[str, int] = {}
    for request_id in sorted(request_ids, key=lambda value: hashlib.sha256(value.encode()).digest()):
        receiver = min(range(ep_size), key=lambda rank: (loads[rank], rank))
        result[request_id] = receiver
        loads[receiver] += 128
    return result


def build_complete_fixture_world(
    *,
    model_key: str = "olmoe",
    model_revision: str = "fixture/olmoe@ric-v1",
    top_k: int = 8,
    num_experts: int = 64,
    ep_size: int = 8,
    ranks_per_node: int = 4,
    trace_index: int = 0,
    seed: int = 202607223001,
    cell: str = "poisson_rho60",
    payload_bytes: int = 4096,
    descriptor_bytes: int = 16,
    alignment_bytes: int = 0,
    mean_microcoflow_gap_us: float = 0.25,
    closure_budget_us: float = 50.0,
) -> ReplayWorld:
    """Build a deterministic 4 x 128 full-load implementation fixture.

    Every top-k sibling shares its token-block arrival.  Ready offsets and
    contribution-stage and once-only combine services are deterministic draws and are
    common to every policy arm.
    """

    if top_k > 16 or top_k <= 0 or num_experts < top_k:
        raise RICValidationError("fixture dimensions violate frozen top-k limits")
    if mean_microcoflow_gap_us <= 0 or closure_budget_us <= 0:
        raise RICValidationError("invalid fixture timing")
    placement = contiguous_expert_placement(num_experts, ep_size)
    request_ids = tuple(
        f"{model_key}-trace{trace_index:02d}-request{index:02d}" for index in range(4)
    )
    origins = _route_blind_equal_load_origins(request_ids, ep_size)
    layer_candidates = (3, 7, 11, 15)
    layers = {
        request_id: layer_candidates[
            int(hashlib.sha256(request_id.encode()).hexdigest(), 16)
            % len(layer_candidates)
        ]
        for request_id in request_ids
    }
    rng = random.Random(seed)
    now = 0.0
    tasks: list[ReplayTask] = []
    for request_index, request_id in enumerate(request_ids):
        forward_id = f"forward-{trace_index:02d}-{request_index:02d}"
        batch_id = f"batch-{trace_index:02d}-{request_index:02d}"
        receiver = origins[request_id]
        layer_id = layers[request_id]
        for token_position in range(128):
            now += rng.expovariate(1.0 / mean_microcoflow_gap_us)
            arrival = now
            token_id = f"{request_id}:token:{token_position:03d}"
            token_block_id = f"{request_id}:block:{token_position:03d}"
            # The arithmetic permutation is injective over slots because
            # num_experts is a multiple of eight in both frozen models.
            base_expert = (
                17 * request_index + 13 * token_position + 5 * layer_id
            ) % num_experts
            selected: list[int] = []
            step = 7 if math.gcd(7, num_experts) == 1 else 5
            candidate = base_expert
            while len(selected) < top_k:
                if candidate not in selected:
                    selected.append(candidate)
                candidate = (candidate + step) % num_experts
            for slot, expert in enumerate(selected):
                sender = placement[expert]
                ready_offset = 0.10 + 0.025 * (expert % 7) + 0.01 * slot
                sender_us = 0.18 + payload_bytes / 32_000.0 + 0.005 * (slot % 3)
                cut_us = 0.10 + (payload_bytes + descriptor_bytes) / 25_000.0
                ingress_us = 0.14 + payload_bytes / 40_000.0 + 0.004 * (expert % 5)
                combine_us = 0.12 + 0.01 * top_k
                stage = StageService(sender_us, cut_us, ingress_us, combine_us)
                identity = ContributionIdentity(
                    request_id=request_id,
                    forward_id=forward_id,
                    batch_id=batch_id,
                    phase="prefill",
                    decode_step=0,
                    layer_id=layer_id,
                    token_id=token_id,
                    token_block_id=token_block_id,
                    topk_slot=slot,
                    expert_id=expert,
                    sender_rank=sender,
                    receiver_rank=receiver,
                    epoch=1,
                )
                record = ContributionRecord(
                    identity=identity,
                    model_revision=model_revision,
                    valid=True,
                    arrival_us=arrival,
                    ready_us=arrival + ready_offset,
                    service_us=stage.total_us,
                    deadline_us=arrival + closure_budget_us,
                    payload_bytes=payload_bytes,
                    descriptor_bytes=descriptor_bytes,
                    alignment_bytes=alignment_bytes,
                    source_tag="derived_from_measured_lut",
                )
                sender_node = sender // ranks_per_node
                receiver_node = receiver // ranks_per_node
                tasks.append(
                    ReplayTask(
                        contribution=record,
                        stage_service=stage,
                        sender_egress_resource=f"sender:{sender}:egress",
                        shared_cut_resource=f"cut:node{sender_node}->node{receiver_node}",
                        receiver_ingress_resource=f"receiver:{receiver}:ingress",
                    )
                )
    records = [task.contribution for task in tasks]
    all_joins = frozenset(task.join_identity for task in tasks)
    audit = validate_full_background(
        records,
        top_k=top_k,
        num_experts=num_experts,
        ep_size=ep_size,
        expected_request_ids=request_ids,
        expected_token_blocks_per_request=128,
        expert_to_sender=placement,
        request_to_receiver=origins,
        expected_layer_by_request=layers,
        expected_model_revision=model_revision,
        score_join_identities=all_joins,
    )
    return ReplayWorld(
        trace_id=f"{model_key}/{cell}/trace-{trace_index:02d}/seed-{seed}",
        workload_seed=seed,
        model_key=model_key,
        model_revision=model_revision,
        cell=cell,
        top_k=top_k,
        num_experts=num_experts,
        ep_size=ep_size,
        ranks_per_node=ranks_per_node,
        tasks=tuple(sorted(tasks, key=lambda task: task.task_id)),
        expected_request_ids=request_ids,
        expert_to_sender=placement,
        request_to_receiver=origins,
        expected_layer_by_request=layers,
        scored_joins=all_joins,
        full_load_audit=audit,
    )


__all__ = [
    "StageService",
    "ReplayTask",
    "ReplayWorld",
    "contiguous_expert_placement",
    "build_complete_fixture_world",
]
