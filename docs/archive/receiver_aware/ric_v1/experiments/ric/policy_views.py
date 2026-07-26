"""Structurally separated sender (S), topology (B), and receiver (R) views."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import Union

from .schema import ContributionIdentity, JoinIdentity, RICValidationError


@dataclass(frozen=True)
class ReadyTaskView:
    task_id: str
    identity: ContributionIdentity
    ready_us: float
    service_us: float
    wire_bytes: int
    deadline_us: float
    age_us: float
    fairness_debt: float
    stage_resources: tuple[str, str, str]
    stage_service_us: tuple[float, float, float]
    combine_resource: str
    combine_service_us: float

    def __post_init__(self) -> None:
        if not self.task_id or type(self.identity) is not ContributionIdentity:
            raise RICValidationError("ready task lacks a stable full identity")
        for name in ("ready_us", "service_us", "deadline_us", "age_us", "fairness_debt"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RICValidationError(f"{name} must be finite")
        if self.ready_us < 0 or self.service_us <= 0 or self.age_us < 0 or self.fairness_debt < 0:
            raise RICValidationError("invalid sender-local task timing/debt")
        if type(self.wire_bytes) is not int or self.wire_bytes < 0:
            raise RICValidationError("wire_bytes must be a non-negative integer")
        if (
            type(self.stage_resources) is not tuple
            or len(self.stage_resources) != 3
            or any(not isinstance(value, str) or not value for value in self.stage_resources)
        ):
            raise RICValidationError("ready task requires an immutable three-resource path")
        if (
            type(self.stage_service_us) is not tuple
            or len(self.stage_service_us) != 3
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
                for value in self.stage_service_us
            )
        ):
            raise RICValidationError("ready task requires positive three-stage service")
        if not isinstance(self.combine_resource, str) or not self.combine_resource:
            raise RICValidationError("ready task lacks its receiver combine resource")
        if (
            not isinstance(self.combine_service_us, (int, float))
            or not math.isfinite(float(self.combine_service_us))
            or self.combine_service_us <= 0
        ):
            raise RICValidationError("ready task combine service must be positive")


@dataclass(frozen=True)
class AggregateResourceView:
    receiver_rank: int
    receiver_qdepth: int

    def __post_init__(self) -> None:
        if type(self.receiver_rank) is not int or not 0 <= self.receiver_rank <= 0xFF:
            raise RICValidationError("aggregate receiver rank is invalid")
        if type(self.receiver_qdepth) is not int or self.receiver_qdepth < 0:
            raise RICValidationError("aggregate qdepth must be non-negative")


@dataclass(frozen=True)
class ResourceBacklogView:
    resource_id: str
    stage: str
    queued_count: int
    remaining_service_us: float

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id:
            raise RICValidationError("resource backlog lacks a resource id")
        if self.stage not in {
            "sender_egress",
            "shared_cut",
            "receiver_ingress",
            "receiver_combine",
        }:
            raise RICValidationError("resource backlog has an unknown stage")
        if type(self.queued_count) is not int or self.queued_count < 0:
            raise RICValidationError("resource queued count must be non-negative")
        if (
            not isinstance(self.remaining_service_us, (int, float))
            or not math.isfinite(float(self.remaining_service_us))
            or self.remaining_service_us < 0
        ):
            raise RICValidationError("resource remaining service must be finite/non-negative")


@dataclass(frozen=True)
class SView:
    sender_rank: int
    now_us: float
    ready_tasks: tuple[ReadyTaskView, ...]


@dataclass(frozen=True)
class BView:
    sender: SView
    aggregate_resources: tuple[AggregateResourceView, ...]
    resource_backlogs: tuple[ResourceBacklogView, ...]


@dataclass(frozen=True)
class ReceiverJoinView:
    join_identity: JoinIdentity
    missing_slot_mask: int
    slack_bucket: int
    epoch: int

    @property
    def missing_count(self) -> int:
        return bin(self.missing_slot_mask).count("1")


@dataclass(frozen=True)
class RView:
    base: BView
    receiver_join_state: tuple[ReceiverJoinView, ...]


def validate_s_view(view: SView) -> None:
    if type(view) is not SView:
        raise RICValidationError("S policy requires the exact SView type")
    if type(view.sender_rank) is not int or not 0 <= view.sender_rank <= 0xFF:
        raise RICValidationError("SView sender rank is invalid")
    if not isinstance(view.now_us, (int, float)) or not math.isfinite(float(view.now_us)) or view.now_us < 0:
        raise RICValidationError("SView time must be finite and non-negative")
    if type(view.ready_tasks) is not tuple:
        raise RICValidationError("SView ready_tasks must be an immutable tuple")
    seen: set[str] = set()
    for task in view.ready_tasks:
        if type(task) is not ReadyTaskView:
            raise RICValidationError("SView contains a non-ReadyTaskView object")
        if task.identity.sender_rank != view.sender_rank:
            raise RICValidationError("SView leaked another sender's local ready task")
        if task.task_id in seen:
            raise RICValidationError("SView contains a duplicate ready task")
        seen.add(task.task_id)


def validate_b_view(view: BView) -> None:
    if type(view) is not BView:
        raise RICValidationError("B policy requires the exact BView type")
    validate_s_view(view.sender)
    if type(view.aggregate_resources) is not tuple:
        raise RICValidationError("BView aggregate resources must be an immutable tuple")
    receivers: set[int] = set()
    for resource in view.aggregate_resources:
        if type(resource) is not AggregateResourceView:
            raise RICValidationError("BView contains a keyed/private receiver object")
        if resource.receiver_rank in receivers:
            raise RICValidationError("BView contains duplicate receiver aggregate state")
        receivers.add(resource.receiver_rank)
    if type(view.resource_backlogs) is not tuple:
        raise RICValidationError("BView resource backlogs must be an immutable tuple")
    resource_ids: set[str] = set()
    for resource in view.resource_backlogs:
        if type(resource) is not ResourceBacklogView:
            raise RICValidationError("BView contains a non-aggregate resource object")
        if resource.resource_id in resource_ids:
            raise RICValidationError("BView contains duplicate resource backlog state")
        resource_ids.add(resource.resource_id)


def validate_r_view(view: RView) -> None:
    if type(view) is not RView:
        raise RICValidationError("R policy requires the exact RView type")
    validate_b_view(view.base)
    if type(view.receiver_join_state) is not tuple:
        raise RICValidationError("RView receiver state must be an immutable tuple")
    seen: set[JoinIdentity] = set()
    for state in view.receiver_join_state:
        if type(state) is not ReceiverJoinView:
            raise RICValidationError("RView contains an invalid receiver join state")
        if type(state.join_identity) is not JoinIdentity:
            raise RICValidationError("receiver state lacks a full join identity")
        if type(state.missing_slot_mask) is not int or not 1 <= state.missing_slot_mask <= 0xFFFF:
            raise RICValidationError("receiver missing-slot mask must be a nonzero u16")
        if type(state.slack_bucket) is not int or not 0 <= state.slack_bucket <= 3:
            raise RICValidationError("receiver slack bucket must be in [0, 3]")
        if state.epoch != state.join_identity.epoch:
            raise RICValidationError("receiver state epoch disagrees with identity epoch")
        if state.join_identity in seen:
            raise RICValidationError("RView contains duplicate keyed receiver state")
        seen.add(state.join_identity)


PolicyView = Union[SView, BView, RView]


def observation_fingerprint(view: PolicyView) -> str:
    """Canonical observation hash used by matched-world equivalence tests."""

    if type(view) is SView:
        validate_s_view(view)
    elif type(view) is BView:
        validate_b_view(view)
    elif type(view) is RView:
        validate_r_view(view)
    else:
        raise RICValidationError("unknown policy view type")
    payload = {
        "type": type(view).__name__,
        "fields": [field.name for field in fields(view)],
        "value": asdict(view),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ReadyTaskView",
    "AggregateResourceView",
    "ResourceBacklogView",
    "SView",
    "BView",
    "ReceiverJoinView",
    "RView",
    "PolicyView",
    "validate_s_view",
    "validate_b_view",
    "validate_r_view",
    "observation_fingerprint",
]
