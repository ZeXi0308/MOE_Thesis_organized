"""Global four-stage discrete-event replay for RIC-v1.

Only sender-local ready-result queues are policy controlled.  Shared cuts and
receiver ingresses use deterministic FCFS service.  The charged and sham arms
both traverse the single canonical codec/apply implementation in ``wire.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

try:
    from .policy_views import (
        AggregateResourceView,
        BView,
        RView,
        ReadyTaskView,
        ReceiverJoinView,
        ResourceBacklogView,
        SView,
        validate_b_view,
        validate_r_view,
        validate_s_view,
    )
    from .scenario import ReplayTask, ReplayWorld
    from .schema import JoinIdentity, RICValidationError
    from .wire import (
        ContractCacheEntry,
        ContractMessage,
        ContractRecord,
        ContractTax,
        HEADER_BYTES,
        RECORD_BYTES,
        IdentityTable,
        SenderContractCache,
        apply_wire_contract,
        encode_contract,
        join_identity_hash_parts,
    )
except ImportError:  # Direct test execution from this directory.
    from policy_views import (  # type: ignore
        AggregateResourceView,
        BView,
        RView,
        ReadyTaskView,
        ReceiverJoinView,
        ResourceBacklogView,
        SView,
        validate_b_view,
        validate_r_view,
        validate_s_view,
    )
    from scenario import ReplayTask, ReplayWorld  # type: ignore
    from schema import JoinIdentity, RICValidationError  # type: ignore
    from wire import (  # type: ignore
        ContractCacheEntry,
        ContractMessage,
        ContractRecord,
        ContractTax,
        HEADER_BYTES,
        RECORD_BYTES,
        IdentityTable,
        SenderContractCache,
        apply_wire_contract,
        encode_contract,
        join_identity_hash_parts,
    )


JOINBLIND_ARMS = frozenset(
    {
        "sender_fcfs",
        "sender_edf",
        "sender_srpt",
        "sender_age_service_drr",
        "sender_remaining_work",
        "sync_token_order",
        "receiver_qdepth",
        "topology_projected_finish",
        "receiver_contention_joinblind",
        "calib_best_joinblind",
    }
)

RIC_ARMS = frozenset(
    {
        "ric_full_zero_delay",
        "ric_compressed_zero_delay",
        "ric_compressed_delayed",
        "ric_wire_charged",
        "ric_sham_feedback",
    }
)


TAX_RECORD_COUNT_GRID = tuple(range(1, 0x100))
TAX_NON_GRID_RULE = "exact_1_to_255_no_interpolation_or_extrapolation"


@dataclass(frozen=True)
class ContractTaxSurface:
    """Frozen measured component tax indexed by actual message record count.

    Every canonical u8 record count 1..255 must have an exact measured point.
    A missing count blocks replay; interpolation and extrapolation are forbidden.
    """

    points: tuple[tuple[int, ContractTax], ...]
    non_grid_rule: str = TAX_NON_GRID_RULE
    source_id: str = "unspecified"

    def __post_init__(self) -> None:
        if self.non_grid_rule != TAX_NON_GRID_RULE:
            raise RICValidationError("unsupported contract tax non-grid rule")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise RICValidationError("contract tax surface requires a source id")
        if type(self.points) is not tuple:
            raise RICValidationError("contract tax surface points must be immutable")
        counts = tuple(count for count, _tax in self.points)
        if any(type(count) is not int for count in counts):
            raise RICValidationError("contract tax surface count must be an integer")
        if counts != TAX_RECORD_COUNT_GRID:
            raise RICValidationError(
                "contract tax surface requires ordered exact points for counts 1..255"
            )
        if any(type(tax) is not ContractTax for _count, tax in self.points):
            raise RICValidationError("contract tax surface contains a non-canonical tax")
        if any(
            float(value) <= 0.0
            for _count, tax in self.points
            for value in tax.__dict__.values()
        ):
            raise RICValidationError(
                "BLOCKED_CONTROL_TAX_SURFACE: additive component must be positive"
            )

    def tax_for(self, record_count: int) -> ContractTax:
        if type(record_count) is not int or not 1 <= record_count <= 0xFF:
            raise RICValidationError("contract record_count must be in [1, 255]")
        by_count = dict(self.points)
        try:
            return by_count[record_count]
        except KeyError as exc:  # Defensive if an object bypassed dataclass validation.
            raise RICValidationError(
                "BLOCKED_CONTROL_TAX_SURFACE: missing exact record_count"
            ) from exc

    @property
    def fingerprint(self) -> str:
        payload = {
            "non_grid_rule": self.non_grid_rule,
            "source_id": self.source_id,
            "points": [
                [count, {name: float(value) for name, value in tax.__dict__.items()}]
                for count, tax in self.points
            ],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _fixture_contract_tax_surface() -> ContractTaxSurface:
    def point(record_count: int) -> ContractTax:
        return ContractTax(
            state_build_us=0.02 * record_count,
            hash_us=0.01 * record_count,
            encode_us=0.02 * record_count,
            transfer_us=0.03 * record_count,
            decode_us=0.02 * record_count,
            lookup_us=0.02 * record_count,
            apply_us=0.01 * record_count,
            policy_lookup_us=0.02 * record_count,
        )

    return ContractTaxSurface(
        points=tuple((count, point(count)) for count in TAX_RECORD_COUNT_GRID),
        source_id="implementation_fixture_not_scientific",
    )


@dataclass(frozen=True)
class ReplayConfig:
    """Frozen policy/control parameters for one replay invocation."""

    compressed_delay_us: float = 5.0
    wire_delay_us: float = 5.0
    starvation_us: float = 200.0
    drr_quantum_us: float = 1.0
    drr_service_fingerprint: str = ""
    calib_best_joinblind: str = "topology_projected_finish"
    contract_tax_surface: ContractTaxSurface = field(
        default_factory=_fixture_contract_tax_surface
    )
    # Faults are keyed by (sender_rank, receiver_rank, emitted_sequence).
    # Supported values intentionally exercise the canonical wire apply path.
    wire_faults: Mapping[tuple[int, int, int], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "compressed_delay_us",
            "wire_delay_us",
            "starvation_us",
            "drr_quantum_us",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise RICValidationError(f"{name} must be finite and non-negative")
        if self.starvation_us <= 0:
            raise RICValidationError("starvation_us must be positive")
        if self.drr_quantum_us <= 0:
            raise RICValidationError("drr_quantum_us must be positive")
        if not isinstance(self.drr_service_fingerprint, str):
            raise RICValidationError("drr_service_fingerprint must be a string")
        if self.calib_best_joinblind not in JOINBLIND_ARMS - {
            "calib_best_joinblind"
        }:
            raise RICValidationError("invalid calibration-selected fallback arm")
        if type(self.contract_tax_surface) is not ContractTaxSurface:
            raise RICValidationError("replay requires a canonical contract tax surface")
        supported = {"malformed", "duplicate", "missing_sequence", "wrong_sender"}
        if set(self.wire_faults.values()) - supported:
            raise RICValidationError("unsupported wire fault fixture")


@dataclass(frozen=True)
class ActionRecord:
    arm: str
    task_id: str
    stage: str
    resource_id: str
    enqueue_us: float
    decision_us: float
    service_start_us: float
    service_end_us: float
    sender_rank: int
    receiver_rank: int
    visible_missing_count: int | None
    fallback: bool
    stale: bool
    starvation_override: bool


@dataclass(frozen=True)
class ControlPlanEvent:
    emission_us: float
    delivery_us: float
    sender_rank: int
    receiver_rank: int
    epoch: int
    sequence: int
    record_count: int
    tax: ContractTax
    payload: bytes
    produced_bytes: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.emission_us)
            or not math.isfinite(self.delivery_us)
            or self.emission_us < 0
            or self.delivery_us < self.emission_us
        ):
            raise RICValidationError("invalid control plan event time")
        if type(self.record_count) is not int or not 1 <= self.record_count <= 0xFF:
            raise RICValidationError("invalid control plan event record count")
        if any(
            type(rank) is not int or not 0 <= rank <= 0xFF
            for rank in (self.sender_rank, self.receiver_rank)
        ):
            raise RICValidationError("invalid control plan event rank")
        if type(self.sequence) is not int or not 1 <= self.sequence <= 0xFFFFFFFF:
            raise RICValidationError("invalid control plan event sequence")
        if type(self.epoch) is not int or not 1 <= self.epoch <= 0xFFFFFFFF:
            raise RICValidationError("invalid control plan event epoch")
        if type(self.tax) is not ContractTax or type(self.payload) is not bytes:
            raise RICValidationError("invalid control plan event tax/payload")
        if type(self.produced_bytes) is not int or self.produced_bytes < len(self.payload):
            raise RICValidationError("control plan produced bytes underflow")


@dataclass(frozen=True)
class ControlPlan:
    """Immutable charged message plan used only by the sham counterfactual."""

    trace_id: str
    task_fingerprint: str
    service_fingerprint: str
    score_mask_fingerprint: str
    resource_demand_fingerprint: str
    contract_tax_surface: ContractTaxSurface
    contract_tax_surface_fingerprint: str
    wire_delay_us: float
    starvation_us: float
    drr_quantum_us: float
    drr_service_fingerprint: str
    calib_best_joinblind: str
    events: tuple[ControlPlanEvent, ...]

    def __post_init__(self) -> None:
        if type(self.contract_tax_surface) is not ContractTaxSurface:
            raise RICValidationError("ControlPlan lacks canonical tax surface")
        if (
            self.contract_tax_surface_fingerprint
            != self.contract_tax_surface.fingerprint
        ):
            raise RICValidationError("ControlPlan tax surface fingerprint mismatch")
        if type(self.events) is not tuple:
            raise RICValidationError("ControlPlan events must be immutable")
        for name in (
            "trace_id",
            "task_fingerprint",
            "service_fingerprint",
            "score_mask_fingerprint",
            "resource_demand_fingerprint",
            "calib_best_joinblind",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise RICValidationError(f"ControlPlan lacks {name}")
        if self.starvation_us <= 0 or self.drr_quantum_us <= 0:
            raise RICValidationError("ControlPlan fairness parameters are invalid")
        for event in self.events:
            if type(event) is not ControlPlanEvent:
                raise RICValidationError("ControlPlan contains an invalid event")
            if event.tax != self.contract_tax_surface.tax_for(event.record_count):
                raise RICValidationError("ControlPlan event tax/surface mismatch")


@dataclass(frozen=True)
class ReplayResult:
    trace_id: str
    workload_seed: int
    model_key: str
    cell: str
    arm: str
    task_count: int
    completed_task_count: int
    completed_stage_count: int
    expected_stage_count: int
    completed_join_count: int
    expected_join_count: int
    task_fingerprint: str
    service_fingerprint: str
    score_mask_fingerprint: str
    resource_demand_fingerprint: str
    payload_bytes: int
    descriptor_bytes: int
    alignment_bytes: int
    contract_bytes: int
    contract_received_bytes: int
    contract_header_bytes: int
    contract_record_bytes: int
    contract_alignment_bytes: int
    contract_messages: int
    contract_record_count_histogram: Mapping[int, int]
    contract_tax_surface_source_id: str
    contract_tax_surface_fingerprint: str
    contract_tax_non_grid_rule: str
    control_component_us: Mapping[str, float]
    stale_decisions: int
    fallback_decisions: int
    sender_decisions: int
    starvation_count: int
    sender_ready_wait_us: tuple[float, ...]
    makespan_us: float
    queue_busy_us: Mapping[str, float]
    resource_service_demand_us: Mapping[str, float]
    source_by_field: Mapping[str, str]
    source_tags: tuple[str, ...]
    full_drain: bool
    completion_by_task_us: Mapping[str, float]
    join_completion_us: Mapping[str, float]
    all_join_latencies_us: Mapping[str, float]
    scored_join_latencies_us: Mapping[str, float]
    action_trace: tuple[ActionRecord, ...]
    fault_counts: Mapping[str, int]
    control_plan: ControlPlan | None


@dataclass
class _Resource:
    resource_id: str
    stage_index: int
    queue: list[tuple[float, str]] = field(default_factory=list)
    busy_task_id: str | None = None
    busy_end_us: float | None = None
    busy_us: float = 0.0


@dataclass(order=True, frozen=True)
class _Event:
    time_us: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False)


@dataclass(frozen=True)
class _StageDone:
    task_id: str
    stage_index: int
    resource_id: str


@dataclass(frozen=True)
class _WireDelivery:
    emission_us: float
    sender_rank: int
    receiver_rank: int
    epoch: int
    sequence: int
    record_count: int
    tax: ContractTax
    payload: bytes
    produced_bytes: int
    charged: bool


@dataclass(frozen=True)
class _ControlTransfer:
    """Charged message after receiver encode, before sender apply FCFS."""

    emission_us: float
    sender_rank: int
    receiver_rank: int
    epoch: int
    sequence: int
    record_count: int
    tax: ContractTax
    payload: bytes
    produced_bytes: int


@dataclass(frozen=True)
class _PendingContract:
    sender_rank: int
    receiver_rank: int
    epoch: int
    join_identity: JoinIdentity
    missing_mask: int
    slack_bucket: int


def _join_label(identity: JoinIdentity) -> str:
    return identity.canonical_bytes().hex()


def _missing_bucket(mask: int) -> int:
    """Exact missing-slot population; retained name is wire-plan ABI only."""

    return bin(mask).count("1")


def _slack_bucket(task: ReplayTask, now_us: float) -> int:
    budget = task.contribution.deadline_us - task.contribution.arrival_us
    slack = task.contribution.deadline_us - now_us
    if slack <= 0:
        return 0
    ratio = slack / budget
    # Timer events are constructed from the same floating-point deadline and
    # budget.  Admit only roundoff at an exact frozen boundary.
    if ratio <= 0.25 + 1e-12:
        return 1
    if ratio <= 0.50 + 1e-12:
        return 2
    return 3


class _Simulator:
    def __init__(
        self,
        world: ReplayWorld,
        arm: str,
        config: ReplayConfig,
        *,
        reference_control_plan: ControlPlan | None = None,
    ) -> None:
        if arm not in JOINBLIND_ARMS | RIC_ARMS:
            raise RICValidationError(f"unsupported RIC replay arm {arm!r}")
        self.world = world
        self.arm = arm
        self.config = config
        self.reference_control_plan = reference_control_plan
        if (
            config.drr_service_fingerprint
            and config.drr_service_fingerprint != world.service_fingerprint
        ):
            raise RICValidationError("DRR quantum is not bound to this service fingerprint")
        if arm == "ric_sham_feedback" and reference_control_plan is None:
            raise RICValidationError(
                "sham feedback requires an immutable charged ControlPlan"
            )
        if arm != "ric_sham_feedback" and reference_control_plan is not None:
            raise RICValidationError("reference ControlPlan is valid only for sham feedback")
        if reference_control_plan is not None:
            if (
                reference_control_plan.trace_id != world.trace_id
                or reference_control_plan.task_fingerprint != world.task_fingerprint
                or reference_control_plan.service_fingerprint != world.service_fingerprint
                or reference_control_plan.score_mask_fingerprint
                != world.score_mask_fingerprint
                or reference_control_plan.resource_demand_fingerprint
                != world.resource_demand_fingerprint
            ):
                raise RICValidationError("sham ControlPlan/workload fingerprint mismatch")
            if (
                reference_control_plan.contract_tax_surface
                != config.contract_tax_surface
                or reference_control_plan.contract_tax_surface_fingerprint
                != config.contract_tax_surface.fingerprint
            ):
                raise RICValidationError("sham ControlPlan tax differs from charged tax")
            if not math.isclose(
                reference_control_plan.wire_delay_us,
                config.wire_delay_us,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RICValidationError("sham ControlPlan delay differs from charged delay")
            if (
                not math.isclose(
                    reference_control_plan.starvation_us,
                    config.starvation_us,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    reference_control_plan.drr_quantum_us,
                    config.drr_quantum_us,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or reference_control_plan.drr_service_fingerprint
                != config.drr_service_fingerprint
                or reference_control_plan.calib_best_joinblind
                != config.calib_best_joinblind
            ):
                raise RICValidationError("sham ControlPlan policy/fairness semantics differ")
        self.tasks = world.task_by_id
        self.joins = world.joins
        self.join_by_label = {_join_label(join): join for join in self.joins}
        self.receiver_ranks = tuple(
            sorted({task.identity.receiver_rank for task in world.tasks})
        )
        self.events: list[_Event] = []
        self.next_event_sequence = 0
        self.resources: dict[str, _Resource] = {}
        self.receiver_outstanding_count: dict[int, int] = {
            receiver: 0 for receiver in range(world.ep_size)
        }
        self.stage_seen: set[tuple[str, int]] = set()
        self.combine_seen: set[JoinIdentity] = set()
        self.join_completion: dict[JoinIdentity, float] = {}
        self.completion_by_task: dict[str, float] = {}
        self.action_trace: list[ActionRecord] = []
        self.sender_waits: list[float] = []
        self.sender_decisions = 0
        self.stale_decisions = 0
        self.fallback_decisions = 0
        self.starvation_count = 0
        self.contract_bytes = 0
        self.contract_received_bytes = 0
        self.contract_header_bytes = 0
        self.contract_record_bytes = 0
        self.contract_alignment_bytes = 0
        self.contract_messages = 0
        self.contract_record_count_histogram: dict[int, int] = {}
        self.control_component_us: dict[str, float] = {
            name: 0.0
            for name in (
                "state_build_us",
                "hash_us",
                "encode_us",
                "configured_delay_us",
                "transfer_us",
                "decode_us",
                "lookup_us",
                "apply_us",
                "policy_lookup_us",
            )
        }
        self.control_resource_busy_us: dict[str, float] = {}
        self.receiver_control_available_us: dict[int, float] = {}
        self.sender_control_available_us: dict[int, float] = {}
        self.fault_counts: dict[str, int] = {}
        self.emitted_control_events: list[ControlPlanEvent] = []
        self.pending_contracts: list[_PendingContract] = []
        self.actual_missing: dict[JoinIdentity, int] = {
            join: (1 << world.top_k) - 1 for join in self.joins
        }
        self.wire_caches = {
            sender: SenderContractCache(sender) for sender in range(world.ep_size)
        }
        self.identity_tables = {
            sender: IdentityTable.from_joins(
                (
                    task.join_identity
                    for task in world.tasks
                    if task.identity.sender_rank == sender
                ),
                top_k=world.top_k,
            )
            for sender in range(world.ep_size)
            if any(task.identity.sender_rank == sender for task in world.tasks)
        }
        self.message_sequences: dict[tuple[int, int, int], int] = {}
        self.fallback_receiver_epochs: set[tuple[int, int, int]] = set()
        self.cumulative_sender_service: dict[tuple[int, str], float] = {}
        self.drr_deficit_us: dict[tuple[int, str], float] = {}
        self.drr_cursor: dict[int, str | None] = {
            sender: None for sender in range(world.ep_size)
        }
        # True means the next call resumes the same already-credited DRR visit;
        # it must not add Q again before consuming residual deficit.
        self.drr_visit_continuation: dict[int, bool] = {
            sender: False for sender in range(world.ep_size)
        }
        self.last_bucket: dict[tuple[int, JoinIdentity], tuple[int, int]] = {}
        for join, siblings in self.joins.items():
            full_mask = (1 << world.top_k) - 1
            representative = siblings[0]
            initial = (
                bin(full_mask).count("1"),
                _slack_bucket(
                    representative, representative.contribution.arrival_us
                ),
            )
            for sender in {task.identity.sender_rank for task in siblings}:
                self.last_bucket[(sender, join)] = initial
            if arm in {
                "ric_compressed_zero_delay",
                "ric_compressed_delayed",
                "ric_wire_charged",
            }:
                arrival = representative.contribution.arrival_us
                budget = representative.contribution.deadline_us - arrival
                # _slack_bucket transitions 3->2, 2->1, and 1->0 at these
                # exact causal times even when no contribution arrives then.
                for fraction in (0.50, 0.75, 1.00):
                    self._push(
                        arrival + fraction * budget,
                        1,
                        "slack_transition",
                        join,
                    )
        for task in world.tasks:
            for index, resource_id in enumerate(task.stage_resources):
                existing = self.resources.get(resource_id)
                if existing is None:
                    self.resources[resource_id] = _Resource(resource_id, index)
                elif existing.stage_index != index:
                    raise RICValidationError("one resource id is reused for two stages")
            self._push(task.contribution.ready_us, 0, "task_ready", task.task_id)
        for siblings in self.joins.values():
            combine_resource = siblings[0].combine_resource
            existing = self.resources.get(combine_resource)
            if existing is None:
                self.resources[combine_resource] = _Resource(combine_resource, 3)
            elif existing.stage_index != 3:
                raise RICValidationError("combine resource id is reused by another stage")
        if reference_control_plan is not None:
            for event in reference_control_plan.events:
                expected_tax = config.contract_tax_surface.tax_for(
                    event.record_count
                )
                if event.tax != expected_tax:
                    raise RICValidationError(
                        "sham ControlPlan event tax/surface mismatch"
                    )
                self._push(
                    event.delivery_us,
                    2,
                    "wire_delivery",
                    _WireDelivery(
                        event.emission_us,
                        event.sender_rank,
                        event.receiver_rank,
                        event.epoch,
                        event.sequence,
                        event.record_count,
                        event.tax,
                        event.payload,
                        event.produced_bytes,
                        True,
                    ),
                )
                self.emitted_control_events.append(event)

    def _push(self, time_us: float, priority: int, kind: str, payload: Any) -> None:
        if not math.isfinite(time_us) or time_us < 0:
            raise RICValidationError("event time must be finite and non-negative")
        self.next_event_sequence += 1
        heapq.heappush(
            self.events,
            _Event(time_us, priority, self.next_event_sequence, kind, payload),
        )

    def _enqueue_stage(
        self,
        task_id: str,
        stage_index: int,
        now_us: float,
        *,
        new_outstanding: bool = False,
    ) -> None:
        if not 0 <= stage_index <= 2:
            raise RICValidationError("contribution stage index must be in [0, 2]")
        task = self.tasks[task_id]
        resource_id = task.stage_resources[stage_index]
        self.resources[resource_id].queue.append((now_us, task_id))
        if new_outstanding:
            receiver = task.identity.receiver_rank
            self.receiver_outstanding_count[receiver] += 1

    def _enqueue_combine(self, join: JoinIdentity, now_us: float) -> None:
        label = _join_label(join)
        if join in self.combine_seen or any(
            label == queued_label
            for resource in self.resources.values()
            for _enqueue, queued_label in resource.queue
        ):
            raise RICValidationError("join combine was enqueued more than once")
        siblings = self.joins[join]
        resource_id = siblings[0].combine_resource
        self.resources[resource_id].queue.append((now_us, label))
        self.receiver_outstanding_count[join.receiver_rank] += 1

    def _label_receiver(self, label: str, stage_index: int) -> int:
        if stage_index == 3:
            return self.join_by_label[label].receiver_rank
        return self.tasks[label].identity.receiver_rank

    def _service_for_label(self, label: str, stage_index: int) -> float:
        if stage_index == 3:
            join = self.join_by_label[label]
            return self.joins[join][0].stage_service.join_combine_us
        return self.tasks[label].stage_service_us[stage_index]

    @staticmethod
    def _stage_name(stage_index: int) -> str:
        try:
            return (
                "sender_egress",
                "shared_cut",
                "receiver_ingress",
                "receiver_combine",
            )[stage_index]
        except IndexError as exc:
            raise RICValidationError("invalid resource stage index") from exc

    def _resource_backlog(
        self, resource: _Resource, now_us: float
    ) -> ResourceBacklogView:
        queued_service = math.fsum(
            self._service_for_label(label, resource.stage_index)
            for _enqueue, label in resource.queue
        )
        residual = 0.0
        busy_count = 0
        if resource.busy_task_id is not None:
            if resource.busy_end_us is None:
                raise RICValidationError("busy resource lacks an end time")
            residual = max(0.0, resource.busy_end_us - now_us)
            busy_count = 1
        elif resource.busy_end_us is not None:
            raise RICValidationError("idle resource retains a busy end time")
        return ResourceBacklogView(
            resource_id=resource.resource_id,
            stage=self._stage_name(resource.stage_index),
            queued_count=len(resource.queue) + busy_count,
            remaining_service_us=queued_service + residual,
        )

    def _aggregate_for_receiver_scan(
        self, receiver_rank: int
    ) -> AggregateResourceView:
        """Slow exact scan retained as an adversarial ledger oracle."""

        qdepth = 0
        for resource in self.resources.values():
            labels = [label for _enqueue, label in resource.queue]
            if resource.busy_task_id is not None:
                labels.append(resource.busy_task_id)
            qdepth += sum(
                self._label_receiver(label, resource.stage_index) == receiver_rank
                for label in labels
            )
        return AggregateResourceView(receiver_rank, qdepth)

    def _assert_aggregate_ledger_matches_scan(self) -> None:
        for receiver_rank in range(self.world.ep_size):
            observed = self._aggregate_for_receiver(receiver_rank)
            expected = self._aggregate_for_receiver_scan(receiver_rank)
            if observed != expected:
                raise RICValidationError("receiver qdepth ledger/scan mismatch")

    def _aggregate_for_receiver(self, receiver_rank: int) -> AggregateResourceView:
        if not 0 <= receiver_rank < self.world.ep_size:
            raise RICValidationError("receiver rank is outside the virtual EP world")
        return AggregateResourceView(
            receiver_rank, self.receiver_outstanding_count[receiver_rank]
        )

    def _views_from_snapshot(
        self,
        sender_rank: int,
        now_us: float,
        queued: Sequence[tuple[float, str]],
        *,
        sort_snapshot: bool,
    ) -> tuple[SView, BView]:
        snapshot: Sequence[tuple[float, str]] = tuple(queued)
        if sort_snapshot:
            snapshot = tuple(sorted(snapshot, key=lambda row: (row[0], row[1])))
        ready = tuple(
            ReadyTaskView(
                task_id=task_id,
                identity=self.tasks[task_id].identity,
                ready_us=self.tasks[task_id].contribution.ready_us,
                service_us=self.tasks[task_id].stage_service.sender_pack_us,
                wire_bytes=self.tasks[task_id].contribution.wire_bytes,
                deadline_us=self.tasks[task_id].contribution.deadline_us,
                age_us=max(0.0, now_us - enqueue_us),
                fairness_debt=self._service_lag_debt(sender_rank, task_id),
                stage_resources=self.tasks[task_id].stage_resources,
                stage_service_us=self.tasks[task_id].stage_service_us,
                combine_resource=self.tasks[task_id].combine_resource,
                combine_service_us=(
                    self.tasks[task_id].stage_service.join_combine_us
                ),
            )
            for enqueue_us, task_id in snapshot
        )
        s_view = SView(sender_rank=sender_rank, now_us=now_us, ready_tasks=ready)
        validate_s_view(s_view)
        b_view = BView(
            sender=s_view,
            aggregate_resources=tuple(
                self._aggregate_for_receiver(receiver)
                for receiver in self.receiver_ranks
            ),
            resource_backlogs=tuple(
                self._resource_backlog(self.resources[resource_id], now_us)
                for resource_id in sorted(self.resources)
            ),
        )
        validate_b_view(b_view)
        return s_view, b_view

    def _views(
        self, sender_rank: int, now_us: float, queued: Sequence[tuple[float, str]]
    ) -> tuple[SView, BView]:
        # Every policy chooses with a complete stable-key min (or an
        # order-invariant count), so queue order is not policy information.
        # Snapshot once to avoid a per-decision O(n log n) sort.
        return self._views_from_snapshot(
            sender_rank, now_us, queued, sort_snapshot=False
        )

    def _views_sorted_reference(
        self, sender_rank: int, now_us: float, queued: Sequence[tuple[float, str]]
    ) -> tuple[SView, BView]:
        """Pre-optimization reference retained for bitwise-equivalence tests."""

        return self._views_from_snapshot(
            sender_rank, now_us, queued, sort_snapshot=True
        )

    def _service_lag_debt(self, sender_rank: int, task_id: str) -> float:
        flow = self.tasks[task_id].identity.request_id
        self.cumulative_sender_service.setdefault((sender_rank, flow), 0.0)
        seen = [
            served
            for (sender, _flow), served in self.cumulative_sender_service.items()
            if sender == sender_rank
        ]
        maximum = max(seen, default=0.0)
        return max(0.0, maximum - self.cumulative_sender_service[(sender_rank, flow)])

    def _resolve_joinblind_arm(self, arm: str) -> str:
        return self.config.calib_best_joinblind if arm == "calib_best_joinblind" else arm

    def _joinblind_key(
        self, arm: str, task: ReadyTaskView, b_view: BView
    ) -> tuple[object, ...]:
        arm = self._resolve_joinblind_arm(arm)
        identity = task.identity
        aggregate = {
            row.receiver_rank: row for row in b_view.aggregate_resources
        }[identity.receiver_rank]
        if arm == "sender_fcfs":
            return (task.ready_us, task.task_id)
        if arm == "sender_edf":
            return (task.deadline_us, task.ready_us, task.task_id)
        if arm == "sender_srpt":
            return (task.service_us, task.ready_us, task.task_id)
        if arm == "sender_age_service_drr":
            raise RICValidationError("standard DRR must use its stateful scheduler")
        if arm == "sender_remaining_work":
            remaining = sum(
                other.identity.request_id == identity.request_id
                for other in b_view.sender.ready_tasks
            )
            return (remaining, task.deadline_us, task.task_id)
        if arm == "sync_token_order":
            return (
                identity.request_id,
                identity.layer_id,
                identity.token_block_id,
                identity.topk_slot,
                task.task_id,
            )
        if arm == "receiver_qdepth":
            return (aggregate.receiver_qdepth, task.deadline_us, task.task_id)
        if arm == "topology_projected_finish":
            backlogs = {
                row.resource_id: row.remaining_service_us
                for row in b_view.resource_backlogs
            }
            projected = (
                task.stage_service_us[0]
                + backlogs[task.stage_resources[1]]
                + task.stage_service_us[1]
                + backlogs[task.stage_resources[2]]
                + task.stage_service_us[2]
                + backlogs[task.combine_resource]
                + task.combine_service_us
            )
            return (projected, task.deadline_us, task.task_id)
        if arm == "receiver_contention_joinblind":
            # Drain the most contended receiver without reading keyed joins.
            return (-aggregate.receiver_qdepth, task.deadline_us, task.task_id)
        raise RICValidationError(f"unknown join-blind policy {arm!r}")

    def _cache_snapshot(self, sender_rank: int) -> Mapping[JoinIdentity, ContractCacheEntry]:
        return self.wire_caches[sender_rank].snapshot()

    def _choose_drr(
        self, sender_rank: int, queue: Sequence[tuple[float, str]]
    ) -> str:
        by_flow: dict[str, list[tuple[float, str]]] = {}
        for enqueue_us, task_id in queue:
            flow = self.tasks[task_id].identity.request_id
            by_flow.setdefault(flow, []).append((enqueue_us, task_id))
            self.drr_deficit_us.setdefault((sender_rank, flow), 0.0)
        active = tuple(sorted(by_flow))
        if not active:
            raise RICValidationError("DRR called without active flows")
        for (sender, flow) in tuple(self.drr_deficit_us):
            if sender == sender_rank and flow not in by_flow:
                self.drr_deficit_us[(sender, flow)] = 0.0
        cursor = self.drr_cursor.get(sender_rank)
        continuing = bool(self.drr_visit_continuation.get(sender_rank, False))
        if cursor not in active:
            cursor = active[0]
            continuing = False
        start = active.index(cursor)

        def next_active_after(flow: str, candidates: Sequence[str]) -> str | None:
            if not candidates:
                return None
            origin = active.index(flow)
            candidate_set = set(candidates)
            for offset in range(1, len(active) + 1):
                candidate = active[(origin + offset) % len(active)]
                if candidate in candidate_set:
                    return candidate
            return None

        # Positive Q guarantees termination.  The first iteration may resume
        # an open visit without adding Q; every subsequent visit adds Q once.
        max_head_service = max(
            self.tasks[min(rows, key=lambda row: (row[0], row[1]))[1]]
            .stage_service.sender_pack_us
            for rows in by_flow.values()
        )
        max_rounds = 2 + int(
            math.ceil(max_head_service / self.config.drr_quantum_us)
        )
        for visit in range(max_rounds * len(active)):
            flow = active[(start + visit) % len(active)]
            key = (sender_rank, flow)
            resume_open_visit = visit == 0 and continuing and flow == cursor
            if not resume_open_visit:
                self.drr_deficit_us[key] += self.config.drr_quantum_us
            _enqueue, task_id = min(by_flow[flow], key=lambda row: (row[0], row[1]))
            service = self.tasks[task_id].stage_service.sender_pack_us
            if self.drr_deficit_us[key] + 1e-12 >= service:
                self.drr_deficit_us[key] -= service
                if self.drr_deficit_us[key] < 0 and self.drr_deficit_us[key] > -1e-9:
                    self.drr_deficit_us[key] = 0.0
                remaining = [row for row in by_flow[flow] if row[1] != task_id]
                if remaining:
                    _next_enqueue, next_task_id = min(
                        remaining, key=lambda row: (row[0], row[1])
                    )
                    next_service = self.tasks[
                        next_task_id
                    ].stage_service.sender_pack_us
                else:
                    next_service = math.inf
                if not remaining:
                    # An inactive flow loses residual credit immediately.
                    self.drr_deficit_us[key] = 0.0
                    remaining_active = tuple(item for item in active if item != flow)
                    self.drr_cursor[sender_rank] = next_active_after(
                        flow, remaining_active
                    )
                    self.drr_visit_continuation[sender_rank] = False
                elif self.drr_deficit_us[key] + 1e-12 >= next_service:
                    self.drr_cursor[sender_rank] = flow
                    self.drr_visit_continuation[sender_rank] = True
                else:
                    self.drr_cursor[sender_rank] = next_active_after(flow, active)
                    self.drr_visit_continuation[sender_rank] = False
                return task_id
        raise RICValidationError("DRR failed to select despite positive quantum")

    def _choose_sender(
        self, resource: _Resource, now_us: float
    ) -> tuple[str, bool, bool, bool, int | None]:
        if not resource.queue:
            raise RICValidationError("sender scheduler called with empty queue")
        sender_rank = self.tasks[resource.queue[0][1]].identity.sender_rank
        if any(
            self.tasks[task_id].identity.sender_rank != sender_rank
            for _enqueue, task_id in resource.queue
        ):
            raise RICValidationError("sender-local queue contains another sender's task")
        s_view, b_view = self._views(sender_rank, now_us, resource.queue)
        oldest = min(resource.queue, key=lambda row: (row[0], row[1]))
        if now_us - oldest[0] >= self.config.starvation_us:
            self.starvation_count += 1
            return oldest[1], False, False, True, None
        affected = any(
            (
                sender_rank,
                task.identity.receiver_rank,
                task.identity.epoch,
            )
            in self.fallback_receiver_epochs
            for task in s_view.ready_tasks
        )
        if affected:
            chosen = min(
                s_view.ready_tasks,
                key=lambda task: self._joinblind_key(
                    self.config.calib_best_joinblind, task, b_view
                ),
            )
            return chosen.task_id, True, False, False, None
        if self.arm in JOINBLIND_ARMS or self.arm == "ric_sham_feedback":
            blind_arm = (
                self.config.calib_best_joinblind
                if self.arm == "ric_sham_feedback"
                else self.arm
            )
            if self._resolve_joinblind_arm(blind_arm) == "sender_age_service_drr":
                chosen_id = self._choose_drr(sender_rank, resource.queue)
                chosen = next(
                    task for task in s_view.ready_tasks if task.task_id == chosen_id
                )
            else:
                chosen = min(
                    s_view.ready_tasks,
                    key=lambda task: self._joinblind_key(blind_arm, task, b_view),
                )
            return chosen.task_id, False, False, False, None

        if self.arm == "ric_full_zero_delay":
            local_ready_joins = {
                task.identity.join_identity for task in s_view.ready_tasks
            }
            states = {
                join: ReceiverJoinView(
                    join_identity=join,
                    missing_slot_mask=mask,
                    slack_bucket=_slack_bucket(self.joins[join][0], now_us),
                    epoch=join.epoch,
                )
                for join, mask in self.actual_missing.items()
                if mask and join in local_ready_joins
            }
        else:
            local_ready_joins = {
                task.identity.join_identity for task in s_view.ready_tasks
            }
            cache = self._cache_snapshot(sender_rank)
            full_mask = (1 << self.world.top_k) - 1
            states = {}
            for join in local_ready_joins:
                entry = cache.get(join)
                states[join] = ReceiverJoinView(
                    join_identity=join,
                    missing_slot_mask=(
                        entry.missing_slot_mask if entry is not None else full_mask
                    ),
                    slack_bucket=(
                        entry.slack_bucket
                        if entry is not None
                        else _slack_bucket(self.joins[join][0], now_us)
                    ),
                    epoch=join.epoch,
                )
        r_view = RView(
            base=b_view,
            receiver_join_state=tuple(
                sorted(states.values(), key=lambda state: state.join_identity)
            ),
        )
        validate_r_view(r_view)
        def aware_key(task: ReadyTaskView) -> tuple[object, ...]:
            state = states.get(task.identity.join_identity)
            if state is None:
                return (1, 17, 255, -task.fairness_debt, task.service_us, task.task_id)
            closes = state.missing_slot_mask == (1 << task.identity.topk_slot)
            return (
                0 if closes else 1,
                state.missing_count,
                state.slack_bucket,
                -task.fairness_debt,
                task.service_us,
                task.task_id,
            )

        chosen = min(s_view.ready_tasks, key=aware_key)
        visible = states.get(chosen.identity.join_identity)
        stale = (
            visible is None
            or visible.missing_slot_mask
            != self.actual_missing[chosen.identity.join_identity]
        )
        return (
            chosen.task_id,
            False,
            stale,
            False,
            visible.missing_count if visible is not None else None,
        )

    def _schedule_resource(self, resource: _Resource, now_us: float) -> None:
        if resource.busy_task_id is not None or not resource.queue:
            return
        fallback = False
        stale = False
        starvation = False
        visible_missing: int | None = None
        decision_us = now_us
        if resource.stage_index == 0:
            task_id, fallback, stale, starvation, visible_missing = self._choose_sender(
                resource, now_us
            )
            self.sender_decisions += 1
            self.fallback_decisions += int(fallback)
            self.stale_decisions += int(stale)
            selected_index = next(
                index
                for index, (_enqueue, candidate) in enumerate(resource.queue)
                if candidate == task_id
            )
            enqueue_us, _ = resource.queue.pop(selected_index)
        else:
            selected_index = min(
                range(len(resource.queue)),
                key=lambda index: (resource.queue[index][0], resource.queue[index][1]),
            )
            enqueue_us, task_id = resource.queue.pop(selected_index)
        service = self._service_for_label(task_id, resource.stage_index)
        start_us = now_us
        end_us = start_us + service
        if resource.stage_index == 0:
            self.sender_waits.append(start_us - enqueue_us)
        resource.busy_task_id = task_id
        resource.busy_end_us = end_us
        resource.busy_us += service
        stage_name = self._stage_name(resource.stage_index)
        if resource.stage_index == 3:
            join = self.join_by_label[task_id]
            sender_rank = min(
                sibling.identity.sender_rank for sibling in self.joins[join]
            )
            receiver_rank = join.receiver_rank
        else:
            task = self.tasks[task_id]
            sender_rank = task.identity.sender_rank
            receiver_rank = task.identity.receiver_rank
        self.action_trace.append(
            ActionRecord(
                arm=self.arm,
                task_id=task_id,
                stage=stage_name,
                resource_id=resource.resource_id,
                enqueue_us=enqueue_us,
                decision_us=decision_us,
                service_start_us=start_us,
                service_end_us=end_us,
                sender_rank=sender_rank,
                receiver_rank=receiver_rank,
                visible_missing_count=visible_missing,
                fallback=fallback,
                stale=stale,
                starvation_override=starvation,
            )
        )
        self._push(
            end_us,
            0,
            "stage_done",
            _StageDone(task_id, resource.stage_index, resource.resource_id),
        )

    def _receiver_arrival(self, task: ReplayTask, now_us: float) -> None:
        join = task.join_identity
        slot_bit = 1 << task.identity.topk_slot
        if not self.actual_missing[join] & slot_bit:
            raise RICValidationError("duplicate contribution completion")
        self.actual_missing[join] &= ~slot_bit
        self.completion_by_task[task.task_id] = now_us
        receiver = task.identity.receiver_rank
        self.receiver_outstanding_count[receiver] -= 1
        if self.receiver_outstanding_count[receiver] < 0:
            raise RICValidationError("receiver outstanding ledger underflow")
        if self.actual_missing[join] == 0:
            self._enqueue_combine(join, now_us)
        else:
            self._queue_contract_updates(join, now_us)

    def _queue_contract_updates(
        self, join: JoinIdentity, now_us: float
    ) -> None:
        remaining_mask = self.actual_missing[join]
        if remaining_mask == 0:
            return
        if self.arm not in {
            "ric_compressed_zero_delay",
            "ric_compressed_delayed",
            "ric_wire_charged",
            "ric_sham_feedback",
        }:
            return
        if self.arm == "ric_sham_feedback" and self.reference_control_plan is not None:
            # The sham arm pays the candidate's immutable message schedule.
            # Its sender policy masks all decoded application semantics.
            return
        remaining_senders = {
            sibling.identity.sender_rank
            for sibling in self.joins[join]
            if remaining_mask & (1 << sibling.identity.topk_slot)
        }
        slack = _slack_bucket(self.joins[join][0], now_us)
        signature = (_missing_bucket(remaining_mask), slack)
        for sender_rank in sorted(remaining_senders):
            key = (sender_rank, join)
            if self.last_bucket[key] == signature:
                continue
            self.last_bucket[key] = signature
            self.pending_contracts.append(
                _PendingContract(
                    sender_rank=sender_rank,
                    receiver_rank=join.receiver_rank,
                    epoch=join.epoch,
                    join_identity=join,
                    missing_mask=remaining_mask,
                    slack_bucket=slack,
                )
            )

    def _next_message_sequence(self, sender: int, receiver: int, epoch: int) -> int:
        key = (sender, receiver, epoch)
        value = self.message_sequences.get(key, 0) + 1
        self.message_sequences[key] = value
        return value

    def _faulted_message(
        self, message: ContractMessage, fault: str | None
    ) -> tuple[bytes, int, int]:
        if fault == "missing_sequence":
            message = ContractMessage(
                sender_rank=message.sender_rank,
                receiver_rank=message.receiver_rank,
                epoch=message.epoch,
                sequence=message.sequence + 1,
                records=message.records,
            )
        elif fault == "wrong_sender":
            message = ContractMessage(
                sender_rank=(message.sender_rank + 1) % self.world.ep_size,
                receiver_rank=message.receiver_rank,
                epoch=message.epoch,
                sequence=message.sequence,
                records=message.records,
            )
        payload = encode_contract(message)
        produced_bytes = len(payload)
        if fault == "malformed":
            payload = payload[:-1]
        return payload, message.sequence, produced_bytes

    def _charged_control_transfer_arrival_us(
        self,
        *,
        now_us: float,
        receiver_rank: int,
        tax: ContractTax,
    ) -> float:
        """Schedule receiver encode and return the causal transfer arrival."""

        encode_service = math.fsum(
            (tax.state_build_us, tax.hash_us, tax.encode_us)
        )
        receiver_start = max(
            now_us, self.receiver_control_available_us.get(receiver_rank, 0.0)
        )
        receiver_end = receiver_start + encode_service
        self.receiver_control_available_us[receiver_rank] = receiver_end
        receiver_resource = f"control:receiver:{receiver_rank}:encode"
        self.control_resource_busy_us[receiver_resource] = (
            self.control_resource_busy_us.get(receiver_resource, 0.0)
            + encode_service
        )
        return (
            receiver_end + self.config.wire_delay_us + tax.transfer_us
        )

    def _schedule_sender_control_apply(
        self, transfer_arrival_us: float, transfer: _ControlTransfer
    ) -> None:
        """Enter sender apply FCFS at transfer arrival, never at emission.

        This is a DES transition: arrivals from different receivers may
        overtake one another before they reach the shared sender-local host.
        The application cache is updated only by the final ``wire_delivery``
        event at ``sender_end``.
        """

        tax = transfer.tax
        apply_service = math.fsum(
            (tax.decode_us, tax.lookup_us, tax.apply_us, tax.policy_lookup_us)
        )
        sender_start = max(
            transfer_arrival_us,
            self.sender_control_available_us.get(transfer.sender_rank, 0.0),
        )
        sender_end = sender_start + apply_service
        self.sender_control_available_us[transfer.sender_rank] = sender_end
        sender_resource = f"control:sender:{transfer.sender_rank}:apply"
        self.control_resource_busy_us[sender_resource] = (
            self.control_resource_busy_us.get(sender_resource, 0.0) + apply_service
        )
        self.emitted_control_events.append(
            ControlPlanEvent(
                emission_us=transfer.emission_us,
                delivery_us=sender_end,
                sender_rank=transfer.sender_rank,
                receiver_rank=transfer.receiver_rank,
                epoch=transfer.epoch,
                sequence=transfer.sequence,
                record_count=transfer.record_count,
                tax=transfer.tax,
                payload=transfer.payload,
                produced_bytes=transfer.produced_bytes,
            )
        )
        self._push(
            sender_end,
            2,
            "wire_delivery",
            _WireDelivery(
                transfer.emission_us,
                transfer.sender_rank,
                transfer.receiver_rank,
                transfer.epoch,
                transfer.sequence,
                transfer.record_count,
                transfer.tax,
                transfer.payload,
                transfer.produced_bytes,
                True,
            ),
        )

    def _flush_contracts(self, now_us: float) -> None:
        if not self.pending_contracts:
            return
        grouped: dict[tuple[int, int, int], dict[JoinIdentity, _PendingContract]] = {}
        for update in self.pending_contracts:
            grouped.setdefault(
                (update.sender_rank, update.receiver_rank, update.epoch), {}
            )[update.join_identity] = update
        self.pending_contracts.clear()
        for (sender, receiver, epoch), by_join in sorted(grouped.items()):
            updates = tuple(
                sorted(by_join.values(), key=lambda row: row.join_identity)
            )
            for start in range(0, len(updates), 255):
                chunk = updates[start : start + 255]
                sequence = self._next_message_sequence(sender, receiver, epoch)
                records = tuple(
                    ContractRecord(
                        join_key_hash64=join_identity_hash_parts(update.join_identity)[0],
                        layer_id=update.join_identity.layer_id,
                        missing_slot_mask=update.missing_mask,
                        identity_tag16=join_identity_hash_parts(update.join_identity)[1],
                        slack_bucket=update.slack_bucket,
                        flags=0,
                    )
                    for update in chunk
                )
                message = ContractMessage(sender, receiver, epoch, sequence, records)
                record_count = len(records)
                charged = self.arm in {"ric_wire_charged", "ric_sham_feedback"}
                message_tax = (
                    self.config.contract_tax_surface.tax_for(record_count)
                    if charged
                    else ContractTax()
                )
                fault = (
                    self.config.wire_faults.get((sender, receiver, sequence))
                    if charged
                    else None
                )
                payload, emitted_sequence, produced_bytes = self._faulted_message(
                    message, fault
                )
                copies = 2 if fault == "duplicate" else 1
                for copy_index in range(copies):
                    if charged:
                        transfer_arrival_us = self._charged_control_transfer_arrival_us(
                            now_us=now_us,
                            receiver_rank=receiver,
                            tax=message_tax,
                        )
                        transfer_arrival_us += copy_index * 1e-12
                        self._push(
                            transfer_arrival_us,
                            2,
                            "control_transfer_arrival",
                            _ControlTransfer(
                                now_us,
                                sender,
                                receiver,
                                epoch,
                                emitted_sequence,
                                record_count,
                                message_tax,
                                payload,
                                produced_bytes,
                            ),
                        )
                    else:
                        delivery_us = now_us + (
                            0.0
                            if self.arm == "ric_compressed_zero_delay"
                            else self.config.compressed_delay_us
                        )
                        delivery_us += copy_index * 1e-12
                        self._push(
                            delivery_us,
                            2,
                            "wire_delivery",
                            _WireDelivery(
                                now_us,
                                sender,
                                receiver,
                                epoch,
                                emitted_sequence,
                                record_count,
                                message_tax,
                                payload,
                                produced_bytes,
                                False,
                            ),
                        )

    def _apply_wire(self, delivery: _WireDelivery) -> None:
        if delivery.sender_rank not in self.identity_tables:
            raise RICValidationError("wire delivery targets sender without identity bindings")
        expected_tax = (
            self.config.contract_tax_surface.tax_for(delivery.record_count)
            if delivery.charged
            else ContractTax()
        )
        if delivery.tax != expected_tax:
            raise RICValidationError("wire delivery tax does not match its arm semantics")
        outcome = apply_wire_contract(
            delivery.payload,
            cache=self.wire_caches[delivery.sender_rank],
            identity_table=self.identity_tables[delivery.sender_rank],
            expected_sender_rank=delivery.sender_rank,
            tax=delivery.tax,
            produced_bytes=delivery.produced_bytes,
        )
        self.contract_bytes += outcome.charged_bytes
        self.contract_received_bytes += outcome.received_bytes
        self.contract_header_bytes += HEADER_BYTES
        self.contract_record_bytes += RECORD_BYTES * delivery.record_count
        alignment = delivery.produced_bytes - (
            HEADER_BYTES + RECORD_BYTES * delivery.record_count
        )
        if alignment < 0:
            raise RICValidationError("contract byte components exceed produced bytes")
        self.contract_alignment_bytes += alignment
        self.contract_messages += 1
        self.contract_record_count_histogram[delivery.record_count] = (
            self.contract_record_count_histogram.get(delivery.record_count, 0) + 1
        )
        if delivery.charged:
            for name, value in delivery.tax.__dict__.items():
                self.control_component_us[name] += float(value)
            self.control_component_us["configured_delay_us"] += (
                self.config.wire_delay_us
            )
            if self.arm == "ric_sham_feedback":
                receiver_resource = (
                    f"control:receiver:{delivery.receiver_rank}:encode"
                )
                sender_resource = f"control:sender:{delivery.sender_rank}:apply"
                self.control_resource_busy_us[receiver_resource] = (
                    self.control_resource_busy_us.get(receiver_resource, 0.0)
                    + delivery.tax.state_build_us
                    + delivery.tax.hash_us
                    + delivery.tax.encode_us
                )
                self.control_resource_busy_us[sender_resource] = (
                    self.control_resource_busy_us.get(sender_resource, 0.0)
                    + delivery.tax.decode_us
                    + delivery.tax.lookup_us
                    + delivery.tax.apply_us
                    + delivery.tax.policy_lookup_us
                )
        if outcome.fallback:
            fault = outcome.fault or "unknown_wire_fault"
            self.fault_counts[fault] = self.fault_counts.get(fault, 0) + 1
            self.fallback_receiver_epochs.add(
                (delivery.sender_rank, delivery.receiver_rank, delivery.epoch)
            )
        else:
            self.fallback_receiver_epochs.discard(
                (delivery.sender_rank, delivery.receiver_rank, delivery.epoch)
            )

    def _process_event(self, event: _Event) -> None:
        if event.kind == "task_ready":
            self._enqueue_stage(
                str(event.payload), 0, event.time_us, new_outstanding=True
            )
            return
        if event.kind == "control_transfer_arrival":
            if type(event.payload) is not _ControlTransfer:
                raise RICValidationError("control transfer event has invalid payload")
            self._schedule_sender_control_apply(event.time_us, event.payload)
            return
        if event.kind == "wire_delivery":
            self._apply_wire(event.payload)
            return
        if event.kind == "slack_transition":
            if type(event.payload) is not JoinIdentity:
                raise RICValidationError("slack transition lacks canonical join identity")
            self._queue_contract_updates(event.payload, event.time_us)
            return
        if event.kind != "stage_done":
            raise RICValidationError(f"unknown replay event {event.kind!r}")
        done: _StageDone = event.payload
        resource = self.resources[done.resource_id]
        if resource.busy_task_id != done.task_id:
            raise RICValidationError("resource completion does not match busy task")
        resource.busy_task_id = None
        resource.busy_end_us = None
        if done.stage_index == 3:
            join = self.join_by_label[done.task_id]
            if join in self.combine_seen:
                raise RICValidationError("duplicate join combine completion")
            self.combine_seen.add(join)
            self.join_completion[join] = event.time_us
            self.receiver_outstanding_count[join.receiver_rank] -= 1
            if self.receiver_outstanding_count[join.receiver_rank] < 0:
                raise RICValidationError("receiver combine ledger underflow")
            return
        stage_key = (done.task_id, done.stage_index)
        if stage_key in self.stage_seen:
            raise RICValidationError("duplicate task stage completion")
        self.stage_seen.add(stage_key)
        if done.stage_index == 0:
            task = self.tasks[done.task_id]
            flow_key = (task.identity.sender_rank, task.identity.request_id)
            self.cumulative_sender_service[flow_key] = (
                self.cumulative_sender_service.get(flow_key, 0.0)
                + task.stage_service.sender_pack_us
            )
        if done.stage_index < 2:
            self._enqueue_stage(done.task_id, done.stage_index + 1, event.time_us)
        else:
            self._receiver_arrival(self.tasks[done.task_id], event.time_us)

    @staticmethod
    def _event_stable_key(event: _Event) -> tuple[object, ...]:
        if event.kind == "task_ready":
            return (event.priority, event.kind, str(event.payload))
        if event.kind == "stage_done":
            done: _StageDone = event.payload
            return (
                event.priority,
                event.kind,
                done.stage_index,
                done.resource_id,
                done.task_id,
            )
        if event.kind == "slack_transition":
            join: JoinIdentity = event.payload
            return (event.priority, event.kind, join.canonical_tuple())
        if event.kind == "control_transfer_arrival":
            transfer: _ControlTransfer = event.payload
            return (
                event.priority,
                event.kind,
                transfer.sender_rank,
                transfer.receiver_rank,
                transfer.epoch,
                transfer.sequence,
                transfer.payload,
            )
        if event.kind == "wire_delivery":
            delivery: _WireDelivery = event.payload
            return (
                event.priority,
                event.kind,
                delivery.sender_rank,
                delivery.receiver_rank,
                delivery.epoch,
                delivery.sequence,
                delivery.payload,
            )
        return (event.priority, event.kind, event.sequence)

    def run(self) -> ReplayResult:
        while self.events:
            now_us = self.events[0].time_us
            # Process all instantaneous work at this timestamp, including
            # zero-delay contracts emitted by simultaneous ingress completions.
            while True:
                progressed = False
                batch = []
                while self.events and self.events[0].time_us == now_us:
                    batch.append(heapq.heappop(self.events))
                for event in sorted(batch, key=self._event_stable_key):
                    self._process_event(event)
                    progressed = True
                if self.pending_contracts:
                    self._flush_contracts(now_us)
                    progressed = True
                    continue
                if not progressed or not (
                    self.events and self.events[0].time_us == now_us
                ):
                    break
            for resource_id in sorted(self.resources):
                self._schedule_resource(self.resources[resource_id], now_us)

        if any(resource.busy_task_id is not None or resource.queue for resource in self.resources.values()):
            raise RICValidationError("event heap drained before resource queues")
        if any(self.receiver_outstanding_count.values()):
            raise RICValidationError("receiver outstanding ledger did not drain")
        self._assert_aggregate_ledger_matches_scan()
        expected_resource_demand: dict[str, float] = {
            resource_id: 0.0 for resource_id in self.resources
        }
        source_by_field: dict[str, str] = {}
        for task in self.world.tasks:
            for resource_id, service, source in zip(
                task.stage_resources,
                task.stage_service_us,
                task.stage_service.source_tags[:3],
            ):
                expected_resource_demand[resource_id] += service
                prior = source_by_field.setdefault(f"resource:{resource_id}", source)
                if prior != source:
                    raise RICValidationError("one resource mixes service source semantics")
        for join, siblings in self.joins.items():
            task = siblings[0]
            expected_resource_demand[task.combine_resource] += (
                task.stage_service.join_combine_us
            )
            prior = source_by_field.setdefault(
                f"resource:{task.combine_resource}",
                task.stage_service.join_combine_source,
            )
            if prior != task.stage_service.join_combine_source:
                raise RICValidationError("one combine resource mixes source semantics")
        actual_resource_demand = {
            resource_id: resource.busy_us
            for resource_id, resource in sorted(self.resources.items())
        }
        for resource_id, expected in expected_resource_demand.items():
            if not math.isclose(
                actual_resource_demand[resource_id],
                expected,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise RICValidationError("resource service demand was not conserved")
        if self.emitted_control_events:
            control_parts: dict[str, list[float]] = {}
            for event in self.emitted_control_events:
                control_parts.setdefault(
                    f"control:receiver:{event.receiver_rank}:encode", []
                ).append(
                    math.fsum(
                        (
                            event.tax.state_build_us,
                            event.tax.hash_us,
                            event.tax.encode_us,
                        )
                    )
                )
                control_parts.setdefault(
                    f"control:sender:{event.sender_rank}:apply", []
                ).append(
                    math.fsum(
                        (
                            event.tax.decode_us,
                            event.tax.lookup_us,
                            event.tax.apply_us,
                            event.tax.policy_lookup_us,
                        )
                    )
                )
            self.control_resource_busy_us = {
                key: math.fsum(values)
                for key, values in sorted(control_parts.items())
            }
        all_resource_demand = dict(actual_resource_demand)
        all_resource_demand.update(
            dict(sorted(self.control_resource_busy_us.items()))
        )
        for resource_id in self.control_resource_busy_us:
            source_by_field[f"resource:{resource_id}"] = "measured_5090_host"
        expected_stage_count = 3 * len(self.tasks) + len(self.joins)
        full_drain = (
            len(self.completion_by_task) == len(self.tasks)
            and len(self.stage_seen) + len(self.combine_seen) == expected_stage_count
            and len(self.join_completion) == len(self.joins)
            and all(mask == 0 for mask in self.actual_missing.values())
        )
        if not full_drain:
            raise RICValidationError("RIC replay full-drain/exactly-once failure")
        join_completion: dict[str, float] = {}
        all_latencies: dict[str, float] = {}
        scored_latencies: dict[str, float] = {}
        for join, siblings in self.joins.items():
            completion = self.join_completion[join]
            release = siblings[0].contribution.arrival_us
            label = _join_label(join)
            join_completion[label] = completion
            all_latencies[label] = completion - release
            if join in self.world.scored_joins:
                scored_latencies[label] = completion - release
        first_ready = min(task.contribution.ready_us for task in self.world.tasks)
        final_completion = max(self.join_completion.values())
        final_horizon = max(final_completion, now_us)
        makespan = final_horizon - first_ready
        source_tags = set(self.world.source_tags)
        if self.arm in {"ric_compressed_delayed", "ric_wire_charged", "ric_sham_feedback"}:
            source_tags.add("synthetic_delay")
        if self.arm in {"ric_wire_charged", "ric_sham_feedback"}:
            source_tags.update({"measured_5090_host", "analytic_network"})
        return ReplayResult(
            trace_id=self.world.trace_id,
            workload_seed=self.world.workload_seed,
            model_key=self.world.model_key,
            cell=self.world.cell,
            arm=self.arm,
            task_count=len(self.tasks),
            completed_task_count=len(self.completion_by_task),
            completed_stage_count=len(self.stage_seen) + len(self.combine_seen),
            expected_stage_count=expected_stage_count,
            completed_join_count=len(self.combine_seen),
            expected_join_count=len(self.joins),
            task_fingerprint=self.world.task_fingerprint,
            service_fingerprint=self.world.service_fingerprint,
            score_mask_fingerprint=self.world.score_mask_fingerprint,
            resource_demand_fingerprint=self.world.resource_demand_fingerprint,
            payload_bytes=self.world.full_load_audit.payload_bytes,
            descriptor_bytes=self.world.full_load_audit.descriptor_bytes,
            alignment_bytes=self.world.full_load_audit.alignment_bytes,
            contract_bytes=self.contract_bytes,
            contract_received_bytes=self.contract_received_bytes,
            contract_header_bytes=self.contract_header_bytes,
            contract_record_bytes=self.contract_record_bytes,
            contract_alignment_bytes=self.contract_alignment_bytes,
            contract_messages=self.contract_messages,
            contract_record_count_histogram=dict(
                sorted(self.contract_record_count_histogram.items())
            ),
            contract_tax_surface_source_id=(
                self.config.contract_tax_surface.source_id
            ),
            contract_tax_surface_fingerprint=(
                self.config.contract_tax_surface.fingerprint
            ),
            contract_tax_non_grid_rule=(
                self.config.contract_tax_surface.non_grid_rule
            ),
            control_component_us=dict(self.control_component_us),
            stale_decisions=self.stale_decisions,
            fallback_decisions=self.fallback_decisions,
            sender_decisions=self.sender_decisions,
            starvation_count=self.starvation_count,
            sender_ready_wait_us=tuple(self.sender_waits),
            makespan_us=makespan,
            queue_busy_us={
                **{
                    resource_id: resource.busy_us
                    for resource_id, resource in sorted(self.resources.items())
                },
                **dict(sorted(self.control_resource_busy_us.items())),
            },
            resource_service_demand_us=all_resource_demand,
            source_by_field=dict(sorted(source_by_field.items())),
            source_tags=tuple(sorted(source_tags)),
            full_drain=full_drain,
            completion_by_task_us=dict(self.completion_by_task),
            join_completion_us=join_completion,
            all_join_latencies_us=all_latencies,
            scored_join_latencies_us=scored_latencies,
            action_trace=tuple(self.action_trace),
            fault_counts=dict(self.fault_counts),
            control_plan=(
                ControlPlan(
                    trace_id=self.world.trace_id,
                    task_fingerprint=self.world.task_fingerprint,
                    service_fingerprint=self.world.service_fingerprint,
                    score_mask_fingerprint=self.world.score_mask_fingerprint,
                    resource_demand_fingerprint=(
                        self.world.resource_demand_fingerprint
                    ),
                    contract_tax_surface=self.config.contract_tax_surface,
                    contract_tax_surface_fingerprint=(
                        self.config.contract_tax_surface.fingerprint
                    ),
                    wire_delay_us=self.config.wire_delay_us,
                    starvation_us=self.config.starvation_us,
                    drr_quantum_us=self.config.drr_quantum_us,
                    drr_service_fingerprint=(
                        self.config.drr_service_fingerprint
                    ),
                    calib_best_joinblind=self.config.calib_best_joinblind,
                    events=tuple(
                        sorted(
                            self.emitted_control_events,
                            key=lambda event: (
                                event.delivery_us,
                                event.sender_rank,
                                event.receiver_rank,
                                event.sequence,
                                event.payload,
                            ),
                        )
                    ),
                )
                if self.arm in {"ric_wire_charged", "ric_sham_feedback"}
                else None
            ),
        )


def run_replay(
    world: ReplayWorld, *, arm: str, config: ReplayConfig | None = None
) -> ReplayResult:
    """Run one deterministic, common-random-number RIC workload arm."""

    if type(world) is not ReplayWorld:
        raise RICValidationError("run_replay requires a validated ReplayWorld")
    return _Simulator(world, arm, config or ReplayConfig()).run()


def run_sham_against_reference(
    world: ReplayWorld,
    *,
    charged_plan: ControlPlan,
    config: ReplayConfig | None = None,
) -> ReplayResult:
    """Run join-blind sham policy against the charged arm's exact control plan.

    The decoded cache is intentionally ignored by the scheduler.  This arm is
    a control-cost counterfactual only; it is not an online baseline claim.
    """

    if type(charged_plan) is not ControlPlan:
        raise RICValidationError("sham requires the canonical ControlPlan type")
    return _Simulator(
        world,
        "ric_sham_feedback",
        config
        or ReplayConfig(
            contract_tax_surface=charged_plan.contract_tax_surface,
            wire_delay_us=charged_plan.wire_delay_us,
            starvation_us=charged_plan.starvation_us,
            drr_quantum_us=charged_plan.drr_quantum_us,
            drr_service_fingerprint=charged_plan.drr_service_fingerprint,
            calib_best_joinblind=charged_plan.calib_best_joinblind,
        ),
        reference_control_plan=charged_plan,
    ).run()


def action_signature(result: ReplayResult, *, stage: str = "sender_egress") -> tuple[str, ...]:
    return tuple(
        record.task_id for record in result.action_trace if record.stage == stage
    )


def action_collapse_matrix(
    results: Sequence[ReplayResult], *, stage: str = "sender_egress"
) -> dict[str, dict[str, bool]]:
    if not results:
        raise RICValidationError("cannot form action collapse matrix from no arms")
    signatures = {result.arm: action_signature(result, stage=stage) for result in results}
    if len(signatures) != len(results):
        raise RICValidationError("duplicate arm in action collapse matrix")
    return {
        left: {right: signatures[left] == signatures[right] for right in signatures}
        for left in signatures
    }


__all__ = [
    "JOINBLIND_ARMS",
    "RIC_ARMS",
    "TAX_RECORD_COUNT_GRID",
    "TAX_NON_GRID_RULE",
    "ContractTaxSurface",
    "ReplayConfig",
    "ActionRecord",
    "ControlPlanEvent",
    "ControlPlan",
    "ReplayResult",
    "run_replay",
    "run_sham_against_reference",
    "action_signature",
    "action_collapse_matrix",
]
