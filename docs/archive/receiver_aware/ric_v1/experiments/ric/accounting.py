"""Trace-level metrics, conservation, and paired inference for RIC-v1.

The statistical unit in this module is deliberately a *complete workload
trace*.  Requests, token blocks, and contributions that shared a replayed
queue never appear as bootstrap units.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Mapping, Protocol, Sequence


class RICAccountingError(ValueError):
    """Fail-closed accounting or statistical validation error."""


ALLOWED_SOURCE_TAGS = frozenset(
    {
        "measured_5090_cuda",
        "measured_5090_host",
        "measured_5090_h2d_not_rdma",
        "derived_from_measured_lut",
        "analytic_network",
        "synthetic_delay",
    }
)

EXACT_CONTROL_TAX_RULE = "exact_1_to_255_no_interpolation_or_extrapolation"
CONTROL_COMPONENT_FIELDS = frozenset(
    {
        "state_build_us",
        "hash_us",
        "encode_us",
        "configured_delay_us",
        "transfer_us",
        "decode_us",
        "lookup_us",
        "apply_us",
        "policy_lookup_us",
    }
)


class ReplayResultLike(Protocol):
    trace_id: str
    workload_seed: int
    model_key: str
    cell: str
    arm: str
    scored_join_latencies_us: Mapping[str, float]
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
    sender_ready_wait_us: Sequence[float]
    makespan_us: float
    queue_busy_us: Mapping[str, float]
    resource_service_demand_us: Mapping[str, float]
    source_by_field: Mapping[str, str]
    source_tags: Sequence[str]
    full_drain: bool


def _finite_values(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise RICAccountingError(f"{name} must contain finite observations")
    return result


def quantile_type1(values: Sequence[float], q: float) -> float:
    """Nearest-rank empirical quantile (Hyndman-Fan type 1)."""

    observations = sorted(_finite_values(values, name="quantile input"))
    if not 0.0 <= q <= 1.0:
        raise RICAccountingError("quantile probability outside [0, 1]")
    if q == 0.0:
        return observations[0]
    rank = max(1, math.ceil(q * len(observations)))
    return observations[rank - 1]


def empirical_cvar(values: Sequence[float], alpha: float = 0.99) -> float:
    """Upper-tail empirical CVaR with a fractional boundary observation.

    This is the uniform empirical Rockafellar-Uryasev CVaR.  Using a
    fractional boundary avoids silently changing the tail mass when
    ``(1-alpha) * n`` is non-integral.
    """

    observations = sorted(
        _finite_values(values, name="CVaR input"), reverse=True
    )
    if not 0.0 <= alpha < 1.0:
        raise RICAccountingError("CVaR alpha must be in [0, 1)")
    tail_mass = (1.0 - alpha) * len(observations)
    if tail_mass <= 0.0:
        raise RICAccountingError("CVaR tail has zero mass")
    full = int(math.floor(tail_mass))
    fractional = tail_mass - full
    weighted_sum = sum(observations[:full])
    if fractional > 0.0:
        weighted_sum += fractional * observations[full]
    return weighted_sum / tail_mass


@dataclass(frozen=True)
class TraceMetrics:
    """One aggregate row for one complete trace and one arm."""

    trace_id: str
    workload_seed: int
    model_key: str
    cell: str
    arm: str
    closure_count: int
    p50_us: float
    p95_us: float
    p99_us: float
    cvar99_us: float
    violation_rate: float
    closure_budget_us: float
    control_bytes_over_payload: float
    stale_rate: float
    fallback_rate: float
    sender_ready_wait_mean_us: float
    sender_ready_wait_p99_us: float
    starvation_count: int
    full_drain_goodput_per_us: float
    queue_utilization: float
    task_fingerprint: str
    service_fingerprint: str
    contract_tax_surface_fingerprint: str
    score_mask_fingerprint: str
    resource_demand_fingerprint: str

    def metric(self, name: str) -> float:
        if name == "cvar99":
            return self.cvar99_us
        if name == "violation":
            return self.violation_rate
        if name == "p99":
            return self.p99_us
        raise RICAccountingError(f"unknown trace metric {name!r}")


def trace_metrics_from_result(
    result: ReplayResultLike, *, closure_budget_us: float
) -> TraceMetrics:
    """Reduce a fully drained replay to one non-IID-safe trace row."""

    if closure_budget_us <= 0 or not math.isfinite(closure_budget_us):
        raise RICAccountingError("closure budget must be finite and positive")
    if not result.full_drain:
        raise RICAccountingError("partial replay cannot produce trace metrics")
    if result.completed_task_count != result.task_count:
        raise RICAccountingError("task conservation failed")
    if result.completed_stage_count != result.expected_stage_count:
        raise RICAccountingError("stage conservation failed")
    if result.completed_join_count != result.expected_join_count:
        raise RICAccountingError("once-only join combine conservation failed")
    unknown_sources = set(result.source_tags) - ALLOWED_SOURCE_TAGS
    if unknown_sources:
        raise RICAccountingError(f"unknown accounting source tags: {unknown_sources}")
    latencies = _finite_values(
        result.scored_join_latencies_us.values(), name="join closure latencies"
    )
    if any(value < 0 for value in latencies):
        raise RICAccountingError("negative join closure latency")
    payload = int(result.payload_bytes)
    if payload <= 0 or result.contract_bytes < 0:
        raise RICAccountingError("invalid byte ledger")
    byte_components = (
        int(result.contract_header_bytes),
        int(result.contract_record_bytes),
        int(result.contract_alignment_bytes),
    )
    if any(value < 0 for value in byte_components) or sum(byte_components) != int(
        result.contract_bytes
    ):
        raise RICAccountingError("contract byte components do not conserve produced bytes")
    if not 0 <= int(result.contract_received_bytes) <= int(result.contract_bytes):
        raise RICAccountingError("contract received/produced byte ledger is invalid")
    if result.sender_decisions < 0:
        raise RICAccountingError("negative decision count")
    if result.contract_tax_non_grid_rule != EXACT_CONTROL_TAX_RULE:
        raise RICAccountingError("control tax surface permits interpolation/extrapolation")
    if (
        not result.contract_tax_surface_source_id
        or len(result.contract_tax_surface_fingerprint) != 64
    ):
        raise RICAccountingError("control tax surface provenance is incomplete")
    histogram = dict(result.contract_record_count_histogram)
    if any(
        type(count) is not int
        or not 1 <= count <= 255
        or type(frequency) is not int
        or frequency <= 0
        for count, frequency in histogram.items()
    ):
        raise RICAccountingError("invalid contract record-count histogram")
    if sum(histogram.values()) != result.contract_messages:
        raise RICAccountingError("contract message/tax histogram conservation failed")
    if result.contract_header_bytes != 16 * result.contract_messages:
        raise RICAccountingError("contract header byte ledger mismatch")
    if result.contract_record_bytes != 16 * sum(
        count * frequency for count, frequency in histogram.items()
    ):
        raise RICAccountingError("contract record byte ledger mismatch")
    component_values = {
        name: float(value) for name, value in result.control_component_us.items()
    }
    if set(component_values) != CONTROL_COMPONENT_FIELDS or any(
        not math.isfinite(value) or value < 0
        for value in component_values.values()
    ):
        raise RICAccountingError("invalid additive control component ledger")
    if result.arm in {"ric_wire_charged", "ric_sham_feedback"} and result.contract_messages and any(
        component_values[name] <= 0
        for name in CONTROL_COMPONENT_FIELDS - {"configured_delay_us"}
    ):
        raise RICAccountingError(
            "charged messages require positive harness-subtracted additive tax"
        )
    stale_rate = (
        result.stale_decisions / result.sender_decisions
        if result.sender_decisions
        else 0.0
    )
    fallback_rate = (
        result.fallback_decisions / result.sender_decisions
        if result.sender_decisions
        else 0.0
    )
    waits = tuple(float(value) for value in result.sender_ready_wait_us)
    if any(value < 0 or not math.isfinite(value) for value in waits):
        raise RICAccountingError("invalid sender ready wait")
    makespan = float(result.makespan_us)
    if makespan <= 0 or not math.isfinite(makespan):
        raise RICAccountingError("invalid replay makespan")
    busy_values = tuple(float(value) for value in result.queue_busy_us.values())
    if not busy_values or any(value < 0 for value in busy_values):
        raise RICAccountingError("invalid resource busy-time ledger")
    utilization = sum(busy_values) / (len(busy_values) * makespan)
    if utilization > 1.0 + 1e-9:
        raise RICAccountingError("resource utilization exceeds one")
    if set(result.queue_busy_us) != set(result.resource_service_demand_us):
        raise RICAccountingError("resource busy/service key set mismatch")
    for resource_id, busy in result.queue_busy_us.items():
        if not math.isclose(
            float(busy),
            float(result.resource_service_demand_us[resource_id]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise RICAccountingError("resource busy/service demand mismatch")
    expected_source_fields = {f"resource:{name}" for name in result.queue_busy_us}
    if set(result.source_by_field) != expected_source_fields:
        raise RICAccountingError("per-resource source provenance is incomplete")
    if set(result.source_by_field.values()) - ALLOWED_SOURCE_TAGS:
        raise RICAccountingError("per-resource source provenance has unknown tags")
    for name in (
        result.task_fingerprint,
        result.service_fingerprint,
        result.score_mask_fingerprint,
        result.resource_demand_fingerprint,
    ):
        if not isinstance(name, str) or len(name) != 64:
            raise RICAccountingError("replay fingerprint binding is incomplete")
    return TraceMetrics(
        trace_id=result.trace_id,
        workload_seed=result.workload_seed,
        model_key=result.model_key,
        cell=result.cell,
        arm=result.arm,
        closure_count=len(latencies),
        p50_us=quantile_type1(latencies, 0.50),
        p95_us=quantile_type1(latencies, 0.95),
        p99_us=quantile_type1(latencies, 0.99),
        cvar99_us=empirical_cvar(latencies, 0.99),
        violation_rate=sum(value > closure_budget_us for value in latencies)
        / len(latencies),
        closure_budget_us=closure_budget_us,
        control_bytes_over_payload=result.contract_bytes / payload,
        stale_rate=stale_rate,
        fallback_rate=fallback_rate,
        sender_ready_wait_mean_us=(sum(waits) / len(waits) if waits else 0.0),
        sender_ready_wait_p99_us=(quantile_type1(waits, 0.99) if waits else 0.0),
        starvation_count=result.starvation_count,
        full_drain_goodput_per_us=result.completed_task_count / makespan,
        queue_utilization=utilization,
        task_fingerprint=result.task_fingerprint,
        service_fingerprint=result.service_fingerprint,
        contract_tax_surface_fingerprint=(
            result.contract_tax_surface_fingerprint
        ),
        score_mask_fingerprint=result.score_mask_fingerprint,
        resource_demand_fingerprint=result.resource_demand_fingerprint,
    )


@dataclass(frozen=True)
class PairedBootstrapSummary:
    baseline_arm: str
    candidate_arm: str
    model_key: str
    cell: str
    n_traces: int
    n_bootstrap: int
    confidence: float
    cvar99_relative_reduction: float
    cvar99_relative_reduction_lcb: float
    cvar99_relative_reduction_ucb: float
    violation_absolute_reduction: float
    violation_absolute_reduction_lcb: float
    violation_absolute_reduction_ucb: float


def _paired_index(
    rows: Sequence[TraceMetrics], required_arms: Sequence[str]
) -> tuple[dict[tuple[str, str], TraceMetrics], tuple[str, ...], str, str]:
    if not rows:
        raise RICAccountingError("empty trace metric table")
    required = tuple(dict.fromkeys(required_arms))
    if len(required) != len(required_arms):
        raise RICAccountingError("duplicate required arm")
    models = {row.model_key for row in rows}
    cells = {row.cell for row in rows}
    if len(models) != 1 or len(cells) != 1:
        raise RICAccountingError("bootstrap must be cellwise and modelwise")
    index: dict[tuple[str, str], TraceMetrics] = {}
    trace_ids: set[str] = set()
    trace_to_seed: dict[str, int] = {}
    seed_to_trace: dict[int, str] = {}
    for row in rows:
        if isinstance(row.workload_seed, bool) or not isinstance(
            row.workload_seed, int
        ):
            raise RICAccountingError("workload seed must be an integer")
        prior_seed = trace_to_seed.setdefault(row.trace_id, row.workload_seed)
        if prior_seed != row.workload_seed:
            raise RICAccountingError("one trace_id is associated with multiple workload seeds")
        prior_trace = seed_to_trace.setdefault(row.workload_seed, row.trace_id)
        if prior_trace != row.trace_id:
            raise RICAccountingError(
                "duplicate workload seed under different trace_id does not add independence"
            )
        key = (row.trace_id, row.arm)
        if key in index:
            raise RICAccountingError(f"duplicate complete trace/arm row: {key}")
        index[key] = row
        trace_ids.add(row.trace_id)
    for trace_id in trace_ids:
        missing = [arm for arm in required if (trace_id, arm) not in index]
        if missing:
            raise RICAccountingError(
                f"unpaired complete trace {trace_id}; missing arms {missing}"
            )
        fingerprints = {
            (
                index[(trace_id, arm)].task_fingerprint,
                index[(trace_id, arm)].service_fingerprint,
                index[(trace_id, arm)].resource_demand_fingerprint,
                index[(trace_id, arm)].score_mask_fingerprint,
                index[(trace_id, arm)].contract_tax_surface_fingerprint,
                index[(trace_id, arm)].closure_count,
                index[(trace_id, arm)].closure_budget_us,
            )
            for arm in required
        }
        if len(fingerprints) != 1:
            raise RICAccountingError("paired arms do not share workload/service fingerprint")
    return index, tuple(sorted(trace_ids)), next(iter(models)), next(iter(cells))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RICAccountingError("mean of empty values")
    return sum(values) / len(values)


def _effects(
    index: Mapping[tuple[str, str], TraceMetrics],
    trace_ids: Sequence[str],
    baseline_arm: str,
    candidate_arm: str,
) -> tuple[float, float]:
    baseline_cvar = _mean(
        [index[(trace_id, baseline_arm)].cvar99_us for trace_id in trace_ids]
    )
    candidate_cvar = _mean(
        [index[(trace_id, candidate_arm)].cvar99_us for trace_id in trace_ids]
    )
    if baseline_cvar <= 0:
        raise RICAccountingError("non-positive baseline CVaR")
    baseline_violation = _mean(
        [index[(trace_id, baseline_arm)].violation_rate for trace_id in trace_ids]
    )
    candidate_violation = _mean(
        [index[(trace_id, candidate_arm)].violation_rate for trace_id in trace_ids]
    )
    return (
        (baseline_cvar - candidate_cvar) / baseline_cvar,
        baseline_violation - candidate_violation,
    )


def paired_trace_bootstrap(
    rows: Sequence[TraceMetrics],
    *,
    baseline_arm: str,
    candidate_arm: str,
    n_bootstrap: int = 10_000,
    confidence: float = 0.9875,
    seed: int = 2026072223,
    expected_trace_count: int | None = None,
) -> PairedBootstrapSummary:
    """Paired cluster bootstrap over complete workload traces only."""

    if n_bootstrap <= 0 or not 0.5 < confidence < 1.0:
        raise RICAccountingError("invalid bootstrap configuration")
    index, trace_ids, model, cell = _paired_index(
        rows, (baseline_arm, candidate_arm)
    )
    if expected_trace_count is not None and len(trace_ids) != expected_trace_count:
        raise RICAccountingError(
            f"expected {expected_trace_count} complete traces, got {len(trace_ids)}"
        )
    point_cvar, point_violation = _effects(
        index, trace_ids, baseline_arm, candidate_arm
    )
    rng = random.Random(seed)
    cvar_samples: list[float] = []
    violation_samples: list[float] = []
    for _ in range(n_bootstrap):
        sampled = tuple(rng.choice(trace_ids) for _ in trace_ids)
        cvar, violation = _effects(index, sampled, baseline_arm, candidate_arm)
        cvar_samples.append(cvar)
        violation_samples.append(violation)
    tail = 1.0 - confidence
    return PairedBootstrapSummary(
        baseline_arm=baseline_arm,
        candidate_arm=candidate_arm,
        model_key=model,
        cell=cell,
        n_traces=len(trace_ids),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        cvar99_relative_reduction=point_cvar,
        cvar99_relative_reduction_lcb=quantile_type1(cvar_samples, tail),
        cvar99_relative_reduction_ucb=quantile_type1(cvar_samples, confidence),
        violation_absolute_reduction=point_violation,
        violation_absolute_reduction_lcb=quantile_type1(violation_samples, tail),
        violation_absolute_reduction_ucb=quantile_type1(
            violation_samples, confidence
        ),
    )


@dataclass(frozen=True)
class RetentionBootstrapSummary:
    metric: str
    n_traces: int
    point: float
    lcb: float
    ucb: float
    invalid_denominator_fraction: float


def paired_retention_bootstrap(
    rows: Sequence[TraceMetrics],
    *,
    baseline_arm: str,
    r0_arm: str,
    charged_arm: str,
    metric: str = "cvar99",
    n_bootstrap: int = 10_000,
    confidence: float = 0.9875,
    seed: int = 2026072224,
    denominator_epsilon: float = 1e-12,
) -> RetentionBootstrapSummary:
    """Bootstrap ``(B - Rwire) / (B - R0)`` without clipping ratios."""

    if n_bootstrap <= 0 or not 0.5 < confidence < 1.0:
        raise RICAccountingError("invalid retention bootstrap configuration")
    if not math.isfinite(denominator_epsilon) or denominator_epsilon < 0:
        raise RICAccountingError("invalid retention denominator epsilon")
    index, trace_ids, _model, _cell = _paired_index(
        rows, (baseline_arm, r0_arm, charged_arm)
    )

    def retention(selected: Sequence[str]) -> float:
        baseline = _mean(
            [index[(trace_id, baseline_arm)].metric(metric) for trace_id in selected]
        )
        r0 = _mean([index[(trace_id, r0_arm)].metric(metric) for trace_id in selected])
        charged = _mean(
            [index[(trace_id, charged_arm)].metric(metric) for trace_id in selected]
        )
        denominator = baseline - r0
        if denominator <= denominator_epsilon:
            raise RICAccountingError(
                "non-positive or near-zero zero-tax information headroom"
            )
        return (baseline - charged) / denominator

    point = retention(trace_ids)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        selected = tuple(rng.choice(trace_ids) for _ in trace_ids)
        # Fail closed on the first invalid denominator.  Dropping such draws
        # conditions on positive headroom and can spuriously raise the LCB.
        samples.append(retention(selected))
    tail = 1.0 - confidence
    return RetentionBootstrapSummary(
        metric=metric,
        n_traces=len(trace_ids),
        point=point,
        lcb=quantile_type1(samples, tail),
        ucb=quantile_type1(samples, confidence),
        invalid_denominator_fraction=0.0,
    )


@dataclass(frozen=True)
class DeploymentDecomposition:
    headroom: float
    compression_loss: float
    delay_loss: float
    wire_tax: float
    net_value: float
    residual: float


def deployment_decomposition(
    rows: Mapping[str, TraceMetrics],
    *,
    baseline_arm: str,
    r0_arm: str,
    compressed_arm: str,
    delayed_arm: str,
    wire_arm: str,
    metric: str = "cvar99",
    tolerance: float = 1e-9,
) -> DeploymentDecomposition:
    """Verify the frozen RIC-v1 telescoping identity for one fixture."""

    required = (baseline_arm, r0_arm, compressed_arm, delayed_arm, wire_arm)
    missing = [arm for arm in required if arm not in rows]
    if missing:
        raise RICAccountingError(f"deployment ledger missing arms {missing}")
    selected = tuple(rows[arm] for arm in required)
    if any(type(row) is not TraceMetrics for row in selected):
        raise RICAccountingError("deployment decomposition requires bound trace rows")
    bindings = {
        (
            row.trace_id,
            row.workload_seed,
            row.model_key,
            row.cell,
            row.task_fingerprint,
            row.service_fingerprint,
            row.resource_demand_fingerprint,
            row.score_mask_fingerprint,
            row.contract_tax_surface_fingerprint,
            row.closure_count,
            row.closure_budget_us,
        )
        for row in selected
    }
    if len(bindings) != 1:
        raise RICAccountingError("deployment rows are not one matched trace world")
    values = {arm: rows[arm].metric(metric) for arm in required}
    if any(not math.isfinite(float(values[arm])) for arm in required):
        raise RICAccountingError("deployment ledger contains non-finite metric")
    headroom = values[baseline_arm] - values[r0_arm]
    compression = values[compressed_arm] - values[r0_arm]
    delay = values[delayed_arm] - values[compressed_arm]
    wire = values[wire_arm] - values[delayed_arm]
    net = values[baseline_arm] - values[wire_arm]
    residual = net - (headroom - compression - delay - wire)
    if abs(residual) > tolerance:
        raise RICAccountingError(
            f"deployment accounting does not telescope; residual={residual}"
        )
    return DeploymentDecomposition(
        headroom=headroom,
        compression_loss=compression,
        delay_loss=delay,
        wire_tax=wire,
        net_value=net,
        residual=residual,
    )


def assert_replay_conservation(results: Sequence[ReplayResultLike]) -> None:
    """Assert workload/service/byte equality and full drain across arms."""

    if not results:
        raise RICAccountingError("no replay results")
    task_counts = {result.task_count for result in results}
    completed = {result.completed_task_count for result in results}
    stage_counts = {
        (result.completed_stage_count, result.expected_stage_count)
        for result in results
    }
    join_counts = {
        (result.completed_join_count, result.expected_join_count)
        for result in results
    }
    task_fingerprints = {result.task_fingerprint for result in results}
    service_fingerprints = {result.service_fingerprint for result in results}
    score_mask_fingerprints = {
        result.score_mask_fingerprint for result in results
    }
    resource_demand_fingerprints = {
        result.resource_demand_fingerprint for result in results
    }
    tax_surface_fingerprints = {
        result.contract_tax_surface_fingerprint for result in results
    }
    workload_seeds = {result.workload_seed for result in results}
    data_ledgers = {
        (result.payload_bytes, result.descriptor_bytes, result.alignment_bytes)
        for result in results
    }
    if len(task_counts) != 1 or completed != task_counts:
        raise RICAccountingError("task/full-drain conservation differs across arms")
    if len(stage_counts) != 1 or next(iter(stage_counts))[0] != next(iter(stage_counts))[1]:
        raise RICAccountingError("stage conservation differs across arms")
    if len(join_counts) != 1 or next(iter(join_counts))[0] != next(iter(join_counts))[1]:
        raise RICAccountingError("join-combine conservation differs across arms")
    if (
        len(task_fingerprints) != 1
        or len(service_fingerprints) != 1
        or len(score_mask_fingerprints) != 1
        or len(resource_demand_fingerprints) != 1
        or len(tax_surface_fingerprints) != 1
        or len(workload_seeds) != 1
    ):
        raise RICAccountingError("workload/service fingerprints differ across arms")
    if len(data_ledgers) != 1:
        raise RICAccountingError("contribution byte ledger differs across arms")
    if not all(result.full_drain for result in results):
        raise RICAccountingError("partial arm present in comparison")


def assert_sham_feedback_cost_equivalence(
    charged: ReplayResultLike, sham: ReplayResultLike
) -> None:
    """Sham feedback must pay the exact charged control path."""

    assert_replay_conservation((charged, sham))
    charged_cost = (
        charged.contract_bytes,
        charged.contract_received_bytes,
        charged.contract_header_bytes,
        charged.contract_record_bytes,
        charged.contract_alignment_bytes,
        charged.contract_messages,
        dict(charged.contract_record_count_histogram),
        dict(charged.control_component_us),
        {
            key: value
            for key, value in charged.resource_service_demand_us.items()
            if key.startswith("control:")
        },
        charged.contract_tax_surface_fingerprint,
    )
    sham_cost = (
        sham.contract_bytes,
        sham.contract_received_bytes,
        sham.contract_header_bytes,
        sham.contract_record_bytes,
        sham.contract_alignment_bytes,
        sham.contract_messages,
        dict(sham.contract_record_count_histogram),
        dict(sham.control_component_us),
        {
            key: value
            for key, value in sham.resource_service_demand_us.items()
            if key.startswith("control:")
        },
        sham.contract_tax_surface_fingerprint,
    )
    if charged_cost != sham_cost:
        raise RICAccountingError(
            f"sham/charged control cost mismatch: {charged_cost} != {sham_cost}"
        )
