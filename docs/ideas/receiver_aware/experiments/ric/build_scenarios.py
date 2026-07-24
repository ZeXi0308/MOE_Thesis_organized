#!/usr/bin/env python3
"""Build validated RIC-v1 virtual-EP worlds from route-real artifacts.

This producer is deliberately policy-free.  It binds the frozen data/route,
placement and measured service surface, constructs exogenous micro-coflow
arrivals, and emits the *complete* workload graph consumed by every arm.
It never filters background work using a score mask and never reads outcomes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import statistics
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:  # Package import.
    from .capture_routes_gpu import (
        _producer_source_sha256 as _capture_routes_source_sha256,
        _route_tuple_sha256,
        assigned_layer,
        expert_sender,
        origin_lpt,
        selected_layers,
    )
    from .formal_provenance import (
        EMBEDDED_PRODUCER_SIGNOFF,
        FormalProvenanceError,
        SIGNOFF_ARTIFACT_FIELDS,
        canonical_reviewed_scope_paths,
        is_sha256,
        load_json_mapping_strict,
        loads_json_mapping_strict,
        materialize_verified_signoff,
        resolve_repo_file,
        validate_data_manifest_fields,
        verify_phase4_signoff,
    )
    from .measure_service_lut_gpu import (
        _producer_source_sha256 as _measure_service_lut_source_sha256,
    )
    from .measure_capability_gpu import (
        _producer_source_sha256 as _measure_capability_source_sha256,
    )
    from .prepare_data import (
        _producer_source_sha256 as _prepare_data_source_sha256,
        add_self_hash,
        sha256_file,
        validate_self_hash,
    )
    from .scenario import ReplayTask, ReplayWorld, StageService
    from .schema import (
        ContributionIdentity,
        ContributionRecord,
        RICValidationError,
        validate_full_background,
    )
except ImportError:  # Direct entrypoint/tests from this directory.
    from capture_routes_gpu import (  # type: ignore
        _producer_source_sha256 as _capture_routes_source_sha256,
        _route_tuple_sha256,
        assigned_layer,
        expert_sender,
        origin_lpt,
        selected_layers,
    )
    from formal_provenance import (  # type: ignore
        EMBEDDED_PRODUCER_SIGNOFF,
        FormalProvenanceError,
        SIGNOFF_ARTIFACT_FIELDS,
        canonical_reviewed_scope_paths,
        is_sha256,
        load_json_mapping_strict,
        loads_json_mapping_strict,
        materialize_verified_signoff,
        resolve_repo_file,
        validate_data_manifest_fields,
        verify_phase4_signoff,
    )
    from measure_service_lut_gpu import (  # type: ignore
        _producer_source_sha256 as _measure_service_lut_source_sha256,
    )
    from measure_capability_gpu import (  # type: ignore
        _producer_source_sha256 as _measure_capability_source_sha256,
    )
    from prepare_data import (  # type: ignore
        _producer_source_sha256 as _prepare_data_source_sha256,
        add_self_hash,
        sha256_file,
        validate_self_hash,
    )
    from scenario import ReplayTask, ReplayWorld, StageService  # type: ignore
    from schema import (  # type: ignore
        ContributionIdentity,
        ContributionRecord,
        RICValidationError,
        validate_full_background,
    )


IDEA_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
DEFAULT_CONFIG = IDEA_ROOT / "configs" / "ric_v1.json"
DEFAULT_PROTOCOL = IDEA_ROOT / "RIC_Phase2_冻结实验协议_2026-07-22.md"
DEFAULT_CONSUMER_AMENDMENT = (
    IDEA_ROOT / "RIC_AmendmentQ_ConsumerMigration_2026-07-22.md"
)
HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256 = (
    "15db8b79ea590fa4c4354835c8ba472928433a685c4df82f8ff7c9d2e155a9b8"
)
FORMAL_AUTHORITATIVE_BUNDLE_ROOT = Path(
    "/root/autodl-tmp/ric_formal_v1_20260722"
)
FORMAL_CENSUS_RELATIVE_ROOTS = (
    ".",
)
FORMAL_PREOUTCOME_ATTESTATION_PATH = (
    FORMAL_AUTHORITATIVE_BUNDLE_ROOT
    / "docs/ideas/receiver_aware/formal_signoff/v6/preoutcome_attestation.json"
)
GLOBAL_SEALED_STATE_DIR = (
    FORMAL_AUTHORITATIVE_BUNDLE_ROOT
    / "docs/ideas/receiver_aware/.formal_state/ric_v1_sealed"
)
GLOBAL_SEALED_RESERVATION = GLOBAL_SEALED_STATE_DIR / "reservation.json"
GLOBAL_SEALED_CONSUMPTION = GLOBAL_SEALED_STATE_DIR / "consumption.json"
GLOBAL_SEALED_EVALUATION_CONSUMPTION = (
    GLOBAL_SEALED_STATE_DIR / "evaluation_consumption.json"
)


class ScenarioBuildError(RuntimeError):
    """An input artifact or frozen workload invariant failed."""


def validate_gpu_environment_artifact(
    artifact: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    label: str,
) -> Mapping[str, str]:
    """Validate one formal 5090 environment and return cross-stage keys."""

    environment = artifact.get("gpu_environment")
    probes = config.get("capability_probes")
    if not isinstance(environment, Mapping) or not isinstance(probes, Mapping):
        raise ScenarioBuildError(f"BLOCKED_GPU_ENVIRONMENT: {label} is missing")
    gate = probes.get("formal_gpu_environment_gate")
    paths = probes.get("gpu_environment_fields")
    if not isinstance(gate, Mapping) or not isinstance(paths, list):
        raise ScenarioBuildError("BLOCKED_GPU_ENVIRONMENT: gate is not frozen")
    def artifact_value(path: object) -> Any:
        if not isinstance(path, str) or not path:
            raise ScenarioBuildError(
                "BLOCKED_GPU_ENVIRONMENT: invalid required path"
            )
        if path.startswith("gpu_environment."):
            field = path.removeprefix("gpu_environment.")
            if not field or "." in field:
                raise ScenarioBuildError(
                    "BLOCKED_GPU_ENVIRONMENT: invalid required path"
                )
            return environment.get(field)
        if "." in path:
            raise ScenarioBuildError(
                "BLOCKED_GPU_ENVIRONMENT: invalid artifact-root path"
            )
        return artifact.get(path)

    for path in paths:
        value = artifact_value(path)
        if value is None or value == "":
            raise ScenarioBuildError(f"BLOCKED_GPU_ENVIRONMENT: {label} lacks {path}")
    if environment.get("gpu_name") != gate.get("gpu_name_exact"):
        raise ScenarioBuildError(f"BLOCKED_GPU_ENVIRONMENT: {label} is not RTX 5090")
    producer_pid = environment.get("producer_pid")
    if type(producer_pid) is not int or producer_pid <= 0:
        raise ScenarioBuildError(f"BLOCKED_GPU_ENVIRONMENT: {label} lacks producer PID")
    for field in (
        "clock_sm_mhz",
        "power_draw_w",
        "memory_used_mib",
        "background_gpu_util_percent",
    ):
        value = environment.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ScenarioBuildError(
                f"BLOCKED_GPU_ENVIRONMENT: {label} has invalid {field}"
            )
    gpu_uuid = str(environment["gpu_uuid"])
    for census_field in ("compute_apps_before", "compute_apps_after"):
        census = environment.get(census_field)
        if not isinstance(census, list):
            raise ScenarioBuildError(
                f"BLOCKED_GPU_ENVIRONMENT: {label} lacks {census_field}"
            )
        for row in census:
            if (
                not isinstance(row, Mapping)
                or row.get("pid") != producer_pid
                or row.get("gpu_uuid") != gpu_uuid
                or not isinstance(row.get("process_name"), str)
                or isinstance(row.get("used_memory_mib"), bool)
                or not isinstance(row.get("used_memory_mib"), (int, float))
            ):
                raise ScenarioBuildError(
                    f"BLOCKED_GPU_ENVIRONMENT: {label} has a foreign GPU process"
                )
    signature: dict[str, str] = {}
    for path in gate["same_model_route_capability_lut_fields_must_match"]:
        value = artifact_value(path)
        if value is None or value == "":
            raise ScenarioBuildError(
                f"BLOCKED_GPU_ENVIRONMENT: {label} lacks cross-stage field {path}"
            )
        signature[str(path)] = str(value)
    return signature


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def causal_arrival_fingerprint(world: ReplayWorld) -> str:
    """Bind link-invariant identity, arrivals, and expert-ready offsets."""

    rows = [
        (
            task.identity.canonical_tuple(),
            task.contribution.arrival_us,
            task.contribution.ready_us - task.contribution.arrival_us,
            task.contribution.payload_bytes,
            task.contribution.descriptor_bytes,
            task.contribution.alignment_bytes,
        )
        for task in sorted(world.tasks, key=lambda item: item.task_id)
    ]
    return object_sha256(rows)


def arrival_schedule_fingerprint(world: ReplayWorld) -> str:
    rows = [
        (join.canonical_tuple(), siblings[0].contribution.arrival_us)
        for join, siblings in sorted(world.joins.items())
    ]
    return object_sha256(rows)


def block_permutation_fingerprint(world: ReplayWorld) -> str:
    block_keys = {
        (
            task.identity.request_id,
            task.identity.forward_id,
            task.identity.batch_id,
            task.identity.layer_id,
            task.identity.token_id,
            task.identity.token_block_id,
        )
        for task in world.tasks
    }
    ordered = sorted(
        block_keys,
        key=lambda value: object_sha256((world.workload_seed, value)),
    )
    return object_sha256(ordered)


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        HERE / "scenario.py",
        HERE / "schema.py",
        HERE / "capture_routes_gpu.py",
        HERE / "measure_capability_gpu.py",
        HERE / "measure_service_lut_gpu.py",
        HERE / "capability_contract.py",
        HERE / "wire.py",
        HERE / "prepare_data.py",
        HERE / "formal_provenance.py",
    ):
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _preoutcome_attestation_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        HERE / "capture_preoutcome_attestation.py",
        Path(__file__),
        HERE / "formal_provenance.py",
    ):
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def guard_mode_role(mode: str, role: str) -> None:
    """Reject dev/sealed before any caller opens a sealed artifact."""

    if mode not in {"dev", "formal"} or role not in {"calibration", "sealed"}:
        raise ScenarioBuildError("invalid mode/data role")
    if mode == "dev" and role == "sealed":
        raise ScenarioBuildError("dev mode is forbidden from reading sealed artifacts")


def validate_consumer_amendment_path(path: Path, *, mode: str) -> str:
    """Formal consumers may use only the exact Amendment Q reviewed in scope."""

    try:
        resolved = path.resolve(strict=True)
        frozen = DEFAULT_CONSUMER_AMENDMENT.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ScenarioBuildError("consumer amendment is unavailable") from exc
    if mode == "formal" and resolved != frozen:
        raise ScenarioBuildError(
            "formal consumer amendment path differs from reviewed Amendment Q"
        )
    return sha256_file(resolved)


def validate_frozen_formal_paths(
    *, config_path: Path, protocol_path: Path, mode: str
) -> None:
    """Formal execution cannot substitute unreviewed config/protocol paths."""

    if mode != "formal":
        return
    for label, supplied, frozen in (
        ("config", config_path, DEFAULT_CONFIG),
        ("protocol", protocol_path, DEFAULT_PROTOCOL),
    ):
        try:
            supplied_resolved = supplied.resolve(strict=True)
            frozen_resolved = frozen.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ScenarioBuildError(f"formal {label} is unavailable") from exc
        if supplied_resolved != frozen_resolved:
            raise ScenarioBuildError(
                f"formal {label} path differs from exact reviewed frozen file"
            )


def validate_authoritative_bundle_root(path: Path, *, mode: str) -> Path:
    """Pin this one formal run to the root reviewed before any sealed outcome."""

    absolute = Path(os.path.abspath(path))
    if mode == "formal" and absolute != FORMAL_AUTHORITATIVE_BUNDLE_ROOT:
        raise ScenarioBuildError(
            "formal authoritative bundle root differs from reviewed run root"
        )
    try:
        resolved = absolute.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ScenarioBuildError("authoritative bundle root is unavailable") from exc
    if absolute.is_symlink() or not resolved.is_dir():
        raise ScenarioBuildError("authoritative bundle root is not a real directory")
    return resolved


def validate_formal_output_path(path: Path, *, mode: str) -> Path:
    """Keep every formal scenario/result inside the censused run namespace."""

    resolved = path.resolve(strict=False)
    if mode != "formal":
        return resolved
    output_root = (FORMAL_AUTHORITATIVE_BUNDLE_ROOT / "formal_outputs").resolve(
        strict=True
    )
    try:
        relative = resolved.relative_to(output_root)
    except ValueError as exc:
        raise ScenarioBuildError(
            "formal output path is outside the reviewed formal_outputs root"
        ) from exc
    if not relative.parts:
        raise ScenarioBuildError("formal output path may not replace formal_outputs")
    return resolved


@contextmanager
def atomic_output_directory(path: Path) -> Iterator[Path]:
    """Yield a sibling temporary directory and atomically commit once."""

    if path.exists():
        raise ScenarioBuildError(f"refusing to overwrite output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{path.name}.partial-", dir=path.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        yield temporary
        if path.exists():
            raise ScenarioBuildError("output appeared while producer was running")
        temporary.rename(path)


def _read_self_hashed_json(
    path: Path, *, schema_version: str, hash_field: str = "manifest_sha256"
) -> dict[str, Any]:
    try:
        value = load_json_mapping_strict(path, label=path.name)
    except FormalProvenanceError as exc:
        raise ScenarioBuildError(f"cannot read JSON artifact {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema_version:
        raise ScenarioBuildError(f"schema mismatch for {path.name}")
    try:
        validate_self_hash(value, hash_field)
    except Exception as exc:
        raise ScenarioBuildError(f"self-hash mismatch for {path.name}") from exc
    return value


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = load_json_mapping_strict(path, label="RIC config")
    except FormalProvenanceError as exc:
        raise ScenarioBuildError(str(exc)) from exc
    if value.get("schema_version") != "ric-config-v1":
        raise ScenarioBuildError("RIC config schema mismatch")
    return value


def partition_requests(
    requests: Sequence[Mapping[str, Any]], *, role: str, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], ...]:
    """Outcome-blind, disjoint 4-request partition from manifest rank hashes."""

    expected = int(config["data"][role]["document_count"])
    per_trace = int(config["workloads"]["requests_per_complete_trace"])
    expected_traces = (
        int(config["workloads"]["calibration_complete_trace_clusters"])
        if role == "calibration"
        else int(config["workloads"]["complete_trace_clusters_per_model_cell"])
    )
    if len(requests) != expected or expected != per_trace * expected_traces:
        raise ScenarioBuildError("manifest request count cannot form frozen traces")
    ranked: list[tuple[str, str, str]] = []
    for request in requests:
        request_id = str(request.get("request_id", ""))
        text_hash = str(request.get("text_sha256", ""))
        rank_hash = str(request.get("rank_sha256", ""))
        if not request_id or len(text_hash) != 64 or len(rank_hash) != 64:
            raise ScenarioBuildError("request lacks frozen id/text/rank hash")
        ranked.append((rank_hash, request_id, text_hash))
    if len({row[1] for row in ranked}) != expected:
        raise ScenarioBuildError("request id reused within split")
    if len({row[2] for row in ranked}) != expected:
        raise ScenarioBuildError("request text reused within split")
    ordered = [request_id for _rank, request_id, _text in sorted(ranked)]
    groups = tuple(
        tuple(ordered[index : index + per_trace])
        for index in range(0, len(ordered), per_trace)
    )
    if len({request for group in groups for request in group}) != expected:
        raise ScenarioBuildError("request partition is not disjoint")
    return groups


@dataclass(frozen=True)
class ArrivalRealization:
    arrivals_us: tuple[float, ...]
    arrival_states: tuple[str, ...]
    state_transitions_us: tuple[tuple[float, str, str], ...]
    mu_per_us: float
    target_rho: float
    realized_offered_rho: float


ARRIVAL_NORMALIZATION_ALGORITHM = (
    "ric-v1-role-cell-dimensionless-common-time-dilation"
)


def validate_load_normalization_contract(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require the exact Amendment-N workload contract before construction."""

    workloads = config.get("workloads")
    topology = config.get("topology_proxy")
    if not isinstance(workloads, Mapping) or not isinstance(topology, Mapping):
        raise ScenarioBuildError("finite-horizon normalization config is missing")
    primary_link = topology.get("primary_link_gbps")
    expected = {
        "claim_label": "exact_load_time_normalized_process_shaped_replay",
        "scope": "one_common_scalar_per_role_and_cell",
        "applies_to_all_main_and_negative_control_cells": True,
        "raw_schedule_space": "dimensionless_mu_equals_1",
        "raw_schedule_count": 512,
        "factor_formula": (
            "mean_i(raw_count/raw_last_arrival_i)/target_utilization"
        ),
        "normalized_arrival_formula": (
            "raw_dimensionless_arrival*factor/model_reference_mu"
        ),
        "normalized_transition_formula": (
            "raw_dimensionless_transition*factor/model_reference_mu"
        ),
        "factor_inputs": [
            "role",
            "cell",
            "sorted_frozen_base_seeds",
            "raw_dimensionless_arrivals",
            "target_utilization",
        ],
        "forbidden_factor_inputs": [
            "model",
            "route",
            "link_variant",
            "arm",
            "policy",
            "oracle",
            "latency",
            "quality",
            "experiment_outcome",
        ],
        "scale_arrivals_and_ctmc_transitions_together": True,
        "normalized_arithmetic_aggregate_abs_tolerance": 1e-12,
        "raw_aggregate_is_diagnostic_not_gate": True,
        "primary_reference_link_gbps": 200,
        "sensitivity_reuses_normalized_schedule_without_rng_or_renormalization": True,
        "metadata_requires_raw_and_normalized_fingerprints": True,
        "no_policy_or_oracle_input": True,
    }
    contract = workloads.get("finite_horizon_load_normalization")
    if (
        contract != expected
        or primary_link != expected["primary_reference_link_gbps"]
        or workloads.get("token_blocks_per_complete_trace")
        != expected["raw_schedule_count"]
        or workloads.get("base_seed_semantics")
        != "literal_python_random_seed"
        or workloads.get("derivation_salt_semantics")
        != "provenance_namespace_only_not_prng_input"
    ):
        raise ScenarioBuildError(
            "finite-horizon load normalization contract is not exact Amendment N"
        )
    return contract


@dataclass(frozen=True)
class NormalizedArrivalTrace:
    """One seed's raw and normalized dimensionless exogenous schedule."""

    seed: int
    raw: ArrivalRealization
    normalized: ArrivalRealization
    raw_fingerprint: str
    normalized_fingerprint: str


@dataclass(frozen=True)
class ArrivalNormalizationCensus:
    """Outcome-blind normalization shared by every model/link in one role/cell."""

    role: str
    cell_name: str
    cell_fingerprint: str
    seed_namespace_label: str
    seeds: tuple[int, ...]
    count_per_trace: int
    target_rho: float
    time_dilation_factor: float
    raw_aggregate_rho: float
    normalized_aggregate_rho: float
    raw_census_fingerprint: str
    normalized_census_fingerprint: str
    traces: tuple[NormalizedArrivalTrace, ...]
    algorithm: str = ARRIVAL_NORMALIZATION_ALGORITHM
    no_policy_or_oracle_input: bool = True

    def trace_for_seed(self, seed: int) -> NormalizedArrivalTrace:
        found = [trace for trace in self.traces if trace.seed == seed]
        if len(found) != 1:
            raise ScenarioBuildError("arrival census lacks exactly one requested seed")
        return found[0]


def _validate_arrival_cell_schema(cell: Mapping[str, Any]) -> None:
    process = cell.get("arrival_process")
    poisson_fields = {"arrival_process", "target_utilization"}
    mmpp_fields = {
        "arrival_process",
        "target_utilization",
        "lambda_low_over_mu",
        "lambda_high_over_mu",
        "mean_high_dwell_service_units",
        "mean_low_dwell_service_units",
    }
    expected = (
        poisson_fields
        if process == "poisson"
        else mmpp_fields
        if process == "continuous_time_two_state_mmpp"
        else None
    )
    if expected is None or set(cell) != expected:
        raise ScenarioBuildError("arrival cell schema is not exact")
    target = cell.get("target_utilization")
    if (
        isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not math.isfinite(float(target))
        or not 0.0 < float(target) < 1.0
    ):
        raise ScenarioBuildError("arrival target utilization is invalid")
    if process == "continuous_time_two_state_mmpp":
        numeric = {
            name: cell.get(name)
            for name in mmpp_fields
            if name not in {"arrival_process", "target_utilization"}
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in numeric.values()
        ):
            raise ScenarioBuildError("MMPP rates and dwell times must be positive")
        low = float(cell["lambda_low_over_mu"])
        high = float(cell["lambda_high_over_mu"])
        mean_low = float(cell["mean_low_dwell_service_units"])
        mean_high = float(cell["mean_high_dwell_service_units"])
        if low >= high:
            raise ScenarioBuildError("MMPP low rate must be below high rate")
        low_to_high = 1.0 / mean_low
        high_to_low = 1.0 / mean_high
        stationary_high = low_to_high / (low_to_high + high_to_low)
        stationary_rho = (1.0 - stationary_high) * low + stationary_high * high
        if not math.isclose(
            stationary_rho, float(target), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ScenarioBuildError("MMPP stationary intensity does not equal target")


def _arrival_trace_fingerprint(realization: ArrivalRealization) -> str:
    return object_sha256(
        {
            "arrivals": list(realization.arrivals_us),
            "arrival_states": list(realization.arrival_states),
            "state_transitions": [list(row) for row in realization.state_transitions_us],
            "mu_per_us": realization.mu_per_us,
            "target_rho": realization.target_rho,
            "realized_offered_rho": realization.realized_offered_rho,
        }
    )


def generate_arrivals(
    *,
    cell: Mapping[str, Any],
    count: int,
    mu_per_us: float,
    bottleneck_work_us: float,
    seed: int,
) -> ArrivalRealization:
    """Generate Poisson or true continuous-time two-state MMPP arrivals."""

    _validate_arrival_cell_schema(cell)
    if (
        type(count) is not int
        or count <= 0
        or isinstance(mu_per_us, bool)
        or not isinstance(mu_per_us, (int, float))
        or not math.isfinite(float(mu_per_us))
        or float(mu_per_us) <= 0.0
        or isinstance(bottleneck_work_us, bool)
        or not isinstance(bottleneck_work_us, (int, float))
        or not math.isfinite(float(bottleneck_work_us))
        or float(bottleneck_work_us) <= 0.0
        or type(seed) is not int
    ):
        raise ScenarioBuildError("invalid arrival calibration inputs")
    target = float(cell["target_utilization"])
    process = str(cell["arrival_process"])
    rng = random.Random(seed)
    arrivals: list[float] = []
    states: list[str] = []
    transitions: list[tuple[float, str, str]] = []
    now = 0.0
    if process == "poisson":
        rate = target * mu_per_us
        for _ in range(count):
            now += rng.expovariate(rate)
            arrivals.append(now)
            states.append("poisson")
    elif process == "continuous_time_two_state_mmpp":
        low_rate = float(cell["lambda_low_over_mu"]) * mu_per_us
        high_rate = float(cell["lambda_high_over_mu"]) * mu_per_us
        low_to_high = mu_per_us / float(cell["mean_low_dwell_service_units"])
        high_to_low = mu_per_us / float(cell["mean_high_dwell_service_units"])
        stationary_high = low_to_high / (low_to_high + high_to_low)
        state = "high" if rng.random() < stationary_high else "low"
        while len(arrivals) < count:
            arrival_rate = high_rate if state == "high" else low_rate
            transition_rate = high_to_low if state == "high" else low_to_high
            arrival_wait = rng.expovariate(arrival_rate)
            transition_wait = rng.expovariate(transition_rate)
            if transition_wait < arrival_wait:
                previous = state
                now += transition_wait
                state = "low" if state == "high" else "high"
                transitions.append((now, previous, state))
            else:
                now += arrival_wait
                arrivals.append(now)
                states.append(state)
    else:
        raise ScenarioBuildError(f"unsupported arrival process {process!r}")
    if len(arrivals) != count or any(
        right <= left for left, right in zip(arrivals, arrivals[1:])
    ):
        raise ScenarioBuildError("arrival process is incomplete or non-monotonic")
    realized = bottleneck_work_us / arrivals[-1]
    return ArrivalRealization(
        arrivals_us=tuple(arrivals),
        arrival_states=tuple(states),
        state_transitions_us=tuple(transitions),
        mu_per_us=mu_per_us,
        target_rho=target,
        realized_offered_rho=realized,
    )


def build_arrival_normalization_census(
    *,
    role: str,
    cell_name: str,
    cell: Mapping[str, Any],
    seeds: Sequence[int],
    seed_namespace_label: str,
    count_per_trace: int,
) -> ArrivalNormalizationCensus:
    """Build a policy-free role/cell census and one common time-dilation factor.

    The namespace is provenance only.  The exact bare integer seeds are passed to
    ``random.Random``; no hidden derivation or outcome-dependent resampling occurs.
    """

    _validate_arrival_cell_schema(cell)
    if role not in {"calibration", "sealed"}:
        raise ScenarioBuildError("arrival census role is invalid")
    if not isinstance(cell_name, str) or not cell_name:
        raise ScenarioBuildError("arrival census cell name is invalid")
    if not isinstance(seed_namespace_label, str) or not seed_namespace_label:
        raise ScenarioBuildError("arrival seed namespace label is invalid")
    if type(count_per_trace) is not int or count_per_trace <= 0:
        raise ScenarioBuildError("arrival census count is invalid")
    seed_tuple = tuple(seeds)
    if (
        not seed_tuple
        or any(type(seed) is not int for seed in seed_tuple)
        or len(set(seed_tuple)) != len(seed_tuple)
        or tuple(sorted(seed_tuple)) != seed_tuple
    ):
        raise ScenarioBuildError("arrival census seeds must be unique sorted integers")

    target = float(cell["target_utilization"])
    raw_rows: list[tuple[int, ArrivalRealization, str]] = []
    for seed in seed_tuple:
        raw = generate_arrivals(
            cell=cell,
            count=count_per_trace,
            mu_per_us=1.0,
            bottleneck_work_us=float(count_per_trace),
            seed=seed,
        )
        raw_rows.append((seed, raw, _arrival_trace_fingerprint(raw)))
    raw_aggregate = statistics.fmean(row[1].realized_offered_rho for row in raw_rows)
    factor = raw_aggregate / target
    if not math.isfinite(factor) or factor <= 0.0:
        raise ScenarioBuildError("arrival normalization factor is invalid")

    traces: list[NormalizedArrivalTrace] = []
    for seed, raw, raw_fingerprint in raw_rows:
        normalized = ArrivalRealization(
            arrivals_us=tuple(value * factor for value in raw.arrivals_us),
            arrival_states=raw.arrival_states,
            state_transitions_us=tuple(
                (time_us * factor, old, new)
                for time_us, old, new in raw.state_transitions_us
            ),
            mu_per_us=1.0,
            target_rho=target,
            realized_offered_rho=raw.realized_offered_rho / factor,
        )
        if (
            normalized.arrival_states != raw.arrival_states
            or any(
                right <= left
                for left, right in zip(
                    normalized.arrivals_us, normalized.arrivals_us[1:]
                )
            )
        ):
            raise ScenarioBuildError("arrival normalization changed state/order")
        traces.append(
            NormalizedArrivalTrace(
                seed=seed,
                raw=raw,
                normalized=normalized,
                raw_fingerprint=raw_fingerprint,
                normalized_fingerprint=_arrival_trace_fingerprint(normalized),
            )
        )
    normalized_aggregate = statistics.fmean(
        trace.normalized.realized_offered_rho for trace in traces
    )
    if not math.isclose(
        normalized_aggregate, target, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ScenarioBuildError("arrival normalization did not hit target exactly")

    cell_fingerprint = object_sha256(cell)
    raw_census_fingerprint = object_sha256(
        {
            "algorithm": ARRIVAL_NORMALIZATION_ALGORITHM,
            "role": role,
            "cell_name": cell_name,
            "cell_fingerprint": cell_fingerprint,
            "seed_namespace_label": seed_namespace_label,
            "seed_namespace_affects_prng": False,
            "seeds": list(seed_tuple),
            "count_per_trace": count_per_trace,
            "trace_fingerprints": [trace.raw_fingerprint for trace in traces],
        }
    )
    normalized_census_fingerprint = object_sha256(
        {
            "raw_census_fingerprint": raw_census_fingerprint,
            "time_dilation_factor": factor,
            "trace_fingerprints": [
                trace.normalized_fingerprint for trace in traces
            ],
            "no_policy_or_oracle_input": True,
        }
    )
    return ArrivalNormalizationCensus(
        role=role,
        cell_name=cell_name,
        cell_fingerprint=cell_fingerprint,
        seed_namespace_label=seed_namespace_label,
        seeds=seed_tuple,
        count_per_trace=count_per_trace,
        target_rho=target,
        time_dilation_factor=factor,
        raw_aggregate_rho=raw_aggregate,
        normalized_aggregate_rho=normalized_aggregate,
        raw_census_fingerprint=raw_census_fingerprint,
        normalized_census_fingerprint=normalized_census_fingerprint,
        traces=tuple(traces),
    )


@dataclass(frozen=True)
class ServiceSurface:
    model_key: str
    model_revision: str
    model_tree_manifest_sha256: str
    payload_bytes: int
    payload_layout_sha256: str
    expert_ready_us_by_layer_expert: Mapping[str, float]
    batching_diagnostic_expert_ready_us: float
    sender_pack_us: float
    receiver_unpack_us: float
    join_combine_us: float
    control_tax_by_record_count: Mapping[str, Mapping[str, float]]
    control_tax_source_id: str
    metadata_sha256: str
    summary_sha256: str
    raw_sha256: str
    producer_source_sha256: str
    producer_signoff_sha256: str | None
    data_producer_signoff_sha256: str | None = None
    gpu_environment_identity: Mapping[str, str] | None = None


def _single_lut_row(
    rows: Sequence[Mapping[str, str]],
    *,
    component: str,
    layer_id: int | None = None,
    expert_id: int | None = None,
    row_count: int = 0,
    record_count: int = 0,
) -> Mapping[str, str]:
    found = [
        row
        for row in rows
        if row.get("component") == component
        and (layer_id is None or int(row.get("layer_id", "-1")) == layer_id)
        and (expert_id is None or int(row.get("expert_id", "-1")) == expert_id)
        and int(row.get("rows", "0")) == row_count
        and int(row.get("record_count", "0")) == record_count
    ]
    if len(found) != 1:
        raise ScenarioBuildError(
            f"service LUT requires exactly one {component}/{row_count}/{record_count} row"
        )
    return found[0]


def _lut_group_key(row: Mapping[str, str]) -> tuple[str, int, int, int, int, str]:
    return (
        str(row.get("component", "")),
        int(row.get("layer_id", "-1")),
        int(row.get("expert_id", "-1")),
        int(row.get("rows", "0")),
        int(row.get("record_count", "0")),
        str(row.get("source", "")),
    )


def _verify_measured_summary(
    summary_row: Mapping[str, str],
    raw_index: Mapping[tuple[str, int, int, int, int, str], Sequence[float]],
) -> None:
    key = _lut_group_key(summary_row)
    values = raw_index.get(key)
    if not values:
        raise ScenarioBuildError(f"summary row lacks raw repeats: {key}")
    median = float(statistics.median(values))
    if not math.isclose(
        median, float(summary_row["median_us"]), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ScenarioBuildError(f"summary median cannot be recomputed from raw repeats: {key}")
    if int(summary_row["trial_count"]) != len(values):
        raise ScenarioBuildError(f"summary/raw trial count mismatch: {key}")


def load_service_surface(
    service_lut_dir: Path,
    *,
    model_key: str,
    model_revision: str,
    config_sha256: str,
    protocol_sha256: str,
    mode: str,
    expected_selected_layers: Sequence[int],
    expected_num_experts: int,
    config: Mapping[str, Any] | None = None,
    expected_data_manifest_sha256: str | None = None,
    expected_data_producer_signoff_sha256: str | None = None,
    historical_reviewed_source_snapshot_path: Path | None = None,
    expected_producer_signoff_file_sha256: str | None = None,
) -> ServiceSurface:
    metadata = _read_self_hashed_json(
        service_lut_dir / "service_lut_metadata.json",
        schema_version="ric-service-lut-v1",
    )
    summary_path = service_lut_dir / "service_lut.csv"
    raw_path = service_lut_dir / "service_lut_raw.csv"
    expected = {
        "model_key": model_key,
        "model_revision": model_revision,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "service_lut_sha256": sha256_file(summary_path),
        "service_lut_raw_sha256": sha256_file(raw_path),
        "measure_service_lut_source_sha256": _measure_service_lut_source_sha256(),
    }
    for field, wanted in expected.items():
        if metadata.get(field) != wanted:
            raise ScenarioBuildError(f"service LUT binding mismatch: {field}")
    if expected_data_manifest_sha256 is not None and metadata.get(
        "data_manifest_sha256"
    ) != expected_data_manifest_sha256:
        raise ScenarioBuildError("service LUT data manifest binding mismatch")
    if metadata.get("data_producer_signoff_sha256") != (
        expected_data_producer_signoff_sha256
    ):
        raise ScenarioBuildError("service LUT data producer signoff binding mismatch")
    model_tree_sha = metadata.get("model_tree_manifest_sha256")
    if not isinstance(model_tree_sha, str) or len(model_tree_sha) != 64:
        raise ScenarioBuildError("service LUT lacks hashed local model tree")
    if mode == "formal" and metadata.get("status") != "LUT_ONLY":
        raise ScenarioBuildError("formal scenario requires formal LUT_ONLY artifact")
    if mode == "dev" and metadata.get("status") != "NOT_TESTED":
        raise ScenarioBuildError("dev scenario accepts only NOT_TESTED LUT artifact")
    producer_signoff_sha256 = metadata.get("signoff_sha256")
    if mode == "formal":
        if historical_reviewed_source_snapshot_path is None:
            raise ScenarioBuildError(
                "formal service LUT requires historical reviewed-source snapshot"
            )
        if not is_sha256(expected_producer_signoff_file_sha256):
            raise ScenarioBuildError(
                "formal service LUT signoff is not pre-outcome registered"
            )
        embedded_signoff = service_lut_dir / EMBEDDED_PRODUCER_SIGNOFF
        if (
            not is_sha256(producer_signoff_sha256)
            or not embedded_signoff.is_file()
            or sha256_file(embedded_signoff) != producer_signoff_sha256
        ):
            raise ScenarioBuildError("formal service LUT lacks embedded producer signoff")
        try:
            verify_immutable_upstream_signoff(
                embedded_signoff,
                snapshot_path=historical_reviewed_source_snapshot_path,
                expected_signoff_file_sha256=str(
                    expected_producer_signoff_file_sha256
                ),
                expected_fields={
                    "stage": "measure_service_lut",
                    "protocol_sha256": protocol_sha256,
                    "config_sha256": config_sha256,
                    "measure_service_lut_source_sha256": (
                        _measure_service_lut_source_sha256()
                    ),
                    "measure_capability_source_sha256": (
                        _measure_capability_source_sha256()
                    ),
                    "capture_routes_source_sha256": (
                        _capture_routes_source_sha256()
                    ),
                    "prepare_data_source_sha256": (
                        _prepare_data_source_sha256()
                    ),
                    "data_manifest_sha256": str(
                        metadata.get("data_manifest_sha256")
                    ),
                    "data_producer_signoff_sha256": str(
                        metadata.get("data_producer_signoff_sha256")
                    ),
                    "model_key": model_key,
                    "model_tree_manifest_sha256": str(
                        metadata.get("model_tree_manifest_sha256")
                    ),
                },
            )
        except Exception as exc:
            raise ScenarioBuildError("service LUT producer signoff is invalid") from exc
    elif producer_signoff_sha256 is not None:
        raise ScenarioBuildError("development service LUT claims a producer signoff")
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with raw_path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not rows:
        raise ScenarioBuildError("empty service LUT")
    if not raw_rows:
        raise ScenarioBuildError("empty raw service LUT")
    raw_index: dict[tuple[str, int, int, int, int, str], list[float]] = {}
    for row in raw_rows:
        try:
            elapsed = float(row["us"])
        except (KeyError, ValueError) as exc:
            raise ScenarioBuildError("invalid raw service repeat") from exc
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ScenarioBuildError("raw service repeat is non-finite/negative")
        raw_index.setdefault(_lut_group_key(row), []).append(elapsed)
    descriptors = {
        (
            row.get("payload_dtype"),
            row.get("payload_elements_per_row"),
            row.get("payload_element_size_bytes"),
            row.get("payload_bytes_per_contribution_row"),
            row.get("payload_layout_sha256"),
        )
        for row in rows
    }
    if len(descriptors) != 1:
        raise ScenarioBuildError("service LUT payload descriptor drift")
    dtype, elements, element_size, payload_bytes, layout_sha = next(iter(descriptors))
    descriptor = {
        "payload_dtype": dtype,
        "payload_elements_per_row": int(str(elements)),
        "payload_element_size_bytes": int(str(element_size)),
        "payload_bytes_per_contribution_row": int(str(payload_bytes)),
    }
    if object_sha256(descriptor) != layout_sha:
        raise ScenarioBuildError("payload layout hash mismatch")
    for field, wanted in {**descriptor, "payload_layout_sha256": layout_sha}.items():
        if metadata.get(field) != wanted:
            raise ScenarioBuildError(f"service LUT metadata payload mismatch: {field}")
    ready = _single_lut_row(
        rows,
        component="expert_execution_conservative_max_selected_median",
        row_count=1,
    )
    selected_layers = [int(value) for value in expected_selected_layers]
    expert_ids = list(range(expected_num_experts))
    if (
        metadata.get("route_specific_selected_layers") != selected_layers
        or metadata.get("route_specific_expert_ids") != expert_ids
        or metadata.get("route_specific_key_count")
        != len(selected_layers) * expected_num_experts
        or metadata.get("route_specific_main_component")
        != "expert_execution_route_specific_row1"
    ):
        raise ScenarioBuildError("route-specific expert service coverage mismatch")
    route_specific: dict[str, float] = {}
    for layer_id in selected_layers:
        for expert_id in expert_ids:
            row = _single_lut_row(
                rows,
                component="expert_execution_route_specific_row1",
                layer_id=layer_id,
                expert_id=expert_id,
                row_count=1,
            )
            _verify_measured_summary(row, raw_index)
            value = float(row["median_us"])
            if row.get("source") != "measured_5090_cuda" or not math.isfinite(
                value
            ) or value <= 0.0:
                raise ScenarioBuildError("invalid route-specific expert service row")
            route_specific[f"{layer_id}:{expert_id}"] = value
    sender = _single_lut_row(rows, component="sender_pack", row_count=1)
    receiver = _single_lut_row(rows, component="receiver_unpack", row_count=1)
    canonical_diagnostic = _single_lut_row(
        rows, component="canonical_reduction", row_count=1
    )
    expert_rows = [
        row
        for row in rows
        if row.get("component") == "expert_execution"
        and int(row.get("rows", "0")) == 1
    ]
    if not expert_rows:
        raise ScenarioBuildError("service LUT lacks measured selected experts")
    for row in (*expert_rows, sender, receiver, canonical_diagnostic):
        _verify_measured_summary(row, raw_index)
    if not math.isclose(
        float(ready["median_us"]),
        max(float(row["median_us"]) for row in expert_rows),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ScenarioBuildError("conservative expert-ready row is not max selected median")
    if ready.get("source") != "measured_5090_cuda":
        raise ScenarioBuildError("expert-ready LUT source is not measured_5090_cuda")
    if (
        sender.get("source") != "measured_5090_cuda"
        or receiver.get("source") != "measured_5090_cuda"
        or canonical_diagnostic.get("source") != "measured_5090_cuda"
    ):
        raise ScenarioBuildError("pack/unpack/reduction LUT source is not measured_5090_cuda")
    data_path_accounting = metadata.get("data_path_measurement_accounting")
    if not isinstance(data_path_accounting, Mapping) or data_path_accounting != {
        "per_contribution_additive_components": [
            "expert_execution_route_specific_row1",
            "sender_pack",
            "receiver_unpack",
        ],
        "per_join_once_only_component": "canonical_reduction",
        "canonical_reduction_charged_once_per_join": True,
    }:
        raise ScenarioBuildError("service LUT data-path accounting contract mismatch")
    additive = (
        ("state_build_us", "state_build_contract_record"),
        ("hash_us", "host_hash_identity"),
        ("encode_us", "host_encode_contract"),
        ("decode_us", "host_decode_contract"),
        ("lookup_us", "collision_checked_identity_lookup"),
        ("apply_us", "epoch_sequence_apply"),
        ("policy_lookup_us", "sender_policy_cache_lookup"),
    )
    accounting = metadata.get("host_measurement_accounting")
    if not isinstance(accounting, Mapping) or accounting.get("additive_components") != [
        component for _field, component in additive
    ]:
        raise ScenarioBuildError("service LUT additive control component list mismatch")
    if accounting.get("end_to_end_diagnostic_not_additive") != "host_apply_wire_contract":
        raise ScenarioBuildError("service LUT does not exclude end-to-end diagnostic")
    if metadata.get("record_count_grid") != list(range(1, 256)):
        raise ScenarioBuildError("formal control tax requires exact record counts 1..255")
    control_surface: dict[str, Mapping[str, float]] = {}
    for count in range(1, 256):
        harness = _single_lut_row(
            rows, component="host_empty_harness", record_count=count
        )
        _verify_measured_summary(harness, raw_index)
        if harness.get("source") != "measured_5090_host":
            raise ScenarioBuildError("control harness source mismatch")
        harness_us = float(harness["median_us"])
        point: dict[str, float] = {}
        for field, component in additive:
            row = _single_lut_row(rows, component=component, record_count=count)
            _verify_measured_summary(row, raw_index)
            if row.get("source") != "measured_5090_host":
                raise ScenarioBuildError(f"control component source mismatch: {component}")
            net = float(row["median_us"]) - harness_us
            if not math.isfinite(net) or net <= 0:
                raise ScenarioBuildError(
                    f"BLOCKED_CONTROL_TAX_MEASUREMENT: {component}/count={count}"
                )
            point[field] = net
        transfer = _single_lut_row(
            rows,
            component="contract_transfer_analytic_primary_link",
            record_count=count,
        )
        transfer_us = float(transfer["median_us"])
        if transfer.get("source") != "analytic_network" or transfer_us <= 0:
            raise ScenarioBuildError("invalid analytic contract transfer surface")
        network = metadata.get("contract_network_accounting")
        if not isinstance(network, Mapping):
            raise ScenarioBuildError("service LUT lacks analytic network accounting")
        message_bytes = int(transfer.get("contract_message_bytes", "0"))
        expected_transfer = message_bytes * 8.0 / (
            float(network["primary_link_gbps"]) * 1000.0
        )
        if not math.isclose(
            transfer_us, expected_transfer, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ScenarioBuildError("analytic transfer row cannot be recomputed")
        point["transfer_us"] = transfer_us
        control_surface[str(count)] = point
    control_source_id = object_sha256(
        {
            "rule": "raw_median_minus_same_count_empty_harness",
            "non_grid_rule": "exact_1_to_255_no_interpolation_or_extrapolation",
            "points": control_surface,
            "service_lut_sha256": metadata["service_lut_sha256"],
        }
    )
    values = (
        float(ready["median_us"]),
        float(sender["median_us"]),
        float(receiver["median_us"]),
        float(canonical_diagnostic["median_us"]),
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ScenarioBuildError("non-positive measured service surface")
    gpu_identity = None
    if mode == "formal":
        if config is None:
            raise ScenarioBuildError("formal service LUT validation requires config")
        gpu_identity = validate_gpu_environment_artifact(
            metadata, config=config, label="service LUT"
        )
    return ServiceSurface(
        model_key=model_key,
        model_revision=model_revision,
        model_tree_manifest_sha256=model_tree_sha,
        payload_bytes=int(descriptor["payload_bytes_per_contribution_row"]),
        payload_layout_sha256=str(layout_sha),
        expert_ready_us_by_layer_expert=route_specific,
        batching_diagnostic_expert_ready_us=values[0],
        sender_pack_us=values[1],
        receiver_unpack_us=values[2],
        join_combine_us=values[3],
        control_tax_by_record_count=control_surface,
        control_tax_source_id=control_source_id,
        metadata_sha256=str(metadata["manifest_sha256"]),
        summary_sha256=str(metadata["service_lut_sha256"]),
        raw_sha256=str(metadata["service_lut_raw_sha256"]),
        producer_source_sha256=str(metadata["measure_service_lut_source_sha256"]),
        producer_signoff_sha256=(
            str(producer_signoff_sha256)
            if producer_signoff_sha256 is not None
            else None
        ),
        data_producer_signoff_sha256=(
            str(metadata.get("data_producer_signoff_sha256"))
            if metadata.get("data_producer_signoff_sha256") is not None
            else None
        ),
        gpu_environment_identity=gpu_identity,
    )


@dataclass(frozen=True)
class ValidatedInputs:
    role: str
    model_key: str
    model_revision: str
    top_k: int
    num_experts: int
    data_manifest: Mapping[str, Any]
    route_rows_by_request: Mapping[str, tuple[Mapping[str, Any], ...]]
    placement: Mapping[str, Any]
    route_metadata: Mapping[str, Any]
    service: ServiceSurface
    service_calibration_data_manifest_sha256: str
    service_calibration_data_manifest_file_sha256: str
    service_calibration_selected_list_sha256: str
    service_calibration_data_producer_signoff_sha256: str | None
    sealed_input_attestation_sha256: str | None
    sealed_input_historical_run_experiment_source_sha256: str | None
    sealed_input_historical_calibration_lock_sha256: str | None
    sealed_input_historical_calibration_signoff_sha256: str | None
    sealed_global_reservation_file_sha256: str | None
    sealed_global_consumption_file_sha256: str | None
    consumer_amendment_sha256: str
    historical_reviewed_source_snapshot_sha256: str
    pre_outcome_attestation_sha256: str
    pre_outcome_producer_signoff_file_sha256: str
    pre_outcome_producer_signoff_self_hash: str
    authoritative_bundle_root: str
    immutable_input_compatibility_sha256: str


def _snapshot_sources(snapshot_path: Path) -> Mapping[str, bytes]:
    """Read the fixed v5 reviewed-source export without trusting tar paths."""

    try:
        if sha256_file(snapshot_path) != HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256:
            raise ScenarioBuildError("historical reviewed-source snapshot hash mismatch")
        result: dict[str, bytes] = {}
        with tarfile.open(snapshot_path, mode="r:*") as archive:
            for member in archive.getmembers():
                raw = member.name
                path = PurePosixPath(raw)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or not path.parts
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or path.as_posix() != raw
                    or raw in result
                ):
                    raise ScenarioBuildError(
                        "historical reviewed-source snapshot has unsafe members"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ScenarioBuildError(
                        "historical reviewed-source snapshot member is unreadable"
                    )
                result[raw] = extracted.read()
    except (OSError, tarfile.TarError) as exc:
        raise ScenarioBuildError(
            "historical reviewed-source snapshot cannot be verified"
        ) from exc
    if not result:
        raise ScenarioBuildError("historical reviewed-source snapshot is empty")
    return result


def verify_immutable_upstream_signoff(
    signoff_path: Path,
    *,
    snapshot_path: Path,
    expected_fields: Mapping[str, Any],
    expected_signoff_file_sha256: str,
) -> Mapping[str, Any]:
    """Re-run the complete historical Phase-4 chain in an isolated export.

    The producer remains historical.  The current consumer accepts its bytes
    only after the original source manifest, full reviewed scope, reports and
    git-head artifact validate against the fixed v5 source snapshot.
    """

    if (
        not is_sha256(expected_signoff_file_sha256)
        or sha256_file(signoff_path) != expected_signoff_file_sha256
    ):
        raise ScenarioBuildError(
            "immutable upstream signoff is not the pre-outcome registered file"
        )
    try:
        signoff = load_json_mapping_strict(
            signoff_path, label="immutable upstream producer signoff"
        )
        validate_self_hash(signoff, "signoff_sha256")
    except Exception as exc:
        raise ScenarioBuildError("immutable upstream signoff is malformed") from exc
    if (
        signoff.get("schema_version") != "ric-phase4-signoff-v1"
        or signoff.get("status") != "SIGNED-OFF"
        or signoff.get("open_p0") != 0
    ):
        raise ScenarioBuildError("immutable upstream signoff is not signed off")
    for field, wanted in expected_fields.items():
        if signoff.get(field) != wanted:
            raise ScenarioBuildError(
                f"immutable upstream signoff mismatch: {field}"
            )

    snapshot = _snapshot_sources(snapshot_path)
    artifact_bytes: dict[str, bytes] = {}
    for field in SIGNOFF_ARTIFACT_FIELDS:
        reference = signoff.get(field)
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"path", "sha256"}
            or not is_sha256(reference.get("sha256"))
        ):
            raise ScenarioBuildError(
                f"immutable upstream signoff artifact reference is invalid: {field}"
            )
        try:
            artifact_path = resolve_repo_file(REPO_ROOT, reference.get("path"))
            payload = artifact_path.read_bytes()
        except (OSError, FormalProvenanceError) as exc:
            raise ScenarioBuildError(
                f"immutable upstream evidence is unavailable: {field}"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
            raise ScenarioBuildError(
                f"immutable upstream evidence hash mismatch: {field}"
            )
        artifact_bytes[str(reference["path"])] = payload

    try:
        source_reference = signoff["source_manifest"]
        scope_reference = signoff["reviewed_patch"]
        source_manifest = loads_json_mapping_strict(
            artifact_bytes[str(source_reference["path"])].decode("utf-8"),
            label="historical source manifest",
        )
        reviewed_scope = loads_json_mapping_strict(
            artifact_bytes[str(scope_reference["path"])].decode("utf-8"),
            label="historical reviewed scope",
        )
        validate_self_hash(source_manifest)
        validate_self_hash(reviewed_scope, "scope_sha256")
    except Exception as exc:
        raise ScenarioBuildError(
            "immutable upstream source/review manifests are invalid"
        ) from exc
    source_rows = source_manifest.get("sources")
    scope_rows = reviewed_scope.get("sources")
    if not isinstance(source_rows, list) or not isinstance(scope_rows, list):
        raise ScenarioBuildError("immutable upstream source/review rows are missing")
    scope_hashes: dict[str, str] = {}
    for row in scope_rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or not isinstance(row.get("path"), str)
            or not is_sha256(row.get("sha256"))
            or row["path"] in scope_hashes
        ):
            raise ScenarioBuildError("historical reviewed scope row is invalid")
        scope_hashes[str(row["path"])] = str(row["sha256"])
    if set(snapshot) != set(scope_hashes):
        raise ScenarioBuildError(
            "historical snapshot is not the exact reviewed source universe"
        )
    if any(
        hashlib.sha256(snapshot[path]).hexdigest() != wanted
        for path, wanted in scope_hashes.items()
    ):
        raise ScenarioBuildError("historical snapshot source hash mismatch")
    for row in source_rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or scope_hashes.get(str(row.get("path"))) != row.get("sha256")
        ):
            raise ScenarioBuildError(
                "historical producer source is not covered by reviewed scope"
            )

    with tempfile.TemporaryDirectory(prefix="ric-v5-attestation-") as directory:
        historical_root = Path(directory)
        for relative, payload in {**snapshot, **artifact_bytes}.items():
            target = historical_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        embedded = historical_root / "compat" / EMBEDDED_PRODUCER_SIGNOFF
        embedded.parent.mkdir(parents=True, exist_ok=True)
        embedded.write_bytes(signoff_path.read_bytes())
        required_sources = tuple(
            historical_root / str(row["path"]) for row in source_rows
        )
        required_reviewed = tuple(
            historical_root / str(row["path"]) for row in scope_rows
        )
        try:
            verified = verify_phase4_signoff(
                embedded,
                repo_root=historical_root,
                expected_fields=expected_fields,
                required_source_paths=required_sources,
                required_reviewed_scope_paths=required_reviewed,
            )
        except FormalProvenanceError as exc:
            raise ScenarioBuildError(
                "immutable upstream historical Phase-4 chain is invalid"
            ) from exc
    return verified


def verify_pre_outcome_attestation(
    path: Path,
    *,
    protocol_sha256: str,
    config_sha256: str,
    consumer_amendment_sha256: str,
    authoritative_bundle_root: Path,
    required_input_paths: Sequence[Path],
    producer_signoff_path: Path,
) -> Mapping[str, Any]:
    """Validate the outcome-blind migration-time bundle census."""

    if path.resolve(strict=True) != FORMAL_PREOUTCOME_ATTESTATION_PATH:
        raise ScenarioBuildError(
            "pre-outcome attestation path differs from reviewed write-once path"
        )

    try:
        value = load_json_mapping_strict(path, label="pre-outcome attestation")
        validate_self_hash(value, "attestation_sha256")
    except Exception as exc:
        raise ScenarioBuildError("pre-outcome attestation is malformed") from exc
    required = {
        "schema_version",
        "status",
        "scientific_result",
        "protocol_sha256",
        "config_sha256",
        "consumer_amendment_sha256",
        "historical_reviewed_source_snapshot_sha256",
        "capture_preoutcome_attestation_source_sha256",
        "producer_signoff_file_sha256",
        "producer_signoff_self_hash",
        "scanned_root",
        "census_roots",
        "path_census",
        "required_inputs",
        "forbidden_hits",
        "attestation_sha256",
    }
    if set(value) != required:
        raise ScenarioBuildError("pre-outcome attestation exact schema mismatch")
    expected = {
        "schema_version": "ric-pre-outcome-attestation-v1",
        "status": "PRE_OUTCOME_CONFIRMED",
        "scientific_result": False,
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "consumer_amendment_sha256": consumer_amendment_sha256,
        "historical_reviewed_source_snapshot_sha256": (
            HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256
        ),
        "forbidden_hits": [],
        "census_roots": list(FORMAL_CENSUS_RELATIVE_ROOTS),
        "capture_preoutcome_attestation_source_sha256": (
            _preoutcome_attestation_source_sha256()
        ),
        "producer_signoff_file_sha256": sha256_file(producer_signoff_path),
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted or type(value.get(field)) is not type(wanted):
            raise ScenarioBuildError(f"pre-outcome attestation mismatch: {field}")
    try:
        producer_signoff = verify_phase4_signoff(
            producer_signoff_path,
            repo_root=REPO_ROOT,
            expected_fields={
                "stage": "capture_preoutcome_attestation",
                "protocol_sha256": protocol_sha256,
                "config_sha256": config_sha256,
                "capture_preoutcome_attestation_source_sha256": (
                    _preoutcome_attestation_source_sha256()
                ),
                "consumer_amendment_sha256": consumer_amendment_sha256,
                "historical_reviewed_source_snapshot_sha256": (
                    HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256
                ),
                "authoritative_bundle_root": str(
                    FORMAL_AUTHORITATIVE_BUNDLE_ROOT
                ),
                "pre_outcome_attestation_path": str(
                    FORMAL_PREOUTCOME_ATTESTATION_PATH
                ),
            },
            required_source_paths=(
                HERE / "capture_preoutcome_attestation.py",
                Path(__file__),
                HERE / "formal_provenance.py",
            ),
            required_reviewed_scope_paths=(
                *canonical_reviewed_scope_paths(
                    REPO_ROOT,
                    (
                        HERE / "capture_preoutcome_attestation.py",
                        Path(__file__),
                        HERE / "formal_provenance.py",
                    ),
                ),
                DEFAULT_CONSUMER_AMENDMENT,
            ),
        )
    except FormalProvenanceError as exc:
        raise ScenarioBuildError(
            "pre-outcome producer Phase-4 signoff is invalid"
        ) from exc
    if value.get("producer_signoff_self_hash") != producer_signoff.get(
        "signoff_sha256"
    ):
        raise ScenarioBuildError("pre-outcome producer signoff self-hash mismatch")
    expected_root = validate_authoritative_bundle_root(
        authoritative_bundle_root, mode="formal"
    )
    if (
        value.get("scanned_root") != str(expected_root)
    ):
        raise ScenarioBuildError("pre-outcome authoritative bundle root mismatch")
    rows = value.get("path_census")
    if not isinstance(rows, list) or not rows:
        raise ScenarioBuildError("pre-outcome attestation has an empty path census")
    prior = ""
    census: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "size_bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or row["path"] <= prior
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] < 0
            or not is_sha256(row.get("sha256"))
        ):
            raise ScenarioBuildError("pre-outcome path census is not canonical")
        prior = str(row["path"])
        census[prior] = row
    registered = value.get("required_inputs")
    if not isinstance(registered, Mapping) or not registered:
        raise ScenarioBuildError("pre-outcome required-input registry is empty")
    registered_by_path: dict[str, Mapping[str, Any]] = {}
    for name, row in registered.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(row, Mapping)
            or set(row) != {"path", "size_bytes", "sha256"}
            or census.get(str(row.get("path"))) != row
            or str(row.get("path")) in registered_by_path
        ):
            raise ScenarioBuildError("pre-outcome required-input row is invalid")
        registered_by_path[str(row["path"])] = row
    for input_path in required_input_paths:
        if input_path.is_symlink():
            raise ScenarioBuildError("registered immutable input may not be a symlink")
        try:
            resolved = input_path.resolve(strict=True)
            relative = resolved.relative_to(expected_root).as_posix()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ScenarioBuildError(
                "immutable input is outside authoritative bundle"
            ) from exc
        row = registered_by_path.get(relative)
        if (
            row is None
            or row["size_bytes"] != resolved.stat().st_size
            or row["sha256"] != sha256_file(resolved)
        ):
            raise ScenarioBuildError(
                "immutable input differs from pre-outcome registry"
            )
    return value


def attested_file_sha256(attestation: Mapping[str, Any], path: Path) -> str:
    """Return the exact registered file digest for one already-validated path."""

    root = Path(str(attestation["scanned_root"])).resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ScenarioBuildError("attested file is outside authoritative bundle") from exc
    rows = attestation.get("required_inputs")
    if not isinstance(rows, Mapping):
        raise ScenarioBuildError("attested required-input registry is invalid")
    matches = [
        row
        for row in rows.values()
        if isinstance(row, Mapping) and row.get("path") == relative
    ]
    if len(matches) != 1 or not is_sha256(matches[0].get("sha256")):
        raise ScenarioBuildError("file is not uniquely registered pre-outcome")
    wanted = str(matches[0]["sha256"])
    if sha256_file(resolved) != wanted:
        raise ScenarioBuildError("registered file changed after pre-outcome census")
    return wanted


def verify_sealed_calibration_manifest_binding(
    sealed_manifest: Mapping[str, Any],
    calibration_manifest: Mapping[str, Any],
    *,
    calibration_manifest_path: Path,
) -> Mapping[str, str]:
    """Forbid substituting another calibration namespace for the sealed LUT."""

    observed = {
        "calibration_manifest_self_hash": str(
            calibration_manifest.get("manifest_sha256")
        ),
        "calibration_manifest_file_sha256": sha256_file(
            calibration_manifest_path
        ),
        "calibration_selected_list_sha256": object_sha256(
            calibration_manifest.get("selected_text_sha256")
        ),
    }
    for field, value in observed.items():
        if sealed_manifest.get(field) != value:
            raise ScenarioBuildError(
                f"sealed calibration manifest substitution detected: {field}"
            )
    return observed


def verify_preconsumer_sealed_data_attestation(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    protocol_sha256: str,
    config_sha256: str,
    historical_reviewed_source_snapshot_path: Path,
    expected_signoff_file_sha256: str,
    historical_calibration_lock_path: Path,
    historical_calibration_signoff_file_sha256: str,
    global_reservation_path: Path,
    global_consumption_path: Path,
) -> Mapping[str, str]:
    """Validate the immutable one-shot input after consumer-only source drift.

    Amendment Q permits this narrow path only because the sealed manifest was
    consumed before any scenario or policy outcome existed.  The old producer
    attestation remains bound byte-for-byte; the new scenario producer is
    separately reviewed and signs the exact old input hashes.
    """

    if manifest.get("role") != "sealed" or manifest.get("mode") != "formal":
        raise ScenarioBuildError("pre-consumer attestation requires formal sealed data")
    try:
        validate_data_manifest_fields(
            manifest,
            mode="formal",
            role="sealed",
            config=config,
            protocol_sha256=protocol_sha256,
            config_sha256=config_sha256,
            expected_prepare_data_source_sha256=_prepare_data_source_sha256(),
            expected_calibration_lock_self_hash=str(
                manifest.get("calibration_lock_self_hash")
            ),
            expected_calibration_lock_file_sha256=str(
                manifest.get("calibration_lock_file_sha256")
            ),
        )
    except Exception as exc:
        raise ScenarioBuildError("sealed data manifest strict validation failed") from exc
    signoff_path = manifest_path.parent / EMBEDDED_PRODUCER_SIGNOFF
    record_path = manifest_path.parent / "sealed_consumption_record.json"
    if not signoff_path.is_file() or not record_path.is_file():
        raise ScenarioBuildError("sealed input lacks producer signoff/consumption record")
    if sha256_file(signoff_path) != manifest.get("signoff_sha256"):
        raise ScenarioBuildError("sealed input producer signoff file mismatch")
    try:
        record = load_json_mapping_strict(record_path, label="sealed consumption record")
        validate_self_hash(record, "record_sha256")
    except Exception as exc:
        raise ScenarioBuildError("sealed input attestation is not self-consistent") from exc
    expected_signoff = {
        "schema_version": "ric-phase4-signoff-v1",
        "status": "SIGNED-OFF",
        "open_p0": 0,
        "stage": "prepare_data",
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "prepare_data_source_sha256": _prepare_data_source_sha256(),
        "data_role": "sealed",
        "calibration_lock_sha256": manifest.get("calibration_lock_self_hash"),
        "calibration_lock_file_sha256": manifest.get(
            "calibration_lock_file_sha256"
        ),
    }
    signoff = verify_immutable_upstream_signoff(
        signoff_path,
        snapshot_path=historical_reviewed_source_snapshot_path,
        expected_signoff_file_sha256=expected_signoff_file_sha256,
        expected_fields=expected_signoff,
    )
    historical_runner = signoff.get("run_experiment_source_sha256")
    if not is_sha256(historical_runner):
        raise ScenarioBuildError("sealed producer lacks historical runner source hash")
    try:
        historical_lock = load_json_mapping_strict(
            historical_calibration_lock_path,
            label="historical calibration lock",
        )
        validate_self_hash(historical_lock)
    except Exception as exc:
        raise ScenarioBuildError("historical calibration lock is malformed") from exc
    expected_lock_fields = {
        "schema_version": "ric-calibration-lock-v1",
        "status": "CALIBRATION_LOCKED",
        "scientific_result": False,
        "mode": "formal",
        "role": "calibration",
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "g1_pass": True,
        "manifest_sha256": manifest.get("calibration_lock_self_hash"),
    }
    for field, wanted in expected_lock_fields.items():
        if (
            historical_lock.get(field) != wanted
            or type(historical_lock.get(field)) is not type(wanted)
        ):
            raise ScenarioBuildError(
                f"historical calibration lock mismatch: {field}"
            )
    if sha256_file(historical_calibration_lock_path) != manifest.get(
        "calibration_lock_file_sha256"
    ):
        raise ScenarioBuildError("historical calibration lock file hash mismatch")
    historical_lock_signoff_path = (
        historical_calibration_lock_path.parent / EMBEDDED_PRODUCER_SIGNOFF
    )
    if (
        not historical_lock_signoff_path.is_file()
        or sha256_file(historical_lock_signoff_path)
        != historical_lock.get("signoff_sha256")
    ):
        raise ScenarioBuildError("historical calibration lock signoff mismatch")
    historical_lock_signoff = verify_immutable_upstream_signoff(
        historical_lock_signoff_path,
        snapshot_path=historical_reviewed_source_snapshot_path,
        expected_signoff_file_sha256=(
            historical_calibration_signoff_file_sha256
        ),
        expected_fields={
            "stage": "calibration",
            "protocol_sha256": protocol_sha256,
            "config_sha256": config_sha256,
            "run_experiment_source_sha256": historical_lock.get(
                "run_experiment_source_sha256"
            ),
            "scenario_tree_sha256": historical_lock.get(
                "scenario_tree_sha256"
            ),
            "scenario_producer_signoff_sha256": historical_lock.get(
                "scenario_producer_signoff_sha256"
            ),
            "capability_probe_sha256": historical_lock.get(
                "capability_probe_sha256"
            ),
            "capability_producer_signoff_sha256": historical_lock.get(
                "capability_producer_signoff_sha256"
            ),
        },
    )
    expected_record = {
        "schema_version": "ric-sealed-consumption-v1",
        "state": "CONSUMED",
        "role": "sealed",
        "mode": "formal",
        "manifest_sha256": manifest.get("manifest_sha256"),
        "signoff_file_sha256": manifest.get("signoff_sha256"),
        "signoff_manifest_sha256": signoff.get("signoff_sha256"),
        "calibration_lock_self_hash": manifest.get("calibration_lock_self_hash"),
        "calibration_lock_file_sha256": manifest.get(
            "calibration_lock_file_sha256"
        ),
        "dataset_slice_canonical_content_sha256": manifest.get(
            "dataset_slice_canonical_content_sha256"
        ),
        "reservation_sha256": manifest.get("sealed_reservation_sha256"),
        "nonce_sha256": manifest.get("sealed_nonce_sha256"),
    }
    for field, wanted in expected_record.items():
        if record.get(field) != wanted:
            raise ScenarioBuildError(f"sealed consumption record mismatch: {field}")
    try:
        reservation = load_json_mapping_strict(
            global_reservation_path, label="global sealed reservation"
        )
        global_consumption = load_json_mapping_strict(
            global_consumption_path, label="global sealed consumption"
        )
        validate_self_hash(reservation, "record_sha256")
        validate_self_hash(global_consumption, "record_sha256")
    except Exception as exc:
        raise ScenarioBuildError("global sealed one-shot ledger is invalid") from exc
    expected_reservation = {
        "schema_version": "ric-sealed-reservation-v1",
        "state": "RESERVED_FAIL_CLOSED",
        "role": "sealed",
        "mode": "formal",
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "record_sha256": manifest.get("sealed_reservation_sha256"),
        "nonce_sha256": manifest.get("sealed_nonce_sha256"),
        "calibration_lock_self_hash": manifest.get(
            "calibration_lock_self_hash"
        ),
        "calibration_lock_file_sha256": manifest.get(
            "calibration_lock_file_sha256"
        ),
        "signoff_file_sha256": manifest.get("signoff_sha256"),
        "signoff_manifest_sha256": signoff.get("signoff_sha256"),
    }
    for field, wanted in expected_reservation.items():
        if reservation.get(field) != wanted:
            raise ScenarioBuildError(f"global sealed reservation mismatch: {field}")
    if global_consumption != record or global_consumption_path.read_bytes() != record_path.read_bytes():
        raise ScenarioBuildError(
            "global/local sealed consumption records are not byte-identical"
        )
    attestation_sha256 = object_sha256(
        {
            "manifest_self_hash": manifest["manifest_sha256"],
            "manifest_file_sha256": sha256_file(manifest_path),
            "producer_signoff_self_hash": signoff["signoff_sha256"],
            "producer_signoff_file_sha256": sha256_file(signoff_path),
            "consumption_record_self_hash": record["record_sha256"],
            "consumption_record_file_sha256": sha256_file(record_path),
            "historical_reviewed_source_snapshot_sha256": (
                HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256
            ),
        }
    )
    return {
        "record_sha256": str(attestation_sha256),
        "historical_run_experiment_source_sha256": str(historical_runner),
        "historical_calibration_lock_sha256": str(
            historical_lock["manifest_sha256"]
        ),
        "historical_calibration_signoff_sha256": str(
            historical_lock_signoff["signoff_sha256"]
        ),
        "consumption_record_self_hash": str(record["record_sha256"]),
        "consumption_record_file_sha256": sha256_file(record_path),
        "global_reservation_file_sha256": sha256_file(global_reservation_path),
        "global_consumption_file_sha256": sha256_file(global_consumption_path),
    }


def _validate_route_tuple_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    parity_row: Mapping[str, Any],
    top_k: int,
    valid_tokens: int,
) -> None:
    """Rebuild one request/layer ordered route tuple from persisted JSONL."""

    import torch

    if len(rows) != valid_tokens * top_k:
        raise ScenarioBuildError("route tuple group has incomplete token/slot coverage")
    experts: list[list[int]] = [[] for _ in range(valid_tokens)]
    weights: list[list[float]] = [[] for _ in range(valid_tokens)]
    expected_dtype = parity_row.get("effective_route_weight_dtype")
    if expected_dtype != "torch.bfloat16":
        raise ScenarioBuildError("formal route effective-weight dtype is not BF16")
    for ordinal, row in enumerate(rows):
        token_position = ordinal // top_k
        topk_slot = ordinal % top_k
        if (
            row.get("token_position") != token_position
            or row.get("topk_slot") != topk_slot
            or row.get("route_weight_dtype") != expected_dtype
        ):
            raise ScenarioBuildError("route tuple token/slot/dtype order mismatch")
        route_weight = row.get("route_weight")
        diagnostic_weight = row.get("route_weight_fp32_precast")
        if (
            isinstance(route_weight, bool)
            or not isinstance(route_weight, (int, float))
            or not math.isfinite(float(route_weight))
            or isinstance(diagnostic_weight, bool)
            or not isinstance(diagnostic_weight, (int, float))
            or not math.isfinite(float(diagnostic_weight))
        ):
            raise ScenarioBuildError("route tuple weight is missing or non-finite")
        experts[token_position].append(int(row["expert_id"]))
        weights[token_position].append(float(route_weight))
    expert_tensor = torch.tensor(experts, dtype=torch.int64)
    weight_tensor = torch.tensor(weights, dtype=torch.bfloat16)
    if _route_tuple_sha256(expert_tensor, weight_tensor) != parity_row.get(
        "native_route_tuple_sha256"
    ):
        raise ScenarioBuildError("persisted route tuple hash differs from native capture")


def load_validated_inputs(
    *,
    role: str,
    mode: str,
    model_key: str,
    data_manifest_path: Path,
    calibration_data_manifest_path: Path | None,
    route_dir: Path,
    service_lut_dir: Path,
    config_path: Path,
    protocol_path: Path,
    consumer_amendment_path: Path,
    historical_reviewed_source_snapshot_path: Path | None,
    pre_outcome_attestation_path: Path | None,
    pre_outcome_producer_signoff_path: Path | None,
    authoritative_bundle_root_path: Path | None,
    historical_calibration_lock_path: Path | None,
) -> ValidatedInputs:
    guard_mode_role(mode, role)
    validate_frozen_formal_paths(
        config_path=config_path, protocol_path=protocol_path, mode=mode
    )
    config = _load_config(config_path)
    if model_key not in config["models"]:
        raise ScenarioBuildError("unknown frozen model")
    config_sha = sha256_file(config_path)
    protocol_sha = sha256_file(protocol_path)
    consumer_amendment_sha = validate_consumer_amendment_path(
        consumer_amendment_path, mode=mode
    )
    pre_outcome_attestation: Mapping[str, Any] | None = None
    if mode == "formal":
        if historical_reviewed_source_snapshot_path is None:
            raise ScenarioBuildError(
                "formal scenario requires --historical-reviewed-source-snapshot"
            )
        if pre_outcome_attestation_path is None:
            raise ScenarioBuildError(
                "formal scenario requires --pre-outcome-attestation"
            )
        if pre_outcome_producer_signoff_path is None:
            raise ScenarioBuildError(
                "formal scenario requires --pre-outcome-producer-signoff"
            )
        if authoritative_bundle_root_path is None:
            raise ScenarioBuildError(
                "formal scenario requires --authoritative-bundle-root"
            )
        if role == "sealed" and historical_calibration_lock_path is None:
            raise ScenarioBuildError(
                "formal sealed scenario requires --historical-calibration-lock"
            )
        if (
            sha256_file(historical_reviewed_source_snapshot_path)
            != HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256
        ):
            raise ScenarioBuildError("historical reviewed-source snapshot mismatch")
        required_input_paths = [
            data_manifest_path,
            data_manifest_path.parent / EMBEDDED_PRODUCER_SIGNOFF,
            route_dir / "capture_metadata.json",
            route_dir / "placement.json",
            route_dir / "route_parity.json",
            route_dir / "route_trace.jsonl",
            route_dir / EMBEDDED_PRODUCER_SIGNOFF,
            service_lut_dir / "service_lut_metadata.json",
            service_lut_dir / "service_lut.csv",
            service_lut_dir / "service_lut_raw.csv",
            service_lut_dir / EMBEDDED_PRODUCER_SIGNOFF,
        ]
        if calibration_data_manifest_path is not None:
            required_input_paths.extend(
                [
                    calibration_data_manifest_path,
                    calibration_data_manifest_path.parent
                    / EMBEDDED_PRODUCER_SIGNOFF,
                ]
            )
        if role == "sealed":
            required_input_paths.append(
                data_manifest_path.parent / "sealed_consumption_record.json"
            )
            assert historical_calibration_lock_path is not None
            required_input_paths.extend(
                [
                    historical_calibration_lock_path,
                    historical_calibration_lock_path.parent
                    / EMBEDDED_PRODUCER_SIGNOFF,
                    GLOBAL_SEALED_RESERVATION,
                    GLOBAL_SEALED_CONSUMPTION,
                ]
            )
        pre_outcome_attestation = verify_pre_outcome_attestation(
            pre_outcome_attestation_path,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
            consumer_amendment_sha256=consumer_amendment_sha,
            authoritative_bundle_root=authoritative_bundle_root_path,
            required_input_paths=required_input_paths,
            producer_signoff_path=pre_outcome_producer_signoff_path,
        )
    elif (
        historical_reviewed_source_snapshot_path is not None
        or pre_outcome_attestation_path is not None
        or pre_outcome_producer_signoff_path is not None
        or authoritative_bundle_root_path is not None
        or historical_calibration_lock_path is not None
    ):
        raise ScenarioBuildError(
            "development scenario must not consume formal migration evidence"
        )
    manifest = _read_self_hashed_json(
        data_manifest_path, schema_version="ric-data-manifest-v1"
    )
    if manifest.get("role") != role:
        raise ScenarioBuildError("declared role/data manifest mismatch")
    sealed_attestation: Mapping[str, str] | None = None
    if mode == "formal":
        manifest_signoff_path = data_manifest_path.parent / EMBEDDED_PRODUCER_SIGNOFF
        if (
            not manifest_signoff_path.is_file()
            or sha256_file(manifest_signoff_path) != manifest.get("signoff_sha256")
        ):
            raise ScenarioBuildError("data manifest/producer signoff file mismatch")
        if role == "sealed":
            assert historical_reviewed_source_snapshot_path is not None
            sealed_attestation = verify_preconsumer_sealed_data_attestation(
                data_manifest_path,
                manifest,
                config=config,
                protocol_sha256=protocol_sha,
                config_sha256=config_sha,
                historical_reviewed_source_snapshot_path=(
                    historical_reviewed_source_snapshot_path
                ),
                expected_signoff_file_sha256=attested_file_sha256(
                    pre_outcome_attestation,
                    data_manifest_path.parent / EMBEDDED_PRODUCER_SIGNOFF,
                ),
                historical_calibration_lock_path=(
                    historical_calibration_lock_path
                ),
                historical_calibration_signoff_file_sha256=(
                    attested_file_sha256(
                        pre_outcome_attestation,
                        historical_calibration_lock_path.parent
                        / EMBEDDED_PRODUCER_SIGNOFF,
                    )
                ),
                global_reservation_path=GLOBAL_SEALED_RESERVATION,
                global_consumption_path=GLOBAL_SEALED_CONSUMPTION,
            )
        else:
            try:
                validate_data_manifest_fields(
                    manifest,
                    mode=mode,
                    role=role,
                    config=config,
                    protocol_sha256=protocol_sha,
                    config_sha256=config_sha,
                    expected_prepare_data_source_sha256=(
                        _prepare_data_source_sha256()
                    ),
                )
                assert historical_reviewed_source_snapshot_path is not None
                verify_immutable_upstream_signoff(
                    data_manifest_path.parent / EMBEDDED_PRODUCER_SIGNOFF,
                    snapshot_path=historical_reviewed_source_snapshot_path,
                    expected_signoff_file_sha256=attested_file_sha256(
                        pre_outcome_attestation,
                        data_manifest_path.parent / EMBEDDED_PRODUCER_SIGNOFF,
                    ),
                    expected_fields={
                        "stage": "prepare_data",
                        "protocol_sha256": protocol_sha,
                        "config_sha256": config_sha,
                        "prepare_data_source_sha256": (
                            _prepare_data_source_sha256()
                        ),
                        "data_role": role,
                    },
                )
            except Exception as exc:
                raise ScenarioBuildError("data producer signoff is invalid") from exc
    if role == "sealed":
        if calibration_data_manifest_path is None:
            raise ScenarioBuildError(
                "sealed scenario requires --calibration-data-manifest"
            )
        calibration_manifest = _read_self_hashed_json(
            calibration_data_manifest_path, schema_version="ric-data-manifest-v1"
        )
        if (
            calibration_manifest.get("role") != "calibration"
            or calibration_manifest.get("mode") != mode
            or calibration_manifest.get("config_sha256") != config_sha
            or calibration_manifest.get("protocol_sha256") != protocol_sha
        ):
            raise ScenarioBuildError("service calibration manifest binding mismatch")
        try:
            validate_data_manifest_fields(
                calibration_manifest,
                mode=mode,
                role="calibration",
                config=config,
                protocol_sha256=protocol_sha,
                config_sha256=config_sha,
                expected_prepare_data_source_sha256=_prepare_data_source_sha256(),
            )
        except Exception as exc:
            raise ScenarioBuildError(
                "service calibration manifest strict validation failed"
            ) from exc
        verify_sealed_calibration_manifest_binding(
            manifest,
            calibration_manifest,
            calibration_manifest_path=calibration_data_manifest_path,
        )
        if mode == "formal":
            try:
                assert historical_reviewed_source_snapshot_path is not None
                calibration_signoff_path = (
                    calibration_data_manifest_path.parent
                    / EMBEDDED_PRODUCER_SIGNOFF
                )
                if (
                    not calibration_signoff_path.is_file()
                    or sha256_file(calibration_signoff_path)
                    != calibration_manifest.get("signoff_sha256")
                ):
                    raise ScenarioBuildError(
                        "calibration manifest/producer signoff file mismatch"
                    )
                verify_immutable_upstream_signoff(
                    calibration_signoff_path,
                    snapshot_path=historical_reviewed_source_snapshot_path,
                    expected_signoff_file_sha256=attested_file_sha256(
                        pre_outcome_attestation,
                        calibration_data_manifest_path.parent
                        / EMBEDDED_PRODUCER_SIGNOFF,
                    ),
                    expected_fields={
                        "stage": "prepare_data",
                        "protocol_sha256": protocol_sha,
                        "config_sha256": config_sha,
                        "prepare_data_source_sha256": (
                            _prepare_data_source_sha256()
                        ),
                        "data_role": "calibration",
                    },
                )
            except Exception as exc:
                raise ScenarioBuildError(
                    "service calibration data producer signoff is invalid"
                ) from exc
    else:
        if calibration_data_manifest_path is not None:
            raise ScenarioBuildError(
                "calibration scenario must not pass --calibration-data-manifest"
            )
        calibration_manifest = manifest
    spec = config["models"][model_key]
    model_revision = f"{spec['repo_id']}@{spec['revision']}"
    if manifest.get("model_revisions", {}).get(model_key) != model_revision:
        raise ScenarioBuildError("data/model revision mismatch")
    route_metadata = _read_self_hashed_json(
        route_dir / "capture_metadata.json", schema_version="ric-route-capture-v1"
    )
    placement = _read_self_hashed_json(
        route_dir / "placement.json", schema_version="ric-placement-v1"
    )
    parity = _read_self_hashed_json(
        route_dir / "route_parity.json", schema_version="ric-route-parity-v1"
    )
    route_path = route_dir / "route_trace.jsonl"
    expected_bindings = {
        "mode": mode,
        "data_role": role,
        "model_key": model_key,
        "model_revision": model_revision,
        "config_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "data_manifest_sha256": manifest["manifest_sha256"],
        "data_producer_signoff_sha256": manifest.get("signoff_sha256"),
        "placement_manifest_sha256": placement["manifest_sha256"],
        "route_parity_sha256": parity["manifest_sha256"],
        "route_trace_sha256": sha256_file(route_path),
        "top_k": int(spec["top_k"]),
        "num_experts": int(spec["num_experts"]),
        "request_count": int(config["data"][role]["document_count"]),
        "capture_routes_source_sha256": _capture_routes_source_sha256(),
    }
    for field, wanted in expected_bindings.items():
        if route_metadata.get(field) != wanted:
            raise ScenarioBuildError(f"route artifact binding mismatch: {field}")
    expected_status = "CAPTURE_ONLY" if mode == "formal" else "NOT_TESTED"
    if route_metadata.get("status") != expected_status or parity.get("status") != expected_status:
        raise ScenarioBuildError("route/parity status does not match execution mode")
    route_signoff_sha256 = route_metadata.get("signoff_sha256")
    if mode == "formal":
        assert historical_reviewed_source_snapshot_path is not None
        embedded_signoff = route_dir / EMBEDDED_PRODUCER_SIGNOFF
        model_tree_sha256 = route_metadata.get("model_tree_manifest_sha256")
        if (
            not is_sha256(route_signoff_sha256)
            or not is_sha256(model_tree_sha256)
            or not embedded_signoff.is_file()
            or sha256_file(embedded_signoff) != route_signoff_sha256
        ):
            raise ScenarioBuildError("formal route lacks embedded producer signoff")
        try:
            verify_immutable_upstream_signoff(
                embedded_signoff,
                snapshot_path=historical_reviewed_source_snapshot_path,
                expected_signoff_file_sha256=attested_file_sha256(
                    pre_outcome_attestation, embedded_signoff
                ),
                expected_fields={
                    "stage": "capture_routes",
                    "protocol_sha256": protocol_sha,
                    "config_sha256": config_sha,
                    "capture_routes_source_sha256": (
                        _capture_routes_source_sha256()
                    ),
                    "prepare_data_source_sha256": (
                        _prepare_data_source_sha256()
                    ),
                    "data_manifest_sha256": str(manifest["manifest_sha256"]),
                    "data_producer_signoff_sha256": str(
                        manifest.get("signoff_sha256")
                    ),
                    "model_key": model_key,
                    "model_tree_manifest_sha256": str(model_tree_sha256),
                },
            )
        except Exception as exc:
            raise ScenarioBuildError("route producer signoff is invalid") from exc
    elif route_signoff_sha256 is not None:
        raise ScenarioBuildError("development route claims a producer signoff")
    if (
        parity.get("all_topk_exact") is not True
        or parity.get("all_route_weights_exact") is not True
        or parity.get("all_native_moe_outputs_within_frozen_tolerance") is not True
        or parity.get("expected_layer_source")
        != "model_config.num_hidden_layers_all_layers_are_moe"
        or parity.get("native_moe_output_tolerance")
        != config["route_capture"].get("native_moe_output_tolerance")
        or parity.get("native_topk_selection_rule")
        != config["route_capture"].get("native_topk_selection_rule")
        or parity.get("stable_sort_substitution_allowed") is not False
        or parity.get("native_topk_dispatch_capture_required") is not True
        or parity.get("route_rows_use_captured_native_topk") is not True
        or parity.get("effective_route_weight")
        != config["route_capture"].get("effective_route_weight")
        or parity.get("native_moe_class") != spec.get("native_moe_class")
        or parity.get("native_moe_forward_source_sha256")
        != spec.get("native_moe_forward_source_sha256")
    ):
        raise ScenarioBuildError("independent native MoE route parity is not valid")
    expected_layers = parity.get("expected_layers")
    if (
        type(parity.get("num_hidden_layers")) is not int
        or expected_layers != list(range(int(parity["num_hidden_layers"])))
        or route_metadata.get("expected_layers") != expected_layers
        or route_metadata.get("expected_layer_source")
        != parity.get("expected_layer_source")
    ):
        raise ScenarioBuildError("model-config-derived MoE layer census mismatch")
    parity_rows = parity.get("parity_rows")
    if not isinstance(parity_rows, list):
        raise ScenarioBuildError("route parity rows are missing")
    parity_grid = {
        (str(row.get("request_id")), int(row.get("layer_id", -1)))
        for row in parity_rows
        if isinstance(row, Mapping)
    }
    expected_parity_grid = {
        (str(request["request_id"]), int(layer))
        for request in manifest["requests"]
        for layer in expected_layers
    }
    hash_fields = (
        "gate_hook_logit_sha256",
        "output_router_logit_sha256",
        "native_topk_indices_sha256",
        "native_topk_precast_values_sha256",
        "native_effective_weights_sha256",
        "native_route_tuple_sha256",
        "replay_route_tuple_sha256",
    )
    if (
        len(parity_rows) != len(expected_parity_grid)
        or parity_grid != expected_parity_grid
        or not all(
            row.get("topk_expert_exact_native_capture") is True
            and row.get("topk_precast_weight_exact_native_capture") is True
            and row.get("topk_effective_weight_exact_native_capture") is True
            and row.get("route_tuple_hash_equal") is True
            and row.get("within_frozen_tolerance") is True
            and row.get("raw_logit_hash_equal") is True
            and not isinstance(row.get("max_logit_abs_error"), bool)
            and isinstance(row.get("max_logit_abs_error"), (int, float))
            and float(row["max_logit_abs_error"]) == 0.0
            and row.get("gate_hook_logit_sha256")
            == row.get("output_router_logit_sha256")
            and row.get("native_route_tuple_sha256")
            == row.get("replay_route_tuple_sha256")
            and all(
                isinstance(row.get(field), str)
                and len(str(row[field])) == 64
                and all(character in "0123456789abcdef" for character in str(row[field]))
                for field in hash_fields
            )
            for row in parity_rows
        )
    ):
        raise ScenarioBuildError("route parity request/layer evidence grid mismatch")
    ep_size = int(config["topology_proxy"]["ep_size"])
    expected_placement = {
        str(expert): expert_sender(expert, int(spec["num_experts"]), ep_size)
        for expert in range(int(spec["num_experts"]))
    }
    expected_origins = origin_lpt(manifest["requests"], ep_size)
    independently_selected_layers = selected_layers(
        [int(value) for value in expected_layers],
        selection_seed=int(config["data"]["selection_seed"]),
        model_revision=model_revision,
        count=int(config["route_capture"]["selected_layer_count_per_model"]),
    )
    if route_metadata.get("selected_layers") != independently_selected_layers:
        raise ScenarioBuildError("selected replay layers cannot be recomputed")
    assigned_layer_by_request = {
        str(request["request_id"]): assigned_layer(
            str(request["request_id"]), independently_selected_layers
        )
        for request in manifest["requests"]
    }
    if placement.get("expert_to_sender") != expected_placement:
        raise ScenarioBuildError("placement cannot be recomputed from frozen config")
    if placement.get("request_to_receiver") != expected_origins:
        raise ScenarioBuildError("request origins cannot be recomputed route-blind")
    required_route_fields = {
        "model_key",
        "model_revision",
        "data_manifest_sha256",
        "placement_manifest_sha256",
        "request_id",
        "forward_id",
        "batch_id",
        "phase",
        "decode_step",
        "layer_id",
        "token_id",
        "token_block_id",
        "token_position",
        "topk_slot",
        "expert_id",
        "sender_rank",
        "receiver_rank",
        "epoch",
        "valid",
        "route_weight",
        "route_weight_dtype",
        "route_weight_fp32_precast",
        "selected_for_replay",
        "route_source",
    }
    by_request: dict[str, list[Mapping[str, Any]]] = {
        str(row["request_id"]): [] for row in manifest["requests"]
    }
    route_rows = 0
    parity_by_key = {
        (str(row["request_id"]), int(row["layer_id"])): row
        for row in parity_rows
    }
    seen_route_tuple_groups: set[tuple[str, int]] = set()
    current_route_tuple_key: tuple[str, int] | None = None
    current_route_tuple_rows: list[Mapping[str, Any]] = []

    def flush_route_tuple_group() -> None:
        nonlocal current_route_tuple_key, current_route_tuple_rows
        if current_route_tuple_key is None:
            return
        if current_route_tuple_key in seen_route_tuple_groups:
            raise ScenarioBuildError("route tuple group is non-contiguous or duplicated")
        parity_row = parity_by_key.get(current_route_tuple_key)
        if parity_row is None:
            raise ScenarioBuildError("route tuple group lacks parity evidence")
        _validate_route_tuple_group(
            current_route_tuple_rows,
            parity_row=parity_row,
            top_k=int(spec["top_k"]),
            valid_tokens=int(config["data"]["sequence_length"]),
        )
        seen_route_tuple_groups.add(current_route_tuple_key)
        current_route_tuple_key = None
        current_route_tuple_rows = []

    with route_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = loads_json_mapping_strict(
                    line, label=f"route JSONL line {line_number}"
                )
            except FormalProvenanceError as exc:
                raise ScenarioBuildError(f"invalid route JSONL line {line_number}") from exc
            route_rows += 1
            if not required_route_fields <= set(row):
                raise ScenarioBuildError("route row schema is incomplete")
            if type(row["valid"]) is not bool or type(row["selected_for_replay"]) is not bool:
                raise ScenarioBuildError("route valid/selected fields must be exact bool")
            for field in (
                "decode_step",
                "layer_id",
                "topk_slot",
                "expert_id",
                "sender_rank",
                "receiver_rank",
                "epoch",
                "token_position",
            ):
                if type(row[field]) is not int:
                    raise ScenarioBuildError(f"route {field} must be an exact integer")
            for field, wanted in (
                ("model_key", model_key),
                ("model_revision", model_revision),
                ("data_manifest_sha256", manifest["manifest_sha256"]),
                ("placement_manifest_sha256", placement["manifest_sha256"]),
                (
                    "route_source",
                    "native_aten_topk_capture_plus_independent_moe_output_parity",
                ),
            ):
                if row.get(field) != wanted:
                    raise ScenarioBuildError(f"route row binding mismatch: {field}")
            request_id = str(row["request_id"])
            if request_id not in by_request:
                raise ScenarioBuildError("route contains request outside manifest")
            expected_selected = (
                int(row["layer_id"]) == assigned_layer_by_request[request_id]
            )
            if row["selected_for_replay"] is not expected_selected:
                raise ScenarioBuildError(
                    "selected_for_replay differs from independent layer assignment"
                )
            route_tuple_key = (request_id, int(row["layer_id"]))
            if current_route_tuple_key != route_tuple_key:
                flush_route_tuple_group()
                current_route_tuple_key = route_tuple_key
            current_route_tuple_rows.append(row)
            if row["selected_for_replay"]:
                by_request[request_id].append(row)
    flush_route_tuple_group()
    if seen_route_tuple_groups != expected_parity_grid:
        raise ScenarioBuildError("route tuple evidence grid is incomplete")
    if route_rows != int(route_metadata["route_rows"]):
        raise ScenarioBuildError("route row count mismatch")
    expected_per_request = 128 * int(spec["top_k"])
    if any(len(rows) != expected_per_request for rows in by_request.values()):
        raise ScenarioBuildError("selected route rows do not cover full 128 x top-k load")
    service = load_service_surface(
        service_lut_dir,
        model_key=model_key,
        model_revision=model_revision,
        config_sha256=config_sha,
        protocol_sha256=protocol_sha,
        mode=mode,
        expected_selected_layers=[
            int(value) for value in route_metadata["selected_layers"]
        ],
        expected_num_experts=int(spec["num_experts"]),
        config=config,
        expected_data_manifest_sha256=str(
            calibration_manifest["manifest_sha256"]
        ),
        expected_data_producer_signoff_sha256=(
            str(calibration_manifest.get("signoff_sha256"))
            if calibration_manifest.get("signoff_sha256") is not None
            else None
        ),
        historical_reviewed_source_snapshot_path=(
            historical_reviewed_source_snapshot_path
        ),
        expected_producer_signoff_file_sha256=(
            attested_file_sha256(
                pre_outcome_attestation,
                service_lut_dir / EMBEDDED_PRODUCER_SIGNOFF,
            )
            if pre_outcome_attestation is not None
            else None
        ),
    )
    if service.model_tree_manifest_sha256 != route_metadata.get(
        "model_tree_manifest_sha256"
    ):
        raise ScenarioBuildError("route/service artifacts use different local model trees")
    if mode == "formal":
        route_gpu = validate_gpu_environment_artifact(
            route_metadata, config=config, label="route capture"
        )
        if route_gpu != service.gpu_environment_identity:
            raise ScenarioBuildError(
                "BLOCKED_GPU_ENVIRONMENT: route/service environments differ"
            )
    calibration_manifest_file_sha256 = (
        sha256_file(calibration_data_manifest_path)
        if calibration_data_manifest_path is not None
        else sha256_file(data_manifest_path)
    )
    calibration_selected_list_sha256 = object_sha256(
        calibration_manifest.get("selected_text_sha256")
    )
    pre_outcome_sha256 = (
        str(pre_outcome_attestation["attestation_sha256"])
        if pre_outcome_attestation is not None
        else ""
    )
    pre_outcome_producer_signoff_file_sha256 = (
        str(pre_outcome_attestation["producer_signoff_file_sha256"])
        if pre_outcome_attestation is not None
        else ""
    )
    pre_outcome_producer_signoff_self_hash = (
        str(pre_outcome_attestation["producer_signoff_self_hash"])
        if pre_outcome_attestation is not None
        else ""
    )
    snapshot_sha256 = (
        HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256 if mode == "formal" else ""
    )
    authoritative_bundle_root = (
        str(authoritative_bundle_root_path.resolve(strict=True))
        if authoritative_bundle_root_path is not None
        else ""
    )
    compatibility_sha256 = object_sha256(
        {
            "consumer_amendment_sha256": consumer_amendment_sha,
            "historical_reviewed_source_snapshot_sha256": snapshot_sha256,
            "pre_outcome_attestation_sha256": pre_outcome_sha256,
            "pre_outcome_producer_signoff_file_sha256": (
                pre_outcome_producer_signoff_file_sha256
            ),
            "pre_outcome_producer_signoff_self_hash": (
                pre_outcome_producer_signoff_self_hash
            ),
            "authoritative_bundle_root": authoritative_bundle_root,
            "workload_data_manifest_self_hash": manifest["manifest_sha256"],
            "workload_data_manifest_file_sha256": sha256_file(data_manifest_path),
            "workload_data_producer_signoff_file_sha256": manifest.get(
                "signoff_sha256"
            ),
            "service_calibration_manifest_self_hash": calibration_manifest[
                "manifest_sha256"
            ],
            "service_calibration_manifest_file_sha256": (
                calibration_manifest_file_sha256
            ),
            "service_calibration_selected_list_sha256": (
                calibration_selected_list_sha256
            ),
            "service_calibration_producer_signoff_file_sha256": (
                calibration_manifest.get("signoff_sha256")
            ),
            "route_capture_manifest_sha256": route_metadata["manifest_sha256"],
            "route_trace_sha256": route_metadata["route_trace_sha256"],
            "route_parity_sha256": parity["manifest_sha256"],
            "placement_manifest_sha256": placement["manifest_sha256"],
            "route_producer_signoff_file_sha256": route_metadata.get(
                "signoff_sha256"
            ),
            "service_lut_metadata_sha256": service.metadata_sha256,
            "service_lut_summary_sha256": service.summary_sha256,
            "service_lut_raw_sha256": service.raw_sha256,
            "service_lut_producer_signoff_file_sha256": (
                service.producer_signoff_sha256
            ),
            "sealed_input_attestation_sha256": (
                sealed_attestation.get("record_sha256")
                if sealed_attestation is not None
                else None
            ),
            "sealed_input_historical_calibration_lock_sha256": (
                sealed_attestation.get("historical_calibration_lock_sha256")
                if sealed_attestation is not None
                else None
            ),
            "sealed_input_historical_calibration_signoff_sha256": (
                sealed_attestation.get(
                    "historical_calibration_signoff_sha256"
                )
                if sealed_attestation is not None
                else None
            ),
            "sealed_global_reservation_file_sha256": (
                sealed_attestation.get("global_reservation_file_sha256")
                if sealed_attestation is not None
                else None
            ),
            "sealed_global_consumption_file_sha256": (
                sealed_attestation.get("global_consumption_file_sha256")
                if sealed_attestation is not None
                else None
            ),
        }
    )
    return ValidatedInputs(
        role=role,
        model_key=model_key,
        model_revision=model_revision,
        top_k=int(spec["top_k"]),
        num_experts=int(spec["num_experts"]),
        data_manifest=manifest,
        route_rows_by_request={
            key: tuple(value) for key, value in by_request.items()
        },
        placement=placement,
        route_metadata=route_metadata,
        service=service,
        service_calibration_data_manifest_sha256=str(
            calibration_manifest["manifest_sha256"]
        ),
        service_calibration_data_manifest_file_sha256=(
            calibration_manifest_file_sha256
        ),
        service_calibration_selected_list_sha256=(
            calibration_selected_list_sha256
        ),
        service_calibration_data_producer_signoff_sha256=(
            str(calibration_manifest.get("signoff_sha256"))
            if calibration_manifest.get("signoff_sha256") is not None
            else None
        ),
        sealed_input_attestation_sha256=(
            str(sealed_attestation["record_sha256"])
            if sealed_attestation is not None
            else None
        ),
        sealed_input_historical_run_experiment_source_sha256=(
            str(sealed_attestation["historical_run_experiment_source_sha256"])
            if sealed_attestation is not None
            else None
        ),
        sealed_input_historical_calibration_lock_sha256=(
            str(sealed_attestation["historical_calibration_lock_sha256"])
            if sealed_attestation is not None
            else None
        ),
        sealed_input_historical_calibration_signoff_sha256=(
            str(
                sealed_attestation[
                    "historical_calibration_signoff_sha256"
                ]
            )
            if sealed_attestation is not None
            else None
        ),
        sealed_global_reservation_file_sha256=(
            str(sealed_attestation["global_reservation_file_sha256"])
            if sealed_attestation is not None
            else None
        ),
        sealed_global_consumption_file_sha256=(
            str(sealed_attestation["global_consumption_file_sha256"])
            if sealed_attestation is not None
            else None
        ),
        consumer_amendment_sha256=consumer_amendment_sha,
        historical_reviewed_source_snapshot_sha256=snapshot_sha256,
        pre_outcome_attestation_sha256=pre_outcome_sha256,
        pre_outcome_producer_signoff_file_sha256=(
            pre_outcome_producer_signoff_file_sha256
        ),
        pre_outcome_producer_signoff_self_hash=(
            pre_outcome_producer_signoff_self_hash
        ),
        authoritative_bundle_root=authoritative_bundle_root,
        immutable_input_compatibility_sha256=compatibility_sha256,
    )


def _cell_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = dict(config["workloads"]["main_cells"])
    result.update(config["workloads"]["negative_control"])
    return result


def role_trace_seeds(
    config: Mapping[str, Any], *, role: str, trace_count: int
) -> tuple[int, ...]:
    """Return the frozen, role-disjoint arrival seeds reused by all links."""

    ranges = config["workloads"].get("role_seed_ranges")
    if not isinstance(ranges, Mapping) or set(ranges) != {"calibration", "sealed"}:
        raise ScenarioBuildError("workload role seed ranges are not frozen")
    seed_sets: dict[str, set[int]] = {}
    for current_role in ("calibration", "sealed"):
        row = ranges[current_role]
        if not isinstance(row, Mapping) or set(row) != {
            "seed_start_inclusive",
            "seed_count",
            "derivation_salt",
        }:
            raise ScenarioBuildError("role seed range schema mismatch")
        start = row["seed_start_inclusive"]
        count = row["seed_count"]
        salt = row["derivation_salt"]
        if (
            type(start) is not int
            or type(count) is not int
            or count <= 0
            or not isinstance(salt, str)
            or not salt
        ):
            raise ScenarioBuildError("invalid frozen role seed range")
        seed_sets[current_role] = set(range(start, start + count))
    if (
        config["workloads"].get("role_seed_sets_must_be_disjoint") is not True
        or seed_sets["calibration"] & seed_sets["sealed"]
    ):
        raise ScenarioBuildError("calibration/sealed arrival seed sets overlap")
    expected_count = (
        int(config["workloads"]["calibration_complete_trace_clusters"])
        if role == "calibration"
        else int(config["workloads"]["complete_trace_clusters_per_model_cell"])
    )
    if trace_count != expected_count or trace_count != len(seed_sets[role]):
        raise ScenarioBuildError("trace count does not match frozen role seed range")
    return tuple(sorted(seed_sets[role]))


def validate_aggregate_utilization(
    world_metadata: Sequence[Mapping[str, Any]],
    *,
    cells: Mapping[str, Mapping[str, Any]],
    tolerance: float,
) -> Mapping[str, float]:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ScenarioBuildError("workload utilization tolerance is invalid")
    realized_by_cell: dict[str, list[float]] = {}
    normalized_rows = True
    for row in world_metadata:
        cell = str(row["cell"])
        if cell not in cells:
            raise ScenarioBuildError("workload metadata contains an unknown cell")
        realized_by_cell.setdefault(cell, []).append(float(row["realized_offered_rho"]))
        normalized_rows = normalized_rows and (
            row.get("arrival_normalization_no_policy_or_oracle_input") is True
            and row.get("arrival_normalization_algorithm")
            == ARRIVAL_NORMALIZATION_ALGORITHM
        )
    if set(realized_by_cell) != set(cells):
        raise ScenarioBuildError("workload utilization grid is incomplete")
    aggregate = {
        cell: sum(values) / len(values) for cell, values in realized_by_cell.items()
    }
    allowed_error = 1e-12 if normalized_rows else tolerance
    blocked = {
        cell: value
        for cell, value in aggregate.items()
        if abs(value - float(cells[cell]["target_utilization"])) > allowed_error
    }
    if blocked:
        raise ScenarioBuildError(f"BLOCKED_WORKLOAD_CALIBRATION: {blocked}")
    return aggregate


def _route_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        str(row["request_id"]),
        str(row["forward_id"]),
        str(row["batch_id"]),
        int(row["layer_id"]),
        str(row["token_block_id"]),
        int(row["topk_slot"]),
        int(row["expert_id"]),
    )


def build_world(
    *,
    inputs: ValidatedInputs,
    request_ids: Sequence[str],
    trace_index: int,
    seed: int,
    cell_name: str,
    cell: Mapping[str, Any],
    arrival_census: ArrivalNormalizationCensus,
    config: Mapping[str, Any],
    link_gbps: int,
) -> tuple[ReplayWorld, dict[str, Any]]:
    if len(request_ids) != 4 or len(set(request_ids)) != 4:
        raise ScenarioBuildError("complete trace requires four unique requests")
    normalization_contract = validate_load_normalization_contract(config)
    expected_seeds = role_trace_seeds(
        config, role=inputs.role, trace_count=len(arrival_census.seeds)
    )
    expected_namespace = str(
        config["workloads"]["role_seed_ranges"][inputs.role]["derivation_salt"]
    )
    if (
        arrival_census.role != inputs.role
        or arrival_census.cell_name != cell_name
        or arrival_census.cell_fingerprint != object_sha256(cell)
        or arrival_census.seed_namespace_label != expected_namespace
        or arrival_census.seeds != expected_seeds
        or arrival_census.count_per_trace != 4 * 128
        or arrival_census.algorithm != ARRIVAL_NORMALIZATION_ALGORITHM
        or arrival_census.no_policy_or_oracle_input is not True
    ):
        raise ScenarioBuildError("arrival census does not match frozen role/cell")
    normalized_trace = arrival_census.trace_for_seed(seed)
    ep_size = int(config["topology_proxy"]["ep_size"])
    ranks_per_node = int(config["topology_proxy"]["ranks_per_node"])
    placement = {int(key): int(value) for key, value in inputs.placement["expert_to_sender"].items()}
    origins = {
        str(key): int(value) for key, value in inputs.placement["request_to_receiver"].items()
    }
    rows = [
        row
        for request_id in request_ids
        for row in inputs.route_rows_by_request[request_id]
    ]
    if len(rows) != 4 * 128 * inputs.top_k:
        raise ScenarioBuildError("trace route slice is not a complete load")
    expected_layers: dict[str, int] = {}
    for request_id in request_ids:
        layer_set = {
            int(row["layer_id"])
            for row in inputs.route_rows_by_request[request_id]
        }
        if len(layer_set) != 1:
            raise ScenarioBuildError("one request maps to multiple selected replay layers")
        expected_layers[request_id] = next(iter(layer_set))
    transport = config["contribution_transport_accounting"]
    descriptor_bytes = int(transport["descriptor_bytes_per_contribution"])
    alignment_boundary = int(transport["alignment_boundary_bytes"])
    if (
        descriptor_bytes != 16
        or alignment_boundary != 16
        or transport.get("descriptor_and_alignment_source") != "analytic_network"
        or bool(transport.get("is_measured_rdma_framing"))
    ):
        raise ScenarioBuildError("contribution transport accounting is not frozen Amendment B")
    alignment_bytes = (
        alignment_boundary
        - ((inputs.service.payload_bytes + descriptor_bytes) % alignment_boundary)
    ) % alignment_boundary
    cut_us = (
        (inputs.service.payload_bytes + descriptor_bytes + alignment_bytes)
        * 8.0
        / (float(link_gbps) * 1000.0)
    )
    if cut_us <= 0:
        raise ScenarioBuildError("analytic cut service must be positive")
    stage = StageService(
        sender_pack_us=inputs.service.sender_pack_us,
        shared_cut_us=cut_us,
        receiver_unpack_us=inputs.service.receiver_unpack_us,
        join_combine_us=inputs.service.join_combine_us,
    )
    def aggregate_resource_work(service: StageService) -> dict[str, float]:
        work: dict[str, float] = {}
        combine_joins: set[tuple[object, ...]] = set()
        for route_row in rows:
            sender = int(route_row["sender_rank"])
            receiver = int(route_row["receiver_rank"])
            resources = (
                (f"sender:{sender}:egress", service.sender_egress_us),
                (
                    f"cut:node{sender // ranks_per_node}->"
                    f"node{receiver // ranks_per_node}",
                    service.shared_cut_us,
                ),
                (f"receiver:{receiver}:ingress", service.receiver_ingress_us),
            )
            for resource, demand in resources:
                work[resource] = work.get(resource, 0.0) + demand
            join_key = (
                str(route_row["request_id"]),
                str(route_row["forward_id"]),
                str(route_row["batch_id"]),
                int(route_row["layer_id"]),
                str(route_row["token_block_id"]),
                receiver,
            )
            if join_key not in combine_joins:
                combine_joins.add(join_key)
                combine_resource = f"receiver:{receiver}:combine"
                work[combine_resource] = (
                    work.get(combine_resource, 0.0) + service.join_combine_us
                )
        return work

    resource_work = aggregate_resource_work(stage)
    bottleneck_work = max(resource_work.values())
    causal = config["topology_proxy"].get("link_sensitivity_causal_world")
    primary_link = int(config["topology_proxy"]["primary_link_gbps"])
    if not isinstance(causal, Mapping) or causal != {
        "reference_link_gbps": primary_link,
        "reuse_exact_arrival_by_block": True,
        "reuse_task_route_and_expert_ready": True,
        "recompute_only": [
            "contribution_shared_cut_service",
            "contract_analytic_transfer",
        ],
        "target_utilization_recalibration_allowed": False,
        "report_realized_utilization_only": True,
    }:
        raise ScenarioBuildError("link-sensitivity causal-world contract mismatch")
    reference_cut_us = (
        (inputs.service.payload_bytes + descriptor_bytes + alignment_bytes)
        * 8.0
        / (float(primary_link) * 1000.0)
    )
    reference_stage = StageService(
        sender_pack_us=inputs.service.sender_pack_us,
        shared_cut_us=reference_cut_us,
        receiver_unpack_us=inputs.service.receiver_unpack_us,
        join_combine_us=inputs.service.join_combine_us,
    )
    reference_bottleneck_work = max(
        aggregate_resource_work(reference_stage).values()
    )
    microcoflow_count = 4 * 128
    reference_mu = microcoflow_count / reference_bottleneck_work
    dimensionless = normalized_trace.normalized
    arrivals = ArrivalRealization(
        arrivals_us=tuple(value / reference_mu for value in dimensionless.arrivals_us),
        arrival_states=dimensionless.arrival_states,
        state_transitions_us=tuple(
            (time_us / reference_mu, old, new)
            for time_us, old, new in dimensionless.state_transitions_us
        ),
        mu_per_us=reference_mu,
        target_rho=dimensionless.target_rho,
        realized_offered_rho=(
            bottleneck_work / (dimensionless.arrivals_us[-1] / reference_mu)
        ),
    )
    block_keys = sorted(
        {
            (
                str(row["request_id"]),
                str(row["forward_id"]),
                str(row["batch_id"]),
                int(row["layer_id"]),
                str(row["token_id"]),
                str(row["token_block_id"]),
            )
            for row in rows
        },
        key=lambda value: object_sha256((seed, value)),
    )
    if len(block_keys) != microcoflow_count:
        raise ScenarioBuildError("route slice does not contain 512 token-block joins")
    block_permutation_sha256 = object_sha256(block_keys)
    arrival_by_block = dict(zip(block_keys, arrivals.arrivals_us))
    tasks: list[ReplayTask] = []
    contribution_path_max_by_join: dict[tuple[object, ...], float] = {}
    deadline_multiplier = float(
        config["closure_and_fairness"]["decision_deadline_multiplier"]
    )
    sorted_rows = sorted(rows, key=_route_key)
    for row in sorted_rows:
        block = (
            str(row["request_id"]),
            str(row["forward_id"]),
            str(row["batch_id"]),
            int(row["layer_id"]),
            str(row["token_id"]),
            str(row["token_block_id"]),
        )
        arrival = arrival_by_block[block]
        service_key = f"{int(row['layer_id'])}:{int(row['expert_id'])}"
        try:
            expert_ready_us = float(
                inputs.service.expert_ready_us_by_layer_expert[service_key]
            )
        except KeyError as exc:
            raise ScenarioBuildError(
                f"route-specific expert service key is missing: {service_key}"
            ) from exc
        contribution_path = expert_ready_us + reference_stage.total_us
        contribution_path_max_by_join[block] = max(
            contribution_path_max_by_join.get(block, 0.0), contribution_path
        )
    if set(contribution_path_max_by_join) != set(block_keys):
        raise ScenarioBuildError("isolated-path census does not cover every join")
    isolated_by_join = {
        block: contribution_path_max + stage.join_combine_us
        for block, contribution_path_max in contribution_path_max_by_join.items()
    }

    for row in sorted_rows:
        block = (
            str(row["request_id"]),
            str(row["forward_id"]),
            str(row["batch_id"]),
            int(row["layer_id"]),
            str(row["token_id"]),
            str(row["token_block_id"]),
        )
        arrival = arrival_by_block[block]
        service_key = f"{int(row['layer_id'])}:{int(row['expert_id'])}"
        expert_ready_us = float(
            inputs.service.expert_ready_us_by_layer_expert[service_key]
        )
        isolated = isolated_by_join[block]
        identity = ContributionIdentity(
            request_id=str(row["request_id"]),
            forward_id=str(row["forward_id"]),
            batch_id=str(row["batch_id"]),
            phase=str(row["phase"]),
            decode_step=int(row["decode_step"]),
            layer_id=int(row["layer_id"]),
            token_id=str(row["token_id"]),
            token_block_id=str(row["token_block_id"]),
            topk_slot=int(row["topk_slot"]),
            expert_id=int(row["expert_id"]),
            sender_rank=int(row["sender_rank"]),
            receiver_rank=int(row["receiver_rank"]),
            epoch=int(row["epoch"]),
        )
        record = ContributionRecord(
            identity=identity,
            model_revision=inputs.model_revision,
            valid=row["valid"],
            arrival_us=arrival,
            ready_us=arrival + expert_ready_us,
            service_us=stage.total_us,
            deadline_us=arrival + deadline_multiplier * isolated,
            payload_bytes=inputs.service.payload_bytes,
            descriptor_bytes=descriptor_bytes,
            alignment_bytes=alignment_bytes,
            source_tag="derived_from_measured_lut",
        )
        sender = identity.sender_rank
        receiver = identity.receiver_rank
        tasks.append(
            ReplayTask(
                contribution=record,
                stage_service=stage,
                sender_egress_resource=f"sender:{sender}:egress",
                shared_cut_resource=(
                    f"cut:node{sender // ranks_per_node}->node{receiver // ranks_per_node}"
                ),
                receiver_ingress_resource=f"receiver:{receiver}:ingress",
            )
        )
    all_joins = frozenset(task.join_identity for task in tasks)
    audit = validate_full_background(
        [task.contribution for task in tasks],
        top_k=inputs.top_k,
        num_experts=inputs.num_experts,
        ep_size=ep_size,
        expected_request_ids=request_ids,
        expected_token_blocks_per_request=128,
        expert_to_sender=placement,
        request_to_receiver={request_id: origins[request_id] for request_id in request_ids},
        expected_layer_by_request=expected_layers,
        expected_model_revision=inputs.model_revision,
        score_join_identities=all_joins,
    )
    trace_id = (
        f"{inputs.role}/{inputs.model_key}/{cell_name}/link{link_gbps}/"
        f"trace-{trace_index:02d}/seed-{seed}"
    )
    world = ReplayWorld(
        trace_id=trace_id,
        workload_seed=seed,
        model_key=inputs.model_key,
        model_revision=inputs.model_revision,
        cell=cell_name,
        top_k=inputs.top_k,
        num_experts=inputs.num_experts,
        ep_size=ep_size,
        ranks_per_node=ranks_per_node,
        tasks=tuple(sorted(tasks, key=lambda task: task.task_id)),
        expected_request_ids=tuple(request_ids),
        expert_to_sender=placement,
        request_to_receiver={request_id: origins[request_id] for request_id in request_ids},
        expected_layer_by_request=expected_layers,
        scored_joins=all_joins,
        full_load_audit=audit,
    )
    isolated_values = tuple(isolated_by_join.values())
    metadata = {
        "trace_id": trace_id,
        "trace_index": trace_index,
        "workload_seed": seed,
        "cell": cell_name,
        "link_gbps": link_gbps,
        "request_ids": list(request_ids),
        "expected_layer_by_request": expected_layers,
        "request_to_receiver": world.request_to_receiver,
        "task_count": len(tasks),
        "join_count": len(all_joins),
        "task_fingerprint": world.task_fingerprint,
        "service_fingerprint": world.service_fingerprint,
        "resource_demand_fingerprint": world.resource_demand_fingerprint,
        "world_causal_arrival_fingerprint": causal_arrival_fingerprint(world),
        "causal_arrival_fingerprint": object_sha256(
            {
                "world_causal_arrival_fingerprint": causal_arrival_fingerprint(world),
                "normalized_arrival_transition_fingerprint": (
                    normalized_trace.normalized_fingerprint
                ),
                "ctmc_state_transition_fingerprint": object_sha256(
                    [list(row) for row in arrivals.state_transitions_us]
                ),
                "block_permutation_fingerprint": block_permutation_sha256,
            }
        ),
        "arrival_schedule_fingerprint": arrival_schedule_fingerprint(world),
        "block_permutation_fingerprint": block_permutation_sha256,
        "ctmc_state_transition_fingerprint": object_sha256(
            [list(row) for row in arrivals.state_transitions_us]
        ),
        "raw_arrival_transition_fingerprint": normalized_trace.raw_fingerprint,
        "normalized_arrival_transition_fingerprint": (
            normalized_trace.normalized_fingerprint
        ),
        "raw_arrival_census_fingerprint": arrival_census.raw_census_fingerprint,
        "normalized_arrival_census_fingerprint": (
            arrival_census.normalized_census_fingerprint
        ),
        "arrival_normalization_algorithm": arrival_census.algorithm,
        "arrival_normalization_claim_label": normalization_contract["claim_label"],
        "arrival_normalization_contract_fingerprint": object_sha256(
            normalization_contract
        ),
        "arrival_time_dilation_factor": arrival_census.time_dilation_factor,
        "raw_aggregate_realized_rho": arrival_census.raw_aggregate_rho,
        "normalized_primary_aggregate_realized_rho": (
            arrival_census.normalized_aggregate_rho
        ),
        "arrival_normalization_no_policy_or_oracle_input": (
            arrival_census.no_policy_or_oracle_input
        ),
        "NO_POLICY_OR_ORACLE_INPUT": arrival_census.no_policy_or_oracle_input,
        "arrival_census_seeds": list(arrival_census.seeds),
        "score_mask_fingerprint": world.score_mask_fingerprint,
        "bottleneck_work_us": bottleneck_work,
        "arrival_reference_link_gbps": primary_link,
        "arrival_reference_bottleneck_work_us": reference_bottleneck_work,
        "arrival_reused_across_link_variants": True,
        "seed_role": inputs.role,
        "seed_namespace_label": arrival_census.seed_namespace_label,
        "seed_namespace_affects_prng": False,
        "mu_per_us": arrivals.mu_per_us,
        "target_rho": arrivals.target_rho,
        "realized_offered_rho": arrivals.realized_offered_rho,
        "arrival_process": str(cell["arrival_process"]),
        "arrival_state_counts": {
            state: arrivals.arrival_states.count(state)
            for state in sorted(set(arrivals.arrival_states))
        },
        "ctmc_state_transitions": [list(row) for row in arrivals.state_transitions_us],
        "isolated_path_median_us": sorted(isolated_values)[len(isolated_values) // 2],
        "isolated_path_min_us": min(isolated_values),
        "isolated_path_max_us": max(isolated_values),
        "isolated_path_values_us": list(isolated_values),
        "decision_deadline_multiplier": deadline_multiplier,
        "decision_deadline_reference_link_gbps": primary_link,
    }
    return world, metadata


def build_link_sensitivity_world_from_primary(
    *,
    inputs: ValidatedInputs,
    primary_world: ReplayWorld,
    primary_metadata: Mapping[str, Any],
    primary_scenario_tree_sha256: str,
    config: Mapping[str, Any],
    link_gbps: int,
) -> tuple[ReplayWorld, dict[str, Any]]:
    """Change only link-causal service using an already-built primary world.

    This path deliberately has no seed, cell, or generator argument.  It cannot
    call the raw arrival generator or recompute an Amendment-N normalization.
    """

    contract = validate_load_normalization_contract(config)
    primary_link = int(contract["primary_reference_link_gbps"])
    allowed_sensitivities = {
        int(value) for value in config["topology_proxy"]["link_sensitivity_gbps"]
    }
    if link_gbps not in allowed_sensitivities or link_gbps == primary_link:
        raise ScenarioBuildError("link sensitivity transform received invalid link")
    if (
        not isinstance(primary_scenario_tree_sha256, str)
        or len(primary_scenario_tree_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in primary_scenario_tree_sha256
        )
    ):
        raise ScenarioBuildError("sensitivity source tree hash is invalid")
    if (
        int(primary_metadata.get("link_gbps", -1)) != primary_link
        or primary_world.model_key != inputs.model_key
        or primary_world.model_revision != inputs.model_revision
        or primary_metadata.get("arrival_reference_link_gbps") != primary_link
        or primary_metadata.get("decision_deadline_reference_link_gbps")
        != primary_link
        or primary_metadata.get("arrival_normalization_claim_label")
        != contract["claim_label"]
        or primary_metadata.get("arrival_normalization_contract_fingerprint")
        != object_sha256(contract)
        or primary_metadata.get("arrival_normalization_no_policy_or_oracle_input")
        is not True
        or primary_metadata.get("NO_POLICY_OR_ORACLE_INPUT") is not True
    ):
        raise ScenarioBuildError("sensitivity source is not a bound primary world")

    transformed_tasks: list[ReplayTask] = []
    for task in primary_world.tasks:
        record = task.contribution
        expected_primary_cut = (
            record.wire_bytes * 8.0 / (float(primary_link) * 1000.0)
        )
        if not math.isclose(
            task.stage_service.shared_cut_us,
            expected_primary_cut,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ScenarioBuildError("sensitivity source task is not primary-link service")
        sensitivity_stage = replace(
            task.stage_service,
            shared_cut_us=(
                record.wire_bytes * 8.0 / (float(link_gbps) * 1000.0)
            ),
        )
        transformed_tasks.append(
            ReplayTask(
                contribution=replace(
                    record,
                    service_us=sensitivity_stage.total_us,
                ),
                stage_service=sensitivity_stage,
                sender_egress_resource=task.sender_egress_resource,
                shared_cut_resource=task.shared_cut_resource,
                receiver_ingress_resource=task.receiver_ingress_resource,
            )
        )
    transformed_tasks_tuple = tuple(
        sorted(transformed_tasks, key=lambda task: task.task_id)
    )
    joins = frozenset(task.join_identity for task in transformed_tasks_tuple)
    audit = validate_full_background(
        [task.contribution for task in transformed_tasks_tuple],
        top_k=primary_world.top_k,
        num_experts=primary_world.num_experts,
        ep_size=primary_world.ep_size,
        expected_request_ids=primary_world.expected_request_ids,
        expected_token_blocks_per_request=128,
        expert_to_sender=primary_world.expert_to_sender,
        request_to_receiver=primary_world.request_to_receiver,
        expected_layer_by_request=primary_world.expected_layer_by_request,
        expected_model_revision=primary_world.model_revision,
        score_join_identities=joins,
    )
    trace_id = (
        f"{inputs.role}/{inputs.model_key}/{primary_world.cell}/link{link_gbps}/"
        f"trace-{int(primary_metadata['trace_index']):02d}/"
        f"seed-{primary_world.workload_seed}"
    )
    world = ReplayWorld(
        trace_id=trace_id,
        workload_seed=primary_world.workload_seed,
        model_key=primary_world.model_key,
        model_revision=primary_world.model_revision,
        cell=primary_world.cell,
        top_k=primary_world.top_k,
        num_experts=primary_world.num_experts,
        ep_size=primary_world.ep_size,
        ranks_per_node=primary_world.ranks_per_node,
        tasks=transformed_tasks_tuple,
        expected_request_ids=primary_world.expected_request_ids,
        expert_to_sender=primary_world.expert_to_sender,
        request_to_receiver=primary_world.request_to_receiver,
        expected_layer_by_request=primary_world.expected_layer_by_request,
        scored_joins=primary_world.scored_joins,
        full_load_audit=audit,
    )

    resource_work: dict[str, float] = {}
    combined: set[tuple[object, ...]] = set()
    for task in transformed_tasks_tuple:
        for resource, demand in (
            (task.sender_egress_resource, task.stage_service.sender_egress_us),
            (task.shared_cut_resource, task.stage_service.shared_cut_us),
            (task.receiver_ingress_resource, task.stage_service.receiver_ingress_us),
        ):
            resource_work[resource] = resource_work.get(resource, 0.0) + demand
        join = task.join_identity.canonical_tuple()
        if join not in combined:
            combined.add(join)
            combine_resource = f"receiver:{task.identity.receiver_rank}:combine"
            resource_work[combine_resource] = (
                resource_work.get(combine_resource, 0.0)
                + task.stage_service.join_combine_us
            )
    bottleneck_work = max(resource_work.values())
    end_arrival = max(
        siblings[0].contribution.arrival_us for siblings in world.joins.values()
    )
    metadata = dict(primary_metadata)
    metadata.update(
        {
            "trace_id": trace_id,
            "link_gbps": link_gbps,
            "task_fingerprint": world.task_fingerprint,
            "service_fingerprint": world.service_fingerprint,
            "resource_demand_fingerprint": world.resource_demand_fingerprint,
            "world_causal_arrival_fingerprint": causal_arrival_fingerprint(world),
            "arrival_schedule_fingerprint": arrival_schedule_fingerprint(world),
            "score_mask_fingerprint": world.score_mask_fingerprint,
            "bottleneck_work_us": bottleneck_work,
            "realized_offered_rho": bottleneck_work / end_arrival,
            "arrival_reused_across_link_variants": True,
            "sensitivity_source_primary_trace_id": primary_world.trace_id,
            "sensitivity_source_primary_scenario_tree_sha256": (
                primary_scenario_tree_sha256
            ),
            "sensitivity_source_primary_causal_fingerprint": primary_metadata[
                "causal_arrival_fingerprint"
            ],
        }
    )
    matched_causal = object_sha256(
        {
            "world_causal_arrival_fingerprint": metadata[
                "world_causal_arrival_fingerprint"
            ],
            "normalized_arrival_transition_fingerprint": metadata[
                "normalized_arrival_transition_fingerprint"
            ],
            "ctmc_state_transition_fingerprint": metadata[
                "ctmc_state_transition_fingerprint"
            ],
            "block_permutation_fingerprint": metadata[
                "block_permutation_fingerprint"
            ],
        }
    )
    metadata["causal_arrival_fingerprint"] = matched_causal
    invariant_fields = (
        "arrival_schedule_fingerprint",
        "block_permutation_fingerprint",
        "raw_arrival_transition_fingerprint",
        "normalized_arrival_transition_fingerprint",
        "ctmc_state_transition_fingerprint",
        "raw_arrival_census_fingerprint",
        "normalized_arrival_census_fingerprint",
        "arrival_time_dilation_factor",
        "isolated_path_values_us",
        "decision_deadline_reference_link_gbps",
    )
    if (
        metadata["causal_arrival_fingerprint"]
        != primary_metadata["causal_arrival_fingerprint"]
        or any(metadata[field] != primary_metadata[field] for field in invariant_fields)
        or any(
            transformed.contribution.deadline_us
            != original.contribution.deadline_us
            for transformed, original in zip(
                transformed_tasks_tuple, primary_world.tasks
            )
        )
    ):
        raise ScenarioBuildError("link sensitivity changed primary causal contract")
    return world, metadata


def _task_row(world: ReplayWorld, task: ReplayTask) -> dict[str, Any]:
    return {
        "schema_version": "ric-task-trace-v1",
        "trace_id": world.trace_id,
        "workload_seed": world.workload_seed,
        "model_key": world.model_key,
        "model_revision": world.model_revision,
        "cell": world.cell,
        "top_k": world.top_k,
        "num_experts": world.num_experts,
        "ep_size": world.ep_size,
        "ranks_per_node": world.ranks_per_node,
        "identity": asdict(task.identity),
        "valid": task.contribution.valid,
        "arrival_us": task.contribution.arrival_us,
        "ready_us": task.contribution.ready_us,
        "service_us": task.contribution.service_us,
        "deadline_us": task.contribution.deadline_us,
        "payload_bytes": task.contribution.payload_bytes,
        "descriptor_bytes": task.contribution.descriptor_bytes,
        "alignment_bytes": task.contribution.alignment_bytes,
        "byte_sources": {
            "payload": "measured_5090_cuda",
            "descriptor": "analytic_network",
            "alignment": "analytic_network",
        },
        "source_tag": task.contribution.source_tag,
        "stage_service": asdict(task.stage_service),
        "resources": {
            "sender_egress": task.sender_egress_resource,
            "shared_cut": task.shared_cut_resource,
            "receiver_ingress": task.receiver_ingress_resource,
        },
    }


def write_scenarios(
    *,
    output_dir: Path,
    inputs: ValidatedInputs,
    worlds: Sequence[ReplayWorld],
    world_metadata: Sequence[Mapping[str, Any]],
    config_path: Path,
    protocol_path: Path,
    mode: str,
    link_gbps: int,
    signoff_path: Path | None = None,
) -> Mapping[str, Any]:
    if len(worlds) != len(world_metadata) or not worlds:
        raise ScenarioBuildError("cannot write incomplete scenario collection")
    frozen_config = _load_config(config_path)
    primary_link = int(frozen_config["topology_proxy"]["primary_link_gbps"])
    service_payload = asdict(inputs.service)
    if link_gbps != primary_link:
        scaled_points: dict[str, dict[str, float]] = {}
        for count, point in inputs.service.control_tax_by_record_count.items():
            adjusted = {key: float(value) for key, value in point.items()}
            adjusted["transfer_us"] *= primary_link / float(link_gbps)
            scaled_points[str(count)] = adjusted
        service_payload["control_tax_by_record_count"] = scaled_points
        service_payload["control_tax_source_id"] = object_sha256(
            {
                "rule": "raw_median_minus_same_count_empty_harness",
                "non_grid_rule": "exact_1_to_255_no_interpolation_or_extrapolation",
                "points": scaled_points,
                "service_lut_sha256": inputs.service.summary_sha256,
            }
        )
    with atomic_output_directory(output_dir) as temporary:
        scenario_signoff_sha256 = None
        if mode == "formal":
            try:
                scenario_signoff_sha256 = materialize_verified_signoff(
                    signoff_path, temporary
                )
            except FormalProvenanceError as exc:
                raise ScenarioBuildError(str(exc)) from exc
        elif signoff_path is not None:
            raise ScenarioBuildError("development scenario cannot carry a signoff")
        trace_path = temporary / "task_trace.jsonl"
        with trace_path.open("x", encoding="utf-8") as handle:
            for world in worlds:
                for task in world.tasks:
                    handle.write(json.dumps(_task_row(world, task), sort_keys=True) + "\n")
        realized_by_cell: dict[str, list[float]] = {}
        normalization_by_cell: dict[str, dict[str, Any]] = {}
        isolated = []
        sensitivity_source_hashes: set[str] = set()
        for row in world_metadata:
            cell_name = str(row["cell"])
            realized_by_cell.setdefault(cell_name, []).append(
                float(row["realized_offered_rho"])
            )
            normalization = {
                "algorithm": row["arrival_normalization_algorithm"],
                "claim_label": row["arrival_normalization_claim_label"],
                "contract_fingerprint": row[
                    "arrival_normalization_contract_fingerprint"
                ],
                "seed_namespace_label": row["seed_namespace_label"],
                "seed_namespace_affects_prng": row[
                    "seed_namespace_affects_prng"
                ],
                "time_dilation_factor": row["arrival_time_dilation_factor"],
                "raw_aggregate_realized_rho": row[
                    "raw_aggregate_realized_rho"
                ],
                "normalized_primary_aggregate_realized_rho": row[
                    "normalized_primary_aggregate_realized_rho"
                ],
                "raw_census_fingerprint": row[
                    "raw_arrival_census_fingerprint"
                ],
                "normalized_census_fingerprint": row[
                    "normalized_arrival_census_fingerprint"
                ],
                "no_policy_or_oracle_input": row[
                    "arrival_normalization_no_policy_or_oracle_input"
                ],
                "NO_POLICY_OR_ORACLE_INPUT": row["NO_POLICY_OR_ORACLE_INPUT"],
                "seeds": row["arrival_census_seeds"],
            }
            prior = normalization_by_cell.setdefault(cell_name, normalization)
            if prior != normalization:
                raise ScenarioBuildError(
                    "one role/cell uses multiple arrival normalization censuses"
                )
            isolated.extend(float(value) for value in row["isolated_path_values_us"])
            source_hash = row.get("sensitivity_source_primary_scenario_tree_sha256")
            if source_hash is not None:
                sensitivity_source_hashes.add(str(source_hash))
        if link_gbps == primary_link:
            if sensitivity_source_hashes:
                raise ScenarioBuildError("primary scenario has a sensitivity source")
            sensitivity_source_tree_sha256 = None
        else:
            if (
                len(sensitivity_source_hashes) != 1
                or len(next(iter(sensitivity_source_hashes), "")) != 64
            ):
                raise ScenarioBuildError(
                    "sensitivity worlds do not bind one primary scenario tree"
                )
            sensitivity_source_tree_sha256 = next(iter(sensitivity_source_hashes))
        aggregate = {
            cell: sum(values) / len(values) for cell, values in realized_by_cell.items()
        }
        payload = add_self_hash(
            {
                "schema_version": "ric-scenario-tree-v1",
                "status": "SCENARIO_ONLY" if mode == "formal" else "NOT_TESTED",
                "scientific_result": False,
                "evidence_boundary": "L2_CALIBRATED_VIRTUAL_EP_NOT_RDMA",
                "mode": mode,
                "role": inputs.role,
                "model_key": inputs.model_key,
                "model_revision": inputs.model_revision,
                "model_tree_manifest_sha256": inputs.service.model_tree_manifest_sha256,
                "gpu_environment_identity": inputs.service.gpu_environment_identity,
                "top_k": inputs.top_k,
                "num_experts": inputs.num_experts,
                "ep_size": int(inputs.placement["ep_size"]),
                "ranks_per_node": int(inputs.placement["ranks_per_node"]),
                "expert_to_sender": inputs.placement["expert_to_sender"],
                "link_gbps": link_gbps,
                "sensitivity_source_primary_scenario_tree_sha256": (
                    sensitivity_source_tree_sha256
                ),
                "link_sensitivity_causal_world": frozen_config["topology_proxy"][
                    "link_sensitivity_causal_world"
                ],
                "target_utilization_calibration_enforced": (
                    link_gbps == primary_link
                ),
                "role_seed_range": frozen_config["workloads"]["role_seed_ranges"][
                    inputs.role
                ],
                "protocol_sha256": sha256_file(protocol_path),
                "config_sha256": sha256_file(config_path),
                "consumer_amendment_sha256": inputs.consumer_amendment_sha256,
                "build_scenarios_source_sha256": _source_sha256(),
                "scenario_producer_signoff_sha256": scenario_signoff_sha256,
                "data_manifest_sha256": inputs.data_manifest["manifest_sha256"],
                "data_producer_signoff_sha256": inputs.data_manifest.get(
                    "signoff_sha256"
                ),
                "service_calibration_data_manifest_sha256": (
                    inputs.service_calibration_data_manifest_sha256
                ),
                "service_calibration_data_manifest_file_sha256": (
                    inputs.service_calibration_data_manifest_file_sha256
                ),
                "service_calibration_selected_list_sha256": (
                    inputs.service_calibration_selected_list_sha256
                ),
                "service_calibration_data_producer_signoff_sha256": (
                    inputs.service_calibration_data_producer_signoff_sha256
                ),
                "sealed_input_attestation_sha256": (
                    inputs.sealed_input_attestation_sha256
                ),
                "sealed_input_historical_run_experiment_source_sha256": (
                    inputs.sealed_input_historical_run_experiment_source_sha256
                ),
                "sealed_input_historical_calibration_lock_sha256": (
                    inputs.sealed_input_historical_calibration_lock_sha256
                ),
                "sealed_input_historical_calibration_signoff_sha256": (
                    inputs.sealed_input_historical_calibration_signoff_sha256
                ),
                "sealed_global_reservation_file_sha256": (
                    inputs.sealed_global_reservation_file_sha256
                ),
                "sealed_global_consumption_file_sha256": (
                    inputs.sealed_global_consumption_file_sha256
                ),
                "historical_reviewed_source_snapshot_sha256": (
                    inputs.historical_reviewed_source_snapshot_sha256
                ),
                "pre_outcome_attestation_sha256": (
                    inputs.pre_outcome_attestation_sha256
                ),
                "pre_outcome_producer_signoff_file_sha256": (
                    inputs.pre_outcome_producer_signoff_file_sha256
                ),
                "pre_outcome_producer_signoff_self_hash": (
                    inputs.pre_outcome_producer_signoff_self_hash
                ),
                "authoritative_bundle_root": inputs.authoritative_bundle_root,
                "immutable_input_compatibility_sha256": (
                    inputs.immutable_input_compatibility_sha256
                ),
                "route_capture_manifest_sha256": inputs.route_metadata["manifest_sha256"],
                "route_trace_sha256": inputs.route_metadata["route_trace_sha256"],
                "capture_routes_source_sha256": inputs.route_metadata[
                    "capture_routes_source_sha256"
                ],
                "route_producer_signoff_sha256": inputs.route_metadata[
                    "signoff_sha256"
                ],
                "placement_manifest_sha256": inputs.placement["manifest_sha256"],
                "service_lut_metadata_sha256": inputs.service.metadata_sha256,
                "service_lut_sha256": inputs.service.summary_sha256,
                "service_lut_raw_sha256": inputs.service.raw_sha256,
                "measure_service_lut_source_sha256": (
                    inputs.service.producer_source_sha256
                ),
                "service_lut_producer_signoff_sha256": (
                    inputs.service.producer_signoff_sha256
                ),
                "payload_bytes": inputs.service.payload_bytes,
                "payload_layout_sha256": inputs.service.payload_layout_sha256,
                "contribution_transport_accounting": {
                    "descriptor_bytes_per_contribution": worlds[0].tasks[0].contribution.descriptor_bytes,
                    "alignment_bytes_per_contribution": worlds[0].tasks[0].contribution.alignment_bytes,
                    "descriptor_and_alignment_source": "analytic_network",
                },
                "service_surface": service_payload,
                "task_trace_sha256": sha256_file(trace_path),
                "world_count": len(worlds),
                "worlds": list(world_metadata),
                "arrival_normalization_by_cell": normalization_by_cell,
                "aggregate_realized_offered_rho": aggregate,
                "calibration_isolated_path_median_us": (
                    sorted(isolated)[len(isolated) // 2]
                    if inputs.role == "calibration"
                    else None
                ),
            }
        )
        (temporary / "scenario_tree.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return payload


def load_worlds(
    scenario_dir: Path, *, expected_role: str, mode: str
) -> tuple[Mapping[str, Any], tuple[ReplayWorld, ...]]:
    guard_mode_role(mode, expected_role)
    tree = _read_self_hashed_json(
        scenario_dir / "scenario_tree.json", schema_version="ric-scenario-tree-v1"
    )
    if tree.get("role") != expected_role or tree.get("mode") != mode:
        raise ScenarioBuildError("scenario role/mode mismatch")
    expected_status = "SCENARIO_ONLY" if mode == "formal" else "NOT_TESTED"
    if tree.get("status") != expected_status:
        raise ScenarioBuildError("scenario status mismatch")
    if tree.get("build_scenarios_source_sha256") != _source_sha256():
        raise ScenarioBuildError("scenario producer source mismatch")
    primary_link = int(
        tree["link_sensitivity_causal_world"]["reference_link_gbps"]
    )
    source_tree_sha = tree.get("sensitivity_source_primary_scenario_tree_sha256")
    if int(tree.get("link_gbps", -1)) == primary_link:
        if source_tree_sha is not None:
            raise ScenarioBuildError("primary scenario binds a sensitivity source")
    elif (
        not isinstance(source_tree_sha, str)
        or len(source_tree_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_tree_sha)
    ):
        raise ScenarioBuildError("sensitivity scenario lacks primary tree hash")
    producer_signoff_sha256 = tree.get("scenario_producer_signoff_sha256")
    if mode == "formal":
        embedded_signoff = scenario_dir / EMBEDDED_PRODUCER_SIGNOFF
        if (
            not is_sha256(producer_signoff_sha256)
            or not embedded_signoff.is_file()
            or sha256_file(embedded_signoff) != producer_signoff_sha256
        ):
            raise ScenarioBuildError("formal scenario lacks embedded producer signoff")
        transitive_hashes = {
            field: tree.get(field)
            for field in (
                "data_manifest_sha256",
                "data_producer_signoff_sha256",
                "route_capture_manifest_sha256",
                "route_trace_sha256",
                "placement_manifest_sha256",
                "service_lut_metadata_sha256",
                "service_lut_sha256",
                "service_lut_raw_sha256",
                "model_tree_manifest_sha256",
                "capture_routes_source_sha256",
                "route_producer_signoff_sha256",
                "measure_service_lut_source_sha256",
                "service_lut_producer_signoff_sha256",
                "service_calibration_data_manifest_sha256",
                "service_calibration_data_manifest_file_sha256",
                "service_calibration_selected_list_sha256",
                "service_calibration_data_producer_signoff_sha256",
                "sealed_input_attestation_sha256",
                "sealed_input_historical_run_experiment_source_sha256",
                "sealed_input_historical_calibration_lock_sha256",
                "sealed_input_historical_calibration_signoff_sha256",
                "sealed_global_reservation_file_sha256",
                "sealed_global_consumption_file_sha256",
                "historical_reviewed_source_snapshot_sha256",
                "pre_outcome_attestation_sha256",
                "pre_outcome_producer_signoff_file_sha256",
                "pre_outcome_producer_signoff_self_hash",
                "authoritative_bundle_root",
                "immutable_input_compatibility_sha256",
            )
        }
        if source_tree_sha is not None:
            transitive_hashes["primary_scenario_tree_sha256"] = source_tree_sha
        _require_formal_signoff(
            embedded_signoff,
            config_sha256=str(tree.get("config_sha256")),
            protocol_sha256=str(tree.get("protocol_sha256")),
            role=expected_role,
            model_key=str(tree.get("model_key")),
            link_gbps=int(tree.get("link_gbps", -1)),
            consumer_amendment_sha256=str(
                tree.get("consumer_amendment_sha256")
            ),
            transitive_hashes=transitive_hashes,
        )
    elif producer_signoff_sha256 is not None:
        raise ScenarioBuildError("development scenario claims a producer signoff")
    task_path = scenario_dir / "task_trace.jsonl"
    if tree.get("task_trace_sha256") != sha256_file(task_path):
        raise ScenarioBuildError("task trace hash mismatch")
    world_meta = {str(row["trace_id"]): row for row in tree["worlds"]}
    normalization_by_cell = tree.get("arrival_normalization_by_cell")
    if not isinstance(normalization_by_cell, Mapping) or not normalization_by_cell:
        raise ScenarioBuildError("scenario tree lacks arrival normalization census")
    grouped: dict[str, list[ReplayTask]] = {trace_id: [] for trace_id in world_meta}
    row_metadata: dict[str, tuple[int, int, int]] = {}
    service_surface = tree.get("service_surface")
    if not isinstance(service_surface, Mapping) or not isinstance(
        service_surface.get("expert_ready_us_by_layer_expert"), Mapping
    ):
        raise ScenarioBuildError("scenario tree lacks route-specific service surface")
    route_specific_service = service_surface["expert_ready_us_by_layer_expert"]
    with task_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = loads_json_mapping_strict(
                    line, label=f"task trace line {line_number}"
                )
            except FormalProvenanceError as exc:
                raise ScenarioBuildError(f"invalid task trace line {line_number}") from exc
            if row.get("schema_version") != "ric-task-trace-v1":
                raise ScenarioBuildError("task trace schema mismatch")
            if type(row.get("valid")) is not bool:
                raise ScenarioBuildError("task trace valid field must be exact bool")
            if row.get("byte_sources") != {
                "payload": "measured_5090_cuda",
                "descriptor": "analytic_network",
                "alignment": "analytic_network",
            }:
                raise ScenarioBuildError("task trace byte source ledger mismatch")
            trace_id = str(row["trace_id"])
            if trace_id not in grouped:
                raise ScenarioBuildError("task references unknown scenario world")
            current_meta = (
                int(row["workload_seed"]),
                int(row["ep_size"]),
                int(row["ranks_per_node"]),
            )
            prior_meta = row_metadata.setdefault(trace_id, current_meta)
            if prior_meta != current_meta:
                raise ScenarioBuildError("task rows disagree on seed/topology")
            identity = ContributionIdentity(**row["identity"])
            stage = StageService(**row["stage_service"])
            record = ContributionRecord(
                identity=identity,
                model_revision=str(row["model_revision"]),
                valid=row["valid"],
                arrival_us=float(row["arrival_us"]),
                ready_us=float(row["ready_us"]),
                service_us=float(row["service_us"]),
                deadline_us=float(row["deadline_us"]),
                payload_bytes=int(row["payload_bytes"]),
                descriptor_bytes=int(row["descriptor_bytes"]),
                alignment_bytes=int(row["alignment_bytes"]),
                source_tag=str(row["source_tag"]),
            )
            service_key = f"{identity.layer_id}:{identity.expert_id}"
            if service_key not in route_specific_service or not math.isclose(
                record.ready_us - record.arrival_us,
                float(route_specific_service[service_key]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ScenarioBuildError("task ready offset differs from exact LUT key")
            if (
                not math.isclose(
                    stage.sender_pack_us,
                    float(service_surface["sender_pack_us"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    stage.receiver_unpack_us,
                    float(service_surface["receiver_unpack_us"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    stage.join_combine_us,
                    float(service_surface["join_combine_us"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ScenarioBuildError("task data-path service differs from LUT surface")
            expected_cut = (
                (record.payload_bytes + record.descriptor_bytes + record.alignment_bytes)
                * 8.0
                / (float(tree["link_gbps"]) * 1000.0)
            )
            if not math.isclose(
                stage.shared_cut_us, expected_cut, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ScenarioBuildError("task cut service differs from byte ledger")
            resources = row.get("resources")
            expected_resources = {
                "sender_egress": f"sender:{identity.sender_rank}:egress",
                "shared_cut": (
                    f"cut:node{identity.sender_rank // current_meta[2]}"
                    f"->node{identity.receiver_rank // current_meta[2]}"
                ),
                "receiver_ingress": f"receiver:{identity.receiver_rank}:ingress",
            }
            if resources != expected_resources:
                raise ScenarioBuildError("task resource path differs from full identity")
            grouped[trace_id].append(
                ReplayTask(
                    contribution=record,
                    stage_service=stage,
                    sender_egress_resource=str(row["resources"]["sender_egress"]),
                    shared_cut_resource=str(row["resources"]["shared_cut"]),
                    receiver_ingress_resource=str(row["resources"]["receiver_ingress"]),
                )
            )
    worlds: list[ReplayWorld] = []
    for trace_id, metadata in world_meta.items():
        metadata_source_tree = metadata.get(
            "sensitivity_source_primary_scenario_tree_sha256"
        )
        if int(tree["link_gbps"]) == primary_link:
            if metadata_source_tree is not None:
                raise ScenarioBuildError(
                    "primary world metadata binds a sensitivity source"
                )
        elif metadata_source_tree != source_tree_sha:
            raise ScenarioBuildError(
                "sensitivity world/tree primary-source hash mismatch"
            )
        tasks = tuple(sorted(grouped[trace_id], key=lambda task: task.task_id))
        if not tasks or trace_id not in row_metadata:
            raise ScenarioBuildError("scenario world has no task rows")
        request_ids = tuple(str(value) for value in metadata["request_ids"])
        placement = {
            int(key): int(value)
            for key, value in tree["expert_to_sender"].items()
        }
        workload_seed, ep_size, ranks_per_node = row_metadata[trace_id]
        if ep_size != int(tree["ep_size"]) or ranks_per_node != int(tree["ranks_per_node"]):
            raise ScenarioBuildError("task/tree topology mismatch")
        recomputed_placement = {
            expert: expert_sender(expert, int(tree["num_experts"]), ep_size)
            for expert in range(int(tree["num_experts"]))
        }
        if placement != recomputed_placement:
            raise ScenarioBuildError("serialized placement is not frozen contiguous placement")
        origins = {
            request_id: int(metadata["request_to_receiver"][request_id])
            for request_id in request_ids
        }
        expected_layers = {
            request_id: int(metadata["expected_layer_by_request"][request_id])
            for request_id in request_ids
        }
        joins = frozenset(task.join_identity for task in tasks)
        audit = validate_full_background(
            [task.contribution for task in tasks],
            top_k=int(tree["top_k"]),
            num_experts=int(tree["num_experts"]),
            ep_size=ep_size,
            expected_request_ids=request_ids,
            expected_token_blocks_per_request=128,
            expert_to_sender=placement,
            request_to_receiver=origins,
            expected_layer_by_request=expected_layers,
            expected_model_revision=str(tree["model_revision"]),
            score_join_identities=joins,
        )
        world = ReplayWorld(
            trace_id=trace_id,
            workload_seed=workload_seed,
            model_key=str(tree["model_key"]),
            model_revision=str(tree["model_revision"]),
            cell=str(metadata["cell"]),
            top_k=int(tree["top_k"]),
            num_experts=int(tree["num_experts"]),
            ep_size=ep_size,
            ranks_per_node=ranks_per_node,
            tasks=tasks,
            expected_request_ids=request_ids,
            expert_to_sender=placement,
            request_to_receiver=origins,
            expected_layer_by_request=expected_layers,
            scored_joins=joins,
            full_load_audit=audit,
        )
        base_causal = causal_arrival_fingerprint(world)
        if metadata.get("world_causal_arrival_fingerprint") != base_causal:
            raise ScenarioBuildError(
                "reloaded world fingerprint mismatch: world_causal_arrival_fingerprint"
            )
        normalized_trace_fingerprint = metadata.get(
            "normalized_arrival_transition_fingerprint"
        )
        serialized_block_permutation_fingerprint = metadata.get(
            "block_permutation_fingerprint"
        )
        transition_fingerprint = object_sha256(
            metadata.get("ctmc_state_transitions")
        )
        if not all(
            isinstance(value, str) and len(value) == 64
            for value in (
                normalized_trace_fingerprint,
                serialized_block_permutation_fingerprint,
                metadata.get("raw_arrival_transition_fingerprint"),
                metadata.get("ctmc_state_transition_fingerprint"),
            )
        ):
            raise ScenarioBuildError("arrival normalization fingerprint is invalid")
        if (
            serialized_block_permutation_fingerprint
            != block_permutation_fingerprint(world)
            or metadata.get("ctmc_state_transition_fingerprint")
            != transition_fingerprint
        ):
            raise ScenarioBuildError(
                "arrival transition or block permutation fingerprint mismatch"
            )
        matched_causal = object_sha256(
            {
                "world_causal_arrival_fingerprint": base_causal,
                "normalized_arrival_transition_fingerprint": (
                    normalized_trace_fingerprint
                ),
                "ctmc_state_transition_fingerprint": transition_fingerprint,
                "block_permutation_fingerprint": (
                    serialized_block_permutation_fingerprint
                ),
            }
        )
        cell_normalization = normalization_by_cell.get(world.cell)
        expected_cell_normalization = {
            "algorithm": metadata.get("arrival_normalization_algorithm"),
            "claim_label": metadata.get("arrival_normalization_claim_label"),
            "contract_fingerprint": metadata.get(
                "arrival_normalization_contract_fingerprint"
            ),
            "seed_namespace_label": metadata.get("seed_namespace_label"),
            "seed_namespace_affects_prng": metadata.get(
                "seed_namespace_affects_prng"
            ),
            "time_dilation_factor": metadata.get("arrival_time_dilation_factor"),
            "raw_aggregate_realized_rho": metadata.get(
                "raw_aggregate_realized_rho"
            ),
            "normalized_primary_aggregate_realized_rho": metadata.get(
                "normalized_primary_aggregate_realized_rho"
            ),
            "raw_census_fingerprint": metadata.get(
                "raw_arrival_census_fingerprint"
            ),
            "normalized_census_fingerprint": metadata.get(
                "normalized_arrival_census_fingerprint"
            ),
            "no_policy_or_oracle_input": metadata.get(
                "arrival_normalization_no_policy_or_oracle_input"
            ),
            "NO_POLICY_OR_ORACLE_INPUT": metadata.get(
                "NO_POLICY_OR_ORACLE_INPUT"
            ),
            "seeds": metadata.get("arrival_census_seeds"),
        }
        if (
            cell_normalization != expected_cell_normalization
            or expected_cell_normalization["algorithm"]
            != ARRIVAL_NORMALIZATION_ALGORITHM
            or expected_cell_normalization["seed_namespace_affects_prng"] is not False
            or expected_cell_normalization["no_policy_or_oracle_input"] is not True
            or expected_cell_normalization["NO_POLICY_OR_ORACLE_INPUT"] is not True
            or expected_cell_normalization["claim_label"]
            != "exact_load_time_normalized_process_shaped_replay"
        ):
            raise ScenarioBuildError("arrival normalization census binding mismatch")

        primary_link = int(
            tree["link_sensitivity_causal_world"]["reference_link_gbps"]
        )
        descriptor = int(
            tree["contribution_transport_accounting"][
                "descriptor_bytes_per_contribution"
            ]
        )
        alignment = int(
            tree["contribution_transport_accounting"][
                "alignment_bytes_per_contribution"
            ]
        )
        reference_cut = (
            (int(tree["payload_bytes"]) + descriptor + alignment)
            * 8.0
            / (float(primary_link) * 1000.0)
        )
        reference_data_path = (
            float(service_surface["sender_pack_us"])
            + reference_cut
            + float(service_surface["receiver_unpack_us"])
        )
        deadline_multiplier = float(metadata["decision_deadline_multiplier"])
        if int(metadata.get("decision_deadline_reference_link_gbps", -1)) != primary_link:
            raise ScenarioBuildError("decision deadline lacks primary-link binding")
        for siblings in world.joins.values():
            isolated = max(
                sibling.contribution.ready_us
                - sibling.contribution.arrival_us
                + reference_data_path
                for sibling in siblings
            ) + float(service_surface["join_combine_us"])
            if any(
                not math.isclose(
                    sibling.contribution.deadline_us
                    - sibling.contribution.arrival_us,
                    deadline_multiplier * isolated,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for sibling in siblings
            ):
                raise ScenarioBuildError("decision deadline drifted from primary link")
        for field, actual in (
            ("task_fingerprint", world.task_fingerprint),
            ("service_fingerprint", world.service_fingerprint),
            ("resource_demand_fingerprint", world.resource_demand_fingerprint),
            ("causal_arrival_fingerprint", matched_causal),
            ("arrival_schedule_fingerprint", arrival_schedule_fingerprint(world)),
            ("score_mask_fingerprint", world.score_mask_fingerprint),
        ):
            if metadata.get(field) != actual:
                raise ScenarioBuildError(f"reloaded world fingerprint mismatch: {field}")
        worlds.append(world)
    if len(worlds) != int(tree["world_count"]):
        raise ScenarioBuildError("scenario world count mismatch")
    return tree, tuple(sorted(worlds, key=lambda world: world.trace_id))


def _require_formal_signoff(
    path: Path | None,
    *,
    config_sha256: str,
    protocol_sha256: str,
    role: str,
    model_key: str,
    link_gbps: int,
    consumer_amendment_sha256: str,
    transitive_hashes: Mapping[str, str] | None = None,
) -> None:
    frozen_amendment_sha256 = validate_consumer_amendment_path(
        DEFAULT_CONSUMER_AMENDMENT, mode="formal"
    )
    if consumer_amendment_sha256 != frozen_amendment_sha256:
        raise ScenarioBuildError(
            "formal signoff binds an unreviewed consumer amendment"
        )
    expected = {
        "stage": "build_scenarios",
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "build_scenarios_source_sha256": _source_sha256(),
        "data_role": role,
        "model_key": model_key,
        "link_gbps": link_gbps,
        "consumer_amendment_sha256": consumer_amendment_sha256,
    }
    if transitive_hashes is not None:
        expected.update(transitive_hashes)
    try:
        verify_phase4_signoff(
            path,
            repo_root=REPO_ROOT,
            expected_fields=expected,
            required_source_paths=(
                Path(__file__),
                HERE / "scenario.py",
                HERE / "schema.py",
                HERE / "capture_routes_gpu.py",
                HERE / "measure_capability_gpu.py",
                HERE / "measure_service_lut_gpu.py",
                HERE / "capability_contract.py",
                HERE / "wire.py",
                HERE / "prepare_data.py",
                HERE / "formal_provenance.py",
            ),
            required_reviewed_scope_paths=(
                *canonical_reviewed_scope_paths(
                    REPO_ROOT,
                    (
                        Path(__file__),
                        HERE / "scenario.py",
                        HERE / "schema.py",
                        HERE / "capture_routes_gpu.py",
                        HERE / "measure_capability_gpu.py",
                        HERE / "measure_service_lut_gpu.py",
                        HERE / "capability_contract.py",
                        HERE / "wire.py",
                        HERE / "prepare_data.py",
                        HERE / "formal_provenance.py",
                    ),
                ),
                DEFAULT_CONSUMER_AMENDMENT,
            ),
        )
    except FormalProvenanceError as exc:
        raise ScenarioBuildError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("calibration", "sealed"), required=True)
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--model-key", choices=("olmoe", "llmjp"), required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument(
        "--calibration-data-manifest",
        type=Path,
        help=(
            "required only for sealed scenarios; binds the calibration-only "
            "service LUT without opening sealed outcomes"
        ),
    )
    parser.add_argument("--route-dir", type=Path, required=True)
    parser.add_argument("--service-lut-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--link-gbps", type=int, default=200)
    parser.add_argument(
        "--primary-scenario-dir",
        type=Path,
        help="required primary-200 scenario source for a 100/400 sensitivity",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--consumer-amendment",
        type=Path,
        default=DEFAULT_CONSUMER_AMENDMENT,
        help="frozen outcome-blind consumer migration amendment",
    )
    parser.add_argument(
        "--historical-reviewed-source-snapshot",
        type=Path,
        help="required in formal mode; exact v5 reviewed-source tar export",
    )
    parser.add_argument(
        "--pre-outcome-attestation",
        type=Path,
        help="required in formal mode; captured before the first sealed scenario",
    )
    parser.add_argument(
        "--pre-outcome-producer-signoff",
        type=Path,
        help="required in formal mode; reviewed signoff for registry capture",
    )
    parser.add_argument(
        "--authoritative-bundle-root",
        type=Path,
        help="required in formal mode; exact root scanned by the pre-outcome registry",
    )
    parser.add_argument(
        "--historical-calibration-lock",
        type=Path,
        help="required for formal sealed scenarios; lock that authorized sealed input",
    )
    parser.add_argument("--signoff", type=Path)
    return parser.parse_args()


def validate_sensitivity_primary_source(
    *,
    primary_scenario_dir: Path | None,
    requested_link_gbps: int,
    primary_link_gbps: int,
    tree: Mapping[str, Any] | None = None,
    inputs: ValidatedInputs | None = None,
    config_path: Path | None = None,
    protocol_path: Path | None = None,
) -> None:
    """Fail closed unless a sensitivity consumes its exact primary artifact."""

    if requested_link_gbps == primary_link_gbps:
        if primary_scenario_dir is not None:
            raise ScenarioBuildError(
                "primary scenario construction must not consume another scenario"
            )
        return
    if primary_scenario_dir is None:
        raise ScenarioBuildError(
            "link sensitivity requires --primary-scenario-dir"
        )
    if any(value is None for value in (tree, inputs, config_path, protocol_path)):
        return
    assert tree is not None
    assert inputs is not None
    assert config_path is not None
    assert protocol_path is not None
    expected = {
        "role": inputs.role,
        "model_key": inputs.model_key,
        "model_revision": inputs.model_revision,
        "link_gbps": primary_link_gbps,
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(protocol_path),
        "data_manifest_sha256": inputs.data_manifest["manifest_sha256"],
        "data_producer_signoff_sha256": inputs.data_manifest.get(
            "signoff_sha256"
        ),
        "route_capture_manifest_sha256": inputs.route_metadata["manifest_sha256"],
        "placement_manifest_sha256": inputs.placement["manifest_sha256"],
        "service_lut_metadata_sha256": inputs.service.metadata_sha256,
        "service_lut_sha256": inputs.service.summary_sha256,
        "service_lut_raw_sha256": inputs.service.raw_sha256,
        "model_tree_manifest_sha256": inputs.service.model_tree_manifest_sha256,
        "build_scenarios_source_sha256": _source_sha256(),
        "consumer_amendment_sha256": inputs.consumer_amendment_sha256,
        "service_calibration_data_manifest_sha256": (
            inputs.service_calibration_data_manifest_sha256
        ),
        "service_calibration_data_manifest_file_sha256": (
            inputs.service_calibration_data_manifest_file_sha256
        ),
        "service_calibration_selected_list_sha256": (
            inputs.service_calibration_selected_list_sha256
        ),
        "service_calibration_data_producer_signoff_sha256": (
            inputs.service_calibration_data_producer_signoff_sha256
        ),
        "sealed_input_attestation_sha256": inputs.sealed_input_attestation_sha256,
        "sealed_input_historical_run_experiment_source_sha256": (
            inputs.sealed_input_historical_run_experiment_source_sha256
        ),
        "sealed_input_historical_calibration_lock_sha256": (
            inputs.sealed_input_historical_calibration_lock_sha256
        ),
        "sealed_input_historical_calibration_signoff_sha256": (
            inputs.sealed_input_historical_calibration_signoff_sha256
        ),
        "sealed_global_reservation_file_sha256": (
            inputs.sealed_global_reservation_file_sha256
        ),
        "sealed_global_consumption_file_sha256": (
            inputs.sealed_global_consumption_file_sha256
        ),
        "historical_reviewed_source_snapshot_sha256": (
            inputs.historical_reviewed_source_snapshot_sha256
        ),
        "pre_outcome_attestation_sha256": inputs.pre_outcome_attestation_sha256,
        "pre_outcome_producer_signoff_file_sha256": (
            inputs.pre_outcome_producer_signoff_file_sha256
        ),
        "pre_outcome_producer_signoff_self_hash": (
            inputs.pre_outcome_producer_signoff_self_hash
        ),
        "authoritative_bundle_root": inputs.authoritative_bundle_root,
        "immutable_input_compatibility_sha256": (
            inputs.immutable_input_compatibility_sha256
        ),
    }
    mismatch = {
        field: (tree.get(field), wanted)
        for field, wanted in expected.items()
        if tree.get(field) != wanted
    }
    if mismatch:
        raise ScenarioBuildError(
            f"sensitivity primary scenario binding mismatch: {sorted(mismatch)}"
        )


def main() -> None:
    args = parse_args()
    validate_formal_output_path(args.output_dir, mode=args.mode)
    guard_mode_role(args.mode, args.role)
    validate_frozen_formal_paths(
        config_path=args.config, protocol_path=args.protocol, mode=args.mode
    )
    config = _load_config(args.config)
    normalization_contract = validate_load_normalization_contract(config)
    allowed_links = {
        int(config["topology_proxy"]["primary_link_gbps"]),
        *[int(value) for value in config["topology_proxy"]["link_sensitivity_gbps"]],
    }
    if args.link_gbps not in allowed_links:
        raise ScenarioBuildError("link rate is outside frozen primary/sensitivity grid")
    primary_link = int(config["topology_proxy"]["primary_link_gbps"])
    validate_sensitivity_primary_source(
        primary_scenario_dir=args.primary_scenario_dir,
        requested_link_gbps=args.link_gbps,
        primary_link_gbps=primary_link,
    )
    if args.mode == "formal":
        _require_formal_signoff(
            args.signoff,
            config_sha256=sha256_file(args.config),
            protocol_sha256=sha256_file(args.protocol),
            role=args.role,
            model_key=args.model_key,
            link_gbps=args.link_gbps,
            consumer_amendment_sha256=sha256_file(args.consumer_amendment),
        )
    inputs = load_validated_inputs(
        role=args.role,
        mode=args.mode,
        model_key=args.model_key,
        data_manifest_path=args.data_manifest,
        calibration_data_manifest_path=args.calibration_data_manifest,
        route_dir=args.route_dir,
        service_lut_dir=args.service_lut_dir,
        config_path=args.config,
        protocol_path=args.protocol,
        consumer_amendment_path=args.consumer_amendment,
        historical_reviewed_source_snapshot_path=(
            args.historical_reviewed_source_snapshot
        ),
        pre_outcome_attestation_path=args.pre_outcome_attestation,
        pre_outcome_producer_signoff_path=args.pre_outcome_producer_signoff,
        authoritative_bundle_root_path=args.authoritative_bundle_root,
        historical_calibration_lock_path=args.historical_calibration_lock,
    )
    if args.mode == "formal":
        _require_formal_signoff(
            args.signoff,
            config_sha256=sha256_file(args.config),
            protocol_sha256=sha256_file(args.protocol),
            role=args.role,
            model_key=args.model_key,
            link_gbps=args.link_gbps,
            consumer_amendment_sha256=inputs.consumer_amendment_sha256,
            transitive_hashes={
                "data_manifest_sha256": str(inputs.data_manifest["manifest_sha256"]),
                "data_producer_signoff_sha256": str(
                    inputs.data_manifest["signoff_sha256"]
                ),
                "route_capture_manifest_sha256": str(
                    inputs.route_metadata["manifest_sha256"]
                ),
                "route_trace_sha256": str(inputs.route_metadata["route_trace_sha256"]),
                "placement_manifest_sha256": str(
                    inputs.placement["manifest_sha256"]
                ),
                "service_lut_metadata_sha256": inputs.service.metadata_sha256,
                "service_lut_sha256": inputs.service.summary_sha256,
                "service_lut_raw_sha256": inputs.service.raw_sha256,
                "model_tree_manifest_sha256": inputs.service.model_tree_manifest_sha256,
                "capture_routes_source_sha256": str(
                    inputs.route_metadata["capture_routes_source_sha256"]
                ),
                "route_producer_signoff_sha256": str(
                    inputs.route_metadata["signoff_sha256"]
                ),
                "measure_service_lut_source_sha256": (
                    inputs.service.producer_source_sha256
                ),
                "service_lut_producer_signoff_sha256": str(
                    inputs.service.producer_signoff_sha256
                ),
                "service_calibration_data_manifest_sha256": (
                    inputs.service_calibration_data_manifest_sha256
                ),
                "service_calibration_data_manifest_file_sha256": (
                    inputs.service_calibration_data_manifest_file_sha256
                ),
                "service_calibration_selected_list_sha256": (
                    inputs.service_calibration_selected_list_sha256
                ),
                "service_calibration_data_producer_signoff_sha256": str(
                    inputs.service_calibration_data_producer_signoff_sha256
                ),
                "historical_reviewed_source_snapshot_sha256": (
                    inputs.historical_reviewed_source_snapshot_sha256
                ),
                "pre_outcome_attestation_sha256": (
                    inputs.pre_outcome_attestation_sha256
                ),
                "pre_outcome_producer_signoff_file_sha256": (
                    inputs.pre_outcome_producer_signoff_file_sha256
                ),
                "pre_outcome_producer_signoff_self_hash": (
                    inputs.pre_outcome_producer_signoff_self_hash
                ),
                "authoritative_bundle_root": inputs.authoritative_bundle_root,
                "immutable_input_compatibility_sha256": (
                    inputs.immutable_input_compatibility_sha256
                ),
                **(
                    {
                        "sealed_input_attestation_sha256": str(
                            inputs.sealed_input_attestation_sha256
                        ),
                        "sealed_input_historical_run_experiment_source_sha256": str(
                            inputs.sealed_input_historical_run_experiment_source_sha256
                        ),
                        "sealed_input_historical_calibration_lock_sha256": str(
                            inputs.sealed_input_historical_calibration_lock_sha256
                        ),
                        "sealed_input_historical_calibration_signoff_sha256": str(
                            inputs.sealed_input_historical_calibration_signoff_sha256
                        ),
                        "sealed_global_reservation_file_sha256": str(
                            inputs.sealed_global_reservation_file_sha256
                        ),
                        "sealed_global_consumption_file_sha256": str(
                            inputs.sealed_global_consumption_file_sha256
                        ),
                    }
                    if inputs.role == "sealed"
                    else {}
                ),
            },
        )
    cells = _cell_map(config)
    worlds: list[ReplayWorld] = []
    metadata: list[Mapping[str, Any]] = []
    if args.link_gbps == primary_link:
        groups = partition_requests(
            inputs.data_manifest["requests"], role=args.role, config=config
        )
        seeds = role_trace_seeds(config, role=args.role, trace_count=len(groups))
        seed_namespace_label = str(
            config["workloads"]["role_seed_ranges"][args.role]["derivation_salt"]
        )
        count_per_trace = int(normalization_contract["raw_schedule_count"])
        arrival_censuses = {
            cell_name: build_arrival_normalization_census(
                role=args.role,
                cell_name=cell_name,
                cell=cell,
                seeds=seeds,
                seed_namespace_label=seed_namespace_label,
                count_per_trace=count_per_trace,
            )
            for cell_name, cell in cells.items()
        }
        for cell_name, cell in cells.items():
            for trace_index, request_ids in enumerate(groups):
                seed = seeds[trace_index]
                world, row = build_world(
                    inputs=inputs,
                    request_ids=request_ids,
                    trace_index=trace_index,
                    seed=seed,
                    cell_name=cell_name,
                    cell=cell,
                    arrival_census=arrival_censuses[cell_name],
                    config=config,
                    link_gbps=args.link_gbps,
                )
                worlds.append(world)
                metadata.append(row)
    else:
        assert args.primary_scenario_dir is not None
        primary_tree, primary_worlds = load_worlds(
            args.primary_scenario_dir,
            expected_role=args.role,
            mode=args.mode,
        )
        validate_sensitivity_primary_source(
            primary_scenario_dir=args.primary_scenario_dir,
            requested_link_gbps=args.link_gbps,
            primary_link_gbps=primary_link,
            tree=primary_tree,
            inputs=inputs,
            config_path=args.config,
            protocol_path=args.protocol,
        )
        primary_tree_sha256 = str(primary_tree["manifest_sha256"])
        if args.mode == "formal":
            _require_formal_signoff(
                args.signoff,
                config_sha256=sha256_file(args.config),
                protocol_sha256=sha256_file(args.protocol),
                role=args.role,
                model_key=args.model_key,
                link_gbps=args.link_gbps,
                consumer_amendment_sha256=inputs.consumer_amendment_sha256,
                transitive_hashes={
                    "primary_scenario_tree_sha256": primary_tree_sha256,
                },
            )
        primary_metadata = {
            str(row["trace_id"]): row for row in primary_tree["worlds"]
        }
        if set(primary_metadata) != {world.trace_id for world in primary_worlds}:
            raise ScenarioBuildError("primary scenario world metadata is incomplete")
        for primary_world in primary_worlds:
            world, row = build_link_sensitivity_world_from_primary(
                inputs=inputs,
                primary_world=primary_world,
                primary_metadata=primary_metadata[primary_world.trace_id],
                primary_scenario_tree_sha256=primary_tree_sha256,
                config=config,
                link_gbps=args.link_gbps,
            )
            worlds.append(world)
            metadata.append(row)
    if args.link_gbps == primary_link:
        tolerance = float(
            normalization_contract["normalized_arithmetic_aggregate_abs_tolerance"]
        )
        validate_aggregate_utilization(metadata, cells=cells, tolerance=tolerance)
    payload = write_scenarios(
        output_dir=args.output_dir,
        inputs=inputs,
        worlds=worlds,
        world_metadata=metadata,
        config_path=args.config,
        protocol_path=args.protocol,
        mode=args.mode,
        link_gbps=args.link_gbps,
        signoff_path=args.signoff,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
