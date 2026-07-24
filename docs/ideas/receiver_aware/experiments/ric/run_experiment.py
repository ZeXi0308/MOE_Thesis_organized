#!/usr/bin/env python3
"""Calibration lock and formal RIC-v1 replay/gate runner.

The runner has two explicit stages:

* ``calibrate`` consumes calibration scenarios only, freezes the strongest
  join-blind arm, the metric closure budget, and the independent starvation
  threshold.
* ``evaluate`` consumes sealed scenarios once.  It runs G2 first and only
  opens the charged G3 arms if G2 passes.

All parallelism is over complete trace clusters.  No request, token, or
contribution is ever treated as an independent statistical unit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import sys
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

try:
    from .accounting import (
        PairedBootstrapSummary,
        RICAccountingError,
        RetentionBootstrapSummary,
        TraceMetrics,
        assert_replay_conservation,
        assert_sham_feedback_cost_equivalence,
        empirical_cvar,
        paired_retention_bootstrap,
        paired_trace_bootstrap,
        quantile_type1,
        trace_metrics_from_result,
    )
    from .build_scenarios import (
        DEFAULT_CONFIG,
        DEFAULT_CONSUMER_AMENDMENT,
        DEFAULT_PROTOCOL,
        HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256,
        GLOBAL_SEALED_EVALUATION_CONSUMPTION,
        ScenarioBuildError,
        _read_self_hashed_json,
        _source_sha256 as _build_scenarios_source_sha256,
        atomic_output_directory,
        attested_file_sha256,
        guard_mode_role,
        load_worlds,
        object_sha256,
        validate_gpu_environment_artifact,
        validate_consumer_amendment_path,
        validate_frozen_formal_paths,
        validate_formal_output_path,
        verify_immutable_upstream_signoff,
        verify_pre_outcome_attestation,
    )
    from .formal_provenance import (
        EMBEDDED_PRODUCER_SIGNOFF,
        FormalProvenanceError,
        add_self_hash,
        canonical_reviewed_scope_paths,
        is_sha256,
        load_json_mapping_strict,
        loads_json_mapping_strict,
        materialize_verified_signoff,
        sha256_file,
        validate_calibration_lock_fields,
        verify_phase4_signoff,
    )
    from .measure_capability_gpu import (
        _require_formal_signoff as _verify_capability_producer_signoff,
    )
    from .run_oracle import (
        ORACLE_SOURCE_PATHS,
        _require_formal_signoff as _verify_oracle_producer_signoff,
        _source_sha256 as _run_oracle_source_sha256,
    )
    from .capability_contract import (
        EXECUTION_ORDER_RULE,
        CapabilityContractError,
        capability_execution_order,
    )
    from .replay import (
        JOINBLIND_ARMS,
        ActionRecord,
        ContractTaxSurface,
        ReplayConfig,
        ReplayResult,
        action_collapse_matrix,
        run_replay,
        run_sham_against_reference,
    )
    from .scenario import ReplayWorld
    from .wire import ContractTax
except ImportError:
    from ric.accounting import (  # type: ignore
        PairedBootstrapSummary,
        RICAccountingError,
        RetentionBootstrapSummary,
        TraceMetrics,
        assert_replay_conservation,
        assert_sham_feedback_cost_equivalence,
        empirical_cvar,
        paired_retention_bootstrap,
        paired_trace_bootstrap,
        quantile_type1,
        trace_metrics_from_result,
    )
    from ric.build_scenarios import (  # type: ignore
        DEFAULT_CONFIG,
        DEFAULT_CONSUMER_AMENDMENT,
        DEFAULT_PROTOCOL,
        HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256,
        GLOBAL_SEALED_EVALUATION_CONSUMPTION,
        ScenarioBuildError,
        _read_self_hashed_json,
        _source_sha256 as _build_scenarios_source_sha256,
        atomic_output_directory,
        attested_file_sha256,
        guard_mode_role,
        load_worlds,
        object_sha256,
        validate_gpu_environment_artifact,
        validate_consumer_amendment_path,
        validate_frozen_formal_paths,
        validate_formal_output_path,
        verify_immutable_upstream_signoff,
        verify_pre_outcome_attestation,
    )
    from ric.formal_provenance import (  # type: ignore
        EMBEDDED_PRODUCER_SIGNOFF,
        FormalProvenanceError,
        add_self_hash,
        canonical_reviewed_scope_paths,
        is_sha256,
        load_json_mapping_strict,
        loads_json_mapping_strict,
        materialize_verified_signoff,
        sha256_file,
        validate_calibration_lock_fields,
        verify_phase4_signoff,
    )
    from ric.measure_capability_gpu import (  # type: ignore
        _require_formal_signoff as _verify_capability_producer_signoff,
    )
    from ric.run_oracle import (  # type: ignore
        ORACLE_SOURCE_PATHS,
        _require_formal_signoff as _verify_oracle_producer_signoff,
        _source_sha256 as _run_oracle_source_sha256,
    )
    from ric.capability_contract import (  # type: ignore
        EXECUTION_ORDER_RULE,
        CapabilityContractError,
        capability_execution_order,
    )
    from ric.replay import (  # type: ignore
        JOINBLIND_ARMS,
        ActionRecord,
        ContractTaxSurface,
        ReplayConfig,
        ReplayResult,
        action_collapse_matrix,
        run_replay,
        run_sham_against_reference,
    )
    from ric.scenario import ReplayWorld  # type: ignore
    from ric.wire import ContractTax  # type: ignore


REPO_ROOT = HERE.parents[4]
CONCRETE_JOINBLIND_ARMS = tuple(sorted(JOINBLIND_ARMS - {"calib_best_joinblind"}))
RUN_EXPERIMENT_SOURCE_PATHS = (
    HERE / "__init__.py",
    Path(__file__),
    HERE / "run_oracle.py",
    HERE / "oracle.py",
    HERE / "build_scenarios.py",
    HERE / "scenario.py",
    HERE / "replay.py",
    HERE / "accounting.py",
    HERE / "wire.py",
    HERE / "policy_views.py",
    HERE / "schema.py",
    HERE / "measure_capability_gpu.py",
    HERE / "measure_service_lut_gpu.py",
    HERE / "capture_routes_gpu.py",
    HERE / "prepare_data.py",
    HERE / "formal_provenance.py",
    HERE / "capability_contract.py",
)
SEALED_RUN_EXPERIMENT_SOURCE_PATHS = tuple(
    dict.fromkeys((*RUN_EXPERIMENT_SOURCE_PATHS, *ORACLE_SOURCE_PATHS))
)


class ExperimentRunnerError(RuntimeError):
    """A runner state, provenance, accounting, or gate invariant failed."""


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for path in RUN_EXPERIMENT_SOURCE_PATHS:
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sealed_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in SEALED_RUN_EXPERIMENT_SOURCE_PATHS:
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _capability_producer_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        HERE / "measure_capability_gpu.py",
        HERE / "capture_routes_gpu.py",
        HERE / "prepare_data.py",
        HERE / "formal_provenance.py",
        HERE / "capability_contract.py",
    ):
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = load_json_mapping_strict(path, label="RIC config")
    except FormalProvenanceError as exc:
        raise ExperimentRunnerError(str(exc)) from exc
    if value.get("schema_version") != "ric-config-v1":
        raise ExperimentRunnerError("RIC config schema mismatch")
    return value


def resolve_worker_count(requested: int, job_count: int) -> int:
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ExperimentRunnerError("workers must be a non-negative integer")
    if job_count <= 0:
        raise ExperimentRunnerError("cannot schedule zero trace jobs")
    if requested:
        return min(requested, job_count)
    available = os.cpu_count() or 1
    return min(max(1, available - 1), job_count)


def capability_paired_lcbs(
    effects: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    order_statistic_one_based: int,
    seed: int,
) -> dict[str, float]:
    if (
        type(replicates) is not int
        or replicates < 1
        or type(order_statistic_one_based) is not int
        or not 1 <= order_statistic_one_based <= replicates
    ):
        raise ExperimentRunnerError("invalid capability bootstrap order statistic")
    lengths = {len(values) for values in effects.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise ExperimentRunnerError("capability paired effects are incomplete")
    count = next(iter(lengths))
    rng = random.Random(seed)
    distributions = {name: [] for name in effects}
    for _ in range(replicates):
        indexes = [rng.randrange(count) for _row in range(count)]
        for name, values in effects.items():
            distributions[name].append(
                math.fsum(float(values[index]) for index in indexes) / count
            )
    rank = order_statistic_one_based - 1
    return {name: sorted(values)[rank] for name, values in distributions.items()}


def contract_tax_surface_from_tree(tree: Mapping[str, Any]) -> ContractTaxSurface:
    service = tree.get("service_surface")
    if not isinstance(service, Mapping):
        raise ExperimentRunnerError("scenario lacks measured service surface")
    raw_points = service.get("control_tax_by_record_count")
    source_id = service.get("control_tax_source_id")
    if not isinstance(raw_points, Mapping) or not isinstance(source_id, str):
        raise ExperimentRunnerError("scenario lacks exact control tax surface")
    points = []
    for count in range(1, 256):
        point = raw_points.get(str(count))
        if not isinstance(point, Mapping):
            raise ExperimentRunnerError(
                f"BLOCKED_CONTROL_TAX_SURFACE: missing record count {count}"
            )
        points.append(
            (
                count,
                ContractTax(
                    state_build_us=float(point["state_build_us"]),
                    hash_us=float(point["hash_us"]),
                    encode_us=float(point["encode_us"]),
                    transfer_us=float(point["transfer_us"]),
                    decode_us=float(point["decode_us"]),
                    lookup_us=float(point["lookup_us"]),
                    apply_us=float(point["apply_us"]),
                    policy_lookup_us=float(point["policy_lookup_us"]),
                ),
            )
        )
    surface = ContractTaxSurface(points=tuple(points), source_id=source_id)
    expected = object_sha256(
        {
            "rule": "raw_median_minus_same_count_empty_harness",
            "non_grid_rule": "exact_1_to_255_no_interpolation_or_extrapolation",
            "points": raw_points,
            "service_lut_sha256": service["summary_sha256"],
        }
    )
    if expected != source_id:
        raise ExperimentRunnerError("control tax source id does not bind LUT points")
    return surface


def validate_capability_action_trace(
    directory: Path,
    *,
    artifact: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, str]],
    trials: int,
    config: Mapping[str, Any],
) -> dict[tuple[int, str, str], list[Mapping[str, Any]]]:
    """Validate the two-ready-task CUDA/NVTX evidence behind G1."""

    try:
        expected_orders = {
            trial: capability_execution_order(config, trial)
            for trial in range(trials)
        }
    except CapabilityContractError as exc:
        raise ExperimentRunnerError(str(exc)) from exc
    trace_path = directory / "capability_action_trace.jsonl"
    if artifact.get("capability_action_trace_sha256") != sha256_file(trace_path):
        raise ExperimentRunnerError("capability action trace hash mismatch")
    actions: list[Mapping[str, Any]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = loads_json_mapping_strict(
                    line, label=f"capability action trace line {line_number}"
                )
            except FormalProvenanceError as exc:
                raise ExperimentRunnerError(
                    f"invalid capability action trace JSON at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ExperimentRunnerError("capability action trace row is not an object")
            actions.append(row)
    expected_count = trials * 2 * 2 * 2
    if (
        artifact.get("action_trace_schema_version") != "ric-capability-action-v1"
        or artifact.get("action_trace_row_count") != expected_count
        or len(actions) != expected_count
        or artifact.get("nvtx_ranges_emitted") is not True
        or artifact.get("queue_snapshot_ready_count") != 2
    ):
        raise ExperimentRunnerError("capability action trace coverage is incomplete")
    fixtures = artifact.get("task_fixtures")
    if not isinstance(fixtures, Mapping) or set(fixtures) != {
        "x_closing",
        "y_nonclosing",
    }:
        raise ExperimentRunnerError("capability task fixtures are incomplete")
    fixture_ids: dict[str, str] = {}
    required_identity_fields = {
        "model_key",
        "model_revision",
        "sender_rank",
        "receiver_rank",
        "sender_local_queue_id",
        "shared_cut_path",
        "receiver_combine_resource",
        "contribution_identities",
        "payload_shape",
        "payload_stride",
        "payload_dtype",
        "payload_bytes",
        "payload_sha256",
    }
    full_contribution_fields = {
        "request_id",
        "forward_id",
        "batch_id",
        "phase",
        "decode_step",
        "layer_id",
        "token_id",
        "token_block_id",
        "topk_slot",
        "expert_id",
        "sender_rank",
        "receiver_rank",
        "epoch",
    }
    try:
        sender_rank = int(artifact["sender_rank"])
        receiver_rank = int(artifact["receiver_rank"])
        ep_size = int(artifact["ep_size"])
        ranks_per_node = int(artifact["ranks_per_node"])
        num_experts = int(artifact["num_experts"])
        top_k = int(artifact["top_k"])
        root_request_id = str(artifact["request_id"])
        root_target_layer = int(artifact["target_layer"])
        root_block_rows = int(artifact["block_rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentRunnerError("capability topology identity is incomplete") from exc
    queue_id = f"cuda:0/ric-capability/sender:{sender_rank}:local-return-queue"
    expected_cut_path = (
        f"node{sender_rank // ranks_per_node}->"
        f"node{receiver_rank // ranks_per_node}"
    )
    if (
        not 0 <= sender_rank < ep_size
        or not 0 <= receiver_rank < ep_size
        or ep_size % ranks_per_node
        or artifact.get("sender_local_queue_id") != queue_id
    ):
        raise ExperimentRunnerError("capability sender-local topology is invalid")
    all_join_ids: set[tuple[Any, ...]] = set()
    for task_name in ("x_closing", "y_nonclosing"):
        fixture = fixtures[task_name]
        if not isinstance(fixture, Mapping):
            raise ExperimentRunnerError("capability task fixture is not an object")
        unhashed = dict(fixture)
        supplied = unhashed.pop("fixture_identity_sha256", None)
        if supplied != object_sha256(unhashed):
            raise ExperimentRunnerError("capability fixture identity hash mismatch")
        if fixture.get("task_name") != task_name or not required_identity_fields.issubset(
            fixture
        ):
            raise ExperimentRunnerError("capability full task identity is incomplete")
        contributions = fixture.get("contribution_identities")
        payload_shape = fixture.get("payload_shape")
        if (
            type(fixture.get("payload_bytes")) is not int
            or int(fixture["payload_bytes"]) <= 0
            or len(str(fixture.get("payload_sha256", ""))) != 64
            or not isinstance(payload_shape, list)
            or not payload_shape
            or type(payload_shape[0]) is not int
            or payload_shape[0] != root_block_rows
            or not isinstance(contributions, list)
            or not contributions
            or payload_shape[0] != len(contributions)
            or fixture.get("sender_rank") != sender_rank
            or fixture.get("receiver_rank") != receiver_rank
            or fixture.get("model_key") != artifact.get("model_key")
            or fixture.get("model_revision") != artifact.get("model_revision")
            or fixture.get("sender_local_queue_id") != queue_id
            or fixture.get("shared_cut_path") != expected_cut_path
            or fixture.get("receiver_combine_resource")
            != f"receiver:{receiver_rank}:combine"
        ):
            raise ExperimentRunnerError("capability task payload identity is invalid")
        task_join_ids: set[tuple[Any, ...]] = set()
        for contribution in contributions:
            if not isinstance(contribution, Mapping) or set(contribution) != full_contribution_fields:
                raise ExperimentRunnerError("capability contribution identity is incomplete")
            integer_fields = (
                "decode_step",
                "layer_id",
                "topk_slot",
                "expert_id",
                "sender_rank",
                "receiver_rank",
                "epoch",
            )
            if any(type(contribution.get(field)) is not int for field in integer_fields):
                raise ExperimentRunnerError("capability contribution integer identity is invalid")
            expert_id = int(contribution["expert_id"])
            expected_sender = min(
                ep_size - 1, expert_id * ep_size // num_experts
            )
            if (
                contribution.get("phase") != "prefill"
                or contribution.get("request_id") != root_request_id
                or contribution.get("forward_id")
                != f"{root_request_id}:capability-forward"
                or contribution.get("batch_id")
                != f"{root_request_id}:capability-batch"
                or contribution.get("decode_step") != 0
                or contribution.get("layer_id") != root_target_layer
                or contribution.get("epoch") != 1
                or contribution.get("sender_rank") != sender_rank
                or contribution.get("receiver_rank") != receiver_rank
                or expected_sender != sender_rank
                or not 0 <= int(contribution["topk_slot"]) < top_k
                or not 0 <= expert_id < num_experts
            ):
                raise ExperimentRunnerError(
                    "capability contribution is not owned by the sender-local queue"
                )
            join_id = (
                contribution["request_id"],
                contribution["forward_id"],
                contribution["batch_id"],
                contribution["layer_id"],
                contribution["token_id"],
                contribution["token_block_id"],
                contribution["receiver_rank"],
                contribution["epoch"],
            )
            if join_id in task_join_ids or join_id in all_join_ids:
                raise ExperimentRunnerError("capability fixture reuses an application join")
            task_join_ids.add(join_id)
        all_join_ids.update(task_join_ids)
        fixture_ids[task_name] = str(supplied)
    selection = artifact.get("sender_local_selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("selection_rule")
        != "route_identity_hash_sender_local_distinct_token_v1"
        or selection.get("sender_rank") != sender_rank
    ):
        raise ExperimentRunnerError("capability sender-local selection evidence is invalid")
    for task_name in ("x_closing", "y_nonclosing"):
        plan_rows = selection.get(task_name)
        fixture_rows = fixtures[task_name]["contribution_identities"]
        if not isinstance(plan_rows, list) or len(plan_rows) != len(fixture_rows):
            raise ExperimentRunnerError("capability sender-local selection coverage mismatch")
        for plan_row, identity in zip(plan_rows, fixture_rows):
            if (
                not isinstance(plan_row, Mapping)
                or plan_row.get("topk_slot") != identity["topk_slot"]
                or plan_row.get("expert_id") != identity["expert_id"]
                or plan_row.get("sender_rank") != sender_rank
                or identity["token_id"]
                != f"{identity['request_id']}:token:{plan_row.get('token_index')}"
                or identity["token_block_id"]
                != f"{identity['request_id']}:token-block:{plan_row.get('token_index')}"
            ):
                raise ExperimentRunnerError("capability route plan/identity mismatch")
    required_profiler_modes = config.get("capability_probes", {}).get(
        "profiler_diagnostics_required_release_modes"
    )
    profiler_diagnostics = artifact.get("profiler_diagnostics")
    if (
        artifact.get("action_trace_evidence_boundary")
        != "CUDA_EVENT_ACTION_TRACE_WITH_EMITTED_NVTX_LABELS"
        or artifact.get("profiler_diagnostic_not_in_timing_trials") is not True
        or artifact.get("profiler_trace_kind")
        != "torch_profiler_chrome_trace_cpu_cuda"
        or required_profiler_modes != ["streaming", "full_layer_barrier"]
        or not isinstance(profiler_diagnostics, Mapping)
        or set(profiler_diagnostics) != set(required_profiler_modes)
    ):
        raise ExperimentRunnerError("capability profiler trace binding mismatch")
    profiler_trials = {
        "streaming": -1_000_000,
        "full_layer_barrier": -1_000_001,
    }
    profiler_gpu_stream_ordinals: set[int] = set()
    for release_mode in required_profiler_modes:
        diagnostic = profiler_diagnostics[release_mode]
        if not isinstance(diagnostic, Mapping):
            raise ExperimentRunnerError("capability profiler diagnostic is not an object")
        profiler_trial = profiler_trials[release_mode]
        profiler_prefix = (
            f"ric_capability/trial={profiler_trial}/policy=candidate_closing_first/"
            f"release={release_mode}"
        )
        required_profiler_labels = [
            profiler_prefix,
            *(
                f"{profiler_prefix}/enqueue={name}"
                for name in ("x_closing", "y_nonclosing")
            ),
            f"{profiler_prefix}/queue_snapshot=both_ready",
            *(
                f"{profiler_prefix}/select={name}"
                for name in ("x_closing", "y_nonclosing")
            ),
            *(
                f"{profiler_prefix}/service={name}"
                for name in ("x_closing", "y_nonclosing")
            ),
        ]
        profiler_file = f"capability_cuda_trace_{release_mode}.json"
        profiler_path = directory / profiler_file
        if (
            diagnostic.get("trial") != profiler_trial
            or diagnostic.get("policy") != "candidate_closing_first"
            or diagnostic.get("release_mode") != release_mode
            or diagnostic.get("trace_file") != profiler_file
            or diagnostic.get("required_labels") != required_profiler_labels
            or diagnostic.get("trace_sha256") != sha256_file(profiler_path)
            or diagnostic.get("sender_local_stream_id")
            != artifact.get("sender_local_stream_id")
            or diagnostic.get("canonical_output_sha256")
            != artifact.get("canonical_reference_sha256")
        ):
            raise ExperimentRunnerError("capability profiler trace binding mismatch")
        try:
            profiler_trace = loads_json_mapping_strict(
                profiler_path.read_text(encoding="utf-8"),
                label=f"capability profiler trace {release_mode}",
            )
        except (OSError, UnicodeError, FormalProvenanceError) as exc:
            raise ExperimentRunnerError(
                "capability profiler trace is invalid JSON"
            ) from exc
        trace_events = profiler_trace.get("traceEvents")
        if not isinstance(trace_events, list):
            raise ExperimentRunnerError("capability profiler trace lacks traceEvents")
        observed_labels = {
            str(event.get("name"))
            for event in trace_events
            if isinstance(event, Mapping)
        }
        gpu_activity_streams: set[int] = set()
        gpu_activity_count = 0
        for event in trace_events:
            if (
                not isinstance(event, Mapping)
                or str(event.get("cat", "")).lower()
                not in {"kernel", "gpu_memcpy", "gpu_memset"}
            ):
                continue
            gpu_activity_count += 1
            args = event.get("args")
            stream = args.get("stream") if isinstance(args, Mapping) else None
            if type(stream) is not int or stream < 0:
                raise ExperimentRunnerError(
                    "capability profiler GPU activity lacks stream ordinal"
                )
            gpu_activity_streams.add(stream)
        if (
            not set(required_profiler_labels).issubset(observed_labels)
            or gpu_activity_count < 1
            or len(gpu_activity_streams) != 1
        ):
            raise ExperimentRunnerError(
                "capability profiler trace has mixed/missing GPU stream activity"
            )
        gpu_stream_ordinal = next(iter(gpu_activity_streams))
        if diagnostic.get("gpu_activity_stream_ordinal") != gpu_stream_ordinal:
            raise ExperimentRunnerError(
                "capability profiler stream ordinal binding mismatch"
            )
        profiler_gpu_stream_ordinals.add(gpu_stream_ordinal)
    if len(profiler_gpu_stream_ordinals) != 1:
        raise ExperimentRunnerError(
            "capability profiler release modes used different GPU streams"
        )
    ready_ids = [fixture_ids["x_closing"], fixture_ids["y_nonclosing"]]
    expected_snapshot = {
        "ready_count": 2,
        "ready_task_ids": ready_ids,
        "all_ready_before_selection": True,
    }
    snapshot_sha = object_sha256(expected_snapshot)
    grouped: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for row in actions:
        try:
            key = (
                int(row["trial"]),
                str(row["policy"]),
                str(row["release_mode"]),
            )
            timestamps = [
                float(row[field])
                for field in (
                    "enqueue_ts_us",
                    "queue_ready_ts_us",
                    "selected_ts_us",
                    "service_start_ts_us",
                    "service_end_ts_us",
                )
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentRunnerError("capability action timestamp schema is invalid") from exc
        task_name = str(row.get("task_name"))
        if task_name not in fixture_ids:
            raise ExperimentRunnerError("capability action names an unknown task")
        fixture = fixtures[task_name]
        if (
            row.get("schema_version") != "ric-capability-action-v1"
            or row.get("fixture_identity_sha256") != fixture_ids[task_name]
            or row.get("task_identity") != fixture
            or row.get("payload_bytes") != fixture["payload_bytes"]
            or row.get("queue_snapshot") != expected_snapshot
            or row.get("queue_snapshot_sha256") != snapshot_sha
            or row.get("source") != "measured_5090_cuda"
            or type(row.get("stream_id")) is not int
            or int(row["stream_id"]) <= 0
        ):
            raise ExperimentRunnerError("capability action provenance mismatch")
        expected_label = (
            f"ric_capability/trial={key[0]}/policy={key[1]}/"
            f"release={key[2]}/service={task_name}"
        )
        trial_prefix = (
            f"ric_capability/trial={key[0]}/policy={key[1]}/release={key[2]}"
        )
        expected_action_labels = {
            "enqueue": f"{trial_prefix}/enqueue={task_name}",
            "queue_snapshot": f"{trial_prefix}/queue_snapshot=both_ready",
            "select": f"{trial_prefix}/select={task_name}",
            "service": expected_label,
        }
        if (
            row.get("nvtx_range_label") != expected_label
            or row.get("nvtx_labels") != expected_action_labels
        ):
            raise ExperimentRunnerError("capability NVTX label mismatch")
        enqueue, ready, selected, service_start, service_end = timestamps
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in timestamps)
            or enqueue > ready
            or ready > selected
            or selected > service_start
            or service_start >= service_end
        ):
            raise ExperimentRunnerError("capability CUDA-event causality is invalid")
        expected_group_id = object_sha256(
            {
                "trial": key[0],
                "execution_order_index": expected_orders[key[0]].index(
                    (key[2], key[1])
                ),
                "policy": key[1],
                "release_mode": key[2],
                "ready_task_ids": ready_ids,
            }
        )
        if row.get("action_trace_group_id") != expected_group_id:
            raise ExperimentRunnerError("capability action group identity mismatch")
        if row.get("execution_order_index") != expected_orders[key[0]].index(
            (key[2], key[1])
        ):
            raise ExperimentRunnerError("capability execution order is not counterbalanced")
        grouped.setdefault(key, []).append(row)
    policies = {"baseline_nonclosing_first", "candidate_closing_first"}
    releases = {"streaming", "full_layer_barrier"}
    expected_grid = {
        (trial, policy, release)
        for trial in range(trials)
        for policy in policies
        for release in releases
    }
    if set(grouped) != expected_grid:
        raise ExperimentRunnerError("capability action trace grid mismatch")
    raw_by_group = {
        (int(row["trial"]), str(row["policy"]), str(row["release_mode"])): row
        for row in raw_rows
    }
    if set(raw_by_group) != expected_grid:
        raise ExperimentRunnerError("capability raw/action group grid mismatch")
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: int(row["service_order_index"]))
        expected_order = (
            ["x_closing", "y_nonclosing"]
            if key[1] == "candidate_closing_first"
            else ["y_nonclosing", "x_closing"]
        )
        if (
            len(ordered) != 2
            or [row["task_name"] for row in ordered] != expected_order
            or [row["service_order_index"] for row in ordered] != [0, 1]
            or float(ordered[0]["service_end_ts_us"])
            > float(ordered[1]["service_start_ts_us"])
            or len({row["stream_id"] for row in ordered}) != 1
        ):
            raise ExperimentRunnerError("capability queue did not execute reviewed order")
        raw = raw_by_group[key]
        if (
            raw.get("action_trace_group_id") != ordered[0]["action_trace_group_id"]
            or int(str(raw.get("execution_order_index", -1)))
            != ordered[0]["execution_order_index"]
            or raw.get("queue_snapshot_sha256") != snapshot_sha
            or int(str(raw.get("stream_id"))) != ordered[0]["stream_id"]
            or raw.get("service_order") != ",".join(expected_order)
        ):
            raise ExperimentRunnerError("capability raw/action trace binding mismatch")
    return grouped


def capability_event_precedence_failures(
    raw_rows: Sequence[Mapping[str, str]],
    grouped_actions: Mapping[
        tuple[int, str, str], Sequence[Mapping[str, Any]]
    ],
    *,
    trials: int,
) -> list[str]:
    """Independently derive the Amendment-O causal event gate."""

    policies = ("baseline_nonclosing_first", "candidate_closing_first")
    releases = ("streaming", "full_layer_barrier")
    raw_by_key = {
        (int(row["trial"]), str(row["policy"]), str(row["release_mode"])): row
        for row in raw_rows
    }
    expected = {
        (trial, policy, release)
        for trial in range(trials)
        for policy in policies
        for release in releases
    }
    if set(raw_by_key) != expected or set(grouped_actions) != expected:
        return ["event_grid_mismatch"]
    failures: list[str] = []

    def ordered(key: tuple[int, str, str]) -> list[Mapping[str, Any]]:
        return sorted(
            grouped_actions[key], key=lambda row: int(row["service_order_index"])
        )

    def named(
        key: tuple[int, str, str], task_name: str
    ) -> Mapping[str, Any]:
        rows = [row for row in grouped_actions[key] if row["task_name"] == task_name]
        if len(rows) != 1:
            raise ExperimentRunnerError(
                "capability event trace task coverage is invalid"
            )
        return rows[0]

    compared_fields = (
        "task_name",
        "fixture_identity_sha256",
        "task_identity",
        "payload_bytes",
        "stream_id",
        "queue_snapshot",
        "queue_snapshot_sha256",
        "service_order_index",
    )
    for trial in range(trials):
        for policy in policies:
            streaming_key = (trial, policy, "streaming")
            barrier_key = (trial, policy, "full_layer_barrier")
            streaming_actions = ordered(streaming_key)
            barrier_actions = ordered(barrier_key)
            if len(streaming_actions) != 2 or len(barrier_actions) != 2 or any(
                streaming[field] != barrier[field]
                for streaming, barrier in zip(streaming_actions, barrier_actions)
                for field in compared_fields
            ):
                failures.append(f"trial={trial}/policy={policy}/cross_release_identity")

            barrier_raw = raw_by_key[barrier_key]
            barrier_release = float(barrier_raw["application_release_us"])
            barrier_downstream = float(barrier_raw["downstream_start_us"])
            if any(
                float(row["service_end_ts_us"]) > barrier_release
                for row in barrier_actions
            ):
                failures.append(f"trial={trial}/policy={policy}/barrier_early_release")
            if barrier_release > barrier_downstream:
                failures.append(
                    f"trial={trial}/policy={policy}/barrier_downstream_before_release"
                )

            streaming_raw = raw_by_key[streaming_key]
            frontier = float(streaming_raw["physical_frontier_us"])
            release = float(streaming_raw["application_release_us"])
            downstream = float(streaming_raw["downstream_start_us"])
            if frontier > release:
                failures.append(f"trial={trial}/policy={policy}/release_before_frontier")
            if release > downstream:
                failures.append(
                    f"trial={trial}/policy={policy}/streaming_downstream_before_release"
                )

            streaming_y = named(streaming_key, "y_nonclosing")
            streaming_x = named(streaming_key, "x_closing")
            if not (
                float(streaming_x["service_start_ts_us"])
                <= frontier
                <= float(streaming_x["service_end_ts_us"])
            ):
                failures.append(
                    f"trial={trial}/policy={policy}/streaming_frontier_outside_closing"
                )
            barrier_x = named(barrier_key, "x_closing")
            barrier_frontier = float(barrier_raw["physical_frontier_us"])
            if not (
                float(barrier_x["service_start_ts_us"])
                <= barrier_frontier
                <= float(barrier_x["service_end_ts_us"])
            ):
                failures.append(
                    f"trial={trial}/policy={policy}/barrier_frontier_outside_closing"
                )
            if policy == "candidate_closing_first":
                y_start = float(streaming_y["service_start_ts_us"])
                if release >= y_start or downstream >= y_start:
                    failures.append(
                        f"trial={trial}/policy={policy}/nonclosing_before_early_use"
                    )
                barrier_y_start = float(
                    named(barrier_key, "y_nonclosing")["service_start_ts_us"]
                )
                if barrier_frontier >= barrier_y_start:
                    failures.append(
                        f"trial={trial}/policy={policy}/barrier_frontier_order"
                    )
            else:
                if float(streaming_y["service_end_ts_us"]) > float(
                    streaming_x["service_start_ts_us"]
                ):
                    failures.append(f"trial={trial}/policy={policy}/baseline_order")
                barrier_y = named(barrier_key, "y_nonclosing")
                if float(barrier_y["service_end_ts_us"]) > float(
                    barrier_x["service_start_ts_us"]
                ):
                    failures.append(
                        f"trial={trial}/policy={policy}/barrier_baseline_order"
                    )
    return failures


def validate_capability_artifact(
    directory: Path,
    *,
    model_key: str,
    model_revision: str,
    mode: str,
    config_sha256: str,
    protocol_sha256: str,
    data_manifest_sha256: str,
    model_tree_manifest_sha256: str,
    config: Mapping[str, Any],
    data_producer_signoff_sha256: str | None = None,
    gpu_environment_identity: Mapping[str, str] | None = None,
    historical_reviewed_source_snapshot_path: Path | None = None,
    pre_outcome_attestation: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], bool]:
    artifact = _read_self_hashed_json(
        directory / "capability_probe.json", schema_version="ric-capability-v1"
    )
    raw_path = directory / "capability_raw.csv"
    contribution_path = directory / "expert_contributions.pt"
    expected = {
        "mode": mode,
        "model_key": model_key,
        "model_revision": model_revision,
        "model_tree_manifest_sha256": model_tree_manifest_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "raw_trials_sha256": sha256_file(raw_path),
        "expert_contributions_sha256": sha256_file(contribution_path),
        "expert_contributions_source": "native_unpatched_model_expert_execution",
        "measure_capability_source_sha256": _capability_producer_source_sha256(),
        "ep_size": int(config["topology_proxy"]["ep_size"]),
        "ranks_per_node": int(config["topology_proxy"]["ranks_per_node"]),
        "num_experts": int(config["models"][model_key]["num_experts"]),
        "top_k": int(config["models"][model_key]["top_k"]),
    }
    for field, wanted in expected.items():
        if artifact.get(field) != wanted:
            raise ExperimentRunnerError(f"capability binding mismatch: {field}")
    if artifact.get("data_producer_signoff_sha256") != data_producer_signoff_sha256:
        raise ExperimentRunnerError(
            "capability data producer signoff binding mismatch"
        )
    if mode == "formal":
        if not is_sha256(data_producer_signoff_sha256):
            raise ExperimentRunnerError(
                "formal capability lacks data producer signoff binding"
            )
        try:
            capability_gpu = validate_gpu_environment_artifact(
                artifact, config=config, label="capability probe"
            )
        except ScenarioBuildError as exc:
            raise ExperimentRunnerError(str(exc)) from exc
        if capability_gpu != gpu_environment_identity:
            raise ExperimentRunnerError(
                "BLOCKED_GPU_ENVIRONMENT: capability/route/LUT environments differ"
            )
    wanted_status = "CAPABILITY_ONLY" if mode == "formal" else "NOT_TESTED"
    if (
        artifact.get("status") != wanted_status
        or type(artifact.get("scientific_result")) is not bool
        or bool(artifact.get("scientific_result"))
    ):
        raise ExperimentRunnerError("capability status/mode mismatch")
    capability_signoff_sha = artifact.get("signoff_sha256")
    if mode == "dev":
        if capability_signoff_sha is not None:
            raise ExperimentRunnerError("dev capability unexpectedly binds a signoff")
    elif (
        not isinstance(capability_signoff_sha, str)
        or len(capability_signoff_sha) != 64
        or any(
            character not in "0123456789abcdef"
            for character in capability_signoff_sha
        )
    ):
        raise ExperimentRunnerError("formal capability lacks producer signoff binding")
    early_trials = artifact.get("trials")
    configured_trials = config.get("capability_probes", {}).get("measured_trials")
    if type(configured_trials) is not int or configured_trials != 30:
        raise ExperimentRunnerError("frozen capability measured-trial count is invalid")
    if mode == "formal" and early_trials != configured_trials:
        raise ExperimentRunnerError("formal capability trial count is not frozen at 30")
    if mode == "formal":
        embedded_signoff = directory / EMBEDDED_PRODUCER_SIGNOFF
        if (
            not embedded_signoff.is_file()
            or sha256_file(embedded_signoff) != capability_signoff_sha
        ):
            raise ExperimentRunnerError(
                "formal capability embedded producer signoff mismatch"
            )
        try:
            if historical_reviewed_source_snapshot_path is None:
                _verify_capability_producer_signoff(
                    embedded_signoff,
                    protocol_sha256=protocol_sha256,
                    config_sha256=config_sha256,
                    source_sha256=_capability_producer_source_sha256(),
                    data_manifest_sha256=data_manifest_sha256,
                    data_producer_signoff_sha256=str(
                        data_producer_signoff_sha256
                    ),
                    model_key=model_key,
                    model_tree_manifest_sha256=model_tree_manifest_sha256,
                )
            else:
                if pre_outcome_attestation is None:
                    raise ExperimentRunnerError(
                        "historical capability reuse requires pre-outcome registry"
                    )
                verify_immutable_upstream_signoff(
                    embedded_signoff,
                    snapshot_path=historical_reviewed_source_snapshot_path,
                    expected_signoff_file_sha256=attested_file_sha256(
                        pre_outcome_attestation, embedded_signoff
                    ),
                    expected_fields={
                        "stage": "measure_capability",
                        "protocol_sha256": protocol_sha256,
                        "config_sha256": config_sha256,
                        "measure_capability_source_sha256": (
                            _capability_producer_source_sha256()
                        ),
                        "data_manifest_sha256": data_manifest_sha256,
                        "data_producer_signoff_sha256": str(
                            data_producer_signoff_sha256
                        ),
                        "model_key": model_key,
                        "model_tree_manifest_sha256": model_tree_manifest_sha256,
                    },
                )
        except Exception as exc:
            raise ExperimentRunnerError(
                "capability producer signoff is invalid"
            ) from exc
    model_spec = config["models"][model_key]
    request_id = artifact.get("request_id")
    target_layer = artifact.get("target_layer")
    frozen_layers = artifact.get("frozen_selected_layers")
    block_rows = artifact.get("block_rows")
    warmups = artifact.get("warmups")
    model_source = artifact.get("model_source")
    selected_layer_count = config.get("route_capture", {}).get(
        "selected_layer_count_per_model"
    )
    if (
        not isinstance(request_id, str)
        or not request_id.startswith("ric:calibration:0000:")
        or type(target_layer) is not int
        or not isinstance(frozen_layers, list)
        or type(selected_layer_count) is not int
        or len(frozen_layers) != selected_layer_count
        or any(type(layer) is not int or layer < 0 for layer in frozen_layers)
        or frozen_layers != sorted(set(frozen_layers))
        or target_layer not in frozen_layers
        or frozen_layers[
            int(hashlib.sha256(request_id.encode("utf-8")).hexdigest(), 16)
            % len(frozen_layers)
        ]
        != target_layer
        or type(block_rows) is not int
        or block_rows <= 0
        or type(warmups) is not int
        or warmups <= 0
        or artifact.get("evidence_boundary")
        != "REAL_5090_EXPERT_OUTPUT_AND_LOCAL_CUDA_STREAM / NOT_NCCL / NOT_RDMA"
        or artifact.get("ready_result_orders")
        != {
            "baseline": ["y_nonclosing", "x_closing"],
            "candidate": ["x_closing", "y_nonclosing"],
        }
        or artifact.get("release_modes")
        != ["streaming", "full_layer_barrier"]
        or not isinstance(artifact.get("transformers_version"), str)
        or not artifact.get("transformers_version")
        or not isinstance(model_source, Mapping)
        or model_source.get("frozen_repo_id") != model_spec["repo_id"]
        or model_source.get("frozen_revision") != model_spec["revision"]
        or model_source.get("expected_tree_manifest_sha256")
        != model_tree_manifest_sha256
        or model_source.get("tree_manifest_sha256")
        != model_tree_manifest_sha256
    ):
        raise ExperimentRunnerError("capability root fixture provenance is invalid")
    formal_block_rows = config.get("capability_probes", {}).get(
        "formal_block_rows"
    )
    formal_warmups = config.get("capability_probes", {}).get("formal_warmups")
    if (
        type(formal_block_rows) is not int
        or formal_block_rows != 32
        or type(formal_warmups) is not int
        or formal_warmups != 10
    ):
        raise ExperimentRunnerError("frozen formal capability shape is invalid")
    if mode == "formal" and (
        block_rows != formal_block_rows or warmups != formal_warmups
    ):
        raise ExperimentRunnerError("formal capability block/warmup shape is invalid")
    with raw_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ExperimentRunnerError("capability raw repeats are missing")
    trials = artifact.get("trials")
    if type(trials) is not int or trials <= 0 or len(rows) != trials * 4:
        raise ExperimentRunnerError("capability raw row count is not trials x 4")
    required_policies = {"baseline_nonclosing_first", "candidate_closing_first"}
    required_release = {"streaming", "full_layer_barrier"}
    if {row.get("policy") for row in rows} != required_policies:
        raise ExperimentRunnerError("capability action rows do not cover both policies")
    if {row.get("release_mode") for row in rows} != required_release:
        raise ExperimentRunnerError("capability rows do not cover streaming/barrier")
    canonical_reference = str(artifact.get("canonical_reference_sha256", ""))
    if (
        len(canonical_reference) != 64
        or any(character not in "0123456789abcdef" for character in canonical_reference)
    ):
        raise ExperimentRunnerError("capability canonical reference hash is invalid")
    observed_grid: set[tuple[int, str, str]] = set()
    for row in rows:
        try:
            trial = int(str(row.get("trial")))
            event_times = [
                float(row[field])
                for field in (
                    "physical_frontier_us",
                    "application_release_us",
                    "downstream_start_us",
                    "total_us",
                )
            ]
            row_layer_id = int(str(row.get("layer_id")))
            row_block_rows = int(str(row.get("block_rows")))
            row_top_k = int(str(row.get("top_k")))
            row_sender_rank = int(str(row.get("sender_rank")))
            row_receiver_rank = int(str(row.get("receiver_rank")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentRunnerError(
                "capability raw event schema is invalid"
            ) from exc
        if not all(math.isfinite(value) and value >= 0.0 for value in event_times):
            raise ExperimentRunnerError("capability raw event time is invalid")
        observed_grid.add((trial, str(row.get("policy")), str(row.get("release_mode"))))
        if (
            row.get("model_key") != model_key
            or row.get("model_revision") != model_revision
            or row.get("request_id") != request_id
            or row_layer_id != target_layer
            or row_block_rows != block_rows
            or row_top_k != int(artifact["top_k"])
            or row.get("source") != "measured_5090_cuda"
            or row.get("canonical_reference_sha256") != canonical_reference
            or row.get("canonical_output_sha256") != canonical_reference
            or row.get("canonical_equal") not in {"True", "true", "1"}
            or row_sender_rank != int(artifact["sender_rank"])
            or row_receiver_rank != int(artifact["receiver_rank"])
        ):
            raise ExperimentRunnerError("capability raw row provenance mismatch")
    canonical_ok = all(
        row.get("canonical_output_sha256") == canonical_reference
        and row.get("canonical_reference_sha256") == canonical_reference
        for row in rows
    )
    if artifact.get("all_canonical_hashes_equal") is not canonical_ok:
        raise ExperimentRunnerError(
            "capability canonical exactness is not raw-derived"
        )
    expected_grid = {
        (trial, policy, release)
        for trial in range(trials)
        for policy in required_policies
        for release in required_release
    }
    if observed_grid != expected_grid:
        raise ExperimentRunnerError("capability raw trial/policy/release grid mismatch")
    if type(artifact.get("sender_local_stream_id")) is not int:
        raise ExperimentRunnerError("capability persistent stream identity is invalid")
    try:
        observed_stream_ids = {int(str(row.get("stream_id"))) for row in rows}
        sender_local_stream_id = int(artifact.get("sender_local_stream_id"))
    except (TypeError, ValueError) as exc:
        raise ExperimentRunnerError("capability persistent stream identity is invalid") from exc
    if (
        config.get("capability_probes", {}).get(
            "persistent_sender_local_cuda_stream_across_all_arms"
        )
        is not True
        or artifact.get("persistent_sender_local_cuda_stream_across_all_arms")
        is not True
        or sender_local_stream_id <= 0
        or observed_stream_ids != {sender_local_stream_id}
    ):
        raise ExperimentRunnerError("capability arms did not reuse one CUDA stream")
    grouped_actions = validate_capability_action_trace(
        directory,
        artifact=artifact,
        raw_rows=rows,
        trials=trials,
        config=config,
    )
    summary = artifact.get("summary")
    if not isinstance(summary, Mapping):
        raise ExperimentRunnerError("capability summary missing")
    capability_cfg = config.get("capability_probes")
    if not isinstance(capability_cfg, Mapping):
        raise ExperimentRunnerError("frozen capability config missing")
    event_gate = capability_cfg.get("event_precedence_gate")
    expected_event_fields = {
        "same_policy_cross_release_task_identity_queue_payload_and_order_exact",
        "barrier_release_after_both_task_service_end",
        "barrier_downstream_not_before_release",
        "streaming_release_at_or_after_physical_frontier",
        "streaming_downstream_not_before_release",
        "physical_frontier_within_closing_service",
        "candidate_streaming_release_and_downstream_before_nonclosing_service",
        "candidate_barrier_frontier_before_nonclosing_service",
        "baseline_nonclosing_service_before_closing_service",
        "all_trials_and_policies_required",
    }
    try:
        existence = capability_cfg["paired_existence_gate"]
        replicates = int(existence["bootstrap_replicates"])
        confidence = float(existence["one_sided_confidence"])
        bootstrap_seed = int(existence["bootstrap_seed"])
        order_statistic_one_based = int(
            existence["lcb_order_statistic_one_based"]
        )
        lcb_threshold_us = float(
            existence["each_effect_lcb_must_be_strictly_greater_than_us"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentRunnerError("capability paired gate config is incomplete") from exc
    expected_effect_definitions = {
        "frontier_advance": (
            "baseline_frontier_release_us-candidate_frontier_release_us"
        ),
        "downstream_start_advance": (
            "baseline_downstream_start_us-candidate_downstream_start_us"
        ),
        "release_interaction": (
            "(baseline-candidate)_streaming_release-"
            "(baseline-candidate)_barrier_release"
        ),
        "downstream_interaction": (
            "(baseline-candidate)_streaming_downstream-"
            "(baseline-candidate)_barrier_downstream"
        ),
    }
    if (
        not isinstance(event_gate, Mapping)
        or set(event_gate) != expected_event_fields
        or any(event_gate[field] is not True for field in expected_event_fields)
        or capability_cfg.get("barrier_cross_policy_timing_is_diagnostic_only")
        is not True
        or capability_cfg.get("artifact_boolean_self_attestation_allowed")
        is not False
        or existence.get("effect_definitions_us") != expected_effect_definitions
        or existence.get("estimator")
        != "mean_of_30_within_trial_paired_differences"
        or existence.get("bootstrap_unit") != "within_trial_pair"
        or existence.get("bootstrap_replicates") != 10000
        or existence.get("bootstrap_seed") != 2026072226
        or type(existence.get("one_sided_confidence")) is not float
        or existence.get("one_sided_confidence") != 0.95
        or existence.get("quantile") != "type1_nearest_rank"
        or existence.get("lcb_order_statistic_one_based") != 500
        or type(
            existence.get("each_effect_lcb_must_be_strictly_greater_than_us")
        )
        not in {int, float}
        or existence.get("each_effect_lcb_must_be_strictly_greater_than_us")
        != 0.0
        or existence.get("logical_operator") != "AND"
    ):
        raise ExperimentRunnerError("frozen Amendment-O capability config is invalid")
    index = {
        (int(row["trial"]), str(row["policy"]), str(row["release_mode"])): row
        for row in rows
    }
    frontier = [
        float(index[(trial, "baseline_nonclosing_first", "streaming")]["application_release_us"])
        - float(index[(trial, "candidate_closing_first", "streaming")]["application_release_us"])
        for trial in range(trials)
    ]
    downstream = [
        float(index[(trial, "baseline_nonclosing_first", "streaming")]["downstream_start_us"])
        - float(index[(trial, "candidate_closing_first", "streaming")]["downstream_start_us"])
        for trial in range(trials)
    ]
    barrier_frontier = [
        float(
            index[
                (trial, "baseline_nonclosing_first", "full_layer_barrier")
            ]["application_release_us"]
        )
        - float(
            index[
                (trial, "candidate_closing_first", "full_layer_barrier")
            ]["application_release_us"]
        )
        for trial in range(trials)
    ]
    barrier_downstream = [
        float(
            index[
                (trial, "baseline_nonclosing_first", "full_layer_barrier")
            ]["downstream_start_us"]
        )
        - float(
            index[
                (trial, "candidate_closing_first", "full_layer_barrier")
            ]["downstream_start_us"]
        )
        for trial in range(trials)
    ]
    release_interaction = [
        streaming - barrier
        for streaming, barrier in zip(frontier, barrier_frontier)
    ]
    downstream_interaction = [
        streaming - barrier
        for streaming, barrier in zip(downstream, barrier_downstream)
    ]
    lcbs = capability_paired_lcbs(
        {
            "frontier": frontier,
            "downstream": downstream,
            "release_interaction": release_interaction,
            "downstream_interaction": downstream_interaction,
        },
        replicates=replicates,
        order_statistic_one_based=order_statistic_one_based,
        seed=bootstrap_seed,
    )
    event_failures = capability_event_precedence_failures(
        rows, grouped_actions, trials=trials
    )
    event_ok = not event_failures

    streaming_baseline = [
        index[(trial, "baseline_nonclosing_first", "streaming")]
        for trial in range(trials)
    ]
    streaming_candidate = [
        index[(trial, "candidate_closing_first", "streaming")]
        for trial in range(trials)
    ]
    barrier_baseline = [
        index[(trial, "baseline_nonclosing_first", "full_layer_barrier")]
        for trial in range(trials)
    ]
    barrier_candidate = [
        index[(trial, "candidate_closing_first", "full_layer_barrier")]
        for trial in range(trials)
    ]

    def median_field(selected_rows: Sequence[Mapping[str, str]], field: str) -> float:
        return float(statistics.median(float(row[field]) for row in selected_rows))

    numeric_summary = {
        "streaming_frontier_advance_us": median_field(
            streaming_baseline, "application_release_us"
        )
        - median_field(streaming_candidate, "application_release_us"),
        "streaming_downstream_advance_us": median_field(
            streaming_baseline, "downstream_start_us"
        )
        - median_field(streaming_candidate, "downstream_start_us"),
        "barrier_application_release_difference_us": median_field(
            barrier_baseline, "application_release_us"
        )
        - median_field(barrier_candidate, "application_release_us"),
        "barrier_downstream_start_difference_us": median_field(
            barrier_baseline, "downstream_start_us"
        )
        - median_field(barrier_candidate, "downstream_start_us"),
        "baseline_streaming_release_median_us": median_field(
            streaming_baseline, "application_release_us"
        ),
        "candidate_streaming_release_median_us": median_field(
            streaming_candidate, "application_release_us"
        ),
        "baseline_barrier_release_median_us": median_field(
            barrier_baseline, "application_release_us"
        ),
        "candidate_barrier_release_median_us": median_field(
            barrier_candidate, "application_release_us"
        ),
        "baseline_barrier_total_median_us": median_field(
            barrier_baseline, "total_us"
        ),
        "streaming_frontier_paired_mean_us": statistics.fmean(frontier),
        "streaming_frontier_paired_lcb_us": lcbs["frontier"],
        "streaming_downstream_paired_mean_us": statistics.fmean(downstream),
        "streaming_downstream_paired_lcb_us": lcbs["downstream"],
        "release_interaction_paired_mean_us": statistics.fmean(
            release_interaction
        ),
        "release_interaction_paired_lcb_us": lcbs["release_interaction"],
        "downstream_interaction_paired_mean_us": statistics.fmean(
            downstream_interaction
        ),
        "downstream_interaction_paired_lcb_us": lcbs[
            "downstream_interaction"
        ],
        "barrier_application_release_paired_mean_us_diagnostic": statistics.fmean(
            barrier_frontier
        ),
        "barrier_application_release_max_abs_paired_difference_us_diagnostic": max(
            abs(value) for value in barrier_frontier
        ),
        "barrier_downstream_paired_mean_us_diagnostic": statistics.fmean(
            barrier_downstream
        ),
        "barrier_downstream_max_abs_paired_difference_us_diagnostic": max(
            abs(value) for value in barrier_downstream
        ),
    }
    if (
        summary.get("event_precedence_all_trials_pass") is not event_ok
        or summary.get("event_precedence_failure_count") != len(event_failures)
        or summary.get("event_precedence_failures") != event_failures
    ):
        raise ExperimentRunnerError(
            "capability event precedence is not raw/action-derived"
        )
    if (
        artifact.get("execution_order_rule") != EXECUTION_ORDER_RULE
        or summary.get("paired_bootstrap_replicates") != replicates
        or summary.get("paired_bootstrap_seed") != bootstrap_seed
        or summary.get("paired_lcb_order_statistic_one_based")
        != order_statistic_one_based
        or not math.isclose(
            float(summary.get("paired_one_sided_confidence", -1.0)),
            confidence,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or summary.get("barrier_cross_policy_timing_diagnostic_only") is not True
        or any(
            not math.isclose(
                float(summary.get(field)),
                value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for field, value in numeric_summary.items()
        )
    ):
        raise ExperimentRunnerError("capability paired gate is not raw/config-derived")
    gate_pass = (
        canonical_ok
        and lcbs["frontier"] > lcb_threshold_us
        and lcbs["downstream"] > lcb_threshold_us
        and lcbs["release_interaction"] > lcb_threshold_us
        and lcbs["downstream_interaction"] > lcb_threshold_us
        and event_ok
    )
    return artifact, gate_pass


def _run_trace_bundle(
    payload: tuple[ReplayWorld, tuple[str, ...], ReplayConfig, bool]
) -> tuple[ReplayResult, ...]:
    world, arms, config, include_sham = payload
    sender_pack_values = {
        float(task.stage_service.sender_pack_us) for task in world.tasks
    }
    if (
        config.drr_service_fingerprint != world.service_fingerprint
        or len(sender_pack_values) != 1
        or not math.isclose(
            config.drr_quantum_us,
            next(iter(sender_pack_values)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ExperimentRunnerError(
            "DRR quantum/fingerprint is not bound to the exact replay world"
        )
    results = [run_replay(world, arm=arm, config=config) for arm in arms]
    if include_sham:
        charged = next(
            (result for result in results if result.arm == "ric_wire_charged"), None
        )
        if charged is None or charged.control_plan is None:
            raise ExperimentRunnerError("sham requested without canonical charged plan")
        results.append(
            run_sham_against_reference(
                world, charged_plan=charged.control_plan, config=config
            )
        )
    # Full three-stage invariants are checked before any persistence-only
    # compaction.  The compact object retains every field consumed by metrics,
    # conservation, action signatures, accounting and wire evidence.
    assert_replay_conservation(tuple(results))
    if include_sham:
        assert_sham_feedback_cost_equivalence(
            next(row for row in results if row.arm == "ric_wire_charged"),
            next(row for row in results if row.arm == "ric_sham_feedback"),
        )
    return tuple(compact_replay_result(result) for result in results)


def compact_replay_result(result: ReplayResult) -> ReplayResult:
    """Drop non-consumed bulk only after a fully validated DES result exists."""

    if not result.full_drain or result.completed_stage_count != result.expected_stage_count:
        raise ExperimentRunnerError("cannot compact a replay before full conservation")
    return replace(
        result,
        action_trace=tuple(
            action for action in result.action_trace if action.stage == "sender_egress"
        ),
        completion_by_task_us={},
        join_completion_us={},
        all_join_latencies_us={},
        control_plan=(None if result.arm == "ric_sham_feedback" else result.control_plan),
    )


def execute_trace_bundles(
    worlds: Sequence[ReplayWorld],
    *,
    arms: Sequence[str],
    configs_by_model: Mapping[str, ReplayConfig],
    workers: int,
    include_sham: bool = False,
) -> tuple[ReplayResult, ...]:
    ordered_worlds = tuple(sorted(worlds, key=lambda row: (row.model_key, row.cell, row.workload_seed)))
    worker_count = resolve_worker_count(workers, len(ordered_worlds))
    jobs = []
    for world in ordered_worlds:
        base = configs_by_model[world.model_key]
        jobs.append(
            (
                world,
                tuple(arms),
                replace(base, drr_service_fingerprint=world.service_fingerprint),
                include_sham,
            )
        )
    if worker_count == 1:
        bundles = [_run_trace_bundle(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            bundles = list(executor.map(_run_trace_bundle, jobs, chunksize=1))
    results = tuple(
        sorted(
            (result for bundle in bundles for result in bundle),
            key=lambda row: (row.model_key, row.cell, row.workload_seed, row.arm),
        )
    )
    expected = len(worlds) * (len(arms) + int(include_sham))
    if len(results) != expected:
        raise ExperimentRunnerError("trace process pool returned an incomplete arm grid")
    return results


def _pooled_latencies(results: Sequence[ReplayResult]) -> tuple[float, ...]:
    values = tuple(
        float(value)
        for result in results
        for value in result.scored_join_latencies_us.values()
    )
    if not values:
        raise ExperimentRunnerError("baseline has no scored closures")
    return values


def _drr_quantum_from_tree(tree: Mapping[str, Any]) -> float:
    try:
        quantum = float(tree["service_surface"]["sender_pack_us"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentRunnerError("scenario lacks measured DRR quantum") from exc
    if not math.isfinite(quantum) or quantum <= 0.0:
        raise ExperimentRunnerError("scenario DRR quantum is not positive and finite")
    return quantum


def select_calibration_baseline(
    results_by_arm: Mapping[str, Sequence[ReplayResult]],
    *,
    frozen_arm_order: Sequence[str],
) -> Mapping[str, Any]:
    if set(results_by_arm) != set(CONCRETE_JOINBLIND_ARMS):
        raise ExperimentRunnerError("calibration baseline grid is incomplete")
    if set(frozen_arm_order) != set(CONCRETE_JOINBLIND_ARMS):
        raise ExperimentRunnerError("frozen concrete baseline order mismatch")
    mean_trace_cvar = {
        arm: statistics.mean(
            empirical_cvar(tuple(result.scored_join_latencies_us.values()), 0.99)
            for result in results
        )
        for arm, results in results_by_arm.items()
    }
    order_index = {arm: index for index, arm in enumerate(frozen_arm_order)}
    selected = min(
        mean_trace_cvar,
        key=lambda arm: (mean_trace_cvar[arm], order_index[arm]),
    )
    budget = quantile_type1(_pooled_latencies(results_by_arm[selected]), 0.95)
    violation = {
        arm: sum(value > budget for value in _pooled_latencies(results))
        / len(_pooled_latencies(results))
        for arm, results in results_by_arm.items()
    }
    return {
        "calib_best_joinblind": selected,
        "closure_budget_us": budget,
        "selection_metric": "mean_complete_trace_cvar99_then_frozen_arm_order",
        "mean_complete_trace_cvar99_us_by_arm": mean_trace_cvar,
        "pooled_violation_rate_report_only_by_arm": violation,
        "selection_used_violation": False,
    }


def _metric_rows(
    results: Sequence[ReplayResult],
    *,
    budgets: Mapping[tuple[str, str], float],
) -> tuple[TraceMetrics, ...]:
    return tuple(
        trace_metrics_from_result(
            result,
            closure_budget_us=float(budgets[(result.model_key, result.cell)]),
        )
        for result in results
    )


def _result_groups(
    results: Sequence[ReplayResult],
) -> Mapping[tuple[str, str, str], tuple[ReplayResult, ...]]:
    grouped: dict[tuple[str, str, str], list[ReplayResult]] = {}
    for result in results:
        grouped.setdefault((result.model_key, result.cell, result.trace_id), []).append(result)
    return {key: tuple(value) for key, value in grouped.items()}


def _assert_result_grid(results: Sequence[ReplayResult]) -> None:
    for grouped in _result_groups(results).values():
        assert_replay_conservation(grouped)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        # Required empty artifacts retain an explicit schema marker.
        path.write_text("empty\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if list(row) != fields:
                raise ExperimentRunnerError("CSV row schema drift")
            writer.writerow(row)


def _accounting_row(
    result: ReplayResult, *, closure_budget_us: float
) -> dict[str, Any]:
    return {
        "trace_id": result.trace_id,
        "workload_seed": result.workload_seed,
        "model_key": result.model_key,
        "cell": result.cell,
        "arm": result.arm,
        "task_count": result.task_count,
        "completed_task_count": result.completed_task_count,
        "completed_stage_count": result.completed_stage_count,
        "expected_stage_count": result.expected_stage_count,
        "completed_join_count": result.completed_join_count,
        "expected_join_count": result.expected_join_count,
        "closure_budget_us": closure_budget_us,
        "payload_bytes": result.payload_bytes,
        "descriptor_bytes": result.descriptor_bytes,
        "alignment_bytes": result.alignment_bytes,
        "contract_bytes": result.contract_bytes,
        "contract_received_bytes": result.contract_received_bytes,
        "contract_header_bytes": result.contract_header_bytes,
        "contract_record_bytes": result.contract_record_bytes,
        "contract_alignment_bytes": result.contract_alignment_bytes,
        "contract_messages": result.contract_messages,
        "contract_record_count_histogram": json.dumps(
            result.contract_record_count_histogram, sort_keys=True
        ),
        "contract_tax_surface_source_id": result.contract_tax_surface_source_id,
        "contract_tax_surface_fingerprint": result.contract_tax_surface_fingerprint,
        "contract_tax_non_grid_rule": result.contract_tax_non_grid_rule,
        "control_component_us": json.dumps(result.control_component_us, sort_keys=True),
        "stale_decisions": result.stale_decisions,
        "fallback_decisions": result.fallback_decisions,
        "sender_decisions": result.sender_decisions,
        "starvation_count": result.starvation_count,
        "makespan_us": result.makespan_us,
        "full_drain": result.full_drain,
        "task_fingerprint": result.task_fingerprint,
        "service_fingerprint": result.service_fingerprint,
        "score_mask_fingerprint": result.score_mask_fingerprint,
        "resource_demand_fingerprint": result.resource_demand_fingerprint,
        "queue_busy_us": json.dumps(result.queue_busy_us, sort_keys=True),
        "resource_service_demand_us": json.dumps(
            result.resource_service_demand_us, sort_keys=True
        ),
        "source_by_field": json.dumps(result.source_by_field, sort_keys=True),
        "source_tags": json.dumps(result.source_tags),
    }


def _validate_persisted_accounting(path: Path, *, expected_rows: int) -> None:
    """Re-read the CSV and independently enforce frozen byte/stage/resource ledgers."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise ExperimentRunnerError("persisted accounting row count mismatch")
    for row in rows:
        try:
            task_count = int(row["task_count"])
            completed_tasks = int(row["completed_task_count"])
            completed_stages = int(row["completed_stage_count"])
            expected_stages = int(row["expected_stage_count"])
            completed_joins = int(row["completed_join_count"])
            expected_joins = int(row["expected_join_count"])
            produced = int(row["contract_bytes"])
            received = int(row["contract_received_bytes"])
            byte_parts = sum(
                int(row[field])
                for field in (
                    "contract_header_bytes",
                    "contract_record_bytes",
                    "contract_alignment_bytes",
                )
            )
            busy = loads_json_mapping_strict(
                row["queue_busy_us"], label="accounting queue_busy_us"
            )
            demand = loads_json_mapping_strict(
                row["resource_service_demand_us"],
                label="accounting resource_service_demand_us",
            )
            sources = loads_json_mapping_strict(
                row["source_by_field"], label="accounting source_by_field"
            )
            budget = float(row["closure_budget_us"])
        except (KeyError, TypeError, ValueError, FormalProvenanceError) as exc:
            raise ExperimentRunnerError("persisted accounting schema is invalid") from exc
        if (
            completed_tasks != task_count
            or completed_joins != expected_joins
            or expected_stages != 3 * task_count + expected_joins
            or completed_stages != expected_stages
            or produced != byte_parts
            or not 0 <= received <= produced
            or not math.isfinite(budget)
            or budget <= 0.0
            or busy != demand
            or set(sources) != {f"resource:{key}" for key in busy}
            or row.get("full_drain") not in {"True", "true", "1"}
        ):
            raise ExperimentRunnerError("persisted accounting conservation failed")
        for field in (
            "task_fingerprint",
            "service_fingerprint",
            "score_mask_fingerprint",
            "resource_demand_fingerprint",
        ):
            value = row.get(field, "")
            if len(value) != 64:
                raise ExperimentRunnerError(
                    "persisted accounting fingerprint is incomplete"
                )


def _action_rows(results: Sequence[ReplayResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for index, action in enumerate(result.action_trace):
            rows.append(
                {
                    "trace_id": result.trace_id,
                    "workload_seed": result.workload_seed,
                    "model_key": result.model_key,
                    "cell": result.cell,
                    "arm": result.arm,
                    "action_index": index,
                    **asdict(action),
                }
            )
    return rows


def _write_wire_trace(path: Path, results: Sequence[ReplayResult]) -> None:
    with path.open("xb") as handle:
        for result in sorted(results, key=lambda row: (row.trace_id, row.arm)):
            if result.control_plan is None:
                continue
            for event in result.control_plan.events:
                header = {
                    "trace_id": result.trace_id,
                    "workload_seed": result.workload_seed,
                    "arm": result.arm,
                    "emission_us": event.emission_us,
                    "delivery_us": event.delivery_us,
                    "sender_rank": event.sender_rank,
                    "receiver_rank": event.receiver_rank,
                    "sequence": event.sequence,
                    "record_count": event.record_count,
                    "tax": asdict(event.tax),
                }
                metadata = json.dumps(header, sort_keys=True).encode("utf-8")
                handle.write(struct.pack("<II", len(metadata), len(event.payload)))
                handle.write(metadata)
                handle.write(event.payload)


def _jsonable_summary(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable_summary(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable_summary(item) for item in value]
    return value


def write_run_artifacts(
    *,
    output_dir: Path,
    mode: str,
    stage: str,
    results: Sequence[ReplayResult],
    metrics: Sequence[TraceMetrics],
    bootstrap: Mapping[str, Any],
    gate_matrix: Mapping[str, Any],
    decision: Mapping[str, Any],
    metadata: Mapping[str, Any],
    calibration_lock: Mapping[str, Any] | None = None,
    sealed_consumption_record: Mapping[str, Any] | None = None,
    producer_signoff_path: Path | None = None,
) -> Mapping[str, Any]:
    with atomic_output_directory(output_dir) as temporary:
        producer_signoff_sha256 = None
        if mode == "formal":
            try:
                producer_signoff_sha256 = materialize_verified_signoff(
                    producer_signoff_path, temporary
                )
            except FormalProvenanceError as exc:
                raise ExperimentRunnerError(str(exc)) from exc
        elif producer_signoff_path is not None:
            raise ExperimentRunnerError("development run cannot carry a signoff")
        metric_index = {
            (row.trace_id, row.model_key, row.cell, row.arm): row
            for row in metrics
        }
        if len(metric_index) != len(results):
            raise ExperimentRunnerError("accounting/result metric grid mismatch")
        _write_csv(
            temporary / "accounting.csv",
            [
                _accounting_row(
                    row,
                    closure_budget_us=metric_index[
                        (row.trace_id, row.model_key, row.cell, row.arm)
                    ].closure_budget_us,
                )
                for row in results
            ],
        )
        _validate_persisted_accounting(
            temporary / "accounting.csv", expected_rows=len(results)
        )
        _write_csv(temporary / "action_trace.csv", _action_rows(results))
        _write_csv(
            temporary / "per_trace_metrics.csv", [asdict(row) for row in metrics]
        )
        _write_wire_trace(temporary / "contract_wire_trace.bin", results)
        bootstrap_payload = add_self_hash(
            {
                "schema_version": "ric-paired-bootstrap-v1",
                "mode": mode,
                "stage": stage,
                "signoff_sha256": producer_signoff_sha256,
                "rows": _jsonable_summary(bootstrap),
            }
        )
        gate_payload = add_self_hash(
            {
                "schema_version": "ric-gate-matrix-v1",
                "mode": mode,
                "stage": stage,
                "gates": _jsonable_summary(gate_matrix),
            }
        )
        decision_payload = add_self_hash(
            {
                "schema_version": "ric-decision-v1",
                "mode": mode,
                "stage": stage,
                **_jsonable_summary(decision),
            }
        )
        for name, payload in (
            ("paired_bootstrap.json", bootstrap_payload),
            ("gate_matrix.json", gate_payload),
            ("decision.json", decision_payload),
        ):
            (temporary / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if calibration_lock is not None:
            (temporary / "calibration_lock.json").write_text(
                json.dumps(calibration_lock, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if sealed_consumption_record is not None:
            sealed_record_copy = temporary / "sealed_consumption_record.json"
            sealed_record_copy.write_text(
                json.dumps(sealed_consumption_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            sealed_record_evidence = {
                "sealed_consumption_record_self_hash": (
                    sealed_consumption_record["manifest_sha256"]
                ),
                "sealed_consumption_record_file_sha256": sha256_file(
                    sealed_record_copy
                ),
            }
        else:
            sealed_record_evidence = {}
        status = add_self_hash(
            {
                "schema_version": "ric-run-status-v1",
                "status": (
                    "NOT_TESTED" if mode == "dev" else str(decision_payload["verdict"])
                ),
                "formal_run_valid": bool(
                    mode == "formal" and decision_payload.get("formal_run_valid")
                ),
                "scientific_result": bool(
                    mode == "formal" and decision_payload.get("scientific_result")
                ),
                "mode": mode,
                "stage": stage,
                "run_experiment_source_sha256": (
                    _sealed_source_sha256() if stage == "sealed" else _source_sha256()
                ),
                **dict(metadata),
                "accounting_sha256": sha256_file(temporary / "accounting.csv"),
                "action_trace_sha256": sha256_file(temporary / "action_trace.csv"),
                "per_trace_metrics_sha256": sha256_file(
                    temporary / "per_trace_metrics.csv"
                ),
                "contract_wire_trace_sha256": sha256_file(
                    temporary / "contract_wire_trace.bin"
                ),
                "paired_bootstrap_sha256": bootstrap_payload["manifest_sha256"],
                "gate_matrix_sha256": gate_payload["manifest_sha256"],
                "decision_sha256": decision_payload["manifest_sha256"],
                **sealed_record_evidence,
            }
        )
        (temporary / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = [
            "# RIC-v1 run report",
            "",
            f"- mode: `{mode}`",
            f"- stage: `{stage}`",
            f"- verdict: `{decision_payload['verdict']}`",
            f"- formal_run_valid: `{decision_payload.get('formal_run_valid', False)}`",
            "",
            "## Primary G0-G3 gate matrix",
            "",
            "```json",
            json.dumps(_jsonable_summary(gate_matrix), indent=2, sort_keys=True),
            "```",
        ]
        primary_cells = gate_matrix.get("G2_cells")
        if isinstance(primary_cells, Mapping):
            report.extend(
                [
                    "",
                    "## Four primary model x workload cells",
                    "",
                    "| Cell | G2 | G3 |",
                    "| --- | ---: | ---: |",
                ]
            )
            g3_cells = gate_matrix.get("G3_cells", {})
            for key in sorted(primary_cells):
                g3_value = (
                    g3_cells.get(key, "NOT_EXECUTED")
                    if isinstance(g3_cells, Mapping)
                    else "NOT_EXECUTED"
                )
                report.append(
                    f"| `{key}` | `{primary_cells[key]}` | `{g3_value}` |"
                )
        report.extend(
            [
                "",
                "## Sensitivities (secondary; never replace primary)",
                "",
                "```json",
                json.dumps(
                    _jsonable_summary(decision_payload.get("sensitivities", {})),
                    indent=2,
                    sort_keys=True,
                ),
                "```",
            ]
        )
        (temporary / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return status


def validate_calibration_signoff_bindings(
    *,
    signoff: Mapping[str, Any] | None,
    trees: Mapping[str, Mapping[str, Any]],
    capabilities: Mapping[str, Mapping[str, Any]],
    required_models: Sequence[str],
) -> None:
    """Require Phase-4 review to bind every formal calibration input."""

    if signoff is None:
        raise ExperimentRunnerError("formal calibration requires reviewed input bindings")
    actual_scenarios = {
        model: str(trees[model]["manifest_sha256"]) for model in required_models
    }
    actual_scenario_signoffs = {
        model: str(trees[model]["scenario_producer_signoff_sha256"])
        for model in required_models
    }
    actual_capabilities = {
        model: str(capabilities[model]["manifest_sha256"])
        for model in required_models
    }
    actual_capability_signoffs = {
        model: str(capabilities[model]["signoff_sha256"])
        for model in required_models
    }
    if signoff.get("scenario_tree_sha256") != actual_scenarios:
        raise ExperimentRunnerError(
            "formal calibration scenario inputs differ from Phase-4 signoff"
        )
    if signoff.get("scenario_producer_signoff_sha256") != actual_scenario_signoffs:
        raise ExperimentRunnerError(
            "formal calibration scenario producer signoffs differ from Phase-4 signoff"
        )
    if signoff.get("capability_probe_sha256") != actual_capabilities:
        raise ExperimentRunnerError(
            "formal calibration capability inputs differ from Phase-4 signoff"
        )
    if (
        signoff.get("capability_producer_signoff_sha256")
        != actual_capability_signoffs
    ):
        raise ExperimentRunnerError(
            "formal calibration capability producer signoffs differ from Phase-4 signoff"
        )


def calibrate_pipeline(
    *,
    scenario_dirs: Sequence[Path],
    capability_dirs: Sequence[Path],
    output_dir: Path,
    mode: str,
    workers: int,
    config_path: Path,
    protocol_path: Path,
    signoff: Mapping[str, Any] | None = None,
    signoff_path: Path | None = None,
    historical_reviewed_source_snapshot_path: Path | None = None,
    pre_outcome_attestation_path: Path | None = None,
    pre_outcome_producer_signoff_path: Path | None = None,
    authoritative_bundle_root_path: Path | None = None,
    consumer_amendment_path: Path = DEFAULT_CONSUMER_AMENDMENT,
) -> Mapping[str, Any]:
    try:
        validate_formal_output_path(output_dir, mode=mode)
    except ScenarioBuildError as exc:
        raise ExperimentRunnerError(str(exc)) from exc
    try:
        validate_frozen_formal_paths(
            config_path=config_path, protocol_path=protocol_path, mode=mode
        )
    except ScenarioBuildError as exc:
        raise ExperimentRunnerError(str(exc)) from exc
    config = _load_config(config_path)
    config_sha = sha256_file(config_path)
    protocol_sha = sha256_file(protocol_path)
    pre_outcome_attestation: Mapping[str, Any] | None = None
    if mode == "formal":
        try:
            validate_consumer_amendment_path(
                consumer_amendment_path, mode="formal"
            )
        except ScenarioBuildError as exc:
            raise ExperimentRunnerError(str(exc)) from exc
        if (
            historical_reviewed_source_snapshot_path is None
            or pre_outcome_attestation_path is None
            or pre_outcome_producer_signoff_path is None
            or authoritative_bundle_root_path is None
        ):
            raise ExperimentRunnerError(
                "formal calibration requires the complete Amendment-Q registry"
            )
        capability_input_paths = [
            directory / name
            for directory in capability_dirs
            for name in (
                "capability_probe.json",
                "capability_raw.csv",
                "expert_contributions.pt",
                "capability_action_trace.jsonl",
                "capability_cuda_trace_full_layer_barrier.json",
                "capability_cuda_trace_streaming.json",
                EMBEDDED_PRODUCER_SIGNOFF,
            )
        ]
        try:
            pre_outcome_attestation = verify_pre_outcome_attestation(
                pre_outcome_attestation_path,
                protocol_sha256=protocol_sha,
                config_sha256=config_sha,
                consumer_amendment_sha256=sha256_file(consumer_amendment_path),
                authoritative_bundle_root=authoritative_bundle_root_path,
                required_input_paths=capability_input_paths,
                producer_signoff_path=pre_outcome_producer_signoff_path,
            )
        except ScenarioBuildError as exc:
            raise ExperimentRunnerError(str(exc)) from exc
    trees: dict[str, Mapping[str, Any]] = {}
    worlds: list[ReplayWorld] = []
    for directory in scenario_dirs:
        tree, loaded = load_worlds(directory, expected_role="calibration", mode=mode)
        model_key = str(tree["model_key"])
        if model_key in trees:
            raise ExperimentRunnerError("duplicate calibration scenario model")
        if int(tree["link_gbps"]) != int(config["topology_proxy"]["primary_link_gbps"]):
            raise ExperimentRunnerError("calibration lock consumes primary link only")
        if tree.get("config_sha256") != config_sha or tree.get("protocol_sha256") != protocol_sha:
            raise ExperimentRunnerError("calibration scenario/config binding mismatch")
        trees[model_key] = tree
        worlds.extend(loaded)
    required_models = tuple(config["go_no_go"]["required_models"])
    if set(trees) != set(required_models):
        raise ExperimentRunnerError("calibration requires exactly both frozen models")
    if mode == "formal":
        assert pre_outcome_attestation is not None
        expected_migration = {
            "consumer_amendment_sha256": sha256_file(consumer_amendment_path),
            "historical_reviewed_source_snapshot_sha256": (
                HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256
            ),
            "pre_outcome_attestation_sha256": pre_outcome_attestation[
                "attestation_sha256"
            ],
            "pre_outcome_producer_signoff_file_sha256": (
                pre_outcome_attestation["producer_signoff_file_sha256"]
            ),
            "pre_outcome_producer_signoff_self_hash": (
                pre_outcome_attestation["producer_signoff_self_hash"]
            ),
            "authoritative_bundle_root": str(
                authoritative_bundle_root_path.resolve(strict=True)
            ),
        }
        for model, tree in trees.items():
            for field, wanted in expected_migration.items():
                if tree.get(field) != wanted:
                    raise ExperimentRunnerError(
                        f"calibration scenario migration mismatch: {model}/{field}"
                    )
        verified_calibration_signoff = _load_signoff(
            signoff_path,
            stage="calibration",
            config=config_path,
            protocol=protocol_path,
            migration_expected_fields=expected_migration,
        )
        if signoff is None or dict(signoff) != dict(verified_calibration_signoff):
            raise ExperimentRunnerError(
                "calibration signoff mapping/file identity mismatch"
            )
        signoff = verified_calibration_signoff
    capabilities: dict[str, Mapping[str, Any]] = {}
    g1_by_model: dict[str, bool] = {}
    for directory in capability_dirs:
        probe = _read_self_hashed_json(
            directory / "capability_probe.json", schema_version="ric-capability-v1"
        )
        model_key = str(probe.get("model_key"))
        if model_key not in trees or model_key in capabilities:
            raise ExperimentRunnerError("capability model grid mismatch")
        artifact, gate = validate_capability_artifact(
            directory,
            model_key=model_key,
            model_revision=str(trees[model_key]["model_revision"]),
            mode=mode,
            config_sha256=config_sha,
            protocol_sha256=protocol_sha,
            data_manifest_sha256=str(trees[model_key]["data_manifest_sha256"]),
            model_tree_manifest_sha256=str(
                trees[model_key]["model_tree_manifest_sha256"]
            ),
            config=config,
            data_producer_signoff_sha256=trees[model_key].get(
                "data_producer_signoff_sha256"
            ),
            gpu_environment_identity=trees[model_key].get(
                "gpu_environment_identity"
            ),
            historical_reviewed_source_snapshot_path=(
                historical_reviewed_source_snapshot_path
            ),
            pre_outcome_attestation=pre_outcome_attestation,
        )
        capabilities[model_key] = artifact
        g1_by_model[model_key] = gate
    if set(capabilities) != set(required_models):
        raise ExperimentRunnerError("calibration requires both capability artifacts")
    if mode == "formal":
        validate_calibration_signoff_bindings(
            signoff=signoff,
            trees=trees,
            capabilities=capabilities,
            required_models=required_models,
        )
        if (
            signoff_path is None
            or signoff_path.is_symlink()
            or not signoff_path.is_file()
        ):
            raise ExperimentRunnerError("formal calibration signoff path is invalid")
    starvation_multiplier = float(
        config["closure_and_fairness"]["starvation_multiplier"]
    )
    starvation = {
        model: starvation_multiplier
        * float(trees[model]["calibration_isolated_path_median_us"])
        for model in required_models
    }
    configs = {
        model: ReplayConfig(
            starvation_us=starvation[model],
            drr_quantum_us=_drr_quantum_from_tree(trees[model]),
            contract_tax_surface=contract_tax_surface_from_tree(trees[model]),
        )
        for model in required_models
    }
    results = execute_trace_bundles(
        worlds,
        arms=CONCRETE_JOINBLIND_ARMS,
        configs_by_model=configs,
        workers=workers,
    )
    _assert_result_grid(results)
    models_payload: dict[str, Any] = {}
    budgets: dict[tuple[str, str], float] = {}
    alias_results: list[ReplayResult] = []
    for model in required_models:
        model_cells: dict[str, Any] = {}
        for cell in sorted({world.cell for world in worlds if world.model_key == model}):
            cell_results = [
                result
                for result in results
                if result.model_key == model and result.cell == cell
            ]
            by_arm = {
                arm: tuple(row for row in cell_results if row.arm == arm)
                for arm in CONCRETE_JOINBLIND_ARMS
            }
            selected = select_calibration_baseline(
                by_arm,
                frozen_arm_order=tuple(
                    arm
                    for arm in config["joinblind_arms"]
                    if arm != "calib_best_joinblind"
                ),
            )
            model_cells[cell] = selected
            budgets[(model, cell)] = float(selected["closure_budget_us"])
            alias_results.extend(
                replace(row, arm="calib_best_joinblind")
                for row in by_arm[str(selected["calib_best_joinblind"])]
            )
        models_payload[model] = {
            "starvation_us": starvation[model],
            "drr_quantum_us": _drr_quantum_from_tree(trees[model]),
            "drr_quantum_source": "calibration_service_lut_sender_pack_us",
            "isolated_path_median_us": trees[model][
                "calibration_isolated_path_median_us"
            ],
            "cells": model_cells,
        }
    all_results = tuple(results) + tuple(alias_results)
    metrics = _metric_rows(all_results, budgets=budgets)
    g1_pass = all(g1_by_model.values())
    lock = add_self_hash(
        {
            "schema_version": "ric-calibration-lock-v1",
            "status": "CALIBRATION_LOCKED" if mode == "formal" else "NOT_TESTED",
            "scientific_result": False,
            "mode": mode,
            "role": "calibration",
            "config_sha256": config_sha,
            "protocol_sha256": protocol_sha,
            "consumer_amendment_sha256": sha256_file(consumer_amendment_path),
            "historical_reviewed_source_snapshot_sha256": (
                HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256
                if mode == "formal"
                else None
            ),
            "pre_outcome_attestation_sha256": (
                pre_outcome_attestation["attestation_sha256"]
                if pre_outcome_attestation is not None
                else None
            ),
            "pre_outcome_producer_signoff_file_sha256": (
                pre_outcome_attestation["producer_signoff_file_sha256"]
                if pre_outcome_attestation is not None
                else None
            ),
            "pre_outcome_producer_signoff_self_hash": (
                pre_outcome_attestation["producer_signoff_self_hash"]
                if pre_outcome_attestation is not None
                else None
            ),
            "authoritative_bundle_root": (
                str(authoritative_bundle_root_path.resolve(strict=True))
                if authoritative_bundle_root_path is not None
                else None
            ),
            "run_experiment_source_sha256": _source_sha256(),
            "scenario_tree_sha256": {
                model: trees[model]["manifest_sha256"] for model in required_models
            },
            "scenario_producer_signoff_sha256": {
                model: trees[model]["scenario_producer_signoff_sha256"]
                for model in required_models
            },
            "service_lut_metadata_sha256": {
                model: trees[model]["service_lut_metadata_sha256"]
                for model in required_models
            },
            "capability_probe_sha256": {
                model: capabilities[model]["manifest_sha256"] for model in required_models
            },
            "capability_producer_signoff_sha256": {
                model: capabilities[model]["signoff_sha256"]
                for model in required_models
            },
            "signoff_sha256": (
                sha256_file(signoff_path) if mode == "formal" else None
            ),
            "g1_by_model": g1_by_model,
            "g1_pass": g1_pass,
            "models": models_payload,
            "policy_semantics_sha256": object_sha256(
                {
                    "deadline": config["closure_and_fairness"]["decision_deadline_source"],
                    "deadline_multiplier": config["closure_and_fairness"][
                        "decision_deadline_multiplier"
                    ],
                    "starvation": config["closure_and_fairness"]["starvation_threshold"],
                    "fallback": config["closure_and_fairness"]["starvation_fallback"],
                }
            ),
        }
    )
    verdict = (
        "NOT_TESTED"
        if mode == "dev"
        else ("CALIBRATION_LOCKED" if g1_pass else "NO_GO_CURRENT_ACTUATOR")
    )
    status = write_run_artifacts(
        output_dir=output_dir,
        mode=mode,
        stage="calibration",
        results=all_results,
        metrics=metrics,
        bootstrap={},
        gate_matrix={"G0": True, "G1": g1_by_model},
        decision={
            "verdict": verdict,
            "formal_run_valid": mode == "formal",
            "scientific_result": mode == "formal" and not g1_pass,
        },
        metadata={
            "config_sha256": config_sha,
            "protocol_sha256": protocol_sha,
            "worker_count": resolve_worker_count(workers, len(worlds)),
            "calibration_lock_sha256": lock["manifest_sha256"],
        },
        calibration_lock=lock,
        producer_signoff_path=signoff_path,
    )
    return status


def _cell_metrics(
    metrics: Sequence[TraceMetrics], *, model: str, cell: str
) -> tuple[TraceMetrics, ...]:
    return tuple(row for row in metrics if row.model_key == model and row.cell == cell)


def _bootstrap_cell(
    metrics: Sequence[TraceMetrics],
    *,
    model: str,
    cell: str,
    candidate: str,
    config: Mapping[str, Any],
    expected_count: int,
) -> PairedBootstrapSummary:
    return paired_trace_bootstrap(
        _cell_metrics(metrics, model=model, cell=cell),
        baseline_arm="calib_best_joinblind",
        candidate_arm=candidate,
        n_bootstrap=10_000,
        confidence=float(config["statistics"]["cellwise_one_sided_confidence"]),
        seed=int(config["data"]["selection_seed"]),
        expected_trace_count=expected_count,
    )


def _double_gate(summary: PairedBootstrapSummary, config: Mapping[str, Any]) -> bool:
    gate = config["go_no_go"]["primary_double_gate"]
    return (
        summary.cvar99_relative_reduction_lcb
        >= float(gate["cvar99_relative_reduction_lcb_min"])
        and 100.0 * summary.violation_absolute_reduction_lcb
        >= float(gate["violation_absolute_reduction_lcb_percentage_points_min"])
    )


def _noninferior_to_frozen_baseline(summary: PairedBootstrapSummary) -> bool:
    """Candidate increase UCB <= 0, expressed as reduction LCB >= 0."""

    return (
        summary.cvar99_relative_reduction_lcb >= 0.0
        and summary.violation_absolute_reduction_lcb >= 0.0
    )


def validate_g3_sensitivity_grid(
    *,
    g3_details: Mapping[str, Mapping[str, Any]],
    link_gates: Mapping[str, bool],
    config: Mapping[str, Any],
) -> None:
    models = tuple(config["go_no_go"]["required_models"])
    cells = tuple(config["go_no_go"]["required_main_cells"])
    expected_cells = {f"{model}/{cell}" for model in models for cell in cells}
    if set(g3_details) != expected_cells:
        raise ExperimentRunnerError("G3 delay sensitivity model/cell grid is incomplete")
    for key, detail in g3_details.items():
        delay = detail.get("delay_sensitivity")
        if not isinstance(delay, Mapping) or set(delay) != {"0", "5", "20", "50"}:
            raise ExperimentRunnerError(f"G3 delay 0/5/20/50 grid is incomplete: {key}")
    expected_links = {
        f"{model}/{cell}/link{int(link)}"
        for model in models
        for cell in cells
        for link in config["topology_proxy"]["link_sensitivity_gbps"]
    }
    if set(link_gates) != expected_links:
        raise ExperimentRunnerError("G3 link sensitivity gate grid is incomplete")


def _strict_jsonl_rows(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentRunnerError(f"{label} must be a regular file")
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(
                    loads_json_mapping_strict(
                        line, label=f"{label} line {line_number}"
                    )
                )
            except FormalProvenanceError as exc:
                raise ExperimentRunnerError(
                    f"invalid {label} line {line_number}"
                ) from exc
    if not rows:
        raise ExperimentRunnerError(f"{label} is empty")
    return rows


def validate_oracle_instance_scheduling_contract(
    row: Mapping[str, Any], calibration_lock: Mapping[str, Any]
) -> None:
    """Bind every persisted MILP instance to the frozen scheduling contract."""

    model = row.get("model_key")
    expected_starvation = calibration_lock.get("models", {}).get(
        str(model), {}
    ).get("starvation_us")
    starvation = row.get("starvation_us")
    nodes = row.get("observation_history_nodes")
    worlds = row.get("worlds")
    world_names = (
        {
            str(world.get("world_name"))
            for world in worlds
            if isinstance(world, Mapping)
        }
        if isinstance(worlds, list)
        else set()
    )
    valid_nodes = isinstance(nodes, Mapping) and set(nodes) == {"S", "B", "R0", "C"}
    if valid_nodes:
        valid_nodes = len(world_names) == 2 and all(
            isinstance(value, Mapping)
            and set(value) == world_names
            and all(type(node_id) is int and node_id >= 0 for node_id in value.values())
            for value in nodes.values()
        )
    if valid_nodes:
        valid_nodes = (
            len(set(nodes["S"].values())) == 1
            and len(set(nodes["B"].values())) == 1
            and len(set(nodes["R0"].values())) == 2
            and len(set(nodes["C"].values())) == 2
        )
    if (
        isinstance(starvation, bool)
        or not isinstance(starvation, (int, float))
        or not isinstance(expected_starvation, (int, float))
        or isinstance(expected_starvation, bool)
        or not math.isfinite(float(starvation))
        or not math.isclose(
            float(starvation),
            float(expected_starvation),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or row.get("downstream_service_discipline")
        != "work_conserving_fcfs_no_overtake"
        or not valid_nodes
    ):
        raise ExperimentRunnerError(
            "oracle instance scheduling contract/lock mismatch"
        )


def _validate_oracle_raw_evidence(
    *,
    oracle_dir: Path,
    oracle: Mapping[str, Any],
    config: Mapping[str, Any],
    calibration_lock: Mapping[str, Any],
) -> None:
    """Recompute every oracle gate from the signed raw solution rows."""

    instance_path = oracle_dir / "milp_instances.jsonl"
    solution_path = oracle_dir / "milp_solutions.jsonl"
    if oracle.get("milp_instances_sha256") != sha256_file(instance_path):
        raise ExperimentRunnerError("oracle instance file hash mismatch")
    if oracle.get("milp_solutions_sha256") != sha256_file(solution_path):
        raise ExperimentRunnerError("oracle solution file hash mismatch")
    instances = _strict_jsonl_rows(instance_path, label="MILP instances")
    solutions = _strict_jsonl_rows(solution_path, label="MILP solutions")
    required_models = tuple(config["matched_world_milp"]["required_models"])
    expected_pairs = int(config["matched_world_milp"]["pairs_per_model"])
    instance_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in instances:
        model = row.get("model_key")
        pair_id = row.get("pair_id")
        if (
            row.get("schema_version") != "ric-milp-instance-v2"
            or model not in required_models
            or not isinstance(pair_id, str)
            or not pair_id
        ):
            raise ExperimentRunnerError("oracle instance schema/model mismatch")
        validate_oracle_instance_scheduling_contract(row, calibration_lock)
        key = (str(model), pair_id)
        if key in instance_index:
            raise ExperimentRunnerError("duplicate oracle instance pair")
        instance_index[key] = row
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in solutions:
        model = row.get("model_key")
        pair_id = row.get("pair_id")
        level = row.get("information_level")
        if (
            row.get("schema_version") != "ric-milp-solution-v2"
            or model not in required_models
            or not isinstance(pair_id, str)
            or level not in {"S", "B", "R0", "C"}
        ):
            raise ExperimentRunnerError("oracle solution schema/model mismatch")
        key = (str(model), pair_id)
        levels = grouped.setdefault(key, {})
        if str(level) in levels:
            raise ExperimentRunnerError("duplicate oracle information level")
        levels[str(level)] = row
    if set(grouped) != set(instance_index):
        raise ExperimentRunnerError("oracle instance/solution pair grid mismatch")
    model_summaries = oracle.get("model_summaries")
    if not isinstance(model_summaries, Mapping) or set(model_summaries) != set(
        required_models
    ):
        raise ExperimentRunnerError("oracle model summary grid mismatch")
    model_gate_values: list[bool] = []
    for model in required_models:
        pairs = [levels for (key_model, _), levels in grouped.items() if key_model == model]
        if len(pairs) != expected_pairs or any(
            set(levels) != {"S", "B", "R0", "C"} for levels in pairs
        ):
            raise ExperimentRunnerError("oracle pair/information grid is incomplete")
        normalized_gaps: list[float] = []
        flip_rates: list[float] = []
        solver_gaps: list[float] = []
        for levels in pairs:
            b_cvar = levels["B"].get("empirical_cvar99_us")
            r0_cvar = levels["R0"].get("empirical_cvar99_us")
            r0_flip = levels["R0"].get("first_action_flip_rate")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (b_cvar, r0_cvar, r0_flip)
            ) or float(b_cvar) <= 0.0:
                raise ExperimentRunnerError("oracle raw metric is invalid")
            if type(levels["R0"].get("unique_optimal_first_action")) is not bool or not levels[
                "R0"
            ]["unique_optimal_first_action"]:
                raise ExperimentRunnerError("oracle R0 optimum is not uniquely actionable")
            normalized_gap = (float(b_cvar) - float(r0_cvar)) / float(b_cvar)
            normalized_gaps.append(normalized_gap)
            flip_rates.append(float(r0_flip))
            for row in levels.values():
                mip_gap = row.get("mip_gap")
                reported_gap = row.get(
                    "normalized_b_to_r0_empirical_cvar99_gap"
                )
                if (
                    isinstance(mip_gap, bool)
                    or not isinstance(mip_gap, (int, float))
                    or not math.isfinite(float(mip_gap))
                    or isinstance(reported_gap, bool)
                    or not isinstance(reported_gap, (int, float))
                    or not math.isclose(
                        float(reported_gap),
                        normalized_gap,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise ExperimentRunnerError("oracle raw solution accounting mismatch")
                solver_gaps.append(float(mip_gap))
        recomputed = {
            "pair_count": float(len(pairs)),
            "median_normalized_empirical_cvar99_gap": float(
                statistics.median(normalized_gaps)
            ),
            "mean_r0_first_action_flip_rate": statistics.fmean(flip_rates),
            "max_solver_gap": max(solver_gaps),
        }
        supplied = model_summaries[model]
        if not isinstance(supplied, Mapping):
            raise ExperimentRunnerError("oracle model summary is not an object")
        for field, wanted in recomputed.items():
            value = supplied.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not math.isclose(
                    float(value), wanted, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise ExperimentRunnerError(
                    f"oracle model summary is not raw-derived: {model}/{field}"
                )
        gap_pass = recomputed[
            "median_normalized_empirical_cvar99_gap"
        ] >= float(
            config["matched_world_milp"][
                "min_exact_oracle_median_normalized_cvar_gap"
            ]
        )
        flip_pass = recomputed["mean_r0_first_action_flip_rate"] >= float(
            config["matched_world_milp"][
                "min_r0_optimal_first_action_flip_rate"
            ]
        )
        if recomputed["max_solver_gap"] > float(
            config["matched_world_milp"]["max_solver_optimality_gap"]
        ):
            raise ExperimentRunnerError("oracle solver gap exceeds frozen maximum")
        expected_gates = {
            "gap_gate_pass": gap_pass,
            "flip_gate_pass": flip_pass,
            "model_gate_pass": gap_pass and flip_pass,
        }
        for field, wanted in expected_gates.items():
            if type(supplied.get(field)) is not bool or supplied.get(field) is not wanted:
                raise ExperimentRunnerError(
                    f"oracle model gate is not raw-derived: {model}/{field}"
                )
        model_gate_values.append(expected_gates["model_gate_pass"])
    oracle_gate = oracle.get("oracle_gate_pass")
    expected_oracle_gate = all(model_gate_values)
    if type(oracle_gate) is not bool or oracle_gate is not expected_oracle_gate:
        raise ExperimentRunnerError("oracle top-level gate is not raw-derived")


def _load_lock_and_oracle(
    *,
    lock_path: Path,
    oracle_dir: Path,
    config_path: Path,
    protocol_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    lock = _read_self_hashed_json(lock_path, schema_version="ric-calibration-lock-v1")
    oracle = _read_self_hashed_json(
        oracle_dir / "status.json", schema_version="ric-oracle-status-v1"
    )
    config = _load_config(config_path)
    try:
        validate_calibration_lock_fields(
            lock,
            config=config,
            protocol_sha256=sha256_file(protocol_path),
            config_sha256=sha256_file(config_path),
            expected_run_experiment_source_sha256=_source_sha256(),
        )
        lock_signoff = lock_path.parent / EMBEDDED_PRODUCER_SIGNOFF
        if (
            not lock_signoff.is_file()
            or sha256_file(lock_signoff) != lock.get("signoff_sha256")
        ):
            raise FormalProvenanceError(
                "calibration lock embedded producer signoff mismatch"
            )
        verify_phase4_signoff(
            lock_signoff,
            repo_root=REPO_ROOT,
            expected_fields={
                "stage": "calibration",
                "config_sha256": lock.get("config_sha256"),
                "protocol_sha256": lock.get("protocol_sha256"),
                "run_experiment_source_sha256": lock.get(
                    "run_experiment_source_sha256"
                ),
                "scenario_tree_sha256": lock.get("scenario_tree_sha256"),
                "scenario_producer_signoff_sha256": lock.get(
                    "scenario_producer_signoff_sha256"
                ),
                "capability_probe_sha256": lock.get("capability_probe_sha256"),
                "capability_producer_signoff_sha256": lock.get(
                    "capability_producer_signoff_sha256"
                ),
                "consumer_amendment_sha256": lock.get(
                    "consumer_amendment_sha256"
                ),
                "historical_reviewed_source_snapshot_sha256": lock.get(
                    "historical_reviewed_source_snapshot_sha256"
                ),
                "pre_outcome_attestation_sha256": lock.get(
                    "pre_outcome_attestation_sha256"
                ),
                "pre_outcome_producer_signoff_file_sha256": lock.get(
                    "pre_outcome_producer_signoff_file_sha256"
                ),
                "pre_outcome_producer_signoff_self_hash": lock.get(
                    "pre_outcome_producer_signoff_self_hash"
                ),
                "authoritative_bundle_root": lock.get(
                    "authoritative_bundle_root"
                ),
            },
            required_source_paths=RUN_EXPERIMENT_SOURCE_PATHS,
            required_reviewed_scope_paths=(
                *canonical_reviewed_scope_paths(
                    REPO_ROOT, RUN_EXPERIMENT_SOURCE_PATHS
                ),
                DEFAULT_CONSUMER_AMENDMENT,
            ),
        )
    except FormalProvenanceError as exc:
        raise ExperimentRunnerError(str(exc)) from exc
    for artifact in (lock, oracle):
        if artifact.get("config_sha256") != sha256_file(config_path):
            raise ExperimentRunnerError("calibration evidence config mismatch")
        if artifact.get("protocol_sha256") != sha256_file(protocol_path):
            raise ExperimentRunnerError("calibration evidence protocol mismatch")
    if (
        oracle.get("status") != "CALIBRATION_ORACLE_COMPLETE"
        or oracle.get("role") != "calibration"
        or oracle.get("mode") != "formal"
    ):
        raise ExperimentRunnerError("oracle status is not formal calibration evidence")
    if oracle.get("calibration_lock_sha256") != lock.get("manifest_sha256"):
        raise ExperimentRunnerError("oracle is not bound to the calibration lock")
    if oracle.get("calibration_lock_file_sha256") != sha256_file(lock_path):
        raise ExperimentRunnerError("oracle calibration-lock file binding mismatch")
    if oracle.get("scenario_tree_sha256") != lock.get("scenario_tree_sha256"):
        raise ExperimentRunnerError("oracle/calibration scenario hash grid mismatch")
    for field in (
        "consumer_amendment_sha256",
        "historical_reviewed_source_snapshot_sha256",
        "pre_outcome_attestation_sha256",
        "pre_outcome_producer_signoff_file_sha256",
        "pre_outcome_producer_signoff_self_hash",
        "authoritative_bundle_root",
    ):
        if oracle.get(field) != lock.get(field) or oracle.get(field) in {None, ""}:
            raise ExperimentRunnerError(
                f"oracle/calibration migration binding mismatch: {field}"
            )
    if oracle.get("run_oracle_source_sha256") != _run_oracle_source_sha256():
        raise ExperimentRunnerError("oracle producer source mismatch")
    if oracle.get("build_scenarios_source_sha256") != _build_scenarios_source_sha256():
        raise ExperimentRunnerError("oracle scenario-consumer source mismatch")
    oracle_signoff = oracle_dir / EMBEDDED_PRODUCER_SIGNOFF
    if (
        not is_sha256(oracle.get("signoff_sha256"))
        or not oracle_signoff.is_file()
        or sha256_file(oracle_signoff) != oracle.get("signoff_sha256")
    ):
        raise ExperimentRunnerError("oracle embedded producer signoff mismatch")
    scenario_tree_hashes = oracle.get("scenario_tree_sha256")
    scenario_tree_file_hashes = oracle.get("scenario_tree_file_sha256")
    scenario_signoffs = oracle.get("scenario_producer_signoff_sha256")
    if not all(
        isinstance(value, Mapping)
        for value in (scenario_tree_hashes, scenario_tree_file_hashes, scenario_signoffs)
    ):
        raise ExperimentRunnerError("oracle scenario provenance grid is missing")
    try:
        _verify_oracle_producer_signoff(
            oracle_signoff,
            config_path=config_path,
            protocol_path=protocol_path,
            calibration_lock_path=lock_path,
            calibration_lock=lock,
            scenario_tree_hashes=scenario_tree_hashes,
            scenario_tree_file_hashes=scenario_tree_file_hashes,
            scenario_producer_signoff_sha256=scenario_signoffs,
        )
    except Exception as exc:
        raise ExperimentRunnerError("oracle producer signoff is invalid") from exc
    _validate_oracle_raw_evidence(
        oracle_dir=oracle_dir,
        oracle=oracle,
        config=config,
        calibration_lock=lock,
    )
    required_models = set(config["go_no_go"]["required_models"])
    if set(lock.get("models", {})) != required_models:
        raise ExperimentRunnerError("calibration lock model grid is incomplete")
    if set(oracle.get("model_summaries", {})) != required_models:
        raise ExperimentRunnerError("oracle model grid is incomplete")
    if lock.get("g1_pass") is not True:
        raise ExperimentRunnerError("G1 did not pass; sealed G2 is forbidden")
    return lock, oracle


def strict_retention_bootstrap(
    rows: Sequence[TraceMetrics],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> tuple[RetentionBootstrapSummary | None, str | None]:
    """Reject the whole retention gate if any paired denominator is nonpositive."""

    arms = ("calib_best_joinblind", "ric_full_zero_delay", "ric_wire_charged")
    index = {(row.trace_id, row.arm): row for row in rows if row.arm in arms}
    trace_ids = sorted({row.trace_id for row in rows})
    if any((trace_id, arm) not in index for trace_id in trace_ids for arm in arms):
        raise ExperimentRunnerError("retention trace/arm grid is incomplete")

    def denominator(selected: Sequence[str]) -> float:
        baseline = statistics.mean(
            index[(trace_id, arms[0])].cvar99_us for trace_id in selected
        )
        r0 = statistics.mean(
            index[(trace_id, arms[1])].cvar99_us for trace_id in selected
        )
        return baseline - r0

    if denominator(trace_ids) <= 1e-12:
        return None, "NONPOSITIVE_POINT_INFORMATION_HEADROOM"
    rng = random.Random(seed)
    for _ in range(n_bootstrap):
        selected = tuple(rng.choice(trace_ids) for _ in trace_ids)
        if denominator(selected) <= 1e-12:
            return None, "NONPOSITIVE_BOOTSTRAP_INFORMATION_HEADROOM"
    return (
        paired_retention_bootstrap(
            rows,
            baseline_arm=arms[0],
            r0_arm=arms[1],
            charged_arm=arms[2],
            metric="cvar99",
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        ),
        None,
    )


def reserve_sealed_consumption(
    path: Path,
    *,
    mode: str,
    output_dir: Path,
    nonce: str,
    config_sha256: str,
    protocol_sha256: str,
    scenario_tree_sha256: Mapping[str, str],
    scenario_producer_signoff_sha256: Mapping[str, str],
    oracle_status_sha256: str,
    oracle_producer_signoff_sha256: str,
) -> Mapping[str, Any]:
    if mode == "formal" and Path(os.path.abspath(path)) != Path(
        os.path.abspath(GLOBAL_SEALED_EVALUATION_CONSUMPTION)
    ):
        raise ExperimentRunnerError(
            "formal sealed ledger path differs from the reviewed global ledger"
        )
    if mode == "formal":
        expected_parent = GLOBAL_SEALED_EVALUATION_CONSUMPTION.parent
        if (
            expected_parent.is_symlink()
            or expected_parent.resolve(strict=True) != expected_parent
        ):
            raise ExperimentRunnerError(
                "formal sealed ledger parent is not the reviewed real directory"
            )
    if not nonce:
        raise ExperimentRunnerError("formal sealed run requires a one-shot nonce")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = add_self_hash(
        {
            "schema_version": "ric-sealed-evaluation-consumption-v1",
            "role": "sealed",
            "nonce": nonce,
            "config_sha256": config_sha256,
            "protocol_sha256": protocol_sha256,
            "sealed_output_dir": str(output_dir.resolve(strict=False)),
            "scenario_tree_sha256": dict(scenario_tree_sha256),
            "scenario_producer_signoff_sha256": dict(
                scenario_producer_signoff_sha256
            ),
            "oracle_status_sha256": oracle_status_sha256,
            "oracle_producer_signoff_sha256": oracle_producer_signoff_sha256,
            "run_experiment_source_sha256": _sealed_source_sha256(),
        }
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError as exc:
        raise ExperimentRunnerError("sealed one-shot was already consumed") from exc
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise ExperimentRunnerError(
                    "sealed one-shot ledger write made no progress"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o400, follow_symlinks=False)
    if path.read_bytes() != encoded:
        raise ExperimentRunnerError("sealed one-shot ledger byte verification failed")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_descriptor = os.open(path.parent, directory_flags)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return payload


def _scenario_grid(
    directories: Sequence[Path], *, mode: str, config: Mapping[str, Any]
) -> tuple[Mapping[tuple[str, int], Mapping[str, Any]], tuple[ReplayWorld, ...]]:
    trees: dict[tuple[str, int], Mapping[str, Any]] = {}
    worlds: list[ReplayWorld] = []
    for directory in directories:
        tree, loaded = load_worlds(directory, expected_role="sealed", mode=mode)
        key = (str(tree["model_key"]), int(tree["link_gbps"]))
        if key in trees:
            raise ExperimentRunnerError("duplicate sealed model/link scenario")
        trees[key] = tree
        worlds.extend(loaded)
    required = {
        (model, link)
        for model in config["go_no_go"]["required_models"]
        for link in (
            int(config["topology_proxy"]["primary_link_gbps"]),
            *[int(value) for value in config["topology_proxy"]["link_sensitivity_gbps"]],
        )
    }
    if mode == "formal" and set(trees) != required:
        raise ExperimentRunnerError("formal sealed grid lacks a model/link sensitivity")
    if mode == "formal":
        globally_identical = (
            "data_manifest_sha256",
            "data_producer_signoff_sha256",
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
            "consumer_amendment_sha256",
            "historical_reviewed_source_snapshot_sha256",
            "pre_outcome_attestation_sha256",
            "pre_outcome_producer_signoff_file_sha256",
            "pre_outcome_producer_signoff_self_hash",
            "authoritative_bundle_root",
        )
        for field in globally_identical:
            values = {tree.get(field) for tree in trees.values()}
            if len(values) != 1 or next(iter(values)) in {None, ""}:
                raise ExperimentRunnerError(
                    f"sealed grid migration binding mismatch: {field}"
                )
    primary_link = int(config["topology_proxy"]["primary_link_gbps"])
    causal_contract = config["topology_proxy"]["link_sensitivity_causal_world"]
    seed_range = config["workloads"]["role_seed_ranges"]["sealed"]

    def causal_index(tree: Mapping[str, Any]) -> dict[tuple[Any, ...], tuple[str, str]]:
        rows = tree.get("worlds")
        if not isinstance(rows, list):
            raise ExperimentRunnerError("sealed scenario lacks world metadata")
        index: dict[tuple[Any, ...], tuple[str, str]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ExperimentRunnerError("sealed world metadata is malformed")
            key = (
                str(row.get("cell")),
                int(row.get("trace_index", -1)),
                int(row.get("workload_seed", -1)),
                tuple(row.get("request_ids", ())),
            )
            value = (
                str(row.get("causal_arrival_fingerprint", "")),
                str(row.get("arrival_schedule_fingerprint", "")),
            )
            if key in index or any(len(item) != 64 for item in value):
                raise ExperimentRunnerError("sealed causal-world identity is invalid")
            index[key] = value
        return index

    for model in config["go_no_go"]["required_models"]:
        primary_tree = trees.get((model, primary_link))
        if primary_tree is None:
            continue
        reference = causal_index(primary_tree)
        for link in (
            primary_link,
            *[int(value) for value in config["topology_proxy"]["link_sensitivity_gbps"]],
        ):
            tree = trees[(model, link)]
            if (
                tree.get("link_sensitivity_causal_world") != causal_contract
                or tree.get("role_seed_range") != seed_range
                or tree.get("target_utilization_calibration_enforced")
                is not (link == primary_link)
                or causal_index(tree) != reference
            ):
                raise ExperimentRunnerError(
                    "link sensitivity does not reuse the exact primary causal world"
                )
    return trees, tuple(worlds)


def evaluate_pipeline(
    *,
    scenario_dirs: Sequence[Path],
    calibration_lock_path: Path,
    oracle_dir: Path,
    output_dir: Path,
    mode: str,
    workers: int,
    config_path: Path,
    protocol_path: Path,
    signoff_path: Path,
) -> Mapping[str, Any]:
    try:
        validate_formal_output_path(output_dir, mode=mode)
    except ScenarioBuildError as exc:
        raise ExperimentRunnerError(str(exc)) from exc
    if output_dir.exists():
        raise ExperimentRunnerError(
            "refusing to consume sealed data for an existing output directory"
        )
    guard_mode_role(mode, "sealed")
    if mode != "formal":
        raise ExperimentRunnerError("sealed evaluation is formal-only")
    try:
        validate_frozen_formal_paths(
            config_path=config_path, protocol_path=protocol_path, mode=mode
        )
    except ScenarioBuildError as exc:
        raise ExperimentRunnerError(str(exc)) from exc
    verified_sealed_signoff = _load_signoff(
        signoff_path,
        stage="sealed",
        config=config_path,
        protocol=protocol_path,
    )
    ledger_path = GLOBAL_SEALED_EVALUATION_CONSUMPTION
    config = _load_config(config_path)
    lock, oracle = _load_lock_and_oracle(
        lock_path=calibration_lock_path,
        oracle_dir=oracle_dir,
        config_path=config_path,
        protocol_path=protocol_path,
    )
    preopen_signoff_expected = {
        "calibration_lock_sha256": lock["manifest_sha256"],
        "oracle_status_sha256": oracle["manifest_sha256"],
        "run_oracle_source_sha256": oracle["run_oracle_source_sha256"],
        "oracle_producer_signoff_sha256": oracle["signoff_sha256"],
        "sealed_consumption_ledger_path": str(ledger_path.resolve(strict=False)),
        "sealed_output_dir": str(output_dir.resolve(strict=False)),
    }
    for field, wanted in preopen_signoff_expected.items():
        if verified_sealed_signoff.get(field) != wanted:
            raise ExperimentRunnerError(
                f"sealed Phase-4 signoff binding mismatch: {field}"
            )
    try:
        signed_tree_hashes = {
            str(key): str(value)
            for key, value in verified_sealed_signoff[
                "scenario_tree_sha256"
            ].items()
        }
        signed_tree_signoffs = {
            str(key): str(value)
            for key, value in verified_sealed_signoff[
                "scenario_producer_signoff_sha256"
            ].items()
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise ExperimentRunnerError(
            "sealed Phase-4 signoff scenario grid is malformed"
        ) from exc
    sealed_consumption_record = reserve_sealed_consumption(
        ledger_path,
        mode="formal",
        output_dir=output_dir,
        nonce=str(verified_sealed_signoff.get("sealed_nonce", "")),
        config_sha256=sha256_file(config_path),
        protocol_sha256=sha256_file(protocol_path),
        scenario_tree_sha256=signed_tree_hashes,
        scenario_producer_signoff_sha256=signed_tree_signoffs,
        oracle_status_sha256=str(oracle["manifest_sha256"]),
        oracle_producer_signoff_sha256=str(oracle["signoff_sha256"]),
    )
    trees, all_worlds = _scenario_grid(scenario_dirs, mode=mode, config=config)
    config_sha = sha256_file(config_path)
    protocol_sha = sha256_file(protocol_path)
    if any(
        tree.get("config_sha256") != config_sha
        or tree.get("protocol_sha256") != protocol_sha
        for tree in trees.values()
    ):
        raise ExperimentRunnerError("sealed scenario protocol/config mismatch")
    for field in (
        "consumer_amendment_sha256",
        "historical_reviewed_source_snapshot_sha256",
        "pre_outcome_attestation_sha256",
        "pre_outcome_producer_signoff_file_sha256",
        "pre_outcome_producer_signoff_self_hash",
        "authoritative_bundle_root",
    ):
        tree_value = next(iter(trees.values())).get(field)
        if lock.get(field) != tree_value:
            raise ExperimentRunnerError(
                f"calibration/sealed migration binding mismatch: {field}"
            )
    for (model, _link), tree in trees.items():
        if tree.get("service_lut_metadata_sha256") != lock.get(
            "service_lut_metadata_sha256", {}
        ).get(model):
            raise ExperimentRunnerError("sealed scenario changed calibration service LUT")
    expected_policy_semantics = object_sha256(
        {
            "deadline": config["closure_and_fairness"]["decision_deadline_source"],
            "deadline_multiplier": config["closure_and_fairness"][
                "decision_deadline_multiplier"
            ],
            "starvation": config["closure_and_fairness"]["starvation_threshold"],
            "fallback": config["closure_and_fairness"]["starvation_fallback"],
        }
    )
    if lock.get("policy_semantics_sha256") != expected_policy_semantics:
        raise ExperimentRunnerError("calibration/sealed policy semantics mismatch")
    actual_tree_hashes = {
        f"{model}/link{link}": str(tree["manifest_sha256"])
        for (model, link), tree in trees.items()
    }
    if sealed_consumption_record.get("scenario_tree_sha256") != actual_tree_hashes:
        raise ExperimentRunnerError("sealed one-shot ledger/scenario hash mismatch")
    actual_scenario_signoffs = {
        f"{model}/link{link}": str(tree["scenario_producer_signoff_sha256"])
        for (model, link), tree in trees.items()
    }
    if (
        sealed_consumption_record.get("scenario_producer_signoff_sha256")
        != actual_scenario_signoffs
    ):
        raise ExperimentRunnerError(
            "sealed one-shot ledger/scenario producer-signoff mismatch"
        )
    if (
        sealed_consumption_record.get("oracle_status_sha256")
        != oracle.get("manifest_sha256")
        or sealed_consumption_record.get("oracle_producer_signoff_sha256")
        != oracle.get("signoff_sha256")
        or sealed_consumption_record.get("run_experiment_source_sha256")
        != _sealed_source_sha256()
    ):
        raise ExperimentRunnerError("sealed one-shot ledger/oracle binding mismatch")
    sealed_signoff_expected = {
        "calibration_lock_sha256": lock["manifest_sha256"],
        "oracle_status_sha256": oracle["manifest_sha256"],
        "run_oracle_source_sha256": oracle["run_oracle_source_sha256"],
        "oracle_producer_signoff_sha256": oracle["signoff_sha256"],
        "sealed_consumption_ledger_path": str(ledger_path.resolve(strict=True)),
        "sealed_output_dir": str(output_dir.resolve(strict=False)),
        "sealed_nonce": sealed_consumption_record["nonce"],
        "scenario_tree_sha256": actual_tree_hashes,
        "scenario_producer_signoff_sha256": actual_scenario_signoffs,
    }
    for field, wanted in sealed_signoff_expected.items():
        if verified_sealed_signoff.get(field) != wanted:
            raise ExperimentRunnerError(
                f"sealed Phase-4 signoff binding mismatch: {field}"
            )
    required_models = tuple(config["go_no_go"]["required_models"])
    main_cells = tuple(config["go_no_go"]["required_main_cells"])
    primary_link = int(config["topology_proxy"]["primary_link_gbps"])
    primary_worlds = tuple(
        world
        for world in all_worlds
        if f"/link{primary_link}/" in world.trace_id
    )
    sealed_run_metadata = {
        "config_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "calibration_lock_sha256": lock["manifest_sha256"],
        "oracle_status_sha256": oracle["manifest_sha256"],
        "sealed_evaluation_ledger_path": str(ledger_path.resolve(strict=True)),
        "sealed_evaluation_ledger_self_hash": (
            sealed_consumption_record["manifest_sha256"]
        ),
        "sealed_evaluation_ledger_file_sha256": sha256_file(ledger_path),
        "worker_count": resolve_worker_count(workers, len(primary_worlds)),
    }
    budgets = {
        (model, cell): float(lock["models"][model]["cells"][cell]["closure_budget_us"])
        for model in required_models
        for cell in {
            *main_cells,
            *config["workloads"]["negative_control"].keys(),
        }
    }
    configs = {
        model: ReplayConfig(
            compressed_delay_us=float(config["contract"]["primary_delay_us"]),
            wire_delay_us=float(config["contract"]["primary_delay_us"]),
            starvation_us=float(lock["models"][model]["starvation_us"]),
            drr_quantum_us=float(lock["models"][model]["drr_quantum_us"]),
            calib_best_joinblind=str(
                lock["models"][model]["cells"][main_cells[0]][
                    "calib_best_joinblind"
                ]
            ),
            contract_tax_surface=contract_tax_surface_from_tree(
                trees[(model, primary_link)]
            ),
        )
        for model in required_models
    }
    for model in required_models:
        locked_quantum = float(lock["models"][model]["drr_quantum_us"])
        for (tree_model, _link), tree in trees.items():
            if tree_model == model and not math.isclose(
                _drr_quantum_from_tree(tree),
                locked_quantum,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ExperimentRunnerError(
                    "sealed/link scenario changed the calibration-frozen DRR quantum"
                )
    # A model may select a different simple baseline by workload cell.  Run
    # each cell with its own frozen alias target while preserving all arms.
    g2_results: list[ReplayResult] = []
    for model in required_models:
        for cell in sorted({world.cell for world in primary_worlds if world.model_key == model}):
            cell_config = replace(
                configs[model],
                calib_best_joinblind=str(
                    lock["models"][model]["cells"][cell]["calib_best_joinblind"]
                ),
            )
            selected_worlds = [
                world
                for world in primary_worlds
                if world.model_key == model and world.cell == cell
            ]
            g2_results.extend(
                execute_trace_bundles(
                    selected_worlds,
                    arms=tuple(sorted(JOINBLIND_ARMS)) + ("ric_full_zero_delay",),
                    configs_by_model={model: cell_config},
                    workers=workers,
                )
            )
    _assert_result_grid(g2_results)
    metrics: list[TraceMetrics] = list(_metric_rows(g2_results, budgets=budgets))
    expected_count = int(config["workloads"]["complete_trace_clusters_per_model_cell"])
    g2_bootstrap: dict[str, Any] = {}
    g2_cells: dict[str, bool] = {}
    reverse_checks: dict[str, bool] = {}
    for model in required_models:
        for cell in main_cells:
            key = f"{model}/{cell}"
            summary = _bootstrap_cell(
                metrics,
                model=model,
                cell=cell,
                candidate="ric_full_zero_delay",
                config=config,
                expected_count=expected_count,
            )
            g2_bootstrap[key] = summary
            g2_cells[key] = _double_gate(summary, config)
            cell_rows = _cell_metrics(metrics, model=model, cell=cell)
            reverse = []
            for arm in CONCRETE_JOINBLIND_ARMS:
                comparison = paired_trace_bootstrap(
                    cell_rows,
                    baseline_arm=arm,
                    candidate_arm="ric_full_zero_delay",
                    n_bootstrap=10_000,
                    confidence=float(config["statistics"]["cellwise_one_sided_confidence"]),
                    seed=int(config["data"]["selection_seed"]),
                    expected_trace_count=expected_count,
                )
                reverse.append(not _noninferior_to_frozen_baseline(comparison))
            reverse_checks[key] = not any(reverse)
    oracle_pass = oracle.get("oracle_gate_pass") is True
    g0_evidence = {
        "formal_mode": mode == "formal",
        "complete_sealed_model_link_grid": len(trees)
        == len(required_models)
        * (1 + len(config["topology_proxy"]["link_sensitivity_gbps"])),
        "all_primary_replays_full_drain": all(row.full_drain for row in g2_results),
        "all_primary_replays_3n_plus_j": all(
            row.completed_stage_count == row.expected_stage_count
            == 3 * row.task_count + row.expected_join_count
            and row.completed_join_count == row.expected_join_count
            for row in g2_results
        ),
        "all_inputs_protocol_config_bound": all(
            tree.get("config_sha256") == config_sha
            and tree.get("protocol_sha256") == protocol_sha
            for tree in trees.values()
        ),
    }
    g0_pass = all(g0_evidence.values())
    if not g0_pass:
        raise ExperimentRunnerError("G0 formal artifact/accounting evidence is incomplete")
    g2_pass = oracle_pass and all(g2_cells.values()) and all(reverse_checks.values())
    bootstrap_payload: dict[str, Any] = {"G2": g2_bootstrap}
    gate_matrix: dict[str, Any] = {
        "G0": g0_pass,
        "G0_evidence": g0_evidence,
        "G1": lock["g1_by_model"],
        "G2_oracle": oracle_pass,
        "G2_cells": g2_cells,
        "G2_no_reverse_vs_frozen_baselines": reverse_checks,
        "G2": g2_pass,
    }
    all_results: list[ReplayResult] = list(g2_results)
    if not g2_pass:
        gate_matrix["action_collapse_matrix"] = {
            group[0].trace_id: action_collapse_matrix(group)
            for group in _result_groups(all_results).values()
        }
        verdict = str(config["g2_scientific_fail_verdict"])
        return write_run_artifacts(
            output_dir=output_dir,
            mode=mode,
            stage="sealed",
            results=all_results,
            metrics=metrics,
            bootstrap=bootstrap_payload,
            gate_matrix=gate_matrix,
            decision={
                "verdict": verdict,
                "formal_run_valid": True,
                "scientific_result": True,
                "charged_g3_executed": False,
            },
            metadata=sealed_run_metadata,
            sealed_consumption_record=sealed_consumption_record,
            producer_signoff_path=signoff_path,
        )
    g3_contract = config["go_no_go"]["g3"]
    if (
        g3_contract.get("retention_metric") != "cvar99"
        or g3_contract.get("control_bytes_over_payload_cell_aggregation")
        != "maximum_complete_trace_ratio"
    ):
        raise ExperimentRunnerError("G3 retention/byte aggregation contract drift")
    charged_results: list[ReplayResult] = []
    for model in required_models:
        for cell in sorted({world.cell for world in primary_worlds if world.model_key == model}):
            cell_config = replace(
                configs[model],
                calib_best_joinblind=str(
                    lock["models"][model]["cells"][cell]["calib_best_joinblind"]
                ),
            )
            selected_worlds = [
                world
                for world in primary_worlds
                if world.model_key == model and world.cell == cell
            ]
            charged_results.extend(
                execute_trace_bundles(
                    selected_worlds,
                    arms=(
                        "ric_compressed_zero_delay",
                        "ric_compressed_delayed",
                        "ric_wire_charged",
                    ),
                    configs_by_model={model: cell_config},
                    workers=workers,
                    include_sham=True,
                )
            )
    all_results.extend(charged_results)
    _assert_result_grid(all_results)
    for group in _result_groups(charged_results).values():
        charged = next(row for row in group if row.arm == "ric_wire_charged")
        sham = next(row for row in group if row.arm == "ric_sham_feedback")
        assert_sham_feedback_cost_equivalence(charged, sham)
    metrics = list(_metric_rows(all_results, budgets=budgets))
    delay_results: list[ReplayResult] = []
    for delay_us in (0.0, 20.0, 50.0):
        delay_label = int(delay_us)
        for model in required_models:
            for cell in main_cells:
                delay_config = replace(
                    configs[model],
                    wire_delay_us=delay_us,
                    calib_best_joinblind=str(
                        lock["models"][model]["cells"][cell]["calib_best_joinblind"]
                    ),
                )
                selected_worlds = [
                    world
                    for world in primary_worlds
                    if world.model_key == model and world.cell == cell
                ]
                delay_results.extend(
                    replace(row, arm=f"ric_wire_charged_delay{delay_label}")
                    for row in execute_trace_bundles(
                        selected_worlds,
                        arms=("ric_wire_charged",),
                        configs_by_model={model: delay_config},
                        workers=workers,
                    )
                )
    all_results.extend(delay_results)
    metrics.extend(_metric_rows(delay_results, budgets=budgets))
    g3_cells: dict[str, bool] = {}
    g3_details: dict[str, Any] = {}
    for model in required_models:
        for cell in main_cells:
            key = f"{model}/{cell}"
            charged = _bootstrap_cell(
                metrics,
                model=model,
                cell=cell,
                candidate="ric_wire_charged",
                config=config,
                expected_count=expected_count,
            )
            stale20 = _bootstrap_cell(
                metrics,
                model=model,
                cell=cell,
                candidate="ric_wire_charged_delay20",
                config=config,
                expected_count=expected_count,
            )
            delay_reports = {
                "0": _bootstrap_cell(
                    metrics,
                    model=model,
                    cell=cell,
                    candidate="ric_wire_charged_delay0",
                    config=config,
                    expected_count=expected_count,
                ),
                "5": charged,
                "20": stale20,
                "50": _bootstrap_cell(
                    metrics,
                    model=model,
                    cell=cell,
                    candidate="ric_wire_charged_delay50",
                    config=config,
                    expected_count=expected_count,
                ),
            }
            cell_rows = _cell_metrics(metrics, model=model, cell=cell)
            retention, retention_failure = strict_retention_bootstrap(
                cell_rows,
                n_bootstrap=10_000,
                confidence=float(config["statistics"]["cellwise_one_sided_confidence"]),
                seed=int(config["data"]["selection_seed"]) + 1,
            )
            charged_metric_rows = [row for row in cell_rows if row.arm == "ric_wire_charged"]
            max_control_ratio = max(row.control_bytes_over_payload for row in charged_metric_rows)
            charged_reverse = []
            for arm in CONCRETE_JOINBLIND_ARMS:
                comparison = paired_trace_bootstrap(
                    cell_rows,
                    baseline_arm=arm,
                    candidate_arm="ric_wire_charged",
                    n_bootstrap=10_000,
                    confidence=float(config["statistics"]["cellwise_one_sided_confidence"]),
                    seed=int(config["data"]["selection_seed"]),
                    expected_trace_count=expected_count,
                )
                charged_reverse.append(
                    not _noninferior_to_frozen_baseline(comparison)
                )
            no_charged_reverse = not any(charged_reverse)
            zero_faults = all(
                not any(int(value) for value in result.fault_counts.values())
                for result in charged_results
                if result.model_key == model and result.cell == cell
            )
            passed = (
                _double_gate(charged, config)
                and retention is not None
                and retention.lcb >= float(config["go_no_go"]["g3"]["charged_retention_lcb_min"])
                and max_control_ratio <= float(config["go_no_go"]["g3"]["control_bytes_over_payload_max"])
                and stale20.cvar99_relative_reduction_lcb >= 0.0
                and stale20.violation_absolute_reduction_lcb >= 0.0
                and no_charged_reverse
                and zero_faults
            )
            g3_cells[key] = passed
            g3_details[key] = {
                "charged": charged,
                "delay_sensitivity": delay_reports,
                "retention": retention,
                "retention_failure": retention_failure,
                "max_control_bytes_over_payload": max_control_ratio,
                "no_reverse_vs_all_frozen_baselines": no_charged_reverse,
                "zero_wire_faults": zero_faults,
            }
    bootstrap_payload["G3"] = g3_details
    # Link sensitivities are always secondary and cannot replace primary gates.
    sensitivity: dict[str, Any] = {}
    link_gates: dict[str, bool] = {}
    for link in config["topology_proxy"]["link_sensitivity_gbps"]:
        link = int(link)
        for model in required_models:
            tree = trees[(model, link)]
            surface = contract_tax_surface_from_tree(tree)
            for cell in main_cells:
                worlds = [
                    world
                    for world in all_worlds
                    if world.model_key == model
                    and world.cell == cell
                    and f"/link{link}/" in world.trace_id
                ]
                sensitivity_config = ReplayConfig(
                    compressed_delay_us=float(config["contract"]["primary_delay_us"]),
                    wire_delay_us=float(config["contract"]["primary_delay_us"]),
                    starvation_us=float(lock["models"][model]["starvation_us"]),
                    drr_quantum_us=float(
                        lock["models"][model]["drr_quantum_us"]
                    ),
                    calib_best_joinblind=str(
                        lock["models"][model]["cells"][cell]["calib_best_joinblind"]
                    ),
                    contract_tax_surface=surface,
                )
                rows = execute_trace_bundles(
                    worlds,
                    arms=(
                        "calib_best_joinblind",
                        "ric_full_zero_delay",
                        "ric_wire_charged",
                    ),
                    configs_by_model={model: sensitivity_config},
                    workers=workers,
                )
                all_results.extend(rows)
                raw_link_metrics = list(
                    _metric_rows(rows, budgets={(model, cell): budgets[(model, cell)]})
                )
                link_bootstrap = paired_trace_bootstrap(
                    raw_link_metrics,
                    baseline_arm="calib_best_joinblind",
                    candidate_arm="ric_wire_charged",
                    n_bootstrap=10_000,
                    confidence=float(config["statistics"]["cellwise_one_sided_confidence"]),
                    seed=int(config["data"]["selection_seed"]),
                    expected_trace_count=expected_count,
                )
                link_key = f"{model}/{cell}/link{link}"
                link_gates[link_key] = (
                    link_bootstrap.cvar99_relative_reduction_lcb >= 0.0
                    and link_bootstrap.violation_absolute_reduction_lcb >= 0.0
                )
                tagged_metrics = [
                    replace(metric, cell=f"{cell}@link{link}")
                    for metric in raw_link_metrics
                ]
                metrics.extend(tagged_metrics)
                sensitivity[link_key] = {
                    "trace_count": len(worlds),
                    "paired_bootstrap": link_bootstrap,
                    "nonnegative_lcb_gate_pass": link_gates[link_key],
                    "contract_tax_surface_source_id": surface.source_id,
                    "contract_tax_surface_fingerprint": surface.fingerprint,
                }
    gate_matrix["G3_cells"] = g3_cells
    validate_g3_sensitivity_grid(
        g3_details=g3_details, link_gates=link_gates, config=config
    )
    gate_matrix["G3_link_sensitivity_nonnegative"] = link_gates
    gate_matrix["G3_full_drain"] = all(result.full_drain for result in all_results)
    gate_matrix["G3_canonical_exactness"] = bool(lock["g1_pass"])
    gate_matrix["G3"] = (
        all(g3_cells.values())
        and all(link_gates.values())
        and gate_matrix["G3_full_drain"]
        and gate_matrix["G3_canonical_exactness"]
    )
    gate_matrix["sensitivities_present"] = sorted(sensitivity)
    _assert_result_grid(all_results)
    gate_matrix["action_collapse_matrix"] = {
        group[0].trace_id: action_collapse_matrix(group)
        for group in _result_groups(all_results).values()
    }
    verdict = (
        config["go_no_go"]["pass_verdict"]
        if gate_matrix["G3"]
        else config["go_no_go"]["g3_scientific_fail_verdict"]
    )
    return write_run_artifacts(
        output_dir=output_dir,
        mode=mode,
        stage="sealed",
        results=all_results,
        metrics=metrics,
        bootstrap=bootstrap_payload,
        gate_matrix=gate_matrix,
        decision={
            "verdict": verdict,
            "formal_run_valid": True,
            "scientific_result": True,
            "charged_g3_executed": True,
            "primary_only_for_verdict": True,
            "sensitivities": sensitivity,
        },
        metadata=sealed_run_metadata,
        sealed_consumption_record=sealed_consumption_record,
        producer_signoff_path=signoff_path,
    )


def _load_signoff(
    path: Path | None,
    *,
    stage: str,
    config: Path,
    protocol: Path,
    migration_expected_fields: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    try:
        validate_frozen_formal_paths(
            config_path=config, protocol_path=protocol, mode="formal"
        )
    except ScenarioBuildError as exc:
        raise ExperimentRunnerError(str(exc)) from exc
    source_paths = (
        SEALED_RUN_EXPERIMENT_SOURCE_PATHS
        if stage == "sealed"
        else RUN_EXPERIMENT_SOURCE_PATHS
    )
    source_sha256 = (
        _sealed_source_sha256() if stage == "sealed" else _source_sha256()
    )
    expected = {
        "stage": stage,
        "config_sha256": sha256_file(config),
        "protocol_sha256": sha256_file(protocol),
        "run_experiment_source_sha256": source_sha256,
    }
    if migration_expected_fields is not None:
        expected.update(migration_expected_fields)
    try:
        value = verify_phase4_signoff(
            path,
            repo_root=REPO_ROOT,
            expected_fields=expected,
            required_source_paths=source_paths,
            required_reviewed_scope_paths=(
                *canonical_reviewed_scope_paths(REPO_ROOT, source_paths),
                DEFAULT_CONSUMER_AMENDMENT,
            ),
        )
    except FormalProvenanceError as exc:
        raise ExperimentRunnerError(str(exc)) from exc
    if not isinstance(value.get("scenario_tree_sha256"), Mapping):
        raise ExperimentRunnerError(f"{stage} signoff lacks reviewed scenario tree hashes")
    if not isinstance(value.get("scenario_producer_signoff_sha256"), Mapping):
        raise ExperimentRunnerError(
            f"{stage} signoff lacks scenario producer-signoff hashes"
        )
    if stage == "calibration" and not isinstance(
        value.get("capability_probe_sha256"), Mapping
    ):
        raise ExperimentRunnerError(
            "calibration signoff lacks reviewed capability-probe hashes"
        )
    if stage == "sealed":
        if not is_sha256(value.get("run_oracle_source_sha256")):
            raise ExperimentRunnerError("sealed signoff lacks oracle source binding")
        if not is_sha256(value.get("oracle_producer_signoff_sha256")):
            raise ExperimentRunnerError(
                "sealed signoff lacks oracle producer-signoff binding"
            )
    if stage == "calibration" and not isinstance(
        value.get("capability_producer_signoff_sha256"), Mapping
    ):
        raise ExperimentRunnerError(
            "calibration signoff lacks capability producer-signoff hashes"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("calibrate", "evaluate"):
        child = subparsers.add_parser(stage)
        child.add_argument("--scenario-dir", type=Path, action="append", required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--mode", choices=("dev", "formal"), default="dev")
        child.add_argument("--workers", type=int, default=0)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        child.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        child.add_argument("--signoff", type=Path)
    calibrate = subparsers.choices["calibrate"]
    calibrate.add_argument("--capability-dir", type=Path, action="append", required=True)
    calibrate.add_argument(
        "--consumer-amendment", type=Path, default=DEFAULT_CONSUMER_AMENDMENT
    )
    calibrate.add_argument(
        "--historical-reviewed-source-snapshot", type=Path
    )
    calibrate.add_argument("--pre-outcome-attestation", type=Path)
    calibrate.add_argument("--pre-outcome-producer-signoff", type=Path)
    calibrate.add_argument("--authoritative-bundle-root", type=Path)
    evaluate = subparsers.choices["evaluate"]
    evaluate.add_argument("--calibration-lock", type=Path, required=True)
    evaluate.add_argument("--oracle-dir", type=Path, required=True)
    evaluate.add_argument("--sealed-consumption-ledger", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "calibrate":
        signoff = None
        if args.mode == "formal":
            try:
                validate_consumer_amendment_path(
                    args.consumer_amendment, mode="formal"
                )
            except ScenarioBuildError as exc:
                raise ExperimentRunnerError(str(exc)) from exc
            if (
                args.pre_outcome_attestation is None
                or args.pre_outcome_producer_signoff is None
                or args.historical_reviewed_source_snapshot is None
                or args.authoritative_bundle_root is None
            ):
                raise ExperimentRunnerError(
                    "formal calibration requires Amendment-Q migration arguments"
                )
            pre_outcome = load_json_mapping_strict(
                args.pre_outcome_attestation,
                label="pre-outcome attestation",
            )
            signoff = _load_signoff(
                args.signoff,
                stage="calibration",
                config=args.config,
                protocol=args.protocol,
                migration_expected_fields={
                    "consumer_amendment_sha256": sha256_file(
                        args.consumer_amendment
                    ),
                    "historical_reviewed_source_snapshot_sha256": (
                        HISTORICAL_REVIEWED_SOURCE_SNAPSHOT_SHA256
                    ),
                    "pre_outcome_attestation_sha256": pre_outcome.get(
                        "attestation_sha256"
                    ),
                    "pre_outcome_producer_signoff_file_sha256": (
                        pre_outcome.get("producer_signoff_file_sha256")
                    ),
                    "pre_outcome_producer_signoff_self_hash": (
                        pre_outcome.get("producer_signoff_self_hash")
                    ),
                    "authoritative_bundle_root": str(
                        args.authoritative_bundle_root.resolve(strict=True)
                    ),
                },
            )
        status = calibrate_pipeline(
            scenario_dirs=args.scenario_dir,
            capability_dirs=args.capability_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            workers=args.workers,
            config_path=args.config,
            protocol_path=args.protocol,
            signoff=signoff,
            signoff_path=args.signoff,
            historical_reviewed_source_snapshot_path=(
                args.historical_reviewed_source_snapshot
            ),
            pre_outcome_attestation_path=args.pre_outcome_attestation,
            pre_outcome_producer_signoff_path=(
                args.pre_outcome_producer_signoff
            ),
            authoritative_bundle_root_path=args.authoritative_bundle_root,
            consumer_amendment_path=args.consumer_amendment,
        )
    else:
        # Hard reject before opening any sealed manifest/scenario/hash.
        guard_mode_role(args.mode, "sealed")
        if Path(os.path.abspath(args.sealed_consumption_ledger)) != Path(
            os.path.abspath(GLOBAL_SEALED_EVALUATION_CONSUMPTION)
        ):
            raise ExperimentRunnerError(
                "formal sealed ledger path differs from the reviewed global ledger"
            )
        try:
            validate_formal_output_path(args.output_dir, mode="formal")
        except ScenarioBuildError as exc:
            raise ExperimentRunnerError(str(exc)) from exc
        if args.output_dir.exists():
            raise ExperimentRunnerError(
                "refusing to consume sealed data for an existing output directory"
            )
        signoff = _load_signoff(
            args.signoff,
            stage="sealed",
            config=args.config,
            protocol=args.protocol,
        )
        # G1/oracle/lock are calibration-only and are checked before the
        # irreversible one-shot reservation.
        lock, oracle = _load_lock_and_oracle(
            lock_path=args.calibration_lock,
            oracle_dir=args.oracle_dir,
            config_path=args.config,
            protocol_path=args.protocol,
        )
        expected_signoff = {
            "calibration_lock_sha256": lock["manifest_sha256"],
            "oracle_status_sha256": oracle["manifest_sha256"],
            "run_oracle_source_sha256": oracle["run_oracle_source_sha256"],
            "oracle_producer_signoff_sha256": oracle["signoff_sha256"],
            "sealed_consumption_ledger_path": str(
                GLOBAL_SEALED_EVALUATION_CONSUMPTION.resolve(strict=False)
            ),
            "sealed_output_dir": str(args.output_dir.resolve(strict=False)),
        }
        for field, wanted in expected_signoff.items():
            if signoff.get(field) != wanted:
                raise ExperimentRunnerError(f"sealed signoff mismatch: {field}")
        status = evaluate_pipeline(
            scenario_dirs=args.scenario_dir,
            calibration_lock_path=args.calibration_lock,
            oracle_dir=args.oracle_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            workers=args.workers,
            config_path=args.config,
            protocol_path=args.protocol,
            signoff_path=args.signoff,
        )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
