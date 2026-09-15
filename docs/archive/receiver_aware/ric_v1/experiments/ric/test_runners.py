from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from ric.accounting import TraceMetrics, trace_metrics_from_result  # noqa: E402
from ric.build_scenarios import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_PROTOCOL,
    _measure_service_lut_source_sha256,
    ScenarioBuildError,
    ServiceSurface,
    ValidatedInputs,
    atomic_output_directory,
    build_arrival_normalization_census,
    build_link_sensitivity_world_from_primary,
    build_world,
    generate_arrivals,
    guard_mode_role,
    load_service_surface,
    load_worlds,
    main as build_scenarios_main,
    object_sha256,
    role_trace_seeds,
    partition_requests,
    validate_aggregate_utilization,
    validate_gpu_environment_artifact,
    validate_load_normalization_contract,
    validate_sensitivity_primary_source,
    write_scenarios,
)
from ric.build_scenarios import _load_config as load_scenario_config  # noqa: E402
from ric.build_scenarios import _validate_route_tuple_group  # noqa: E402
from ric.capture_routes_gpu import _route_tuple_sha256, expert_sender  # noqa: E402
from ric.prepare_data import add_self_hash, sha256_file, validate_self_hash  # noqa: E402
from ric.replay import ReplayConfig, action_signature, run_replay  # noqa: E402
from ric.run_experiment import (  # noqa: E402
    CONCRETE_JOINBLIND_ARMS,
    ExperimentRunnerError,
    _accounting_row,
    _capability_producer_source_sha256,
    _load_config as load_experiment_config,
    _noninferior_to_frozen_baseline,
    _validate_persisted_accounting,
    _write_csv as write_experiment_csv,
    capability_paired_lcbs,
    compact_replay_result,
    contract_tax_surface_from_tree,
    execute_trace_bundles,
    evaluate_pipeline,
    reserve_sealed_consumption,
    resolve_worker_count,
    select_calibration_baseline,
    strict_retention_bootstrap,
    validate_capability_artifact,
    validate_calibration_signoff_bindings,
    validate_g3_sensitivity_grid,
    validate_oracle_instance_scheduling_contract,
)
from ric.capability_contract import (  # noqa: E402
    EXECUTION_ORDER_RULE,
    capability_execution_order,
)


CONFIG = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


class StrictConfigTests(unittest.TestCase):
    def test_capability_type1_lcb_uses_frozen_500th_order_statistic(self) -> None:
        class BoundaryRandom:
            def __init__(self) -> None:
                self.calls = 0

            def randrange(self, _count: int) -> int:
                replicate = self.calls // 2
                self.calls += 1
                return 0 if replicate < 500 else 1

        with patch(
            "ric.run_experiment.random.Random", return_value=BoundaryRandom()
        ):
            lcb = capability_paired_lcbs(
                {"boundary": [0.0, 1.0]},
                replicates=10000,
                order_statistic_one_based=500,
                seed=2026072226,
            )
        self.assertEqual(lcb["boundary"], 0.0)

    def test_noninferiority_rejects_reduction_interval_crossing_zero(self) -> None:
        crossing = SimpleNamespace(
            cvar99_relative_reduction_lcb=-0.01,
            cvar99_relative_reduction_ucb=0.03,
            violation_absolute_reduction_lcb=-0.001,
            violation_absolute_reduction_ucb=0.004,
        )
        boundary = SimpleNamespace(
            cvar99_relative_reduction_lcb=0.0,
            cvar99_relative_reduction_ucb=0.03,
            violation_absolute_reduction_lcb=0.0,
            violation_absolute_reduction_ucb=0.004,
        )
        self.assertFalse(_noninferior_to_frozen_baseline(crossing))
        self.assertTrue(_noninferior_to_frozen_baseline(boundary))

    def test_capability_counterbalance_exact_mod4_golden(self) -> None:
        baseline = "baseline_nonclosing_first"
        candidate = "candidate_closing_first"
        early = "streaming"
        barrier = "full_layer_barrier"
        expected = (
            ((early, baseline), (early, candidate), (barrier, baseline), (barrier, candidate)),
            ((barrier, candidate), (barrier, baseline), (early, candidate), (early, baseline)),
            ((early, candidate), (early, baseline), (barrier, candidate), (barrier, baseline)),
            ((barrier, baseline), (barrier, candidate), (early, baseline), (early, candidate)),
        )
        observed = tuple(capability_execution_order(CONFIG, trial) for trial in range(8))
        self.assertEqual(observed[:4], expected)
        self.assertEqual(observed[4:], expected)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text(
                '{"schema_version":"ric-config-v1","topology_proxy":'
                '{"evidence_level":"a","evidence_level":"b"}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ExperimentRunnerError, "duplicate JSON.*key|duplicate JSON key"
            ):
                load_experiment_config(path)

            with self.assertRaisesRegex(
                ScenarioBuildError, "duplicate JSON.*key|duplicate JSON key"
            ):
                load_scenario_config(path)

    def test_gpu_environment_accepts_environment_and_artifact_root_fields(self) -> None:
        pid = 1234
        gpu_uuid = "GPU-test"
        app = {
            "pid": pid,
            "gpu_uuid": gpu_uuid,
            "process_name": "python",
            "used_memory_mib": 128.0,
        }
        artifact = {
            "transformers_version": "4.53.3",
            "model_tree_manifest_sha256": "a" * 64,
            "gpu_environment": {
                "producer_pid": pid,
                "gpu_uuid": gpu_uuid,
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "driver_version": "580.105.08",
                "cuda_version": "12.8",
                "pytorch_version": "2.7.1",
                "clock_sm_mhz": 2400.0,
                "power_draw_w": 100.0,
                "memory_used_mib": 128.0,
                "background_gpu_util_percent": 0.0,
                "compute_apps_before": [],
                "compute_apps_after": [app],
            },
        }
        signature = validate_gpu_environment_artifact(
            artifact, config=CONFIG, label="fixture"
        )
        self.assertEqual(signature["transformers_version"], "4.53.3")
        self.assertEqual(
            signature["model_tree_manifest_sha256"], "a" * 64
        )
        self.assertEqual(signature["gpu_environment.gpu_uuid"], gpu_uuid)


def _requests(count: int, role: str) -> list[dict[str, object]]:
    return [
        {
            "request_id": f"{role}-request-{index:03d}",
            "text_sha256": f"{index + 1:064x}",
            "rank_sha256": f"{count - index:064x}",
        }
        for index in range(count)
    ]


def _control_points(summary_sha: str) -> tuple[dict[str, dict[str, float]], str]:
    points = {
        str(count): {
            "state_build_us": 0.01 * count,
            "hash_us": 0.02 * count,
            "encode_us": 0.03 * count,
            "transfer_us": 0.04 * count,
            "decode_us": 0.05 * count,
            "lookup_us": 0.06 * count,
            "apply_us": 0.07 * count,
            "policy_lookup_us": 0.08 * count,
        }
        for count in range(1, 256)
    }
    source = object_sha256(
        {
            "rule": "raw_median_minus_same_count_empty_harness",
            "non_grid_rule": "exact_1_to_255_no_interpolation_or_extrapolation",
            "points": points,
            "service_lut_sha256": summary_sha,
        }
    )
    return points, source


def _fixture_inputs(*, invalid_valid: object = True, split_layer: bool = False):
    model_key = "fixture"
    model_revision = "fixture/model@revision"
    request_ids = tuple(f"request-{index}" for index in range(4))
    placement = {expert: expert_sender(expert, 8, 8) for expert in range(8)}
    origins = {request_id: index for index, request_id in enumerate(request_ids)}
    by_request: dict[str, tuple[dict[str, object], ...]] = {}
    for request_index, request_id in enumerate(request_ids):
        rows = []
        for token in range(128):
            for slot in range(2):
                expert = (2 * token + slot + request_index) % 8
                layer = 7
                if split_layer and request_index == 0 and token == 127 and slot == 1:
                    layer = 8
                rows.append(
                    {
                        "request_id": request_id,
                        "forward_id": f"{request_id}:forward",
                        "batch_id": f"{request_id}:batch",
                        "phase": "prefill",
                        "decode_step": 0,
                        "layer_id": layer,
                        "token_id": f"{request_id}:token:{token}",
                        "token_block_id": f"{request_id}:block:{token}",
                        "topk_slot": slot,
                        "expert_id": expert,
                        "sender_rank": placement[expert],
                        "receiver_rank": origins[request_id],
                        "epoch": 1,
                        "valid": invalid_valid,
                    }
                )
        by_request[request_id] = tuple(rows)
    summary_sha = "a" * 64
    points, source = _control_points(summary_sha)
    surface = ServiceSurface(
        model_key=model_key,
        model_revision=model_revision,
        model_tree_manifest_sha256="9" * 64,
        payload_bytes=30,
        payload_layout_sha256="b" * 64,
        expert_ready_us_by_layer_expert={
            **{f"7:{expert}": 2.0 + expert / 10.0 for expert in range(8)},
            **{f"8:{expert}": 3.0 + expert / 10.0 for expert in range(8)},
        },
        batching_diagnostic_expert_ready_us=2.0,
        sender_pack_us=1.0,
        receiver_unpack_us=3.0,
        join_combine_us=97.0,
        control_tax_by_record_count=points,
        control_tax_source_id=source,
        metadata_sha256="c" * 64,
        summary_sha256=summary_sha,
        raw_sha256="d" * 64,
        producer_source_sha256="e" * 64,
        producer_signoff_sha256=None,
    )
    inputs = ValidatedInputs(
        role="calibration",
        model_key=model_key,
        model_revision=model_revision,
        top_k=2,
        num_experts=8,
        data_manifest={
            "manifest_sha256": "e" * 64,
            "requests": [
                {
                    "request_id": request_id,
                    "text_sha256": f"{index + 1:064x}",
                    "rank_sha256": f"{index + 1:064x}",
                }
                for index, request_id in enumerate(request_ids)
            ],
        },
        route_rows_by_request=by_request,
        placement={
            "manifest_sha256": "f" * 64,
            "ep_size": 8,
            "ranks_per_node": 4,
            "expert_to_sender": {str(key): value for key, value in placement.items()},
            "request_to_receiver": origins,
        },
        route_metadata={
            "manifest_sha256": "1" * 64,
            "model_tree_manifest_sha256": "9" * 64,
            "route_trace_sha256": "2" * 64,
            "capture_routes_source_sha256": "3" * 64,
            "signoff_sha256": None,
        },
        service=surface,
        service_calibration_data_manifest_sha256="e" * 64,
        service_calibration_data_manifest_file_sha256="f" * 64,
        service_calibration_selected_list_sha256="a" * 64,
        service_calibration_data_producer_signoff_sha256=None,
        sealed_input_attestation_sha256=None,
        sealed_input_historical_run_experiment_source_sha256=None,
        sealed_input_historical_calibration_lock_sha256=None,
        sealed_input_historical_calibration_signoff_sha256=None,
        sealed_global_reservation_file_sha256=None,
        sealed_global_consumption_file_sha256=None,
        consumer_amendment_sha256="b" * 64,
        historical_reviewed_source_snapshot_sha256="",
        pre_outcome_attestation_sha256="",
        pre_outcome_producer_signoff_file_sha256="",
        pre_outcome_producer_signoff_self_hash="",
        authoritative_bundle_root="",
        immutable_input_compatibility_sha256="c" * 64,
    )
    world, metadata = build_world(
        inputs=inputs,
        request_ids=request_ids,
        trace_index=0,
        seed=202607223001,
        cell_name="poisson_rho60",
        cell=CONFIG["workloads"]["main_cells"]["poisson_rho60"],
        arrival_census=build_arrival_normalization_census(
            role="calibration",
            cell_name="poisson_rho60",
            cell=CONFIG["workloads"]["main_cells"]["poisson_rho60"],
            seeds=role_trace_seeds(CONFIG, role="calibration", trace_count=16),
            seed_namespace_label=CONFIG["workloads"]["role_seed_ranges"][
                "calibration"
            ]["derivation_salt"],
            count_per_trace=512,
        ),
        config=CONFIG,
        link_gbps=200,
    )
    return inputs, world, metadata


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _service_lut_fixture(
    root: Path,
    *,
    bad_net_component: bool = False,
    missing_route_key: bool = False,
) -> tuple[str, str]:
    model_key = "olmoe"
    spec = CONFIG["models"][model_key]
    model_revision = f"{spec['repo_id']}@{spec['revision']}"
    descriptor = {
        "payload_dtype": "bfloat16",
        "payload_elements_per_row": 15,
        "payload_element_size_bytes": 2,
        "payload_bytes_per_contribution_row": 30,
    }
    layout_sha = object_sha256(descriptor)
    common = {
        "model_key": model_key,
        "layer_id": 1,
        "rows": 0,
        "record_count": 0,
        "source": "measured_5090_host",
        **descriptor,
        "payload_layout_sha256": layout_sha,
        "contract_message_bytes": 0,
        "configured_delay_us": 0.0,
    }
    raw: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    def measured(
        component: str,
        median: float,
        *,
        expert_id: int = -1,
        rows: int = 0,
        record_count: int = 0,
        message_bytes: int = 0,
        source: str = "measured_5090_host",
    ) -> None:
        base = {
            **common,
            "component": component,
            "expert_id": expert_id,
            "rows": rows,
            "record_count": record_count,
            "source": source,
            "contract_message_bytes": message_bytes,
        }
        raw.append({**base, "trial": 0, "us": median})
        summary.append(
            {
                **base,
                "trial_count": 1,
                "median_us": median,
                "p95_us": median,
                "max_us": median,
            }
        )

    measured("expert_execution", 4.0, expert_id=1, rows=1, source="measured_5090_cuda")
    measured("expert_execution", 5.0, expert_id=2, rows=1, source="measured_5090_cuda")
    summary.append(
        {
            **summary[-1],
            "component": "expert_execution_conservative_max_selected_median",
            "expert_id": -1,
        }
    )
    measured("sender_pack", 2.0, rows=1, source="measured_5090_cuda")
    measured("receiver_unpack", 3.0, rows=1, source="measured_5090_cuda")
    measured("canonical_reduction", 97.0, rows=1, source="measured_5090_cuda")
    selected_layers = [1, 3, 5, 7]
    for layer_id in selected_layers:
        for expert_id in range(int(spec["num_experts"])):
            if missing_route_key and layer_id == 3 and expert_id == 4:
                continue
            measured(
                "expert_execution_route_specific_row1",
                10.0 + layer_id + expert_id / 100.0,
                expert_id=expert_id,
                rows=1,
                source="measured_5090_cuda",
            )
            raw[-1]["layer_id"] = layer_id
            summary[-1]["layer_id"] = layer_id
    components = (
        "state_build_contract_record",
        "host_hash_identity",
        "host_encode_contract",
        "host_decode_contract",
        "collision_checked_identity_lookup",
        "epoch_sequence_apply",
        "sender_policy_cache_lookup",
    )
    for count in range(1, 256):
        message_bytes = 16 + 16 * count
        harness = 0.5 + count * 1e-4
        measured(
            "host_empty_harness",
            harness,
            record_count=count,
            message_bytes=message_bytes,
        )
        for index, component in enumerate(components):
            net = 0.0 if bad_net_component and count == 17 and index == 0 else 1.0 + index
            measured(
                component,
                harness + net,
                record_count=count,
                message_bytes=message_bytes,
            )
        transfer = message_bytes * 8.0 / (200.0 * 1000.0)
        derived = {
            **common,
            "component": "contract_transfer_analytic_primary_link",
            "expert_id": -1,
            "record_count": count,
            "source": "analytic_network",
            "contract_message_bytes": message_bytes,
            "trial_count": 1,
            "median_us": transfer,
            "p95_us": transfer,
            "max_us": transfer,
        }
        summary.append(derived)
    raw_path = root / "service_lut_raw.csv"
    summary_path = root / "service_lut.csv"
    _csv(raw_path, raw)
    _csv(summary_path, summary)
    metadata = add_self_hash(
        {
            "schema_version": "ric-service-lut-v1",
            "status": "NOT_TESTED",
            "scientific_result": False,
            "mode": "dev",
            "model_key": model_key,
            "model_revision": model_revision,
            "model_tree_manifest_sha256": "9" * 64,
            **descriptor,
            "payload_layout_sha256": layout_sha,
            "record_count_grid": list(range(1, 256)),
            "route_specific_selected_layers": selected_layers,
            "route_specific_expert_ids": list(range(int(spec["num_experts"]))),
            "route_specific_key_count": len(selected_layers)
            * int(spec["num_experts"]),
            "route_specific_main_component": (
                "expert_execution_route_specific_row1"
            ),
            "config_sha256": sha256_file(DEFAULT_CONFIG),
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
            "service_lut_raw_sha256": sha256_file(raw_path),
            "service_lut_sha256": sha256_file(summary_path),
            "measure_service_lut_source_sha256": (
                _measure_service_lut_source_sha256()
            ),
            "signoff_sha256": None,
            "host_measurement_accounting": {
                "additive_components": list(components),
                "end_to_end_diagnostic_not_additive": "host_apply_wire_contract",
            },
            "data_path_measurement_accounting": {
                "per_contribution_additive_components": [
                    "expert_execution_route_specific_row1",
                    "sender_pack",
                    "receiver_unpack",
                ],
                "per_join_once_only_component": "canonical_reduction",
                "canonical_reduction_charged_once_per_join": True,
            },
            "contract_network_accounting": {
                "primary_link_gbps": 200.0,
            },
        }
    )
    (root / "service_lut_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return model_key, model_revision


def _trace_metric(
    trace_id: str, seed: int, arm: str, cvar: float
) -> TraceMetrics:
    return TraceMetrics(
        trace_id=trace_id,
        workload_seed=seed,
        model_key="olmoe",
        cell="poisson_rho60",
        arm=arm,
        closure_count=1,
        p50_us=cvar,
        p95_us=cvar,
        p99_us=cvar,
        cvar99_us=cvar,
        violation_rate=0.0,
        closure_budget_us=10.0,
        control_bytes_over_payload=0.0,
        stale_rate=0.0,
        fallback_rate=0.0,
        sender_ready_wait_mean_us=0.0,
        sender_ready_wait_p99_us=0.0,
        starvation_count=0,
        full_drain_goodput_per_us=1.0,
        queue_utilization=0.5,
        task_fingerprint="task",
        score_mask_fingerprint="score-mask",
        service_fingerprint="service",
        resource_demand_fingerprint="resource-demand",
        contract_tax_surface_fingerprint="tax",
    )


def _capability_fixture(
    root: Path,
    *,
    bad_row_revision: bool = False,
    mixed_sender: bool = False,
) -> dict[str, object]:
    model_key = "olmoe"
    spec = CONFIG["models"][model_key]
    revision = f"{spec['repo_id']}@{spec['revision']}"
    reference = "a" * 64
    sender_rank = 0
    receiver_rank = 1
    ep_size = 8
    ranks_per_node = 4
    request_id = "ric:calibration:0000:8a882c4c7d16"
    target_layer = 1
    queue_id = f"cuda:0/ric-capability/sender:{sender_rank}:local-return-queue"
    fixtures: dict[str, dict[str, object]] = {}
    plan_rows: dict[str, list[dict[str, int]]] = {}
    for index, task_name in enumerate(("x_closing", "y_nonclosing")):
        expert_id = 16 if mixed_sender and task_name == "y_nonclosing" else index
        contribution_sender = (
            2 if mixed_sender and task_name == "y_nonclosing" else sender_rank
        )
        contribution = {
            "request_id": request_id,
            "forward_id": f"{request_id}:capability-forward",
            "batch_id": f"{request_id}:capability-batch",
            "phase": "prefill",
            "decode_step": 0,
            "layer_id": target_layer,
            "token_id": f"{request_id}:token:{index}",
            "token_block_id": f"{request_id}:token-block:{index}",
            "topk_slot": 1 - index,
            "expert_id": expert_id,
            "sender_rank": contribution_sender,
            "receiver_rank": receiver_rank,
            "epoch": 1,
        }
        identity = {
            "schema_version": "ric-capability-task-v1",
            "task_name": task_name,
            "model_key": model_key,
            "model_revision": revision,
            "sender_rank": sender_rank,
            "receiver_rank": receiver_rank,
            "sender_local_queue_id": queue_id,
            "shared_cut_path": "node0->node0",
            "receiver_combine_resource": "receiver:1:combine",
            "contribution_identities": [contribution],
            "payload_shape": [1, 1],
            "payload_stride": [1, 1],
            "payload_dtype": "torch.float32",
            "payload_bytes": 4,
            "payload_sha256": str(index + 1) * 64,
        }
        fixtures[task_name] = {
            **identity,
            "fixture_identity_sha256": object_sha256(identity),
        }
        plan_rows[task_name] = [
            {
                "token_index": index,
                "topk_slot": 1 - index,
                "expert_id": expert_id,
                "sender_rank": contribution_sender,
            }
        ]
    ready_ids = [
        str(fixtures["x_closing"]["fixture_identity_sha256"]),
        str(fixtures["y_nonclosing"]["fixture_identity_sha256"]),
    ]
    snapshot = {
        "ready_count": 2,
        "ready_task_ids": ready_ids,
        "all_ready_before_selection": True,
    }
    snapshot_sha = object_sha256(snapshot)
    rows = []
    actions = []
    for release in ("streaming", "full_layer_barrier"):
        for policy in ("baseline_nonclosing_first", "candidate_closing_first"):
            execution_index = capability_execution_order(CONFIG, 0).index(
                (release, policy)
            )
            order = (
                ("x_closing", "y_nonclosing")
                if policy == "candidate_closing_first"
                else ("y_nonclosing", "x_closing")
            )
            group_id = object_sha256(
                {
                    "trial": 0,
                    "execution_order_index": execution_index,
                    "policy": policy,
                    "release_mode": release,
                    "ready_task_ids": ready_ids,
                }
            )
            for order_index, task_name in enumerate(order):
                start = 1.0 + order_index * 2.0
                actions.append(
                    {
                        "schema_version": "ric-capability-action-v1",
                        "action_trace_group_id": group_id,
                        "trial": 0,
                        "execution_order_index": execution_index,
                        "policy": policy,
                        "release_mode": release,
                        "service_order_index": order_index,
                        "task_name": task_name,
                        "fixture_identity_sha256": fixtures[task_name][
                            "fixture_identity_sha256"
                        ],
                        "task_identity": fixtures[task_name],
                        "payload_bytes": 4,
                        "stream_id": 7,
                        "queue_snapshot": snapshot,
                        "queue_snapshot_sha256": snapshot_sha,
                        "enqueue_ts_us": 0.1,
                        "queue_ready_ts_us": 0.2,
                        "selected_ts_us": start - 0.1,
                        "service_start_ts_us": start,
                        "service_end_ts_us": start + 1.0,
                        "nvtx_range_label": (
                            f"ric_capability/trial=0/policy={policy}/"
                            f"release={release}/service={task_name}"
                        ),
                        "nvtx_labels": {
                            "enqueue": (
                                f"ric_capability/trial=0/policy={policy}/"
                                f"release={release}/enqueue={task_name}"
                            ),
                            "queue_snapshot": (
                                f"ric_capability/trial=0/policy={policy}/"
                                f"release={release}/queue_snapshot=both_ready"
                            ),
                            "select": (
                                f"ric_capability/trial=0/policy={policy}/"
                                f"release={release}/select={task_name}"
                            ),
                            "service": (
                                f"ric_capability/trial=0/policy={policy}/"
                                f"release={release}/service={task_name}"
                            ),
                        },
                        "source": "measured_5090_cuda",
                    }
                )
            rows.append(
                {
                    "trial": 0,
                    "execution_order_index": execution_index,
                    "model_key": model_key,
                    "model_revision": (
                        "wrong@revision" if bad_row_revision and not rows else revision
                    ),
                    "request_id": request_id,
                    "layer_id": target_layer,
                    "block_rows": 1,
                    "top_k": int(spec["top_k"]),
                    "sender_rank": sender_rank,
                    "receiver_rank": receiver_rank,
                    "policy": policy,
                    "release_mode": release,
                    "service_order": ",".join(order),
                    "action_trace_group_id": group_id,
                    "queue_snapshot_sha256": snapshot_sha,
                    "stream_id": 7,
                    "canonical_equal": True,
                    "canonical_reference_sha256": reference,
                    "canonical_output_sha256": reference,
                    "physical_frontier_us": (
                        3.5
                        if policy == "baseline_nonclosing_first"
                        else 1.5
                    ),
                    "application_release_us": (
                        3.5
                        if release == "streaming"
                        and policy == "baseline_nonclosing_first"
                        else 1.5
                        if release == "streaming"
                        else 5.0
                    ),
                    "downstream_start_us": (
                        3.5
                        if release == "streaming"
                        and policy == "baseline_nonclosing_first"
                        else 1.5
                        if release == "streaming"
                        else 5.0
                    ),
                    "total_us": 6.0,
                    "source": "measured_5090_cuda",
                }
            )
    raw_path = root / "capability_raw.csv"
    _csv(raw_path, rows)
    action_path = root / "capability_action_trace.jsonl"
    action_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in actions),
        encoding="utf-8",
    )
    profiler_diagnostics = {}
    for profiler_release, profiler_trial in (
        ("streaming", -1_000_000),
        ("full_layer_barrier", -1_000_001),
    ):
        profiler_prefix = (
            f"ric_capability/trial={profiler_trial}/policy=candidate_closing_first/"
            f"release={profiler_release}"
        )
        profiler_labels = [
            profiler_prefix,
            *(f"{profiler_prefix}/enqueue={name}" for name in fixtures),
            f"{profiler_prefix}/queue_snapshot=both_ready",
            *(f"{profiler_prefix}/select={name}" for name in fixtures),
            *(f"{profiler_prefix}/service={name}" for name in fixtures),
        ]
        profiler_file = f"capability_cuda_trace_{profiler_release}.json"
        profiler_path = root / profiler_file
        profiler_path.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        *(
                            {"name": label, "cat": "user_annotation"}
                            for label in profiler_labels
                        ),
                        {
                            "name": "fixtureKernel",
                            "cat": "kernel",
                            "args": {"stream": 13},
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        profiler_diagnostics[profiler_release] = {
            "trial": profiler_trial,
            "policy": "candidate_closing_first",
            "release_mode": profiler_release,
            "trace_file": profiler_file,
            "required_labels": profiler_labels,
            "trace_sha256": sha256_file(profiler_path),
            "sender_local_stream_id": 7,
            "gpu_activity_stream_ordinal": 13,
            "canonical_output_sha256": reference,
        }
    contribution_path = root / "expert_contributions.pt"
    contribution_path.write_bytes(b"real-contribution-fixture")
    artifact = add_self_hash(
        {
            "schema_version": "ric-capability-v1",
            "status": "NOT_TESTED",
            "scientific_result": False,
            "mode": "dev",
            "model_key": model_key,
            "model_revision": revision,
            "model_tree_manifest_sha256": "9" * 64,
            "transformers_version": "fixture",
            "model_source": {
                "kind": "explicit_local_directory",
                "frozen_repo_id": spec["repo_id"],
                "frozen_revision": spec["revision"],
                "expected_tree_manifest_sha256": "9" * 64,
                "tree_manifest_sha256": "9" * 64,
            },
            "data_manifest_sha256": "8" * 64,
            "config_sha256": sha256_file(DEFAULT_CONFIG),
            "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
            "trials": 1,
            "warmups": 1,
            "block_rows": 1,
            "request_id": request_id,
            "target_layer": target_layer,
            "frozen_selected_layers": [0, 1, 2, 3],
            "evidence_boundary": (
                "REAL_5090_EXPERT_OUTPUT_AND_LOCAL_CUDA_STREAM / NOT_NCCL / NOT_RDMA"
            ),
            "ready_result_orders": {
                "baseline": ["y_nonclosing", "x_closing"],
                "candidate": ["x_closing", "y_nonclosing"],
            },
            "release_modes": ["streaming", "full_layer_barrier"],
            "execution_order_rule": EXECUTION_ORDER_RULE,
            "persistent_sender_local_cuda_stream_across_all_arms": True,
            "sender_local_stream_id": 7,
            "sender_local_queue_id": queue_id,
            "sender_rank": sender_rank,
            "receiver_rank": receiver_rank,
            "ep_size": ep_size,
            "ranks_per_node": ranks_per_node,
            "num_experts": int(spec["num_experts"]),
            "top_k": int(spec["top_k"]),
            "sender_local_selection": {
                "selection_rule": "route_identity_hash_sender_local_distinct_token_v1",
                "sender_rank": sender_rank,
                "x_closing": plan_rows["x_closing"],
                "y_nonclosing": plan_rows["y_nonclosing"],
                "support_by_sender": {"0": 2},
            },
            "task_fixtures": fixtures,
            "queue_snapshot_ready_count": 2,
            "action_trace_schema_version": "ric-capability-action-v1",
            "action_trace_row_count": len(actions),
            "capability_action_trace_sha256": sha256_file(action_path),
            "nvtx_ranges_emitted": True,
            "action_trace_evidence_boundary": (
                "CUDA_EVENT_ACTION_TRACE_WITH_EMITTED_NVTX_LABELS"
            ),
            "profiler_diagnostic_not_in_timing_trials": True,
            "profiler_trace_kind": "torch_profiler_chrome_trace_cpu_cuda",
            "profiler_diagnostics": profiler_diagnostics,
            "canonical_reference_sha256": reference,
            "all_canonical_hashes_equal": True,
            "raw_trials_sha256": sha256_file(raw_path),
            "expert_contributions_sha256": sha256_file(contribution_path),
            "expert_contributions_source": "native_unpatched_model_expert_execution",
            "measure_capability_source_sha256": _capability_producer_source_sha256(),
            "signoff_sha256": None,
            "summary": {
                "streaming_frontier_advance_us": 2.0,
                "streaming_downstream_advance_us": 2.0,
                "streaming_frontier_paired_mean_us": 2.0,
                "streaming_frontier_paired_lcb_us": 2.0,
                "streaming_downstream_paired_mean_us": 2.0,
                "streaming_downstream_paired_lcb_us": 2.0,
                "release_interaction_paired_mean_us": 2.0,
                "release_interaction_paired_lcb_us": 2.0,
                "downstream_interaction_paired_mean_us": 2.0,
                "downstream_interaction_paired_lcb_us": 2.0,
                "paired_bootstrap_replicates": 10000,
                "paired_bootstrap_seed": 2026072226,
                "paired_one_sided_confidence": 0.95,
                "paired_lcb_order_statistic_one_based": 500,
                "barrier_application_release_difference_us": 0.0,
                "barrier_downstream_start_difference_us": 0.0,
                "baseline_streaming_release_median_us": 3.5,
                "candidate_streaming_release_median_us": 1.5,
                "baseline_barrier_release_median_us": 5.0,
                "candidate_barrier_release_median_us": 5.0,
                "baseline_barrier_total_median_us": 6.0,
                "barrier_cross_policy_timing_diagnostic_only": True,
                "barrier_application_release_paired_mean_us_diagnostic": 0.0,
                "barrier_application_release_max_abs_paired_difference_us_diagnostic": 0.0,
                "barrier_downstream_paired_mean_us_diagnostic": 0.0,
                "barrier_downstream_max_abs_paired_difference_us_diagnostic": 0.0,
                "event_precedence_all_trials_pass": True,
                "event_precedence_failure_count": 0,
                "event_precedence_failures": [],
            },
        }
    )
    (root / "capability_probe.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    return artifact


class PartitionAndArrivalTests(unittest.TestCase):
    def test_route_tuple_consumer_rejects_tied_slot_swap_and_bf16_mutation(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        experts = torch.tensor([[1, 2]], dtype=torch.int64)
        weights = torch.tensor([[0.5, 0.5]], dtype=torch.bfloat16)
        parity = {
            "effective_route_weight_dtype": "torch.bfloat16",
            "native_route_tuple_sha256": _route_tuple_sha256(experts, weights),
        }
        rows = [
            {
                "token_position": 0,
                "topk_slot": slot,
                "expert_id": expert,
                "route_weight": 0.5,
                "route_weight_fp32_precast": 0.5,
                "route_weight_dtype": "torch.bfloat16",
            }
            for slot, expert in enumerate((1, 2))
        ]
        _validate_route_tuple_group(rows, parity_row=parity, top_k=2, valid_tokens=1)
        swapped = [dict(row) for row in rows]
        swapped[0]["expert_id"], swapped[1]["expert_id"] = (
            swapped[1]["expert_id"],
            swapped[0]["expert_id"],
        )
        with self.assertRaisesRegex(ScenarioBuildError, "tuple hash"):
            _validate_route_tuple_group(
                swapped, parity_row=parity, top_k=2, valid_tokens=1
            )
        mutated = [dict(row) for row in rows]
        mutated[0]["route_weight"] = 0.50390625
        with self.assertRaisesRegex(ScenarioBuildError, "tuple hash"):
            _validate_route_tuple_group(
                mutated, parity_row=parity, top_k=2, valid_tokens=1
            )

    def test_frozen_partition_is_disjoint_16_and_32(self) -> None:
        calibration = partition_requests(
            _requests(64, "calibration"), role="calibration", config=CONFIG
        )
        sealed = partition_requests(_requests(128, "sealed"), role="sealed", config=CONFIG)
        self.assertEqual((len(calibration), len(sealed)), (16, 32))
        self.assertTrue(all(len(group) == 4 for group in calibration + sealed))
        self.assertEqual(len({item for group in sealed for item in group}), 128)

    def test_partition_rejects_text_reuse(self) -> None:
        requests = _requests(64, "calibration")
        requests[1]["text_sha256"] = requests[0]["text_sha256"]
        with self.assertRaisesRegex(ScenarioBuildError, "text reused"):
            partition_requests(requests, role="calibration", config=CONFIG)

    def test_dev_rejects_sealed_before_artifact_read(self) -> None:
        with self.assertRaisesRegex(ScenarioBuildError, "forbidden"):
            guard_mode_role("dev", "sealed")

    def test_poisson_and_true_continuous_time_mmpp(self) -> None:
        poisson = generate_arrivals(
            cell=CONFIG["workloads"]["main_cells"]["poisson_rho60"],
            count=4096,
            mu_per_us=1.0,
            bottleneck_work_us=4096.0,
            seed=7,
        )
        self.assertLess(abs(poisson.realized_offered_rho - 0.60), 0.04)
        mmpp = generate_arrivals(
            cell=CONFIG["workloads"]["main_cells"]["ctmc_mmpp_rho85"],
            count=4096,
            mu_per_us=1.0,
            bottleneck_work_us=4096.0,
            seed=9,
        )
        self.assertGreater(len(mmpp.state_transitions_us), 20)
        arrival_times = set(mmpp.arrivals_us)
        self.assertTrue(
            any(time_us not in arrival_times for time_us, _old, _new in mmpp.state_transitions_us)
        )
        self.assertLess(abs(mmpp.realized_offered_rho - 0.85), 0.15)

    def test_all_cells_use_one_role_cell_factor_and_hit_exact_target(self) -> None:
        seeds = role_trace_seeds(CONFIG, role="calibration", trace_count=16)
        cells = {
            **CONFIG["workloads"]["main_cells"],
            **CONFIG["workloads"]["negative_control"],
        }
        censuses = {
            name: build_arrival_normalization_census(
                role="calibration",
                cell_name=name,
                cell=cell,
                seeds=seeds,
                seed_namespace_label="ric-v1-calibration-arrival",
                count_per_trace=512,
            )
            for name, cell in cells.items()
        }
        self.assertEqual(len({row.algorithm for row in censuses.values()}), 1)
        for name, census in censuses.items():
            target = float(cells[name]["target_utilization"])
            self.assertAlmostEqual(census.normalized_aggregate_rho, target, places=12)
            self.assertTrue(census.no_policy_or_oracle_input)
            self.assertEqual(len(census.traces), 16)
            self.assertEqual(
                len({census.time_dilation_factor for _trace in census.traces}), 1
            )

    def test_normalization_preserves_mmpp_state_order_and_burst_shape(self) -> None:
        cell = CONFIG["workloads"]["main_cells"]["ctmc_mmpp_rho85"]
        census = build_arrival_normalization_census(
            role="calibration",
            cell_name="ctmc_mmpp_rho85",
            cell=cell,
            seeds=(202607223001, 202607223002),
            seed_namespace_label="namespace-a",
            count_per_trace=512,
        )
        trace = census.traces[0]
        self.assertEqual(trace.raw.arrival_states, trace.normalized.arrival_states)
        self.assertEqual(
            [(old, new) for _time, old, new in trace.raw.state_transitions_us],
            [
                (old, new)
                for _time, old, new in trace.normalized.state_transitions_us
            ],
        )
        for raw, normalized in zip(
            trace.raw.arrivals_us, trace.normalized.arrivals_us
        ):
            self.assertAlmostEqual(
                normalized, raw * census.time_dilation_factor, places=12
            )
        for raw, normalized in zip(
            trace.raw.state_transitions_us,
            trace.normalized.state_transitions_us,
        ):
            self.assertAlmostEqual(
                normalized[0], raw[0] * census.time_dilation_factor, places=12
            )
        self.assertTrue(
            all(
                right > left
                for left, right in zip(
                    trace.normalized.arrivals_us,
                    trace.normalized.arrivals_us[1:],
                )
            )
        )

    def test_seed_namespace_is_provenance_only_and_bad_schema_fails(self) -> None:
        cell = CONFIG["workloads"]["main_cells"]["poisson_rho60"]
        first = build_arrival_normalization_census(
            role="calibration",
            cell_name="poisson_rho60",
            cell=cell,
            seeds=(7, 8),
            seed_namespace_label="namespace-a",
            count_per_trace=64,
        )
        second = build_arrival_normalization_census(
            role="calibration",
            cell_name="poisson_rho60",
            cell=cell,
            seeds=(7, 8),
            seed_namespace_label="namespace-b",
            count_per_trace=64,
        )
        self.assertEqual(
            [trace.raw_fingerprint for trace in first.traces],
            [trace.raw_fingerprint for trace in second.traces],
        )
        self.assertNotEqual(
            first.raw_census_fingerprint, second.raw_census_fingerprint
        )
        with self.assertRaisesRegex(ScenarioBuildError, "unique sorted"):
            build_arrival_normalization_census(
                role="calibration",
                cell_name="poisson_rho60",
                cell=cell,
                seeds=(7, 7),
                seed_namespace_label="namespace-a",
                count_per_trace=64,
            )
        illegal = dict(cell)
        illegal["unexpected"] = True
        with self.assertRaisesRegex(ScenarioBuildError, "schema"):
            build_arrival_normalization_census(
                role="calibration",
                cell_name="poisson_rho60",
                cell=illegal,
                seeds=(7, 8),
                seed_namespace_label="namespace-a",
                count_per_trace=64,
            )
        broken_mmpp = dict(
            CONFIG["workloads"]["main_cells"]["ctmc_mmpp_rho85"]
        )
        broken_mmpp["target_utilization"] = 0.75
        with self.assertRaisesRegex(ScenarioBuildError, "stationary intensity"):
            build_arrival_normalization_census(
                role="calibration",
                cell_name="ctmc_mmpp_rho85",
                cell=broken_mmpp,
                seeds=(7, 8),
                seed_namespace_label="namespace-a",
                count_per_trace=64,
            )

    def test_normalized_aggregate_cannot_use_legacy_wide_tolerance(self) -> None:
        cells = {"poisson_rho60": {"target_utilization": 0.60}}
        row = {
            "cell": "poisson_rho60",
            "realized_offered_rho": 0.600001,
            "arrival_normalization_no_policy_or_oracle_input": True,
            "arrival_normalization_algorithm": (
                "ric-v1-role-cell-dimensionless-common-time-dilation"
            ),
        }
        with self.assertRaisesRegex(ScenarioBuildError, "BLOCKED_WORKLOAD"):
            validate_aggregate_utilization((row,), cells=cells, tolerance=0.03)

    def test_amendment_n_config_contract_is_exact(self) -> None:
        contract = validate_load_normalization_contract(CONFIG)
        self.assertEqual(contract["raw_schedule_count"], 512)
        self.assertTrue(contract["no_policy_or_oracle_input"])
        broken = json.loads(json.dumps(CONFIG))
        broken["workloads"]["finite_horizon_load_normalization"][
            "forbidden_factor_inputs"
        ].remove("experiment_outcome")
        with self.assertRaisesRegex(ScenarioBuildError, "Amendment N"):
            validate_load_normalization_contract(broken)

    def test_bad_aggregate_rho_blocks_before_artifact_commit(self) -> None:
        cells = {"poisson_rho60": {"target_utilization": 0.60}}
        with self.assertRaisesRegex(ScenarioBuildError, "BLOCKED_WORKLOAD"):
            validate_aggregate_utilization(
                ({"cell": "poisson_rho60", "realized_offered_rho": 0.90},),
                cells=cells,
                tolerance=0.03,
            )


class ScenarioBuilderTests(unittest.TestCase):
    def test_sensitivity_requires_explicit_primary_scenario(self) -> None:
        with self.assertRaisesRegex(ScenarioBuildError, "primary-scenario-dir"):
            validate_sensitivity_primary_source(
                primary_scenario_dir=None,
                requested_link_gbps=100,
                primary_link_gbps=200,
            )

    def test_sensitivity_main_never_calls_rng_or_census(self) -> None:
        inputs, primary_world, primary_metadata = _fixture_inputs()
        args = SimpleNamespace(
            role="calibration",
            mode="dev",
            model_key="fixture",
            data_manifest=Path("data.json"),
            calibration_data_manifest=None,
            route_dir=Path("route"),
            service_lut_dir=Path("lut"),
            output_dir=Path("out"),
            link_gbps=100,
            primary_scenario_dir=Path("primary"),
            config=DEFAULT_CONFIG,
            protocol=DEFAULT_PROTOCOL,
            consumer_amendment=Path("amendment.md"),
            historical_reviewed_source_snapshot=None,
            pre_outcome_attestation=None,
            pre_outcome_producer_signoff=None,
            authoritative_bundle_root=None,
            historical_calibration_lock=None,
            signoff=None,
        )
        primary_tree = {"manifest_sha256": "a" * 64, "worlds": [primary_metadata]}
        with patch(
            "ric.build_scenarios.parse_args", return_value=args
        ), patch(
            "ric.build_scenarios.load_validated_inputs", return_value=inputs
        ), patch(
            "ric.build_scenarios.load_worlds",
            return_value=(primary_tree, (primary_world,)),
        ), patch(
            "ric.build_scenarios.validate_sensitivity_primary_source"
        ), patch(
            "ric.build_scenarios.build_link_sensitivity_world_from_primary",
            return_value=(primary_world, primary_metadata),
        ), patch(
            "ric.build_scenarios.write_scenarios", return_value={}
        ), patch(
            "ric.build_scenarios.build_arrival_normalization_census",
            side_effect=AssertionError("sensitivity must not build census"),
        ) as census_builder, patch(
            "ric.build_scenarios.generate_arrivals",
            side_effect=AssertionError("sensitivity must not use RNG"),
        ) as arrival_generator, patch("builtins.print"):
            build_scenarios_main()
        census_builder.assert_not_called()
        arrival_generator.assert_not_called()
        with self.assertRaisesRegex(ScenarioBuildError, "must not consume"):
            validate_sensitivity_primary_source(
                primary_scenario_dir=Path("unexpected"),
                requested_link_gbps=200,
                primary_link_gbps=200,
            )

    def test_role_seed_ranges_are_disjoint_and_exact(self) -> None:
        calibration = role_trace_seeds(CONFIG, role="calibration", trace_count=16)
        sealed = role_trace_seeds(CONFIG, role="sealed", trace_count=32)
        self.assertFalse(set(calibration) & set(sealed))
        self.assertEqual(calibration[0], 202607223001)
        self.assertEqual(sealed[0], 202607224001)
        with self.assertRaisesRegex(ScenarioBuildError, "trace count"):
            role_trace_seeds(CONFIG, role="calibration", trace_count=15)

    def test_full_world_uses_amendment_b_and_shared_sibling_arrival(self) -> None:
        _inputs, world, metadata = _fixture_inputs()
        self.assertEqual(world.workload_seed, 202607223001)
        self.assertEqual(len(world.tasks), 4 * 128 * 2)
        self.assertEqual(world.tasks[0].contribution.descriptor_bytes, 16)
        self.assertEqual(world.tasks[0].contribution.alignment_bytes, 2)
        expected_cut = (30 + 16 + 2) * 8.0 / (200.0 * 1000.0)
        self.assertAlmostEqual(world.tasks[0].stage_service.shared_cut_us, expected_cut)
        for siblings in world.joins.values():
            self.assertEqual(len({row.contribution.arrival_us for row in siblings}), 1)
            self.assertEqual(len({row.contribution.deadline_us for row in siblings}), 1)
            expected_contribution_path = max(
                _inputs.service.expert_ready_us_by_layer_expert[
                    f"{sibling.identity.layer_id}:{sibling.identity.expert_id}"
                ]
                + sibling.stage_service.total_us
                for sibling in siblings
            )
            expected_isolated = (
                expected_contribution_path
                + siblings[0].stage_service.join_combine_us
            )
            for sibling in siblings:
                identity = sibling.identity
                expected_ready = _inputs.service.expert_ready_us_by_layer_expert[
                    f"{identity.layer_id}:{identity.expert_id}"
                ]
                self.assertAlmostEqual(
                    sibling.contribution.ready_us - sibling.contribution.arrival_us,
                    expected_ready,
                )
                self.assertAlmostEqual(
                    sibling.contribution.deadline_us
                    - sibling.contribution.arrival_us,
                    2.0 * expected_isolated,
                )
        self.assertEqual(metadata["join_count"], 512)

    def test_non_bool_valid_and_multi_layer_fail_closed(self) -> None:
        with self.assertRaisesRegex(Exception, "valid"):
            _fixture_inputs(invalid_valid="false")
        with self.assertRaisesRegex(ScenarioBuildError, "multiple selected"):
            _fixture_inputs(split_layer=True)

    def test_roundtrip_hash_seed_and_link_tax_scaling(self) -> None:
        inputs, world, metadata = _fixture_inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            slow = root / "slow"
            primary_payload = write_scenarios(
                output_dir=primary,
                inputs=inputs,
                worlds=(world,),
                world_metadata=(metadata,),
                config_path=DEFAULT_CONFIG,
                protocol_path=DEFAULT_PROTOCOL,
                mode="dev",
                link_gbps=200,
            )
            with patch(
                "ric.build_scenarios.generate_arrivals",
                side_effect=AssertionError("sensitivity must not use RNG"),
            ) as arrival_generator, patch(
                "ric.build_scenarios.build_arrival_normalization_census",
                side_effect=AssertionError("sensitivity must not renormalize"),
            ) as census_builder:
                slow_world, slow_meta = build_link_sensitivity_world_from_primary(
                    inputs=inputs,
                    primary_world=world,
                    primary_metadata=metadata,
                    primary_scenario_tree_sha256=primary_payload["manifest_sha256"],
                    config=CONFIG,
                    link_gbps=100,
                )
            arrival_generator.assert_not_called()
            census_builder.assert_not_called()
            write_scenarios(
                output_dir=slow,
                inputs=inputs,
                worlds=(slow_world,),
                world_metadata=(slow_meta,),
                config_path=DEFAULT_CONFIG,
                protocol_path=DEFAULT_PROTOCOL,
                mode="dev",
                link_gbps=100,
            )
            tree200, loaded = load_worlds(primary, expected_role="calibration", mode="dev")
            tree100, loaded100 = load_worlds(
                slow, expected_role="calibration", mode="dev"
            )
            validate_sensitivity_primary_source(
                primary_scenario_dir=primary,
                requested_link_gbps=100,
                primary_link_gbps=200,
                tree=tree200,
                inputs=inputs,
                config_path=DEFAULT_CONFIG,
                protocol_path=DEFAULT_PROTOCOL,
            )
            self.assertEqual(loaded[0].workload_seed, world.workload_seed)
            self.assertEqual(loaded100[0].workload_seed, world.workload_seed)
            self.assertEqual(
                tree100["sensitivity_source_primary_scenario_tree_sha256"],
                tree200["manifest_sha256"],
            )
            point200 = tree200["service_surface"]["control_tax_by_record_count"]["7"]
            point100 = tree100["service_surface"]["control_tax_by_record_count"]["7"]
            self.assertEqual(point100["state_build_us"], point200["state_build_us"])
            self.assertAlmostEqual(point100["transfer_us"], 2 * point200["transfer_us"])
            self.assertNotEqual(
                tree100["service_surface"]["control_tax_source_id"],
                tree200["service_surface"]["control_tax_source_id"],
            )
            arrivals200 = {
                join: siblings[0].contribution.arrival_us
                for join, siblings in world.joins.items()
            }
            arrivals100 = {
                join: siblings[0].contribution.arrival_us
                for join, siblings in slow_world.joins.items()
            }
            self.assertEqual(arrivals100, arrivals200)
            self.assertEqual(
                slow_meta["causal_arrival_fingerprint"],
                metadata["causal_arrival_fingerprint"],
            )
            self.assertEqual(
                slow_meta["normalized_arrival_transition_fingerprint"],
                metadata["normalized_arrival_transition_fingerprint"],
            )
            self.assertEqual(
                slow_meta["block_permutation_fingerprint"],
                metadata["block_permutation_fingerprint"],
            )
            self.assertEqual(
                slow_meta["isolated_path_values_us"],
                metadata["isolated_path_values_us"],
            )
            deadlines200 = {
                task.task_id: (
                    task.contribution.deadline_us - task.contribution.arrival_us
                )
                for task in world.tasks
            }
            deadlines100 = {
                task.task_id: (
                    task.contribution.deadline_us - task.contribution.arrival_us
                )
                for task in slow_world.tasks
            }
            self.assertEqual(deadlines100, deadlines200)
            self.assertNotEqual(
                slow_world.tasks[0].stage_service.shared_cut_us,
                world.tasks[0].stage_service.shared_cut_us,
            )
            self.assertNotEqual(
                slow_meta["resource_demand_fingerprint"],
                metadata["resource_demand_fingerprint"],
            )
            with self.assertRaisesRegex(ScenarioBuildError, "overwrite"):
                write_scenarios(
                    output_dir=primary,
                    inputs=inputs,
                    worlds=(world,),
                    world_metadata=(metadata,),
                    config_path=DEFAULT_CONFIG,
                    protocol_path=DEFAULT_PROTOCOL,
                    mode="dev",
                    link_gbps=200,
                )

    def test_task_trace_tamper_is_detected(self) -> None:
        inputs, world, metadata = _fixture_inputs()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scenario"
            write_scenarios(
                output_dir=output,
                inputs=inputs,
                worlds=(world,),
                world_metadata=(metadata,),
                config_path=DEFAULT_CONFIG,
                protocol_path=DEFAULT_PROTOCOL,
                mode="dev",
                link_gbps=200,
            )
            with (output / "task_trace.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ScenarioBuildError, "hash mismatch"):
                load_worlds(output, expected_role="calibration", mode="dev")

    def test_rehashed_task_resource_identity_tamper_is_detected(self) -> None:
        inputs, world, metadata = _fixture_inputs()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scenario"
            write_scenarios(
                output_dir=output,
                inputs=inputs,
                worlds=(world,),
                world_metadata=(metadata,),
                config_path=DEFAULT_CONFIG,
                protocol_path=DEFAULT_PROTOCOL,
                mode="dev",
                link_gbps=200,
            )
            task_path = output / "task_trace.jsonl"
            rows = [
                json.loads(line)
                for line in task_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["resources"]["sender_egress"] = "sender:999:egress"
            task_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            tree_path = output / "scenario_tree.json"
            tree = json.loads(tree_path.read_text(encoding="utf-8"))
            tree.pop("manifest_sha256")
            tree["task_trace_sha256"] = sha256_file(task_path)
            tree_path.write_text(
                json.dumps(add_self_hash(tree), sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ScenarioBuildError, "resource path"):
                load_worlds(output, expected_role="calibration", mode="dev")


class ServiceSurfaceTests(unittest.TestCase):
    def test_raw_repeats_and_exact_255_tax_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, revision = _service_lut_fixture(root)
            surface = load_service_surface(
                root,
                model_key=model,
                model_revision=revision,
                config_sha256=sha256_file(DEFAULT_CONFIG),
                protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                mode="dev",
                expected_selected_layers=[1, 3, 5, 7],
                expected_num_experts=int(CONFIG["models"][model]["num_experts"]),
            )
            self.assertEqual(surface.payload_bytes, 30)
            self.assertEqual(surface.sender_pack_us, 2.0)
            self.assertEqual(surface.receiver_unpack_us, 3.0)
            self.assertEqual(surface.join_combine_us, 97.0)
            self.assertEqual(surface.expert_ready_us_by_layer_expert["3:4"], 13.04)
            self.assertEqual(len(surface.control_tax_by_record_count), 255)
            self.assertAlmostEqual(
                surface.control_tax_by_record_count["17"]["state_build_us"], 1.0
            )

    def test_join_combine_is_once_per_join_not_per_topk_contribution(self) -> None:
        inputs, world, _metadata = _fixture_inputs()
        self.assertEqual(inputs.top_k, 2)
        self.assertEqual(inputs.service.join_combine_us, 97.0)
        self.assertEqual(len(world.joins), 4 * 128)
        for task in world.tasks:
            self.assertEqual(task.stage_service.sender_egress_us, 1.0)
            self.assertEqual(task.stage_service.receiver_ingress_us, 3.0)
            self.assertNotEqual(task.stage_service.receiver_ingress_us, 97.0)
            self.assertEqual(task.stage_service.join_combine_us, 97.0)
        once_only_combine = sum(
            siblings[0].stage_service.join_combine_us
            for siblings in world.joins.values()
        )
        self.assertEqual(once_only_combine, len(world.joins) * 97.0)
        self.assertNotEqual(once_only_combine, len(world.tasks) * 97.0)

    def test_nonpositive_harness_adjusted_component_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, revision = _service_lut_fixture(root, bad_net_component=True)
            with self.assertRaisesRegex(ScenarioBuildError, "BLOCKED_CONTROL_TAX"):
                load_service_surface(
                    root,
                    model_key=model,
                    model_revision=revision,
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    mode="dev",
                    expected_selected_layers=[1, 3, 5, 7],
                    expected_num_experts=int(CONFIG["models"][model]["num_experts"]),
                )

    def test_missing_route_specific_expert_service_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, revision = _service_lut_fixture(root, missing_route_key=True)
            with self.assertRaisesRegex(
                ScenarioBuildError, "expert_execution_route_specific_row1"
            ):
                load_service_surface(
                    root,
                    model_key=model,
                    model_revision=revision,
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    mode="dev",
                    expected_selected_layers=[1, 3, 5, 7],
                    expected_num_experts=int(CONFIG["models"][model]["num_experts"]),
                )


class CapabilityArtifactTests(unittest.TestCase):
    def test_capability_binds_raw_payload_tree_and_exact_trial_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            loaded, gate = validate_capability_artifact(
                root,
                model_key="olmoe",
                model_revision=str(artifact["model_revision"]),
                mode="dev",
                config_sha256=sha256_file(DEFAULT_CONFIG),
                protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                data_manifest_sha256="8" * 64,
                model_tree_manifest_sha256="9" * 64,
                config=CONFIG,
            )
            self.assertTrue(gate)
            self.assertEqual(loaded["expert_contributions_source"], "native_unpatched_model_expert_execution")

    def test_formal_capability_rejects_dev_sized_one_trial_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            unhashed.update(
                {
                    "mode": "formal",
                    "status": "CAPABILITY_ONLY",
                    "signoff_sha256": "d" * 64,
                    "data_producer_signoff_sha256": "e" * 64,
                }
            )
            formal = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(formal), encoding="utf-8"
            )
            gpu_identity = {"gpu_uuid": "fixture"}
            with patch(
                "ric.run_experiment.validate_gpu_environment_artifact",
                return_value=gpu_identity,
            ):
                with self.assertRaisesRegex(
                    ExperimentRunnerError, "trial count.*30"
                ):
                    validate_capability_artifact(
                        root,
                        model_key="olmoe",
                        model_revision=str(formal["model_revision"]),
                        mode="formal",
                        config_sha256=sha256_file(DEFAULT_CONFIG),
                        protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                        data_manifest_sha256="8" * 64,
                        model_tree_manifest_sha256="9" * 64,
                        config=CONFIG,
                        data_producer_signoff_sha256="e" * 64,
                        gpu_environment_identity=gpu_identity,
                    )

    def test_capability_event_gate_blocks_barrier_release_before_both_tasks_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            trace_path = root / "capability_action_trace.jsonl"
            actions = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            for row in actions:
                if (
                    row["policy"] == "candidate_closing_first"
                    and row["release_mode"] == "full_layer_barrier"
                    and row["task_name"] == "y_nonclosing"
                ):
                    row["service_end_ts_us"] = 5.5
            trace_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in actions),
                encoding="utf-8",
            )
            failure = (
                "trial=0/policy=candidate_closing_first/barrier_early_release"
            )
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            unhashed["capability_action_trace_sha256"] = sha256_file(trace_path)
            summary = dict(unhashed["summary"])
            summary.update(
                {
                    "event_precedence_all_trials_pass": False,
                    "event_precedence_failure_count": 1,
                    "event_precedence_failures": [failure],
                }
            )
            unhashed["summary"] = summary
            tampered = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            _loaded, gate = validate_capability_artifact(
                root,
                model_key="olmoe",
                model_revision=str(tampered["model_revision"]),
                mode="dev",
                config_sha256=sha256_file(DEFAULT_CONFIG),
                protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                data_manifest_sha256="8" * 64,
                model_tree_manifest_sha256="9" * 64,
                config=CONFIG,
            )
            self.assertFalse(gate)

    def test_capability_event_gate_blocks_candidate_downstream_after_nonclosing_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            raw_path = root / "capability_raw.csv"
            with raw_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if (
                    row["policy"] == "candidate_closing_first"
                    and row["release_mode"] == "streaming"
                ):
                    row["downstream_start_us"] = "3.1"
            raw_path.unlink()
            _csv(raw_path, rows)
            failure = (
                "trial=0/policy=candidate_closing_first/nonclosing_before_early_use"
            )
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            unhashed["raw_trials_sha256"] = sha256_file(raw_path)
            summary = dict(unhashed["summary"])
            summary.update(
                {
                    "streaming_downstream_advance_us": 0.4,
                    "streaming_downstream_paired_mean_us": 0.4,
                    "streaming_downstream_paired_lcb_us": 0.4,
                    "downstream_interaction_paired_mean_us": 0.4,
                    "downstream_interaction_paired_lcb_us": 0.4,
                    "event_precedence_all_trials_pass": False,
                    "event_precedence_failure_count": 1,
                    "event_precedence_failures": [failure],
                }
            )
            unhashed["summary"] = summary
            tampered = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            _loaded, gate = validate_capability_artifact(
                root,
                model_key="olmoe",
                model_revision=str(tampered["model_revision"]),
                mode="dev",
                config_sha256=sha256_file(DEFAULT_CONFIG),
                protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                data_manifest_sha256="8" * 64,
                model_tree_manifest_sha256="9" * 64,
                config=CONFIG,
            )
            self.assertFalse(gate)

    def test_capability_event_gate_binds_frontier_to_closing_service(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            raw_path = root / "capability_raw.csv"
            with raw_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if (
                    row["policy"] == "candidate_closing_first"
                    and row["release_mode"] == "streaming"
                ):
                    row["physical_frontier_us"] = "0.1"
            raw_path.unlink()
            _csv(raw_path, rows)
            failure = (
                "trial=0/policy=candidate_closing_first/"
                "streaming_frontier_outside_closing"
            )
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            unhashed["raw_trials_sha256"] = sha256_file(raw_path)
            summary = dict(unhashed["summary"])
            summary.update(
                {
                    "event_precedence_all_trials_pass": False,
                    "event_precedence_failure_count": 1,
                    "event_precedence_failures": [failure],
                }
            )
            unhashed["summary"] = summary
            tampered = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            _loaded, gate = validate_capability_artifact(
                root,
                model_key="olmoe",
                model_revision=str(tampered["model_revision"]),
                mode="dev",
                config_sha256=sha256_file(DEFAULT_CONFIG),
                protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                data_manifest_sha256="8" * 64,
                model_tree_manifest_sha256="9" * 64,
                config=CONFIG,
            )
            self.assertFalse(gate)

    def test_capability_rejects_self_attested_interaction_lcb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            summary = dict(unhashed["summary"])
            summary["release_interaction_paired_lcb_us"] = 999.0
            unhashed["summary"] = summary
            tampered = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ExperimentRunnerError, "paired gate.*raw/config-derived"
            ):
                validate_capability_artifact(
                    root,
                    model_key="olmoe",
                    model_revision=str(tampered["model_revision"]),
                    mode="dev",
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    data_manifest_sha256="8" * 64,
                    model_tree_manifest_sha256="9" * 64,
                    config=CONFIG,
                )

    def test_capability_profiler_rejects_default_stream_gpu_activity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            trace_path = root / "capability_cuda_trace_streaming.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace["traceEvents"].append(
                {
                    "name": "unreviewedDefaultStreamKernel",
                    "cat": "kernel",
                    "args": {"stream": 7},
                }
            )
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            diagnostics = {
                key: dict(value)
                for key, value in unhashed["profiler_diagnostics"].items()
            }
            diagnostics["streaming"]["trace_sha256"] = sha256_file(trace_path)
            unhashed["profiler_diagnostics"] = diagnostics
            tampered = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ExperimentRunnerError, "mixed/missing GPU stream"
            ):
                validate_capability_artifact(
                    root,
                    model_key="olmoe",
                    model_revision=str(tampered["model_revision"]),
                    mode="dev",
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    data_manifest_sha256="8" * 64,
                    model_tree_manifest_sha256="9" * 64,
                    config=CONFIG,
                )

    def test_capability_rejects_zero_default_stream_pointer_even_if_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            raw_path = root / "capability_raw.csv"
            with raw_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["stream_id"] = "0"
            raw_path.unlink()
            _csv(raw_path, rows)
            action_path = root / "capability_action_trace.jsonl"
            actions = [
                json.loads(line)
                for line in action_path.read_text(encoding="utf-8").splitlines()
            ]
            for row in actions:
                row["stream_id"] = 0
            action_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in actions),
                encoding="utf-8",
            )
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            unhashed["sender_local_stream_id"] = 0
            unhashed["raw_trials_sha256"] = sha256_file(raw_path)
            unhashed["capability_action_trace_sha256"] = sha256_file(action_path)
            diagnostics = {
                key: {**value, "sender_local_stream_id": 0}
                for key, value in unhashed["profiler_diagnostics"].items()
            }
            unhashed["profiler_diagnostics"] = diagnostics
            tampered = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ExperimentRunnerError,
                "reuse one CUDA stream|persistent stream|action provenance",
            ):
                validate_capability_artifact(
                    root,
                    model_key="olmoe",
                    model_revision=str(tampered["model_revision"]),
                    mode="dev",
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    data_manifest_sha256="8" * 64,
                    model_tree_manifest_sha256="9" * 64,
                    config=CONFIG,
                )

    def test_capability_rejects_self_attested_canonical_equal_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            raw_path = root / "capability_raw.csv"
            with raw_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["canonical_output_sha256"] = "c" * 64
            rows[0]["canonical_equal"] = "True"
            raw_path.unlink()
            _csv(raw_path, rows)
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            unhashed["raw_trials_sha256"] = sha256_file(raw_path)
            tampered = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ExperimentRunnerError, "canonical|row provenance"
            ):
                validate_capability_artifact(
                    root,
                    model_key="olmoe",
                    model_revision=str(tampered["model_revision"]),
                    mode="dev",
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    data_manifest_sha256="8" * 64,
                    model_tree_manifest_sha256="9" * 64,
                    config=CONFIG,
                )

    def test_capability_rejects_unbound_producer_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            unhashed["measure_capability_source_sha256"] = "0" * 64
            tampered = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ExperimentRunnerError, "measure_capability_source_sha256"
            ):
                validate_capability_artifact(
                    root,
                    model_key="olmoe",
                    model_revision=str(tampered["model_revision"]),
                    mode="dev",
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    data_manifest_sha256="8" * 64,
                    model_tree_manifest_sha256="9" * 64,
                    config=CONFIG,
                )

    def test_capability_rejects_superseded_v4_single_profiler_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            diagnostics = unhashed.pop("profiler_diagnostics")
            streaming = diagnostics["streaming"]
            unhashed.update(
                {
                    "profiler_diagnostic_trial": streaming["trial"],
                    "profiler_required_labels": streaming["required_labels"],
                    "capability_cuda_trace_sha256": streaming["trace_sha256"],
                }
            )
            legacy = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(legacy), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ExperimentRunnerError, "profiler trace binding"
            ):
                validate_capability_artifact(
                    root,
                    model_key="olmoe",
                    model_revision=str(legacy["model_revision"]),
                    mode="dev",
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    data_manifest_sha256="8" * 64,
                    model_tree_manifest_sha256="9" * 64,
                    config=CONFIG,
                )

    def test_capability_rejects_mixed_sender_block_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root, mixed_sender=True)
            with self.assertRaisesRegex(
                ExperimentRunnerError, "sender-local|owned by"
            ):
                validate_capability_artifact(
                    root,
                    model_key="olmoe",
                    model_revision=str(artifact["model_revision"]),
                    mode="dev",
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    data_manifest_sha256="8" * 64,
                    model_tree_manifest_sha256="9" * 64,
                    config=CONFIG,
                )

    def test_capability_rejects_queue_snapshot_without_both_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root)
            trace_path = root / "capability_action_trace.jsonl"
            actions = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            actions[0]["queue_snapshot"]["ready_count"] = 1
            trace_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in actions),
                encoding="utf-8",
            )
            unhashed = dict(artifact)
            unhashed.pop("manifest_sha256")
            unhashed["capability_action_trace_sha256"] = sha256_file(trace_path)
            tampered = add_self_hash(unhashed)
            (root / "capability_probe.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ExperimentRunnerError, "action provenance|snapshot"
            ):
                validate_capability_artifact(
                    root,
                    model_key="olmoe",
                    model_revision=str(tampered["model_revision"]),
                    mode="dev",
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    data_manifest_sha256="8" * 64,
                    model_tree_manifest_sha256="9" * 64,
                    config=CONFIG,
                )

    def test_capability_rejects_row_level_revision_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _capability_fixture(root, bad_row_revision=True)
            with self.assertRaisesRegex(ExperimentRunnerError, "row provenance"):
                validate_capability_artifact(
                    root,
                    model_key="olmoe",
                    model_revision=str(artifact["model_revision"]),
                    mode="dev",
                    config_sha256=sha256_file(DEFAULT_CONFIG),
                    protocol_sha256=sha256_file(DEFAULT_PROTOCOL),
                    data_manifest_sha256="8" * 64,
                    model_tree_manifest_sha256="9" * 64,
                    config=CONFIG,
                )


class ReplayRunnerTests(unittest.TestCase):
    def test_persisted_accounting_roundtrip_and_tamper_gate(self) -> None:
        _inputs, world, _metadata = _fixture_inputs()
        result = run_replay(world, arm="sender_fcfs")
        row = _accounting_row(result, closure_budget_us=100.0)
        required = {
            "completed_join_count",
            "expected_join_count",
            "contract_received_bytes",
            "contract_header_bytes",
            "contract_record_bytes",
            "contract_alignment_bytes",
            "score_mask_fingerprint",
            "resource_demand_fingerprint",
            "queue_busy_us",
            "resource_service_demand_us",
            "source_by_field",
        }
        self.assertTrue(required <= set(row))
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.csv"
            write_experiment_csv(valid, [row])
            _validate_persisted_accounting(valid, expected_rows=1)
            tampered_row = dict(row)
            tampered_row["contract_received_bytes"] = int(row["contract_bytes"]) + 1
            tampered = Path(directory) / "tampered.csv"
            write_experiment_csv(tampered, [tampered_row])
            with self.assertRaisesRegex(
                ExperimentRunnerError, "accounting conservation"
            ):
                _validate_persisted_accounting(tampered, expected_rows=1)

    def test_trace_bundle_rejects_wrong_measured_drr_quantum(self) -> None:
        inputs, world, metadata = _fixture_inputs()
        with tempfile.TemporaryDirectory() as directory:
            tree = write_scenarios(
                output_dir=Path(directory) / "scenario",
                inputs=inputs,
                worlds=(world,),
                world_metadata=(metadata,),
                config_path=DEFAULT_CONFIG,
                protocol_path=DEFAULT_PROTOCOL,
                mode="dev",
                link_gbps=200,
            )
            wrong = ReplayConfig(
                starvation_us=1000.0,
                drr_quantum_us=float(tree["service_surface"]["sender_pack_us"])
                + 1.0,
                contract_tax_surface=contract_tax_surface_from_tree(tree),
            )
            with self.assertRaisesRegex(
                ExperimentRunnerError, "DRR quantum/fingerprint"
            ):
                execute_trace_bundles(
                    (world,),
                    arms=("sender_age_service_drr",),
                    configs_by_model={world.model_key: wrong},
                    workers=1,
                )

    def test_compaction_preserves_metrics_and_sender_signature(self) -> None:
        inputs, world, metadata = _fixture_inputs()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scenario"
            tree = write_scenarios(
                output_dir=output,
                inputs=inputs,
                worlds=(world,),
                world_metadata=(metadata,),
                config_path=DEFAULT_CONFIG,
                protocol_path=DEFAULT_PROTOCOL,
                mode="dev",
                link_gbps=200,
            )
            config = ReplayConfig(
                starvation_us=1000.0,
                contract_tax_surface=contract_tax_surface_from_tree(tree),
            )
            full = run_replay(world, arm="sender_fcfs", config=config)
            compact = compact_replay_result(full)
            before = trace_metrics_from_result(full, closure_budget_us=100.0)
            after = trace_metrics_from_result(compact, closure_budget_us=100.0)
            self.assertEqual(before, after)
            self.assertEqual(action_signature(full), action_signature(compact))
            self.assertEqual(
                {row.stage for row in compact.action_trace}, {"sender_egress"}
            )
            self.assertFalse(compact.completion_by_task_us)

    def test_trace_process_pool_path_keeps_seed_and_all_arms(self) -> None:
        inputs, world, metadata = _fixture_inputs()
        with tempfile.TemporaryDirectory() as directory:
            tree = write_scenarios(
                output_dir=Path(directory) / "scenario",
                inputs=inputs,
                worlds=(world,),
                world_metadata=(metadata,),
                config_path=DEFAULT_CONFIG,
                protocol_path=DEFAULT_PROTOCOL,
                mode="dev",
                link_gbps=200,
            )
            config = ReplayConfig(
                starvation_us=1000.0,
                contract_tax_surface=contract_tax_surface_from_tree(tree),
            )
            results = execute_trace_bundles(
                (world,),
                arms=("sender_fcfs", "ric_full_zero_delay"),
                configs_by_model={world.model_key: config},
                workers=1,
            )
            self.assertEqual({row.arm for row in results}, {"sender_fcfs", "ric_full_zero_delay"})
            self.assertEqual({row.workload_seed for row in results}, {world.workload_seed})
            self.assertTrue(all(row.full_drain for row in results))

    def test_worker_resolution_is_deterministic_and_bounded(self) -> None:
        self.assertEqual(resolve_worker_count(99, 3), 3)
        self.assertGreaterEqual(resolve_worker_count(0, 3), 1)
        with self.assertRaises(ExperimentRunnerError):
            resolve_worker_count(-1, 3)


class LockAndSealedDisciplineTests(unittest.TestCase):
    def test_oracle_instance_scheduling_contract_rejects_mutation(self) -> None:
        lock = {"models": {"olmoe": {"starvation_us": 123.0}}}
        row = {
            "model_key": "olmoe",
            "starvation_us": 123.0,
            "downstream_service_discipline": "work_conserving_fcfs_no_overtake",
            "worlds": [{"world_name": "a"}, {"world_name": "b"}],
            "observation_history_nodes": {
                "S": {"a": 0, "b": 0},
                "B": {"a": 0, "b": 0},
                "R0": {"a": 0, "b": 1},
                "C": {"a": 0, "b": 1},
            },
        }
        validate_oracle_instance_scheduling_contract(row, lock)
        for field, value in (
            ("starvation_us", 122.0),
            ("downstream_service_discipline", "reorder_allowed"),
            ("observation_history_nodes", {"S": {"a": 0}}),
        ):
            with self.subTest(field=field):
                mutated = dict(row)
                mutated[field] = value
                with self.assertRaisesRegex(
                    ExperimentRunnerError, "scheduling contract/lock mismatch"
                ):
                    validate_oracle_instance_scheduling_contract(mutated, lock)

    def test_formal_calibration_signoff_binds_scenarios_and_capabilities(self) -> None:
        models = ("olmoe", "qwen")
        trees = {
            "olmoe": {
                "manifest_sha256": "scenario-o",
                "scenario_producer_signoff_sha256": "scenario-signoff-o",
            },
            "qwen": {
                "manifest_sha256": "scenario-q",
                "scenario_producer_signoff_sha256": "scenario-signoff-q",
            },
        }
        capabilities = {
            "olmoe": {
                "manifest_sha256": "capability-o",
                "signoff_sha256": "producer-signoff-o",
            },
            "qwen": {
                "manifest_sha256": "capability-q",
                "signoff_sha256": "producer-signoff-q",
            },
        }
        signoff = {
            "scenario_tree_sha256": {
                "olmoe": "scenario-o",
                "qwen": "scenario-q",
            },
            "capability_probe_sha256": {
                "olmoe": "capability-o",
                "qwen": "capability-q",
            },
            "scenario_producer_signoff_sha256": {
                "olmoe": "scenario-signoff-o",
                "qwen": "scenario-signoff-q",
            },
            "capability_producer_signoff_sha256": {
                "olmoe": "producer-signoff-o",
                "qwen": "producer-signoff-q",
            },
        }
        validate_calibration_signoff_bindings(
            signoff=signoff,
            trees=trees,
            capabilities=capabilities,
            required_models=models,
        )
        tampered = dict(signoff)
        tampered["capability_probe_sha256"] = {
            "olmoe": "capability-o",
            "qwen": "different",
        }
        with self.assertRaisesRegex(ExperimentRunnerError, "capability inputs"):
            validate_calibration_signoff_bindings(
                signoff=tampered,
                trees=trees,
                capabilities=capabilities,
                required_models=models,
            )
        tampered_signoff = dict(signoff)
        tampered_signoff["capability_producer_signoff_sha256"] = {
            "olmoe": "producer-signoff-o",
            "qwen": "different",
        }
        with self.assertRaisesRegex(
            ExperimentRunnerError, "producer signoffs"
        ):
            validate_calibration_signoff_bindings(
                signoff=tampered_signoff,
                trees=trees,
                capabilities=capabilities,
                required_models=models,
            )

    def test_amendment_e_uses_mean_trace_cvar_not_pooled_tail(self) -> None:
        frozen = tuple(
            arm for arm in CONFIG["joinblind_arms"] if arm != "calib_best_joinblind"
        )
        first, second = frozen[:2]
        rows = {
            arm: (
                SimpleNamespace(scored_join_latencies_us={"x": 20.0}),
                SimpleNamespace(scored_join_latencies_us={"x": 20.0}),
            )
            for arm in CONCRETE_JOINBLIND_ARMS
        }
        rows[first] = (
            SimpleNamespace(scored_join_latencies_us={"x": 1.0}),
            SimpleNamespace(scored_join_latencies_us={"x": 9.0}),
        )
        rows[second] = (
            SimpleNamespace(scored_join_latencies_us={"x": 6.0}),
            SimpleNamespace(scored_join_latencies_us={"x": 6.0}),
        )
        selected = select_calibration_baseline(rows, frozen_arm_order=frozen)
        self.assertEqual(selected["calib_best_joinblind"], first)
        self.assertEqual(selected["closure_budget_us"], 9.0)
        self.assertFalse(selected["selection_used_violation"])

    def test_exact_tie_uses_frozen_arm_order(self) -> None:
        frozen = tuple(
            arm for arm in CONFIG["joinblind_arms"] if arm != "calib_best_joinblind"
        )
        rows = {
            arm: (SimpleNamespace(scored_join_latencies_us={"x": 5.0}),)
            for arm in CONCRETE_JOINBLIND_ARMS
        }
        selected = select_calibration_baseline(rows, frozen_arm_order=frozen)
        self.assertEqual(selected["calib_best_joinblind"], frozen[0])

    def test_any_nonpositive_retention_denominator_is_valid_gate_fail(self) -> None:
        rows = []
        # Point headroom is positive, but resampling trace b alone is negative.
        for trace, seed, baseline, r0, charged in (
            ("a", 1, 20.0, 10.0, 15.0),
            ("b", 2, 5.0, 6.0, 5.5),
        ):
            rows.extend(
                (
                    _trace_metric(trace, seed, "calib_best_joinblind", baseline),
                    _trace_metric(trace, seed, "ric_full_zero_delay", r0),
                    _trace_metric(trace, seed, "ric_wire_charged", charged),
                )
            )
        summary, reason = strict_retention_bootstrap(
            rows, n_bootstrap=100, confidence=0.9, seed=3
        )
        self.assertIsNone(summary)
        self.assertEqual(reason, "NONPOSITIVE_BOOTSTRAP_INFORMATION_HEADROOM")

    def test_sealed_ledger_is_atomic_one_shot_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.json"
            record = reserve_sealed_consumption(
                path,
                mode="dev",
                output_dir=Path(directory) / "output",
                nonce="nonce-1",
                config_sha256="c",
                protocol_sha256="p",
                scenario_tree_sha256={"olmoe/link200": "s"},
                scenario_producer_signoff_sha256={"olmoe/link200": "ss"},
                oracle_status_sha256="o",
                oracle_producer_signoff_sha256="os",
            )
            validate_self_hash(record)
            self.assertEqual(record["scenario_tree_sha256"], {"olmoe/link200": "s"})
            with self.assertRaisesRegex(ExperimentRunnerError, "already consumed"):
                reserve_sealed_consumption(
                    path,
                    mode="dev",
                    output_dir=Path(directory) / "output",
                    nonce="nonce-2",
                    config_sha256="c",
                    protocol_sha256="p",
                    scenario_tree_sha256={"olmoe/link200": "s"},
                    scenario_producer_signoff_sha256={"olmoe/link200": "ss"},
                    oracle_status_sha256="o",
                    oracle_producer_signoff_sha256="os",
                )

    def test_formal_sealed_ledger_rejects_alternate_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixed = root / "state/evaluation_consumption.json"
            alternate = root / "state/alternate.json"
            with patch.dict(
                reserve_sealed_consumption.__globals__,
                {"GLOBAL_SEALED_EVALUATION_CONSUMPTION": fixed},
            ), self.assertRaisesRegex(
                ExperimentRunnerError, "reviewed global ledger"
            ):
                reserve_sealed_consumption(
                    alternate,
                    mode="formal",
                    output_dir=root / "output",
                    nonce="nonce-1",
                    config_sha256="c",
                    protocol_sha256="p",
                    scenario_tree_sha256={"olmoe/link200": "s"},
                    scenario_producer_signoff_sha256={"olmoe/link200": "ss"},
                    oracle_status_sha256="o",
                    oracle_producer_signoff_sha256="os",
                )

    def test_sealed_ledger_handles_short_writes_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.json"
            real_write = os.write

            def short_write(descriptor: int, data: bytes) -> int:
                return real_write(descriptor, data[: max(1, len(data) // 2)])

            with patch.object(os, "write", side_effect=short_write):
                record = reserve_sealed_consumption(
                    path,
                    mode="dev",
                    output_dir=Path(directory) / "output",
                    nonce="nonce-short",
                    config_sha256="c",
                    protocol_sha256="p",
                    scenario_tree_sha256={"olmoe/link200": "s"},
                    scenario_producer_signoff_sha256={"olmoe/link200": "ss"},
                    oracle_status_sha256="o",
                    oracle_producer_signoff_sha256="os",
                )
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), record
            )

    def test_formal_evaluate_api_reserves_one_shot_internally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            ledger = root / "state/evaluation_consumption.json"
            ledger.parent.mkdir()
            output = root / "formal_outputs/sealed"
            lock = {"manifest_sha256": "l" * 64}
            oracle = {
                "manifest_sha256": "o" * 64,
                "run_oracle_source_sha256": "r" * 64,
                "signoff_sha256": "s" * 64,
            }
            signoff = {
                "calibration_lock_sha256": lock["manifest_sha256"],
                "oracle_status_sha256": oracle["manifest_sha256"],
                "run_oracle_source_sha256": oracle["run_oracle_source_sha256"],
                "oracle_producer_signoff_sha256": oracle["signoff_sha256"],
                "sealed_consumption_ledger_path": str(ledger),
                "sealed_output_dir": str(output),
                "sealed_nonce": "nonce-api",
                "scenario_tree_sha256": {"olmoe/link200": "t" * 64},
                "scenario_producer_signoff_sha256": {
                    "olmoe/link200": "p" * 64
                },
            }
            replacements = {
                "GLOBAL_SEALED_EVALUATION_CONSUMPTION": ledger,
                "validate_formal_output_path": lambda *args, **kwargs: output,
                "validate_frozen_formal_paths": lambda *args, **kwargs: None,
                "_load_signoff": lambda *args, **kwargs: signoff,
                "_load_config": lambda *args, **kwargs: {},
                "_load_lock_and_oracle": lambda *args, **kwargs: (lock, oracle),
                "_scenario_grid": Mock(
                    side_effect=ExperimentRunnerError("stop after reservation")
                ),
            }
            kwargs = {
                "scenario_dirs": (),
                "calibration_lock_path": root / "lock.json",
                "oracle_dir": root / "oracle",
                "output_dir": output,
                "mode": "formal",
                "workers": 1,
                "config_path": DEFAULT_CONFIG,
                "protocol_path": DEFAULT_PROTOCOL,
                "signoff_path": root / "signoff.json",
            }
            with patch.dict(evaluate_pipeline.__globals__, replacements):
                output.mkdir(parents=True)
                with self.assertRaisesRegex(
                    ExperimentRunnerError, "existing output directory"
                ):
                    evaluate_pipeline(**kwargs)
                self.assertFalse(ledger.exists())
                output.rmdir()
                with self.assertRaisesRegex(
                    ExperimentRunnerError, "stop after reservation"
                ):
                    evaluate_pipeline(**kwargs)
                self.assertTrue(ledger.is_file())
                with self.assertRaisesRegex(
                    ExperimentRunnerError, "already consumed"
                ):
                    evaluate_pipeline(**kwargs)

    def test_g3_requires_complete_delay_and_link_lcb_grid(self) -> None:
        cell_keys = {
            f"{model}/{cell}"
            for model in CONFIG["go_no_go"]["required_models"]
            for cell in CONFIG["go_no_go"]["required_main_cells"]
        }
        details = {
            key: {"delay_sensitivity": {name: {} for name in ("0", "5", "20", "50")}}
            for key in cell_keys
        }
        links = {
            f"{key}/link{int(link)}": True
            for key in cell_keys
            for link in CONFIG["topology_proxy"]["link_sensitivity_gbps"]
        }
        validate_g3_sensitivity_grid(
            g3_details=details, link_gates=links, config=CONFIG
        )
        broken = dict(details)
        first = next(iter(broken))
        broken[first] = {"delay_sensitivity": {"0": {}, "5": {}, "20": {}}}
        with self.assertRaisesRegex(ExperimentRunnerError, "0/5/20/50"):
            validate_g3_sensitivity_grid(
                g3_details=broken, link_gates=links, config=CONFIG
            )

    def test_atomic_directory_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "artifact"
            with atomic_output_directory(target) as temporary:
                (temporary / "value").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ScenarioBuildError, "overwrite"):
                with atomic_output_directory(target):
                    pass


if __name__ == "__main__":
    unittest.main()
