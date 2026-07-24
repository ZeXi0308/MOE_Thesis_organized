from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cjc_policy import (
    ACK_HEADER_BYTES,
    ACK_RECORD_BYTES,
    JOIN_BLIND_ARMS,
    AckConfig,
    AckWireRecord,
    CJCValidationError,
    EpisodeMetrics,
    JoinBlindTask,
    LUTPoint,
    PlacementManifest,
    RouteContribution,
    ServiceLUT,
    Task,
    WorkloadSpec,
    ack_message_bytes,
    assert_arm_equivalence,
    build_tasks_from_routes,
    canonical_reduction_signature,
    choose_global_causal_join,
    choose_join_blind,
    decode_ack_message,
    encode_ack_message,
    episode_metrics,
    paired_hierarchical_bootstrap,
    simulate,
    to_join_blind,
    validate_route_contributions,
)
from run_cjc_oracle import (
    load_routes,
    select_replay_routes,
    sha256_file,
    validate_environment,
    validate_signoff,
)


REVISION = "revision-a"
DATA_SHA = "d" * 64
PLACEMENT_SHA = "p" * 64


def placement(requests: tuple[str, ...] = ("r0", "r1")) -> PlacementManifest:
    return PlacementManifest(
        sha256=PLACEMENT_SHA,
        ep_size=4,
        gpus_per_node=2,
        expert_to_sender={(REVISION, 0): 0, (REVISION, 1): 1, (REVISION, 2): 2, (REVISION, 3): 3},
        request_to_receiver={request_id: 0 for request_id in requests},
    )


def route_rows(request_id: str = "r0", token_count: int = 2) -> list[RouteContribution]:
    rows: list[RouteContribution] = []
    for token_position in range(token_count):
        for slot, expert in enumerate((0, 1)):
            rows.append(
                RouteContribution(
                    schema_version="cjc-route-v1",
                    model_revision=REVISION,
                    data_manifest_sha256=DATA_SHA,
                    request_id=request_id,
                    forward_id=f"f-{request_id}",
                    batch_id="0",
                    phase="prefill",
                    decode_step=0,
                    layer_id=0,
                    token_id=f"{request_id}-t{token_position}",
                    token_position=token_position,
                    topk_slot=slot,
                    expert_id=expert,
                    sender_rank=expert,
                    receiver_rank=0,
                    valid=True,
                    route_weight=0.5,
                    route_source="native_model_forward",
                    placement_manifest_sha256=PLACEMENT_SHA,
                )
            )
    return rows


def task(
    name: str,
    join: str,
    slot: int,
    *,
    ready: float = 0.0,
    service: float = 1.0,
    receiver: int = 0,
    resource: str = "node0:combine_ingress",
) -> Task:
    return Task(
        task_id=name,
        route_key=("r0", "f0", 0, join, slot),
        join_key=("r0", "f0", 0, join),
        episode_id="r0",
        model_revision=REVISION,
        cell="steady_rho50",
        seed=2026072201,
        layer_id=0,
        token_position=0 if join == "a" else 1,
        topk_slot=slot,
        expert_id=slot,
        sender_rank=slot,
        receiver_rank=receiver,
        resource_id=resource,
        release_us=0.0,
        ready_us=ready,
        service_us=service,
        deadline_us=100.0,
        payload_bytes=32,
        descriptor_bytes=16,
        alignment_bytes=0,
    )


def four_tasks() -> list[Task]:
    return [
        task("a0", "a", 0),
        task("a1", "a", 1),
        task("b0", "b", 0),
        task("b1", "b", 1),
    ]


class IdentityTests(unittest.TestCase):
    def test_replay_selection_preserves_all_siblings(self) -> None:
        routes: list[RouteContribution] = []
        for layer in range(3):
            for row in route_rows(token_count=4):
                routes.append(replace(row, layer_id=layer))
        selected = select_replay_routes(
            routes,
            {
                "method": "smallest_sha256_of_seed_and_identity",
                "seed": 20260722,
                "layers_per_model": 2,
                "token_positions_per_request_layer": 2,
                "preserve_all_topk_siblings": True,
            },
        )
        self.assertEqual(len(selected), 2 * 2 * 2)
        by_join: dict[tuple[object, ...], int] = {}
        for route in selected:
            by_join[route.join_key] = by_join.get(route.join_key, 0) + 1
        self.assertEqual(set(by_join.values()), {2})

    def test_identity_complete_topk_closure(self) -> None:
        rows = route_rows()
        validate_route_contributions(
            rows,
            expected_model_revision=REVISION,
            top_k=2,
            num_experts=4,
            placement=placement(("r0",)),
            expected_data_manifest_sha256=DATA_SHA,
            formal=True,
        )

    def test_duplicate_forward_token_slot_hard_fails(self) -> None:
        rows = route_rows()
        with self.assertRaisesRegex(CJCValidationError, "duplicate route identity"):
            validate_route_contributions(
                rows + [rows[0]],
                expected_model_revision=REVISION,
                top_k=2,
                num_experts=4,
                placement=placement(("r0",)),
                formal=True,
            )

    def test_missing_slot_and_padding_hard_fail(self) -> None:
        rows = route_rows()
        with self.assertRaises(CJCValidationError):
            validate_route_contributions(
                rows[:-1],
                expected_model_revision=REVISION,
                top_k=2,
                num_experts=4,
                placement=placement(("r0",)),
            )
        invalid = list(rows)
        invalid[0] = replace(invalid[0], valid=False)
        with self.assertRaisesRegex(CJCValidationError, "padding/drop"):
            validate_route_contributions(
                invalid,
                expected_model_revision=REVISION,
                top_k=2,
                num_experts=4,
                placement=placement(("r0",)),
            )

    def test_sender_and_receiver_must_come_from_manifests(self) -> None:
        rows = route_rows()
        wrong_sender = list(rows)
        wrong_sender[0] = replace(wrong_sender[0], sender_rank=3)
        with self.assertRaisesRegex(CJCValidationError, "placement-manifest owner"):
            validate_route_contributions(
                wrong_sender,
                expected_model_revision=REVISION,
                top_k=2,
                num_experts=4,
                placement=placement(("r0",)),
            )
        wrong_receiver = list(rows)
        wrong_receiver[0] = replace(wrong_receiver[0], receiver_rank=1)
        with self.assertRaisesRegex(CJCValidationError, "origin-manifest owner"):
            validate_route_contributions(
                wrong_receiver,
                expected_model_revision=REVISION,
                top_k=2,
                num_experts=4,
                placement=placement(("r0",)),
            )

    def test_topk_slot_is_not_sender_rank(self) -> None:
        rows = route_rows()
        # slot=0 legitimately maps to sender 1 after swapping placement; the
        # validator follows expert ownership, never interprets slot as rank.
        swapped = PlacementManifest(
            sha256=PLACEMENT_SHA,
            ep_size=4,
            gpus_per_node=2,
            expert_to_sender={(REVISION, 0): 1, (REVISION, 1): 0, (REVISION, 2): 2, (REVISION, 3): 3},
            request_to_receiver={"r0": 0},
        )
        remapped = [replace(row, sender_rank=swapped.expert_to_sender[(row.model_revision, row.expert_id)]) for row in rows]
        validate_route_contributions(
            remapped,
            expected_model_revision=REVISION,
            top_k=2,
            num_experts=4,
            placement=swapped,
        )

    def test_legacy_csv_and_route_timing_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "routes.csv"
            csv_path.write_text("sample_id,layer,rank,expert_id\n", encoding="utf-8")
            with self.assertRaisesRegex(CJCValidationError, "legacy CSV"):
                load_routes([csv_path])
            jsonl_path = Path(directory) / "routes.jsonl"
            raw = route_rows(token_count=1)[0].__dict__.copy()
            raw["expert_ready_us"] = 1.0
            jsonl_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CJCValidationError, "must not inject"):
                load_routes([jsonl_path])


class PolicyTests(unittest.TestCase):
    def test_join_blind_view_has_no_missing_or_join_field(self) -> None:
        blind = to_join_blind(task("a0", "a", 0))
        self.assertNotIn("join", blind.__dataclass_fields__)
        self.assertNotIn("missing", blind.__dataclass_fields__)

    def test_future_mutation_cannot_change_current_causal_action(self) -> None:
        ready = [task("a0", "a", 0), task("b0", "b", 0)]
        current = {ready[0].join_key: 1, ready[1].join_key: 2}
        first = choose_global_causal_join(ready, now_us=0, visible_missing=current, fairness_debt={})
        # A future task/service exists but is not in the causal API.
        _future = task("future", "b", 1, ready=1000.0, service=999.0)
        second = choose_global_causal_join(ready, now_us=0, visible_missing=current, fairness_debt={})
        self.assertEqual(first, second)
        self.assertEqual(first, "a0")

    def test_indistinguishable_worlds_separate_information_value(self) -> None:
        blind = [
            JoinBlindTask("x", 0, 1, 48, 0, "q", 100, ("r", 0, "x", 0)),
            JoinBlindTask("y", 0, 1, 48, 0, "q", 100, ("r", 0, "y", 0)),
        ]
        blind_a = choose_join_blind(
            "fifo", blind, now_us=0, receiver_queue_depth={0: 2}, resource_backlog_us={"q": 2}
        )
        blind_b = choose_join_blind(
            "fifo", blind, now_us=0, receiver_queue_depth={0: 2}, resource_backlog_us={"q": 2}
        )
        x = task("x", "a", 0)
        y = task("y", "b", 0)
        aware_a = choose_global_causal_join(
            [x, y], now_us=0, visible_missing={x.join_key: 1, y.join_key: 2}, fairness_debt={}
        )
        aware_b = choose_global_causal_join(
            [x, y], now_us=0, visible_missing={x.join_key: 2, y.join_key: 1}, fairness_debt={}
        )
        self.assertEqual(blind_a, blind_b)
        self.assertNotEqual(aware_a, aware_b)

    def test_sync_token_order_is_stable(self) -> None:
        rows = [to_join_blind(task("z", "b", 0)), to_join_blind(task("a", "a", 0))]
        first = choose_join_blind(
            "sync_token_order", rows, now_us=0, receiver_queue_depth={0: 2}, resource_backlog_us={}
        )
        second = choose_join_blind(
            "sync_token_order", list(reversed(rows)), now_us=0, receiver_queue_depth={0: 2}, resource_backlog_us={}
        )
        self.assertEqual(first, second)


class AccountingAndSimulationTests(unittest.TestCase):
    def test_ack_wire_identity(self) -> None:
        self.assertEqual(ack_message_bytes(0), 0)
        self.assertEqual(ack_message_bytes(1), ACK_HEADER_BYTES + ACK_RECORD_BYTES)
        self.assertEqual(ack_message_bytes(2), 48)
        records = (
            AckWireRecord(1, 2, 3, 4),
            AckWireRecord(5, 6, 7, 8),
        )
        payload = encode_ack_message(records)
        self.assertEqual(len(payload), ack_message_bytes(2))
        self.assertEqual(decode_ack_message(payload), records)
        with self.assertRaises(CJCValidationError):
            decode_ack_message(payload[:-1])

    def test_nonpreemptive_full_drain_and_task_equivalence(self) -> None:
        tasks = four_tasks()
        fifo = simulate(tasks, arm="fifo")
        join = simulate(tasks, arm="global_causal_join", ack=AckConfig(enabled=False))
        assert_arm_equivalence([fifo, join], len(tasks))
        self.assertEqual(set(fifo.completion_by_task), {task.task_id for task in tasks})
        for left, right in zip(fifo.action_trace, fifo.action_trace[1:]):
            self.assertGreaterEqual(right.start_us, left.completion_us)
        self.assertEqual(fifo.data_bytes, join.data_bytes)

    def test_late_malformed_ack_falls_back_but_keeps_tax(self) -> None:
        tasks = four_tasks()
        result = simulate(
            tasks,
            arm="global_causal_join",
            ack=AckConfig(
                enabled=True,
                staleness_us=0,
                build_us=0.1,
                serialize_us=0.1,
                wire_us=0.1,
                parse_us=0.1,
                malformed_task_ids=frozenset({"a0", "a1", "b0", "b1"}),
            ),
        )
        self.assertGreater(result.fallback_count, 0)
        self.assertEqual(result.ack_bytes, len(tasks) * ack_message_bytes(1))
        self.assertAlmostEqual(result.coordination_charged_us, len(tasks) * 0.4)

    def test_no_overlap_evidence_charges_full_coordination_time(self) -> None:
        tasks = four_tasks()
        charged = simulate(
            tasks,
            arm="global_causal_join",
            ack=AckConfig(enabled=True, staleness_us=20, build_us=1, serialize_us=2, wire_us=3, parse_us=4),
        )
        self.assertEqual(charged.coordination_charged_us, len(tasks) * 10)
        self.assertGreater(charged.action_trace[-1].completion_us, len(tasks))
        self.assertGreater(charged.fallback_count, 0)

    def test_canonical_reduction_is_action_order_independent(self) -> None:
        rows = [(1, 5, b"right"), (0, 2, b"left")]
        self.assertEqual(
            canonical_reduction_signature(rows),
            canonical_reduction_signature(list(reversed(rows))),
        )

    def test_multiple_receiver_resources_drain_independently(self) -> None:
        tasks = four_tasks()
        other = [
            replace(
                task(f"c{slot}", "c", slot, receiver=2, resource="node1:combine_ingress"),
                route_key=("r0", "f0", 0, "c", slot),
                join_key=("r0", "f0", 0, "c"),
            )
            for slot in range(2)
        ]
        result = simulate(tasks + other, arm="fifo")
        self.assertEqual(len(result.completion_by_task), 6)


class LUTAndTaskBuildTests(unittest.TestCase):
    def lut(self) -> ServiceLUT:
        return ServiceLUT(
            [
                LUTPoint(REVISION, -1, 1, 1, 0.1, 0.1, 0.1, 0.1, "measured_same_gpu"),
                LUTPoint(REVISION, -1, 2, 2, 0.2, 0.2, 0.2, 0.2, "measured_same_gpu"),
                LUTPoint(REVISION, -1, 4, 3, 0.4, 0.4, 0.4, 0.4, "measured_same_gpu"),
            ]
        )

    def test_lut_interpolates_but_never_extrapolates(self) -> None:
        point = self.lut().lookup(REVISION, 0, 3)
        self.assertAlmostEqual(point.expert_us, 2.5)
        with self.assertRaisesRegex(CJCValidationError, "extrapolation forbidden"):
            self.lut().lookup(REVISION, 0, 5)

    def test_tasks_derive_timing_from_lut_and_workload(self) -> None:
        routes = route_rows(token_count=2)
        tasks = build_tasks_from_routes(
            routes,
            lut=self.lut(),
            placement=placement(("r0",)),
            workload=WorkloadSpec("steady_rho50", 0.1, 10, 100),
            seed=1,
            hidden_size=16,
            dtype_bytes=2,
            descriptor_bytes=16,
            alignment_bytes=0,
            link_gbps=200,
        )
        self.assertEqual(len(tasks), len(routes))
        self.assertTrue(all(task.time_source == "derived_from_measured_lut" for task in tasks))
        self.assertTrue(all(task.ready_us >= task.release_us for task in tasks))


class StatisticsAndFormalGateTests(unittest.TestCase):
    def test_episode_is_the_bootstrap_unit(self) -> None:
        rows: list[EpisodeMetrics] = []
        for episode in ("d0", "d1"):
            for seed in (1, 2):
                rows.extend(
                    [
                        EpisodeMetrics(REVISION, "steady_rho50", seed, episode, "fifo", (10.0, 12.0), 11.0),
                        EpisodeMetrics(REVISION, "steady_rho50", seed, episode, "join", (8.0, 9.0), 11.0),
                    ]
                )
        summary = paired_hierarchical_bootstrap(
            rows, candidate_arm="join", baseline_arms=["fifo"], n_bootstrap=50, seed=3
        )
        self.assertEqual(summary.n_episodes, 2)
        self.assertEqual(summary.n_seeds, 2)
        self.assertGreater(summary.p99_gain_ci_low, 0)

    def test_unpaired_arm_hard_fails(self) -> None:
        rows = [
            EpisodeMetrics(REVISION, "steady_rho50", 1, "d0", "fifo", (10.0,), 11.0)
        ]
        with self.assertRaisesRegex(CJCValidationError, "unpaired arms"):
            paired_hierarchical_bootstrap(
                rows, candidate_arm="join", baseline_arms=["fifo"], n_bootstrap=10
            )

    def test_full_barrier_capability_blocks_formal(self) -> None:
        config = {
            "formal": {
                "required_capabilities": [
                    "identity_complete_native_route_producer",
                    "ready_block_reorderable",
                    "token_block_early_release_effect",
                ]
            }
        }
        environment = {
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "h2d_boundary": "NOT_RDMA",
            "capabilities": {
                "identity_complete_native_route_producer": True,
                "ready_block_reorderable": True,
                "token_block_early_release_effect": False,
            },
        }
        with self.assertRaisesRegex(CJCValidationError, "capability missing"):
            validate_environment(environment, config)

    def test_signoff_hash_drift_hard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signoff.json"
            path.write_text(
                json.dumps({"status": "SIGNED-OFF", "protocol_sha256": "old"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CJCValidationError, "hash drift"):
                validate_signoff(str(path), bindings={"protocol_sha256": "new"})

    def test_episode_metrics_use_token_join_not_contributions(self) -> None:
        tasks = four_tasks()
        result = simulate(tasks, arm="fifo")
        metrics = episode_metrics(tasks, result, slo_us=100)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(len(metrics[0].token_latencies_us), 2)


class RunnerSmokeTests(unittest.TestCase):
    def test_dev_runner_is_runnable_and_cannot_emit_scientific_go(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            protocol_rel = "docs/ideas/receiver_aware/CJC_Phase2_冻结实验协议_2026-07-22.md"
            config = {
                "protocol_version": "cjc-v1",
                "protocol_path": protocol_rel,
                "evidence_boundary": "DEV_PROXY_NOT_RDMA_NOT_SERVING_P99",
                "required_models": {
                    "tiny": {
                        "revision": REVISION,
                        "top_k": 2,
                        "num_experts": 4,
                        "hidden_size": 16,
                    }
                },
                "topology": {
                    "ep_size": 4,
                    "gpus_per_node": 2,
                    "placement": "contiguous",
                    "link_gbps_primary": 200.0,
                    "link_gbps_sensitivity": [100.0, 200.0, 400.0],
                },
                "wire": {
                    "dtype": "bfloat16",
                    "dtype_bytes": 2,
                    "descriptor_bytes_per_contribution": 16,
                    "alignment_bytes_per_contribution": 0,
                    "host_staging_label": "NOT_RDMA",
                },
                "workload": {
                    "cells": ["steady_rho50"],
                    "main_seeds": [1],
                    "reserve_seeds": [],
                    "main_block_size": 1,
                    "sensitivity_block_size": 8,
                },
                "replay_selection": {
                    "method": "smallest_sha256_of_seed_and_identity",
                    "seed": 20260722,
                    "layers_per_model": 1,
                    "token_positions_per_request_layer": 1,
                    "preserve_all_topk_siblings": True,
                },
                "arms": {
                    "join_blind": list(JOIN_BLIND_ARMS),
                    "candidate": "global_causal_join",
                    "fallback": "topology_join_blind",
                },
                "ack": {
                    "header_bytes": 16,
                    "record_bytes": 16,
                    "alignment_bytes": 16,
                    "main_staleness_us": 5.0,
                    "sensitivity_staleness_us": [0.0, 20.0, 50.0],
                    "require_measured_build_serialize_parse": True,
                    "free_piggyback_can_support_go": False,
                },
                "statistics": {
                    "n_bootstrap": 10,
                    "bootstrap_seed": 2,
                    "unit": "document_request_episode_then_seed",
                    "p99_gain_lcb_gate": 0.05,
                    "violation_reduction_lcb_gate": 0.03,
                    "required_cells_per_model": 1,
                },
                "formal": {
                    "required_capabilities": [],
                    "require_phase4_signed_off": True,
                },
            }
            config_path = root / "cjc_v1.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            data = {
                "dataset": "wikitext/wikitext-103-raw-v1",
                "dataset_split": "train",
                "calibration_hashes": ["calib"],
                "sealed_hashes": ["sealed"],
                "historical_hashes": ["historical"],
                "calibration_manifest_sha256": "c" * 64,
                "sealed_manifest_sha256": "s" * 64,
                "sealed": False,
            }
            data_path = root / "data.json"
            data_path.write_text(json.dumps(data), encoding="utf-8")
            data_sha = sha256_file(data_path)

            placement_raw = {
                "schema_version": 1,
                "expert_to_sender_by_model": {
                    REVISION: {"0": 0, "1": 1, "2": 2, "3": 3}
                },
                "request_to_receiver": {"r0": 0},
            }
            placement_raw["manifest_sha256"] = __import__("hashlib").sha256(
                json.dumps(placement_raw, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            placement_path = root / "placement.json"
            placement_path.write_text(json.dumps(placement_raw), encoding="utf-8")
            placement_sha = placement_raw["manifest_sha256"]

            routes = []
            for row in route_rows(token_count=2):
                routes.append(
                    replace(
                        row,
                        data_manifest_sha256="s" * 64,
                        placement_manifest_sha256=placement_sha,
                    )
                )
            route_path = root / "routes.jsonl"
            route_path.write_text(
                "".join(json.dumps(row.__dict__) + "\n" for row in routes),
                encoding="utf-8",
            )

            lut_path = root / "lut.csv"
            lut_path.write_text(
                "model_revision,layer_id,rows,expert_us,pack_us,launch_us,host_staging_us,reduction_us,source\n"
                f"{REVISION},-1,1,1,0.1,0.1,0.1,0.1,measured_same_gpu\n"
                f"{REVISION},-1,2,2,0.2,0.2,0.2,0.2,measured_same_gpu\n",
                encoding="utf-8",
            )
            calibration = {
                "models": {
                    "tiny": {
                        "cells": {
                            "steady_rho50": {
                                "arrival_rate_per_us": 0.1,
                                "layer_period_us": 10.0,
                                "slo_us": 110.0,
                                "calibration_best_joinblind_p99_us": 100.0,
                                "slo_definition": "1.10_x_calibration_best_joinblind_p99",
                                "target_rho": 0.50,
                                "calib_best_static": "fifo",
                            }
                        },
                        "ack_timing": {
                            "schema_version": "cjc-ack-timing-v1",
                            "components": {
                                "build_us": {"value_us": 0.01, "source": "measured_same_run_host_monotonic_ns"},
                                "serialize_us": {"value_us": 0.01, "source": "measured_same_run_host_monotonic_ns"},
                                "wire_us": {"value_us": 0.01, "source": "analytic_link", "message_bytes": 32},
                                "parse_us": {"value_us": 0.01, "source": "measured_same_run_host_monotonic_ns"},
                                "policy_lookup_us": {"value_us": 0.01, "source": "measured_same_run_host_monotonic_ns"},
                            },
                        },
                    }
                }
            }
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
            environment_path = root / "environment.json"
            environment_path.write_text(json.dumps({}), encoding="utf-8")

            implementation_files = [
                HERE / "cjc_policy.py",
                HERE / "run_cjc_oracle.py",
                HERE / "test_cjc_policy.py",
                config_path,
                HERE / "capture_cjc_routes_gpu.py",
                HERE / "prepare_cjc_data_manifest.py",
                HERE / "build_cjc_data_registry.py",
                HERE / "prepare_cjc_calibration.py",
                HERE / "run_cjc_lut_gpu.py",
                HERE / "merge_cjc_luts.py",
                HERE.parents[3] / "experiments/shared/capture_moe.py",
            ]
            source_manifest = {
                "files": {str(path): sha256_file(path) for path in implementation_files}
            }
            source_path = root / "source.json"
            source_path.write_text(json.dumps(source_manifest), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_cjc_oracle.py"),
                    "--config", str(config_path),
                    "--route-trace", str(route_path),
                    "--lut", str(lut_path),
                    "--placement", str(placement_path),
                    "--data-manifest", str(data_path),
                    "--calibration-manifest", str(calibration_path),
                    "--environment", str(environment_path),
                    "--source-manifest", str(source_path),
                    "--mode", "dev",
                    "--output-dir", str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "PARTIAL")
            self.assertFalse(status["formal_run_valid"])
            self.assertIsNone(status["scientific_verdict"])
            self.assertEqual(decision["verdict"], "NOT_TESTED")
            self.assertFalse(decision["go"])


if __name__ == "__main__":
    unittest.main()
