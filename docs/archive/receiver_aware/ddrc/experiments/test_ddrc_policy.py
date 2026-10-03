from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from ddrc_policy import (
    AccountingConfig,
    CreditEnvelope,
    CreditRecord,
    CreditTiming,
    FormatTiming,
    INFORMATION_GLOBAL_ORACLE,
    INFORMATION_RECEIVER_CREDIT,
    INFORMATION_SENDER_LOCAL,
    LaneMatrix,
    PolicyState,
    ReceiverResourceView,
    Topology,
    account_step,
    apply_receiver_credit_messages,
    build_receiver_credit_messages,
    decode_credit_message,
    deterministic_origin_lpt,
    encode_credit_message,
    high_lane_bytes,
    lane_byte_breakdown,
    low_lane_bytes,
    make_receiver_resource_views,
    make_sender_local_views,
    plan_ddrc,
    plan_global_oracle,
    plan_sender_local,
    positive_net_gate,
)
from run_ddrc_existence_gpu import (
    build_decision,
    build_status,
    canonical_json_hash,
    load_traces,
    paired_bootstrap,
    parse_trace_record,
    sha256_file,
    validate_formal_gate,
    validate_formal_inputs,
)


class DDRCPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = Topology(
            ep_size=4,
            node_by_rank=(0, 0, 1, 1),
            receive_resource_by_rank=("node0_nic0", "node0_nic0", "node1_nic0", "node1_nic0"),
            link_gbps=0.01,
        )
        zero = FormatTiming(0.0, 0.0, 0.0, 128, "measured_same_run")
        self.cfg = AccountingConfig(
            hidden_dim=32,
            lane_descriptor_bytes=16,
            lane_alignment_bytes=16,
            codec_tile_rows=32,
            codec_tax_mode="serialized_tiles",
            high_timing=zero,
            low_timing=zero,
            credit_timing=CreditTiming(
                build_us=0.0,
                aggregate_us=0.0,
                transfer_us=0.0,
                parse_us=0.0,
                pack_deadline_slack_us=1.0,
                overlap_proven=True,
                source="measured_same_run",
            ),
        )
        self.matrix = LaneMatrix(
            lane_counts={
                (0, 0): 1,
                (2, 0): 1,
                (1, 1): 1,
                (2, 1): 1,
                (0, 2): 1,
                (2, 2): 1,
                (0, 3): 1,
                (3, 3): 1,
            },
            valid_origin_tokens={0: 1, 1: 1, 2: 1, 3: 1},
            top_k=2,
            step=3,
            layer=7,
            trace_id="canonical",
        )

    def test_identity_remote_local_and_receiver_closure(self) -> None:
        self.matrix.validate_closure(self.topology)
        remote = self.matrix.remote_counts(self.topology)
        self.assertEqual(set(remote), {(2, 0), (2, 1), (0, 2), (0, 3)})
        receiver_rows = {
            receiver: sum(
                count for (_sender, lane_receiver), count in self.matrix.lane_counts.items()
                if lane_receiver == receiver
            )
            for receiver in range(4)
        }
        self.assertEqual(receiver_rows, {0: 2, 1: 2, 2: 2, 3: 2})

    def test_receiver_closure_rejects_missing_or_double_counted_pair(self) -> None:
        broken = LaneMatrix(
            lane_counts={**self.matrix.lane_counts, (3, 0): 1},
            valid_origin_tokens=self.matrix.valid_origin_tokens,
            top_k=2,
            step=3,
            layer=7,
        )
        with self.assertRaisesRegex(ValueError, "receiver closure failed"):
            broken.validate_closure(self.topology)

    def test_origin_lpt_is_deterministic_and_route_free(self) -> None:
        weights = {"req-b": 8, "req-a": 8, "req-c": 2, "req-d": 1}
        first = deterministic_origin_lpt(weights, 2)
        second = deterministic_origin_lpt(dict(reversed(list(weights.items()))), 2)
        self.assertEqual(first, second)
        self.assertEqual(first["req-a"], 0)
        self.assertNotIn("route", inspect.signature(deterministic_origin_lpt).parameters)

    def test_formal_trace_recomputes_lpt_and_origin_token_closure(self) -> None:
        raw = {
            "trace_id": "formal-lpt",
            "stream_id": "stream-0",
            "split": "sealed",
            "step": 3,
            "layer": 7,
            "top_k": 2,
            "valid_origin_tokens": {"0": 1, "1": 1, "2": 1, "3": 1},
            "lane_counts": [
                {"sender": sender, "receiver": receiver, "count": count}
                for (sender, receiver), count in self.matrix.lane_counts.items()
            ],
            "origin_balancing": "scheduler_visible_token_count_lpt",
            "request_weights": {"req0": 1, "req1": 1, "req2": 1, "req3": 1},
            "request_assignment": {"req0": 0, "req1": 1, "req2": 2, "req3": 3},
        }
        split, parsed = parse_trace_record(raw, self.topology, formal=True)
        self.assertEqual(split, "sealed")
        self.assertEqual(parsed.trace_id, "formal-lpt")
        bad = dict(raw)
        bad["request_assignment"] = {"req0": 1, "req1": 0, "req2": 2, "req3": 3}
        with self.assertRaisesRegex(ValueError, "not deterministic origin LPT"):
            parse_trace_record(bad, self.topology, formal=True)

    def test_sender_local_signature_and_view_cannot_read_global_matrix(self) -> None:
        parameters = set(inspect.signature(plan_sender_local).parameters)
        self.assertEqual(parameters, {"view", "threshold_bytes", "cfg", "topology"})
        views = make_sender_local_views(self.matrix, self.topology)
        original = plan_sender_local(
            views[2], threshold_bytes=0, cfg=self.cfg, topology=self.topology
        )
        changed = LaneMatrix(
            lane_counts={
                (0, 0): 1,
                (2, 0): 1,
                (1, 1): 1,
                (2, 1): 1,
                (1, 2): 1,
                (2, 2): 1,
                (1, 3): 1,
                (3, 3): 1,
            },
            valid_origin_tokens=self.matrix.valid_origin_tokens,
            top_k=2,
            step=3,
            layer=7,
        )
        changed_view = make_sender_local_views(changed, self.topology)[2]
        replay = plan_sender_local(
            changed_view, threshold_bytes=0, cfg=self.cfg, topology=self.topology
        )
        self.assertEqual(original.requested_lanes, replay.requested_lanes)
        self.assertEqual(original.low_lanes, replay.low_lanes)
        self.assertEqual(original.information_set, INFORMATION_SENDER_LOCAL)

    def test_receiver_credit_is_local_to_one_receive_resource(self) -> None:
        view = make_receiver_resource_views(self.matrix, self.topology)["node0_nic0"]
        threshold = high_lane_bytes(1, self.cfg).wire_bytes
        messages, lanes = build_receiver_credit_messages(
            view, threshold_bytes=threshold, cfg=self.cfg
        )
        modified_other_resource = ReceiverResourceView(
            resource="node0_nic0",
            receiver_ranks=view.receiver_ranks,
            lane_counts=dict(view.lane_counts),
            step=view.step,
            layer=view.layer,
        )
        replay_messages, replay_lanes = build_receiver_credit_messages(
            modified_other_resource, threshold_bytes=threshold, cfg=self.cfg
        )
        self.assertEqual(lanes, replay_lanes)
        self.assertEqual([message.payload for message in messages], [message.payload for message in replay_messages])

    def test_credit_header_record_and_alignment_accounting(self) -> None:
        one = encode_credit_message((CreditRecord(2, 0),), step=3, layer=7, alignment=16)
        two = encode_credit_message(
            (CreditRecord(2, 0), CreditRecord(2, 1)), step=3, layer=7, alignment=16
        )
        self.assertEqual(len(one), 32)  # align16(16 + 8)
        self.assertEqual(len(two), 32)  # align16(16 + 16)
        step, layer, records = decode_credit_message(one, 16)
        self.assertEqual((step, layer), (3, 7))
        self.assertEqual(records, (CreditRecord(2, 0),))
        malformed = bytearray(one)
        malformed[-1] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            decode_credit_message(bytes(malformed), 16)

    def test_protocol_hardgate_example_rejects_10_minus_8_minus_3(self) -> None:
        self.assertFalse(positive_net_gate(10.0, 8.0, 3.0))
        self.assertTrue(positive_net_gate(10.0, 8.0, 1.0))
        self.assertFalse(positive_net_gate(10.0, 8.0, 2.0))  # equality is not positive

    def test_late_credit_fallback_rolls_back_state_but_keeps_credit_tax(self) -> None:
        late_cfg = replace(
            self.cfg,
            credit_timing=CreditTiming(1.0, 1.0, 1.0, 1.0, 1.0, False, "measured_same_run"),
        )
        view = make_sender_local_views(self.matrix, self.topology)[2]
        payload = encode_credit_message((CreditRecord(2, 0),), 3, 7, 16)
        envelope = CreditEnvelope("node0_nic0", 2, 3, 7, payload, arrival_us=4.0, deadline_us=1.0)
        state = PolicyState(committed_low_lanes=((2, 1),), action_epoch=9)
        before = state.snapshot()
        plan = apply_receiver_credit_messages(
            view, (envelope,), cfg=late_cfg, topology=self.topology, state=state
        )
        self.assertEqual(plan.low_lanes, ())
        self.assertEqual(plan.fallback_reason, "late_credit")
        self.assertEqual(state.snapshot(), before)
        self.assertEqual(plan.credit_bytes, 32)
        self.assertEqual(plan.credit_visible_us, 4.0)

    def test_duplicate_credit_falls_back_and_keeps_bytes(self) -> None:
        view = make_sender_local_views(self.matrix, self.topology)[2]
        payload = encode_credit_message((CreditRecord(2, 0),), 3, 7, 16)
        message = CreditEnvelope("node0_nic0", 2, 3, 7, payload, 0.0, 1.0)
        state = PolicyState()
        plan = apply_receiver_credit_messages(
            view, (message, message), cfg=self.cfg, topology=self.topology, state=state
        )
        self.assertEqual(plan.low_lanes, ())
        self.assertEqual(plan.fallback_reason, "malformed_or_duplicate_credit")
        self.assertEqual(plan.credit_bytes, 64)

    def test_hardgate_reject_rolls_back_state_but_keeps_credit_tax(self) -> None:
        expensive_credit = replace(
            self.cfg,
            credit_timing=CreditTiming(0.5, 0.5, 0.5, 0.5, 5.0, False, "measured_same_run"),
        )
        fast_topology = replace(self.topology, link_gbps=800.0)
        view = make_sender_local_views(self.matrix, fast_topology)[2]
        payload = encode_credit_message((CreditRecord(2, 0),), 3, 7, 16)
        message = CreditEnvelope("node0_nic0", 2, 3, 7, payload, 2.0, 5.0)
        state = PolicyState(committed_low_lanes=((2, 1),), action_epoch=4)
        before = state.snapshot()
        plan = apply_receiver_credit_messages(
            view, (message,), cfg=expensive_credit, topology=fast_topology, state=state
        )
        self.assertEqual(plan.low_lanes, ())
        self.assertEqual(plan.fallback_reason, "hardgate_reject")
        self.assertEqual(state.snapshot(), before)
        self.assertGreater(plan.credit_visible_us, 0.0)

    def test_byte_ledger_closes_with_payload_scale_descriptor_padding(self) -> None:
        for low, breakdown in (
            (False, high_lane_bytes(3, self.cfg)),
            (True, low_lane_bytes(3, self.cfg)),
        ):
            self.assertEqual(
                breakdown.payload_bytes
                + breakdown.scale_bytes
                + breakdown.descriptor_bytes
                + breakdown.padding_bytes,
                breakdown.wire_bytes,
                msg=f"low={low}",
            )
            self.assertEqual(breakdown.wire_bytes % self.cfg.lane_alignment_bytes, 0)

    def test_no_overlap_proof_charges_full_credit_time_and_boundary_is_not_rdma(self) -> None:
        timing = CreditTiming(1.0, 2.0, 3.0, 4.0, 100.0, False, "assumed")
        self.assertEqual(timing.visible_us, 10.0)
        self.assertIn("NOT_RDMA", self.cfg.evidence_boundary)
        with self.assertRaisesRegex(ValueError, "NOT_RDMA"):
            replace(self.cfg, evidence_boundary="RDMA")

    def test_current_action_has_no_future_or_label_input(self) -> None:
        parameters = set(inspect.signature(plan_ddrc).parameters)
        self.assertNotIn("future", parameters)
        self.assertNotIn("labels", parameters)
        threshold = {"node0_nic0": 0, "node1_nic0": 0}
        first = plan_ddrc(
            self.matrix,
            receiver_threshold_bytes=threshold,
            cfg=self.cfg,
            topology=self.topology,
        )
        second = plan_ddrc(
            self.matrix,
            receiver_threshold_bytes=threshold,
            cfg=self.cfg,
            topology=self.topology,
        )
        self.assertEqual(first.requested_lanes, second.requested_lanes)
        self.assertEqual(first.low_lanes, second.low_lanes)
        self.assertEqual(first.information_set, INFORMATION_RECEIVER_CREDIT)

    def test_global_oracle_is_separate_and_not_worse_than_ddrc(self) -> None:
        threshold = {"node0_nic0": 0, "node1_nic0": 0}
        ddrc = plan_ddrc(
            self.matrix,
            receiver_threshold_bytes=threshold,
            cfg=self.cfg,
            topology=self.topology,
        )
        oracle = plan_global_oracle(
            self.matrix,
            cfg=self.cfg,
            topology=self.topology,
            deployable_seed_plans=(ddrc,),
        )
        self.assertEqual(oracle.information_set, INFORMATION_GLOBAL_ORACLE)
        self.assertLessEqual(
            account_step(self.matrix, oracle, cfg=self.cfg, topology=self.topology).total_us,
            account_step(self.matrix, ddrc, cfg=self.cfg, topology=self.topology).total_us,
        )

    def test_formal_gate_requires_signed_off_and_exact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = root / "protocol.md"
            data = root / "data.json"
            attestation = root / "review.json"
            protocol.write_text("frozen", encoding="utf-8")
            data.write_text("{}", encoding="utf-8")
            config = {"x": 1}
            source = {"sha256": "source-hash"}
            payload = {
                "status": "SIGNED-OFF",
                "protocol_sha256": sha256_file(protocol),
                "config_sha256": canonical_json_hash(config),
                "source_sha256": "source-hash",
                "data_manifest_sha256": sha256_file(data),
            }
            attestation.write_text(json.dumps(payload), encoding="utf-8")
            args = SimpleNamespace(
                review_attestation=attestation,
                data_manifest=data,
                protocol=protocol,
            )
            self.assertEqual(validate_formal_gate(args, config, source)["status"], "SIGNED-OFF")
            payload["status"] = "BLOCKED"
            attestation.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not SIGNED-OFF"):
                validate_formal_gate(args, config, source)

    def test_formal_manifest_hashes_trace_and_quality_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = root / "trace.jsonl"
            quality = root / "quality.csv"
            trace.write_text("{}\n", encoding="utf-8")
            quality.write_text("header\n", encoding="utf-8")
            args = SimpleNamespace(trace_jsonl=trace, quality_csv=quality)
            manifest = {
                "sealed": True,
                "trace_jsonl_sha256": sha256_file(trace),
                "quality_csv_sha256": sha256_file(quality),
            }
            validate_formal_inputs(args, manifest)
            manifest["quality_csv_sha256"] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "quality_csv_sha256 mismatch"):
                validate_formal_inputs(args, manifest)

    def test_quality_action_signature_mismatch_is_rejected(self) -> None:
        accounting = []
        quality = []
        for arm in (
            "uniform_full",
            "uniform_low",
            "calib_static",
            "causal_prev_step",
            "sender_local_exact_handle",
            "DDRC",
        ):
            accounting.append({
                "trace_id": "t0",
                "model": "olmoe",
                "arm": arm,
                "net_saving_fraction": 0.0,
                "action_signature": f"sig-{arm}",
            })
            quality.append({
                "trace_id": "t0",
                "model": "olmoe",
                "arm": arm,
                "incremental_accuracy_harm": 0.0,
                "cvar10_positive_harm": 0.0,
                "action_signature": "wrong" if arm == "DDRC" else f"sig-{arm}",
            })
        with self.assertRaisesRegex(ValueError, "quality/action signature mismatch"):
            paired_bootstrap(accounting, quality, lambdas=(0.0,), repeats=2, seed=1)

    def test_formal_decision_fails_closed_when_capabilities_are_incomplete(self) -> None:
        capabilities = {
            "native_route_capture": False,
            "native_action_matched_gpu_quality": False,
            "measured_credit_timing": False,
            "burst_block_bootstrap": False,
        }
        decision = build_decision(
            "formal",
            accounting_rows=({"model": "olmoe"}, {"model": "llmjp"}),
            bootstrap={"status": "COMPLETE", "rows": []},
            config={
                "decision": {"required_models": ["olmoe", "llmjp"]},
                "formal": {"capabilities": capabilities},
            },
        )
        self.assertEqual(decision["status"], "PARTIAL")
        self.assertFalse(decision["go"])
        self.assertEqual(
            decision["missing_capabilities"],
            sorted(capabilities),
        )
        self.assertIn("no scientific GO/NO-GO", decision["evidence_boundary"])

    def test_formal_single_model_cannot_emit_go(self) -> None:
        decision = build_decision(
            "formal",
            accounting_rows=({"model": "olmoe"},),
            bootstrap={"status": "COMPLETE", "rows": []},
            config={
                "decision": {"required_models": ["olmoe", "llmjp"]},
                "formal": {"capabilities": {
                    "native_route_capture": True,
                    "native_action_matched_gpu_quality": True,
                    "measured_credit_timing": True,
                    "burst_block_bootstrap": True,
                }},
            },
        )
        self.assertEqual(decision["status"], "PARTIAL")
        self.assertFalse(decision["go"])
        self.assertEqual(decision["observed_models"], ["olmoe"])
        self.assertEqual(decision["missing_models"], ["llmjp"])

    def test_formal_trace_rejects_reverse_order_within_stream(self) -> None:
        base = {
            "stream_id": "stream-0",
            "model": "olmoe",
            "split": "sealed",
            "layer": 7,
            "top_k": 2,
            "valid_origin_tokens": {"0": 1, "1": 1, "2": 1, "3": 1},
            "lane_counts": [
                {"sender": sender, "receiver": receiver, "count": count}
                for (sender, receiver), count in self.matrix.lane_counts.items()
            ],
            "origin_balancing": "scheduler_visible_token_count_lpt",
            "request_weights": {"req0": 1, "req1": 1, "req2": 1, "req3": 1},
            "request_assignment": {"req0": 0, "req1": 1, "req2": 2, "req3": 3},
        }
        rows = [
            {**base, "trace_id": "later", "step": 2},
            {**base, "trace_id": "earlier", "step": 1},
        ]
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "reverse.jsonl"
            trace.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not strictly monotonic"):
                load_traces(
                    trace,
                    {"models": {"olmoe": {"top_k": 2}}},
                    self.topology,
                    formal=True,
                )

    def test_formal_trace_rejects_wrong_top_k_and_main_cell_drop(self) -> None:
        base = {
            "trace_id": "formal-cell",
            "stream_id": "stream-0",
            "model": "olmoe",
            "split": "sealed",
            "step": 1,
            "layer": 7,
            "top_k": 2,
            "valid_origin_tokens": {"0": 1, "1": 1, "2": 1, "3": 1},
            "lane_counts": [
                {"sender": sender, "receiver": receiver, "count": count}
                for (sender, receiver), count in self.matrix.lane_counts.items()
            ],
            "origin_balancing": "scheduler_visible_token_count_lpt",
            "request_weights": {"req0": 1, "req1": 1, "req2": 1, "req3": 1},
            "request_assignment": {"req0": 0, "req1": 1, "req2": 2, "req3": 3},
        }
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / "cell.jsonl"
            trace.write_text(json.dumps(base) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top_k mismatch"):
                load_traces(
                    trace,
                    {"models": {"olmoe": {"top_k": 3}}},
                    self.topology,
                    formal=True,
                )

            dropped = dict(base)
            dropped["trace_id"] = "formal-drop"
            dropped["dropped_pairs_by_receiver"] = {"0": 1}
            dropped["lane_counts"] = [
                row for row in base["lane_counts"]
                if not (row["sender"] == 0 and row["receiver"] == 0)
            ]
            trace.write_text(json.dumps(dropped) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires dropped_pairs.*empty"):
                load_traces(
                    trace,
                    {"models": {"olmoe": {"top_k": 2}}},
                    self.topology,
                    formal=True,
                )

    def test_status_separates_formal_validity_verdict_and_go(self) -> None:
        partial = build_status(
            "formal", {"status": "PARTIAL", "go": False}, {"status": "SIGNED-OFF"}
        )
        self.assertFalse(partial["formal_run_valid"])
        self.assertIsNone(partial["scientific_verdict"])
        self.assertFalse(partial["go"])
        no_go = build_status(
            "formal", {"status": "NO_GO", "go": False}, {"status": "SIGNED-OFF"}
        )
        self.assertTrue(no_go["formal_run_valid"])
        self.assertEqual(no_go["scientific_verdict"], "NO_GO")
        self.assertFalse(no_go["go"])


if __name__ == "__main__":
    unittest.main()
