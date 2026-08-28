from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_shape_lane_continuous_cost_gate as gate


class FakeTensor:
    def __init__(self, values: object) -> None:
        self.array = np.asarray(values, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> object:
        return self.array.dtype

    def __getitem__(self, key: object) -> "FakeTensor":
        return FakeTensor(self.array[key])

    def new_zeros(self, shape: tuple[int, ...]) -> "FakeTensor":
        return FakeTensor(np.zeros(shape, dtype=self.array.dtype))


class RecordingExpert:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []
        self.inputs: list[np.ndarray] = []

    def __call__(self, value: FakeTensor) -> FakeTensor:
        self.shapes.append(value.shape)
        self.inputs.append(value.array.copy())
        return FakeTensor(value.array * 2.0 + 1.0)


def fake_torch_module() -> object:
    return types.SimpleNamespace(
        cat=lambda values, dim=0: FakeTensor(
            np.concatenate([value.array for value in values], axis=dim)
        )
    )


def roster_fixture() -> tuple[list[dict[str, object]], list[str]]:
    identities = [
        {"request_id": "r0", "arrival_us": 1.0, "sample_id": 0},
        {"request_id": "r1", "arrival_us": 2.0, "sample_id": 1},
    ]
    common = {
        "schema_version": "stablebatch-shape-lane-native-roster-row-v1",
        "batch_size": 2,
        "active_request_ids": ["r0", "r1"],
        "pending_request_count": 0,
        "request_ids": ["r0", "r1"],
        "admission_identity": identities,
        "native_route_membership_sha256": "a" * 64,
        "native_final_logits_sha256": "b" * 64,
    }
    rows = [
        {
            **common,
            "batch_index": 0,
            "decode_steps": [0, 0],
            "input_token_ids": [10, 20],
            "native_predicted_next_token_ids": [11, 21],
            "position_ids": [3, 4],
            "prior_cache_lengths": [3, 4],
            "left_padding": [1, 0],
        },
        {
            **common,
            "batch_index": 1,
            "decode_steps": [1, 1],
            "input_token_ids": [11, 21],
            "native_predicted_next_token_ids": [12, 22],
            "position_ids": [4, 5],
            "prior_cache_lengths": [4, 5],
            "left_padding": [1, 0],
        },
    ]
    return rows, ["r0", "r1"]


def correctness_fixture() -> dict[str, dict[str, object]]:
    return {
        arm: {
            "raw_repeat_mismatches": 0,
            "route_repeat_mismatches": 0,
            "final_logits_repeat_mismatches": 0,
        }
        for arm in gate.ARMS
    }


class ExpertExecutorTests(unittest.TestCase):
    def test_all_three_arms_and_c8_slice(self) -> None:
        source = FakeTensor([[1, 2], [3, 4], [5, 6]])
        expected = source.array * 2.0 + 1.0
        expected_shapes = {
            gate.ARM_NATIVE: [(3, 2)],
            gate.ARM_SERIAL: [(1, 2), (1, 2), (1, 2)],
            gate.ARM_C8: [(8, 2)],
        }
        with mock.patch.dict(sys.modules, {"torch": fake_torch_module()}):
            for arm in gate.ARMS:
                expert = RecordingExpert()
                result = gate.execute_expert_policy(expert, source, arm)
                self.assertEqual(expert.shapes, expected_shapes[arm])
                np.testing.assert_array_equal(result.output.array, expected)
                self.assertEqual(result.output.shape, source.shape)
                if arm == gate.ARM_C8:
                    self.assertEqual(result.physical_m, 8)
                    self.assertEqual(result.padding_rows, 5)
                    np.testing.assert_array_equal(expert.inputs[0][3:], 0.0)
                if arm == gate.ARM_SERIAL:
                    self.assertEqual(result.kernel_calls, 3)

    def test_m_greater_than_eight_fails_closed(self) -> None:
        expert = RecordingExpert()
        with self.assertRaisesRegex(gate.ProtocolError, "exceeds frozen C"):
            gate.execute_expert_policy(expert, FakeTensor(np.ones((9, 2))), gate.ARM_C8)
        self.assertEqual(expert.shapes, [])


class RosterTests(unittest.TestCase):
    def test_roster_conserves_steps_tokens_kv_and_admission_identity(self) -> None:
        roster, request_ids = roster_fixture()
        audit = gate.validate_roster_conservation(
            roster,
            expected_request_ids=request_ids,
            expected_steps_per_request=2,
            max_batch_size=8,
        )
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["request_steps"], 4)

    def test_roster_rejects_missing_duplicate_and_token_or_kv_drift(self) -> None:
        roster, request_ids = roster_fixture()
        mutations = []
        missing = copy.deepcopy(roster)
        for field in (
            "request_ids",
            "decode_steps",
            "input_token_ids",
            "native_predicted_next_token_ids",
            "position_ids",
            "prior_cache_lengths",
            "left_padding",
            "admission_identity",
        ):
            missing[1][field] = missing[1][field][:-1]
        missing[1]["batch_size"] = 1
        mutations.append(missing)
        duplicate = copy.deepcopy(roster)
        duplicate[0]["request_ids"] = ["r0", "r0"]
        mutations.append(duplicate)
        token_drift = copy.deepcopy(roster)
        token_drift[1]["input_token_ids"][0] = 999
        mutations.append(token_drift)
        kv_drift = copy.deepcopy(roster)
        kv_drift[1]["prior_cache_lengths"][0] = 9
        kv_drift[1]["position_ids"][0] = 9
        kv_drift[1]["left_padding"] = [0, 4]
        mutations.append(kv_drift)
        for mutated in mutations:
            with self.subTest(mutated=mutated), self.assertRaises(gate.ProtocolError):
                gate.validate_roster_conservation(
                    mutated,
                    expected_request_ids=request_ids,
                    expected_steps_per_request=2,
                    max_batch_size=8,
                )


class RepeatAndGateTests(unittest.TestCase):
    def test_replay_result_is_adapted_to_mapping_comparator(self) -> None:
        signature = {
            "batch_index": 0,
            "roster_identity_sha256": "r",
            "raw_calls_sha256": "raw",
            "route_membership_sha256": "route",
            "final_logits_sha256": "final",
            "greedy_token_ids": [1],
        }
        replay = types.SimpleNamespace(
            arm=gate.ARM_C8,
            step_signatures=[signature],
        )
        payload = gate.replay_comparison_payload(replay)
        self.assertEqual(payload, {"arm": gate.ARM_C8, "step_signatures": [signature]})
        result = gate.compare_policy_repeats(payload, copy.deepcopy(payload))
        self.assertTrue(result["bitwise_repeat_stable"])

    def test_policy_repeat_comparison_is_within_arm(self) -> None:
        signature = {
            "batch_index": 0,
            "roster_identity_sha256": "r",
            "raw_calls_sha256": "raw",
            "route_membership_sha256": "route",
            "final_logits_sha256": "final",
            "greedy_token_ids": [1, 2],
        }
        first = {"arm": gate.ARM_C8, "step_signatures": [signature]}
        second = copy.deepcopy(first)
        result = gate.compare_policy_repeats(first, second)
        self.assertTrue(result["bitwise_repeat_stable"])
        second["step_signatures"][0]["raw_calls_sha256"] = "changed"
        result = gate.compare_policy_repeats(first, second)
        self.assertEqual(result["raw_repeat_mismatches"], 1)

    def test_frozen_threshold_boundaries_are_inclusive(self) -> None:
        config = json.loads(
            (HERE / "configs" / "shape_lane_continuous_cost_gate_v1.json").read_text()
        )
        metrics = {
            gate.ARM_NATIVE: {
                "median_total_expert_gpu_ms": 4.0,
                "median_token_step_p99_ms": 10.0,
            },
            gate.ARM_SERIAL: {
                "median_total_expert_gpu_ms": 10.0,
                "median_token_step_p99_ms": 20.0,
            },
            gate.ARM_C8: {
                "median_total_expert_gpu_ms": 8.0,
                "median_token_step_p99_ms": 10.5,
            },
        }
        result = gate.classify_gate(correctness_fixture(), metrics, config["gate"])
        self.assertEqual(result["verdict"], config["gate"]["pass"])
        metrics[gate.ARM_C8]["median_total_expert_gpu_ms"] = 8.000001
        result = gate.classify_gate(correctness_fixture(), metrics, config["gate"])
        self.assertEqual(result["verdict"], config["gate"]["cost_fail"])
        metrics[gate.ARM_C8]["median_total_expert_gpu_ms"] = 8.0
        correctness = correctness_fixture()
        correctness[gate.ARM_C8]["route_repeat_mismatches"] = 1
        result = gate.classify_gate(correctness, metrics, config["gate"])
        self.assertEqual(result["verdict"], config["gate"]["correctness_fail"])


class IntegrityAndBoundaryTests(unittest.TestCase):
    def test_manifest_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence.json").write_text("{}\n", encoding="utf-8")
            manifest = gate.build_manifest(root, required_names=("evidence.json",))
            gate.verify_manifest(root, manifest, required_names=("evidence.json",))
            (root / "evidence.json").write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(gate.ProtocolError, "manifest"):
                gate.verify_manifest(root, manifest, required_names=("evidence.json",))

    def test_claim_boundary_and_official_bi_are_fail_closed(self) -> None:
        config = json.loads(
            (HERE / "configs" / "shape_lane_continuous_cost_gate_v1.json").read_text()
        )
        gate.validate_frozen_config(config)
        self.assertEqual(
            config["execution"]["native_roster_serial_equivalence_audit"],
            gate.BCRD_SERIAL_AUDIT_STATUS,
        )
        self.assertEqual(config["official_batch_invariance"]["status"], "NOT_EXECUTABLE")
        self.assertIn("not_bcrd_formal_producer", gate.EXPERIMENT_BOUNDARY)
        self.assertIn("not_serving", gate.EXPERIMENT_BOUNDARY)
        serial_drift = copy.deepcopy(config)
        serial_drift["execution"]["native_roster_serial_equivalence_audit"] = "PASS"
        with self.assertRaises(gate.ProtocolError):
            gate.validate_frozen_config(serial_drift)
        bi_drift = copy.deepcopy(config)
        bi_drift["official_batch_invariance"]["status"] = "PASS"
        with self.assertRaises(gate.ProtocolError):
            gate.validate_frozen_config(bi_drift)

    def test_timed_moe_forward_has_no_host_materialization_or_uniqueness_sync(self) -> None:
        source = inspect.getsource(gate.MoEPolicyController.forward)
        for forbidden in (".tolist(", ".cpu(", ".item(", "torch.unique", ".clone("):
            self.assertNotIn(forbidden, source)
        finish_source = inspect.getsource(gate.MoEPolicyController.finish_step)
        self.assertIn("by_layer_token", finish_source)
        self.assertIn("len(set(expert_ids))", finish_source)


if __name__ == "__main__":
    unittest.main()
