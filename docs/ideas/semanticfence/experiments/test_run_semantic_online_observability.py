#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_semantic_online_observability_5090 as subject


@dataclass
class FakeRecord:
    document_sha256: str
    document_index: int
    offset: int
    layer: int
    token_position: int
    route_rank: int
    expert_id: int

    def identity_payload(self):
        return {
            "split": "calibration",
            "document_sha256": self.document_sha256,
            "document_index": self.document_index,
            "offset": self.offset,
            "token_position": self.token_position,
            "layer": self.layer,
            "expert_id": self.expert_id,
            "route_rank": self.route_rank,
            "hidden_sha256": f"{self.token_position:064x}",
        }


@dataclass
class FakeContext:
    window_id: str
    absolute_token_position: int
    routing_weight: float = 0.1


@dataclass
class FakeRow:
    row_id: str
    record: FakeRecord
    context: FakeContext


def fake_rows(count: int, *, document="a" * 64, window="window-a", offset=0):
    return [
        FakeRow(
            row_id=f"row-{window}-{index}",
            record=FakeRecord(
                document_sha256=document,
                document_index=0,
                offset=offset,
                layer=3,
                token_position=index,
                route_rank=1,
                expert_id=7,
            ),
            context=FakeContext(window, offset + index),
        )
        for index in range(count)
    ]


def edge(edge_id, left, right, *, index=0, document="a" * 64):
    return {
        "edge_id": edge_id,
        "row_ids": [left, right],
        "schedule_index": index,
        "logical_split": "validation",
        "document_sha256": document,
        "global_arrival_indices": [index, index + 1],
        "row_records": [{"document_index": 0}, {"document_index": 0}],
    }


class ConfigTests(unittest.TestCase):
    def test_frozen_config(self):
        path = HERE / "configs" / "semantic_online_observability_v1.json"
        value = subject.validate_config(json.loads(path.read_text()))
        self.assertEqual(
            value["dataset"]["selection_salt"],
            "semanticfence-p0-eval-20260809-v1",
        )
        self.assertEqual(
            value["candidate_schedule"]["stream_boundary"],
            "document_and_capture_window",
        )

    def test_strict_json_rejects_nonfinite(self):
        with self.assertRaises(subject.GateError):
            subject.canonical_json_bytes({"bad": float("inf")})

    def test_stack_comparison_ignores_only_physical_gpu_uuid(self):
        def bind(payload):
            return {**payload, "stack_digest": subject.canonical_sha256(payload)}

        frozen_payload = {
            "gpu": {
                "uuid": "GPU-frozen",
                "name": "NVIDIA GeForce RTX 5090",
                "driver_version": "580.65.06",
            },
            "torch": "2.8.0+cu128",
            "cuda": "12.8",
            "matmul_state": {"allow_tf32": False},
        }
        frozen = bind(frozen_payload)
        observed_payload = {
            **frozen_payload,
            "gpu": {**frozen_payload["gpu"], "uuid": "GPU-current"},
        }
        observed = bind(observed_payload)
        result = subject.validate_frozen_stack_except_gpu_uuid(observed, frozen)
        self.assertTrue(result["all_non_uuid_stack_fields_exact"])
        self.assertEqual(result["ignored_identity_leaf"], "gpu.uuid")
        self.assertEqual(
            result["derived_fields_recomputed_not_directly_compared"],
            ["stack_digest"],
        )
        self.assertEqual(result["observed_gpu_uuid"], "GPU-current")
        self.assertNotEqual(
            result["observed_stack_digest"], result["frozen_stack_digest"]
        )

        changed_driver_payload = {
            **observed_payload,
            "gpu": {**observed_payload["gpu"], "driver_version": "different"},
        }
        changed_driver = bind(changed_driver_payload)
        with self.assertRaisesRegex(subject.GateError, "outside gpu.uuid"):
            subject.validate_frozen_stack_except_gpu_uuid(changed_driver, frozen)
        changed_torch = bind({**observed_payload, "torch": "different"})
        with self.assertRaisesRegex(subject.GateError, "outside gpu.uuid"):
            subject.validate_frozen_stack_except_gpu_uuid(changed_torch, frozen)
        missing_uuid_payload = {
            **observed_payload,
            "gpu": {
                key: value
                for key, value in observed_payload["gpu"].items()
                if key != "uuid"
            },
        }
        missing_uuid = bind(missing_uuid_payload)
        with self.assertRaisesRegex(subject.GateError, "identity keys"):
            subject.validate_frozen_stack_except_gpu_uuid(missing_uuid, frozen)

        tampered_digest = {**observed, "stack_digest": "0" * 64}
        with self.assertRaisesRegex(subject.GateError, "does not bind"):
            subject.validate_frozen_stack_except_gpu_uuid(tampered_digest, frozen)


class CandidateTests(unittest.TestCase):
    def test_rolling_graph_is_window_bounded(self):
        document = "a" * 64
        first = fake_rows(4, document=document, window="w0", offset=0)
        second = fake_rows(4, document=document, window="w256", offset=256)
        schedule = subject.build_candidate_schedule(
            list(reversed(first + second)), {document: "test"}
        )
        self.assertEqual(len(schedule), 12)
        self.assertTrue(
            all(
                row["endpoint_context"][0]["window_id"]
                == row["endpoint_context"][1]["window_id"]
                for row in schedule
            )
        )

    def test_w8_and_document_cap(self):
        document = "b" * 64
        schedule = subject.build_candidate_schedule(
            fake_rows(10, document=document), {document: "test"}
        )
        self.assertEqual(len(schedule), 32)
        self.assertTrue(
            all(
                1
                <= row["compatible_arrival_indices"][1]
                - row["compatible_arrival_indices"][0]
                <= 8
                for row in schedule
            )
        )

    def test_input_order_does_not_change_schedule(self):
        document = "c" * 64
        rows = fake_rows(7, document=document)
        left = subject.build_candidate_schedule(rows, {document: "train"})
        right = subject.build_candidate_schedule(list(reversed(rows)), {document: "train"})
        self.assertEqual(
            [row["edge_id"] for row in left], [row["edge_id"] for row in right]
        )


class MatchingTests(unittest.TestCase):
    def test_triangle_is_general_graph(self):
        schedule = [
            edge("e01", "0", "1", index=0),
            edge("e12", "1", "2", index=1),
            edge("e02", "0", "2", index=2),
        ]
        result = subject.general_maximum_matching(
            schedule, {row["edge_id"]: True for row in schedule}
        )
        self.assertEqual(result["matching_edges"], 1)
        self.assertEqual(result["unique_vertices"], 3)

    def test_path_beats_bad_greedy(self):
        schedule = [
            edge("e12", "1", "2", index=0),
            edge("e01", "0", "1", index=1),
            edge("e23", "2", "3", index=2),
        ]
        result = subject.general_maximum_matching(
            schedule, {row["edge_id"]: True for row in schedule}
        )
        self.assertEqual(result["matching_edges"], 2)


class WitnessTests(unittest.TestCase):
    def test_conservative_unique_row_label(self):
        values = [
            {
                "endpoints": [
                    {"row_id": "a", "semantic_safe": True},
                    {"row_id": "b", "semantic_safe": True},
                ]
            },
            {
                "endpoints": [
                    {"row_id": "a", "semantic_safe": False},
                    {"row_id": "c", "semantic_safe": True},
                ]
            },
        ]
        self.assertEqual(
            subject.conservative_row_labels(values),
            {"a": False, "b": True, "c": True},
        )

    def test_missing_bank_abstains(self):
        score = subject.witness_score(
            "q",
            features={"q": [0.0], "s": [0.0]},
            row_cells={"q": (0, 0, subject.ABI)},
            banks={(0, 0, subject.ABI): {"safe": ["s"], "unsafe": []}},
        )
        self.assertFalse(score["eligible"])
        self.assertIsNone(score["score"])

    def test_score_orientation(self):
        score = subject.witness_score(
            "q",
            features={"q": [0.1], "s": [0.0], "u": [10.0]},
            row_cells={"q": (0, 0, subject.ABI)},
            banks={(0, 0, subject.ABI): {"safe": ["s"], "unsafe": ["u"]}},
        )
        self.assertTrue(score["eligible"])
        self.assertGreater(score["score"], 0)

    def test_pair_safe_is_endpoint_and(self):
        self.assertTrue(
            subject.pair_semantic_safe(
                [{"semantic_safe": True}, {"semantic_safe": True}]
            )
        )
        self.assertFalse(
            subject.pair_semantic_safe(
                [{"semantic_safe": True}, {"semantic_safe": False}]
            )
        )

    def test_normalization_fits_train_only(self):
        raw = {"train-a": [0.0, 2.0], "train-b": [2.0, 4.0], "test": [1e9, -1e9]}
        _, manifest = subject.normalize_features(raw, ["train-a", "train-b"])
        self.assertEqual(manifest["population_mean"], [1.0, 3.0])
        self.assertEqual(manifest["train_row_count"], 2)


class ThresholdTests(unittest.TestCase):
    def test_zero_unsafe_and_higher_threshold_tie(self):
        schedule = [
            edge("a", "r4", "r3", index=0),
            edge("b", "r3", "r2", index=1),
        ]
        scores = {
            "r4": {"eligible": True, "score": 4.0},
            "r3": {"eligible": True, "score": 3.0},
            "r2": {"eligible": True, "score": 2.0},
        }
        labels = {"r4": True, "r3": True, "r2": True}
        result = subject.select_validation_threshold(
            schedule,
            scores,
            labels,
            c1_ms=1.0,
            c2_ms=1.0,
            fixed_online_overhead_ms=0.0,
        )
        self.assertEqual(result["threshold"]["value"], 3.0)

    def test_expensive_m2_selects_abstain_all(self):
        schedule = [edge("a", "safe1", "safe2", index=0)]
        scores = {
            "safe1": {"eligible": True, "score": 2.0},
            "safe2": {"eligible": True, "score": 2.0},
        }
        result = subject.select_validation_threshold(
            schedule,
            scores,
            {"safe1": True, "safe2": True},
            c1_ms=1.0,
            c2_ms=3.0,
            fixed_online_overhead_ms=0.0,
        )
        self.assertEqual(
            result["threshold"]["mode"], "ABSTAIN_ALL_POSITIVE_INFINITY"
        )
        self.assertIsNone(result["threshold"]["value"])


class DiagnosticsTests(unittest.TestCase):
    def test_tie_group_risk_and_auc(self):
        scores = {
            "a": {"eligible": True, "score": 3.0},
            "b": {"eligible": True, "score": 2.0},
            "c": {"eligible": True, "score": 2.0},
            "d": {"eligible": True, "score": 1.0},
            "e": {"eligible": False, "score": None},
        }
        labels = {"a": True, "b": True, "c": False, "d": False, "e": True}
        result = subject.risk_coverage_diagnostics(scores, labels)
        self.assertAlmostEqual(result["auroc"], 0.875)
        self.assertAlmostEqual(result["auprc_average_precision"], 5 / 6)
        self.assertAlmostEqual(result["coverage_at_risk"]["0pct"], 0.2)
        self.assertEqual(
            [row["coverage"] for row in result["risk_coverage_curve"]],
            [0.0, 0.2, 0.6, 0.8],
        )

    def test_go_rejects_unsafe_admissible_unexecuted_edge(self):
        oracle = {
            "cost_projection": {"gross_saved_fraction": 0.1},
            "matching": {"row_coverage": 0.1, "matching_edges": 20},
            "positive_action_documents": 4,
        }
        certificate = {
            "unsafe_admissible_candidate_edges": 1,
            "unsafe_greedy_executed_pairs": 0,
            "greedy_executed_pairs": 20,
            "admitted_row_coverage": 0.1,
            "positive_action_documents": 4,
            "cost_projection": {"net_saved_fraction": 0.02},
        }
        self.assertEqual(
            subject.decide_verdict(oracle, certificate), "PIVOT_TO_SHADOW_VERIFY"
        )

    def test_test_outcome_refuses_missing_predecision(self):
        schedule = [{"logical_split": "test"}]
        with self.assertRaisesRegex(subject.GateError, "frozen threshold"):
            subject.execute_semantic_split(
                object(),
                schedule,
                {},
                {},
                pre_outcome_lock=Path(__file__),
                threshold_artifact=None,
            )

    def test_complete_hash_closure_excludes_complete_itself(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "A.json").write_text("{}\n", encoding="utf-8")
            (root / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
            hashes = subject._artifact_hashes(root, exclude={"COMPLETE.json"})
            self.assertEqual(set(hashes), {"A.json"})


if __name__ == "__main__":
    unittest.main()
