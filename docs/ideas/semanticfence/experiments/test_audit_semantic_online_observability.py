#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
import sys

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_semantic_online_observability as subject


def write_json(path: Path, value: object) -> None:
    path.write_bytes(subject.gate.canonical_json_bytes(value))


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(subject.gate.canonical_json_bytes(row) for row in rows))


def schedule_edge(split: str = "train") -> dict[str, object]:
    records = [
        {"document_index": 0, "layer": 1, "expert_id": 2, "token_position": index}
        for index in range(2)
    ]
    return {
        "schedule_index": 0,
        "edge_id": "edge-0",
        "logical_split": split,
        "document_sha256": "d" * 64,
        "window_id": "window-0",
        "layer": 1,
        "expert_id": 2,
        "abi": "abi-v1",
        "row_ids": ["row-a", "row-b"],
        "row_records": records,
        "endpoint_context": [
            {"window_id": "window-0"},
            {"window_id": "window-0"},
        ],
    }


def feature_row(
    row_id: str, record: dict[str, object], value: float
) -> dict[str, object]:
    vector = [value] * 76
    return {
        "row_id": row_id,
        "logical_split": "train",
        "document_sha256": "d" * 64,
        "row_record": record,
        "window_id": "window-0",
        "cell": [1, 2, "abi-v1"],
        "feature_vector": vector,
        "feature_vector_sha256": subject.gate.canonical_sha256(vector),
    }


class NumericComparisonTests(unittest.TestCase):
    def test_accepts_cross_python_ulp_drift(self) -> None:
        actual = {
            "candidate_evaluations": [
                {"threshold_value": 0.1486107919801413, "eligible": True}
            ]
        }
        expected = {
            "candidate_evaluations": [
                {"threshold_value": 0.14861079198014113, "eligible": True}
            ]
        }
        subject.require_numeric_close(actual, expected, "threshold")

    def test_rejects_material_float_difference(self) -> None:
        with self.assertRaisesRegex(subject.AuditError, "absolute_delta"):
            subject.require_numeric_close(
                {"score": 0.1}, {"score": 0.100000001}, "score"
            )

    def test_keeps_discrete_values_and_scalar_types_exact(self) -> None:
        with self.assertRaisesRegex(subject.AuditError, "eligible"):
            subject.require_numeric_close(
                {"eligible": True}, {"eligible": False}, "threshold"
            )
        with self.assertRaisesRegex(subject.AuditError, "types"):
            subject.require_numeric_close({"count": 1}, {"count": 1.0}, "count")

    def test_oracle_separates_networkx_provenance_only(self) -> None:
        actual = {
            "matching": {
                "networkx_version": "3.2.1",
                "algorithm": "blossom",
                "matching_edges": 2,
            }
        }
        expected = {
            "matching": {
                "networkx_version": "3.5",
                "algorithm": "blossom",
                "matching_edges": 2,
            }
        }
        self.assertEqual(
            subject.compare_oracle_recompute(actual, expected), ("3.5", "3.2.1")
        )
        expected["matching"]["matching_edges"] = 3
        with self.assertRaisesRegex(subject.AuditError, "matching_edges"):
            subject.compare_oracle_recompute(actual, expected)


class WitnessBankTests(unittest.TestCase):
    def test_reconstructs_exact_membership_cells_and_digests(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            edge = schedule_edge()
            feature_rows = [
                feature_row("row-a", edge["row_records"][0], 0.0),
                feature_row("row-b", edge["row_records"][1], 1.0),
            ]
            train_results = [
                {
                    "edge_id": "edge-0",
                    "endpoints": [
                        {"row_id": "row-a", "semantic_safe": True},
                        {"row_id": "row-b", "semantic_safe": False},
                    ],
                }
            ]
            write_jsonl(root / "PRE_OUTCOME_FEATURES.jsonl", feature_rows)
            write_jsonl(root / "TRAIN_EDGE_RESULTS.jsonl", train_results)
            expected_manifest = {
                "schema_version": "semanticfence-online-witness-bank-v1",
                "label_aggregation": (
                    "unique_row_safe_iff_all_incident_train_endpoint_observations_safe"
                ),
                "cell_key": ["layer", "expert_id", "abi"],
                "safe_rows": 1,
                "unsafe_rows": 1,
                "unique_rows": 2,
                "cells": [
                    {
                        "cell": [1, 2, "abi-v1"],
                        "safe_row_ids": ["row-a"],
                        "unsafe_row_ids": ["row-b"],
                        "safe_count": 1,
                        "unsafe_count": 1,
                        "safe_feature_digest": subject.gate.canonical_sha256(
                            [feature_rows[0]["feature_vector"]]
                        ),
                        "unsafe_feature_digest": subject.gate.canonical_sha256(
                            [feature_rows[1]["feature_vector"]]
                        ),
                    }
                ],
                "pre_outcome_features_sha256": subject.gate.sha256_file(
                    root / "PRE_OUTCOME_FEATURES.jsonl"
                ),
                "train_edge_results_sha256": subject.gate.sha256_file(
                    root / "TRAIN_EDGE_RESULTS.jsonl"
                ),
            }
            write_json(root / "TRAIN_WITNESS_BANK.json", expected_manifest)

            features, cells, banks = subject.reconstruct_bank(
                root, [edge], train_results
            )
            self.assertEqual(set(features), {"row-a", "row-b"})
            self.assertEqual(cells["row-a"], (1, 2, "abi-v1"))
            self.assertEqual(
                banks[(1, 2, "abi-v1")],
                {"safe": ["row-a"], "unsafe": ["row-b"]},
            )

            expected_manifest["cells"][0]["safe_row_ids"] = ["row-b"]
            expected_manifest["cells"][0]["unsafe_row_ids"] = ["row-a"]
            write_json(root / "TRAIN_WITNESS_BANK.json", expected_manifest)
            with self.assertRaisesRegex(subject.AuditError, "reconstructed train witness"):
                subject.reconstruct_bank(root, [edge], train_results)


class TimingTests(unittest.TestCase):
    def microcost(self) -> dict[str, object]:
        m1 = [8.0, 9.0, 10.0, 11.0, 12.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        m2 = [4.0, 5.0, 6.0, 7.0, 8.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        return {
            "candidate_edges": 2,
            "raw_aggregate": {
                "warmups": 3,
                "repeats": 10,
                "m1_two_single_calls_ms": m1,
                "m2_one_pair_call_ms": m2,
                "m1_median_ms": 10.0,
                "m2_median_ms": 6.0,
                "ratio_of_medians_m1_over_m2": 10.0 / 6.0,
            },
            "estimated_single_m1_ms": 2.5,
            "estimated_pair_m2_ms": 3.0,
        }

    def test_recomputes_microcost_from_raw_arrays(self) -> None:
        value = self.microcost()
        self.assertEqual(
            subject.recompute_microcost(value, expected_edges=2, label="test"),
            (2.5, 3.0),
        )
        value["estimated_single_m1_ms"] = 9.0
        with self.assertRaisesRegex(subject.AuditError, "estimated M1"):
            subject.recompute_microcost(value, expected_edges=2, label="test")

    def test_recomputes_online_overhead_median(self) -> None:
        schedule = [
            {"row_ids": ["a", "b"]},
            {"row_ids": ["b", "c"]},
        ]
        value = {
            "repeats": 5,
            "elapsed_ms": [1.0, 5.0, 3.0, 2.0, 4.0],
            "median_total_ms": 3.0,
            "candidate_edges": 2,
            "unique_endpoints": 3,
        }
        self.assertEqual(
            subject.recompute_online_overhead(value, schedule=schedule, label="test"),
            3.0,
        )
        value["median_total_ms"] = 4.0
        with self.assertRaisesRegex(subject.AuditError, "median"):
            subject.recompute_online_overhead(value, schedule=schedule, label="test")


class CompletionTests(unittest.TestCase):
    def make_complete(self, root: Path) -> dict[str, object]:
        for name in (
            "PRE_OUTCOME_LOCK.json",
            "VALIDATION_THRESHOLD.json",
            "TEST_ADMISSION_PLAN.json",
            "SUMMARY.json",
            "A.json",
        ):
            write_json(root / name, {"name": name})
        actual = subject._actual_artifact_hashes(root)
        return {
            "schema_version": subject.gate.COMPLETE_SCHEMA,
            "status": "SUCCESS_COMPLETE",
            "verdict": "PIVOT_TO_SHADOW_VERIFY",
            "completion_last": True,
            "pre_outcome_lock_sha256": actual["PRE_OUTCOME_LOCK.json"],
            "validation_threshold_sha256": actual["VALIDATION_THRESHOLD.json"],
            "test_admission_plan_sha256": actual["TEST_ADMISSION_PLAN.json"],
            "summary_sha256": actual["SUMMARY.json"],
            "artifact_sha256": actual,
            "paper_result": False,
        }

    def test_exact_complete_closure_and_direct_digests(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            complete = self.make_complete(root)
            write_json(root / "COMPLETE.json", complete)
            self.assertEqual(subject.validate_complete(root), complete)

            write_json(root / "UNDECLARED.json", {})
            with self.assertRaisesRegex(subject.AuditError, "exact artifact closure"):
                subject.validate_complete(root)

    def test_rejects_wrong_direct_digest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            complete = self.make_complete(root)
            complete["summary_sha256"] = "0" * 64
            write_json(root / "COMPLETE.json", complete)
            with self.assertRaisesRegex(subject.AuditError, "summary_sha256"):
                subject.validate_complete(root)

    def test_audit_output_must_be_outside_formal_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(subject.AuditError, "outside"):
                subject.ensure_output_outside_formal_root(root, root / "AUDIT.json")
            subject.ensure_output_outside_formal_root(root, root.parent / "AUDIT.json")


if __name__ == "__main__":
    unittest.main()
