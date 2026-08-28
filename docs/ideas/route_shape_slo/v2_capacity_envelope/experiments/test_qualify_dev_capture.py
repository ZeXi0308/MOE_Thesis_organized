#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qualifier = load_module(HERE / "qualify_dev_capture.py", "qualify_dev_capture")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QualifierTest(unittest.TestCase):
    def reseal(self, capture: Path) -> None:
        complete = json.loads((capture / "CAPTURE_COMPLETE.json").read_text())
        complete["serial_audit"] = json.loads(
            (capture / "serial_audit.json").read_text()
        )
        complete["files"] = {
            name: sha256_file(capture / name) for name in complete["files"]
        }
        write_json(capture / "CAPTURE_COMPLETE.json", complete)

    def make_capture(self, root: Path) -> Path:
        capture = root / "capture"
        capture.mkdir()
        serial_audit = {
                "status": "PASS",
                "requests": 1,
                "steps": 1,
                "route_identity_match_fraction": 1.0,
                "token_match_fraction": 1.0,
                "route_identity_semantics": "per_layer_expert_assignment_multiset",
                "batch_dependent_route_observed": False,
                "topk_order_checked": True,
                "gate_weight_checked": False,
                "reference_type": (
                    "same-model serial cached-decode engineering equivalence"
                ),
                "scientific_ground_truth": False,
            }
        write_json(capture / "serial_audit.json", serial_audit)
        write_json(
            capture / "workload_manifest.json",
            {"route_capacity_envelope": {"episode_id": "e", "arrival_regime": "steady"}},
        )
        request = {
            "request_id": "r0", "document_id": "d0", "arrival_us": 0,
            "deadline_us": 100, "prompt_tokens": 2, "steps": [],
        }
        (capture / "request_ledger.jsonl").write_text(json.dumps(request) + "\n")
        batch = {
            "batch_index": 0, "start_us": 10, "end_us": 20, "batch_size": 1,
            "active_request_ids": ["r0"], "request_ids": ["r0"],
            "decode_steps": [0], "prior_cache_lengths": [2], "left_padding": [0],
        }
        (capture / "decode_batches.jsonl").write_text(json.dumps(batch) + "\n")
        with (capture / "routes.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "request_id", "decode_step", "layer_id", "topk_slot", "expert_id",
                "gate_weight", "layer_ready_us", "route_end_us", "document_id",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "request_id": "r0", "decode_step": 0, "layer_id": 0,
                "topk_slot": 0, "expert_id": 1, "gate_weight": 1.0,
                "layer_ready_us": 10, "route_end_us": 20, "document_id": "d0",
            })
        write_json(capture / "preregistration.json", {"run_class": "development"})
        write_json(capture / "environment.json", {"device": "fixture"})
        names = {
            "routes.csv", "decode_batches.jsonl", "request_ledger.jsonl",
            "workload_manifest.json", "preregistration.json", "environment.json",
            "serial_audit.json",
        }
        write_json(
            capture / "RUN_STATUS.json",
            {"status": "COMPLETE", "required_sentinel": "CAPTURE_COMPLETE.json"},
        )
        write_json(
            capture / "CAPTURE_COMPLETE.json",
            {
                "schema": "bcrd-continuous-capture-complete-v1",
                "status": "CAPTURE_COMPLETE",
                "serial_audit": serial_audit,
                "files": {name: sha256_file(capture / name) for name in names},
            },
        )
        return capture

    def test_alignment_passes_but_missing_hook_off_stays_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = self.make_capture(Path(temporary))
            result = qualifier.qualify(capture)
            self.assertEqual(result["status"], "P0_READY_WITH_PROXY_RUNTIME")
            self.assertTrue(result["exploratory_p1_ready"])
            self.assertFalse(result["representative_serving_p1_ready"])
            self.assertEqual(
                result["checks"]["route_request_step_latency_alignment"], "PASS"
            )
            self.assertEqual(result["checks"]["hook_no_hook_distortion"], "NOT_MEASURED")

    def test_fixed_batch_overhead_check_is_labeled_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = self.make_capture(root)
            overhead = root / "overhead.json"
            write_json(
                overhead,
                {
                    "schema": "route-capacity-envelope-telemetry-overhead-v1",
                    "status": "TELEMETRY_OVERHEAD_OK",
                    "token_output_match": True,
                    "logit_output_match": True,
                    "on_route_trace_stable": True,
                    "completion_trace_match": True,
                    "same_requests": True,
                    "same_batch_schedule": True,
                    "same_decode_steps": True,
                    "same_dtype": True,
                    "same_arrival_trace": False,
                    "arrival_policy_applied_to_timing": False,
                },
            )
            result = qualifier.qualify(capture, overhead)
            self.assertEqual(
                result["checks"]["hook_no_hook_distortion"],
                "PASS_FIXED_BATCH_PROXY",
            )

    def test_batch_dependent_route_is_exposed_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = self.make_capture(Path(temporary))
            audit = json.loads((capture / "serial_audit.json").read_text())
            audit.update(
                {
                    "status": "PASS_TOKEN_PARITY_ROUTE_BATCH_DEPENDENT",
                    "route_identity_match_fraction": 0.98,
                    "batch_dependent_route_observed": True,
                    "layers": 1,
                    "route_identity_semantics": (
                        "per_layer_expert_assignment_multiset"
                    ),
                    "topk_order_checked": True,
                    "reference_type": (
                        "same-model serial cached-decode conformance diagnostic"
                    ),
                }
            )
            write_json(capture / "serial_audit.json", audit)
            self.reseal(capture)
            result = qualifier.qualify(capture)
            self.assertTrue(result["batch_dependent_route_observed"])
            self.assertTrue(result["diagnostic_only"])
            self.assertEqual(
                result["checks"]["serial_route_conformance"], "BATCH_DEPENDENT"
            )

    def test_route_window_misalignment_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = self.make_capture(Path(temporary))
            rows = (capture / "routes.csv").read_text().replace(",20,d0", ",21,d0")
            (capture / "routes.csv").write_text(rows)
            self.reseal(capture)
            with self.assertRaisesRegex(qualifier.QualificationError, "misaligned"):
                qualifier.qualify(capture)

    def test_serial_audit_tamper_after_sentinel_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = self.make_capture(Path(temporary))
            audit = json.loads((capture / "serial_audit.json").read_text())
            audit["route_identity_match_fraction"] = 0.99
            write_json(capture / "serial_audit.json", audit)
            with self.assertRaisesRegex(qualifier.QualificationError, "hash mismatch"):
                qualifier.qualify(capture)


if __name__ == "__main__":
    unittest.main()
