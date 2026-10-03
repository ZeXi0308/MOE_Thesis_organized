#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_n0d_matched_router_gate.py"
spec = importlib.util.spec_from_file_location("run_n0d_matched_router_gate", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {SOURCE}")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


def row(
    *,
    request: str = "req-0",
    step: int,
    layer: int,
    logits: list[float],
    experts: list[int],
):
    return {
        "request_id": request,
        "decode_step": step,
        "layer": layer,
        "router_logits": logits,
        "router_logits_dtype_before_float32_copy": "torch.float32",
        "selected_experts": experts,
        "topk_margins": {"selection_boundary": 0.001},
    }


class N0dMatchedRouterGateTest(unittest.TestCase):
    def test_current_runtime_sources_are_hash_bound(self) -> None:
        root = next(parent for parent in SOURCE.parents if (parent / ".git").exists())
        identity = gate.validate_source_identity(root)
        self.assertEqual(identity["repo_head"], gate.EXPECTED_REPO_HEAD)
        self.assertEqual(
            set(identity["files_sha256"]), set(gate.REQUIRED_REPOSITORY_FILES)
        )

    def test_runtime_imports_bind_to_the_frozen_source_tree(self) -> None:
        root = next(parent for parent in SOURCE.parents if (parent / ".git").exists())
        comparator = gate.load_comparator(root)
        binding = gate.validate_import_binding(
            comparator, root, require_modeling=False
        )
        self.assertEqual(
            set(binding),
            {"comparator", "development_wrapper", "producer", "core"},
        )

    def test_frontier_orders_layer_before_request_within_a_step(self) -> None:
        serial = {
            "router": [
                row(
                    request="req-a",
                    step=0,
                    layer=1,
                    logits=[2.0, 1.0, 0.0],
                    experts=[0, 1],
                ),
                row(
                    request="req-z",
                    step=0,
                    layer=0,
                    logits=[2.0, 1.0, 0.0],
                    experts=[0, 1],
                ),
            ]
        }
        batched = {
            "router": [
                row(
                    request="req-a",
                    step=0,
                    layer=1,
                    logits=[2.0, 0.9, 1.1],
                    experts=[0, 2],
                ),
                row(
                    request="req-z",
                    step=0,
                    layer=0,
                    logits=[2.0, 0.9, 1.1],
                    experts=[0, 2],
                ),
            ]
        }
        result = gate.first_divergences(serial, batched)
        self.assertEqual(
            result["causal_frontier_signature"]["layer"], 0
        )
        self.assertEqual(
            result["causal_frontier_signature"]["records"][0]["request_id"],
            "req-z",
        )

    def test_earlier_logit_difference_classifies_later_assignment_as_pretopk(self) -> None:
        serial = {
            "router": [
                row(step=0, layer=0, logits=[2.0, 1.0, 0.0], experts=[0, 1]),
                row(step=0, layer=1, logits=[2.0, 1.0, 1.0], experts=[0, 1]),
            ]
        }
        batched = {
            "router": [
                row(step=0, layer=0, logits=[2.0, 1.0 + 1e-8, 0.0], experts=[0, 1]),
                row(step=0, layer=1, logits=[2.0, 1.0, 1.0], experts=[0, 2]),
            ]
        }
        signature = gate.first_divergence_signature(
            gate.first_divergences(serial, batched)
        )
        self.assertEqual(signature["category"], "PRE_TOPK_NUMERICAL_DIVERGENCE")
        self.assertEqual(
            signature["records"][0]["first_router_logit_position"],
            {"decode_step": 0, "layer": 0},
        )

    def test_prior_step_logit_difference_cannot_explain_later_step_assignment(self) -> None:
        serial = {
            "router": [
                row(step=0, layer=15, logits=[2.0, 1.0, 0.0], experts=[0, 1]),
                row(step=1, layer=0, logits=[2.0, 1.0, 1.0], experts=[0, 1]),
            ]
        }
        batched = {
            "router": [
                row(step=0, layer=15, logits=[2.0, 1.0 + 1e-8, 0.0], experts=[0, 1]),
                row(step=1, layer=0, logits=[2.0, 1.0, 1.0], experts=[0, 2]),
            ]
        }
        result = gate.first_divergences(serial, batched)
        signature = gate.first_divergence_signature(result)
        self.assertEqual(signature["category"], "RECONSTRUCTED_TOPK_INCONSISTENCY")
        self.assertIsNone(signature["records"][0]["first_router_logit_position"])
        later_cell = next(
            row
            for row in result["per_request_step"]
            if row["request_id"] == "req-0" and row["decode_step"] == 1
        )
        self.assertIsNone(later_cell["first_router_logit_value_difference"])

    def test_later_layer_logit_difference_cannot_explain_earlier_assignment(self) -> None:
        serial = {
            "router": [
                row(step=0, layer=1, logits=[2.0, 1.0, 1.0], experts=[0, 1]),
                row(step=0, layer=2, logits=[2.0, 1.0, 0.0], experts=[0, 1]),
            ]
        }
        batched = {
            "router": [
                row(step=0, layer=1, logits=[2.0, 1.0, 1.0], experts=[0, 2]),
                row(step=0, layer=2, logits=[2.0, 1.0 + 1e-8, 0.0], experts=[0, 1]),
            ]
        }
        signature = gate.first_divergence_signature(
            gate.first_divergences(serial, batched)
        )
        self.assertEqual(signature["category"], "RECONSTRUCTED_TOPK_INCONSISTENCY")
        self.assertIsNone(signature["records"][0]["first_router_logit_position"])

    def test_missing_raw_router_dtype_fails_closed(self) -> None:
        serial_row = row(step=0, layer=0, logits=[2.0, 1.0, 0.0], experts=[0, 1])
        batch_row = row(step=0, layer=0, logits=[2.0, 0.9, 1.1], experts=[0, 2])
        serial_row.pop("router_logits_dtype_before_float32_copy")
        batch_row.pop("router_logits_dtype_before_float32_copy")
        with self.assertRaisesRegex(gate.GateError, "source dtype"):
            gate.first_divergences(
                {"router": [serial_row]}, {"router": [batch_row]}
            )

    def test_simultaneous_requests_are_kept_as_a_frontier_set(self) -> None:
        serial = {
            "router": [
                row(request=request, step=0, layer=0, logits=[2.0, 1.0, 0.0], experts=[0, 1])
                for request in ("req-a", "req-b")
            ]
        }
        batched = {
            "router": [
                row(request=request, step=0, layer=0, logits=[2.0, 0.9, 1.1], experts=[0, 2])
                for request in ("req-a", "req-b")
            ]
        }
        signature = gate.first_divergence_signature(
            gate.first_divergences(serial, batched)
        )
        self.assertEqual(
            [record["request_id"] for record in signature["records"]],
            ["req-a", "req-b"],
        )

    def test_gate_precedence_fails_negative_control_before_token_parity(self) -> None:
        status = gate.classify_gate(
            source_batch_dependence=True,
            serial_negative_control_exact=False,
            token_parity=False,
            assignment_changed=False,
            double_sided_signature_match=False,
            first_category=None,
        )
        self.assertEqual(status, "INVALID_SERIAL_NEGATIVE_CONTROL")

    def test_gate_accepts_only_fully_qualified_comparator_status(self) -> None:
        status = gate.classify_gate(
            source_batch_dependence=True,
            serial_negative_control_exact=True,
            token_parity=True,
            assignment_changed=True,
            double_sided_signature_match=True,
            first_category="PRE_TOPK_NUMERICAL_DIVERGENCE",
        )
        self.assertEqual(
            status,
            "PROCESS_CANDIDATE_PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION",
        )

    def test_sub_tolerance_nonzero_delta_is_still_pre_topk(self) -> None:
        serial = {
            "router": [row(step=0, layer=3, logits=[2.0, 1.0, 0.0], experts=[0, 1])]
        }
        batched = {
            "router": [
                row(step=0, layer=3, logits=[2.0, 1.0 - 1e-8, 1e-8], experts=[0, 2])
            ]
        }
        signature = gate.first_divergence_signature(
            gate.first_divergences(serial, batched)
        )
        self.assertEqual(signature["category"], "PRE_TOPK_NUMERICAL_DIVERGENCE")


if __name__ == "__main__":
    unittest.main()
