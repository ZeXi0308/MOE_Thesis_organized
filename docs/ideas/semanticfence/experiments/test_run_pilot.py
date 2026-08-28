"""CPU-only integrity tests for the SemanticFence pilot runner."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("run_pilot_5090.py")
SPEC = importlib.util.spec_from_file_location("semanticfence_run_pilot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def frozen_config() -> dict:
    return {
        "schema_version": MODULE.SCHEMA,
        "status": "FROZEN",
        "evidence_boundary": "single_gpu_expert_stage_only",
        "model": {
            "repo_id": "allenai/OLMoE-1B-7B-0924",
            "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
            "local_path_candidates": ["/model"],
            "dtype": "bfloat16",
            "num_hidden_layers": 16,
            "hidden_size": 2048,
            "intermediate_size": 1024,
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "file_sha256": {"config.json": "a" * 64},
        },
        "data": {
            "calibration_manifest": "cal.jsonl",
            "calibration_manifest_sha256": "b" * 64,
            "calibration_document_count": 8,
            "calibration_provenance": "cal-provenance.json",
            "calibration_provenance_sha256": "d" * 64,
            "historical_hash_registry": "history.json",
            "historical_hash_registry_sha256": "e" * 64,
            "evaluation_manifest": "eval.jsonl",
            "evaluation_manifest_sha256": "c" * 64,
            "evaluation_document_count": 32,
            "evaluation_provenance": "eval-provenance.json",
            "evaluation_provenance_sha256": "f" * 64,
            "evaluation_exclusion_report": "exclusions.json",
            "evaluation_exclusion_report_sha256": "1" * 64,
            "evaluation_artifact_hashes": "artifact-hashes.json",
            "evaluation_artifact_hashes_sha256": "2" * 64,
            "token_offsets": [0, 256],
            "window_tokens": 16,
            "add_special_tokens": False,
            "calibration_positions": list(range(16)),
            "evaluation_position": 15,
        },
        "intervention": {
            "m_values": [1, 2, 4, 8, 16, 32, 64],
            "reference_m": 1,
            "fixed_padding_m": 64,
            "repeats": 10,
            "warmups": 3,
            "min_calibration_packs": 3,
            "min_calibration_documents": 3,
            "cublaslt_log_level": 5,
            "cublaslt_log_mask": 31,
        },
        "decision": {
            "minimum_unrestricted_mismatch_victims": 8,
            "minimum_semanticfence_covered_victims": 8,
            "minimum_distinct_admitted_m_gt_1": 2,
            "minimum_latency_reduction_fraction": 0.10,
        },
        "budget": {"max_gpu_seconds": 5400},
    }


def row(index: int, *, expert: int = 3):
    return MODULE.CONTRACT.RowRecord(
        split="evaluation",
        document_sha256=f"{index + 1:064x}",
        document_index=index,
        offset=0,
        token_position=15,
        layer=2,
        expert_id=expert,
        route_rank=1,
        hidden_sha256=f"{index + 100:064x}",
    )


BINDING_CONTEXT = {
    "config_sha256": "1" * 64,
    "stack_digest": "2" * 64,
    "model_bindings_sha256": "3" * 64,
    "source_bindings_sha256": "4" * 64,
    "math_state_sha256": "5" * 64,
    "hidden_size": 2048,
    "intermediate_size": 1024,
    "dtype": "bfloat16",
}


def call(call_index: int, rows, *, arm: str = "semanticfence", padding: int = 0):
    record = {
        "schema_version": MODULE.CALL_SCHEMA,
        "call_index": call_index,
        "arm": arm,
        "layer": rows[0].layer,
        "expert_id": rows[0].expert_id,
        "m": len(rows) + padding,
        "row_ids": [item.row_id for item in rows],
        "padding_rows": padding,
        "expected_signatures": [],
    }
    descriptor = MODULE._descriptor_from_call_fields(record, BINDING_CONTEXT)
    return record | {
        "pre_call_descriptor": descriptor,
        "pre_call_descriptor_sha256": MODULE.canonical_sha256(descriptor),
    }


class ConfigAndRawBitsTest(unittest.TestCase):
    def test_frozen_config_accepts_and_treatment_change_fails(self) -> None:
        config = frozen_config()
        self.assertEqual(MODULE.validate_config(config)["status"], "FROZEN")
        changed = copy.deepcopy(config)
        changed["intervention"]["reference_m"] = 64
        with self.assertRaisesRegex(MODULE.ProtocolError, "reference_m changed"):
            MODULE.validate_config(changed)

    def test_signed_zero_is_a_raw_bf16_mismatch(self) -> None:
        positive_zero = bytes.fromhex("0000")
        negative_zero = bytes.fromhex("0080")
        self.assertEqual(
            MODULE.strict_bf16_mismatch_count(positive_zero, negative_zero), 1
        )
        self.assertEqual(
            MODULE.strict_bf16_mismatch_count(positive_zero, positive_zero), 0
        )

    def test_raw_bf16_shape_mismatch_fails(self) -> None:
        with self.assertRaises(MODULE.ProtocolError):
            MODULE.strict_bf16_mismatch_count(b"\x00\x00", b"\x00")

    def test_acceptance_requires_valid_complete_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "ACCEPTANCE.json"
            stack_payload = {"gpu": {"uuid": "GPU-test"}}
            stack = stack_payload | {
                "stack_digest": MODULE.canonical_sha256(stack_payload)
            }
            path.write_text(
                json.dumps(
                    {
                        "schema_version": MODULE.ACCEPTANCE_SCHEMA,
                        "status": "ACCEPTED_REAL_GPU_CAPABILITY_ONLY",
                        "stack": stack,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MODULE.ProtocolError, "sentinel"):
                MODULE.load_acceptance(path)
            (root / "ACCEPTANCE_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "schema_version": MODULE.ACCEPTANCE_COMPLETE_SCHEMA,
                        "status": "COMPLETE",
                        "acceptance_sha256": MODULE.sha256_file(path),
                        "paper_result": False,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(MODULE.load_acceptance(path)["stack"], stack)


class LedgerAndDecisionTest(unittest.TestCase):
    def test_call_ledger_requires_exact_once_coverage(self) -> None:
        rows = [row(index) for index in range(3)]
        calls = [call(0, rows[:2]), call(1, rows[2:])]
        result = MODULE.validate_call_ledger(
            rows, calls, binding_context=BINDING_CONTEXT
        )
        self.assertEqual(result["row_count"], 3)

        duplicated = [call(0, rows[:2]), call(1, rows[1:])]
        with self.assertRaisesRegex(MODULE.ProtocolError, "duplicates"):
            MODULE.validate_call_ledger(
                rows, duplicated, binding_context=BINDING_CONTEXT
            )

    def test_padding_is_explicit_not_a_real_row(self) -> None:
        rows = [row(0), row(1)]
        result = MODULE.validate_call_ledger(
            rows, [call(0, rows, padding=62)], binding_context=BINDING_CONTEXT
        )
        self.assertEqual(result["row_count"], 2)

    def test_call_descriptor_tamper_fails_closed(self) -> None:
        rows = [row(0), row(1)]
        value = call(0, rows)
        value["pre_call_descriptor"]["m"] = 64
        with self.assertRaisesRegex(MODULE.ProtocolError, "descriptor"):
            MODULE.validate_call_ledger(
                rows, [value], binding_context=BINDING_CONTEXT
            )

    def test_paired_latency_uses_pairwise_ratios(self) -> None:
        reduction = MODULE.paired_latency_reduction([10.0, 20.0], [8.0, 16.0])
        self.assertAlmostEqual(reduction, 0.2)

    def test_decision_support_weaken_and_unable(self) -> None:
        config = frozen_config()
        positive = {
            "reference_all_stable": True,
            "unrestricted_mismatch_victims": 8,
            "semanticfence_mismatch_rows": 0,
            "semanticfence_covered_victims": 8,
            "semanticfence_distinct_m_gt_1": 2,
            "semanticfence_padding_rows": 0,
            "semanticfence_latency_reduction_fraction": 0.10,
            "fixed_control_dominates": False,
            "evidence_complete": True,
        }
        self.assertEqual(MODULE.decide_summary(positive, config), "SUPPORT")
        mismatch = positive | {"semanticfence_mismatch_rows": 1}
        self.assertEqual(MODULE.decide_summary(mismatch, config), "WEAKEN")
        incomplete = positive | {"evidence_complete": False}
        self.assertEqual(MODULE.decide_summary(incomplete, config), "UNABLE")


class CompletionAuthorityTest(unittest.TestCase):
    def test_complete_is_written_last_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "raw.json").write_text("{}\n", encoding="utf-8")
            MODULE.finalize_complete(
                output,
                {
                    "schema_version": MODULE.SUMMARY_SCHEMA,
                    "decision": "SUPPORT",
                },
            )
            complete = json.loads((output / "COMPLETE.json").read_text())
            self.assertEqual(complete["status"], "SUCCESS_COMPLETE")
            self.assertIn("summary.json", complete["artifact_sha256"])
            with self.assertRaises(MODULE.ProtocolError):
                MODULE.finalize_complete(output, {"decision": "WEAKEN"})


class TraceBindingTest(unittest.TestCase):
    def test_trace_and_numeric_output_hash_must_close(self) -> None:
        identity = {
            "call_index": 0,
            "arm": "semanticfence",
            "layer": 2,
            "expert_id": 3,
            "m": 2,
            "row_ids": ["a" * 64, "b" * 64],
        }
        trace = [identity | {"signature_sha256": "c" * 64}]
        outputs = [identity | {"full_output_sha256": "d" * 64}]
        numeric = [
            identity | {"representative_full_output_sha256": "d" * 64}
        ]
        merged = MODULE.bind_trace_to_numeric(
            trace_rows=trace,
            trace_call_outputs=outputs,
            numeric_calls=numeric,
        )
        self.assertEqual(merged[0]["signature_sha256"], "c" * 64)
        bad = [identity | {"full_output_sha256": "e" * 64}]
        with self.assertRaisesRegex(MODULE.ProtocolError, "output hash mismatch"):
            MODULE.bind_trace_to_numeric(
                trace_rows=trace,
                trace_call_outputs=bad,
                numeric_calls=numeric,
            )


if __name__ == "__main__":
    unittest.main()
