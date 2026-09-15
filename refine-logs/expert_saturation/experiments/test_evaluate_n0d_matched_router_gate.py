#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("evaluate_n0d_matched_router_gate.py")
SPEC = importlib.util.spec_from_file_location("evaluate_n0d_matched_router_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _base_logits() -> list[float]:
    logits = [1.0 - 0.01 * index for index in range(MODULE.EXPERTS)]
    # Keep experts 7 and 8 exactly tied at the top-k boundary.  This makes both
    # possible tie selections valid while allowing the numerical fixture below
    # to move expert 8 strictly above expert 7.
    boundary = (logits[MODULE.TOP_K - 1] + logits[MODULE.TOP_K]) / 2.0
    logits[MODULE.TOP_K - 1] = boundary
    logits[MODULE.TOP_K] = boundary
    return logits


def _token_rows() -> list[dict[str, object]]:
    return [
        {
            "request_id": request_id,
            "decode_step": step,
            "input_token_id": 1000 + request_index * 100 + step,
            "predicted_next_token_id": 1001 + request_index * 100 + step,
        }
        for request_index, request_id in enumerate(MODULE.REQUEST_IDS)
        for step in range(MODULE.DECODE_STEPS)
    ]


def _router_rows(arm: str, *, divergence: bool, numerical: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tokens = {(row["request_id"], row["decode_step"]): row for row in _token_rows()}
    first_key = (MODULE.REQUEST_IDS[0], 0, 0)
    for request_id in MODULE.REQUEST_IDS:
        for step in range(MODULE.DECODE_STEPS):
            token = tokens[(request_id, step)]
            for layer in range(MODULE.LAYERS):
                logits = _base_logits()
                experts = list(range(MODULE.TOP_K))
                if arm == "batch_4" and divergence and (request_id, step, layer) == first_key:
                    experts[-1] = MODULE.TOP_K
                    if numerical:
                        logits[MODULE.TOP_K - 1] -= 1e-8
                        logits[MODULE.TOP_K] += 1e-8
                rows.append(
                    {
                        "request_id": request_id,
                        "decode_step": step,
                        "layer": layer,
                        "input_token_id": token["input_token_id"],
                        "predicted_next_token_id": token["predicted_next_token_id"],
                        "router_logits": logits,
                        "router_logits_dtype_before_float32_copy": "torch.float32",
                        "selected_experts": experts,
                    }
                )
    return rows


def _payload(process: int, *, divergence: bool = True, numerical: bool = True) -> dict[str, object]:
    tokens = _token_rows()
    traces = {
        arm: {
            "tokens": copy.deepcopy(tokens),
            "router": _router_rows(arm, divergence=divergence, numerical=numerical),
        }
        for arm in MODULE.ARMS
    }
    prompt_rows = [
        {
            "request_id": request_id,
            "prompt_token_count": count,
            "prompt_token_ids_sha256": digest,
        }
        for request_id, count, digest in MODULE.PROMPT_IDENTITY
    ]
    runner_sha = MODULE._sha256(MODULE.RUNNER_PATH)
    capture_audit = {"batch_dependent_route_observed": True, "status": "fixture"}
    if not divergence:
        runner_status = "STOP_MATCHED_PRESTATE_NO_ASSIGNMENT_DIVERGENCE"
    elif numerical:
        runner_status = "PROCESS_CANDIDATE_PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION"
    else:
        runner_status = "INCONCLUSIVE_RECONSTRUCTED_TOPK_INCONSISTENCY"
    return {
        "schema": MODULE.ARM_SCHEMA,
        "status": runner_status,
        "claim_ceiling": MODULE.CLAIM_CEILING,
        "capacity_claim_authorized": False,
        "action_oracle_authorized": False,
        "controller_authorized": False,
        "source_identity": {
            "repo_head": MODULE.EXPECTED_REPO_HEAD,
            "relevant_paths_clean": True,
            "files_sha256": copy.deepcopy(MODULE.SOURCE_FILES),
            "runner": f"/fixture/{MODULE.RUNNER_PATH.name}",
            "runner_sha256": runner_sha,
            "actual_import_paths": {
                role: f"/fixture/source/{relative}"
                for role, relative in MODULE.IMPORT_RELATIVE_PATHS.items()
            },
        },
        "fresh_capture": {
            "capture_dir": "/fixture/capture",
            "capture_complete_sha256": "c" * 64,
            "serial_audit": capture_audit,
            "source_batch_dependence": True,
            "reference_tokens": copy.deepcopy(tokens),
        },
        "execution": {
            "process_identity": {
                "pid": 1000 + process,
                "start_time_ticks": 2000 + process,
                "boot_id": "12345678-1234-1234-1234-123456789abc",
            },
            "requests": len(MODULE.REQUEST_IDS),
            "request_ids": list(MODULE.REQUEST_IDS),
            "decode_steps": MODULE.DECODE_STEPS,
            "batch_width": len(MODULE.REQUEST_IDS),
            "fresh_process_repeat": process,
            "planned_fresh_process_repeats": len(MODULE.PROCESSES),
            "arm_order": list(MODULE.ARM_ORDERS[process]),
            "canonical_state_advance": "serial_a_only",
            "batch_state_propagated_to_next_step": False,
            "matched_prestate_fork_checks": [
                {
                    "arms": sorted(MODULE.ARMS),
                    "requests": len(MODULE.REQUEST_IDS),
                    "independent_equal_tensors_checked": 384,
                    "independent_equal_elements_checked": 1024,
                }
                for _ in range(MODULE.DECODE_STEPS)
            ],
            "seed_reset_before_process_trajectory": 20260812,
            "warmup_trajectory_discarded": True,
            "runtime_validation": {"status": "MATCH_CAPTURED_RUNTIME"},
            "environment": {"torch": "fixture", "gpu": "fixture"},
            "token_identity": {
                "tokenizer": {"revision": "fixture"},
                "selected_request_prompt_tokens": prompt_rows,
            },
            "gpu_isolation": {
                "status": "PASS_SAMPLED_PROCESS_ISOLATION",
                "violations": [],
            },
        },
        "serial_negative_control": {"exact": True, "comparison": {}},
        "reference_token_parity": {"passed": True},
        "serial_vs_batch_4": {},
        "traces": traces,
    }


def _campaign(**kwargs: object) -> list[dict[str, object]]:
    return [_payload(process, **kwargs) for process in MODULE.PROCESSES]


def _sealed_capture() -> dict[str, object]:
    payload = _payload(0)
    capture = payload["fresh_capture"]
    return {
        "schema": MODULE.CAPTURE_CONTRACT.SCHEMA,
        "capture_dir": capture["capture_dir"],
        "capture_complete_sha256": capture["capture_complete_sha256"],
        "source_batch_dependence": capture["source_batch_dependence"],
        "serial_audit": copy.deepcopy(capture["serial_audit"]),
        "request_ids": list(MODULE.REQUEST_IDS),
        "decode_steps": MODULE.DECODE_STEPS,
        "reference_tokens": copy.deepcopy(capture["reference_tokens"]),
        "workload_manifest_sha256": "a" * 64,
        "request_ledger_sha256": "b" * 64,
        "serial_audit_sha256": "d" * 64,
    }


def _evaluate(payloads: list[dict[str, object]]) -> dict[str, object]:
    return MODULE.evaluate_payloads(
        payloads,
        sealed_capture=_sealed_capture(),
    )


def _router_row(payload: dict[str, object], arm: str, key: tuple[str, int, int]) -> dict[str, object]:
    rows = payload["traces"][arm]["router"]  # type: ignore[index]
    return next(
        row
        for row in rows
        if (row["request_id"], row["decode_step"], row["layer"]) == key
    )


def _normalized_trace(payload: dict[str, object], arm: str) -> dict[str, object]:
    errors: list[str] = []
    trace = MODULE._normalize_trace(payload["traces"][arm], arm, errors)  # type: ignore[index]
    if errors:
        raise AssertionError(errors)
    return trace


def _make_trace_token_parity_fail(payloads: list[dict[str, object]]) -> None:
    request_id = MODULE.REQUEST_IDS[0]
    for payload in payloads:
        for arm in MODULE.ARMS:
            trace = payload["traces"][arm]  # type: ignore[index]
            token = next(
                row
                for row in trace["tokens"]
                if row["request_id"] == request_id and row["decode_step"] == 0
            )
            token["input_token_id"] += 1
            for row in trace["router"]:
                if row["request_id"] == request_id and row["decode_step"] == 0:
                    row["input_token_id"] += 1
        payload["reference_token_parity"]["passed"] = False  # type: ignore[index]
        payload["status"] = "STOP_TOKEN_PARITY_FAILED"


class N0dEvaluatorTest(unittest.TestCase):
    def test_positive_requires_three_reproducible_double_sided_processes(self) -> None:
        report = _evaluate(_campaign())
        self.assertEqual(
            report["status"],
            "PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED",
        )
        self.assertTrue(report["structurally_valid"])
        self.assertTrue(report["checks"]["identical_double_sided_signature_across_processes"])
        self.assertFalse(report["capacity_claim_authorized"])
        self.assertFalse(report["action_oracle_authorized"])
        self.assertFalse(report["controller_authorized"])
        self.assertTrue(
            report["checks"]["selected_experts_topk_value_consistent"]
        )
        self.assertFalse(
            report["checks"]["selected_experts_exact_tie_break_recomputed"]
        )
        self.assertGreater(report["checks"]["exact_topk_boundary_tie_rows"], 0)

    def test_selected_experts_must_be_topk_value_consistent_with_router_logits(self) -> None:
        payloads = _campaign()
        key = (MODULE.REQUEST_IDS[1], 1, 1)
        row = _router_row(payloads[0], "serial_a", key)
        row["selected_experts"][-1] = MODULE.EXPERTS - 1
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any(
                "selected_experts_not_topk_value_consistent_with_router_logits"
                in error
                for error in report["errors"]
            )
        )

    def test_exact_boundary_tie_accepts_either_valid_topk_member(self) -> None:
        payloads = _campaign(numerical=False)
        key = (MODULE.REQUEST_IDS[0], 0, 0)
        serial = _router_row(payloads[0], "serial_a", key)
        batch = _router_row(payloads[0], "batch_4", key)
        self.assertEqual(
            serial["router_logits"][MODULE.TOP_K - 1],
            batch["router_logits"][MODULE.TOP_K],
        )
        self.assertEqual(_evaluate(payloads)["status"], "INCONCLUSIVE")

    def test_no_assignment_divergence_is_a_closed_negative_result(self) -> None:
        report = _evaluate(_campaign(divergence=False))
        self.assertEqual(report["status"], "NO_DIVERGENCE")
        self.assertTrue(report["checks"]["no_assignment_divergence"])

    def test_exact_logit_tie_with_assignment_change_is_inconclusive(self) -> None:
        report = _evaluate(_campaign(numerical=False))
        self.assertEqual(report["status"], "INCONCLUSIVE")

    def test_serial_control_has_precedence_over_every_scientific_failure(self) -> None:
        payloads = _campaign()
        key = (MODULE.REQUEST_IDS[0], 0, 0)
        _router_row(payloads[0], "serial_b", key)["selected_experts"][-1] = MODULE.TOP_K
        _make_trace_token_parity_fail(payloads)
        payloads[0]["serial_negative_control"]["exact"] = False  # type: ignore[index]
        payloads[0]["status"] = "INVALID_SERIAL_NEGATIVE_CONTROL"
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "SERIAL_CONTROL_UNSTABLE")

    def test_reference_token_parity_precedes_cross_process_stability(self) -> None:
        payloads = _campaign()
        _make_trace_token_parity_fail(payloads)
        key = (MODULE.REQUEST_IDS[1], 1, 1)
        _router_row(payloads[1], "batch_4", key)["router_logits"][0] += 1e-2
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "TOKEN_PARITY_FAILED")

    def test_reported_control_booleans_cannot_override_or_hide_the_gate(self) -> None:
        payloads = _campaign()
        payloads[0]["serial_negative_control"]["exact"] = False  # type: ignore[index]
        self.assertEqual(
            _evaluate(payloads)["status"], "SERIAL_CONTROL_UNSTABLE"
        )
        payloads = _campaign()
        payloads[0]["reference_token_parity"]["passed"] = False  # type: ignore[index]
        self.assertEqual(_evaluate(payloads)["status"], "TOKEN_PARITY_FAILED")

    def test_cross_process_selected_expert_drift_fails_closed(self) -> None:
        payloads = _campaign()
        key = (MODULE.REQUEST_IDS[1], 1, 1)
        _router_row(payloads[2], "batch_4", key)["selected_experts"][-1] = MODULE.TOP_K
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "CROSS_PROCESS_UNSTABLE")

    def test_cross_process_router_logit_allclose_accepts_sub_tolerance_noise(self) -> None:
        payloads = _campaign()
        for arm in MODULE.ARMS:
            for row in payloads[1]["traces"][arm]["router"]:  # type: ignore[index]
                row["router_logits"][12] += 5e-7
        report = _evaluate(payloads)
        self.assertEqual(
            report["status"],
            "PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED",
        )

    def test_cross_process_router_logit_material_drift_fails(self) -> None:
        payloads = _campaign()
        key = (MODULE.REQUEST_IDS[1], 1, 1)
        _router_row(payloads[1], "batch_4", key)["router_logits"][0] += 1e-3
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "CROSS_PROCESS_UNSTABLE")

    def test_category_drift_makes_first_divergence_inconsistent(self) -> None:
        payloads = _campaign()
        key = (MODULE.REQUEST_IDS[0], 0, 0)
        batch = _router_row(payloads[1], "batch_4", key)
        serial = _router_row(payloads[1], "serial_a", key)
        batch["router_logits"] = copy.deepcopy(serial["router_logits"])
        payloads[1]["status"] = "INCONCLUSIVE_RECONSTRUCTED_TOPK_INCONSISTENCY"
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "INCONSISTENT_FIRST_DIVERGENCE")

    def test_frontier_orders_layers_before_cross_request_lexical_order(self) -> None:
        payload = _payload(0, divergence=False)
        earlier_request_later_layer = (MODULE.REQUEST_IDS[0], 0, 5)
        later_request_earlier_layer = (MODULE.REQUEST_IDS[1], 0, 1)
        for key in (earlier_request_later_layer, later_request_earlier_layer):
            row = _router_row(payload, "batch_4", key)
            row["selected_experts"][-1] = 8
            row["router_logits"][8] += 1e-8
        signature = MODULE._first_signature(
            _normalized_trace(payload, "serial_a"),
            _normalized_trace(payload, "batch_4"),
        )
        self.assertEqual((signature["decode_step"], signature["layer"]), (0, 1))
        self.assertEqual(
            [row["request_id"] for row in signature["records"]],
            [MODULE.REQUEST_IDS[1]],
        )

    def test_earlier_logit_delta_qualifies_a_later_assignment_as_pretopk(self) -> None:
        payload = _payload(0, divergence=False)
        logit_key = (MODULE.REQUEST_IDS[0], 0, 1)
        assignment_key = (MODULE.REQUEST_IDS[0], 0, 3)
        _router_row(payload, "batch_4", logit_key)["router_logits"][20] += 1e-8
        _router_row(payload, "batch_4", assignment_key)["selected_experts"][-1] = 8
        signature = MODULE._first_signature(
            _normalized_trace(payload, "serial_a"),
            _normalized_trace(payload, "batch_4"),
        )
        self.assertEqual(signature["category"], "PRE_TOPK_NUMERICAL_DIVERGENCE")
        self.assertEqual((signature["decode_step"], signature["layer"]), (0, 3))
        self.assertEqual(
            signature["records"][0]["first_router_logit_position"],
            {"decode_step": 0, "layer": 1},
        )

    def test_prior_step_logit_delta_cannot_qualify_later_step_assignment(self) -> None:
        payload = _payload(0, divergence=False)
        prior_logit = (MODULE.REQUEST_IDS[0], 0, MODULE.LAYERS - 1)
        later_assignment = (MODULE.REQUEST_IDS[0], 1, 0)
        _router_row(payload, "batch_4", prior_logit)["router_logits"][20] += 1e-8
        _router_row(payload, "batch_4", later_assignment)["selected_experts"][-1] = 8
        signature = MODULE._first_signature(
            _normalized_trace(payload, "serial_a"),
            _normalized_trace(payload, "batch_4"),
        )
        self.assertEqual(signature["category"], "RECONSTRUCTED_TOPK_INCONSISTENCY")
        self.assertIsNone(signature["records"][0]["first_router_logit_position"])

    def test_later_layer_logit_delta_cannot_qualify_earlier_assignment(self) -> None:
        payload = _payload(0, divergence=False)
        assignment = (MODULE.REQUEST_IDS[0], 0, 1)
        later_logit = (MODULE.REQUEST_IDS[0], 0, 2)
        _router_row(payload, "batch_4", assignment)["selected_experts"][-1] = 8
        _router_row(payload, "batch_4", later_logit)["router_logits"][20] += 1e-8
        signature = MODULE._first_signature(
            _normalized_trace(payload, "serial_a"),
            _normalized_trace(payload, "batch_4"),
        )
        self.assertEqual(signature["category"], "RECONSTRUCTED_TOPK_INCONSISTENCY")
        self.assertIsNone(signature["records"][0]["first_router_logit_position"])

    def test_simultaneous_global_frontier_preserves_all_request_records(self) -> None:
        payload = _payload(0, divergence=False)
        for request_id in MODULE.REQUEST_IDS[:2]:
            row = _router_row(payload, "batch_4", (request_id, 0, 2))
            row["selected_experts"][-1] = 8
            row["router_logits"][8] += 1e-8
        signature = MODULE._first_signature(
            _normalized_trace(payload, "serial_a"),
            _normalized_trace(payload, "batch_4"),
        )
        self.assertEqual((signature["decode_step"], signature["layer"]), (0, 2))
        self.assertEqual(
            [row["request_id"] for row in signature["records"]],
            list(MODULE.REQUEST_IDS[:2]),
        )

    def test_source_hash_drift_is_invalid(self) -> None:
        payloads = _campaign()
        payloads[1]["source_identity"]["files_sha256"][next(iter(MODULE.SOURCE_FILES))] = "0" * 64  # type: ignore[index]
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("source_file_identity" in error for error in report["errors"]))

    def test_synchronized_embedded_reference_forgery_cannot_replace_sealed_ledger(self) -> None:
        payloads = _campaign()
        for payload in payloads:
            payload["fresh_capture"]["reference_tokens"][0]["input_token_id"] += 1  # type: ignore[index]
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(
            any("reference_tokens_not_bound_to_sealed_ledger" in error for error in report["errors"])
        )

    def test_sealed_capture_contract_is_mandatory(self) -> None:
        report = MODULE.evaluate_payloads(_campaign(), sealed_capture=None)
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("sealed_capture_contract_missing", report["errors"])

    def test_payload_capture_hash_and_path_must_match_independent_capture(self) -> None:
        payloads = _campaign()
        for payload in payloads:
            payload["fresh_capture"]["capture_complete_sha256"] = "0" * 64  # type: ignore[index]
            payload["fresh_capture"]["capture_dir"] = "/fixture/other-capture"  # type: ignore[index]
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("capture_hash_not_bound" in error for error in report["errors"]))
        self.assertTrue(any("capture_path_not_bound" in error for error in report["errors"]))

    def test_runner_hash_drift_is_invalid(self) -> None:
        payloads = _campaign()
        payloads[2]["source_identity"]["runner_sha256"] = "0" * 64  # type: ignore[index]
        self.assertEqual(_evaluate(payloads)["status"], "INVALID")

    def test_false_source_capture_dependence_is_invalid(self) -> None:
        payloads = _campaign()
        for payload in payloads:
            payload["fresh_capture"]["source_batch_dependence"] = False  # type: ignore[index]
            payload["fresh_capture"]["serial_audit"]["batch_dependent_route_observed"] = False  # type: ignore[index]
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("source_capture_batch_dependence_not_true" in error for error in report["errors"]))

    def test_actual_import_path_drift_is_invalid(self) -> None:
        payloads = _campaign()
        payloads[0]["source_identity"]["actual_import_paths"]["modeling"] = (  # type: ignore[index]
            "/fixture/other/experiments/shared/modeling.py"
        )
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("actual_import_source_root_mismatch" in error for error in report["errors"]))

    def test_router_raw_dtype_is_required_and_arm_bound(self) -> None:
        for mutation in ("missing", "mismatch"):
            with self.subTest(mutation=mutation):
                payloads = _campaign()
                key = (MODULE.REQUEST_IDS[0], 0, 0)
                row = _router_row(payloads[0], "batch_4", key)
                if mutation == "missing":
                    row.pop("router_logits_dtype_before_float32_copy")
                else:
                    row["router_logits_dtype_before_float32_copy"] = "torch.bfloat16"
                report = _evaluate(payloads)
                self.assertEqual(report["status"], "INVALID")
                self.assertTrue(any("router_raw_dtype" in error for error in report["errors"]))

    def test_runner_status_is_recomputed_not_trusted(self) -> None:
        payloads = _campaign()
        payloads[1]["status"] = "PROCESS_CANDIDATE_PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_TAMPERED"
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("runner_status_mismatch" in error for error in report["errors"]))

    def test_process_indices_must_be_exactly_zero_one_two(self) -> None:
        payloads = _campaign()
        payloads[2]["execution"]["fresh_process_repeat"] = 1  # type: ignore[index]
        self.assertEqual(_evaluate(payloads)["status"], "INVALID")

    def test_fresh_process_identities_must_be_unique(self) -> None:
        payloads = _campaign()
        payloads[2]["execution"]["process_identity"] = copy.deepcopy(  # type: ignore[index]
            payloads[1]["execution"]["process_identity"]  # type: ignore[index]
        )
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("fresh_process_identities_not_unique", report["errors"])

    def test_fresh_process_identities_must_share_one_boot(self) -> None:
        payloads = _campaign()
        payloads[2]["execution"]["process_identity"]["boot_id"] = (  # type: ignore[index]
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )
        report = _evaluate(payloads)
        self.assertEqual(report["status"], "INVALID")
        self.assertIn("fresh_process_boot_id_drift", report["errors"])

    def test_process_identity_fields_are_fail_closed(self) -> None:
        for field, value in (
            ("pid", 0),
            ("start_time_ticks", "12"),
            ("boot_id", "not-a-boot-id"),
        ):
            with self.subTest(field=field):
                payloads = _campaign()
                payloads[0]["execution"]["process_identity"][field] = value  # type: ignore[index]
                report = _evaluate(payloads)
                self.assertEqual(report["status"], "INVALID")
                self.assertTrue(any("invalid_process_identity" in error for error in report["errors"]))

    def test_arm_order_is_bound_to_process_index(self) -> None:
        payloads = _campaign()
        payloads[1]["execution"]["arm_order"] = list(MODULE.ARM_ORDERS[0])  # type: ignore[index]
        self.assertEqual(_evaluate(payloads)["status"], "INVALID")

    def test_exactly_three_inputs_are_required(self) -> None:
        report = _evaluate(_campaign()[:2])
        self.assertEqual(report["status"], "INVALID")
        self.assertTrue(any("expected_exactly_3_inputs" in error for error in report["errors"]))

    def test_any_attempted_claim_unlock_is_invalid(self) -> None:
        for field in ("capacity_claim_authorized", "action_oracle_authorized", "controller_authorized"):
            with self.subTest(field=field):
                payloads = _campaign()
                payloads[0][field] = True
                self.assertEqual(_evaluate(payloads)["status"], "INVALID")

    def test_cli_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "verdict.json"
            output.write_text("sentinel", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--input",
                    str(root / "missing-0.json"),
                    "--input",
                    str(root / "missing-1.json"),
                    "--input",
                    str(root / "missing-2.json"),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel")

    def test_atomic_writer_publishes_once_without_temp_residue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "verdict.json"
            MODULE._write_once(output, {"status": "fixture"})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"status": "fixture"})
            self.assertEqual([path.name for path in root.iterdir()], ["verdict.json"])
            with self.assertRaises(FileExistsError):
                MODULE._write_once(output, {"status": "replacement"})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"status": "fixture"})


if __name__ == "__main__":
    unittest.main()
