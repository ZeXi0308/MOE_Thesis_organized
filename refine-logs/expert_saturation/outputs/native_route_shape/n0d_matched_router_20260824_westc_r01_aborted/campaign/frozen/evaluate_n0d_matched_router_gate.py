#!/usr/bin/env python3
"""Fail-closed evaluator for the three-process N0d matched-prestate Gate.

The evaluator recomputes every scientific control from the retained traces.  A
runner status or boolean is never accepted as evidence by itself.  This Gate
localizes execution-conformance divergence only; it cannot authorize a
capacity claim, action Oracle, scheduling policy, or Controller.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA = "n0d-matched-router-evaluation-v1"
ARM_SCHEMA = "n0d-matched-prestate-router-gate-v1"
CLAIM_CEILING = "CUSTOM_TRANSFORMERS_MATCHED_PRESTATE_CONFORMANCE_ONLY"
EXPECTED_REPO_HEAD = "b141c1d587fe2c918643c3c7c3a8f5f5157d4c8a"
PROCESSES = (0, 1, 2)
ARMS = ("serial_a", "batch_4", "serial_b")
ARM_ORDERS = (
    ("serial_a", "batch_4", "serial_b"),
    ("batch_4", "serial_b", "serial_a"),
    ("serial_b", "serial_a", "batch_4"),
)
REQUEST_IDS = tuple(f"olmoe-dev-steady-{index:03d}" for index in range(4))
PROMPT_IDENTITY = (
    ("olmoe-dev-steady-000", 123, "1cf194a163db3212363e7859c61ad6329be8744cd22f9317ef7152f5be267bd0"),
    ("olmoe-dev-steady-001", 6, "01e1061c59a58212858a1216d5cf732d1b545d484cd895406bfbe14f79bb5120"),
    ("olmoe-dev-steady-002", 128, "26fe688dc850010138ebfe61921f3821d0f3c646aad7e18a2757808ea1750921"),
    ("olmoe-dev-steady-003", 9, "22464a4f0ca50b6539504f30052f6915e82e4c73b91e7bf95d7bbf6fb38046ad"),
)
SOURCE_FILES = {
    "docs/ideas/bcrd/experiments/capture_continuous_decode.py": "564d9fb6734462789eaca9bf0cf5cfd1ff8a04271a923cacf021015c6893b2db",
    "docs/ideas/bcrd/experiments/configs/gate0_continuous_decode_v1.json": "5664e1e457548b6564a1bf3d24af5c3d2d98c1d1ddbd6510a93556ea49042de4",
    "docs/ideas/bcrd/experiments/configs/workloads/olmoe.formal.json": "2bf4b4897c15b165fea90d730ed9136d0777535daab7f6807336c09a7c70cdbe",
    "docs/ideas/bcrd/experiments/core.py": "9115acf75ab60eeb9145521e1de7fb8be14455c85b7d1d85a4ad6ac7ab8be575",
    "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/capture_dev_continuous_decode.py": "5cda4159c94662e07efc07ff02ba42df31f1c9c5268b37d6757ff378df490f86",
    "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/compare_serial_batched_router_logits.py": "ab4cb2e1f3091d55f8f1952b00de4a45673f52465211d8d70a6c19e0a816cd1b",
    "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/olmoe_dev_workload.json": "0fc7fccd168b231d62812907cf1df6d352a130798fa4b7efbe521b9c62be60f6",
    "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/prepare_dev_workloads.py": "1d3f753fa837222cfe892e9ce2cc9cbe3c3d14aff2466503cfbe76a711a01d88",
    "experiments/shared/modeling.py": "f98269bd3084988cc952a272c8d6eec97f50e189d7689b9c581c2c170c4a623e",
}
DECODE_STEPS = 8
LAYERS = 16
EXPERTS = 64
TOP_K = 8
LOGIT_ATOL = 1e-6
LOGIT_RTOL = 1e-5
ROUTER_RAW_DTYPES = frozenset(
    {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
)
IMPORT_RELATIVE_PATHS = {
    "comparator": (
        "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/"
        "compare_serial_batched_router_logits.py"
    ),
    "development_wrapper": (
        "docs/ideas/route_shape_slo/v2_capacity_envelope/experiments/"
        "capture_dev_continuous_decode.py"
    ),
    "producer": "docs/ideas/bcrd/experiments/capture_continuous_decode.py",
    "core": "docs/ideas/bcrd/experiments/core.py",
    "modeling": "experiments/shared/modeling.py",
}
RUNNER_PATH = Path(__file__).with_name("run_n0d_matched_router_gate.py")
CAPTURE_CONTRACT_PATH = Path(__file__).with_name("n0d_capture_contract.py")


def _load_capture_contract_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "n0d_capture_contract_for_evaluator", CAPTURE_CONTRACT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import capture contract: {CAPTURE_CONTRACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE_CONTRACT = _load_capture_contract_module()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _allclose(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        abs(a - b) <= LOGIT_ATOL + LOGIT_RTOL * abs(b)
        for a, b in zip(left, right)
    )


def _token_index(
    value: Any, label: str, errors: list[str]
) -> dict[tuple[str, int], tuple[int, int]]:
    result: dict[tuple[str, int], tuple[int, int]] = {}
    if not isinstance(value, list):
        errors.append(f"{label}:tokens_not_list")
        return result
    for row_index, raw in enumerate(value):
        row = _mapping(raw)
        if row is None:
            errors.append(f"{label}:token_row_not_object:{row_index}")
            continue
        request_id = row.get("request_id")
        step = row.get("decode_step")
        input_token = row.get("input_token_id")
        predicted = row.get("predicted_next_token_id")
        if (
            request_id not in REQUEST_IDS
            or not _is_int(step)
            or not _is_int(input_token)
            or not _is_int(predicted)
            or int(step) not in range(DECODE_STEPS)
            or int(input_token) < 0
            or int(predicted) < 0
        ):
            errors.append(f"{label}:invalid_token_row:{row_index}")
            continue
        key = (str(request_id), int(step))
        if key in result:
            errors.append(f"{label}:duplicate_token_identity:{key}")
        else:
            result[key] = (int(input_token), int(predicted))
    expected = {(request_id, step) for request_id in REQUEST_IDS for step in range(DECODE_STEPS)}
    if set(result) != expected:
        errors.append(f"{label}:token_identity_set_mismatch")
    return result


def _router_index(
    value: Any,
    tokens: Mapping[tuple[str, int], tuple[int, int]],
    label: str,
    errors: list[str],
) -> dict[
    tuple[str, int, int], tuple[tuple[float, ...], tuple[int, ...], str]
]:
    result: dict[
        tuple[str, int, int], tuple[tuple[float, ...], tuple[int, ...], str]
    ] = {}
    if not isinstance(value, list):
        errors.append(f"{label}:router_not_list")
        return result
    for row_index, raw in enumerate(value):
        row = _mapping(raw)
        if row is None:
            errors.append(f"{label}:router_row_not_object:{row_index}")
            continue
        request_id = row.get("request_id")
        step = row.get("decode_step")
        layer = row.get("layer")
        if (
            request_id not in REQUEST_IDS
            or not _is_int(step)
            or int(step) not in range(DECODE_STEPS)
            or not _is_int(layer)
            or int(layer) not in range(LAYERS)
        ):
            errors.append(f"{label}:invalid_router_identity:{row_index}")
            continue
        token_key = (str(request_id), int(step))
        expected_tokens = tokens.get(token_key)
        if expected_tokens is None or (
            row.get("input_token_id"), row.get("predicted_next_token_id")
        ) != expected_tokens:
            errors.append(f"{label}:router_token_identity_mismatch:{row_index}")
        raw_logits = row.get("router_logits")
        if not isinstance(raw_logits, list) or len(raw_logits) != EXPERTS:
            errors.append(f"{label}:invalid_router_logits:{row_index}")
            continue
        logits = tuple(_finite_float(item) for item in raw_logits)
        if any(item is None for item in logits):
            errors.append(f"{label}:nonfinite_router_logits:{row_index}")
            continue
        raw_dtype = row.get("router_logits_dtype_before_float32_copy")
        if raw_dtype not in ROUTER_RAW_DTYPES:
            errors.append(f"{label}:invalid_router_raw_dtype:{row_index}")
            continue
        raw_experts = row.get("selected_experts")
        if (
            not isinstance(raw_experts, list)
            or len(raw_experts) != TOP_K
            or any(not _is_int(item) or int(item) not in range(EXPERTS) for item in raw_experts)
            or len(set(int(item) for item in raw_experts)) != TOP_K
        ):
            errors.append(f"{label}:invalid_selected_experts:{row_index}")
            continue
        key = (str(request_id), int(step), int(layer))
        if key in result:
            errors.append(f"{label}:duplicate_router_identity:{key}")
        else:
            result[key] = (
                tuple(float(item) for item in logits if item is not None),
                tuple(int(item) for item in raw_experts),
                str(raw_dtype),
            )
    expected = {
        (request_id, step, layer)
        for request_id in REQUEST_IDS
        for step in range(DECODE_STEPS)
        for layer in range(LAYERS)
    }
    if set(result) != expected:
        errors.append(f"{label}:router_identity_set_mismatch")
    return result


def _normalize_trace(
    value: Any, label: str, errors: list[str]
) -> dict[str, Any]:
    trace = _mapping(value)
    if trace is None:
        errors.append(f"{label}:trace_not_object")
        return {"tokens": {}, "router": {}}
    tokens = _token_index(trace.get("tokens"), label, errors)
    router = _router_index(trace.get("router"), tokens, label, errors)
    return {"tokens": tokens, "router": router}


def _validate_prompt_identity(execution: Mapping[str, Any], label: str, errors: list[str]) -> Any:
    token_identity = _mapping(execution.get("token_identity"))
    if token_identity is None:
        errors.append(f"{label}:missing_token_identity")
        return None
    rows = token_identity.get("selected_request_prompt_tokens")
    if not isinstance(rows, list):
        errors.append(f"{label}:invalid_prompt_identity")
        return token_identity
    observed = []
    for raw in rows:
        row = _mapping(raw)
        if row is None:
            observed.append(None)
        else:
            observed.append(
                (
                    row.get("request_id"),
                    row.get("prompt_token_count"),
                    row.get("prompt_token_ids_sha256"),
                )
            )
    if tuple(observed) != PROMPT_IDENTITY:
        errors.append(f"{label}:prompt_identity_mismatch")
    if not isinstance(token_identity.get("tokenizer"), Mapping):
        errors.append(f"{label}:missing_tokenizer_identity")
    return token_identity


def _validate_actual_import_paths(
    source: Mapping[str, Any], label: str, errors: list[str]
) -> dict[str, str]:
    raw = _mapping(source.get("actual_import_paths"))
    if raw is None or set(raw) != set(IMPORT_RELATIVE_PATHS):
        errors.append(f"{label}:actual_import_path_role_set_mismatch")
        return {}
    roots: set[str] = set()
    observed: dict[str, str] = {}
    for role, relative in IMPORT_RELATIVE_PATHS.items():
        value = raw.get(role)
        if not isinstance(value, str) or not value:
            errors.append(f"{label}:invalid_actual_import_path:{role}")
            continue
        path = Path(value)
        relative_parts = Path(relative).parts
        if (
            not path.is_absolute()
            or ".." in path.parts
            or len(path.parts) <= len(relative_parts)
            or tuple(path.parts[-len(relative_parts) :]) != relative_parts
        ):
            errors.append(f"{label}:actual_import_path_mismatch:{role}")
            continue
        root = path.parents[len(relative_parts) - 1]
        roots.add(str(root))
        observed[role] = str(path)
    if len(observed) == len(IMPORT_RELATIVE_PATHS) and len(roots) != 1:
        errors.append(f"{label}:actual_import_source_root_mismatch")
    return observed


def _validate_payload(
    value: Any,
    label: str,
    runner_sha: str,
    sealed_capture: Mapping[str, Any],
    sealed_reference: Mapping[tuple[str, int], tuple[int, int]],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    payload = _mapping(value)
    if payload is None:
        return None, [f"{label}:root_not_object"]
    if payload.get("schema") != ARM_SCHEMA:
        errors.append(f"{label}:invalid_schema")
    if payload.get("claim_ceiling") != CLAIM_CEILING:
        errors.append(f"{label}:invalid_claim_ceiling")
    for field in ("capacity_claim_authorized", "action_oracle_authorized", "controller_authorized"):
        if payload.get(field) is not False:
            errors.append(f"{label}:forbidden_unlock:{field}")

    source = _mapping(payload.get("source_identity"))
    if source is None:
        errors.append(f"{label}:missing_source_identity")
        source = {}
    if source.get("repo_head") != EXPECTED_REPO_HEAD or source.get("relevant_paths_clean") is not True:
        errors.append(f"{label}:source_checkout_identity_mismatch")
    if source.get("files_sha256") != SOURCE_FILES:
        errors.append(f"{label}:source_file_identity_mismatch")
    if source.get("runner_sha256") != runner_sha or not _is_sha(source.get("runner_sha256")):
        errors.append(f"{label}:runner_hash_mismatch")
    runner = source.get("runner")
    if not isinstance(runner, str) or Path(runner).name != RUNNER_PATH.name:
        errors.append(f"{label}:runner_path_mismatch")
    actual_import_paths = _validate_actual_import_paths(source, label, errors)

    capture = _mapping(payload.get("fresh_capture"))
    if capture is None:
        errors.append(f"{label}:missing_fresh_capture")
        capture = {}
    capture_path = capture.get("capture_dir")
    if not isinstance(capture_path, str) or not capture_path:
        errors.append(f"{label}:invalid_capture_path")
    elif str(Path(capture_path).resolve()) != sealed_capture.get("capture_dir"):
        errors.append(f"{label}:capture_path_not_bound_to_sealed_capture")
    if (
        not _is_sha(capture.get("capture_complete_sha256"))
        or capture.get("capture_complete_sha256")
        != sealed_capture.get("capture_complete_sha256")
    ):
        errors.append(f"{label}:capture_hash_not_bound_to_sealed_capture")
    audit = _mapping(capture.get("serial_audit"))
    if audit is None or not isinstance(capture.get("source_batch_dependence"), bool):
        errors.append(f"{label}:invalid_capture_audit")
    elif capture.get("source_batch_dependence") != bool(audit.get("batch_dependent_route_observed", False)):
        errors.append(f"{label}:capture_audit_binding_mismatch")
    elif (
        capture.get("source_batch_dependence") is not True
        or audit.get("batch_dependent_route_observed") is not True
    ):
        errors.append(f"{label}:source_capture_batch_dependence_not_true")
    if (
        audit != sealed_capture.get("serial_audit")
        or capture.get("source_batch_dependence")
        != sealed_capture.get("source_batch_dependence")
    ):
        errors.append(f"{label}:capture_audit_not_bound_to_sealed_capture")
    reference_errors: list[str] = []
    embedded_reference = _token_index(
        capture.get("reference_tokens"), f"{label}:reference", reference_errors
    )
    errors.extend(reference_errors)
    if embedded_reference != sealed_reference:
        errors.append(f"{label}:reference_tokens_not_bound_to_sealed_ledger")
    reference = dict(sealed_reference)

    execution = _mapping(payload.get("execution"))
    if execution is None:
        errors.append(f"{label}:missing_execution")
        execution = {}
    process = execution.get("fresh_process_repeat")
    if not _is_int(process) or int(process) not in PROCESSES:
        errors.append(f"{label}:invalid_process_index")
        process = -1
    expected_order = ARM_ORDERS[int(process)] if process in PROCESSES else None
    if (
        execution.get("requests") != len(REQUEST_IDS)
        or execution.get("request_ids") != list(REQUEST_IDS)
        or execution.get("decode_steps") != DECODE_STEPS
        or execution.get("batch_width") != len(REQUEST_IDS)
        or execution.get("planned_fresh_process_repeats") != len(PROCESSES)
        or execution.get("arm_order") != (list(expected_order) if expected_order else None)
    ):
        errors.append(f"{label}:scale_or_arm_order_mismatch")
    if (
        execution.get("canonical_state_advance") != "serial_a_only"
        or execution.get("batch_state_propagated_to_next_step") is not False
        or execution.get("warmup_trajectory_discarded") is not True
    ):
        errors.append(f"{label}:matched_prestate_contract_mismatch")
    fork_checks = execution.get("matched_prestate_fork_checks")
    if (
        not isinstance(fork_checks, list)
        or len(fork_checks) != DECODE_STEPS
        or any(
            not isinstance(item, Mapping)
            or item.get("arms") != sorted(ARMS)
            or item.get("requests") != len(REQUEST_IDS)
            or not _is_int(item.get("independent_equal_tensors_checked"))
            or int(item.get("independent_equal_tensors_checked", 0)) <= 0
            or not _is_int(item.get("independent_equal_elements_checked"))
            or int(item.get("independent_equal_elements_checked", 0)) <= 0
            for item in fork_checks
        )
    ):
        errors.append(f"{label}:fork_check_contract_mismatch")
    gpu = _mapping(execution.get("gpu_isolation"))
    if gpu is None or gpu.get("status") != "PASS_SAMPLED_PROCESS_ISOLATION" or gpu.get("violations") != []:
        errors.append(f"{label}:gpu_isolation_invalid")
    token_identity = _validate_prompt_identity(execution, label, errors)
    process_identity = _mapping(execution.get("process_identity"))
    if process_identity is None:
        errors.append(f"{label}:missing_process_identity")
        process_identity = {}
    pid = process_identity.get("pid")
    start_ticks = process_identity.get("start_time_ticks")
    boot_id = process_identity.get("boot_id")
    if (
        not _is_int(pid)
        or int(pid) <= 0
        or not _is_int(start_ticks)
        or int(start_ticks) <= 0
        or not isinstance(boot_id, str)
        or re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            boot_id,
        )
        is None
    ):
        errors.append(f"{label}:invalid_process_identity")

    traces_value = _mapping(payload.get("traces"))
    traces: dict[str, Any] = {}
    if traces_value is None or set(traces_value) != set(ARMS):
        errors.append(f"{label}:trace_arm_set_mismatch")
        traces_value = {}
    for arm in ARMS:
        traces[arm] = _normalize_trace(traces_value.get(arm), f"{label}:{arm}", errors)
    if all(set(traces[arm]["router"]) for arm in ARMS):
        common_keys = traces[ARMS[0]]["router"].keys()
        if all(traces[arm]["router"].keys() == common_keys for arm in ARMS[1:]):
            for key in common_keys:
                dtypes = {traces[arm]["router"][key][2] for arm in ARMS}
                if len(dtypes) != 1:
                    errors.append(f"{label}:router_raw_dtype_arm_mismatch:{key}")
                    break

    control_summary = _mapping(payload.get("serial_negative_control"))
    parity_summary = _mapping(payload.get("reference_token_parity"))
    contrast_summary = _mapping(payload.get("serial_vs_batch_4"))
    if control_summary is None or not isinstance(control_summary.get("exact"), bool):
        errors.append(f"{label}:invalid_serial_control_summary")
        reported_serial_exact = False
    else:
        reported_serial_exact = bool(control_summary["exact"])
    if parity_summary is None or not isinstance(parity_summary.get("passed"), bool):
        errors.append(f"{label}:invalid_reference_parity_summary")
        reported_reference_parity = False
    else:
        reported_reference_parity = bool(parity_summary["passed"])
    if contrast_summary is None:
        errors.append(f"{label}:missing_serial_batch_contrast")

    return {
        "process": int(process),
        "source_identity": source,
        "actual_import_paths": actual_import_paths,
        "capture_identity": capture,
        "reference": reference,
        "execution_identity": {
            "request_ids": execution.get("request_ids"),
            "seed": execution.get("seed_reset_before_process_trajectory"),
            "environment": execution.get("environment"),
            "runtime_validation": execution.get("runtime_validation"),
            "token_identity": token_identity,
        },
        "runner_status": payload.get("status"),
        "process_identity": dict(process_identity),
        "reported_serial_exact": reported_serial_exact,
        "reported_reference_parity": reported_reference_parity,
        "traces": traces,
    }, errors


def _serial_exact(cell: Mapping[str, Any]) -> bool:
    left = cell["traces"]["serial_a"]
    right = cell["traces"]["serial_b"]
    if (
        cell["reported_serial_exact"] is not True
        or left["tokens"] != right["tokens"]
        or left["router"].keys() != right["router"].keys()
    ):
        return False
    return all(left["router"][key] == right["router"][key] for key in left["router"])


def _serial_trace_exact(cell: Mapping[str, Any]) -> bool:
    left = cell["traces"]["serial_a"]
    right = cell["traces"]["serial_b"]
    return bool(
        left["tokens"] == right["tokens"]
        and left["router"].keys() == right["router"].keys()
        and all(left["router"][key] == right["router"][key] for key in left["router"])
    )


def _reference_parity(cell: Mapping[str, Any]) -> bool:
    return cell["reported_reference_parity"] is True and all(
        cell["traces"][arm]["tokens"] == cell["reference"] for arm in ARMS
    )


def _reference_trace_parity(cell: Mapping[str, Any]) -> bool:
    return all(cell["traces"][arm]["tokens"] == cell["reference"] for arm in ARMS)


def _cross_process_stable(cells: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    reference = cells[0]
    for cell in cells[1:]:
        for arm in ARMS:
            left = reference["traces"][arm]
            right = cell["traces"][arm]
            if left["tokens"] != right["tokens"]:
                errors.append(f"{arm}:token_drift:p0_vs_p{cell['process']}")
                continue
            if left["router"].keys() != right["router"].keys():
                errors.append(f"{arm}:router_identity_drift:p0_vs_p{cell['process']}")
                continue
            for key in left["router"]:
                left_logits, left_experts, left_dtype = left["router"][key]
                right_logits, right_experts, right_dtype = right["router"][key]
                if left_dtype != right_dtype:
                    errors.append(
                        f"{arm}:router_raw_dtype_drift:{key}:p0_vs_p{cell['process']}"
                    )
                    break
                if left_experts != right_experts:
                    errors.append(f"{arm}:selected_expert_drift:{key}:p0_vs_p{cell['process']}")
                    break
                if not _allclose(left_logits, right_logits):
                    errors.append(f"{arm}:router_logit_drift:{key}:p0_vs_p{cell['process']}")
                    break
    return not errors, errors


def _first_signature(serial: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any] | None:
    """Mirror the runner's request-step-local causal-frontier signature."""

    per_request_step: list[dict[str, Any]] = []
    for request_id in REQUEST_IDS:
        for decode_step in range(DECODE_STEPS):
            keys = sorted(
                (
                    key
                    for key in serial["router"]
                    if key[0] == request_id and key[1] == decode_step
                ),
                key=lambda key: key[2],
            )
            first_logit_layer = None
            first_assignment = None
            for key in keys:
                serial_logits, serial_experts, serial_dtype = serial["router"][key]
                batch_logits, batch_experts, batch_dtype = batch["router"][key]
                if serial_dtype != batch_dtype:
                    raise ValueError(f"router raw dtype mismatch at {key}")
                if first_logit_layer is None and any(
                    left != right for left, right in zip(serial_logits, batch_logits)
                ):
                    first_logit_layer = key[2]
                if (
                    first_assignment is None
                    and sorted(serial_experts) != sorted(batch_experts)
                ):
                    first_assignment = {
                        "position": (decode_step, key[2]),
                        "serial_experts": sorted(serial_experts),
                        "batched_experts": sorted(batch_experts),
                    }
            category = None
            qualifying_logit_position = None
            if first_assignment is not None:
                if (
                    first_logit_layer is not None
                    and first_logit_layer <= first_assignment["position"][1]
                ):
                    category = "PRE_TOPK_NUMERICAL_DIVERGENCE"
                    qualifying_logit_position = (decode_step, first_logit_layer)
                else:
                    category = "RECONSTRUCTED_TOPK_INCONSISTENCY"
            per_request_step.append(
                {
                    "request_id": request_id,
                    "decode_step": decode_step,
                    "qualifying_logit_position": qualifying_logit_position,
                    "first_assignment": first_assignment,
                    "category": category,
                }
            )

    assignment_rows = [
        row for row in per_request_step if row["first_assignment"] is not None
    ]
    if not assignment_rows:
        return None
    position = min(row["first_assignment"]["position"] for row in assignment_rows)
    frontier = sorted(
        [row for row in assignment_rows if row["first_assignment"]["position"] == position],
        key=lambda row: row["request_id"],
    )
    categories = {row["category"] for row in frontier}
    return {
        "decode_step": position[0],
        "layer": position[1],
        "category": next(iter(categories)) if len(categories) == 1 else "MIXED_CAUSAL_FRONTIER",
        "records": [
            {
                "request_id": row["request_id"],
                "serial_experts": row["first_assignment"]["serial_experts"],
                "batched_experts": row["first_assignment"]["batched_experts"],
                "assignment_source_category": row["category"],
                "first_router_logit_position": (
                    {
                        "decode_step": row["qualifying_logit_position"][0],
                        "layer": row["qualifying_logit_position"][1],
                    }
                    if row["qualifying_logit_position"] is not None
                    else None
                ),
            }
            for row in frontier
        ],
    }


def _recompute_runner_status(cell: Mapping[str, Any]) -> str:
    serial_a = _first_signature(cell["traces"]["serial_a"], cell["traces"]["batch_4"])
    serial_b = _first_signature(cell["traces"]["serial_b"], cell["traces"]["batch_4"])
    source_batch_dependence = cell["capture_identity"].get("source_batch_dependence") is True
    serial_exact = _serial_trace_exact(cell)
    token_parity = _reference_trace_parity(cell)
    assignment_changed = serial_a is not None and serial_b is not None
    double_sided_match = assignment_changed and serial_a == serial_b
    if not source_batch_dependence:
        return "STOP_FRESH_CAPTURE_NO_BATCH_DEPENDENT_ROUTE"
    if not serial_exact:
        return "INVALID_SERIAL_NEGATIVE_CONTROL"
    if not token_parity:
        return "STOP_TOKEN_PARITY_FAILED"
    if not assignment_changed:
        return "STOP_MATCHED_PRESTATE_NO_ASSIGNMENT_DIVERGENCE"
    if not double_sided_match:
        return "STOP_BATCH_CONTRAST_NOT_DOUBLE_SIDED"
    if serial_a["category"] == "PRE_TOPK_NUMERICAL_DIVERGENCE":
        return "PROCESS_CANDIDATE_PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION"
    return "INCONCLUSIVE_RECONSTRUCTED_TOPK_INCONSISTENCY"


def evaluate_payloads(
    payloads: Sequence[Any],
    input_metadata: Sequence[Mapping[str, Any]] | None = None,
    *,
    sealed_capture: Mapping[str, Any] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(payloads) != len(PROCESSES):
        errors.append(f"expected_exactly_3_inputs:found_{len(payloads)}")
    if not RUNNER_PATH.is_file():
        errors.append("runner_file_missing")
        runner_sha = ""
    else:
        runner_sha = _sha256(RUNNER_PATH)
    capture_contract = _mapping(sealed_capture)
    if capture_contract is None:
        errors.append("sealed_capture_contract_missing")
        capture_contract = {}
    elif (
        capture_contract.get("schema") != CAPTURE_CONTRACT.SCHEMA
        or capture_contract.get("source_batch_dependence") is not True
        or capture_contract.get("request_ids") != list(REQUEST_IDS)
        or capture_contract.get("decode_steps") != DECODE_STEPS
        or not _is_sha(capture_contract.get("capture_complete_sha256"))
    ):
        errors.append("sealed_capture_contract_invalid")
    sealed_reference_errors: list[str] = []
    sealed_reference = _token_index(
        capture_contract.get("reference_tokens"),
        "sealed_capture",
        sealed_reference_errors,
    )
    errors.extend(sealed_reference_errors)
    cells: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        cell, cell_errors = _validate_payload(
            payload,
            f"input[{index}]",
            runner_sha,
            capture_contract,
            sealed_reference,
        )
        errors.extend(cell_errors)
        if cell is not None:
            cells.append(cell)
    observed_process_indices = [cell["process"] for cell in cells]
    if (
        sorted(observed_process_indices) != list(PROCESSES)
        or len(set(observed_process_indices)) != len(observed_process_indices)
    ):
        errors.append(f"process_index_set_mismatch:{observed_process_indices}")
    cells.sort(key=lambda cell: cell["process"])
    process_indices = [cell["process"] for cell in cells]
    process_identities = [cell["process_identity"] for cell in cells]
    process_identity_keys = [
        (
            identity.get("pid"),
            identity.get("start_time_ticks"),
            identity.get("boot_id"),
        )
        for identity in process_identities
    ]
    unique_process_identities = bool(
        len(process_identity_keys) == len(PROCESSES)
        and len(set(process_identity_keys)) == len(PROCESSES)
    )
    boot_ids = {identity.get("boot_id") for identity in process_identities}
    same_boot_id = bool(len(process_identities) == len(PROCESSES) and len(boot_ids) == 1)
    if not unique_process_identities:
        errors.append("fresh_process_identities_not_unique")
    if not same_boot_id:
        errors.append("fresh_process_boot_id_drift")

    if len(cells) == len(PROCESSES):
        source_identities = [cell["source_identity"] for cell in cells]
        if any(value != source_identities[0] for value in source_identities[1:]):
            errors.append("source_identity_drift_across_processes")
        capture_identities = [cell["capture_identity"] for cell in cells]
        if any(value != capture_identities[0] for value in capture_identities[1:]):
            errors.append("capture_identity_drift_across_processes")
        execution_identities = [cell["execution_identity"] for cell in cells]
        if any(value != execution_identities[0] for value in execution_identities[1:]):
            errors.append("workload_or_runtime_identity_drift_across_processes")

    runner_status_checks: list[dict[str, Any]] = []
    if not errors and len(cells) == len(PROCESSES):
        for cell in cells:
            expected_status = _recompute_runner_status(cell)
            reported_status = cell["runner_status"]
            runner_status_checks.append(
                {
                    "process": cell["process"],
                    "reported": reported_status,
                    "recomputed": expected_status,
                    "match": reported_status == expected_status,
                }
            )
            if reported_status != expected_status:
                errors.append(
                    f"runner_status_mismatch:p{cell['process']}:"
                    f"reported={reported_status}:recomputed={expected_status}"
                )

    metadata = list(input_metadata or [])
    base = {
        "schema": SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "capacity_claim_authorized": False,
        "action_oracle_authorized": False,
        "controller_authorized": False,
        "method_go_authorized": False,
        "inputs": metadata,
        "process_indices": process_indices,
        "process_identities": process_identities,
        "fresh_process_identity_unique": unique_process_identities,
        "same_boot_id": same_boot_id,
        "runner_sha256": runner_sha,
        "capture_contract_helper_sha256": (
            _sha256(CAPTURE_CONTRACT_PATH) if CAPTURE_CONTRACT_PATH.is_file() else None
        ),
        "sealed_capture_evidence": {
            key: value
            for key, value in capture_contract.items()
            if key not in {"reference_tokens", "serial_audit"}
        },
        "runner_status_recomputation": runner_status_checks,
        "errors": errors,
    }
    if errors:
        return {
            **base,
            "status": "INVALID",
            "structurally_valid": False,
            "failure_category": "INPUT_SOURCE_OR_IDENTITY_INVALID",
        }

    serial_exact = [_serial_exact(cell) for cell in cells]
    token_parity = [_reference_parity(cell) for cell in cells]
    cross_stable, stability_errors = _cross_process_stable(cells)
    signatures = []
    for cell in cells:
        serial_a = _first_signature(cell["traces"]["serial_a"], cell["traces"]["batch_4"])
        serial_b = _first_signature(cell["traces"]["serial_b"], cell["traces"]["batch_4"])
        signatures.append(
            {
                "process": cell["process"],
                "vs_serial_a": serial_a,
                "vs_serial_b": serial_b,
                "double_sided_match": serial_a == serial_b,
            }
        )
    all_signatures = [item["vs_serial_a"] for item in signatures]
    no_divergence = all(signature is None for signature in all_signatures) and all(
        item["vs_serial_b"] is None for item in signatures
    )
    consistent = bool(
        not no_divergence
        and all(item["double_sided_match"] and item["vs_serial_a"] is not None for item in signatures)
        and all(signature == all_signatures[0] for signature in all_signatures[1:])
    )

    if not all(serial_exact):
        status = "SERIAL_CONTROL_UNSTABLE"
    elif not all(token_parity):
        status = "TOKEN_PARITY_FAILED"
    elif not cross_stable:
        status = "CROSS_PROCESS_UNSTABLE"
    elif no_divergence:
        status = "NO_DIVERGENCE"
    elif not consistent:
        status = "INCONSISTENT_FIRST_DIVERGENCE"
    elif all_signatures[0]["category"] == "PRE_TOPK_NUMERICAL_DIVERGENCE":
        status = "PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED"
    else:
        status = "INCONCLUSIVE"

    return {
        **base,
        "status": status,
        "structurally_valid": True,
        "failure_category": (
            None
            if status == "PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED"
            else status
        ),
        "evidence_type": "CUSTOM_TRANSFORMERS_MATCHED_PRESTATE_THREE_FRESH_PROCESS",
        "checks": {
            "serial_a_b_exact_by_process": serial_exact,
            "reference_token_parity_by_process": token_parity,
            "within_arm_cross_process_stable": cross_stable,
            "cross_process_stability_errors": stability_errors,
            "first_divergence_signatures": signatures,
            "identical_double_sided_signature_across_processes": consistent,
            "no_assignment_divergence": no_divergence,
            "router_logit_allclose": {"atol": LOGIT_ATOL, "rtol": LOGIT_RTOL},
            "fresh_process_identity_unique": unique_process_identities,
            "same_boot_id": same_boot_id,
        },
        "runner_statuses": [cell["runner_status"] for cell in cells],
    }


def evaluate_paths(paths: Sequence[Path], capture_dir: Path) -> dict[str, Any]:
    payloads: list[Any] = []
    metadata: list[dict[str, Any]] = []
    load_errors: list[str] = []
    sealed_capture: Mapping[str, Any] | None = None
    try:
        sealed_capture = CAPTURE_CONTRACT.load_n0d_capture_contract(capture_dir)
    except (OSError, CAPTURE_CONTRACT.CaptureContractError) as exc:
        load_errors.append(f"sealed_capture_contract_failed:{exc}")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        load_errors.append("duplicate_input_paths")
    for path in resolved:
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
            metadata.append({"path": str(path), "sha256": _sha256(path)})
        except (OSError, json.JSONDecodeError) as exc:
            load_errors.append(f"cannot_load_input:{path}:{exc}")
    report = evaluate_payloads(
        payloads,
        metadata,
        sealed_capture=sealed_capture,
    )
    if load_errors:
        report["status"] = "INVALID"
        report["structurally_valid"] = False
        report["failure_category"] = "INPUT_SOURCE_OR_IDENTITY_INVALID"
        report["errors"] = load_errors + list(report.get("errors", []))
    return report


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    report = evaluate_paths(
        [Path(value) for value in args.input],
        Path(args.capture_dir),
    )
    _write_once(output, report)
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
