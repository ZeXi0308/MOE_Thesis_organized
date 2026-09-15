#!/usr/bin/env python3
"""Fail-closed evaluator for the N0c fresh-process capture triage."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "n0c-capture-stage-evaluation-v1"
CONFIG_SCHEMA = "n0c-capture-stage-arm-config-v1"
RESULT_SCHEMA = "n0c-capture-stage-arm-result-v1"
CLAIM_CEILING = "FRESH_PROCESS_ASSOCIATIONAL_CAPTURE_TRIAGE_ONLY"
ARMS = ("n_a", "capture_only", "full_export", "n_b")
ROUNDS = tuple(range(4))
CAPTURE_MODES = {
    "n_a": "off",
    "capture_only": "device",
    "full_export": "full_export",
    "n_b": "off",
}
RUNTIME_SHAPES = {
    "stock": {"batch": 8, "prefix_cells": 12},
    "valid-window": {"batch": 16, "prefix_cells": 36},
}
TARGET_RUNTIMES = {
    "stock_p512_b8_g2_w0": "stock",
    "valid_window_p512_b16_g1_w0": "valid-window",
}
LATIN_ARM_ORDERS = (
    ("n_a", "capture_only", "full_export", "n_b"),
    ("capture_only", "n_b", "n_a", "full_export"),
    ("full_export", "n_a", "n_b", "capture_only"),
    ("n_b", "full_export", "capture_only", "n_a"),
)
IDENTITY_FIELDS = (
    "target_id",
    "target_runtime",
    "round",
    "arm",
    "capture_mode",
    "logical_runtime_variant",
    "runtime_import_root_verified",
    "runtime_patch_id",
    "claim_ceiling",
    "target_spec_sha256",
    "prefix_plan_sha256",
    "runtime_package_manifest_sha256",
    "target_input_artifact_sha256",
    "target_prompt_token_ids_sha256",
)
HASH_FIELDS = (
    "target_spec_sha256",
    "prefix_plan_sha256",
    "runtime_package_manifest_sha256",
    "target_input_artifact_sha256",
    "target_prompt_token_ids_sha256",
)
ROUTE_FIELDS = {
    "full_export_includes_prompt_tail",
    "route_mapping",
    "route_artifact",
    "route_artifact_sha256",
    "route_shape",
}
VALID_PRECEDENCE = (
    "BASELINE_DISCRETE_NONDETERMINISM",
    "CAPTURE_NO_EXPORT_ASSOCIATION",
    "EXPORT_PATH_ASSOCIATION",
    "INTERMITTENT_OR_UNRESOLVED",
    "NOT_REPRODUCED",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required:{path}")
    return value


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _token_matrix(value: Any, batch: int) -> list[list[int]] | None:
    if not isinstance(value, list) or len(value) != batch:
        return None
    result: list[list[int]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 16
            or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in row)
        ):
            return None
        result.append(list(row))
    return result


def _safe_artifact(bundle: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    try:
        if Path(relative).is_absolute():
            return None
        path = (bundle / relative).resolve()
        path.relative_to(bundle.resolve())
    except (OSError, ValueError):
        return None
    return path if path.is_file() and not path.is_symlink() else None


def _package_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
        and path.suffix != ".pyc" and "__pycache__" not in path.parts
    }


def _load_bundle(bundle: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    paths = {name: bundle / name for name in ("config.json", "result.json", "RUN_COMPLETE.json")}
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            errors.append(f"{bundle}:missing_or_symlink:{name}")
    if errors:
        return None, errors
    try:
        config = _load_json(paths["config.json"])
        result = _load_json(paths["result.json"])
        seal = _load_json(paths["RUN_COMPLETE.json"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"{bundle}:invalid_json:{exc}"]

    if config.get("schema") != CONFIG_SCHEMA:
        errors.append(f"{bundle}:invalid_config_schema")
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != "COMPLETE":
        errors.append(f"{bundle}:invalid_result_schema_or_status")
    if seal.get("status") != "RUN_COMPLETE":
        errors.append(f"{bundle}:invalid_seal_status")
    for filename, seal_field in (("config.json", "config_sha256"), ("result.json", "result_sha256")):
        if not _is_sha(seal.get(seal_field)) or seal.get(seal_field) != _sha256(paths[filename]):
            errors.append(f"{bundle}:seal_mismatch:{filename}")

    for field in IDENTITY_FIELDS:
        if config.get(field) != result.get(field):
            errors.append(f"{bundle}:config_result_identity_mismatch:{field}")
    if config.get("claim_ceiling") != CLAIM_CEILING:
        errors.append(f"{bundle}:invalid_claim_ceiling")
    for field in HASH_FIELDS:
        if not _is_sha(config.get(field)):
            errors.append(f"{bundle}:invalid_identity_hash:{field}")
    if not isinstance(config.get("runtime_patch_id"), str) or not config.get("runtime_patch_id"):
        errors.append(f"{bundle}:invalid_runtime_patch_id")

    target = config.get("target_id")
    runtime = config.get("target_runtime")
    round_id = config.get("round")
    arm = config.get("arm")
    valid_runtime = isinstance(runtime, str) and runtime in RUNTIME_SHAPES
    valid_round = isinstance(round_id, int) and not isinstance(round_id, bool) and round_id in ROUNDS
    valid_arm = isinstance(arm, str) and arm in ARMS
    if not isinstance(target, str) or not target:
        errors.append(f"{bundle}:invalid_target_id")
    if not valid_runtime:
        errors.append(f"{bundle}:invalid_target_runtime")
    if not valid_round:
        errors.append(f"{bundle}:invalid_round")
    if not valid_arm or config.get("capture_mode") != (CAPTURE_MODES[arm] if valid_arm else None):
        errors.append(f"{bundle}:invalid_arm_or_capture_mode")
    if (
        not isinstance(target, str) or not target
        or not valid_runtime
        or not valid_round
        or not valid_arm
    ):
        return None, errors
    if bundle.name != f"{target}-r{round_id}-{arm}":
        errors.append(f"{bundle}:bundle_name_identity_mismatch")

    expected = RUNTIME_SHAPES[runtime]
    tokens = _token_matrix(result.get("output_token_ids"), expected["batch"])
    if tokens is None:
        errors.append(f"{bundle}:invalid_output_token_ids")
    if result.get("warmup_count") != 6:
        errors.append(f"{bundle}:invalid_warmup_count")
    if result.get("prefix_cells_executed") != expected["prefix_cells"]:
        errors.append(f"{bundle}:invalid_prefix_cells_executed")

    routes: np.ndarray[Any, Any] | None = None
    if arm != "full_export":
        unexpected = sorted(ROUTE_FIELDS.intersection(result))
        if unexpected:
            errors.append(f"{bundle}:route_fields_in_nonexport_arm:{','.join(unexpected)}")
        if seal.get("route_sha256") is not None:
            errors.append(f"{bundle}:nonexport_route_seal_not_null")
    else:
        shape = [expected["batch"], 16, 16, 8]
        mapping = result.get("route_mapping")
        expected_mapping = [
            {"route_row": index, "input_position": 511 + index, "produces_output_token_index": index}
            for index in range(16)
        ]
        if result.get("full_export_includes_prompt_tail") is not True:
            errors.append(f"{bundle}:prompt_tail_not_retained")
        if mapping != expected_mapping:
            errors.append(f"{bundle}:invalid_route_mapping")
        if result.get("route_shape") != shape:
            errors.append(f"{bundle}:invalid_declared_route_shape")
        route_path = _safe_artifact(bundle, result.get("route_artifact"))
        actual_route_sha = _sha256(route_path) if route_path is not None else None
        if route_path is None:
            errors.append(f"{bundle}:missing_or_unsafe_route_artifact")
        if (
            not _is_sha(result.get("route_artifact_sha256"))
            or result.get("route_artifact_sha256") != actual_route_sha
            or seal.get("route_sha256") != actual_route_sha
        ):
            errors.append(f"{bundle}:route_seal_mismatch")
        if route_path is not None:
            try:
                with np.load(route_path, allow_pickle=False) as archive:
                    if set(archive.files) != {"routes"}:
                        raise ValueError("routes.npz must contain only 'routes'")
                    routes = np.array(archive["routes"], copy=True)
                if list(routes.shape) != shape or not np.issubdtype(routes.dtype, np.integer):
                    raise ValueError("invalid route array shape or dtype")
                if routes.size == 0 or int(routes.min()) < 0 or int(routes.max()) > 63:
                    raise ValueError("expert IDs outside [0,63]")
            except (OSError, EOFError, TypeError, ValueError, zipfile.BadZipFile) as exc:
                errors.append(f"{bundle}:invalid_route_artifact:{exc}")

    return {
        "path": str(bundle),
        "config": config,
        "result": result,
        "tokens": tokens,
        "routes": routes,
    }, errors


def _target_report(target_id: str, cells: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    canonical = cells[(0, "n_a")]["tokens"]
    noop_mismatches = [
        {"round": round_id, "arm": arm}
        for round_id in ROUNDS
        for arm in ("n_a", "n_b")
        if cells[(round_id, arm)]["tokens"] != canonical
    ]
    def first_divergence(observed: list[list[int]]) -> tuple[int, int, int, int] | None:
        # Decode time is the primary order; request row only breaks ties within
        # the same output position.  Downstream tokens may amplify a single
        # numerical flip, so the repeated first divergence is the frozen unit.
        for output_token_index in range(16):
            for request_row in range(len(canonical)):
                baseline_token = canonical[request_row][output_token_index]
                observed_token = observed[request_row][output_token_index]
                if observed_token != baseline_token:
                    return (
                        output_token_index,
                        request_row,
                        baseline_token,
                        observed_token,
                    )
        return None

    def signature_payload(signature: tuple[int, int, int, int] | None) -> dict[str, int] | None:
        if signature is None:
            return None
        return dict(
            zip(
                ("output_token_index", "request_row", "baseline_token", "observed_token"),
                signature,
            )
        )

    round_contrasts: list[dict[str, Any]] = []
    common_signatures: list[tuple[int, int, int, int]] = []
    export_only_signatures: list[tuple[int, int, int, int]] = []
    for round_id in ROUNDS:
        capture_signature = first_divergence(cells[(round_id, "capture_only")]["tokens"])
        export_signature = first_divergence(cells[(round_id, "full_export")]["tokens"])
        if capture_signature is not None and capture_signature == export_signature:
            contrast = "COMMON_CAPTURE"
            common_signatures.append(capture_signature)
        elif capture_signature is None and export_signature is not None:
            contrast = "EXPORT_ONLY"
            export_only_signatures.append(export_signature)
        elif capture_signature is not None and export_signature is None:
            contrast = "CAPTURE_ONLY_CONTRADICTION"
        elif capture_signature is not None and export_signature is not None:
            contrast = "DISCORDANT"
        else:
            contrast = "NO_DIVERGENCE"
        round_contrasts.append(
            {
                "round": round_id,
                "contrast": contrast,
                "capture_only_first_divergence": signature_payload(capture_signature),
                "full_export_first_divergence": signature_payload(export_signature),
            }
        )

    common_counts = Counter(common_signatures)
    export_counts = Counter(export_only_signatures)
    matching_common = common_counts.most_common(1)[0] if common_counts else (None, 0)
    matching_export = export_counts.most_common(1)[0] if export_counts else (None, 0)
    capture_drift = [
        row["round"] for row in round_contrasts
        if row["capture_only_first_divergence"] is not None
    ]
    export_drift = [
        row["round"] for row in round_contrasts
        if row["full_export_first_divergence"] is not None
    ]

    route_reference = cells[(0, "full_export")]["routes"]
    exact_route_drift: list[int] = []
    set_route_drift: list[dict[str, int]] = []
    for round_id in ROUNDS[1:]:
        current = cells[(round_id, "full_export")]["routes"]
        if not np.array_equal(current, route_reference):
            exact_route_drift.append(round_id)
        changed = np.any(
            np.sort(current, axis=-1) != np.sort(route_reference, axis=-1), axis=-1
        )
        count = int(np.count_nonzero(changed))
        if count:
            set_route_drift.append({"round": round_id, "changed_request_step_layer_positions": count})

    if noop_mismatches:
        status = "BASELINE_DISCRETE_NONDETERMINISM"
    elif matching_common[1] >= 3:
        status = "CAPTURE_NO_EXPORT_ASSOCIATION"
    elif not capture_drift and matching_export[1] >= 3:
        status = "EXPORT_PATH_ASSOCIATION"
    elif capture_drift or export_drift:
        status = "INTERMITTENT_OR_UNRESOLVED"
    else:
        status = "NOT_REPRODUCED"
    sample = cells[(0, "n_a")]["config"]
    return {
        "target_id": target_id,
        "target_runtime": sample["target_runtime"],
        "status": status,
        "noop_token_mismatches": noop_mismatches,
        "capture_only_token_drift_rounds": capture_drift,
        "full_export_token_drift_rounds": export_drift,
        "per_round_token_contrasts": round_contrasts,
        "matching_common_capture": {
            "first_divergence": signature_payload(matching_common[0]),
            "round_count": matching_common[1],
        },
        "matching_export_only": {
            "first_divergence": signature_payload(matching_export[0]),
            "round_count": matching_export[1],
        },
        "route_diagnostic_only": {
            "canonical_full_export_round": 0,
            "exact_route_drift_rounds": exact_route_drift,
            "topk_set_drift": set_route_drift,
            "used_for_token_attribution": False,
        },
    }


def _schedule_errors(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        plan.get("schema") != "n0c-capture-stage-orchestration-v1"
        or plan.get("claim_ceiling") != CLAIM_CEILING
        or plan.get("rounds") != 4
    ):
        errors.append("invalid_run_plan_schema_claim_or_rounds")
    schedule = plan.get("schedule")
    if not isinstance(schedule, list):
        return errors + ["run_plan_schedule_must_be_a_list"]
    targets = plan.get("targets")
    if not isinstance(targets, dict) or set(targets) != set(TARGET_RUNTIMES):
        return errors + ["run_plan_targets_invalid"]
    target_ids = tuple(TARGET_RUNTIMES)
    expected: list[dict[str, Any]] = []
    for round_id, arms in enumerate(LATIN_ARM_ORDERS):
        target_order = target_ids if round_id % 2 == 0 else tuple(reversed(target_ids))
        for target in target_order:
            for arm in arms:
                target_plan = targets.get(target)
                base_patch = target_plan.get("base_runtime_patch_id") if isinstance(target_plan, dict) else None
                patch_id = (
                    f"{base_patch}+device-capture-no-export-v1"
                    if arm == "capture_only" else base_patch
                )
                expected.append(
                    {
                        "target_id": target,
                        "target_runtime": TARGET_RUNTIMES[target],
                        "round": round_id,
                        "arm": arm,
                        "capture_mode": CAPTURE_MODES[arm],
                        "runtime_patch_id": patch_id,
                        "bundle": f"{target}-r{round_id}-{arm}",
                    }
                )
    if schedule != expected:
        errors.append("run_plan_schedule_does_not_match_frozen_latin_order")
    return errors


def _campaign_identity_errors(
    root: Path,
    plan: dict[str, Any],
    cells_by_target: dict[str, dict[tuple[int, str], dict[str, Any]]],
) -> list[str]:
    errors: list[str] = []
    frozen = plan.get("frozen_input_sha256")
    if not isinstance(frozen, dict) or not frozen:
        return ["missing_frozen_input_sha256"]
    frozen_root = root / "frozen"
    actual_frozen = _package_manifest(frozen_root) if frozen_root.is_dir() else {}
    actual_frozen = {f"frozen/{relative}": digest for relative, digest in actual_frozen.items()}
    if frozen != actual_frozen:
        errors.append("frozen_input_manifest_mismatch")

    plan_targets = plan.get("targets")
    if not isinstance(plan_targets, dict):
        return errors + ["missing_plan_targets"]
    for target_id, runtime in TARGET_RUNTIMES.items():
        target_plan = plan_targets.get(target_id)
        cells = cells_by_target.get(target_id, {})
        if not isinstance(target_plan, dict) or not cells:
            errors.append(f"missing_target_plan_or_cells:{target_id}")
            continue
        spec_relative = f"frozen/targets/{target_id}/target-spec.json"
        spec_path = root / spec_relative
        try:
            spec = _load_json(spec_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid_target_spec:{target_id}:{exc}")
            continue
        spec_sha = _sha256(spec_path)
        configs = [cell["config"] for cell in cells.values()]
        if frozen.get(spec_relative) != spec_sha or any(
            config.get("target_spec_sha256") != spec_sha for config in configs
        ):
            errors.append(f"target_spec_hash_mismatch:{target_id}")
        records = spec.get("prefix_records")
        expected_prefix = RUNTIME_SHAPES[runtime]["prefix_cells"]
        if (
            spec.get("schema") != "n0c-capture-target-spec-v1"
            or spec.get("target_id") != target_id
            or spec.get("target_runtime") != runtime
            or not isinstance(records, list)
            or len(records) != expected_prefix
            or any(not isinstance(row, dict) for row in records)
            or [row.get("execution_order") for row in records] != list(range(expected_prefix))
            or spec.get("target_record") != records[-1]
            or spec.get("prefix_plan_sha256") != _json_sha256(records)
        ):
            errors.append(f"target_spec_structure_mismatch:{target_id}")
            continue
        if any(config.get("prefix_plan_sha256") != spec["prefix_plan_sha256"] for config in configs):
            errors.append(f"prefix_plan_binding_mismatch:{target_id}")
        for record in records:
            artifact_relative = f"frozen/targets/{target_id}/{record.get('input_artifact')}"
            artifact = _safe_artifact(root, artifact_relative)
            if (
                artifact is None
                or frozen.get(artifact_relative) != record.get("input_artifact_sha256")
            ):
                errors.append(f"prefix_input_hash_mismatch:{target_id}:{record.get('execution_order')}")
                continue
            try:
                with np.load(artifact, allow_pickle=False) as archive:
                    if set(archive.files) != {"prompt_token_ids"}:
                        raise ValueError("input NPZ must contain only prompt_token_ids")
                    prompts = np.array(archive["prompt_token_ids"], copy=True)
                expected_shape = (int(record["batch_size"]), int(record["prompt_length"]))
                if prompts.shape != expected_shape or not np.issubdtype(prompts.dtype, np.integer):
                    raise ValueError("prompt shape or dtype mismatch")
                if _json_sha256(prompts.astype(np.int64).tolist()) != record.get("prompt_token_ids_sha256"):
                    raise ValueError("prompt-token hash mismatch")
            except (KeyError, OSError, EOFError, TypeError, ValueError, zipfile.BadZipFile) as exc:
                errors.append(f"invalid_prefix_input:{target_id}:{record.get('execution_order')}:{exc}")
        terminal = records[-1]
        expected_group = 2 if runtime == "stock" else 1
        if (
            terminal.get("prompt_length") != 512
            or terminal.get("batch_size") != RUNTIME_SHAPES[runtime]["batch"]
            or terminal.get("group") != expected_group
            or terminal.get("within_process_repeat") != 0
            or terminal.get("execution_order") != expected_prefix - 1
        ):
            errors.append(f"terminal_target_shape_mismatch:{target_id}")
        bindings = {
            "target_runtime": runtime,
            "batch_size": RUNTIME_SHAPES[runtime]["batch"],
            "batch_id": terminal.get("batch_id"),
            "execution_order": expected_prefix - 1,
            "target_input_artifact_sha256": terminal.get("input_artifact_sha256"),
            "target_prompt_token_ids_sha256": terminal.get("prompt_token_ids_sha256"),
        }
        if any(target_plan.get(field) != value for field, value in bindings.items()):
            errors.append(f"run_plan_target_binding_mismatch:{target_id}")
        if (
            target_plan.get("source_bundle") != spec.get("source_bundle")
            or target_plan.get("source_batches_sha256") != spec.get("source_batches_sha256")
        ):
            errors.append(f"run_plan_target_provenance_mismatch:{target_id}")
        if any(
            config.get("target_input_artifact_sha256") != terminal.get("input_artifact_sha256")
            or config.get("target_prompt_token_ids_sha256") != terminal.get("prompt_token_ids_sha256")
            for config in configs
        ):
            errors.append(f"bundle_target_input_binding_mismatch:{target_id}")

    try:
        manifests = _load_json(root / "runtime-package-manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return errors + [f"invalid_runtime_package_manifest:{exc}"]
    variants = {"stock", "stock-device", "valid-window", "valid-window-device"}
    plan_manifest_sha = plan.get("runtime_package_manifest_sha256")
    if set(manifests) != variants or not isinstance(plan_manifest_sha, dict) or set(plan_manifest_sha) != variants:
        return errors + ["runtime_manifest_variants_invalid"]
    campaign_imports = plan.get("campaign_runtime_imports")
    if not isinstance(campaign_imports, dict) or set(campaign_imports) != variants:
        errors.append("campaign_runtime_import_variants_invalid")
        campaign_imports = {}
    for variant in sorted(variants):
        manifest = manifests.get(variant)
        runtime_root = root / "runtime" / variant
        if not isinstance(manifest, dict) or _package_manifest(runtime_root) != manifest:
            errors.append(f"runtime_package_files_mismatch:{variant}")
            continue
        digest = _json_sha256(manifest)
        if plan_manifest_sha.get(variant) != digest:
            errors.append(f"runtime_package_plan_hash_mismatch:{variant}")
        probe = campaign_imports.get(variant)
        if not isinstance(probe, dict):
            errors.append(f"campaign_runtime_import_probe_invalid:{variant}")
        else:
            expected_root = probe.get("expected_runtime_root")
            source_root = probe.get("source_root")
            module_file = probe.get("module_file")
            if (
                probe.get("logical_runtime_variant") != variant
                or probe.get("runtime_import_root_verified") is not True
                or probe.get("version") != "0.26.0"
                or not isinstance(expected_root, str)
                or not Path(expected_root).is_absolute()
                or source_root != expected_root
                or not isinstance(module_file, str)
                or Path(module_file).parent.parent != Path(expected_root)
                or Path(expected_root).parts[-2:] != ("runtime", variant)
            ):
                errors.append(f"campaign_runtime_import_probe_invalid:{variant}")
    try:
        n0b_manifest = _load_json(root / "frozen" / "n0b-runtime-package-manifest.json")
        if manifests.get("stock") != n0b_manifest.get("stock"):
            errors.append("stock_base_runtime_not_n0b_sealed")
        if manifests.get("valid-window") != n0b_manifest.get("optimized"):
            errors.append("valid_window_base_runtime_not_n0b_sealed")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_frozen_n0b_runtime_manifest:{exc}")
    for base in ("stock", "valid-window"):
        device = f"{base}-device"
        base_manifest = manifests.get(base)
        device_manifest = manifests.get(device)
        if not isinstance(base_manifest, dict) or not isinstance(device_manifest, dict):
            errors.append(f"device_runtime_manifest_invalid:{base}")
            continue
        changed = sorted(
            relative for relative in set(base_manifest) | set(device_manifest)
            if base_manifest.get(relative) != device_manifest.get(relative)
        )
        if changed != ["vllm/v1/worker/gpu_model_runner.py"]:
            errors.append(f"device_runtime_patch_allowlist_mismatch:{base}")

    workload_sha = frozen.get("frozen/workload.json")
    producer_sha = frozen.get("frozen/run_n0c_capture_stage_arm.py")
    for target_id, cells in cells_by_target.items():
        target_plan = plan_targets.get(target_id, {})
        base_patch = target_plan.get("base_runtime_patch_id")
        for (_, arm), cell in cells.items():
            config = cell["config"]
            variant = f"{config['target_runtime']}{'-device' if arm == 'capture_only' else ''}"
            expected_patch = f"{base_patch}+device-capture-no-export-v1" if arm == "capture_only" else base_patch
            import_probe = campaign_imports.get(variant)
            expected_import_root = (
                import_probe.get("expected_runtime_root")
                if isinstance(import_probe, dict) else None
            )
            if (
                config.get("logical_runtime_variant") != variant
                or config.get("runtime_import_root_verified") is not True
                or config.get("runtime_patch_id") != expected_patch
                or config.get("runtime_package_manifest_sha256") != plan_manifest_sha.get(variant)
                or config.get("workload_manifest_sha256") != workload_sha
                or config.get("producer_source_sha256") != producer_sha
            ):
                errors.append(f"bundle_source_binding_mismatch:{target_id}:{config.get('round')}:{arm}")
            identity = config.get("runtime_identity")
            expected_flag = "1" if arm == "capture_only" else "0"
            source_hashes = identity.get("source_sha256") if isinstance(identity, dict) else None
            variant_manifest = manifests.get(variant)
            if (
                not isinstance(identity, dict)
                or not isinstance(variant_manifest, dict)
                or identity.get("vllm") != "0.26.0"
                or identity.get("logical_runtime_variant") != variant
                or identity.get("runtime_import_root_verified") is not True
                or identity.get("expected_runtime_root") != expected_import_root
                or identity.get("vllm_source_root") != expected_import_root
                or identity.get("vllm_package") != str(Path(str(expected_import_root)) / "vllm")
                or identity.get("vllm_module_file")
                != str(Path(str(expected_import_root)) / "vllm" / "__init__.py")
                or identity.get("vllm_batch_invariant") != "0"
                or identity.get("n0c_device_capture_only") != expected_flag
                or not isinstance(source_hashes, dict)
                or any(
                    source_hashes.get(relative.removeprefix("vllm/")) != variant_manifest.get(relative)
                    for relative in (
                        "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py",
                        "vllm/v1/worker/gpu_model_runner.py",
                    )
                )
            ):
                errors.append(f"runtime_identity_mismatch:{target_id}:{config.get('round')}:{arm}")
    return errors


def evaluate_campaign(campaign_root: Path) -> dict[str, Any]:
    root = campaign_root.resolve()
    errors: list[str] = []
    bundle_root = root / "bundles"
    if not bundle_root.is_dir():
        errors.append("missing_bundles_directory")
        candidates: list[Path] = []
    else:
        candidates = sorted(
            {path.parent for name in ("config.json", "result.json", "RUN_COMPLETE.json") for path in bundle_root.rglob(name)}
        )
    if len(candidates) != 32:
        errors.append(f"expected_32_bundle_directories:found_{len(candidates)}")

    cells_by_target: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    for bundle in candidates:
        loaded, bundle_errors = _load_bundle(bundle)
        errors.extend(bundle_errors)
        if loaded is None:
            continue
        config = loaded["config"]
        target = str(config["target_id"])
        key = (int(config["round"]), str(config["arm"]))
        target_cells = cells_by_target.setdefault(target, {})
        if key in target_cells:
            errors.append(f"duplicate_schedule_cell:{target}:r{key[0]}:{key[1]}")
        else:
            target_cells[key] = loaded

    expected_keys = {(round_id, arm) for round_id in ROUNDS for arm in ARMS}
    if set(cells_by_target) != set(TARGET_RUNTIMES):
        errors.append(f"unexpected_target_ids:{sorted(cells_by_target)}")
    runtimes: list[str] = []
    for target, cells in sorted(cells_by_target.items()):
        if set(cells) != expected_keys:
            errors.append(f"incomplete_or_extra_schedule:{target}")
            continue
        configs = [cell["config"] for cell in cells.values()]
        runtimes.append(str(configs[0]["target_runtime"]))
        for field in (
            "target_runtime", "target_spec_sha256", "prefix_plan_sha256",
            "target_input_artifact_sha256", "target_prompt_token_ids_sha256",
        ):
            if len({str(config.get(field)) for config in configs}) != 1:
                errors.append(f"target_identity_drift:{target}:{field}")
        base_configs = [
            cell["config"] for (round_id, arm), cell in cells.items()
            if arm in {"n_a", "n_b", "full_export"}
        ]
        device_configs = [
            cell["config"] for (round_id, arm), cell in cells.items()
            if arm == "capture_only"
        ]
        for variant, variant_configs in (("base", base_configs), ("device", device_configs)):
            pairs = {
                (config.get("runtime_patch_id"), config.get("runtime_package_manifest_sha256"))
                for config in variant_configs
            }
            if len(pairs) != 1:
                errors.append(f"runtime_variant_identity_drift:{target}:{variant}")
        if base_configs and device_configs and (
            base_configs[0].get("runtime_patch_id") == device_configs[0].get("runtime_patch_id")
        ):
            errors.append(f"device_patch_id_not_distinct:{target}")
        if base_configs and device_configs and (
            base_configs[0].get("runtime_package_manifest_sha256")
            == device_configs[0].get("runtime_package_manifest_sha256")
        ):
            errors.append(f"device_package_manifest_not_distinct:{target}")
    if sorted(runtimes) != sorted(RUNTIME_SHAPES):
        errors.append("targets_must_cover_stock_and_valid-window_once")
    for target, runtime in TARGET_RUNTIMES.items():
        cells = cells_by_target.get(target, {})
        if cells and any(cell["config"]["target_runtime"] != runtime for cell in cells.values()):
            errors.append(f"target_runtime_binding_mismatch:{target}")
    if not errors:
        try:
            plan = _load_json(root / "run_plan.json")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid_or_missing_run_plan:{exc}")
        else:
            errors.extend(_schedule_errors(plan))
            errors.extend(_campaign_identity_errors(root, plan, cells_by_target))

    if errors:
        return {
            "schema": SCHEMA,
            "status": "INVALID_CAMPAIGN",
            "failure_category": "IDENTITY_SEAL_INPUT_OR_SOURCE_INVALID",
            "claim_ceiling": CLAIM_CEILING,
            "structurally_valid": False,
            "controller_unlocked": False,
            "bundle_count": len(candidates),
            "errors": errors,
        }

    target_reports = [
        _target_report(target, cells_by_target[target]) for target in sorted(cells_by_target)
    ]
    selected = next(
        status for status in VALID_PRECEDENCE
        if any(target["status"] == status for target in target_reports)
    )
    return {
        "schema": SCHEMA,
        "status": selected,
        "failure_category": None if selected == "NOT_REPRODUCED" else selected,
        "claim_ceiling": CLAIM_CEILING,
        "evidence_type": "NATIVE_SINGLE_GPU_FRESH_PROCESS_ASSOCIATIONAL",
        "structurally_valid": True,
        "controller_unlocked": False,
        "action_oracle_unlocked": False,
        "bundle_count": len(candidates),
        "threshold": {"rounds": 4, "required_matching_first_divergence_rounds": 3},
        "decision_basis": "matched_first_output_token_divergence_signature_only",
        "float_hashes_used_for_token_explanation": False,
        "full_export_route_drift_is_diagnostic_only": True,
        "identity_verification": {
            "frozen_input_files_recomputed": True,
            "target_specs_and_prompt_npz_recomputed": True,
            "runtime_package_files_recomputed": True,
            "arm_runtime_identity_bound": True,
        },
        "targets": target_reports,
        "anti_claims": [
            "fresh-process association is not a same-prestate causal intervention",
            "capture-only retains scheduler route-manager bookkeeping and is not a pure device-kernel treatment",
            "route or float-hash drift does not explain token drift",
            "this Gate does not unlock a decode-cap action or Controller",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; N0c evaluation artifacts are write-once")
    report = evaluate_campaign(args.campaign_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report.get(key) for key in ("status", "failure_category")}))
    if report["status"] == "INVALID_CAMPAIGN":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
