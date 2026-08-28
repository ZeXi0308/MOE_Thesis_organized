#!/usr/bin/env python3
"""Fail-closed stock versus valid-window route-telemetry qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "vllm-route-telemetry-implementation-gate-v3"
CLAIM_CEILING = "NATIVE_OFFLINE_FIXED_BATCH_TELEMETRY_IMPLEMENTATION_ONLY"
MINIMUM_PROCESS_REPEATS = 2
FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT = 5.0
EXPECTED_PATCH_IDS = {
    "stock": "stock-vllm-0.26.0",
    "optimized": "valid-window-clear-v1",
}
QUALIFIED_PAIR_STATUSES = {
    "TELEMETRY_OVERHEAD_QUALIFIED",  # compatibility with v1 output name
    "TELEMETRY_TIMING_DEVIATION_QUALIFIED",
}

# Campaign aggregation must preserve the strongest failure seen in any retained
# repeat.  In particular, a timing-only failure in an earlier repeat must not
# hide output drift in a later repeat.
CAMPAIGN_INVALID_PRECEDENCE = (
    "INVALID_IMPLEMENTATION_IDENTITY",
    "INVALID_IMPLEMENTATION_PAIR",
    "INVALID_PATCH_CONTROL",
)
CAMPAIGN_FAILURE_PRECEDENCE = (
    "VALID_WINDOW_NOT_TRANSPARENT",
    "VALID_WINDOW_ROUTE_SEMANTICS_MISMATCH",
    "VALID_WINDOW_ROUTE_SEMANTICS_INCONCLUSIVE",
    "VALID_WINDOW_TOO_PERTURBATIVE",
)


def _load_module(filename: str, module_name: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARATOR = _load_module("compare_vllm_route_probe_runs.py", "route_probe_comparator")
VALIDATOR = _load_module(
    "vllm_patches/validate_valid_window_patch.py", "valid_window_source_validator"
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["prompt_length"]),
        int(row["batch_size"]),
        int(row["group"]),
        int(row["within_process_repeat"]),
    )


def _tokens(row: dict[str, Any]) -> list[list[int]]:
    return [list(request["token_ids"]) for request in row["request_metrics"]]


def _paths(value: Path | Sequence[Path]) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    return [Path(path) for path in value]


def _bundle(path: Path) -> dict[str, Any]:
    path = path.resolve()
    integrity = COMPARATOR.verify_bundle(path)
    config = _json(path / "config.json") if integrity["valid"] else {}
    rows = _jsonl(path / "batches.jsonl") if integrity["valid"] else []
    return {
        "path": path,
        "integrity": integrity,
        "config": config,
        "rows": rows,
        "map": {_key(row): row for row in rows},
        "process_repeat": int(config.get("process_repeat", -1)),
    }


def _bundle_set(paths: Path | Sequence[Path]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    bundles = [_bundle(path) for path in _paths(paths)]
    errors: list[str] = []
    result: dict[int, dict[str, Any]] = {}
    for bundle in bundles:
        repeat = bundle["process_repeat"]
        if repeat < 0:
            errors.append(f"invalid_process_repeat:{bundle['path']}")
        elif repeat in result:
            errors.append(f"duplicate_process_repeat:{repeat}")
        else:
            result[repeat] = bundle
    return result, errors


def _runtime_without_sources(identity: Any) -> Any:
    if not isinstance(identity, dict):
        return identity
    return {key: value for key, value in identity.items() if key != "vllm_runtime_sources"}


def _cross_runtime_config_drift(
    stock: dict[str, Any], optimized: dict[str, Any]
) -> dict[str, list[Any]]:
    ignored = {"runtime_patch_id", "runtime_identity"}
    fields = set(COMPARATOR.MATCHED_CONFIG_FIELDS) - ignored
    drift = {
        field: [stock.get(field), optimized.get(field)]
        for field in sorted(fields)
        if stock.get(field) != optimized.get(field)
    }
    stock_runtime = _runtime_without_sources(stock.get("runtime_identity"))
    optimized_runtime = _runtime_without_sources(optimized.get("runtime_identity"))
    if stock_runtime != optimized_runtime:
        drift["runtime_identity_without_sources"] = [stock_runtime, optimized_runtime]
    return drift


def _repeat_config_drift(bundles: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
    repeats = sorted(bundles)
    if not repeats:
        return {"empty": True}
    reference = bundles[repeats[0]]["config"]
    fields = set(COMPARATOR.MATCHED_CONFIG_FIELDS) - {"process_repeat"}
    drift: dict[str, Any] = {}
    for repeat in repeats[1:]:
        current = bundles[repeat]["config"]
        changed = {
            field: [reference.get(field), current.get(field)]
            for field in sorted(fields)
            if reference.get(field) != current.get(field)
        }
        if changed:
            drift[str(repeat)] = changed
    return drift


def _token_parity(
    left: Mapping[tuple[int, int, int, int], dict[str, Any]],
    right: Mapping[tuple[int, int, int, int], dict[str, Any]],
) -> tuple[bool, list[list[int]], list[list[int]]]:
    keys = sorted(set(left) | set(right))
    token_mismatches: list[list[int]] = []
    prompt_mismatches: list[list[int]] = []
    for key in keys:
        if key not in left or key not in right:
            token_mismatches.append(list(key))
            prompt_mismatches.append(list(key))
            continue
        if _tokens(left[key]) != _tokens(right[key]):
            token_mismatches.append(list(key))
        if left[key].get("prompt_token_ids_sha256") != right[key].get(
            "prompt_token_ids_sha256"
        ):
            prompt_mismatches.append(list(key))
    return not token_mismatches and not prompt_mismatches, token_mismatches, prompt_mismatches


def _normalized_source_hashes(config: dict[str, Any]) -> dict[str, str]:
    identity = config.get("runtime_identity")
    if not isinstance(identity, dict):
        return {}
    raw = identity.get("vllm_runtime_sources")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for relative, value in raw.items():
        key = str(relative)
        if not key.startswith("vllm/"):
            key = f"vllm/{key}"
        if isinstance(value, dict):
            digest = value.get("sha256")
        else:
            digest = value
        if isinstance(digest, str):
            result[key] = digest
    return result


def _validate_source_identity(
    bundle_sets: Mapping[str, Mapping[int, dict[str, Any]]]
) -> dict[str, Any]:
    errors: list[str] = []
    expected_runner = _sha256(Path(__file__).with_name("run_vllm_route_shape_probe.py"))
    checked: list[dict[str, Any]] = []
    for label, bundles in bundle_sets.items():
        implementation = "stock" if label.startswith("stock_") else "optimized"
        expected_sources = {
            relative: states["original" if implementation == "stock" else "patched"]
            for relative, states in VALIDATOR.FILES.items()
        }
        expected_patch_id = EXPECTED_PATCH_IDS[implementation]
        for repeat, bundle in sorted(bundles.items()):
            config = bundle["config"]
            runtime = config.get("runtime_identity")
            sources = _normalized_source_hashes(config)
            item_errors: list[str] = []
            if not isinstance(runtime, dict) or runtime.get("vllm") != "0.26.0":
                item_errors.append("unexpected_vllm_version")
            if config.get("runtime_patch_id") != expected_patch_id:
                item_errors.append("unexpected_runtime_patch_id")
            if sources != expected_sources:
                item_errors.append("runtime_source_hashes_not_validator_approved")
            producer_relative = config.get("producer_source_artifact")
            producer_hash = config.get("producer_source_artifact_sha256")
            if producer_relative != "producer_source.py" or producer_hash != expected_runner:
                item_errors.append("producer_source_identity_mismatch")
            else:
                producer_path = bundle["path"] / producer_relative
                try:
                    if not producer_path.is_file() or _sha256(producer_path) != expected_runner:
                        item_errors.append("embedded_producer_source_mismatch")
                    manifest = _json(bundle["path"] / "ARTIFACT_HASHES.json")
                    if manifest.get(producer_relative) != expected_runner:
                        item_errors.append("producer_source_not_sealed")
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    item_errors.append("producer_source_validation_error")
            if config.get("probe_script_sha256") != expected_runner:
                item_errors.append("stale_probe_script_sha256")
            if item_errors:
                errors.extend(f"{label}:r{repeat}:{error}" for error in item_errors)
            checked.append(
                {
                    "arm": label,
                    "process_repeat": repeat,
                    "implementation": implementation,
                    "runtime_patch_id": config.get("runtime_patch_id"),
                    "runtime_source_hashes": sources,
                    "errors": item_errors,
                }
            )
    return {
        "valid": not errors,
        "expected_vllm_version": "0.26.0",
        "expected_patch_sha256": VALIDATOR.PATCH_SHA256,
        "expected_probe_script_sha256": expected_runner,
        "errors": errors,
        "checked_arms": checked,
    }


def _validated_route_array(bundle: dict[str, Any], row: dict[str, Any]) -> np.ndarray:
    relative = Path(str(row.get("route_artifact", "")))
    path = (bundle["path"] / relative).resolve()
    try:
        path.relative_to(bundle["path"])
    except ValueError as exc:
        raise ValueError(f"route artifact escapes bundle:{relative}") from exc
    expected_hash = row.get("route_artifact_sha256")
    manifest = _json(bundle["path"] / "ARTIFACT_HASHES.json")
    if not path.is_file() or not isinstance(expected_hash, str):
        raise ValueError(f"missing route artifact:{relative}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash or manifest.get(str(relative)) != actual_hash:
        raise ValueError(f"route artifact hash mismatch:{relative}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"routes"}:
            raise ValueError(f"invalid route NPZ keys:{payload.files}")
        routes = np.asarray(payload["routes"])
    model_shape = bundle["config"]["model_shape"]
    expected_shape = (
        int(row["batch_size"]),
        int(bundle["config"]["output_tokens"]) - 1,
        int(model_shape["num_layers"]),
        int(model_shape["top_k"]),
    )
    if routes.shape != expected_shape:
        raise ValueError(f"route shape mismatch:{routes.shape}:expected:{expected_shape}")
    if not np.issubdtype(routes.dtype, np.integer):
        raise ValueError(f"route dtype is not integer:{routes.dtype}")
    num_experts = int(model_shape["num_experts"])
    if np.any(routes < 0) or np.any(routes >= num_experts):
        raise ValueError("expert ID outside configured range")
    if routes.shape[-1] > 1 and np.any(np.diff(np.sort(routes, axis=-1), axis=-1) == 0):
        raise ValueError("duplicate expert IDs inside top-k")
    return routes


def _route_semantics(stock_on: dict[str, Any], optimized_on: dict[str, Any]) -> dict[str, Any]:
    expected = sorted(set(stock_on["map"]) | set(optimized_on["map"]))
    exact = 0
    comparable = 0
    route_mismatches: list[list[int]] = []
    token_drift: list[list[int]] = []
    prompt_drift: list[list[int]] = []
    missing: list[list[int]] = []
    validation_errors: list[str] = []
    for key in expected:
        stock_row = stock_on["map"].get(key)
        optimized_row = optimized_on["map"].get(key)
        if stock_row is None or optimized_row is None:
            missing.append(list(key))
            continue
        if stock_row.get("prompt_token_ids_sha256") != optimized_row.get(
            "prompt_token_ids_sha256"
        ):
            prompt_drift.append(list(key))
            continue
        if _tokens(stock_row) != _tokens(optimized_row):
            token_drift.append(list(key))
            continue
        comparable += 1
        try:
            stock_routes = _validated_route_array(stock_on, stock_row)
            optimized_routes = _validated_route_array(optimized_on, optimized_row)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            validation_errors.append(f"{list(key)}:{exc}")
            continue
        if np.array_equal(stock_routes, optimized_routes):
            exact += 1
        else:
            route_mismatches.append(list(key))
    qualified = bool(
        expected
        and comparable == len(expected)
        and exact == len(expected)
        and not missing
        and not prompt_drift
        and not token_drift
        and not route_mismatches
        and not validation_errors
    )
    return {
        "expected_cells": len(expected),
        "comparable_cells": comparable,
        "exact_route_cells": exact,
        "comparable_fraction": comparable / len(expected) if expected else 0.0,
        "route_mismatch_keys": route_mismatches,
        "token_drift_keys": token_drift,
        "prompt_digest_mismatch_keys": prompt_drift,
        "missing_keys": missing,
        "validation_errors": validation_errors,
        "qualified": qualified,
    }


def _pair_structure_errors(pair: Mapping[str, Any]) -> bool:
    return bool(
        pair.get("config_drift")
        or pair.get("duplicate_keys")
        or pair.get("missing_on")
        or pair.get("missing_off")
        or pair.get("incomplete_on")
        or pair.get("incomplete_off")
        or pair.get("unexpected_on")
        or pair.get("unexpected_off")
        or pair.get("prompt_digest_mismatches")
        or pair.get("timing_errors")
        or pair.get("row_schema_errors")
        or pair.get("threshold_valid") is False
        or int(pair.get("pair_count", 0)) <= 0
    )


def choose_status(
    stock_pair: Mapping[str, Any],
    optimized_pair: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    cross_off_token_parity: bool,
    cross_config_drift: Mapping[str, Any],
    route_semantics: Mapping[str, Any],
) -> tuple[str, str | None]:
    if not source_identity.get("valid"):
        return "INVALID_IMPLEMENTATION_IDENTITY", "SOURCE_IDENTITY_NOT_VALIDATOR_APPROVED"
    if (
        _pair_structure_errors(stock_pair)
        or _pair_structure_errors(optimized_pair)
        or cross_config_drift
        or route_semantics.get("missing_keys")
        or route_semantics.get("prompt_digest_mismatch_keys")
        or route_semantics.get("validation_errors")
    ):
        return "INVALID_IMPLEMENTATION_PAIR", "CONFIG_COVERAGE_OR_ARTIFACT_DRIFT"
    if not cross_off_token_parity:
        return "INVALID_PATCH_CONTROL", "PATCH_AFFECTS_ROUTE_OFF_EXECUTION"
    if not optimized_pair.get("token_parity"):
        return "VALID_WINDOW_NOT_TRANSPARENT", "TELEMETRY_TOKEN_DRIFT"
    if (
        int(route_semantics.get("expected_cells", 0)) <= 0
        or int(route_semantics.get("comparable_cells", 0))
        != int(route_semantics.get("expected_cells", 0))
        or route_semantics.get("token_drift_keys")
    ):
        return (
            "VALID_WINDOW_ROUTE_SEMANTICS_INCONCLUSIVE",
            "CROSS_IMPLEMENTATION_TOKEN_DRIFT_OR_ZERO_SUPPORT",
        )
    if (
        route_semantics.get("route_mismatch_keys")
        or int(route_semantics.get("exact_route_cells", 0))
        != int(route_semantics.get("expected_cells", 0))
        or not route_semantics.get("qualified")
    ):
        return "VALID_WINDOW_ROUTE_SEMANTICS_MISMATCH", "LOSSLESS_ROUTE_CONTRACT_FAILED"
    if optimized_pair.get("status") not in QUALIFIED_PAIR_STATUSES:
        return "VALID_WINDOW_TOO_PERTURBATIVE", "TELEMETRY_TIMING_DEVIATION_ABOVE_THRESHOLD"
    return "VALID_WINDOW_TELEMETRY_QUALIFIED", None


def _repeat_coverage_errors(
    bundle_sets: Mapping[str, Mapping[int, dict[str, Any]]]
) -> list[str]:
    errors: list[str] = []
    repeat_sets = {name: set(bundles) for name, bundles in bundle_sets.items()}
    union = set().union(*repeat_sets.values()) if repeat_sets else set()
    if len(union) < MINIMUM_PROCESS_REPEATS:
        errors.append(f"minimum_process_repeats:{len(union)}<{MINIMUM_PROCESS_REPEATS}")
    for name, repeats in repeat_sets.items():
        if repeats != union:
            errors.append(f"repeat_coverage:{name}:{sorted(repeats)}!=expected:{sorted(union)}")
    return errors


def _select_campaign_verdict(
    repeat_reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for status in CAMPAIGN_INVALID_PRECEDENCE:
        selected = next(
            (row for row in repeat_reports if row.get("status") == status),
            None,
        )
        if selected is not None:
            return selected
    unknown_invalid = next(
        (
            row
            for row in repeat_reports
            if str(row.get("status", "")).startswith("INVALID_")
        ),
        None,
    )
    if unknown_invalid is not None:
        return unknown_invalid
    for status in CAMPAIGN_FAILURE_PRECEDENCE:
        selected = next(
            (row for row in repeat_reports if row.get("status") == status),
            None,
        )
        if selected is not None:
            return selected
    return next(
        (
            row
            for row in repeat_reports
            if row.get("status") != "VALID_WINDOW_TELEMETRY_QUALIFIED"
        ),
        None,
    )


def compare_implementations(
    stock_off_paths: Path | Sequence[Path],
    stock_on_paths: Path | Sequence[Path],
    optimized_off_paths: Path | Sequence[Path],
    optimized_on_paths: Path | Sequence[Path],
    max_p95_overhead_pct: float = FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT,
) -> dict[str, Any]:
    try:
        requested_threshold = float(max_p95_overhead_pct)
    except (TypeError, ValueError):
        requested_threshold = None
        threshold_error = "threshold_is_not_numeric"
    else:
        if not math.isfinite(requested_threshold):
            threshold_error = "threshold_is_not_finite"
        elif requested_threshold < 0:
            threshold_error = "threshold_is_negative"
        elif requested_threshold != FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT:
            threshold_error = "threshold_does_not_match_frozen_gate"
        else:
            threshold_error = None

    threshold_base = {
        "schema": SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "minimum_process_repeats": MINIMUM_PROCESS_REPEATS,
        "max_p95_absolute_timing_deviation_pct": (
            FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT
        ),
        "requested_max_p95_overhead_pct": (
            requested_threshold
            if requested_threshold is not None and math.isfinite(requested_threshold)
            else repr(max_p95_overhead_pct)
        ),
    }
    if threshold_error is not None:
        return threshold_base | {
            "status": "INVALID_INPUT",
            "failure_category": "INVALID_THRESHOLD",
            "errors": {
                "threshold": [
                    threshold_error,
                    (
                        "required_exact_value:"
                        f"{FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT}"
                    ),
                ]
            },
        }

    raw_sets = {
        "stock_off": stock_off_paths,
        "stock_on": stock_on_paths,
        "optimized_off": optimized_off_paths,
        "optimized_on": optimized_on_paths,
    }
    bundle_sets: dict[str, dict[int, dict[str, Any]]] = {}
    set_errors: list[str] = []
    for name, paths in raw_sets.items():
        bundle_sets[name], errors = _bundle_set(paths)
        set_errors.extend(f"{name}:{error}" for error in errors)
    integrity_errors = {
        f"{name}:r{repeat}": bundle["integrity"]["errors"]
        for name, bundles in bundle_sets.items()
        for repeat, bundle in bundles.items()
        if not bundle["integrity"]["valid"]
    }
    base = threshold_base | {
        "bundles": {
            name: [str(path) for path in _paths(paths)] for name, paths in raw_sets.items()
        },
        "bundle_integrity": {
            f"{name}:r{repeat}": bundle["integrity"]
            for name, bundles in bundle_sets.items()
            for repeat, bundle in bundles.items()
        },
    }
    coverage_errors = _repeat_coverage_errors(bundle_sets)
    if set_errors or integrity_errors or coverage_errors:
        return base | {
            "status": "INVALID_INPUT",
            "failure_category": "BUNDLE_INTEGRITY_OR_REPEAT_COVERAGE",
            "errors": {
                "bundle_sets": set_errors,
                "bundle_integrity": integrity_errors,
                "repeat_coverage": coverage_errors,
            },
        }

    environment_errors = {
        f"{name}:r{repeat}": "exclusive_gpu_not_verified"
        for name, bundles in bundle_sets.items()
        for repeat, bundle in bundles.items()
        if bundle["integrity"].get("exclusive_gpu_verified") is not True
    }
    if environment_errors:
        return base | {
            "status": "INVALID_INPUT",
            "failure_category": "ENVIRONMENT_NOT_QUALIFIED",
            "errors": {"environment": environment_errors},
        }

    expected_capture = {
        "stock_off": False,
        "stock_on": True,
        "optimized_off": False,
        "optimized_on": True,
    }
    capture_errors = [
        f"{name}:r{repeat}"
        for name, expected in expected_capture.items()
        for repeat, bundle in bundle_sets[name].items()
        if bool(bundle["config"].get("capture_routes")) != expected
    ]
    repeat_config_drift = {
        name: drift
        for name, bundles in bundle_sets.items()
        if (drift := _repeat_config_drift(bundles))
    }
    source_identity = _validate_source_identity(bundle_sets)
    if capture_errors or repeat_config_drift:
        return base | {
            "status": "INVALID_INPUT",
            "failure_category": "ARM_OR_REPEAT_IDENTITY",
            "capture_errors": capture_errors,
            "repeat_config_drift": repeat_config_drift,
            "source_identity": source_identity,
        }

    repeats = sorted(bundle_sets["stock_off"])
    repeat_reports: list[dict[str, Any]] = []
    for repeat in repeats:
        stock_off = bundle_sets["stock_off"][repeat]
        stock_on = bundle_sets["stock_on"][repeat]
        optimized_off = bundle_sets["optimized_off"][repeat]
        optimized_on = bundle_sets["optimized_on"][repeat]
        stock_pair = COMPARATOR.compare_runs(
            stock_on["config"], stock_off["config"], stock_on["rows"], stock_off["rows"],
            FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT,
        )
        optimized_pair = COMPARATOR.compare_runs(
            optimized_on["config"], optimized_off["config"], optimized_on["rows"],
            optimized_off["rows"], FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT,
        )
        cross_config = _cross_runtime_config_drift(
            stock_off["config"], optimized_off["config"]
        )
        off_parity, off_token_mismatches, off_prompt_mismatches = _token_parity(
            stock_off["map"], optimized_off["map"]
        )
        route_semantics = _route_semantics(stock_on, optimized_on)
        status, failure = choose_status(
            stock_pair,
            optimized_pair,
            source_identity,
            off_parity,
            cross_config,
            route_semantics,
        )
        repeat_reports.append(
            {
                "process_repeat": repeat,
                "status": status,
                "failure_category": failure,
                "stock_pair": stock_pair,
                "stock_control_interpretation": (
                    "PERTURBATIVE_NEGATIVE_CONTROL"
                    if not stock_pair.get("token_parity")
                    else "TOKEN_TRANSPARENT_CONTROL"
                ),
                "optimized_pair": optimized_pair,
                "cross_runtime_config_drift": cross_config,
                "cross_runtime_route_OFF_token_parity": off_parity,
                "cross_runtime_route_OFF_token_mismatch_keys": off_token_mismatches,
                "cross_runtime_route_OFF_prompt_mismatch_keys": off_prompt_mismatches,
                "lossless_route_semantics": route_semantics,
            }
        )

    selected = _select_campaign_verdict(repeat_reports)
    status = selected["status"] if selected else "VALID_WINDOW_TELEMETRY_QUALIFIED"
    failure = selected["failure_category"] if selected else None
    return base | {
        "status": status,
        "failure_category": failure,
        "process_repeats": repeats,
        "all_required_repeats_retained": True,
        "source_identity": source_identity,
        "repeat_reports": repeat_reports,
        "campaign_repeat_failures": [
            {
                "process_repeat": row["process_repeat"],
                "status": row["status"],
                "failure_category": row["failure_category"],
            }
            for row in repeat_reports
            if row["status"] != "VALID_WINDOW_TELEMETRY_QUALIFIED"
        ],
        "anti_claims": [
            "telemetry qualification is not a pressure-latency result",
            "telemetry qualification is not scheduling or admission headroom",
            "stock route-ON is a control, not ground-truth execution",
            "single-GPU eager evidence is not Expert Parallel evidence",
        ],
    }


def exit_code(status: str) -> int:
    if status == "VALID_WINDOW_TELEMETRY_QUALIFIED":
        return 0
    if status.startswith("INVALID_"):
        return 2
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-off", type=Path, nargs="+", required=True)
    parser.add_argument("--stock-on", type=Path, nargs="+", required=True)
    parser.add_argument("--optimized-off", type=Path, nargs="+", required=True)
    parser.add_argument("--optimized-on", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-p95-overhead-pct",
        type=float,
        default=FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT,
        help=(
            "frozen Gate threshold; only the exact value "
            f"{FROZEN_MAX_P95_ABSOLUTE_TIMING_DEVIATION_PCT} is accepted"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; comparison artifacts are write-once")
    report = compare_implementations(
        args.stock_off,
        args.stock_on,
        args.optimized_off,
        args.optimized_on,
        args.max_p95_overhead_pct,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps({key: report.get(key) for key in ("status", "failure_category")}))
    code = exit_code(str(report["status"]))
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
