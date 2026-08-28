#!/usr/bin/env python3
"""Localize same-arm repeat divergence in a sealed N0b campaign.

This is an associational diagnostic.  Separate fresh processes are compared;
the result cannot attribute divergence causally to telemetry and never consumes
timing as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "n0b-repeat-divergence-localization-v2"
CLAIM_CEILING = "NATIVE_OFFLINE_FIXED_BATCH_ASSOCIATIONAL_LOCALIZATION"
ARMS = ("stock_off", "stock_on", "optimized_off", "optimized_on")


def _load_gate() -> Any:
    path = Path(__file__).with_name("compare_vllm_telemetry_implementations.py")
    spec = importlib.util.spec_from_file_location("n0b_gate_v3", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(row: Mapping[str, Any]) -> list[list[int]]:
    return [list(request["token_ids"]) for request in row["request_metrics"]]


def _first_token_differences(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> list[dict[str, Any]]:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if len(left_tokens) != len(right_tokens):
        raise ValueError("request count drift")
    result: list[dict[str, Any]] = []
    for row, (left_ids, right_ids) in enumerate(zip(left_tokens, right_tokens)):
        if len(left_ids) != len(right_ids):
            raise ValueError(f"output length drift at request row {row}")
        changed = [index for index, pair in enumerate(zip(left_ids, right_ids)) if pair[0] != pair[1]]
        if changed:
            result.append(
                {
                    "request_row": row,
                    "first_output_token_index": changed[0],
                    "changed_output_token_count": len(changed),
                }
            )
    return result


def _route_differences(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"route shape drift:{left.shape}!={right.shape}")
    if left.ndim != 4:
        raise ValueError(f"expected [request,step,layer,top_k], got {left.shape}")
    raw_changed = np.any(left != right, axis=-1)
    set_changed = np.any(
        np.sort(left, axis=-1) != np.sort(right, axis=-1), axis=-1
    )
    first_by_request: dict[int, list[int]] = {}
    for request_row in range(left.shape[0]):
        positions = np.argwhere(set_changed[request_row])
        if len(positions):
            first_by_request[request_row] = [
                int(positions[0, 0]),
                int(positions[0, 1]),
            ]
    return {
        "raw_route_exact": bool(not np.any(raw_changed)),
        "topk_set_exact": bool(not np.any(set_changed)),
        "changed_raw_route_positions": int(np.count_nonzero(raw_changed)),
        "changed_topk_set_positions": int(np.count_nonzero(set_changed)),
        "total_request_step_layer_positions": int(np.prod(left.shape[:-1])),
        "first_topk_set_divergence_by_request": {
            str(key): value for key, value in sorted(first_by_request.items())
        },
    }


def _align_route_to_output(
    first_route_step_layer: list[int] | None,
    first_output_token_index: int,
) -> dict[str, Any]:
    # The producer intentionally stores full_routes[1:]. full_routes[0] is the
    # prompt-tail forward that produces output token 0; retained route step s
    # is therefore the decode forward that produces output token s + 1.
    produced_output_token_index = (
        first_route_step_layer[0] + 1
        if first_route_step_layer is not None
        else None
    )
    return {
        "first_route_step_layer": first_route_step_layer,
        "route_forward_produced_output_token_index": produced_output_token_index,
        "first_output_token_index": first_output_token_index,
        "captured_route_forward_no_later_than_output_divergence": bool(
            produced_output_token_index is not None
            and produced_output_token_index <= first_output_token_index
        ),
    }


def _cell_report(
    left_bundle: Mapping[str, Any],
    right_bundle: Mapping[str, Any],
    key: tuple[int, int, int, int],
    *,
    capture_routes: bool,
) -> dict[str, Any]:
    left_row, right_row = left_bundle["map"][key], right_bundle["map"][key]
    if left_row.get("prompt_token_ids_sha256") != right_row.get(
        "prompt_token_ids_sha256"
    ):
        raise ValueError(f"prompt digest drift:{list(key)}")
    token_differences = _first_token_differences(left_row, right_row)
    result: dict[str, Any] = {
        "key": list(key),
        "token_exact": not token_differences,
        "token_differences": token_differences,
    }
    if capture_routes:
        left_routes = GATE._validated_route_array(left_bundle, left_row)
        right_routes = GATE._validated_route_array(right_bundle, right_row)
        route = _route_differences(left_routes, right_routes)
        first_routes = route["first_topk_set_divergence_by_request"]
        aligned: list[dict[str, Any]] = []
        for token in token_differences:
            request = str(token["request_row"])
            first_route = first_routes.get(request)
            aligned.append(
                {"request_row": token["request_row"]}
                | _align_route_to_output(
                    first_route, token["first_output_token_index"]
                )
            )
        result["route"] = route
        result["route_to_output_alignment"] = aligned
    return result


def _arm_report(
    bundles: Mapping[int, Mapping[str, Any]], *, capture_routes: bool
) -> dict[str, Any]:
    if set(bundles) != {0, 1}:
        raise ValueError(f"expected process repeats [0,1], got {sorted(bundles)}")
    left, right = bundles[0], bundles[1]
    keys = sorted(set(left["map"]) | set(right["map"]))
    if set(left["map"]) != set(right["map"]):
        raise ValueError("cell coverage drift across process repeats")
    cells = [
        _cell_report(left, right, key, capture_routes=capture_routes)
        for key in keys
    ]
    token_drift = [row for row in cells if not row["token_exact"]]
    route_drift = [
        row
        for row in cells
        if capture_routes and not row["route"]["topk_set_exact"]
    ]
    return {
        "cell_count": len(cells),
        "token_exact_cells": len(cells) - len(token_drift),
        "token_drift_keys": [row["key"] for row in token_drift],
        "topk_set_exact_cells": (
            len(cells) - len(route_drift) if capture_routes else None
        ),
        "topk_set_drift_keys": [row["key"] for row in route_drift],
        "divergent_cells": [
            row
            for row in cells
            if not row["token_exact"]
            or (capture_routes and not row["route"]["topk_set_exact"])
        ],
    }


def analyze(campaign_root: Path) -> dict[str, Any]:
    campaign_root = campaign_root.resolve()
    terminal = campaign_root / "CAMPAIGN_COMPLETE.json"
    raw_report = campaign_root / "valid-window-gate.json"
    if not terminal.is_file() or not raw_report.is_file():
        raise ValueError("campaign terminal or raw report is missing")
    terminal_data = json.loads(terminal.read_text())
    if terminal_data.get("comparison_sha256") != _sha256(raw_report):
        raise ValueError("raw report hash does not match campaign terminal")

    paths = {
        arm: [campaign_root / "bundles" / f"{arm}-r{repeat}" for repeat in (0, 1)]
        for arm in ARMS
    }
    gate_report = GATE.compare_implementations(
        paths["stock_off"],
        paths["stock_on"],
        paths["optimized_off"],
        paths["optimized_on"],
    )
    if str(gate_report.get("status", "")).startswith("INVALID_"):
        raise ValueError(f"campaign is not structurally valid:{gate_report['status']}")

    bundle_sets: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ARMS:
        bundles, errors = GATE._bundle_set(paths[arm])
        if errors or any(not item["integrity"]["valid"] for item in bundles.values()):
            raise ValueError(f"bundle integrity failure:{arm}:{errors}")
        bundle_sets[arm] = bundles
    arm_reports = {
        arm: _arm_report(bundle_sets[arm], capture_routes=arm.endswith("_on"))
        for arm in ARMS
    }

    route_off_stable = all(
        not arm_reports[arm]["token_drift_keys"]
        for arm in ("stock_off", "optimized_off")
    )
    route_on_drift = any(
        arm_reports[arm]["token_drift_keys"]
        for arm in ("stock_on", "optimized_on")
    )
    selected_targets = [
        {"arm": arm, "key": key}
        for arm in ("stock_on", "optimized_on")
        for key in arm_reports[arm]["token_drift_keys"]
    ]
    aligned = [
        item["captured_route_forward_no_later_than_output_divergence"]
        for arm in ("stock_on", "optimized_on")
        for cell in arm_reports[arm]["divergent_cells"]
        for item in cell.get("route_to_output_alignment", [])
    ]
    if not route_off_stable:
        status = "BASELINE_REPEAT_NONDETERMINISM"
        failure = "ROUTE_OFF_OUTPUT_DRIFT"
        next_step = "Diagnose baseline runtime/kernel determinism before telemetry attribution."
    elif route_on_drift and aligned and all(aligned):
        status = "TELEMETRY_ENABLED_REPEAT_DIVERGENCE_LOCALIZED"
        failure = "ASSOCIATIONAL_ROUTE_BEFORE_OUTPUT_DIVERGENCE"
        next_step = (
            "Replay only the frozen divergent cells with independent OFF/OFF controls "
            "and step-local hidden/router-logit/top-k diagnostics; do not time the run."
        )
    elif route_on_drift and any(aligned):
        status = "TELEMETRY_ENABLED_REPEAT_DIVERGENCE_PARTIALLY_LOCALIZED"
        failure = "PROMPT_TAIL_ROUTE_MISSING_OR_ROUTE_AFTER_OUTPUT_FOR_SUBSET"
        next_step = (
            "Replay only the frozen divergent cells, retain the prompt-tail route, "
            "and record explicit forward-to-output indices plus hidden/router/logit "
            "digests; do not time the run."
        )
    elif route_on_drift:
        status = "TELEMETRY_ENABLED_OUTPUT_DIVERGENCE_UNLOCALIZED"
        failure = "ROUTE_TO_OUTPUT_ORDER_NOT_CLOSED"
        next_step = "Capture the first hidden/router/logit divergence before any mechanism Gate."
    else:
        status = "NO_SAME_ARM_REPEAT_DIVERGENCE"
        failure = None
        next_step = "Do not infer causality; obtain a matched same-prestate intervention."

    return {
        "schema": SCHEMA,
        "status": status,
        "failure_category": failure,
        "evidence_tier": "NATIVE_OFFLINE_FIXED_BATCH",
        "claim_ceiling": CLAIM_CEILING,
        "campaign_terminal_sha256": _sha256(terminal),
        "raw_gate_report_sha256": _sha256(raw_report),
        "gate_replay_status": gate_report["status"],
        "gate_replay_failure_category": gate_report["failure_category"],
        "route_off_output_repeat_stable": route_off_stable,
        "route_on_output_repeat_drift": route_on_drift,
        "all_output_drift_requests_have_no_later_captured_route_forward": bool(
            aligned and all(aligned)
        ),
        "output_drift_requests_with_no_later_captured_route_forward": int(
            sum(aligned)
        ),
        "output_drift_request_count": len(aligned),
        "arms": arm_reports,
        "selected_targets": selected_targets,
        "same_prestate": False,
        "timing_used": False,
        "causal_telemetry_attribution_authorized": False,
        "anti_claims": [
            "fresh-process association is not a same-prestate causal intervention",
            "route divergence is not proof that KV cache caused later output drift",
            "the retained route artifact omits the prompt-tail forward that produces output token 0",
            "this diagnostic is not telemetry overhead, capacity, or controller evidence",
        ],
        "one_next_experiment": next_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; localization artifacts are write-once")
    try:
        report = analyze(args.campaign_root)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema": SCHEMA,
            "status": "INVALID_INPUT",
            "failure_category": "CAMPAIGN_OR_ALIGNMENT_INVALID",
            "claim_ceiling": CLAIM_CEILING,
            "error": str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps({key: report.get(key) for key in ("status", "failure_category")}))
    if report["status"] == "INVALID_INPUT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
