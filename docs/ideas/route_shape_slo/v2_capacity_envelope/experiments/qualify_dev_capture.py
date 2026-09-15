#!/usr/bin/env python3
"""Qualify a Route Capacity Envelope development capture for P1 input.

Only the four lightweight checks requested by the v2 protocol are evaluated.
The current producer cannot emit a route-hook-OFF arm, so a real capture stops
at P0_READY_WITH_PROXY_RUNTIME until a paired ON/OFF timing check is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class QualificationError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualificationError(f"{path} must contain an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise QualificationError(f"{path}:{number} must contain an object")
        rows.append(value)
    if not rows:
        raise QualificationError(f"{path} is empty")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualify(
    capture_dir: Path, overhead_path: Path | None = None
) -> dict[str, Any]:
    complete = read_json(capture_dir / "CAPTURE_COMPLETE.json")
    status = read_json(capture_dir / "RUN_STATUS.json")
    workload = read_json(capture_dir / "workload_manifest.json")
    serial_audit = read_json(capture_dir / "serial_audit.json")
    batches = read_jsonl(capture_dir / "decode_batches.jsonl")
    requests = read_jsonl(capture_dir / "request_ledger.jsonl")
    if (
        complete.get("schema") != "bcrd-continuous-capture-complete-v1"
        or complete.get("status") != "CAPTURE_COMPLETE"
        or status.get("status") != "COMPLETE"
        or status.get("required_sentinel") != "CAPTURE_COMPLETE.json"
    ):
        raise QualificationError("capture is not complete")
    expected_files = {
        "routes.csv",
        "decode_batches.jsonl",
        "request_ledger.jsonl",
        "workload_manifest.json",
        "preregistration.json",
        "environment.json",
        "serial_audit.json",
    }
    file_hashes = complete.get("files")
    if not isinstance(file_hashes, Mapping) or set(file_hashes) != expected_files:
        raise QualificationError("capture sentinel has an invalid file manifest")
    for name in sorted(expected_files):
        path = capture_dir / name
        if not path.is_file() or sha256_file(path) != str(file_hashes[name]):
            raise QualificationError(f"capture file hash mismatch: {name}")
    if complete.get("serial_audit") != serial_audit:
        raise QualificationError("serial audit differs from the final sentinel")
    audit_status = str(serial_audit.get("status", ""))
    try:
        route_match = float(serial_audit.get("route_identity_match_fraction"))
        token_match = float(serial_audit.get("token_match_fraction"))
    except (TypeError, ValueError) as exc:
        raise QualificationError("serial audit fractions are missing or invalid") from exc
    batch_dependent_value = serial_audit.get(
        "batch_dependent_route_observed", False
    )
    if not isinstance(batch_dependent_value, bool):
        raise QualificationError("batch-dependent route marker must be boolean")
    batch_dependent_route = batch_dependent_value
    common_audit_ok = (
        math.isfinite(route_match)
        and 0.0 <= route_match <= 1.0
        and token_match == 1.0
        and int(serial_audit.get("requests", 0)) > 0
        and int(serial_audit.get("steps", -1)) >= 0
        and serial_audit.get("scientific_ground_truth") is False
    )
    exact_equivalence = (
        audit_status == "PASS"
        and route_match == 1.0
        and not batch_dependent_route
        and serial_audit.get("reference_type")
        in {
            "same-model serial cached-decode engineering equivalence",
            "same-model serial cached-decode conformance diagnostic",
        }
    )
    exposed_batch_dependence = (
        audit_status == "PASS_TOKEN_PARITY_ROUTE_BATCH_DEPENDENT"
        and batch_dependent_route
        and route_match < 1.0
        and int(serial_audit.get("steps", 0)) > 0
        and int(serial_audit.get("layers", 0)) > 0
        and serial_audit.get("route_identity_semantics")
        == "per_layer_expert_assignment_multiset"
        and serial_audit.get("topk_order_checked") is True
        and serial_audit.get("reference_type")
        == "same-model serial cached-decode conformance diagnostic"
    )
    if not common_audit_ok or not (exact_equivalence or exposed_batch_dependence):
        raise QualificationError(
            "producer serial route/token engineering audit is missing or failed"
        )

    request_index = {str(row["request_id"]): row for row in requests}
    if len(request_index) != len(requests):
        raise QualificationError("duplicate request identity")
    route_events: dict[tuple[str, int], set[tuple[int, int]]] = {}
    route_windows: dict[tuple[str, int], tuple[float, float]] = {}
    with (capture_dir / "routes.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "request_id", "decode_step", "layer_id", "topk_slot", "expert_id",
            "gate_weight", "layer_ready_us", "route_end_us", "document_id",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise QualificationError(f"routes.csv lacks {sorted(missing)}")
        for row in reader:
            request_id = str(row["request_id"])
            step = int(row["decode_step"])
            if request_id not in request_index:
                raise QualificationError("route references an unknown request")
            if str(row["document_id"]) != str(request_index[request_id]["document_id"]):
                raise QualificationError("route/request document identity is misaligned")
            weight = float(row["gate_weight"])
            if not math.isfinite(weight) or not 0 <= weight <= 1:
                raise QualificationError("invalid gate weight")
            key = (request_id, step)
            slot = (int(row["layer_id"]), int(row["topk_slot"]))
            if slot in route_events.setdefault(key, set()):
                raise QualificationError("duplicate layer/top-k route identity")
            route_events[key].add(slot)
            bounds = (float(row["layer_ready_us"]), float(row["route_end_us"]))
            if key in route_windows and route_windows[key] != bounds:
                raise QualificationError("one route event has inconsistent timing")
            route_windows[key] = bounds

    seen_events: set[tuple[str, int]] = set()
    prior_end = -math.inf
    for expected_index, batch in enumerate(sorted(batches, key=lambda row: int(row["batch_index"]))):
        if int(batch["batch_index"]) != expected_index:
            raise QualificationError("batch indices are not contiguous")
        start, end = float(batch["start_us"]), float(batch["end_us"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start or start < prior_end:
            raise QualificationError("batch timing is invalid or overlaps")
        prior_end = end
        request_ids = [str(value) for value in batch["request_ids"]]
        active_ids = [str(value) for value in batch["active_request_ids"]]
        steps = [int(value) for value in batch["decode_steps"]]
        lengths = [int(value) for value in batch["prior_cache_lengths"]]
        if not (
            len(request_ids) == len(steps) == len(lengths) == int(batch["batch_size"])
            and len(set(request_ids)) == len(request_ids)
            and set(request_ids) <= set(active_ids)
        ):
            raise QualificationError("batch/request/KV identity is misaligned")
        for request_id, step in zip(request_ids, steps):
            key = (request_id, step)
            if key not in route_events or key in seen_events:
                raise QualificationError("route event is missing or reused")
            seen_events.add(key)
            if route_windows[key] != (start, end):
                raise QualificationError("route and model-call boundaries are misaligned")
    if seen_events != set(route_events):
        raise QualificationError("orphan route events remain")

    overhead = read_json(overhead_path) if overhead_path is not None else None
    fixed_batch_proxy_pass = bool(
        overhead
        and overhead.get("schema")
        == "route-capacity-envelope-telemetry-overhead-v1"
        and overhead.get("status") == "TELEMETRY_OVERHEAD_OK"
        and overhead.get("token_output_match") is True
        and overhead.get("logit_output_match") is True
        and overhead.get("on_route_trace_stable") is True
        and overhead.get("completion_trace_match") is True
        and overhead.get("same_requests") is True
        and overhead.get("same_batch_schedule") is True
        and overhead.get("same_decode_steps") is True
        and overhead.get("same_dtype") is True
        and overhead.get("same_arrival_trace") is False
        and overhead.get("arrival_policy_applied_to_timing") is False
    )
    workload_marker = workload.get("route_capacity_envelope", {})
    return {
        "schema": "route-capacity-envelope-p0-qualification-v1",
        "capture_dir": str(capture_dir),
        "episode_id": workload_marker.get("episode_id", "unknown"),
        "arrival_regime": workload_marker.get("arrival_regime", "unknown"),
        "checks": {
            "causal_future_information": "PASS_BY_RAW_CAPTURE_SCHEMA_NO_SHIFTED_TARGETS",
            "route_request_step_latency_alignment": "PASS",
            "serial_token_parity": "PASS",
            "serial_route_conformance": (
                "BATCH_DEPENDENT"
                if batch_dependent_route
                else "PASS_EXPERT_ASSIGNMENT_MULTISET"
            ),
            "hook_no_hook_distortion": (
                "PASS_FIXED_BATCH_PROXY"
                if fixed_batch_proxy_pass
                else "NOT_MEASURED"
            ),
            "same_data_same_action_space": "PASS_P0_NO_ACTION_ALL_METHODS_SHARE_WINDOWS",
        },
        "status": "P0_READY_WITH_PROXY_RUNTIME",
        "p1_runtime_ready": False,
        "exploratory_p1_ready": True,
        "diagnostic_only": batch_dependent_route,
        "representative_serving_p1_ready": False,
        "telemetry_arrival_replay_measured": False,
        "request_count": len(requests),
        "decode_windows": len(batches),
        "route_events": len(route_events),
        "native_serving_runtime": False,
        "native_queue": False,
        "custom_waiting_count_semantics": "len(active_request_ids)-batch_size",
        "custom_running_set_semantics": "scheduled request_ids",
        "pending_request_count_consumed_as_queue": False,
        "serial_route_identity_match_fraction": route_match,
        "batch_dependent_route_observed": batch_dependent_route,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--overhead")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = qualify(
        Path(args.capture_dir).resolve(),
        Path(args.overhead).resolve() if args.overhead else None,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
