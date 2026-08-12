#!/usr/bin/env python3
"""Fail-closed, read-only inventory for RouteShape-SLO source artifacts.

The inspector understands two evidence layouts:

* StableBatch's sealed single-GPU frozen-roster cost run.  The only accepted
  replay slice is ``arm=native_variable_m, phase=measured, repeat=0``.
* BCRD's canonical continuous-decode producer bundle
  (``routes.csv`` + ``decode_batches.jsonl`` + ``request_ledger.jsonl``).

Inspection never changes a source run.  It deliberately reports evidence
boundaries such as missing gate weights and ``runtime_representative=false``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


STABLE_ARM = "native_variable_m"
STABLE_PHASE = "measured"
STABLE_REPEAT = 0

STABLE_MANIFEST_SCHEMA = "stablebatch-shape-lane-cost-manifest-v1"
STABLE_STATUS_SCHEMA = "stablebatch-shape-lane-cost-run-status-v1"
STABLE_COMPLETE_SCHEMA = "stablebatch-shape-lane-cost-complete-v1"
STABLE_CONFIG_SCHEMA = "stablebatch-shape-lane-continuous-cost-gate-v1"
STABLE_SUMMARY_SCHEMA = "stablebatch-shape-lane-continuous-cost-summary-v1"
STABLE_ROSTER_SCHEMA = "stablebatch-shape-lane-native-roster-row-v1"
STABLE_STAGE_SCHEMA = "stablebatch-shape-lane-expert-stage-v1"
STABLE_STEP_SCHEMA = "stablebatch-shape-lane-decode-step-v1"
STABLE_CALL_SCHEMA = "stablebatch-shape-lane-expert-call-v1"

BCRD_COMPLETE_SCHEMA = "bcrd-continuous-capture-complete-v1"

STABLE_REQUIRED = (
    "RUN_STATUS.json",
    "COMPLETE.json",
    "MANIFEST.json",
    "config_snapshot.json",
    "summary.json",
    "workload_snapshot.json",
    "native_roster.jsonl",
    "expert_stage_ledger.jsonl",
    "decode_step_ledger.jsonl",
    "expert_call_ledger.jsonl",
)

BCRD_REQUIRED = (
    "RUN_STATUS.json",
    "CAPTURE_COMPLETE.json",
    "routes.csv",
    "decode_batches.jsonl",
    "request_ledger.jsonl",
    "workload_manifest.json",
)


class AssetError(RuntimeError):
    """An input/schema/integrity failure that invalidates the adapter."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssetError(message)


def mapping(value: Any, where: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), "%s must be a JSON object" % where)
    return value


def integer(value: Any, where: str) -> int:
    require(not isinstance(value, bool) and isinstance(value, int), "%s must be an integer" % where)
    return int(value)


def finite(value: Any, where: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), "%s must be numeric" % where)
    result = float(value)
    require(math.isfinite(result), "%s must be finite" % where)
    return result


def safe_file(path: Path, where: str) -> Path:
    require(path.exists(), "%s is missing: %s" % (where, path))
    require(path.is_file(), "%s is not a regular file: %s" % (where, path))
    require(not path.is_symlink(), "%s may not be a symlink: %s" % (where, path))
    return path


def safe_run_dir(path: Path) -> Path:
    require(path.exists(), "run directory is missing: %s" % path)
    require(path.is_dir(), "run path is not a directory: %s" % path)
    require(not path.is_symlink(), "run directory may not be a symlink: %s" % path)
    return path.resolve()


def load_json(path: Path) -> Mapping[str, Any]:
    safe_file(path, "JSON input")
    try:
        with path.open("r", encoding="utf-8") as stream:
            return mapping(json.load(stream), path.name)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssetError("cannot read %s: %s" % (path, exc)) from exc


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Mapping[str, Any]]]:
    safe_file(path, "JSONL input")
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                require(line.endswith("\n"), "%s:%d is not newline terminated" % (path.name, line_number))
                require(bool(line.strip()), "%s:%d is blank" % (path.name, line_number))
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AssetError("invalid JSON at %s:%d: %s" % (path.name, line_number, exc)) from exc
                yield line_number, mapping(value, "%s:%d" % (path.name, line_number))
    except (OSError, UnicodeError) as exc:
        raise AssetError("cannot stream %s: %s" % (path, exc)) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AssetError("cannot hash %s: %s" % (path, exc)) from exc
    return digest.hexdigest()


def stable_identity(row: Mapping[str, Any], where: str) -> Tuple[str, str, int]:
    arm = str(row.get("arm", ""))
    phase = str(row.get("phase", ""))
    repeat = integer(row.get("repeat"), "%s.repeat" % where)
    return arm, phase, repeat


def is_stable_target(row: Mapping[str, Any], where: str) -> bool:
    return stable_identity(row, where) == (STABLE_ARM, STABLE_PHASE, STABLE_REPEAT)


def verify_manifest_file(
    run_dir: Path,
    manifest: Mapping[str, Any],
    name: str,
    verify_sha256: bool,
) -> Dict[str, Any]:
    files = mapping(manifest.get("files"), "MANIFEST.files")
    entry = mapping(files.get(name), "MANIFEST.files.%s" % name)
    path = safe_file(run_dir / name, "manifest artifact")
    expected_size = integer(entry.get("size_bytes"), "MANIFEST.files.%s.size_bytes" % name)
    observed_size = path.stat().st_size
    require(observed_size == expected_size, "%s size differs from MANIFEST" % name)
    expected_sha = str(entry.get("sha256", ""))
    require(len(expected_sha) == 64, "%s has no valid manifest SHA-256" % name)
    observed_sha: Optional[str] = None
    if verify_sha256:
        observed_sha = sha256_file(path)
        require(observed_sha == expected_sha, "%s SHA-256 differs from MANIFEST" % name)
    return {
        "name": name,
        "size_bytes": observed_size,
        "manifest_sha256": expected_sha,
        "sha256_recomputed": verify_sha256,
        "observed_sha256": observed_sha,
    }


def inspect_stablebatch(run_dir: Path, verify_large_ledger_sha256: bool = False) -> Dict[str, Any]:
    run_dir = safe_run_dir(run_dir)
    for name in STABLE_REQUIRED:
        safe_file(run_dir / name, "StableBatch required artifact")

    status = load_json(run_dir / "RUN_STATUS.json")
    complete = load_json(run_dir / "COMPLETE.json")
    manifest = load_json(run_dir / "MANIFEST.json")
    config = load_json(run_dir / "config_snapshot.json")
    summary = load_json(run_dir / "summary.json")
    workload = load_json(run_dir / "workload_snapshot.json")

    require(status.get("schema_version") == STABLE_STATUS_SCHEMA, "StableBatch RUN_STATUS schema drifted")
    require(complete.get("schema_version") == STABLE_COMPLETE_SCHEMA, "StableBatch COMPLETE schema drifted")
    require(manifest.get("schema_version") == STABLE_MANIFEST_SCHEMA, "StableBatch MANIFEST schema drifted")
    require(config.get("schema_version") == STABLE_CONFIG_SCHEMA, "StableBatch config schema drifted")
    require(summary.get("schema_version") == STABLE_SUMMARY_SCHEMA, "StableBatch summary schema drifted")
    require(status.get("status") == "COMPLETE", "StableBatch RUN_STATUS is not COMPLETE")
    require(complete.get("status") == "COMPLETE" and complete.get("completion_last") is True,
            "StableBatch completion sentinel is not final/complete")
    require(summary.get("status") == "COMPLETE", "StableBatch summary is not COMPLETE")
    require(status.get("serving_result") is False, "StableBatch status unexpectedly claims serving")
    require(complete.get("serving_result") is False, "StableBatch sentinel unexpectedly claims serving")
    require(summary.get("serving_result") is False, "StableBatch summary unexpectedly claims serving")
    require(summary.get("teacher_forced_frozen_roster") is True,
            "StableBatch run is not the frozen teacher-forced roster")
    require(summary.get("formal_bcrd_producer") is False,
            "StableBatch adapter refuses a run claiming to be a formal BCRD producer")

    require(sha256_file(run_dir / "MANIFEST.json") == str(complete.get("manifest_sha256")),
            "COMPLETE manifest hash does not match MANIFEST.json")
    require(sha256_file(run_dir / "RUN_STATUS.json") == str(complete.get("run_status_sha256")),
            "COMPLETE run-status hash does not match RUN_STATUS.json")
    require(sha256_file(run_dir / "summary.json") == str(complete.get("summary_sha256")),
            "COMPLETE summary hash does not match summary.json")

    execution = mapping(config.get("execution"), "config.execution")
    model = mapping(config.get("model"), "config.model")
    config_workload = mapping(config.get("workload"), "config.workload")
    arms = tuple(str(value) for value in execution.get("arms", []))
    require(STABLE_ARM in arms, "StableBatch config has no native_variable_m arm")
    measured_orders = execution.get("measured_arm_orders")
    require(isinstance(measured_orders, list) and bool(measured_orders),
            "StableBatch config has no measured arm order")
    first_order = tuple(str(value) for value in measured_orders[0])
    require(first_order and first_order[0] == STABLE_ARM,
            "canonical adapter requires repeat-0 measured native arm to be first")

    expected_requests = integer(config_workload.get("expected_requests"), "config.workload.expected_requests")
    expected_request_steps = integer(
        config_workload.get("expected_request_steps"), "config.workload.expected_request_steps"
    )
    expected_decode_steps = integer(
        config_workload.get("expected_decode_steps_per_request"),
        "config.workload.expected_decode_steps_per_request",
    )
    max_batch_size = integer(config_workload.get("max_batch_size"), "config.workload.max_batch_size")
    num_layers = integer(model.get("num_hidden_layers"), "config.model.num_hidden_layers")
    num_experts = integer(model.get("num_experts"), "config.model.num_experts")
    top_k = integer(model.get("num_experts_per_tok"), "config.model.num_experts_per_tok")
    require(integer(workload.get("expected_requests"), "workload.expected_requests") == expected_requests,
            "workload request count differs from config")
    workload_model = mapping(workload.get("model"), "workload.model")
    require(str(workload_model.get("id")) == str(model.get("repo_id")), "workload model id drifted")
    require(str(workload_model.get("revision")) == str(model.get("revision")), "workload model revision drifted")
    workload_requests = workload.get("requests")
    require(isinstance(workload_requests, list) and len(workload_requests) == expected_requests,
            "workload request rows are incomplete")
    workload_ids = [str(mapping(row, "workload request").get("request_id")) for row in workload_requests]
    require(len(set(workload_ids)) == expected_requests, "workload request ids are not unique")

    roster_count = 0
    seen_request_steps: set = set()
    roster_batches: Dict[int, Dict[str, Any]] = {}
    for line_number, row in iter_jsonl(run_dir / "native_roster.jsonl"):
        where = "native_roster.jsonl:%d" % line_number
        require(row.get("schema_version") == STABLE_ROSTER_SCHEMA, "%s schema drifted" % where)
        batch_index = integer(row.get("batch_index"), "%s.batch_index" % where)
        require(batch_index == roster_count, "native roster batch indexes are not contiguous")
        request_ids = [str(value) for value in row.get("request_ids", [])]
        decode_steps = [integer(value, "%s.decode_steps" % where) for value in row.get("decode_steps", [])]
        prior_lengths = [integer(value, "%s.prior_cache_lengths" % where)
                         for value in row.get("prior_cache_lengths", [])]
        require(0 < len(request_ids) <= max_batch_size, "%s batch width is invalid" % where)
        require(len(request_ids) == len(decode_steps) == len(prior_lengths),
                "%s roster columns have different widths" % where)
        require(integer(row.get("batch_size"), "%s.batch_size" % where) == len(request_ids),
                "%s batch_size differs from request_ids" % where)
        require(set(request_ids).issubset(set(str(value) for value in row.get("active_request_ids", []))),
                "%s selected requests are outside active_request_ids" % where)
        for request_id, step in zip(request_ids, decode_steps):
            pair = (request_id, step)
            require(pair not in seen_request_steps, "duplicate roster request/decode pair: %r" % (pair,))
            seen_request_steps.add(pair)
        roster_batches[batch_index] = {
            "request_ids": request_ids,
            "decode_steps": decode_steps,
            "route_sha256": str(row.get("native_route_membership_sha256", "")),
            "logits_sha256": str(row.get("native_final_logits_sha256", "")),
        }
        roster_count += 1
    require(len(seen_request_steps) == expected_request_steps,
            "native roster does not close expected request/decode steps")
    require(len({request_id for request_id, _ in seen_request_steps}) == expected_requests,
            "native roster does not cover every workload request")

    target_steps: Dict[int, Mapping[str, Any]] = {}
    step_total = 0
    step_identities: set = set()
    for line_number, row in iter_jsonl(run_dir / "decode_step_ledger.jsonl"):
        where = "decode_step_ledger.jsonl:%d" % line_number
        require(row.get("schema_version") == STABLE_STEP_SCHEMA, "%s schema drifted" % where)
        identity = stable_identity(row, where)
        require(identity[0] in arms and identity[1] in ("warmup", "measured") and identity[2] >= 0,
                "%s has an unsupported replay identity" % where)
        step_identities.add(identity)
        step_total += 1
        if identity != (STABLE_ARM, STABLE_PHASE, STABLE_REPEAT):
            continue
        batch_index = integer(row.get("batch_index"), "%s.batch_index" % where)
        require(batch_index not in target_steps, "duplicate selected decode batch %d" % batch_index)
        target_steps[batch_index] = row
    require(set(target_steps) == set(roster_batches),
            "selected decode_step ledger does not align one-to-one with native roster")

    target_stage_keys: set = set()
    stage_total = 0
    stage_identities: set = set()
    for line_number, row in iter_jsonl(run_dir / "expert_stage_ledger.jsonl"):
        where = "expert_stage_ledger.jsonl:%d" % line_number
        require(row.get("schema_version") == STABLE_STAGE_SCHEMA, "%s schema drifted" % where)
        identity = stable_identity(row, where)
        require(identity[0] in arms and identity[1] in ("warmup", "measured") and identity[2] >= 0,
                "%s has an unsupported replay identity" % where)
        stage_identities.add(identity)
        stage_total += 1
        if identity != (STABLE_ARM, STABLE_PHASE, STABLE_REPEAT):
            continue
        batch_index = integer(row.get("batch_index"), "%s.batch_index" % where)
        layer = integer(row.get("layer"), "%s.layer" % where)
        require(0 <= layer < num_layers, "%s layer is out of range" % where)
        require(0 <= batch_index < roster_count, "%s batch is out of range" % where)
        require(integer(row.get("occupied_experts"), "%s.occupied_experts" % where) > 0,
                "%s has no occupied experts" % where)
        require(finite(row.get("expert_stage_gpu_ms"), "%s.expert_stage_gpu_ms" % where) > 0.0,
                "%s expert-stage latency must be positive" % where)
        key = (batch_index, layer)
        require(key not in target_stage_keys, "duplicate selected expert-stage row: %r" % (key,))
        target_stage_keys.add(key)
    expected_stage_keys = {(batch, layer) for batch in range(roster_count) for layer in range(num_layers)}
    require(target_stage_keys == expected_stage_keys,
            "selected expert_stage ledger does not cover every batch/layer")

    for batch_index, step in target_steps.items():
        roster = roster_batches[batch_index]
        require([str(value) for value in step.get("request_ids", [])] == roster["request_ids"],
                "decode-step/request roster drift at batch %d" % batch_index)
        require([integer(value, "decode step") for value in step.get("decode_steps", [])]
                == roster["decode_steps"], "decode-step index drift at batch %d" % batch_index)
        require(str(step.get("native_roster_route_membership_sha256")) == roster["route_sha256"],
                "decode-step/native route hash drift at batch %d" % batch_index)
        require(str(step.get("native_roster_final_logits_sha256")) == roster["logits_sha256"],
                "decode-step/native logits hash drift at batch %d" % batch_index)

    manifest_checks: List[Dict[str, Any]] = []
    for name in (
        "config_snapshot.json",
        "summary.json",
        "workload_snapshot.json",
        "native_roster.jsonl",
        "expert_stage_ledger.jsonl",
        "decode_step_ledger.jsonl",
        "expert_call_ledger.jsonl",
    ):
        verify_hash = name != "expert_call_ledger.jsonl" or verify_large_ledger_sha256
        manifest_checks.append(verify_manifest_file(run_dir, manifest, name, verify_hash))

    return {
        "adapter": "stablebatch_shape_lane_cost_v1",
        "status": "READY_FOR_NON_SERVING_ROUTE_WINDOW_EXTRACTION",
        "run_dir": str(run_dir),
        "selected_slice": {
            "arm": STABLE_ARM,
            "phase": STABLE_PHASE,
            "repeat": STABLE_REPEAT,
            "decode_batches": roster_count,
            "request_steps": len(seen_request_steps),
            "layers": num_layers,
            "experts": num_experts,
            "top_k": top_k,
        },
        "source_replay_identities": {
            "decode_step": [list(value) for value in sorted(step_identities)],
            "expert_stage": [list(value) for value in sorted(stage_identities)],
            "decode_step_rows_total": step_total,
            "expert_stage_rows_total": stage_total,
        },
        "manifest_checks": manifest_checks,
        "large_expert_call_ledger": {
            "sha256_recomputed": verify_large_ledger_sha256,
            "selection_strategy": (
                "target contiguous byte range only; expert_stage ledger remains the primary timing source"
            ),
        },
        "gate_weight": {
            "available": False,
            "reason": "StableBatch call/stage/step ledgers do not record router gate weights",
        },
        "runtime_representative": False,
        "evidence_type": "[Observed isolated GPU primitive]",
        "evidence_boundary": (
            "single-GPU teacher-forced frozen-roster mechanism-cost artifact; "
            "not continuous serving, admission control, vLLM, EP, or capacity evidence"
        ),
        "verdict_inherited_not_reinterpreted": str(summary.get("verdict")),
    }


def inspect_bcrd(run_dir: Path) -> Dict[str, Any]:
    run_dir = safe_run_dir(run_dir)
    for name in BCRD_REQUIRED:
        safe_file(run_dir / name, "BCRD required artifact")
    status = load_json(run_dir / "RUN_STATUS.json")
    complete = load_json(run_dir / "CAPTURE_COMPLETE.json")
    workload = load_json(run_dir / "workload_manifest.json")
    require(status.get("status") == "COMPLETE", "BCRD RUN_STATUS is not COMPLETE")
    require(status.get("required_sentinel") == "CAPTURE_COMPLETE.json",
            "BCRD RUN_STATUS does not name the completion sentinel")
    require(complete.get("schema") == BCRD_COMPLETE_SCHEMA, "BCRD completion schema drifted")
    require(complete.get("status") == "CAPTURE_COMPLETE", "BCRD completion sentinel is not complete")
    require(complete.get("producer_formal_eligible") is False,
            "BCRD adapter refuses an unexpected formal-eligibility claim")
    require(complete.get("scientific_result_eligible") is False,
            "BCRD adapter refuses an unexpected scientific-eligibility claim")
    require(complete.get("gate0_complete") is False and complete.get("gate1_authorized") is False,
            "BCRD producer bundle unexpectedly claims downstream authorization")
    files = mapping(complete.get("files"), "CAPTURE_COMPLETE.files")
    for name in ("routes.csv", "decode_batches.jsonl", "request_ledger.jsonl", "workload_manifest.json"):
        expected = str(files.get(name, ""))
        require(len(expected) == 64, "BCRD sentinel has no hash for %s" % name)
        require(sha256_file(run_dir / name) == expected, "BCRD %s hash differs from sentinel" % name)

    batch_count = 0
    request_step_pairs: set = set()
    for line_number, row in iter_jsonl(run_dir / "decode_batches.jsonl"):
        where = "decode_batches.jsonl:%d" % line_number
        require(integer(row.get("batch_index"), "%s.batch_index" % where) == batch_count,
                "BCRD decode batch indexes are not contiguous")
        request_ids = [str(value) for value in row.get("request_ids", [])]
        decode_steps = [integer(value, "%s.decode_steps" % where) for value in row.get("decode_steps", [])]
        require(request_ids and len(request_ids) == len(decode_steps), "%s columns do not align" % where)
        for pair in zip(request_ids, decode_steps):
            require(pair not in request_step_pairs, "duplicate BCRD request/decode pair: %r" % (pair,))
            request_step_pairs.add(pair)
        batch_count += 1
    require(batch_count > 0, "BCRD decode_batches ledger is empty")

    request_count = 0
    ledger_pairs: set = set()
    for line_number, row in iter_jsonl(run_dir / "request_ledger.jsonl"):
        where = "request_ledger.jsonl:%d" % line_number
        request_id = str(row.get("request_id", ""))
        require(bool(request_id), "%s has no request_id" % where)
        require(row.get("completion_us") is not None and bool(row.get("stop_reason")),
                "%s is not terminal" % where)
        steps = row.get("steps")
        require(isinstance(steps, list), "%s.steps must be a list" % where)
        for step in steps:
            item = mapping(step, "%s step" % where)
            pair = (request_id, integer(item.get("decode_step"), "%s.decode_step" % where))
            require(pair not in ledger_pairs, "duplicate request-ledger pair: %r" % (pair,))
            ledger_pairs.add(pair)
        request_count += 1
    require(ledger_pairs == request_step_pairs, "BCRD request and decode-batch ledgers do not align")

    try:
        with (run_dir / "routes.csv").open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required_columns = {
                "model", "phase", "request_id", "decode_step", "layer_id",
                "topk_slot", "expert_id", "gate_weight", "input_event_id",
            }
            require(reader.fieldnames is not None and required_columns.issubset(set(reader.fieldnames)),
                    "BCRD routes.csv header is missing canonical route columns")
            route_rows = 0
            route_pairs: set = set()
            gate_weight_missing = 0
            for row_number, row in enumerate(reader, 2):
                require(row.get("phase") == "decode", "routes.csv:%d is not decode" % row_number)
                request_id = str(row.get("request_id", ""))
                step = int(str(row.get("decode_step")))
                require((request_id, step) in request_step_pairs,
                        "routes.csv:%d has no matching decode batch" % row_number)
                layer = int(str(row.get("layer_id")))
                slot = int(str(row.get("topk_slot")))
                expert = int(str(row.get("expert_id")))
                require(layer >= 0 and slot >= 0 and expert >= 0,
                        "routes.csv:%d has negative route identity" % row_number)
                weight_text = str(row.get("gate_weight", ""))
                if not weight_text:
                    gate_weight_missing += 1
                else:
                    require(math.isfinite(float(weight_text)),
                            "routes.csv:%d gate_weight is not finite" % row_number)
                route_pairs.add((request_id, step))
                route_rows += 1
    except (OSError, UnicodeError, ValueError) as exc:
        raise AssetError("cannot validate BCRD routes.csv: %s" % exc) from exc
    require(route_rows > 0 and route_pairs == request_step_pairs,
            "BCRD routes do not cover every request/decode pair")

    return {
        "adapter": "bcrd_continuous_capture_v1",
        "status": "READY_FOR_PRODUCER_CANDIDATE_ROUTE_WINDOW_EXTRACTION",
        "run_dir": str(run_dir),
        "run_class": str(complete.get("run_class")),
        "decode_batches": batch_count,
        "requests": request_count,
        "request_steps": len(request_step_pairs),
        "route_rows": route_rows,
        "gate_weight": {
            "available": gate_weight_missing == 0,
            "missing_rows": gate_weight_missing,
        },
        "runtime_representative": False,
        "evidence_type": "[Observed real runtime]",
        "evidence_boundary": (
            "BCRD continuous-decode producer candidate; model-call timing and route identity only; "
            "not independently qualified serving, stage, SLO-capacity, or EP evidence"
        ),
        "formal_blockers": list(complete.get("formal_blockers_outside_gate0_a", [])),
        "workload_model": dict(mapping(workload.get("model"), "workload.model")),
    }


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def discover_stablebatch(repo_root: Path) -> List[Path]:
    root = repo_root / "docs" / "ideas" / "stablebatch" / "experiments" / "outputs"
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.glob("shape_lane_continuous_cost_*")
        if path.is_dir() and (path / "expert_call_ledger.jsonl").is_file()
    )


def discover_bcrd(repo_root: Path) -> List[Path]:
    root = repo_root / "artifacts" / "bcrd_gate0" / "formal"
    if not root.is_dir():
        return []
    return sorted(
        path.parent for path in root.glob("*/*/CAPTURE_COMPLETE.json")
        if (path.parent / "routes.csv").is_file()
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--stablebatch-run", type=Path, action="append", default=[])
    parser.add_argument("--bcrd-run", type=Path, action="append", default=[])
    parser.add_argument(
        "--verify-large-ledger-sha256",
        action="store_true",
        help="also hash the 1.2GB StableBatch expert-call ledger",
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument("--output", type=Path, help="also write the JSON inventory")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    stable_runs = list(args.stablebatch_run) or discover_stablebatch(repo_root)
    bcrd_runs = list(args.bcrd_run) or discover_bcrd(repo_root)
    reports: List[Dict[str, Any]] = []
    invalid = False
    for adapter, paths in (("stablebatch", stable_runs), ("bcrd", bcrd_runs)):
        for path in paths:
            try:
                report = (
                    inspect_stablebatch(path, args.verify_large_ledger_sha256)
                    if adapter == "stablebatch"
                    else inspect_bcrd(path)
                )
            except AssetError as exc:
                invalid = True
                report = {
                    "adapter": adapter,
                    "status": "INVALID_FAIL_CLOSED",
                    "run_dir": str(path),
                    "error": str(exc),
                }
            reports.append(report)
    payload = {
        "schema_version": "route-shape-slo-existing-assets-v1",
        "repo_root": str(repo_root),
        "status": "INVALID_FAIL_CLOSED" if invalid else "INSPECTION_COMPLETE",
        "assets": reports,
        "discovery": {
            "stablebatch_runs": len(stable_runs),
            "bcrd_canonical_runs": len(bcrd_runs),
            "bcrd_adapter_supported_but_no_workspace_bundle": not bool(bcrd_runs),
        },
        "claim_boundary": (
            "Asset availability and schema closure only. No source is upgraded to serving, "
            "capacity, controller, multi-GPU, or formal GO evidence."
        ),
    }
    rendered = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if args.compact
        else json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 2 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
