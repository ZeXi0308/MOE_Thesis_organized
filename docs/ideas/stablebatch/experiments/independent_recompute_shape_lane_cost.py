#!/usr/bin/env python3
"""Read-only, independent recomputation for the frozen D10 shape-lane gate.

This module intentionally does not import the producer, its classifier, or any
model/runtime helper.  It consumes the sealed JSON/JSONL evidence, reconstructs
all quantities supported by those ledgers, prints one JSON report, and never
writes inside the run directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterator, Mapping, Sequence


ARM_NATIVE = "native_variable_m"
ARM_SERIAL = "serial_m1"
ARM_C8 = "fixed_c8"
ARMS = (ARM_NATIVE, ARM_SERIAL, ARM_C8)
CANONICAL_M = 8

CONFIG_SCHEMA = "stablebatch-shape-lane-continuous-cost-gate-v1"
SUMMARY_SCHEMA = "stablebatch-shape-lane-continuous-cost-summary-v1"
STATUS_SCHEMA = "stablebatch-shape-lane-cost-run-status-v1"
COMPLETE_SCHEMA = "stablebatch-shape-lane-cost-complete-v1"
MANIFEST_SCHEMA = "stablebatch-shape-lane-cost-manifest-v1"
ROSTER_SCHEMA = "stablebatch-shape-lane-native-roster-row-v1"
CALL_SCHEMA = "stablebatch-shape-lane-expert-call-v1"
STAGE_SCHEMA = "stablebatch-shape-lane-expert-stage-v1"
STEP_SCHEMA = "stablebatch-shape-lane-decode-step-v1"

INVALID_VERDICT = "INVALID_D10_CONTINUOUS_COST_GATE"
FROZEN_GATE = {
    "require_fixed_c8_raw_repeat_mismatches": 0,
    "require_fixed_c8_route_repeat_mismatches": 0,
    "require_fixed_c8_final_logits_repeat_mismatches": 0,
    "maximum_c8_over_serial_expert_gpu_time_ratio": 0.8,
    "maximum_c8_over_native_token_step_p99_ratio": 1.05,
    "pass": "PROVISIONAL_SUPPORT_VS_SERIAL / EXTERNAL_BI_OPEN",
    "correctness_fail": "NO_GO_D10_C8_CURRENT_STACK",
    "cost_fail": "NO_GO_D10_HEADLINE_COST",
    "invalid": INVALID_VERDICT,
}

PRODUCER_REQUIRED_ARTIFACTS = (
    "run_request.json",
    "environment.json",
    "static_bindings.json",
    "config_snapshot.json",
    "workload_snapshot.json",
    "native_prefill_ledger.jsonl",
    "native_roster.jsonl",
    "expert_call_ledger.jsonl",
    "expert_stage_ledger.jsonl",
    "decode_step_ledger.jsonl",
    "replay_ledger.jsonl",
    "runtime_final.json",
    "summary.json",
)

AUDIT_INPUTS = (
    "RUN_STATUS.json",
    "COMPLETE.json",
    "MANIFEST.json",
    "config_snapshot.json",
    "summary.json",
    "replay_ledger.jsonl",
    "decode_step_ledger.jsonl",
    "expert_stage_ledger.jsonl",
    "expert_call_ledger.jsonl",
    "native_roster.jsonl",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SLOT = re.compile(
    r"^(?P<request>.*):decode:(?P<step>[0-9]{6}):"
    r"layer:(?P<layer>[0-9]{2}):topk:(?P<rank>[0-9]+)$"
)


class RecomputeError(RuntimeError):
    """An integrity/protocol failure that invalidates independent recomputation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecomputeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{where} must be a JSON object")
    return value


def load_json_object(path: Path) -> Mapping[str, Any]:
    _require(path.is_file(), f"required input is missing: {path.name}")
    _require(not path.is_symlink(), f"required input may not be a symlink: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            return _mapping(json.load(stream), path.name)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecomputeError(f"cannot read {path.name}: {exc}") from exc


def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    _require(path.is_file(), f"required input is missing: {path.name}")
    _require(not path.is_symlink(), f"required input may not be a symlink: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                _require(bool(line.strip()), f"{path.name}:{line_number} is blank")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RecomputeError(
                        f"{path.name}:{line_number} is invalid JSON: {exc}"
                    ) from exc
                yield _mapping(row, f"{path.name}:{line_number}")
    except (OSError, UnicodeError) as exc:
        raise RecomputeError(f"cannot read {path.name}: {exc}") from exc


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return list(iter_jsonl(path))


def _integer(value: Any, where: str) -> int:
    _require(not isinstance(value, bool), f"{where} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecomputeError(f"{where} must be an integer") from exc
    _require(converted == value, f"{where} must be an exact integer")
    return converted


def _finite(value: Any, where: str, *, positive: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecomputeError(f"{where} must be numeric") from exc
    _require(math.isfinite(converted), f"{where} must be finite")
    if positive:
        _require(converted > 0.0, f"{where} must be positive")
    return converted


def _digest(value: Any, where: str) -> str:
    converted = str(value)
    _require(bool(_HEX64.fullmatch(converted)), f"{where} is not a lowercase SHA-256")
    return converted


def _close(observed: Any, expected: Any, where: str) -> None:
    left = _finite(observed, where)
    right = _finite(expected, f"recomputed {where}")
    _require(
        math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9),
        f"{where} mismatch: ledger={left!r}, recomputed={right!r}",
    )


def percentile(values: Sequence[float], q: float) -> float:
    _require(bool(values) and 0.0 <= q <= 1.0, "percentile requires data and q in [0,1]")
    ordered = sorted(_finite(value, "percentile value") for value in values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def verify_manifest(
    output_dir: Path,
    manifest: Mapping[str, Any],
    required_names: Sequence[str] = PRODUCER_REQUIRED_ARTIFACTS,
) -> dict[str, Any]:
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA, "manifest schema drifted")
    declared_required = manifest.get("required_artifacts")
    _require(isinstance(declared_required, list), "manifest required_artifacts is not a list")
    _require(
        tuple(str(name) for name in declared_required) == tuple(required_names),
        "manifest required_artifacts differs from the frozen producer set",
    )
    files = _mapping(manifest.get("files"), "manifest.files")
    _require(set(required_names).issubset(files), "manifest omits a required artifact")
    excluded_control_files = {
        "MANIFEST.json", "RUN_STATUS.json", "COMPLETE.json", "FAILURE.json"
    }
    observed_bindable_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in excluded_control_files
    }
    _require(
        observed_bindable_files == set(files),
        "output directory has a missing or unbound non-control file",
    )
    verified: list[str] = []
    for raw_name, raw_metadata in sorted(files.items()):
        name = str(raw_name)
        _require(Path(name).name == name and name not in (".", ".."), "unsafe manifest path")
        metadata = _mapping(raw_metadata, f"manifest.files[{name!r}]")
        path = output_dir / name
        _require(path.is_file(), f"manifest-bound file is missing: {name}")
        _require(not path.is_symlink(), f"manifest-bound file may not be a symlink: {name}")
        expected_size = _integer(metadata.get("size_bytes"), f"manifest size for {name}")
        _require(path.stat().st_size == expected_size, f"manifest size mismatch: {name}")
        expected_hash = _digest(metadata.get("sha256"), f"manifest hash for {name}")
        _require(sha256_file(path) == expected_hash, f"manifest hash mismatch: {name}")
        verified.append(name)
    return {
        "status": "PASS",
        "schema_version": MANIFEST_SCHEMA,
        "verified_file_count": len(verified),
        "verified_files": verified,
        "control_files_not_bound_by_manifest": [
            "MANIFEST.json",
            "RUN_STATUS.json",
            "COMPLETE.json",
        ],
    }


def verify_completion(
    output_dir: Path,
    complete: Mapping[str, Any],
    status: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    _require(complete.get("schema_version") == COMPLETE_SCHEMA, "completion schema drifted")
    _require(complete.get("status") == "COMPLETE", "completion sentinel is not COMPLETE")
    _require(complete.get("completion_last") is True, "completion_last is not true")
    _require(not (output_dir / "FAILURE.json").exists(),
             "complete output also contains FAILURE.json")
    for field, name in (
        ("manifest_sha256", "MANIFEST.json"),
        ("summary_sha256", "summary.json"),
        ("run_status_sha256", "RUN_STATUS.json"),
    ):
        expected = _digest(complete.get(field), f"COMPLETE.{field}")
        _require(sha256_file(output_dir / name) == expected, f"COMPLETE hash mismatch: {name}")
    _require(complete.get("verdict") == summary.get("verdict"), "completion/summary verdict drift")
    _require(status.get("verdict") == summary.get("verdict"), "status/summary verdict drift")
    return {
        "status": "PASS",
        "completion_last": True,
        "bound_files": ["MANIFEST.json", "summary.json", "RUN_STATUS.json"],
    }


def validate_frozen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drifted")
    _require(config.get("status") == "FROZEN_PRE_RUN", "config is not frozen pre-run")
    workload = _mapping(config.get("workload"), "config.workload")
    model = _mapping(config.get("model"), "config.model")
    execution = _mapping(config.get("execution"), "config.execution")
    gate = _mapping(config.get("gate"), "config.gate")

    frozen_dimensions = {
        "expected_requests": 128,
        "expected_decode_steps_per_request": 16,
        "expected_request_steps": 2048,
        "max_batch_size": 8,
    }
    for field, expected in frozen_dimensions.items():
        _require(_integer(workload.get(field), f"config.workload.{field}") == expected,
                 f"frozen workload field drifted: {field}")
    _require(workload.get("teacher_force_frozen_native_tokens") is True,
             "teacher-forced roster flag drifted")
    _require(model.get("num_hidden_layers") == 16, "frozen layer count drifted")
    _require(model.get("num_experts") == 64, "frozen expert count drifted")
    _require(model.get("num_experts_per_tok") == 8, "frozen top-k drifted")

    _require(tuple(execution.get("arms", [])) == ARMS, "execution arms drifted")
    _require(execution.get("canonical_m") == CANONICAL_M, "canonical C drifted")
    _require(execution.get("warmup_replays_per_arm") == 1, "warmup count drifted")
    _require(execution.get("measured_replays_per_arm") == 2, "measured count drifted")
    _require(tuple(execution.get("warmup_arm_order", [])) == ARMS, "warmup order drifted")
    measured_orders = tuple(tuple(row) for row in execution.get("measured_arm_orders", []))
    _require(
        measured_orders
        == ((ARM_NATIVE, ARM_SERIAL, ARM_C8), (ARM_SERIAL, ARM_C8, ARM_NATIVE)),
        "measured replay orders drifted",
    )
    _require(execution.get("immediate_flush") is True, "immediate_flush drifted")
    _require(execution.get("cross_step_wait") is False, "cross_step_wait drifted")
    _require(execution.get("maximum_wall_seconds") == 2700, "wall budget drifted")
    _require(dict(gate) == FROZEN_GATE, "frozen gate thresholds or verdicts drifted")
    return {
        "status": "PASS",
        "schema_version": CONFIG_SCHEMA,
        "arms": list(ARMS),
        "canonical_m": CANONICAL_M,
        "requests": 128,
        "decode_steps_per_request": 16,
        "request_steps": 2048,
        "layers": 16,
        "top_k": 8,
        "gate": dict(FROZEN_GATE),
    }


def audit_roster(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    workload = _mapping(config["workload"], "config.workload")
    expected_requests = _integer(workload["expected_requests"], "expected requests")
    expected_steps = _integer(
        workload["expected_decode_steps_per_request"], "expected decode steps"
    )
    expected_request_steps = _integer(
        workload["expected_request_steps"], "expected request steps"
    )
    max_batch = _integer(workload["max_batch_size"], "maximum batch size")
    _require(bool(rows), "native roster is empty")

    observed: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    admission_by_request: dict[str, dict[str, Any]] = {}
    seen_pairs: set[tuple[str, int]] = set()
    normalized: dict[int, dict[str, Any]] = {}
    histogram: Counter[str] = Counter()
    for expected_batch, row in enumerate(rows):
        where = f"native_roster[{expected_batch}]"
        _require(row.get("schema_version") == ROSTER_SCHEMA, f"{where} schema drifted")
        batch_index = _integer(row.get("batch_index"), f"{where}.batch_index")
        _require(batch_index == expected_batch, "native roster batch indexes are not contiguous")
        request_ids = [str(value) for value in row.get("request_ids", [])]
        decode_steps = [_integer(value, f"{where}.decode_steps") for value in row.get("decode_steps", [])]
        input_tokens = [_integer(value, f"{where}.input_token_ids") for value in row.get("input_token_ids", [])]
        predicted = [_integer(value, f"{where}.native_predicted_next_token_ids")
                     for value in row.get("native_predicted_next_token_ids", [])]
        position_ids = [_integer(value, f"{where}.position_ids") for value in row.get("position_ids", [])]
        prior_lengths = [_integer(value, f"{where}.prior_cache_lengths")
                         for value in row.get("prior_cache_lengths", [])]
        left_padding = [_integer(value, f"{where}.left_padding") for value in row.get("left_padding", [])]
        admission = row.get("admission_identity", [])
        _require(isinstance(admission, list), f"{where}.admission_identity must be a list")
        width = len(request_ids)
        _require(0 < width <= max_batch, f"{where} batch width violates maximum")
        _require(len(set(request_ids)) == width, f"{where} duplicates a request")
        _require(
            all(len(column) == width for column in (
                decode_steps, input_tokens, predicted, position_ids,
                prior_lengths, left_padding, admission,
            )),
            f"{where} columns have different widths",
        )
        _require(_integer(row.get("batch_size"), f"{where}.batch_size") == width,
                 f"{where}.batch_size drifted")
        _require(_integer(row.get("pending_request_count"), f"{where}.pending_request_count") >= 0,
                 f"{where}.pending_request_count is negative")
        active = {str(value) for value in row.get("active_request_ids", [])}
        _require(set(request_ids).issubset(active), f"{where} is outside its active set")
        maximum_prior = max(prior_lengths)
        for index, (request_id, step) in enumerate(zip(request_ids, decode_steps)):
            pair = (request_id, step)
            _require(pair not in seen_pairs, f"duplicate native roster request-step: {pair}")
            seen_pairs.add(pair)
            _require(position_ids[index] == prior_lengths[index], f"{where} position/KV mismatch")
            _require(left_padding[index] == maximum_prior - prior_lengths[index],
                     f"{where} left-padding mismatch")
            identity = _mapping(admission[index], f"{where}.admission_identity[{index}]")
            _require(str(identity.get("request_id")) == request_id,
                     f"{where} admission identity lost alignment")
            normalized_identity = {
                "request_id": request_id,
                "arrival_us": _finite(identity.get("arrival_us"), f"{where}.arrival_us"),
                "sample_id": _integer(identity.get("sample_id"), f"{where}.sample_id"),
            }
            previous = admission_by_request.setdefault(request_id, normalized_identity)
            _require(previous == normalized_identity, f"admission identity drifted: {request_id}")
            observed[request_id].append(
                (step, input_tokens[index], predicted[index], prior_lengths[index])
            )
        route_hash = _digest(
            row.get("native_route_membership_sha256"), f"{where}.native route hash"
        )
        final_hash = _digest(
            row.get("native_final_logits_sha256"), f"{where}.native logits hash"
        )
        histogram[str(width)] += 1
        normalized[batch_index] = {
            "request_ids": request_ids,
            "decode_steps": decode_steps,
            "input_token_ids": input_tokens,
            "native_predicted_next_token_ids": predicted,
            "position_ids": position_ids,
            "native_route_membership_sha256": route_hash,
            "native_final_logits_sha256": final_hash,
        }

    _require(len(observed) == expected_requests, "native roster request count is incomplete")
    frozen_steps = list(range(expected_steps))
    for request_id, records in observed.items():
        records.sort(key=lambda item: item[0])
        _require([item[0] for item in records] == frozen_steps,
                 f"native roster steps do not close: {request_id}")
        initial_prior = records[0][3]
        _require(initial_prior > 0, f"native roster initial KV length is invalid: {request_id}")
        for index, record in enumerate(records):
            _require(record[3] == initial_prior + index,
                     f"native roster KV chain drifted: {request_id}")
            if index:
                _require(records[index - 1][2] == record[1],
                         f"native roster token chain drifted: {request_id}")
    _require(len(seen_pairs) == expected_request_steps, "native roster denominator is incomplete")
    report = {
        "status": "PASS",
        "requests": len(observed),
        "request_steps": len(seen_pairs),
        "decode_batches": len(rows),
        "batch_size_histogram": dict(sorted(histogram.items())),
        "maximum_batch_size": max(len(row["request_ids"]) for row in normalized.values()),
        "roster_sha256": canonical_sha256(list(rows)),
    }
    return report, normalized


ReplayKey = tuple[str, str, int]


def _replay_key(row: Mapping[str, Any], where: str) -> ReplayKey:
    arm = str(row.get("arm"))
    phase = str(row.get("phase"))
    repeat = _integer(row.get("repeat"), f"{where}.repeat")
    _require(arm in ARMS, f"{where} has an unknown arm")
    _require(phase in ("warmup", "measured"), f"{where} has an unknown phase")
    return arm, phase, repeat


def expected_replay_order(config: Mapping[str, Any]) -> list[ReplayKey]:
    execution = _mapping(config["execution"], "config.execution")
    output = [(str(arm), "warmup", 0) for arm in execution["warmup_arm_order"]]
    for repeat, order in enumerate(execution["measured_arm_orders"]):
        output.extend((str(arm), "measured", repeat) for arm in order)
    return output


def group_ledger(
    rows: Sequence[Mapping[str, Any]],
    name: str,
    expected_keys: set[ReplayKey],
    schema: str | None,
) -> dict[ReplayKey, list[Mapping[str, Any]]]:
    grouped: dict[ReplayKey, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        where = f"{name}[{index}]"
        if schema is not None:
            _require(row.get("schema_version") == schema, f"{where} schema drifted")
        key = _replay_key(row, where)
        _require(key in expected_keys, f"{where} has an unexpected replay identity: {key}")
        grouped[key].append(row)
    _require(set(grouped) == expected_keys, f"{name} does not close every arm/phase/repeat")
    return dict(grouped)


def iter_ordered_call_batches(
    path: Path,
    order: Sequence[ReplayKey],
    num_batches: int,
) -> Iterator[tuple[ReplayKey, int, list[Mapping[str, Any]]]]:
    """Stream the largest ledger while enforcing producer replay/batch order."""

    rank_by_key = {key: rank for rank, key in enumerate(order)}
    current_identity: tuple[ReplayKey, int] | None = None
    current_rows: list[Mapping[str, Any]] = []
    previous_position = (-1, -1)
    for row_index, row in enumerate(iter_jsonl(path)):
        where = f"expert_call_ledger[{row_index}]"
        _require(row.get("schema_version") == CALL_SCHEMA, f"{where} schema drifted")
        key = _replay_key(row, where)
        _require(key in rank_by_key, f"{where} has an unexpected replay identity: {key}")
        batch_index = _integer(row.get("batch_index"), f"{where}.batch_index")
        _require(0 <= batch_index < num_batches, f"{where} batch is out of range")
        position = (rank_by_key[key], batch_index)
        _require(position >= previous_position, f"{where} replay/batch order drifted")
        previous_position = position
        identity = (key, batch_index)
        if current_identity is not None and identity != current_identity:
            yield current_identity[0], current_identity[1], current_rows
            current_rows = []
        current_identity = identity
        current_rows.append(row)
    if current_identity is not None:
        yield current_identity[0], current_identity[1], current_rows


def _validate_call_policy(row: Mapping[str, Any], arm: str, where: str) -> None:
    logical_m = _integer(row.get("logical_m"), f"{where}.logical_m")
    physical_m = _integer(row.get("physical_m"), f"{where}.physical_m")
    padding = _integer(row.get("padding_rows"), f"{where}.padding_rows")
    kernels = _integer(row.get("kernel_calls"), f"{where}.kernel_calls")
    per_kernel = [_integer(value, f"{where}.physical_m_per_kernel")
                  for value in row.get("physical_m_per_kernel", [])]
    _require(1 <= logical_m <= CANONICAL_M, f"{where}.logical_m is outside [1,8]")
    if arm == ARM_NATIVE:
        expected = (logical_m, 0, 1, [logical_m])
    elif arm == ARM_SERIAL:
        expected = (1, 0, logical_m, [1] * logical_m)
    else:
        expected = (CANONICAL_M, CANONICAL_M - logical_m, 1, [CANONICAL_M])
    _require((physical_m, padding, kernels, per_kernel) == expected,
             f"{where} violates the {arm} execution mechanics")


def reconstruct_batch_evidence(
    *,
    key: ReplayKey,
    batch_index: int,
    step: Mapping[str, Any],
    roster: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    num_layers: int,
    num_experts: int,
    top_k: int,
) -> dict[str, Any]:
    """Reconstruct one batch solely from roster/call/stage/decode ledgers."""

    arm, phase, repeat = key
    where = f"{arm}/{phase}/{repeat}/batch/{batch_index}"
    _require(step.get("schema_version") == STEP_SCHEMA, f"{where} step schema drifted")
    _require(_replay_key(step, f"{where}.step") == key, f"{where} step key drifted")
    _require(_integer(step.get("batch_index"), f"{where}.batch_index") == batch_index,
             f"{where} step batch identity drifted")

    request_ids = [str(value) for value in step.get("request_ids", [])]
    decode_steps = [_integer(value, f"{where}.decode_steps")
                    for value in step.get("decode_steps", [])]
    input_tokens = [_integer(value, f"{where}.input_token_ids")
                    for value in step.get("input_token_ids", [])]
    targets = [_integer(value, f"{where}.teacher targets")
               for value in step.get("teacher_forced_target_token_ids", [])]
    nll_values = [_finite(value, f"{where}.frozen_token_nll")
                  for value in step.get("frozen_token_nll", [])]
    greedy = [_integer(value, f"{where}.greedy_token_ids")
              for value in step.get("greedy_token_ids", [])]
    width = len(request_ids)
    _require(width > 0 and len(set(request_ids)) == width, f"{where} request width is invalid")
    _require(all(len(values) == width for values in (
        decode_steps, input_tokens, targets, nll_values, greedy,
    )), f"{where} decode columns have different widths")
    _require(request_ids == roster["request_ids"], f"{where} request roster drift")
    _require(decode_steps == roster["decode_steps"], f"{where} decode-step roster drift")
    _require(input_tokens == roster["input_token_ids"], f"{where} input-token roster drift")
    _require(targets == roster["native_predicted_next_token_ids"], f"{where} teacher target drift")
    _require(all(value >= 0.0 for value in nll_values), f"{where} has negative NLL")
    _require(
        step.get("timing_boundaries")
        == {
            "whole_step": "cuda_sync_before_model_to_logits_ready_cuda_sync",
            "expert_stage": (
                "sum_of_layer_cuda_events_after_router_topk_through_"
                "dispatch_padding_expert_and_index_add"
            ),
        },
        f"{where} decode timing boundaries drifted",
    )

    roster_identity = {
        "batch_index": batch_index,
        "request_ids": request_ids,
        "decode_steps": decode_steps,
        "input_token_ids": input_tokens,
        "position_ids": roster["position_ids"],
    }
    roster_digest = canonical_sha256(roster_identity)
    _require(step.get("roster_identity_sha256") == roster_digest,
             f"{where} roster identity digest mismatch")
    _require(
        step.get("native_roster_route_membership_sha256")
        == roster["native_route_membership_sha256"],
        f"{where} native route reference drift",
    )
    _require(
        step.get("native_roster_final_logits_sha256")
        == roster["native_final_logits_sha256"],
        f"{where} native logits reference drift",
    )

    stage_by_layer: dict[int, Mapping[str, Any]] = {}
    for index, stage in enumerate(stages):
        stage_where = f"{where}.stage[{index}]"
        _require(stage.get("schema_version") == STAGE_SCHEMA, f"{stage_where} schema drifted")
        _require(_replay_key(stage, stage_where) == key, f"{stage_where} key drifted")
        _require(_integer(stage.get("batch_index"), f"{stage_where}.batch_index") == batch_index,
                 f"{stage_where} batch drifted")
        layer = _integer(stage.get("layer"), f"{stage_where}.layer")
        _require(layer not in stage_by_layer, f"{where} duplicates expert-stage layer {layer}")
        _require(0 <= layer < num_layers, f"{stage_where} layer is out of range")
        _finite(stage.get("expert_stage_gpu_ms"), f"{stage_where}.gpu_ms", positive=True)
        _require(
            stage.get("timing_boundary")
            == "after_router_topk_through_dispatch_padding_expert_and_index_add",
            f"{stage_where} timing boundary drifted",
        )
        stage_by_layer[layer] = stage
    _require(set(stage_by_layer) == set(range(num_layers)), f"{where} stage layers do not close")

    calls_by_layer: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    observed_call_order: list[tuple[int, int]] = []
    for index, call in enumerate(calls):
        call_where = f"{where}.call[{index}]"
        _require(call.get("schema_version") == CALL_SCHEMA, f"{call_where} schema drifted")
        _require(_replay_key(call, call_where) == key, f"{call_where} key drifted")
        _require(_integer(call.get("batch_index"), f"{call_where}.batch_index") == batch_index,
                 f"{call_where} batch drifted")
        layer = _integer(call.get("layer"), f"{call_where}.layer")
        expert = _integer(call.get("expert_id"), f"{call_where}.expert_id")
        _require(0 <= layer < num_layers, f"{call_where} layer is out of range")
        _require(0 <= expert < num_experts, f"{call_where} expert is out of range")
        observed_call_order.append((layer, expert))
        calls_by_layer[layer].append(call)
    _require(observed_call_order == sorted(observed_call_order), f"{where} call order is not canonical")

    token_index_by_request = {request_id: index for index, request_id in enumerate(request_ids)}
    route_entries: list[dict[str, Any]] = []
    raw_payload: list[dict[str, Any]] = []
    kernel_calls = 0
    real_rows = 0
    dummy_rows = 0
    natural_histogram: Counter[str] = Counter()
    seen_row_ids: set[str] = set()
    for layer in range(num_layers):
        layer_calls = calls_by_layer.get(layer, [])
        _require(bool(layer_calls), f"{where} layer {layer} has no occupied expert")
        experts = [_integer(call["expert_id"], f"{where}.expert_id") for call in layer_calls]
        _require(len(experts) == len(set(experts)), f"{where} layer {layer} duplicates an expert call")
        layer_slots: set[str] = set()
        by_token: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for call_index, call in enumerate(layer_calls):
            call_where = f"{where}.layer[{layer}].call[{call_index}]"
            _validate_call_policy(call, arm, call_where)
            expert = _integer(call["expert_id"], f"{call_where}.expert_id")
            logical_m = _integer(call["logical_m"], f"{call_where}.logical_m")
            padding = _integer(call["padding_rows"], f"{call_where}.padding_rows")
            kernels = _integer(call["kernel_calls"], f"{call_where}.kernel_calls")
            slot_ids = [str(value) for value in call.get("slot_ids", [])]
            row_ids = [str(value) for value in call.get("row_ids", [])]
            raw_rows = call.get("raw_row_sha256", [])
            _require(isinstance(raw_rows, list), f"{call_where}.raw_row_sha256 must be a list")
            _require(len(slot_ids) == len(row_ids) == len(raw_rows) == logical_m,
                     f"{call_where} routed-row widths drifted")
            _require(len(set(slot_ids)) == logical_m and len(set(row_ids)) == logical_m,
                     f"{call_where} duplicates a slot or row")
            _require(call.get("route_membership_sha256") == canonical_sha256(row_ids),
                     f"{call_where} row-membership digest mismatch")
            _digest(call.get("raw_bf16_sha256"), f"{call_where}.raw_bf16_sha256")
            call_positions: list[tuple[int, int]] = []
            for offset, (slot_id, row_id, raw_row) in enumerate(zip(slot_ids, row_ids, raw_rows)):
                match = _SLOT.fullmatch(slot_id)
                _require(match is not None, f"{call_where} has malformed slot_id")
                request_id = match.group("request")
                step_value = int(match.group("step"))
                slot_layer = int(match.group("layer"))
                rank = int(match.group("rank"))
                _require(request_id in token_index_by_request, f"{call_where} has unknown request slot")
                token_index = token_index_by_request[request_id]
                _require(step_value == decode_steps[token_index], f"{call_where} slot decode step drifted")
                _require(slot_layer == layer, f"{call_where} slot layer drifted")
                _require(0 <= rank < top_k, f"{call_where} top-k rank is out of range")
                expected_row_id = f"{slot_id}:expert:{expert:02d}"
                _require(row_id == expected_row_id, f"{call_where} row identity drifted")
                _require(slot_id not in layer_slots, f"{where} layer {layer} duplicates a route slot")
                _require(row_id not in seen_row_ids, f"{where} duplicates a route row")
                layer_slots.add(slot_id)
                seen_row_ids.add(row_id)
                raw_row_map = _mapping(raw_row, f"{call_where}.raw_row_sha256[{offset}]")
                _require(raw_row_map.get("slot_id") == slot_id and raw_row_map.get("row_id") == row_id,
                         f"{call_where} raw-row identity drifted")
                _digest(raw_row_map.get("sha256"), f"{call_where}.raw row hash")
                route_entries.append({
                    "layer": layer,
                    "token_index": token_index,
                    "request_id": request_id,
                    "decode_step": step_value,
                    "topk_rank": rank,
                    "expert_id": expert,
                    "slot_id": slot_id,
                    "row_id": row_id,
                })
                by_token[token_index].append((rank, expert))
                call_positions.append((token_index, rank))
            _require(
                call_positions
                == sorted(call_positions, key=lambda item: (item[1], item[0])),
                f"{call_where} row order is not producer rank-major order",
            )
            raw_payload.append({
                "layer": layer,
                "expert_id": expert,
                "logical_m": logical_m,
                "physical_m": _integer(call["physical_m"], f"{call_where}.physical_m"),
                "kernel_calls": kernels,
                "row_ids": row_ids,
                "raw_bf16_sha256": call["raw_bf16_sha256"],
                "raw_row_sha256": raw_rows,
            })
            kernel_calls += kernels
            real_rows += logical_m
            dummy_rows += padding
            natural_histogram[str(logical_m)] += 1
        _require(len(layer_slots) == width * top_k, f"{where} layer {layer} route slots do not close")
        _require(set(by_token) == set(range(width)), f"{where} layer {layer} omits a token")
        for token_index, ranked in by_token.items():
            ranked.sort()
            _require([rank for rank, _ in ranked] == list(range(top_k)),
                     f"{where} layer {layer} token {token_index} top-k ranks do not close")
            _require(len({expert for _, expert in ranked}) == top_k,
                     f"{where} layer {layer} token {token_index} repeats an expert")
        _require(
            _integer(stage_by_layer[layer].get("occupied_experts"),
                     f"{where}.stage[{layer}].occupied_experts") == len(layer_calls),
            f"{where} layer {layer} stage/call occupancy drift",
        )

    route_entries.sort(key=lambda row: (row["layer"], row["token_index"], row["topk_rank"]))
    expected_routes = width * num_layers * top_k
    _require(len(route_entries) == expected_routes, f"{where} total routes do not close")
    route_digest = canonical_sha256(route_entries)
    raw_digest = canonical_sha256(raw_payload)
    _require(step.get("route_membership_sha256") == route_digest,
             f"{where} independently rebuilt route digest mismatch")
    _require(step.get("raw_calls_sha256") == raw_digest,
             f"{where} independently rebuilt raw-call digest mismatch")
    final_hash = _digest(step.get("final_logits_sha256"), f"{where}.final logits hash")
    expert_ms = sum(
        _finite(stage_by_layer[layer]["expert_stage_gpu_ms"], f"{where}.stage gpu ms")
        for layer in range(num_layers)
    )
    wall_ms = _finite(step.get("whole_step_wall_ms"), f"{where}.whole_step_wall_ms", positive=True)
    _close(step.get("expert_stage_gpu_ms"), expert_ms, f"{where}.expert_stage_gpu_ms")
    batch_mean_nll = statistics.fmean(nll_values)
    _close(step.get("mean_frozen_token_nll"), batch_mean_nll,
           f"{where}.mean_frozen_token_nll")
    return {
        "signature": {
            "batch_index": batch_index,
            "roster_identity_sha256": roster_digest,
            "raw_calls_sha256": raw_digest,
            "route_membership_sha256": route_digest,
            "final_logits_sha256": final_hash,
            "greedy_token_ids": greedy,
        },
        "request_steps": width,
        "expert_gpu_ms": expert_ms,
        "wall_ms": wall_ms,
        "nll_values": nll_values,
        "kernel_calls": kernel_calls,
        "real_rows": real_rows,
        "dummy_rows": dummy_rows,
        "occupied_experts": len(calls),
        "natural_m_histogram": dict(sorted(natural_histogram.items())),
        "route_rows": len(route_entries),
    }


def _compare_replay_metrics(
    ledger: Mapping[str, Any], computed: Mapping[str, Any], where: str
) -> None:
    for field in (
        "decode_batches", "request_steps", "kernel_calls", "real_rows",
        "dummy_rows", "occupied_experts",
    ):
        _require(_integer(ledger.get(field), f"{where}.{field}") == computed[field],
                 f"{where}.{field} mismatch")
    _require(
        {str(key): int(value) for key, value in _mapping(
            ledger.get("natural_m_histogram"), f"{where}.natural_m_histogram"
        ).items()} == computed["natural_m_histogram"],
        f"{where}.natural_m_histogram mismatch",
    )
    for field in (
        "total_expert_gpu_ms", "total_whole_step_wall_ms", "token_step_p99_ms",
        "padding_fraction", "mean_frozen_token_nll",
    ):
        _close(ledger.get(field), computed[field], f"{where}.{field}")
    prefill = _finite(
        ledger.get("prefill_wall_ms_excluded_from_decode_metrics"),
        f"{where}.prefill_wall_ms_excluded_from_decode_metrics",
    )
    _require(prefill >= 0.0, f"{where}.prefill_wall_ms is negative")


def recompute_replays(
    *,
    replay_rows: Sequence[Mapping[str, Any]],
    step_rows: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
    call_ledger_path: Path,
    roster: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[ReplayKey, dict[str, Any]], dict[str, Any]]:
    order = expected_replay_order(config)
    expected_keys = set(order)
    _require(len(order) == len(expected_keys) == 9, "frozen replay identities are not unique")
    observed_order = [_replay_key(row, f"replay_ledger[{index}]")
                      for index, row in enumerate(replay_rows)]
    _require(observed_order == order, "replay ledger arm/phase/repeat order drifted")
    replay_grouped = group_ledger(replay_rows, "replay_ledger", expected_keys, None)
    _require(all(len(rows) == 1 for rows in replay_grouped.values()),
             "replay ledger duplicates an arm/phase/repeat")
    steps_grouped = group_ledger(step_rows, "decode_step_ledger", expected_keys, STEP_SCHEMA)
    stages_grouped = group_ledger(stage_rows, "expert_stage_ledger", expected_keys, STAGE_SCHEMA)

    num_batches = len(roster)
    model = _mapping(config["model"], "config.model")
    num_layers = _integer(model["num_hidden_layers"], "model.num_hidden_layers")
    num_experts = _integer(model["num_experts"], "model.num_experts")
    top_k = _integer(model["num_experts_per_tok"], "model.num_experts_per_tok")
    expected_request_steps = _integer(config["workload"]["expected_request_steps"],
                                      "workload.expected_request_steps")

    _require(len(step_rows) == len(order) * num_batches, "decode-step critical row count mismatch")
    _require(len(stage_rows) == len(order) * num_batches * num_layers,
             "expert-stage critical row count mismatch")
    recomputed: dict[ReplayKey, dict[str, Any]] = {}
    total_route_rows = 0
    total_call_rows = 0
    call_batches = iter_ordered_call_batches(call_ledger_path, order, num_batches)
    for key in order:
        steps = steps_grouped[key]
        stages = stages_grouped[key]
        step_by_batch: dict[int, Mapping[str, Any]] = {}
        for row in steps:
            batch = _integer(row.get("batch_index"), "decode step batch_index")
            _require(batch not in step_by_batch, f"{key} duplicates decode batch {batch}")
            step_by_batch[batch] = row
        _require(set(step_by_batch) == set(range(num_batches)), f"{key} decode batches do not close")
        stage_by_batch: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in stages:
            stage_by_batch[_integer(row.get("batch_index"), "stage batch_index")].append(row)
        _require(set(stage_by_batch) == set(range(num_batches)), f"{key} stage batches do not close")

        batch_results: list[dict[str, Any]] = []
        for batch_index in range(num_batches):
            try:
                call_key, call_batch_index, batch_calls = next(call_batches)
            except StopIteration as exc:
                raise RecomputeError(
                    f"expert_call_ledger ends before {key} batch {batch_index}"
                ) from exc
            _require(
                (call_key, call_batch_index) == (key, batch_index),
                f"expert_call_ledger expected {(key, batch_index)!r}, "
                f"observed {(call_key, call_batch_index)!r}",
            )
            total_call_rows += len(batch_calls)
            result = reconstruct_batch_evidence(
                key=key,
                batch_index=batch_index,
                step=step_by_batch[batch_index],
                roster=roster[batch_index],
                stages=stage_by_batch[batch_index],
                calls=batch_calls,
                num_layers=num_layers,
                num_experts=num_experts,
                top_k=top_k,
            )
            batch_results.append(result)
            total_route_rows += int(result["route_rows"])
        request_steps = sum(int(row["request_steps"]) for row in batch_results)
        _require(request_steps == expected_request_steps, f"{key} request-step denominator drifted")
        real_rows = sum(int(row["real_rows"]) for row in batch_results)
        dummy_rows = sum(int(row["dummy_rows"]) for row in batch_results)
        histogram: Counter[str] = Counter()
        for row in batch_results:
            histogram.update(row["natural_m_histogram"])
        token_latencies = [
            float(row["wall_ms"])
            for row in batch_results
            for _ in range(int(row["request_steps"]))
        ]
        nll_values = [float(value) for row in batch_results for value in row["nll_values"]]
        computed = {
            "arm": key[0],
            "phase": key[1],
            "repeat": key[2],
            "decode_batches": num_batches,
            "request_steps": request_steps,
            "total_expert_gpu_ms": sum(float(row["expert_gpu_ms"]) for row in batch_results),
            "total_whole_step_wall_ms": sum(float(row["wall_ms"]) for row in batch_results),
            "token_step_p99_ms": percentile(token_latencies, 0.99),
            "kernel_calls": sum(int(row["kernel_calls"]) for row in batch_results),
            "real_rows": real_rows,
            "dummy_rows": dummy_rows,
            "padding_fraction": dummy_rows / (real_rows + dummy_rows)
            if real_rows + dummy_rows else 0.0,
            "occupied_experts": sum(int(row["occupied_experts"]) for row in batch_results),
            "natural_m_histogram": dict(sorted(histogram.items())),
            "mean_frozen_token_nll": statistics.fmean(nll_values),
            "step_signatures": [row["signature"] for row in batch_results],
        }
        ledger_row = replay_grouped[key][0]
        _compare_replay_metrics(ledger_row, computed, f"replay {key}")
        computed["prefill_wall_ms_excluded_from_decode_metrics"] = _finite(
            ledger_row["prefill_wall_ms_excluded_from_decode_metrics"], "prefill wall ms"
        )
        computed["ledger_row"] = dict(ledger_row)
        recomputed[key] = computed

    try:
        extra_call_batch = next(call_batches)
    except StopIteration:
        extra_call_batch = None
    _require(extra_call_batch is None, "expert_call_ledger has trailing replay/batch rows")

    counts = {
        "status": "PASS",
        "native_roster_rows": {
            "observed": num_batches,
            "expected": "data_dependent; reconciled to every replay batch index",
        },
        "replay_rows": {"observed": len(replay_rows), "expected": 9},
        "decode_step_rows": {
            "observed": len(step_rows), "expected": 9 * num_batches,
        },
        "expert_stage_rows": {
            "observed": len(stage_rows), "expected": 9 * num_batches * num_layers,
        },
        "expert_call_rows": {
            "observed": total_call_rows,
            "expected": "data_dependent; reconciled to occupied_experts per replay",
        },
        "route_rows": {
            "observed": total_route_rows,
            "expected": 9 * expected_request_steps * num_layers * top_k,
        },
    }
    _require(counts["route_rows"]["observed"] == counts["route_rows"]["expected"],
             "global route-row denominator drifted")
    return recomputed, counts


def compare_within_arm_repeats(
    replays: Mapping[ReplayKey, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        left_rows = replays[(arm, "measured", 0)]["step_signatures"]
        right_rows = replays[(arm, "measured", 1)]["step_signatures"]
        left = {int(row["batch_index"]): row for row in left_rows}
        right = {int(row["batch_index"]): row for row in right_rows}
        _require(len(left) == len(left_rows) and len(right) == len(right_rows),
                 f"{arm} repeat comparison duplicates a batch")
        _require(set(left) == set(right), f"{arm} repeats cover different batches")
        raw = route = final = greedy = 0
        for batch_index in sorted(left):
            lhs, rhs = left[batch_index], right[batch_index]
            _require(lhs["roster_identity_sha256"] == rhs["roster_identity_sha256"],
                     f"{arm} repeat changed roster identity at batch {batch_index}")
            raw += lhs["raw_calls_sha256"] != rhs["raw_calls_sha256"]
            route += lhs["route_membership_sha256"] != rhs["route_membership_sha256"]
            final += lhs["final_logits_sha256"] != rhs["final_logits_sha256"]
            left_tokens = lhs["greedy_token_ids"]
            right_tokens = rhs["greedy_token_ids"]
            _require(len(left_tokens) == len(right_tokens),
                     f"{arm} repeat greedy-token widths differ")
            greedy += sum(a != b for a, b in zip(left_tokens, right_tokens))
        output[arm] = {
            "arm": arm,
            "comparison_scope": "within_policy_repeat0_vs_repeat1",
            "m1_is_ground_truth": False,
            "batches": len(left),
            "raw_repeat_mismatches": int(raw),
            "route_repeat_mismatches": int(route),
            "final_logits_repeat_mismatches": int(final),
            "greedy_token_repeat_mismatches": int(greedy),
            "bitwise_repeat_stable": not (raw or route or final),
        }
    return output


def summarize_measured_arms(
    replays: Mapping[ReplayKey, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        values = [replays[(arm, "measured", repeat)] for repeat in (0, 1)]
        real = sum(int(row["real_rows"]) for row in values)
        dummy = sum(int(row["dummy_rows"]) for row in values)
        histogram: Counter[str] = Counter()
        for row in values:
            histogram.update(row["natural_m_histogram"])
        output[arm] = {
            "arm": arm,
            "measured_replays": [row["ledger_row"] for row in values],
            "median_total_expert_gpu_ms": statistics.median(
                float(row["total_expert_gpu_ms"]) for row in values
            ),
            "median_total_whole_step_wall_ms": statistics.median(
                float(row["total_whole_step_wall_ms"]) for row in values
            ),
            "median_token_step_p99_ms": statistics.median(
                float(row["token_step_p99_ms"]) for row in values
            ),
            "mean_frozen_token_nll": statistics.fmean(
                float(row["mean_frozen_token_nll"]) for row in values
            ),
            "kernel_calls": sum(int(row["kernel_calls"]) for row in values),
            "real_rows": real,
            "dummy_rows": dummy,
            "padding_fraction": dummy / (real + dummy) if real + dummy else 0.0,
            "occupied_experts": sum(int(row["occupied_experts"]) for row in values),
            "natural_m_histogram": dict(sorted(histogram.items())),
        }
    return output


def classify_independent(
    correctness: Mapping[str, Mapping[str, Any]],
    arm_metrics: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(correctness) == set(ARMS), "gate correctness arms are incomplete")
    _require(set(arm_metrics) == set(ARMS), "gate metric arms are incomplete")
    fixed = correctness[ARM_C8]
    correctness_failures: list[str] = []
    checks = (
        ("raw_repeat_mismatches", "require_fixed_c8_raw_repeat_mismatches",
         "fixed_c8_raw_repeat_mismatch"),
        ("route_repeat_mismatches", "require_fixed_c8_route_repeat_mismatches",
         "fixed_c8_route_repeat_mismatch"),
        ("final_logits_repeat_mismatches", "require_fixed_c8_final_logits_repeat_mismatches",
         "fixed_c8_final_logits_repeat_mismatch"),
    )
    for metric, requirement, failure in checks:
        if _integer(fixed.get(metric), f"correctness.{metric}") != _integer(
            gate.get(requirement), f"gate.{requirement}"
        ):
            correctness_failures.append(failure)
    serial_expert = _finite(
        arm_metrics[ARM_SERIAL]["median_total_expert_gpu_ms"], "serial expert median", positive=True
    )
    c8_expert = _finite(
        arm_metrics[ARM_C8]["median_total_expert_gpu_ms"], "C8 expert median", positive=True
    )
    native_p99 = _finite(
        arm_metrics[ARM_NATIVE]["median_token_step_p99_ms"], "native p99 median", positive=True
    )
    c8_p99 = _finite(
        arm_metrics[ARM_C8]["median_token_step_p99_ms"], "C8 p99 median", positive=True
    )
    expert_ratio = c8_expert / serial_expert
    p99_ratio = c8_p99 / native_p99
    expert_threshold = _finite(
        gate["maximum_c8_over_serial_expert_gpu_time_ratio"], "expert ratio threshold", positive=True
    )
    p99_threshold = _finite(
        gate["maximum_c8_over_native_token_step_p99_ratio"], "p99 ratio threshold", positive=True
    )
    cost_failures: list[str] = []
    if expert_ratio > expert_threshold:
        cost_failures.append("fixed_c8_expert_gpu_time_over_serial_threshold")
    if p99_ratio > p99_threshold:
        cost_failures.append("fixed_c8_token_step_p99_over_native_threshold")
    if correctness_failures:
        verdict = str(gate["correctness_fail"])
    elif cost_failures:
        verdict = str(gate["cost_fail"])
    else:
        verdict = str(gate["pass"])
    return {
        "verdict": verdict,
        "correctness_failures": correctness_failures,
        "cost_failures": cost_failures,
        "fixed_c8_over_serial_expert_gpu_time_ratio": expert_ratio,
        "fixed_c8_over_native_token_step_p99_ratio": p99_ratio,
        "expert_ratio_threshold": expert_threshold,
        "p99_ratio_threshold": p99_threshold,
        "m1_is_ground_truth": False,
    }


def _compare_arm_summary(
    observed: Mapping[str, Any], recomputed: Mapping[str, Any], arm: str
) -> None:
    _require(observed.get("arm") == arm, f"summary arm identity drifted: {arm}")
    for field in ("kernel_calls", "real_rows", "dummy_rows", "occupied_experts"):
        _require(_integer(observed.get(field), f"summary.{arm}.{field}") == recomputed[field],
                 f"summary.{arm}.{field} mismatch")
    _require(observed.get("natural_m_histogram") == recomputed["natural_m_histogram"],
             f"summary.{arm}.natural_m_histogram mismatch")
    for field in (
        "median_total_expert_gpu_ms", "median_total_whole_step_wall_ms",
        "median_token_step_p99_ms", "mean_frozen_token_nll", "padding_fraction",
    ):
        _close(observed.get(field), recomputed[field], f"summary.{arm}.{field}")
    _require(observed.get("measured_replays") == recomputed["measured_replays"],
             f"summary.{arm}.measured_replays differs from replay ledger")


def independent_recompute(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    _require(output_dir.is_dir(), f"output directory does not exist: {output_dir}")
    for name in AUDIT_INPUTS:
        _require((output_dir / name).is_file(), f"required input is missing: {name}")

    manifest = load_json_object(output_dir / "MANIFEST.json")
    manifest_report = verify_manifest(output_dir, manifest)
    config = load_json_object(output_dir / "config_snapshot.json")
    protocol_report = validate_frozen_config(config)
    summary = load_json_object(output_dir / "summary.json")
    status = load_json_object(output_dir / "RUN_STATUS.json")
    complete = load_json_object(output_dir / "COMPLETE.json")
    completion_report = verify_completion(output_dir, complete, status, summary)

    _require(summary.get("schema_version") == SUMMARY_SCHEMA, "summary schema drifted")
    _require(summary.get("status") == "COMPLETE", "summary status is not COMPLETE")
    _require(status.get("schema_version") == STATUS_SCHEMA, "RUN_STATUS schema drifted")
    _require(status.get("status") == "COMPLETE", "RUN_STATUS is not COMPLETE")
    _require(status.get("scientific_result_eligible") is True,
             "RUN_STATUS does not mark the result eligible")
    _require(status.get("required_sentinel") == "COMPLETE.json",
             "RUN_STATUS completion sentinel drifted")

    roster_rows = load_jsonl(output_dir / "native_roster.jsonl")
    replay_rows = load_jsonl(output_dir / "replay_ledger.jsonl")
    step_rows = load_jsonl(output_dir / "decode_step_ledger.jsonl")
    stage_rows = load_jsonl(output_dir / "expert_stage_ledger.jsonl")

    roster_report, roster = audit_roster(roster_rows, config)
    _require(summary.get("native_roster") == roster_report,
             "summary native_roster differs from independent recomputation")
    replays, row_counts = recompute_replays(
        replay_rows=replay_rows,
        step_rows=step_rows,
        stage_rows=stage_rows,
        call_ledger_path=output_dir / "expert_call_ledger.jsonl",
        roster=roster,
        config=config,
    )
    repeat_correctness = compare_within_arm_repeats(replays)
    measured_metrics = summarize_measured_arms(replays)

    observed_warmups = _mapping(summary.get("warmups"), "summary.warmups")
    _require(set(observed_warmups) == set(ARMS), "summary warmup arms are incomplete")
    for arm in ARMS:
        _require(
            observed_warmups[arm] == replays[(arm, "warmup", 0)]["ledger_row"],
            f"summary warmup {arm} differs from replay ledger",
        )
    observed_correctness = _mapping(
        summary.get("policy_repeat_correctness"), "summary.policy_repeat_correctness"
    )
    _require(observed_correctness == repeat_correctness,
             "summary repeat correctness differs from independent recomputation")
    observed_metrics = _mapping(summary.get("arm_metrics"), "summary.arm_metrics")
    _require(set(observed_metrics) == set(ARMS), "summary arm metrics are incomplete")
    for arm in ARMS:
        _compare_arm_summary(
            _mapping(observed_metrics[arm], f"summary.arm_metrics.{arm}"),
            measured_metrics[arm],
            arm,
        )

    gate_result = classify_independent(repeat_correctness, measured_metrics, config["gate"])
    _require(summary.get("gate") == gate_result,
             "summary gate object differs from independent classification")
    _require(summary.get("verdict") == gate_result["verdict"],
             "summary verdict differs from independent classification")
    _require(status.get("verdict") == gate_result["verdict"],
             "RUN_STATUS verdict differs from independent classification")
    _require(complete.get("verdict") == gate_result["verdict"],
             "COMPLETE verdict differs from independent classification")

    replay_metrics_for_report: dict[str, Any] = {}
    for key in expected_replay_order(config):
        value = replays[key]
        report_key = f"{key[1]}:{key[2]}:{key[0]}"
        replay_metrics_for_report[report_key] = {
            field: value[field]
            for field in (
                "decode_batches", "request_steps", "total_expert_gpu_ms",
                "total_whole_step_wall_ms", "token_step_p99_ms", "kernel_calls",
                "real_rows", "dummy_rows", "padding_fraction", "occupied_experts",
                "natural_m_histogram", "mean_frozen_token_nll",
            )
        }

    return {
        "schema_version": "stablebatch-shape-lane-independent-recompute-v1",
        "status": "PASS",
        "read_only": True,
        "output_dir": str(output_dir),
        "implementation_independence": {
            "producer_imported": False,
            "producer_classifier_imported": False,
            "runtime_or_model_imported": False,
            "semantic_inputs": list(AUDIT_INPUTS),
            "additional_reads": "byte hashing of every MANIFEST-bound artifact",
        },
        "integrity": {
            "manifest": manifest_report,
            "completion": completion_report,
        },
        "frozen_protocol": protocol_report,
        "critical_row_counts": row_counts,
        "arm_repeat_closure": {
            "status": "PASS",
            "observed_order": [
                {"arm": arm, "phase": phase, "repeat": repeat}
                for arm, phase, repeat in expected_replay_order(config)
            ],
            "warmups_per_arm": 1,
            "measured_repeats_per_arm": 2,
        },
        "native_roster": roster_report,
        "recomputed_replays": replay_metrics_for_report,
        "measured_arm_metrics": measured_metrics,
        "within_arm_repeat_mismatches": repeat_correctness,
        "evidence_support": {
            "roster_identity_sha256": {
                "status": "SUPPORTED",
                "method": "rebuilt from roster and decode-step identity fields",
            },
            "route_membership_sha256": {
                "status": "SUPPORTED",
                "method": "rebuilt from call slot/row identities and token alignment",
            },
            "raw_calls_sha256": {
                "status": "PARTIAL",
                "method": "rebuilt over stored per-call and per-row hashes",
                "unsupported": "raw BF16 tensor bytes are absent from the ledgers",
            },
            "final_logits_sha256": {
                "status": "UNSUPPORTED_FOR_VALUE_REHASH",
                "supported": "stored hash equality across within-arm repeats",
                "unsupported": "final-logit tensor bytes are absent from the ledgers",
            },
            "cuda_timing": {
                "status": "UNSUPPORTED_FOR_REMEASUREMENT",
                "supported": "ledger aggregation, p99, medians, and ratios",
                "unsupported": "no model/runtime execution is performed",
            },
        },
        "independent_gate": gate_result,
        "independent_verdict": gate_result["verdict"],
        "summary_status_completion_consistency": "PASS",
    }


def invalid_report(output_dir: Path, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": "stablebatch-shape-lane-independent-recompute-v1",
        "status": "FAIL",
        "read_only": True,
        "output_dir": str(output_dir.resolve()),
        "independent_verdict": INVALID_VERDICT,
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently recompute the sealed D10 shape-lane cost gate"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = independent_recompute(args.output_dir)
        exit_code = 0
    except (RecomputeError, OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        report = invalid_report(args.output_dir, exc)
        exit_code = 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
