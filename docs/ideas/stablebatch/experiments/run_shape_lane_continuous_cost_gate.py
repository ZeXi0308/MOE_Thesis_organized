#!/usr/bin/env python3
"""D10 fixed-C8 continuous-decode mechanism-cost gate.

The runner first captures one immutable native continuous-decode roster, then
teacher-forces that exact request/step schedule through three same-stack expert
execution policies: native variable M, serial M=1, and fixed C=8.  Correctness
is repeatability *within* a policy.  Serial M=1 and native execution are behavior
references, never scientific ground truth.

This is deliberately an eager single-GPU mechanism experiment.  Its timings do
not include a serving queue and are not vLLM/SGLang TPOT measurements.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import types
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "stablebatch-shape-lane-continuous-cost-gate-v1"
ARM_NATIVE = "native_variable_m"
ARM_SERIAL = "serial_m1"
ARM_C8 = "fixed_c8"
ARMS = (ARM_NATIVE, ARM_SERIAL, ARM_C8)
CANONICAL_M = 8
BCRD_SERIAL_AUDIT_STATUS = "NOT_EXECUTED_NOT_PART_OF_D10_GATE"
EXPERIMENT_BOUNDARY = (
    "d10_specific_frozen_roster_teacher_forced_single_gpu_mechanism_cost_"
    "not_bcrd_formal_producer_not_serving_not_vllm_batch_invariance"
)
REQUIRED_ARTIFACTS = (
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


class ProtocolError(RuntimeError):
    """A violation that makes the result uninterpretable."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class JsonlWriter:
    """Exclusive, durable JSONL sink for raw evidence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("x", encoding="utf-8")

    def write(self, row: Mapping[str, Any]) -> None:
        self._stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        self._stream.write("\n")

    def checkpoint(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._stream.closed:
            self.checkpoint()
            self._stream.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def percentile(values: Sequence[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ProtocolError("percentile requires values and q in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


@dataclass(frozen=True)
class LogicalRouteRow:
    batch_index: int
    layer: int
    expert_id: int
    token_index: int
    topk_rank: int
    slot_id: str
    row_id: str


@dataclass(frozen=True)
class ExpertCallSpec:
    arm: str
    batch_index: int
    layer: int
    expert_id: int
    call_ordinal: int
    rows: tuple[LogicalRouteRow, ...]
    physical_m: int
    padding_rows: int


def plan_expert_calls(
    rows: Iterable[LogicalRouteRow],
    arm: str,
    *,
    canonical_m: int = CANONICAL_M,
) -> tuple[ExpertCallSpec, ...]:
    """Plan calls without ever coalescing rows from different decode epochs."""

    records = tuple(rows)
    if arm not in ARMS:
        raise ProtocolError(f"unknown arm {arm!r}")
    if int(canonical_m) != CANONICAL_M:
        raise ProtocolError(f"canonical M is frozen at {CANONICAL_M}")
    if not records:
        return ()
    slot_ids = [row.slot_id for row in records]
    row_ids = [row.row_id for row in records]
    if len(slot_ids) != len(set(slot_ids)) or len(row_ids) != len(set(row_ids)):
        raise ProtocolError("logical route rows contain duplicate identities")

    grouped: dict[tuple[int, int, int], list[LogicalRouteRow]] = {}
    for row in records:
        key = (int(row.batch_index), int(row.layer), int(row.expert_id))
        grouped.setdefault(key, []).append(row)

    calls: list[ExpertCallSpec] = []
    for (batch_index, layer, expert_id), group_values in sorted(grouped.items()):
        group = tuple(
            sorted(
                group_values,
                key=lambda row: (row.token_index, row.topk_rank, row.slot_id),
            )
        )
        logical_m = len(group)
        if logical_m > canonical_m:
            raise ProtocolError(
                f"natural M={logical_m} exceeds frozen C={canonical_m} at "
                f"batch={batch_index}, layer={layer}, expert={expert_id}"
            )
        if arm == ARM_SERIAL:
            for ordinal, row in enumerate(group):
                calls.append(
                    ExpertCallSpec(
                        arm=arm,
                        batch_index=batch_index,
                        layer=layer,
                        expert_id=expert_id,
                        call_ordinal=ordinal,
                        rows=(row,),
                        physical_m=1,
                        padding_rows=0,
                    )
                )
        else:
            physical_m = logical_m if arm == ARM_NATIVE else canonical_m
            calls.append(
                ExpertCallSpec(
                    arm=arm,
                    batch_index=batch_index,
                    layer=layer,
                    expert_id=expert_id,
                    call_ordinal=0,
                    rows=group,
                    physical_m=physical_m,
                    padding_rows=physical_m - logical_m,
                )
            )

    observed = [row.slot_id for call in calls for row in call.rows]
    if len(observed) != len(set(observed)) or set(observed) != set(slot_ids):
        raise ProtocolError("call plan does not conserve every logical route slot once")
    return tuple(calls)


@dataclass
class PolicyExecution:
    output: Any
    physical_m: int
    padding_rows: int
    kernel_calls: int


def execute_expert_policy(
    expert: Any,
    current_state: Any,
    arm: str,
    *,
    canonical_m: int = CANONICAL_M,
) -> PolicyExecution:
    """Execute one non-empty expert group according to one frozen policy."""

    if arm not in ARMS:
        raise ProtocolError(f"unknown arm {arm!r}")
    if int(canonical_m) != CANONICAL_M:
        raise ProtocolError(f"canonical M is frozen at {CANONICAL_M}")
    if getattr(current_state, "ndim", None) != 2:
        raise ProtocolError("expert input must be a rank-two tensor")
    logical_m = int(current_state.shape[0])
    hidden_size = int(current_state.shape[1])
    if logical_m <= 0:
        raise ProtocolError("empty experts must be skipped, not executed")
    if logical_m > canonical_m:
        raise ProtocolError(
            f"natural M={logical_m} exceeds frozen C={canonical_m}; refusing to split"
        )

    if arm == ARM_NATIVE:
        output = expert(current_state)
        physical_m = logical_m
        padding_rows = 0
        kernel_calls = 1
    elif arm == ARM_SERIAL:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - GPU environment concern
            raise ProtocolError("serial M1 execution requires PyTorch") from exc
        pieces = [expert(current_state[index : index + 1]) for index in range(logical_m)]
        output = torch.cat(pieces, dim=0)
        physical_m = 1
        padding_rows = 0
        kernel_calls = logical_m
    else:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - GPU environment concern
            raise ProtocolError("fixed C8 execution requires PyTorch") from exc
        padding_rows = canonical_m - logical_m
        padded = current_state
        if padding_rows:
            padding = current_state.new_zeros((padding_rows, hidden_size))
            padded = torch.cat((current_state, padding), dim=0)
        physical_output = expert(padded)
        expected_physical = (canonical_m, hidden_size)
        if tuple(physical_output.shape) != expected_physical:
            raise ProtocolError(
                f"fixed C8 expert returned {tuple(physical_output.shape)}, "
                f"expected {expected_physical}"
            )
        output = physical_output[:logical_m]
        physical_m = canonical_m
        kernel_calls = 1

    if tuple(output.shape) != (logical_m, hidden_size):
        raise ProtocolError("expert policy changed the logical output shape")
    if getattr(output, "dtype", None) != getattr(current_state, "dtype", None):
        raise ProtocolError("expert policy changed the logical output dtype")
    return PolicyExecution(
        output=output,
        physical_m=physical_m,
        padding_rows=padding_rows,
        kernel_calls=kernel_calls,
    )


def validate_roster_conservation(
    roster: Sequence[Mapping[str, Any]],
    *,
    expected_request_ids: Sequence[str],
    expected_steps_per_request: int,
    max_batch_size: int,
) -> dict[str, Any]:
    """Require one immutable, complete request-step DAG."""

    expected_ids = tuple(str(value) for value in expected_request_ids)
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ProtocolError("expected request IDs must be non-empty and unique")
    if expected_steps_per_request <= 0 or max_batch_size <= 0:
        raise ProtocolError("roster dimensions must be positive")
    observed: dict[str, list[tuple[int, int, int, int, int]]] = {
        request_id: [] for request_id in expected_ids
    }
    admission_by_request: dict[str, dict[str, Any]] = {}
    pairs: set[tuple[str, int]] = set()
    batch_histogram: dict[str, int] = {}
    for expected_batch_index, row in enumerate(roster):
        if int(row.get("batch_index", -1)) != expected_batch_index:
            raise ProtocolError("roster batch indexes are not contiguous")
        request_ids = [str(value) for value in row.get("request_ids", [])]
        decode_steps = [int(value) for value in row.get("decode_steps", [])]
        input_tokens = [int(value) for value in row.get("input_token_ids", [])]
        predicted_tokens = [
            int(value) for value in row.get("native_predicted_next_token_ids", [])
        ]
        position_ids = [int(value) for value in row.get("position_ids", [])]
        prior_lengths = [int(value) for value in row.get("prior_cache_lengths", [])]
        left_padding = [int(value) for value in row.get("left_padding", [])]
        admission = list(row.get("admission_identity", []))
        width = len(request_ids)
        if not 0 < width <= max_batch_size:
            raise ProtocolError("roster batch size violates the frozen maximum")
        if len(set(request_ids)) != width:
            raise ProtocolError("one roster batch duplicates a request")
        if not all(
            len(values) == width
            for values in (
                decode_steps,
                input_tokens,
                predicted_tokens,
                position_ids,
                prior_lengths,
                left_padding,
                admission,
            )
        ):
            raise ProtocolError("roster batch columns have different lengths")
        if int(row.get("batch_size", -1)) != width:
            raise ProtocolError("roster batch_size disagrees with request identities")
        if int(row.get("pending_request_count", -1)) < 0:
            raise ProtocolError("roster pending request count is invalid")
        maximum_prior = max(prior_lengths)
        active_ids = {str(value) for value in row.get("active_request_ids", [])}
        if not set(request_ids).issubset(active_ids):
            raise ProtocolError("roster batch is not a subset of its frozen active set")
        for index, (request_id, step) in enumerate(zip(request_ids, decode_steps)):
            if request_id not in observed:
                raise ProtocolError(f"roster contains unknown request {request_id}")
            pair = (request_id, step)
            if pair in pairs:
                raise ProtocolError(f"duplicate roster request-step {pair}")
            pairs.add(pair)
            if position_ids[index] != prior_lengths[index]:
                raise ProtocolError("roster position ID and KV length disagree")
            if left_padding[index] != maximum_prior - prior_lengths[index]:
                raise ProtocolError("roster left padding and KV lengths disagree")
            identity = admission[index]
            if not isinstance(identity, Mapping) or str(identity.get("request_id")) != request_id:
                raise ProtocolError("roster admission identity lost request alignment")
            normalized_identity = {
                "request_id": request_id,
                "arrival_us": float(identity.get("arrival_us")),
                "sample_id": int(identity.get("sample_id")),
            }
            previous_identity = admission_by_request.setdefault(
                request_id, normalized_identity
            )
            if previous_identity != normalized_identity:
                raise ProtocolError("roster admission identity drifted across steps")
            observed[request_id].append(
                (
                    step,
                    input_tokens[index],
                    predicted_tokens[index],
                    prior_lengths[index],
                    expected_batch_index,
                )
            )
        for hash_field in (
            "native_route_membership_sha256",
            "native_final_logits_sha256",
        ):
            digest = str(row.get(hash_field, ""))
            if len(digest) != 64:
                raise ProtocolError(f"roster {hash_field} is not sealed")
        key = str(width)
        batch_histogram[key] = batch_histogram.get(key, 0) + 1

    frozen_steps = list(range(expected_steps_per_request))
    for request_id, records in observed.items():
        records.sort(key=lambda value: value[0])
        steps = [value[0] for value in records]
        if steps != frozen_steps:
            raise ProtocolError(
                f"request {request_id} roster steps {steps!r} do not close"
            )
        initial_prior = records[0][3]
        if initial_prior <= 0:
            raise ProtocolError("roster starts with an invalid KV length")
        for index, record in enumerate(records):
            if record[3] != initial_prior + index:
                raise ProtocolError(f"request {request_id} KV length chain drifted")
            if index and records[index - 1][2] != record[1]:
                raise ProtocolError(f"request {request_id} frozen token chain drifted")
    expected_total = len(expected_ids) * expected_steps_per_request
    if len(pairs) != expected_total:
        raise ProtocolError("roster request-step denominator is incomplete")
    return {
        "status": "PASS",
        "requests": len(expected_ids),
        "request_steps": len(pairs),
        "decode_batches": len(roster),
        "batch_size_histogram": batch_histogram,
        "maximum_batch_size": max((len(row["request_ids"]) for row in roster), default=0),
        "roster_sha256": canonical_sha256(list(roster)),
    }


def compare_policy_repeats(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two replays of the same policy; no cross-policy oracle is used."""

    arm = str(first.get("arm"))
    if arm not in ARMS or str(second.get("arm")) != arm:
        raise ProtocolError("policy repeat comparison cannot cross arms")
    left = {int(row["batch_index"]): row for row in first.get("step_signatures", [])}
    right = {int(row["batch_index"]): row for row in second.get("step_signatures", [])}
    if len(left) != len(first.get("step_signatures", [])) or len(right) != len(
        second.get("step_signatures", [])
    ):
        raise ProtocolError("policy replay duplicates a batch identity")
    if set(left) != set(right):
        raise ProtocolError("policy repeats do not cover the same batch identities")
    raw_mismatches = 0
    route_mismatches = 0
    final_mismatches = 0
    greedy_mismatches = 0
    for batch_index in sorted(left):
        lhs = left[batch_index]
        rhs = right[batch_index]
        if lhs.get("roster_identity_sha256") != rhs.get("roster_identity_sha256"):
            raise ProtocolError("policy repeat changed the frozen roster identity")
        raw_mismatches += lhs.get("raw_calls_sha256") != rhs.get("raw_calls_sha256")
        route_mismatches += lhs.get("route_membership_sha256") != rhs.get(
            "route_membership_sha256"
        )
        final_mismatches += lhs.get("final_logits_sha256") != rhs.get(
            "final_logits_sha256"
        )
        left_tokens = list(lhs.get("greedy_token_ids", []))
        right_tokens = list(rhs.get("greedy_token_ids", []))
        if len(left_tokens) != len(right_tokens):
            raise ProtocolError("policy repeat greedy-token widths differ")
        greedy_mismatches += sum(a != b for a, b in zip(left_tokens, right_tokens))
    return {
        "arm": arm,
        "comparison_scope": "within_policy_repeat0_vs_repeat1",
        "m1_is_ground_truth": False,
        "batches": len(left),
        "raw_repeat_mismatches": int(raw_mismatches),
        "route_repeat_mismatches": int(route_mismatches),
        "final_logits_repeat_mismatches": int(final_mismatches),
        "greedy_token_repeat_mismatches": int(greedy_mismatches),
        "bitwise_repeat_stable": not (
            raw_mismatches or route_mismatches or final_mismatches
        ),
    }


def classify_gate(
    policy_correctness: Mapping[str, Mapping[str, Any]],
    arm_metrics: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    if set(policy_correctness) != set(ARMS) or set(arm_metrics) != set(ARMS):
        raise ProtocolError("gate requires all three frozen arms")
    fixed = policy_correctness[ARM_C8]
    required_raw = int(gate["require_fixed_c8_raw_repeat_mismatches"])
    required_route = int(gate["require_fixed_c8_route_repeat_mismatches"])
    required_final = int(gate["require_fixed_c8_final_logits_repeat_mismatches"])
    correctness_failures: list[str] = []
    if int(fixed["raw_repeat_mismatches"]) != required_raw:
        correctness_failures.append("fixed_c8_raw_repeat_mismatch")
    if int(fixed["route_repeat_mismatches"]) != required_route:
        correctness_failures.append("fixed_c8_route_repeat_mismatch")
    if int(fixed["final_logits_repeat_mismatches"]) != required_final:
        correctness_failures.append("fixed_c8_final_logits_repeat_mismatch")

    serial_expert = float(arm_metrics[ARM_SERIAL]["median_total_expert_gpu_ms"])
    c8_expert = float(arm_metrics[ARM_C8]["median_total_expert_gpu_ms"])
    native_p99 = float(arm_metrics[ARM_NATIVE]["median_token_step_p99_ms"])
    c8_p99 = float(arm_metrics[ARM_C8]["median_token_step_p99_ms"])
    if min(serial_expert, c8_expert, native_p99, c8_p99) <= 0 or not all(
        math.isfinite(value)
        for value in (serial_expert, c8_expert, native_p99, c8_p99)
    ):
        raise ProtocolError("gate timing denominators must be finite and positive")
    expert_ratio = c8_expert / serial_expert
    p99_ratio = c8_p99 / native_p99
    cost_failures: list[str] = []
    if expert_ratio > float(gate["maximum_c8_over_serial_expert_gpu_time_ratio"]):
        cost_failures.append("fixed_c8_expert_gpu_time_over_serial_threshold")
    if p99_ratio > float(gate["maximum_c8_over_native_token_step_p99_ratio"]):
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
        "expert_ratio_threshold": float(
            gate["maximum_c8_over_serial_expert_gpu_time_ratio"]
        ),
        "p99_ratio_threshold": float(
            gate["maximum_c8_over_native_token_step_p99_ratio"]
        ),
        "m1_is_ground_truth": False,
    }


def build_manifest(
    output_dir: Path, required_names: Sequence[str] = REQUIRED_ARTIFACTS
) -> dict[str, Any]:
    missing = [name for name in required_names if not (output_dir / name).is_file()]
    if missing:
        raise ProtocolError(f"manifest required artifacts are missing: {missing}")
    excluded = {"MANIFEST.json", "RUN_STATUS.json", "COMPLETE.json", "FAILURE.json"}
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name not in excluded:
            files[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": "stablebatch-shape-lane-cost-manifest-v1",
        "created_at": utc_now(),
        "required_artifacts": list(required_names),
        "files": files,
    }


def verify_manifest(
    output_dir: Path,
    manifest: Mapping[str, Any],
    required_names: Sequence[str] = REQUIRED_ARTIFACTS,
) -> None:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ProtocolError("manifest files entry is not an object")
    if set(required_names) - set(files):
        raise ProtocolError("manifest does not bind every required artifact")
    for name, metadata in files.items():
        path = output_dir / str(name)
        if not path.is_file():
            raise ProtocolError(f"manifest file disappeared: {name}")
        if int(metadata["size_bytes"]) != path.stat().st_size:
            raise ProtocolError(f"manifest size mismatch: {name}")
        if str(metadata["sha256"]) != sha256_file(path):
            raise ProtocolError(f"manifest hash mismatch: {name}")


def _load_module(name: str, path: Path) -> Any:
    cached = sys.modules.get(name)
    if cached is not None:
        cached_path = Path(str(getattr(cached, "__file__", ""))).resolve()
        if cached_path != path.resolve():
            raise ProtocolError(f"module name {name!r} is already bound to {cached_path}")
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot import helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_helpers(repo_root: Path) -> tuple[Any, Any]:
    bcrd_dir = repo_root / "docs/ideas/bcrd/experiments"
    if str(bcrd_dir) not in sys.path:
        sys.path.insert(0, str(bcrd_dir))
    core_path = bcrd_dir / "core.py"
    existing_core = sys.modules.get("core")
    if existing_core is None:
        _load_module("core", core_path)
    elif Path(str(getattr(existing_core, "__file__", ""))).resolve() != core_path.resolve():
        raise ProtocolError("the generic module name 'core' is bound to the wrong file")
    capture = _load_module(
        "d10_capture_continuous_decode", bcrd_dir / "capture_continuous_decode.py"
    )
    stable = _load_module(
        "d10_single_contribution_helper",
        repo_root
        / "docs/ideas/stablebatch/experiments/run_single_contribution_pilot.py",
    )
    return capture, stable


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA or config.get("status") != "FROZEN_PRE_RUN":
        raise ProtocolError("config is not the frozen D10 v1 protocol")
    execution = config.get("execution")
    if not isinstance(execution, Mapping):
        raise ProtocolError("config.execution is missing")
    if tuple(execution.get("arms", [])) != ARMS:
        raise ProtocolError("the three execution arms drifted")
    if int(execution.get("canonical_m", -1)) != CANONICAL_M:
        raise ProtocolError("C is frozen at 8")
    if int(execution.get("warmup_replays_per_arm", -1)) != 1:
        raise ProtocolError("warmup count drifted")
    if int(execution.get("measured_replays_per_arm", -1)) != 2:
        raise ProtocolError("measured replay count drifted")
    if tuple(execution.get("warmup_arm_order", [])) != ARMS:
        raise ProtocolError("warmup arm order drifted")
    measured_orders = tuple(tuple(row) for row in execution.get("measured_arm_orders", []))
    if measured_orders != (
        (ARM_NATIVE, ARM_SERIAL, ARM_C8),
        (ARM_SERIAL, ARM_C8, ARM_NATIVE),
    ):
        raise ProtocolError("measured arm orders drifted")
    if not bool(execution.get("immediate_flush")) or bool(
        execution.get("cross_step_wait")
    ):
        raise ProtocolError("D10 must immediate-flush and never wait across steps")
    if int(execution.get("maximum_wall_seconds", 0)) != 2700:
        raise ProtocolError("wall-time budget drifted")
    if execution.get("native_roster_serial_equivalence_audit") != (
        BCRD_SERIAL_AUDIT_STATUS
    ):
        raise ProtocolError("BCRD serial-audit claim boundary drifted")
    if config.get("official_batch_invariance", {}).get("status") != "NOT_EXECUTABLE":
        raise ProtocolError("this runner must not claim an official BI arm")


def verify_static_bindings(
    config: Mapping[str, Any], repo_root: Path, config_path: Path, runner_path: Path
) -> dict[str, Any]:
    validate_frozen_config(config)
    git_metadata_present = (repo_root / ".git").exists()
    observed_sources: dict[str, str] = {}
    for relative, expected in config["source_bindings"].items():
        path = repo_root / str(relative)
        if not path.is_file():
            raise ProtocolError(f"bound source is missing: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ProtocolError(f"bound source hash drifted: {relative} -> {observed}")
        observed_sources[str(relative)] = observed
    plan_cfg = config["plan"]
    plan_path = repo_root / str(plan_cfg["path"])
    if not plan_path.is_file() or sha256_file(plan_path) != plan_cfg["sha256"]:
        raise ProtocolError("frozen experiment plan path/hash mismatch")
    workload_cfg = config["workload"]
    workload_path = repo_root / str(workload_cfg["path"])
    if not workload_path.is_file() or sha256_file(workload_path) != workload_cfg["sha256"]:
        raise ProtocolError("frozen workload path/hash mismatch")
    model_root = Path(str(config["model"]["local_path"])).resolve()
    observed_model: dict[str, str] = {}
    for relative, expected in config["model"]["file_sha256"].items():
        path = model_root / str(relative)
        if not path.is_file():
            raise ProtocolError(f"model file is missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ProtocolError(f"model file hash drifted: {relative}")
        observed_model[str(relative)] = observed
    return {
        "schema_version": "stablebatch-shape-lane-static-bindings-v1",
        "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "workload_path": str(workload_path),
        "workload_sha256": sha256_file(workload_path),
        "source_sha256": observed_sources,
        "model_path": str(model_root),
        "model_file_sha256": observed_model,
        "experiment_boundary": EXPERIMENT_BOUNDARY,
        "bcrd_serial_audit": BCRD_SERIAL_AUDIT_STATUS,
        "git_provenance": (
            "AVAILABLE_BUT_NOT_USED_AS_AUTHORITY"
            if git_metadata_present
            else "UNAVAILABLE_CONTENT_HASH_BOUND_ONLY"
        ),
    }


@dataclass
class Deadline:
    maximum_seconds: float
    started_monotonic: float = field(default_factory=time.monotonic)

    def check(self, where: str) -> None:
        elapsed = time.monotonic() - self.started_monotonic
        if elapsed > self.maximum_seconds:
            raise TimeoutError(
                f"D10 exceeded {self.maximum_seconds:.0f}s at {where} ({elapsed:.1f}s)"
            )


@dataclass
class DecodeState:
    spec: Any
    cache: Any
    attention_mask: Any
    next_token: Any
    prompt_length: int
    decode_step: int = 0


def _tensor_storage_sha256(tensor: Any, *, require_bf16: bool = False) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - GPU environment concern
        raise ProtocolError("tensor hashing requires PyTorch") from exc
    value = tensor.detach().contiguous()
    if require_bf16 and value.dtype != torch.bfloat16:
        raise ProtocolError(f"expected BF16 storage, observed {value.dtype}")
    raw = value.view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _route_entries(
    route_batches: Sequence[Mapping[str, Any]],
    request_ids: Sequence[str],
    decode_steps: Sequence[int],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for batch in route_batches:
        layer = int(batch["layer"])
        selected = batch["selected_experts"]
        if int(selected.shape[0]) != len(request_ids):
            raise ProtocolError("router output lost the frozen batch width")
        for token_index, (request_id, step) in enumerate(
            zip(request_ids, decode_steps)
        ):
            experts = [int(value) for value in selected[token_index].tolist()]
            if len(experts) != len(set(experts)):
                raise ProtocolError("top-k returned a duplicate expert for one token")
            for rank, expert_id in enumerate(experts):
                slot_id = f"{request_id}:decode:{step:06d}:layer:{layer:02d}:topk:{rank}"
                entries.append(
                    {
                        "layer": layer,
                        "token_index": token_index,
                        "request_id": request_id,
                        "decode_step": int(step),
                        "topk_rank": rank,
                        "expert_id": expert_id,
                        "slot_id": slot_id,
                        "row_id": f"{slot_id}:expert:{expert_id:02d}",
                    }
                )
    entries.sort(key=lambda row: (row["layer"], row["token_index"], row["topk_rank"]))
    return entries


@dataclass
class RosterCapture:
    prefill_rows: list[dict[str, Any]]
    roster_rows: list[dict[str, Any]]


def capture_native_roster(
    model: Any,
    requests: Sequence[Any],
    *,
    capture_helper: Any,
    max_decode_steps: int,
    max_batch_size: int,
    deadline: Deadline,
) -> RosterCapture:
    """Capture the only runtime-derived schedule; EOS never changes its denominator."""

    import torch

    ordered = sorted(requests, key=lambda item: (item.arrival_us, item.request_id))
    if not ordered:
        raise ProtocolError("cannot capture an empty request roster")
    pending = list(ordered)
    active: list[DecodeState] = []
    clock_us = float(pending[0].arrival_us)
    prefill_rows: list[dict[str, Any]] = []
    roster: list[dict[str, Any]] = []

    while pending or active:
        deadline.check("native roster capture")
        if not active and pending and float(pending[0].arrival_us) > clock_us:
            clock_us = float(pending[0].arrival_us)
        while pending and float(pending[0].arrival_us) <= clock_us:
            deadline.check("native roster prefill")
            spec = pending.pop(0)
            start_us = clock_us
            with torch.inference_mode():
                output, elapsed_us = capture_helper._timed_call(
                    model,
                    "d10_native_roster_prefill",
                    1,
                    None,
                    input_ids=spec.input_ids,
                    attention_mask=spec.attention_mask,
                    use_cache=True,
                    output_router_logits=False,
                    return_dict=True,
                )
            clock_us += float(elapsed_us)
            cache = output.past_key_values
            logits = output.logits
            prompt_length = int(spec.input_ids.shape[1])
            if cache is None or capture_helper._cache_length(cache) != prompt_length:
                raise ProtocolError(f"native roster prefill cache failed for {spec.request_id}")
            if logits is None or not bool(torch.isfinite(logits).all().item()):
                raise ProtocolError(f"native roster prefill logits failed for {spec.request_id}")
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            active.append(
                DecodeState(
                    spec=spec,
                    cache=cache,
                    attention_mask=spec.attention_mask,
                    next_token=next_token,
                    prompt_length=prompt_length,
                )
            )
            prefill_rows.append(
                {
                    "request_id": spec.request_id,
                    "arrival_us": float(spec.arrival_us),
                    "start_us": start_us,
                    "end_us": clock_us,
                    "wall_ms": float(elapsed_us) / 1000.0,
                    "prompt_tokens": prompt_length,
                    "first_decode_input_token_id": int(next_token.item()),
                }
            )

        if not active:
            continue
        active.sort(key=lambda item: (item.spec.arrival_us, item.spec.request_id))
        batch = active[:max_batch_size]
        input_ids, attention_mask, position_ids, cache, lengths, prior_max = (
            capture_helper._pad_decode_inputs(batch)
        )
        request_ids = [state.spec.request_id for state in batch]
        decode_steps = [int(state.decode_step) for state in batch]
        input_token_ids = [int(state.next_token.item()) for state in batch]
        start_us = clock_us
        with torch.inference_mode():
            output, elapsed_us = capture_helper._timed_call(
                model,
                "d10_native_roster_decode",
                len(batch),
                None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=torch.tensor(
                    [prior_max], dtype=torch.long, device=input_ids.device
                ),
                past_key_values=cache,
                use_cache=True,
                output_router_logits=True,
                return_dict=True,
            )
        clock_us += float(elapsed_us)
        logits = output.logits
        output_cache = output.past_key_values
        if logits is None or output_cache is None:
            raise ProtocolError("native roster decode returned no logits/cache")
        if not bool(torch.isfinite(logits).all().item()):
            raise ProtocolError("native roster decode produced NaN/Inf logits")
        split_caches = capture_helper.split_left_padded_cache(
            output_cache, prior_lengths=lengths, prior_max_length=prior_max
        )
        predicted = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        route_batches = capture_helper._native_route_batches(
            output, expected_rows=len(batch), config=model.config
        )
        routes = _route_entries(route_batches, request_ids, decode_steps)
        roster.append(
            {
                "schema_version": "stablebatch-shape-lane-native-roster-row-v1",
                "batch_index": len(roster),
                "start_us": start_us,
                "end_us": clock_us,
                "native_capture_wall_ms": float(elapsed_us) / 1000.0,
                "batch_size": len(batch),
                "active_request_ids": [state.spec.request_id for state in active],
                "pending_request_count": len(pending),
                "request_ids": request_ids,
                "decode_steps": decode_steps,
                "input_token_ids": input_token_ids,
                "position_ids": [int(value) for value in position_ids[:, 0].tolist()],
                "prior_cache_lengths": [int(value) for value in lengths],
                "left_padding": [prior_max - int(value) for value in lengths],
                "native_predicted_next_token_ids": [
                    int(value) for value in predicted[:, 0].tolist()
                ],
                "native_route_membership_sha256": canonical_sha256(routes),
                "native_final_logits_sha256": _tensor_storage_sha256(
                    logits[:, -1, :], require_bf16=True
                ),
                "admission_identity": [
                    {
                        "request_id": state.spec.request_id,
                        "arrival_us": float(state.spec.arrival_us),
                        "sample_id": int(state.spec.sample_id),
                    }
                    for state in batch
                ],
            }
        )
        for index, state in enumerate(batch):
            state.cache = split_caches[index]
            state.attention_mask = torch.cat(
                (state.attention_mask, state.attention_mask.new_ones((1, 1))), dim=1
            )
            state.decode_step += 1
            state.next_token = predicted[index : index + 1]
            if state.decode_step >= max_decode_steps:
                active.remove(state)

    return RosterCapture(prefill_rows=prefill_rows, roster_rows=roster)


@dataclass
class _PendingExpertCall:
    layer: int
    expert_id: int
    top_x: Any
    topk_rank: Any
    raw_output: Any
    logical_m: int
    physical_m: int
    padding_rows: int
    kernel_calls: int


@dataclass
class _PendingLayer:
    layer: int
    start_event: Any
    end_event: Any
    calls: list[_PendingExpertCall]


class MoEPolicyController:
    """Live OLMoE decode-only policy patch and per-step CUDA ledger."""

    def __init__(self, model: Any, arm: str, canonical_m: int) -> None:
        if arm not in ARMS:
            raise ProtocolError(f"unknown arm {arm}")
        self.model = model
        self.arm = arm
        self.canonical_m = canonical_m
        self.num_layers = int(model.config.num_hidden_layers)
        self._context: dict[str, Any] | None = None
        self._layers: list[_PendingLayer] = []

    def begin_step(
        self,
        *,
        batch_index: int,
        request_ids: Sequence[str],
        decode_steps: Sequence[int],
    ) -> None:
        if self._context is not None or self._layers:
            raise ProtocolError("MoE policy context leaked across decode epochs")
        if len(request_ids) != len(decode_steps) or not request_ids:
            raise ProtocolError("MoE policy received an invalid decode batch identity")
        self._context = {
            "batch_index": int(batch_index),
            "request_ids": tuple(str(value) for value in request_ids),
            "decode_steps": tuple(int(value) for value in decode_steps),
        }

    def forward(self, block: Any, layer: int, hidden_states: Any) -> tuple[Any, Any]:
        import torch
        import torch.nn.functional as functional

        if self._context is None:
            raise ProtocolError("patched MoE executed outside a decode-step context")
        if any(item.layer == layer for item in self._layers):
            raise ProtocolError("one decode epoch invoked the same MoE layer twice")
        if hidden_states.ndim != 3:
            raise ProtocolError("OLMoE hidden states must be rank three")
        batch_size, sequence_length, hidden_dim = map(int, hidden_states.shape)
        if sequence_length != 1:
            raise ProtocolError("D10 patch is decode-only and refuses prompt/prefill rows")
        if batch_size != len(self._context["request_ids"]):
            raise ProtocolError("MoE hidden batch no longer matches the frozen roster")
        flat = hidden_states.view(-1, hidden_dim)
        router_logits = block.gate(flat)
        routing_weights = functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, block.top_k, dim=-1
        )
        if block.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        final = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=flat.dtype,
            device=flat.device,
        )
        expert_mask = functional.one_hot(
            selected_experts, num_classes=block.num_experts
        ).permute(2, 1, 0)
        pending_calls: list[_PendingExpertCall] = []
        for expert_id in range(int(block.num_experts)):
            topk_rank, top_x = torch.where(expert_mask[expert_id])
            logical_m = int(top_x.numel())
            if logical_m == 0:
                continue
            if logical_m > self.canonical_m:
                raise ProtocolError(
                    f"natural M={logical_m} exceeds C={self.canonical_m} in "
                    f"batch={self._context['batch_index']}, layer={layer}, "
                    f"expert={expert_id}"
                )
            current_state = flat[None, top_x].reshape(-1, hidden_dim)
            execution = execute_expert_policy(
                block.experts[expert_id],
                current_state,
                self.arm,
                canonical_m=self.canonical_m,
            )
            weighted = execution.output * routing_weights[top_x, topk_rank, None]
            final.index_add_(0, top_x, weighted.to(flat.dtype))
            pending_calls.append(
                _PendingExpertCall(
                    layer=layer,
                    expert_id=expert_id,
                    top_x=top_x.detach(),
                    topk_rank=topk_rank.detach(),
                    raw_output=execution.output.detach(),
                    logical_m=logical_m,
                    physical_m=execution.physical_m,
                    padding_rows=execution.padding_rows,
                    kernel_calls=execution.kernel_calls,
                )
            )
        final = final.reshape(batch_size, sequence_length, hidden_dim)
        end_event.record()
        self._layers.append(
            _PendingLayer(
                layer=layer,
                start_event=start_event,
                end_event=end_event,
                calls=pending_calls,
            )
        )
        return final, router_logits

    def finish_step(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        import torch

        if self._context is None:
            raise ProtocolError("cannot finish an inactive MoE decode context")
        if {item.layer for item in self._layers} != set(range(self.num_layers)):
            raise ProtocolError("decode step did not execute every frozen MoE layer once")
        torch.cuda.synchronize(self.model.device)
        context = self._context
        call_rows: list[dict[str, Any]] = []
        layer_rows: list[dict[str, Any]] = []
        route_entries: list[dict[str, Any]] = []
        for pending_layer in sorted(self._layers, key=lambda item: item.layer):
            layer_ms = float(
                pending_layer.start_event.elapsed_time(pending_layer.end_event)
            )
            if not math.isfinite(layer_ms) or layer_ms <= 0:
                raise ProtocolError("expert-stage CUDA event returned invalid time")
            layer_rows.append(
                {
                    "batch_index": context["batch_index"],
                    "layer": pending_layer.layer,
                    "expert_stage_gpu_ms": layer_ms,
                    "occupied_experts": len(pending_layer.calls),
                    "timing_boundary": (
                        "after_router_topk_through_dispatch_padding_expert_and_index_add"
                    ),
                }
            )
            layer_slots: list[str] = []
            for pending in pending_layer.calls:
                if not bool(torch.isfinite(pending.raw_output).all().item()):
                    raise ProtocolError("expert execution produced NaN/Inf")
                token_indexes = [int(value) for value in pending.top_x.cpu().tolist()]
                ranks = [int(value) for value in pending.topk_rank.cpu().tolist()]
                if len(token_indexes) != pending.logical_m or len(ranks) != pending.logical_m:
                    raise ProtocolError("expert call lost a routed row")
                slot_ids: list[str] = []
                row_ids: list[str] = []
                raw_rows: list[dict[str, str]] = []
                for row_offset, (token_index, rank) in enumerate(
                    zip(token_indexes, ranks)
                ):
                    request_id = context["request_ids"][token_index]
                    decode_step = context["decode_steps"][token_index]
                    slot_id = (
                        f"{request_id}:decode:{decode_step:06d}:"
                        f"layer:{pending.layer:02d}:topk:{rank}"
                    )
                    row_id = f"{slot_id}:expert:{pending.expert_id:02d}"
                    slot_ids.append(slot_id)
                    row_ids.append(row_id)
                    raw_rows.append(
                        {
                            "slot_id": slot_id,
                            "row_id": row_id,
                            "sha256": _tensor_storage_sha256(
                                pending.raw_output[row_offset], require_bf16=True
                            ),
                        }
                    )
                    route_entries.append(
                        {
                            "layer": pending.layer,
                            "token_index": token_index,
                            "request_id": request_id,
                            "decode_step": decode_step,
                            "topk_rank": rank,
                            "expert_id": pending.expert_id,
                            "slot_id": slot_id,
                            "row_id": row_id,
                        }
                    )
                layer_slots.extend(slot_ids)
                call_rows.append(
                    {
                        "batch_index": context["batch_index"],
                        "layer": pending.layer,
                        "expert_id": pending.expert_id,
                        "logical_m": pending.logical_m,
                        "physical_m": pending.physical_m,
                        "physical_m_per_kernel": [pending.physical_m]
                        * pending.kernel_calls,
                        "padding_rows": pending.padding_rows,
                        "kernel_calls": pending.kernel_calls,
                        "slot_ids": slot_ids,
                        "row_ids": row_ids,
                        "route_membership_sha256": canonical_sha256(row_ids),
                        "raw_bf16_sha256": _tensor_storage_sha256(
                            pending.raw_output, require_bf16=True
                        ),
                        "raw_row_sha256": raw_rows,
                    }
                )
            expected_slots = len(context["request_ids"]) * int(self.model.config.num_experts_per_tok)
            if len(layer_slots) != expected_slots or len(set(layer_slots)) != expected_slots:
                raise ProtocolError("one layer did not conserve every route slot exactly once")

        route_entries.sort(
            key=lambda row: (row["layer"], row["token_index"], row["topk_rank"])
        )
        by_layer_token: dict[tuple[int, int], list[int]] = {}
        for row in route_entries:
            key = (int(row["layer"]), int(row["token_index"]))
            by_layer_token.setdefault(key, []).append(int(row["expert_id"]))
        expected_top_k = int(self.model.config.num_experts_per_tok)
        for key, expert_ids in by_layer_token.items():
            if len(expert_ids) != expected_top_k or len(set(expert_ids)) != expected_top_k:
                raise ProtocolError(
                    f"top-k expert identity is not closed and unique for {key}"
                )
        expected_total = (
            len(context["request_ids"])
            * self.num_layers
            * int(self.model.config.num_experts_per_tok)
        )
        slot_ids = [str(row["slot_id"]) for row in route_entries]
        if len(slot_ids) != expected_total or len(set(slot_ids)) != expected_total:
            raise ProtocolError("decode epoch route-slot conservation failed")
        route_hash = canonical_sha256(route_entries)
        self._context = None
        self._layers = []
        return call_rows, layer_rows, route_hash

    def abort_step(self) -> None:
        self._context = None
        self._layers = []


@contextlib.contextmanager
def patched_moe_policy(model: Any, arm: str, canonical_m: int) -> Iterable[MoEPolicyController]:
    layers = tuple(model.model.layers)
    controller = MoEPolicyController(model, arm, canonical_m)
    originals: list[tuple[Any, Any]] = []
    try:
        for layer_index, layer in enumerate(layers):
            block = layer.mlp
            if not hasattr(block, "gate") or not hasattr(block, "experts"):
                raise ProtocolError("model layer is not the expected OLMoE sparse block")
            originals.append((block, block.forward))

            def patched_forward(
                bound_block: Any, hidden_states: Any, _layer_index: int = layer_index
            ) -> tuple[Any, Any]:
                return controller.forward(bound_block, _layer_index, hidden_states)

            block.forward = types.MethodType(patched_forward, block)
        yield controller
    finally:
        controller.abort_step()
        for block, original in originals:
            block.forward = original


def _prepare_replay_states(
    model: Any,
    requests: Sequence[Any],
    *,
    roster: Sequence[Mapping[str, Any]],
    capture_helper: Any,
    deadline: Deadline,
) -> tuple[dict[str, DecodeState], float]:
    import torch

    first_inputs: dict[str, int] = {}
    for row in roster:
        for request_id, step, token_id in zip(
            row["request_ids"], row["decode_steps"], row["input_token_ids"]
        ):
            if int(step) == 0:
                first_inputs[str(request_id)] = int(token_id)
    states: dict[str, DecodeState] = {}
    total_prefill_ms = 0.0
    for spec in sorted(requests, key=lambda item: (item.arrival_us, item.request_id)):
        deadline.check("teacher-forced replay prefill")
        with torch.inference_mode():
            output, elapsed_us = capture_helper._timed_call(
                model,
                "d10_replay_prefill",
                1,
                None,
                input_ids=spec.input_ids,
                attention_mask=spec.attention_mask,
                use_cache=True,
                output_router_logits=False,
                return_dict=True,
            )
        total_prefill_ms += float(elapsed_us) / 1000.0
        cache = output.past_key_values
        logits = output.logits
        prompt_length = int(spec.input_ids.shape[1])
        if cache is None or capture_helper._cache_length(cache) != prompt_length:
            raise ProtocolError(f"replay prefill cache failed for {spec.request_id}")
        if logits is None or not bool(torch.isfinite(logits).all().item()):
            raise ProtocolError(f"replay prefill logits failed for {spec.request_id}")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if int(next_token.item()) != first_inputs.get(spec.request_id):
            raise ProtocolError(
                f"replay prefill changed frozen step-0 token for {spec.request_id}"
            )
        states[spec.request_id] = DecodeState(
            spec=spec,
            cache=cache,
            attention_mask=spec.attention_mask,
            next_token=next_token,
            prompt_length=prompt_length,
        )
    return states, total_prefill_ms


@dataclass
class ReplayResult:
    arm: str
    phase: str
    repeat: int
    step_signatures: list[dict[str, Any]]
    total_expert_gpu_ms: float
    total_whole_step_wall_ms: float
    token_step_p99_ms: float
    request_steps: int
    kernel_calls: int
    real_rows: int
    dummy_rows: int
    occupied_experts: int
    natural_m_histogram: dict[str, int]
    mean_frozen_token_nll: float
    prefill_wall_ms: float

    def public(self) -> dict[str, Any]:
        physical_rows = self.real_rows + self.dummy_rows
        return {
            "arm": self.arm,
            "phase": self.phase,
            "repeat": self.repeat,
            "decode_batches": len(self.step_signatures),
            "request_steps": self.request_steps,
            "total_expert_gpu_ms": self.total_expert_gpu_ms,
            "total_whole_step_wall_ms": self.total_whole_step_wall_ms,
            "token_step_p99_ms": self.token_step_p99_ms,
            "kernel_calls": self.kernel_calls,
            "real_rows": self.real_rows,
            "dummy_rows": self.dummy_rows,
            "padding_fraction": (
                self.dummy_rows / physical_rows if physical_rows else 0.0
            ),
            "occupied_experts": self.occupied_experts,
            "natural_m_histogram": dict(sorted(self.natural_m_histogram.items())),
            "mean_frozen_token_nll": self.mean_frozen_token_nll,
            "prefill_wall_ms_excluded_from_decode_metrics": self.prefill_wall_ms,
        }


def replay_comparison_payload(replay: ReplayResult) -> dict[str, Any]:
    """Expose only the fields consumed by the frozen within-policy comparator."""

    return {
        "arm": replay.arm,
        "step_signatures": replay.step_signatures,
    }


def run_teacher_forced_replay(
    model: Any,
    requests: Sequence[Any],
    roster: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    phase: str,
    repeat: int,
    capture_helper: Any,
    canonical_m: int,
    deadline: Deadline,
    expert_calls_sink: JsonlWriter,
    expert_stages_sink: JsonlWriter,
    steps_sink: JsonlWriter,
) -> ReplayResult:
    import torch

    states, prefill_wall_ms = _prepare_replay_states(
        model,
        requests,
        roster=roster,
        capture_helper=capture_helper,
        deadline=deadline,
    )
    future_inputs = {
        (str(request_id), int(step)): int(token_id)
        for row in roster
        for request_id, step, token_id in zip(
            row["request_ids"], row["decode_steps"], row["input_token_ids"]
        )
    }
    step_signatures: list[dict[str, Any]] = []
    token_step_latencies: list[float] = []
    total_expert_ms = 0.0
    total_wall_ms = 0.0
    kernel_calls = 0
    real_rows = 0
    dummy_rows = 0
    occupied_experts = 0
    natural_histogram: dict[str, int] = {}
    nll_values: list[float] = []

    with patched_moe_policy(model, arm, canonical_m) as controller:
        for frozen in roster:
            deadline.check(f"{phase} {arm} repeat {repeat} batch")
            batch_index = int(frozen["batch_index"])
            request_ids = [str(value) for value in frozen["request_ids"]]
            decode_steps = [int(value) for value in frozen["decode_steps"]]
            input_tokens = [int(value) for value in frozen["input_token_ids"]]
            batch = [states[request_id] for request_id in request_ids]
            for state, expected_step, token_id in zip(
                batch, decode_steps, input_tokens
            ):
                if state.decode_step != expected_step:
                    raise ProtocolError("teacher-forced replay changed request-step order")
                state.next_token = torch.tensor(
                    [[token_id]], dtype=torch.long, device=model.device
                )
            (
                input_ids,
                attention_mask,
                position_ids,
                cache,
                prior_lengths,
                prior_max,
            ) = capture_helper._pad_decode_inputs(batch)
            if [int(value) for value in prior_lengths] != [
                int(value) for value in frozen["prior_cache_lengths"]
            ]:
                raise ProtocolError("teacher-forced replay cache lengths drifted")
            if [int(value) for value in position_ids[:, 0].tolist()] != [
                int(value) for value in frozen["position_ids"]
            ]:
                raise ProtocolError("teacher-forced replay position IDs drifted")
            controller.begin_step(
                batch_index=batch_index,
                request_ids=request_ids,
                decode_steps=decode_steps,
            )
            try:
                with torch.inference_mode():
                    output, elapsed_us = capture_helper._timed_call(
                        model,
                        f"d10_{phase}_{arm}_decode",
                        len(batch),
                        None,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        cache_position=torch.tensor(
                            [prior_max], dtype=torch.long, device=input_ids.device
                        ),
                        past_key_values=cache,
                        use_cache=True,
                        output_router_logits=True,
                        return_dict=True,
                    )
                call_rows, layer_rows, patched_route_hash = controller.finish_step()
            except BaseException:
                controller.abort_step()
                raise
            logits = output.logits
            output_cache = output.past_key_values
            if logits is None or output_cache is None:
                raise ProtocolError("teacher-forced decode returned no logits/cache")
            if logits.dtype != torch.bfloat16 or not bool(torch.isfinite(logits).all().item()):
                raise ProtocolError("teacher-forced decode logits are not finite BF16")
            route_batches = capture_helper._native_route_batches(
                output, expected_rows=len(batch), config=model.config
            )
            observed_routes = _route_entries(route_batches, request_ids, decode_steps)
            observed_route_hash = canonical_sha256(observed_routes)
            if observed_route_hash != patched_route_hash:
                raise ProtocolError("patched route ledger disagrees with returned router logits")
            split_caches = capture_helper.split_left_padded_cache(
                output_cache,
                prior_lengths=prior_lengths,
                prior_max_length=prior_max,
            )
            predicted = torch.argmax(logits[:, -1, :], dim=-1)
            targets = torch.tensor(
                frozen["native_predicted_next_token_ids"],
                dtype=torch.long,
                device=logits.device,
            )
            nll = -torch.log_softmax(logits[:, -1, :].float(), dim=-1).gather(
                1, targets[:, None]
            )[:, 0]
            if not bool(torch.isfinite(nll).all().item()):
                raise ProtocolError("frozen-token NLL proxy produced NaN/Inf")
            batch_nll = [float(value) for value in nll.detach().cpu().tolist()]
            nll_values.extend(batch_nll)
            final_hash = _tensor_storage_sha256(
                logits[:, -1, :], require_bf16=True
            )

            annotated_calls: list[dict[str, Any]] = []
            for row in call_rows:
                annotated = {
                    "schema_version": "stablebatch-shape-lane-expert-call-v1",
                    "arm": arm,
                    "phase": phase,
                    "repeat": repeat,
                    **row,
                }
                expert_calls_sink.write(annotated)
                annotated_calls.append(annotated)
                logical_m = int(row["logical_m"])
                natural_histogram[str(logical_m)] = (
                    natural_histogram.get(str(logical_m), 0) + 1
                )
                kernel_calls += int(row["kernel_calls"])
                real_rows += logical_m
                dummy_rows += int(row["padding_rows"])
                occupied_experts += 1
            for row in layer_rows:
                expert_stages_sink.write(
                    {
                        "schema_version": "stablebatch-shape-lane-expert-stage-v1",
                        "arm": arm,
                        "phase": phase,
                        "repeat": repeat,
                        **row,
                    }
                )
            expert_ms = sum(float(row["expert_stage_gpu_ms"]) for row in layer_rows)
            wall_ms = float(elapsed_us) / 1000.0
            total_expert_ms += expert_ms
            total_wall_ms += wall_ms
            token_step_latencies.extend([wall_ms] * len(batch))
            roster_identity = {
                "batch_index": batch_index,
                "request_ids": request_ids,
                "decode_steps": decode_steps,
                "input_token_ids": input_tokens,
                "position_ids": [int(value) for value in frozen["position_ids"]],
            }
            raw_call_digest = canonical_sha256(
                [
                    {
                        "layer": row["layer"],
                        "expert_id": row["expert_id"],
                        "logical_m": row["logical_m"],
                        "physical_m": row["physical_m"],
                        "kernel_calls": row["kernel_calls"],
                        "row_ids": row["row_ids"],
                        "raw_bf16_sha256": row["raw_bf16_sha256"],
                        "raw_row_sha256": row["raw_row_sha256"],
                    }
                    for row in annotated_calls
                ]
            )
            step_signature = {
                "batch_index": batch_index,
                "roster_identity_sha256": canonical_sha256(roster_identity),
                "raw_calls_sha256": raw_call_digest,
                "route_membership_sha256": observed_route_hash,
                "final_logits_sha256": final_hash,
                "greedy_token_ids": [int(value) for value in predicted.cpu().tolist()],
                "mean_frozen_token_nll": statistics.fmean(batch_nll),
            }
            step_signatures.append(step_signature)
            steps_sink.write(
                {
                    "schema_version": "stablebatch-shape-lane-decode-step-v1",
                    "arm": arm,
                    "phase": phase,
                    "repeat": repeat,
                    "batch_index": batch_index,
                    "request_ids": request_ids,
                    "decode_steps": decode_steps,
                    "input_token_ids": input_tokens,
                    "teacher_forced_target_token_ids": [
                        int(value)
                        for value in frozen["native_predicted_next_token_ids"]
                    ],
                    "frozen_token_nll": batch_nll,
                    "whole_step_wall_ms": wall_ms,
                    "expert_stage_gpu_ms": expert_ms,
                    "timing_boundaries": {
                        "whole_step": "cuda_sync_before_model_to_logits_ready_cuda_sync",
                        "expert_stage": (
                            "sum_of_layer_cuda_events_after_router_topk_through_"
                            "dispatch_padding_expert_and_index_add"
                        ),
                    },
                    "native_roster_route_membership_sha256": frozen[
                        "native_route_membership_sha256"
                    ],
                    "native_roster_final_logits_sha256": frozen[
                        "native_final_logits_sha256"
                    ],
                    **step_signature,
                }
            )
            for index, state in enumerate(batch):
                state.cache = split_caches[index]
                state.attention_mask = torch.cat(
                    (state.attention_mask, state.attention_mask.new_ones((1, 1))),
                    dim=1,
                )
                state.decode_step += 1
                next_key = (state.spec.request_id, state.decode_step)
                if next_key in future_inputs:
                    state.next_token = torch.tensor(
                        [[future_inputs[next_key]]],
                        dtype=torch.long,
                        device=model.device,
                    )

    expected_steps = sum(len(row["request_ids"]) for row in roster)
    if sum(state.decode_step for state in states.values()) != expected_steps:
        raise ProtocolError("teacher-forced replay did not close request-step state")
    if len(step_signatures) != len(roster):
        raise ProtocolError("teacher-forced replay did not cover every roster batch")
    return ReplayResult(
        arm=arm,
        phase=phase,
        repeat=repeat,
        step_signatures=step_signatures,
        total_expert_gpu_ms=total_expert_ms,
        total_whole_step_wall_ms=total_wall_ms,
        token_step_p99_ms=percentile(token_step_latencies, 0.99),
        request_steps=len(token_step_latencies),
        kernel_calls=kernel_calls,
        real_rows=real_rows,
        dummy_rows=dummy_rows,
        occupied_experts=occupied_experts,
        natural_m_histogram=natural_histogram,
        mean_frozen_token_nll=statistics.fmean(nll_values),
        prefill_wall_ms=prefill_wall_ms,
    )


def summarize_arm_replays(replays: Sequence[ReplayResult]) -> dict[str, Any]:
    if len(replays) != 2 or len({row.repeat for row in replays}) != 2:
        raise ProtocolError("arm summary requires exactly two measured replays")
    arm = replays[0].arm
    if any(row.arm != arm or row.phase != "measured" for row in replays):
        raise ProtocolError("arm summary mixed policies or warmups")
    total_real = sum(row.real_rows for row in replays)
    total_dummy = sum(row.dummy_rows for row in replays)
    histogram: dict[str, int] = {}
    for replay in replays:
        for natural_m, count in replay.natural_m_histogram.items():
            histogram[natural_m] = histogram.get(natural_m, 0) + int(count)
    return {
        "arm": arm,
        "measured_replays": [row.public() for row in sorted(replays, key=lambda x: x.repeat)],
        "median_total_expert_gpu_ms": statistics.median(
            row.total_expert_gpu_ms for row in replays
        ),
        "median_total_whole_step_wall_ms": statistics.median(
            row.total_whole_step_wall_ms for row in replays
        ),
        "median_token_step_p99_ms": statistics.median(
            row.token_step_p99_ms for row in replays
        ),
        "mean_frozen_token_nll": statistics.fmean(
            row.mean_frozen_token_nll for row in replays
        ),
        "kernel_calls": sum(row.kernel_calls for row in replays),
        "real_rows": total_real,
        "dummy_rows": total_dummy,
        "padding_fraction": total_dummy / (total_real + total_dummy)
        if total_real + total_dummy
        else 0.0,
        "occupied_experts": sum(row.occupied_experts for row in replays),
        "natural_m_histogram": dict(sorted(histogram.items())),
    }


def cross_arm_behavior_diagnostics(
    measured: Mapping[str, Sequence[ReplayResult]]
) -> dict[str, Any]:
    native_by_repeat = {row.repeat: row for row in measured[ARM_NATIVE]}
    output: dict[str, Any] = {
        "reference": "native_variable_m_behavior_reference_not_ground_truth",
        "m1_is_ground_truth": False,
        "arms": {},
    }
    for arm in (ARM_SERIAL, ARM_C8):
        route_mismatches = 0
        final_mismatches = 0
        greedy_mismatches = 0
        nll_deltas: list[float] = []
        for replay in measured[arm]:
            native = native_by_repeat[replay.repeat]
            left = {row["batch_index"]: row for row in native.step_signatures}
            right = {row["batch_index"]: row for row in replay.step_signatures}
            if set(left) != set(right):
                raise ProtocolError("cross-arm diagnostic roster coverage drifted")
            for batch_index in left:
                lhs, rhs = left[batch_index], right[batch_index]
                if lhs["roster_identity_sha256"] != rhs["roster_identity_sha256"]:
                    raise ProtocolError("cross-arm diagnostic changed roster identity")
                route_mismatches += lhs["route_membership_sha256"] != rhs[
                    "route_membership_sha256"
                ]
                final_mismatches += lhs["final_logits_sha256"] != rhs[
                    "final_logits_sha256"
                ]
                greedy_mismatches += sum(
                    a != b
                    for a, b in zip(lhs["greedy_token_ids"], rhs["greedy_token_ids"])
                )
                nll_deltas.append(
                    float(rhs["mean_frozen_token_nll"])
                    - float(lhs["mean_frozen_token_nll"])
                )
        output["arms"][arm] = {
            "route_mismatch_batches_vs_native": int(route_mismatches),
            "final_logits_mismatch_batches_vs_native": int(final_mismatches),
            "greedy_token_mismatches_vs_native": int(greedy_mismatches),
            "mean_frozen_token_nll_delta_vs_native": statistics.fmean(nll_deltas),
            "interpretation": "behavior diagnostic only; equality is not a gate",
        }
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args(argv)


def _default_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise ProtocolError("cannot infer repository root")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner_path = Path(__file__).resolve()
    repo_root = (args.repo_root or _default_repo_root()).resolve()
    config_path = (
        args.config
        or repo_root
        / "docs/ideas/stablebatch/experiments/configs/shape_lane_continuous_cost_gate_v1.json"
    ).resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ProtocolError(f"refusing to reuse output directory {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    started_wall = time.time()
    config = load_json(config_path)
    write_json_new(
        output_dir / "run_request.json",
        {
            "schema_version": "stablebatch-shape-lane-cost-run-request-v1",
            "started_at": utc_now(),
            "argv": sys.argv if argv is None else [str(runner_path), *argv],
            "pid": os.getpid(),
            "repo_root": str(repo_root),
            "runner_path": str(runner_path),
            "runner_sha256": sha256_file(runner_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "experiment_scope": (
                "single_gpu_same_stack_mechanism_cost_not_serving_not_vllm_bi"
            ),
            "experiment_boundary": EXPERIMENT_BOUNDARY,
            "bcrd_serial_audit": BCRD_SERIAL_AUDIT_STATUS,
            "serving_result": False,
        },
    )
    try:
        static = verify_static_bindings(config, repo_root, config_path, runner_path)
        deadline = Deadline(float(config["execution"]["maximum_wall_seconds"]))
        capture_helper, stable_helper = load_helpers(repo_root)
        pre_import_gpu = stable_helper.gpu_snapshot()
        environment = stable_helper.verify_environment(config, pre_import_gpu)
        write_json_new(output_dir / "environment.json", environment)
        write_json_new(output_dir / "static_bindings.json", static)
        write_json_new(output_dir / "config_snapshot.json", config)

        workload_path = Path(static["workload_path"])
        workload = capture_helper.load_workload_manifest(workload_path)
        workload_cfg = config["workload"]
        if len(workload["requests"]) != int(workload_cfg["expected_requests"]):
            raise ProtocolError("workload request count differs from the frozen D10 config")
        if int(workload["generation"]["max_decode_steps"]) != int(
            workload_cfg["expected_decode_steps_per_request"]
        ):
            raise ProtocolError("workload decode-step count drifted")
        if int(workload["scheduler"]["max_batch_size"]) != int(
            workload_cfg["max_batch_size"]
        ):
            raise ProtocolError("workload max batch size drifted")
        if str(workload["generation"]["mode"]) != str(
            workload_cfg["generation_mode"]
        ):
            raise ProtocolError("workload generation mode drifted")
        if str(workload["model"]["revision"]) != str(config["model"]["revision"]):
            raise ProtocolError("workload/model revisions disagree")
        write_json_new(output_dir / "workload_snapshot.json", workload)

        seed = int(workload["seed"])
        random.seed(seed)
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model_load_started = time.monotonic()
        model, tokenizer = stable_helper.load_model(config)
        model_load_seconds = time.monotonic() - model_load_started
        requests = capture_helper._prepare_requests(workload, tokenizer, model.device)
        deadline.check("model and request preparation")

        native_capture = capture_native_roster(
            model,
            requests,
            capture_helper=capture_helper,
            max_decode_steps=int(workload_cfg["expected_decode_steps_per_request"]),
            max_batch_size=int(workload_cfg["max_batch_size"]),
            deadline=deadline,
        )
        request_ids = [str(row["request_id"]) for row in workload["requests"]]
        roster_audit = validate_roster_conservation(
            native_capture.roster_rows,
            expected_request_ids=request_ids,
            expected_steps_per_request=int(
                workload_cfg["expected_decode_steps_per_request"]
            ),
            max_batch_size=int(workload_cfg["max_batch_size"]),
        )
        if int(roster_audit["request_steps"]) != int(
            workload_cfg["expected_request_steps"]
        ):
            raise ProtocolError("native roster denominator differs from the frozen config")
        write_jsonl_new(
            output_dir / "native_prefill_ledger.jsonl", native_capture.prefill_rows
        )
        write_jsonl_new(output_dir / "native_roster.jsonl", native_capture.roster_rows)

        measured: dict[str, list[ReplayResult]] = {arm: [] for arm in ARMS}
        warmups: dict[str, ReplayResult] = {}
        replay_rows: list[dict[str, Any]] = []
        with contextlib.ExitStack() as stack:
            expert_calls_sink = stack.enter_context(
                JsonlWriter(output_dir / "expert_call_ledger.jsonl")
            )
            expert_stages_sink = stack.enter_context(
                JsonlWriter(output_dir / "expert_stage_ledger.jsonl")
            )
            steps_sink = stack.enter_context(
                JsonlWriter(output_dir / "decode_step_ledger.jsonl")
            )
            replay_sink = stack.enter_context(
                JsonlWriter(output_dir / "replay_ledger.jsonl")
            )
            for arm in config["execution"]["warmup_arm_order"]:
                result = run_teacher_forced_replay(
                    model,
                    requests,
                    native_capture.roster_rows,
                    arm=str(arm),
                    phase="warmup",
                    repeat=0,
                    capture_helper=capture_helper,
                    canonical_m=CANONICAL_M,
                    deadline=deadline,
                    expert_calls_sink=expert_calls_sink,
                    expert_stages_sink=expert_stages_sink,
                    steps_sink=steps_sink,
                )
                warmups[str(arm)] = result
                public = result.public()
                replay_rows.append(public)
                replay_sink.write(public)
                expert_calls_sink.checkpoint()
                expert_stages_sink.checkpoint()
                steps_sink.checkpoint()
                replay_sink.checkpoint()
            for repeat, order in enumerate(config["execution"]["measured_arm_orders"]):
                for arm in order:
                    result = run_teacher_forced_replay(
                        model,
                        requests,
                        native_capture.roster_rows,
                        arm=str(arm),
                        phase="measured",
                        repeat=repeat,
                        capture_helper=capture_helper,
                        canonical_m=CANONICAL_M,
                        deadline=deadline,
                        expert_calls_sink=expert_calls_sink,
                        expert_stages_sink=expert_stages_sink,
                        steps_sink=steps_sink,
                    )
                    measured[str(arm)].append(result)
                    public = result.public()
                    replay_rows.append(public)
                    replay_sink.write(public)
                    expert_calls_sink.checkpoint()
                    expert_stages_sink.checkpoint()
                    steps_sink.checkpoint()
                    replay_sink.checkpoint()

        policy_correctness = {
            arm: compare_policy_repeats(
                *(
                    replay_comparison_payload(row)
                    for row in sorted(measured[arm], key=lambda row: row.repeat)
                )
            )
            for arm in ARMS
        }
        arm_metrics = {arm: summarize_arm_replays(measured[arm]) for arm in ARMS}
        gate_result = classify_gate(policy_correctness, arm_metrics, config["gate"])
        behavior = cross_arm_behavior_diagnostics(measured)
        runtime_final = stable_helper.verify_final_runtime(config)
        write_json_new(output_dir / "runtime_final.json", runtime_final)
        summary = {
            "schema_version": "stablebatch-shape-lane-continuous-cost-summary-v1",
            "status": "COMPLETE",
            "verdict": gate_result["verdict"],
            "claim_ceiling": config["claim_ceiling"],
            "research_boundary": config["research_boundary"],
            "experiment_boundary": EXPERIMENT_BOUNDARY,
            "bcrd_serial_audit": BCRD_SERIAL_AUDIT_STATUS,
            "formal_bcrd_producer": False,
            "teacher_forced_frozen_roster": True,
            "serving_result": False,
            "vllm_batch_invariance_result": False,
            "official_batch_invariance": dict(config["official_batch_invariance"]),
            "scientific_ground_truth": None,
            "m1_is_ground_truth": False,
            "correctness_scope": "within_each_policy_across_two_independent_replays",
            "timing_scope": {
                "expert_gpu_ms": (
                    "layer CUDA events after router/topk through dispatch, policy "
                    "packing, expert calls, and index_add combine"
                ),
                "token_step_wall_ms": (
                    "CUDA-synchronized base-model-plus-lm-head decode call; each "
                    "request-step inherits its containing batch latency"
                ),
                "serving_queue_wait_included": False,
            },
            "native_roster": roster_audit,
            "warmups": {arm: warmups[arm].public() for arm in ARMS},
            "policy_repeat_correctness": policy_correctness,
            "arm_metrics": arm_metrics,
            "cross_arm_behavior_diagnostics": behavior,
            "gate": gate_result,
            "model_load_seconds": model_load_seconds,
            "wall_seconds": time.time() - started_wall,
            "completed_at": utc_now(),
        }
        write_json_new(output_dir / "summary.json", summary)
        manifest = build_manifest(output_dir)
        write_json_new(output_dir / "MANIFEST.json", manifest)
        verify_manifest(output_dir, manifest)
        status = {
            "schema_version": "stablebatch-shape-lane-cost-run-status-v1",
            "status": "COMPLETE",
            "scientific_result_eligible": True,
            "verdict": summary["verdict"],
            "claim_ceiling": config["claim_ceiling"],
            "experiment_boundary": EXPERIMENT_BOUNDARY,
            "bcrd_serial_audit": BCRD_SERIAL_AUDIT_STATUS,
            "serving_result": False,
            "official_batch_invariance_status": "NOT_EXECUTABLE",
            "required_sentinel": "COMPLETE.json",
            "completed_at": utc_now(),
            "wall_seconds": time.time() - started_wall,
        }
        write_json_new(output_dir / "RUN_STATUS.json", status)
        complete = {
            "schema_version": "stablebatch-shape-lane-cost-complete-v1",
            "status": "COMPLETE",
            "verdict": summary["verdict"],
            "claim_ceiling": config["claim_ceiling"],
            "experiment_boundary": EXPERIMENT_BOUNDARY,
            "bcrd_serial_audit": BCRD_SERIAL_AUDIT_STATUS,
            "serving_result": False,
            "manifest_sha256": sha256_file(output_dir / "MANIFEST.json"),
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            "run_status_sha256": sha256_file(output_dir / "RUN_STATUS.json"),
            "completion_last": True,
            "completed_at": utc_now(),
        }
        # Scientific consumers must require this final filesystem mutation.
        write_json_new(output_dir / "COMPLETE.json", complete)
        print(json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            "schema_version": "stablebatch-shape-lane-cost-failure-v1",
            "status": "INVALID",
            "scientific_result_eligible": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at": utc_now(),
            "wall_seconds": time.time() - started_wall,
        }
        if not (output_dir / "FAILURE.json").exists():
            write_json_new(output_dir / "FAILURE.json", failure)
        if not (output_dir / "RUN_STATUS.json").exists():
            write_json_new(output_dir / "RUN_STATUS.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
