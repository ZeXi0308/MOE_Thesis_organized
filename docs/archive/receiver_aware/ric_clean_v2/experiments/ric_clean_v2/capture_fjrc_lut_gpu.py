#!/usr/bin/env python3
"""Produce the frozen FJRC RTX-5090 primitive LUT.

The artifact contains BF16 sender-pack, dense-depth receiver-unpack queue,
once-per-join canonical-combine, and auxiliary host keyed-lookup timings.  It
is an exploratory calibration input, not an RDMA or serving result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parents[1] / "FJRC_Phase2_FrozenOraclePilot_2026-07-23.md"
MODEL_SHAPES: Mapping[str, Mapping[str, Any]] = {
    "olmoe": {
        "model_revision": "allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5",
        "hidden": 2048,
        "top_k": 8,
    },
    "llmjp": {
        "model_revision": "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M@1d5983076dfc67aee4a77ec06a27027f5bab6055",
        "hidden": 512,
        "top_k": 16,
    },
}
UNPACK_DEPTHS = tuple(range(17))
WARMUPS = 20
MEASURED = 100
HOST_LOOKUP_REPEATS = 4096
CUDA_BOUNDARY = "REAL_5090_CUDA_SINGLE_SERIAL_STREAM_NOT_NETWORK"
HOST_BOUNDARY = "HOST_PYTHON_KEYED_LOOKUP_AUXILIARY_NOT_ZERO_TAX_R0"
SOURCES = {
    "sender_pack": "measured_5090_cuda_sender_pack_row1",
    "receiver_unpack": "measured_5090_cuda_receiver_unpack_row1_queue",
    "canonical_combine": "measured_5090_cuda_canonical_combine_once_per_join",
    "host_lookup_tax": "measured_host_perf_counter_keyed_receiver_lookup",
}


class FJRCLUTError(RuntimeError):
    """The frozen LUT schema, census, environment, or publication failed."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    if "artifact_sha256" in value:
        raise FJRCLUTError("payload already contains artifact_sha256")
    result = dict(value)
    result["artifact_sha256"] = object_sha256(result)
    return result


def validate_self_hash(value: Mapping[str, Any]) -> None:
    expected = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if not isinstance(expected, str) or expected != object_sha256(payload):
        raise FJRCLUTError("artifact self hash mismatch")


def nearest_rank_p95(values: Sequence[float]) -> float:
    if not values:
        raise FJRCLUTError("cannot summarize empty values")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def tensor_descriptor(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "element_size_bytes": int(tensor.element_size()),
        "payload_bytes": int(tensor.numel()) * int(tensor.element_size()),
    }


def tensor_sha256(tensor: Any) -> str:
    import torch

    original = tensor.detach().cpu()
    contiguous = original.contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(str(tuple(original.shape)).encode())
    digest.update(b"\0")
    digest.update(str(original.dtype).encode())
    digest.update(b"\0")
    digest.update(str(tuple(original.stride())).encode())
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _validate_cuda_row(row: Mapping[str, Any]) -> None:
    model = row.get("model_key")
    component = row.get("component")
    phase = row.get("phase")
    depth = row.get("queue_depth")
    cuda_us = row.get("cuda_event_us")
    wall_us = row.get("wall_time_us")
    completions = row.get("invocation_completion_us")
    if (
        model not in MODEL_SHAPES
        or row.get("model_revision") != MODEL_SHAPES[str(model)]["model_revision"]
        or row.get("hidden") != MODEL_SHAPES[str(model)]["hidden"]
        or row.get("top_k") != MODEL_SHAPES[str(model)]["top_k"]
        or row.get("dtype") != "torch.bfloat16"
        or component not in {"sender_pack", "receiver_unpack", "canonical_combine"}
        or phase not in {"warmup", "measured"}
        or type(row.get("trial_index")) is not int
        or type(row.get("execution_ordinal")) is not int
        or not isinstance(row.get("stream_id"), int)
        or not isinstance(cuda_us, (int, float))
        or isinstance(cuda_us, bool)
        or not math.isfinite(float(cuda_us))
        or float(cuda_us) < 0
        or not isinstance(wall_us, (int, float))
        or isinstance(wall_us, bool)
        or not math.isfinite(float(wall_us))
        or float(wall_us) <= 0
        or float(cuda_us) > float(wall_us)
        or row.get("source") != SOURCES[str(component)]
        or row.get("evidence_boundary") != CUDA_BOUNDARY
        or not isinstance(row.get("producer_source_sha256"), str)
        or len(row["producer_source_sha256"]) != 64
        or not isinstance(row.get("input_tensor_sha256"), str)
        or len(row["input_tensor_sha256"]) != 64
        or not isinstance(row.get("input_descriptor_sha256"), str)
        or len(row["input_descriptor_sha256"]) != 64
        or not isinstance(row.get("output_descriptor_sha256"), str)
        or len(row["output_descriptor_sha256"]) != 64
    ):
        raise FJRCLUTError("CUDA raw row violates frozen schema")
    if component == "receiver_unpack":
        if (
            type(depth) is not int
            or depth not in UNPACK_DEPTHS
            or row.get("primitive_invocations") != depth
            or row.get("candidate_included") is not False
            or not isinstance(completions, list)
            or len(completions) != depth
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0 for value in completions)
            or any(float(completions[index]) > float(completions[index + 1]) for index in range(len(completions) - 1))
            or (depth > 0 and (float(cuda_us) <= 0 or abs(float(completions[-1]) - float(cuda_us)) > 1e-9))
        ):
            raise FJRCLUTError("unpack completion timestamp schema mismatch")
    elif depth is not None or row.get("primitive_invocations") != 1 or completions != [] or float(cuda_us) <= 0:
        raise FJRCLUTError("single-invocation CUDA component schema mismatch")


def _validate_host_row(row: Mapping[str, Any]) -> None:
    tax = row.get("lookup_tax_us_per_joint_decision")
    if (
        row.get("component") != "host_lookup_tax"
        or row.get("phase") not in {"warmup", "measured"}
        or type(row.get("trial_index")) is not int
        or type(row.get("execution_ordinal")) is not int
        or row.get("lookup_repeats") != HOST_LOOKUP_REPEATS
        or row.get("lookups_per_joint_decision") != 6
        or row.get("enters_zero_tax_r0") is not False
        or not isinstance(tax, (int, float))
        or isinstance(tax, bool)
        or not math.isfinite(float(tax))
        or float(tax) <= 0
        or row.get("source") != SOURCES["host_lookup_tax"]
        or row.get("evidence_boundary") != HOST_BOUNDARY
        or not isinstance(row.get("producer_source_sha256"), str)
        or len(row["producer_source_sha256"]) != 64
    ):
        raise FJRCLUTError("host lookup row violates frozen schema")


def _trial_census(rows: Sequence[Mapping[str, Any]], warmups: int, measured: int) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    first = [row for row in rows if row["phase"] == "warmup"]
    second = [row for row in rows if row["phase"] == "measured"]
    if (
        len(first) != warmups
        or len(second) != measured
        or {row["trial_index"] for row in first} != set(range(warmups))
        or {row["trial_index"] for row in second} != set(range(measured))
    ):
        raise FJRCLUTError("20+100 trial census mismatch")
    return first, second


def validate_and_summarize(raw_trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cuda_points_per_model = 2 + len(UNPACK_DEPTHS)
    expected = len(MODEL_SHAPES) * cuda_points_per_model * (WARMUPS + MEASURED) + WARMUPS + MEASURED
    if len(raw_trials) != expected:
        raise FJRCLUTError("raw LUT census mismatch")
    identities: set[tuple[Any, ...]] = set()
    ordinals = []
    cuda_groups: dict[tuple[str, str, int | None], list[Mapping[str, Any]]] = {}
    host_rows = []
    for row in raw_trials:
        ordinal = row.get("execution_ordinal")
        ordinals.append(ordinal)
        if row.get("component") == "host_lookup_tax":
            _validate_host_row(row)
            key = ("host_lookup_tax", row["phase"], row["trial_index"])
            host_rows.append(row)
        else:
            _validate_cuda_row(row)
            key = (row["model_key"], row["component"], row.get("queue_depth"), row["phase"], row["trial_index"])
            cuda_groups.setdefault((row["model_key"], row["component"], row.get("queue_depth")), []).append(row)
        if key in identities:
            raise FJRCLUTError("duplicate raw trial identity")
        identities.add(key)
    if ordinals != list(range(expected)):
        raise FJRCLUTError("raw execution ordinals are not contiguous and ordered")

    expected_cuda = {
        (model, component, depth)
        for model in MODEL_SHAPES
        for component, depths in (
            ("sender_pack", (None,)),
            ("canonical_combine", (None,)),
            ("receiver_unpack", UNPACK_DEPTHS),
        )
        for depth in depths
    }
    if set(cuda_groups) != expected_cuda:
        raise FJRCLUTError("CUDA point surface is incomplete")

    summary: list[dict[str, Any]] = []
    for model in MODEL_SHAPES:
        shape = MODEL_SHAPES[model]
        for component, depths in (
            ("sender_pack", (None,)),
            ("receiver_unpack", UNPACK_DEPTHS),
            ("canonical_combine", (None,)),
        ):
            for depth in depths:
                rows = cuda_groups[(model, component, depth)]
                _warmup, measured_rows = _trial_census(rows, WARMUPS, MEASURED)
                immutable = (
                    "model_revision", "hidden", "top_k", "dtype", "primitive_invocations",
                    "candidate_included", "stream_id", "source", "evidence_boundary",
                    "producer_source_sha256", "input_tensor_sha256", "input_descriptor_sha256",
                    "output_descriptor_sha256",
                )
                if any(len({row.get(field) for row in rows}) != 1 for field in immutable):
                    raise FJRCLUTError("CUDA point immutable field drift")
                cuda = [float(row["cuda_event_us"]) for row in measured_rows]
                wall = [float(row["wall_time_us"]) for row in measured_rows]
                completion_medians: list[float] = []
                if component == "receiver_unpack" and depth:
                    completion_medians = [
                        float(statistics.median(float(row["invocation_completion_us"][index]) for row in measured_rows))
                        for index in range(int(depth))
                    ]
                summary.append(
                    {
                        "model_key": model,
                        "model_revision": shape["model_revision"],
                        "hidden": shape["hidden"],
                        "top_k": shape["top_k"],
                        "dtype": "torch.bfloat16",
                        "component": component,
                        "queue_depth": depth,
                        "warmup_count": WARMUPS,
                        "measured_count": MEASURED,
                        "median_cuda_event_us": float(statistics.median(cuda)),
                        "p95_cuda_event_us": nearest_rank_p95(cuda),
                        "max_cuda_event_us": max(cuda),
                        "median_wall_time_us": float(statistics.median(wall)),
                        "p95_wall_time_us": nearest_rank_p95(wall),
                        "max_wall_time_us": max(wall),
                        "median_invocation_completion_us": completion_medians,
                        "backlog_only_queue_work_us": 0.0 if component == "receiver_unpack" and depth == 0 else float(statistics.median(cuda)),
                        "candidate_included": False if component == "receiver_unpack" else None,
                        "source": SOURCES[component],
                        "evidence_boundary": CUDA_BOUNDARY,
                        "producer_source_sha256": rows[0]["producer_source_sha256"],
                        "input_tensor_sha256": rows[0]["input_tensor_sha256"],
                        "input_descriptor_sha256": rows[0]["input_descriptor_sha256"],
                        "output_descriptor_sha256": rows[0]["output_descriptor_sha256"],
                    }
                )

    _host_warmup, host_measured = _trial_census(host_rows, WARMUPS, MEASURED)
    taxes = [float(row["lookup_tax_us_per_joint_decision"]) for row in host_measured]
    summary.append(
        {
            "model_key": None,
            "component": "host_lookup_tax",
            "warmup_count": WARMUPS,
            "measured_count": MEASURED,
            "lookups_per_joint_decision": 6,
            "lookup_repeats_per_trial": HOST_LOOKUP_REPEATS,
            "median_lookup_tax_us_per_joint_decision": float(statistics.median(taxes)),
            "p95_lookup_tax_us_per_joint_decision": nearest_rank_p95(taxes),
            "max_lookup_tax_us_per_joint_decision": max(taxes),
            "enters_zero_tax_r0": False,
            "source": SOURCES["host_lookup_tax"],
            "evidence_boundary": HOST_BOUNDARY,
            "producer_source_sha256": host_rows[0]["producer_source_sha256"],
        }
    )
    return summary


def _query_compute_apps() -> list[dict[str, Any]]:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory", "--format=csv,noheader,nounits", "-i", "0"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise FJRCLUTError("compute-app query failed") from exc
    if not output or "No running processes found" in output:
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise FJRCLUTError("malformed compute-app query")
        try:
            rows.append({"pid": int(fields[0]), "gpu_uuid": fields[1], "process_name": fields[2], "used_gpu_memory_mib": float(fields[3])})
        except ValueError as exc:
            raise FJRCLUTError("invalid compute-app row") from exc
    return sorted(rows, key=lambda row: (row["pid"], row["process_name"]))


def _reject_foreign_apps(rows: Sequence[Mapping[str, Any]]) -> None:
    foreign = [dict(row) for row in rows if row.get("pid") != os.getpid()]
    if foreign:
        raise FJRCLUTError(f"foreign GPU processes are active: {foreign}")


def capture_environment(torch: Any, before: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,clocks.sm,power.limit,temperature.gpu", "--format=csv,noheader,nounits", "-i", "0"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise FJRCLUTError("GPU environment query failed") from exc
    fields = [field.strip() for field in output.split(",")]
    after = _query_compute_apps()
    _reject_foreign_apps(after)
    if len(fields) != 6 or fields[1] != "NVIDIA GeForce RTX 5090" or torch.cuda.get_device_name(0) != fields[1] or torch.version.cuda is None:
        raise FJRCLUTError("RTX 5090 identity mismatch")
    try:
        numeric = [float(fields[index]) for index in (3, 4, 5)]
    except ValueError as exc:
        raise FJRCLUTError("non-numeric GPU environment") from exc
    return {
        "producer_pid": os.getpid(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_uuid": fields[0], "gpu_name": fields[1], "driver_version": fields[2],
        "clock_sm_mhz": numeric[0], "power_limit_w": numeric[1], "temperature_c": numeric[2],
        "compute_apps_before": [dict(row) for row in before], "compute_apps_after": after,
        "host_platform": platform.platform(), "host_processor": platform.processor(),
    }


def _make_model_primitives(torch: Any, model: str, stream: Any) -> tuple[dict[str, Callable[[], Any]], dict[str, Any]]:
    shape = MODEL_SHAPES[model]
    hidden, top_k = int(shape["hidden"]), int(shape["top_k"])
    with torch.cuda.stream(stream):
        packed_input = (torch.arange(hidden, device="cuda:0", dtype=torch.int64) % 31).to(torch.bfloat16).reshape(1, hidden)
        siblings = (torch.arange(top_k * hidden, device="cuda:0", dtype=torch.int64) % 37).to(torch.bfloat16).reshape(1, top_k, hidden)
        reverse_index = torch.arange(0, -1, -1, device="cuda:0", dtype=torch.int64)
        unpacked = torch.empty_like(packed_input)
        accumulator = torch.empty_like(packed_input)
    stream.synchronize()

    def pack() -> Any:
        return torch.index_select(packed_input, 0, reverse_index)

    def unpack() -> None:
        unpacked.index_copy_(0, reverse_index, packed_input)

    def combine() -> None:
        accumulator.copy_(siblings[:, 0, :])
        for slot in range(1, top_k):
            torch.add(accumulator, siblings[:, slot, :], out=accumulator)

    row_descriptor = tensor_descriptor(packed_input)
    sibling_descriptor = tensor_descriptor(siblings)
    return {"sender_pack": pack, "receiver_unpack": unpack, "canonical_combine": combine}, {
        "packed_input_tensor_sha256": tensor_sha256(packed_input),
        "siblings_tensor_sha256": tensor_sha256(siblings),
        "row_descriptor": row_descriptor,
        "row_descriptor_sha256": object_sha256(row_descriptor),
        "siblings_descriptor": sibling_descriptor,
        "siblings_descriptor_sha256": object_sha256(sibling_descriptor),
    }


def _measure_single(torch: Any, stream: Any, primitive: Callable[[], Any]) -> tuple[float, float]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    with torch.cuda.stream(stream), torch.inference_mode():
        start.record(stream)
        primitive()
        end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0, (time.perf_counter_ns() - wall_start) / 1000.0


def _measure_unpack(torch: Any, stream: Any, primitive: Callable[[], Any], depth: int) -> tuple[float, float, list[float]]:
    start = torch.cuda.Event(enable_timing=True)
    completion_events = [torch.cuda.Event(enable_timing=True) for _ in range(depth)]
    empty_end = torch.cuda.Event(enable_timing=True) if depth == 0 else None
    wall_start = time.perf_counter_ns()
    with torch.cuda.stream(stream), torch.inference_mode():
        start.record(stream)
        for index in range(depth):
            primitive()
            completion_events[index].record(stream)
        if empty_end is not None:
            empty_end.record(stream)
    final_event = completion_events[-1] if completion_events else empty_end
    if final_event is None:
        raise FJRCLUTError("unpack event construction failed")
    final_event.synchronize()
    completions = [float(start.elapsed_time(event)) * 1000.0 for event in completion_events]
    cuda_us = completions[-1] if completions else float(start.elapsed_time(final_event)) * 1000.0
    return cuda_us, (time.perf_counter_ns() - wall_start) / 1000.0, completions


def _measure_host_lookup() -> float:
    receiver_work = {rank: float(rank * rank + 1) for rank in range(8)}
    keys = (7, 1, 5, 2, 6, 0)
    checksum = 0.0
    started = time.perf_counter_ns()
    for _ in range(HOST_LOOKUP_REPEATS):
        checksum += sum(receiver_work[key] for key in keys)
    elapsed = time.perf_counter_ns() - started
    if checksum <= 0:
        raise FJRCLUTError("host lookup checksum failed")
    return elapsed / 1000.0 / HOST_LOOKUP_REPEATS


def run() -> dict[str, Any]:
    if not PROTOCOL.is_file():
        raise FJRCLUTError("frozen protocol is missing")
    try:
        import torch
    except ImportError as exc:
        raise FJRCLUTError("PyTorch is unavailable") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise FJRCLUTError("exactly one CUDA device is required")
    before = _query_compute_apps()
    _reject_foreign_apps(before)
    producer_sha = file_sha256(Path(__file__))
    raw: list[dict[str, Any]] = []
    model_inputs: dict[str, Any] = {}
    ordinal = 0

    for model, shape in MODEL_SHAPES.items():
        stream = torch.cuda.Stream(device=0)
        primitives, descriptors = _make_model_primitives(torch, model, stream)
        model_inputs[model] = {**dict(shape), "dtype": "torch.bfloat16", **descriptors}
        points = [("sender_pack", None), ("canonical_combine", None)] + [("receiver_unpack", depth) for depth in UNPACK_DEPTHS]
        for phase, rounds in (("warmup", WARMUPS), ("measured", MEASURED)):
            for trial_index in range(rounds):
                offset = trial_index % len(points)
                for component, depth in points[offset:] + points[:offset]:
                    if component == "receiver_unpack":
                        cuda_us, wall_us, completions = _measure_unpack(torch, stream, primitives[component], int(depth))
                        input_sha = descriptors["packed_input_tensor_sha256"]
                        input_descriptor = descriptors["row_descriptor_sha256"]
                    else:
                        cuda_us, wall_us = _measure_single(torch, stream, primitives[component])
                        completions = []
                        input_sha = descriptors["siblings_tensor_sha256"] if component == "canonical_combine" else descriptors["packed_input_tensor_sha256"]
                        input_descriptor = descriptors["siblings_descriptor_sha256"] if component == "canonical_combine" else descriptors["row_descriptor_sha256"]
                    raw.append(
                        {
                            "model_key": model, "model_revision": shape["model_revision"],
                            "hidden": shape["hidden"], "top_k": shape["top_k"], "dtype": "torch.bfloat16",
                            "component": component, "queue_depth": depth,
                            "primitive_invocations": int(depth) if component == "receiver_unpack" else 1,
                            "candidate_included": False if component == "receiver_unpack" else None,
                            "phase": phase, "trial_index": trial_index, "execution_ordinal": ordinal,
                            "cuda_event_us": cuda_us, "wall_time_us": wall_us,
                            "invocation_completion_us": completions,
                            "stream_id": int(stream.cuda_stream),
                            "source": SOURCES[component], "evidence_boundary": CUDA_BOUNDARY,
                            "producer_source_sha256": producer_sha,
                            "input_tensor_sha256": input_sha,
                            "input_descriptor_sha256": input_descriptor,
                            "output_descriptor_sha256": descriptors["row_descriptor_sha256"],
                        }
                    )
                    ordinal += 1

    for phase, rounds in (("warmup", WARMUPS), ("measured", MEASURED)):
        for trial_index in range(rounds):
            raw.append(
                {
                    "model_key": None, "component": "host_lookup_tax", "phase": phase,
                    "trial_index": trial_index, "execution_ordinal": ordinal,
                    "lookup_repeats": HOST_LOOKUP_REPEATS, "lookups_per_joint_decision": 6,
                    "lookup_tax_us_per_joint_decision": _measure_host_lookup(),
                    "enters_zero_tax_r0": False,
                    "source": SOURCES["host_lookup_tax"], "evidence_boundary": HOST_BOUNDARY,
                    "producer_source_sha256": producer_sha,
                }
            )
            ordinal += 1

    summary = validate_and_summarize(raw)
    environment = capture_environment(torch, before)
    return add_self_hash(
        {
            "schema_version": "fjrc-primitive-lut-v1",
            "status": "EXPLORATORY_CALIBRATION_INPUT_ONLY",
            "scientific_result": False,
            "protocol_file": str(PROTOCOL), "protocol_sha256": file_sha256(PROTOCOL),
            "producer_source_sha256": producer_sha,
            "warmups_per_point": WARMUPS, "measured_trials_per_point": MEASURED,
            "unpack_depths": list(UNPACK_DEPTHS), "maximum_supported_unpack_depth": 16,
            "depth_above_16_policy": "BLOCKED_NO_INTERPOLATION_NO_EXTRAPOLATION",
            "shared_cut": {
                "bandwidth_gbps": 200,
                "source": "ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA",
                "payload_formula": "hidden*2",
                "descriptor_bytes": 16,
                "alignment_bytes": 16,
            },
            "host_lookup_contract": {
                "receiver_map_entries": 8, "lookups_per_joint_decision": 6,
                "repeats_per_trial": HOST_LOOKUP_REPEATS, "enters_zero_tax_r0": False,
            },
            "model_inputs": model_inputs, "environment": environment,
            "raw_trials": raw, "summary": summary,
        }
    )


def write_json_atomic_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    validate_self_hash(value)
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise FJRCLUTError("refusing to overwrite output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise FJRCLUTError("output parent may not be a symlink")
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    try:
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise FJRCLUTError("atomic write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FJRCLUTError("output appeared during atomic publish") from exc
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FJRCLUTError("refusing to overwrite output")
    result = run()
    write_json_atomic_no_overwrite(args.output, result)
    print(json.dumps({"output": str(args.output), "artifact_sha256": result["artifact_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
