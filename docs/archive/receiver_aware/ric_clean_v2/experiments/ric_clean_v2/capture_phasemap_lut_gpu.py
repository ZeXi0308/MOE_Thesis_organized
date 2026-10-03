#!/usr/bin/env python3
"""Capture the frozen PhaseMap row-1 RTX-5090 primitive LUT.

This producer measures one BF16 sender pack, one BF16 receiver unpack, and one
once-per-join canonical combine for each frozen model.  The artifact is an L2
calibration input only; it is not an RDMA, NCCL, or serving measurement.
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
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parents[1] / "PhaseMap_Phase2_FrozenOracleGate_2026-07-23.md"
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
COMPONENTS = ("sender_pack", "receiver_unpack", "canonical_combine")
WARMUPS = 20
MEASURED = 100
SCHEMA_VERSION = "phasemap-primitive-lut-v1"
STATUS = "EXPLORATORY_CALIBRATION_INPUT_ONLY"
CUDA_BOUNDARY = "REAL_5090_CUDA_SINGLE_SERIAL_STREAM_NOT_NETWORK"
SOURCES = {
    "sender_pack": "measured_5090_cuda_sender_pack_row1",
    "receiver_unpack": "measured_5090_cuda_receiver_unpack_row1",
    "canonical_combine": "measured_5090_cuda_canonical_combine_once_per_join",
}


class PhaseMapLUTError(RuntimeError):
    """The frozen LUT schema, provenance, environment, or publication failed."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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
        raise PhaseMapLUTError("payload already contains artifact_sha256")
    result = dict(value)
    result["artifact_sha256"] = object_sha256(result)
    return result


def validate_self_hash(value: Mapping[str, Any]) -> None:
    expected = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if not isinstance(expected, str) or expected != object_sha256(payload):
        raise PhaseMapLUTError("artifact self hash mismatch")


def nearest_rank_p95(values: Sequence[float]) -> float:
    if not values:
        raise PhaseMapLUTError("cannot summarize empty values")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def tensor_descriptor(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
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


def _bf16_bits(tensor: Any) -> list[int]:
    import torch

    value = tensor.detach().cpu().contiguous()
    if value.dtype != torch.bfloat16:
        raise PhaseMapLUTError("correctness tensor is not BF16")
    return [int(item) & 0xFFFF for item in value.view(torch.int16).reshape(-1).tolist()]


def _bf16_bits_to_float(value: int) -> float:
    return struct.unpack(">f", struct.pack(">I", int(value) << 16))[0]


def _float32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def _float_to_bf16_bits(value: float) -> int:
    raw = struct.unpack(">I", struct.pack(">f", _float32(value)))[0]
    rounded = raw + 0x7FFF + ((raw >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def _contiguous_stride(shape: Sequence[int]) -> list[int]:
    stride = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= int(extent)
    return list(reversed(stride))


def _tensor_sha256_from_bf16_bits(
    bits: Sequence[int], descriptor: Mapping[str, Any]
) -> str:
    shape = descriptor.get("shape")
    stride = descriptor.get("stride")
    if (
        not isinstance(shape, list)
        or any(type(value) is not int or value <= 0 for value in shape)
        or stride != _contiguous_stride(shape)
        or descriptor.get("dtype") != "torch.bfloat16"
        or descriptor.get("numel") != math.prod(shape)
        or descriptor.get("element_size_bytes") != 2
        or descriptor.get("payload_bytes") != len(bits) * 2
        or len(bits) != math.prod(shape)
        or any(type(value) is not int or not 0 <= value <= 0xFFFF for value in bits)
    ):
        raise PhaseMapLUTError("BF16 correctness descriptor/bit census mismatch")
    raw = b"".join(int(value).to_bytes(2, byteorder=sys.byteorder) for value in bits)
    digest = hashlib.sha256()
    digest.update(str(tuple(shape)).encode())
    digest.update(b"\0")
    digest.update(b"torch.bfloat16")
    digest.update(b"\0")
    digest.update(str(tuple(stride)).encode())
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _reference_bits(
    component: str, input_bits: Sequence[int], hidden: int, top_k: int
) -> list[int]:
    if component in {"sender_pack", "receiver_unpack"}:
        if len(input_bits) != hidden:
            raise PhaseMapLUTError("row-1 correctness input census mismatch")
        return list(input_bits)
    if component != "canonical_combine" or len(input_bits) != hidden * top_k:
        raise PhaseMapLUTError("combine correctness input census mismatch")
    result = list(input_bits[:hidden])
    for slot in range(1, top_k):
        offset = slot * hidden
        result = [
            _float_to_bf16_bits(
                _float32(
                    _bf16_bits_to_float(result[index])
                    + _bf16_bits_to_float(input_bits[offset + index])
                )
            )
            for index in range(hidden)
        ]
    return result


def build_correctness_record(
    *,
    model: str,
    component: str,
    input_bits: Sequence[int],
    observed_bits: Sequence[int],
    input_descriptor: Mapping[str, Any],
    output_descriptor: Mapping[str, Any],
    input_tensor_sha256: str,
) -> dict[str, Any]:
    """Build a timing-excluded, independently recomputable BF16 certificate."""

    if model not in MODEL_SHAPES or component not in COMPONENTS:
        raise PhaseMapLUTError("unknown correctness model/component")
    shape = MODEL_SHAPES[model]
    hidden, top_k = int(shape["hidden"]), int(shape["top_k"])
    input_values = list(input_bits)
    observed = list(observed_bits)
    if _tensor_sha256_from_bf16_bits(input_values, input_descriptor) != input_tensor_sha256:
        raise PhaseMapLUTError("correctness input bits do not match frozen tensor hash")
    reference = _reference_bits(component, input_values, hidden, top_k)
    if len(observed) != hidden or len(reference) != hidden:
        raise PhaseMapLUTError("correctness output census mismatch")
    reference_values = [_bf16_bits_to_float(value) for value in reference]
    observed_values = [_bf16_bits_to_float(value) for value in observed]
    absolute = [abs(left - right) for left, right in zip(observed_values, reference_values)]
    relative = [
        difference / max(abs(reference_value), 1e-12)
        for difference, reference_value in zip(absolute, reference_values)
    ]
    bitwise_equal = observed == reference
    payload = {
        "model_key": model,
        "model_revision": shape["model_revision"],
        "component": component,
        "dtype": "torch.bfloat16",
        "rows": 1,
        "primitive_invocations": 1,
        "timing_exclusion": "EXECUTED_ON_CAPTURE_STREAM_BEFORE_WARMUP_OUTSIDE_TIMED_REGION",
        "comparison_rule": "BF16_BITWISE_EXACT",
        "max_abs_tolerance": 0.0,
        "max_rel_tolerance": 0.0,
        "input_tensor_sha256": input_tensor_sha256,
        "input_descriptor": dict(input_descriptor),
        "input_descriptor_sha256": object_sha256(input_descriptor),
        "output_descriptor": dict(output_descriptor),
        "output_descriptor_sha256": object_sha256(output_descriptor),
        "input_bf16_bits": input_values,
        "reference_bf16_bits": reference,
        "observed_bf16_bits": observed,
        "input_bits_sha256": object_sha256(input_values),
        "reference_bits_sha256": object_sha256(reference),
        "observed_bits_sha256": object_sha256(observed),
        "reference_tensor_sha256": _tensor_sha256_from_bf16_bits(
            reference, output_descriptor
        ),
        "observed_tensor_sha256": _tensor_sha256_from_bf16_bits(
            observed, output_descriptor
        ),
        "max_abs_error": max(absolute, default=0.0),
        "max_rel_error": max(relative, default=0.0),
        "bitwise_equal": bitwise_equal,
        "combine_sibling_count": top_k if component == "canonical_combine" else None,
        "combine_passes": 1 if component == "canonical_combine" else 0,
        "passed": bitwise_equal and max(absolute, default=0.0) == 0.0,
    }
    return payload


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_raw_row(row: Mapping[str, Any]) -> None:
    model = row.get("model_key")
    component = row.get("component")
    cuda_us = row.get("cuda_event_us")
    wall_us = row.get("wall_time_us")
    if (
        model not in MODEL_SHAPES
        or component not in COMPONENTS
        or row.get("model_revision") != MODEL_SHAPES[str(model)]["model_revision"]
        or row.get("hidden") != MODEL_SHAPES[str(model)]["hidden"]
        or row.get("top_k") != MODEL_SHAPES[str(model)]["top_k"]
        or row.get("dtype") != "torch.bfloat16"
        or row.get("rows") != 1
        or row.get("primitive_invocations") != 1
        or row.get("phase") not in {"warmup", "measured"}
        or type(row.get("trial_index")) is not int
        or type(row.get("execution_ordinal")) is not int
        or type(row.get("stream_id")) is not int
        or not isinstance(cuda_us, (int, float))
        or isinstance(cuda_us, bool)
        or not math.isfinite(float(cuda_us))
        or float(cuda_us) <= 0
        or not isinstance(wall_us, (int, float))
        or isinstance(wall_us, bool)
        or not math.isfinite(float(wall_us))
        or float(wall_us) <= 0
        or float(cuda_us) > float(wall_us)
        or row.get("source") != SOURCES[str(component)]
        or row.get("evidence_boundary") != CUDA_BOUNDARY
        or not _is_sha256(row.get("producer_source_sha256"))
        or not _is_sha256(row.get("protocol_sha256"))
        or not _is_sha256(row.get("input_tensor_sha256"))
        or not _is_sha256(row.get("input_descriptor_sha256"))
        or not _is_sha256(row.get("output_descriptor_sha256"))
    ):
        raise PhaseMapLUTError("raw CUDA row violates frozen schema")


def _trial_census(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    warmups = [row for row in rows if row["phase"] == "warmup"]
    measured = [row for row in rows if row["phase"] == "measured"]
    if (
        len(warmups) != WARMUPS
        or len(measured) != MEASURED
        or {row["trial_index"] for row in warmups} != set(range(WARMUPS))
        or {row["trial_index"] for row in measured} != set(range(MEASURED))
    ):
        raise PhaseMapLUTError("20+100 trial census mismatch")
    return warmups, measured


def validate_and_summarize(raw_trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = len(MODEL_SHAPES) * len(COMPONENTS) * (WARMUPS + MEASURED)
    if len(raw_trials) != expected:
        raise PhaseMapLUTError("raw LUT census mismatch")

    identities: set[tuple[Any, ...]] = set()
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    ordinals: list[int] = []
    for row in raw_trials:
        _validate_raw_row(row)
        identity = (row["model_key"], row["component"], row["phase"], row["trial_index"])
        if identity in identities:
            raise PhaseMapLUTError("duplicate raw trial identity")
        identities.add(identity)
        groups.setdefault((row["model_key"], row["component"]), []).append(row)
        ordinals.append(row["execution_ordinal"])

    if ordinals != list(range(expected)):
        raise PhaseMapLUTError("raw execution ordinals are not contiguous and ordered")
    expected_groups = {(model, component) for model in MODEL_SHAPES for component in COMPONENTS}
    if set(groups) != expected_groups:
        raise PhaseMapLUTError("primitive surface is incomplete")

    summary: list[dict[str, Any]] = []
    immutable = (
        "model_revision",
        "hidden",
        "top_k",
        "dtype",
        "rows",
        "primitive_invocations",
        "stream_id",
        "source",
        "evidence_boundary",
        "producer_source_sha256",
        "protocol_sha256",
        "input_tensor_sha256",
        "input_descriptor_sha256",
        "output_descriptor_sha256",
    )
    for model, shape in MODEL_SHAPES.items():
        for component in COMPONENTS:
            rows = groups[(model, component)]
            _warmups, measured = _trial_census(rows)
            if any(len({row.get(field) for row in rows}) != 1 for field in immutable):
                raise PhaseMapLUTError("primitive immutable field drift")
            cuda = [float(row["cuda_event_us"]) for row in measured]
            wall = [float(row["wall_time_us"]) for row in measured]
            summary.append(
                {
                    "model_key": model,
                    "model_revision": shape["model_revision"],
                    "hidden": shape["hidden"],
                    "top_k": shape["top_k"],
                    "dtype": "torch.bfloat16",
                    "rows": 1,
                    "component": component,
                    "warmup_count": WARMUPS,
                    "measured_count": MEASURED,
                    "median_cuda_event_us": float(statistics.median(cuda)),
                    "p95_cuda_event_us": nearest_rank_p95(cuda),
                    "max_cuda_event_us": max(cuda),
                    "median_wall_time_us": float(statistics.median(wall)),
                    "p95_wall_time_us": nearest_rank_p95(wall),
                    "max_wall_time_us": max(wall),
                    "source": SOURCES[component],
                    "evidence_boundary": CUDA_BOUNDARY,
                    "producer_source_sha256": rows[0]["producer_source_sha256"],
                    "protocol_sha256": rows[0]["protocol_sha256"],
                    "input_tensor_sha256": rows[0]["input_tensor_sha256"],
                    "input_descriptor_sha256": rows[0]["input_descriptor_sha256"],
                    "output_descriptor_sha256": rows[0]["output_descriptor_sha256"],
                }
            )
    return summary


def _validate_capture_environment(environment: Mapping[str, Any]) -> None:
    required_text = (
        "python_executable",
        "python_version",
        "pytorch_version",
        "cuda_version",
        "gpu_uuid",
        "driver_version",
    )
    producer_pid = environment.get("producer_pid")
    before = environment.get("compute_apps_before")
    after = environment.get("compute_apps_after")
    if (
        any(not isinstance(environment.get(field), str) or not environment[field] for field in required_text)
        or environment.get("gpu_name") != "NVIDIA GeForce RTX 5090"
        or not str(environment.get("gpu_uuid", "")).startswith("GPU-")
        or type(producer_pid) is not int
        or producer_pid <= 0
        or not isinstance(before, list)
        or not isinstance(after, list)
    ):
        raise PhaseMapLUTError("RTX 5090 capture environment identity is incomplete")
    for field in ("clock_sm_mhz", "power_limit_w", "temperature_c"):
        value = environment.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (field != "temperature_c" and float(value) <= 0.0)
        ):
            raise PhaseMapLUTError("RTX 5090 capture environment numeric field is invalid")
    for census_name, rows in (("before", before), ("after", after)):
        seen: set[tuple[int, str]] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise PhaseMapLUTError(f"compute-app {census_name} census is malformed")
            pid = row.get("pid")
            used = row.get("used_gpu_memory_mib")
            identity = (pid, str(row.get("process_name")))
            if (
                set(row) != {"pid", "gpu_uuid", "process_name", "used_gpu_memory_mib"}
                or pid != producer_pid
                or row.get("gpu_uuid") != environment["gpu_uuid"]
                or not isinstance(row.get("process_name"), str)
                or not row["process_name"]
                or isinstance(used, bool)
                or not isinstance(used, (int, float))
                or not math.isfinite(float(used))
                or float(used) < 0.0
                or identity in seen
            ):
                raise PhaseMapLUTError(
                    f"compute-app {census_name} census violates producer-only isolation"
                )
            seen.add(identity)
def validate_artifact(value: Mapping[str, Any]) -> None:
    validate_self_hash(value)
    protocol_sha = file_sha256(PROTOCOL)
    producer_sha = file_sha256(Path(__file__))
    source_manifest = value.get("source_manifest")
    expected_cut = {
        "bandwidth_gbps": 200,
        "source": "ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA",
        "payload_formula": "hidden*2",
        "descriptor_bytes": 16,
        "alignment_bytes": 16,
    }
    model_inputs = value.get("model_inputs")
    environment = value.get("environment")
    raw_trials = value.get("raw_trials")
    correctness = value.get("correctness_certificates")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != STATUS
        or value.get("scientific_result") is not False
        or value.get("protocol_file") != str(PROTOCOL)
        or value.get("protocol_sha256") != protocol_sha
        or value.get("producer_source_sha256") != producer_sha
        or value.get("warmups_per_point") != WARMUPS
        or value.get("measured_trials_per_point") != MEASURED
        or value.get("row_count") != 1
        or value.get("components") != list(COMPONENTS)
        or source_manifest
        != {
            "protocol": {"path": str(PROTOCOL), "sha256": protocol_sha},
            "producer": {"path": str(Path(__file__).resolve()), "sha256": producer_sha},
        }
        or value.get("shared_cut") != expected_cut
        or not isinstance(model_inputs, Mapping)
        or set(model_inputs) != set(MODEL_SHAPES)
        or not isinstance(environment, Mapping)
        or not isinstance(raw_trials, list)
        or not isinstance(value.get("summary"), list)
        or not isinstance(correctness, list)
        or len(correctness) != len(MODEL_SHAPES) * len(COMPONENTS)
    ):
        raise PhaseMapLUTError("artifact provenance or frozen contract mismatch")
    _validate_capture_environment(environment)

    for model, shape in MODEL_SHAPES.items():
        inputs = model_inputs[model]
        if (
            not isinstance(inputs, Mapping)
            or inputs.get("model_revision") != shape["model_revision"]
            or inputs.get("hidden") != shape["hidden"]
            or inputs.get("top_k") != shape["top_k"]
            or inputs.get("dtype") != "torch.bfloat16"
            or inputs.get("rows") != 1
            or not _is_sha256(inputs.get("row_tensor_sha256"))
            or not _is_sha256(inputs.get("siblings_tensor_sha256"))
            or not isinstance(inputs.get("row_descriptor"), Mapping)
            or not isinstance(inputs.get("siblings_descriptor"), Mapping)
            or inputs.get("row_descriptor_sha256") != object_sha256(inputs["row_descriptor"])
            or inputs.get("siblings_descriptor_sha256") != object_sha256(inputs["siblings_descriptor"])
        ):
            raise PhaseMapLUTError("model input descriptor provenance mismatch")
        for row in raw_trials:
            if row.get("model_key") != model:
                continue
            combine = row.get("component") == "canonical_combine"
            if (
                row.get("producer_source_sha256") != producer_sha
                or row.get("protocol_sha256") != protocol_sha
                or row.get("input_tensor_sha256")
                != (inputs["siblings_tensor_sha256"] if combine else inputs["row_tensor_sha256"])
                or row.get("input_descriptor_sha256")
                != (inputs["siblings_descriptor_sha256"] if combine else inputs["row_descriptor_sha256"])
                or row.get("output_descriptor_sha256") != inputs["row_descriptor_sha256"]
            ):
                raise PhaseMapLUTError("raw trial is not bound to artifact provenance")

        by_component = {
            row.get("component"): row
            for row in correctness
            if isinstance(row, Mapping) and row.get("model_key") == model
        }
        if set(by_component) != set(COMPONENTS):
            raise PhaseMapLUTError("correctness primitive surface is incomplete")
        for component in COMPONENTS:
            row = by_component[component]
            combine = component == "canonical_combine"
            input_descriptor = (
                inputs["siblings_descriptor"] if combine else inputs["row_descriptor"]
            )
            input_tensor_sha = (
                inputs["siblings_tensor_sha256"] if combine else inputs["row_tensor_sha256"]
            )
            try:
                expected = build_correctness_record(
                    model=model,
                    component=component,
                    input_bits=row["input_bf16_bits"],
                    observed_bits=row["observed_bf16_bits"],
                    input_descriptor=input_descriptor,
                    output_descriptor=inputs["row_descriptor"],
                    input_tensor_sha256=input_tensor_sha,
                )
            except (KeyError, TypeError, ValueError, PhaseMapLUTError) as exc:
                raise PhaseMapLUTError("correctness certificate cannot be recomputed") from exc
            if dict(row) != expected or row.get("passed") is not True:
                raise PhaseMapLUTError("correctness certificate mismatch or failed primitive")

    recomputed = validate_and_summarize(raw_trials)
    if value["summary"] != recomputed:
        raise PhaseMapLUTError("artifact summary does not match raw trials")


def _query_compute_apps() -> list[dict[str, Any]]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PhaseMapLUTError("compute-app query failed") from exc
    if not output or "No running processes found" in output:
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise PhaseMapLUTError("malformed compute-app query")
        try:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "gpu_uuid": fields[1],
                    "process_name": fields[2],
                    "used_gpu_memory_mib": float(fields[3]),
                }
            )
        except ValueError as exc:
            raise PhaseMapLUTError("invalid compute-app row") from exc
    return sorted(rows, key=lambda row: (row["pid"], row["process_name"]))


def _reject_foreign_apps(rows: Sequence[Mapping[str, Any]]) -> None:
    foreign = [dict(row) for row in rows if row.get("pid") != os.getpid()]
    if foreign:
        raise PhaseMapLUTError(f"foreign GPU processes are active: {foreign}")


def capture_environment(
    torch: Any,
    before: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,clocks.sm,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PhaseMapLUTError("GPU environment query failed") from exc
    fields = [field.strip() for field in output.split(",")]
    after = _query_compute_apps()
    _reject_foreign_apps(after)
    if (
        len(fields) != 6
        or fields[1] != "NVIDIA GeForce RTX 5090"
        or torch.cuda.get_device_name(0) != fields[1]
        or torch.version.cuda is None
    ):
        raise PhaseMapLUTError("RTX 5090 identity mismatch")
    try:
        clock, power_limit, temperature = (float(fields[index]) for index in (3, 4, 5))
    except ValueError as exc:
        raise PhaseMapLUTError("non-numeric GPU environment") from exc
    return {
        "producer_pid": os.getpid(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_uuid": fields[0],
        "gpu_name": fields[1],
        "driver_version": fields[2],
        "clock_sm_mhz": clock,
        "power_limit_w": power_limit,
        "temperature_c": temperature,
        "compute_apps_before": [dict(row) for row in before],
        "compute_apps_after": after,
        "host_platform": platform.platform(),
        "host_processor": platform.processor(),
    }


def _make_model_primitives(
    torch: Any,
    model: str,
    stream: Any,
) -> tuple[dict[str, Callable[[], Any]], dict[str, Any], dict[str, Any]]:
    shape = MODEL_SHAPES[model]
    hidden, top_k = int(shape["hidden"]), int(shape["top_k"])
    with torch.cuda.stream(stream):
        row = (torch.arange(hidden, device="cuda:0", dtype=torch.int64) % 31).to(torch.bfloat16).reshape(1, hidden)
        siblings = (torch.arange(top_k * hidden, device="cuda:0", dtype=torch.int64) % 37).to(torch.bfloat16).reshape(1, top_k, hidden)
        row_index = torch.zeros(1, device="cuda:0", dtype=torch.int64)
        unpacked = torch.empty_like(row)
        accumulator = torch.empty_like(row)
    stream.synchronize()

    def pack() -> Any:
        return torch.index_select(row, 0, row_index)

    def unpack() -> Any:
        unpacked.index_copy_(0, row_index, row)
        return unpacked

    def combine() -> Any:
        accumulator.copy_(siblings[:, 0, :])
        for slot in range(1, top_k):
            torch.add(accumulator, siblings[:, slot, :], out=accumulator)
        return accumulator

    row_descriptor = tensor_descriptor(row)
    sibling_descriptor = tensor_descriptor(siblings)
    return (
        {"sender_pack": pack, "receiver_unpack": unpack, "canonical_combine": combine},
        {
            "row_tensor_sha256": tensor_sha256(row),
            "siblings_tensor_sha256": tensor_sha256(siblings),
            "row_descriptor": row_descriptor,
            "row_descriptor_sha256": object_sha256(row_descriptor),
            "siblings_descriptor": sibling_descriptor,
            "siblings_descriptor_sha256": object_sha256(sibling_descriptor),
        },
        {
            "row": row,
            "siblings": siblings,
            "unpacked": unpacked,
            "accumulator": accumulator,
        },
    )


def _capture_model_correctness(
    torch: Any,
    model: str,
    stream: Any,
    primitives: Mapping[str, Callable[[], Any]],
    descriptors: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run one untimed invocation per primitive and materialize its certificate."""

    with torch.cuda.stream(stream), torch.inference_mode():
        observed = {
            component: primitives[component]() for component in COMPONENTS
        }
    stream.synchronize()
    row_bits = _bf16_bits(tensors["row"])
    sibling_bits = _bf16_bits(tensors["siblings"])
    rows = []
    for component in COMPONENTS:
        combine = component == "canonical_combine"
        rows.append(
            build_correctness_record(
                model=model,
                component=component,
                input_bits=sibling_bits if combine else row_bits,
                observed_bits=_bf16_bits(observed[component]),
                input_descriptor=(
                    descriptors["siblings_descriptor"]
                    if combine
                    else descriptors["row_descriptor"]
                ),
                output_descriptor=descriptors["row_descriptor"],
                input_tensor_sha256=(
                    descriptors["siblings_tensor_sha256"]
                    if combine
                    else descriptors["row_tensor_sha256"]
                ),
            )
        )
    return rows


def _measure_single(
    torch: Any,
    stream: Any,
    primitive: Callable[[], Any],
) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    with torch.cuda.stream(stream), torch.inference_mode():
        start.record(stream)
        primitive()
        end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0, (time.perf_counter_ns() - wall_start) / 1000.0


def run() -> dict[str, Any]:
    if not PROTOCOL.is_file():
        raise PhaseMapLUTError("frozen PhaseMap protocol is missing")
    try:
        import torch
    except ImportError as exc:
        raise PhaseMapLUTError("PyTorch is unavailable") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseMapLUTError("exactly one CUDA device is required")

    before = _query_compute_apps()
    _reject_foreign_apps(before)
    protocol_sha = file_sha256(PROTOCOL)
    producer_path = Path(__file__).resolve()
    producer_sha = file_sha256(producer_path)
    raw: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    model_inputs: dict[str, Any] = {}
    ordinal = 0

    for model, shape in MODEL_SHAPES.items():
        stream = torch.cuda.Stream(device=0)
        primitives, descriptors, tensors = _make_model_primitives(torch, model, stream)
        model_inputs[model] = {**dict(shape), "dtype": "torch.bfloat16", "rows": 1, **descriptors}
        correctness.extend(
            _capture_model_correctness(
                torch, model, stream, primitives, descriptors, tensors
            )
        )
        for phase, rounds in (("warmup", WARMUPS), ("measured", MEASURED)):
            for trial_index in range(rounds):
                offset = trial_index % len(COMPONENTS)
                for component in COMPONENTS[offset:] + COMPONENTS[:offset]:
                    cuda_us, wall_us = _measure_single(torch, stream, primitives[component])
                    combine = component == "canonical_combine"
                    raw.append(
                        {
                            "model_key": model,
                            "model_revision": shape["model_revision"],
                            "hidden": shape["hidden"],
                            "top_k": shape["top_k"],
                            "dtype": "torch.bfloat16",
                            "rows": 1,
                            "component": component,
                            "primitive_invocations": 1,
                            "phase": phase,
                            "trial_index": trial_index,
                            "execution_ordinal": ordinal,
                            "cuda_event_us": cuda_us,
                            "wall_time_us": wall_us,
                            "stream_id": int(stream.cuda_stream),
                            "source": SOURCES[component],
                            "evidence_boundary": CUDA_BOUNDARY,
                            "producer_source_sha256": producer_sha,
                            "protocol_sha256": protocol_sha,
                            "input_tensor_sha256": descriptors["siblings_tensor_sha256"] if combine else descriptors["row_tensor_sha256"],
                            "input_descriptor_sha256": descriptors["siblings_descriptor_sha256"] if combine else descriptors["row_descriptor_sha256"],
                            "output_descriptor_sha256": descriptors["row_descriptor_sha256"],
                        }
                    )
                    ordinal += 1

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "scientific_result": False,
        "protocol_file": str(PROTOCOL),
        "protocol_sha256": protocol_sha,
        "producer_source_sha256": producer_sha,
        "source_manifest": {
            "protocol": {"path": str(PROTOCOL), "sha256": protocol_sha},
            "producer": {"path": str(producer_path), "sha256": producer_sha},
        },
        "warmups_per_point": WARMUPS,
        "measured_trials_per_point": MEASURED,
        "row_count": 1,
        "components": list(COMPONENTS),
        "shared_cut": {
            "bandwidth_gbps": 200,
            "source": "ANALYTIC_NETWORK_L2_PROXY_NOT_RDMA",
            "payload_formula": "hidden*2",
            "descriptor_bytes": 16,
            "alignment_bytes": 16,
        },
        "model_inputs": model_inputs,
        "correctness_certificates": correctness,
        "environment": capture_environment(torch, before),
        "raw_trials": raw,
        "summary": validate_and_summarize(raw),
    }
    result = add_self_hash(payload)
    validate_artifact(result)
    return result


def write_json_atomic_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    validate_artifact(value)
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise PhaseMapLUTError("refusing to overwrite output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise PhaseMapLUTError("output parent may not be a symlink")
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise PhaseMapLUTError("atomic write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PhaseMapLUTError("output appeared during atomic publish") from exc
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
        raise PhaseMapLUTError("refusing to overwrite output")
    result = run()
    write_json_atomic_no_overwrite(args.output, result)
    print(json.dumps({"output": str(args.output), "artifact_sha256": result["artifact_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
