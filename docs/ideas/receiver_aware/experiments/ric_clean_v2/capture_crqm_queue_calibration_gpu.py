#!/usr/bin/env python3
"""Capture the frozen CRQM virtual-receiver queue-drain calibration.

This is an exploratory RTX 5090 calibration producer, not a serving or
network benchmark. It measures a serial CUDA-stream queue containing zero or
more copies of the frozen BF16 row-1 receiver-unpack primitive. It never emits
a scientific result.
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
PROTOCOL = HERE.parents[1] / "CRQM_Phase2_FrozenPilot_2026-07-23.md"
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
DEPTHS = (0, 1, 2, 4, 8, 16)
WARMUPS = 20
MEASURED = 100
SOURCE = "measured_5090_cuda_serial_virtual_receiver"
EVIDENCE_BOUNDARY = "SINGLE_GPU_SERIAL_CUDA_STREAM_NOT_NIC_NOT_RDMA_NOT_MULTI_RANK"
MEASUREMENT_SEMANTICS = "BACKLOG_ONLY_D_PRIOR_RECEIVER_PRIMITIVES_CANDIDATE_EXCLUDED"


class CRQMCalibrationError(RuntimeError):
    """The frozen calibration contract or output integrity failed."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def add_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    if "artifact_sha256" in value:
        raise CRQMCalibrationError("artifact payload already contains a self hash")
    result = dict(value)
    result["artifact_sha256"] = object_sha256(result)
    return result


def validate_self_hash(value: Mapping[str, Any]) -> None:
    expected = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if not isinstance(expected, str) or expected != object_sha256(payload):
        raise CRQMCalibrationError("artifact self hash mismatch")


def nearest_rank_p95(values: Sequence[float]) -> float:
    if not values:
        raise CRQMCalibrationError("cannot summarize an empty sample")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _validate_trial(row: Mapping[str, Any]) -> None:
    model_key = row.get("model_key")
    depth = row.get("queue_depth")
    phase = row.get("phase")
    cuda_us = row.get("cuda_event_us")
    wall_us = row.get("wall_time_us")
    if (
        model_key not in MODEL_SHAPES
        or row.get("model_revision") != MODEL_SHAPES[str(model_key)]["model_revision"]
        or row.get("hidden") != MODEL_SHAPES[str(model_key)]["hidden"]
        or row.get("top_k") != MODEL_SHAPES[str(model_key)]["top_k"]
        or row.get("dtype") != "torch.bfloat16"
        or type(depth) is not int
        or depth not in DEPTHS
        or row.get("primitive_invocations") != depth
        or row.get("candidate_included") is not False
        or row.get("measurement_semantics") != MEASUREMENT_SEMANTICS
        or phase not in {"warmup", "measured"}
        or type(row.get("trial_index")) is not int
        or row["trial_index"] < 0
        or type(row.get("execution_ordinal")) is not int
        or row["execution_ordinal"] < 0
        or not isinstance(cuda_us, (int, float))
        or isinstance(cuda_us, bool)
        or not math.isfinite(float(cuda_us))
        or float(cuda_us) < 0
        or (depth > 0 and float(cuda_us) <= 0)
        or not isinstance(wall_us, (int, float))
        or isinstance(wall_us, bool)
        or not math.isfinite(float(wall_us))
        or float(wall_us) <= 0
        or float(cuda_us) > float(wall_us)
        or row.get("source") != SOURCE
        or row.get("evidence_boundary") != EVIDENCE_BOUNDARY
        or not isinstance(row.get("stream_id"), int)
        or not isinstance(row.get("producer_source_sha256"), str)
        or len(row["producer_source_sha256"]) != 64
    ):
        raise CRQMCalibrationError("raw trial violates the frozen schema")


def validate_and_summarize(raw_trials: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected_total = len(MODEL_SHAPES) * len(DEPTHS) * (WARMUPS + MEASURED)
    if len(raw_trials) != expected_total:
        raise CRQMCalibrationError("raw trial census mismatch")
    ordinals: list[int] = []
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    identities: set[tuple[str, int, str, int]] = set()
    for row in raw_trials:
        _validate_trial(row)
        identity = (
            str(row["model_key"]),
            int(row["queue_depth"]),
            str(row["phase"]),
            int(row["trial_index"]),
        )
        if identity in identities:
            raise CRQMCalibrationError("duplicate raw trial identity")
        identities.add(identity)
        ordinals.append(int(row["execution_ordinal"]))
        groups.setdefault((str(row["model_key"]), int(row["queue_depth"])), []).append(row)
    if ordinals != list(range(expected_total)):
        raise CRQMCalibrationError("execution ordinals are not contiguous and ordered")
    expected_groups = {(model, depth) for model in MODEL_SHAPES for depth in DEPTHS}
    if set(groups) != expected_groups:
        raise CRQMCalibrationError("model/depth surface is incomplete")

    summaries: list[dict[str, Any]] = []
    for model_key in MODEL_SHAPES:
        shape = MODEL_SHAPES[model_key]
        for depth in DEPTHS:
            rows = groups[(model_key, depth)]
            warmups = [row for row in rows if row["phase"] == "warmup"]
            measured = [row for row in rows if row["phase"] == "measured"]
            if (
                len(warmups) != WARMUPS
                or len(measured) != MEASURED
                or {row["trial_index"] for row in warmups} != set(range(WARMUPS))
                or {row["trial_index"] for row in measured} != set(range(MEASURED))
            ):
                raise CRQMCalibrationError("20+100 point census mismatch")
            immutable = (
                "model_revision", "hidden", "top_k", "dtype", "primitive_invocations",
                "stream_id", "source", "evidence_boundary", "producer_source_sha256",
                "packed_tensor_sha256", "candidate_included",
                "measurement_semantics",
            )
            if any(len({row.get(field) for row in rows}) != 1 for field in immutable):
                raise CRQMCalibrationError("immutable trial field drift")
            cuda = [float(row["cuda_event_us"]) for row in measured]
            wall = [float(row["wall_time_us"]) for row in measured]
            summaries.append(
                {
                    "model_key": model_key,
                    "model_revision": shape["model_revision"],
                    "hidden": shape["hidden"],
                    "top_k": shape["top_k"],
                    "dtype": "torch.bfloat16",
                    "queue_depth": depth,
                    "primitive_invocations": depth,
                    "warmup_count": WARMUPS,
                    "measured_count": MEASURED,
                    "median_cuda_event_us": float(statistics.median(cuda)),
                    "p95_cuda_event_us": nearest_rank_p95(cuda),
                    "max_cuda_event_us": max(cuda),
                    "median_wall_time_us": float(statistics.median(wall)),
                    "p95_wall_time_us": nearest_rank_p95(wall),
                    "max_wall_time_us": max(wall),
                    # Depth zero still records the real event/wall harness above,
                    # but it represents no prior receiver work by definition.
                    "backlog_only_queue_work_us": 0.0 if depth == 0 else float(statistics.median(cuda)),
                    "candidate_included": False,
                    "measurement_semantics": MEASUREMENT_SEMANTICS,
                    "source": SOURCE,
                    "evidence_boundary": EVIDENCE_BOUNDARY,
                    "producer_source_sha256": rows[0]["producer_source_sha256"],
                    "packed_tensor_sha256": rows[0]["packed_tensor_sha256"],
                }
            )
    return summaries


def _query_compute_apps() -> list[dict[str, Any]]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
                "-i", "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise CRQMCalibrationError("GPU compute-app query failed") from exc
    if not output or "No running processes found" in output:
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise CRQMCalibrationError("malformed compute-app query")
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
            raise CRQMCalibrationError("invalid compute-app row") from exc
    return sorted(rows, key=lambda row: (row["pid"], row["process_name"]))


def _reject_foreign_compute_apps(rows: Sequence[Mapping[str, Any]]) -> None:
    foreign = [dict(row) for row in rows if row.get("pid") != os.getpid()]
    if foreign:
        raise CRQMCalibrationError(f"foreign GPU processes are active: {foreign}")


def capture_environment(torch: Any, before: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,clocks.sm,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
                "-i", "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise CRQMCalibrationError("GPU identity query failed") from exc
    fields = [field.strip() for field in output.split(",")]
    after = _query_compute_apps()
    _reject_foreign_compute_apps(after)
    if (
        len(fields) != 6
        or fields[1] != "NVIDIA GeForce RTX 5090"
        or torch.cuda.get_device_name(0) != fields[1]
        or torch.version.cuda is None
    ):
        raise CRQMCalibrationError("formal RTX 5090 identity mismatch")
    try:
        numeric = [float(fields[index]) for index in (3, 4, 5)]
    except ValueError as exc:
        raise CRQMCalibrationError("non-numeric GPU environment field") from exc
    return {
        "producer_pid": os.getpid(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_uuid": fields[0],
        "gpu_name": fields[1],
        "driver_version": fields[2],
        "clock_sm_mhz": numeric[0],
        "power_limit_w": numeric[1],
        "temperature_c": numeric[2],
        "compute_apps_before": [dict(row) for row in before],
        "compute_apps_after": after,
    }


def _tensor_sha256(tensor: Any) -> str:
    original = tensor.detach().cpu()
    contiguous = original.contiguous()
    raw = contiguous.view(__import__("torch").uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(str(tuple(original.shape)).encode())
    digest.update(b"\0")
    digest.update(str(original.dtype).encode())
    digest.update(b"\0")
    digest.update(str(tuple(original.stride())).encode())
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _make_primitive(torch: Any, model_key: str, stream: Any) -> tuple[Callable[[], None], dict[str, Any]]:
    shape = MODEL_SHAPES[model_key]
    hidden = int(shape["hidden"])
    with torch.cuda.stream(stream):
        packed = (torch.arange(hidden, device="cuda:0", dtype=torch.int64) % 31).to(torch.bfloat16).reshape(1, hidden)
        reverse_index = torch.arange(0, -1, -1, device="cuda:0", dtype=torch.int64)
        unpacked = torch.empty_like(packed)
    stream.synchronize()

    def primitive() -> None:
        unpacked.index_copy_(0, reverse_index, packed)

    return primitive, {
        "packed_tensor_sha256": _tensor_sha256(packed),
    }


def _measure_trial(torch: Any, stream: Any, primitive: Callable[[], None], depth: int) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    with torch.cuda.stream(stream), torch.inference_mode():
        start.record(stream)
        for _ in range(depth):
            primitive()
        end.record(stream)
    end.synchronize()
    wall_us = (time.perf_counter_ns() - wall_start) / 1000.0
    cuda_us = float(start.elapsed_time(end)) * 1000.0
    return cuda_us, wall_us


def run() -> dict[str, Any]:
    if not PROTOCOL.is_file():
        raise CRQMCalibrationError("frozen protocol is missing")
    try:
        import torch
    except ImportError as exc:
        raise CRQMCalibrationError("PyTorch is unavailable") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise CRQMCalibrationError("exactly one CUDA device is required")
    before = _query_compute_apps()
    _reject_foreign_compute_apps(before)
    producer_sha = file_sha256(Path(__file__))
    raw_trials: list[dict[str, Any]] = []
    model_inputs: dict[str, Any] = {}
    ordinal = 0
    for model_key, shape in MODEL_SHAPES.items():
        stream = torch.cuda.Stream(device=0)
        primitive, tensor_hashes = _make_primitive(torch, model_key, stream)
        model_inputs[model_key] = {**dict(shape), "dtype": "torch.bfloat16", **tensor_hashes}
        for phase, rounds in (("warmup", WARMUPS), ("measured", MEASURED)):
            for trial_index in range(rounds):
                rotated = DEPTHS[trial_index % len(DEPTHS):] + DEPTHS[:trial_index % len(DEPTHS)]
                for depth in rotated:
                    cuda_us, wall_us = _measure_trial(torch, stream, primitive, depth)
                    raw_trials.append(
                        {
                            "model_key": model_key,
                            "model_revision": shape["model_revision"],
                            "hidden": shape["hidden"],
                            "top_k": shape["top_k"],
                            "dtype": "torch.bfloat16",
                            "queue_depth": depth,
                            "primitive_invocations": depth,
                            "candidate_included": False,
                            "measurement_semantics": MEASUREMENT_SEMANTICS,
                            "phase": phase,
                            "trial_index": trial_index,
                            "execution_ordinal": ordinal,
                            "cuda_event_us": cuda_us,
                            "wall_time_us": wall_us,
                            "stream_id": int(stream.cuda_stream),
                            "source": SOURCE,
                            "evidence_boundary": EVIDENCE_BOUNDARY,
                            "producer_source_sha256": producer_sha,
                            **tensor_hashes,
                        }
                    )
                    ordinal += 1
    summary = validate_and_summarize(raw_trials)
    environment = capture_environment(torch, before)
    return add_self_hash(
        {
            "schema_version": "crqm-queue-drain-calibration-v1",
            "status": "EXPLORATORY_CALIBRATION_INPUT_ONLY",
            "scientific_result": False,
            "evidence_boundary": EVIDENCE_BOUNDARY,
            "protocol_file": str(PROTOCOL),
            "protocol_sha256": file_sha256(PROTOCOL),
            "producer_source_sha256": producer_sha,
            "frozen_depths": list(DEPTHS),
            "warmups_per_point": WARMUPS,
            "measured_trials_per_point": MEASURED,
            "primitive": "BF16_ROW1_INDEX_COPY_RECEIVER_UNPACK_PER_CONTRIBUTION",
            "measurement_semantics": MEASUREMENT_SEMANTICS,
            "summary_consumer_field": "backlog_only_queue_work_us",
            "model_inputs": model_inputs,
            "environment": environment,
            "raw_trials": raw_trials,
            "summary": summary,
        }
    )


def write_json_atomic_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    validate_self_hash(value)
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise CRQMCalibrationError("refusing to overwrite output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise CRQMCalibrationError("output parent identity mismatch")
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o444)
    except FileExistsError as exc:
        raise CRQMCalibrationError("temporary output already exists") from exc
    try:
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise CRQMCalibrationError("atomic output write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CRQMCalibrationError("output appeared during atomic publish") from exc
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
        raise CRQMCalibrationError("refusing to overwrite output")
    result = run()
    write_json_atomic_no_overwrite(args.output, result)
    print(json.dumps({"output": str(args.output), "artifact_sha256": result["artifact_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
