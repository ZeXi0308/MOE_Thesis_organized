from __future__ import annotations

"""Run a fail-closed CUDA/NVML capability probe for RouteSlack Gate 0.

This is deliberately not a model, serving, Energy-SLO, or formal Gate result.
It only verifies that one CUDA device can be bound to NVML, that the cumulative
energy counter is readable, and that timestamped power/thermal state can be
recorded without exceeding the frozen sampling-gap limit.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--interval-ms", type=float, default=10.0)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=3)
    return parser.parse_args()


def _safe_nvml(callable_, *args):
    try:
        return callable_(*args)
    except Exception as exc:  # NVML capability differs by driver/GPU.
        return {"unsupported": f"{type(exc).__name__}: {exc}"}


def _telemetry(backend) -> dict[str, object]:
    nvml = backend._pynvml
    handle = backend._handle
    utilization = _safe_nvml(nvml.nvmlDeviceGetUtilizationRates, handle)
    if not isinstance(utilization, dict):
        utilization = {
            "gpu_percent": int(utilization.gpu),
            "memory_percent": int(utilization.memory),
        }
    constraints = _safe_nvml(nvml.nvmlDeviceGetPowerManagementLimitConstraints, handle)
    if not isinstance(constraints, dict):
        constraints = [float(value) / 1000.0 for value in constraints]
    return {
        "timestamp_monotonic_ns": time.monotonic_ns(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "temperature_c": _safe_nvml(
            nvml.nvmlDeviceGetTemperature, handle, nvml.NVML_TEMPERATURE_GPU
        ),
        "graphics_clock_mhz": _safe_nvml(
            nvml.nvmlDeviceGetClockInfo, handle, nvml.NVML_CLOCK_GRAPHICS
        ),
        "sm_clock_mhz": _safe_nvml(
            nvml.nvmlDeviceGetClockInfo, handle, nvml.NVML_CLOCK_SM
        ),
        "memory_clock_mhz": _safe_nvml(
            nvml.nvmlDeviceGetClockInfo, handle, nvml.NVML_CLOCK_MEM
        ),
        "power_limit_w": (
            lambda value: value
            if isinstance(value, dict)
            else float(value) / 1000.0
        )(_safe_nvml(nvml.nvmlDeviceGetPowerManagementLimit, handle)),
        "power_limit_constraints_w": constraints,
        "power_draw_w": backend.read_power_w(),
        "utilization": utilization,
        "clock_throttle_reasons": _safe_nvml(
            nvml.nvmlDeviceGetCurrentClocksThrottleReasons, handle
        ),
        "total_energy_j": backend.read_total_energy_j(),
    }


def main() -> None:
    args = parse_args()
    if args.seconds <= 0 or args.interval_ms <= 0:
        raise SystemExit("seconds and interval-ms must be positive")
    if args.matrix_size <= 0 or args.warmups < 0:
        raise SystemExit("matrix-size must be positive and warmups non-negative")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the GPU meter preflight")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise SystemExit(f"invalid CUDA device index: {args.device}")

    route_row = next(
        candidate / "docs/ideas/energy_slo/route_row_fp8/experiments"
        for candidate in Path(__file__).resolve().parents
        if (candidate / "docs/ideas/energy_slo/route_row_fp8/experiments").is_dir()
    )
    import sys

    sys.path.insert(0, str(route_row))
    from power_accounting import (
        MonotonicNVMLSampler,
        PynvmlPowerBackend,
        assert_matching_gpu_uuid,
        integrate_power_samples,
    )

    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    cuda_uuid = str(properties.uuid)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PynvmlPowerBackend(args.device) as backend:
        assert_matching_gpu_uuid(backend.gpu_uuid, cuda_uuid)
        before = _telemetry(backend)

        size = args.matrix_size
        left = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        for _ in range(args.warmups):
            torch.mm(left, right)
        torch.cuda.synchronize()

        sampler = MonotonicNVMLSampler(
            backend,
            interval_s=args.interval_ms / 1000.0,
            formal=True,
        )
        sampler.start()
        workload_started_ns = time.monotonic_ns()
        deadline_ns = workload_started_ns + int(args.seconds * 1_000_000_000)
        iterations = 0
        while time.monotonic_ns() < deadline_ns:
            torch.mm(left, right)
            iterations += 1
            if iterations % 8 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        workload_ended_ns = time.monotonic_ns()
        trace = sampler.stop()
        after = _telemetry(backend)

    power_integral_j = integrate_power_samples(trace.samples)
    result = {
        "schema": "routeslack-gpu-meter-preflight-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_result": False,
        "scientific_result_eligible": False,
        "gate0_eligible": False,
        "evidence_boundary": (
            "CUDA/NVML capability and sampling-gap probe only; no model route, "
            "serving timeline, matched completion set, SLO, or policy comparison"
        ),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "gpu": {
            "index": args.device,
            "name": properties.name,
            "cuda_uuid": cuda_uuid,
            "nvml_uuid": trace.gpu_uuid,
            "uuid_match": True,
            "total_memory_bytes": int(properties.total_memory),
        },
        "config": {
            "seconds_requested": args.seconds,
            "interval_ms_requested": args.interval_ms,
            "matrix_size": args.matrix_size,
            "dtype": "bfloat16",
            "warmups": args.warmups,
        },
        "workload": {
            "iterations": iterations,
            "started_monotonic_ns": workload_started_ns,
            "ended_monotonic_ns": workload_ended_ns,
            "duration_s": (workload_ended_ns - workload_started_ns) / 1e9,
        },
        "telemetry_before": before,
        "telemetry_after": after,
        "trace": {
            "start_ns": trace.start_ns,
            "end_ns": trace.end_ns,
            "sample_count": len(trace.samples),
            "max_sample_gap_s": max(
                (right.timestamp_ns - left.timestamp_ns) / 1e9
                for left, right in zip(trace.samples, trace.samples[1:])
            ),
            "total_energy_counter_delta_j": trace.total_energy_counter_delta_j,
            "power_integral_j": power_integral_j,
            "samples": [asdict(sample) for sample in trace.samples],
        },
        "checks": {
            "cuda_available": True,
            "uuid_match": True,
            "counter_supported": trace.total_energy_counter_delta_j is not None,
            "sample_gap_within_20ms": True,
            "telemetry_logged": True,
        },
        "remaining_blockers": [
            "no natural continuous-serving producer or per-request KV ownership",
            "no matched completion/output identity or end-to-end measurement window",
            "no two-model service-energy surface or tier intervention",
            "no physical expert-parallel replica actuator",
        ],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("schema", "formal_result", "checks")}, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
