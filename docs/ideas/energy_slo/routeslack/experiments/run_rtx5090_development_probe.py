#!/usr/bin/env python3
"""Fail-closed RTX 5090 measurement-path probe for RouteSlack.

This is deliberately a *development* probe.  It measures two exact-work
dispatch orders over the same synthetic BF16 expert jobs.  Every window uses
one execution for both CUDA-event latency and raw NVML board energy, logs the
power trace plus thermal/clock telemetry, and uses an ABBA schedule with one
frozen repeat denominator.

It is not a natural continuous-serving producer, not an EP experiment, not a
RouteSlack surface, and can never emit a formal result.  Its sole purpose is to
exercise the RTX 5090 measurement plumbing before a real serving backend is
available.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    candidate for candidate in HERE.parents if (candidate / "experiments" / "shared").is_dir()
)
POWER_ACCOUNTING_DIR = (
    REPO_ROOT / "docs" / "ideas" / "energy_slo" / "route_row_fp8" / "experiments"
)
if str(POWER_ACCOUNTING_DIR) not in sys.path:
    sys.path.insert(0, str(POWER_ACCOUNTING_DIR))

from power_accounting import (  # noqa: E402
    MAX_FORMAL_SAMPLE_INTERVAL_S,
    MonotonicNVMLSampler,
    assert_matching_gpu_uuid,
    integrate_power_samples,
    max_sample_gap_s,
)


SCHEMA = "routeslack-rtx5090-development-probe-v1"
VERDICT = "DEVELOPMENT_MEASUREMENT_PATH_ONLY"
ARM_A = "forward_dispatch"
ARM_B = "reverse_dispatch"
ABBA = (ARM_A, ARM_B, ARM_B, ARM_A)


class ProbeError(RuntimeError):
    """Raised when a development artifact would otherwise be ambiguous."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--blocks", type=_positive_int, default=3)
    parser.add_argument("--warmup-repeats", type=_positive_int, default=10)
    parser.add_argument("--minimum-window-seconds", type=_positive_float, default=0.5)
    parser.add_argument("--maximum-repeats", type=_positive_int, default=65536)
    parser.add_argument("--power-sample-interval-seconds", type=_positive_float, default=0.005)
    parser.add_argument("--maximum-observed-power-gap-seconds", type=_positive_float, default=0.020)
    parser.add_argument("--maximum-window-temperature-drift-c", type=_positive_float, default=3.0)
    parser.add_argument("--maximum-run-start-temperature-c", type=_positive_float, default=78.0)
    parser.add_argument("--hidden-size", type=_positive_int, default=1024)
    parser.add_argument("--intermediate-size", type=_positive_int, default=2048)
    parser.add_argument("--row-grid", type=_positive_int, nargs="+", default=(1, 4, 16, 64))
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def abba_schedule(blocks: int) -> tuple[tuple[int, int, str], ...]:
    """Return ``(block, position, arm)`` entries for a strict ABBA schedule."""

    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
        raise ProbeError("ABBA block count must be a positive integer")
    return tuple(
        (block, position, arm)
        for block in range(blocks)
        for position, arm in enumerate(ABBA)
    )


def choose_equal_repeats(
    measure_seconds: Callable[[str, int], float],
    *,
    minimum_window_s: float,
    maximum_repeats: int,
) -> tuple[int, dict[str, float]]:
    """Choose one repeat denominator that reaches the floor for both arms.

    Calibration calls are not independent trials.  The returned denominator is
    frozen before any ABBA window and is identical for every measured arm.
    """

    if not math.isfinite(minimum_window_s) or minimum_window_s <= 0:
        raise ProbeError("minimum window duration must be finite and positive")
    if (
        isinstance(maximum_repeats, bool)
        or not isinstance(maximum_repeats, int)
        or maximum_repeats <= 0
    ):
        raise ProbeError("maximum repeats must be a positive integer")
    repeats = 1
    while True:
        durations = {
            arm: float(measure_seconds(arm, repeats)) for arm in (ARM_A, ARM_B)
        }
        if any(not math.isfinite(value) or value <= 0 for value in durations.values()):
            raise ProbeError("CUDA repeat calibration returned an invalid duration")
        if min(durations.values()) >= minimum_window_s:
            return repeats, durations
        if repeats >= maximum_repeats:
            raise ProbeError(
                "both arms did not reach the minimum window before maximum repeats"
            )
        repeats = min(maximum_repeats, repeats * 2)


@dataclass(frozen=True)
class TimedResult:
    latency_ms: float
    payload: object


@dataclass(frozen=True)
class EnergyEnvelope:
    raw_board_energy_j: float
    energy_source: str
    power_integral_j: float
    counter_energy_j: float | None
    energy_window_start_ns: int
    energy_window_end_ns: int
    host_cuda_record_start_ns: int
    host_cuda_sync_end_ns: int
    power_samples: tuple[Mapping[str, object], ...]
    maximum_power_sample_gap_s: float
    telemetry_start: Mapping[str, object]
    telemetry_end: Mapping[str, object]
    counter_reads: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class SameWindowResult:
    timed: TimedResult
    energy: EnergyEnvelope


def measure_same_window(
    workload: Callable[[], object],
    *,
    timer: Callable[[Callable[[], object]], TimedResult],
    meter: Callable[[Callable[[], None]], EnergyEnvelope],
) -> SameWindowResult:
    """Nest one CUDA-event timing around the one workload invoked by the meter."""

    timed_results: list[TimedResult] = []

    def timed_workload() -> None:
        if timed_results:
            raise ProbeError("energy meter invoked one logical workload more than once")
        timed_results.append(timer(workload))

    energy = meter(timed_workload)
    if len(timed_results) != 1:
        raise ProbeError("energy meter did not invoke exactly one logical workload")
    timed = timed_results[0]
    if not math.isfinite(timed.latency_ms) or timed.latency_ms <= 0:
        raise ProbeError("CUDA-event latency must be finite and positive")
    if not math.isfinite(energy.raw_board_energy_j) or energy.raw_board_energy_j <= 0:
        raise ProbeError("raw board energy must be finite and positive")
    return SameWindowResult(timed=timed, energy=energy)


def _normalise_uuid(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    text = str(value).strip()
    return text if text.upper().startswith("GPU-") else f"GPU-{text}"


class RecordingPowerBackend:
    """Record the raw NVML counter boundaries consumed by the shared sampler."""

    def __init__(self, backend: "NVMLTelemetryBackend") -> None:
        self.backend = backend
        self.counter_reads: list[dict[str, object]] = []

    @property
    def gpu_uuid(self) -> str:
        return self.backend.gpu_uuid

    def read_power_w(self) -> float:
        return self.backend.read_power_w()

    def read_total_energy_j(self) -> float | None:
        before_ns = time.monotonic_ns()
        value = self.backend.read_total_energy_j()
        after_ns = time.monotonic_ns()
        self.counter_reads.append(
            {
                "read_start_ns": before_ns,
                "read_end_ns": after_ns,
                "energy_j": value,
            }
        )
        return value


class NVMLTelemetryBackend:
    """Read-only CUDA/NVML matched backend with required 5090 telemetry."""

    def __init__(self, torch_module: Any, device: object, device_index: int) -> None:
        try:
            import pynvml  # type: ignore
        except ImportError as exc:  # pragma: no cover - remote capability
            raise ProbeError("pynvml is required for RTX 5090 board-energy measurement") from exc
        self.nvml = pynvml
        self.nvml.nvmlInit()
        self._closed = False
        self.device = device
        self.device_index = device_index
        properties = torch_module.cuda.get_device_properties(device)
        self.gpu_name = str(properties.name)
        if "RTX 5090" not in self.gpu_name.upper().replace("GEFORCE ", ""):
            # Keep the literal identity in the error; never silently run on a
            # different GPU and label it a 5090 artifact.
            raise ProbeError(f"RTX 5090 required, observed CUDA device {self.gpu_name!r}")
        cuda_uuid = _normalise_uuid(properties.uuid)
        self.handle = None
        self.nvml_physical_index = None
        self._gpu_uuid = ""
        for nvml_index in range(self.nvml.nvmlDeviceGetCount()):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(nvml_index)
            raw_nvml_uuid = self.nvml.nvmlDeviceGetUUID(handle)
            candidate_uuid = _normalise_uuid(raw_nvml_uuid)
            try:
                assert_matching_gpu_uuid(candidate_uuid, cuda_uuid)
            except RuntimeError:
                continue
            self.handle = handle
            self.nvml_physical_index = nvml_index
            self._gpu_uuid = candidate_uuid
            break
        if self.handle is None:
            raise ProbeError(
                f"no NVML physical device matches CUDA-visible UUID {cuda_uuid!r}"
            )
        assert_matching_gpu_uuid(self._gpu_uuid, cuda_uuid)
        driver = self.nvml.nvmlSystemGetDriverVersion()
        self.driver_version = driver.decode() if isinstance(driver, bytes) else str(driver)

    @property
    def gpu_uuid(self) -> str:
        return self._gpu_uuid

    def read_power_w(self) -> float:
        value = float(self.nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
        if not math.isfinite(value) or value <= 0:
            raise ProbeError("NVML returned invalid instantaneous board power")
        return value

    def read_total_energy_j(self) -> float | None:
        fn = getattr(self.nvml, "nvmlDeviceGetTotalEnergyConsumption", None)
        if fn is None:
            return None
        try:
            return float(fn(self.handle)) / 1000.0
        except self.nvml.NVMLError_NotSupported:
            return None

    def telemetry(self) -> dict[str, object]:
        """Read required temperature, clocks, power, utilization, and limit state."""

        nvml = self.nvml
        utilization = nvml.nvmlDeviceGetUtilizationRates(self.handle)
        power_limit_fn = getattr(nvml, "nvmlDeviceGetEnforcedPowerLimit", None)
        if power_limit_fn is None:
            power_limit_fn = nvml.nvmlDeviceGetPowerManagementLimit
        snapshot = {
            "timestamp_ns": time.monotonic_ns(),
            "temperature_c": float(
                nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)
            ),
            "power_w": self.read_power_w(),
            "sm_clock_mhz": int(
                nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_SM)
            ),
            "graphics_clock_mhz": int(
                nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_GRAPHICS)
            ),
            "memory_clock_mhz": int(
                nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_MEM)
            ),
            "power_limit_w": float(power_limit_fn(self.handle)) / 1000.0,
            "performance_state": int(nvml.nvmlDeviceGetPerformanceState(self.handle)),
            "clock_throttle_reasons": int(
                nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
            ),
            "gpu_utilization_percent": int(utilization.gpu),
            "memory_utilization_percent": int(utilization.memory),
        }
        validate_telemetry(snapshot)
        return snapshot

    def compute_processes(self) -> list[dict[str, object]]:
        """Return compute-process state, tolerating pynvml API naming drift."""

        function = None
        for name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            candidate = getattr(self.nvml, name, None)
            if candidate is not None:
                function = candidate
                break
        if function is None:
            raise ProbeError("NVML compute-process telemetry is unavailable")
        rows = []
        for process in function(self.handle):
            used = getattr(process, "usedGpuMemory", None)
            rows.append(
                {
                    "pid": int(process.pid),
                    "used_gpu_memory_bytes": None if used is None else int(used),
                }
            )
        return rows

    def close(self) -> None:
        if not self._closed:
            self.nvml.nvmlShutdown()
            self._closed = True

    def __enter__(self) -> "NVMLTelemetryBackend":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def validate_telemetry(snapshot: Mapping[str, object]) -> None:
    required_positive = (
        "timestamp_ns",
        "temperature_c",
        "power_w",
        "sm_clock_mhz",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "power_limit_w",
    )
    for key in required_positive:
        value = snapshot.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProbeError(f"telemetry field {key!r} is missing or non-numeric")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ProbeError(f"telemetry field {key!r} must be finite and positive")
    for key in (
        "performance_state",
        "clock_throttle_reasons",
        "gpu_utilization_percent",
        "memory_utilization_percent",
    ):
        value = snapshot.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProbeError(f"telemetry field {key!r} is missing or invalid")


class DevelopmentWindowMeter:
    def __init__(
        self,
        backend: NVMLTelemetryBackend,
        *,
        sample_interval_s: float,
        maximum_observed_gap_s: float,
    ) -> None:
        if not 0 < sample_interval_s < maximum_observed_gap_s:
            raise ProbeError("power sampler interval must be below the observed-gap gate")
        if maximum_observed_gap_s > MAX_FORMAL_SAMPLE_INTERVAL_S:
            raise ProbeError("observed power-sample gap gate must be <=20 ms")
        self.backend = backend
        self.sample_interval_s = sample_interval_s
        self.maximum_observed_gap_s = maximum_observed_gap_s

    def measure(self, workload: Callable[[], None]) -> EnergyEnvelope:
        recording = RecordingPowerBackend(self.backend)
        telemetry_start = self.backend.telemetry()
        sampler = MonotonicNVMLSampler(
            recording,
            interval_s=self.sample_interval_s,
            formal=True,
        )
        sampler.start()
        host_cuda_record_start_ns = time.monotonic_ns()
        try:
            workload()
            host_cuda_sync_end_ns = time.monotonic_ns()
            trace = sampler.stop()
        except BaseException:
            # Best effort stop only to avoid leaving a reader thread alive.  The
            # original exception remains authoritative.
            try:
                sampler.stop()
            except BaseException:
                pass
            raise
        telemetry_end = self.backend.telemetry()
        observed_gap = max_sample_gap_s(trace.samples)
        if observed_gap > self.maximum_observed_gap_s + 1e-12:
            raise ProbeError(
                "observed power-sample gap exceeded the development gate: "
                f"observed={observed_gap:.9f}, gate={self.maximum_observed_gap_s:.9f}"
            )
        power_integral_j = integrate_power_samples(trace.samples)
        counter_energy_j = trace.total_energy_counter_delta_j
        raw_energy_j = counter_energy_j if counter_energy_j is not None else power_integral_j
        source = (
            "nvml_total_energy_counter"
            if counter_energy_j is not None
            else "monotonic_power_integral"
        )
        if power_integral_j <= 0 or raw_energy_j <= 0:
            raise ProbeError("raw power integral and selected board energy must be positive")
        power_samples = tuple(
            {"timestamp_ns": row.timestamp_ns, "power_w": row.power_w}
            for row in trace.samples
        )
        return EnergyEnvelope(
            raw_board_energy_j=raw_energy_j,
            energy_source=source,
            power_integral_j=power_integral_j,
            counter_energy_j=counter_energy_j,
            energy_window_start_ns=trace.start_ns,
            energy_window_end_ns=trace.end_ns,
            host_cuda_record_start_ns=host_cuda_record_start_ns,
            host_cuda_sync_end_ns=host_cuda_sync_end_ns,
            power_samples=power_samples,
            maximum_power_sample_gap_s=observed_gap,
            telemetry_start=telemetry_start,
            telemetry_end=telemetry_end,
            counter_reads=tuple(recording.counter_reads),
        )


class CUDAEventTimer:
    def __init__(self, torch_module: Any, device: object) -> None:
        self.torch = torch_module
        self.device = device

    def measure(self, workload: Callable[[], object]) -> TimedResult:
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        payload = workload()
        end.record()
        end.synchronize()
        return TimedResult(latency_ms=float(start.elapsed_time(end)), payload=payload)


class SyntheticExpertJobs:
    """Exact equal-work BF16 expert jobs; only their dispatch order changes."""

    def __init__(
        self,
        torch_module: Any,
        device: object,
        *,
        hidden_size: int,
        intermediate_size: int,
        row_grid: Sequence[int],
        seed: int,
    ) -> None:
        self.torch = torch_module
        self.device = device
        self.dtype = torch_module.bfloat16
        generator = torch_module.Generator(device=device)
        generator.manual_seed(seed)
        hidden_scale = 1.0 / math.sqrt(hidden_size)
        intermediate_scale = 1.0 / math.sqrt(intermediate_size)
        self.gate_weight = (
            torch_module.randn(
                (intermediate_size, hidden_size),
                device=device,
                dtype=self.dtype,
                generator=generator,
            )
            * hidden_scale
        )
        self.up_weight = (
            torch_module.randn(
                (intermediate_size, hidden_size),
                device=device,
                dtype=self.dtype,
                generator=generator,
            )
            * hidden_scale
        )
        self.down_weight = (
            torch_module.randn(
                (hidden_size, intermediate_size),
                device=device,
                dtype=self.dtype,
                generator=generator,
            )
            * intermediate_scale
        )
        self.jobs: tuple[tuple[str, object], ...] = tuple(
            (
                f"synthetic-expert-0/rows-{rows}",
                torch_module.randn(
                    (rows, hidden_size),
                    device=device,
                    dtype=self.dtype,
                    generator=generator,
                ),
            )
            for rows in row_grid
        )
        if len({name for name, _ in self.jobs}) != len(self.jobs):
            raise ProbeError("row grid produced duplicate logical work identities")

    @property
    def logical_work_ids(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.jobs)

    @property
    def rows_per_repeat(self) -> int:
        return sum(int(value.shape[0]) for _, value in self.jobs)

    def _expert(self, activation: object) -> object:
        functional = self.torch.nn.functional
        gate = functional.linear(activation, self.gate_weight)
        up = functional.linear(activation, self.up_weight)
        return functional.linear(functional.silu(gate) * up, self.down_weight)

    def run(self, arm: str, repeats: int) -> dict[str, object]:
        if arm not in (ARM_A, ARM_B):
            raise ProbeError(f"unknown development arm {arm!r}")
        if repeats <= 0:
            raise ProbeError("workload repeats must be positive")
        indices = range(len(self.jobs))
        order = tuple(indices if arm == ARM_A else reversed(tuple(indices)))
        outputs: dict[str, object] = {}
        for _ in range(repeats):
            for index in order:
                work_id, activation = self.jobs[index]
                outputs[work_id] = self._expert(activation)
        return outputs


def hash_outputs(outputs: Mapping[str, object]) -> str:
    if not outputs:
        raise ProbeError("cannot hash an empty output set")
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        raise ProbeError("torch must be imported before hashing CUDA outputs")
    digest = hashlib.sha256()
    for work_id in sorted(outputs):
        tensor = outputs[work_id]
        digest.update(work_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        # ``Tensor.numpy`` does not support bfloat16 on every torch build.  A
        # byte view preserves the exact device result without numeric casting.
        raw_uint8 = (
            tensor.detach()
            .contiguous()
            .view(torch_module.uint8)
            .cpu()
            .numpy()
            .tobytes()
        )
        digest.update(raw_uint8)
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def _record_from_measurement(
    *,
    block: int,
    position: int,
    arm: str,
    repeats: int,
    work_ids: Sequence[str],
    rows_per_repeat: int,
    result: SameWindowResult,
    output_sha256: str,
    gpu_name: str,
    gpu_uuid: str,
) -> dict[str, object]:
    envelope = result.energy
    return {
        "schema": SCHEMA,
        "formal_result": False,
        "scientific_result_eligible": False,
        "block": block,
        "position": position,
        "arm": arm,
        "actuator": "dispatch_order_only",
        "repeats": repeats,
        "logical_work_ids": list(work_ids),
        "logical_jobs_per_repeat": len(work_ids),
        "rows_per_repeat": rows_per_repeat,
        "cuda_latency_ms": result.timed.latency_ms,
        "cuda_latency_ms_per_repeat": result.timed.latency_ms / repeats,
        "raw_board_energy_j": envelope.raw_board_energy_j,
        "raw_board_energy_j_per_repeat": envelope.raw_board_energy_j / repeats,
        "energy_source": envelope.energy_source,
        "counter_energy_j": envelope.counter_energy_j,
        "power_integral_j": envelope.power_integral_j,
        "energy_window_start_ns": envelope.energy_window_start_ns,
        "energy_window_end_ns": envelope.energy_window_end_ns,
        "host_cuda_record_start_ns": envelope.host_cuda_record_start_ns,
        "host_cuda_sync_end_ns": envelope.host_cuda_sync_end_ns,
        "counter_sample_boundary_relation": "SEQUENTIAL_BRACKETING_NOT_ATOMIC",
        "same_execution_cuda_latency_and_energy": True,
        "power_samples": list(envelope.power_samples),
        "maximum_power_sample_gap_s": envelope.maximum_power_sample_gap_s,
        "telemetry_start": dict(envelope.telemetry_start),
        "telemetry_end": dict(envelope.telemetry_end),
        "counter_reads": list(envelope.counter_reads),
        "output_sha256": output_sha256,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
    }


def validate_development_records(
    records: Sequence[Mapping[str, object]],
    *,
    blocks: int,
    expected_repeats: int,
    maximum_observed_gap_s: float,
    maximum_temperature_drift_c: float,
) -> None:
    """Validate raw records without ever promoting them to formal evidence."""

    schedule = abba_schedule(blocks)
    if len(records) != len(schedule):
        raise ProbeError("raw window count does not close the ABBA schedule")
    reference_hash: str | None = None
    reference_work: tuple[str, ...] | None = None
    reference_gpu: tuple[str, str] | None = None
    reference_power_limit_w: float | None = None
    start_temperatures_by_block: dict[int, list[float]] = {
        block: [] for block in range(blocks)
    }
    for record, (block, position, arm) in zip(records, schedule):
        if record.get("formal_result") is not False:
            raise ProbeError("development record attempted to become formal")
        if record.get("scientific_result_eligible") is not False:
            raise ProbeError("development record attempted scientific promotion")
        if (record.get("block"), record.get("position"), record.get("arm")) != (
            block,
            position,
            arm,
        ):
            raise ProbeError("raw windows drifted from strict ABBA order")
        if record.get("same_execution_cuda_latency_and_energy") is not True:
            raise ProbeError("CUDA latency and energy did not use the same execution")
        if record.get("repeats") != expected_repeats:
            raise ProbeError("ABBA windows have unequal repeat denominators")
        for key in (
            "cuda_latency_ms",
            "raw_board_energy_j",
            "power_integral_j",
        ):
            value = record.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ProbeError(f"raw window has invalid {key}")
        gap = record.get("maximum_power_sample_gap_s")
        if not isinstance(gap, (int, float)) or gap <= 0 or gap > maximum_observed_gap_s + 1e-12:
            raise ProbeError("raw window violates the power-sample gap gate")
        start_ns = record.get("energy_window_start_ns")
        cuda_start_ns = record.get("host_cuda_record_start_ns")
        cuda_end_ns = record.get("host_cuda_sync_end_ns")
        end_ns = record.get("energy_window_end_ns")
        if not all(isinstance(value, int) for value in (start_ns, cuda_start_ns, cuda_end_ns, end_ns)):
            raise ProbeError("measurement-window timestamps are missing")
        if not start_ns <= cuda_start_ns <= cuda_end_ns <= end_ns:
            raise ProbeError("energy window does not bracket the timed CUDA execution")
        samples = record.get("power_samples")
        if not isinstance(samples, list) or len(samples) < 2:
            raise ProbeError("raw power samples are missing")
        sample_times = [row.get("timestamp_ns") for row in samples if isinstance(row, Mapping)]
        if len(sample_times) != len(samples) or any(not isinstance(value, int) for value in sample_times):
            raise ProbeError("raw power-sample timestamps are invalid")
        if any(right <= left for left, right in zip(sample_times, sample_times[1:])):
            raise ProbeError("raw power-sample timestamps are not strictly monotonic")
        if sample_times[0] != start_ns or sample_times[-1] != end_ns:
            raise ProbeError("raw power samples do not contain both window boundaries")
        counter_reads = record.get("counter_reads")
        if not isinstance(counter_reads, list) or len(counter_reads) != 2:
            raise ProbeError("raw energy-counter boundary reads are missing")
        first_counter, second_counter = counter_reads
        if not isinstance(first_counter, Mapping) or not isinstance(second_counter, Mapping):
            raise ProbeError("raw energy-counter boundary reads are invalid")
        if not (
            isinstance(first_counter.get("read_end_ns"), int)
            and isinstance(second_counter.get("read_start_ns"), int)
            and first_counter["read_end_ns"] <= start_ns
            and end_ns <= second_counter["read_start_ns"]
        ):
            raise ProbeError("energy counter reads do not bracket the raw sample window")
        source = record.get("energy_source")
        counter_energy = record.get("counter_energy_j")
        raw_energy = float(record["raw_board_energy_j"])
        integral_energy = float(record["power_integral_j"])
        if counter_energy is None:
            if source != "monotonic_power_integral" or not math.isclose(
                raw_energy, integral_energy, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ProbeError("power-integral energy source is internally inconsistent")
        elif (
            not isinstance(counter_energy, (int, float))
            or counter_energy <= 0
            or source != "nvml_total_energy_counter"
            or not math.isclose(raw_energy, float(counter_energy), rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ProbeError("NVML counter energy source is internally inconsistent")
        start_telemetry = record.get("telemetry_start")
        end_telemetry = record.get("telemetry_end")
        if not isinstance(start_telemetry, Mapping) or not isinstance(end_telemetry, Mapping):
            raise ProbeError("thermal/clock/power telemetry is missing")
        validate_telemetry(start_telemetry)
        validate_telemetry(end_telemetry)
        if not (
            int(start_telemetry["timestamp_ns"]) <= start_ns
            and end_ns <= int(end_telemetry["timestamp_ns"])
        ):
            raise ProbeError("telemetry snapshots do not bracket the energy window")
        thermal_delta = abs(
            float(end_telemetry["temperature_c"])
            - float(start_telemetry["temperature_c"])
        )
        if thermal_delta > maximum_temperature_drift_c:
            raise ProbeError(
                f"window thermal drift {thermal_delta:.3f} C exceeds the gate"
            )
        start_temperatures_by_block[block].append(
            float(start_telemetry["temperature_c"])
        )
        power_limit_w = float(start_telemetry["power_limit_w"])
        if not math.isclose(
            power_limit_w,
            float(end_telemetry["power_limit_w"]),
            rel_tol=0.0,
            abs_tol=0.001,
        ):
            raise ProbeError("power limit changed inside a measurement window")
        work = tuple(str(value) for value in record.get("logical_work_ids", ()))
        if not work or len(work) != len(set(work)):
            raise ProbeError("logical equal-work identities are empty or duplicated")
        output_hash = record.get("output_sha256")
        if not isinstance(output_hash, str) or len(output_hash) != 64:
            raise ProbeError("exact output SHA-256 is missing")
        gpu_name = str(record.get("gpu_name", ""))
        gpu_uuid = str(record.get("gpu_uuid", ""))
        if "RTX 5090" not in gpu_name.upper().replace("GEFORCE ", ""):
            raise ProbeError("raw artifact is not from an RTX 5090")
        if reference_hash is None:
            reference_hash = output_hash
            reference_work = work
            reference_gpu = (gpu_name, gpu_uuid)
            reference_power_limit_w = power_limit_w
        elif output_hash != reference_hash or work != reference_work:
            raise ProbeError("ABBA arms do not have exact equal work/output identity")
        elif (gpu_name, gpu_uuid) != reference_gpu:
            raise ProbeError("CUDA/NVML GPU identity changed during the run")
        elif not math.isclose(
            power_limit_w,
            float(reference_power_limit_w),
            rel_tol=0.0,
            abs_tol=0.001,
        ):
            raise ProbeError("power limit changed across ABBA windows")
    for block, values in start_temperatures_by_block.items():
        if max(values) - min(values) > maximum_temperature_drift_c:
            raise ProbeError(
                f"ABBA block {block} start-temperature spread exceeds the gate"
            )


def _block_summaries(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    blocks = sorted({int(row["block"]) for row in records})
    for block in blocks:
        group = [row for row in records if int(row["block"]) == block]
        arms: dict[str, dict[str, float]] = {}
        for arm in (ARM_A, ARM_B):
            rows = [row for row in group if row["arm"] == arm]
            arms[arm] = {
                "raw_board_energy_j_per_repeat_mean": statistics.fmean(
                    float(row["raw_board_energy_j_per_repeat"]) for row in rows
                ),
                "cuda_latency_ms_per_repeat_mean": statistics.fmean(
                    float(row["cuda_latency_ms_per_repeat"]) for row in rows
                ),
            }
        summaries.append(
            {
                "block": block,
                "arms": arms,
                "forward_minus_reverse_raw_energy_j_per_repeat": (
                    arms[ARM_A]["raw_board_energy_j_per_repeat_mean"]
                    - arms[ARM_B]["raw_board_energy_j_per_repeat_mean"]
                ),
                "forward_minus_reverse_cuda_latency_ms_per_repeat": (
                    arms[ARM_A]["cuda_latency_ms_per_repeat_mean"]
                    - arms[ARM_B]["cuda_latency_ms_per_repeat_mean"]
                ),
            }
        )
    return summaries


def write_manifest(output_dir: Path, *, status: str, source_hashes: Mapping[str, str]) -> None:
    files = {
        str(path.relative_to(output_dir)): _sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(
        output_dir / "manifest.json",
        {
            "schema": SCHEMA,
            "status": status,
            "verdict": VERDICT,
            "formal_result": False,
            "scientific_result_eligible": False,
            "source_sha256": dict(source_hashes),
            "artifact_sha256": files,
        },
    )


def _config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "formal_result": False,
        "device_index": args.device_index,
        "blocks": args.blocks,
        "warmup_repeats": args.warmup_repeats,
        "minimum_window_seconds": args.minimum_window_seconds,
        "maximum_repeats": args.maximum_repeats,
        "power_sample_interval_seconds": args.power_sample_interval_seconds,
        "maximum_observed_power_gap_seconds": args.maximum_observed_power_gap_seconds,
        "maximum_window_temperature_drift_c": args.maximum_window_temperature_drift_c,
        "maximum_run_start_temperature_c": args.maximum_run_start_temperature_c,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "row_grid": list(args.row_grid),
        "seed": args.seed,
        "arms": [ARM_A, ARM_B],
        "schedule": list(ABBA),
        "actuator": "dispatch_order_only",
        "energy_basis": "raw_board_energy_during_same_cuda_timed_execution",
    }


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProbeError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw" / "windows.jsonl"
    config = _config_from_args(args)
    _write_json(output_dir / "config.json", config)
    source_hashes = {
        str(Path(__file__).resolve().relative_to(REPO_ROOT)): _sha256_file(Path(__file__)),
        str((POWER_ACCOUNTING_DIR / "power_accounting.py").relative_to(REPO_ROOT)): _sha256_file(
            POWER_ACCOUNTING_DIR / "power_accounting.py"
        ),
    }
    records: list[dict[str, object]] = []
    try:
        import torch

        if not torch.cuda.is_available():
            raise ProbeError("CUDA is unavailable")
        if args.device_index < 0 or args.device_index >= torch.cuda.device_count():
            raise ProbeError("CUDA device index is out of range")
        device = torch.device(f"cuda:{args.device_index}")
        torch.cuda.set_device(device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        jobs = SyntheticExpertJobs(
            torch,
            device,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            row_grid=args.row_grid,
            seed=args.seed,
        )
        timer = CUDAEventTimer(torch, device)
        with NVMLTelemetryBackend(torch, device, args.device_index) as backend:
            environment = {
                "schema": SCHEMA,
                "formal_result": False,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu_name": backend.gpu_name,
                "gpu_uuid": backend.gpu_uuid,
                "nvml_physical_index": backend.nvml_physical_index,
                "driver_version": backend.driver_version,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "initial_telemetry": backend.telemetry(),
                "compute_processes": backend.compute_processes(),
            }
            competitors = [
                row for row in environment["compute_processes"] if row["pid"] != os.getpid()
            ]
            if competitors:
                raise ProbeError(f"competing GPU compute processes detected: {competitors!r}")
            if float(environment["initial_telemetry"]["temperature_c"]) > args.maximum_run_start_temperature_c:
                raise ProbeError("run-start GPU temperature exceeds the configured gate")
            for _ in range(args.warmup_repeats):
                jobs.run(ARM_A, 1)
                jobs.run(ARM_B, 1)
            torch.cuda.synchronize(device)
            environment["post_warmup_telemetry"] = backend.telemetry()
            if (
                float(environment["post_warmup_telemetry"]["temperature_c"])
                > args.maximum_run_start_temperature_c
            ):
                raise ProbeError("post-warmup GPU temperature exceeds the configured gate")
            _write_json(output_dir / "environment.json", environment)

            def calibrate(arm: str, repeats: int) -> float:
                return timer.measure(lambda: jobs.run(arm, repeats)).latency_ms / 1000.0

            repeats, calibration = choose_equal_repeats(
                calibrate,
                minimum_window_s=args.minimum_window_seconds,
                maximum_repeats=args.maximum_repeats,
            )
            _write_json(
                output_dir / "raw" / "calibration.json",
                {
                    "formal_result": False,
                    "selected_equal_repeats": repeats,
                    "final_calibration_seconds": calibration,
                    "calibration_is_independent_evidence": False,
                },
            )
            meter = DevelopmentWindowMeter(
                backend,
                sample_interval_s=args.power_sample_interval_seconds,
                maximum_observed_gap_s=args.maximum_observed_power_gap_seconds,
            )
            for block, position, arm in abba_schedule(args.blocks):
                torch.cuda.synchronize(device)
                result = measure_same_window(
                    lambda arm=arm: jobs.run(arm, repeats),
                    timer=timer.measure,
                    meter=meter.measure,
                )
                outputs = result.timed.payload
                if not isinstance(outputs, Mapping):
                    raise ProbeError("synthetic workload did not return output mapping")
                output_sha256 = hash_outputs(outputs)
                record = _record_from_measurement(
                    block=block,
                    position=position,
                    arm=arm,
                    repeats=repeats,
                    work_ids=jobs.logical_work_ids,
                    rows_per_repeat=jobs.rows_per_repeat,
                    result=result,
                    output_sha256=output_sha256,
                    gpu_name=backend.gpu_name,
                    gpu_uuid=backend.gpu_uuid,
                )
                records.append(record)
                _append_jsonl(raw_path, record)

            validate_development_records(
                records,
                blocks=args.blocks,
                expected_repeats=repeats,
                maximum_observed_gap_s=args.maximum_observed_power_gap_seconds,
                maximum_temperature_drift_c=args.maximum_window_temperature_drift_c,
            )
            environment["post_run_compute_processes"] = backend.compute_processes()
            post_run_competitors = [
                row
                for row in environment["post_run_compute_processes"]
                if row["pid"] != os.getpid()
            ]
            if post_run_competitors:
                raise ProbeError(
                    f"competing GPU compute processes detected after run: {post_run_competitors!r}"
                )
            environment["post_run_telemetry"] = backend.telemetry()
            _write_json(output_dir / "environment.json", environment)
            summary = {
                "schema": SCHEMA,
                "status": "DEVELOPMENT_PROBE_COMPLETE",
                "verdict": VERDICT,
                "formal_result": False,
                "scientific_result_eligible": False,
                "reason": (
                    "synthetic single-GPU dispatch-order jobs are measurement-path checks; "
                    "they are not native continuous serving, EP, or RouteSlack evidence"
                ),
                "gpu_name": backend.gpu_name,
                "gpu_uuid": backend.gpu_uuid,
                "equal_repeats": repeats,
                "windows": len(records),
                "blocks": args.blocks,
                "block_summaries": _block_summaries(records),
                "power_tier_controlled": False,
                "natural_route_events": 0,
                "completed_serving_tokens": 0,
            }
            _write_json(output_dir / "processed" / "summary.json", summary)
            _write_json(
                output_dir / "verdict.json",
                {
                    "verdict": VERDICT,
                    "formal_result": False,
                    "gate0": "FAIL",
                    "open_items": [
                        "native_continuous_decode",
                        "real_route_identity",
                        "EP_execution",
                        "controlled_power_tiers",
                        "matched_completed_serving_tokens",
                    ],
                },
            )
        write_manifest(output_dir, status="DEVELOPMENT_PROBE_COMPLETE", source_hashes=source_hashes)
        return summary
    except BaseException as exc:
        _write_json(
            output_dir / "failure.json",
            {
                "schema": SCHEMA,
                "status": "DEVELOPMENT_PROBE_FAILED_CLOSED",
                "formal_result": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_windows": len(records),
            },
        )
        write_manifest(
            output_dir,
            status="DEVELOPMENT_PROBE_FAILED_CLOSED",
            source_hashes=source_hashes,
        )
        raise


def main() -> int:
    args = parse_args()
    summary = run_probe(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
