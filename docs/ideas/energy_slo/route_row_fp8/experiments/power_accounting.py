from __future__ import annotations

"""GPU board-power accounting for the frozen Energy-SLO protocol.

The primary clock is always ``time.monotonic_ns``.  A total-energy NVML
counter, when available, is preferred for total board energy; timestamped
power samples remain necessary for the auxiliary idle-subtracted accounting.
This module deliberately knows nothing about model FLOPs or route-row proxy
costs: its denominator is an externally verified count of *completed output
tokens*.
"""

from dataclasses import dataclass
import math
import operator
import threading
import time
from typing import Callable, Protocol, Sequence


NS_PER_SECOND = 1_000_000_000
MAX_FORMAL_SAMPLE_INTERVAL_S = 0.020


@dataclass(frozen=True)
class PowerSample:
    timestamp_ns: int
    power_w: float

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not math.isfinite(self.power_w) or self.power_w < 0:
            raise ValueError("power_w must be finite and non-negative")


@dataclass(frozen=True)
class PowerTrace:
    samples: tuple[PowerSample, ...]
    start_ns: int
    end_ns: int
    gpu_uuid: str | None = None
    total_energy_counter_delta_j: float | None = None

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError("power trace must have positive duration")
        if len(self.samples) < 2:
            raise ValueError("power trace requires explicit start/end samples")
        if self.samples[0].timestamp_ns != self.start_ns:
            raise ValueError("first sample must be the explicit start boundary")
        if self.samples[-1].timestamp_ns != self.end_ns:
            raise ValueError("last sample must be the explicit end boundary")
        _validate_strictly_monotonic(self.samples)
        if (
            self.total_energy_counter_delta_j is not None
            and (
                not math.isfinite(self.total_energy_counter_delta_j)
                or self.total_energy_counter_delta_j < 0
            )
        ):
            raise ValueError("total-energy counter delta must be finite and non-negative")


@dataclass(frozen=True)
class PowerAccountingResult:
    total_energy_j: float
    dynamic_energy_j: float | None
    completed_output_tokens: int
    total_j_per_completed_token: float
    dynamic_j_per_completed_token: float | None
    idle_power_w: float | None
    total_source: str
    start_ns: int
    end_ns: int
    sample_count: int
    max_sample_gap_s: float
    gpu_uuid: str | None


class PowerBackend(Protocol):
    """Small adapter surface used by :class:`MonotonicNVMLSampler`."""

    @property
    def gpu_uuid(self) -> str:
        ...

    def read_power_w(self) -> float:
        ...

    def read_total_energy_j(self) -> float | None:
        ...


def _validate_strictly_monotonic(samples: Sequence[PowerSample]) -> None:
    for left, right in zip(samples, samples[1:]):
        if right.timestamp_ns <= left.timestamp_ns:
            raise ValueError("power sample timestamps must be strictly monotonic")


def max_sample_gap_s(samples: Sequence[PowerSample]) -> float:
    """Return the largest observed sampling gap, using monotonic timestamps."""

    if len(samples) < 2:
        raise ValueError("at least two power samples are required")
    _validate_strictly_monotonic(samples)
    return max(
        (right.timestamp_ns - left.timestamp_ns) / NS_PER_SECOND
        for left, right in zip(samples, samples[1:])
    )


def validate_formal_sample_intervals(
    samples: Sequence[PowerSample],
    *,
    max_interval_s: float = MAX_FORMAL_SAMPLE_INTERVAL_S,
) -> float:
    """Fail closed when an observed formal NVML gap exceeds the frozen limit."""

    if not math.isfinite(max_interval_s) or max_interval_s <= 0:
        raise ValueError("max_interval_s must be finite and positive")
    observed = max_sample_gap_s(samples)
    if observed > max_interval_s + 1e-12:
        raise RuntimeError(
            "formal power trace exceeds the observed sampling-gap limit: "
            f"max_gap_s={observed:.9f}, limit_s={max_interval_s:.9f}"
        )
    return observed


def integrate_power_samples(
    samples: Sequence[PowerSample],
    *,
    idle_power_w: float | None = None,
) -> float:
    """Integrate explicit-boundary samples with the trapezoidal rule.

    When ``idle_power_w`` is supplied, each sample is transformed with
    ``max(power-idle, 0)`` before integration.  Boundary interpolation is not
    guessed: callers must provide the start and end boundary samples.
    """

    if len(samples) < 2:
        raise ValueError("at least two power samples are required")
    _validate_strictly_monotonic(samples)
    if idle_power_w is not None and (
        not math.isfinite(idle_power_w) or idle_power_w < 0
    ):
        raise ValueError("idle_power_w must be finite and non-negative")

    def adjusted(sample: PowerSample) -> float:
        if idle_power_w is None:
            return sample.power_w
        return max(sample.power_w - idle_power_w, 0.0)

    energy_j = 0.0
    for left, right in zip(samples, samples[1:]):
        dt_s = (right.timestamp_ns - left.timestamp_ns) / NS_PER_SECOND
        energy_j += 0.5 * (adjusted(left) + adjusted(right)) * dt_s
    return energy_j


def account_power_trace(
    trace: PowerTrace,
    *,
    completed_output_tokens: int,
    idle_power_w: float | None,
    formal: bool = False,
) -> PowerAccountingResult:
    """Compute total/dynamic GPU J per completed output token.

    A counter delta is the frozen protocol's primary total-energy source when
    present.  Otherwise the total is integrated from monotonic samples.
    Unfinished or merely scheduled tokens must never be passed in this count.
    """

    if isinstance(completed_output_tokens, bool):
        raise ValueError("completed_output_tokens must be a positive integer")
    try:
        completed_output_tokens = operator.index(completed_output_tokens)
    except TypeError as exc:
        raise ValueError("completed_output_tokens must be a positive integer") from exc
    if completed_output_tokens <= 0:
        raise ValueError("completed_output_tokens must be a positive integer")

    observed_max_gap_s = max_sample_gap_s(trace.samples)
    if formal:
        validate_formal_sample_intervals(trace.samples)

    if trace.total_energy_counter_delta_j is not None:
        total_energy_j = trace.total_energy_counter_delta_j
        total_source = "nvml_total_energy_counter"
    else:
        total_energy_j = integrate_power_samples(trace.samples)
        total_source = "monotonic_power_integral"

    dynamic_energy_j = None
    dynamic_j_per_token = None
    if idle_power_w is not None:
        dynamic_energy_j = integrate_power_samples(
            trace.samples, idle_power_w=idle_power_w
        )
        dynamic_j_per_token = dynamic_energy_j / completed_output_tokens

    return PowerAccountingResult(
        total_energy_j=total_energy_j,
        dynamic_energy_j=dynamic_energy_j,
        completed_output_tokens=completed_output_tokens,
        total_j_per_completed_token=total_energy_j / completed_output_tokens,
        dynamic_j_per_completed_token=dynamic_j_per_token,
        idle_power_w=idle_power_w,
        total_source=total_source,
        start_ns=trace.start_ns,
        end_ns=trace.end_ns,
        sample_count=len(trace.samples),
        max_sample_gap_s=observed_max_gap_s,
        gpu_uuid=trace.gpu_uuid,
    )


class MonotonicNVMLSampler:
    """Background power sampler with explicit monotonic boundary points."""

    def __init__(
        self,
        backend: PowerBackend,
        *,
        interval_s: float = MAX_FORMAL_SAMPLE_INTERVAL_S,
        formal: bool = True,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if formal and interval_s > MAX_FORMAL_SAMPLE_INTERVAL_S:
            raise ValueError("formal sampling interval must be <=20ms")
        self.backend = backend
        self.interval_s = float(interval_s)
        self.formal = bool(formal)
        self.clock_ns = clock_ns
        self._samples: list[PowerSample] = []
        self._counter_start_j: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _read_sample(self) -> PowerSample:
        return PowerSample(
            timestamp_ns=int(self.clock_ns()),
            power_w=float(self.backend.read_power_w()),
        )

    def _append_sample(self, sample: PowerSample) -> None:
        with self._lock:
            if self._samples and sample.timestamp_ns <= self._samples[-1].timestamp_ns:
                if sample.timestamp_ns == self._samples[-1].timestamp_ns:
                    self._samples[-1] = sample
                    return
                raise RuntimeError("monotonic clock moved backwards")
            self._samples.append(sample)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._append_sample(self._read_sample())

    def start(self) -> int:
        if self._thread is not None:
            raise RuntimeError("power sampler is already running")
        self._samples = []
        self._stop.clear()
        self._counter_start_j = self.backend.read_total_energy_j()
        self._append_sample(self._read_sample())
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self._samples[0].timestamp_ns

    def stop(self, synchronize: Callable[[], None] | None = None) -> PowerTrace:
        if self._thread is None:
            raise RuntimeError("power sampler is not running")
        if synchronize is not None:
            synchronize()
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 4))
        if self._thread.is_alive():
            raise RuntimeError("power sampler thread did not stop")
        self._thread = None
        self._append_sample(self._read_sample())
        counter_end_j = self.backend.read_total_energy_j()

        counter_delta_j = None
        if self._counter_start_j is not None and counter_end_j is not None:
            counter_delta_j = float(counter_end_j - self._counter_start_j)
            if counter_delta_j < 0:
                raise RuntimeError("NVML total-energy counter moved backwards")

        with self._lock:
            samples = tuple(self._samples)
        trace = PowerTrace(
            samples=samples,
            start_ns=samples[0].timestamp_ns,
            end_ns=samples[-1].timestamp_ns,
            gpu_uuid=self.backend.gpu_uuid,
            total_energy_counter_delta_j=counter_delta_j,
        )
        if self.formal:
            validate_formal_sample_intervals(trace.samples)
        return trace


class PynvmlPowerBackend:
    """NVML adapter.  Import/initialization is intentionally lazy."""

    def __init__(self, device_index: int = 0) -> None:
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        raw_uuid = pynvml.nvmlDeviceGetUUID(self._handle)
        self._gpu_uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else str(raw_uuid)
        self._closed = False

    @property
    def gpu_uuid(self) -> str:
        return self._gpu_uuid

    def read_power_w(self) -> float:
        return float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0

    def read_total_energy_j(self) -> float | None:
        fn = getattr(self._pynvml, "nvmlDeviceGetTotalEnergyConsumption", None)
        if fn is None:
            return None
        try:
            return float(fn(self._handle)) / 1000.0
        except self._pynvml.NVMLError_NotSupported:
            return None

    def close(self) -> None:
        if not self._closed:
            self._pynvml.nvmlShutdown()
            self._closed = True

    def __enter__(self) -> "PynvmlPowerBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def normalize_gpu_uuid(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    return normalized.replace("-", "")


def assert_matching_gpu_uuid(nvml_uuid: str, cuda_uuid: str) -> None:
    if normalize_gpu_uuid(nvml_uuid) != normalize_gpu_uuid(cuda_uuid):
        raise RuntimeError(
            f"NVML/CUDA device mismatch: nvml={nvml_uuid!r}, cuda={cuda_uuid!r}"
        )
