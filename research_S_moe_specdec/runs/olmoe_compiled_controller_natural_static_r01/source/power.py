"""Board-energy measurement with NVML counters and 50 ms power samples.

Usage (the caller chooses the CUDA synchronization boundary)::

    meter = PowerSampler(device=0)
    meter.measure_idle(2.0, synchronize=torch.cuda.synchronize)
    with meter.measure(synchronize=torch.cuda.synchronize) as measurement:
        for _ in range(repetitions):
            run_kernel()
    measurement.write_json("power.json")
    meter.close()

No torch dependency or implicit device synchronization is introduced. ``device``
is an NVML physical index or UUID, NOT a CUDA_VISIBLE_DEVICES logical ordinal.
Measure sufficiently long repeated work; 50 ms polling does not establish the
hardware sensor update interval. All energy is whole-device, not process-only.

Adapted from the July PowerSampler in MOE_Thesis_organized at commit
5593b5ff0fb602f479b10f82ca682ba8f9adc86d:
docs/ideas/energy_slo/route_row_fp8/experiments/run_energy_slo_power_probe.py
(blob b0d15bbba61bdaf0295f3d71cf5735797b91b32b, retrieved with the GitHub
connector and kept under ref_july/). Retains its NVML initialization, background
thread, mW-to-W sampling and trapezoidal-integration pattern; extends it with
energy counters, monotonic bracketing, idle accounting and self-contained data.
The old sampler's stop() tuple is deliberately replaced with a documented dict.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable


def integrate_power(samples: list[dict], start_ns: int, end_ns: int) -> float | None:
    """Trapezoids clipped to [start, end]; no extrapolation beyond raw samples."""
    valid = sorted((s["monotonic_ns"], s["power_w"]) for s in samples
                   if s.get("power_w") is not None)
    if end_ns < start_ns:
        raise ValueError("end precedes start")
    if len(valid) < 2 or valid[0][0] > start_ns or valid[-1][0] < end_ns:
        return None
    energy = 0.0
    for (a, pa), (b, pb) in zip(valid, valid[1:]):
        lo, hi = max(a, start_ns), min(b, end_ns)
        if hi <= lo or b == a:
            continue
        p_lo = pa + (pb - pa) * (lo - a) / (b - a)
        p_hi = pa + (pb - pa) * (hi - a) / (b - a)
        energy += (p_lo + p_hi) * 0.5 * (hi - lo) / 1e9
    return energy


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class Measurement:
    def __init__(self, meter: "PowerSampler", synchronize: Callable | None):
        self.meter = meter
        self.synchronize = synchronize
        self.result: dict | None = None

    def __enter__(self) -> "Measurement":
        if self.synchronize is not None:
            self.synchronize()
        self.meter.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        sync_error = None
        try:
            if self.synchronize is not None:
                self.synchronize()
        except BaseException as error:
            sync_error = error
        finally:
            self.result = self.meter.stop()
            self.result["synchronization"] = "caller_callback" if self.synchronize else "caller_managed"
            self.result["workload_failed"] = exc_type is not None or sync_error is not None
        if sync_error is not None and exc_type is None:
            raise sync_error
        return False

    def write_json(self, path: str | Path) -> None:
        if self.result is None:
            raise RuntimeError("measurement has not finished")
        Path(path).write_text(json.dumps(self.result, indent=2, allow_nan=False) + "\n")


class PowerSampler:
    """Reusable sampler; call close() after the final measurement.

    ``start``/``stop`` support existing imperative benchmark harnesses. Each stop
    returns a self-contained JSON-ready dict including the raw NVML readings.
    Negative idle-subtracted energy is retained, not clipped to hide noise.
    ``nvml_module`` is an injectable backend for CPU-only measurement checks.
    """

    def __init__(self, device: int | str = 0, interval_s: float = 0.05,
                 idle_power_w: float | None = None, *, device_index: int | None = None,
                 nvml_module=None):
        if device_index is not None:
            if device != 0:
                raise ValueError("specify either device or legacy device_index")
            device = device_index
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("interval_s must be positive and finite")
        if idle_power_w is not None and (not math.isfinite(idle_power_w) or idle_power_w < 0):
            raise ValueError("idle_power_w must be nonnegative and finite")
        if nvml_module is None:
            import pynvml as nvml_module
        self.nvml = nvml_module
        self.interval_s = interval_s
        self.idle_power_w = idle_power_w
        self.idle_measurement = None
        self.result = None
        self._running = False
        self._closed = False
        self.nvml.nvmlInit()
        try:
            if isinstance(device, str):
                self.handle = self.nvml.nvmlDeviceGetHandleByUUID(device)
            else:
                self.handle = self.nvml.nvmlDeviceGetHandleByIndex(device)
        except BaseException:
            self.nvml.nvmlShutdown()
            raise
        self.device = {"requested_nvml_device": device}
        for key, getter in (("uuid", "nvmlDeviceGetUUID"), ("name", "nvmlDeviceGetName")):
            try:
                self.device[key] = _text(getattr(self.nvml, getter)(self.handle))
            except Exception as error:
                self.device[key + "_error"] = str(error)
        try:
            self.device["driver_version"] = _text(self.nvml.nvmlSystemGetDriverVersion())
        except Exception as error:
            self.device["driver_version_error"] = str(error)

    def _read(self, function: str, key: str, divisor: float = 1.0) -> dict:
        before = time.monotonic_ns()
        wall = time.time_ns()
        value, error = None, None
        try:
            raw = getattr(self.nvml, function)(self.handle)
            value = int(raw) if key == "energy_mj" else float(raw) / divisor
            if value < 0 or not math.isfinite(value):
                raise ValueError("NVML returned a negative or non-finite value")
        except Exception as exception:
            value, error = None, f"{type(exception).__name__}: {exception}"
        after = time.monotonic_ns()
        result = {"monotonic_ns": (before + after) // 2, "read_start_ns": before,
                  "read_end_ns": after, "unix_ns_at_read_start": wall, key: value}
        if error is not None:
            result["error"] = error
        return result

    def _sample(self) -> dict:
        return self._read("nvmlDeviceGetPowerUsage", "power_w", 1000.0)

    def _counter(self) -> dict:
        return self._read("nvmlDeviceGetTotalEnergyConsumption", "energy_mj")

    def _loop(self) -> None:
        deadline = time.monotonic() + self.interval_s
        while not self._stop_event.wait(max(0.0, deadline - time.monotonic())):
            self.samples.append(self._sample())
            deadline += self.interval_s
            # Skip missed deadlines instead of issuing a burst of catch-up reads.
            now = time.monotonic()
            if deadline < now:
                deadline = now + self.interval_s

    def start(self) -> "PowerSampler":
        if self._closed:
            raise RuntimeError("sampler is closed")
        if self._running:
            raise RuntimeError("measurement already running")
        self.result = None
        self.samples = [self._sample()]
        self._counter_start = self._counter()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="nvml-power")
        self._running = True
        self._thread.start()
        self._workload_start_ns = time.monotonic_ns()
        return self

    def stop(self) -> dict:
        if not self._running:
            raise RuntimeError("no measurement running")
        workload_end = time.monotonic_ns()
        self._stop_event.set()
        self._thread.join()
        counter_end = self._counter()
        self.samples.append(self._sample())
        self._running = False
        start = self._counter_start["monotonic_ns"]
        end = counter_end["monotonic_ns"]
        elapsed = (end - start) / 1e9
        sampled = integrate_power(self.samples, start, end)
        a, b = self._counter_start["energy_mj"], counter_end["energy_mj"]
        counter = (b - a) / 1000.0 if a is not None and b is not None and b >= a else None
        warnings = []
        # A nonadvancing counter for a short positive-power window is unusable,
        # e.g. a cached sensor value. Preserve it and explicitly label fallback.
        if counter == 0 and sampled is not None and sampled > 0:
            warnings.append("energy_counter_did_not_advance")
            counter_valid = False
        else:
            counter_valid = counter is not None
        if a is not None and b is not None and b < a:
            warnings.append("energy_counter_decreased_or_reset")
        if counter_valid:
            energy, method = counter, "nvml_total_energy_counter"
        else:
            energy, method = sampled, "nvml_power_trapezoid"
        if energy is None:
            method = "unavailable"
            warnings.append("no_valid_counter_or_bracketing_power_samples")
        if elapsed < 1.0:
            warnings.append("short_window_repeat_work_for_energy_resolution")
        if any("error" in sample for sample in self.samples):
            warnings.append("power_read_failures_see_raw_samples")
        if elapsed <= 0:
            warnings.append("nonpositive_measurement_duration")
        idle_energy = self.idle_power_w * elapsed if self.idle_power_w is not None else None
        sample_times = [s["monotonic_ns"] for s in self.samples if s["power_w"] is not None]
        self.result = {
            "schema_version": 1, "device": dict(self.device), "method": method,
            "elapsed_s": elapsed, "energy_j": energy,
            "dynamic_energy_j": energy - idle_energy if energy is not None and idle_energy is not None else None,
            "idle_power_w": self.idle_power_w, "idle_energy_j": idle_energy,
            "mean_power_w": energy / elapsed if energy is not None and elapsed > 0 else None,
            "sampled_energy_j": sampled, "counter_energy_j": counter,
            "counter_valid": counter_valid,
            "counter_start": self._counter_start, "counter_end": counter_end,
            "boundary": {"clock": "time.monotonic_ns", "start_ns": start, "end_ns": end,
                         "definition": "midpoints_of_boundary_NVML_energy_counter_attempts",
                         "workload_start_ns": self._workload_start_ns,
                         "workload_end_ns": workload_end,
                         "workload_elapsed_s": (workload_end - self._workload_start_ns) / 1e9},
            "sampling": {"requested_interval_s": self.interval_s,
                         "count": len(self.samples),
                         "max_valid_sample_gap_s": max(((b - a) / 1e9 for a, b in zip(sample_times, sample_times[1:])), default=None),
                         "integration": "linear_interpolation_at_boundaries_then_trapezoids",
                         "sensor_update_interval_s": None,
                         "scope": "whole_NVML_device_including_other_processes"},
            "idle_baseline": self.idle_measurement,
            "samples": list(self.samples), "warnings": warnings,
            "synchronization": "caller_managed", "workload_failed": False,
        }
        return self.result

    def measure(self, synchronize: Callable | None = None) -> Measurement:
        return Measurement(self, synchronize)

    def measure_idle(self, duration_s: float = 2.0, synchronize: Callable | None = None) -> dict:
        """Measure a quiescent device; the caller must ensure no other GPU work."""
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be positive and finite")
        with self.measure(synchronize=synchronize) as measurement:
            time.sleep(duration_s)
        result = measurement.result
        if result["mean_power_w"] is None:
            raise RuntimeError("idle measurement unavailable; inspect meter.result")
        self.idle_power_w = result["mean_power_w"]
        self.idle_measurement = {
            "power_w": self.idle_power_w, "elapsed_s": result["elapsed_s"],
            "method": result["method"], "boundary": result["boundary"],
            "warnings": result["warnings"], "samples": result["samples"],
            "counter_start": result["counter_start"], "counter_end": result["counter_end"],
        }
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._running:
                self.stop()
        finally:
            self.nvml.nvmlShutdown()
            self._closed = True
