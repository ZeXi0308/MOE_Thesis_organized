from __future__ import annotations

"""DEV-only RTX 5090 BF16 expert-stage energy characterization.

This runner deliberately cannot emit a formal RouteSlack result.  It measures
one isolated expert at the provider's current power/clock state using real
captured BF16 activations, a cumulative NVML energy counter, explicit raw
telemetry, and non-overlapping activation groups.  It does not measure a
continuous-decode request, an SLO-completed token, a power-tier actuator, or EP.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shlex
import statistics
import sys
import threading
import time
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping, Sequence


EVIDENCE_SCOPE = "RTX5090_BF16_EXPERT_STAGE_CHARACTERIZATION"
FORMAL_BLOCKERS = (
    "isolated expert execution is not a natural continuous-decode request window",
    "activation source is calibration prefill rather than cached continuous decode",
    "provider denies power-limit and graphics-clock actuator changes",
    "denominator is processed expert rows, not matched SLO-completed output tokens",
    "single-GPU execution has no expert-parallel dispatch, A2A, combine, or rank slack",
)


class CharacterizationError(RuntimeError):
    """Raised when a physical measurement would otherwise become ambiguous."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    _write_text(
        path,
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
    )


def counter_delta_j(start_mj: int, end_mj: int) -> float:
    """Return raw counter energy and reject an undeclared wrap/reset."""

    if isinstance(start_mj, bool) or isinstance(end_mj, bool):
        raise CharacterizationError("energy counters must be integer millijoules")
    if start_mj < 0 or end_mj < 0:
        raise CharacterizationError("energy counters must be non-negative")
    if end_mj < start_mj:
        raise CharacterizationError(
            "NVML total-energy counter moved backwards; wrap modulus is unavailable"
        )
    return (end_mj - start_mj) / 1000.0


def exact_expert_row_denominator(*, repeats: int, rows: int) -> int:
    if (
        isinstance(repeats, bool)
        or isinstance(rows, bool)
        or not isinstance(repeats, int)
        or not isinstance(rows, int)
        or repeats <= 0
        or rows <= 0
    ):
        raise CharacterizationError("repeats and rows must be positive integers")
    return repeats * rows


def characterization_status(
    valid_counts: Mapping[int, int], *, requested_rows: Sequence[int], trials: int
) -> str:
    if trials < 2 or not requested_rows:
        raise CharacterizationError("status requires rows and at least two trials")
    if all(valid_counts.get(rows, 0) == trials for rows in set(requested_rows)):
        return "CHARACTERIZATION_COMPLETE"
    if all(valid_counts.get(rows, 0) >= 2 for rows in set(requested_rows)):
        return "CHARACTERIZATION_COMPLETE_WITH_FILTERED_WINDOWS"
    return "CHARACTERIZATION_INCOMPLETE_INVALID_WINDOWS"


@dataclass(frozen=True)
class ActivationRecord:
    request_id: str
    forward_id: str
    row_count: int
    record_index: int


@dataclass(frozen=True)
class ActivationGroupPlan:
    rows: int
    trial: int
    record_indices: tuple[int, ...]
    request_ids: tuple[str, ...]


def plan_disjoint_activation_groups(
    records: Sequence[ActivationRecord],
    *,
    row_grid: Sequence[int],
    trials: int,
) -> dict[tuple[int, int], ActivationGroupPlan]:
    """Allocate whole source requests to one outer trial only.

    The final source record may contribute only the requested prefix, but its
    unused rows are intentionally discarded so no request leaks into another
    independent activation group.
    """

    if trials <= 0 or not row_grid or any(row <= 0 for row in row_grid):
        raise CharacterizationError("row grid and trial count must be positive")
    ordered = sorted(
        records,
        key=lambda record: (-record.row_count, record.request_id, record.record_index),
    )
    if len({record.request_id for record in ordered}) != len(ordered):
        raise CharacterizationError("selected expert has duplicate request records")
    cursor = 0
    plans: dict[tuple[int, int], ActivationGroupPlan] = {}
    # Allocate the most demanding groups first; execution order is frozen later.
    for rows in sorted(set(row_grid), reverse=True):
        for trial in range(trials):
            chosen: list[ActivationRecord] = []
            available_rows = 0
            while available_rows < rows and cursor < len(ordered):
                record = ordered[cursor]
                cursor += 1
                chosen.append(record)
                available_rows += record.row_count
            if available_rows < rows:
                raise CharacterizationError(
                    f"not enough disjoint captured rows for rows={rows}, trial={trial}"
                )
            plans[(rows, trial)] = ActivationGroupPlan(
                rows=rows,
                trial=trial,
                record_indices=tuple(record.record_index for record in chosen),
                request_ids=tuple(record.request_id for record in chosen),
            )
    all_requests = [request for plan in plans.values() for request in plan.request_ids]
    if len(all_requests) != len(set(all_requests)):
        raise CharacterizationError("activation groups are not request-disjoint")
    return plans


def validate_telemetry_window(
    samples: Sequence[Mapping[str, object]],
    *,
    max_gap_s: float,
    max_temperature_range_c: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if len(samples) < 2:
        return ("fewer_than_two_telemetry_samples",)
    timestamps = [int(sample["monotonic_ns"]) for sample in samples]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        reasons.append("telemetry_timestamps_not_strictly_monotonic")
    else:
        observed_gap = max(
            (right - left) / 1_000_000_000
            for left, right in zip(timestamps, timestamps[1:])
        )
        if observed_gap > max_gap_s + 1e-12:
            reasons.append("telemetry_gap_exceeds_limit")
    temperatures = [float(sample["temperature_c"]) for sample in samples]
    if any(not math.isfinite(value) for value in temperatures):
        reasons.append("non_finite_temperature")
    elif max(temperatures) - min(temperatures) > max_temperature_range_c:
        reasons.append("temperature_range_exceeds_limit")
    uuids = {str(sample["gpu_uuid"]) for sample in samples}
    if len(uuids) != 1:
        reasons.append("gpu_uuid_changed")
    power_limits = {float(sample["power_limit_w"]) for sample in samples}
    if len(power_limits) != 1:
        reasons.append("power_limit_changed")
    return tuple(reasons)


class NVMLBackend:
    def __init__(self, device_index: int = 0) -> None:
        try:
            import pynvml
        except ImportError as exc:  # pragma: no cover - remote capability
            raise CharacterizationError("pynvml/nvidia-ml-py is required") from exc
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        raw_uuid = pynvml.nvmlDeviceGetUUID(self.handle)
        self.gpu_uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else str(raw_uuid)
        self.driver_version = str(pynvml.nvmlSystemGetDriverVersion())
        self._closed = False
        function = getattr(pynvml, "nvmlDeviceGetTotalEnergyConsumption", None)
        if function is None:
            self.close()
            raise CharacterizationError("NVML cumulative total-energy counter is unavailable")
        self._energy_function = function
        try:
            self.total_energy_mj()
        except BaseException as exc:
            self.close()
            raise CharacterizationError(
                "NVML cumulative total-energy counter is unsupported"
            ) from exc

    def total_energy_mj(self) -> int:
        return int(self._energy_function(self.handle))

    def sample(self, *, kind: str) -> dict[str, object]:
        utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
        return {
            "kind": kind,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_uuid": self.gpu_uuid,
            "power_w": float(self.nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0,
            "temperature_c": float(
                self.nvml.nvmlDeviceGetTemperature(
                    self.handle, self.nvml.NVML_TEMPERATURE_GPU
                )
            ),
            "graphics_clock_mhz": float(
                self.nvml.nvmlDeviceGetClockInfo(
                    self.handle, self.nvml.NVML_CLOCK_GRAPHICS
                )
            ),
            "memory_clock_mhz": float(
                self.nvml.nvmlDeviceGetClockInfo(
                    self.handle, self.nvml.NVML_CLOCK_MEM
                )
            ),
            "power_limit_w": float(
                self.nvml.nvmlDeviceGetPowerManagementLimit(self.handle)
            )
            / 1000.0,
            "power_draw_w": float(self.nvml.nvmlDeviceGetPowerUsage(self.handle))
            / 1000.0,
            "gpu_utilization_pct": float(utilization.gpu),
            "memory_utilization_pct": float(utilization.memory),
            "throttle_reasons": int(
                self.nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
            ),
        }

    def close(self) -> None:
        if not self._closed:
            self.nvml.nvmlShutdown()
            self._closed = True


class TelemetrySampler:
    def __init__(
        self,
        backend: object,
        *,
        interval_s: float,
        sample_function: Callable[..., Mapping[str, object]] | None = None,
    ) -> None:
        if interval_s <= 0:
            raise CharacterizationError("telemetry interval must be positive")
        self.backend = backend
        self.interval_s = float(interval_s)
        self._sample_function = sample_function or getattr(backend, "sample")
        self._samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    def _record(self, kind: str) -> None:
        value = dict(self._sample_function(kind=kind))
        with self._lock:
            self._samples.append(value)

    def _loop(self) -> None:
        try:
            while not self._stop.wait(self.interval_s):
                self._record("periodic")
        except BaseException as exc:
            self._error = exc
            self._stop.set()

    def start(self) -> None:
        if self._thread is not None:
            raise CharacterizationError("telemetry sampler already started")
        self._samples = []
        self._stop.clear()
        self._error = None
        self._record("boundary_start")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[dict[str, object], ...]:
        if self._thread is None:
            raise CharacterizationError("telemetry sampler is not running")
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 4))
        if self._thread.is_alive():
            raise CharacterizationError("telemetry sampler thread did not stop")
        self._thread = None
        if self._error is not None:
            error = self._error
            self._error = None
            raise CharacterizationError("telemetry sampler background read failed") from error
        self._record("boundary_end")
        with self._lock:
            return tuple(sorted(self._samples, key=lambda row: int(row["monotonic_ns"])))


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, replicates: int
) -> tuple[float, float, float]:
    if not values or replicates < 100:
        raise CharacterizationError("bootstrap requires values and at least 100 replicates")
    converted = tuple(float(value) for value in values)
    rng = random.Random(seed)
    draws = [
        sum(converted[rng.randrange(len(converted))] for _ in converted)
        / len(converted)
        for _ in range(replicates)
    ]
    return (
        statistics.mean(converted),
        _percentile(draws, 0.025),
        _percentile(draws, 0.975),
    )


def _summarize_trials(
    trials: Sequence[Mapping[str, object]], *, bootstrap: int, seed: int
) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for trial in trials:
        if bool(trial["valid"]):
            grouped[int(trial["rows"])].append(trial)
    output: list[dict[str, object]] = []
    for rows, values in sorted(grouped.items()):
        energy = [float(value["raw_j_per_expert_row"]) for value in values]
        latency = [float(value["cuda_us_per_logical_batch"]) for value in values]
        energy_mean, energy_low, energy_high = _bootstrap_mean_ci(
            energy, seed=seed + rows, replicates=bootstrap
        )
        latency_mean, latency_low, latency_high = _bootstrap_mean_ci(
            latency, seed=seed + 100_000 + rows, replicates=bootstrap
        )
        output.append(
            {
                "rows": rows,
                "valid_outer_trials": len(values),
                "independent_unit": "non-overlapping captured request activation group",
                "raw_j_per_expert_row_mean": energy_mean,
                "raw_j_per_expert_row_median": statistics.median(energy),
                "raw_j_per_expert_row_p95": _percentile(energy, 0.95),
                "raw_j_per_expert_row_p99": _percentile(energy, 0.99),
                "raw_j_per_expert_row_mean_ci95": [energy_low, energy_high],
                "cuda_us_per_logical_batch_mean": latency_mean,
                "cuda_us_per_logical_batch_median": statistics.median(latency),
                "cuda_us_per_logical_batch_p95": _percentile(latency, 0.95),
                "cuda_us_per_logical_batch_p99": _percentile(latency, 0.99),
                "cuda_us_per_logical_batch_mean_ci95": [latency_low, latency_high],
            }
        )
    return output


def _calibrate_repeats(
    expert: object,
    activation: object,
    *,
    minimum_window_s: float,
    maximum_repeats: int,
) -> int:
    import torch

    repeats = 256
    with torch.inference_mode():
        for _ in range(20):
            expert(activation)
        while True:
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(repeats):
                expert(activation)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if elapsed <= 0:
                raise CharacterizationError(
                    "repeat calibration has non-positive duration"
                )
            if elapsed >= minimum_window_s:
                return repeats
            estimate = math.ceil(
                repeats * minimum_window_s / elapsed * 1.20
            )
            repeats = max(repeats + 1, estimate)
            if repeats > maximum_repeats:
                raise CharacterizationError(
                    f"required repeats {repeats} exceed maximum {maximum_repeats}"
                )


def _run_warmup(
    expert: object,
    activation: object,
    backend: NVMLBackend,
    *,
    interval_s: float,
    minimum_s: float,
    maximum_s: float,
    stable_window_s: float,
    maximum_temperature_range_c: float,
) -> tuple[tuple[dict[str, object], ...], bool]:
    import torch

    sampler = TelemetrySampler(backend, interval_s=interval_s)
    sampler.start()
    started_ns = time.monotonic_ns()
    stable = False
    try:
        with torch.inference_mode():
            while True:
                for _ in range(512):
                    expert(activation)
                torch.cuda.synchronize()
                elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
                if elapsed_s < minimum_s:
                    continue
                with sampler._lock:
                    snapshot = tuple(sampler._samples)
                cutoff_ns = time.monotonic_ns() - int(stable_window_s * 1_000_000_000)
                recent = [
                    sample
                    for sample in snapshot
                    if int(sample["monotonic_ns"]) >= cutoff_ns
                ]
                temperatures = [float(sample["temperature_c"]) for sample in recent]
                if len(temperatures) >= 2 and max(temperatures) - min(temperatures) <= maximum_temperature_range_c:
                    stable = True
                    break
                if elapsed_s >= maximum_s:
                    break
    finally:
        torch.cuda.synchronize()
        samples = sampler.stop()
    return samples, stable


def _run_trial(
    expert: object,
    activation: object,
    backend: NVMLBackend,
    *,
    rows: int,
    trial: int,
    repeats: int,
    request_ids: Sequence[str],
    interval_s: float,
    max_gap_s: float,
    max_temperature_range_c: float,
    minimum_window_s: float,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    import torch

    with torch.inference_mode():
        for _ in range(20):
            expert(activation)
        torch.cuda.synchronize()

    sampler = TelemetrySampler(backend, interval_s=interval_s)
    sampler.start()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    start_event.synchronize()
    counter_start_timestamp_ns = time.monotonic_ns()
    counter_start_mj = backend.total_energy_mj()
    host_start_ns = time.monotonic_ns()
    with torch.inference_mode():
        for _ in range(repeats):
            expert(activation)
    end_event.record()
    end_event.synchronize()
    host_end_ns = time.monotonic_ns()
    counter_end_mj = backend.total_energy_mj()
    counter_end_timestamp_ns = time.monotonic_ns()
    samples = sampler.stop()

    energy_j = counter_delta_j(counter_start_mj, counter_end_mj)
    denominator = exact_expert_row_denominator(repeats=repeats, rows=rows)
    cuda_elapsed_ms = float(start_event.elapsed_time(end_event))
    invalid_reasons = list(validate_telemetry_window(
        samples,
        max_gap_s=max_gap_s,
        max_temperature_range_c=max_temperature_range_c,
    ))
    host_elapsed_s = (host_end_ns - host_start_ns) / 1_000_000_000
    if host_elapsed_s < minimum_window_s:
        invalid_reasons.append("workload_window_below_minimum")
    trial_row: dict[str, object] = {
        "rows": rows,
        "trial": trial,
        "tier": "provider_default_uncontrolled",
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "source_request_ids": list(request_ids),
        "source_request_count": len(request_ids),
        "inner_repeats": repeats,
        "logical_batches": repeats,
        "processed_expert_rows": denominator,
        "counter_start_mj": counter_start_mj,
        "counter_end_mj": counter_end_mj,
        "counter_start_timestamp_ns": counter_start_timestamp_ns,
        "counter_end_timestamp_ns": counter_end_timestamp_ns,
        "boundary_semantics": "SEQUENTIAL_BRACKETING_NOT_ATOMIC",
        "raw_board_energy_j": energy_j,
        "raw_j_per_logical_batch": energy_j / repeats,
        "raw_j_per_expert_row": energy_j / denominator,
        "cuda_elapsed_ms": cuda_elapsed_ms,
        "cuda_us_per_logical_batch": cuda_elapsed_ms * 1000.0 / repeats,
        "host_start_ns": host_start_ns,
        "host_end_ns": host_end_ns,
        "host_elapsed_s": host_elapsed_s,
        "telemetry_samples": len(samples),
        "telemetry_max_gap_s": max(
            (int(right["monotonic_ns"]) - int(left["monotonic_ns"]))
            / 1_000_000_000
            for left, right in zip(samples, samples[1:])
        ),
        "temperature_min_c": min(float(sample["temperature_c"]) for sample in samples),
        "temperature_max_c": max(float(sample["temperature_c"]) for sample in samples),
        "graphics_clock_median_mhz": statistics.median(
            float(sample["graphics_clock_mhz"]) for sample in samples
        ),
        "memory_clock_median_mhz": statistics.median(
            float(sample["memory_clock_mhz"]) for sample in samples
        ),
        "power_limit_w": float(samples[0]["power_limit_w"]),
        "gpu_uuid": backend.gpu_uuid,
        "formal_result": False,
    }
    tagged_samples = tuple(
        {"rows": rows, "trial": trial, **sample} for sample in samples
    )
    return trial_row, tagged_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("olmoe", "llm_jp"), required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--outer-trials", type=int, default=4)
    parser.add_argument("--minimum-window-seconds", type=float, default=10.0)
    parser.add_argument("--maximum-inner-repeats", type=int, default=1_048_576)
    parser.add_argument("--sample-interval-ms", type=float, default=5.0)
    parser.add_argument("--maximum-sample-gap-ms", type=float, default=20.0)
    parser.add_argument("--thermal-warmup-seconds", type=float, default=60.0)
    parser.add_argument("--maximum-warmup-seconds", type=float, default=180.0)
    parser.add_argument("--thermal-stable-window-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-temperature-range-c", type=float, default=2.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")
    if (
        any(row <= 0 for row in args.rows)
        or args.outer_trials < 2
        or args.minimum_window_seconds < 10.0
        or args.sample_interval_ms <= 0
        or args.maximum_sample_gap_ms < args.sample_interval_ms
        or args.thermal_warmup_seconds < 60.0
        or args.thermal_stable_window_seconds < 30.0
        or args.maximum_warmup_seconds < args.thermal_warmup_seconds
    ):
        raise SystemExit("characterization timing/grid arguments violate the frozen minimums")
    output_dir.mkdir(parents=True)
    for child in ("raw", "processed", "logs"):
        (output_dir / child).mkdir()

    import torch

    if not torch.cuda.is_available():
        raise CharacterizationError("CUDA is required; CPU fallback is forbidden")

    here = Path(__file__).resolve().parent
    repo_root = next(
        candidate for candidate in here.parents if (candidate / "experiments/shared").is_dir()
    )
    cjc_experiments = repo_root / "docs/archive/receiver_aware/cjc/experiments"
    joulequeue_experiments = repo_root / "docs/ideas/energy_slo/joulequeue/experiments"
    shared = repo_root / "experiments/shared"
    for path in (cjc_experiments, joulequeue_experiments, shared):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from capture_cjc_routes_gpu import MODEL_SPECS
    from capture_joulequeue_expert_inputs_gpu import (
        _expert_modules,
        _load_model_and_tokenizer,
    )

    capture_path = args.capture.resolve()
    capture = torch.load(capture_path, map_location="cpu", weights_only=False)
    if not isinstance(capture, dict) or not isinstance(capture.get("records"), list):
        raise CharacterizationError("capture artifact has an invalid schema")
    metadata = capture.get("metadata")
    if not isinstance(metadata, dict):
        raise CharacterizationError("capture artifact has no metadata")
    spec = MODEL_SPECS[args.model_key]
    expected_revision = f"{spec['model_id']}@{spec['revision']}"
    if metadata.get("model_revision") != expected_revision:
        raise CharacterizationError("capture/model revision mismatch")
    if metadata.get("input_source") != "measured_same_gpu_model_forward":
        raise CharacterizationError("capture is not a real same-GPU activation artifact")

    records_by_expert: dict[tuple[int, int], list[tuple[int, Mapping[str, object]]]] = defaultdict(list)
    for index, record in enumerate(capture["records"]):
        if not isinstance(record, dict):
            raise CharacterizationError("capture record is not a mapping")
        key = int(record["layer_id"]), int(record["expert_id"])
        records_by_expert[key].append((index, record))

    selected_key: tuple[int, int] | None = None
    selected_plans: dict[tuple[int, int], ActivationGroupPlan] | None = None
    for key, candidates in sorted(
        records_by_expert.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        descriptors = [
            ActivationRecord(
                request_id=str(record["request_id"]),
                forward_id=str(record["forward_id"]),
                row_count=int(record["row_count"]),
                record_index=index,
            )
            for index, record in candidates
        ]
        try:
            plans = plan_disjoint_activation_groups(
                descriptors, row_grid=args.rows, trials=args.outer_trials
            )
        except CharacterizationError:
            continue
        selected_key = key
        selected_plans = plans
        break
    if selected_key is None or selected_plans is None:
        raise CharacterizationError("no captured expert can supply disjoint activation groups")

    model, _tokenizer = _load_model_and_tokenizer(
        SimpleNamespace(cache_dir=args.cache_dir, allow_download=args.allow_download),
        spec,
    )
    experts = _expert_modules(model)
    if selected_key not in experts:
        raise CharacterizationError(f"selected expert {selected_key} is absent from model")
    expert = experts[selected_key]

    activation_groups: dict[tuple[int, int], object] = {}
    for key, plan in selected_plans.items():
        pieces = [capture["records"][index]["activation"] for index in plan.record_indices]
        if any(not isinstance(piece, torch.Tensor) or piece.ndim != 2 for piece in pieces):
            raise CharacterizationError("capture contains malformed activation tensors")
        activation = torch.cat(pieces, dim=0)[: plan.rows]
        if int(activation.shape[0]) != plan.rows:
            raise CharacterizationError("activation group has the wrong row count")
        activation_groups[key] = activation.to(device="cuda:0", dtype=torch.bfloat16)

    backend = NVMLBackend(0)
    warmup_samples: tuple[dict[str, object], ...] = ()
    raw_trials: list[dict[str, object]] = []
    raw_telemetry: list[dict[str, object]] = []
    try:
        properties = torch.cuda.get_device_properties(0)
        if "RTX 5090" not in properties.name:
            raise CharacterizationError(f"unexpected GPU for this runner: {properties.name}")
        representative = activation_groups[(max(args.rows), 0)]
        warmup_samples, thermal_stable = _run_warmup(
            expert,
            representative,
            backend,
            interval_s=args.sample_interval_ms / 1000.0,
            minimum_s=args.thermal_warmup_seconds,
            maximum_s=args.maximum_warmup_seconds,
            stable_window_s=args.thermal_stable_window_seconds,
            maximum_temperature_range_c=args.maximum_temperature_range_c,
        )
        warmup_reasons = validate_telemetry_window(
            warmup_samples,
            # The 20 ms gap threshold is a workload-window accounting gate.
            # Warmup is not integrated into a scientific energy sample, so an
            # occasional host scheduling gap must not invalidate the later,
            # independently sampled workload windows.  Warmup still fails on
            # non-monotonic timestamps, UUID/power-limit drift, bad thermal
            # readings, or failure to reach the frozen final-window stability.
            max_gap_s=math.inf,
            max_temperature_range_c=max(
                args.maximum_temperature_range_c,
                max(float(row["temperature_c"]) for row in warmup_samples)
                - min(float(row["temperature_c"]) for row in warmup_samples),
            ),
        )
        if warmup_reasons or not thermal_stable:
            raise CharacterizationError(
                "thermal warmup failed: "
                + ",".join((*warmup_reasons, *( () if thermal_stable else ("not_stable",) )))
            )

        repeats_by_rows = {
            rows: _calibrate_repeats(
                expert,
                activation_groups[(rows, 0)],
                minimum_window_s=args.minimum_window_seconds,
                maximum_repeats=args.maximum_inner_repeats,
            )
            for rows in sorted(set(args.rows))
        }
        # Forward/reverse blocks distribute row-size order across thermal time.
        for trial in range(args.outer_trials):
            row_order = sorted(set(args.rows), reverse=bool(trial % 2))
            for rows in row_order:
                plan = selected_plans[(rows, trial)]
                trial_row, telemetry = _run_trial(
                    expert,
                    activation_groups[(rows, trial)],
                    backend,
                    rows=rows,
                    trial=trial,
                    repeats=repeats_by_rows[rows],
                    request_ids=plan.request_ids,
                    interval_s=args.sample_interval_ms / 1000.0,
                    max_gap_s=args.maximum_sample_gap_ms / 1000.0,
                    max_temperature_range_c=args.maximum_temperature_range_c,
                    minimum_window_s=args.minimum_window_seconds,
                )
                raw_trials.append(trial_row)
                raw_telemetry.extend(telemetry)
    finally:
        backend.close()

    summaries = _summarize_trials(
        raw_trials, bootstrap=args.bootstrap, seed=args.seed
    )
    warmup_max_gap_s = max(
        (
            int(right["monotonic_ns"]) - int(left["monotonic_ns"])
        )
        / 1_000_000_000
        for left, right in zip(warmup_samples, warmup_samples[1:])
    )
    valid_counts = {
        rows: sum(
            bool(trial["valid"]) and int(trial["rows"]) == rows
            for trial in raw_trials
        )
        for rows in set(args.rows)
    }
    requested_cells = set(args.rows)
    status = characterization_status(
        valid_counts,
        requested_rows=tuple(requested_cells),
        trials=args.outer_trials,
    )
    config = {
        "schema": "routeslack-rtx5090-expert-energy-characterization-v1",
        "formal_result": False,
        "scope": EVIDENCE_SCOPE,
        "model_key": args.model_key,
        "model_revision": expected_revision,
        "capture": str(capture_path),
        "capture_sha256": _sha256(capture_path),
        "selected_layer": selected_key[0],
        "selected_expert": selected_key[1],
        "rows": sorted(set(args.rows)),
        "outer_trials": args.outer_trials,
        "minimum_window_seconds": args.minimum_window_seconds,
        "sample_interval_ms": args.sample_interval_ms,
        "maximum_sample_gap_ms": args.maximum_sample_gap_ms,
        "thermal_warmup_seconds": args.thermal_warmup_seconds,
        "thermal_stable_window_seconds": args.thermal_stable_window_seconds,
        "maximum_temperature_range_c": args.maximum_temperature_range_c,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "formal_blockers": list(FORMAL_BLOCKERS),
    }
    environment = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_uuid": raw_trials[0]["gpu_uuid"] if raw_trials else None,
        "driver_version": backend.driver_version,
        "power_limit_w": raw_trials[0]["power_limit_w"] if raw_trials else None,
        "total_energy_counter_required": True,
        "power_or_clock_actuator_available": False,
    }
    command = " ".join(shlex.quote(value) for value in sys.argv)
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "environment.json", environment)
    _write_text(output_dir / "commands.sh", command + "\n")
    _write_jsonl(
        output_dir / "raw/warmup_telemetry.jsonl",
        ({"phase": "warmup", **sample} for sample in warmup_samples),
    )
    _write_jsonl(output_dir / "raw/trials.jsonl", raw_trials)
    _write_jsonl(output_dir / "raw/telemetry.jsonl", raw_telemetry)
    _write_json(
        output_dir / "processed/summary.json",
        {
            "schema": "routeslack-rtx5090-expert-energy-characterization-v1",
            "status": status,
            "formal_result": False,
            "scope": EVIDENCE_SCOPE,
            "physical_windows": len(raw_trials),
            "valid_physical_windows": sum(bool(row["valid"]) for row in raw_trials),
            "invalid_physical_windows": sum(not bool(row["valid"]) for row in raw_trials),
            "warmup_telemetry_samples": len(warmup_samples),
            "warmup_max_gap_s": warmup_max_gap_s,
            "workload_maximum_sample_gap_s": args.maximum_sample_gap_ms / 1000.0,
            "cells": summaries,
            "formal_blockers": list(FORMAL_BLOCKERS),
            "interpretation": (
                "Physical default-tier expert-stage characterization only. "
                "These values are not J/SLO-completed-token and cannot authorize Gate 1."
            ),
        },
    )
    _write_text(
        output_dir / "verdict.md",
        "# Verdict\n\nMEASUREMENT_ONLY\n\n"
        "This artifact contains physical RTX 5090 expert-stage measurements but "
        "is not a RouteSlack formal result.\n",
    )
    source_files = (
        Path(__file__).resolve(),
        (joulequeue_experiments / "capture_joulequeue_expert_inputs_gpu.py").resolve(),
        (cjc_experiments / "capture_cjc_routes_gpu.py").resolve(),
        (shared / "capture_moe.py").resolve(),
    )
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "schema": "routeslack-rtx5090-expert-energy-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "formal_result": False,
        "scope": EVIDENCE_SCOPE,
        "model_revision": expected_revision,
        "capture_sha256": _sha256(capture_path),
        "source_sha256": {
            str(path.relative_to(repo_root)): _sha256(path) for path in source_files
        },
        "files": {
            str(path.relative_to(output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        },
        "formal_blockers": list(FORMAL_BLOCKERS),
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest | {"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
