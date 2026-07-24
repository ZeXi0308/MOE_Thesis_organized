#!/usr/bin/env python3
"""Measure paired separate-vs-coalesced BF16 expert calls on one CUDA GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
RECEIVER_EXPERIMENTS = REPO_ROOT / "docs" / "ideas" / "receiver_aware" / "experiments"
if str(RECEIVER_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(RECEIVER_EXPERIMENTS))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_cjc_routes_gpu import MODEL_SPECS, sha256_file  # noqa: E402
from capture_joulequeue_expert_inputs_gpu import (  # noqa: E402
    _expert_modules,
    _load_model_and_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=HERE / "configs" / "joulequeue_surface_v1.json"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE.parent / "JouleQueue_Phase2_冻结实验协议_2026-07-22.md",
    )
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_config(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError("surface config schema_version must be 1")
    if value.get("row_grid") != [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        raise RuntimeError("row grid drifted from JouleQueue v1")
    interval = float(value["power_sample_interval_seconds"])
    hard_gap = float(value["formal_max_observed_gap_seconds"])
    if not 0 < interval < hard_gap <= 0.020:
        raise RuntimeError("power sampler must leave margin below the 20 ms hard gate")
    expected_contract = {
        "evidence_level": "REAL_5090_NATIVE_ACTIVATIONS",
        "energy_basis": "total_during_launch",
        "paired_order": "AB_BA",
        "counter_sample_logical_window_bracketed": True,
        "background_sampler_exceptions_propagated": True,
    }
    if value.get("artifact_contract") != expected_contract:
        raise RuntimeError("surface artifact contract drifted from JouleQueue v1")
    if int(value.get("warmup_calls", 0)) < 20:
        raise RuntimeError("formal surface requires at least 20 warm-up calls")
    if int(value.get("independent_trials", 0)) < 10:
        raise RuntimeError("formal surface requires at least 10 independent trials")
    if float(value.get("minimum_window_seconds", 0.0)) < 2.0:
        raise RuntimeError("formal surface requires a measurement window of at least 2 seconds")
    if value.get("numerical_gate") != {
        "max_abs_error": 0.02,
        "mean_abs_error": 0.002,
        "max_cosine_error": 0.0001,
    }:
        raise RuntimeError("surface numerical gate drifted from JouleQueue v1")
    return value


def _source_hash() -> str:
    paths = (
        Path(__file__),
        HERE / "capture_joulequeue_expert_inputs_gpu.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _normalise_uuid(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    result = str(value).strip().upper()
    return result[4:] if result.startswith("GPU-") else result


class NVMLWindowMeter:
    """Sequentially bracket one logical workload with counter and 5 ms samples."""

    def __init__(self, device: torch.device, interval_s: float, max_gap_s: float) -> None:
        try:
            import pynvml  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment capability
            raise RuntimeError("pynvml is required for board-energy measurement") from exc
        self.nvml = pynvml
        self.nvml.nvmlInit()
        self.device = device
        self.interval_s = interval_s
        self.max_gap_s = max_gap_s
        cuda_uuid = _normalise_uuid(torch.cuda.get_device_properties(device).uuid)
        self.handle = None
        self.nvml_uuid = None
        for index in range(self.nvml.nvmlDeviceGetCount()):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(index)
            uuid = _normalise_uuid(self.nvml.nvmlDeviceGetUUID(handle))
            if uuid == cuda_uuid:
                self.handle = handle
                self.nvml_uuid = uuid
                break
        if self.handle is None or self.nvml_uuid != cuda_uuid:
            raise RuntimeError(f"CUDA/NVML UUID mismatch: cuda={cuda_uuid}, nvml={self.nvml_uuid}")
        try:
            self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.has_counter = True
        except (AttributeError, self.nvml.NVMLError_NotSupported):
            self.has_counter = False

    @property
    def gpu_uuid(self) -> str:
        return f"GPU-{self.nvml_uuid}"

    def _power_sample(self) -> tuple[int, float]:
        assert self.handle is not None
        timestamp_ns = time.monotonic_ns()
        power_w = float(self.nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
        if not math.isfinite(power_w) or power_w < 0:
            raise RuntimeError("NVML returned invalid board power")
        return timestamp_ns, power_w

    def measure(self, workload: Callable[[], None]) -> dict[str, object]:
        assert self.handle is not None
        torch.cuda.synchronize(self.device)
        samples: list[tuple[int, float]] = []
        errors: list[BaseException] = []
        stop = threading.Event()

        def worker() -> None:
            try:
                while not stop.wait(self.interval_s):
                    samples.append(self._power_sample())
            except BaseException as exc:  # propagate from sampler thread
                errors.append(exc)
                stop.set()

        start_counter_read_ns = time.monotonic_ns()
        start_mj = (
            int(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
            if self.has_counter
            else None
        )
        samples.append(self._power_sample())
        workload_start_ns = time.monotonic_ns()
        thread = threading.Thread(target=worker, name="joulequeue-nvml", daemon=True)
        thread.start()
        try:
            workload()
            torch.cuda.synchronize(self.device)
        finally:
            workload_end_ns = time.monotonic_ns()
            stop.set()
            thread.join(timeout=max(1.0, 5 * self.interval_s))
            if thread.is_alive():
                raise RuntimeError("NVML sampler thread did not stop")
            samples.append(self._power_sample())
        end_mj = (
            int(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
            if self.has_counter
            else None
        )
        end_counter_read_ns = time.monotonic_ns()
        if errors:
            raise RuntimeError("NVML sampler background failure") from errors[0]
        samples.sort(key=lambda sample: sample[0])
        gaps = [
            (right[0] - left[0]) / 1e9 for left, right in zip(samples, samples[1:])
        ]
        if any(gap <= 0 for gap in gaps):
            raise RuntimeError("non-monotonic power samples")
        max_gap = max(gaps, default=0.0)
        if max_gap > self.max_gap_s + 1e-12:
            raise RuntimeError(
                f"observed NVML sample gap {max_gap:.6f}s exceeds {self.max_gap_s:.6f}s"
            )
        if start_mj is not None and end_mj is not None:
            if end_mj < start_mj:
                raise RuntimeError("NVML total-energy counter moved backwards")
            energy_j = (end_mj - start_mj) / 1000.0
            source = "nvml_total_energy_counter"
        else:
            energy_j = 0.0
            for left, right in zip(samples, samples[1:]):
                energy_j += (
                    0.5 * (left[1] + right[1]) * (right[0] - left[0]) / 1e9
                )
            source = "monotonic_power_integral"
        if energy_j <= 0:
            raise RuntimeError("measured board energy must be positive")
        first_sample_ns = samples[0][0]
        last_sample_ns = samples[-1][0]
        logical_window_bracketed = (
            start_counter_read_ns <= first_sample_ns <= workload_start_ns
            and workload_start_ns <= workload_end_ns
            and workload_end_ns <= last_sample_ns <= end_counter_read_ns
        )
        if not logical_window_bracketed:
            raise RuntimeError("counter/sample measurements do not bracket one workload window")
        return {
            "energy_j": energy_j,
            "source": source,
            "gpu_uuid": self.gpu_uuid,
            "workload_start_ns": workload_start_ns,
            "workload_end_ns": workload_end_ns,
            "counter_start_read_ns": start_counter_read_ns,
            "counter_end_read_ns": end_counter_read_ns,
            "sample_count": len(samples),
            "max_sample_gap_s": max_gap,
            "first_sample_ns": first_sample_ns,
            "last_sample_ns": last_sample_ns,
            "counter_sample_logical_window_bracketed": logical_window_bracketed,
            "counter_sample_boundary_relation": "SEQUENTIAL_BRACKETING_NOT_ATOMIC",
        }


def _run_arm(expert: Any, activation: torch.Tensor, arm: str, repeats: int) -> None:
    output = None
    rows = int(activation.shape[0])
    for _ in range(repeats):
        if arm == "coalesced":
            output = expert(activation)
        elif arm == "separate":
            pieces = [expert(activation[index : index + 1]) for index in range(rows)]
            output = torch.cat(pieces, dim=0)
        else:
            raise ValueError(f"unknown arm: {arm}")
    if output is None:
        raise AssertionError("measurement repeats must be positive")


def _cuda_elapsed_s(
    device: torch.device, expert: Any, activation: torch.Tensor, arm: str, repeats: int
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    _run_arm(expert, activation, arm, repeats)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


def _choose_repeats(
    device: torch.device,
    expert: Any,
    activation: torch.Tensor,
    arm: str,
    minimum_seconds: float,
    maximum: int,
) -> int:
    repeats = 1
    while True:
        elapsed = _cuda_elapsed_s(device, expert, activation, arm, repeats)
        if elapsed >= minimum_seconds:
            return repeats
        if repeats >= maximum:
            raise RuntimeError("cannot reach the frozen minimum measurement window")
        repeats = min(maximum, repeats * 2)


def _numerical_errors(expert: Any, activation: torch.Tensor) -> dict[str, float]:
    separate = torch.cat(
        [expert(activation[index : index + 1]) for index in range(activation.shape[0])],
        dim=0,
    ).float()
    coalesced = expert(activation).float()
    delta = separate - coalesced
    cosine = torch.nn.functional.cosine_similarity(separate, coalesced, dim=-1)
    return {
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "max_cosine_error": float((1.0 - cosine).max().item()),
    }


def _bootstrap_mean_ci(
    values: Sequence[float], resamples: int, seed: int
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap requires non-empty values")
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return statistics.fmean(values), lower, upper


def _load_capture(path: Path, expected_model_revision: str) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, Mapping) or artifact.get("schema_version") != 1:
        raise RuntimeError("expert-input capture schema_version must be 1")
    metadata = artifact.get("metadata")
    records = artifact.get("records")
    if not isinstance(metadata, Mapping) or not isinstance(records, list) or not records:
        raise RuntimeError("capture requires metadata and non-empty records")
    expected = {
        "capture_kind": "joulequeue_real_bf16_expert_inputs",
        "input_source": "measured_same_gpu_model_forward",
        "model_revision": expected_model_revision,
        "data_split": "calibration",
    }
    for key, wanted in expected.items():
        if metadata.get(key) != wanted:
            raise RuntimeError(f"capture contract mismatch for {key}")
    selected_raw = metadata.get("selected_experts")
    if not isinstance(selected_raw, list):
        raise RuntimeError("capture lacks frozen selected-expert identities")
    selected = [tuple(map(int, key)) for key in selected_raw]
    if len(selected) != 16 or len(set(selected)) != 16:
        raise RuntimeError("capture must contain 16 unique selected experts")
    selected_layers = {layer for layer, _expert in selected}
    if len(selected_layers) != 4 or any(
        sum(layer == selected_layer for layer, _expert in selected) != 4
        for selected_layer in selected_layers
    ):
        raise RuntimeError("capture must select four experts in each of four layers")
    return metadata, records


def _require_signoff(
    path: Path | None,
    *,
    protocol_sha: str,
    config_sha: str,
    source_sha: str,
    capture_sha: str,
) -> Mapping[str, Any]:
    if path is None or not path.is_file():
        raise RuntimeError("formal surface requires Phase-4 SIGNED-OFF")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "SIGNED-OFF",
        "joulequeue_protocol_sha256": protocol_sha,
        "joulequeue_surface_config_sha256": config_sha,
        "joulequeue_surface_source_sha256": source_sha,
        "joulequeue_capture_sha256": capture_sha,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeError(f"formal signoff mismatch for {key}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _surface_curves(
    trials: Sequence[Mapping[str, object]],
    selected: Sequence[tuple[int, int]],
    config: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Summarise independent trials without treating inner repeats as samples."""

    expected_trials = int(config["independent_trials"])
    resamples = int(config["bootstrap_resamples"])
    base_seed = int(config["bootstrap_seed"])
    maximum_cv = float(config["maximum_energy_cv"])
    curves: list[dict[str, object]] = []
    for expert_index, (layer_id, expert_id) in enumerate(selected):
        points: list[dict[str, object]] = []
        for row_index, rows in enumerate(map(int, config["row_grid"])):
            group = [
                trial
                for trial in trials
                if int(trial["layer_id"]) == layer_id
                and int(trial["expert_id"]) == expert_id
                and int(trial["rows"]) == rows
            ]
            if len(group) != expected_trials or {
                int(trial["trial"]) for trial in group
            } != set(range(expected_trials)):
                raise RuntimeError(
                    f"incomplete independent trials for expert={(layer_id, expert_id)}, rows={rows}"
                )
            orders = {str(trial["arm_order"]) for trial in group}
            paired_order_complete = {
                "separate>coalesced",
                "coalesced>separate",
            }.issubset(orders)
            energy_values = [float(trial["coalesced_energy_j"]) for trial in group]
            latency_values = [float(trial["coalesced_latency_us"]) for trial in group]
            delta_values = [float(trial["delta_energy_j"]) for trial in group]
            seed = base_seed + expert_index * 1000 + row_index * 10
            energy_mean, energy_lcb, energy_ucb = _bootstrap_mean_ci(
                energy_values, resamples, seed
            )
            latency_mean, latency_lcb, latency_ucb = _bootstrap_mean_ci(
                latency_values, resamples, seed + 1
            )
            delta_mean, delta_lcb, delta_ucb = _bootstrap_mean_ci(
                delta_values, resamples, seed + 2
            )
            energy_cv = (
                statistics.stdev(energy_values) / energy_mean
                if len(energy_values) > 1
                else 0.0
            )
            numerical_pass = all(bool(trial["numerical_pass"]) for trial in group)
            point_measurement_valid = (
                energy_cv <= maximum_cv
                and paired_order_complete
                and all(
                    bool(trial["counter_sample_logical_window_bracketed"])
                    for trial in group
                )
            )
            points.append(
                {
                    "rows": rows,
                    "energy_j": energy_mean,
                    "energy_lcb95_j": energy_lcb,
                    "energy_ucb95_j": energy_ucb,
                    "latency_us": latency_mean,
                    "latency_lcb95_us": latency_lcb,
                    "latency_ucb95_us": latency_ucb,
                    "separate_minus_coalesced_energy_mean_j": delta_mean,
                    "separate_minus_coalesced_energy_lcb95_j": delta_lcb,
                    "separate_minus_coalesced_energy_ucb95_j": delta_ucb,
                    "independent_trials": len(group),
                    "energy_cv_across_trials": energy_cv,
                    "paired_order_complete": paired_order_complete,
                    "measurement_valid": point_measurement_valid,
                    "numerical_gate_passed": numerical_pass,
                }
            )
        curves.append(
            {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "energy_basis": "total_during_launch",
                "all_measurements_valid": all(
                    bool(point["measurement_valid"]) for point in points
                ),
                "all_numerical_gates_passed": all(
                    bool(point["numerical_gate_passed"]) for point in points
                ),
                "points": points,
            }
        )
    return curves


def _surface_metadata(
    *,
    mode: str,
    signoff_verified: bool,
    signoff_sha256: str | None,
    model_revision: str,
    gpu_name: str,
    gpu_uuid: str,
    capture_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    trials: Sequence[Mapping[str, object]],
    curves: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a fail-closed metadata contract for the formal surface loader."""

    energy_sources = sorted({str(trial["energy_source"]) for trial in trials})
    measured_uuids = {str(trial["gpu_uuid"]) for trial in trials}
    max_gap_ms = 1000.0 * max(float(trial["max_sample_gap_s"]) for trial in trials)
    observed_window_s = min(
        (
            int(trial[f"{arm}_workload_end_ns"])
            - int(trial[f"{arm}_workload_start_ns"])
        )
        / 1e9
        for trial in trials
        for arm in ("separate", "coalesced")
    )
    logical_window_bracketed = all(
        bool(trial["counter_sample_logical_window_bracketed"]) for trial in trials
    )
    paired_order_complete = all(
        bool(point["paired_order_complete"])
        for curve in curves
        for point in curve["points"]  # type: ignore[index]
    )
    all_measurements_valid = all(
        bool(curve["all_measurements_valid"]) for curve in curves
    )
    all_numerical_gates_passed = all(
        bool(curve["all_numerical_gates_passed"]) for curve in curves
    )
    native_activations = (
        capture_metadata.get("capture_kind")
        == "joulequeue_real_bf16_expert_inputs"
        and capture_metadata.get("input_source")
        == "measured_same_gpu_model_forward"
    )
    is_rtx_5090 = "RTX 5090" in gpu_name
    evidence_level = (
        "REAL_5090_NATIVE_ACTIVATIONS"
        if native_activations and is_rtx_5090
        else "NON_FORMAL_GPU_OR_INPUT_EVIDENCE"
    )
    contract = config["artifact_contract"]
    sampling_interval_ms = 1000.0 * float(config["power_sample_interval_seconds"])
    signoff_gate = (
        signoff_verified
        and signoff_sha256 is not None
        and len(signoff_sha256) == 64
        and all(character in "0123456789abcdef" for character in signoff_sha256)
    )
    reasons: list[str] = []
    gates = {
        "formal_mode": mode == "formal",
        "phase4_signoff_verified": signoff_gate,
        "evidence_level": evidence_level == contract["evidence_level"],
        "native_activations": native_activations,
        "gpu_uuid_consistent": measured_uuids == {gpu_uuid} and bool(gpu_uuid),
        "single_energy_source": len(energy_sources) == 1,
        "paired_ab_ba": paired_order_complete,
        "counter_sample_logical_window_bracketed": logical_window_bracketed,
        "minimum_window": observed_window_s
        >= float(config["minimum_window_seconds"])
        >= 2.0,
        "independent_trials": int(config["independent_trials"]) >= 10,
        "sampling_interval": sampling_interval_ms == 5.0,
        "observed_gap": max_gap_ms <= 20.0,
        "measurement_repeatability": all_measurements_valid,
        "numerical_equivalence": all_numerical_gates_passed,
    }
    for name, passed in gates.items():
        if not passed:
            reasons.append(name)
    formal_eligible = not reasons
    return {
        "artifact_schema": "joulequeue-expert-surface-v1",
        "model_revision": model_revision,
        "evidence_level": evidence_level,
        "native_activations": native_activations,
        "formal_eligible": formal_eligible,
        "formal_eligibility_failed_gates": reasons,
        "phase4_signoff_verified": signoff_gate,
        "phase4_signoff_sha256": signoff_sha256,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "energy_source": energy_sources[0] if len(energy_sources) == 1 else energy_sources,
        "energy_basis": contract["energy_basis"],
        "paired_order": contract["paired_order"] if paired_order_complete else "INCOMPLETE",
        "counter_sample_logical_window_bracketed": logical_window_bracketed,
        "counter_sample_boundary_relation": "SEQUENTIAL_BRACKETING_NOT_ATOMIC",
        "background_sampler_exceptions_propagated": contract[
            "background_sampler_exceptions_propagated"
        ],
        "minimum_window_s": observed_window_s,
        "configured_minimum_window_s": float(config["minimum_window_seconds"]),
        "independent_trials": int(config["independent_trials"]),
        "sampling_interval_ms": sampling_interval_ms,
        "max_observed_gap_ms": max_gap_ms,
        "all_measurements_valid": all_measurements_valid,
        "all_numerical_gates_passed": all_numerical_gates_passed,
        "numerical_gate": dict(config["numerical_gate"]),
        "measurement_scope": "BF16_EXPERT_STAGE_ONLY_NOT_FULL_SERVING",
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; analytical surface fallback is forbidden")
    if args.output_dir.exists():
        raise RuntimeError("refusing to overwrite surface output directory")
    config = _load_config(args.config)
    spec = MODEL_SPECS[args.model_key]
    model_revision = f"{spec['model_id']}@{spec['revision']}"
    capture_metadata, records = _load_capture(args.capture, model_revision)
    if int(capture_metadata.get("selection_seed", -1)) != int(config["selection_seed"]):
        raise RuntimeError("capture selected-expert seed drifted from surface config")
    hashes = {
        "protocol": sha256_file(args.protocol),
        "config": sha256_file(args.config),
        "source": _source_hash(),
        "capture": sha256_file(args.capture),
    }
    signoff = None
    if args.mode == "formal":
        signoff = _require_signoff(
            args.signoff,
            protocol_sha=hashes["protocol"],
            config_sha=hashes["config"],
            source_sha=hashes["source"],
            capture_sha=hashes["capture"],
        )

    model, _tokenizer = _load_model_and_tokenizer(args, spec)
    experts = _expert_modules(model)
    selected = tuple(tuple(map(int, key)) for key in capture_metadata["selected_experts"])
    if any(key not in experts for key in selected):
        raise RuntimeError("capture selected-expert identity missing from pinned model")
    pools: dict[tuple[int, int], list[torch.Tensor]] = {key: [] for key in selected}
    seen_record_ids: set[tuple[str, int, int]] = set()
    for row in records:
        key = int(row["layer_id"]), int(row["expert_id"])
        if key not in pools:
            raise RuntimeError("capture contains an unselected expert")
        identity = str(row["forward_id"]), key[0], key[1]
        if identity in seen_record_ids:
            raise RuntimeError(f"duplicate expert-input event: {identity}")
        seen_record_ids.add(identity)
        activation = row["activation"]
        if not isinstance(activation, torch.Tensor) or activation.ndim != 2:
            raise RuntimeError("capture activation must be rank-2 Tensor")
        if activation.shape[0] != int(row["row_count"]):
            raise RuntimeError("capture activation/row_count mismatch")
        pools[key].append(activation.to(dtype=torch.bfloat16))
    maximum_rows = max(int(value) for value in config["row_grid"])
    for key, chunks in pools.items():
        if sum(int(chunk.shape[0]) for chunk in chunks) < maximum_rows:
            raise RuntimeError(f"selected expert {key} has fewer than {maximum_rows} real rows")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    meter = NVMLWindowMeter(
        device,
        float(config["power_sample_interval_seconds"]),
        float(config["formal_max_observed_gap_seconds"]),
    )
    trials: list[dict[str, object]] = []
    gate = config["numerical_gate"]
    with torch.inference_mode():
        for expert_index, key in enumerate(selected):
            expert = experts[key]
            pool = torch.cat(pools[key], dim=0)
            for rows in map(int, config["row_grid"]):
                activation = pool[:rows].to(device=device, dtype=torch.bfloat16)
                for _ in range(int(config["warmup_calls"])):
                    _run_arm(expert, activation, "separate", 1)
                    _run_arm(expert, activation, "coalesced", 1)
                torch.cuda.synchronize(device)
                errors = _numerical_errors(expert, activation)
                numerical_pass = (
                    errors["max_abs_error"] <= float(gate["max_abs_error"])
                    and errors["mean_abs_error"] <= float(gate["mean_abs_error"])
                    and errors["max_cosine_error"] <= float(gate["max_cosine_error"])
                )
                if not numerical_pass:
                    raise RuntimeError(f"numerical coalescing gate failed for expert={key}, rows={rows}")
                repeats = {
                    arm: _choose_repeats(
                        device,
                        expert,
                        activation,
                        arm,
                        float(config["minimum_window_seconds"]),
                        int(config["maximum_inner_repeats"]),
                    )
                    for arm in ("separate", "coalesced")
                }
                for trial in range(int(config["independent_trials"])):
                    order = (
                        ("coalesced", "separate")
                        if (trial + expert_index) % 2
                        else ("separate", "coalesced")
                    )
                    measured: dict[str, dict[str, object]] = {}
                    for arm in order:
                        latency_s = _cuda_elapsed_s(
                            device, expert, activation, arm, repeats[arm]
                        ) / repeats[arm]
                        energy = meter.measure(
                            lambda arm=arm: _run_arm(
                                expert, activation, arm, repeats[arm]
                            )
                        )
                        measured[arm] = {
                            **energy,
                            "energy_j_per_logical_batch": float(energy["energy_j"])
                            / repeats[arm],
                            "latency_us_per_logical_batch": latency_s * 1e6,
                            "repeats": repeats[arm],
                        }
                    if measured["separate"]["source"] != measured["coalesced"]["source"]:
                        raise RuntimeError("paired arms used different energy sources")
                    trials.append(
                        {
                            "model_key": args.model_key,
                            "layer_id": key[0],
                            "expert_id": key[1],
                            "rows": rows,
                            "trial": trial,
                            "arm_order": ">".join(order),
                            "energy_source": measured["separate"]["source"],
                            "gpu_uuid": measured["separate"]["gpu_uuid"],
                            "separate_repeats": measured["separate"]["repeats"],
                            "coalesced_repeats": measured["coalesced"]["repeats"],
                            "separate_energy_j": measured["separate"]["energy_j_per_logical_batch"],
                            "coalesced_energy_j": measured["coalesced"]["energy_j_per_logical_batch"],
                            "delta_energy_j": float(measured["separate"]["energy_j_per_logical_batch"])
                            - float(measured["coalesced"]["energy_j_per_logical_batch"]),
                            "separate_latency_us": measured["separate"]["latency_us_per_logical_batch"],
                            "coalesced_latency_us": measured["coalesced"]["latency_us_per_logical_batch"],
                            "delta_latency_us": float(measured["separate"]["latency_us_per_logical_batch"])
                            - float(measured["coalesced"]["latency_us_per_logical_batch"]),
                            "max_sample_gap_s": max(
                                float(measured["separate"]["max_sample_gap_s"]),
                                float(measured["coalesced"]["max_sample_gap_s"]),
                            ),
                            "counter_sample_logical_window_bracketed": bool(
                                measured["separate"]["counter_sample_logical_window_bracketed"]
                            )
                            and bool(
                                measured["coalesced"]["counter_sample_logical_window_bracketed"]
                            ),
                            "separate_workload_start_ns": measured["separate"]["workload_start_ns"],
                            "separate_workload_end_ns": measured["separate"]["workload_end_ns"],
                            "separate_counter_start_read_ns": measured["separate"]["counter_start_read_ns"],
                            "separate_counter_end_read_ns": measured["separate"]["counter_end_read_ns"],
                            "separate_first_sample_ns": measured["separate"]["first_sample_ns"],
                            "separate_last_sample_ns": measured["separate"]["last_sample_ns"],
                            "separate_sample_count": measured["separate"]["sample_count"],
                            "coalesced_workload_start_ns": measured["coalesced"]["workload_start_ns"],
                            "coalesced_workload_end_ns": measured["coalesced"]["workload_end_ns"],
                            "coalesced_counter_start_read_ns": measured["coalesced"]["counter_start_read_ns"],
                            "coalesced_counter_end_read_ns": measured["coalesced"]["counter_end_read_ns"],
                            "coalesced_first_sample_ns": measured["coalesced"]["first_sample_ns"],
                            "coalesced_last_sample_ns": measured["coalesced"]["last_sample_ns"],
                            "coalesced_sample_count": measured["coalesced"]["sample_count"],
                            "numerical_pass": numerical_pass,
                            **errors,
                        }
                    )

    curves = _surface_curves(trials, selected, config)
    summaries: list[dict[str, object]] = []
    surface_rows: dict[str, object] = {}
    for row_index, rows in enumerate(map(int, config["row_grid"])):
        unit_delta_energy = []
        unit_coalesced_energy = []
        unit_coalesced_latency = []
        for key in selected:
            group = [
                row for row in trials
                if int(row["rows"]) == rows
                and (int(row["layer_id"]), int(row["expert_id"])) == key
            ]
            unit_delta_energy.append(statistics.fmean(float(row["delta_energy_j"]) for row in group))
            unit_coalesced_energy.append(statistics.fmean(float(row["coalesced_energy_j"]) for row in group))
            unit_coalesced_latency.append(statistics.fmean(float(row["coalesced_latency_us"]) for row in group))
        delta_mean, delta_lcb, delta_ucb = _bootstrap_mean_ci(
            unit_delta_energy,
            int(config["bootstrap_resamples"]),
            int(config["bootstrap_seed"]) + row_index,
        )
        energy_mean, energy_lcb, energy_ucb = _bootstrap_mean_ci(
            unit_coalesced_energy,
            int(config["bootstrap_resamples"]),
            int(config["bootstrap_seed"]) + 100 + row_index,
        )
        latency_mean, latency_lcb, latency_ucb = _bootstrap_mean_ci(
            unit_coalesced_latency,
            int(config["bootstrap_resamples"]),
            int(config["bootstrap_seed"]) + 200 + row_index,
        )
        energy_cv = statistics.stdev(unit_coalesced_energy) / energy_mean if len(unit_coalesced_energy) > 1 else 0.0
        per_expert_points = [
            point
            for curve in curves
            for point in curve["points"]  # type: ignore[index]
            if int(point["rows"]) == rows
        ]
        summary = {
            "rows": rows,
            "independent_experts": len(selected),
            "delta_energy_mean_j": delta_mean,
            "delta_energy_lcb95_j": delta_lcb,
            "delta_energy_ucb95_j": delta_ucb,
            "coalesced_energy_mean_j": energy_mean,
            "coalesced_energy_lcb95_j": energy_lcb,
            "coalesced_energy_ucb95_j": energy_ucb,
            "coalesced_latency_mean_us": latency_mean,
            "coalesced_latency_lcb95_us": latency_lcb,
            "coalesced_latency_ucb95_us": latency_ucb,
            "coalesced_energy_cv_across_experts": energy_cv,
            "all_expert_cells_measurement_valid": all(
                bool(point["measurement_valid"]) for point in per_expert_points
            ),
            "all_expert_cells_numerical_gate_passed": all(
                bool(point["numerical_gate_passed"]) for point in per_expert_points
            ),
        }
        summaries.append(summary)
        surface_rows[str(rows)] = summary

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(args.output_dir / "expert_surface_trials.csv", trials)
    _write_csv(args.output_dir / "expert_surface.csv", summaries)
    gpu_name = str(torch.cuda.get_device_name(device))
    signoff_sha256 = sha256_file(args.signoff) if signoff else None
    metadata = _surface_metadata(
        mode=args.mode,
        signoff_verified=signoff is not None,
        signoff_sha256=signoff_sha256,
        model_revision=model_revision,
        gpu_name=gpu_name,
        gpu_uuid=meter.gpu_uuid,
        capture_metadata=capture_metadata,
        config=config,
        trials=trials,
        curves=curves,
    )
    status = {
        "schema_version": 1,
        "status": (
            "SURFACE_ONLY"
            if bool(metadata["formal_eligible"])
            else ("BLOCKED_SURFACE" if args.mode == "formal" else "NOT_TESTED")
        ),
        "scientific_result": False,
        "evidence_boundary": "REAL_5090_EXPERT_STAGE_ONLY / NOT_FULL_SERVING",
        "model_revision": model_revision,
        "gpu_uuid": meter.gpu_uuid,
        "energy_source": metadata["energy_source"],
        "all_measurements_valid": metadata["all_measurements_valid"],
        "all_numerical_gates_passed": metadata["all_numerical_gates_passed"],
        "hashes": hashes,
        "metadata": metadata,
        "curves": curves,
        "surface": surface_rows,
        "signoff_sha256": signoff_sha256,
    }
    (args.output_dir / "surface.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
