#!/usr/bin/env python3
"""Build a calibration-proxy route-row BF16-vs-FP8 expert surface on CUDA.

The runner consumes expert inputs captured from a *real continuous decode* hot
path.  It intentionally has no random-activation, full-forward, analytical, or
isolated pre-cast GEMM fallback.  A missing/invalid capture is an infrastructure
BLOCKED condition, not a scientific No-Go.  Even a positive local crossover is
only a mechanism diagnostic: this runner does not integrate the policy into a
continuous serving hot path and therefore cannot emit a scientific verdict.

Expected capture (``torch.save``) schema::

    {
      "schema_version": 1,
      "metadata": {
        "capture_kind": "continuous_decode_expert_inputs",
        "model_revision": "model@40-char-revision",
        "phase": "decode",
        "source_precision": "BF16",
        "prefill_once_per_request": true,
        "decode_input_length": 1,
        "kv_reused": true,
        "kv_monotonic": true,
        "independent_policy_state": true,
        "route_replay": false,
        "future_information_used": false,
        "cache_repack_complexity": "O(1)_metadata_only",
        "engine_name": "...", "engine_commit": "...",
        "capture_producer_sha256": "...", "data_manifest_sha256": "...",
        "arrival_config_sha256": "...", "data_split": "calibration"
      },
      "records": [{
        "iteration_id": 0, "batch_id": "...", "layer_id": 0,
        "active_decode_tokens": 4,
        "selected_experts": LongTensor[4, top_k],
        "expert_inputs": {expert_id: BFloat16Tensor[m, hidden]}
      }, ...]
    }

Records must cover every layer and must contain an active-batch trace with at
least one increase and one decrease.  ``selected_experts`` is recounted here;
the supplied expert input row counts must match it exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

try:
    from route_row_policy import (
        ACTION_BF16,
        ACTION_FP8,
        FP8_RECIPE_E4M3FN,
        ROW_BIN_LABELS,
        DualResidentExpertBank,
        RouteRowLUT,
        RuntimeCounters,
        SurfaceCell,
        SurfaceKey,
        code_config_sha256,
        require_cuda_fp8,
        route_row_counts,
        row_bin_for_count,
        sha256_file,
        verify_formal_signoff,
        write_json_atomic,
    )
except ImportError:  # pragma: no cover - supports package-style invocation
    from .route_row_policy import (
        ACTION_BF16,
        ACTION_FP8,
        FP8_RECIPE_E4M3FN,
        ROW_BIN_LABELS,
        DualResidentExpertBank,
        RouteRowLUT,
        RuntimeCounters,
        SurfaceCell,
        SurfaceKey,
        code_config_sha256,
        require_cuda_fp8,
        route_row_counts,
        row_bin_for_count,
        sha256_file,
        verify_formal_signoff,
        write_json_atomic,
    )


HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class ExpertObservation:
    event_id: str
    iteration_id: int
    batch_id: str
    layer_id: int
    expert_id: int
    row_count: int
    row_bin: str
    activation_cpu: torch.Tensor


@dataclass(frozen=True)
class ArmMeasurement:
    energy_j_per_call: float
    latency_ms_per_call: float
    repeats: int
    energy_method: str
    counter_delta: Mapping[str, int]


def parse_args() -> argparse.Namespace:
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Route-row full-expert FP8 break-even surface; no proxy fallback"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment_dir / "configs" / "route_row_break_even_v1.json",
    )
    parser.add_argument("--model-key", choices=("olmoe", "llm_jp"))
    parser.add_argument(
        "--capture",
        type=Path,
        help="torch.save artifact from the audited continuous-decode capture hook",
    )
    parser.add_argument(
        "--quality-artifact",
        type=Path,
        help="calibration in-loop quality-gate JSON; mandatory in --formal mode",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--formal",
        action="store_true",
        help=(
            "require Phase-4 SIGNED-OFF for a reviewed calibration run; "
            "this does not make the surface a scientific result"
        ),
    )
    parser.add_argument("--signoff", type=Path)
    parser.add_argument(
        "--print-code-config-hash",
        action="store_true",
        help="print the Phase-4 signoff hash and exit without touching CUDA",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "docs").is_dir():
            return candidate
    raise RuntimeError("cannot locate repository root for formal hash gate")


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid frozen config: {path}") from exc
    if config.get("schema_version") != 1:
        raise RuntimeError("unsupported config schema")
    if tuple(config.get("fixed_row_bins", ())) != ROW_BIN_LABELS:
        raise RuntimeError("config row bins drifted from the frozen Phase-2 bins")
    if config.get("fp8_recipe") != FP8_RECIPE_E4M3FN:
        raise RuntimeError("config FP8 recipe drifted from the implemented recipe")
    return config


def _current_code_config_hash(config: Mapping[str, Any], config_path: Path) -> str:
    root = _repo_root()
    manifest = config["formal_gate"]["hash_manifest"]
    named_paths = {str(name): root / str(name) for name in manifest}
    config_logical_names = [
        name for name in manifest if str(name).endswith("route_row_break_even_v1.json")
    ]
    if len(config_logical_names) != 1:
        raise RuntimeError("formal hash manifest must contain exactly one frozen config")
    manifest_config = (root / str(config_logical_names[0])).resolve()
    if config_path.resolve() != manifest_config:
        raise RuntimeError(
            "formal hash gate refuses a config outside the frozen hash manifest: "
            f"{config_path}"
        )
    return code_config_sha256(named_paths)


def _apply_formal_gate(
    config: Mapping[str, Any], config_path: Path, signoff_path: Optional[Path]
) -> tuple[str, Mapping[str, Any]]:
    current_hash = _current_code_config_hash(config, config_path)
    root = _repo_root()
    resolved_signoff = signoff_path
    if resolved_signoff is None:
        resolved_signoff = root / config["formal_gate"]["default_signoff_path"]
    signoff = verify_formal_signoff(resolved_signoff, current_hash)
    return current_hash, signoff


def _require_hex64(metadata: Mapping[str, Any], field_name: str) -> None:
    value = metadata.get(field_name)
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise RuntimeError(f"capture {field_name} must be a lowercase SHA-256")


def _load_capture(
    path: Path,
    model_spec: Mapping[str, Any],
    capture_contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[ExpertObservation]]:
    if not path.is_file():
        raise RuntimeError(f"continuous-decode capture missing; run is BLOCKED: {path}")
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"cannot read continuous-decode capture: {path}") from exc
    if not isinstance(artifact, Mapping) or artifact.get("schema_version") != 1:
        raise RuntimeError("capture schema_version must be 1")
    metadata = artifact.get("metadata")
    records = artifact.get("records")
    if not isinstance(metadata, Mapping) or not isinstance(records, list) or not records:
        raise RuntimeError("capture requires non-empty metadata and records")

    expected_model_revision = f"{model_spec['model_id']}@{model_spec['revision']}"
    exact_fields = {
        "capture_kind": capture_contract["capture_kind"],
        "model_revision": expected_model_revision,
        "phase": capture_contract["phase"],
        "source_precision": "BF16",
        "prefill_once_per_request": capture_contract["prefill_once_per_request"],
        "decode_input_length": capture_contract["decode_input_length"],
        "kv_reused": capture_contract["kv_reused"],
        "kv_monotonic": capture_contract["kv_monotonic"],
        "independent_policy_state": capture_contract["independent_policy_state"],
        "route_replay": capture_contract["route_replay"],
        "future_information_used": capture_contract["future_information_used"],
        "cache_repack_complexity": capture_contract["cache_repack_complexity"],
        "data_split": "calibration",
    }
    for field_name, expected in exact_fields.items():
        if metadata.get(field_name) != expected:
            raise RuntimeError(
                f"capture contract violation: {field_name}={metadata.get(field_name)!r}, "
                f"expected={expected!r}"
            )
    if metadata.get("capture_kind") in set(capture_contract["forbidden_capture_kinds"]):
        raise RuntimeError("forbidden proxy capture kind")
    engine_name = metadata.get("engine_name")
    engine_commit = metadata.get("engine_commit")
    if not isinstance(engine_name, str) or not engine_name.strip():
        raise RuntimeError("capture must name its continuous serving engine")
    if not isinstance(engine_commit, str) or GIT_REVISION.fullmatch(engine_commit) is None:
        raise RuntimeError("capture must pin an auditable serving-engine commit")
    for field_name in (
        "capture_producer_sha256",
        "data_manifest_sha256",
        "arrival_config_sha256",
    ):
        _require_hex64(metadata, field_name)

    num_layers = int(model_spec["num_layers"])
    num_experts = int(model_spec["num_experts"])
    top_k = int(model_spec["top_k"])
    hidden_size = int(model_spec["hidden_size"])
    observations: list[ExpertObservation] = []
    iteration_active: dict[int, int] = {}
    iteration_batch: dict[int, str] = {}
    iteration_layers: dict[int, set[int]] = {}
    seen_iteration_layers: set[tuple[int, int]] = set()
    seen_layers: set[int] = set()
    event_ids: set[str] = set()

    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"capture record {record_index} is not a mapping")
        try:
            iteration_id = int(record["iteration_id"])
            batch_id = str(record["batch_id"])
            layer_id = int(record["layer_id"])
            active_tokens = int(record["active_decode_tokens"])
            selected_experts = record["selected_experts"]
            expert_inputs = record["expert_inputs"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"malformed capture record {record_index}") from exc
        if iteration_id < 0 or not batch_id:
            raise RuntimeError(f"invalid iteration/batch identity in record {record_index}")
        if not 0 <= layer_id < num_layers:
            raise RuntimeError(f"layer id out of range in record {record_index}: {layer_id}")
        if active_tokens <= 0:
            raise RuntimeError("a captured decode iteration must have at least one active token")
        previous_active = iteration_active.setdefault(iteration_id, active_tokens)
        if previous_active != active_tokens:
            raise RuntimeError(f"active batch disagrees across layers at iteration {iteration_id}")
        previous_batch = iteration_batch.setdefault(iteration_id, batch_id)
        if previous_batch != batch_id:
            raise RuntimeError(f"batch identity disagrees across layers at iteration {iteration_id}")
        iteration_layer = (iteration_id, layer_id)
        if iteration_layer in seen_iteration_layers:
            raise RuntimeError(
                f"duplicate layer record at iteration={iteration_id}, layer={layer_id}"
            )
        seen_iteration_layers.add(iteration_layer)
        iteration_layers.setdefault(iteration_id, set()).add(layer_id)
        if not isinstance(expert_inputs, Mapping):
            raise RuntimeError(f"record {record_index} expert_inputs is not a mapping")

        counts = route_row_counts(
            selected_experts,
            num_experts,
            expected_active_decode_tokens=active_tokens,
            expected_top_k=top_k,
        )
        nonempty_ids = {int(index) for index in torch.nonzero(counts, as_tuple=False).flatten()}
        supplied_ids: set[int] = set()
        for raw_expert_id in expert_inputs:
            if isinstance(raw_expert_id, bool):
                raise RuntimeError("boolean expert id is forbidden")
            try:
                expert_id = int(raw_expert_id)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid expert id key: {raw_expert_id!r}") from exc
            if str(expert_id) != str(raw_expert_id) and not isinstance(raw_expert_id, int):
                raise RuntimeError(f"non-canonical expert id key: {raw_expert_id!r}")
            supplied_ids.add(expert_id)
        if supplied_ids != nonempty_ids:
            raise RuntimeError(
                f"record {record_index} expert-input identities mismatch router counts; "
                f"missing={sorted(nonempty_ids - supplied_ids)}, "
                f"extra={sorted(supplied_ids - nonempty_ids)}"
            )

        for raw_expert_id, activation in expert_inputs.items():
            expert_id = int(raw_expert_id)
            if not isinstance(activation, torch.Tensor):
                raise RuntimeError("expert activation must be a Tensor")
            if activation.ndim != 2 or activation.shape[1] != hidden_size:
                raise RuntimeError(
                    f"expert input shape mismatch at layer={layer_id}, expert={expert_id}: "
                    f"got={tuple(activation.shape)}, expected=[m,{hidden_size}]"
                )
            if not bool(torch.isfinite(activation).all().item()):
                raise RuntimeError(
                    f"non-finite expert input at layer={layer_id}, expert={expert_id}"
                )
            row_count = int(counts[expert_id].item())
            if activation.shape[0] != row_count:
                raise RuntimeError(
                    f"row-count identity mismatch at layer={layer_id}, expert={expert_id}: "
                    f"activation_rows={activation.shape[0]}, router_rows={row_count}"
                )
            if row_count == 0:
                raise RuntimeError("empty experts must not appear in expert_inputs")
            event_id = f"{iteration_id}:{batch_id}:{layer_id}:{expert_id}"
            if event_id in event_ids:
                raise RuntimeError(f"duplicate expert observation identity: {event_id}")
            event_ids.add(event_id)
            label = row_bin_for_count(row_count)
            if label is None:
                raise AssertionError("nonempty capture mapped to empty row bin")
            observations.append(
                ExpertObservation(
                    event_id=event_id,
                    iteration_id=iteration_id,
                    batch_id=batch_id,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    row_count=row_count,
                    row_bin=label,
                    activation_cpu=(
                        activation.detach()
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous()
                    ),
                )
            )
        seen_layers.add(layer_id)

    if seen_layers != set(range(num_layers)):
        raise RuntimeError(
            f"capture does not cover every layer: seen={sorted(seen_layers)}, "
            f"expected=0..{num_layers - 1}"
        )
    expected_layers = set(range(num_layers))
    incomplete_iterations = [
        iteration_id
        for iteration_id, layers in iteration_layers.items()
        if layers != expected_layers
    ]
    if incomplete_iterations:
        raise RuntimeError(
            "capture must include every layer for every retained iteration; "
            f"incomplete iterations={incomplete_iterations[:8]}"
        )
    active_trace = [iteration_active[index] for index in sorted(iteration_active)]
    differences = [right - left for left, right in zip(active_trace, active_trace[1:])]
    if not any(value > 0 for value in differences) or not any(value < 0 for value in differences):
        raise RuntimeError(
            "capture active batch did not exhibit both an increase and a decrease; "
            "fixed-batch traces are forbidden substitutes"
        )
    if not observations:
        raise RuntimeError("capture contains no non-empty expert observation")
    return metadata, observations


def _load_quality_gate(
    path: Optional[Path],
    model_revision: str,
    config: Mapping[str, Any],
    capture_metadata: Mapping[str, Any],
    *,
    formal: bool,
) -> tuple[bool, Optional[str], str]:
    if path is None:
        if formal:
            raise RuntimeError("formal surface BLOCKED: calibration in-loop quality artifact missing")
        return False, None, "quality_artifact_missing"
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid quality artifact: {path}") from exc
    required_exact = {
        "schema_version": 1,
        "model_revision": model_revision,
        "metric": config["quality_gate"]["metric"],
        "data_split": "calibration",
        "in_loop_decode": True,
        "independent_kv": True,
        "route_replay": False,
        "fp8_recipe": config["fp8_recipe"],
        "data_manifest_sha256": capture_metadata["data_manifest_sha256"],
        "arrival_config_sha256": capture_metadata["arrival_config_sha256"],
    }
    for field_name, expected in required_exact.items():
        if artifact.get(field_name) != expected:
            raise RuntimeError(
                f"quality artifact contract violation: {field_name}="
                f"{artifact.get(field_name)!r}, expected={expected!r}"
            )
    try:
        ci_upper = float(artifact["ci95_upper"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("quality artifact lacks finite ci95_upper") from exc
    if not math.isfinite(ci_upper):
        raise RuntimeError("quality ci95_upper must be finite")
    maximum = float(config["quality_gate"]["maximum_ci_upper"])
    passed = artifact.get("status") == "PASS" and ci_upper < maximum
    reason = "quality_gate_passed" if passed else "quality_gate_failed"
    return passed, sha256_file(path), reason


def _normalise_uuid(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    text = str(value).strip().upper()
    if text.startswith("GPU-"):
        text = text[4:]
    return text


class NvmlBoardEnergyMeter:
    """NVML total-energy delta, with a <=20 ms monotonic power fallback."""

    def __init__(self, device: torch.device, sampling_interval_s: float):
        if sampling_interval_s <= 0 or sampling_interval_s > 0.020:
            raise RuntimeError("fallback power sampling interval must be in (0, 20ms]")
        try:
            import pynvml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pynvml is required for GPU board-energy accounting") from exc
        self.nvml = pynvml
        self.nvml.nvmlInit()
        self.device = device
        self.sampling_interval_s = sampling_interval_s
        properties = torch.cuda.get_device_properties(device)
        torch_uuid_raw = getattr(properties, "uuid", None)
        if torch_uuid_raw is None:
            raise RuntimeError("CUDA device exposes no UUID; CUDA/NVML identity cannot be audited")
        self.torch_uuid = _normalise_uuid(torch_uuid_raw)
        self.handle = None
        self.nvml_uuid = None
        for index in range(self.nvml.nvmlDeviceGetCount()):
            candidate = self.nvml.nvmlDeviceGetHandleByIndex(index)
            candidate_uuid = _normalise_uuid(self.nvml.nvmlDeviceGetUUID(candidate))
            if candidate_uuid == self.torch_uuid:
                self.handle = candidate
                self.nvml_uuid = candidate_uuid
                break
        if self.handle is None or self.nvml_uuid != self.torch_uuid:
            raise RuntimeError(
                f"CUDA/NVML UUID mismatch: cuda={self.torch_uuid}, nvml={self.nvml_uuid}"
            )
        try:
            self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.has_total_energy_counter = True
        except (self.nvml.NVMLError_NotSupported, AttributeError):
            self.has_total_energy_counter = False

    @property
    def gpu_uuid(self) -> str:
        return f"GPU-{self.nvml_uuid}"

    def measure(self, workload: Callable[[], None]) -> tuple[float, str]:
        torch.cuda.synchronize(self.device)
        if self.has_total_energy_counter:
            start_mj = int(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
            workload()
            torch.cuda.synchronize(self.device)
            end_mj = int(self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
            if end_mj < start_mj:
                raise RuntimeError("NVML total-energy counter moved backwards")
            return (end_mj - start_mj) / 1000.0, "nvml_total_energy_counter"

        samples: list[tuple[int, float]] = []
        stop_event = threading.Event()

        def read_sample() -> tuple[int, float]:
            timestamp_ns = time.monotonic_ns()
            watts = float(self.nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
            return timestamp_ns, watts

        def worker() -> None:
            while not stop_event.wait(self.sampling_interval_s):
                samples.append(read_sample())

        samples.append(read_sample())  # explicit t0 boundary point
        thread = threading.Thread(target=worker, name="nvml-power-sampler", daemon=True)
        thread.start()
        try:
            workload()
            torch.cuda.synchronize(self.device)
        finally:
            stop_event.set()
            thread.join(timeout=max(1.0, 5 * self.sampling_interval_s))
            if thread.is_alive():
                raise RuntimeError("NVML power-sampling thread did not stop")
            samples.append(read_sample())  # explicit t1 boundary point
        if len(samples) < 2:
            raise RuntimeError("insufficient NVML power samples")
        energy_j = 0.0
        for (left_ns, left_w), (right_ns, right_w) in zip(samples, samples[1:]):
            if right_ns <= left_ns:
                raise RuntimeError("non-monotonic NVML sampling timestamp")
            energy_j += 0.5 * (left_w + right_w) * (right_ns - left_ns) / 1e9
        return energy_j, "nvml_power_trapezoid_monotonic"


def _cuda_elapsed_ms(device: torch.device, workload: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    workload()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _choose_repeats(
    expert: Any,
    activation: torch.Tensor,
    minimum_seconds: float,
    maximum_repeats: int,
) -> int:
    repeats = 1
    while True:
        elapsed_ms = _cuda_elapsed_ms(
            activation.device,
            lambda: _repeat_expert(expert, activation, ACTION_BF16, repeats),
        )
        if elapsed_ms / 1000.0 >= minimum_seconds:
            return repeats
        if repeats >= maximum_repeats:
            raise RuntimeError(
                "cannot reach minimum measurement duration before maximum_inner_repeats"
            )
        repeats = min(maximum_repeats, repeats * 2)


def _repeat_expert(expert: Any, activation: torch.Tensor, action: str, repeats: int) -> None:
    output = None
    for _ in range(repeats):
        output = expert(activation, action)
    # Retain the last output until all kernels have been enqueued.
    if output is None:
        raise AssertionError("measurement repeats must be positive")


def _measure_arm(
    expert: Any,
    activation: torch.Tensor,
    action: str,
    repeats: int,
    energy_meter: NvmlBoardEnergyMeter,
) -> ArmMeasurement:
    before_counters = expert.counters.snapshot()
    workload = lambda: _repeat_expert(expert, activation, action, repeats)
    elapsed_ms = _cuda_elapsed_ms(activation.device, workload)
    energy_j, energy_method = energy_meter.measure(workload)
    after_counters = expert.counters.snapshot()
    counter_delta = {
        name: after_counters[name] - before_counters[name] for name in before_counters
    }
    if elapsed_ms <= 0 or energy_j <= 0:
        raise RuntimeError("latency and board energy measurements must both be positive")
    if counter_delta["weight_casts"] != 0:
        raise AssertionError("weight cast occurred inside an arm measurement")
    expected_calls = 2 * repeats  # one timed workload plus one energy workload
    if action == ACTION_FP8:
        if counter_delta["fp8_expert_calls"] != expected_calls:
            raise AssertionError("FP8 expert-call counter does not match measured work")
        if counter_delta["activation_casts"] != 3 * expected_calls:
            raise AssertionError("FP8 activation-cast counter does not cover three projections")
        if counter_delta["scaled_mm_calls"] != 3 * expected_calls:
            raise AssertionError("FP8 scaled-mm counter does not cover three projections")
        if counter_delta["bf16_expert_calls"] != 0:
            raise AssertionError("FP8 arm executed BF16 expert calls")
    elif action == ACTION_BF16:
        if counter_delta["bf16_expert_calls"] != expected_calls:
            raise AssertionError("BF16 expert-call counter does not match measured work")
        if any(
            counter_delta[name] != 0
            for name in ("fp8_expert_calls", "activation_casts", "scaled_mm_calls")
        ):
            raise AssertionError("BF16 arm executed FP8 work")
    return ArmMeasurement(
        energy_j_per_call=energy_j / repeats,
        latency_ms_per_call=elapsed_ms / repeats,
        repeats=repeats,
        energy_method=energy_method,
        counter_delta=counter_delta,
    )


def _measure_observation(
    observation: ExpertObservation,
    expert: Any,
    device: torch.device,
    energy_meter: NvmlBoardEnergyMeter,
    surface_config: Mapping[str, Any],
    observation_index: int,
) -> dict[str, Any]:
    activation = observation.activation_cpu.to(device=device, dtype=torch.bfloat16)
    if tuple(activation.shape) != (observation.row_count, expert.input_features):
        raise RuntimeError("capture/model expert shape mismatch")
    warmups = int(surface_config["warmup_calls"])
    for _ in range(warmups):
        expert(activation, ACTION_BF16)
        expert(activation, ACTION_FP8)
    torch.cuda.synchronize(device)
    repeats = _choose_repeats(
        expert,
        activation,
        float(surface_config["minimum_measurement_seconds_per_arm"]),
        int(surface_config["maximum_inner_repeats"]),
    )
    trials = int(surface_config["paired_trials_per_event"])
    if trials <= 0:
        raise RuntimeError("paired_trials_per_event must be positive")
    if surface_config.get("alternate_arm_order") is not True:
        raise RuntimeError("frozen surface requires alternating paired arm order")
    pair_rows: list[dict[str, Any]] = []
    for trial in range(trials):
        fp8_first = bool((trial + observation_index) % 2)
        order = (ACTION_FP8, ACTION_BF16) if fp8_first else (ACTION_BF16, ACTION_FP8)
        measured: dict[str, ArmMeasurement] = {}
        for action in order:
            measured[action] = _measure_arm(
                expert, activation, action, repeats, energy_meter
            )
        if measured[ACTION_BF16].energy_method != measured[ACTION_FP8].energy_method:
            raise RuntimeError("paired arms used different board-energy methods")
        pair_rows.append(
            {
                "trial": trial,
                "arm_order": list(order),
                "repeats": repeats,
                "energy_method": measured[ACTION_BF16].energy_method,
                "bf16_counter_delta": measured[ACTION_BF16].counter_delta,
                "fp8_counter_delta": measured[ACTION_FP8].counter_delta,
                "bf16_energy_j_per_call": measured[ACTION_BF16].energy_j_per_call,
                "fp8_energy_j_per_call": measured[ACTION_FP8].energy_j_per_call,
                "delta_energy_j": (
                    measured[ACTION_BF16].energy_j_per_call
                    - measured[ACTION_FP8].energy_j_per_call
                ),
                "bf16_latency_ms_per_call": measured[ACTION_BF16].latency_ms_per_call,
                "fp8_latency_ms_per_call": measured[ACTION_FP8].latency_ms_per_call,
                "delta_latency_ms": (
                    measured[ACTION_FP8].latency_ms_per_call
                    - measured[ACTION_BF16].latency_ms_per_call
                ),
            }
        )
    return {
        "event_id": observation.event_id,
        "iteration_id": observation.iteration_id,
        "batch_id": observation.batch_id,
        "layer_id": observation.layer_id,
        "expert_id": observation.expert_id,
        "row_count": observation.row_count,
        "row_bin": observation.row_bin,
        "trials": pair_rows,
        # Trials characterize one captured expert call; the independent
        # bootstrap unit below is this event, not each repeated trial.
        "event_mean_bf16_energy_j": statistics.fmean(
            row["bf16_energy_j_per_call"] for row in pair_rows
        ),
        "event_mean_delta_energy_j": statistics.fmean(
            row["delta_energy_j"] for row in pair_rows
        ),
        "event_mean_delta_latency_ms": statistics.fmean(
            row["delta_latency_ms"] for row in pair_rows
        ),
    }


def _select_observations(
    observations: Sequence[ExpertObservation], surface_config: Mapping[str, Any]
) -> tuple[list[ExpertObservation], dict[str, int], dict[str, int]]:
    method = surface_config.get("event_subsampling")
    if method != "smallest_sha256_of_seed_and_event_id":
        raise RuntimeError(f"unsupported or mutable event-subsampling rule: {method!r}")
    maximum = int(surface_config["max_profiled_events_per_bin"])
    if maximum <= 0:
        raise RuntimeError("max_profiled_events_per_bin must be positive")
    seed = int(surface_config["bootstrap_seed"])
    grouped: dict[str, list[ExpertObservation]] = {label: [] for label in ROW_BIN_LABELS}
    for observation in observations:
        grouped[observation.row_bin].append(observation)
    selected: list[ExpertObservation] = []
    population_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for label in ROW_BIN_LABELS:
        population_counts[label] = len(grouped[label])
        ranked = sorted(
            grouped[label],
            key=lambda observation: hashlib.sha256(
                f"{seed}:{observation.event_id}".encode("utf-8")
            ).digest(),
        )
        chosen = ranked[:maximum]
        selected.extend(chosen)
        selected_counts[label] = len(chosen)
    selected.sort(key=lambda item: (item.iteration_id, item.layer_id, item.expert_id))
    return selected, population_counts, selected_counts


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def _bootstrap_mean_ci(
    values: Sequence[float], *, resamples: int, seed: int
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one independent event")
    if resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be finite")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    count = len(values)
    bootstrap_means = [
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    bootstrap_means.sort()
    return mean, _quantile(bootstrap_means, 0.025), _quantile(bootstrap_means, 0.975)


def _surface_cells(
    event_results: Sequence[Mapping[str, Any]],
    surface_config: Mapping[str, Any],
    population_counts: Mapping[str, int],
) -> dict[str, SurfaceCell]:
    grouped: dict[str, list[Mapping[str, Any]]] = {label: [] for label in ROW_BIN_LABELS}
    for result in event_results:
        grouped[str(result["row_bin"])].append(result)
    cells: dict[str, SurfaceCell] = {}
    resamples = int(surface_config["bootstrap_resamples"])
    base_seed = int(surface_config["bootstrap_seed"])
    for bin_index, label in enumerate(ROW_BIN_LABELS):
        rows = grouped[label]
        if not rows:
            continue
        # Multiple layers/experts in one scheduler iteration share load and
        # power state.  Treating them as independent would be pseudo-replication,
        # so the CI bootstrap unit is the iteration cluster.
        clusters: dict[int, list[Mapping[str, Any]]] = {}
        for row in rows:
            clusters.setdefault(int(row["iteration_id"]), []).append(row)
        energy = [
            statistics.fmean(
                float(row["event_mean_delta_energy_j"]) for row in cluster_rows
            )
            for cluster_rows in clusters.values()
        ]
        latency = [
            statistics.fmean(
                float(row["event_mean_delta_latency_ms"]) for row in cluster_rows
            )
            for cluster_rows in clusters.values()
        ]
        energy_mean, energy_lcb, energy_ucb = _bootstrap_mean_ci(
            energy, resamples=resamples, seed=base_seed + 2 * bin_index
        )
        latency_mean, latency_lcb, latency_ucb = _bootstrap_mean_ci(
            latency, resamples=resamples, seed=base_seed + 2 * bin_index + 1
        )
        cells[label] = SurfaceCell(
            row_bin=label,
            sample_count=len(clusters),
            delta_energy_mean_j=energy_mean,
            delta_energy_lcb95_j=energy_lcb,
            delta_energy_ucb95_j=energy_ucb,
            delta_latency_mean_ms=latency_mean,
            delta_latency_lcb95_ms=latency_lcb,
            delta_latency_ucb95_ms=latency_ucb,
            # Hash-based event subsampling is outcome-blind.  Estimate the full
            # captured-workload BF16 energy mass by sampled mean x population.
            bf16_energy_mass_j=(
                statistics.fmean(float(row["event_mean_bf16_energy_j"]) for row in rows)
                * int(population_counts[label])
            ),
        )
    return cells


def _load_model(model_spec: Mapping[str, Any], device: torch.device, local_only: bool) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the pinned MoE model") from exc
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        trust_remote_code=bool(model_spec.get("trust_remote_code", False)),
        local_files_only=local_only,
    )
    model.eval()
    model.requires_grad_(False)
    return model


def _expert_shape(bank: DualResidentExpertBank) -> tuple[int, int, int]:
    shapes = {
        (expert.input_features, expert.intermediate_features, expert.output_features)
        for expert in bank.experts.values()
    }
    if len(shapes) != 1:
        raise RuntimeError(f"experts do not share one MLP shape: {sorted(shapes)}")
    return next(iter(shapes))


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.model_key is None or args.capture is None or args.output_dir is None:
        raise RuntimeError("surface run requires --model-key, --capture, and --output-dir")
    config_path = args.config.resolve()
    config = _load_config(config_path)
    code_hash: Optional[str] = None
    signoff_summary: Optional[Mapping[str, Any]] = None
    if args.formal:
        code_hash, signoff = _apply_formal_gate(config, config_path, args.signoff)
        signoff_summary = {
            "status": signoff["status"],
            "phase": signoff["phase"],
            "code_config_sha256": signoff["code_config_sha256"],
        }

    device = require_cuda_fp8(args.device, probe_kernel=True)
    model_spec = config["models"][args.model_key]
    model_revision = f"{model_spec['model_id']}@{model_spec['revision']}"
    capture_metadata, observations = _load_capture(
        args.capture, model_spec, config["capture_contract"]
    )
    selected_observations, population_counts, selected_counts = _select_observations(
        observations, config["surface"]
    )
    quality_passed, quality_sha256, quality_reason = _load_quality_gate(
        args.quality_artifact,
        model_revision,
        config,
        capture_metadata,
        formal=args.formal,
    )

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = _load_model(model_spec, device, args.local_files_only)
    counters = RuntimeCounters()
    bank = DualResidentExpertBank(
        model,
        expected_target_linears=int(model_spec["expected_target_linears"]),
        expected_num_layers=int(model_spec["num_layers"]),
        expected_num_experts=int(model_spec["num_experts"]),
        counters=counters,
    )
    bank.eval()
    torch.cuda.synchronize(device)
    post_residency_allocated_bytes = int(torch.cuda.memory_allocated(device))
    residency = bank.residency_bytes()
    shape = _expert_shape(bank)
    energy_meter = NvmlBoardEnergyMeter(
        device, float(config["surface"]["power_sampling_interval_seconds"])
    )
    key = SurfaceKey(
        model_revision=model_revision,
        gpu_uuid=energy_meter.gpu_uuid,
        fp8_recipe=config["fp8_recipe"],
        expert_mlp_shape=shape,
    )

    before_measurement = counters.snapshot()
    event_results: list[Mapping[str, Any]] = []
    with torch.inference_mode():
        for index, observation in enumerate(selected_observations):
            event_results.append(
                _measure_observation(
                    observation,
                    bank.get(observation.layer_id, observation.expert_id),
                    device,
                    energy_meter,
                    config["surface"],
                    index,
                )
            )
    torch.cuda.synchronize(device)
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    after_measurement = counters.snapshot()
    if after_measurement["weight_casts"] != before_measurement["weight_casts"]:
        raise AssertionError("weight casting occurred inside the measurement phase")
    if after_measurement["activation_casts"] <= before_measurement["activation_casts"]:
        raise AssertionError("FP8 measurement executed no activation casts")
    if after_measurement["scaled_mm_calls"] <= before_measurement["scaled_mm_calls"]:
        raise AssertionError("FP8 measurement executed no scaled_mm calls")

    cells = _surface_cells(event_results, config["surface"], population_counts)
    lut = RouteRowLUT(
        key=key,
        cells=cells,
        quality_gate_passed=quality_passed,
        min_samples_per_bin=int(
            config["surface"]["min_independent_iteration_clusters_per_bin"]
        ),
        quality_artifact_sha256=quality_sha256,
    )
    existence = lut.existence_summary(
        float(
            config["existence_gate"][
                "minimum_bf16_expert_energy_mass_fraction_per_region"
            ]
        )
    )
    if not quality_passed:
        diagnostic_status = "LOCAL_DIAGNOSTIC_QUALITY_NOT_READY"
    elif existence["passed"]:
        diagnostic_status = "LOCAL_CROSSOVER_DIAGNOSTIC_OBSERVED"
    else:
        diagnostic_status = "LOCAL_CROSSOVER_DIAGNOSTIC_NOT_OBSERVED"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lut_path = args.output_dir / "route_row_lut.json"
    raw_path = args.output_dir / "route_row_raw_pairs.json"
    summary_path = args.output_dir / "route_row_surface.json"
    write_json_atomic(lut_path, lut.to_dict())
    write_json_atomic(
        raw_path,
        {
            "schema_version": 1,
            "independent_unit": "captured_expert_event",
            "capture_sha256": sha256_file(args.capture),
            "events": event_results,
        },
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "phase": "Phase3_SURFACE_CALIBRATION",
        "phase3_status": "BLOCKED_PENDING_INTEGRATED_CONTINUOUS_DYNAMIC_EXPERT_HOT_PATH",
        "scientific_eligibility": "INELIGIBLE_CALIBRATION_PROXY",
        "artifact_scope": "CAPABILITY_AND_CALIBRATION_PROXY_ONLY",
        "run_mode": "REVIEWED_CALIBRATION" if args.formal else "PREFLIGHT",
        "local_diagnostic_status": diagnostic_status,
        "blocked_reason": (
            "no audited continuous-serving integration executes this LUT as a "
            "dynamic per-expert BF16/FP8 hot-path policy"
        ),
        "model_key": args.model_key,
        "model_revision": model_revision,
        "gpu_uuid": energy_meter.gpu_uuid,
        "fp8_recipe": config["fp8_recipe"],
        "expert_mlp_shape": list(shape),
        "capture": {
            "path": str(args.capture),
            "sha256": sha256_file(args.capture),
            "engine_name": capture_metadata["engine_name"],
            "engine_commit": capture_metadata["engine_commit"],
            "capture_producer_sha256": capture_metadata["capture_producer_sha256"],
            "data_manifest_sha256": capture_metadata["data_manifest_sha256"],
            "arrival_config_sha256": capture_metadata["arrival_config_sha256"],
            "captured_expert_events": len(observations),
            "profiled_expert_events": len(selected_observations),
            "population_events_by_bin": population_counts,
            "profiled_events_by_bin": selected_counts,
            "event_subsampling": config["surface"]["event_subsampling"],
        },
        "quality_gate": {
            "passed": quality_passed,
            "reason": quality_reason,
            "artifact_sha256": quality_sha256,
        },
        "existence_gate": existence,
        "residency_accounting": {
            **residency,
            "post_residency_cuda_allocated_bytes": post_residency_allocated_bytes,
            "peak_cuda_allocated_bytes": peak_allocated_bytes,
        },
        "runtime_counters_before_measurement": before_measurement,
        "runtime_counters_after_measurement": after_measurement,
        "surface_cells": {label: asdict(cell) for label, cell in cells.items()},
        "surface_ci_unit": "scheduler_iteration_cluster",
        "bf16_energy_mass_estimator": "sampled_bin_mean_times_captured_bin_population",
        "formal_gate": {
            "code_config_sha256": code_hash,
            "signoff": signoff_summary,
        },
        "outputs": {
            "lut": str(lut_path),
            "raw_pairs": str(raw_path),
            "summary": str(summary_path),
        },
    }
    write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.print_code_config_hash:
        config_path = args.config.resolve()
        config = _load_config(config_path)
        print(_current_code_config_hash(config, config_path))
        return
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
