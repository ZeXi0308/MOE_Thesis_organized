"""Fail-closed route-row FP8 policy and dual-resident expert wrappers.

This module implements the Phase-2 frozen *mechanism*.  It does not implement a
serving engine and does not produce a scientific result by itself.  In
particular, a missing, under-powered, mismatched, or quality-unsafe LUT entry
always falls back to BF16.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import torch
from torch import nn


ROW_BIN_LABELS: tuple[str, ...] = (
    "1",
    "2",
    "3-4",
    "5-8",
    "9-16",
    "17-32",
    "33-64",
    "65-128",
    ">=129",
)

FP8_RECIPE_E4M3FN = (
    "E4M3FN_weight_per_tensor_absmax_once_"
    "activation_per_projection_absmax_scaled_mm_bf16"
)

ACTION_FP8 = "FP8"
ACTION_BF16 = "BF16"
ACTION_SKIP = "SKIP"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def row_bin_for_count(row_count: int) -> Optional[str]:
    """Return the frozen row-bin label; zero means no expert invocation."""

    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise TypeError("row_count must be an integer")
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if row_count == 0:
        return None
    if row_count == 1:
        return "1"
    if row_count == 2:
        return "2"
    if row_count <= 4:
        return "3-4"
    if row_count <= 8:
        return "5-8"
    if row_count <= 16:
        return "9-16"
    if row_count <= 32:
        return "17-32"
    if row_count <= 64:
        return "33-64"
    if row_count <= 128:
        return "65-128"
    return ">=129"


def route_row_counts(
    selected_experts: torch.Tensor,
    num_experts: int,
    *,
    expected_active_decode_tokens: Optional[int] = None,
    expected_top_k: Optional[int] = None,
) -> torch.Tensor:
    """Count current-layer router selections and assert the accounting identity.

    ``selected_experts`` must be the router/top-k result for the current layer,
    before any expert executes.  This API deliberately has no argument through
    which future tokens, future arrivals, or later-layer routes can enter.
    """

    if not isinstance(selected_experts, torch.Tensor):
        raise TypeError("selected_experts must be a torch.Tensor")
    if selected_experts.ndim != 2:
        raise ValueError("selected_experts must have shape [active_tokens, top_k]")
    if selected_experts.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("selected_experts must contain integer expert ids")
    if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")

    active_decode_tokens, top_k = selected_experts.shape
    if expected_active_decode_tokens is not None and active_decode_tokens != expected_active_decode_tokens:
        raise ValueError(
            f"active token mismatch: tensor={active_decode_tokens}, "
            f"expected={expected_active_decode_tokens}"
        )
    if expected_top_k is not None and top_k != expected_top_k:
        raise ValueError(f"top-k mismatch: tensor={top_k}, expected={expected_top_k}")

    flat = selected_experts.reshape(-1).to(dtype=torch.int64)
    if flat.numel():
        minimum = int(flat.min().item())
        maximum = int(flat.max().item())
        if minimum < 0 or maximum >= num_experts:
            raise ValueError(
                f"expert id out of range [0, {num_experts}): min={minimum}, max={maximum}"
            )
        if top_k > 1:
            ordered, _ = torch.sort(selected_experts.to(dtype=torch.int64), dim=1)
            if bool(torch.any(ordered[:, 1:] == ordered[:, :-1]).item()):
                raise ValueError("one token cannot select the same expert in two top-k slots")
    counts = torch.bincount(flat, minlength=num_experts)
    expected_rows = active_decode_tokens * top_k
    if counts.numel() != num_experts or int(counts.sum().item()) != expected_rows:
        raise AssertionError(
            "route-row accounting failed: "
            f"sum={int(counts.sum().item())}, active_tokens*top_k={expected_rows}"
        )
    return counts


def routing_blind_expected_rows(
    active_decode_tokens: int, top_k: int, num_experts: int
) -> float:
    """The frozen B3 signal; it contains no realized routing information."""

    for name, value in (
        ("active_decode_tokens", active_decode_tokens),
        ("top_k", top_k),
        ("num_experts", num_experts),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if top_k > num_experts:
        raise ValueError("top_k cannot exceed num_experts")
    return active_decode_tokens * top_k / num_experts


def energy_per_completed_token(
    total_board_energy_j: float,
    completed_output_tokens: int,
    *,
    duration_s: Optional[float] = None,
    idle_power_w: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """Apply the frozen completed-token denominator and optional idle sensitivity."""

    if not math.isfinite(total_board_energy_j) or total_board_energy_j < 0:
        raise ValueError("total_board_energy_j must be finite and non-negative")
    if (
        isinstance(completed_output_tokens, bool)
        or not isinstance(completed_output_tokens, int)
        or completed_output_tokens <= 0
    ):
        raise ValueError("completed_output_tokens must be a positive integer")
    dynamic_energy_j: Optional[float] = None
    dynamic_j_per_token: Optional[float] = None
    if (duration_s is None) != (idle_power_w is None):
        raise ValueError("duration_s and idle_power_w must be supplied together")
    if duration_s is not None and idle_power_w is not None:
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s must be finite and non-negative")
        if not math.isfinite(idle_power_w) or idle_power_w < 0:
            raise ValueError("idle_power_w must be finite and non-negative")
        dynamic_energy_j = max(total_board_energy_j - duration_s * idle_power_w, 0.0)
        dynamic_j_per_token = dynamic_energy_j / completed_output_tokens
    return {
        "board_j_per_completed_output_token": total_board_energy_j
        / completed_output_tokens,
        "dynamic_energy_j": dynamic_energy_j,
        "dynamic_j_per_completed_output_token": dynamic_j_per_token,
    }


@dataclass(frozen=True)
class SurfaceKey:
    model_revision: str
    gpu_uuid: str
    fp8_recipe: str
    expert_mlp_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not self.model_revision or "@" not in self.model_revision:
            raise ValueError("model_revision must be a pinned '<model>@<revision>' string")
        if not self.gpu_uuid:
            raise ValueError("gpu_uuid is required")
        if not self.fp8_recipe:
            raise ValueError("fp8_recipe is required")
        if len(self.expert_mlp_shape) != 3 or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in self.expert_mlp_shape
        ):
            raise ValueError("expert_mlp_shape must be three positive integers")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SurfaceKey":
        return cls(
            model_revision=str(value["model_revision"]),
            gpu_uuid=str(value["gpu_uuid"]),
            fp8_recipe=str(value["fp8_recipe"]),
            expert_mlp_shape=tuple(int(v) for v in value["expert_mlp_shape"]),
        )


@dataclass(frozen=True)
class SurfaceCell:
    row_bin: str
    sample_count: int
    delta_energy_mean_j: float
    delta_energy_lcb95_j: float
    delta_energy_ucb95_j: float
    delta_latency_mean_ms: float
    delta_latency_lcb95_ms: float
    delta_latency_ucb95_ms: float
    bf16_energy_mass_j: float

    def __post_init__(self) -> None:
        if self.row_bin not in ROW_BIN_LABELS:
            raise ValueError(f"unknown frozen row bin: {self.row_bin}")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise ValueError("sample_count must be non-negative")
        numeric = (
            self.delta_energy_mean_j,
            self.delta_energy_lcb95_j,
            self.delta_energy_ucb95_j,
            self.delta_latency_mean_ms,
            self.delta_latency_lcb95_ms,
            self.delta_latency_ucb95_ms,
            self.bf16_energy_mass_j,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("surface values must be finite")
        if self.bf16_energy_mass_j < 0:
            raise ValueError("bf16_energy_mass_j must be non-negative")
        if self.delta_energy_lcb95_j > self.delta_energy_ucb95_j:
            raise ValueError("energy CI lower bound exceeds upper bound")
        if self.delta_latency_lcb95_ms > self.delta_latency_ucb95_ms:
            raise ValueError("latency CI lower bound exceeds upper bound")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SurfaceCell":
        return cls(**{field_name: value[field_name] for field_name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str
    row_bin: Optional[str]


@dataclass
class RouteRowLUT:
    key: SurfaceKey
    cells: Mapping[str, SurfaceCell]
    quality_gate_passed: bool
    min_samples_per_bin: int
    quality_artifact_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.min_samples_per_bin, bool) or self.min_samples_per_bin <= 0:
            raise ValueError("min_samples_per_bin must be positive")
        if not isinstance(self.quality_gate_passed, bool):
            raise TypeError("quality_gate_passed must be bool")
        if self.quality_artifact_sha256 is not None and (
            not isinstance(self.quality_artifact_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.quality_artifact_sha256) is None
        ):
            raise ValueError(
                "quality_artifact_sha256 must be null or a lowercase SHA-256"
            )
        if self.quality_gate_passed and self.quality_artifact_sha256 is None:
            raise ValueError(
                "quality_gate_passed=true requires a valid quality artifact SHA-256"
            )
        copied: dict[str, SurfaceCell] = {}
        for label, cell in self.cells.items():
            if label != cell.row_bin:
                raise ValueError(f"cell key {label!r} does not match row_bin {cell.row_bin!r}")
            if label in copied:
                raise ValueError(f"duplicate row bin: {label}")
            copied[label] = cell
        unknown = set(copied) - set(ROW_BIN_LABELS)
        if unknown:
            raise ValueError(f"unknown row bins: {sorted(unknown)}")
        self.cells = copied

    def decide(self, row_count: int, query_key: SurfaceKey) -> PolicyDecision:
        label = row_bin_for_count(row_count)
        if label is None:
            return PolicyDecision(ACTION_SKIP, "empty_expert", None)
        if query_key != self.key:
            return PolicyDecision(ACTION_BF16, "surface_key_mismatch", label)
        if not self.quality_gate_passed:
            return PolicyDecision(ACTION_BF16, "quality_gate_failed_or_missing", label)
        cell = self.cells.get(label)
        if cell is None:
            return PolicyDecision(ACTION_BF16, "missing_bin", label)
        if cell.sample_count < self.min_samples_per_bin:
            return PolicyDecision(ACTION_BF16, "underpowered_bin", label)
        if cell.delta_energy_lcb95_j <= 0:
            return PolicyDecision(ACTION_BF16, "energy_lcb_not_positive", label)
        if cell.delta_latency_ucb95_ms > 0:
            return PolicyDecision(ACTION_BF16, "latency_ucb_positive", label)
        return PolicyDecision(ACTION_FP8, "all_frozen_gates_passed", label)

    def existence_summary(self, min_energy_mass_fraction: float = 0.10) -> dict[str, Any]:
        if not 0 < min_energy_mass_fraction < 1:
            raise ValueError("min_energy_mass_fraction must be in (0, 1)")
        total_mass = sum(cell.bf16_energy_mass_j for cell in self.cells.values())
        safe_bins: list[str] = []
        fallback_bins: list[str] = []
        safe_mass = 0.0
        fallback_mass = 0.0
        for label in ROW_BIN_LABELS:
            cell = self.cells.get(label)
            if cell is None:
                continue
            decision = self.decide(_representative_row_count(label), self.key)
            if decision.action == ACTION_FP8:
                safe_bins.append(label)
                safe_mass += cell.bf16_energy_mass_j
            else:
                fallback_bins.append(label)
                fallback_mass += cell.bf16_energy_mass_j
        safe_fraction = safe_mass / total_mass if total_mass > 0 else 0.0
        fallback_fraction = fallback_mass / total_mass if total_mass > 0 else 0.0
        passed = (
            self.quality_gate_passed
            and bool(safe_bins)
            and bool(fallback_bins)
            and safe_fraction >= min_energy_mass_fraction
            and fallback_fraction >= min_energy_mass_fraction
        )
        return {
            "passed": passed,
            "quality_gate_passed": self.quality_gate_passed,
            "safe_bins": safe_bins,
            "fallback_bins": fallback_bins,
            "total_bf16_energy_mass_j": total_mass,
            "safe_bf16_energy_mass_fraction": safe_fraction,
            "fallback_bf16_energy_mass_fraction": fallback_fraction,
            "minimum_required_fraction": min_energy_mass_fraction,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "key": asdict(self.key),
            "quality_gate_passed": self.quality_gate_passed,
            "quality_artifact_sha256": self.quality_artifact_sha256,
            "min_samples_per_bin": self.min_samples_per_bin,
            "fixed_row_bins": list(ROW_BIN_LABELS),
            "cells": {label: asdict(cell) for label, cell in self.cells.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RouteRowLUT":
        if not isinstance(value, Mapping):
            raise TypeError("LUT document must be a mapping")
        if tuple(value.get("fixed_row_bins", ())) != ROW_BIN_LABELS:
            raise ValueError("LUT row bins differ from the Phase-2 frozen bins")
        if "quality_gate_passed" not in value:
            raise ValueError("LUT is missing required quality_gate_passed")
        quality_gate_passed = value["quality_gate_passed"]
        if type(quality_gate_passed) is not bool:
            raise TypeError("quality_gate_passed must be a JSON boolean")
        quality_artifact_sha256 = value.get("quality_artifact_sha256")
        if quality_artifact_sha256 is not None and (
            not isinstance(quality_artifact_sha256, str)
            or _SHA256_PATTERN.fullmatch(quality_artifact_sha256) is None
        ):
            raise ValueError(
                "quality_artifact_sha256 must be null or a lowercase SHA-256"
            )
        if quality_gate_passed and quality_artifact_sha256 is None:
            raise ValueError(
                "quality_gate_passed=true requires quality_artifact_sha256"
            )
        cells = {
            str(label): SurfaceCell.from_dict(cell)
            for label, cell in value["cells"].items()
        }
        return cls(
            key=SurfaceKey.from_dict(value["key"]),
            cells=cells,
            quality_gate_passed=quality_gate_passed,
            min_samples_per_bin=int(value["min_samples_per_bin"]),
            quality_artifact_sha256=quality_artifact_sha256,
        )


def _representative_row_count(label: str) -> int:
    representatives = {
        "1": 1,
        "2": 2,
        "3-4": 3,
        "5-8": 5,
        "9-16": 9,
        "17-32": 17,
        "33-64": 33,
        "65-128": 65,
        ">=129": 129,
    }
    return representatives[label]


@dataclass
class RuntimeCounters:
    weight_casts: int = 0
    activation_casts: int = 0
    scaled_mm_calls: int = 0
    fp8_expert_calls: int = 0
    bf16_expert_calls: int = 0
    empty_expert_calls: int = 0

    def snapshot(self) -> dict[str, int]:
        return asdict(self)


def require_cuda_fp8(device: torch.device | str, *, probe_kernel: bool = False) -> torch.device:
    """Hard-fail when the requested device cannot execute the frozen FP8 path."""

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError("CUDA is mandatory; CPU/proxy execution is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; FP8 surface is BLOCKED")
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("this PyTorch build lacks E4M3FN/_scaled_mm support")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(resolved)
    if capability < (8, 9):
        raise RuntimeError(
            f"GPU compute capability {capability} lacks the required native FP8 path"
        )
    if probe_kernel:
        try:
            lhs = torch.zeros((16, 16), device=resolved, dtype=torch.float8_e4m3fn)
            rhs_storage = torch.zeros((16, 16), device=resolved, dtype=torch.float8_e4m3fn)
            rhs = rhs_storage.t()
            scale = torch.ones((), device=resolved, dtype=torch.float32)
            torch._scaled_mm(
                lhs,
                rhs,
                scale_a=scale,
                scale_b=scale,
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )
            torch.cuda.synchronize(resolved)
        except Exception as exc:  # pragma: no cover - requires a CUDA FP8 host
            raise RuntimeError("native torch._scaled_mm FP8 probe failed") from exc
    return resolved


def _fp8_quantize_per_tensor(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = 448.0
    value_f32 = value.detach().float()
    scale = value_f32.abs().amax().clamp_min(1e-12) / fp8_max
    quantized = (value_f32 / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return quantized, scale


class DualResidentFP8Linear(nn.Module):
    """One BF16 linear plus a statically-created E4M3FN weight resident copy."""

    def __init__(self, original: nn.Linear, counters: RuntimeCounters):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError("DualResidentFP8Linear requires nn.Linear")
        if original.weight.device.type != "cuda":
            raise RuntimeError("dual-resident FP8 weights must be created on CUDA")
        if original.weight.dtype != torch.bfloat16:
            raise RuntimeError(
                f"frozen baseline requires BF16 weights, got {original.weight.dtype}"
            )
        require_cuda_fp8(original.weight.device)
        self.original = original
        self.counters = counters
        weight_fp8, weight_scale = _fp8_quantize_per_tensor(original.weight)
        # _scaled_mm consumes [M,K] @ [K,N].  Keep a transposed column-major
        # view; making it contiguous here changes the expected operand layout.
        self.register_buffer("weight_fp8_t", weight_fp8.t(), persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        self.counters.weight_casts += 1

    @property
    def out_features(self) -> int:
        return self.original.out_features

    def forward_fp8(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("expert linear input must be rank-2 [rows, hidden]")
        if hidden_states.shape[0] == 0:
            return hidden_states.new_empty((0, self.out_features), dtype=torch.bfloat16)
        if hidden_states.device != self.weight_fp8_t.device:
            raise RuntimeError("activation and dual-resident weight are on different devices")
        activation_fp8, activation_scale = _fp8_quantize_per_tensor(hidden_states)
        self.counters.activation_casts += 1
        result = torch._scaled_mm(
            activation_fp8,
            self.weight_fp8_t,
            scale_a=activation_scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        self.counters.scaled_mm_calls += 1
        if self.original.bias is not None:
            result = result + self.original.bias
        return result


def _projection_names(expert: nn.Module) -> tuple[str, str, str]:
    if all(isinstance(getattr(expert, name, None), nn.Linear) for name in (
        "gate_proj",
        "up_proj",
        "down_proj",
    )):
        return "gate_proj", "up_proj", "down_proj"
    if all(isinstance(getattr(expert, name, None), nn.Linear) for name in ("w1", "w3", "w2")):
        return "w1", "w3", "w2"
    raise TypeError("unsupported expert: expected gate/up/down or w1/w3/w2 linears")


class DualResidentExpertMLP(nn.Module):
    """Full three-projection expert path with one precision per invocation."""

    def __init__(self, original_expert: nn.Module, counters: RuntimeCounters):
        super().__init__()
        gate_name, up_name, down_name = _projection_names(original_expert)
        if not hasattr(original_expert, "act_fn"):
            raise TypeError("expert has no act_fn; refusing to guess its MLP semantics")
        self.original_expert = original_expert
        self.gate = DualResidentFP8Linear(getattr(original_expert, gate_name), counters)
        self.up = DualResidentFP8Linear(getattr(original_expert, up_name), counters)
        self.down = DualResidentFP8Linear(getattr(original_expert, down_name), counters)
        self.counters = counters

    @property
    def input_features(self) -> int:
        return self.gate.original.in_features

    @property
    def output_features(self) -> int:
        return self.down.original.out_features

    @property
    def intermediate_features(self) -> int:
        return self.gate.original.out_features

    def forward(self, hidden_states: torch.Tensor, action: str) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.input_features:
            raise ValueError(
                f"expected [rows,{self.input_features}] expert input, got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] == 0:
            self.counters.empty_expert_calls += 1
            return hidden_states.new_empty((0, self.output_features), dtype=torch.bfloat16)
        if action == ACTION_BF16:
            self.counters.bf16_expert_calls += 1
            output = self.original_expert(hidden_states)
            if not isinstance(output, torch.Tensor):
                raise TypeError("expert forward must return a Tensor")
            return output
        if action == ACTION_FP8:
            before_casts = self.counters.activation_casts
            before_mm = self.counters.scaled_mm_calls
            gate = self.gate.forward_fp8(hidden_states)
            up = self.up.forward_fp8(hidden_states)
            activated = self.original_expert.act_fn(gate) * up
            output = self.down.forward_fp8(activated)
            if self.counters.activation_casts - before_casts != 3:
                raise AssertionError("a non-empty FP8 expert call must cast all three activations")
            if self.counters.scaled_mm_calls - before_mm != 3:
                raise AssertionError("a non-empty FP8 expert call must execute all three scaled GEMMs")
            self.counters.fp8_expert_calls += 1
            return output
        raise ValueError(f"unsupported expert action: {action!r}")


_EXPERT_PATH = re.compile(r"(?:^|\.)layers\.(\d+)\..*experts\.(\d+)$")


class DualResidentExpertBank(nn.Module):
    """Own all per-expert FP8 copies before any measurement begins."""

    def __init__(
        self,
        model: nn.Module,
        *,
        expected_target_linears: int,
        expected_num_layers: Optional[int] = None,
        expected_num_experts: Optional[int] = None,
        counters: Optional[RuntimeCounters] = None,
    ) -> None:
        super().__init__()
        if expected_target_linears <= 0 or expected_target_linears % 3:
            raise ValueError("expected_target_linears must be a positive multiple of three")
        self.counters = counters or RuntimeCounters()
        found: list[tuple[int, int, str, nn.Module]] = []
        for name, module in model.named_modules():
            try:
                _projection_names(module)
            except TypeError:
                continue
            match = _EXPERT_PATH.search(name)
            if match is None:
                raise RuntimeError(f"expert projection module has unparseable layer/expert path: {name}")
            found.append((int(match.group(1)), int(match.group(2)), name, module))
        actual_linears = len(found) * 3
        if actual_linears != expected_target_linears:
            raise RuntimeError(
                "expert target-linear count mismatch: "
                f"actual={actual_linears}, expected={expected_target_linears}"
            )
        if (expected_num_layers is None) != (expected_num_experts is None):
            raise ValueError("expected_num_layers and expected_num_experts must be supplied together")
        if expected_num_layers is not None and expected_num_experts is not None:
            if expected_num_layers <= 0 or expected_num_experts <= 0:
                raise ValueError("expected layer/expert counts must be positive")
            actual_identities = {(layer_id, expert_id) for layer_id, expert_id, _, _ in found}
            expected_identities = {
                (layer_id, expert_id)
                for layer_id in range(expected_num_layers)
                for expert_id in range(expected_num_experts)
            }
            if actual_identities != expected_identities:
                raise RuntimeError(
                    "expert identity grid mismatch: "
                    f"missing={sorted(expected_identities - actual_identities)[:8]}, "
                    f"extra={sorted(actual_identities - expected_identities)[:8]}"
                )
        found.sort(key=lambda item: (item[0], item[1]))
        modules: MutableMapping[str, DualResidentExpertMLP] = {}
        self._index: dict[tuple[int, int], str] = {}
        for layer_id, expert_id, _name, expert in found:
            key = (layer_id, expert_id)
            if key in self._index:
                raise RuntimeError(f"duplicate expert identity: layer={layer_id}, expert={expert_id}")
            module_key = f"layer_{layer_id}_expert_{expert_id}"
            modules[module_key] = DualResidentExpertMLP(expert, self.counters)
            self._index[key] = module_key
        self.experts = nn.ModuleDict(modules)
        if self.counters.weight_casts != expected_target_linears:
            raise AssertionError(
                f"weight casts={self.counters.weight_casts}, expected={expected_target_linears}"
            )

    def get(self, layer_id: int, expert_id: int) -> DualResidentExpertMLP:
        try:
            return self.experts[self._index[(layer_id, expert_id)]]
        except KeyError as exc:
            raise KeyError(f"unknown expert ({layer_id}, {expert_id})") from exc

    def residency_bytes(self) -> dict[str, int]:
        bf16_tensors: list[torch.Tensor] = []
        fp8_tensors: list[torch.Tensor] = []
        scale_tensors: list[torch.Tensor] = []
        for expert in self.experts.values():
            for linear in (expert.gate, expert.up, expert.down):
                bf16_tensors.append(linear.original.weight)
                if linear.original.bias is not None:
                    bf16_tensors.append(linear.original.bias)
                fp8_tensors.append(linear.weight_fp8_t)
                scale_tensors.append(linear.weight_scale)
        return {
            "bf16_expert_bytes": _unique_storage_bytes(bf16_tensors),
            "fp8_expert_bytes": _unique_storage_bytes(fp8_tensors),
            "fp8_scale_bytes": _unique_storage_bytes(scale_tensors),
        }


def _unique_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if identity in seen:
            continue
        seen.add(identity)
        total += storage.nbytes()
    return total


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_config_sha256(named_paths: Mapping[str, Path | str]) -> str:
    """Hash exact named bytes, independent of checkout absolute paths."""

    if not named_paths:
        raise ValueError("hash manifest cannot be empty")
    digest = hashlib.sha256()
    for logical_name in sorted(named_paths):
        if not logical_name or logical_name.startswith("/") or ".." in Path(logical_name).parts:
            raise ValueError(f"unsafe logical hash-manifest name: {logical_name!r}")
        path = Path(named_paths[logical_name])
        if not path.is_file():
            raise FileNotFoundError(path)
        name_bytes = logical_name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def verify_formal_signoff(
    signoff_path: Path | str,
    expected_code_config_sha256: str,
) -> Mapping[str, Any]:
    """Require Phase-4 SIGNED-OFF and an exact code/config hash match."""

    path = Path(signoff_path)
    if not path.is_file():
        raise RuntimeError(f"formal run BLOCKED: Phase-4 signoff is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"formal run BLOCKED: unreadable signoff: {path}") from exc
    if document.get("status") != "SIGNED-OFF":
        raise RuntimeError("formal run BLOCKED: Phase-4 status is not SIGNED-OFF")
    if document.get("phase") != "Phase4":
        raise RuntimeError("formal run BLOCKED: signoff phase is not Phase4")
    actual_hash = document.get("code_config_sha256")
    if actual_hash != expected_code_config_sha256:
        raise RuntimeError(
            "formal run BLOCKED: code/config hash drift "
            f"(signed={actual_hash}, current={expected_code_config_sha256})"
        )
    return document


def write_json_atomic(path: Path | str, value: Mapping[str, Any]) -> None:
    """Write an artifact without leaving a partially-written JSON file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
