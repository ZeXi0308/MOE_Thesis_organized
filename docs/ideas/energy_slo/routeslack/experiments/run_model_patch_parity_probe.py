#!/usr/bin/env python3
"""Development-only native-versus-patched-full GPU parity probe.

This probe qualifies one exact Hugging Face MoE revision at a time.  It runs
the same built-in token IDs through the unmodified model and through
``experiments/shared/capture_moe.py``'s ``full`` patch, then compares:

* full-prompt prefill logits and router decisions;
* two forced, cached, one-token decode steps;
* KV-cache ``+1`` growth and layer/top-k closure.

The output is always non-formal.  Even a passing artifact only establishes a
single-GPU instrumentation-parity fact for one model revision; it is not a
continuous-serving, expert-parallel, latency, energy, or Energy-SLO result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(
    candidate for candidate in HERE.parents if (candidate / "experiments" / "shared").is_dir()
)
SHARED_DIR = REPO_ROOT / "experiments" / "shared"
BCRD_DIR = REPO_ROOT / "docs" / "ideas" / "bcrd" / "experiments"

SCHEMA = "routeslack-model-patch-parity-development-v1"
VERDICT = "DEVELOPMENT_MODEL_PATCH_PARITY_ONLY"
REQUIRED_DECODE_STEPS = 2
EXACT_HF_REVISION = re.compile(r"[0-9a-fA-F]{40}")

PERMANENT_FORMAL_BLOCKERS = (
    "development-only single-GPU instrumentation parity for one model revision",
    "no natural arrivals, continuous batching, scheduler, or serving-engine timing",
    "no expert parallelism, NCCL, RDMA, dispatch, execution, or combine ledger",
    "no latency, board-energy, idle-energy, thermal, or completed-token denominator",
    "does not qualify any other model, revision, dtype, backend, or GPU",
)


class ProbeError(RuntimeError):
    """Raised whenever a parity statement would otherwise be ambiguous."""


@dataclass(frozen=True)
class Thresholds:
    max_abs_logit_error: float
    max_kl_divergence: float
    max_route_weight_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "max_abs_logit_error": self.max_abs_logit_error,
            "max_kl_divergence": self.max_kl_divergence,
            "max_route_weight_error": self.max_route_weight_error,
        }


@dataclass(frozen=True)
class PrefillObservation:
    logits: Any
    cache_length: int | None
    route_batches: tuple[Mapping[str, Any], ...]


class NativeRouterRecorder:
    """Minimal recorder compatible with ``run_cached_decode_steps``."""

    def __init__(self) -> None:
        self.current_sample_id = -1
        self.route_batches: list[dict[str, Any]] = []
        self.routing_weight_batches: list[Any] = []

    def set_sample_id(self, sample_id: int) -> None:
        self.current_sample_id = int(sample_id)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _token_id(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("token ids must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="one Hugging Face model id")
    parser.add_argument(
        "--revision",
        required=True,
        help="exact 40-hex Hugging Face commit; branches and tags are rejected",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--prompt-token-ids",
        type=_token_id,
        nargs="+",
        default=(1, 2, 3, 4),
        help="built-in prompt ids; recorded verbatim and shared by both arms",
    )
    parser.add_argument(
        "--forced-decode-token-ids",
        type=_token_id,
        nargs=REQUIRED_DECODE_STEPS,
        default=(5, 6),
        help="exactly two forced cached-decode ids",
    )
    parser.add_argument("--max-abs-logit-error", type=_positive_float, default=1e-4)
    parser.add_argument("--max-kl-divergence", type=_positive_float, default=1e-7)
    parser.add_argument("--max-route-weight-error", type=_positive_float, default=1e-4)
    return parser.parse_args()


def require_exact_revision(revision: str) -> str:
    if not EXACT_HF_REVISION.fullmatch(revision):
        raise ProbeError("--revision must be an exact 40-hex Hugging Face commit")
    return revision.lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    metadata = json.dumps(
        {"shape": list(value.shape), "dtype": str(value.dtype)},
        sort_keys=True,
    ).encode("utf-8")
    try:
        raw = value.numpy().tobytes(order="C")
    except TypeError:
        raw = value.view(-1).to(dtype=getattr(__import__("torch"), "float32")).numpy().tobytes(order="C")
    return hashlib.sha256(metadata + b"\0" + raw).hexdigest()


def validate_token_ids(
    prompt_token_ids: Sequence[int],
    forced_decode_token_ids: Sequence[int],
    *,
    vocab_size: int,
) -> None:
    if not prompt_token_ids:
        raise ProbeError("prompt token ids must be non-empty")
    if len(forced_decode_token_ids) != REQUIRED_DECODE_STEPS:
        raise ProbeError(f"exactly {REQUIRED_DECODE_STEPS} forced decode ids are required")
    for token_id in (*prompt_token_ids, *forced_decode_token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ProbeError("token ids must be integers")
        if token_id < 0 or token_id >= vocab_size:
            raise ProbeError(f"token id {token_id} is outside vocab_size={vocab_size}")


def logit_metrics(native_logits: Any, patched_logits: Any) -> dict[str, object]:
    """Return finite max-error and KL metrics without hiding a shape drift."""

    import torch

    native = native_logits.detach().to(dtype=torch.float64, device="cpu")
    patched = patched_logits.detach().to(dtype=torch.float64, device="cpu")
    if tuple(native.shape) != tuple(patched.shape):
        raise ProbeError(
            f"native/patched logit shape mismatch: {tuple(native.shape)} != {tuple(patched.shape)}"
        )
    if native.numel() == 0:
        raise ProbeError("logit tensors must be non-empty")
    if not bool(torch.isfinite(native).all()) or not bool(torch.isfinite(patched).all()):
        raise ProbeError("native/patched logits contain non-finite values")
    absolute = (native - patched).abs()
    native_log_prob = torch.log_softmax(native, dim=-1)
    patched_log_prob = torch.log_softmax(patched, dim=-1)
    native_prob = native_log_prob.exp()
    kl_rows = (native_prob * (native_log_prob - patched_log_prob)).sum(dim=-1)
    # Float64 reduction can still produce tiny negative roundoff around zero.
    kl_rows = kl_rows.clamp_min(0.0)
    return {
        "shape": list(native.shape),
        "max_abs_logit_error": float(absolute.max().item()),
        "mean_abs_logit_error": float(absolute.mean().item()),
        "max_kl_divergence_native_to_patched": float(kl_rows.max().item()),
        "mean_kl_divergence_native_to_patched": float(kl_rows.mean().item()),
        "native_logits_sha256": _tensor_sha256(native_logits),
        "patched_logits_sha256": _tensor_sha256(patched_logits),
    }


def _route_map(
    batches: Sequence[Mapping[str, Any]], *, name: str
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for batch in batches:
        try:
            layer = int(batch["layer"])
            selected = batch["selected_experts"]
            weights = batch["routing_weights"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProbeError(f"{name} contains a malformed route batch") from exc
        if layer in result:
            raise ProbeError(f"{name} recorded layer {layer} more than once")
        if selected.ndim != 2 or tuple(weights.shape) != tuple(selected.shape):
            raise ProbeError(f"{name} layer {layer} is not a matching [rows, top_k] pair")
        result[layer] = batch
    if not result:
        raise ProbeError(f"{name} produced no router records")
    return result


def route_metrics(
    native_batches: Sequence[Mapping[str, Any]],
    patched_batches: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
) -> dict[str, object]:
    """Compare route closure, selected expert ids, and captured gate weights."""

    import torch

    native = _route_map(native_batches, name="native")
    patched = _route_map(patched_batches, name="patched")
    native_layers = sorted(native)
    patched_layers = sorted(patched)
    common_layers = sorted(set(native) & set(patched))
    rows: list[dict[str, object]] = []
    selected_equal = native_layers == patched_layers
    shape_equal = True
    expected_rows_closed = True
    max_weight_error = 0.0
    for layer in common_layers:
        left_selected = native[layer]["selected_experts"]
        right_selected = patched[layer]["selected_experts"]
        left_weights = native[layer]["routing_weights"]
        right_weights = patched[layer]["routing_weights"]
        same_shape = tuple(left_selected.shape) == tuple(right_selected.shape)
        shape_equal = shape_equal and same_shape
        expected_rows_closed = expected_rows_closed and (
            int(left_selected.shape[0]) == expected_rows
            and int(right_selected.shape[0]) == expected_rows
        )
        layer_selected_equal = same_shape and bool(torch.equal(left_selected, right_selected))
        selected_equal = selected_equal and layer_selected_equal
        weight_error: float | None = None
        if same_shape:
            weight_error = float(
                (
                    left_weights.detach().float().cpu()
                    - right_weights.detach().float().cpu()
                )
                .abs()
                .max()
                .item()
            )
            if not math.isfinite(weight_error):
                raise ProbeError("router weights contain a non-finite difference")
            max_weight_error = max(max_weight_error, weight_error)
        rows.append(
            {
                "layer": layer,
                "native_shape": list(left_selected.shape),
                "patched_shape": list(right_selected.shape),
                "selected_experts_equal": layer_selected_equal,
                "max_route_weight_error": weight_error,
                "native_selected_experts_sha256": _tensor_sha256(left_selected),
                "patched_selected_experts_sha256": _tensor_sha256(right_selected),
                "native_routing_weights_sha256": _tensor_sha256(left_weights),
                "patched_routing_weights_sha256": _tensor_sha256(right_weights),
            }
        )
    missing_from_patched = sorted(set(native) - set(patched))
    missing_from_native = sorted(set(patched) - set(native))
    if missing_from_patched or missing_from_native:
        shape_equal = False
        expected_rows_closed = False
    native_signature = [
        [layer, int(native[layer]["selected_experts"].shape[1])] for layer in native_layers
    ]
    patched_signature = [
        [layer, int(patched[layer]["selected_experts"].shape[1])] for layer in patched_layers
    ]
    return {
        "expected_rows": expected_rows,
        "native_layer_topk_signature": native_signature,
        "patched_layer_topk_signature": patched_signature,
        "layer_topk_signature_equal": native_signature == patched_signature,
        "expected_row_count_closed": expected_rows_closed,
        "route_tensor_shapes_equal": shape_equal,
        "selected_experts_equal": selected_equal,
        "max_route_weight_error": max_weight_error,
        "missing_layers_from_patched": missing_from_patched,
        "missing_layers_from_native": missing_from_native,
        "layers": rows,
    }


def validate_cache_growth(
    cache_lengths: Sequence[int | None], *, prompt_length: int
) -> dict[str, object]:
    expected = [prompt_length + index + 1 for index in range(len(cache_lengths))]
    observed = [None if value is None else int(value) for value in cache_lengths]
    return {
        "prompt_length": prompt_length,
        "expected_cache_lengths": expected,
        "observed_cache_lengths": observed,
        "cache_advanced_by_one": observed == expected,
    }


def _metrics_pass(metrics: Mapping[str, object], thresholds: Thresholds) -> bool:
    return (
        float(metrics["max_abs_logit_error"]) <= thresholds.max_abs_logit_error
        and float(metrics["max_kl_divergence_native_to_patched"])
        <= thresholds.max_kl_divergence
    )


def _routes_pass(metrics: Mapping[str, object], thresholds: Thresholds) -> bool:
    return (
        bool(metrics["layer_topk_signature_equal"])
        and bool(metrics["expected_row_count_closed"])
        and bool(metrics["route_tensor_shapes_equal"])
        and bool(metrics["selected_experts_equal"])
        and float(metrics["max_route_weight_error"])
        <= thresholds.max_route_weight_error
    )


def compare_observations(
    native_prefill: PrefillObservation,
    patched_prefill: PrefillObservation,
    native_decode_steps: Sequence[Any],
    patched_decode_steps: Sequence[Any],
    *,
    prompt_length: int,
    forced_decode_ids: Sequence[int],
    thresholds: Thresholds,
) -> dict[str, object]:
    if len(native_decode_steps) != REQUIRED_DECODE_STEPS:
        raise ProbeError(
            f"native cached decode produced {len(native_decode_steps)} steps; "
            f"expected {REQUIRED_DECODE_STEPS}"
        )
    if len(patched_decode_steps) != REQUIRED_DECODE_STEPS:
        raise ProbeError(
            f"patched cached decode produced {len(patched_decode_steps)} steps; "
            f"expected {REQUIRED_DECODE_STEPS}"
        )

    prefill_logits = logit_metrics(native_prefill.logits, patched_prefill.logits)
    prefill_routes = route_metrics(
        native_prefill.route_batches,
        patched_prefill.route_batches,
        expected_rows=prompt_length,
    )
    prefill_cache = {
        "expected_cache_length": prompt_length,
        "native_cache_length": native_prefill.cache_length,
        "patched_cache_length": patched_prefill.cache_length,
        "cache_length_equal_to_prompt": (
            native_prefill.cache_length == prompt_length
            and patched_prefill.cache_length == prompt_length
        ),
    }

    decode_rows: list[dict[str, object]] = []
    token_ids_equal = True
    absolute_positions_equal = True
    for index, (native, patched) in enumerate(zip(native_decode_steps, patched_decode_steps)):
        expected_token = int(forced_decode_ids[index])
        row_token_equal = native.token_id == patched.token_id == expected_token
        row_position_equal = (
            native.absolute_position == patched.absolute_position == prompt_length + index
        )
        token_ids_equal = token_ids_equal and row_token_equal
        absolute_positions_equal = absolute_positions_equal and row_position_equal
        if native.logits is None or patched.logits is None:
            raise ProbeError("cached decode parity requires captured logits on both arms")
        logits = logit_metrics(native.logits, patched.logits)
        routes = route_metrics(
            native.route_batches,
            patched.route_batches,
            expected_rows=1,
        )
        decode_rows.append(
            {
                "decode_step": index,
                "forced_token_id": expected_token,
                "native_token_id": native.token_id,
                "patched_token_id": patched.token_id,
                "token_id_equal": row_token_equal,
                "expected_absolute_position": prompt_length + index,
                "native_absolute_position": native.absolute_position,
                "patched_absolute_position": patched.absolute_position,
                "absolute_position_equal": row_position_equal,
                "native_cache_length": native.cache_length,
                "patched_cache_length": patched.cache_length,
                "logits": logits,
                "routes": routes,
            }
        )

    native_cache = validate_cache_growth(
        [step.cache_length for step in native_decode_steps], prompt_length=prompt_length
    )
    patched_cache = validate_cache_growth(
        [step.cache_length for step in patched_decode_steps], prompt_length=prompt_length
    )
    all_signatures = [
        prefill_routes["native_layer_topk_signature"],
        prefill_routes["patched_layer_topk_signature"],
        *[row["routes"]["native_layer_topk_signature"] for row in decode_rows],
        *[row["routes"]["patched_layer_topk_signature"] for row in decode_rows],
    ]
    layer_topk_closed_across_phases = all(
        signature == all_signatures[0] for signature in all_signatures[1:]
    )

    passed = (
        _metrics_pass(prefill_logits, thresholds)
        and _routes_pass(prefill_routes, thresholds)
        and bool(prefill_cache["cache_length_equal_to_prompt"])
        and token_ids_equal
        and absolute_positions_equal
        and bool(native_cache["cache_advanced_by_one"])
        and bool(patched_cache["cache_advanced_by_one"])
        and layer_topk_closed_across_phases
        and all(
            _metrics_pass(row["logits"], thresholds)
            and _routes_pass(row["routes"], thresholds)
            for row in decode_rows
        )
    )
    return {
        "parity_pass": passed,
        "thresholds": thresholds.as_dict(),
        "prefill": {
            "logits": prefill_logits,
            "routes": prefill_routes,
            "cache": prefill_cache,
        },
        "decode": decode_rows,
        "forced_token_ids_equal": token_ids_equal,
        "absolute_positions_equal": absolute_positions_equal,
        "native_cache_growth": native_cache,
        "patched_cache_growth": patched_cache,
        "layer_topk_closed_across_prefill_and_decode": layer_topk_closed_across_phases,
    }


def _moe_layer_ids(model: Any) -> tuple[int, ...]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ProbeError("model does not expose model.layers required by the shared patch")
    result = []
    for layer_id, layer in enumerate(layers):
        if hasattr(layer, "block_sparse_moe"):
            result.append(layer_id)
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "experts") and hasattr(layer.mlp, "gate"):
            result.append(layer_id)
    if not result:
        raise ProbeError("shared patch found no supported MoE layers")
    return tuple(result)


def _normalise_selected_router_weights(config: Any) -> bool:
    if hasattr(config, "norm_topk_prob"):
        return bool(config.norm_topk_prob)
    if str(getattr(config, "model_type", "")).lower() == "mixtral":
        return True
    raise ProbeError(
        "native router-weight normalization is unknown for this architecture; "
        "refusing an ambiguous parity comparison"
    )


class NativeRouterObserver:
    """Passively derive route records from native ``output_router_logits``."""

    def __init__(self, model: Any, recorder: NativeRouterRecorder, layer_ids: Sequence[int]) -> None:
        import torch

        self.model = model
        self.recorder = recorder
        self.layer_ids = tuple(int(value) for value in layer_ids)
        self.top_k = int(model.config.num_experts_per_tok)
        self.normalise_topk = _normalise_selected_router_weights(model.config)
        self.routing_dtype = next(model.parameters()).dtype
        self.torch = torch

    def __call__(self, **kwargs: Any) -> Any:
        kwargs["output_router_logits"] = True
        kwargs.pop("return_dict", None)
        # Call the still-unmodified decoder body directly.  Some Transformers
        # CausalLM wrappers compute an auxiliary router loss whose attention-
        # mask bookkeeping is not defined for a one-token cached step.  The
        # decoder body is the native inference implementation we need, and its
        # hidden state followed by the unchanged lm_head is exactly the logits
        # path compared with the patched CausalLM call.
        output = self.model.model(return_dict=True, **kwargs)
        router_logits = getattr(output, "router_logits", None)
        if router_logits is None:
            raise ProbeError("native model did not expose router_logits")
        router_logits = tuple(value for value in router_logits if value is not None)
        if len(router_logits) != len(self.layer_ids):
            raise ProbeError(
                "native router-logit layer count does not close over shared-patch MoE layers "
                f"(native={len(router_logits)}, patchable={len(self.layer_ids)})"
            )
        for layer_id, logits in zip(self.layer_ids, router_logits):
            if logits.ndim != 2:
                raise ProbeError("native router logits must have shape [tokens, experts]")
            probability = self.torch.softmax(logits.float(), dim=-1)
            weights, experts = self.torch.topk(probability, k=self.top_k, dim=-1)
            if self.normalise_topk:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            weights = weights.to(dtype=self.routing_dtype)
            selected_cpu = experts.detach().cpu()
            weights_cpu = weights.detach().float().cpu()
            self.recorder.route_batches.append(
                {
                    "sample_id": self.recorder.current_sample_id,
                    "layer": layer_id,
                    "selected_experts": selected_cpu,
                    "routing_weights": weights_cpu,
                }
            )
            self.recorder.routing_weight_batches.append(weights_cpu)
        return SimpleNamespace(
            logits=self.model.lm_head(output.last_hidden_state),
            past_key_values=output.past_key_values,
            router_logits=router_logits,
        )


def _load_reused_components() -> tuple[Any, Any, Any, Any]:
    # Both legacy modules use flat imports (not package-relative imports) and
    # both trees contain a ``policies.py``.  Put shared first so capture_moe's
    # policy contract cannot accidentally resolve to BCRD's unrelated module;
    # capture_native_routes' ``core.py`` still resolves from BCRD afterwards.
    for path in (BCRD_DIR, SHARED_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from capture_moe import patch_mixtral_moe
    from capture_native_routes import (
        _cache_sequence_length,
        _clear_recorder,
        _snapshot_route_batches,
        run_cached_decode_steps,
    )

    return (
        patch_mixtral_moe,
        _cache_sequence_length,
        _clear_recorder,
        (_snapshot_route_batches, run_cached_decode_steps),
    )


def _run_prefill(
    model: Any,
    recorder: Any,
    inputs: Mapping[str, Any],
    *,
    cache_length_fn: Any,
    clear_recorder_fn: Any,
    snapshot_fn: Any,
) -> PrefillObservation:
    import torch

    input_ids = inputs["input_ids"]
    expected_rows = int(input_ids.shape[1])
    clear_recorder_fn(recorder)
    with torch.inference_mode():
        output = model(**dict(inputs), use_cache=True, return_dict=True)
    logits = getattr(output, "logits", None)
    cache = getattr(output, "past_key_values", None)
    if logits is None or cache is None:
        raise ProbeError("prefill did not return logits and past_key_values")
    result = PrefillObservation(
        logits=logits.detach().float().cpu(),
        cache_length=cache_length_fn(cache),
        route_batches=snapshot_fn(recorder, expected_batch=expected_rows),
    )
    clear_recorder_fn(recorder)
    return result


def _dtype(torch_module: Any, name: str) -> Any:
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[name]


def _git_metadata() -> dict[str, object]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                args,
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status_porcelain_for_probe_sources": run(
            "git",
            "status",
            "--porcelain",
            "--",
            str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            str((BCRD_DIR / "capture_native_routes.py").relative_to(REPO_ROOT)),
            str((SHARED_DIR / "capture_moe.py").relative_to(REPO_ROOT)),
        ),
    }


def _cuda_environment(torch_module: Any, device: Any) -> dict[str, object]:
    properties = torch_module.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    if uuid is None or not str(uuid).strip():
        raise ProbeError("CUDA device UUID is unavailable")
    cuda_uuid = str(uuid)
    normalised_cuda_uuid = (
        cuda_uuid if cuda_uuid.upper().startswith("GPU-") else f"GPU-{cuda_uuid}"
    )
    driver_version: str | None = None
    nvml_uuid: str | None = None
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            driver_version = driver.decode() if isinstance(driver, bytes) else str(driver)
            for physical_index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
                observed_uuid = pynvml.nvmlDeviceGetUUID(handle)
                candidate = (
                    observed_uuid.decode()
                    if isinstance(observed_uuid, bytes)
                    else str(observed_uuid)
                )
                normalised_candidate = (
                    candidate
                    if candidate.upper().startswith("GPU-")
                    else f"GPU-{candidate}"
                )
                if normalised_candidate.lower() == normalised_cuda_uuid.lower():
                    nvml_uuid = normalised_candidate
                    break
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001 - recorded, then failed closed below
        raise ProbeError(f"NVML CUDA identity metadata is unavailable: {exc}") from exc

    if nvml_uuid is None:
        raise ProbeError(f"no NVML physical GPU matches CUDA UUID {normalised_cuda_uuid}")
    normalised_nvml_uuid = nvml_uuid
    if normalised_cuda_uuid.lower() != normalised_nvml_uuid.lower():
        raise ProbeError(
            f"CUDA/NVML UUID mismatch: {normalised_cuda_uuid} != {normalised_nvml_uuid}"
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "torch": torch_module.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_runtime": torch_module.version.cuda,
        "cudnn": torch_module.backends.cudnn.version(),
        "driver": driver_version,
        "device_index": device.index,
        "gpu_name": str(properties.name),
        "gpu_uuid": normalised_cuda_uuid,
        "nvml_gpu_uuid": normalised_nvml_uuid,
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "deterministic_algorithms_enabled": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
    }


def hash_snapshot_files(snapshot_dir: Path) -> dict[str, object]:
    if not snapshot_dir.is_dir():
        raise ProbeError(f"resolved model snapshot directory does not exist: {snapshot_dir}")
    rows: list[dict[str, object]] = []
    weight_files = 0
    for path in sorted(candidate for candidate in snapshot_dir.rglob("*") if candidate.is_file()):
        relative = path.relative_to(snapshot_dir).as_posix()
        suffixes = tuple(path.name.lower().split("."))
        if path.name.endswith((".safetensors", ".bin", ".pt")):
            weight_files += 1
        resolved = path.resolve()
        rows.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "resolved_blob_name": resolved.name,
                "resolved_blob_name_is_sha256": bool(re.fullmatch(r"[0-9a-f]{64}", resolved.name)),
                "name_components": list(suffixes),
            }
        )
    if not rows:
        raise ProbeError("resolved model snapshot is empty")
    if weight_files == 0:
        raise ProbeError("resolved model snapshot contains no checkpoint weight file")
    return {
        "snapshot_dir": str(snapshot_dir),
        "files": rows,
        "aggregate_sha256": canonical_json_sha256(rows),
        "all_files_hashed": True,
        "weight_file_count": weight_files,
    }


def _model_state_schema(model: Any) -> dict[str, object]:
    rows = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in model.state_dict().items()
    ]
    return {
        "tensors": len(rows),
        "parameters_and_buffers": sum(int(row["numel"]) for row in rows),
        "schema_sha256": canonical_json_sha256(rows),
    }


def _source_hashes() -> dict[str, str]:
    files = (
        Path(__file__).resolve(),
        BCRD_DIR / "capture_native_routes.py",
        SHARED_DIR / "capture_moe.py",
    )
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in files
    }


def _resolve_snapshot(model_id: str, revision: str, *, offline: bool) -> tuple[Path, str]:
    from transformers.utils.hub import cached_file

    config_path = cached_file(
        model_id,
        "config.json",
        revision=revision,
        local_files_only=offline,
    )
    if config_path is None:
        raise ProbeError("could not resolve config.json for the exact revision")
    snapshot_dir = Path(config_path).resolve().parent
    # The public path normally contains snapshots/<commit>/config.json.  The
    # resolved blob path does not, so inspect the original symlink path first.
    unresolved_parent = Path(config_path).parent
    resolved_revision = unresolved_parent.name
    if not EXACT_HF_REVISION.fullmatch(resolved_revision):
        candidates = [part for part in Path(config_path).parts if EXACT_HF_REVISION.fullmatch(part)]
        if candidates:
            resolved_revision = candidates[-1]
    if not EXACT_HF_REVISION.fullmatch(resolved_revision):
        raise ProbeError("could not recover an exact commit from the Hugging Face snapshot path")
    # Hash the snapshot symlink tree rather than the single resolved blob dir.
    return unresolved_parent, resolved_revision.lower()


def execute_probe(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM

    revision = require_exact_revision(args.revision)
    if args.device_index < 0:
        raise ProbeError("device index must be non-negative")
    if not torch.cuda.is_available():
        raise ProbeError("CUDA is required; CPU parity is only covered by unit tests")
    if args.device_index >= torch.cuda.device_count():
        raise ProbeError(
            f"device index {args.device_index} is outside CUDA device count {torch.cuda.device_count()}"
        )
    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)
    torch.manual_seed(20260728)
    torch.cuda.manual_seed_all(20260728)

    snapshot_dir, snapshot_revision = _resolve_snapshot(
        args.model, revision, offline=bool(args.offline)
    )
    if snapshot_revision != revision:
        raise ProbeError(
            f"resolved Hugging Face revision drifted: requested={revision}, resolved={snapshot_revision}"
        )
    snapshot_hashes = hash_snapshot_files(snapshot_dir)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        local_files_only=bool(args.offline),
        dtype=_dtype(torch, args.dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    resolved_from_config = str(getattr(model.config, "_commit_hash", "")).lower()
    if resolved_from_config != revision:
        raise ProbeError(
            "loaded model config does not confirm the requested exact revision "
            f"(requested={revision}, config={resolved_from_config or 'missing'})"
        )
    vocab_size = int(model.config.vocab_size)
    prompt_ids = tuple(int(value) for value in args.prompt_token_ids)
    forced_ids = tuple(int(value) for value in args.forced_decode_token_ids)
    validate_token_ids(prompt_ids, forced_ids, vocab_size=vocab_size)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    forced_decode_ids = torch.tensor([forced_ids], dtype=torch.long, device=device)

    (
        patch_mixtral_moe,
        cache_length_fn,
        clear_recorder_fn,
        (snapshot_fn, run_cached_decode_steps),
    ) = _load_reused_components()
    layer_ids = _moe_layer_ids(model)

    native_recorder = NativeRouterRecorder()
    native_recorder.set_sample_id(0)
    native_model = NativeRouterObserver(model, native_recorder, layer_ids)
    native_prefill = _run_prefill(
        native_model,
        native_recorder,
        inputs,
        cache_length_fn=cache_length_fn,
        clear_recorder_fn=clear_recorder_fn,
        snapshot_fn=snapshot_fn,
    )
    native_decode = run_cached_decode_steps(
        native_model,
        native_recorder,
        inputs,
        max_steps=REQUIRED_DECODE_STEPS,
        eos_token_id=None,
        forced_decode_ids=forced_decode_ids,
        capture_logits=True,
    )
    torch.cuda.synchronize(device)

    patched_recorder = patch_mixtral_moe(
        model,
        "full",
        num_receiver_groups=1,
        record_routes=True,
        record_diagnostics=False,
    )
    patched_recorder.set_sample_id(0)
    patched_prefill = _run_prefill(
        model,
        patched_recorder,
        inputs,
        cache_length_fn=cache_length_fn,
        clear_recorder_fn=clear_recorder_fn,
        snapshot_fn=snapshot_fn,
    )
    patched_decode = run_cached_decode_steps(
        model,
        patched_recorder,
        inputs,
        max_steps=REQUIRED_DECODE_STEPS,
        eos_token_id=None,
        forced_decode_ids=forced_decode_ids,
        capture_logits=True,
    )
    torch.cuda.synchronize(device)

    thresholds = Thresholds(
        max_abs_logit_error=float(args.max_abs_logit_error),
        max_kl_divergence=float(args.max_kl_divergence),
        max_route_weight_error=float(args.max_route_weight_error),
    )
    comparison = compare_observations(
        native_prefill,
        patched_prefill,
        native_decode,
        patched_decode,
        prompt_length=len(prompt_ids),
        forced_decode_ids=forced_ids,
        thresholds=thresholds,
    )
    config_dict = model.config.to_dict()
    environment = _cuda_environment(torch, device)
    input_identity = {
        "source": "builtin_cli_token_ids_no_tokenizer",
        "prompt_token_ids": list(prompt_ids),
        "forced_decode_token_ids": list(forced_ids),
        "required_decode_steps": REQUIRED_DECODE_STEPS,
    }
    input_identity["sha256"] = canonical_json_sha256(input_identity)
    return {
        "status": (
            "DEVELOPMENT_PARITY_PASS"
            if comparison["parity_pass"]
            else "DEVELOPMENT_PARITY_FAIL"
        ),
        "parity": comparison,
        "model": {
            "id": args.model,
            "requested_revision": revision,
            "resolved_snapshot_revision": snapshot_revision,
            "resolved_config_revision": resolved_from_config,
            "class": type(model).__name__,
            "model_type": str(model.config.model_type),
            "architectures": list(getattr(model.config, "architectures", None) or ()),
            "dtype": args.dtype,
            "vocab_size": vocab_size,
            "moe_layer_ids": list(layer_ids),
            "num_experts_per_tok": int(model.config.num_experts_per_tok),
            "config": config_dict,
            "config_sha256": canonical_json_sha256(config_dict),
            "state_schema": _model_state_schema(model),
            "snapshot": snapshot_hashes,
        },
        "input_identity": input_identity,
        "environment": environment,
        "environment_sha256": canonical_json_sha256(environment),
        "git": _git_metadata(),
        "source_sha256": _source_hashes(),
        "native_route_provenance": "derived from unmodified output_router_logits",
        "patched_route_provenance": "shared patch full policy recorder",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_result_bundle(
    output_dir: Path,
    payload: Mapping[str, object],
    *,
    error: BaseException | None = None,
) -> dict[str, object]:
    """Write a non-promotable result plus an external artifact hash manifest."""

    parity_pass = bool(payload.get("parity", {}).get("parity_pass", False)) if isinstance(payload.get("parity"), Mapping) else False
    blockers = list(PERMANENT_FORMAL_BLOCKERS)
    if not parity_pass:
        blockers.append("native-versus-patched-full parity gate did not pass")
    if error is not None:
        blockers.append(f"probe execution failed closed: {type(error).__name__}")
    result = {
        **dict(payload),
        "schema": SCHEMA,
        "verdict": VERDICT,
        "formal_eligible": False,
        "formal_result": False,
        "scientific_result_eligible": False,
        "development_only": True,
        "formal_blockers": blockers,
        "error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
    }
    if error is not None:
        result["status"] = "DEVELOPMENT_PARITY_BLOCKED"
    result_path = output_dir / "result.json"
    _write_json(result_path, result)
    manifest = {
        "schema": f"{SCHEMA}-manifest",
        "verdict": VERDICT,
        "formal_eligible": False,
        "formal_result": False,
        "scientific_result_eligible": False,
        "result_path": result_path.name,
        "result_sha256": sha256_file(result_path),
        "source_sha256": payload.get("source_sha256", {}),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return result


def main() -> None:
    args = parse_args()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}") from exc

    error: BaseException | None = None
    try:
        payload = execute_probe(args)
    except BaseException as exc:  # noqa: BLE001 - persist every fail-closed boundary
        error = exc
        payload = {
            "status": "DEVELOPMENT_PARITY_BLOCKED",
            "model": {
                "id": args.model,
                "requested_revision": args.revision,
                "dtype": args.dtype,
            },
            "input_identity": {
                "source": "builtin_cli_token_ids_no_tokenizer",
                "prompt_token_ids": list(args.prompt_token_ids),
                "forced_decode_token_ids": list(args.forced_decode_token_ids),
            },
            "git": _git_metadata(),
            "source_sha256": _source_hashes(),
        }
    result = write_result_bundle(args.output_dir, payload, error=error)
    print(
        json.dumps(
            {
                "status": result["status"],
                "formal_eligible": False,
                "result": str(args.output_dir / "result.json"),
            },
            sort_keys=True,
        )
    )
    if error is not None or not bool(result.get("parity", {}).get("parity_pass", False)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
