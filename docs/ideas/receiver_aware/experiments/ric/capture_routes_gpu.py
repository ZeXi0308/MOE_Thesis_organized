#!/usr/bin/env python3
"""Capture identity-complete native RIC-v1 routes on CUDA.

The model is never monkey-patched.  Each native forward exposes router logits
through ``output_router_logits`` while an independent hook records every MoE
gate output.  The producer requires per-layer top-k/weight parity before a
route row is written.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prepare_data import (  # noqa: E402
    DataPreparationError,
    _producer_source_sha256 as _prepare_data_source_sha256,
    add_self_hash,
    canonical_json_bytes,
    frozen_concat_sha256,
    sha256_bytes,
    sha256_file,
    validate_self_hash,
    verify_embedded_formal_signoff,
)
from formal_provenance import (  # noqa: E402
    FormalProvenanceError,
    is_sha256,
    load_json_mapping_strict,
    materialize_verified_signoff,
    validate_data_manifest_fields,
    verify_phase4_signoff,
)


REPO_ROOT = HERE.parents[4]
IDEA_ROOT = HERE.parents[1]
DEFAULT_CONFIG = IDEA_ROOT / "configs" / "ric_v1.json"
DEFAULT_PROTOCOL = IDEA_ROOT / "RIC_Phase2_冻结实验协议_2026-07-22.md"
ROUTE_EPOCH = 1
NATIVE_TOPK_SELECTION_RULE = (
    "torch.topk(torch.softmax(raw_gate_logits,dim=-1,dtype=torch.float32),"
    "k,dim=-1);preserve_returned_slot_order"
)


class RouteCaptureError(RuntimeError):
    """A native-route, identity, or provenance invariant failed."""


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        value = load_json_mapping_strict(path, label="RIC config")
    except FormalProvenanceError as exc:
        raise RouteCaptureError(str(exc)) from exc
    if value.get("schema_version") != "ric-config-v1":
        raise RouteCaptureError("not a RIC-v1 config")
    if value.get("status") != "PHASE2_FROZEN_NO_SCIENTIFIC_RESULT":
        raise RouteCaptureError("RIC config is not frozen")
    return value


def _load_data_manifest(
    path: Path,
    *,
    mode: str,
    model_key: str,
    config: Mapping[str, Any],
    protocol_sha256: str,
    config_sha256: str,
) -> Mapping[str, Any]:
    try:
        value = load_json_mapping_strict(path, label="RIC data manifest")
    except FormalProvenanceError as exc:
        raise RouteCaptureError(str(exc)) from exc
    if value.get("schema_version") != "ric-data-manifest-v1":
        raise RouteCaptureError("not a RIC data manifest")
    try:
        validate_self_hash(value)
    except DataPreparationError as exc:
        raise RouteCaptureError(str(exc)) from exc
    role = value.get("role")
    if role not in {"calibration", "sealed"}:
        raise RouteCaptureError("invalid data role")
    if mode == "dev" and role == "sealed":
        raise RouteCaptureError("dev mode is forbidden from reading sealed data")
    try:
        validate_data_manifest_fields(
            value,
            mode=mode,
            role=str(role),
            config=config,
            protocol_sha256=protocol_sha256,
            config_sha256=config_sha256,
            expected_prepare_data_source_sha256=_prepare_data_source_sha256(),
        )
    except (FormalProvenanceError, DataPreparationError) as exc:
        raise RouteCaptureError(str(exc)) from exc
    if mode == "formal":
        try:
            verify_embedded_formal_signoff(
                path,
                value,
                protocol_sha256=protocol_sha256,
                config_sha256=config_sha256,
            )
        except DataPreparationError as exc:
            raise RouteCaptureError(str(exc)) from exc
    data_cfg = config["data"]
    role_cfg = data_cfg[role]
    if value.get("candidate_window") != [
        int(role_cfg["candidate_row_start_inclusive"]),
        int(role_cfg["candidate_row_end_exclusive"]),
    ]:
        raise RouteCaptureError("data candidate window mismatch")
    if value.get("selection_seed") != int(data_cfg["selection_seed"]):
        raise RouteCaptureError("data selection seed mismatch")
    if value.get("sequence_tokens") != int(data_cfg["sequence_length"]):
        raise RouteCaptureError("sequence length mismatch")
    requests = value.get("requests")
    if not isinstance(requests, list) or len(requests) != int(role_cfg["document_count"]):
        raise RouteCaptureError("data request count mismatch")
    ids = [str(row.get("request_id")) for row in requests if isinstance(row, Mapping)]
    hashes = [str(row.get("text_sha256")) for row in requests if isinstance(row, Mapping)]
    if len(ids) != len(requests) or len(set(ids)) != len(ids):
        raise RouteCaptureError("request identity is missing or duplicated")
    if len(set(hashes)) != len(hashes):
        raise RouteCaptureError("request text hashes are duplicated")
    model_spec = config["models"][model_key]
    expected_revision = f"{model_spec['repo_id']}@{model_spec['revision']}"
    revisions = value.get("model_revisions")
    if not isinstance(revisions, Mapping) or revisions.get(model_key) != expected_revision:
        raise RouteCaptureError("data/model revision mismatch")
    return value


def _producer_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        HERE / "prepare_data.py",
        HERE / "formal_provenance.py",
    ):
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_formal_signoff(
    path: Path | None,
    *,
    protocol_sha256: str,
    config_sha256: str,
    source_sha256: str,
    data_manifest_sha256: str,
    data_producer_signoff_sha256: str,
    model_key: str,
    model_tree_manifest_sha256: str,
) -> Mapping[str, Any]:
    try:
        return verify_phase4_signoff(
            path,
            repo_root=REPO_ROOT,
            expected_fields={
                "stage": "capture_routes",
                "protocol_sha256": protocol_sha256,
                "config_sha256": config_sha256,
                "capture_routes_source_sha256": source_sha256,
                "data_manifest_sha256": data_manifest_sha256,
                "data_producer_signoff_sha256": data_producer_signoff_sha256,
                "model_key": model_key,
                "model_tree_manifest_sha256": model_tree_manifest_sha256,
                "prepare_data_source_sha256": _prepare_data_source_sha256(),
            },
            required_source_paths=(
                Path(__file__),
                HERE / "prepare_data.py",
                HERE / "formal_provenance.py",
            ),
        )
    except (FormalProvenanceError, DataPreparationError) as exc:
        raise RouteCaptureError(str(exc)) from exc


def _canonical_request_hash(request_id: str) -> str:
    return sha256_bytes(request_id.encode("utf-8"))


def origin_lpt(requests: Sequence[Mapping[str, Any]], ep_size: int) -> dict[str, int]:
    if ep_size < 1:
        raise RouteCaptureError("ep_size must be positive")
    loads = [0] * ep_size
    result: dict[str, int] = {}
    weighted = [
        (str(row["request_id"]), 128, _canonical_request_hash(str(row["request_id"])))
        for row in requests
    ]
    for request_id, weight, _request_hash in sorted(
        weighted, key=lambda item: (-item[1], item[2])
    ):
        # The frozen equal-load tie break is request-specific.  Concatenation is
        # exact UTF-8(request_id) || ASCII(decimal rank), with no delimiter.
        rank = min(
            range(ep_size),
            key=lambda candidate: (
                loads[candidate],
                sha256_bytes(f"{request_id}{candidate}".encode("utf-8")),
                candidate,
            ),
        )
        result[request_id] = rank
        loads[rank] += weight
    return result


def expert_sender(expert_id: int, num_experts: int, ep_size: int) -> int:
    if not 0 <= expert_id < num_experts:
        raise RouteCaptureError("expert outside frozen range")
    return min(ep_size - 1, expert_id * ep_size // num_experts)


def selected_layers(
    layer_ids: Sequence[int], *, selection_seed: int, model_revision: str, count: int
) -> list[int]:
    if len(set(layer_ids)) != len(layer_ids):
        raise RouteCaptureError("duplicate discovered layer ids")
    ranked = sorted(
        layer_ids,
        key=lambda layer_id: frozen_concat_sha256(
            selection_seed, model_revision, layer_id
        ),
    )
    if len(ranked) < count:
        raise RouteCaptureError("model exposes fewer MoE layers than frozen selection")
    return sorted(ranked[:count])


def assigned_layer(request_id: str, frozen_layers: Sequence[int]) -> int:
    if not frozen_layers:
        raise RouteCaptureError("no frozen layers")
    index = int(sha256_bytes(request_id.encode("utf-8")), 16) % len(frozen_layers)
    return int(frozen_layers[index])


def validate_full_tokenizer_length(
    request: Mapping[str, Any],
    *,
    model_key: str,
    observed_length: int,
    minimum_length: int,
) -> None:
    lengths = request.get("token_lengths")
    if not isinstance(lengths, Mapping) or type(lengths.get(model_key)) is not int:
        raise RouteCaptureError("manifest tokenizer length is missing")
    if observed_length != int(lengths[model_key]):
        raise RouteCaptureError("local model-tree tokenizer length differs from manifest")
    if observed_length < minimum_length:
        raise RouteCaptureError("selected request is below frozen tokenizer minimum")


def _layer_index(name: str) -> int:
    matches = re.findall(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if len(matches) != 1:
        raise RouteCaptureError(f"cannot derive one decoder layer id from {name!r}")
    return int(matches[0])


def discover_moe_modules(model: Any) -> list[tuple[int, str, Any]]:
    discovered: list[tuple[int, str, Any]] = []
    seen_layers: set[int] = set()
    for name, module in model.named_modules():
        if not hasattr(module, "gate") or not hasattr(module, "experts"):
            continue
        experts = getattr(module, "experts")
        try:
            expert_count = len(experts)
        except TypeError:
            continue
        if expert_count < 2:
            continue
        layer_id = _layer_index(name)
        if layer_id in seen_layers:
            raise RouteCaptureError(f"multiple MoE modules in decoder layer {layer_id}")
        seen_layers.add(layer_id)
        discovered.append((layer_id, name, module))
    if not discovered:
        raise RouteCaptureError("no native MoE gate+experts modules discovered")
    return sorted(discovered, key=lambda row: row[0])


def _config_values(model_config: Any, names: Sequence[str]) -> set[int]:
    objects = [model_config]
    nested = getattr(model_config, "text_config", None)
    if nested is not None:
        objects.append(nested)
    values: set[int] = set()
    for obj in objects:
        for name in names:
            value = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
            if value is None or isinstance(value, bool):
                continue
            try:
                values.add(int(value))
            except (TypeError, ValueError) as exc:
                raise RouteCaptureError(f"model config {name} is not an integer") from exc
    return values


def validate_model_config_layer_census(
    model_config: Any,
    modules: Sequence[tuple[int, str, Any]],
    *,
    expected_num_experts: int,
    expected_top_k: int,
) -> dict[str, Any]:
    """Derive the all-MoE layer census from model config, not hook output."""

    layer_values = _config_values(model_config, ("num_hidden_layers",))
    expert_values = _config_values(
        model_config,
        ("num_experts", "num_local_experts", "n_routed_experts", "n_experts"),
    )
    topk_values = _config_values(
        model_config,
        (
            "num_experts_per_tok",
            "num_experts_per_token",
            "num_selected_experts",
            "moe_top_k",
        ),
    )
    if len(layer_values) != 1:
        raise RouteCaptureError("model config lacks one exact num_hidden_layers")
    if expert_values != {int(expected_num_experts)}:
        raise RouteCaptureError("model-config expert count differs from frozen spec")
    if topk_values != {int(expected_top_k)}:
        raise RouteCaptureError("model-config top-k differs from frozen spec")
    num_hidden_layers = next(iter(layer_values))
    expected_layers = list(range(num_hidden_layers))
    discovered_layers = [int(layer) for layer, _name, _module in modules]
    if discovered_layers != expected_layers:
        missing = sorted(set(expected_layers) - set(discovered_layers))
        extra = sorted(set(discovered_layers) - set(expected_layers))
        raise RouteCaptureError(
            f"native MoE layer census differs from model config: missing={missing}, extra={extra}"
        )
    if any(len(module.experts) != expected_num_experts for _, _, module in modules):
        raise RouteCaptureError("native expert count differs from frozen model spec")
    return {
        "expected_layer_source": "model_config.num_hidden_layers_all_layers_are_moe",
        "num_hidden_layers": num_hidden_layers,
        "expected_layers": expected_layers,
        "model_config_num_experts": expected_num_experts,
        "model_config_top_k": expected_top_k,
    }


def _normalizes_topk(moe: Any) -> bool:
    if hasattr(moe, "norm_topk_prob"):
        return bool(moe.norm_topk_prob)
    if "mixtral" in type(moe).__name__.lower():
        return True
    raise RouteCaptureError(
        f"unknown native top-k normalization for {type(moe).__name__}"
    )


def validate_native_moe_implementation(
    modules: Sequence[tuple[int, str, Any]],
    *,
    model_spec: Mapping[str, Any],
    route_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind exact native route semantics, including tied-slot ordering."""

    if route_config.get("native_topk_selection_rule") != NATIVE_TOPK_SELECTION_RULE:
        raise RouteCaptureError("native top-k selection rule is not frozen")
    if route_config.get("native_topk_dispatch_capture_required") is not True:
        raise RouteCaptureError("native top-k dispatch capture must be required")
    if route_config.get("route_rows_use_captured_native_topk") is not True:
        raise RouteCaptureError("route rows must use captured native top-k")
    if route_config.get("effective_route_weight") != (
        "topk_fp32_then_optional_renorm_then_cast_native_hidden_dtype"
    ):
        raise RouteCaptureError("effective route-weight rule is not frozen")
    if route_config.get("stable_sort_substitution_allowed") is not False:
        raise RouteCaptureError("stable-sort substitution must be forbidden")
    expected_class = model_spec.get("native_moe_class")
    expected_source_sha256 = model_spec.get("native_moe_forward_source_sha256")
    if not isinstance(expected_class, str) or not isinstance(expected_source_sha256, str):
        raise RouteCaptureError("model spec lacks frozen native MoE implementation")
    observed_classes: set[str] = set()
    observed_source_hashes: set[str] = set()
    expected_top_k = int(model_spec["top_k"])
    for _layer_id, _name, module in modules:
        module_class = f"{type(module).__module__}.{type(module).__name__}"
        try:
            forward_source = inspect.getsource(type(module).forward)
        except (OSError, TypeError) as exc:
            raise RouteCaptureError("cannot inspect native MoE forward source") from exc
        source_sha256 = sha256_bytes(forward_source.encode("utf-8"))
        observed_classes.add(module_class)
        observed_source_hashes.add(source_sha256)
        if getattr(module, "top_k", None) != expected_top_k:
            raise RouteCaptureError("native MoE module top-k differs from frozen spec")
    if observed_classes != {expected_class}:
        raise RouteCaptureError("native MoE class differs from frozen spec")
    if observed_source_hashes != {expected_source_sha256}:
        raise RouteCaptureError("native MoE forward source differs from frozen spec")
    return {
        "native_topk_selection_rule": NATIVE_TOPK_SELECTION_RULE,
        "native_topk_dispatch_capture_required": True,
        "route_rows_use_captured_native_topk": True,
        "effective_route_weight": (
            "topk_fp32_then_optional_renorm_then_cast_native_hidden_dtype"
        ),
        "stable_sort_substitution_allowed": False,
        "native_moe_class": expected_class,
        "native_moe_forward_source_sha256": expected_source_sha256,
    }


def _tensor_sha256(tensor: Any) -> str:
    import torch

    cpu = tensor.detach().cpu()
    contiguous = cpu.contiguous()
    raw_bytes = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode())
    digest.update(b"\0")
    digest.update(str(cpu.dtype).encode())
    digest.update(b"\0")
    digest.update(str(tuple(cpu.stride())).encode())
    digest.update(b"\0")
    digest.update(raw_bytes)
    return digest.hexdigest()


def _route_tuple_sha256(experts: Any, effective_weights: Any) -> str:
    """Hash the ordered (token, slot, expert, effective-weight) tensor pair."""

    import torch

    if tuple(experts.shape) != tuple(effective_weights.shape):
        raise RouteCaptureError("route tuple expert/weight shape mismatch")
    if experts.dtype != torch.int64 or not effective_weights.dtype.is_floating_point:
        raise RouteCaptureError("route tuple expert/weight dtype mismatch")
    if not bool(torch.isfinite(effective_weights).all()):
        raise RouteCaptureError("route tuple contains non-finite weight")
    return frozen_concat_sha256(
        "ric-native-route-tuple-v1",
        _tensor_sha256(experts),
        _tensor_sha256(effective_weights),
    )


def _effective_route_weights(
    topk_values: Any,
    *,
    normalize_topk: bool,
    output_dtype: Any,
) -> Any:
    return _precast_route_weights(
        topk_values, normalize_topk=normalize_topk
    ).to(dtype=output_dtype)


def _precast_route_weights(topk_values: Any, *, normalize_topk: bool) -> Any:
    weights = topk_values
    if normalize_topk:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights


def validate_raw_router_tensor_identity(
    gate_logits: Any,
    output_router_logits: Any,
    *,
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    """Validate raw router identity before any reshape can erase provenance."""

    if (
        tuple(gate_logits.shape) != expected_shape
        or tuple(output_router_logits.shape) != expected_shape
        or gate_logits.dtype != output_router_logits.dtype
        or tuple(gate_logits.stride()) != tuple(output_router_logits.stride())
    ):
        raise RouteCaptureError(
            "native/gate raw router shape, dtype, or stride mismatch"
        )
    gate_sha256 = _tensor_sha256(gate_logits)
    output_sha256 = _tensor_sha256(output_router_logits)
    if gate_sha256 != output_sha256:
        raise RouteCaptureError(
            "framework output_router_logits differs from raw gate hook"
        )
    return {
        "gate_hook_logit_sha256": gate_sha256,
        "output_router_logit_sha256": output_sha256,
        "raw_logit_hash_equal": True,
    }


def make_native_topk_capture_mode(
    *,
    active_layer: dict[str, int | None],
    expected_num_experts: int,
    expected_top_k: int,
    expected_tokens: int,
) -> Any:
    """Observe the actual native aten.topk outputs without changing them."""

    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class NativeTopKCapture(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.calls: dict[int, tuple[Any, Any]] = {}

        def __torch_dispatch__(
            self,
            func: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: Mapping[str, Any] | None = None,
        ) -> Any:
            call_kwargs = {} if kwargs is None else dict(kwargs)
            output = func(*args, **call_kwargs)
            if func != torch.ops.aten.topk.default:
                return output
            layer_id = active_layer.get("value")
            if layer_id is None:
                return output
            source = args[0]
            k = int(args[1])
            dim = int(args[2]) if len(args) > 2 else int(call_kwargs.get("dim", -1))
            largest = bool(args[3]) if len(args) > 3 else bool(
                call_kwargs.get("largest", True)
            )
            sorted_output = bool(args[4]) if len(args) > 4 else bool(
                call_kwargs.get("sorted", True)
            )
            if (
                tuple(source.shape) != (expected_tokens, expected_num_experts)
                or source.dtype != torch.float32
                or k != expected_top_k
                or dim not in {-1, 1}
                or not largest
                or not sorted_output
            ):
                raise RouteCaptureError("native MoE aten.topk signature drift")
            if layer_id in self.calls:
                raise RouteCaptureError(f"native layer {layer_id} called aten.topk twice")
            values, indices = output
            if tuple(values.shape) != (expected_tokens, expected_top_k):
                raise RouteCaptureError("native aten.topk output shape mismatch")
            self.calls[layer_id] = (
                values.detach().clone(),
                indices.detach().clone(),
            )
            return output

    return NativeTopKCapture()


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "shape"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if hasattr(item, "shape"):
                return item
    raise RouteCaptureError("native MoE value contains no tensor")


def reconstruct_native_moe_output(
    *,
    moe: Any,
    hidden_states: Any,
    selected_experts: Any,
    effective_weights: Any,
) -> tuple[Any, Any, Any]:
    """Independently execute experts using the observed native route tuple."""

    import torch

    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    experts = selected_experts
    weights = effective_weights
    if (
        len(experts.shape) != 2
        or tuple(experts.shape) != tuple(weights.shape)
        or int(experts.shape[0]) != int(hidden.shape[0])
        or weights.dtype != hidden.dtype
    ):
        raise RouteCaptureError("captured native route tuple shape/dtype mismatch")
    reconstructed = torch.zeros_like(hidden)
    with torch.inference_mode():
        for expert_id in range(len(moe.experts)):
            token_idx, slot_idx = torch.where(experts == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_output = _first_tensor(moe.experts[expert_id](hidden[token_idx]))
            if expert_output.shape != hidden[token_idx].shape:
                raise RouteCaptureError("native expert output shape mismatch")
            weighted = expert_output * weights[token_idx, slot_idx].to(
                dtype=expert_output.dtype
            ).unsqueeze(-1)
            reconstructed.index_add_(0, token_idx, weighted.to(reconstructed.dtype))
    return reconstructed.reshape(hidden_states.shape), experts, weights


def validate_native_moe_output_parity(
    native_output: Any,
    reconstructed_output: Any,
    *,
    tolerance_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless independent expert reconstruction matches native MoE."""

    import torch

    if (
        tuple(native_output.shape) != tuple(reconstructed_output.shape)
        or native_output.dtype != reconstructed_output.dtype
    ):
        raise RouteCaptureError("native/reconstructed MoE output shape or dtype mismatch")
    dtype_name = str(native_output.dtype)
    try:
        if (
            tolerance_rule["rule"] != "finfo_scaled_source_only"
            or tolerance_rule["atol_scale"]
            != "max(1,max_abs_native_layer_output)"
            or tolerance_rule["topk_indexes_must_match_exactly"] is not True
            or tolerance_rule["outcome_tuning_allowed"] is not False
        ):
            raise RouteCaptureError("native MoE parity tolerance rule drift")
        eps = float(torch.finfo(native_output.dtype).eps)
        rtol_multiplier = float(tolerance_rule["rtol_finfo_eps_multiplier"])
        atol_multiplier = float(tolerance_rule["atol_finfo_eps_multiplier"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RouteCaptureError("invalid frozen MoE parity tolerance") from exc
    if rtol_multiplier <= 0.0 or atol_multiplier <= 0.0:
        raise RouteCaptureError("negative MoE parity tolerance")
    native_float = native_output.float()
    reconstructed_float = reconstructed_output.float()
    max_abs_native = float(native_float.abs().max().item())
    rtol = rtol_multiplier * eps
    atol = atol_multiplier * eps * max(1.0, max_abs_native)
    absolute = (native_float - reconstructed_float).abs()
    denominator = native_float.abs().clamp_min(max(atol, 1e-12))
    parity = bool(
        torch.allclose(
            native_output,
            reconstructed_output,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
    )
    evidence = {
        "dtype": dtype_name,
        "shape": [int(value) for value in native_output.shape],
        "atol": atol,
        "rtol": rtol,
        "finfo_eps": eps,
        "max_abs_native": max_abs_native,
        "tolerance_rule": dict(tolerance_rule),
        "native_output_sha256": _tensor_sha256(native_output),
        "independent_reconstruction_sha256": _tensor_sha256(reconstructed_output),
        "hash_equal": _tensor_sha256(native_output)
        == _tensor_sha256(reconstructed_output),
        "max_abs_error": float(absolute.max().item()),
        "max_relative_error": float((absolute / denominator).max().item()),
        "within_frozen_tolerance": parity,
    }
    if not parity:
        raise RouteCaptureError(f"independent native MoE output parity failed: {evidence}")
    return evidence


def _routes_from_logits(
    logits: Any,
    *,
    top_k: int,
    normalize_topk: bool,
    selection_rule: str,
    output_dtype: Any,
) -> tuple[Any, Any]:
    import torch

    if selection_rule != NATIVE_TOPK_SELECTION_RULE:
        raise RouteCaptureError("native top-k selection rule drift")
    flattened = logits.reshape(-1, logits.shape[-1])
    probabilities = torch.softmax(flattened, dim=-1, dtype=torch.float32)
    weights, experts = torch.topk(probabilities, top_k, dim=-1)
    return experts, _effective_route_weights(
        weights,
        normalize_topk=normalize_topk,
        output_dtype=output_dtype,
    )


def _extract_router_logits(outputs: Any) -> list[Any]:
    value = getattr(outputs, "router_logits", None)
    if value is None and isinstance(outputs, Mapping):
        value = outputs.get("router_logits")
    if value is None:
        raise RouteCaptureError("native output_router_logits returned no router_logits")
    if hasattr(value, "shape"):
        return [value]
    if not isinstance(value, (tuple, list)):
        raise RouteCaptureError("native router_logits has unknown container")
    tensors = [item for item in value if hasattr(item, "shape")]
    if not tensors:
        raise RouteCaptureError("native router_logits container has no tensors")
    return tensors


def _gpu_compute_apps() -> list[dict[str, Any]]:
    """Capture the exact GPU-0 compute-process census or fail closed."""

    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RouteCaptureError("BLOCKED_GPU_ENVIRONMENT: compute-app query failed") from exc
    if not output or "No running processes found" in output:
        return []
    rows = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            raise RouteCaptureError(
                "BLOCKED_GPU_ENVIRONMENT: malformed compute-app query"
            )
        try:
            pid = int(fields[0])
            used_memory_mib = float(fields[3])
        except ValueError as exc:
            raise RouteCaptureError(
                "BLOCKED_GPU_ENVIRONMENT: invalid compute-app values"
            ) from exc
        rows.append(
            {
                "pid": pid,
                "gpu_uuid": fields[1],
                "process_name": fields[2],
                "used_memory_mib": used_memory_mib,
            }
        )
    return sorted(rows, key=lambda row: (row["pid"], row["process_name"]))


def _gpu_environment(
    *, compute_apps_before: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    import torch

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,clocks.sm,power.draw,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RouteCaptureError("BLOCKED_GPU_ENVIRONMENT: GPU query failed") from exc
    fields = [item.strip() for item in query.split(",")]
    if len(fields) != 7 or any(not value for value in fields):
        raise RouteCaptureError("BLOCKED_GPU_ENVIRONMENT: malformed GPU query")
    try:
        clock_sm_mhz = float(fields[3])
        power_draw_w = float(fields[4])
        memory_used_mib = float(fields[5])
        utilization = float(fields[6])
    except ValueError as exc:
        raise RouteCaptureError("BLOCKED_GPU_ENVIRONMENT: invalid GPU telemetry") from exc
    torch_name = torch.cuda.get_device_name(0)
    if torch_name != fields[1] or torch.version.cuda is None:
        raise RouteCaptureError("BLOCKED_GPU_ENVIRONMENT: CUDA device identity drift")
    return {
        "producer_pid": os.getpid(),
        "gpu_uuid": fields[0],
        "gpu_name": fields[1],
        "driver_version": fields[2],
        "clock_sm_mhz": clock_sm_mhz,
        "power_draw_w": power_draw_w,
        "memory_used_mib": memory_used_mib,
        "background_gpu_util_percent": utilization,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "compute_apps_before": [dict(row) for row in compute_apps_before],
        "compute_apps_after": _gpu_compute_apps(),
    }


def model_load_reference(
    spec: Mapping[str, Any], model_path: Path | None
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Resolve a frozen HF revision or an explicit offline model directory."""

    if model_path is not None:
        resolved = model_path.resolve()
        if not resolved.is_dir():
            raise RouteCaptureError(f"local model directory does not exist: {resolved}")
        files = []
        for path in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
            relative = str(path.relative_to(resolved))
            before = path.stat()
            file_sha256 = sha256_file(path)
            after = path.stat()
            stable_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            stable_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if stable_before != stable_after:
                raise RouteCaptureError(f"local model file changed while hashing: {relative}")
            row: dict[str, Any] = {
                "path": relative,
                "size_bytes": after.st_size,
                "sha256": file_sha256,
            }
            files.append(row)
        if not files:
            raise RouteCaptureError("local model directory is empty")
        manifest = {
            "kind": "explicit_local_directory",
            "path": str(resolved),
            "frozen_repo_id": spec["repo_id"],
            "frozen_revision": spec["revision"],
            "file_count": len(files),
            "tree_manifest_sha256": sha256_bytes(canonical_json_bytes(files)),
            "files": files,
        }
        expected_tree_sha256 = spec.get(
            "expected_local_model_tree_manifest_sha256"
        )
        if (
            not isinstance(expected_tree_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_tree_sha256)
        ):
            raise RouteCaptureError(
                "explicit local model spec lacks a canonical expected tree hash"
            )
        if manifest["tree_manifest_sha256"] != expected_tree_sha256:
            raise RouteCaptureError(
                "explicit local model tree hash differs from frozen config: "
                f"actual={manifest['tree_manifest_sha256']}, "
                f"expected={expected_tree_sha256}"
            )
        manifest["expected_tree_manifest_sha256"] = expected_tree_sha256
        # No revision/cache lookup is allowed when the operator supplied a
        # complete offline directory.
        return str(resolved), {"local_files_only": True}, manifest
    manifest = {
        "kind": "huggingface_frozen_revision",
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
    }
    return str(spec["repo_id"]), {"revision": spec["revision"]}, manifest


def _load_model_and_tokenizer(
    spec: Mapping[str, Any],
    *,
    cache_dir: Path | None,
    allow_download: bool,
    model_path: Path | None = None,
    model_reference: tuple[str, dict[str, Any], dict[str, Any]] | None = None,
) -> tuple[Any, Any, str, Mapping[str, Any]]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment capability
        raise RouteCaptureError("torch and transformers are required") from exc
    source, common, source_manifest = (
        model_reference if model_reference is not None else model_load_reference(spec, model_path)
    )
    common = dict(common)
    if model_path is None:
        common["cache_dir"] = str(cache_dir) if cache_dir else None
        common["local_files_only"] = not allow_download
    tokenizer = AutoTokenizer.from_pretrained(source, **common)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        **common,
    )
    model.eval()
    model.config.output_router_logits = True
    return model, tokenizer, transformers.__version__, source_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("olmoe", "llmjp"), required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment capability
        raise RouteCaptureError("CUDA PyTorch is required") from exc
    if not torch.cuda.is_available():
        raise RouteCaptureError("CUDA is required; proxy route fallback is forbidden")
    if args.output_dir.exists():
        raise RouteCaptureError("refusing to overwrite route output directory")
    compute_apps_before = _gpu_compute_apps()
    config = _load_config(args.config)
    protocol_sha = sha256_file(args.protocol)
    config_sha = sha256_file(args.config)
    manifest = _load_data_manifest(
        args.data_manifest,
        mode=args.mode,
        model_key=args.model_key,
        config=config,
        protocol_sha256=protocol_sha,
        config_sha256=config_sha,
    )
    source_sha = _producer_source_sha256()
    manifest_sha = str(manifest["manifest_sha256"])
    data_producer_signoff_sha = manifest.get("signoff_sha256")
    spec = config["models"][args.model_key]
    if args.mode == "formal" and args.model_path is None:
        raise RouteCaptureError("formal capture requires an explicit hashed --model-path")
    model_reference = model_load_reference(spec, args.model_path)
    model_tree_sha = model_reference[2].get("tree_manifest_sha256")
    signoff = None
    if args.mode == "formal":
        if not is_sha256(data_producer_signoff_sha):
            raise RouteCaptureError("formal data producer signoff hash is missing")
        if not isinstance(model_tree_sha, str):
            raise RouteCaptureError("formal local model tree has no manifest hash")
        signoff = _require_formal_signoff(
            args.signoff,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
            source_sha256=source_sha,
            data_manifest_sha256=manifest_sha,
            data_producer_signoff_sha256=str(data_producer_signoff_sha),
            model_key=args.model_key,
            model_tree_manifest_sha256=model_tree_sha,
        )

    model_revision = f"{spec['repo_id']}@{spec['revision']}"
    model, tokenizer, transformers_version, model_source = _load_model_and_tokenizer(
        spec,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
        model_path=args.model_path,
        model_reference=model_reference,
    )
    modules = discover_moe_modules(model)
    expected_top_k = int(spec["top_k"])
    expected_experts = int(spec["num_experts"])
    census = validate_model_config_layer_census(
        model.config,
        modules,
        expected_num_experts=expected_experts,
        expected_top_k=expected_top_k,
    )
    native_implementation = validate_native_moe_implementation(
        modules,
        model_spec=spec,
        route_config=config["route_capture"],
    )
    layer_ids = list(census["expected_layers"])
    frozen_layers = selected_layers(
        layer_ids,
        selection_seed=int(config["data"]["selection_seed"]),
        model_revision=model_revision,
        count=int(config["route_capture"]["selected_layer_count_per_model"]),
    )
    normalize_by_layer = {layer: _normalizes_topk(module) for layer, _, module in modules}
    module_by_layer = {layer: module for layer, _, module in modules}
    tolerance_rule = config["route_capture"].get("native_moe_output_tolerance")
    if not isinstance(tolerance_rule, Mapping):
        raise RouteCaptureError("config lacks native MoE output parity tolerance")
    gate_outputs: dict[int, Any] = {}
    moe_inputs: dict[int, Any] = {}
    moe_outputs: dict[int, Any] = {}
    active_native_moe: dict[str, int | None] = {"value": None}
    native_topk_capture = make_native_topk_capture_mode(
        active_layer=active_native_moe,
        expected_num_experts=expected_experts,
        expected_top_k=expected_top_k,
        expected_tokens=int(config["data"]["sequence_length"]),
    )
    handles = []

    def make_gate_hook(layer_id: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if layer_id in gate_outputs:
                raise RouteCaptureError(f"native gate called twice in layer {layer_id}")
            if not hasattr(output, "shape"):
                raise RouteCaptureError("native gate hook emitted non-tensor")
            gate_outputs[layer_id] = output.detach()

        return hook

    def make_moe_hook(layer_id: int):
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if layer_id in moe_inputs or layer_id in moe_outputs:
                raise RouteCaptureError(f"native MoE called twice in layer {layer_id}")
            moe_inputs[layer_id] = _first_tensor(inputs).detach()
            moe_outputs[layer_id] = _first_tensor(output).detach()

        return hook

    def make_moe_pre_hook(layer_id: int):
        def hook(_module: Any, _inputs: tuple[Any, ...]) -> None:
            if active_native_moe["value"] is not None:
                raise RouteCaptureError("nested native MoE execution is unsupported")
            active_native_moe["value"] = layer_id

        return hook

    def make_moe_clear_hook(layer_id: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
            if active_native_moe["value"] != layer_id:
                raise RouteCaptureError("native MoE active-layer identity drift")
            active_native_moe["value"] = None

        return hook

    for layer_id, _name, module in modules:
        handles.append(module.register_forward_pre_hook(make_moe_pre_hook(layer_id)))
        handles.append(module.gate.register_forward_hook(make_gate_hook(layer_id)))
        handles.append(module.register_forward_hook(make_moe_hook(layer_id)))
        handles.append(module.register_forward_hook(make_moe_clear_hook(layer_id)))

    requests = manifest["requests"]
    assert isinstance(requests, list)
    ep_size = int(config["topology_proxy"]["ep_size"])
    request_to_receiver = origin_lpt(requests, ep_size)
    expert_to_sender = {
        str(expert_id): expert_sender(expert_id, expected_experts, ep_size)
        for expert_id in range(expected_experts)
    }
    placement = add_self_hash(
        {
            "schema_version": "ric-placement-v1",
            "model_revision": model_revision,
            "ep_size": ep_size,
            "virtual_nodes": int(config["topology_proxy"]["virtual_nodes"]),
            "ranks_per_node": int(config["topology_proxy"]["ranks_per_node"]),
            "expert_placement": "contiguous",
            "request_origin": "route_blind_token_count_lpt",
            "expert_to_sender": expert_to_sender,
            "request_to_receiver": request_to_receiver,
        }
    )

    parity_rows: list[dict[str, Any]] = []
    route_rows = 0
    join_sets = 0
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{args.output_dir.name}.partial-", dir=args.output_dir.parent
        ) as temporary_directory:
            temporary = Path(temporary_directory)
            trace_path = temporary / "route_trace.jsonl"
            with trace_path.open("x", encoding="utf-8") as trace:
                for request_index, request in enumerate(requests):
                    request_id = str(request["request_id"])
                    text = str(request["text"])
                    if sha256_bytes(text.encode("utf-8")) != request.get("text_sha256"):
                        raise RouteCaptureError(f"text hash mismatch for {request_id}")
                    encoded = tokenizer(
                        text,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                    full_input_ids = encoded["input_ids"]
                    if len(full_input_ids.shape) != 2 or int(full_input_ids.shape[0]) != 1:
                        raise RouteCaptureError("tokenizer did not return one full sequence")
                    validate_full_tokenizer_length(
                        request,
                        model_key=args.model_key,
                        observed_length=int(full_input_ids.shape[1]),
                        minimum_length=int(
                            config["data"]["min_tokens_both_frozen_tokenizers"]
                        ),
                    )
                    input_ids = full_input_ids[
                        :, : int(config["data"]["sequence_length"])
                    ].contiguous()
                    if tuple(input_ids.shape) != (1, int(config["data"]["sequence_length"])):
                        raise RouteCaptureError(f"request {request_id} is not exactly 128 tokens")
                    gate_outputs.clear()
                    moe_inputs.clear()
                    moe_outputs.clear()
                    native_topk_capture.calls.clear()
                    with torch.inference_mode(), native_topk_capture:
                        outputs = model(
                            input_ids=input_ids.to("cuda:0"),
                            use_cache=False,
                            output_router_logits=True,
                            return_dict=True,
                        )
                    torch.cuda.synchronize()
                    native_outputs = _extract_router_logits(outputs)
                    if sorted(gate_outputs) != layer_ids:
                        raise RouteCaptureError("gate-hook layer census is incomplete")
                    if sorted(moe_inputs) != layer_ids or sorted(moe_outputs) != layer_ids:
                        raise RouteCaptureError(
                            "native MoE input/output layer census is incomplete"
                        )
                    if active_native_moe["value"] is not None:
                        raise RouteCaptureError("native MoE observer did not close")
                    if sorted(native_topk_capture.calls) != layer_ids:
                        raise RouteCaptureError(
                            "native aten.topk capture layer census is incomplete"
                        )
                    if len(native_outputs) != len(layer_ids):
                        raise RouteCaptureError(
                            "output_router_logits count differs from independent gate census"
                        )
                    forward_id = f"{request_id}:prefill:0"
                    batch_id = f"batch:{request_id}:prefill:0"
                    request_assigned_layer = assigned_layer(request_id, frozen_layers)
                    for route_event_index, (layer_id, native_logits) in enumerate(
                        zip(layer_ids, native_outputs)
                    ):
                        gate_logits = gate_outputs[layer_id]
                        raw_router_identity = validate_raw_router_tensor_identity(
                            gate_logits,
                            native_logits,
                            expected_shape=(
                                int(config["data"]["sequence_length"]),
                                expected_experts,
                            ),
                        )
                        gate_flat = gate_logits.reshape(-1, gate_logits.shape[-1])
                        native_flat = native_logits.reshape(-1, native_logits.shape[-1])
                        if gate_flat.shape != native_flat.shape or tuple(gate_flat.shape) != (
                            int(config["data"]["sequence_length"]),
                            expected_experts,
                        ):
                            raise RouteCaptureError("native/gate router shape mismatch")
                        replay_probabilities = torch.softmax(
                            gate_flat, dim=-1, dtype=torch.float32
                        )
                        replay_precast_weights, replay_experts = torch.topk(
                            replay_probabilities, expected_top_k, dim=-1
                        )
                        if str(config["route_capture"]["native_topk_selection_rule"]) != (
                            NATIVE_TOPK_SELECTION_RULE
                        ):
                            raise RouteCaptureError("native top-k selection rule drift")
                        replay_effective_weights = _effective_route_weights(
                            replay_precast_weights,
                            normalize_topk=normalize_by_layer[layer_id],
                            output_dtype=moe_inputs[layer_id].dtype,
                        )
                        native_precast_weights, native_experts = (
                            native_topk_capture.calls[layer_id]
                        )
                        native_effective_weights = _effective_route_weights(
                            native_precast_weights,
                            normalize_topk=normalize_by_layer[layer_id],
                            output_dtype=moe_inputs[layer_id].dtype,
                        )
                        native_fp32_precast_weights = _precast_route_weights(
                            native_precast_weights,
                            normalize_topk=normalize_by_layer[layer_id],
                        )
                        gate_experts, gate_weights = _routes_from_logits(
                            gate_flat,
                            top_k=expected_top_k,
                            normalize_topk=normalize_by_layer[layer_id],
                            selection_rule=str(
                                config["route_capture"]["native_topk_selection_rule"]
                            ),
                            output_dtype=moe_inputs[layer_id].dtype,
                        )
                        reconstructed, reconstructed_experts, reconstructed_weights = (
                            reconstruct_native_moe_output(
                                moe=module_by_layer[layer_id],
                                hidden_states=moe_inputs[layer_id],
                                selected_experts=native_experts,
                                effective_weights=native_effective_weights,
                            )
                        )
                        exact_experts = bool(
                            torch.equal(replay_experts, native_experts)
                            and torch.equal(gate_experts, native_experts)
                            and torch.equal(reconstructed_experts, native_experts)
                        )
                        exact_precast_weights = bool(
                            torch.equal(replay_precast_weights, native_precast_weights)
                        )
                        exact_effective_weights = bool(
                            torch.equal(replay_effective_weights, native_effective_weights)
                            and torch.equal(gate_weights, native_effective_weights)
                            and torch.equal(
                                reconstructed_weights, native_effective_weights
                            )
                        )
                        native_route_tuple_sha256 = _route_tuple_sha256(
                            native_experts, native_effective_weights
                        )
                        replay_route_tuple_sha256 = _route_tuple_sha256(
                            replay_experts, replay_effective_weights
                        )
                        route_tuple_hash_equal = (
                            native_route_tuple_sha256 == replay_route_tuple_sha256
                        )
                        output_parity = validate_native_moe_output_parity(
                            moe_outputs[layer_id],
                            reconstructed,
                            tolerance_rule=tolerance_rule,
                        )
                        max_logit_abs = float(
                            (gate_flat.float() - native_flat.float()).abs().max().item()
                        )
                        raw_logit_hash_equal = bool(
                            raw_router_identity["raw_logit_hash_equal"]
                        )
                        selected_tie_tokens = int(
                            (
                                native_precast_weights.unsqueeze(-1)
                                == native_precast_weights.unsqueeze(-2)
                            )
                            .sum(dim=(-1, -2))
                            .gt(expected_top_k)
                            .sum()
                            .item()
                        )
                        boundary = native_precast_weights[:, -1]
                        boundary_tie_tokens = int(
                            (
                                (replay_probabilities == boundary.unsqueeze(-1)).sum(dim=-1)
                                > (native_precast_weights == boundary.unsqueeze(-1)).sum(dim=-1)
                            )
                            .sum()
                            .item()
                        )
                        parity = {
                            "request_id": request_id,
                            "forward_id": forward_id,
                            "layer_id": layer_id,
                            **raw_router_identity,
                            "framework_router_container_consistency_only": True,
                            "topk_expert_exact_native_capture": exact_experts,
                            "topk_precast_weight_exact_native_capture": (
                                exact_precast_weights
                            ),
                            "topk_effective_weight_exact_native_capture": (
                                exact_effective_weights
                            ),
                            "native_topk_indices_sha256": _tensor_sha256(native_experts),
                            "native_topk_precast_values_sha256": _tensor_sha256(
                                native_precast_weights
                            ),
                            "native_effective_weights_sha256": _tensor_sha256(
                                native_effective_weights
                            ),
                            "native_route_tuple_sha256": native_route_tuple_sha256,
                            "replay_route_tuple_sha256": replay_route_tuple_sha256,
                            "route_tuple_hash_equal": route_tuple_hash_equal,
                            "effective_route_weight_dtype": str(
                                native_effective_weights.dtype
                            ),
                            "selected_weight_tie_token_count": selected_tie_tokens,
                            "selection_boundary_tie_token_count": boundary_tie_tokens,
                            "max_logit_abs_error": max_logit_abs,
                            **output_parity,
                        }
                        parity_rows.append(parity)
                        if not raw_logit_hash_equal or max_logit_abs != 0.0:
                            raise RouteCaptureError(
                                "framework output_router_logits differs from raw gate hook"
                            )
                        if not (
                            exact_experts
                            and exact_precast_weights
                            and exact_effective_weights
                            and route_tuple_hash_equal
                        ):
                            raise RouteCaptureError(
                                "exact native top-k capture/replay route tuple differs"
                            )
                        selected = native_experts.detach().cpu()
                        weights = native_effective_weights.detach().cpu()
                        for token_position in range(int(config["data"]["sequence_length"])):
                            expert_ids = [int(value) for value in selected[token_position].tolist()]
                            if len(set(expert_ids)) != expected_top_k:
                                raise RouteCaptureError("duplicate expert in one top-k join set")
                            token_id = f"{request_id}:token:{token_position:03d}"
                            token_block_id = token_id
                            for topk_slot, expert_id in enumerate(expert_ids):
                                row = {
                                    "schema_version": "ric-route-v1",
                                    "model_key": args.model_key,
                                    "model_revision": model_revision,
                                    "data_manifest_sha256": manifest_sha,
                                    "placement_manifest_sha256": placement["manifest_sha256"],
                                    "request_id": request_id,
                                    "forward_id": forward_id,
                                    "batch_id": batch_id,
                                    "phase": "prefill",
                                    "decode_step": 0,
                                    "layer_id": layer_id,
                                    "token_id": token_id,
                                    "token_block_id": token_block_id,
                                    "token_position": token_position,
                                    "topk_slot": topk_slot,
                                    "expert_id": expert_id,
                                    "sender_rank": expert_sender(
                                        expert_id, expected_experts, ep_size
                                    ),
                                    "receiver_rank": request_to_receiver[request_id],
                                    "epoch": ROUTE_EPOCH,
                                    "valid": True,
                                    "route_weight": float(weights[token_position, topk_slot]),
                                    "route_weight_dtype": str(
                                        native_effective_weights.dtype
                                    ),
                                    "route_weight_fp32_precast": float(
                                        native_fp32_precast_weights[
                                            token_position, topk_slot
                                        ]
                                    ),
                                    "route_event_index": route_event_index,
                                    "selected_for_replay": layer_id
                                    == request_assigned_layer,
                                    "route_source": (
                                        "native_aten_topk_capture_plus_independent_moe_output_parity"
                                    ),
                                }
                                trace.write(json.dumps(row, sort_keys=True) + "\n")
                                route_rows += 1
                            join_sets += 1
            route_trace_sha = sha256_file(trace_path)
            parity = add_self_hash(
                {
                    "schema_version": "ric-route-parity-v1",
                    "status": "CAPTURE_ONLY" if args.mode == "formal" else "NOT_TESTED",
                    "scientific_result": False,
                    "model_key": args.model_key,
                    "model_revision": model_revision,
                    **census,
                    "selected_layers": frozen_layers,
                    **native_implementation,
                    "parity_rows": parity_rows,
                    "all_topk_exact": all(
                        row["topk_expert_exact_native_capture"]
                        for row in parity_rows
                    ),
                    "all_route_weights_exact": all(
                        row["topk_precast_weight_exact_native_capture"]
                        and row["topk_effective_weight_exact_native_capture"]
                        and row["route_tuple_hash_equal"]
                        for row in parity_rows
                    ),
                    "all_native_moe_outputs_within_frozen_tolerance": all(
                        row["within_frozen_tolerance"] for row in parity_rows
                    ),
                    "native_moe_output_tolerance": tolerance_rule,
                    "max_logit_abs_error": max(
                        float(row["max_logit_abs_error"]) for row in parity_rows
                    ),
                    "max_moe_output_abs_error": max(
                        float(row["max_abs_error"]) for row in parity_rows
                    ),
                }
            )
            (temporary / "route_parity.json").write_text(
                json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (temporary / "placement.json").write_text(
                json.dumps(placement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            embedded_signoff_sha256 = (
                materialize_verified_signoff(args.signoff, temporary)
                if signoff is not None
                else None
            )
            metadata = add_self_hash(
                {
                    "schema_version": "ric-route-capture-v1",
                    "status": "CAPTURE_ONLY" if args.mode == "formal" else "NOT_TESTED",
                    "scientific_result": False,
                    "evidence_boundary": (
                        "ROUTE_REAL_NATIVE_FORWARD / NO_READY_OR_NETWORK_TIMESTAMPS / NOT_RDMA"
                    ),
                    "mode": args.mode,
                    "data_role": manifest["role"],
                    "model_key": args.model_key,
                    "model_revision": model_revision,
                    "transformers_version": transformers_version,
                    "model_source": model_source,
                    "model_tree_manifest_sha256": model_tree_sha,
                    "protocol_sha256": protocol_sha,
                    "config_sha256": config_sha,
                    "capture_routes_source_sha256": source_sha,
                    "data_manifest_sha256": manifest_sha,
                    "data_producer_signoff_sha256": data_producer_signoff_sha,
                    "placement_manifest_sha256": placement["manifest_sha256"],
                    "route_trace_sha256": route_trace_sha,
                    "route_parity_sha256": parity["manifest_sha256"],
                    "route_rows": route_rows,
                    "join_sets": join_sets,
                    "request_count": len(requests),
                    "expected_layers": list(census["expected_layers"]),
                    "expected_layer_source": census["expected_layer_source"],
                    "selected_layers": frozen_layers,
                    "top_k": expected_top_k,
                    "num_experts": expected_experts,
                    "gpu_environment": _gpu_environment(
                        compute_apps_before=compute_apps_before
                    ),
                    "signoff_sha256": embedded_signoff_sha256,
                }
            )
            (temporary / "capture_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.rename(args.output_dir)
    finally:
        for handle in handles:
            handle.remove()
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
