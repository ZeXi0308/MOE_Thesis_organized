#!/usr/bin/env python3
"""Pure native-MoE route observation primitives for RIC-Clean-v2.

This module deliberately contains no CLI, file-system provenance, formal
signoff, or experiment-result logic.  It is the reviewed mechanism layer used
by the clean calibration route producer.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from typing import Any, Mapping, Sequence


NATIVE_TOPK_SELECTION_RULE = (
    "torch.topk(torch.softmax(raw_gate_logits,dim=-1,dtype=torch.float32),"
    "k,dim=-1);preserve_returned_slot_order"
)


class NativeRouteError(RuntimeError):
    """A native-route semantic or parity invariant failed."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def frozen_concat_sha256(*parts: object) -> str:
    return sha256_bytes("".join(str(part) for part in parts).encode("utf-8"))


def origin_lpt(requests: Sequence[Mapping[str, Any]], ep_size: int) -> dict[str, int]:
    """Frozen route-blind equal-token LPT placement with request/rank tie-break."""

    if ep_size < 1:
        raise NativeRouteError("ep_size must be positive")
    loads = [0] * ep_size
    result: dict[str, int] = {}
    weighted = [
        (str(row["request_id"]), sha256_bytes(str(row["request_id"]).encode("utf-8")))
        for row in requests
    ]
    if len({request_id for request_id, _digest in weighted}) != len(weighted):
        raise NativeRouteError("request identities are duplicated")
    for request_id, _digest in sorted(weighted, key=lambda item: item[1]):
        rank = min(
            range(ep_size),
            key=lambda candidate: (
                loads[candidate],
                sha256_bytes(f"{request_id}{candidate}".encode("utf-8")),
                candidate,
            ),
        )
        result[request_id] = rank
        loads[rank] += 128
    return result


def expert_sender(expert_id: int, num_experts: int, ep_size: int) -> int:
    if ep_size < 1 or num_experts < 1 or not 0 <= expert_id < num_experts:
        raise NativeRouteError("expert placement is outside the frozen domain")
    return min(ep_size - 1, expert_id * ep_size // num_experts)


def selected_layers(
    layer_ids: Sequence[int], *, selection_seed: int, model_revision: str, count: int
) -> list[int]:
    if len(set(layer_ids)) != len(layer_ids):
        raise NativeRouteError("duplicate discovered layer ids")
    ranked = sorted(
        layer_ids,
        key=lambda layer_id: frozen_concat_sha256(
            selection_seed, model_revision, layer_id
        ),
    )
    if count < 1 or len(ranked) < count:
        raise NativeRouteError("model exposes fewer MoE layers than frozen selection")
    return sorted(int(value) for value in ranked[:count])


def assigned_layer(request_id: str, frozen_layers: Sequence[int]) -> int:
    if not frozen_layers:
        raise NativeRouteError("no frozen layers")
    return int(frozen_layers[int(sha256_bytes(request_id.encode("utf-8")), 16) % len(frozen_layers)])


def _layer_index(name: str) -> int:
    matches = re.findall(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if len(matches) != 1:
        raise NativeRouteError(f"cannot derive one decoder layer id from {name!r}")
    return int(matches[0])


def discover_moe_modules(model: Any) -> list[tuple[int, str, Any]]:
    discovered: list[tuple[int, str, Any]] = []
    seen_layers: set[int] = set()
    for name, module in model.named_modules():
        if not hasattr(module, "gate") or not hasattr(module, "experts"):
            continue
        try:
            expert_count = len(module.experts)
        except TypeError:
            continue
        if expert_count < 2:
            continue
        layer_id = _layer_index(name)
        if layer_id in seen_layers:
            raise NativeRouteError(f"multiple MoE modules in decoder layer {layer_id}")
        seen_layers.add(layer_id)
        discovered.append((layer_id, name, module))
    if not discovered:
        raise NativeRouteError("no native MoE gate+experts modules discovered")
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
                raise NativeRouteError(f"model config {name} is not an integer") from exc
    return values


def validate_model_config_layer_census(
    model_config: Any,
    modules: Sequence[tuple[int, str, Any]],
    *,
    expected_num_experts: int,
    expected_top_k: int,
) -> dict[str, Any]:
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
        raise NativeRouteError("model config lacks one exact num_hidden_layers")
    if expert_values != {int(expected_num_experts)}:
        raise NativeRouteError("model-config expert count differs from frozen spec")
    if topk_values != {int(expected_top_k)}:
        raise NativeRouteError("model-config top-k differs from frozen spec")
    num_hidden_layers = next(iter(layer_values))
    expected_layers = list(range(num_hidden_layers))
    discovered_layers = [int(layer) for layer, _name, _module in modules]
    if discovered_layers != expected_layers:
        missing = sorted(set(expected_layers) - set(discovered_layers))
        extra = sorted(set(discovered_layers) - set(expected_layers))
        raise NativeRouteError(
            "native MoE layer census differs from model config: "
            f"missing={missing}, extra={extra}"
        )
    if any(len(module.experts) != expected_num_experts for _, _, module in modules):
        raise NativeRouteError("native expert count differs from frozen model spec")
    return {
        "expected_layer_source": "model_config.num_hidden_layers_all_layers_are_moe",
        "num_hidden_layers": num_hidden_layers,
        "expected_layers": expected_layers,
        "model_config_num_experts": expected_num_experts,
        "model_config_top_k": expected_top_k,
    }


def normalizes_topk(moe: Any) -> bool:
    if hasattr(moe, "norm_topk_prob"):
        return bool(moe.norm_topk_prob)
    if "mixtral" in type(moe).__name__.lower():
        return True
    raise NativeRouteError(f"unknown native top-k normalization for {type(moe).__name__}")


def validate_native_moe_implementation(
    modules: Sequence[tuple[int, str, Any]],
    *,
    model_spec: Mapping[str, Any],
    route_config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_contract = {
        "native_topk_selection_rule": NATIVE_TOPK_SELECTION_RULE,
        "native_topk_dispatch_capture_required": True,
        "route_rows_use_captured_native_topk": True,
        "effective_route_weight": (
            "topk_fp32_then_optional_renorm_then_cast_native_hidden_dtype"
        ),
        "stable_sort_substitution_allowed": False,
    }
    for field, expected in expected_contract.items():
        if route_config.get(field) != expected:
            raise NativeRouteError(f"native route contract drift: {field}")
    expected_class = model_spec.get("native_moe_class")
    expected_source = model_spec.get("native_moe_forward_source_sha256")
    if not isinstance(expected_class, str) or not isinstance(expected_source, str):
        raise NativeRouteError("model spec lacks frozen native MoE implementation")
    classes: set[str] = set()
    source_hashes: set[str] = set()
    for _layer, _name, module in modules:
        classes.add(f"{type(module).__module__}.{type(module).__name__}")
        try:
            forward_source = inspect.getsource(type(module).forward)
        except (OSError, TypeError) as exc:
            raise NativeRouteError("cannot inspect native MoE forward source") from exc
        source_hashes.add(sha256_bytes(forward_source.encode("utf-8")))
        if getattr(module, "top_k", None) != int(model_spec["top_k"]):
            raise NativeRouteError("native MoE module top-k differs from frozen spec")
    if classes != {expected_class} or source_hashes != {expected_source}:
        raise NativeRouteError("native MoE implementation differs from frozen spec")
    return {**expected_contract, "native_moe_class": expected_class, "native_moe_forward_source_sha256": expected_source}


def tensor_sha256(tensor: Any) -> str:
    import torch

    original = tensor.detach().cpu()
    contiguous = original.contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(str(tuple(original.shape)).encode())
    digest.update(b"\0")
    digest.update(str(original.dtype).encode())
    digest.update(b"\0")
    digest.update(str(tuple(original.stride())).encode())
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def route_tuple_sha256(experts: Any, effective_weights: Any) -> str:
    import torch

    if experts.shape != effective_weights.shape or experts.dtype != torch.int64:
        raise NativeRouteError("route tuple expert/weight shape or dtype mismatch")
    if not bool(torch.isfinite(effective_weights).all().item()):
        raise NativeRouteError("route tuple contains non-finite weight")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema": "ric-clean-v2-native-route-tuple-v1",
                "experts_sha256": tensor_sha256(experts),
                "effective_weights_sha256": tensor_sha256(effective_weights),
            }
        )
    )


def precast_route_weights(topk_values: Any, *, normalize_topk: bool) -> Any:
    weights = topk_values
    if normalize_topk:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights


def effective_route_weights(
    topk_values: Any, *, normalize_topk: bool, output_dtype: Any
) -> Any:
    return precast_route_weights(topk_values, normalize_topk=normalize_topk).to(
        dtype=output_dtype
    )


def routes_from_logits(
    router_logits: Any,
    *,
    top_k: int,
    normalize_topk: bool,
    selection_rule: str,
    output_dtype: Any,
) -> tuple[Any, Any]:
    import torch

    if selection_rule != NATIVE_TOPK_SELECTION_RULE:
        raise NativeRouteError("native selection rule drift")
    flattened = router_logits.reshape(-1, router_logits.shape[-1])
    probabilities = torch.softmax(flattened, dim=-1, dtype=torch.float32)
    values, experts = torch.topk(probabilities, top_k, dim=-1)
    return experts, effective_route_weights(
        values, normalize_topk=normalize_topk, output_dtype=output_dtype
    )


def validate_raw_router_tensor_identity(
    gate_logits: Any, output_router_logits: Any, *, expected_shape: tuple[int, int]
) -> dict[str, Any]:
    if (
        tuple(gate_logits.shape) != expected_shape
        or tuple(output_router_logits.shape) != expected_shape
        or gate_logits.dtype != output_router_logits.dtype
        or tuple(gate_logits.stride()) != tuple(output_router_logits.stride())
    ):
        raise NativeRouteError("native/gate raw router shape, dtype, or stride mismatch")
    gate_hash = tensor_sha256(gate_logits)
    output_hash = tensor_sha256(output_router_logits)
    if gate_hash != output_hash:
        raise NativeRouteError("framework output_router_logits differs from raw gate hook")
    return {
        "gate_logit_sha256": gate_hash,
        "output_router_logit_sha256": output_hash,
        "raw_logit_hash_equal": True,
    }


def make_native_topk_capture_mode(
    *,
    active_layer: Mapping[str, int | None],
    expected_num_experts: int,
    expected_top_k: int,
    expected_tokens: int,
) -> Any:
    """Observe the actual native ``aten.topk`` result without changing it."""

    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class NativeTopkCapture(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.calls: dict[int, tuple[Any, Any]] = {}

        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            del types
            call_kwargs = {} if kwargs is None else dict(kwargs)
            result = func(*args, **call_kwargs)
            if func != torch.ops.aten.topk.default:
                return result
            layer_id = active_layer.get("value")
            if layer_id is None:
                return result
            values, indices = result
            source = args[0]
            k = int(args[1])
            dim = int(args[2]) if len(args) > 2 else int(call_kwargs.get("dim", -1))
            largest = bool(args[3]) if len(args) > 3 else bool(call_kwargs.get("largest", True))
            sorted_output = bool(args[4]) if len(args) > 4 else bool(call_kwargs.get("sorted", True))
            if (
                k != expected_top_k
                or dim not in {-1, source.ndim - 1}
                or tuple(source.shape) != (expected_tokens, expected_num_experts)
                or source.dtype != torch.float32
                or not largest
                or not sorted_output
                or tuple(values.shape) != (expected_tokens, expected_top_k)
                or indices.dtype != torch.int64
            ):
                raise NativeRouteError("native MoE aten.topk signature drift")
            if int(layer_id) in self.calls:
                raise NativeRouteError(f"native layer {layer_id} called aten.topk twice")
            self.calls[int(layer_id)] = (
                values.detach().clone(),
                indices.detach().clone(),
            )
            return result

    return NativeTopkCapture()


def first_tensor(value: Any) -> Any:
    if hasattr(value, "shape"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if hasattr(item, "shape"):
                return item
    raise NativeRouteError("native value contains no tensor")


def reconstruct_native_moe_output(
    *, moe: Any, hidden_states: Any, selected_experts: Any, effective_weights: Any
) -> tuple[Any, Any, Any]:
    import torch

    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    if selected_experts.shape != effective_weights.shape or selected_experts.shape[0] != hidden.shape[0]:
        raise NativeRouteError("captured native route tuple shape mismatch")
    output = torch.zeros_like(hidden)
    for expert_id in range(len(moe.experts)):
        token_idx, slot_idx = torch.where(selected_experts == expert_id)
        if token_idx.numel() == 0:
            continue
        expert_output = first_tensor(moe.experts[expert_id](hidden[token_idx]))
        if expert_output.shape != hidden[token_idx].shape:
            raise NativeRouteError("native expert output shape mismatch")
        output.index_add_(
            0,
            token_idx,
            expert_output * effective_weights[token_idx, slot_idx].unsqueeze(-1),
        )
    return output.reshape(hidden_states.shape), selected_experts, effective_weights


def validate_native_moe_output_parity(
    native_output: Any,
    reconstructed_output: Any,
    *,
    tolerance_rule: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    if native_output.shape != reconstructed_output.shape or native_output.dtype != reconstructed_output.dtype:
        raise NativeRouteError("native/reconstructed MoE output identity mismatch")
    if (
        tolerance_rule.get("rule") != "finfo_scaled_source_only"
        or tolerance_rule.get("topk_indexes_must_match_exactly") is not True
        or tolerance_rule.get("outcome_tuning_allowed") is not False
    ):
        raise NativeRouteError("native MoE parity tolerance rule drift")
    eps = float(torch.finfo(native_output.dtype).eps)
    rtol = eps * float(tolerance_rule["rtol_finfo_eps_multiplier"])
    native_float = native_output.float()
    reconstructed_float = reconstructed_output.float()
    scale = max(1.0, float(native_float.abs().max().item()))
    atol = eps * float(tolerance_rule["atol_finfo_eps_multiplier"]) * scale
    absolute = (native_float - reconstructed_float).abs()
    denominator = native_float.abs().clamp_min(atol)
    parity = bool(torch.all(absolute <= atol + rtol * native_float.abs()).item())
    evidence = {
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": float(absolute.max().item()),
        "max_relative_error": float((absolute / denominator).max().item()),
        "within_frozen_tolerance": parity,
    }
    if not parity:
        raise NativeRouteError(f"independent native MoE output parity failed: {evidence}")
    return evidence
