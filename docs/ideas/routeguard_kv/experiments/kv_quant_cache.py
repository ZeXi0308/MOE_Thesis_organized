#!/usr/bin/env python3
"""Dedicated decode-cache QDQ implementation for RouteGuard-KV R0-A."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, Iterable

import torch
from transformers.cache_utils import DynamicCache


class QuantizedCacheError(RuntimeError):
    pass


def asymmetric_int4_qdq(x: torch.Tensor) -> torch.Tensor:
    """Frozen per-head/per-token asymmetric INT4 QDQ over head_dim.

    Input is expected to have cache shape [batch, kv_head, token, head_dim].
    Min/max, scale, zero point, quantization and dequantization are FP32. A
    constant head vector is returned exactly, including nonzero constants.
    """

    if x.ndim != 4:
        raise QuantizedCacheError(f"expected rank-4 KV tensor, got {tuple(x.shape)}")
    original_dtype = x.dtype
    work = x.float()
    minimum = work.amin(dim=-1, keepdim=True)
    maximum = work.amax(dim=-1, keepdim=True)
    span = maximum - minimum
    constant = span == 0
    safe_scale = torch.where(constant, torch.ones_like(span), span / 15.0)
    zero = torch.clamp(torch.round(-minimum / safe_scale), 0, 15)
    quantized = torch.clamp(torch.round(work / safe_scale) + zero, 0, 15)
    dequantized = (quantized - zero) * safe_scale
    result = torch.where(constant, work, dequantized)
    return result.to(original_dtype)


def identity_qdq(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise QuantizedCacheError(f"expected rank-4 KV tensor, got {tuple(x.shape)}")
    return x.clone()


@dataclass
class QuantizerEvent:
    phase: str
    layer_idx: int
    target: str
    key_tokens: int
    value_tokens: int


@dataclass
class QuantizerLedger:
    events: list[QuantizerEvent] = field(default_factory=list)

    def record(
        self,
        *,
        phase: str,
        layer_idx: int,
        target: str,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        self.events.append(
            QuantizerEvent(
                phase=phase,
                layer_idx=layer_idx,
                target=target,
                key_tokens=int(key_states.shape[-2]),
                value_tokens=int(value_states.shape[-2]),
            )
        )


def _validate_target(target: str) -> None:
    if target not in {"identity", "k_only", "v_only", "kv"}:
        raise QuantizedCacheError(f"unknown cache quantization target: {target}")


def _transform_pair(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    target: str,
    quantizer: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_target(target)
    key_out = quantizer(key_states) if target in {"k_only", "kv"} else key_states.clone()
    value_out = quantizer(value_states) if target in {"v_only", "kv"} else value_states.clone()
    if key_out.dtype != key_states.dtype or value_out.dtype != value_states.dtype:
        raise QuantizedCacheError("quantizer promoted cache dtype")
    return key_out, value_out


class QuantizedDynamicCache(DynamicCache):
    """A DynamicCache that QDQs both the prompt snapshot and every new write."""

    def __init__(
        self,
        *,
        target: str,
        quantizer: Callable[[torch.Tensor], torch.Tensor] = asymmetric_int4_qdq,
        config: Any | None = None,
        ledger: QuantizerLedger | None = None,
    ) -> None:
        _validate_target(target)
        super().__init__(config=config)
        self.target = target
        self.quantizer = quantizer
        self.ledger = ledger if ledger is not None else QuantizerLedger()

    @classmethod
    def from_prompt_cache(
        cls,
        prompt_cache: DynamicCache,
        *,
        target: str,
        quantizer: Callable[[torch.Tensor], torch.Tensor] = asymmetric_int4_qdq,
        config: Any | None = None,
        ledger: QuantizerLedger | None = None,
    ) -> "QuantizedDynamicCache":
        result = cls(target=target, quantizer=quantizer, config=config, ledger=ledger)
        if len(result.layers) not in {0, len(prompt_cache.layers)}:
            raise QuantizedCacheError("prompt cache/model layer-count mismatch")
        if len(result.layers) == 0:
            # Lazy DynamicCache: create a layer for every populated prompt layer.
            result.layers = [result.layer_class_to_replicate() for _ in prompt_cache.layers]
        for layer_idx, (source, destination) in enumerate(zip(prompt_cache.layers, result.layers)):
            if not getattr(source, "is_initialized", False):
                continue
            keys, values = _transform_pair(
                source.keys,
                source.values,
                target=target,
                quantizer=quantizer,
            )
            destination.lazy_initialization(keys)
            destination.keys = keys
            destination.values = values
            result.ledger.record(
                phase="snapshot",
                layer_idx=layer_idx,
                target=target,
                key_states=keys,
                value_states=values,
            )
        assert_no_storage_aliases([prompt_cache, result])
        return result

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = _transform_pair(
            key_states,
            value_states,
            target=self.target,
            quantizer=self.quantizer,
        )
        self.ledger.record(
            phase="update",
            layer_idx=layer_idx,
            target=self.target,
            key_states=keys,
            value_states=values,
        )
        return super().update(keys, values, layer_idx, cache_kwargs)


def clone_dynamic_cache(prompt_cache: DynamicCache, *, config: Any | None = None) -> DynamicCache:
    result = DynamicCache(config=config)
    if len(result.layers) not in {0, len(prompt_cache.layers)}:
        raise QuantizedCacheError("prompt cache/model layer-count mismatch")
    if len(result.layers) == 0:
        result.layers = [result.layer_class_to_replicate() for _ in prompt_cache.layers]
    for source, destination in zip(prompt_cache.layers, result.layers):
        if not getattr(source, "is_initialized", False):
            continue
        keys = source.keys.clone()
        values = source.values.clone()
        destination.lazy_initialization(keys)
        destination.keys = keys
        destination.values = values
    assert_no_storage_aliases([prompt_cache, result])
    return result


def cache_storage_pointers(cache: DynamicCache) -> set[int]:
    pointers: set[int] = set()
    for layer in cache.layers:
        if not getattr(layer, "is_initialized", False):
            continue
        for tensor in (layer.keys, layer.values):
            if tensor.numel():
                pointers.add(int(tensor.untyped_storage().data_ptr()))
    return pointers


def assert_no_storage_aliases(caches: Iterable[DynamicCache]) -> None:
    seen: set[int] = set()
    for cache_index, cache in enumerate(caches):
        pointers = cache_storage_pointers(cache)
        overlap = seen & pointers
        if overlap:
            raise QuantizedCacheError(
                f"cache {cache_index} shares {len(overlap)} storage pointer(s) with a prior arm"
            )
        seen.update(pointers)


def cache_structure_fingerprint(cache: DynamicCache) -> str:
    rows: list[str] = []
    for layer_idx, layer in enumerate(cache.layers):
        if not getattr(layer, "is_initialized", False):
            rows.append(f"{layer_idx}:uninitialized")
            continue
        for name, tensor in (("k", layer.keys), ("v", layer.values)):
            rows.append(
                f"{layer_idx}:{name}:{tuple(tensor.shape)}:{tensor.dtype}:{tensor.device}:"
                f"{tensor.untyped_storage().data_ptr()}"
            )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()

