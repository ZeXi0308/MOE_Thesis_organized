"""Runtime primitives for same-state TriageAudit execution.

The module contains no dataset or sealed-result logic.  It owns the two pieces
that previously caused false positive evidence: cache forking and a persistent
INT4 quality-proxy backend whose weights are prepared once, not per decode step.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F


class TriageRuntimeError(RuntimeError):
    pass


def find_expert_linears(model: torch.nn.Module, scope: str = "all") -> list[torch.nn.Linear]:
    if scope not in {"all", "gate_up_only", "down_only"}:
        raise TriageRuntimeError(f"unsupported expert scope: {scope}")
    names: list[str] = []
    if scope in {"all", "gate_up_only"}:
        names.extend(["gate_proj", "up_proj", "w1", "w3"])
    if scope in {"all", "down_only"}:
        names.extend(["down_proj", "w2"])
    try:
        layers = model.model.layers
    except AttributeError as exc:
        raise TriageRuntimeError("model has no model.layers collection") from exc
    result: list[torch.nn.Linear] = []
    for layer in layers:
        if hasattr(layer, "block_sparse_moe"):
            moe = layer.block_sparse_moe
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            moe = layer.mlp
        else:
            continue
        for expert in moe.experts:
            for name in names:
                value = getattr(expert, name, None)
                if isinstance(value, torch.nn.Linear):
                    result.append(value)
    if not result:
        raise TriageRuntimeError("no expert linear modules found")
    if len({id(item) for item in result}) != len(result):
        raise TriageRuntimeError("duplicate expert linear identity")
    return result


def symmetric_int4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric RTN quantize-dequantize to [-7, 7]."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise TriageRuntimeError("expert weight must be a floating 2D tensor")
    source = weight.detach().float()
    scale = source.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 7.0
    quantized = torch.round(source / scale).clamp(-7, 7)
    return (quantized * scale).to(device=weight.device, dtype=weight.dtype)


class _PreparedInt4Linear:
    def __init__(self, linear: torch.nn.Linear):
        self.weight = symmetric_int4_weight(linear.weight)
        self.bias = linear.bias
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return F.linear(x, self.weight, self.bias)


class PreparedInt4ExpertBackend:
    """Switch expert linears to persistent INT4-dequantized proxy weights.

    Preparation happens once. Entering a low-precision step only switches
    Python forward dispatch; it never requantizes or allocates weights.
    """

    def __init__(self, model: torch.nn.Module, expected_linears: int, scope: str = "all"):
        if type(expected_linears) is not int or expected_linears <= 0:
            raise TriageRuntimeError("expected_linears must be a positive integer")
        self.linears = find_expert_linears(model, scope)
        if len(self.linears) != expected_linears:
            raise TriageRuntimeError(
                f"expert linear count mismatch: expected {expected_linears}, got {len(self.linears)}"
            )
        self.proxies = [_PreparedInt4Linear(linear) for linear in self.linears]
        self._active = False
        self.low_model_forwards = 0

    def __enter__(self) -> "PreparedInt4ExpertBackend":
        if self._active:
            raise TriageRuntimeError("nested INT4 backend activation")
        for linear, proxy in zip(self.linears, self.proxies):
            if "forward" in linear.__dict__:
                raise TriageRuntimeError("expert linear already has an instance forward override")
            linear.forward = proxy
        self._active = True
        self.low_model_forwards += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for linear in self.linears:
            if "forward" in linear.__dict__:
                del linear.__dict__["forward"]
        self._active = False

    @property
    def expert_linear_calls(self) -> int:
        return sum(proxy.calls for proxy in self.proxies)


def _iter_tensors(value: Any, seen: set[int] | None = None) -> Iterator[torch.Tensor]:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(key, seen)
            yield from _iter_tensors(item, seen)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item, seen)
        return
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        for item in attributes.values():
            yield from _iter_tensors(item, seen)


def _storage_keys(value: Any) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for tensor in _iter_tensors(value):
        if tensor.numel() == 0 or tensor.device.type == "meta":
            continue
        storage = tensor.untyped_storage()
        keys.add((str(tensor.device), storage.data_ptr(), storage.nbytes()))
    return keys


def clone_cache(cache: Any) -> Any:
    """Deep-copy a HF cache and reject any tensor-storage alias."""
    cloned = copy.deepcopy(cache)
    original_keys = _storage_keys(cache)
    cloned_keys = _storage_keys(cloned)
    if not original_keys or not cloned_keys:
        raise TriageRuntimeError("cache clone contained no inspectable tensor storage")
    overlap = original_keys & cloned_keys
    if overlap:
        raise TriageRuntimeError(f"cache clone aliases {len(overlap)} tensor storages")
    return cloned


def fork_cache_pair(cache: Any) -> tuple[Any, Any]:
    high = clone_cache(cache)
    low = clone_cache(cache)
    overlap = _storage_keys(high) & _storage_keys(low)
    if overlap:
        raise TriageRuntimeError(f"high/low cache forks alias {len(overlap)} storages")
    return high, low


def cache_sequence_length(cache: Any) -> int:
    method = getattr(cache, "get_seq_length", None)
    if callable(method):
        value = method()
        if isinstance(value, torch.Tensor):
            value = value.item()
        if type(value) is not int:
            raise TriageRuntimeError("cache get_seq_length did not return an integer")
        return value
    tensors = list(_iter_tensors(cache))
    if not tensors:
        raise TriageRuntimeError("cannot infer cache sequence length")
    candidates = [int(tensor.shape[-2]) for tensor in tensors if tensor.ndim >= 3]
    if not candidates:
        raise TriageRuntimeError("cache has no tensor with a sequence dimension")
    if len(set(candidates)) != 1:
        raise TriageRuntimeError(f"inconsistent inferred cache lengths: {set(candidates)}")
    return candidates[0]


def per_step_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    ref = reference_logits.float()
    cand = candidate_logits.float()
    if ref.shape != cand.shape or ref.ndim < 1:
        raise TriageRuntimeError("KL logits must have the same non-scalar shape")
    log_p = F.log_softmax(ref, dim=-1)
    log_q = F.log_softmax(cand, dim=-1)
    value = (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()
    if not torch.isfinite(value) or value < -1e-6:
        raise TriageRuntimeError("KL produced a non-finite or negative value")
    return max(0.0, float(value.detach().cpu().item()))


@dataclass(frozen=True)
class BranchResult:
    logits: torch.Tensor
    cache: Any
    pre_length: int
    post_length: int

    def validate(self) -> None:
        if self.post_length != self.pre_length + 1:
            raise TriageRuntimeError(
                f"decode branch cache length did not grow by one: {self.pre_length}->{self.post_length}"
            )


@dataclass(frozen=True)
class SameStateStep:
    high: BranchResult
    low: BranchResult
    discrepancy: float
    served_action: str

    @property
    def served(self) -> BranchResult:
        return self.high if self.served_action == "high" else self.low


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def cache_sha256(cache: Any) -> str:
    digest = hashlib.sha256()
    tensors = list(_iter_tensors(cache))
    if not tensors:
        raise TriageRuntimeError("cannot fingerprint a cache without tensors")
    for tensor in tensors:
        contiguous = tensor.detach().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def execute_single_action_step(
    cache: Any,
    input_token: torch.Tensor,
    *,
    forward: Callable[[torch.Tensor, Any], Any],
) -> BranchResult:
    """Advance one canonical cache without creating a diagnostic fork."""
    if input_token.ndim != 2 or input_token.shape[1] != 1:
        raise TriageRuntimeError("input_token must have shape [batch,1]")
    pre_length = cache_sequence_length(cache)
    output = forward(input_token, cache)
    if not hasattr(output, "logits") or not hasattr(output, "past_key_values"):
        raise TriageRuntimeError("single-action forward lacks logits/past_key_values")
    result = BranchResult(
        logits=output.logits,
        cache=output.past_key_values,
        pre_length=pre_length,
        post_length=cache_sequence_length(output.past_key_values),
    )
    result.validate()
    return result


def execute_same_state_step(
    cache: Any,
    input_token: torch.Tensor,
    *,
    high_forward: Callable[[torch.Tensor, Any], Any],
    low_forward: Callable[[torch.Tensor, Any], Any],
    served_action: str,
) -> SameStateStep:
    """Execute high/low from non-aliased copies of one pre-step cache.

    ``served_action`` is selected by the caller's already-frozen state. For an
    audit, callers may first execute with either placeholder action, inspect
    ``discrepancy``, and construct a final SameStateStep selecting the branch;
    neither forward is repeated.
    """
    if served_action not in {"high", "low"}:
        raise TriageRuntimeError("served_action must be high or low")
    if input_token.ndim != 2 or input_token.shape[1] != 1:
        raise TriageRuntimeError("input_token must have shape [batch,1]")
    pre_length = cache_sequence_length(cache)
    high_cache, low_cache = fork_cache_pair(cache)
    high_output = high_forward(input_token, high_cache)
    low_output = low_forward(input_token, low_cache)
    for name, output in (("high", high_output), ("low", low_output)):
        if not hasattr(output, "logits") or not hasattr(output, "past_key_values"):
            raise TriageRuntimeError(f"{name} forward lacks logits/past_key_values")
    high = BranchResult(
        logits=high_output.logits,
        cache=high_output.past_key_values,
        pre_length=pre_length,
        post_length=cache_sequence_length(high_output.past_key_values),
    )
    low = BranchResult(
        logits=low_output.logits,
        cache=low_output.past_key_values,
        pre_length=pre_length,
        post_length=cache_sequence_length(low_output.past_key_values),
    )
    high.validate()
    low.validate()
    if _storage_keys(high.cache) & _storage_keys(low.cache):
        raise TriageRuntimeError("post-forward high/low caches alias")
    discrepancy = per_step_kl(high.logits[:, -1, :], low.logits[:, -1, :])
    return SameStateStep(
        high=high,
        low=low,
        discrepancy=discrepancy,
        served_action=served_action,
    )
