"""Compact route-pressure capture contract for a vLLM 0.26 experiment.

This module is deliberately not wired into vLLM by itself.  It contains:

* a NumPy reference implementation used to prove the metric semantics; and
* an optional Triton/PyTorch device implementation that can be bound to the
  existing ``BaseRouter.capture_fn`` hook in vLLM 0.26.

The device implementation is intentionally scoped to one TP/DP rank, eager
execution, normal (non-speculative) decode.  Broader support needs explicit
qualification rather than silently reusing token/request mappings that do not
hold under SP, DBO, or asynchronous batch queues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

try:  # The repository's lightweight local test environment has no torch.
    import torch
    from vllm.triton_utils import tl, triton
except ModuleNotFoundError:  # pragma: no cover - exercised on the GPU host.
    torch = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    triton = None  # type: ignore[assignment]


class RoutePressureReference(NamedTuple):
    """Lossy per-step pressure summary plus an exact last-token signature."""

    load_counts: np.ndarray
    layer_max_load: np.ndarray
    layer_active_experts: np.ndarray
    request_signature: np.ndarray


def aggregate_reference(
    routes: np.ndarray,
    req_indices: np.ndarray,
    query_start_loc: np.ndarray,
    num_experts: int,
) -> RoutePressureReference:
    """Compute the contract from lossless routes.

    Args:
        routes: Logical expert ids, shape ``[valid_tokens, layers, top_k]``.
        req_indices: Flattened token-to-request mapping, shape ``[valid_tokens]``.
        query_start_loc: Exclusive request offsets, shape ``[num_reqs + 1]``.
        num_experts: Number of logical experts in each MoE layer.

    ``request_signature[r]`` is the exact ``[layer, top_k]`` route of the last
    token scheduled for request ``r`` in this step.  It is a pre-action signal
    only for the *following* synchronous scheduler step.
    """

    routes = np.asarray(routes)
    req_indices = np.asarray(req_indices)
    query_start_loc = np.asarray(query_start_loc)
    if routes.ndim != 3:
        raise ValueError(f"routes must be rank 3, got {routes.shape}")
    num_tokens, num_layers, top_k = routes.shape
    if req_indices.shape != (num_tokens,):
        raise ValueError(
            f"req_indices must have shape {(num_tokens,)}, got {req_indices.shape}"
        )
    if query_start_loc.ndim != 1 or query_start_loc.size < 2:
        raise ValueError("query_start_loc must contain at least [0, end]")
    if int(query_start_loc[0]) != 0 or int(query_start_loc[-1]) != num_tokens:
        raise ValueError("query_start_loc must span exactly the valid token rows")
    if np.any(np.diff(query_start_loc) <= 0):
        raise ValueError("every scheduled request must own at least one token")
    expected_req_indices = np.repeat(
        np.arange(query_start_loc.size - 1, dtype=req_indices.dtype),
        np.diff(query_start_loc),
    )
    if not np.array_equal(req_indices, expected_req_indices):
        raise ValueError("req_indices and query_start_loc describe different ownership")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    load_counts = np.zeros((num_layers, num_experts), dtype=np.int32)
    for layer_id in range(num_layers):
        layer_ids = routes[:, layer_id, :].reshape(-1)
        valid = (layer_ids >= 0) & (layer_ids < num_experts)
        np.add.at(load_counts[layer_id], layer_ids[valid].astype(np.int64), 1)

    last_rows = query_start_loc[1:].astype(np.int64) - 1
    request_signature = routes[last_rows].astype(np.int16, copy=True)
    invalid = (request_signature < 0) | (request_signature >= num_experts)
    request_signature[invalid] = -1
    return RoutePressureReference(
        load_counts=load_counts,
        layer_max_load=load_counts.max(axis=1),
        layer_active_experts=(load_counts > 0).sum(axis=1, dtype=np.int32),
        request_signature=request_signature.reshape(-1, num_layers, top_k),
    )


def compact_snapshot_nbytes(
    num_reqs: int,
    num_layers: int,
    top_k: int,
) -> int:
    """Bytes copied to host by the proposed compact snapshot.

    Two int32 values per layer (max load and active experts), plus exact int16
    ``[request, layer, top_k]`` last-token signatures.
    """

    return num_layers * 2 * 4 + num_reqs * num_layers * top_k * 2


def lossless_snapshot_nbytes(
    num_tokens: int,
    num_layers: int,
    top_k: int,
) -> int:
    """Bytes copied by vLLM's current route ids + int64 slot mapping."""

    return num_tokens * num_layers * top_k * 4 + num_tokens * 8


@dataclass(frozen=True)
class InitialScope:
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    async_scheduling: bool = False
    speculative_decode: bool = False
    enforce_eager: bool = True
    dual_batch_overlap: bool = False

    def validate(self) -> None:
        unsupported: list[str] = []
        if self.tensor_parallel_size != 1:
            unsupported.append("TP/SP")
        if self.data_parallel_size != 1:
            unsupported.append("DP")
        if self.pipeline_parallel_size != 1:
            unsupported.append("PP")
        if self.async_scheduling:
            unsupported.append("async scheduling")
        if self.speculative_decode:
            unsupported.append("speculative decode")
        if not self.enforce_eager:
            unsupported.append("CUDA graph/compiled execution")
        if self.dual_batch_overlap:
            unsupported.append("DBO")
        if unsupported:
            raise ValueError("unqualified route-pressure scope: " + ", ".join(unsupported))


if triton is not None and torch is not None:

    @triton.jit
    def _capture_pressure_kernel(
        topk_ids_ptr,
        req_indices_ptr,
        query_start_loc_ptr,
        load_counts_ptr,
        signature_ptr,
        valid_tokens,
        layer_id,
        NUM_LAYERS: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        numel = valid_tokens * TOP_K
        valid_elem = offs < numel
        token_idx = offs // TOP_K
        topk_idx = offs % TOP_K
        expert_id = tl.load(topk_ids_ptr + offs, mask=valid_elem, other=-1).to(tl.int32)
        valid_expert = valid_elem & (expert_id >= 0) & (expert_id < NUM_EXPERTS)
        safe_expert = tl.where(valid_expert, expert_id, 0)

        # Counts every top-k occurrence for this logical layer.
        tl.atomic_add(
            load_counts_ptr + layer_id * NUM_EXPERTS + safe_expert,
            1,
            mask=valid_expert,
        )

        # Keep only the last token scheduled for each request.  req_indices and
        # query_start_loc are vLLM's existing persistent GPU input buffers.
        req_idx = tl.load(req_indices_ptr + token_idx, mask=valid_elem, other=0).to(
            tl.int64
        )
        last_token_idx = tl.load(
            query_start_loc_ptr + req_idx + 1, mask=valid_elem, other=0
        ).to(tl.int64) - 1
        is_last = valid_expert & (token_idx == last_token_idx)
        sig_off = (req_idx * NUM_LAYERS + layer_id) * TOP_K + topk_idx
        tl.store(signature_ptr + sig_off, expert_id.to(tl.int16), mask=is_last)


    @triton.jit
    def _reduce_pressure_kernel(
        load_counts_ptr,
        summary_ptr,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        layer_id = tl.program_id(0)
        expert_offs = tl.arange(0, BLOCK_SIZE)
        valid = expert_offs < NUM_EXPERTS
        counts = tl.load(
            load_counts_ptr + layer_id * NUM_EXPERTS + expert_offs,
            mask=valid,
            other=0,
        ).to(tl.int32)
        max_load = tl.max(counts, axis=0)
        active = tl.sum((counts > 0).to(tl.int32), axis=0)
        tl.store(summary_ptr + layer_id * 2, max_load)
        tl.store(summary_ptr + layer_id * 2 + 1, active)


    class GPURoutePressureSketch:
        """Single-rank device capturer for the first native overhead gate.

        Bind ``capture`` to every ``BaseRouter`` exactly as vLLM currently binds
        ``RoutedExpertsCapturer.capture``.  Call ``begin_step`` after
        ``GPUModelRunner._prepare_inputs`` and before model forward, then call
        ``finalize`` after forward.  All operations enqueue on the current
        stream; this class performs no host synchronization or D2H itself.
        """

        def __init__(
            self,
            max_num_reqs: int,
            num_layers: int,
            num_experts: int,
            top_k: int,
            device: Any,
        ) -> None:
            if num_experts > np.iinfo(np.int16).max:
                raise ValueError("initial sketch stores expert ids as int16")
            self.max_num_reqs = max_num_reqs
            self.num_layers = num_layers
            self.num_experts = num_experts
            self.top_k = top_k
            self.load_counts = torch.zeros(
                (num_layers, num_experts), dtype=torch.int32, device=device
            )
            self.summary = torch.empty((num_layers, 2), dtype=torch.int32, device=device)
            self.request_signature = torch.full(
                (max_num_reqs, num_layers, top_k),
                -1,
                dtype=torch.int16,
                device=device,
            )
            self._req_indices = None
            self._query_start_loc = None
            self._num_reqs = 0
            self._valid_tokens = 0

        def begin_step(
            self,
            req_indices: Any,
            query_start_loc: Any,
            num_reqs: int,
            valid_tokens: int,
        ) -> None:
            if not 0 < num_reqs <= self.max_num_reqs:
                raise ValueError(f"invalid num_reqs={num_reqs}")
            if valid_tokens <= 0:
                raise ValueError(f"invalid valid_tokens={valid_tokens}")
            if req_indices.numel() < valid_tokens or query_start_loc.numel() < num_reqs + 1:
                raise ValueError("token/request mapping buffers are too small")
            self.load_counts.zero_()
            self.request_signature[:num_reqs].fill_(-1)
            self._req_indices = req_indices
            self._query_start_loc = query_start_loc
            self._num_reqs = num_reqs
            self._valid_tokens = valid_tokens

        def capture(self, layer_id: int, topk_ids: Any) -> None:
            if self._req_indices is None or self._query_start_loc is None:
                raise RuntimeError("begin_step must run before capture")
            if not 0 <= layer_id < self.num_layers:
                return
            if topk_ids.ndim != 2 or topk_ids.shape[1] != self.top_k:
                raise ValueError(f"unexpected topk_ids shape {tuple(topk_ids.shape)}")
            if topk_ids.shape[0] < self._valid_tokens:
                raise ValueError("topk_ids does not cover all valid token rows")
            if not topk_ids.is_contiguous():
                raise ValueError("topk_ids must be contiguous; refusing a hidden copy")
            numel = self._valid_tokens * self.top_k
            grid = (triton.cdiv(numel, 256),)
            _capture_pressure_kernel[grid](
                topk_ids,
                self._req_indices,
                self._query_start_loc,
                self.load_counts,
                self.request_signature,
                self._valid_tokens,
                layer_id,
                NUM_LAYERS=self.num_layers,
                NUM_EXPERTS=self.num_experts,
                TOP_K=self.top_k,
                BLOCK_SIZE=256,
            )

        def finalize(self) -> tuple[Any, Any]:
            if self._num_reqs <= 0:
                raise RuntimeError("begin_step must run before finalize")
            block = triton.next_power_of_2(self.num_experts)
            _reduce_pressure_kernel[(self.num_layers,)](
                self.load_counts,
                self.summary,
                NUM_EXPERTS=self.num_experts,
                BLOCK_SIZE=block,
            )
            return self.summary, self.request_signature[: self._num_reqs]

else:

    class GPURoutePressureSketch:  # pragma: no cover - error-only local stub.
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("GPURoutePressureSketch requires torch and vLLM Triton")

