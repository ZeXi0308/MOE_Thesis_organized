"""Eager, single-GPU dual-format expert execution for actual vLLM 0.26 APIs.

This module contains real resident weights and delegates to the same fused
backends as ``format_backends.build_backend``. It is a post-load prototype,
NOT a registered ``vllm serve --quantization`` option or a complete serving
integration. A CUDA test and model-quality evaluation are still required.

Minimal worker-side wiring, AFTER normal model weight post-processing and
BEFORE KV-cache sizing, warmup, compilation, or graph capture::

    # Run inside the GPU worker, not in the API/frontend process.
    from dual_format_moe import install_dual_format, format_phase
    method = install_dual_format(
        runner.routed_experts,
        fp8_weights="layer_00_fp8.pt",
        lowbit_weights="layer_00_awq.pt", lowbit_format="awq",
        threshold=32, policy="threshold", release_original_weights=True)

    # Do this for EVERY intended MoE layer with that layer's sidecar artifacts.
    # For the phase rule, wrap the worker's entire model forward:
    with format_phase("mixed"):  # or "prefill" / "decode", from scheduler
        output = model(...)

The runner already selects top-k ONCE before ``RoutedExperts.forward_modular``;
this method consumes those IDs/weights and does not repeat router selection.
The existing runner also owns shared-expert execution, so this method returns
only the routed contribution. It does not call or sum shared experts again.

Use enforce_eager=True (CLI: --enforce-eager), TP=DP=EP=1, no LoRA/EPLB/DBO,
and no torch.compile. ``bincount`` has data-dependent output allocation and
``max().item()`` waits for CUDA work on the host; both costs belong to selection
overhead. Python branching cannot change a previously captured CUDA graph.
Capture/compilation are rejected rather than silently freezing a format.

``format_phase`` must be set in the worker execution context using actual
scheduler semantics; token count alone cannot distinguish decode from prefill.
Pure phases bypass the histogram; mixed steps use the threshold. Missing phase
metadata is an error for the phase policy. Expert-split v2 is NOT implemented:
the backends have not established one shared mask/sentinel contract.

Both artifacts must be genuine checkpoint exports (FP8 is never synthesized).
Include base_checkpoint/base_checkpoint_revision in each artifact when their
quantized checkpoint names differ. Existing artifacts without these fields
require an explicit allow_unverified_source=True and are marked unverified;
that option is suitable for mechanism checks, not a same-model quality claim.

Installation temporarily holds original AND both new sets of weights. For
Qwen3 use an FP8-loaded base model or otherwise budget this peak; installing
over a BF16 model can exceed VRAM before old weights are released. Patching an
already-serving engine after its KV capacity is sized is unsupported. Retain
the normal loader for non-expert weights; sidecars only replace routed experts.
"""

from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from typing import Literal
import weakref

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)

try:  # Both ``python code/script.py`` and package imports are supported.
    from .format_backends import ExpertShape, build_backend
except ImportError:
    from format_backends import ExpertShape, build_backend


Phase = Literal["prefill", "decode", "mixed"]
_phase: ContextVar[Phase | None] = ContextVar("dual_format_phase", default=None)


@contextmanager
def format_phase(phase: Phase):
    """Set a phase for all patched layers in this worker's current forward."""
    if phase not in ("prefill", "decode", "mixed"):
        raise ValueError("phase must be prefill, decode, or mixed")
    token = _phase.set(phase)
    try:
        yield
    finally:
        _phase.reset(token)


class _ResidentWeights(torch.nn.Module):
    """Register the exact tensors captured by delegates, without copying them."""

    def __init__(self, backends):
        super().__init__()
        for name, backend in backends.items():
            arm = torch.nn.Module()
            for key, value in backend.tensors.items():
                if torch.is_tensor(value):
                    arm.register_buffer(key, value, persistent=True)
            self.add_module(name, arm)

    def _apply(self, fn, recurse=True):
        # Backend closures and pre-built quant configs retain the exact tensor
        # objects. Allowing Module.to() to replace buffers would stale them.
        raise RuntimeError(
            "Dual-format tensors are fixed to their install device/dtype; "
            "rebuild from artifacts instead of calling .to()/.half() on this model"
        )


def _base_identity(metadata):
    if metadata.get("base_checkpoint") and metadata.get("base_checkpoint_revision"):
        return metadata["base_checkpoint"], metadata["base_checkpoint_revision"]
    return None


class DualFormatMoEMethod(FusedMoEMethodBase):
    """Post-load vLLM method: FP8 Triton versus calibrated AWQ/NVFP4 delegate.

    Constructor loads both complete artifacts on CUDA. ``create_weights`` then
    registers those tensors on a RoutedExperts module. It is not a checkpoint
    schema for vLLM's stock loader; use ``install_dual_format`` after loading.
    ``apply`` has the real v0.26 modular method signature.
    """

    def __init__(
        self,
        moe,
        *,
        fp8_weights: str | Path,
        lowbit_weights: str | Path,
        lowbit_format: Literal["awq", "nvfp4"],
        threshold: int,
        policy: Literal["threshold", "phase"] = "threshold",
        nvfp4_backend: str = "marlin",
        device: str | torch.device | None = None,
        allow_unverified_source: bool = False,
        record_limit: int = 1024,
        record_cuda_events: bool = False,
        validate_route_values: bool = False,
    ):
        super().__init__(moe)
        self._validate_config()
        if policy not in ("threshold", "phase"):
            raise NotImplementedError(
                "Only threshold and phase are implemented; expert-split masks "
                "are not validated across these real backends"
            )
        if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 1:
            raise ValueError("threshold must be a positive integer")
        if lowbit_format not in ("awq", "nvfp4"):
            raise ValueError("lowbit_format must be calibrated awq or nvfp4")
        if not fp8_weights or not lowbit_weights:
            raise ValueError("Both checkpoint artifact paths are required")
        if record_limit < 0:
            raise ValueError("record_limit cannot be negative")
        self.shape = ExpertShape(
            moe.hidden_dim, moe.intermediate_size_per_partition,
            moe.num_experts, moe.experts_per_token,
        )
        self.shape.validate()
        self.device = torch.device(device if device is not None else moe.device)
        if self.device.type != "cuda":
            raise ValueError("This prototype requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.threshold, self.policy = threshold, policy
        self.lowbit_format = lowbit_format
        self.record_cuda_events = record_cuda_events
        self.validate_route_values = validate_route_values
        self.records = deque(maxlen=record_limit)
        self.selection_counts = Counter()
        self.switch_count = 0
        self.calls = 0
        self._previous_format = None
        self._layer = None
        self.selection_host_ns_total = 0
        self.dispatch_host_ns_total = 0
        self.unregistered_original_tensor_bytes = 0
        self.retained_original_bytes = 0

        # build_backend resets RNG for synthetic characterization too. Preserve
        # application RNG states while loading genuine checkpoint artifacts.
        with torch.random.fork_rng(devices=[self.device.index]):
            fp8 = build_backend(
                "fp8", self.shape, device=str(self.device), weights=fp8_weights,
                max_tokens=moe.max_num_tokens,
            )
            low = build_backend(
                lowbit_format, self.shape, device=str(self.device),
                weights=lowbit_weights, nvfp4_backend=nvfp4_backend,
                max_tokens=moe.max_num_tokens,
            )
        if fp8.metadata.get("weight_source", "").startswith("synthetic"):
            raise ValueError("Synthetic FP8 cannot be used for dual-format execution")
        if str(fp8.metadata["layer"]) != str(low.metadata["layer"]):
            raise ValueError("The two artifacts identify different model layers")
        left, right = _base_identity(fp8.metadata), _base_identity(low.metadata)
        # Same original checkpoint identity is also sufficient when both
        # exports explicitly name it rather than distinct quantized repos.
        if left is None and right is None:
            a = (fp8.metadata["checkpoint"], fp8.metadata["checkpoint_revision"])
            b = (low.metadata["checkpoint"], low.metadata["checkpoint_revision"])
            if a == b:
                left = right = a
        self.same_base_verified = left is not None and left == right
        if left is not None and right is not None and left != right:
            raise ValueError("Artifacts have different base checkpoint identities")
        if not self.same_base_verified and not allow_unverified_source:
            raise ValueError(
                "Cannot establish a common base checkpoint. Add matching "
                "base_checkpoint/base_checkpoint_revision metadata to both "
                "artifacts; mechanism-only runs may explicitly allow unverified source"
            )
        self.backends = {"fp8": fp8, lowbit_format: low}
        self.resident_weights = _ResidentWeights(self.backends)
        # The runner sees unquantized hidden states; each delegate owns its
        # distinct quant configuration internally. Do NOT expose the FP8 config
        # as a shared config that could prequantize the low-bit branch's input.
        self.moe_quant_config = FusedMoEQuantConfig.make()

    def _validate_config(self):
        moe, pc = self.moe, self.moe.moe_parallel_config
        for name in ("tp_size", "dp_size", "ep_size", "pcp_size", "sp_size"):
            if getattr(pc, name, 1) != 1:
                raise NotImplementedError("Dual-format prototype is single-GPU only")
        if pc.use_ep or pc.enable_eplb:
            raise NotImplementedError("EP/EPLB are unsupported")
        if moe.num_local_experts != moe.num_experts:
            raise NotImplementedError("Every routed expert must be local")
        if moe.num_logical_experts != moe.num_experts:
            raise NotImplementedError("Redundant/fused shared expert IDs are unsupported")
        if moe.activation != MoEActivation.SILU or not moe.is_act_and_mul:
            raise NotImplementedError("Delegates implement standard gated SiLU only")
        if moe.has_bias or moe.is_lora_enabled:
            raise NotImplementedError("Bias and LoRA are not supported by these delegates")
        if any(getattr(moe, key, None) is not None for key in (
            "swiglu_limit", "swiglu_alpha", "swiglu_beta"
        )):
            raise NotImplementedError("Custom SwiGLU clamps/scales are unsupported")
        if moe.in_dtype != torch.bfloat16:
            raise NotImplementedError("The measured delegate contract is BF16 input/output")
        if moe.hidden_dim_unpadded not in (None, moe.hidden_dim):
            raise NotImplementedError("Padded hidden dimensions are unsupported")
        if moe.intermediate_size_per_partition_unpadded not in (
            None, moe.intermediate_size_per_partition
        ):
            raise NotImplementedError("Padded intermediate dimensions are unsupported")

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition,
                            act_dtype, moe_parallel_config):
        return hidden_size, intermediate_size_per_partition

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        expected = (self.shape.experts, self.shape.d, self.shape.intermediate)
        if (num_experts, hidden_size, intermediate_size_per_partition) != expected:
            raise ValueError("RoutedExperts shape does not match both artifacts")
        if params_dtype != torch.bfloat16:
            raise ValueError("RoutedExperts must expose BF16 activation parameters")
        if getattr(layer, "apply_router_weight_on_input", False):
            raise NotImplementedError("Delegates apply router weights on outputs only")
        if layer.expert_map is not None:
            raise NotImplementedError("Expert remapping/masking is not supported")
        existing = getattr(layer, "_dual_format_resident", None)
        if existing is not None and existing is not self.resident_weights:
            raise ValueError("This layer already owns a different dual-format method")
        if self._layer is not None and self._layer() is not layer:
            raise ValueError("A method instance cannot be shared across model layers")
        layer.add_module("_dual_format_resident", self.resident_weights)
        self._layer = weakref.ref(layer)

    def process_weights_after_loading(self, layer):
        # The input artifacts are already prepared by their respective real
        # backend; repacking them here would corrupt the prepared layouts.
        if self._layer is None or self._layer() is not layer:
            raise RuntimeError("Install/register the prepared sidecars before processing")

    def get_fused_moe_quant_config(self, layer):
        return self.moe_quant_config

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # MoERunner.maybe_init_modular_kernel must leave this direct single-GPU
        # apply method in place. Both delegates already prepare/finalize locally.
        if routing_tables is not None:
            raise NotImplementedError("Distributed routing tables are unsupported")
        return None

    @property
    def topk_indices_dtype(self):
        return torch.int32

    @property
    def is_monolithic(self):
        return False

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts=None, shared_experts_input=None):
        if torch.compiler.is_compiling():
            raise RuntimeError("Dual-format Python dispatch requires eager execution")
        if x.device != self.device or x.dtype != torch.bfloat16:
            raise ValueError("Input must be BF16 on the artifact installation device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Dual-format host decisions cannot run during CUDA graph capture")
        if self._layer is None or self._layer() is not layer:
            raise RuntimeError("Call create_weights/install_dual_format for this layer first")
        if x.ndim != 2 or x.shape[1] != self.shape.d or not x.is_contiguous():
            raise ValueError("Expected contiguous [tokens, hidden_dim] input")
        expected = (x.shape[0], self.shape.top_k)
        if tuple(topk_ids.shape) != expected or tuple(topk_weights.shape) != expected:
            raise ValueError("Routing shapes must be [tokens, top_k]")
        if topk_ids.device != self.device or topk_weights.device != self.device:
            raise ValueError("Routing tensors must share the input CUDA device")
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Routing IDs must be int32/int64")
        if x.shape[0] > self.moe.max_num_tokens:
            raise ValueError("Input exceeds the installed backend's max_num_tokens")
        phase = _phase.get()
        if self.policy == "phase" and phase is None:
            raise RuntimeError("Phase policy requires worker-side format_phase context")
        if not x.shape[0]:
            return torch.empty_like(x)

        # Shared experts are intentionally handled by the stock MoERunner;
        # mk_can_overlap_shared_experts remains False (moe_kernel is None).
        start = end = None
        if self.record_cuda_events:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
        select_start = perf_counter_ns()
        if self.validate_route_values:
            # Optional extra synchronization for direct callers. The ordinary
            # vLLM router already guarantees ID validity; disclose this flag.
            bounds = torch.stack(torch.aminmax(topk_ids)).tolist()
            if bounds[0] < 0 or bounds[1] >= self.shape.experts:
                raise ValueError("Routing contains invalid or masked expert IDs")
        max_tokens = None
        if self.policy == "phase" and phase != "mixed":
            chosen = "fp8" if phase == "prefill" else self.lowbit_format
        else:
            counts = torch.bincount(topk_ids.reshape(-1).long(),
                                    minlength=self.shape.experts)
            if counts.numel() != self.shape.experts:
                raise ValueError("Routing ID exceeds the number of experts")
            max_tokens = int(counts.max().item())  # CUDA -> host wait INCLUDED.
            chosen = "fp8" if max_tokens >= self.threshold else self.lowbit_format
        select_end = perf_counter_ns()
        output = self.backends[chosen](x, topk_weights, topk_ids)
        dispatch_end = perf_counter_ns()
        if end is not None:
            end.record()
        selection_ns = select_end - select_start
        dispatch_ns = dispatch_end - select_end
        switched = self._previous_format is not None and self._previous_format != chosen
        self.switch_count += int(switched)
        self._previous_format = chosen
        self.calls += 1
        self.selection_counts[chosen] += 1
        self.selection_host_ns_total += selection_ns
        self.dispatch_host_ns_total += dispatch_ns
        self.records.append({
            "call": self.calls, "phase": phase, "format": chosen,
            "input_tokens": x.shape[0], "max_expert_tokens": max_tokens,
            "switched": switched, "selection_host_ns": selection_ns,
            "dispatch_host_ns": dispatch_ns,
            "selection_has_host_sync": max_tokens is not None or self.validate_route_values,
            "cuda_events": (start, end) if start is not None else None,
        })
        return output

    def take_records(self, *, synchronize=False):
        """Drain bounded records outside the timed loop; CUDA timing is optional.

        Host dispatch time is not GPU kernel latency. A CUDA interval, when
        enabled, includes the host-decision-induced idle gap. Incomplete event
        pairs report None unless synchronize=True is explicitly requested.
        """
        result = []
        while self.records:
            record = self.records.popleft()
            events = record.pop("cuda_events")
            record["layer_elapsed_cuda_ms"] = None
            if events is not None:
                start, end = events
                if synchronize:
                    end.synchronize()
                if end.query():
                    record["layer_elapsed_cuda_ms"] = start.elapsed_time(end)
            result.append(record)
        return result

    def summary(self):
        return {
            "implementation": "vllm0.26_post_load_eager_dual_format_prototype",
            "serving_integration_validated": False,
            "expert_split_implemented": False,
            "shape": asdict(self.shape), "policy": self.policy,
            "threshold": self.threshold, "device": str(self.device),
            "same_base_verified": self.same_base_verified,
            "calls": self.calls, "selection_counts": dict(self.selection_counts),
            "switch_count": self.switch_count,
            "selection_host_ns_total": self.selection_host_ns_total,
            "dispatch_host_ns_total": self.dispatch_host_ns_total,
            "threshold_selection_includes_cuda_host_wait": True,
            "record_cuda_events": self.record_cuda_events,
            "validate_route_values": self.validate_route_values,
            "resident_weight_and_scale_bytes": sum(
                b.metadata["resident_weight_and_scale_bytes"] for b in self.backends.values()
            ),
            "unregistered_original_tensor_bytes": self.unregistered_original_tensor_bytes,
            "retained_original_bytes": self.retained_original_bytes,
            "memory_accounting": "logical tensor bytes; measure actual CUDA memory separately",
            "backends": {name: dict(b.metadata) for name, b in self.backends.items()},
        }


def install_dual_format(routed_experts, *, release_original_weights=True, **kwargs):
    """Patch a normally loaded RoutedExperts before worker memory profiling.

    Only direct expert parameters and weight-related buffers are released;
    expert-map metadata remains. No old method is retained internally. Caller
    references to old tensors/methods can retain memory and must be accounted
    for. Reload the original model to restore the original quant method.
    """
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    if not isinstance(routed_experts, RoutedExperts):
        raise TypeError("Expected actual vLLM RoutedExperts, not a legacy FusedMoE layer")
    if isinstance(routed_experts.quant_method, DualFormatMoEMethod):
        raise ValueError("Layer is already patched")
    method = DualFormatMoEMethod(routed_experts.moe_config, **kwargs)
    original_params = dict(routed_experts.named_parameters(recurse=False))
    # RoutedExperts' direct parameters are expert weights/scales. Preserve
    # routing metadata buffers; only release recognized weight buffers.
    original_buffers = {
        name: value for name, value in routed_experts.named_buffers(recurse=False)
        if name.startswith(("w13_", "w2_", "w1_", "g_idx", "sort_indices"))
    }
    old_bytes = sum(t.numel() * t.element_size()
                    for t in [*original_params.values(), *original_buffers.values()])
    method.create_weights(
        routed_experts, routed_experts.local_num_experts,
        routed_experts.hidden_size, routed_experts.intermediate_size_per_partition,
        routed_experts.params_dtype,
    )
    method.process_weights_after_loading(routed_experts)
    # Ensure pending loader/backend-preparation work has completed before
    # retiring original tensors. Installation must never race live serving.
    torch.cuda.synchronize(method.device)
    routed_experts._replace_quant_method(method)
    if release_original_weights:
        for name in [*original_params, *original_buffers]:
            delattr(routed_experts, name)
        method.unregistered_original_tensor_bytes = old_bytes
    else:
        method.retained_original_bytes = old_bytes
    return method
