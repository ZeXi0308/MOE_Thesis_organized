"""Eager, single-GPU expert execution for actual vLLM 0.26 APIs.

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
        activation_basis_id=verified_worker_layer_input_basis_id,
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
metadata is an error for the phase policy. Expert-split v2 supports FP8 Triton
and AWQ Marlin using native expert_map entries of identity or -1. Routing IDs
remain valid. Both full weight sets stay resident and no weights move per step.
The FP8 arm uses TritonExperts' zero-writing masked-block path, rather than the
functional path that can leave masked intermediates unwritten before amax.
NVFP4 expert splitting is rejected until its mask path is validated.

Both artifacts must be genuine checkpoint exports (FP8 is never synthesized).
Include base_checkpoint/base_checkpoint_revision in each artifact when their
quantized checkpoint names differ. Existing artifacts without these fields
require an explicit allow_unverified_source=True and are marked unverified;
that option is suitable for mechanism checks, not a same-model quality claim.

Common base weights do NOT establish compatible expert inputs. For example,
AWQ smoothing can change the preceding layernorm and inversely scale router
and expert weights. Installation therefore requires an explicit, matching
``activation_basis_id`` in BOTH artifact manifests and as an installation
argument identifying this worker layer's actual input coordinates. The ID
must describe the verified per-layer input basis (including smoothing/scaling
or permutations), not merely a base repository or revision. The caller must
check the installed non-expert weights and router; copying a sidecar's ID into
the argument without that check is not verification. No input adapter is
implemented, and metadata claiming an adapter cannot bypass this guard.
Current exports without this evidence are rejected by default.

``mechanism_only=True`` explicitly permits missing or incompatible basis IDs
for dispatch/mask diagnostics; records identify this scope and make no output
semantics or quality claim. ``allow_unverified_source=True`` is separately
needed if checkpoint lineage is unknown and never bypasses the basis guard.
Matching IDs establish a declared input-basis contract, not model quality.

``install_single_format(..., weights=..., single_format="fp8")`` installs one
real BF16/FP8/AWQ/NVFP4 artifact through the same backend and basis checks. It
does not construct a second arm, compute a histogram or select a format. It
requires release_original_weights=True. For a comparison with matched nonexpert
parameters, load the same complete common-BF16 master for the FP8, AWQ and dual
arms; do not add global online FP8 quantization only to the FP8-sidecar arm.
NVFP4 must use a master matching its own verified activation basis.

Installation temporarily holds original expert tensors AND the new one/two
sets for the current layer; budget this peak before using a large BF16 host.
Patching an already-serving engine after its KV capacity is sized is unsupported.
Retain the normal loader for non-expert weights; sidecars replace routed experts.
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
_phase_source: ContextVar[str | None] = ContextVar("dual_format_phase_source", default=None)


@contextmanager
def format_phase(phase: Phase, *, source: str = "caller"):
    """Set a phase for all patched layers in this worker's current forward."""
    if phase not in ("prefill", "decode", "mixed"):
        raise ValueError("phase must be prefill, decode, or mixed")
    token = _phase.set(phase)
    source_token = _phase_source.set(source)
    try:
        yield
    finally:
        _phase.reset(token)
        _phase_source.reset(source_token)


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


def _known_revision(value):
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = "_".join(value.strip().lower().replace("-", "_").split())
    return normalized not in ("main", "master", "head", "latest", "none", "null", "n/a", "na", "tbd") and not any(
        marker in normalized for marker in ("unknown", "unreported", "unverified", "unspecified",
            "not_known", "not_reported", "not_available", "unavailable", "pending", "unresolved", "<", ">")
    )


def _base_identity(metadata):
    if metadata.get("exact_base_revision_known") is False:
        return None
    if metadata.get("base_checkpoint") and _known_revision(metadata.get("base_checkpoint_revision")):
        return metadata["base_checkpoint"], metadata["base_checkpoint_revision"]
    return None


def _activation_basis_contract(fp8_metadata, low_metadata, worker_basis_id,
                               *, mechanism_only=False):
    """Check explicit input-coordinate identities, independently of lineage.

    A caller supplies the worker identity only after checking its preceding
    normalization/router. IDs in sidecars alone cannot establish that input
    contract. This implementation has no branch-specific input adapter.
    """
    if not isinstance(mechanism_only, bool):
        raise TypeError("mechanism_only must be a boolean")

    def declared_id(value, metadata=None):
        if metadata is not None and metadata.get("activation_basis_verified") is False:
            return None
        # Reuse the rejection of empty values and unknown/placeholder markers;
        # a basis ID need not be a commit hash.
        return value.strip() if _known_revision(value) else None

    ids = {
        "fp8": declared_id(fp8_metadata.get("activation_basis_id"), fp8_metadata),
        "lowbit": declared_id(low_metadata.get("activation_basis_id"), low_metadata),
        "worker_input": declared_id(worker_basis_id),
    }
    matched = ids["fp8"] is not None and ids["fp8"] == ids["lowbit"] == ids["worker_input"]
    if not matched and not mechanism_only:
        raise ValueError(
            "Cannot establish a common activation basis for FP8, low-bit, and "
            "the installed worker layer. Both artifacts must declare the same "
            "activation_basis_id and installation must pass that verified worker "
            "input ID. Base checkpoint identity and allow_unverified_source do "
            "not establish this contract; AWQ smoothing can change it. No input "
            "adapter is implemented. Only dispatch/mask diagnostics may use "
            "mechanism_only=True, without numerical semantics or quality claims."
        )
    return {
        "activation_basis_ids": ids,
        "activation_basis_contract_matched": matched,
        "input_adapter_implemented": False,
        "mechanism_only": mechanism_only,
        "execution_scope": "mechanism_only" if mechanism_only else "input_basis_contract_checked_prototype",
        "quality_validated": False,
    }


class NativeExpertSplit:
    """Two real fused calls over identity/-1 expert maps, without weight copies.

    Construction also accepts synthetic FP8 for the separate mechanism validator;
    the production DualFormatMoEMethod constructor still rejects synthetic data.
    Empty/full masks are supported by the native delegates for validation. The
    policy avoids the unnecessary empty call for an all-hot or all-cold step.
    """

    def __init__(self, fp8_backend, low_backend, moe, lowbit_format="awq"):
        if lowbit_format != "awq" or low_backend.metadata.get("requested_backend") != "marlin":
            raise NotImplementedError("Expert split currently supports calibrated AWQ Marlin only; NVFP4 is unvalidated")
        from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
        self.fp8_backend, self.low_backend, self.moe = fp8_backend, low_backend, moe
        t = fp8_backend.tensors
        self.quant_config = fp8_w8a8_moe_quant_config(t["w1_scale"], t["w2_scale"])
        self.fp8_experts = TritonExperts(moe_config=moe, quant_config=self.quant_config)
        self.identity = torch.arange(moe.num_experts, dtype=torch.int32, device=t["w1"].device)
        self.resident_metadata_bytes = self.identity.numel() * self.identity.element_size()

    def maps(self, hot_experts):
        if hot_experts.shape != self.identity.shape or hot_experts.dtype != torch.bool:
            raise ValueError("Expected an E-element boolean hot-expert mask")
        return (torch.where(hot_experts, self.identity, -1),
                torch.where(hot_experts, -1, self.identity))

    def fp8_masked(self, x, scores, ids, expert_map):
        from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
        t, qc = self.fp8_backend.tensors, self.quant_config
        quant_x, input_scale = moe_kernel_quantize_input(x, None, qc.quant_dtype, False)
        m, k = x.shape
        topk, n = ids.shape[1], t["w1"].shape[1]
        # TritonExperts uses the same workspace2 for GEMM1 and GEMM2, while
        # workspace13 holds the activated intermediate. Every masked token row
        # is explicitly zero-written by native GEMM early-return branches.
        workspace13 = torch.empty((m, topk, max(n // 2, k)), dtype=x.dtype, device=x.device)
        workspace2 = torch.empty((m, topk, max(n, k)), dtype=x.dtype, device=x.device)
        output = torch.empty_like(x)
        self.fp8_experts.apply(output, quant_x, t["w1"], t["w2"], scores, ids,
            MoEActivation.SILU, self.moe.num_experts, expert_map, input_scale, None,
            workspace13, workspace2, None, False)
        return output

    def low_masked(self, x, scores, ids, expert_map):
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.scalar_type import scalar_types
        t, meta = self.low_backend.tensors, self.low_backend.metadata
        quant_type = meta.get("quant_type", "uint4b8" if meta.get("symmetric") else "uint4")
        if quant_type not in ("uint4", "uint4b8"):
            raise ValueError("Expected AWQ int4 scalar type")
        optional = ("g_idx1", "g_idx2", "sort_indices1", "sort_indices2")
        extra = {key: t[key] for key in optional if key in t}
        if quant_type == "uint4":
            extra.update(w1_zeros=t["w1_zeros"], w2_zeros=t["w2_zeros"])
        return fused_marlin_moe(x, t["w1"], t["w2"], None, None,
            t["w1_scale"], t["w2_scale"], scores, ids,
            getattr(scalar_types, quant_type).id,
            global_num_experts=self.moe.num_experts, expert_map=expert_map,
            input_dtype=None, **extra)

    def apply_maps(self, x, scores, ids, fp8_map, low_map):
        return self.fp8_masked(x, scores, ids, fp8_map) + self.low_masked(x, scores, ids, low_map)


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
        policy: Literal["threshold", "phase", "expert_split"] = "threshold",
        nvfp4_backend: str = "marlin",
        device: str | torch.device | None = None,
        allow_unverified_source: bool = False,
        activation_basis_id: str | None = None,
        mechanism_only: bool = False,
        record_limit: int = 1024,
        record_cuda_events: bool = False,
        validate_route_values: bool = False,
    ):
        super().__init__(moe)
        self._validate_config()
        if policy not in ("threshold", "phase", "expert_split"):
            raise ValueError("policy must be threshold, phase, or expert_split")
        if policy == "expert_split" and lowbit_format != "awq":
            raise NotImplementedError("Expert split currently supports FP8 Triton + AWQ Marlin only")
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
        self.activation_basis_contract = _activation_basis_contract(
            fp8.metadata, low.metadata, activation_basis_id,
            mechanism_only=mechanism_only,
        )
        self.mechanism_only = mechanism_only
        left, right = _base_identity(fp8.metadata), _base_identity(low.metadata)
        # This checks lineage only. Even identical source weights do not prove
        # input-coordinate compatibility after quantization transformations.
        if left is None and right is None:
            a = (fp8.metadata["checkpoint"], fp8.metadata["checkpoint_revision"])
            b = (low.metadata["checkpoint"], low.metadata["checkpoint_revision"])
            if a == b and _known_revision(a[1]):
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
        self.expert_split = NativeExpertSplit(fp8, low, moe, lowbit_format) if policy == "expert_split" else None
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
        max_tokens = hot_experts_count = active_experts_count = None
        split_maps = None
        if self.policy == "phase" and phase != "mixed":
            chosen = "fp8" if phase == "prefill" else self.lowbit_format
        else:
            counts = torch.bincount(topk_ids.reshape(-1).long(),
                                    minlength=self.shape.experts)
            if counts.numel() != self.shape.experts:
                raise ValueError("Routing ID exceeds the number of experts")
            max_tokens = int(counts.max().item())  # CUDA -> host wait INCLUDED.
            if self.policy == "expert_split":
                hot = counts >= self.threshold
                hot_experts_count = int(hot.sum().item())
                active_experts_count = int((counts > 0).sum().item())
                if hot_experts_count == 0:
                    chosen = self.lowbit_format
                elif hot_experts_count == active_experts_count:
                    chosen = "fp8"
                else:
                    chosen = "expert_split"
                    split_maps = self.expert_split.maps(hot)
            else:
                chosen = "fp8" if max_tokens >= self.threshold else self.lowbit_format
        select_end = perf_counter_ns()
        if chosen == "expert_split":
            output = self.expert_split.apply_maps(x, topk_weights, topk_ids, *split_maps)
        else:
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
            "phase_source": _phase_source.get(),
            "mechanism_only": self.mechanism_only,
            "activation_basis_contract_matched": self.activation_basis_contract["activation_basis_contract_matched"],
            "input_tokens": x.shape[0], "max_expert_tokens": max_tokens,
            "hot_experts": hot_experts_count,
            "active_experts": active_experts_count,
            "fused_expert_calls": 2 if chosen == "expert_split" else 1,
            "weight_transfer_bytes": 0,
            "expert_map_bytes": 2 * self.shape.experts * 4 if split_maps is not None else 0,
            "switched": switched, "selection_host_ns": selection_ns,
            "dispatch_host_ns": dispatch_ns,
            "selection_has_host_sync": max_tokens is not None or self.validate_route_values,
            "cuda_events": (start, end) if start is not None else None,
        })
        return output

    def reset_statistics(self):
        """Start a fresh measurement after startup; resident weights are intact."""
        self.records.clear()
        self.selection_counts.clear()
        self.switch_count = self.calls = 0
        self._previous_format = None
        self.selection_host_ns_total = self.dispatch_host_ns_total = 0

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
            "expert_split_implemented": True,
            "expert_split_runtime_validation": "run validate_expert_split.py on this installed kernel build",
            "expert_split_supported": ["FP8 TritonExperts native expert_map", "AWQ Marlin native expert_map"],
            "expert_split_entrypoints": [
                "vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts.apply",
                "vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe"],
            "expert_split_mask_contract": "valid global topk_ids; identity/-1 expert_map; no router-weight-only masking",
            "expert_split_fp8_quantization_note": "dynamic second-activation tensor scale is recomputed on the selected expert subset",
            "expert_split_weight_transfer_bytes_per_step": 0,
            "expert_split_resident_metadata_bytes": self.expert_split.resident_metadata_bytes if self.expert_split else 0,
            "expert_split_costs": "both full weight sets resident; 2 expert maps; 2 fused calls; full-shape activation workspaces and final sum",
            "shape": asdict(self.shape), "policy": self.policy,
            "threshold": self.threshold, "device": str(self.device),
            "same_base_verified": self.same_base_verified,
            "same_base_verified_scope": "checkpoint lineage only; does not establish activation basis or numerical semantics",
            **self.activation_basis_contract,
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

    Pass activation_basis_id for the verified input basis of this worker layer;
    both artifact manifests must match it. Missing/mismatched IDs are rejected
    before the method is installed. An explicit mechanism_only=True permits
    diagnostics with no numerical semantics or quality claim; it is not a
    production installation. There is currently no low-bit input adapter.
    """
    return _install_resident_method(routed_experts, DualFormatMoEMethod,
                                    release_original_weights=release_original_weights, **kwargs)


class SingleFormatMoEMethod(DualFormatMoEMethod):
    """One real resident artifact, with no second arm or format selection.

    Inherit only configuration/registration/statistics utilities. Deliberately
    do not call DualFormatMoEMethod.__init__: it would allocate both formats.
    The same delegate entrypoints are used by the dual method for fair baselines.
    """

    def __init__(self, moe, *, weights, single_format, nvfp4_backend="cutlass",
                 device=None, allow_unverified_source=False, activation_basis_id=None,
                 mechanism_only=False, record_limit=1024, record_cuda_events=False,
                 validate_route_values=False):
        FusedMoEMethodBase.__init__(self, moe)
        self._validate_config()
        if single_format not in ("bf16", "fp8", "awq", "nvfp4") or not weights:
            raise ValueError("Single policy requires one real BF16/FP8/AWQ/NVFP4 artifact")
        if record_limit < 0:
            raise ValueError("record_limit cannot be negative")
        if allow_unverified_source and not mechanism_only:
            raise ValueError("Unknown source lineage is restricted to mechanism-only diagnostics")
        self.shape = ExpertShape(moe.hidden_dim, moe.intermediate_size_per_partition,
                                 moe.num_experts, moe.experts_per_token)
        self.shape.validate()
        self.device = torch.device(device if device is not None else moe.device)
        if self.device.type != "cuda":
            raise ValueError("This prototype requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.policy, self.single_format = "single", single_format
        self.record_cuda_events, self.validate_route_values = record_cuda_events, validate_route_values
        self.records = deque(maxlen=record_limit)
        self.selection_counts = Counter()
        self.switch_count = self.calls = 0
        self._previous_format = self._layer = None
        self.selection_host_ns_total = self.dispatch_host_ns_total = 0
        self.route_validation_host_ns_total = 0
        self.unregistered_original_tensor_bytes = self.retained_original_bytes = 0
        self.mechanism_only = mechanism_only
        with torch.random.fork_rng(devices=[self.device.index]):
            backend = build_backend(single_format, self.shape, device=str(self.device), weights=weights,
                                    nvfp4_backend=nvfp4_backend, max_tokens=moe.max_num_tokens)
        metadata = backend.metadata
        if metadata.get("weight_source", "").startswith("synthetic"):
            raise ValueError("Single-format serving requires an actual checkpoint artifact")
        contract = _activation_basis_contract(metadata, metadata, activation_basis_id,
                                               mechanism_only=mechanism_only)
        contract["activation_basis_ids"] = {
            "single": contract["activation_basis_ids"]["fp8"],
            "worker_input": contract["activation_basis_ids"]["worker_input"]}
        self.activation_basis_contract = contract
        self.base_identity = _base_identity(metadata)
        if self.base_identity is None and not allow_unverified_source:
            raise ValueError("Single-format artifact must retain its verified original base identity")
        self.backends = {single_format: backend}
        self.resident_weights = _ResidentWeights(self.backends)
        self.moe_quant_config = FusedMoEQuantConfig.make()

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts=None, shared_experts_input=None):
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Single-format comparison uses the same eager runtime as the dual prototype")
        if self._layer is None or self._layer() is not layer:
            raise RuntimeError("Install the single-format method on this layer first")
        if x.device != self.device or x.dtype != torch.bfloat16:
            raise ValueError("Input must be BF16 on the artifact installation device")
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
            raise ValueError("Input exceeds installed max_num_tokens")
        if not x.shape[0]:
            return torch.empty_like(x)
        start = end = None
        if self.record_cuda_events:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
        validation_ns = 0
        if self.validate_route_values:
            validation_start = perf_counter_ns()
            bounds = torch.stack(torch.aminmax(topk_ids)).tolist()
            if bounds[0] < 0 or bounds[1] >= self.shape.experts:
                raise ValueError("Routing contains invalid or masked expert IDs")
            validation_ns = perf_counter_ns() - validation_start
        dispatch_start = perf_counter_ns()
        output = self.backends[self.single_format](x, topk_weights, topk_ids)
        dispatch_ns = perf_counter_ns() - dispatch_start
        if end is not None:
            end.record()
        self.calls += 1
        self.selection_counts[self.single_format] += 1
        self.dispatch_host_ns_total += dispatch_ns
        self.route_validation_host_ns_total += validation_ns
        self.records.append({"call": self.calls, "phase": None, "phase_source": "single_policy_no_phase_collection",
            "format": self.single_format, "mechanism_only": self.mechanism_only,
            "activation_basis_contract_matched": self.activation_basis_contract["activation_basis_contract_matched"],
            "input_tokens": x.shape[0], "max_expert_tokens": None, "hot_experts": None, "active_experts": None,
            "fused_expert_calls": 1, "weight_transfer_bytes": 0, "expert_map_bytes": 0, "switched": False,
            "selection_host_ns": 0, "dispatch_host_ns": dispatch_ns,
            "route_validation_host_ns": validation_ns, "selection_has_host_sync": False,
            "route_validation_has_host_sync": self.validate_route_values,
            "cuda_events": (start, end) if start is not None else None})
        return output

    def reset_statistics(self):
        super().reset_statistics()
        self.route_validation_host_ns_total = 0

    def summary(self):
        return {"implementation": "vllm0.26_post_load_eager_single_format_prototype",
            "serving_integration_validated": False, "shape": asdict(self.shape),
            "policy": "single", "single_format": self.single_format, "device": str(self.device),
            "resident_expert_weight_sets": 1, "format_selection_implemented": False,
            "histogram_computed": False, "base_identity": self.base_identity,
            **self.activation_basis_contract,
            "calls": self.calls, "selection_counts": dict(self.selection_counts), "switch_count": 0,
            "selection_host_ns_total": 0, "dispatch_host_ns_total": self.dispatch_host_ns_total,
            "route_validation_host_ns_total": self.route_validation_host_ns_total,
            "threshold_selection_includes_cuda_host_wait": False,
            "record_cuda_events": self.record_cuda_events, "validate_route_values": self.validate_route_values,
            "resident_weight_and_scale_bytes": self.backends[self.single_format].metadata["resident_weight_and_scale_bytes"],
            "unregistered_original_tensor_bytes": self.unregistered_original_tensor_bytes,
            "retained_original_bytes": self.retained_original_bytes,
            "memory_accounting": "one artifact's logical tensor bytes; worker measures actual net CUDA allocation before KV sizing",
            "backends": {self.single_format: dict(self.backends[self.single_format].metadata)}}


def install_single_format(routed_experts, *, release_original_weights=True, **kwargs):
    """Install exactly one sidecar and retire original expert tensors before KV sizing."""
    if release_original_weights is not True:
        raise ValueError("A single-residency baseline must release original expert weights")
    return _install_resident_method(routed_experts, SingleFormatMoEMethod,
                                    release_original_weights=True, **kwargs)


def _install_resident_method(routed_experts, method_class, *, release_original_weights, **kwargs):
    """Shared post-load installation; selection and backend behavior stay in each method."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    if not isinstance(routed_experts, RoutedExperts):
        raise TypeError("Expected actual vLLM RoutedExperts, not a legacy FusedMoE layer")
    if isinstance(routed_experts.quant_method, DualFormatMoEMethod):
        raise ValueError("Layer is already patched")
    method = method_class(routed_experts.moe_config, **kwargs)
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
