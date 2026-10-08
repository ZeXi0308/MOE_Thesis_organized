# SPDX-License-Identifier: Apache-2.0
# Native dispatch sequence adapted from vLLM v0.26.0 fused_experts_impl.
# Copyright contributors to the vLLM project.
"""Preserve token/top-k contribution slots across expert residency waves.

Only unquantized BF16 SiLU, no bias, single GPU, eager execution. Each top-k
slot must belong to exactly one wave. Existing native matmuls/activation and
moe_sum are used. This is a small execution probe, not a novelty claim.
"""
import importlib


class DeferredExpertReduction:
    def __init__(self, x, weights, ids, w1, w2, *, activation,
                 global_num_experts, apply_router_weight_on_input=False, config=None):
        import torch
        from vllm.triton_utils import tl
        self.fm = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
        self.torch, self.compute_type = torch, tl.bfloat16
        if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.ndim != 2:
            raise ValueError("requires contiguous BF16 hidden states")
        if weights.shape != ids.shape or ids.shape[0] != x.shape[0]:
            raise ValueError("top-k row identity differs")
        if activation != self.fm.MoEActivation.SILU:
            raise ValueError("only SiLU is qualified")
        if w1.dtype != x.dtype or w2.dtype != x.dtype or w1.shape[2] != x.shape[1]:
            raise ValueError("invalid BF16 expert weights")
        self.x, self.weights, self.ids = x, weights, ids
        self.activation = activation
        self.global_num_experts = global_num_experts
        self.apply_input = apply_router_weight_on_input
        self.m, self.k = ids.shape
        self.n, self.h = w1.shape[1], w2.shape[1]
        self.config = config or self.fm.try_get_optimal_moe_config(
            w1.size(), w2.size(), self.k,
            self.fm._get_config_dtype_str(dtype=x.dtype), self.m, block_shape=None)
        self.fc1 = torch.empty((self.m, self.k, self.n), device=x.device, dtype=x.dtype)
        self.act = torch.empty((self.m*self.k, self.n//2), device=x.device, dtype=x.dtype)
        # FC1 and FC2 cannot alias: a later wave must preserve earlier FC2 slots.
        self.contributions = torch.empty((self.m, self.k, self.h), device=x.device, dtype=x.dtype)
        self.output = torch.empty_like(x)
        self.seen_experts = set()
        self.wave_count = 0
        self.workspace_bytes = sum(t.numel()*t.element_size() for t in
                                   (self.fc1, self.act, self.contributions, self.output))

    def wave(self, w1, w2, expert_map, executed_experts):
        # Host plan is already available in the pager; no added route D2H here.
        current = set(executed_experts)
        if not current or current & self.seen_experts:
            raise ValueError("empty or duplicate executed expert wave")
        self.seen_experts.update(current)
        fm = self.fm
        sorted_ids, expert_ids, padded = fm._prepare_expert_assignment(
            self.ids, self.config, self.m, self.k, self.global_num_experts,
            expert_map, ignore_invalid_experts=True)
        # ignore_invalid_experts=True excludes other waves from assignment.
        # No FC2 zero_: earlier token/top-k slots remain intact.
        fm.dispatch_fused_moe_kernel(
            self.x, w1, self.fc1, None, None, None, self.weights,
            sorted_ids, expert_ids, padded, self.apply_input, self.k,
            self.config, compute_type=self.compute_type, use_fp8_w8a8=False,
            use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False, block_shape=None, B_bias=None)
        fm.apply_moe_activation(self.activation, self.act, self.fc1.view(-1, self.n))
        fm.dispatch_fused_moe_kernel(
            self.act, w2, self.contributions, None, None, None, self.weights,
            sorted_ids, expert_ids, padded, not self.apply_input, 1,
            self.config, compute_type=self.compute_type, use_fp8_w8a8=False,
            use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False, block_shape=None, B_bias=None)
        self.wave_count += 1

    def finish(self, active_experts):
        if self.seen_experts != set(active_experts):
            raise ValueError("missing/extra expert contributions before final reduction")
        self.fm.ops.moe_sum(self.contributions, self.output)
        return self.output
