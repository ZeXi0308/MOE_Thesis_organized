"""Mac-side survival experiments for Graceful EP and QTree-EP.

This is an end-to-end fake-quant / tail-miss quality experiment on OLMoE, plus
a logical combine-wire accounting proxy.  It does NOT measure RDMA priority,
kernel latency, TPOT, or P99.

Graceful EP:
  * normal layers use FP8 head + MXFP4 tail;
  * congested layers deterministically miss a configurable fraction of tokens'
    lowest-rank contributions;
  * compare raw miss with gate-mass-preserving renormalization.

QTree-EP:
  * compare per-expert quantization against exact owner-node partial sums that
    are quantized only before the next topology level;
  * include uniform-FP8, rank-segmented FP8/MXFP4, and a double-quantization
    anti-control.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fake_quant import apply_precision
from metrics import MetricAccumulator
from modeling import load_model, load_tokenizer
from prompts import get_prompts


@dataclass(frozen=True)
class Strategy:
    name: str
    family: str
    mode: str
    congested_start: int = 0
    congested_layers: int = 0
    miss_tail_count: int = 0
    miss_token_fraction: float = 0.0
    renorm: bool = False
    alpha_compensation: bool = False
    delivery_semantics: str = "normal"
    ep_size: int = 8
    gpus_per_node: int = 4


@dataclass
class WireStats:
    # All wire fields below are combine-only.  ``bytes``/``vectors`` count
    # only owner-node -> token-origin-node traffic; local-node contributions
    # are deliberately excluded.
    bytes: float = 0.0
    vectors: int = 0
    dropped_vectors: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--dataset", default="wikitext2_docs")
    p.add_argument("--dataset-split", default="test")
    p.add_argument("--calibration-split", default="validation")
    p.add_argument("--test-samples", type=int, default=16)
    p.add_argument("--test-offset", type=int, default=0)
    p.add_argument("--calibration-samples", type=int, default=8)
    p.add_argument("--calibration-offset", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument(
        "--strategy-set",
        choices=("finalists", "all"),
        default="finalists",
        help="Run pre-registered primary strategies or the full exploratory sweep.",
    )
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--output-dir",
        default="experiments/idea_a_mac/outputs/paper_validation/"
        "graceful_qtree_survival_2026-07-13",
    )
    return p.parse_args()


def vector_wire_bytes(precision: str, hidden: int) -> int:
    if precision in ("bf16", "full"):
        return 2 * hidden
    if precision == "fp8":
        # fp8_e4m3_quant_dequant uses one FP32 scale per row.
        return hidden + 4
    if precision == "mxfp4":
        return math.ceil(hidden / 2) + math.ceil(hidden / 32)
    raise ValueError(precision)


def build_strategies(num_layers: int, top_k: int) -> list[Strategy]:
    n_tail = top_k // 2
    strategies = [
        Strategy("full_bf16", "baseline", "full"),
        Strategy("uniform_fp8_per_expert", "baseline", "uniform_fp8"),
        Strategy(
            "uniform_fp8_per_expert_ep16",
            "baseline",
            "uniform_fp8",
            ep_size=16,
            gpus_per_node=4,
        ),
        Strategy("r_layout_fp8head_mxfp4tail", "baseline", "r_layout"),
    ]
    graceful_cases = [
        # Four consecutive early layers: scan congestion prevalence.
        (0, 4, n_tail, 0.25, "early4_tailall_tok25"),
        (0, 4, n_tail, 0.50, "early4_tailall_tok50"),
        (0, 4, n_tail, 1.00, "early4_tailall_tok100"),
        # Longer congestion window.
        (0, 8, n_tail, 0.50, "early8_tailall_tok50"),
        (0, 8, n_tail, 1.00, "early8_tailall_tok100"),
        # Finer progressive completion: only the last two ranks miss.
        (0, 4, min(2, n_tail), 1.00, "early4_last2_tok100"),
        # Layer-position control.
        (max(0, num_layers - 4), 4, n_tail, 1.00, "late4_tailall_tok100"),
    ]
    for start, count, tail_count, fraction, label in graceful_cases:
        for renorm in (False, True):
            suffix = "mass_renorm" if renorm else "raw"
            strategies.append(
                Strategy(
                    f"graceful_{label}_{suffix}",
                    "graceful",
                    "graceful",
                    congested_start=start,
                    congested_layers=count,
                    miss_tail_count=tail_count,
                    miss_token_fraction=fraction,
                    renorm=renorm,
                    delivery_semantics="sender_cancel",
                )
            )
    # Same numerical miss as the mildest sender-cancel case, but the late
    # vectors have already traversed the network.  This separates quality loss
    # from wire saving and prevents deadline misses from being counted as free.
    strategies.append(
        Strategy(
            "graceful_early4_tailall_tok25_receiver_ignore_late",
            "graceful",
            "graceful",
            congested_start=0,
            congested_layers=4,
            miss_tail_count=n_tail,
            miss_token_fraction=0.25,
            delivery_semantics="receiver_ignore_late",
        )
    )
    # Follow-up after raw/renorm screening: a scalar fitted on disjoint
    # calibration prompts rescales the received partial combine.  It adds no
    # wire bytes and tests whether a minimal compensation can widen the safe
    # region without a learned neural module.
    alpha_cases = [
        (0, 4, n_tail, 0.25, "early4_tailall_tok25"),
        (0, 4, n_tail, 0.50, "early4_tailall_tok50"),
        (0, 8, n_tail, 0.50, "early8_tailall_tok50"),
        (0, 4, min(2, n_tail), 1.00, "early4_last2_tok100"),
        (max(0, num_layers - 4), 4, n_tail, 1.00, "late4_tailall_tok100"),
    ]
    for start, count, tail_count, fraction, label in alpha_cases:
        strategies.append(
            Strategy(
                f"graceful_{label}_calibrated_alpha",
                "graceful",
                "graceful",
                congested_start=start,
                congested_layers=count,
                miss_tail_count=tail_count,
                miss_token_fraction=fraction,
                alpha_compensation=True,
                delivery_semantics="sender_cancel",
            )
        )
    for ep_size, gpus_per_node in ((8, 4), (16, 4)):
        strategies.extend(
            [
                Strategy(
                    f"qtree_ep{ep_size}_node_uniform_fp8",
                    "qtree",
                    "qtree_uniform",
                    ep_size=ep_size,
                    gpus_per_node=gpus_per_node,
                ),
                Strategy(
                    f"qtree_ep{ep_size}_node_uniform_mxfp4",
                    "qtree",
                    "qtree_uniform_mxfp4",
                    ep_size=ep_size,
                    gpus_per_node=gpus_per_node,
                ),
                Strategy(
                    f"qtree_ep{ep_size}_node_critical_single",
                    "qtree",
                    "qtree_critical_single",
                    ep_size=ep_size,
                    gpus_per_node=gpus_per_node,
                ),
                Strategy(
                    f"qtree_ep{ep_size}_node_two_lane",
                    "qtree",
                    "qtree_two_lane",
                    ep_size=ep_size,
                    gpus_per_node=gpus_per_node,
                ),
            ]
        )
    strategies.append(
        Strategy(
            "qtree_ep8_node_two_lane_double_quant",
            "qtree_control",
            "qtree_two_lane_double",
            ep_size=8,
            gpus_per_node=4,
        )
    )
    return strategies


def select_strategies(strategies: list[Strategy], strategy_set: str) -> list[Strategy]:
    if strategy_set == "all":
        return strategies
    finalists = {
        "full_bf16",
        "uniform_fp8_per_expert",
        "uniform_fp8_per_expert_ep16",
        "r_layout_fp8head_mxfp4tail",
        "graceful_early4_tailall_tok25_raw",
        "graceful_early4_tailall_tok25_mass_renorm",
        "graceful_early4_tailall_tok50_raw",
        "graceful_late4_tailall_tok100_raw",
        "graceful_early4_tailall_tok25_receiver_ignore_late",
        "qtree_ep8_node_uniform_fp8",
        "qtree_ep8_node_uniform_mxfp4",
        "qtree_ep8_node_critical_single",
        "qtree_ep8_node_two_lane",
        "qtree_ep16_node_uniform_fp8",
        "qtree_ep16_node_uniform_mxfp4",
        "qtree_ep16_node_critical_single",
        "qtree_ep16_node_two_lane",
    }
    selected = [strategy for strategy in strategies if strategy.name in finalists]
    missing = finalists - {strategy.name for strategy in selected}
    if missing:
        raise RuntimeError(f"missing finalist strategies: {sorted(missing)}")
    return selected


def _raw_expert_outputs(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(flat)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(flat.dtype)

    raw = torch.zeros(
        (flat.shape[0], self.top_k, hidden_dim),
        dtype=flat.dtype,
        device=flat.device,
    )
    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        rank_idx, token_idx = torch.where(expert_mask[expert_idx])
        current = flat[None, token_idx].reshape(-1, hidden_dim)
        raw[token_idx, rank_idx, :] = self.experts[expert_idx](current)
    return (
        batch_size,
        sequence_length,
        hidden_dim,
        router_logits,
        routing_weights,
        selected_experts,
        raw,
    )


def _r_layout_outputs(raw: torch.Tensor, top_k: int) -> torch.Tensor:
    n_head = top_k // 2
    out = apply_precision(raw, "fp8")
    for rank_idx in range(n_head, top_k):
        out[:, rank_idx, :] = apply_precision(raw[:, rank_idx, :], "mxfp4")
    return out


def _combine_expert_order(
    outputs: torch.Tensor,
    weights: torch.Tensor,
    selected: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Match the pretrained OLMoE block's BF16 accumulation order.

    The Transformers implementation visits experts in expert-id order and uses
    ``index_add_`` for each expert.  A simple ``sum(dim=1)`` changes BF16
    associativity, perturbs logits, and can alter downstream routing.  Keeping
    this order makes the unquantized patched path a valid reference.
    """
    total_tokens, _top_k, hidden = outputs.shape
    final = torch.zeros(
        (total_tokens, hidden), dtype=outputs.dtype, device=outputs.device
    )
    expert_mask = F.one_hot(selected, num_classes=num_experts).permute(2, 1, 0)
    for expert_idx in range(num_experts):
        rank_idx, token_idx = torch.where(expert_mask[expert_idx])
        current = outputs[token_idx, rank_idx, :] * weights[
            token_idx, rank_idx, None
        ]
        final.index_add_(0, token_idx, current.to(outputs.dtype))
    return final


def _deterministic_token_mask(
    total_tokens: int,
    fraction: float,
    sample_id: int,
    layer_id: int,
    device: torch.device,
) -> torch.Tensor:
    if fraction <= 0:
        return torch.zeros(total_tokens, dtype=torch.bool, device=device)
    if fraction >= 1:
        return torch.ones(total_tokens, dtype=torch.bool, device=device)
    idx = torch.arange(total_tokens, dtype=torch.int64, device=device)
    hashed = (
        idx * 1103515245
        + int(sample_id) * 12345
        + int(layer_id) * 2654435761
        + 1013904223
    ) % 10000
    return hashed < int(round(fraction * 10000))


def _token_origin_node(
    total_tokens: int,
    sample_id: int,
    ep_size: int,
    gpus_per_node: int,
    device: torch.device,
) -> torch.Tensor:
    """Assign a stable synthetic token origin for inter-node accounting.

    The assignment is invariant across layers and strategies.  It is still a
    proxy, but unlike the old accounting it distinguishes local-node combine
    from traffic that must cross a slow inter-node link.
    """
    idx = torch.arange(total_tokens, dtype=torch.int64, device=device)
    origin_rank = (
        idx * 1103515245 + int(sample_id) * 2654435761 + 1013904223
    ) % ep_size
    return torch.div(origin_rank, gpus_per_node, rounding_mode="floor")


def _add_per_expert_wire(
    stats: WireStats,
    selected: torch.Tensor,
    origin_node: torch.Tensor,
    hidden: int,
    ep_size: int,
    gpus_per_node: int,
    head_precision: str,
    tail_precision: str,
    drop_mask: torch.Tensor | None = None,
    count_dropped_as_transmitted: bool = False,
) -> None:
    total_tokens, top_k = selected.shape
    n_head = top_k // 2
    owner_rank = selected % ep_size
    owner_node = torch.div(owner_rank, gpus_per_node, rounding_mode="floor")
    remote = owner_node != origin_node[:, None]
    if drop_mask is None:
        drop_mask = torch.zeros_like(remote)
    for rank_idx in range(top_k):
        precision = head_precision if rank_idx < n_head else tail_precision
        remote_rank = remote[:, rank_idx]
        dropped_rank = remote_rank & drop_mask[:, rank_idx]
        transmitted = remote_rank & (
            torch.ones_like(remote_rank)
            if count_dropped_as_transmitted
            else ~drop_mask[:, rank_idx]
        )
        count = int(transmitted.sum().item())
        dropped = int(dropped_rank.sum().item())
        stats.bytes += count * vector_wire_bytes(precision, hidden)
        stats.vectors += count
        stats.dropped_vectors += dropped


def _node_partial_sum(
    raw: torch.Tensor,
    weights: torch.Tensor,
    selected: torch.Tensor,
    strategy: Strategy,
) -> torch.Tensor:
    total_tokens, top_k, hidden = raw.shape
    n_head = top_k // 2
    owner_rank = selected % strategy.ep_size
    owner_node = torch.div(owner_rank, strategy.gpus_per_node, rounding_mode="floor")
    num_nodes = strategy.ep_size // strategy.gpus_per_node
    final = torch.zeros((total_tokens, hidden), dtype=raw.dtype, device=raw.device)

    if strategy.mode == "qtree_two_lane_double":
        source = _r_layout_outputs(raw, top_k)
    else:
        source = raw
    contrib = source * weights[:, :, None]

    for node in range(num_nodes):
        node_mask = owner_node == node
        if strategy.mode in (
            "qtree_uniform",
            "qtree_uniform_mxfp4",
            "qtree_critical_single",
        ):
            active = node_mask.any(dim=1)
            if not active.any():
                continue
            partial = (contrib * node_mask[:, :, None]).sum(dim=1)
            if strategy.mode == "qtree_uniform_mxfp4":
                final[active] += apply_precision(partial[active], "mxfp4")
            elif strategy.mode == "qtree_critical_single":
                # Exactly one partial per active node.  A partial containing at
                # least one head-rank contribution uses FP8; a tail-only partial
                # uses MXFP4.  This avoids the duplicate vectors of two lanes.
                has_head = node_mask[:, :n_head].any(dim=1)
                head_active = active & has_head
                tail_only = active & ~has_head
                if head_active.any():
                    final[head_active] += apply_precision(partial[head_active], "fp8")
                if tail_only.any():
                    final[tail_only] += apply_precision(partial[tail_only], "mxfp4")
            else:
                final[active] += apply_precision(partial[active], "fp8")
            continue

        for start, end, precision in (
            (0, n_head, "fp8"),
            (n_head, top_k, "mxfp4"),
        ):
            lane_mask = node_mask[:, start:end]
            active = lane_mask.any(dim=1)
            if not active.any():
                continue
            partial = (
                contrib[:, start:end, :] * lane_mask[:, :, None]
            ).sum(dim=1)
            final[active] += apply_precision(partial[active], precision)
    return final


def _account_qtree_wire(
    selected: torch.Tensor,
    origin_node: torch.Tensor,
    strategy: Strategy,
    hidden: int,
    stats: WireStats,
) -> None:
    """Count frozen-route inter-node combine bytes for a QTree strategy."""
    _total_tokens, top_k = selected.shape
    n_head = top_k // 2
    owner_rank = selected % strategy.ep_size
    owner_node = torch.div(owner_rank, strategy.gpus_per_node, rounding_mode="floor")
    num_nodes = strategy.ep_size // strategy.gpus_per_node
    for node in range(num_nodes):
        node_mask = owner_node == node
        remote_node = origin_node != node
        active = node_mask.any(dim=1) & remote_node
        if strategy.mode in (
            "qtree_uniform",
            "qtree_uniform_mxfp4",
            "qtree_critical_single",
        ):
            if strategy.mode == "qtree_uniform_mxfp4":
                precision_counts = [("mxfp4", active)]
            elif strategy.mode == "qtree_critical_single":
                has_head = node_mask[:, :n_head].any(dim=1)
                precision_counts = [
                    ("fp8", active & has_head),
                    ("mxfp4", active & ~has_head),
                ]
            else:
                precision_counts = [("fp8", active)]
            for precision, mask in precision_counts:
                count = int(mask.sum().item())
                stats.bytes += count * vector_wire_bytes(precision, hidden)
                stats.vectors += count
            continue

        for start, end, precision in (
            (0, n_head, "fp8"),
            (n_head, top_k, "mxfp4"),
        ):
            lane_active = node_mask[:, start:end].any(dim=1) & remote_node
            count = int(lane_active.sum().item())
            stats.bytes += count * vector_wire_bytes(precision, hidden)
            stats.vectors += count


def _patched_forward(self, hidden_states: torch.Tensor):
    (
        batch_size,
        sequence_length,
        hidden_dim,
        router_logits,
        routing_weights,
        selected_experts,
        raw,
    ) = _raw_expert_outputs(self, hidden_states)
    strategy: Strategy = self._sg_strategy
    stats: WireStats = self._sg_wire_stats
    total_tokens = raw.shape[0]
    top_k = raw.shape[1]
    key = (self._sg_sample_id, self._sg_layer_id)
    if strategy.mode == "full":
        self._sg_frozen_routes[key] = selected_experts.detach().cpu().clone()
        wire_selected = selected_experts
    else:
        if key not in self._sg_frozen_routes:
            raise RuntimeError(f"missing frozen baseline route for {key}")
        wire_selected = self._sg_frozen_routes[key].to(selected_experts.device)
    origin_node = _token_origin_node(
        total_tokens,
        self._sg_sample_id,
        strategy.ep_size,
        strategy.gpus_per_node,
        raw.device,
    )

    if strategy.mode == "full":
        final = _combine_expert_order(
            raw, routing_weights, selected_experts, self.num_experts
        )
        _add_per_expert_wire(
            stats,
            wire_selected,
            origin_node,
            hidden_dim,
            strategy.ep_size,
            strategy.gpus_per_node,
            "bf16",
            "bf16",
        )
    elif strategy.mode == "uniform_fp8":
        out = apply_precision(raw, "fp8")
        final = _combine_expert_order(
            out, routing_weights, selected_experts, self.num_experts
        )
        _add_per_expert_wire(
            stats,
            wire_selected,
            origin_node,
            hidden_dim,
            strategy.ep_size,
            strategy.gpus_per_node,
            "fp8",
            "fp8",
        )
    elif strategy.mode == "r_layout":
        out = _r_layout_outputs(raw, top_k)
        final = _combine_expert_order(
            out, routing_weights, selected_experts, self.num_experts
        )
        _add_per_expert_wire(
            stats,
            wire_selected,
            origin_node,
            hidden_dim,
            strategy.ep_size,
            strategy.gpus_per_node,
            "fp8",
            "mxfp4",
        )
    elif strategy.mode == "graceful":
        out = _r_layout_outputs(raw, top_k)
        weights = routing_weights
        in_window = (
            strategy.congested_start
            <= self._sg_layer_id
            < strategy.congested_start + strategy.congested_layers
        )
        drop_mask = torch.zeros(
            (total_tokens, top_k), dtype=torch.bool, device=raw.device
        )
        missed_token_mask = torch.zeros(total_tokens, dtype=torch.bool, device=raw.device)
        if in_window:
            token_mask = _deterministic_token_mask(
                total_tokens,
                strategy.miss_token_fraction,
                self._sg_sample_id,
                self._sg_layer_id,
                raw.device,
            )
            missed_token_mask = token_mask
            first_missed = max(0, top_k - strategy.miss_tail_count)
            drop_mask[:, first_missed:] = token_mask[:, None]
            out = out.clone()
            out[drop_mask] = 0
            if strategy.renorm:
                weights = routing_weights.clone()
                original_mass = weights.sum(dim=-1, keepdim=True)
                weights[drop_mask] = 0
                kept_mass = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                weights = weights * (original_mass / kept_mass)
        final = _combine_expert_order(
            out, weights, selected_experts, self.num_experts
        )
        if strategy.alpha_compensation and in_window and missed_token_mask.any():
            alpha = float(self._sg_alphas[(self._sg_layer_id, strategy.miss_tail_count)])
            final[missed_token_mask] *= alpha
        _add_per_expert_wire(
            stats,
            wire_selected,
            origin_node,
            hidden_dim,
            strategy.ep_size,
            strategy.gpus_per_node,
            "fp8",
            "mxfp4",
            drop_mask=drop_mask,
            count_dropped_as_transmitted=(
                strategy.delivery_semantics == "receiver_ignore_late"
            ),
        )
    elif strategy.mode.startswith("qtree_"):
        final = _node_partial_sum(
            raw, routing_weights, selected_experts, strategy
        )
        _account_qtree_wire(
            wire_selected, origin_node, strategy, hidden_dim, stats
        )
    else:
        raise ValueError(strategy.mode)

    return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


def patch_model(model, strategy: Strategy, stats: WireStats) -> None:
    frozen_routes = getattr(model, "_sg_frozen_routes", {})
    model._sg_frozen_routes = frozen_routes
    for layer_id, layer in enumerate(model.model.layers):
        moe = layer.mlp
        if not (hasattr(moe, "experts") and hasattr(moe, "gate")):
            raise TypeError("this survival script currently supports OLMoE-style layers")
        moe._sg_strategy = strategy
        moe._sg_wire_stats = stats
        moe._sg_layer_id = layer_id
        moe._sg_sample_id = -1
        moe._sg_alphas = getattr(model, "_sg_alphas", {})
        moe._sg_frozen_routes = frozen_routes
        moe.forward = MethodType(_patched_forward, moe)


def _patched_alpha_calibration_forward(self, hidden_states: torch.Tensor):
    (
        batch_size,
        sequence_length,
        _hidden_dim,
        router_logits,
        routing_weights,
        _selected_experts,
        raw,
    ) = _raw_expert_outputs(self, hidden_states)
    target = _combine_expert_order(
        raw, routing_weights, _selected_experts, self.num_experts
    )
    normal = _r_layout_outputs(raw, raw.shape[1])
    for tail_count in self._sg_alpha_tail_counts:
        kept = normal.clone()
        kept[:, raw.shape[1] - tail_count :, :] = 0
        kept_final = _combine_expert_order(
            kept, routing_weights, _selected_experts, self.num_experts
        )
        key = (self._sg_layer_id, tail_count)
        self._sg_alpha_accum[key][0] += float(
            (kept_final.float() * target.float()).sum().item()
        )
        self._sg_alpha_accum[key][1] += float(
            kept_final.float().pow(2).sum().item()
        )
    # Preserve the unmodified trajectory during calibration.
    return target.reshape(batch_size, sequence_length, -1), router_logits


def calibrate_alphas(model, tokenizer, texts: list[str], seq_len: int, tail_counts: list[int]):
    accum = {
        (layer_id, tail_count): [0.0, 0.0]
        for layer_id in range(len(model.model.layers))
        for tail_count in tail_counts
    }
    for layer_id, layer in enumerate(model.model.layers):
        moe = layer.mlp
        moe._sg_layer_id = layer_id
        moe._sg_alpha_tail_counts = tail_counts
        moe._sg_alpha_accum = accum
        moe.forward = MethodType(_patched_alpha_calibration_forward, moe)
    for sample_id, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        with torch.no_grad():
            model(**inputs)
        print(f"  alpha calibration: {sample_id + 1}/{len(texts)}", flush=True)
    return {
        key: numerator / max(denominator, 1e-12)
        for key, (numerator, denominator) in accum.items()
    }


def validate_exact_full_path(model, tokenizer, text: str, seq_len: int) -> dict[str, float]:
    """Fail fast unless patched full combine reproduces the pretrained model."""
    original_forwards = [layer.mlp.forward for layer in model.model.layers]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    with torch.no_grad():
        original = model(**inputs).logits.detach().cpu()

    model._sg_frozen_routes = {}
    patch_model(model, Strategy("equivalence_full", "diagnostic", "full"), WireStats())
    for layer in model.model.layers:
        layer.mlp._sg_sample_id = -999
    with torch.no_grad():
        patched = model(**inputs).logits.detach().cpu()

    for layer, original_forward in zip(model.model.layers, original_forwards):
        layer.mlp.forward = original_forward
    model._sg_frozen_routes = {}

    diff = (patched.float() - original.float()).abs()
    result = {
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
    }
    # Exact equality is expected because both paths now preserve expert-order
    # BF16 index_add accumulation.  Do not silently run paper experiments on a
    # numerically different baseline.
    if not torch.equal(original, patched):
        raise RuntimeError(f"patched full path is not exact: {result}")
    return result


def data_manifest(tokenizer, texts: list[str], split: str, seq_len: int) -> list[dict]:
    rows = []
    for sample_id, text in enumerate(texts):
        token_count = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "title_prefix": text[:120],
                "characters": len(text),
                "tokens_before_truncation": token_count,
                "tokens_used": min(token_count, seq_len),
            }
        )
    return rows


def run_strategy(
    model,
    tokenizer,
    texts: list[str],
    seq_len: int,
    strategy: Strategy,
    baseline_logits: list[torch.Tensor] | None,
) -> tuple[MetricAccumulator, list[torch.Tensor], WireStats]:
    stats = WireStats()
    patch_model(model, strategy, stats)
    metrics = MetricAccumulator()
    logits_out: list[torch.Tensor] = []
    for sample_id, text in enumerate(texts):
        for layer in model.model.layers:
            layer.mlp._sg_sample_id = sample_id
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu()
        metrics.add(
            sample_id,
            logits,
            inputs["input_ids"],
            baseline_logits=(
                baseline_logits[sample_id] if baseline_logits is not None else None
            ),
            attention_mask=inputs.get("attention_mask"),
        )
        logits_out.append(logits)
        print(
            f"  {strategy.name}: {sample_id + 1}/{len(texts)}",
            flush=True,
        )
    return metrics, logits_out, stats


def paired_bootstrap(
    candidate: MetricAccumulator,
    reference: MetricAccumulator,
    n_bootstrap: int,
    seed: int = 20260713,
) -> dict[str, float]:
    if len(candidate.samples) != len(reference.samples):
        raise ValueError("paired samples differ")
    n = len(candidate.samples)

    def weighted_kl(rows, indices):
        return sum(rows[i].kl_sum for i in indices) / max(
            sum(rows[i].token_count for i in indices), 1
        )

    all_idx = np.arange(n)
    point = weighted_kl(candidate.samples, all_idx) - weighted_kl(
        reference.samples, all_idx
    )
    if n < 2 or n_bootstrap <= 0:
        return {"paired_kl_delta": point, "paired_ci_low": point, "paired_ci_high": point}
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values[b] = weighted_kl(candidate.samples, idx) - weighted_kl(
            reference.samples, idx
        )
    return {
        "paired_kl_delta": float(point),
        "paired_ci_low": float(np.quantile(values, 0.025)),
        "paired_ci_high": float(np.quantile(values, 0.975)),
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [header, sep]
    for _, row in df[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = get_prompts(
        args.dataset,
        args.test_samples,
        offset=args.test_offset,
        split=args.dataset_split,
    )
    calibration_texts = get_prompts(
        args.dataset,
        args.calibration_samples,
        offset=args.calibration_offset,
        split=args.calibration_split,
    )
    if set(texts) & set(calibration_texts):
        raise RuntimeError("calibration and test prompts overlap")
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline)
    model, load_seconds = load_model(
        args.model, dtype_name=args.dtype, local_files_only=args.offline
    )
    equivalence = validate_exact_full_path(
        model, tokenizer, calibration_texts[0], args.seq_len
    )
    num_layers = len(model.model.layers)
    top_k = int(model.config.num_experts_per_tok)
    hidden = int(model.config.hidden_size)
    strategies = select_strategies(
        build_strategies(num_layers, top_k), args.strategy_set
    )
    print(
        f"loaded in {load_seconds:.1f}s; layers={num_layers}; top_k={top_k}; "
        f"strategies={len(strategies)}",
        flush=True,
    )
    alpha_tail_counts = sorted(
        {strategy.miss_tail_count for strategy in strategies if strategy.alpha_compensation}
    )
    if alpha_tail_counts:
        print("calibrating layer/tail scalar compensation...", flush=True)
        alphas = calibrate_alphas(
            model, tokenizer, calibration_texts, args.seq_len, alpha_tail_counts
        )
    else:
        alphas = {}
    model._sg_alphas = alphas

    config = vars(args).copy()
    config.update(
        {
            "num_layers": num_layers,
            "top_k": top_k,
            "hidden_size": hidden,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "runtime_versions": {
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "datasets", "pandas", "numpy")
            },
            "baseline_equivalence": equivalence,
            "strategies": [asdict(s) for s in strategies],
            "calibrated_alphas": {
                f"layer{layer}_tail{tail}": value
                for (layer, tail), value in alphas.items()
            },
            "boundary": (
                "article-level fake-quant quality + frozen-route synthetic-origin "
                "inter-node combine wire proxy; no kernel/RDMA/TPOT/P99"
            ),
        }
    )
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "calibration": data_manifest(
            tokenizer, calibration_texts, args.calibration_split, args.seq_len
        ),
        "test": data_manifest(tokenizer, texts, args.dataset_split, args.seq_len),
    }
    (out / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metrics_by_name: dict[str, MetricAccumulator] = {}
    wire_by_name: dict[str, WireStats] = {}
    sample_rows: list[dict[str, float | int | str]] = []
    baseline_logits: list[torch.Tensor] | None = None

    for idx, strategy in enumerate(strategies):
        print(f"[{idx + 1}/{len(strategies)}] {strategy.name}", flush=True)
        metrics, logits, wire = run_strategy(
            model,
            tokenizer,
            texts,
            args.seq_len,
            strategy,
            baseline_logits if strategy.mode != "full" else None,
        )
        if strategy.mode == "full":
            baseline_logits = logits
        metrics_by_name[strategy.name] = metrics
        wire_by_name[strategy.name] = wire
        sample_rows.extend(metrics.sample_rows(strategy.name))
        pd.DataFrame(sample_rows).to_csv(out / "sample_metrics.partial.csv", index=False)

    full_wire = wire_by_name["full_bf16"].bytes
    fp8_wire = wire_by_name["uniform_fp8_per_expert"].bytes
    summary_rows = []
    for strategy in strategies:
        metrics = metrics_by_name[strategy.name]
        matching_fp8_name = (
            "uniform_fp8_per_expert_ep16"
            if strategy.ep_size == 16
            else "uniform_fp8_per_expert"
        )
        matching_fp8_wire = wire_by_name[matching_fp8_name].bytes
        node_fp8_name = f"qtree_ep{strategy.ep_size}_node_uniform_fp8"
        node_fp8_wire = (
            wire_by_name[node_fp8_name].bytes
            if node_fp8_name in wire_by_name
            else None
        )
        row = metrics.bootstrap_summary(args.bootstrap)
        row.update(
            {
                "strategy": strategy.name,
                "family": strategy.family,
                "corpus_ppl_delta_vs_full": metrics.corpus_ppl
                - metrics_by_name["full_bf16"].corpus_ppl,
                "logical_wire_bytes": wire_by_name[strategy.name].bytes,
                "logical_wire_vectors": wire_by_name[strategy.name].vectors,
                "dropped_vectors": wire_by_name[strategy.name].dropped_vectors,
                "saving_vs_ep8_per_expert_bf16_reference": 1.0
                - wire_by_name[strategy.name].bytes / max(full_wire, 1.0),
                "saving_vs_ep8_per_expert_fp8_reference": 1.0
                - wire_by_name[strategy.name].bytes / max(fp8_wire, 1.0),
                "saving_vs_matching_per_expert_fp8": 1.0
                - wire_by_name[strategy.name].bytes / max(matching_fp8_wire, 1.0),
                "saving_vs_matching_node_fp8": (
                    1.0
                    - wire_by_name[strategy.name].bytes / max(node_fp8_wire, 1.0)
                    if strategy.family in ("qtree", "qtree_control")
                    and node_fp8_wire is not None
                    else float("nan")
                ),
            }
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "survival_summary.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(out / "sample_metrics.csv", index=False)

    references = {
        "uniform_fp8_per_expert": "full_bf16",
        "uniform_fp8_per_expert_ep16": "full_bf16",
        "r_layout_fp8head_mxfp4tail": "uniform_fp8_per_expert",
    }
    for strategy in strategies:
        if strategy.family == "graceful":
            references[strategy.name] = "r_layout_fp8head_mxfp4tail"
        elif strategy.name.endswith("node_uniform_fp8"):
            references[strategy.name] = (
                "uniform_fp8_per_expert_ep16"
                if strategy.ep_size == 16
                else "uniform_fp8_per_expert"
            )
        elif strategy.name.endswith("node_uniform_mxfp4"):
            references[strategy.name] = strategy.name.replace(
                "node_uniform_mxfp4", "node_uniform_fp8"
            )
        elif strategy.name.endswith("node_critical_single"):
            references[strategy.name] = strategy.name.replace(
                "node_critical_single", "node_uniform_fp8"
            )
        elif strategy.name.endswith("node_two_lane"):
            references[strategy.name] = strategy.name.replace(
                "node_two_lane", "node_uniform_fp8"
            )
        elif strategy.mode == "qtree_two_lane_double":
            references[strategy.name] = "qtree_ep8_node_two_lane"
    comparison_rows = []
    for candidate, reference in references.items():
        paired = paired_bootstrap(
            metrics_by_name[candidate], metrics_by_name[reference], args.bootstrap
        )
        comparison_rows.append(
            {"candidate": candidate, "reference": reference, **paired}
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)

    main_columns = [
        "strategy",
        "mean_token_kl",
        "corpus_ppl_delta_vs_full",
        "saving_vs_matching_per_expert_fp8",
        "saving_vs_matching_node_fp8",
        "dropped_vectors",
    ]
    report = [
        "# Graceful EP + QTree-EP Mac 生死实验",
        "",
        "> 边界：这是按独立 WikiText 文档 bootstrap 的端到端 fake-quant / tail-miss 质量实验，以及 frozen-route、synthetic-origin 的跨节点 combine wire proxy；不是多 GPU kernel、RDMA priority、TPOT 或 P99 证据。",
        "",
        f"- model: `{args.model}`",
        f"- test: offset `{args.test_offset}`, samples `{args.test_samples}`, seq_len `{args.seq_len}`",
        f"- layers/top-k: `{num_layers}/{top_k}`",
        "",
        "## 汇总",
        "",
        markdown_table(summary, main_columns),
        "",
        "## Paired KL 差（candidate - reference）",
        "",
        markdown_table(
            comparisons,
            [
                "candidate",
                "reference",
                "paired_kl_delta",
                "paired_ci_low",
                "paired_ci_high",
            ],
        ),
        "",
        "## 解释边界",
        "",
        "1. Graceful sender_cancel 表示发送前取消；receiver_ignore_late 表示字节已传输但 combine 不再等待。两者不能互换解释。",
        "2. Wire bytes 只统计 synthetic token origin 之外的 frozen-route inter-node combine；未计 dispatch、节点内 reduction、packetization 或 kernel 开销。",
        "3. QTree 必须以 matching-EP hierarchical FP8 为主 baseline；proxy bytes 不能直接转化为 latency。",
    ]
    (out / "生死实验报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote results to {out}", flush=True)


if __name__ == "__main__":
    main()
