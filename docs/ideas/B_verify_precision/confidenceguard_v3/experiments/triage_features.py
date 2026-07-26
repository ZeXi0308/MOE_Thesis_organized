"""Frozen prefill features for TriageAudit v2.

The equations match the eight ``full_route_*`` summaries used by the prior
quality-isolation experiment; this module deliberately has no dependency on
that experiment's analysis stack.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class FeatureError(RuntimeError):
    pass


def extract_full_route_features(
    route_batches: Sequence[Mapping[str, object]], num_experts: int
) -> dict[str, float]:
    if type(num_experts) is not int or num_experts <= 0:
        raise FeatureError("num_experts must be a positive integer")
    ordered = sorted(route_batches, key=lambda batch: int(batch["layer"]))
    if not ordered:
        raise FeatureError("route recorder produced no batches")
    experts_by_layer: list[np.ndarray] = []
    weights_by_layer: list[np.ndarray] = []
    layers: list[int] = []
    for batch in ordered:
        selected = batch.get("selected_experts")
        weights = batch.get("routing_weights")
        if not isinstance(selected, torch.Tensor) or not isinstance(weights, torch.Tensor):
            raise FeatureError("route batch lacks tensor experts/weights")
        experts = selected.detach().cpu().numpy()
        probabilities = weights.detach().float().cpu().numpy()
        if experts.ndim != 2 or probabilities.shape != experts.shape or len(experts) == 0:
            raise FeatureError("invalid route tensor shapes")
        if np.any(experts < 0) or np.any(experts >= num_experts):
            raise FeatureError("expert id outside model range")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise FeatureError("routing weights must be finite and non-negative")
        layers.append(int(batch["layer"]))
        experts_by_layer.append(experts)
        weights_by_layer.append(probabilities)
    if len(set(layers)) != len(layers):
        raise FeatureError("duplicate route batch layer")
    weights = np.concatenate(weights_by_layer, axis=0)
    top1 = weights[:, 0]
    margin = top1 - weights[:, 1] if weights.shape[1] > 1 else top1
    tail_start = max(1, weights.shape[1] // 2)
    tail_mass = weights[:, tail_start:].sum(axis=1)
    normalized = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(normalized * np.log(normalized + 1e-12)).sum(axis=1)
    if weights.shape[1] > 1:
        entropy /= math.log(weights.shape[1])
    hhis: list[float] = []
    active_fractions: list[float] = []
    adjacent_same: list[float] = []
    for experts in experts_by_layer:
        counts = np.bincount(experts[:, 0].astype(int), minlength=num_experts).astype(float)
        shares = counts / max(counts.sum(), 1.0)
        hhis.append(float(np.square(shares).sum()))
        active_fractions.append(float(len(np.unique(experts)) / num_experts))
    for current, following in zip(experts_by_layer[:-1], experts_by_layer[1:]):
        count = min(len(current), len(following))
        if count:
            adjacent_same.append(float((current[:count, 0] == following[:count, 0]).mean()))
    values = {
        "full_route_top1_weight_mean": float(top1.mean()),
        "full_route_top1_weight_std": float(top1.std()),
        "full_route_top1_top2_margin_mean": float(margin.mean()),
        "full_route_tail_mass_mean": float(tail_mass.mean()),
        "full_route_routing_entropy_mean": float(entropy.mean()),
        "full_route_rank1_hhi_mean": float(np.mean(hhis)),
        "full_route_active_expert_fraction_mean": float(np.mean(active_fractions)),
        "full_route_same_id_adjacent_layer_rate": float(np.mean(adjacent_same)) if adjacent_same else 0.0,
    }
    if not np.isfinite(np.asarray(list(values.values()))).all():
        raise FeatureError("route features are non-finite")
    return values


def prefill_mean_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise FeatureError("prefill logits/input shapes do not align")
    if input_ids.shape[1] < 2:
        raise FeatureError("prefill NLL needs at least two tokens")
    value = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="mean",
    )
    if not torch.isfinite(value):
        raise FeatureError("prefill NLL is non-finite")
    return float(value.detach().cpu().item())
