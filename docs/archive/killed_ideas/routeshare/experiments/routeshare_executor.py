from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def execute_expert_stage(
    experts: Sequence[nn.Module],
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Transformers-style dispatch, real expert execution, weighted combine."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [tokens, hidden]")
    if selected_experts.ndim != 2 or selected_experts.shape[0] != hidden_states.shape[0]:
        raise ValueError("selected_experts must be [tokens, top_k]")
    if routing_weights.shape != selected_experts.shape:
        raise ValueError("routing_weights shape mismatch")
    if selected_experts.dtype != torch.long:
        raise TypeError("selected_experts must be torch.long")
    if selected_experts.numel() and (
        int(selected_experts.min()) < 0 or int(selected_experts.max()) >= len(experts)
    ):
        raise ValueError("selected expert id out of range")

    token_count, top_k = selected_experts.shape
    output = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=len(experts)
    ).permute(2, 1, 0)
    for expert_id, expert in enumerate(experts):
        rank_index, token_index = torch.where(expert_mask[expert_id])
        if token_index.numel() == 0:
            continue
        expert_input = hidden_states[token_index]
        expert_output = expert(expert_input)
        if not isinstance(expert_output, torch.Tensor) or expert_output.shape != expert_input.shape:
            raise RuntimeError("expert must return a tensor matching [rows, hidden]")
        weighted = expert_output * routing_weights[token_index, rank_index, None]
        output.index_add_(0, token_index, weighted.to(output.dtype))
    return output


def execute_tenants_separately(
    experts: Sequence[nn.Module],
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    tenant_ids: torch.Tensor,
) -> torch.Tensor:
    if tenant_ids.shape != (hidden_states.shape[0],):
        raise ValueError("tenant_ids shape mismatch")
    unique = torch.unique(tenant_ids, sorted=True)
    if unique.tolist() != [0, 1]:
        raise ValueError("expected exactly tenant ids 0 and 1")
    outputs = []
    indices = []
    for tenant_id in (0, 1):
        index = torch.where(tenant_ids == tenant_id)[0]
        indices.append(index)
        outputs.append(
            execute_expert_stage(
                experts,
                hidden_states[index],
                selected_experts[index],
                routing_weights[index],
            )
        )
    combined = torch.empty_like(hidden_states)
    for index, value in zip(indices, outputs):
        combined[index] = value
    return combined


def closure_error(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    if reference.shape != candidate.shape:
        raise ValueError("closure tensors have different shapes")
    delta = (reference.float() - candidate.float()).abs()
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    scale = max(float(reference.float().abs().max().item()), 1e-12)
    return max_abs, max_abs / scale
