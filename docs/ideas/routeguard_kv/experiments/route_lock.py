#!/usr/bin/env python3
"""Step-aware OLMoE route recording and counterfactual route locks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import types
from typing import Any, Iterator

import torch
import torch.nn.functional as F


class RouteLockError(RuntimeError):
    pass


@dataclass
class RouteReference:
    selected_experts: torch.Tensor
    native_weights: torch.Tensor
    boundary_margin: torch.Tensor


@dataclass
class RouteObservation:
    step: int
    layer: int
    natural_experts: torch.Tensor
    executed_experts: torch.Tensor
    natural_weights: torch.Tensor
    executed_weights: torch.Tensor
    boundary_margin: torch.Tensor


class RouteController:
    """One-document controller keyed by prompt length, decode step and layer."""

    MODES = {"record", "free", "set_locked", "fully_locked"}

    def __init__(
        self,
        *,
        mode: str,
        text_sha256: str,
        prompt_length: int,
        expected_layers: int,
        references: dict[tuple[int, int], RouteReference] | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise RouteLockError(f"unknown route mode: {mode}")
        self.mode = mode
        self.text_sha256 = text_sha256
        self.prompt_length = int(prompt_length)
        self.expected_layers = int(expected_layers)
        self.references = references if references is not None else {}
        self.observations: list[RouteObservation] = []
        self.current_step: int | None = None
        self._visited: set[int] = set()

    def begin_step(self, step: int) -> None:
        if self.current_step is not None:
            raise RouteLockError("previous decode step was not closed")
        self.current_step = int(step)
        self._visited = set()

    def end_step(self) -> None:
        if self.current_step is None:
            raise RouteLockError("no active decode step")
        expected = set(range(self.expected_layers))
        if self._visited != expected:
            raise RouteLockError(
                f"step {self.current_step} visited layers {sorted(self._visited)}, expected {sorted(expected)}"
            )
        self.current_step = None
        self._visited = set()

    def select(
        self,
        *,
        layer_idx: int,
        full_probabilities: torch.Tensor,
        natural_experts: torch.Tensor,
        natural_weights: torch.Tensor,
        boundary_margin: torch.Tensor,
        norm_topk_prob: bool,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.current_step is None:
            raise RouteLockError("route block called outside begin_step/end_step")
        if layer_idx in self._visited:
            raise RouteLockError(f"layer {layer_idx} executed twice at step {self.current_step}")
        self._visited.add(layer_idx)
        key = (self.current_step, layer_idx)

        if self.mode == "record":
            self.references[key] = RouteReference(
                selected_experts=natural_experts.detach().cpu().clone(),
                native_weights=natural_weights.detach().cpu().clone(),
                boundary_margin=boundary_margin.detach().cpu().clone(),
            )
            executed_experts = natural_experts
            executed_weights = natural_weights
        elif self.mode == "free":
            executed_experts = natural_experts
            executed_weights = natural_weights
        else:
            if key not in self.references:
                raise RouteLockError(f"missing reference route for step/layer {key}")
            reference = self.references[key]
            executed_experts = reference.selected_experts.to(full_probabilities.device)
            if tuple(executed_experts.shape) != tuple(natural_experts.shape):
                raise RouteLockError("reference route shape differs from treatment route shape")
            if self.mode == "set_locked":
                executed_weights = torch.gather(full_probabilities, 1, executed_experts)
                if norm_topk_prob:
                    executed_weights = executed_weights / executed_weights.sum(dim=-1, keepdim=True)
                executed_weights = executed_weights.to(output_dtype)
            else:
                executed_weights = reference.native_weights.to(
                    device=full_probabilities.device, dtype=output_dtype
                )

        self.observations.append(
            RouteObservation(
                step=self.current_step,
                layer=layer_idx,
                natural_experts=natural_experts.detach().cpu().clone(),
                executed_experts=executed_experts.detach().cpu().clone(),
                natural_weights=natural_weights.detach().cpu().clone(),
                executed_weights=executed_weights.detach().cpu().clone(),
                boundary_margin=boundary_margin.detach().cpu().clone(),
            )
        )
        return executed_experts, executed_weights


def olmoe_locked_forward(
    block: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    controller: RouteController,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy the Transformers 4.57.6 OLMoE accumulation path, changing selection only."""

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat_states = hidden_states.view(-1, hidden_dim)
    router_logits = block.gate(flat_states)
    full_probabilities = F.softmax(router_logits, dim=1, dtype=torch.float)
    natural_weights, natural_experts = torch.topk(full_probabilities, block.top_k, dim=-1)
    boundary_values = torch.topk(router_logits.float(), block.top_k + 1, dim=-1).values
    boundary_margin = boundary_values[:, block.top_k - 1] - boundary_values[:, block.top_k]
    if block.norm_topk_prob:
        natural_weights = natural_weights / natural_weights.sum(dim=-1, keepdim=True)
    natural_weights = natural_weights.to(flat_states.dtype)

    selected_experts, routing_weights = controller.select(
        layer_idx=layer_idx,
        full_probabilities=full_probabilities,
        natural_experts=natural_experts,
        natural_weights=natural_weights,
        boundary_margin=boundary_margin,
        norm_topk_prob=bool(block.norm_topk_prob),
        output_dtype=flat_states.dtype,
    )

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=flat_states.dtype,
        device=flat_states.device,
    )
    expert_mask = F.one_hot(selected_experts, num_classes=block.num_experts).permute(2, 1, 0)
    for expert_idx in range(block.num_experts):
        expert_layer = block.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        # Empty experts have no numerical contribution. Skipping their empty
        # linear kernels preserves the official expert-order accumulation while
        # avoiding 56 useless expert calls per top-8 decode token.
        if top_x.numel() == 0:
            continue
        current_state = flat_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(flat_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def find_olmoe_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    blocks = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "OlmoeSparseMoeBlock"
        or (
            hasattr(module, "gate")
            and hasattr(module, "experts")
            and hasattr(module, "top_k")
            and hasattr(module, "norm_topk_prob")
        )
    ]
    # Do not include a parent that merely aggregates child blocks.
    return [block for block in blocks if hasattr(block, "num_experts")]


@contextmanager
def patch_olmoe_routes(model: torch.nn.Module, controller: RouteController) -> Iterator[None]:
    blocks = find_olmoe_blocks(model)
    if len(blocks) != controller.expected_layers:
        raise RouteLockError(
            f"found {len(blocks)} OLMoE blocks, expected {controller.expected_layers}"
        )
    originals: list[tuple[torch.nn.Module, Any]] = []
    try:
        for layer_idx, block in enumerate(blocks):
            originals.append((block, block.forward))

            def replacement(
                bound_block: torch.nn.Module,
                hidden_states: torch.Tensor,
                *,
                _layer_idx: int = layer_idx,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return olmoe_locked_forward(
                    bound_block,
                    hidden_states,
                    controller=controller,
                    layer_idx=_layer_idx,
                )

            block.forward = types.MethodType(replacement, block)
        yield
    finally:
        for block, original in originals:
            block.forward = original


def exact_set_equal(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise RouteLockError(f"route shape mismatch: {left.shape} != {right.shape}")
    return torch.sort(left, dim=-1).values.eq(torch.sort(right, dim=-1).values).all(dim=-1)


def jaccard_overlap(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise RouteLockError("Jaccard expects equal [token, top_k] tensors")
    intersections = (left.unsqueeze(-1) == right.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    unions = 2 * left.shape[-1] - intersections
    return intersections.float() / unions.float()
