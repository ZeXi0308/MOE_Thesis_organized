from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from route_lock import (
    RouteController,
    RouteLockError,
    exact_set_equal,
    jaccard_overlap,
    olmoe_locked_forward,
    patch_olmoe_routes,
)


class FakeBlock(nn.Module):
    def __init__(self, *, norm_topk_prob: bool = False) -> None:
        super().__init__()
        self.num_experts = 4
        self.top_k = 2
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(3, 4, bias=False)
        self.experts = nn.ModuleList([nn.Identity() for _ in range(4)])
        with torch.no_grad():
            self.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [-1.0, -1.0, -1.0],
                    ]
                )
            )

    def forward(self, hidden_states: torch.Tensor):
        batch, sequence, hidden = hidden_states.shape
        flat = hidden_states.view(-1, hidden)
        logits = self.gate(flat)
        weights = F.softmax(logits, dim=-1, dtype=torch.float)
        weights, experts = torch.topk(weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(flat.dtype)
        output = torch.zeros_like(flat)
        mask = F.one_hot(experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx, expert in enumerate(self.experts):
            idx, top_x = torch.where(mask[expert_idx])
            current = flat[None, top_x].reshape(-1, hidden)
            output.index_add_(0, top_x, expert(current) * weights[top_x, idx, None])
        return output.reshape(batch, sequence, hidden), logits


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x, _ = block(x)
        return x


def _run_block(block: FakeBlock, controller: RouteController, hidden: torch.Tensor) -> torch.Tensor:
    controller.begin_step(0)
    output, _ = olmoe_locked_forward(block, hidden, controller=controller, layer_idx=0)
    controller.end_step()
    return output


def test_record_and_free_are_native_equivalent() -> None:
    block = FakeBlock()
    hidden = torch.tensor([[[0.7, 0.2, -0.1]]])
    native, _ = block(hidden)
    references = {}
    record = RouteController(
        mode="record", text_sha256="a" * 64, prompt_length=8, expected_layers=1, references=references
    )
    patched = _run_block(block, record, hidden)
    torch.testing.assert_close(patched, native, rtol=0, atol=0)
    free = RouteController(
        mode="free", text_sha256="a" * 64, prompt_length=8, expected_layers=1, references=references
    )
    torch.testing.assert_close(_run_block(block, free, hidden), native, rtol=0, atol=0)


def test_set_locked_gathers_full_softmax_without_renormalizing_for_olmoe() -> None:
    block = FakeBlock(norm_topk_prob=False)
    reference_hidden = torch.tensor([[[0.7, 0.2, -0.1]]])
    treatment_hidden = torch.tensor([[[-0.8, 0.1, 0.9]]])
    references = {}
    record = RouteController(
        mode="record", text_sha256="b" * 64, prompt_length=8, expected_layers=1, references=references
    )
    _run_block(block, record, reference_hidden)
    locked = RouteController(
        mode="set_locked", text_sha256="b" * 64, prompt_length=8, expected_layers=1, references=references
    )
    _run_block(block, locked, treatment_hidden)
    observation = locked.observations[0]
    probabilities = F.softmax(block.gate(treatment_hidden.view(1, 3)), dim=-1, dtype=torch.float)
    expected = torch.gather(probabilities, 1, references[(0, 0)].selected_experts)
    torch.testing.assert_close(observation.executed_weights, expected)
    assert float(observation.executed_weights.sum()) < 1.0


def test_native_norm_semantics_are_preserved_when_enabled() -> None:
    block = FakeBlock(norm_topk_prob=True)
    references = {}
    record = RouteController(
        mode="record", text_sha256="c" * 64, prompt_length=8, expected_layers=1, references=references
    )
    _run_block(block, record, torch.tensor([[[0.7, 0.2, -0.1]]]))
    locked = RouteController(
        mode="set_locked", text_sha256="c" * 64, prompt_length=8, expected_layers=1, references=references
    )
    _run_block(block, locked, torch.tensor([[[-0.8, 0.1, 0.9]]]))
    torch.testing.assert_close(locked.observations[0].executed_weights.sum(dim=-1), torch.ones(1))


def test_step_layer_cursor_and_patch_restore() -> None:
    model = FakeModel()
    hidden = torch.tensor([[[0.7, 0.2, -0.1]]])
    controller = RouteController(
        mode="record", text_sha256="d" * 64, prompt_length=8, expected_layers=2
    )
    originals = [block.forward for block in model.blocks]
    controller.begin_step(3)
    with patch_olmoe_routes(model, controller):
        model(hidden)
    controller.end_step()
    assert sorted(controller.references) == [(3, 0), (3, 1)]
    assert all(block.forward.__func__ is original.__func__ for block, original in zip(model.blocks, originals))
    controller.begin_step(4)
    try:
        controller.end_step()
    except RouteLockError:
        pass
    else:
        raise AssertionError("missing-layer cursor must fail closed")


def test_route_metrics_are_set_based() -> None:
    left = torch.tensor([[1, 2], [1, 3]])
    right = torch.tensor([[2, 1], [2, 3]])
    assert exact_set_equal(left, right).tolist() == [True, False]
    torch.testing.assert_close(jaccard_overlap(left, right), torch.tensor([1.0, 1.0 / 3.0]))

