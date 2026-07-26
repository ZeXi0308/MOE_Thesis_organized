from __future__ import annotations

import unittest

import torch
from types import SimpleNamespace

from triage_runtime import (
    PreparedInt4ExpertBackend,
    TriageRuntimeError,
    cache_sequence_length,
    clone_cache,
    execute_same_state_step,
    fork_cache_pair,
    per_step_kl,
    symmetric_int4_weight,
)


class FakeExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 8, bias=False)
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)


class FakeMoe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList([FakeExpert(), FakeExpert()])


class FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = FakeMoe()


class FakeInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer()])


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeInner()


class TriageRuntimeTests(unittest.TestCase):
    def test_cache_forks_have_no_storage_alias(self) -> None:
        cache = ((torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),)
        high, low = fork_cache_pair(cache)
        high[0][0].add_(10)
        self.assertFalse(torch.equal(high[0][0], low[0][0]))
        self.assertFalse(torch.equal(high[0][0], cache[0][0]))
        self.assertEqual(cache_sequence_length(cache), 3)

    def test_clone_rejects_no_tensor_cache(self) -> None:
        with self.assertRaises(TriageRuntimeError):
            clone_cache({"length": 4})

    def test_int4_backend_prepares_once_and_restores_forward(self) -> None:
        model = FakeModel()
        backend = PreparedInt4ExpertBackend(model, expected_linears=6)
        original = model.model.layers[0].mlp.experts[0].gate_proj(torch.ones(1, 4))
        with backend:
            low = model.model.layers[0].mlp.experts[0].gate_proj(torch.ones(1, 4))
        restored = model.model.layers[0].mlp.experts[0].gate_proj(torch.ones(1, 4))
        self.assertTrue(torch.equal(original, restored))
        self.assertFalse(torch.equal(original, low))
        self.assertEqual(backend.low_model_forwards, 1)
        self.assertEqual(backend.expert_linear_calls, 1)

    def test_int4_backend_rejects_wrong_count_and_nesting(self) -> None:
        model = FakeModel()
        with self.assertRaises(TriageRuntimeError):
            PreparedInt4ExpertBackend(model, expected_linears=5)
        backend = PreparedInt4ExpertBackend(model, expected_linears=6)
        with backend:
            with self.assertRaises(TriageRuntimeError):
                backend.__enter__()

    def test_int4_range_and_kl(self) -> None:
        weight = torch.tensor([[0.0, 1.0, -2.0], [3.0, -4.0, 5.0]])
        quantized = symmetric_int4_weight(weight)
        self.assertEqual(quantized.shape, weight.shape)
        self.assertEqual(per_step_kl(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]])), 0.0)
        self.assertGreater(per_step_kl(torch.tensor([[5.0, 0.0]]), torch.tensor([[0.0, 5.0]])), 1.0)

    def test_same_state_step_selects_one_valid_branch(self) -> None:
        cache = ((torch.zeros(1, 1, 2, 3),),)

        def forward(offset: float):
            def call(token, branch_cache):
                old = branch_cache[0][0]
                new = torch.cat([old, torch.full((1, 1, 1, 3), offset)], dim=-2)
                logits = torch.tensor([[[offset, -offset]]])
                return SimpleNamespace(logits=logits, past_key_values=((new,),))
            return call

        result = execute_same_state_step(
            cache,
            torch.ones(1, 1, dtype=torch.long),
            high_forward=forward(1.0),
            low_forward=forward(-1.0),
            served_action="low",
        )
        self.assertEqual(result.served.post_length, 3)
        self.assertGreater(result.discrepancy, 0.0)
        self.assertTrue(torch.equal(cache[0][0], torch.zeros(1, 1, 2, 3)))


if __name__ == "__main__":
    unittest.main()
