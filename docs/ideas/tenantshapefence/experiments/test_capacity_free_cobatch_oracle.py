#!/usr/bin/env python3
"""Small integrity tests for the expert-only domain split."""

from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_capacity_free_cobatch_oracle_5090 as pilot


class DomainSplitTest(unittest.TestCase):
    def test_unstable_validation_cannot_be_no_go(self) -> None:
        row = {
            "validated_strong_hit": False,
            "validated_numeric_hit": False,
            "all_arms_stable": False,
            "unprotected_effect_stable": True,
            "protected_exact_noninterference": True,
        }
        self.assertEqual(pilot.decide_verdict([row]), "INCONCLUSIVE_UNSTABLE")

    def test_foreign_m_must_precede_route_divergence(self) -> None:
        self.assertTrue(pilot.foreign_m_precedes_divergence([{"layer": 4}], [5]))
        self.assertFalse(pilot.foreign_m_precedes_divergence([{"layer": 5}], [5]))
        self.assertTrue(pilot.foreign_m_precedes_divergence([{"layer": 15}], []))

    def test_split_removes_foreign_m_from_victim_expert_call(self) -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Gate(nn.Module):
            def forward(self, hidden):
                return torch.stack((hidden[:, 0], -hidden[:, 0]), dim=-1)

        class ShapeSensitiveExpert(nn.Module):
            def forward(self, hidden):
                return hidden + float(hidden.shape[0])

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 2
                self.top_k = 1
                self.norm_topk_prob = False
                self.gate = Gate()
                self.experts = nn.ModuleList(
                    [ShapeSensitiveExpert(), ShapeSensitiveExpert()]
                )

            def forward(self, hidden_states):
                batch_size, sequence_length, hidden_dim = hidden_states.shape
                flat_hidden = hidden_states.view(-1, hidden_dim)
                router_logits = self.gate(flat_hidden)
                weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                weights, selected = torch.topk(weights, self.top_k, dim=-1)
                weights = weights.to(flat_hidden.dtype)
                final = torch.zeros_like(flat_hidden)
                mask = F.one_hot(selected, num_classes=self.num_experts).permute(2, 1, 0)
                for expert_idx, expert in enumerate(self.experts):
                    idx, top_x = torch.where(mask[expert_idx])
                    current = flat_hidden[None, top_x].reshape(-1, hidden_dim)
                    contribution = expert(current) * weights[top_x, idx, None]
                    final.index_add_(0, top_x, contribution)
                return final.reshape(batch_size, sequence_length, hidden_dim), router_logits

        block = Block()
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                layers=[types.SimpleNamespace(mlp=block)]
            )
        )
        # Victim is the last request and stays fixed.  The two foreign arms
        # route either two or zero foreign rows to the victim's expert 0.
        victim = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
        foreign_same = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        foreign_other = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]])
        arm_a = torch.stack((foreign_same, victim), dim=0)
        arm_b = torch.stack((foreign_other, victim), dim=0)
        domains = [0, 1]

        with pilot.expert_execution_mode(
            model, mode="unprotected", request_domains=domains
        ):
            native_a = block(arm_a)[0][1].detach().clone()
        with pilot.expert_execution_mode(
            model, mode="unprotected", request_domains=domains
        ):
            native_b = block(arm_b)[0][1].detach().clone()
        self.assertFalse(torch.equal(native_a, native_b))

        with pilot.expert_execution_mode(
            model, mode="domain_split", request_domains=domains
        ) as trace_a:
            split_a = block(arm_a)[0][1].detach().clone()
        with pilot.expert_execution_mode(
            model, mode="domain_split", request_domains=domains
        ) as trace_b:
            split_b = block(arm_b)[0][1].detach().clone()
        self.assertTrue(torch.equal(split_a, split_b))
        self.assertEqual(
            trace_a["layers"]["0"]["victim_domain_m_by_expert"],
            trace_b["layers"]["0"]["victim_domain_m_by_expert"],
        )
        self.assertEqual(trace_a["layers"]["0"]["processed_contributions"], 4)
        self.assertEqual(trace_b["layers"]["0"]["processed_contributions"], 4)


if __name__ == "__main__":
    unittest.main()
