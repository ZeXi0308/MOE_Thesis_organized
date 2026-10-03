#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_single_contribution_pilot as pilot  # noqa: E402


class ToyExpert(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale


class ToyMoe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 3
        self.top_k = 2
        self.norm_topk_prob = False
        self.gate = torch.nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            )
        self.experts = torch.nn.ModuleList(
            [ToyExpert(1.0), ToyExpert(2.0), ToyExpert(3.0)]
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights.to(flat.dtype)
        final = torch.zeros_like(flat)
        mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(mask[expert_idx])
            current = flat[None, top_x].reshape(-1, hidden_dim)
            raw = self.experts[expert_idx](current)
            final.index_add_(0, top_x, raw * routing_weights[top_x, idx, None])
        return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


class SingleContributionTests(unittest.TestCase):
    def test_patch_noop_is_bitwise_and_replacement_is_local(self) -> None:
        block = ToyMoe()
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                layers=[types.SimpleNamespace(mlp=block)]
            )
        )
        hidden = torch.tensor(
            [
                [
                    [2.0, 0.1, 0.0, 1.0],
                    [0.2, 3.0, 0.1, 1.0],
                    [0.1, 0.2, 4.0, 1.0],
                ]
            ]
        )
        native, logits = block(hidden)
        _, experts = pilot.topk_from_logits(logits[1], block.top_k)
        identity = pilot.PairIdentity(
            layer=0,
            flat_token_idx=1,
            topk_rank=0,
            expert_id=int(experts[0]),
        )
        with pilot.patched_single_contribution(
            model, identity, None, "self"
        ) as noop_trace:
            noop, noop_logits = block(hidden)
        self.assertTrue(torch.equal(native, noop))
        self.assertTrue(torch.equal(logits, noop_logits))
        self.assertEqual(noop_trace["pair_match_count"], 1)
        self.assertEqual(
            noop_trace["target_native_raw_sha256"],
            noop_trace["target_applied_raw_sha256"],
        )

        replacement = torch.zeros(4)
        with pilot.patched_single_contribution(
            model, identity, replacement, "replacement"
        ) as replacement_trace:
            changed, changed_logits = block(hidden)
        self.assertTrue(torch.equal(logits, changed_logits))
        self.assertTrue(torch.equal(native[:, 0], changed[:, 0]))
        self.assertFalse(torch.equal(native[:, 1], changed[:, 1]))
        self.assertTrue(torch.equal(native[:, 2], changed[:, 2]))
        self.assertEqual(replacement_trace["pair_match_count"], 1)
        self.assertEqual(replacement_trace["routing_weight_apply_count"], 1)

    def test_selection_is_balanced_and_capped(self) -> None:
        candidates = []
        bands = [[0, 3], [4, 7], [8, 11], [12, 14]]
        for victim_index in range(16):
            for layer in range(15):
                candidates.append(
                    {
                        "victim_id": f"v{victim_index:02d}",
                        "layer": layer,
                        "topk_rank": 0,
                        "expert_id": victim_index % 3,
                        "selection_score": 1000.0 - victim_index * 10.0 - layer,
                    }
                )
        selection = {
            "layer_bands": bands,
            "targets_per_band": 8,
            "target_count": 32,
            "max_targets_per_victim": 2,
        }
        selected = pilot.select_targets(candidates, selection)
        self.assertEqual(len(selected), 32)
        counts = {}
        for row in selected:
            counts[row["victim_id"]] = counts.get(row["victim_id"], 0) + 1
        self.assertLessEqual(max(counts.values()), 2)
        self.assertEqual(
            [sum(row["layer_band_index"] == idx for row in selected) for idx in range(4)],
            [8, 8, 8, 8],
        )
        self.assertEqual(selected, pilot.select_targets(candidates, selection))

    def test_verdicts_are_predefined(self) -> None:
        gate = {
            "support_min_reproducible_route_targets": 4,
            "support_min_distinct_victims": 2,
            "suggestive_min_reproducible_route_targets": 1,
            "suggestive_max_reproducible_route_targets": 3,
        }

        def rows(route_count: int, local_count: int):
            return [
                {
                    "victim_id": f"v{index % 2}",
                    "reproducible_route_propagation": index < route_count,
                    "local_replacement_changed": index < local_count,
                    "reproducible_token_flip": False,
                }
                for index in range(32)
            ]

        self.assertEqual(pilot.classify_summary(rows(4, 32), gate)["verdict"], "SUPPORT")
        self.assertEqual(
            pilot.classify_summary(rows(2, 32), gate)["verdict"],
            "SUGGESTIVE_TARGETED_RERUN",
        )
        self.assertEqual(
            pilot.classify_summary(rows(0, 0), gate)["verdict"],
            "LOCAL_EXECUTION_SHAPE_SIGNAL_ABSENT",
        )
        self.assertEqual(
            pilot.classify_summary(rows(0, 32), gate)["verdict"],
            "NO_REPRODUCIBLE_DOWNSTREAM_ROUTE_PROPAGATION",
        )

    def test_config_has_frozen_scope_and_exact_counts(self) -> None:
        config_path = HERE / "configs" / "single_contribution_pilot_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "FROZEN_PRE_RUN")
        self.assertEqual(len(config["data"]["document_indices"]), 16)
        self.assertEqual(config["selection"]["target_count"], 32)
        self.assertEqual(config["intervention"]["repeats_per_arm"], 3)
        self.assertIn("not_kernel_algorithm", config["research_boundary"])

    def test_membership_difference_ignores_order(self) -> None:
        left = [[1, 2], [3, 4], [5, 6]]
        right = [[2, 1], [4, 3], [5, 7]]
        self.assertEqual(pilot.changed_membership_layers(left, right, 1), [2])


if __name__ == "__main__":
    unittest.main()
