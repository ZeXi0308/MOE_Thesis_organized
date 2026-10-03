#!/usr/bin/env python3

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_native_c8_replay_bridge as bridge  # noqa: E402
import run_single_contribution_pilot as base  # noqa: E402


class LoggedShapeExpert(torch.nn.Module):
    def __init__(self, expert_id: int, calls: list[tuple[int, int]]) -> None:
        super().__init__()
        self.expert_id = expert_id
        self.calls = calls

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.calls.append((self.expert_id, int(values.shape[0])))
        return values * float(self.expert_id + 1) + float(values.shape[0]) / 32.0


class ToyMoe(torch.nn.Module):
    def __init__(self, calls: list[tuple[int, int]]) -> None:
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
            [LoggedShapeExpert(index, calls) for index in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        logits = self.gate(flat)
        weights = F.softmax(logits, dim=1, dtype=torch.float)
        weights, experts = torch.topk(weights, self.top_k, dim=-1)
        weights = weights.to(flat.dtype)
        final = torch.zeros_like(flat)
        mask = F.one_hot(experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(mask[expert_idx])
            current = flat[None, top_x].reshape(-1, hidden_dim)
            raw = self.experts[expert_idx](current)
            final.index_add_(0, top_x, raw * weights[top_x, idx, None])
        return final.reshape(batch_size, sequence_length, hidden_dim), logits


def toy_cell(block: ToyMoe, hidden: torch.Tensor) -> dict:
    flat = hidden.reshape(-1, hidden.shape[-1])
    with torch.inference_mode():
        logits = block.gate(flat)
        weights = F.softmax(logits, dim=1, dtype=torch.float)
        weights, experts = torch.topk(weights, block.top_k, dim=-1)
    token = 1
    return {
        "layer": 0,
        "flat_token_idx": token,
        "expert_ids": list(map(int, experts[token].tolist())),
        "gate_weights": list(map(float, weights[token].tolist())),
        "target_hidden_sha256": base.tensor_sha256(flat[token]),
        "target_router_logits_sha256": base.tensor_sha256(logits[token]),
    }


class NativeReplayBridgeTests(unittest.TestCase):
    def test_strict_native_then_replay_and_native_noop_equivalence(self) -> None:
        calls: list[tuple[int, int]] = []
        block = ToyMoe(calls)
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(layers=[types.SimpleNamespace(mlp=block)])
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
        cell = toy_cell(block, hidden)
        native, native_logits = block(hidden)

        calls.clear()
        with bridge.native_then_c8_replay(model, cell, None) as native_trace:
            copied, copied_logits = block(hidden)
        self.assertTrue(torch.equal(native, copied))
        self.assertTrue(torch.equal(native_logits, copied_logits))
        self.assertEqual(len(calls), block.num_experts)
        self.assertEqual(native_trace["execution_order"], "all_native_then_optional_c8_then_combine")

        calls.clear()
        with bridge.native_then_c8_replay(model, cell, 0) as replay_trace:
            patched, patched_logits = block(hidden)
        self.assertEqual(len(calls), block.num_experts + 1)
        self.assertEqual([item[0] for item in calls[: block.num_experts]], [0, 1, 2])
        self.assertEqual(calls[-1], (cell["expert_ids"][0], 8))
        self.assertTrue(torch.equal(native_logits, patched_logits))
        self.assertFalse(torch.equal(native[:, 1], patched[:, 1]))
        self.assertEqual(replay_trace["dummy_rows"], 7)
        self.assertEqual(
            replay_trace["target_native_raw_sha256_by_rank"],
            native_trace["target_native_raw_sha256_by_rank"],
        )
        self.assertEqual(
            replay_trace["non_target_contributions_sha256"],
            native_trace["non_target_contributions_sha256"],
        )
        other = "1"
        self.assertEqual(
            replay_trace["target_applied_raw_sha256_by_rank"][other],
            native_trace["target_native_raw_sha256_by_rank"][other],
        )

    def test_timing_plan_uses_only_rank_and_natural_m_extremes(self) -> None:
        rows = [
            {
                "action_id": f"cell-{cell:03d}-rank-{rank}",
                "cell_id": f"cell-{cell:03d}",
                "rank": rank,
                "expert_id": rank,
                "natural_m": cell + 2,
                "route_net_reward": 1000 - cell,
            }
            for cell in range(3)
            for rank in range(8)
        ]
        selected = bridge.freeze_timing_plan(rows, 16)
        self.assertEqual(len(selected), 16)
        for rank in range(8):
            values = [row["natural_m"] for row in selected if row["rank"] == rank]
            self.assertEqual(values, [2, 4])
        self.assertTrue(all(not row["selection_used_utility"] for row in selected))
        self.assertTrue(all(not row["selection_used_latency"] for row in selected))

    def test_native_aggregation_random_and_abstaining_oracle(self) -> None:
        cells = []
        for cell_index in range(2):
            actions = {}
            nets = [2, -1, 0, 1, 0, 0, 0, 0] if cell_index == 0 else [0] * 8
            for rank, net in enumerate(nets):
                recovered = max(net, 0)
                harmed = max(-net, 0)
                actions[str(rank)] = {
                    "rank": rank,
                    "route_recovered_count": recovered,
                    "route_harmed_count": harmed,
                    "route_persistent_count": 0,
                    "route_net_reward": net,
                }
            cells.append(
                {
                    "document_index": cell_index,
                    "frozen_m1_rank": 0,
                    "actions": actions,
                }
            )
        result = bridge.native_metrics(cells)
        self.assertEqual(bridge.fraction_value(result["same_rank"]["net"]), 2)
        self.assertEqual(
            bridge.fraction_value(result["matched_random"]["net"]), Fraction(2, 8)
        )
        self.assertEqual(
            bridge.fraction_value(result["abstaining_oracle"]["net"]), 2
        )
        self.assertEqual(result["oracle_selected_cell_count"], 1)
        self.assertEqual(result["same_rank_cell_signs"], {"positive": 1, "zero": 1, "negative": 0})

    def test_classification_priority(self) -> None:
        def native(same: int, random: int, oracle: int):
            def metric(value: int):
                return {"net": bridge.fraction_payload(value)}

            return {
                "same_rank": metric(same),
                "matched_random": metric(random),
                "abstaining_oracle": metric(oracle),
            }

        low_cost = {"paired_relative_direct_overhead": {"median": 0.5}}
        high_cost = {"paired_relative_direct_overhead": {"median": 1.2}}
        self.assertEqual(
            bridge.classify_bridge(0, native(1, 0, 2), high_cost, 1.0),
            "NO_NATIVE_OPPORTUNITY",
        )
        self.assertEqual(
            bridge.classify_bridge(1, native(0, 0, 0), low_cost, 1.0),
            "PROXY_BACKGROUND_DEPENDENT",
        )
        self.assertEqual(
            bridge.classify_bridge(1, native(1, 0, 2), high_cost, 1.0),
            "NATIVE_ACTION_VALID_DIRECT_COST_HIGH",
        )
        self.assertEqual(
            bridge.classify_bridge(1, native(2, 1, 3), low_cost, 1.0),
            "NATIVE_REPLAY_AND_RANK_SPECIFICITY_TRANSFER",
        )
        self.assertEqual(
            bridge.classify_bridge(1, native(1, 1, 3), low_cost, 1.0),
            "NATIVE_REPLAY_TRANSFERS_RANK_SIGNAL_WEAK",
        )

    def test_report_has_exact_required_sections(self) -> None:
        metric = {
            "recovered": bridge.fraction_payload(2),
            "harmed": bridge.fraction_payload(0),
            "net": bridge.fraction_payload(2),
        }
        cost_band = {"median": 1.0, "p10": 0.9, "p90": 1.1}
        summary = {
            "bridge_classification": "NATIVE_REPLAY_AND_RANK_SPECIFICITY_TRANSFER",
            "proxy_metrics": {"same_rank": metric},
            "native_metrics": {
                "same_rank": metric,
                "matched_random": metric,
                "abstaining_oracle": metric,
                "same_rank_cell_signs": {"positive": 1, "zero": 0, "negative": 0},
                "same_rank_document_signs": {"positive": 1, "zero": 0, "negative": 0},
            },
            "bridge_metrics": {
                "same_rank_bridge_transfer": 1.0,
                "oracle_bridge_transfer": 1.0,
                "native_specificity_gap": 0.0,
            },
            "direct_cost": {
                "target_moe_stage_native_ms": cost_band,
                "target_moe_stage_native_plus_replay_ms": cost_band,
                "paired_direct_patch_delta_ms": cost_band,
                "paired_relative_direct_overhead": {"median": 0.1},
                "per_protected_action_delta_ms": 0.1,
                "timing_subset_per_net_recovered_route_ms": 0.05,
            },
            "document_count": 1,
            "native_opportunity": {"route_count": 2, "cell_count": 1},
        }
        report = bridge.build_report(summary)
        self.assertEqual(report.count("\n## ") + int(report.startswith("## ")), 8)
        for heading in (
            "Bridge classification",
            "Result table",
            "Bridge metrics",
            "Direct cost",
            "Mechanistic interpretation",
            "System implication",
            "Scope",
            "Next minimal experiment",
        ):
            self.assertIn(f"## {heading}", report)


if __name__ == "__main__":
    unittest.main()
