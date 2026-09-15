from __future__ import annotations


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

import math
import unittest

import torch

from grouped_owner_combine import (
    combine_owner_order,
    estimate_expert_gain_profile,
    expert_owner_ids,
    fixed_quota_high_mask,
    global_high_mask,
    group_owner_outputs,
    grouped_owner_combine,
    token_high_mask,
)


class GroupedOwnerCombineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]],
            ],
            dtype=torch.bfloat16,
        )
        self.weights = torch.tensor(
            [[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]],
            dtype=torch.bfloat16,
        )
        self.experts = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    def test_owner_mappings(self) -> None:
        ids = torch.arange(8)
        self.assertTrue(
            torch.equal(
                expert_owner_ids(ids, 8, 4, "contiguous"),
                torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
            )
        )
        self.assertTrue(
            torch.equal(
                expert_owner_ids(ids, 8, 4, "round_robin"),
                torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
            )
        )

    def test_group_counts_and_ep1_identity(self) -> None:
        grouped = group_owner_outputs(
            self.raw, self.weights, self.experts, 4, 2, "contiguous"
        )
        self.assertEqual(int(grouped.pair_count.sum().item()), 8)
        self.assertEqual(int(grouped.present.sum().item()), 4)
        self.assertTrue(torch.equal(grouped.pair_count, torch.full((2, 2), 2)))

        ep1 = group_owner_outputs(
            self.raw, self.weights, self.experts, 4, 1, "contiguous"
        )
        # Explicit expert-id-ordered BF16 accumulation.
        expected = torch.zeros((2, 2), dtype=torch.bfloat16)
        for expert in range(4):
            token, rank = torch.where(self.experts == expert)
            expected.index_add_(
                0,
                token,
                self.raw[token, rank]
                * self.weights[token, rank, None],
            )
        self.assertTrue(torch.equal(combine_owner_order(ep1.vectors), expected))

    def test_fixed_quota_is_exact_per_owner_tile(self) -> None:
        scores = torch.arange(18, dtype=torch.float32).reshape(9, 2)
        present = torch.ones_like(scores, dtype=torch.bool)
        high, rows = fixed_quota_high_mask(scores, present, 4, 0.5)
        for row in rows:
            self.assertEqual(
                row["high_vectors"], math.floor(row["vectors"] * 0.5)
            )
        # Owner streams have tiles 4, 4, 1 => 2 + 2 + 0 high vectors.
        self.assertEqual(int(high[:, 0].sum().item()), 4)
        self.assertEqual(int(high[:, 1].sum().item()), 4)

    def test_mixed_endpoints_match_uniform_formats(self) -> None:
        kwargs = dict(
            raw_outputs=self.raw,
            routing_weights=self.weights,
            selected_experts=self.experts,
            num_experts=4,
            ep_size=2,
            mapping="contiguous",
            tile_vectors=8,
        )
        _, uniform_high, _ = grouped_owner_combine(
            **kwargs, policy="uniform_fp8"
        )
        _, all_high, _ = grouped_owner_combine(
            **kwargs, policy="mixed_gate_mass", high_fraction=1.0
        )
        self.assertTrue(torch.equal(uniform_high, all_high))

        _, uniform_low, _ = grouped_owner_combine(
            **kwargs, policy="uniform_mxfp4"
        )
        _, all_low, _ = grouped_owner_combine(
            **kwargs, policy="mixed_gate_mass", high_fraction=0.0
        )
        self.assertTrue(torch.equal(uniform_low, all_low))

    def test_global_and_token_quota_counts(self) -> None:
        scores = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        present = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
                [True, True, True, False],
            ]
        )
        global_mask, _ = global_high_mask(scores, present, 0.5)
        token_mask, _ = token_high_mask(scores, present, 0.5)
        self.assertEqual(int(global_mask.sum().item()), 4)
        self.assertEqual(int(token_mask[0].sum().item()), 2)
        self.assertEqual(int(token_mask[1].sum().item()), 1)
        self.assertEqual(int(token_mask[2].sum().item()), 1)

    def test_origin_available_and_profiled_scores(self) -> None:
        input_norm = torch.tensor([2.0, 4.0])
        profile = torch.tensor([1.0, 2.0, 3.0, 4.0])
        common = dict(
            raw_outputs=self.raw,
            routing_weights=self.weights,
            selected_experts=self.experts,
            num_experts=4,
            ep_size=2,
            mapping="contiguous",
            tile_vectors=8,
            high_fraction=0.5,
            input_norm=input_norm,
        )
        _, _, input_diag = grouped_owner_combine(
            **common, policy="mixed_inputnorm_gate"
        )
        _, _, profiled_diag = grouped_owner_combine(
            **common,
            policy="mixed_profiled_gain",
            expert_gain_profile=profile,
        )
        self.assertEqual(input_diag["high_vectors"], 2)
        self.assertEqual(profiled_diag["high_vectors"], 2)

        sums, counts = estimate_expert_gain_profile(
            self.raw, input_norm, self.experts, 4
        )
        self.assertTrue(torch.equal(counts, torch.full((4,), 2.0, dtype=torch.float64)))
        self.assertTrue(bool(torch.isfinite(sums).all()))

    def test_oracle_runs_and_preserves_counts(self) -> None:
        reference, approximation, diagnostics = grouped_owner_combine(
            self.raw,
            self.weights,
            self.experts,
            num_experts=4,
            ep_size=2,
            mapping="round_robin",
            policy="mixed_oracle",
            tile_vectors=4,
            high_fraction=0.5,
        )
        self.assertEqual(reference.shape, approximation.shape)
        self.assertEqual(diagnostics["routed_pairs"], 8)
        self.assertEqual(
            diagnostics["high_vectors"] + diagnostics["low_vectors"],
            diagnostics["grouped_vectors"],
        )


if __name__ == "__main__":
    unittest.main()
