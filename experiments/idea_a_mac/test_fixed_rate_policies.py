from __future__ import annotations

import unittest

import torch

from policies import ApproxPolicy, _block_budget_mask


class FixedRatePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.tokens = 5
        self.top_k = 8
        self.hidden = 64
        self.raw = torch.randn(self.tokens, self.top_k, self.hidden, dtype=torch.bfloat16)
        weights = torch.rand(self.tokens, self.top_k)
        self.weights = torch.sort(weights, dim=-1, descending=True).values.to(torch.bfloat16)
        self.experts = torch.arange(self.tokens * self.top_k).reshape(
            self.tokens, self.top_k
        ) % 32

    def _bytes(self, name: str) -> torch.Tensor:
        return ApproxPolicy(name).bytes_per_element_for_selected(
            self.experts,
            num_experts=32,
            routing_weights=self.weights,
        )

    def test_every_direct_score_has_exact_half_composition(self) -> None:
        for mode in ("contrib", "qenergy", "qerr", "qbenefit", "random", "reversegate"):
            name = f"block_{mode}4_mxfp4"
            values = self._bytes(name)
            self.assertEqual(int((values == 0.5).sum()), 20, name)
            approx, returned_weights = ApproxPolicy(name).apply(
                self.raw, self.weights, self.experts, 32
            )
            self.assertEqual(approx.shape, self.raw.shape)
            self.assertTrue(torch.isfinite(approx.float()).all())
            self.assertTrue(torch.equal(returned_weights, self.weights))

    def test_every_residual_score_has_exact_half_refinement(self) -> None:
        for mode in ("contrib", "resenergy", "reserr", "resbenefit", "random", "reversegate"):
            name = f"block_{mode}4_residual_mxfp4"
            values = self._bytes(name)
            self.assertEqual(int((values == 0.5).sum()), 20, name)
            self.assertEqual(int((values == 1.0).sum()), 20, name)
            approx, _ = ApproxPolicy(name).apply(
                self.raw, self.weights, self.experts, 32
            )
            self.assertTrue(torch.isfinite(approx.float()).all())

    def test_encoded_matched_fraction_is_exact_per_full_block(self) -> None:
        values = self._bytes("block_qerr4_f436_mxfp4")
        # 4 tokens x 8 pairs -> round(32 * .436) = 14; final 1-token block -> 3.
        self.assertEqual(int((values == 0.5).sum()), 17)

    def test_random_control_is_reproducible(self) -> None:
        first = self._bytes("block_random4_mxfp4")
        second = self._bytes("block_random4_mxfp4")
        self.assertTrue(torch.equal(first, second))

    def test_owner_group_policies_preserve_fixed_cardinality(self) -> None:
        for name in (
            "peerblock_gate8_mxfp4",
            "peerblock_qerr8_mxfp4",
            "peerblock_gate8_f436_mxfp4",
            "peerblock_qbenefit8_f436_mxfp4",
            "peerblock_reserr8_residual_mxfp4",
        ):
            values = ApproxPolicy(name).bytes_per_element_for_selected(
                self.experts,
                num_experts=32,
                num_receiver_groups=4,
                routing_weights=self.weights,
            )
            approx, _ = ApproxPolicy(name).apply(
                self.raw,
                self.weights,
                self.experts,
                32,
                num_receiver_groups=4,
            )
            self.assertTrue(torch.isfinite(approx.float()).all(), name)
            self.assertTrue(((values == 0.5) | (values == 1.0)).all(), name)

    def test_block_one_gate_matches_rank_tail_under_ties(self) -> None:
        tied = torch.ones(3, 8)
        mask = _block_budget_mask(tied, block_tokens=1, low_bit_fraction=0.5)
        expected = torch.zeros_like(mask)
        expected[:, 4:] = True
        self.assertTrue(torch.equal(mask, expected))

    def test_keep_drop_renorm_removes_tail_and_normalizes(self) -> None:
        policy = ApproxPolicy("keep4_drop_renorm")
        values = self._bytes(policy.name)
        self.assertTrue((values[:, :4] == 2.0).all())
        self.assertTrue((values[:, 4:] == 0.0).all())
        approx, weights = policy.apply(
            self.raw,
            self.weights,
            self.experts,
            32,
        )
        self.assertTrue((approx[:, 4:, :] == 0).all())
        self.assertTrue(torch.allclose(weights[:, 4:], torch.zeros_like(weights[:, 4:])))
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=-1),
                torch.ones(self.tokens, dtype=weights.dtype),
                atol=1e-2,
            )
        )


if __name__ == "__main__":
    unittest.main()
