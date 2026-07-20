from __future__ import annotations

import math
import unittest

import torch

from creditreduce_reference import (
    build_expert_to_rank,
    creditreduce_reference,
    rank_domain_ids,
)


def _bf16(values: object) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bfloat16)


class CreditReduceReferenceTest(unittest.TestCase):
    def _run(
        self,
        raw: torch.Tensor,
        experts: torch.Tensor,
        *,
        num_experts: int,
        ep_size: int,
        ranks_per_domain: int = 1,
        placement: str = "contiguous",
        weights: torch.Tensor | None = None,
        home_ranks: torch.Tensor | None = None,
        home_domains: torch.Tensor | None = None,
        threshold: float = math.inf,
    ):
        if weights is None:
            weights = torch.ones(raw.shape[:2], dtype=torch.bfloat16)
        if home_ranks is None and home_domains is None:
            home_ranks = torch.zeros(raw.shape[0], dtype=torch.int64)
        return creditreduce_reference(
            raw,
            weights,
            experts,
            num_experts=num_experts,
            ep_size=ep_size,
            ranks_per_domain=ranks_per_domain,
            placement=placement,
            home_ranks=home_ranks,
            home_domains=home_domains,
            residual_rms_threshold=threshold,
        )

    def test_hand_arithmetic_and_canonical_expert_order(self) -> None:
        # Input top-k order is deliberately reversed/mixed.  The arithmetic
        # oracle must expose and consume expert order 0,1,2,3.
        raw = _bf16([[[40, 4], [10, 1], [30, 3], [20, 2]]])
        weights = _bf16([[0.5, 0.5, 0.5, 0.5]])
        experts = torch.tensor([[3, 0, 2, 1]])
        result = self._run(
            raw,
            experts,
            weights=weights,
            num_experts=4,
            ep_size=4,
            ranks_per_domain=2,
            home_ranks=torch.tensor([0]),
        )

        self.assertTrue(
            torch.equal(result.canonical_expert_ids, torch.tensor([[0, 1, 2, 3]]))
        )
        expected_contributions = _bf16([[[5, 0.5], [10, 1], [15, 1.5], [20, 2]]])
        self.assertTrue(
            torch.equal(result.canonical_weighted_contributions, expected_contributions)
        )
        torch.testing.assert_close(
            result.domain_subtotals_fp32,
            torch.tensor([[[15, 1.5], [35, 3.5]]], dtype=torch.float32),
        )
        self.assertTrue(
            torch.equal(result.outputs["late_bf16"], _bf16([[50, 5]]))
        )

    def test_stock_bf16_and_clean_fp32_subtotal_are_distinct(self) -> None:
        # At magnitude 256, sequential BF16 additions of 1 round away.  The
        # clean endpoint performs the subtotal in FP32 and casts only once.
        raw = _bf16([[[256], [1], [1]]])
        experts = torch.tensor([[0, 1, 2]])
        result = self._run(
            raw,
            experts,
            num_experts=8,
            ep_size=2,
            home_ranks=torch.tensor([1]),
        )
        self.assertTrue(
            torch.equal(result.outputs["stock_early_bf16"], _bf16([[256]]))
        )
        self.assertTrue(
            torch.equal(result.outputs["clean_early_bf16"], _bf16([[258]]))
        )

    def test_home_domain_is_excluded_from_all_wire_counts(self) -> None:
        # Experts 0/1 are local (domain 0); expert 4 is remote (domain 2).
        raw = _bf16([[[2, 4], [3, 5], [7, 11]]])
        experts = torch.tensor([[0, 1, 4]])
        result = self._run(
            raw,
            experts,
            num_experts=8,
            ep_size=4,
            home_domains=torch.tensor([0]),
        )
        diag = result.diagnostics
        self.assertTrue(torch.equal(diag.group_multiplicities, torch.tensor([[2, 0, 1, 0]])))
        self.assertEqual(int(diag.k_remote[0]), 1)
        self.assertEqual(int(diag.d_remote[0]), 1)
        self.assertEqual(int(diag.c_remote[0]), 0)
        self.assertEqual(int(diag.d_total[0]), 2)
        hidden = raw.shape[-1]
        self.assertEqual(
            int(diag.endpoints["late_bf16"].logical_payload_bytes[0]), 2 * hidden
        )
        self.assertEqual(
            int(diag.endpoints["pd_full"].logical_payload_bytes[0]), 2 * hidden
        )

    def test_precision_dividend_theorem_per_token(self) -> None:
        raw = _bf16(
            [
                [[1], [2], [3], [4]],
                [[5], [6], [7], [8]],
                [[9], [10], [11], [12]],
            ]
        )
        # contiguous E8/EP4 gives expert pairs per rank.
        experts = torch.tensor(
            [
                [0, 1, 2, 3],  # two collided remote groups
                [0, 1, 4, 6],  # one collision plus two singletons
                [0, 2, 4, 6],  # all singleton groups
            ]
        )
        result = self._run(
            raw,
            experts,
            num_experts=8,
            ep_size=4,
            home_domains=torch.tensor([3, 1, 1]),
            threshold=0.0,
        )
        diag = result.diagnostics
        self.assertTrue(bool((diag.c_remote <= diag.k_remote - diag.d_remote).all()))
        for name in ("pd_full", "pd_gated"):
            endpoint = diag.endpoints[name]
            self.assertTrue(bool((endpoint.n32 <= diag.c_remote).all()))
            expected = torch.where(
                diag.eligible,
                2 * raw.shape[-1] * (diag.d_remote + endpoint.n32),
                2 * raw.shape[-1] * diag.k_remote,
            ) if name == "pd_gated" else 2 * raw.shape[-1] * (
                diag.d_remote + endpoint.n32
            )
            self.assertTrue(torch.equal(endpoint.logical_payload_bytes, expected))
            self.assertTrue(bool(endpoint.payload_cap_holds.all()))

    def test_d_total_one_is_ineligible_even_with_remote_collision(self) -> None:
        raw = _bf16([[[256], [1], [1]]])
        experts = torch.tensor([[0, 1, 2]])
        result = self._run(
            raw,
            experts,
            num_experts=8,
            ep_size=2,
            home_domains=torch.tensor([1]),
            threshold=0.0,
        )
        diag = result.diagnostics
        self.assertEqual(int(diag.d_total[0]), 1)
        self.assertEqual(int(diag.c_remote[0]), 1)
        self.assertFalse(bool(diag.eligible[0]))
        self.assertEqual(int(diag.endpoints["pd_gated"].n32[0]), 0)
        self.assertEqual(int(diag.endpoints["pd_gated"].minimal_bitmap_bytes[0]), 0)
        self.assertTrue(
            torch.equal(result.outputs["pd_gated"], result.outputs["late_bf16"])
        )

    def test_all_singleton_groups_are_ineligible(self) -> None:
        raw = _bf16([[[1], [2], [3], [4]]])
        experts = torch.tensor([[0, 2, 4, 6]])
        result = self._run(
            raw,
            experts,
            num_experts=8,
            ep_size=4,
            home_domains=torch.tensor([0]),
            threshold=0.0,
        )
        diag = result.diagnostics
        self.assertEqual(int(diag.d_total[0]), 4)
        self.assertEqual(int(diag.c_remote[0]), 0)
        self.assertFalse(bool(diag.eligible[0]))
        self.assertTrue(
            torch.equal(result.outputs["pd_gated"], result.outputs["late_bf16"])
        )

    def test_top2_is_an_automatic_dynamic_noop(self) -> None:
        raw = _bf16([[[256], [1]], [[2], [3]]])
        experts = torch.tensor([[0, 1], [0, 2]])
        result = self._run(
            raw,
            experts,
            num_experts=4,
            ep_size=2,
            home_domains=torch.tensor([1, 1]),
            threshold=0.0,
        )
        self.assertFalse(bool(result.diagnostics.eligible.any()))
        self.assertTrue(
            torch.equal(result.outputs["pd_gated"], result.outputs["late_bf16"])
        )
        self.assertTrue(
            torch.equal(
                result.diagnostics.endpoints["pd_gated"].minimal_bitmap_bytes,
                torch.zeros(2, dtype=torch.int64),
            )
        )

    def test_pd_full_equals_uniform_fp32_with_no_more_payload(self) -> None:
        # Home singleton in domain 0, collided remote domain 1, remote singleton
        # in domain 2.  PD-Full saves one BF16 vector versus uniform FP32.
        raw = _bf16([[[1, -1], [256, 2], [1, 3], [4, 5]]])
        experts = torch.tensor([[0, 2, 3, 4]])
        result = self._run(
            raw,
            experts,
            num_experts=6,
            ep_size=3,
            home_domains=torch.tensor([0]),
        )
        self.assertTrue(
            torch.equal(
                result.outputs["pd_full"], result.outputs["uniform_early_fp32"]
            )
        )
        pd_bytes = result.diagnostics.endpoints["pd_full"].logical_payload_bytes
        uniform_bytes = result.diagnostics.endpoints[
            "uniform_early_fp32"
        ].logical_payload_bytes
        self.assertTrue(bool((pd_bytes <= uniform_bytes).all()))
        self.assertLess(int(pd_bytes[0]), int(uniform_bytes[0]))

    def test_residual_threshold_endpoints_and_dynamic_bitmap(self) -> None:
        # Domain 1 subtotal is 257: BF16 recast is 256 and residual RMS is 1.
        # The home value -256 makes the final endpoint difference observable.
        raw = _bf16([[[-256], [256], [1]]])
        experts = torch.tensor([[0, 2, 3]])
        common = dict(
            raw=raw,
            experts=experts,
            num_experts=4,
            ep_size=2,
            home_domains=torch.tensor([0]),
        )
        all_fp32 = self._run(**common, threshold=0.0)
        all_bf16 = self._run(**common, threshold=math.inf)

        self.assertTrue(bool(all_fp32.diagnostics.eligible[0]))
        self.assertEqual(int(all_fp32.diagnostics.endpoints["pd_gated"].n32[0]), 1)
        self.assertEqual(int(all_bf16.diagnostics.endpoints["pd_gated"].n32[0]), 0)
        self.assertEqual(
            int(all_fp32.diagnostics.endpoints["pd_gated"].minimal_bitmap_bytes[0]),
            1,
        )
        self.assertTrue(
            torch.equal(all_fp32.outputs["pd_gated"], all_fp32.outputs["pd_full"])
        )
        self.assertTrue(
            torch.equal(
                all_bf16.outputs["pd_gated"],
                all_bf16.outputs["clean_early_bf16"],
            )
        )
        self.assertTrue(
            torch.equal(all_fp32.outputs["pd_gated"], _bf16([[1]]))
        )
        self.assertTrue(
            torch.equal(all_bf16.outputs["pd_gated"], _bf16([[0]]))
        )

    def test_contiguous_round_robin_and_ranks_per_domain(self) -> None:
        ids = torch.arange(8)
        contiguous = build_expert_to_rank(8, 4, "contiguous")
        round_robin = build_expert_to_rank(8, 4, "round_robin")
        self.assertTrue(
            torch.equal(contiguous, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
        )
        self.assertTrue(
            torch.equal(round_robin, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
        )
        self.assertTrue(
            torch.equal(
                rank_domain_ids(contiguous, ep_size=4, ranks_per_domain=2),
                torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            )
        )

        raw = torch.ones((1, 8, 1), dtype=torch.bfloat16)
        experts = ids.unsqueeze(0)
        contiguous_result = self._run(
            raw,
            experts,
            num_experts=8,
            ep_size=4,
            ranks_per_domain=2,
            placement="contiguous",
            home_domains=torch.tensor([0]),
        )
        round_robin_result = self._run(
            raw,
            experts,
            num_experts=8,
            ep_size=4,
            ranks_per_domain=2,
            placement="round_robin",
            home_domains=torch.tensor([0]),
        )
        self.assertTrue(
            torch.equal(
                contiguous_result.diagnostics.group_multiplicities,
                torch.tensor([[4, 4]]),
            )
        )
        self.assertTrue(
            torch.equal(
                round_robin_result.diagnostics.group_multiplicities,
                torch.tensor([[4, 4]]),
            )
        )
        self.assertTrue(
            torch.equal(
                contiguous_result.canonical_source_ranks,
                torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]]),
            )
        )
        self.assertTrue(
            torch.equal(
                round_robin_result.canonical_source_ranks,
                torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            )
        )

        records = round_robin_result.diagnostics.recorder_records()
        self.assertEqual(records["aggregate"]["tokens"], 1)
        self.assertEqual(records["aggregate"]["eligible_k_remote"], 4)
        self.assertEqual(records["aggregate"]["eligible_credit_units"], 3)
        self.assertEqual(records["aggregate"]["remote_credit_units"], 3)
        self.assertEqual(len(records["token_rows"]), 1)
        self.assertEqual(len(records["group_rows"]), 2)
        self.assertTrue(
            all(
                not isinstance(value, torch.Tensor)
                for section in records.values()
                for row in ([section] if isinstance(section, dict) else section)
                for value in row.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
