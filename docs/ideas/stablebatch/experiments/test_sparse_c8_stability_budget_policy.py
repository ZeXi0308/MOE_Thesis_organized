#!/usr/bin/env python3
"""Unit tests for the pure-CPU Sparse-C8 StabilityBudget policy."""

from __future__ import annotations

from fractions import Fraction
import unittest

from docs.ideas.stablebatch.experiments import sparse_c8_stability_budget_policy as policy


def action(net: int) -> dict[str, int]:
    return {
        "recovered": max(net, 0),
        "harmed": max(-net, 0),
        "net": net,
    }


def cell(
    identity: str,
    document: str,
    layer: int,
    nets: list[int] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "cell_identity": identity,
        "document_id": document,
        "layer": layer,
        "expert_ids": list(range(8)),
        "gate_weights": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        "current_layer_topk_cutoff_margin": 0.05 + layer * 0.01,
    }
    if nets is not None:
        row["actions"] = {str(rank): action(net) for rank, net in enumerate(nets)}
    return row


def exact(payload: dict[str, object]) -> Fraction:
    return Fraction(int(payload["numerator"]), int(payload["denominator"]))


class SparseC8StabilityBudgetPolicyTest(unittest.TestCase):
    def test_flatten_derives_net_and_rejects_mismatched_label(self):
        rows = policy.flatten_outcome_cells(
            [cell("c0", "d0", 0, [3, 0, -2, 0, 0, 0, 0, 0])]
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["net"], 3)
        self.assertEqual(rows[2]["net"], -2)

        broken = cell("bad", "d0", 0, [0] * 8)
        broken["actions"]["0"]["reward"] = 1  # type: ignore[index]
        with self.assertRaises(policy.PolicyError):
            policy.flatten_outcome_cells([broken])

    def test_ridge_uses_only_frozen_action_pre_feature_family(self):
        training = policy.flatten_outcome_cells(
            [
                cell("c0", "d0", 0, [4, 0, 0, 0, 0, 0, 0, 0]),
                cell("c1", "d1", 1, [3, 0, 0, 0, 0, 0, 0, 0]),
            ]
        )
        for row in training:
            row["historical_sensitivity"] = 999999
        model = policy.fit_action_pre_ridge(
            training, num_layers=2, num_experts=8, top_k=8
        )
        self.assertEqual(model["alpha"], 1.0)
        self.assertFalse(model["intercept_penalized"])
        self.assertEqual(model["outcome_derived_features"], [])
        self.assertEqual(model["label"], "net=recovered-harmed")
        self.assertNotIn("historical_sensitivity", model["feature_names"])
        self.assertEqual(
            set(model["continuous_mean"]), set(policy.CONTINUOUS_FEATURES)
        )
        scored = policy.predict_action_scores(training, model)
        self.assertEqual(len(scored), 16)
        self.assertTrue(all("net" not in row for row in scored))

    def test_selector_enforces_rank_then_cell_tie_break_and_exact_budget(self):
        scored = []
        for identity in ("cell-c", "cell-a", "cell-b"):
            for rank in reversed(range(8)):
                scored.append(
                    {
                        "cell_identity": identity,
                        "document_id": identity[-1],
                        "layer": 0,
                        "rank": rank,
                        "expert_id": rank,
                        "predicted_utility": 1.0,
                    }
                )
        plan = policy.select_global_exact_b(scored, budget=2)
        self.assertEqual(len(plan["selected"]), 2)
        self.assertEqual(
            [(row["cell_identity"], row["rank"]) for row in plan["selected"]],
            [("cell-a", 0), ("cell-b", 0)],
        )

    def test_exact_random_baselines_gains_oracles_and_headroom(self):
        outcomes = policy.flatten_outcome_cells(
            [
                cell("a", "doc-1", 0, [8, 0, 0, 0, 0, 0, 0, 0]),
                cell("b", "doc-1", 1, [4, 4, 4, 4, 4, 4, 4, 4]),
                cell("c", "doc-2", 0, [-8, 0, 0, 0, 0, 0, 0, 0]),
            ]
        )
        selected = [
            {"cell_identity": "a", "rank": 0},
            {"cell_identity": "c", "rank": 1},
        ]
        result = policy.evaluate_budget_decomposition(
            outcomes, selected, budget=2
        )
        self.assertEqual(exact(result["global_matched_random"]["net"]), Fraction(8, 3))
        self.assertEqual(exact(result["cell_matched_random_rank"]["net"]), 0)
        self.assertEqual(result["selector"]["net"], 8)
        self.assertEqual(result["oracle_exact_B"]["net"], 12)
        self.assertEqual(result["oracle_at_most_B"]["net"], 12)
        self.assertEqual(
            exact(result["decomposition"]["cell_selection_gain"]), Fraction(-8, 3)
        )
        self.assertEqual(exact(result["decomposition"]["rank_selection_gain"]), 8)
        self.assertEqual(
            exact(result["decomposition"]["rank_headroom_capture"]), Fraction(2, 3)
        )
        self.assertEqual(set(result["per_document"]), {"doc-1", "doc-2"})
        self.assertEqual(result["per_document"]["doc-2"]["selector"]["actions"], 1)

    def test_oracle_exact_b_keeps_negative_action_while_at_most_b_abstains(self):
        outcomes = policy.flatten_outcome_cells(
            [
                cell("a", "doc", 0, [5, 1, 1, 1, 1, 1, 1, 1]),
                cell("b", "doc", 1, [-1, -2, -2, -2, -2, -2, -2, -2]),
            ]
        )
        result = policy.evaluate_budget_decomposition(
            outcomes,
            [{"cell_identity": "a", "rank": 0}, {"cell_identity": "b", "rank": 0}],
            budget=2,
        )
        self.assertEqual(result["oracle_exact_B"]["actions"], 2)
        self.assertEqual(result["oracle_exact_B"]["net"], 4)
        self.assertEqual(result["oracle_at_most_B"]["actions"], 1)
        self.assertEqual(result["oracle_at_most_B"]["net"], 5)

    def test_selector_rejects_multiple_actions_from_one_cell(self):
        outcomes = policy.flatten_outcome_cells(
            [cell("a", "doc", 0, [1] * 8), cell("b", "doc", 1, [0] * 8)]
        )
        with self.assertRaises(policy.PolicyError):
            policy.evaluate_budget_decomposition(
                outcomes,
                [{"cell_identity": "a", "rank": 0}, {"cell_identity": "a", "rank": 1}],
                budget=2,
            )


if __name__ == "__main__":
    unittest.main()
