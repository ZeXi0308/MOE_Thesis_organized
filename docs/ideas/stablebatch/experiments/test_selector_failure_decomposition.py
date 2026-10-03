#!/usr/bin/env python3
"""Pure-CPU contract tests for selector failure decomposition."""

from __future__ import annotations

from fractions import Fraction
import copy
import json
import unittest

import numpy as np

from docs.ideas.stablebatch.experiments import selector_failure_decomposition as decomp


TOP_K = 8


def raw_cell(
    identity: str,
    document: str,
    layer: int,
    utilities: list[int],
    *,
    weight: float = 1.0,
) -> dict[str, object]:
    """Build the raw Sparse-C8 ledger schema, not the policy schema."""

    if len(utilities) != TOP_K:
        raise ValueError("synthetic cell must contain eight utilities")
    actions: dict[str, dict[str, int]] = {}
    for rank, utility in enumerate(utilities):
        recovered = max(utility, 0)
        harmed = max(-utility, 0)
        actions[str(rank)] = {
            "expert_id": rank,
            "route_recovered_count": recovered,
            "route_harmed_count": harmed,
            "route_net_reward": recovered - harmed,
            "utility": recovered - harmed,
            # Deliberately unrelated: decomposition labels must be route labels.
            "final_logit_recovered_count": 1000 + rank,
            "final_logit_harmed_count": 2000 + rank,
            "final_logit_net_reward": -1000,
        }
    return {
        "cell_identity": identity,
        "document_text_sha256": document,
        "document_index": int(identity.removeprefix("c") or 0)
        if identity.removeprefix("c").isdigit()
        else 0,
        "layer": layer,
        "expert_ids": list(range(TOP_K)),
        "gate_weights": [float(weight)] * TOP_K,
        "current_layer_topk_cutoff_margin": 0.125,
        "c8_actions": actions,
    }


def exact(value: object) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, dict):
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    raise TypeError(f"not an exact value: {type(value)!r}")


class SelectorFailureDecompositionTest(unittest.TestCase):
    def test_raw_schema_conversion_uses_route_labels(self):
        actions = decomp.surface_actions(
            [raw_cell("c0", "doc-a", 0, [3, -2, 0, 0, 0, 0, 0, 0])]
        )
        self.assertEqual(len(actions), TOP_K)
        self.assertEqual(
            (actions[0]["recovered"], actions[0]["harmed"], actions[0]["utility"]),
            (3, 0, 3),
        )
        self.assertEqual(
            (actions[1]["recovered"], actions[1]["harmed"], actions[1]["utility"]),
            (0, 2, -2),
        )
        self.assertNotEqual(actions[0]["utility"], -1000)

        broken = raw_cell("c0", "doc-a", 0, [0] * TOP_K)
        broken["c8_actions"]["0"]["route_net_reward"] = 1  # type: ignore[index]
        with self.assertRaises(decomp.DecompositionError):
            decomp.surface_actions([broken])

    def test_residual8_closes_exactly_in_every_cell(self):
        actions = decomp.surface_actions(
            [
                raw_cell("c0", "doc-a", 0, [3, -1, 0, 0, 0, 0, 0, 0]),
                raw_cell("c1", "doc-b", 1, [1, 1, 1, 1, 1, 1, 1, 2]),
            ]
        )
        by_cell: dict[str, list[dict[str, object]]] = {}
        for row in actions:
            by_cell.setdefault(str(row["cell_identity"]), []).append(row)
        self.assertEqual(set(by_cell), {"c0", "c1"})
        for rows in by_cell.values():
            self.assertEqual(len(rows), TOP_K)
            self.assertTrue(all(isinstance(row["residual8"], int) for row in rows))
            self.assertEqual(sum(int(row["residual8"]) for row in rows), 0)
        self.assertEqual([int(row["residual8"]) for row in by_cell["c0"]], [22, -10] + [-2] * 6)

    def test_generic_ridge_never_penalizes_optional_intercept(self):
        matrix = np.zeros((5, 3), dtype=np.float64)
        target = np.full(5, 7.0, dtype=np.float64)

        with_intercept = decomp.fit_ridge(
            matrix, target, alpha=1.0, fit_intercept=True
        )
        self.assertFalse(with_intercept["intercept_penalized"])
        self.assertAlmostEqual(float(with_intercept["intercept"]), 7.0)
        np.testing.assert_allclose(
            decomp.predict_ridge(matrix, with_intercept), target, atol=1e-12
        )

        without_intercept = decomp.fit_ridge(
            matrix, target, alpha=1.0, fit_intercept=False
        )
        self.assertEqual(float(without_intercept["intercept"]), 0.0)
        np.testing.assert_allclose(
            decomp.predict_ridge(matrix, without_intercept), np.zeros(5), atol=1e-12
        )

    def test_profile_keeps_integer_residual8_and_rank_ties_choose_smallest(self):
        # residual8 is [7, -1, ..., -1]. Dividing by eight and int-casting
        # would collapse every training label to zero and fail these signs.
        actions = decomp.surface_actions(
            [raw_cell("c0", "doc-a", 0, [1, 0, 0, 0, 0, 0, 0, 0])]
        )
        model = decomp.fit_hierarchical_profile(actions, shrinkage_lambda=4.0)
        positive = decomp.profile_score(actions[0], model)["score"]
        negative = decomp.profile_score(actions[1], model)["score"]
        self.assertGreater(positive, 0.0)
        self.assertLess(negative, 0.0)

        flat = decomp.surface_actions(
            [raw_cell("c1", "doc-b", 1, [0] * TOP_K)]
        )
        flat_model = decomp.fit_hierarchical_profile(flat, shrinkage_lambda=4.0)
        scored = [
            {**row, "profile_score": decomp.profile_score(row, flat_model)["score"]}
            for row in reversed(flat)
        ]
        selected = decomp.select_ranks(
            scored, score_key="profile_score", maximize=True
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["rank"], 0)

    def test_exact_budget_random_baselines_and_one_rank_per_cell(self):
        actions = decomp.surface_actions(
            [
                raw_cell(f"c{index}", f"doc-{index // 4}", index % 2, [index] * TOP_K)
                for index in range(40)
            ]
        )
        selected = [
            {"cell_identity": f"c{index}", "rank": 0} for index in range(33)
        ]
        result = decomp.evaluate_exact_budget(actions, selected, budget=33)
        self.assertEqual(result["selector"]["actions"], 33)
        self.assertEqual(len(result["selector_selected"]), 33)
        self.assertEqual(
            len({row["cell_identity"] for row in result["selector_selected"]}), 33
        )
        self.assertEqual(
            exact(result["global_matched_random"]["net"]), Fraction(1287, 2)
        )
        self.assertEqual(
            exact(result["cell_matched_random_rank"]["net"]), Fraction(528)
        )
        self.assertEqual(result["selector"]["net"], 528)
        # Every exact quantity must already be encoded for durable JSON output.
        json.dumps(result)

        duplicate_cell = selected[:32] + [{"cell_identity": "c0", "rank": 1}]
        with self.assertRaises(decomp.DecompositionError):
            decomp.evaluate_exact_budget(actions, duplicate_cell, budget=33)

    def test_fresh_transform_reuses_training_normalizer(self):
        training = decomp.surface_actions(
            [
                raw_cell("c0", "doc-a", 0, [0] * TOP_K, weight=1.0),
                raw_cell("c1", "doc-b", 1, [0] * TOP_K, weight=2.0),
            ]
        )
        fresh = decomp.surface_actions(
            [raw_cell("c2", "doc-c", 0, [0] * TOP_K, weight=10.0)]
        )
        schema = decomp.fit_action_feature_schema(
            training, num_layers=2, num_experts=TOP_K, top_k=TOP_K
        )
        frozen_schema = copy.deepcopy(schema)
        matrix = decomp.transform_action_features(fresh, schema)
        self.assertEqual(schema, frozen_schema)
        self.assertAlmostEqual(schema["continuous_mean"]["topk_mass"], 12.0)
        self.assertAlmostEqual(schema["continuous_std"]["topk_mass"], 4.0)
        column = schema["feature_names"].index("z_topk_mass")
        # Fresh mass is 80. Reusing broad mean/std gives (80-12)/4 = 17;
        # refitting on fresh would incorrectly yield zero.
        np.testing.assert_allclose(matrix[:, column], np.full(TOP_K, 17.0))


if __name__ == "__main__":
    unittest.main()
