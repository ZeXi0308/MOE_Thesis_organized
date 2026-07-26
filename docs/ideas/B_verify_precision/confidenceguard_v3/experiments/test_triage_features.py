from __future__ import annotations

import math
import unittest

import torch

from triage_features import FeatureError, extract_full_route_features, prefill_mean_nll


class FeatureTests(unittest.TestCase):
    def test_route_fixture(self) -> None:
        batches = [
            {"layer": 0, "selected_experts": torch.tensor([[0, 1], [0, 2]]), "routing_weights": torch.tensor([[0.8, 0.2], [0.6, 0.4]])},
            {"layer": 1, "selected_experts": torch.tensor([[0, 2], [1, 2]]), "routing_weights": torch.tensor([[0.7, 0.3], [0.9, 0.1]])},
        ]
        result = extract_full_route_features(batches, 4)
        self.assertAlmostEqual(result["full_route_top1_weight_mean"], 0.75)
        self.assertAlmostEqual(result["full_route_tail_mass_mean"], 0.25)
        self.assertAlmostEqual(result["full_route_rank1_hhi_mean"], 0.75)
        self.assertAlmostEqual(result["full_route_active_expert_fraction_mean"], 0.75)
        self.assertAlmostEqual(result["full_route_same_id_adjacent_layer_rate"], 0.5)
        self.assertTrue(0 <= result["full_route_routing_entropy_mean"] <= 1)

    def test_nll(self) -> None:
        logits = torch.zeros(1, 3, 5)
        ids = torch.tensor([[0, 1, 2]])
        self.assertAlmostEqual(prefill_mean_nll(logits, ids), math.log(5), places=6)

    def test_empty_routes_fail(self) -> None:
        with self.assertRaises(FeatureError):
            extract_full_route_features([], 4)


if __name__ == "__main__":
    unittest.main()
