#!/usr/bin/env python3

import unittest

try:
    from .explore_receiver_matched_milp import Candidate, MatchedWindow, empirical_cvar, evaluate_order, exhaustive_optimum, milp_optimum, policy_order
except ImportError:
    from explore_receiver_matched_milp import Candidate, MatchedWindow, empirical_cvar, evaluate_order, exhaustive_optimum, milp_optimum, policy_order


class MatchedMilpCoreTests(unittest.TestCase):
    def fixture(self) -> MatchedWindow:
        tasks = tuple(Candidate(str(i), str(i), "r", 0.0, 1.0, i % 2, 0, i, i) for i in range(4))
        return MatchedWindow("m", "holdout_half", 0, 0, tasks,
                             ({"0": 2.0, "1": 0.0, "2": 2.0, "3": 0.0},
                              {"0": 0.0, "1": 2.0, "2": 0.0, "3": 2.0}),
                             0.8, 0.0, 0.0, 2.0, "f")

    def test_worlds_have_equal_aggregate_but_different_keys(self) -> None:
        fixture = self.fixture()
        self.assertEqual(sum(fixture.residual_by_world[0].values()), sum(fixture.residual_by_world[1].values()))
        self.assertNotEqual(fixture.residual_by_world[0], fixture.residual_by_world[1])

    def test_b_uses_same_order_both_worlds(self) -> None:
        fixture = self.fixture()
        optimum = exhaustive_optimum(fixture, (0, 1))
        cvar, mean, flows = evaluate_order(fixture, optimum["order"], (0, 1))
        self.assertAlmostEqual(cvar, optimum["cvar99"])
        self.assertAlmostEqual(mean, optimum["mean"])
        self.assertEqual(len(flows), 8)

    def test_receiver_policy_changes_with_keyed_world(self) -> None:
        fixture = self.fixture()
        self.assertNotEqual(policy_order(fixture, "receiver_shadow_price", 0)[0],
                            policy_order(fixture, "receiver_shadow_price", 1)[0])

    def test_empirical_cvar_not_hardcoded_to_max(self) -> None:
        self.assertAlmostEqual(empirical_cvar([1.0, 2.0], 0.0), 1.5)
        self.assertAlmostEqual(empirical_cvar([1.0, 2.0], 0.99), 2.0)

    def test_milp_matches_enumeration_when_scipy_available(self) -> None:
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("SciPy is unavailable")
        result = milp_optimum(self.fixture(), (0, 1))
        exact = exhaustive_optimum(self.fixture(), (0, 1))
        self.assertAlmostEqual(result["cvar99"], exact["cvar99"], places=6)
        self.assertAlmostEqual(result["mean"], exact["mean"], places=6)


if __name__ == "__main__":
    unittest.main()
