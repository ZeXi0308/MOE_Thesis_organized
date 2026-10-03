#!/usr/bin/env python3

from dataclasses import replace
import copy
import importlib.util
import unittest

try:
    from . import phasemap_milp_crosscheck as crosscheck
    from . import phasemap_oracle_core as core
    from .test_phasemap_oracle_core import primary_fixture
except ImportError:  # pragma: no cover
    import phasemap_milp_crosscheck as crosscheck  # type: ignore
    import phasemap_oracle_core as core  # type: ignore
    from test_phasemap_oracle_core import primary_fixture  # type: ignore


HAS_SCIPY = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(HAS_SCIPY, "SciPy MILP is unavailable in this environment")
class PhaseMapMILPCrosscheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = primary_fixture()
        cls.core_report = core.optimize_information_lattice(cls.scenario)

    def test_all_information_arms_match_enumerator(self):
        report = crosscheck.crosscheck_information_lattice(self.scenario, self.core_report)
        self.assertTrue(report["passed"])
        self.assertLessEqual(report["maximum_solver_mip_gap"], 1e-7)
        self.assertLessEqual(report["maximum_core_objective_gap"], 1e-7)
        self.assertEqual(set(report["arms"]), {"B0", "Q", "J", "R"})
        self.assertIn("not_independent_simulator", report["crosscheck_scope"])
        self.assertEqual(
            {arm: report["arms"][arm]["observation_class_count"] for arm in report["arms"]},
            {"B0": 1, "Q": 2, "J": 2, "R": 4},
        )
        for arm in ("B0", "Q", "J", "R"):
            with self.subTest(arm=arm):
                actual = report["arms"][arm]
                expected = self.core_report["arms"][arm]
                self.assertEqual(actual["optimal_policy_count"], len(expected["optimal_policies"]))
                self.assertEqual(actual["optimal_policies"], expected["optimal_policies"])
                self.assertEqual(
                    actual["selected_canonical_policy"], expected["selected_canonical_policy"]
                )
                self.assertEqual(
                    actual["lexicographic_tolerance"]["relative"],
                    core.LEX_REL_TOLERANCE,
                )

    def test_independent_event_ledger_recomputation_rejects_count_drift(self):
        result = core.simulate(self.scenario, 0, core.enumerate_actions(self.scenario)[0])
        self.assertEqual(len(crosscheck._validate_replay_ledger(result)), 64)
        broken = copy.deepcopy(result)
        broken["accounting"]["decision_cut_count"] += 1
        with self.assertRaisesRegex(crosscheck.PhaseMapMILPError, "accounting"):
            crosscheck._validate_replay_ledger(broken)

    def test_action_sets_are_exact_projections_of_optimal_policies(self):
        arm_report = crosscheck.crosscheck_arm(
            self.scenario,
            "R",
            self.core_report["arms"]["R"],
        )
        policies = arm_report["optimal_policies"]
        projected = {
            key: {dict(policy)[key] for policy in policies}
            for key, _indices in core.observation_partitions(self.scenario, "R")
        }
        self.assertEqual(
            {key: set(actions) for key, actions in arm_report["optimal_action_sets"]},
            projected,
        )

    def test_core_objective_corruption_is_rejected(self):
        report = copy.deepcopy(self.core_report)
        arm = report["arms"]["Q"]
        arm["metrics"] = replace(
            arm["metrics"],
            expected_miss_count=arm["metrics"].expected_miss_count + 0.25,
        )
        with self.assertRaisesRegex(crosscheck.PhaseMapMILPError, "objective mismatch"):
            crosscheck.crosscheck_information_lattice(self.scenario, report)

    def test_core_optimal_policy_set_corruption_is_rejected(self):
        report = copy.deepcopy(self.core_report)
        arm = report["arms"]["R"]
        original = arm["optimal_policies"]
        arm["optimal_policies"] = original[:-1] if len(original) > 1 else ()
        with self.assertRaisesRegex(crosscheck.PhaseMapMILPError, "optimal policy set mismatch"):
            crosscheck.crosscheck_information_lattice(self.scenario, report)

    def test_core_canonical_tie_break_corruption_is_rejected(self):
        # Pick an arm with a tie if one exists; otherwise append the same policy
        # in reversed class order, which is objective-equivalent but not a valid
        # canonical report and must still fail closed.
        report = copy.deepcopy(self.core_report)
        target = next(
            (
                arm
                for arm in ("B0", "Q", "J", "R")
                if len(report["arms"][arm]["optimal_policies"]) > 1
            ),
            None,
        )
        if target is None:
            self.skipTest("fixture has no tied optimal arm")
        arm = report["arms"][target]
        arm["selected_canonical_policy"] = arm["optimal_policies"][-1]
        if arm["selected_canonical_policy"] == arm["optimal_policies"][0]:
            self.skipTest("fixture optimal policies serialize identically")
        with self.assertRaisesRegex(crosscheck.PhaseMapMILPError, "canonical policy mismatch"):
            crosscheck.crosscheck_information_lattice(self.scenario, report)


if __name__ == "__main__":
    unittest.main()
