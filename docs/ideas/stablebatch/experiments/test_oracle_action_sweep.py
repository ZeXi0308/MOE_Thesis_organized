#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_oracle_action_sweep as oracle


def config_fixture() -> dict:
    return {
        "source": {
            "expected_closure": {
                "cell_count": 2,
                "independent_documents": 2,
                "maxgate_rank": 0,
                "maxgate_total_reward": -1,
                "frozen_shuffle_total_reward": 1,
            }
        },
        "action_space": {"candidate_ranks": [0, 1]},
        "budget_curve": {"budgets_in_protected_cells": [1, 2]},
        "signal": {
            "strong_if": {
                "min_abstaining_oracle_recovery_fraction": 0.5,
                "min_abstaining_oracle_advantage_over_each_budget_matched_random_expected_reward": 1.0,
                "min_distinct_victims_with_positive_oracle_reward": 1,
            }
        },
        "scope_of_failure": "fixture",
    }


def row_fixture(
    victim_id: str,
    layer: int,
    d_u: int,
    rewards: list[int],
    shuffled_rank: int,
) -> dict:
    best_rank = min(range(len(rewards)), key=lambda rank: (-rewards[rank], rank))
    best_reward = rewards[best_rank]
    return {
        "victim_id": victim_id,
        "layer": layer,
        "integrity_status": "PASS",
        "unprotected_distance_vs_R": d_u,
        "source_shuffled_rank": shuffled_rank,
        "actions": {
            str(rank): {
                "reward": reward,
                "full_restoration": bool(d_u > 0 and reward == d_u),
            }
            for rank, reward in enumerate(rewards)
        },
        "forced_oracle_rank": best_rank,
        "forced_oracle_reward": best_reward,
        "selected_positive_action_confirmation": (
            {"status": "PASS"} if best_reward > 0 else None
        ),
    }


class OracleActionSweepTests(unittest.TestCase):
    def test_candidate_surface_has_exactly_one_protected_rank(self) -> None:
        self.assertEqual(
            oracle.candidate_surface(2, 4),
            {0: 64, 1: 64, 2: 1, 3: 64},
        )
        with self.assertRaises(oracle.ProtocolError):
            oracle.candidate_surface(4, 4)

    def test_all_surfaces_enumerates_reference_unprotected_and_every_rank(self) -> None:
        surfaces = oracle.all_surfaces(3)
        self.assertEqual(set(surfaces), {"R", "U", "A0", "A1", "A2"})
        self.assertEqual(surfaces["R"], {0: 1, 1: 1, 2: 1})
        self.assertEqual(surfaces["U"], {0: 64, 1: 64, 2: 64})
        for rank in range(3):
            values = list(surfaces[f"A{rank}"].values())
            self.assertEqual(values.count(1), 1)
            self.assertEqual(values.count(64), 2)

    def test_deterministic_arm_order_is_complete_and_repeatable(self) -> None:
        labels = ["R", "U", "A0", "A1"]
        first = oracle.deterministic_arm_order("victim|layer=00", labels, "seed")
        second = oracle.deterministic_arm_order("victim|layer=00", labels, "seed")
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(labels))

    def test_classify_results_recomputes_oracle_random_and_closures(self) -> None:
        rows = [
            row_fixture("doc-a", 0, 2, [0, 2], 1),
            row_fixture("doc-b", 0, 1, [-1, 0], 0),
        ]
        summary = oracle.classify_results(rows, config_fixture())
        self.assertEqual(summary["maxgate_v1_total_reward"], -1)
        self.assertEqual(summary["frozen_shuffle_total_reward"], 1)
        self.assertEqual(
            summary["uniform_random_one_action_per_cell_expected_total_reward"], 0.5
        )
        self.assertEqual(summary["forced_oracle_total_reward"], 2)
        self.assertEqual(summary["abstaining_oracle_total_reward"], 2)
        self.assertEqual(summary["abstaining_oracle_action_budget"], 1)
        self.assertEqual(summary["budget_matched_global_random_expected_reward"], 0.25)
        self.assertEqual(
            summary["budget_matched_conditional_random_expected_reward"], 1.0
        )
        self.assertAlmostEqual(summary["abstaining_oracle_recovery_fraction"], 2 / 3)
        self.assertEqual(summary["positive_oracle_cell_count"], 1)
        self.assertEqual(summary["verdict"], "STRONG_ORACLE_ACTION_VALUE_SIGNAL")
        self.assertEqual(summary["budget_curve"][0]["oracle_total_reward"], 2)
        self.assertEqual(
            summary["budget_curve"][0][
                "conditional_uniform_random_rank_expected_reward"
            ],
            1.0,
        )

    def test_classify_results_fails_closed_on_source_closure_mismatch(self) -> None:
        rows = [
            row_fixture("doc-a", 0, 2, [0, 2], 1),
            row_fixture("doc-b", 0, 1, [-1, 0], 0),
        ]
        config = config_fixture()
        config["source"]["expected_closure"]["maxgate_total_reward"] = 99
        with self.assertRaises(oracle.ProtocolError):
            oracle.classify_results(rows, config)

    def test_classify_results_requires_confirmation_for_positive_choice(self) -> None:
        rows = [
            row_fixture("doc-a", 0, 2, [0, 2], 1),
            row_fixture("doc-b", 0, 1, [-1, 0], 0),
        ]
        rows[0]["selected_positive_action_confirmation"] = None
        with self.assertRaises(oracle.ProtocolError):
            oracle.classify_results(rows, config_fixture())


if __name__ == "__main__":
    unittest.main()
