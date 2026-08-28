from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import replay_capacity_oracle  # noqa: E402
import run_causal_controller  # noqa: E402


class GateTests(unittest.TestCase):
    def test_frozen_action_is_active_token_budget_across_gates(self) -> None:
        self.assertEqual(
            replay_capacity_oracle.ACTION, "next_window_active_token_budget"
        )
        self.assertEqual(
            run_causal_controller.ACTION, "next_window_active_token_budget"
        )

    def test_p2_rejects_smoke_signal(self) -> None:
        result = replay_capacity_oracle.gate(
            {
                "p1_status": "WEAK_SIGNAL_NEEDS_MORE_EVENTS",
                "scientific_result_eligible": False,
            }
        )
        self.assertEqual(result["status"], "BLOCKED_P1_NOT_ELIGIBLE")
        self.assertEqual(result["action"], "next_window_active_token_budget")
        self.assertFalse(result["executed"])

    def test_p3_rejects_unpassed_oracle(self) -> None:
        result = run_causal_controller.gate({"status": "READY_TO_IMPLEMENT_P2_REPLAY"})
        self.assertEqual(result["status"], "BLOCKED_P2_NOT_PASSED")
        self.assertFalse(result["executed"])

    def test_p2_rejects_truthy_string_and_wrong_action(self) -> None:
        result = replay_capacity_oracle.gate(
            {
                "schema": "route-shape-slo-p1-summary-v1",
                "action": "wrong_action",
                "p1_status": "P1_INCREMENTAL_SIGNAL_PASS",
                "scientific_result_eligible": "false",
                "eligibility_checks": {"all": True},
                "eligibility_blockers": [],
            }
        )
        self.assertEqual(result["status"], "BLOCKED_P1_NOT_ELIGIBLE")

    def test_p2_rejects_invented_or_incomplete_eligibility_map(self) -> None:
        base = {
            "schema": "route-shape-slo-p1-summary-v1",
            "action": "next_window_active_token_budget",
            "p1_status": "P1_INCREMENTAL_SIGNAL_PASS",
            "scientific_result_eligible": True,
            "p1_gate_eligible": True,
            "eligibility_blockers": [],
        }
        for checks in ({"invented": True}, {}):
            result = replay_capacity_oracle.gate(
                {**base, "eligibility_checks": checks}
            )
            self.assertEqual(result["status"], "BLOCKED_P1_NOT_ELIGIBLE")

        exact_checks = {
            name: True
            for name in replay_capacity_oracle.REQUIRED_P1_ELIGIBILITY_CHECKS
        }
        for blockers in (None, "", {}):
            result = replay_capacity_oracle.gate(
                {
                    **base,
                    "eligibility_checks": exact_checks,
                    "eligibility_blockers": blockers,
                }
            )
            self.assertEqual(result["status"], "BLOCKED_P1_NOT_ELIGIBLE")

    def test_p2_accepts_only_exact_eligible_contract(self) -> None:
        result = replay_capacity_oracle.gate(
            {
                "schema": "route-shape-slo-p1-summary-v1",
                "action": "next_window_active_token_budget",
                "p1_status": "P1_INCREMENTAL_SIGNAL_PASS",
                "scientific_result_eligible": True,
                "p1_gate_eligible": True,
                "eligibility_checks": {
                    name: True
                    for name in replay_capacity_oracle.REQUIRED_P1_ELIGIBILITY_CHECKS
                },
                "eligibility_blockers": [],
            }
        )
        self.assertEqual(result["status"], "READY_TO_IMPLEMENT_P2_REPLAY")
        self.assertFalse(result["executed"])

    def test_p3_rejects_unbound_positive_status(self) -> None:
        result = run_causal_controller.gate(
            {"status": "P2_ORACLE_HEADROOM_PASS"}
        )
        self.assertEqual(result["status"], "BLOCKED_P2_NOT_PASSED")
        self.assertEqual(result["action"], "next_window_active_token_budget")

    def test_p3_accepts_only_executed_bound_p2_contract(self) -> None:
        result = run_causal_controller.gate(
            {
                "schema": "route-shape-slo-p2-summary-v1",
                "action": "next_window_active_token_budget",
                "status": "P2_ORACLE_HEADROOM_PASS",
                "scientific_result_eligible": True,
                "executed": True,
            }
        )
        self.assertEqual(result["status"], "READY_TO_IMPLEMENT_P3_CONTROLLER")
        self.assertFalse(result["executed"])


if __name__ == "__main__":
    unittest.main()
