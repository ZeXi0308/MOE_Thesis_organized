from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from triage_executor import execute_policy_trajectory


def forward(offset: float):
    def call(token, branch_cache):
        old = branch_cache[0][0]
        new = torch.cat([old, torch.full((1, 1, 1, 2), offset)], dim=-2)
        logits = torch.tensor([[[offset, -offset]]])
        return SimpleNamespace(logits=logits, past_key_values=((new,),))
    return call


class ExecutorTests(unittest.TestCase):
    def test_periodic_audit_counters_and_diagnostic_separation(self) -> None:
        summary, rows = execute_policy_trajectory(
            policy="triage_2_4_8",
            initial_cache=((torch.zeros(1, 1, 1, 2),),),
            decode_tokens=torch.ones(1, 4, dtype=torch.long),
            reference_logits=torch.tensor([[1.0, -1.0]] * 4),
            high_forward=forward(1.0),
            low_forward=forward(-1.0),
            discrepancy_threshold=0.1,
            period=2,
            phase=0,
            lockout_following_steps=0,
        )
        self.assertEqual(summary["audit_events"], 2)
        self.assertEqual(summary["total_candidate_forward_calls"], 6)
        self.assertEqual(summary["diagnostic_forward_calls"], 2)
        self.assertEqual(summary["diagnostic_high_forward_calls"] + summary["diagnostic_low_forward_calls"], 2)
        self.assertEqual(summary["physical_low_forward_calls"], 4)
        self.assertEqual(sum(row["candidate_forward_calls"] for row in rows), 6)

    def test_always_low_diagnostics_do_not_change_candidate_cost(self) -> None:
        summary, _ = execute_policy_trajectory(
            policy="always_low",
            initial_cache=((torch.zeros(1, 1, 1, 2),),),
            decode_tokens=torch.ones(1, 3, dtype=torch.long),
            reference_logits=torch.tensor([[1.0, -1.0]] * 3),
            high_forward=forward(1.0),
            low_forward=forward(-1.0),
            discrepancy_threshold=0.1,
        )
        self.assertEqual(summary["total_candidate_forward_calls"], 3)
        self.assertEqual(summary["diagnostic_forward_calls"], 3)
        self.assertEqual(summary["diagnostic_high_forward_calls"], 3)
        self.assertEqual(summary["diagnostic_low_forward_calls"], 0)
        self.assertEqual(summary["served_low_steps"], 3)

    def test_disabling_diagnostics_preserves_periodic_action_and_logits(self) -> None:
        kwargs = dict(
            policy="fixed_2",
            decode_tokens=torch.ones(1, 4, dtype=torch.long),
            reference_logits=torch.tensor([[1.0, -1.0]] * 4),
            high_forward=forward(1.0),
            low_forward=forward(-1.0),
            discrepancy_threshold=10.0,
            period=2,
            phase=0,
        )
        enabled, enabled_rows = execute_policy_trajectory(
            initial_cache=((torch.zeros(1, 1, 1, 2),),), fingerprint_final_cache=True, **kwargs
        )
        disabled, disabled_rows = execute_policy_trajectory(
            initial_cache=((torch.zeros(1, 1, 1, 2),),),
            collect_diagnostics=False,
            fingerprint_final_cache=True,
            **kwargs,
        )
        self.assertEqual(
            [row["served_action"] for row in enabled_rows],
            [row["served_action"] for row in disabled_rows],
        )
        self.assertEqual(
            [row["served_logits_sha256"] for row in enabled_rows],
            [row["served_logits_sha256"] for row in disabled_rows],
        )
        self.assertEqual(enabled["total_candidate_forward_calls"], disabled["total_candidate_forward_calls"])
        self.assertGreater(enabled["diagnostic_forward_calls"], disabled["diagnostic_forward_calls"])
        self.assertIsNone(disabled["dangerous_step_recall"])
        self.assertEqual(enabled["final_cache_sha256"], disabled["final_cache_sha256"])


if __name__ == "__main__":
    unittest.main()
