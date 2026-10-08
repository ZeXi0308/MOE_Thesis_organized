"""Boundary checks for the joint model, with no claims about GPU performance."""
from dataclasses import replace
import unittest

from recovery_lease import LeaseConfig, LeaseState, choose_lease, continuation_reason


BASE = LeaseState(prompt_tokens=96, output_tokens=10, max_output_tokens=100,
                  held_blocks=0, free_blocks=8, inflight_reserved_blocks=0,
                  target_age_s=1.2, peer_ages_s=(0.4,),
                  overhead_s=0.06, decode_interval_s=0.01)


class LeaseBoundaries(unittest.TestCase):
    def test_cost_changes_service_only_with_same_feasible_state(self):
        cheap = choose_lease(replace(BASE, overhead_s=0.01))
        expensive = choose_lease(BASE)
        self.assertEqual((cheap.quantum, expensive.quantum), (1, 6))
        self.assertTrue(expensive.amortization_target_met)

    def test_page_boundary_truncates_even_when_q1_fits(self):
        # Q1 at token 111 fits seven blocks; only Q1 and Q2 fit. The sixth
        # output requires an eighth block, so a pre-funding fixed Q6 is invalid.
        decision = choose_lease(replace(BASE, prompt_tokens=101, free_blocks=7))
        self.assertEqual(decision.quantum, 2)
        self.assertEqual(decision.required_total_blocks, 7)
        self.assertEqual(decision.reason, "FRONTIER_TRUNCATED")
        self.assertFalse(decision.amortization_target_met)

    def test_peer_age_changes_quantum_without_changing_kv_or_cost(self):
        # Peer can wait 80ms: 60ms fixed service plus two decode intervals.
        decision = choose_lease(replace(BASE, peer_ages_s=(1.12,)))
        self.assertEqual(decision.quantum, 2)
        self.assertLessEqual(1.12 + decision.estimated_duration_s, 1.2 + 1e-12)
        urgent = choose_lease(replace(BASE, peer_ages_s=(1.19,)))
        self.assertEqual((urgent.quantum, urgent.reason), (1, "Q1_PEER_BUDGET_EXHAUSTED"))

    def test_native_reservations_cannot_be_spent_twice(self):
        decision = choose_lease(replace(BASE, free_blocks=7, inflight_reserved_blocks=1))
        self.assertEqual((decision.quantum, decision.reason), (0, "Q1_PHYSICALLY_UNFUNDED"))

    def test_no_future_eos_needed_and_unknown_cost_preserves_q1(self):
        cap = choose_lease(replace(BASE, max_output_tokens=13))
        self.assertEqual(cap.quantum, 3)
        unknown = choose_lease(replace(BASE, overhead_s=None))
        self.assertEqual(unknown.quantum, 1)

    def test_stale_estimate_new_peer_revokes_extra_service(self):
        args = dict(output_at_start=10, current_output=11, quantum=6,
                    prompt_tokens=96, held_blocks=7, free_blocks=1,
                    inflight_reserved_blocks=0, peer_ages_s=(1.195,),
                    peer_age_limit_s=1.2, next_decode_interval_s=0.01)
        self.assertEqual(continuation_reason(**args), "RELEASE_PEER_AGE_FRONTIER")
        args.update(peer_ages_s=(0.4,), current_output=16)
        self.assertEqual(continuation_reason(**args), "OUTPUT_GOAL_REACHED")


if __name__ == "__main__":
    unittest.main()
