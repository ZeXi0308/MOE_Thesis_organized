"""Causal and accounting checks for capture-quantized coalesced admission.

These cover exactly the four risks the protocol requires before a GPU run:
future-information leakage, identity/ordering preservation, request accounting,
and action legality. They are not a general coverage exercise.
"""
import unittest

from admission_coalescer import (DEFAULT_CAPTURE_SIZES, CoalescingAdmission,
                                 ceil_capture, padding_waste)


class CaptureGeometry(unittest.TestCase):
    def test_ceil_capture_matches_engine_logged_bucket_edges(self):
        # Edges independently confirmed twice: recovered from the sealed 44-episode
        # timing reconstruction, and printed by the engine itself as
        # cudagraph_capture_sizes [1,2,4,8,16,24,32,...] with 7 decode FULL graphs.
        for width, expected in ((1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8),
                                (9, 16), (16, 16), (17, 24), (24, 24), (25, 32), (32, 32)):
            self.assertEqual(ceil_capture(width, DEFAULT_CAPTURE_SIZES), expected)

    def test_padding_waste_is_zero_only_at_capture_points(self):
        for width in range(1, 33):
            waste = padding_waste(width, DEFAULT_CAPTURE_SIZES)
            self.assertEqual(waste == 0.0, width in DEFAULT_CAPTURE_SIZES)
            self.assertLess(waste, 1.0)
        # The regime the failed ITL feedback controller sat in.
        self.assertAlmostEqual(padding_waste(9, DEFAULT_CAPTURE_SIZES), 1 - 9 / 16)
        self.assertAlmostEqual(padding_waste(10, DEFAULT_CAPTURE_SIZES), 0.375)


class Causality(unittest.TestCase):
    def make(self, **kwargs):
        kwargs.setdefault("group_size", 4)
        kwargs.setdefault("hold_budget_s", 0.150)
        return CoalescingAdmission(**kwargs)

    def test_rejects_offer_before_arrival(self):
        policy = self.make()
        with self.assertRaises(ValueError):
            policy.offer(index=0, arrival_s=1.0, prompt_tokens=128, now_s=0.5)

    def test_rejects_out_of_order_arrivals(self):
        policy = self.make()
        policy.offer(index=0, arrival_s=1.0, prompt_tokens=128, now_s=1.0)
        with self.assertRaises(ValueError):
            policy.offer(index=1, arrival_s=0.5, prompt_tokens=128, now_s=1.0)

    def test_decision_uses_only_already_offered_requests(self):
        policy = self.make(group_size=8, hold_budget_s=10.0)
        for i in range(3):
            policy.offer(index=i, arrival_s=0.05 * i, prompt_tokens=128, now_s=0.05 * i)
        released, row = policy.poll(now_s=0.10, running=0, waiting=0)
        self.assertEqual(released, [])
        self.assertEqual(row["reason"], "hold")
        self.assertEqual(row["held_before"], 3)

    def test_hold_budget_is_a_hard_elapsed_time_deadline(self):
        policy = self.make(group_size=8, hold_budget_s=0.150)
        policy.offer(index=0, arrival_s=0.0, prompt_tokens=128, now_s=0.0)
        self.assertEqual(policy.poll(now_s=0.149, running=0, waiting=0)[0], [])
        released, row = policy.poll(now_s=0.150, running=0, waiting=0)
        self.assertEqual(released, [0])
        self.assertEqual(row["reason"], "hold_budget_expired")
        self.assertAlmostEqual(row["oldest_hold_s"], 0.150)

    def test_zero_hold_budget_never_delays_anything(self):
        """The degenerate setting must reproduce immediate-release semantics."""
        policy = self.make(group_size=8, hold_budget_s=0.0)
        policy.offer(index=0, arrival_s=0.0, prompt_tokens=128, now_s=0.0)
        released, row = policy.poll(now_s=0.0, running=0, waiting=0)
        self.assertEqual(released, [0])
        self.assertEqual(row["reason"], "hold_budget_expired")

    def test_next_deadline_tracks_oldest_held_request(self):
        policy = self.make(hold_budget_s=0.2)
        self.assertIsNone(policy.next_deadline_s())
        policy.offer(index=0, arrival_s=1.0, prompt_tokens=128, now_s=1.0)
        policy.offer(index=1, arrival_s=1.1, prompt_tokens=128, now_s=1.1)
        self.assertAlmostEqual(policy.next_deadline_s(), 1.2)
        policy.poll(now_s=1.2, running=0, waiting=0)
        self.assertIsNone(policy.next_deadline_s())


class ReleaseRules(unittest.TestCase):
    def test_group_completion_releases_in_fcfs_order(self):
        policy = CoalescingAdmission(group_size=4, hold_budget_s=10.0)
        for i in range(4):
            policy.offer(index=10 + i, arrival_s=0.05 * i, prompt_tokens=128, now_s=0.05 * i)
        released, row = policy.poll(now_s=0.15, running=0, waiting=0)
        self.assertEqual(released, [10, 11, 12, 13])
        self.assertEqual(row["reason"], "group_complete")
        self.assertEqual(policy.pending(), 0)

    def test_snapping_releases_early_to_land_on_a_capture_point(self):
        policy = CoalescingAdmission(group_size=8, hold_budget_s=10.0)
        for i in range(3):
            policy.offer(index=i, arrival_s=0.0, prompt_tokens=128, now_s=0.0)
        released, row = policy.poll(now_s=0.01, running=13, waiting=0)
        self.assertEqual(released, [0, 1, 2])
        self.assertEqual(row["reason"], "lands_on_capture_point")
        self.assertEqual(row["projected_width"], 16)
        self.assertEqual(row["projected_padded_width"], 16)

    def test_no_snap_when_projection_stays_inside_a_bucket(self):
        policy = CoalescingAdmission(group_size=8, hold_budget_s=10.0)
        for i in range(2):
            policy.offer(index=i, arrival_s=0.0, prompt_tokens=128, now_s=0.0)
        released, row = policy.poll(now_s=0.01, running=13, waiting=0)
        self.assertEqual(released, [])
        self.assertEqual(row["reason"], "hold")

    def test_snapping_can_be_disabled_for_the_ablation_arm(self):
        policy = CoalescingAdmission(group_size=8, hold_budget_s=10.0,
                                     snap_to_capture_point=False)
        for i in range(3):
            policy.offer(index=i, arrival_s=0.0, prompt_tokens=128, now_s=0.0)
        self.assertEqual(policy.poll(now_s=0.01, running=13, waiting=0)[0], [])

    def test_single_held_request_never_triggers_a_snap_release(self):
        policy = CoalescingAdmission(group_size=8, hold_budget_s=10.0)
        policy.offer(index=0, arrival_s=0.0, prompt_tokens=128, now_s=0.0)
        released, row = policy.poll(now_s=0.01, running=15, waiting=0)
        self.assertEqual(released, [])
        self.assertEqual(row["reason"], "hold")

    def test_waiting_requests_count_towards_projected_width(self):
        policy = CoalescingAdmission(group_size=8, hold_budget_s=10.0)
        for i in range(2):
            policy.offer(index=i, arrival_s=0.0, prompt_tokens=128, now_s=0.0)
        released, row = policy.poll(now_s=0.01, running=12, waiting=2)
        self.assertEqual(released, [0, 1])
        self.assertEqual(row["projected_width"], 16)


class Budgets(unittest.TestCase):
    def test_engine_capacity_limits_the_release_without_dropping_requests(self):
        policy = CoalescingAdmission(group_size=4, hold_budget_s=0.0, engine_max_seqs=32)
        for i in range(4):
            policy.offer(index=i, arrival_s=0.0, prompt_tokens=128, now_s=0.0)
        released, row = policy.poll(now_s=0.0, running=30, waiting=0)
        self.assertEqual(released, [0, 1])
        self.assertEqual(row["releasable"], 2)
        self.assertEqual(policy.pending(), 2)

    def test_full_engine_blocks_release_and_retains_every_request(self):
        policy = CoalescingAdmission(group_size=4, hold_budget_s=0.0, engine_max_seqs=32)
        for i in range(3):
            policy.offer(index=i, arrival_s=0.0, prompt_tokens=128, now_s=0.0)
        released, row = policy.poll(now_s=5.0, running=32, waiting=0)
        self.assertEqual(released, [])
        self.assertEqual(row["reason"], "blocked_by_engine_or_token_budget")
        self.assertEqual(policy.pending(), 3)

    def test_prefill_token_budget_splits_a_release(self):
        policy = CoalescingAdmission(group_size=8, hold_budget_s=0.0,
                                     max_prefill_tokens_per_release=300)
        for i in range(4):
            policy.offer(index=i, arrival_s=0.0, prompt_tokens=128, now_s=0.0)
        released, _ = policy.poll(now_s=0.0, running=0, waiting=0)
        self.assertEqual(released, [0, 1])
        self.assertEqual(policy.pending(), 2)

    def test_oversized_single_request_still_releases_alone(self):
        """A prompt larger than the budget must not deadlock behind it."""
        policy = CoalescingAdmission(group_size=4, hold_budget_s=0.0,
                                     max_prefill_tokens_per_release=100)
        policy.offer(index=0, arrival_s=0.0, prompt_tokens=1024, now_s=0.0)
        released, _ = policy.poll(now_s=0.0, running=0, waiting=0)
        self.assertEqual(released, [0])


class Accounting(unittest.TestCase):
    def test_every_offered_request_is_released_exactly_once(self):
        policy = CoalescingAdmission(group_size=4, hold_budget_s=0.05, engine_max_seqs=32)
        offered, released = [], []
        clock = 0.0
        for i in range(20):
            clock = 0.01 * i
            policy.offer(index=i, arrival_s=clock, prompt_tokens=128, now_s=clock)
            offered.append(i)
            out, _ = policy.poll(now_s=clock, running=0, waiting=0)
            released.extend(out)
        while policy.pending():
            clock += 0.05
            released.extend(policy.poll(now_s=clock, running=0, waiting=0)[0])
        self.assertEqual(released, offered)
        self.assertEqual(len(released), len(set(released)))

    def test_every_poll_is_recorded_for_the_cost_denominator(self):
        policy = CoalescingAdmission(group_size=4, hold_budget_s=10.0)
        for _ in range(5):
            policy.poll(now_s=0.0, running=0, waiting=0)
        self.assertEqual(len(policy.decisions), 5)
        self.assertEqual([d["decision_index"] for d in policy.decisions], list(range(5)))

    def test_rejects_invalid_construction_and_state(self):
        for kwargs in (dict(group_size=0, hold_budget_s=0.1),
                       dict(group_size=4, hold_budget_s=-1.0),
                       dict(group_size=4, hold_budget_s=0.1, capture_sizes=(4, 2)),
                       dict(group_size=4, hold_budget_s=0.1, engine_max_seqs=0),
                       dict(group_size=4, hold_budget_s=0.1, max_prefill_tokens_per_release=0)):
            with self.assertRaises(ValueError):
                CoalescingAdmission(**kwargs)
        policy = CoalescingAdmission(group_size=4, hold_budget_s=0.1)
        with self.assertRaises(ValueError):
            policy.poll(now_s=0.0, running=-1, waiting=0)


if __name__ == "__main__":
    unittest.main()
