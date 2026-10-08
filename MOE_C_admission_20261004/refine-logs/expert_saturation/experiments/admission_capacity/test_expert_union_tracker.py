"""Correctness checks for the batched expert-union tracker.

The tracker's output decides whether an entire mechanism family stays open, so the
risks worth covering are: silent shape errors that would fabricate a union, wrong
union semantics, wrong horizon accumulation, and a verdict that does not follow
from its own numbers.
"""
from __future__ import annotations

import unittest

from expert_union_tracker import ExpertUnionTracker, topk_from_logits


class Validation(unittest.TestCase):
    def tracker(self, **kw):
        kw.setdefault("experts_total", 64)
        kw.setdefault("experts_per_token", 8)
        return ExpertUnionTracker(**kw)

    def test_rejects_wrong_k(self):
        t = self.tracker()
        with self.assertRaises(ValueError):
            t.record_step(layer=0, selected_per_token=[list(range(7))])

    def test_rejects_duplicate_expert_within_a_token(self):
        t = self.tracker()
        with self.assertRaises(ValueError):
            t.record_step(layer=0, selected_per_token=[[0, 0, 1, 2, 3, 4, 5, 6]])

    def test_rejects_out_of_range_expert(self):
        t = self.tracker()
        with self.assertRaises(ValueError):
            t.record_step(layer=0, selected_per_token=[[0, 1, 2, 3, 4, 5, 6, 64]])

    def test_rejects_empty_step(self):
        t = self.tracker()
        with self.assertRaises(ValueError):
            t.record_step(layer=0, selected_per_token=[])

    def test_rejects_bad_construction(self):
        for kw in (dict(experts_total=1, experts_per_token=1),
                   dict(experts_total=64, experts_per_token=0),
                   dict(experts_total=64, experts_per_token=65),
                   dict(experts_total=64, experts_per_token=8, horizons=(0,))):
            with self.assertRaises(ValueError):
                ExpertUnionTracker(**kw)


class UnionSemantics(unittest.TestCase):
    def test_single_token_union_equals_k(self):
        t = ExpertUnionTracker(experts_total=64, experts_per_token=8)
        size = t.record_step(layer=0, selected_per_token=[list(range(8))])
        self.assertEqual(size, 8)
        self.assertAlmostEqual(t.per_step_union_fractions(0)[0], 8 / 64)
        self.assertAlmostEqual(t.per_step_idle_fractions(0)[0], 1 - 8 / 64)

    def test_identical_tokens_do_not_grow_the_union(self):
        """Fully concentrated routing: the best possible case for reclamation."""
        t = ExpertUnionTracker(experts_total=64, experts_per_token=8)
        size = t.record_step(layer=0, selected_per_token=[list(range(8))] * 32)
        self.assertEqual(size, 8)
        self.assertAlmostEqual(t.per_step_idle_fractions(0)[0], 0.875)

    def test_disjoint_tokens_saturate_the_union(self):
        """Worst case: 8 tokens x 8 disjoint experts covers all 64."""
        t = ExpertUnionTracker(experts_total=64, experts_per_token=8)
        rows = [list(range(8 * i, 8 * i + 8)) for i in range(8)]
        size = t.record_step(layer=0, selected_per_token=rows)
        self.assertEqual(size, 64)
        self.assertEqual(t.per_step_idle_fractions(0)[0], 0.0)

    def test_uniform_null_matches_closed_form(self):
        t = ExpertUnionTracker(experts_total=64, experts_per_token=8)
        self.assertAlmostEqual(t.uniform_null_idle_fraction(1), 0.875)
        self.assertAlmostEqual(t.uniform_null_idle_fraction(16), (0.875) ** 16)


class HorizonGrowth(unittest.TestCase):
    def test_horizon_union_accumulates_across_steps(self):
        t = ExpertUnionTracker(experts_total=64, experts_per_token=8, horizons=(1, 2, 4))
        # step 0 uses experts 0-7, step 1 uses 8-15, step 2 uses 16-23
        for start in (0, 8, 16):
            t.record_step(layer=0, selected_per_token=[list(range(start, start + 8))])
        self.assertEqual(t.horizon_union_fractions(0, 1), [8 / 64] * 3)
        self.assertEqual(t.horizon_union_fractions(0, 2), [16 / 64] * 2)
        self.assertEqual(t.horizon_union_fractions(0, 4), [])

    def test_saturation_window_detects_immediate_reuse(self):
        """Disjoint halves alternating: union saturates at window 2 only if it covers all."""
        t = ExpertUnionTracker(experts_total=16, experts_per_token=8, horizons=(1, 2, 4))
        for _ in range(6):
            t.record_step(layer=0, selected_per_token=[list(range(0, 8))])
            t.record_step(layer=0, selected_per_token=[list(range(8, 16))])
        self.assertIsNone(t.saturation_window(0, 0.99) == 1 or None)
        self.assertEqual(t.saturation_window(0, 0.99), 2)

    def test_stable_subset_never_saturates(self):
        t = ExpertUnionTracker(experts_total=64, experts_per_token=8, horizons=(1, 2, 4, 8))
        for _ in range(20):
            t.record_step(layer=0, selected_per_token=[list(range(8))] * 4)
        self.assertIsNone(t.saturation_window(0, 0.99))


class LoadSkew(unittest.TestCase):
    def test_skew_counts_zero_token_experts_in_the_mean(self):
        t = ExpertUnionTracker(experts_total=64, experts_per_token=8)
        t.record_step(layer=0, selected_per_token=[list(range(8))])
        skew = t.load_skew(0)
        # 8 experts hit once, 56 never; mean over all 64 = 8/64 = 0.125
        self.assertAlmostEqual(skew["max_over_mean"], 1 / (8 / 64))
        self.assertEqual(skew["experts_never_selected"], 56)
        self.assertEqual(skew["total_selections"], 8)


class Verdict(unittest.TestCase):
    def test_saturated_routing_gives_no_headroom(self):
        t = ExpertUnionTracker(experts_total=64, experts_per_token=8)
        rows = [list(range(8 * i, 8 * i + 8)) for i in range(8)]
        for _ in range(10):
            t.record_step(layer=0, selected_per_token=rows)
        v = t.verdict()
        self.assertEqual(v["verdict"], "NO_RESIDENCY_HEADROOM")
        self.assertEqual(v["max_layer_median_idle_fraction"], 0.0)

    def test_stable_subset_gives_candidate_headroom(self):
        t = ExpertUnionTracker(experts_total=64, experts_per_token=8, horizons=(1, 2, 4, 8))
        for _ in range(20):
            t.record_step(layer=0, selected_per_token=[list(range(8))] * 8)
        v = t.verdict()
        self.assertEqual(v["verdict"], "CANDIDATE_RESIDENCY_HEADROOM")
        self.assertAlmostEqual(v["max_layer_median_idle_fraction"], 0.875)

    def test_alternating_halves_flagged_as_immediate_reuse(self):
        t = ExpertUnionTracker(experts_total=16, experts_per_token=8, horizons=(1, 2, 4))
        for _ in range(10):
            t.record_step(layer=0, selected_per_token=[list(range(0, 8))])
            t.record_step(layer=0, selected_per_token=[list(range(8, 16))])
        v = t.verdict()
        self.assertEqual(v["verdict"], "HEADROOM_BUT_IMMEDIATE_REUSE")
        self.assertAlmostEqual(v["max_layer_median_idle_fraction"], 0.5)

    def test_multiple_layers_are_summarised_separately(self):
        t = ExpertUnionTracker(experts_total=64, experts_per_token=8)
        t.record_step(layer=0, selected_per_token=[list(range(8))] * 4)
        t.record_step(layer=5, selected_per_token=[list(range(8 * i, 8 * i + 8)) for i in range(8)])
        summary = t.summary()
        self.assertEqual(set(summary), {"0", "5"})
        self.assertAlmostEqual(summary["0"]["idle_fraction_p50"], 0.875)
        self.assertAlmostEqual(summary["5"]["idle_fraction_p50"], 0.0)


class TopK(unittest.TestCase):
    def test_topk_is_deterministic_and_sorted(self):
        logits = [0.1, 0.9, 0.5, 0.9, 0.2]
        self.assertEqual(topk_from_logits(logits, 2), (1, 3))
        self.assertEqual(topk_from_logits(logits, 3), (1, 2, 3))

    def test_topk_rejects_bad_k(self):
        with self.assertRaises(ValueError):
            topk_from_logits([0.1, 0.2], 3)


if __name__ == "__main__":
    unittest.main()
