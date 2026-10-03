#!/usr/bin/env python3
"""Regression tests for the expert-union collection discipline.

The collector records router choices via a forward hook, so a step that mixes
a chunked prefill with decode would silently contribute up to
`max_num_batched_tokens` routes to what is supposed to be a decode-width
union. That would make U a function of chunk size instead of decode batch
width and invalidate the entire measurement.

These tests exercise that guard without a GPU by driving the same hook logic
against synthetic logits. They check the *discipline*, not the model.
"""

import unittest

from expert_union_tracker import ExpertUnionTracker


class FakeLogits:
    """Minimal stand-in for a 2-D logits tensor as seen by the hook."""

    def __init__(self, rows):
        self._rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def dim(self):
        return 2

    def float(self):
        return self

    def tolist(self):
        return self._rows


def topk_indices(rows, k):
    out = []
    for row in rows:
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))
        out.append(order[:k])
    return out


def make_hook(tracker, state, layer, n_experts, top_k):
    """Mirror of the hook body in run_expert_union.py."""
    def hook(logits):
        if not state["collect"]:
            return "not_collecting"
        if logits.dim() != 2 or logits.shape[-1] != n_experts:
            return "wrong_shape"
        if logits.shape[0] != state["expected_tokens"]:
            state["token_count_mismatch"] = state.get("token_count_mismatch", 0) + 1
            return "token_count_mismatch"
        chosen = topk_indices(logits.tolist(), top_k)
        tracker.record_step(layer=layer,
                            selected_per_token=[tuple(sorted(r)) for r in chosen])
        return "recorded"
    return hook


def logits_for(selections, n_experts):
    """Build logits whose top-k equals each requested selection."""
    rows = []
    for sel in selections:
        row = [0.0] * n_experts
        for rank, e in enumerate(sel):
            row[e] = 100.0 - rank
        rows.append(row)
    return FakeLogits(rows)


class TestCollectionDiscipline(unittest.TestCase):
    def setUp(self):
        self.E, self.K = 64, 8
        self.tracker = ExpertUnionTracker(experts_total=self.E, experts_per_token=self.K)
        self.state = dict(collect=False, expected_tokens=-1)
        self.hook = make_hook(self.tracker, self.state, 0, self.E, self.K)

    def test_pure_decode_step_is_recorded(self):
        self.state.update(collect=True, expected_tokens=3)
        sel = [tuple(range(self.K)), tuple(range(8, 16)), tuple(range(16, 24))]
        self.assertEqual(self.hook(logits_for(sel, self.E)), "recorded")
        self.assertEqual(len(self.tracker.steps[0]), 1)
        # Union of three disjoint 8-expert selections is 24 of 64.
        self.assertAlmostEqual(self.tracker.per_step_union_fractions(0)[0], 24 / 64)

    def test_mixed_step_with_prefill_chunk_is_rejected(self):
        """The decisive guard: a 1024-token chunk must not enter a width-3 union."""
        self.state.update(collect=True, expected_tokens=3)
        chunk = [tuple(range((i * 7) % 56, (i * 7) % 56 + self.K)) for i in range(1024)]
        self.assertEqual(self.hook(logits_for(chunk, self.E)), "token_count_mismatch")
        self.assertEqual(self.tracker.steps.get(0, []), [])
        self.assertEqual(self.state["token_count_mismatch"], 1)

    def test_collection_off_records_nothing(self):
        self.state.update(collect=False, expected_tokens=-1)
        sel = [tuple(range(self.K))]
        self.assertEqual(self.hook(logits_for(sel, self.E)), "not_collecting")
        self.assertEqual(self.tracker.steps.get(0, []), [])

    def test_expected_tokens_sentinel_blocks_all_recording(self):
        """When a step is not pure, expected_tokens is -1 and nothing can match."""
        self.state.update(collect=True, expected_tokens=-1)
        for width in (1, 3, 16, 1024):
            sel = [tuple(range(self.K))] * width
            self.assertEqual(self.hook(logits_for(sel, self.E)), "token_count_mismatch")
        self.assertEqual(self.tracker.steps.get(0, []), [])

    def test_wrong_expert_dimension_is_ignored_not_counted(self):
        """A non-router linear on the same module must not be mistaken for logits."""
        self.state.update(collect=True, expected_tokens=2)
        rows = [[0.0] * 32, [0.0] * 32]           # 32 != n_experts
        self.assertEqual(self.hook(FakeLogits(rows)), "wrong_shape")
        self.assertEqual(self.tracker.steps.get(0, []), [])
        self.assertEqual(self.state.get("token_count_mismatch", 0), 0)

    def test_union_grows_with_width_not_with_token_count(self):
        """Two widths, same routing rule: union must track width."""
        for width in (1, 4, 16):
            tracker = ExpertUnionTracker(experts_total=self.E, experts_per_token=self.K)
            state = dict(collect=True, expected_tokens=width)
            hook = make_hook(tracker, state, 0, self.E, self.K)
            sel = [tuple(range((i * self.K) % self.E, (i * self.K) % self.E + self.K))
                   for i in range(width)]
            sel = [tuple(e % self.E for e in s) for s in sel]
            self.assertEqual(hook(logits_for(sel, self.E)), "recorded")
            frac = tracker.per_step_union_fractions(0)[0]
            self.assertLessEqual(frac, 1.0)
            self.assertGreaterEqual(frac, self.K / self.E)


class TestPurityPredicate(unittest.TestCase):
    """The driver's own decision of whether a step is pure."""

    @staticmethod
    def is_pure(n_decoding, n_prefilling, n_waiting):
        return n_decoding > 0 and not n_prefilling and not n_waiting

    def test_pure_only_when_decode_alone(self):
        self.assertTrue(self.is_pure(16, 0, 0))
        self.assertFalse(self.is_pure(16, 1, 0), "prefill present")
        self.assertFalse(self.is_pure(16, 0, 1), "queue could admit this step")
        self.assertFalse(self.is_pure(0, 0, 0), "nothing decoding")
        self.assertFalse(self.is_pure(0, 4, 0), "prefill only")


if __name__ == "__main__":
    unittest.main()


class TestVerdictBoundary(unittest.TestCase):
    """The verdict must not call a 4-step reuse cycle 'candidate headroom'.

    The original rule only checked saturation <= 2 steps, so a layer that
    re-read its evicted experts every 4 steps fell through to CANDIDATE and
    would have overstated the result. CANDIDATE now requires at least one
    layer that never saturates within any tracked horizon.
    """

    E, K = 64, 8

    def _tracker(self, selections_per_step, horizons=(1, 2, 4, 8, 16, 32)):
        t = ExpertUnionTracker(experts_total=self.E, experts_per_token=self.K,
                               horizons=horizons)
        for rows in selections_per_step:
            t.record_step(layer=0, selected_per_token=rows)
        return t

    def test_no_headroom_when_union_covers_almost_everything(self):
        """Width large enough that each step needs nearly all experts."""
        rows = [tuple(range(i, i + self.K)) for i in range(0, self.E, self.K)]
        rows = [tuple(e % self.E for e in r) for r in rows]        # 8 rows x 8 = 64
        t = self._tracker([rows] * 20)
        v = t.verdict()
        self.assertEqual(v["verdict"], "NO_RESIDENCY_HEADROOM")
        self.assertLess(v["max_layer_median_idle_fraction"], 0.10)

    def test_recurring_reuse_is_not_candidate(self):
        """Idle now, but the whole expert set is re-touched within the horizons.

        Eight disjoint groups of 8 experts tile all 64. Each step uses two
        groups (16 experts, idle 0.75), and the rotation returns to any given
        group within 4 steps, so the union over a short window is everything
        and no expert stays idle. This must NOT read as candidate headroom.
        """
        groups = [tuple(range(g, g + self.K)) for g in range(0, self.E, self.K)]
        self.assertEqual(len({e for g in groups for e in g}), self.E,
                         "fixture must tile every expert, else idleness is real")
        n = len(groups)
        steps = [[groups[i % n], groups[(i + 1) % n]] for i in range(40)]
        t = self._tracker(steps)
        v = t.verdict()
        self.assertGreaterEqual(v["max_layer_median_idle_fraction"], 0.10,
                                "this fixture is meant to have instantaneous headroom")
        self.assertEqual(v["verdict"], "HEADROOM_BUT_IMMEDIATE_REUSE")
        self.assertEqual(v["n_layers_never_saturating"], 0)
        self.assertIsNotNone(t.saturation_window(0, 0.99))

    def test_stable_idle_subset_is_candidate(self):
        """A fixed subset is never selected, so it stays idle past every horizon."""
        rows = [tuple(range(0, self.K)), tuple(range(8, 16))]
        t = self._tracker([rows] * 40)
        v = t.verdict()
        self.assertEqual(v["verdict"], "CANDIDATE_RESIDENCY_HEADROOM")
        self.assertEqual(v["n_layers_never_saturating"], 1)
        # 48 of 64 experts are never touched.
        self.assertAlmostEqual(v["max_layer_median_idle_fraction"], 48 / 64)

    def test_reuse_horizon_is_explicit_not_implied_by_grid(self):
        v = self._tracker([[tuple(range(self.K))]] * 10).verdict(immediate_reuse_steps=8)
        self.assertEqual(v["immediate_reuse_steps"], 8)
        self.assertEqual(v["max_tracked_horizon"], 32)
