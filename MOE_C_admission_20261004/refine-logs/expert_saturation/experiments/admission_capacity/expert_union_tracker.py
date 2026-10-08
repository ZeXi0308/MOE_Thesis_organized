"""Measure the per-step, per-layer union of experts selected by a decode batch.

This is the prerequisite named in `20260908_kv_budget_r01/REPORT.md` and never run:

    "若之后检查稀疏专家回收，应先测实际 batched expert union、跨 step 复用与 KV
     压力交集；单 token 未选中专家算术不构成可回收 HBM。"

and the reason my own `20260908_kv_pressure_probe_r01` idle-expert figure had to be
withdrawn (see its ADDENDUM_BATCHED_UNIT_CORRECTION.md).

The decisive quantity is not the per-token top-k ratio but, for each layer L and
each decode step s:

    U(L, s) = |{ expert e : e selected by ANY token in step s at layer L }| / E

Because a layer must hold every expert some token in the batch routes to, `1 - U`
is the only fraction of that layer's expert weights that could conceivably be
absent from HBM during step s. Two derived quantities decide whether any paging
action could exist at all:

  * `idle_fraction = 1 - U`, the instantaneous headroom;
  * `residency_horizon`: how the union grows when accumulated over W consecutive
    steps. If W-step union saturates to E within a few steps, an expert evicted
    now is needed again almost immediately, so transfer cost recurs every step and
    no static residency decision survives.

This module is pure analysis over routing decisions. It computes nothing about
transfer cost and makes no paging claim; a near-zero idle fraction is a negative
result that closes the reclamation family, and a large one only opens a cost
question.

Routing is recomputed from the model's own router logits with its own
softmax/top-k, matching the approach already used elsewhere in this repository.
It is a structural signal: it is NOT measured HBM traffic, NOT a claim about what
the fused backend actually loads, and NOT evidence that an idle expert's bytes are
reclaimable in practice.
"""
from __future__ import annotations

import math
from collections import Counter


class ExpertUnionTracker:
    """Accumulate per-layer expert-selection unions over a decode episode."""

    def __init__(self, *, experts_total, experts_per_token, horizons=(1, 2, 4, 8, 16, 32)):
        if type(experts_total) is not int or experts_total < 2:
            raise ValueError("experts_total must be an integer of at least two")
        if type(experts_per_token) is not int or not 1 <= experts_per_token <= experts_total:
            raise ValueError("experts_per_token must be within [1, experts_total]")
        horizons = tuple(sorted(set(int(h) for h in horizons)))
        if not horizons or horizons[0] < 1:
            raise ValueError("horizons must be positive integers")
        self.E = experts_total
        self.k = experts_per_token
        self.horizons = horizons
        # steps[layer] = list of frozensets, one per recorded step
        self.steps = {}
        self.token_counts = {}
        self.expert_hits = {}

    def record_step(self, *, layer, selected_per_token):
        """Register one decode step at one layer.

        `selected_per_token` is a sequence of sequences: for each token in the
        batch, the expert ids it routed to. Validation is strict because a silent
        shape error here would fabricate the entire result."""
        if type(layer) is not int or layer < 0:
            raise ValueError("layer must be a nonnegative integer")
        rows = [tuple(int(e) for e in row) for row in selected_per_token]
        if not rows:
            raise ValueError("a recorded step must contain at least one token")
        for row in rows:
            if len(row) != self.k:
                raise ValueError(f"expected {self.k} experts per token, got {len(row)}")
            if len(set(row)) != self.k:
                raise ValueError("a token selected the same expert twice")
            if any(not 0 <= e < self.E for e in row):
                raise ValueError("expert id out of range")
        union = frozenset(e for row in rows for e in row)
        self.steps.setdefault(layer, []).append(union)
        self.token_counts.setdefault(layer, []).append(len(rows))
        counter = self.expert_hits.setdefault(layer, Counter())
        for row in rows:
            counter.update(row)
        return len(union)

    # ---- per-step statistics -------------------------------------------------

    def per_step_union_fractions(self, layer):
        return [len(u) / self.E for u in self.steps.get(layer, [])]

    def per_step_idle_fractions(self, layer):
        return [1.0 - f for f in self.per_step_union_fractions(layer)]

    def uniform_null_idle_fraction(self, batch_width):
        """Idle fraction expected if routing were independent and uniform."""
        return (1.0 - self.k / self.E) ** batch_width

    # ---- horizon growth -----------------------------------------------------

    def horizon_union_fractions(self, layer, window):
        """Union accumulated over each sliding window of `window` steps."""
        sequence = self.steps.get(layer, [])
        if window < 1 or len(sequence) < window:
            return []
        out = []
        for start in range(0, len(sequence) - window + 1):
            merged = set()
            for u in sequence[start:start + window]:
                merged |= u
            out.append(len(merged) / self.E)
        return out

    def saturation_window(self, layer, threshold=0.99):
        """Smallest horizon whose median union reaches `threshold` of all experts.

        A small value means an evicted expert is needed again within that many
        steps, so no static residency decision holds."""
        for window in self.horizons:
            values = self.horizon_union_fractions(layer, window)
            if not values:
                continue
            values.sort()
            median = values[len(values) // 2] if len(values) % 2 else \
                0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2])
            if median >= threshold:
                return window
        return None

    # ---- load skew ----------------------------------------------------------

    def load_skew(self, layer):
        """C = max expert token count / mean over ALL experts, zeros included."""
        counter = self.expert_hits.get(layer)
        if not counter:
            return None
        totals = [counter.get(e, 0) for e in range(self.E)]
        mean = sum(totals) / self.E
        if mean <= 0:
            return None
        never = sum(1 for t in totals if t == 0)
        return dict(max_over_mean=max(totals) / mean,
                    experts_never_selected=never,
                    experts_never_selected_fraction=never / self.E,
                    total_selections=sum(totals))

    # ---- summary ------------------------------------------------------------

    def summary(self):
        layers = sorted(self.steps)
        out = {}
        for layer in layers:
            fractions = self.per_step_union_fractions(layer)
            widths = self.token_counts[layer]
            ordered = sorted(fractions)
            n = len(ordered)
            median_width = sorted(widths)[len(widths) // 2]
            out[str(layer)] = dict(
                n_steps=n,
                median_batch_width=median_width,
                union_fraction_min=ordered[0], union_fraction_max=ordered[-1],
                union_fraction_p50=ordered[n // 2],
                idle_fraction_p50=1.0 - ordered[n // 2],
                idle_fraction_max=1.0 - ordered[0],
                uniform_null_idle_at_median_width=self.uniform_null_idle_fraction(median_width),
                saturation_window_99=self.saturation_window(layer, 0.99),
                saturation_window_95=self.saturation_window(layer, 0.95),
                horizon_union_p50={str(w): (lambda v: sorted(v)[len(v) // 2] if v else None)(
                    self.horizon_union_fractions(layer, w)) for w in self.horizons},
                load_skew=self.load_skew(layer))
        return out

    def verdict(self, *, action_space_threshold=0.10, immediate_reuse_steps=4):
        """Mechanical read: is there any instantaneous residency headroom?

        `action_space_threshold` is the minimum median idle fraction that could
        plausibly repay a transfer, frozen before measurement. Below it, no
        reclamation action exists regardless of transfer cost.

        `immediate_reuse_steps` is the horizon within which re-needing an
        expert makes a *static* residency decision meaningless, because the
        transfer cost recurs on that period. A layer that saturates in 4 steps
        re-reads its evicted experts roughly every 4 steps; calling that
        "candidate headroom" would overstate the result, so the boundary is
        explicit rather than implied by the horizon grid.

        A layer whose 99% saturation window is `None` never saturated within
        the tracked horizons. That is the only case where a genuinely stable
        idle subset can exist, so it is required for CANDIDATE.
        """
        summary = self.summary()
        if not summary:
            return dict(verdict="NO_DATA")
        idles = [v["idle_fraction_p50"] for v in summary.values()]
        windows = [v["saturation_window_99"] for v in summary.values()]
        best = max(idles)
        # Saturating within the reuse horizon => transfer cost recurs.
        recurring = [w for w in windows if w is not None and w <= immediate_reuse_steps]
        # Never saturating within any tracked horizon => a stable idle subset.
        stable = [w for w in windows if w is None]
        if best < action_space_threshold:
            verdict = "NO_RESIDENCY_HEADROOM"
        elif not stable:
            verdict = "HEADROOM_BUT_IMMEDIATE_REUSE"
        else:
            verdict = "CANDIDATE_RESIDENCY_HEADROOM"
        return dict(verdict=verdict, action_space_threshold=action_space_threshold,
                    immediate_reuse_steps=immediate_reuse_steps,
                    max_layer_median_idle_fraction=best,
                    min_layer_median_idle_fraction=min(idles),
                    n_layers=len(summary),
                    max_tracked_horizon=self.horizons[-1],
                    n_layers_reusing_within_horizon=len(recurring),
                    n_layers_never_saturating=len(stable),
                    interpretation=(
                        "NO_RESIDENCY_HEADROOM: every step needs nearly all experts; "
                        "HEADROOM_BUT_IMMEDIATE_REUSE: every layer re-needs its experts "
                        "within the tracked horizons, so transfer cost recurs and no "
                        "static residency decision survives; "
                        "CANDIDATE_RESIDENCY_HEADROOM: at least one layer keeps experts "
                        "idle beyond every tracked horizon, cost question remains open"))


def topk_from_logits(logits_row, k):
    """Deterministic top-k by value then by index, matching a router tie-break."""
    if k < 1 or k > len(logits_row):
        raise ValueError("invalid k for this logit row")
    ordered = sorted(range(len(logits_row)), key=lambda i: (-logits_row[i], i))
    return tuple(sorted(ordered[:k]))
