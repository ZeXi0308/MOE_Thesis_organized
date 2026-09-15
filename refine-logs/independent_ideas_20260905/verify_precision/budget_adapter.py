"""Stateful, matched-call selector qualification using existing KV primitives."""
from __future__ import annotations

import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "docs/ideas/B_verify_precision/confidenceguard_v3/experiments"))
from triage_runtime import clone_cache, execute_same_state_step, per_step_kl

STEPS, HIGH_BUDGET = 32, 4
POLICIES = ("fixed_period", "budget_capped_reactive", "budget_matched_random")


class Selector:
    def __init__(self, policy, threshold, seed=20260905):
        if policy not in POLICIES or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("need a supported policy and finite nonnegative frozen threshold")
        self.policy, self.threshold, self.used = policy, threshold, 0
        self.forced = self.threshold_selected = self.threshold_requested = self.blocked = 0
        self.random_steps = set(random.Random(seed).sample(range(STEPS), HIGH_BUDGET))

    def choose(self, step, discrepancy):
        if not 0 <= step < STEPS or not math.isfinite(discrepancy) or discrepancy < 0:
            raise ValueError("invalid step/discrepancy")
        requested = (step % 8 == 0 if self.policy == "fixed_period" else
                     step in self.random_steps if self.policy == "budget_matched_random" else
                     discrepancy >= self.threshold)
        remaining = HIGH_BUDGET - self.used
        forced = self.policy == "budget_capped_reactive" and not requested and STEPS - step == remaining
        high = remaining > 0 and (requested or forced)
        if self.policy == "budget_capped_reactive":
            self.threshold_requested += int(requested)
            self.threshold_selected += int(high and requested)
            self.blocked += int(requested and not high)
        self.forced += int(high and forced)
        self.used += int(high)
        return "high" if high else "low", bool(high and forced), requested


def run_policy(initial_cache, decode_ids, references, *, high_forward, low_forward,
               policy, threshold, seed=20260905, sync=lambda: None, journal=None):
    """Each arm commits its own H/L cache; every step pays one real H and L call.

    References are all-high logits on the fixed teacher-forced token sequence;
    they only measure outcomes and never select actions.
    """
    if tuple(decode_ids.shape) != (1, STEPS) or len(references) != STEPS:
        raise ValueError("qualification requires exactly one 32-step sequence")
    selector = Selector(policy, threshold, seed)
    ledger = dict(physical_high_calls=0, physical_low_calls=0,
                  high_forward_s=0.0, low_forward_s=0.0, served_high_steps=0,
                  served_low_steps=0, discarded_calls=0, cache_clone_events=1)
    sync()
    started = time.perf_counter()
    cache = clone_cache(initial_cache)
    rows = []

    def timed(action, forward, token, state):
        sync()
        before = time.perf_counter()
        output = forward(token, state)
        sync()
        ledger[f"{action}_forward_s"] += time.perf_counter() - before
        ledger[f"physical_{action}_calls"] += 1
        return output

    for step in range(STEPS):
        paired = execute_same_state_step(cache, decode_ids[:, step:step + 1],
            high_forward=lambda token, state: timed("high", high_forward, token, state),
            low_forward=lambda token, state: timed("low", low_forward, token, state), served_action="low")
        action, forced, requested = selector.choose(step, paired.discrepancy)
        selected = paired.high if action == "high" else paired.low
        cache = selected.cache
        ledger[f"served_{action}_steps"] += 1
        ledger["discarded_calls"] += 1
        ledger["cache_clone_events"] += 2
        row = dict(step=step, action=action, forced_budget_fill=forced,
                   requested_high_on_realized_history=requested,
                   same_state_kl=paired.discrepancy,
                   reference_kl=per_step_kl(references[step], selected.logits[:, -1, :]),
                   committed_cache_length=selected.post_length)
        rows.append(row)
        if journal is not None:
            journal(row)
    sync()
    ledger.update(whole_policy_wall_s=time.perf_counter() - started,
                  forced_high_steps=selector.forced, threshold_selected_high_steps=selector.threshold_selected,
                  threshold_requested_high_steps=selector.threshold_requested,
                  budget_blocked_high_steps=selector.blocked,
                  accumulated_kl=sum(row["reference_kl"] for row in rows))
    if (ledger["served_high_steps"], ledger["physical_high_calls"], ledger["physical_low_calls"]) != (4, 32, 32):
        raise ValueError("served or physical call budgets did not match")
    return dict(policy=policy, summary=ledger, steps=rows,
                actual_cost_scope="wall time includes fork, both forwards, KL, decision, commit and journal",
                unforced_counterfactual="UNMEASURED: threshold counts describe this realized KV history only",
                evidence_ceiling="DOUBLE_SHADOW_TEACHER_FORCED_QDQ_SELECTOR_QUALIFICATION")
