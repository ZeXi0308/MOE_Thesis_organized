"""Capture-quantized coalesced admission (CQA): a client-side admission policy.

Motivation is the reconstruction in
`outputs/admission_capacity/20260908_step_cost_surface_r01`:

  * pure-decode step time is a staircase over the engine's CUDA-graph capture
    sizes, flat to within ~4% inside a bucket and jumping ~12-30% across one;
  * every admission event injects a prefill interleave whose cost is ~77% fixed
    at a 128-token prompt, and that cost is charged to every co-resident
    decoding request.

Both consequences are properties of *when requests are handed to the engine*,
not of the engine's concurrency limit. This module decides only that: given the
arrivals that have already become due, and only state observable at or before
the decision instant, which of them to release into the engine now.

The policy is deliberately parameter-poor and has no trained component:

  release the held group when either
    (a) releasing it would land the running width exactly on a capture point, or
    (b) the group has reached `group_size`, or
    (c) the oldest held request has consumed its TTFT hold budget.

Rule (c) is a hard causal deadline: it depends only on elapsed time for a
request that has already arrived, never on how many arrivals are still to come.
The policy never inspects future arrivals, never reorders (FCFS within the held
set is preserved), never preempts, evicts KV, pauses decoding or changes model
semantics. Holding is pure added queueing delay and is fully charged to TTFT.
"""
from __future__ import annotations

import math

# vLLM V1 default piecewise capture sizes at or below max_num_seqs=32. The
# runner must overwrite this from the live engine rather than trust the default.
DEFAULT_CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32)


def ceil_capture(width, capture_sizes):
    """Padded graph width actually executed for a batch of `width` requests."""
    for size in capture_sizes:
        if width <= size:
            return size
    return capture_sizes[-1]


def padding_waste(width, capture_sizes):
    if width <= 0:
        return 0.0
    return 1.0 - width / ceil_capture(width, capture_sizes)


class CoalescingAdmission:
    """Decide which already-arrived requests to release into the engine now.

    `hold_budget_s` is the maximum time a request may be held after its arrival.
    It must be set from the TTFT SLO minus the measured non-hold TTFT of the
    comparison arm, before the run, and never tuned on the outcome.
    """

    def __init__(self, *, group_size, hold_budget_s, capture_sizes=DEFAULT_CAPTURE_SIZES,
                 snap_to_capture_point=True, engine_max_seqs=None,
                 max_prefill_tokens_per_release=None):
        sizes = tuple(capture_sizes)
        if not sizes or sorted(set(sizes)) != list(sizes) or any(
                type(s) is not int or s < 1 for s in sizes):
            raise ValueError("capture sizes must be strictly increasing positive integers")
        if type(group_size) is not int or group_size < 1:
            raise ValueError("group size must be a positive integer")
        if not isinstance(hold_budget_s, (int, float)) or not math.isfinite(hold_budget_s) or hold_budget_s < 0:
            raise ValueError("hold budget must be finite and nonnegative")
        if engine_max_seqs is not None and (type(engine_max_seqs) is not int or engine_max_seqs < 1):
            raise ValueError("engine max seqs must be a positive integer when given")
        if max_prefill_tokens_per_release is not None and (
                type(max_prefill_tokens_per_release) is not int or max_prefill_tokens_per_release < 1):
            raise ValueError("prefill token budget must be a positive integer when given")
        self.capture_sizes = sizes
        self.group_size = group_size
        self.hold_budget_s = float(hold_budget_s)
        self.snap = bool(snap_to_capture_point)
        self.engine_max_seqs = engine_max_seqs
        self.token_budget = max_prefill_tokens_per_release
        self.held = []          # FCFS: (arrival_s, index, prompt_tokens)
        self.decisions = []

    def offer(self, *, index, arrival_s, prompt_tokens, now_s):
        """Register a request whose arrival time has already passed."""
        if not math.isfinite(arrival_s) or not math.isfinite(now_s) or now_s + 1e-9 < arrival_s:
            raise ValueError("a request may only be offered at or after its arrival time")
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise ValueError("prompt token count must be a positive integer")
        if self.held and arrival_s + 1e-9 < self.held[-1][0]:
            raise ValueError("arrivals must be offered in nondecreasing arrival order")
        self.held.append((arrival_s, index, prompt_tokens))

    def _release_size(self, running):
        """How many held requests to release, honouring engine and token budgets."""
        size = len(self.held)
        if self.engine_max_seqs is not None:
            size = min(size, max(0, self.engine_max_seqs - running))
        if self.token_budget is not None:
            allowed, total = 0, 0
            for _, _, tokens in self.held[:size]:
                if total + tokens > self.token_budget and allowed:
                    break
                total += tokens
                allowed += 1
            size = allowed
        return size

    def poll(self, *, now_s, running, waiting):
        """Return (released_indices, decision_row). Uses only present state."""
        if not math.isfinite(now_s) or running < 0 or waiting < 0:
            raise ValueError("invalid present scheduler state")
        available = self._release_size(running)
        oldest_wait_s = now_s - self.held[0][0] if self.held else 0.0
        reason = "hold"
        count = 0
        if available:
            projected = running + waiting + available
            lands_on_capture_point = (
                self.snap and projected == ceil_capture(projected, self.capture_sizes))
            if oldest_wait_s >= self.hold_budget_s:
                reason, count = "hold_budget_expired", available
            elif available >= self.group_size:
                reason, count = "group_complete", available
            elif lands_on_capture_point and available > 1:
                reason, count = "lands_on_capture_point", available
        elif self.held:
            reason = "blocked_by_engine_or_token_budget"
        released = [entry[1] for entry in self.held[:count]]
        row = dict(decision_index=len(self.decisions), now_s=now_s, running=running,
                   waiting=waiting, held_before=len(self.held), releasable=available,
                   oldest_hold_s=oldest_wait_s, reason=reason, released=list(released),
                   released_count=count,
                   projected_width=running + waiting + count,
                   projected_padded_width=ceil_capture(running + waiting + count,
                                                       self.capture_sizes)
                   if running + waiting + count > 0 else 0)
        self.decisions.append(row)
        if count:
            self.held = self.held[count:]
        return released, row

    def pending(self):
        return len(self.held)

    def next_deadline_s(self):
        """Earliest time a held request must be released; None when nothing is held."""
        return self.held[0][0] + self.hold_budget_s if self.held else None
