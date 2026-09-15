#!/usr/bin/env python3
"""Choose a waiting request and a replacement victim from current KV state.

The sealed native trajectories contained concentrated recovery waits. Their
122-step preemption spacing and total absence are observations of that path,
not invariants or counterfactual bounds for this policy. Rotation changes
future batches, recomputation, release times and potentially other requests.

This module proposes a pair only. A native adapter must actually preempt,
prioritize recovery, preserve worker/block-table semantics, and account for
any held requests needed to protect recovery allocation. CPU decisions do
not establish executable action, tail latency, throughput or quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RotationConfig:
    """All thresholds are explicit so nothing is hidden in code."""

    # Rotate only after a victim has been absent this long. Below the cost of a
    # swap (one recompute) this cannot pay for itself.
    min_absence_steps: int = 30
    # Never swap more often than this, so recompute cost stays bounded.
    min_steps_between_swaps: int = 20
    # Do not rotate when the pool already holds enough blocks to resume the
    # victim on its own: a swap would then add two recomputes for something
    # that was about to happen anyway.
    #
    # This must be compared against the blocks the VICTIM needs, not against
    # zero. The sealed run shows why: at step 931 a preemption freed 245 blocks
    # while the waiting victim needed 237, yet the survivors absorbed them at
    # 1.94 blocks/step and the victim stayed out 94 further steps. A rule of
    # "free_blocks > 0 means do nothing" is silent in exactly the situation the
    # mechanism exists for.
    free_block_slack: int = 0
    # Minimum steps a request must have been resident before it may be evicted
    # again. Without this only the swap cooldown limits churn, which lets a
    # just-evicted request be swapped back almost immediately.
    min_residency_steps: int = 30
    # Protect requests that are nearly done: evicting them discards the most
    # progress and their KV frees soon anyway.
    protect_progress_fraction: float = 0.90
    # Never rotate a request that has already served as a victim this often.
    max_absences_per_request: int = 8
    enabled: bool = True


@dataclass
class RequestView:
    """Exactly the per-request state an online scheduler can see."""

    request_id: str
    num_computed_tokens: int
    num_prompt_tokens: int
    max_total_tokens: int
    # Engine output already produced before this decision; excludes recompute.
    # Kept optional for existing least-progress CPU fixtures.
    num_output_tokens: int | None = None

    @property
    def generated(self) -> int:
        return max(0, self.num_computed_tokens - self.num_prompt_tokens)

    @property
    def progress(self) -> float:
        span = max(1, self.max_total_tokens - self.num_prompt_tokens)
        return min(1.0, self.generated / span)


@dataclass
class RotationDecision:
    step: int
    action: str                     # "rotate" | "noop"
    reason: str
    resume_id: str | None = None
    victim_id: str | None = None
    absence_steps: int = 0
    free_blocks: int = 0


@dataclass
class AbsenceRotation:
    """Pair selector, deterministic given observed state and event history."""

    config: RotationConfig = field(default_factory=RotationConfig)
    absent_since: dict = field(default_factory=dict)   # rid -> step
    absence_count: dict = field(default_factory=dict)  # rid -> times evicted
    resident_since: dict = field(default_factory=dict)  # rid -> step resumed
    last_swap_step: int = -10 ** 9
    decisions: list = field(default_factory=list)
    victim_order: str = "least_progress"
    applied_rotations: int = 0

    def __post_init__(self):
        if self.victim_order not in ("least_progress", "most_output", "first_most_then_least"):
            raise ValueError("unsupported victim order")

    @property
    def effective_victim_order(self):
        if self.victim_order == "first_most_then_least":
            return "most_output" if self.applied_rotations == 0 else "least_progress"
        return self.victim_order

    def note_rotation_applied(self):
        """Called only after a successful native forced exchange, not a proposal."""
        self.applied_rotations += 1

    # ---- bookkeeping driven by what the engine actually did ----------------

    def note_preempted(self, step: int, request_ids) -> None:
        """Record an eviction, whether natural or caused by this controller.

        `absence_count` is the guard behind `max_absences_per_request`, so it
        has to count rotations too. Counting only natural preemptions made the
        cap unreachable and allowed one request to be rotated without limit --
        which would manufacture the very tail the mechanism is meant to remove.
        """
        for rid in request_ids:
            self.absent_since.setdefault(rid, step)
            self.absence_count[rid] = self.absence_count.get(rid, 0) + 1
            self.resident_since.pop(rid, None)

    def note_resumed(self, step: int, request_ids) -> None:
        for rid in request_ids:
            self.absent_since.pop(rid, None)
            self.resident_since[rid] = step

    # ---- the decision ------------------------------------------------------

    def decide(self, step: int, running: list, waiting_ids: list,
               free_blocks: int, blocks_needed: dict | None = None
               ) -> RotationDecision:
        """Pick (resume, victim) or do nothing. Uses only present state.

        `blocks_needed` maps a waiting request id to the blocks its resume
        would require. When supplied, the pool check compares free blocks
        against that requirement instead of against zero.
        """
        cfg = self.config

        def record(action, reason, **kw):
            d = RotationDecision(step=step, action=action, reason=reason,
                                 free_blocks=free_blocks, **kw)
            self.decisions.append(d)
            return d

        if not cfg.enabled:
            return record("noop", "disabled")
        if not waiting_ids:
            return record("noop", "nothing waiting")
        if step - self.last_swap_step < cfg.min_steps_between_swaps:
            return record("noop", "swap cooldown")

        # Longest-waiting victim first. Ties broken by id for determinism.
        candidates = [(step - self.absent_since[r], r) for r in waiting_ids
                      if r in self.absent_since]
        if not candidates:
            return record("noop", "no tracked absentee waiting")
        absence, resume_id = max(candidates, key=lambda t: (t[0], t[1]))

        # The pool is only "comfortable" if it can actually resume THIS victim.
        # Comparing against zero would keep the controller silent in precisely
        # the measured situation it exists for: blocks free, yet absorbed by
        # the survivors before the victim can use them.
        need = (blocks_needed or {}).get(resume_id)
        threshold = cfg.free_block_slack if need is None else need
        if (free_blocks > threshold if need is None else free_blocks >= threshold):
            return record("noop", "pool can already resume the victim",
                          resume_id=resume_id, absence_steps=absence)

        if absence < cfg.min_absence_steps:
            return record("noop", f"absence {absence} below threshold",
                          resume_id=resume_id, absence_steps=absence)

        # Choose the next victim: least progressed, so the swap discards the
        # least computed state and does not delay a nearly-finished request
        # whose KV is about to be released anyway. A request that has only just
        # been resumed is protected, otherwise the pair would thrash.
        eligible = [r for r in running
                    if r.progress < cfg.protect_progress_fraction
                    and self.absence_count.get(r.request_id, 0)
                    < cfg.max_absences_per_request
                    and (step - self.resident_since.get(r.request_id, -10 ** 9)
                         >= cfg.min_residency_steps)]
        if not eligible:
            return record("noop", "no eligible victim",
                          resume_id=resume_id, absence_steps=absence)
        if self.effective_victim_order == "most_output":
            if any(type(r.num_output_tokens) is not int or r.num_output_tokens < 0
                   for r in eligible):
                raise ValueError("most_output requires observed output counts")
            victim = min(eligible, key=lambda r: (-r.num_output_tokens, r.request_id))
        else:
            victim = min(eligible, key=lambda r: (r.progress, r.request_id))
        if victim.request_id == resume_id:
            return record("noop", "victim is the resumer",
                          resume_id=resume_id, absence_steps=absence)

        self.last_swap_step = step
        return record("rotate", "longest absence swapped in",
                      resume_id=resume_id, victim_id=victim.request_id,
                      absence_steps=absence)

    # ---- summary -----------------------------------------------------------

    def summary(self) -> dict:
        rotations = [d for d in self.decisions if d.action == "rotate"]
        absences = [d.absence_steps for d in rotations]
        return dict(
            n_decisions=len(self.decisions), n_rotations=len(rotations),
            n_requests_rotated=len(self.absence_count),
            max_absence_at_rotation=max(absences) if absences else 0,
            absence_count_per_request=dict(sorted(self.absence_count.items())),
            noop_reasons={r: sum(1 for d in self.decisions
                                 if d.action == "noop" and d.reason == r)
                          for r in sorted({d.reason for d in self.decisions
                                           if d.action == "noop"})},
            config=vars(self.config), victim_order=self.victim_order,
            applied_rotations=self.applied_rotations)
