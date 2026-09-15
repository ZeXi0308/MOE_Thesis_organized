"""KV deficit primitives for the long-context admission/capacity regime.

Scope and claim ceiling
-----------------------
These are *model primitives*, not a controller and not a fitted predictor.
Every function is pure arithmetic. Admission bounds are online-visible, while
release times and victim yields can be post-hoc inputs; callers must retain
that distinction. These helpers do not certify a policy-independent bound.

The primitives were written to answer one question on already-sealed native
vLLM ledgers: when a closed request cohort's terminal KV demand exceeds the
block pool by a small margin, (a) is the resulting preemption predictable, and
(b) at which decision point can a legal action still change it.

Conventions
-----------
* ``block_size`` is inferred from data, never assumed.
* A request's block footprint is ``ceil(computed_tokens / block_size)``.  This
  holds only without prefix sharing; the caller must assert that.
* "step" means one ``engine.step`` attempt, matching the sealed ledgers.
* Drain rate is expressed in blocks per step and is only valid while the
  running set is in pure decode at constant width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


def infer_block_size(pairs: Sequence[tuple[int, int]]) -> int:
    """Infer block size from observed ``(computed_tokens, blocks)`` pairs.

    Returns the unique integer ``B`` with ``ceil(tokens / B) == blocks`` for
    every pair.  Raises if no unique candidate exists, so a wrong assumption
    fails loudly rather than silently biasing the accounting.
    """
    if not pairs:
        raise ValueError("need at least one (computed_tokens, blocks) pair")
    for tokens, blocks in pairs:
        if tokens <= 0 or blocks <= 0:
            raise ValueError(f"non-positive observation: {(tokens, blocks)}")
    candidates = [
        b
        for b in range(1, 1025)
        if all(math.ceil(t / b) == k for t, k in pairs)
    ]
    if len(candidates) != 1:
        raise ValueError(f"block size not identified; candidates={candidates}")
    return candidates[0]


def request_blocks(computed_tokens: int, block_size: int) -> int:
    """Logical block footprint of one request (no prefix sharing)."""
    if computed_tokens < 0:
        raise ValueError("computed_tokens must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return math.ceil(computed_tokens / block_size)


@dataclass(frozen=True)
class StaticFeasibility:
    """Admission-time feasibility of a closed cohort, using declared bounds.

    All inputs are known before the first token is generated. ``margin < 0``
    means declared maximum contexts cannot all be resident simultaneously.
    This is not inevitable preemption: completion may release KV before that
    simultaneous state occurs. The final emitted token need not itself be cached.
    """

    n_requests: int
    prompt_tokens: int
    max_output_tokens: int
    block_size: int
    usable_blocks: int

    @property
    def blocks_per_request(self) -> int:
        return request_blocks(self.prompt_tokens + self.max_output_tokens, self.block_size)

    @property
    def terminal_demand_blocks(self) -> int:
        return self.n_requests * self.blocks_per_request

    @property
    def margin_blocks(self) -> int:
        return self.usable_blocks - self.terminal_demand_blocks

    @property
    def infeasible(self) -> bool:
        return self.margin_blocks < 0

    @property
    def deficit_blocks(self) -> int:
        return max(0, -self.margin_blocks)

    @property
    def feasible_concurrency(self) -> int:
        """Largest cohort size that is co-resident-feasible by construction."""
        return self.usable_blocks // self.blocks_per_request

    def as_dict(self) -> dict:
        return {
            "n_requests": self.n_requests,
            "blocks_per_request": self.blocks_per_request,
            "terminal_demand_blocks": self.terminal_demand_blocks,
            "usable_blocks": self.usable_blocks,
            "margin_blocks": self.margin_blocks,
            "deficit_blocks": self.deficit_blocks,
            "terminal_co_residency_exceeds_pool": self.infeasible,
            "scope": "Declared maximum co-residency only; preemption depends on progression and release.",
            "feasible_concurrency": self.feasible_concurrency,
        }


def decode_drain_rate(width: int, block_size: int) -> float:
    """Blocks consumed per step by ``width`` requests each emitting one token.

    This is an expectation over block-boundary crossings, not a per-step exact
    count; individual steps consume an integer number of blocks.
    """
    if width < 0 or block_size <= 0:
        raise ValueError("width must be non-negative and block_size positive")
    return width / block_size


@dataclass(frozen=True)
class DepletionForecast:
    """Linear extrapolation of the free-block pool during pure decode."""

    step0: int
    free_blocks0: int
    drain_rate: float

    def free_at(self, step: int) -> float:
        return self.free_blocks0 - self.drain_rate * (step - self.step0)

    @property
    def exhaustion_step(self) -> int:
        """First step at which the forecast free pool is non-positive."""
        if self.drain_rate <= 0:
            raise ValueError("non-draining pool has no exhaustion step")
        return self.step0 + math.ceil(self.free_blocks0 / self.drain_rate)

    def lead_time_steps(self, action_deadline_step: int) -> int:
        """Steps between the last actionable decision point and exhaustion.

        Positive means the constraint binds *after* the action window closed,
        i.e. a reactive controller observing the pool cannot help.
        """
        return self.exhaustion_step - action_deadline_step


def first_release_step(
    prefill_steps: int,
    max_output_tokens: int,
    first_service_step: int = 0,
) -> int:
    """Zero-based completion step under uninterrupted one-output-per-step service.

    The last prefill step emits the first output. Subsequent outputs take one
    step each. This conditional model assumes a positive prefill step count;
    scheduling holds/preemption invalidate its calendar, and early stopping
    releases earlier. Declared output bounds alone do not bound waiting time.
    """
    if prefill_steps <= 0 or max_output_tokens <= 0 or first_service_step < 0:
        raise ValueError("invalid release-step inputs")
    return first_service_step + prefill_steps + max_output_tokens - 2


def preemption_race(exhaustion_step: int, release_step: int) -> bool:
    """Compare a supplied depletion forecast and release estimate.

    True is a capacity-risk signal under that growth path, not inevitable
    preemption. Zero free blocks alone need not fail: a later step may allocate
    nothing or complete a request. A changed schedule changes the growth path.
    """
    return exhaustion_step < release_step


@dataclass(frozen=True)
class BridgeRequirement:
    """Descriptive reclaim estimate under a supplied constant-growth path.

    Keeping this path until the supplied release consumes bridge_blocks. The
    victim estimate assumes the supplied fixed yield. Changed decode width,
    retained-KV holds or earlier releases invalidate it as a lower bound.
    Actual future release times make this post-hoc reconstruction only.
    """

    exhaustion_step: int
    first_release_step: int
    drain_rate: float
    victim_yield_blocks: int

    @property
    def bridge_steps(self) -> int:
        return max(0, self.first_release_step - self.exhaustion_step)

    @property
    def bridge_blocks(self) -> int:
        return math.ceil(self.bridge_steps * self.drain_rate)

    @property
    def fixed_path_victim_estimate(self) -> int:
        if self.bridge_blocks == 0:
            return 0
        if self.victim_yield_blocks <= 0:
            raise ValueError("victim_yield_blocks must be positive")
        return math.ceil(self.bridge_blocks / self.victim_yield_blocks)

    def as_dict(self) -> dict:
        return {
            "exhaustion_step": self.exhaustion_step,
            "first_release_step": self.first_release_step,
            "bridge_steps": self.bridge_steps,
            "bridge_blocks": self.bridge_blocks,
            "victim_yield_blocks": self.victim_yield_blocks,
            "fixed_path_victim_estimate": self.fixed_path_victim_estimate,
            "scope": "Constant supplied growth and victim yield; not a cross-policy lower bound.",
        }


def predict_victim_stall(
    preempt_time_s: float,
    release_times_s: Sequence[float],
    restore_rank: int,
    recompute_span_s: float,
) -> float:
    """Reconstruct victim stall from supplied release times and restore rank.

    A preempted request cannot be restored until enough blocks are free, and in
    a saturated pool the only source is a natural completion.  ``restore_rank``
    is the victim's 0-based position in the restore order, so it waits for the
    ``restore_rank``-th completion after its preemption.

    Returns ``wait + recompute`` under that rank model. Measured future release
    times and recompute span make this descriptive, not an online forecast.
    """
    if restore_rank < 0:
        raise ValueError("restore_rank must be non-negative")
    later = [t for t in sorted(release_times_s) if t >= preempt_time_s]
    if restore_rank >= len(later):
        raise ValueError("no release event available for that restore rank")
    return (later[restore_rank] - preempt_time_s) + recompute_span_s


@dataclass(frozen=True)
class StaggerFeasibility:
    """Leverage available to admission staggering in a prefill-heavy cohort.

    Restricted model: all prompts are resident before the first completion,
    with synchronous unshifted progress and fluid block accounting. Occupancy
    grows monotonically until that first completion, so the
    binding instant is just before it.  Admitting a request ``d`` steps later
    lowers its footprint at that instant by ``min(d / block_size,
    decode_blocks)`` -- capped because a request cannot hold less than its
    prompt.  Deferring the *whole* cohort equally shifts the binding instant
    too and buys nothing, so at least one request must stay on time.

    Hence a stagger schedule avoids preemption iff

        k * min(d / block_size, decode_blocks) >= deficit_blocks

    for some ``k <= n_requests - 1`` deferred requests and delay ``d``. It excludes
    delaying whole prefills past the first release or changing decode service.
    It is not a bound over general admission schedules. When
    the prompt dominates the footprint, ``decode_blocks`` is small and ``k``
    must be large, which is why staggering degenerates into plain concurrency
    reduction on long-prompt workloads.
    """

    n_requests: int
    prompt_tokens: int
    max_output_tokens: int
    block_size: int
    usable_blocks: int

    @property
    def prefill_blocks(self) -> int:
        return request_blocks(self.prompt_tokens, self.block_size)

    @property
    def full_blocks(self) -> int:
        return request_blocks(self.prompt_tokens + self.max_output_tokens, self.block_size)

    @property
    def decode_blocks(self) -> int:
        """Per-request footprint that is created during decode, not prefill.

        This is the entire dynamic range available to any admission-time
        deferral action.
        """
        return self.full_blocks - self.prefill_blocks

    @property
    def prefill_share(self) -> float:
        return self.prefill_blocks / self.full_blocks

    @property
    def deficit_blocks(self) -> int:
        return StaticFeasibility(
            self.n_requests,
            self.prompt_tokens,
            self.max_output_tokens,
            self.block_size,
            self.usable_blocks,
        ).deficit_blocks

    @property
    def min_deferred_requests(self) -> int:
        """Fewest requests that must be deferred, at maximum useful delay."""
        if self.deficit_blocks == 0:
            return 0
        if self.decode_blocks <= 0:
            return self.n_requests  # no dynamic range at all
        return math.ceil(self.deficit_blocks / self.decode_blocks)

    def min_delay_steps(self, k_deferred: int) -> int | None:
        """Delay each of ``k_deferred`` requests needs, or ``None`` if infeasible."""
        if k_deferred <= 0:
            return None if self.deficit_blocks > 0 else 0
        per_request = math.ceil(self.deficit_blocks / k_deferred)
        if per_request > self.decode_blocks:
            return None
        return per_request * self.block_size

    @property
    def stagger_helps(self) -> bool:
        """True iff some split with at least one on-time request works."""
        if self.deficit_blocks == 0:
            return True
        return self.min_deferred_requests <= self.n_requests - 1

    def as_dict(self) -> dict:
        k = self.min_deferred_requests
        return {
            "prefill_blocks": self.prefill_blocks,
            "full_blocks": self.full_blocks,
            "decode_blocks": self.decode_blocks,
            "prefill_share_of_footprint": self.prefill_share,
            "deficit_blocks": self.deficit_blocks,
            "min_deferred_requests": k,
            "min_delay_steps_at_that_k": self.min_delay_steps(k),
            "stagger_admits_full_cohort": self.stagger_helps,
        }


@dataclass
class DeficitLedger:
    """Mutually exclusive accounting of one deficit-absorption episode.

    Buckets must not overlap: a second of victim wall time is either lost to
    restore wait or spent in recompute, never both.
    """

    victims: list[dict] = field(default_factory=list)

    def add_victim(
        self,
        request_id: str,
        preempt_time_s: float,
        wait_s: float,
        recompute_s: float,
        released_blocks: int,
    ) -> None:
        if wait_s < 0 or recompute_s < 0:
            raise ValueError("stall components must be non-negative")
        if released_blocks <= 0:
            raise ValueError("a victim must release at least one block")
        if any(v["request_id"] == request_id for v in self.victims):
            raise ValueError(f"duplicate victim {request_id}")
        self.victims.append(
            {
                "request_id": request_id,
                "preempt_time_s": preempt_time_s,
                "wait_s": wait_s,
                "recompute_s": recompute_s,
                "stall_s": wait_s + recompute_s,
                "released_blocks": released_blocks,
            }
        )

    @property
    def total_stall_s(self) -> float:
        return sum(v["stall_s"] for v in self.victims)

    @property
    def total_wait_s(self) -> float:
        return sum(v["wait_s"] for v in self.victims)

    @property
    def total_recompute_s(self) -> float:
        return sum(v["recompute_s"] for v in self.victims)

    @property
    def wait_share(self) -> float:
        total = self.total_stall_s
        return self.total_wait_s / total if total > 0 else 0.0

    @property
    def released_blocks(self) -> int:
        return sum(v["released_blocks"] for v in self.victims)

    def max_stall_s(self) -> float:
        return max((v["stall_s"] for v in self.victims), default=0.0)

    def as_dict(self) -> dict:
        return {
            "n_victims": len(self.victims),
            "victims": self.victims,
            "total_stall_s": self.total_stall_s,
            "total_wait_s": self.total_wait_s,
            "total_recompute_s": self.total_recompute_s,
            "wait_share_of_stall": self.wait_share,
            "max_stall_s": self.max_stall_s(),
            "released_blocks": self.released_blocks,
        }


def relative_error(predicted: float, observed: float) -> float:
    if observed == 0:
        raise ValueError("cannot take relative error against zero observation")
    return abs(predicted - observed) / abs(observed)
