"""Pure online decision for a funded, interruption-limited recovery episode.

This is a local model, not a throughput prediction or a wall-clock SLO proof.
The native adapter owns admission, KV ownership verification and execution.
Only past service observations and current state are accepted here. `overhead_s`
is an observed host service-episode overhead estimate: first-output latency minus
one decode interval. It must not be summed with overlapping device-copy times.

For q new outputs, the estimated episode costs h + q*d, and its full KV endpoint
is prompt + output + q - 1. The feasible frontier intersects physical capacity,
the known output cap, and other requests' interruption headroom. Among feasible
q, choose the smallest meeting h/(h+q*d) <= overhead_fraction. If unattainable,
choose the feasible endpoint and expose that the amortization target was missed.
The established Q1 fallback is retained even if a peer is already overdue.
"""
from dataclasses import asdict, dataclass
from math import ceil, floor, isfinite
from typing import Optional, Tuple


@dataclass(frozen=True)
class LeaseState:
    prompt_tokens: int
    output_tokens: int
    max_output_tokens: int
    held_blocks: int
    # Actual free blocks AFTER the proposed victim's blocks are reclaimed.
    free_blocks: int
    # Native allocations promised but not yet reflected in actual free blocks.
    inflight_reserved_blocks: int
    target_age_s: float
    # Conservatively include every other output-bearing live request; a running
    # peer may become blocked by growth while the target owns the lease.
    peer_ages_s: Tuple[float, ...]
    overhead_s: Optional[float]
    decode_interval_s: Optional[float]
    block_size: int = 16


@dataclass(frozen=True)
class LeaseConfig:
    max_quantum: int = 16
    overhead_fraction: float = 0.5
    interruption_floor_s: float = 1.0


@dataclass(frozen=True)
class LeaseDecision:
    quantum: int
    required_total_blocks: int
    reserved_growth_blocks: int
    desired_quantum: int
    memory_quantum_cap: int
    peer_quantum_cap: int
    peer_age_limit_s: float
    estimated_duration_s: Optional[float]
    estimated_overhead_fraction: Optional[float]
    amortization_target_met: bool
    reason: str

    def to_dict(self):
        return asdict(self)


def _nonnegative_real(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) \
        and isfinite(value) and value >= 0


def required_blocks(prompt_tokens, output_tokens, quantum, block_size=16):
    """KV through the computation producing the final protected output."""
    if quantum < 1:
        raise ValueError("quantum must be positive")
    return (prompt_tokens + output_tokens + quantum - 1 + block_size - 1) // block_size


def choose_lease(state: LeaseState, config: LeaseConfig = LeaseConfig(),
                 *, requested_quantum=None) -> LeaseDecision:
    if requested_quantum is not None and (type(requested_quantum) is not int or requested_quantum<1):
        raise ValueError('requested_quantum must be a positive integer or None')
    ints = (state.prompt_tokens, state.output_tokens, state.max_output_tokens,
            state.held_blocks, state.free_blocks, state.inflight_reserved_blocks)
    if any(type(n) is not int or n < 0 for n in ints):
        raise ValueError("token and physical block counts must be nonnegative integers")
    if state.output_tokens > state.max_output_tokens:
        raise ValueError("output count exceeds known cap")
    if type(state.block_size) is not int or state.block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if type(config.max_quantum) is not int or config.max_quantum <= 0:
        raise ValueError("max_quantum must be a positive integer")
    if not _nonnegative_real(config.interruption_floor_s) \
            or not 0 < config.overhead_fraction < 1:
        raise ValueError("invalid lease configuration")
    if not _nonnegative_real(state.target_age_s) \
            or any(not _nonnegative_real(a) for a in state.peer_ages_s):
        raise ValueError("output ages must be finite and nonnegative")

    age_limit = max(config.interruption_floor_s, state.target_age_s)
    usable = state.free_blocks - state.inflight_reserved_blocks
    memory_cap = max(0, (state.held_blocks + usable) * state.block_size
                     - state.prompt_tokens - state.output_tokens + 1)
    hard_cap = min(config.max_quantum, state.max_output_tokens - state.output_tokens)
    base_cap = min(memory_cap, hard_cap)

    def result(q, desired, peer_cap, reason, h=None, d=None):
        need = required_blocks(state.prompt_tokens, state.output_tokens, q,
                               state.block_size) if q else 0
        duration = h + q * d if h is not None and q else None
        fraction = h / duration if duration is not None else None
        return LeaseDecision(q, need, max(0, need - state.held_blocks), desired,
                             memory_cap, peer_cap, age_limit, duration, fraction,
                             fraction is not None and fraction <= config.overhead_fraction + 1e-12,
                             reason)

    if state.output_tokens == state.max_output_tokens:
        return result(0, 0, 0, "TERMINAL_OUTPUT_CAP")
    if usable < 0 or base_cap < 1:
        return result(0, 1, 0, "Q1_PHYSICALLY_UNFUNDED")
    h, d = state.overhead_s, state.decode_interval_s
    if not _nonnegative_real(h) or not _nonnegative_real(d) or d == 0:
        return result(1, 1, 1, "Q1_UNKNOWN_SERVICE_COST")

    desired = max(1, ceil(h * (1 - config.overhead_fraction)
                          / (config.overhead_fraction * d) - 1e-12))
    if requested_quantum is not None:desired=requested_quantum
    if state.peer_ages_s:
        headroom = age_limit - max(state.peer_ages_s)
        peer_cap = max(0, floor((headroom - h) / d + 1e-12))
    else:
        peer_cap = hard_cap
    feasible_cap = min(base_cap, peer_cap)
    if feasible_cap < 1:
        # The old Q1 progress contract takes precedence over optional extension.
        # This does NOT claim that already-overdue peers satisfy the age bound.
        return result(1, desired, peer_cap, "Q1_PEER_BUDGET_EXHAUSTED", h, d)
    q = min(desired, feasible_cap)
    reason = "AMORTIZED_WITHIN_FRONTIER" if q == desired else "FRONTIER_TRUNCATED"
    return result(q, desired, peer_cap, reason, h, d)


def continuation_reason(*, output_at_start, current_output, quantum,
                        prompt_tokens, held_blocks, free_blocks,
                        inflight_reserved_blocks, peer_ages_s,
                        peer_age_limit_s, next_decode_interval_s, block_size=16):
    """Check optional extension after Q1; returns CONTINUE or a release reason.

    The adapter also checks native status/ownership/transfer/execution state.
    Updating clocks before this check handles new peers and cost underestimates.
    The initial first output keeps the established Q1 lifecycle, including loads.
    """
    emitted = current_output - output_at_start
    if emitted < 0:
        raise ValueError("output count moved backwards")
    if emitted >= quantum:
        return "OUTPUT_GOAL_REACHED"
    if emitted == 0:
        return "Q1_IN_PROGRESS"
    if not _nonnegative_real(next_decode_interval_s) or next_decode_interval_s == 0:
        return "RELEASE_UNKNOWN_DECODE_INTERVAL"
    if any(not _nonnegative_real(a) for a in peer_ages_s):
        return "RELEASE_UNKNOWN_PEER_AGE"
    if peer_ages_s and max(peer_ages_s) + next_decode_interval_s > peer_age_limit_s:
        return "RELEASE_PEER_AGE_FRONTIER"
    required = required_blocks(prompt_tokens, output_at_start, quantum, block_size)
    if free_blocks - inflight_reserved_blocks < max(0, required - held_blocks):
        return "RELEASE_GROWTH_RESERVATION_LOST"
    return "CONTINUE"
