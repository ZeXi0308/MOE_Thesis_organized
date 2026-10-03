"""Bounded, pre-action service-window qualification; not a trajectory simulator.

All timings are supplied action-specific profiles, never the alternative arm's
future trace. A profile includes engine-returned new outputs, not client receipt.
Window length is an opportunity conditional on the request not reaching EOS.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, isfinite


@dataclass(frozen=True)
class Request:
    name: str
    history_tokens: int  # All current input/output history, including pending input.
    computed_tokens: int  # Valid, contiguous GPU prefix; partial recovery survives.
    gpu_blocks: int  # Actually held blocks, including any existing reservation.
    output_age_ms: float
    host_prefix_tokens: int = 0
    host_valid: bool = False  # Same request and numerical/KV epoch, not mere presence.
    waiting_for_remote_kv: bool = False  # Native blocks may be held before usable KV.
    remaining_output_cap: int | None = None  # Declared limit minus returned IDs, never true EOS distance.


@dataclass(frozen=True)
class State:
    target: Request
    peers: tuple[Request, ...]
    block_tokens: int
    free_gpu_blocks: int  # After the one fixed victim rule's actual release.
    host_used_bytes: int
    host_budget_bytes: int
    max_context_tokens: int | None = None  # Installed engine limit, if supplied.


@dataclass(frozen=True)
class Node:
    name: str
    duration_ms: float
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Action:
    name: str
    tokens: int
    first_output_path: tuple[Node, ...]  # Includes queues, load-ready notification/next schedule, output return.
    batch_ms: float  # Subsequent target/peer decode batch, from this action profile.
    co_batch: tuple[str, ...] = ()
    recovery_peer_outputs: tuple[tuple[str, tuple[float, ...]], ...] = ()
    transfer_prefix_tokens: int = 0  # Prefix endpoint, not number of transferred tokens.
    recompute_tokens: int = 0  # Execution coverage: old history plus first-output input.
    restore_scratch_blocks: int = 0
    staging_bytes: int = 0
    prospective_tax_ms: float | None = None  # Remaining EXPOSED marginal cost only.


def critical_span(nodes):
    """max along declared dependencies; shared-resource serialization is an edge.

    Input must already encode the backend's actual dependency/resource order.
    No automatic claim of overlap is made by this arithmetic helper.
    """
    ends = {}
    for node in nodes:
        if node.name in ends or any(p not in ends for p in node.after):
            raise ValueError('dependency order, duplicate, or cycle')
        if not isfinite(node.duration_ms) or node.duration_ms < 0:
            raise ValueError('invalid phase duration')
        ends[node.name] = max((ends[p] for p in node.after), default=0) + node.duration_ms
    if not ends:
        raise ValueError('first output dependency path is required')
    return max(ends.values())


def growth(request, outputs, block_tokens):
    # The first new output consumes the already pending last history token.
    # Output n's own KV is not materialized until the following token-step.
    if outputs == 0:
        return 0
    return max(0, ceil((request.history_tokens + outputs - 1) / block_tokens)
               - request.gpu_blocks)


def maximum_age(initial_age, outputs, end_ms):
    previous, longest = -initial_age, 0.0
    for when in outputs:
        longest, previous = max(longest, when - previous), when
    return max(longest, end_ms - previous)


def qualify(state, action, *, gap_budget_ms, tax_budget_ms_per_output):
    """A finite action's resource, urgency, and amortization conditions, separately.

    A full-current-history reservation is conservative during recovery. Its
    rejection applies only to this action and reservation scheme. No future EOS,
    arrival, victim search, cancellation, or resource release is inferred.
    An already pending native load defers execution qualification until native
    promotion; neither DMA completion nor a speculative duration clears it here.
    """
    if not 1 <= action.tokens <= 32 or state.block_tokens <= 0:
        raise ValueError('bounded opportunity must contain 1..32 output tokens')
    if min(state.free_gpu_blocks, state.host_used_bytes, state.host_budget_bytes,
           action.staging_bytes, action.restore_scratch_blocks) < 0:
        raise ValueError('negative resource state')
    if not isfinite(action.batch_ms) or action.batch_ms <= 0:
        raise ValueError('invalid measured batch duration')
    if gap_budget_ms <= 0 or tax_budget_ms_per_output <= 0:
        raise ValueError('positive stated diagnostic budgets are required')
    requests = (state.target,) + state.peers
    if state.max_context_tokens is not None and state.max_context_tokens < 1:
        raise ValueError('invalid declared context limit')
    if state.target.computed_tokens >= state.target.history_tokens:
        raise ValueError('target must still have at least the last input pending')
    if len({r.name for r in requests}) != len(requests):
        raise ValueError('duplicate request identity')
    for r in requests:
        if r.remaining_output_cap is not None and r.remaining_output_cap < 1:
            raise ValueError('finished requests do not belong in the active window state')
        if r.waiting_for_remote_kv and r.computed_tokens:
            raise ValueError('pending native computed counter is not a usable GPU prefix')
        if (not 0 <= r.computed_tokens <= r.history_tokens or r.history_tokens < 1
                or min(r.gpu_blocks, r.output_age_ms, r.host_prefix_tokens) < 0
                or r.computed_tokens > r.gpu_blocks * state.block_tokens):
            raise ValueError('invalid KV prefix, ownership, or output age')
    peers = {r.name: r for r in state.peers}
    if len(set(action.co_batch)) != len(action.co_batch) or set(action.co_batch) - peers.keys():
        raise ValueError('unknown or duplicated co-batched peer')
    first = critical_span(action.first_output_path)
    recovery = dict(action.recovery_peer_outputs)
    if len(recovery) != len(action.recovery_peer_outputs) or recovery.keys() - peers.keys():
        raise ValueError('invalid recovery output identity')
    for times in recovery.values():
        if tuple(sorted(set(times))) != times or any(t <= 0 or t > first for t in times):
            raise ValueError('invalid new-output profile')
    for rid, peer in peers.items():
        will_output = bool(recovery.get(rid)) or (rid in action.co_batch and action.tokens > 1)
        if will_output and (peer.waiting_for_remote_kv or
                            peer.computed_tokens != peer.history_tokens - 1):
            raise ValueError('this model only serves resident peers with pending=1')
    readiness_ok = not state.target.waiting_for_remote_kv
    r, prefix = state.target, action.transfer_prefix_tokens
    if prefix < 0 or prefix >= r.history_tokens:
        raise ValueError('transfer must leave the last input for first-output compute')
    source_ok = not prefix or (r.host_valid and r.host_prefix_tokens >= prefix)
    remaining = r.history_tokens - max(r.computed_tokens, prefix if source_ok else 0)
    source_ok &= action.recompute_tokens >= remaining
    if action.prospective_tax_ms is not None and (
            not isfinite(action.prospective_tax_ms) or not 0 <= action.prospective_tax_ms <= first):
        raise ValueError('tax must be remaining exposed marginal time, not summed work')
    end = first + (action.tokens - 1) * action.batch_ms
    target_times = [first + k * action.batch_ms for k in range(action.tokens)]
    peer_times = {rid: list(recovery.get(rid, ())) + (
        [first + k * action.batch_ms for k in range(1, action.tokens)]
        if rid in action.co_batch else []) for rid in peers}
    all_times = {r.name: target_times, **peer_times}
    for request in requests:
        count = len(all_times[request.name])
        if request.remaining_output_cap is not None and count > request.remaining_output_cap:
            raise ValueError('output profile exceeds declared remaining output cap')
        if (count and state.max_context_tokens is not None and
                request.history_tokens + count > state.max_context_tokens):
            raise ValueError('output profile exceeds declared context limit')
    recovery_extra = growth(r, 1, state.block_tokens) + action.restore_scratch_blocks
    recovery_extra += sum(growth(peers[rid], len(recovery.get(rid, ())), state.block_tokens)
                          for rid in peers)
    service_extra = growth(r, action.tokens, state.block_tokens)
    service_extra += sum(growth(peers[rid], len(times), state.block_tokens)
                         for rid, times in peer_times.items())
    peak_extra = max(recovery_extra, service_extra)
    # A profile reaching a declared output cap completes at that return. Its
    # later absence is not starvation. KV is still conservatively retained for
    # peak accounting: this helper never borrows a future completion's release.
    ages = {}
    for request in requests:
        times = all_times[request.name]
        complete = (request.remaining_output_cap == len(times) or
                    bool(times) and state.max_context_tokens == request.history_tokens + len(times))
        ages[request.name] = maximum_age(request.output_age_ms, times,
                                        times[-1] if complete else end)
    resource_ok = (peak_extra <= state.free_gpu_blocks and
                   state.host_used_bytes + action.staging_bytes <= state.host_budget_bytes)
    lower = None if action.prospective_tax_ms is None else max(
        1, ceil(action.prospective_tax_ms / tax_budget_ms_per_output))
    urgency_ok = max(ages.values()) <= gap_budget_ms
    amortized = None if lower is None else action.tokens >= lower
    return dict(action=action.name, source_ok=source_ok, resource_ok=resource_ok,
                readiness_ok=readiness_ok,
                deferred_reason=None if readiness_ok else 'WAIT_FOR_NATIVE_LOAD_PROMOTION',
                urgency_ok=urgency_ok, amortized=amortized, min_opportunity_tokens=lower,
                eligible=readiness_ok and source_ok and resource_ok and urgency_ok and amortized is True,
                first_output_ms=first, window_end_ms=end, peak_extra_blocks=peak_extra,
                free_at_peak=state.free_gpu_blocks - peak_extra, maximum_ages_ms=ages,
                evidence='STRUCTURAL_MODEL_ONLY_CONDITIONAL_ON_NO_EARLY_EOS')


def retention_efficiency(keep_from_now_ms, switch_base_from_now_ms,
                         return_probability, future_marginal_restore_ms):
    """Compare the SAME future service milestone; sunk work is not an input.

    Supplied costs exclude overlap/double counting. Future restore is incremental
    to switch_base; probability/restore intervals need observable calibration.
    This efficiency preference does not waive qualify's peer-delay constraints.
    """
    lo_p, hi_p = return_probability
    lo_r, hi_r = future_marginal_restore_ms
    if not (0 <= lo_p <= hi_p <= 1 and 0 <= lo_r <= hi_r and
            min(keep_from_now_ms, switch_base_from_now_ms) >= 0):
        raise ValueError('invalid prospective cost or uncertainty interval')
    switch = (switch_base_from_now_ms + lo_p * lo_r,
              switch_base_from_now_ms + hi_p * hi_r)
    preference = ('KEEP' if keep_from_now_ms < switch[0] else
                  'SWITCH' if keep_from_now_ms > switch[1] else 'UNRESOLVED')
    return dict(efficiency_preference=preference, switch_cost_ms_interval=switch,
                keep_cost_ms=keep_from_now_ms, peer_delay_constraints_still_required=True)
