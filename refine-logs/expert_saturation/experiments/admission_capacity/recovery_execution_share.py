"""Current-state compute allocation for ONE externally chosen recovery.

Native recompute, one unshared full-attention KV pool, synchronous calls,
resident peers with one pending decode position. No victim/rank/arrival search,
EOS predictor or time predictor. The caller must honor KV escrow and suppress
other admissions until an actual new output or completion ends the obligation.
"""
from dataclasses import dataclass


def blocks(tokens, size):
    return (tokens + size - 1) // size


@dataclass(frozen=True)
class Request:
    request_id: str
    history: int
    computed: int
    allocated: int
    outputs: int
    remaining_cap: int
    remote_pending: bool = False


@dataclass(frozen=True)
class State:
    target: Request
    peers: tuple[Request, ...]  # Existing order supplied by the upper layer.
    free_blocks: int
    block_size: int = 16
    token_budget: int = 1024


@dataclass(frozen=True)
class Protection:
    request_id: str
    initial_outputs: int

    def released(self, *, observed_outputs, completed=False):
        if observed_outputs < self.initial_outputs:
            raise ValueError('returned output count regressed')
        return completed or observed_outputs > self.initial_outputs


def allocate(state, rule='ready_first'):
    """Return a plan, never an execution/output receipt.

    ready_first extracts the existing most recovery sharing rule: reserve the
    full current target history, serve feasible ready peers, restore in residue.
    preserve_calls additionally keeps the target's compute-only lower bound
    ceil(pending / budget) from increasing due to peer token allocation.
    This bound counts successful calls, NOT milliseconds or future outputs.
    """
    if rule not in ('ready_first', 'preserve_calls', 'restore_first'):
        raise ValueError('unknown execution rule')
    t, peers, B, b, F = (state.target, state.peers, state.token_budget,
                          state.block_size, state.free_blocks)
    if min(B, b) <= 0 or F < 0:
        raise ValueError('invalid physical or compute budget')
    rows = (t,) + peers
    if len({r.request_id for r in rows}) != len(rows):
        raise ValueError('duplicate request identity')
    for r in rows:
        if (not 0 <= r.computed <= r.history or r.history < 1
                or min(r.allocated, r.outputs) < 0 or r.remaining_cap < 1
                or (not r.remote_pending and r.computed > r.allocated * b)):
            raise ValueError('invalid active request or usable KV')
    if any(r.remote_pending or r.computed != r.history - 1 or not r.outputs for r in peers):
        raise ValueError('peers must be resident pending-one decode requests')
    R = t.history - t.computed
    escrow = max(0, blocks(t.history, b) - t.allocated)
    result = dict(rule=rule, status='READY', scheduled={}, held={},
                  full_history_escrow_blocks=escrow, allocation_blocks=0,
                  free_after_allocations=F, remaining_escrow_blocks=escrow,
                  target_pending=R, target_positions=0, peer_output_opportunities=0,
                  target_output_eligible=False, target_min_positions=0,
                  compute_only_min_calls=blocks(R, B), unused_token_budget=B)
    if t.remote_pending:
        result['status'] = 'WAIT_NATIVE_READY'
        return result
    if not R:
        result['status'] = 'WAIT_OUTPUT_RETURN'
        return result
    if F < escrow:
        result['status'] = 'UNFUNDED_RETURN_TO_RESOURCE_LAYER'
        return result
    k = blocks(R, B)
    minimum = (R - (k - 1) * B if rule == 'preserve_calls' else
               min(R, B) if rule == 'restore_first' else 1)
    slots = B - minimum
    extra, selected = 0, []
    for r in peers:
        cost = max(0, blocks(r.history, b) - r.allocated)
        if len(selected) >= slots:
            result['held'][r.request_id] = 'COMPUTE_SHARE'
        elif extra + cost > F - escrow:
            result['held'][r.request_id] = 'KV_ESCROW'
        else:
            selected.append(r.request_id)
            extra += cost
    amount = min(R, B - len(selected))
    target_cost = max(0, blocks(t.computed + amount, b) - t.allocated)
    allocation = extra + target_cost
    assert amount >= minimum and F - allocation >= escrow - target_cost
    result.update(scheduled={**{rid: 1 for rid in selected}, t.request_id: amount},
                  target_min_positions=minimum, target_positions=amount,
                  peer_output_opportunities=len(selected), target_output_eligible=amount == R,
                  allocation_blocks=allocation, free_after_allocations=F - allocation,
                  remaining_escrow_blocks=escrow - target_cost,
                  unused_token_budget=B - len(selected) - amount)
    return result


def decode_release_envelope(state):
    """Bounds for resident decode while existing blocks cannot be reclaimed.

    No arrivals, external releases, preemption, block sharing, or early EOS.
    A declared cap is a worst-case stop bound, never a predicted EOS. These are
    output opportunities, not elapsed time. Pool release and safe GPU overwrite
    are backend-specific, separate events; neither is inferred from a cap plan.
    """
    allocate(state)  # Reuse identity, ownership and pending-one validation.
    t = state.target
    if t.remote_pending or not t.outputs or t.computed != t.history - 1:
        raise ValueError('target must also be resident pending-one decode')
    b, F = state.block_size, state.free_blocks
    rows = []
    for r in (t,) + state.peers:
        rows.append(dict(request_id=r.request_id,
            next_decode_blocks=max(0, blocks(r.computed + 1, b) - r.allocated),
            allocated_unused_positions=r.allocated*b-r.computed,
            remaining_declared_cap=r.remaining_cap,
            blocks_to_declared_cap=max(0, blocks(r.computed+r.remaining_cap, b)-r.allocated)))
    least = min(r['blocks_to_declared_cap'] for r in rows)
    impossible = least > F
    return dict(requests=rows, free_blocks=F,
        target_next_decode_feasible=rows[0]['next_decode_blocks'] <= F,
        all_resident_next_decode_blocks=sum(r['next_decode_blocks'] for r in rows),
        all_resident_next_decode_feasible=sum(r['next_decode_blocks'] for r in rows) <= F,
        min_blocks_to_any_declared_cap=least,
        no_early_eos_first_completion_possible=not impossible,
        no_early_eos_total_output_opportunities_before_exhaustion=(
            sum(r['allocated_unused_positions'] for r in rows)+b*F if impossible else None))


def allocate_completion_bridge(state, finisher_id=None):
    """CPU execution component: target progress plus a cap-bounded finisher.

    Reserve only still-missing blocks for target and finisher through the
    finisher's declared cap. Other resident decode uses the remainder. Stop
    on an OBSERVED completion and re-read the pool after native output handling.
    Keep the connector's flush-before-overwrite fence: an allocatable block is
    not proof that its old content has finished saving or that reuse is wait-free.
    This supplies a conditional output/cap path, not a service-time guarantee.
    """
    decode_release_envelope(state)
    t, b, F, B = state.target, state.block_size, state.free_blocks, state.token_budget
    ordered = (t,) + state.peers
    candidates = []
    for r in ordered:
        if finisher_id is not None and r.request_id != finisher_id:
            continue
        if r is not t and r.remaining_cap >= t.remaining_cap:
            continue  # Reaching the target's cap is a shorter sufficient path.
        required = (t,) if r is t else (t, r)
        horizon = r.remaining_cap
        reserve = sum(max(0, blocks(x.computed+horizon, b)-x.allocated) for x in required)
        if reserve <= F and len(required) <= B:
            candidates.append((horizon, reserve, r, required))
    if not candidates:
        return dict(status='NO_CAP_BRIDGE', scheduled={})
    # Stable source order breaks ties; no request identity or observed future.
    horizon, reserve, finisher, required = min(candidates, key=lambda x: (x[0], x[1]))
    ids = {r.request_id for r in required}
    scheduled = {r.request_id: 1 for r in required}
    held = {}
    extra = 0
    for r in state.peers:
        if r.request_id in ids:
            continue
        cost = max(0, blocks(r.computed+1, b)-r.allocated)
        if len(scheduled) >= B:
            held[r.request_id] = 'COMPUTE_BUDGET'
        elif extra+cost > F-reserve:
            held[r.request_id] = 'COMPLETION_BRIDGE_BLOCKS'
        else:
            scheduled[r.request_id] = 1
            extra += cost
    current = sum(max(0, blocks(r.computed+1, b)-r.allocated) for r in required)
    assert F-extra-current >= reserve-current
    return dict(status='READY', finisher_id=finisher.request_id,
        remaining_cap_calls=horizon, reserved_blocks=reserve,
        scheduled=scheduled, held=held, allocation_blocks=extra+current,
        free_after_allocations=F-extra-current, remaining_reserved_blocks=reserve-current)


def decode_service_envelope(state, horizon, required_ids):
    """Exact output-volume bound and forced zero-service count for a fixed slice.

    Every required request gets one output opportunity per call. All requests
    are resident pending-one, no early EOS/arrivals/preemption/sharing, and no
    held block returns before the last output of this slice. A cap may end at
    the final call, never before it. Compute can accommodate the whole cohort.
    This is a conditional allocation bound, not a time or global-policy bound.
    """
    decode_release_envelope(state)
    rows = (state.target,) + state.peers
    required = set(required_ids)
    if (not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1
            or not required or not required <= {r.request_id for r in rows}
            or any(r.remaining_cap < horizon for r in rows)
            or state.token_budget < len(rows)):
        raise ValueError('outside fixed-slice no-early-release/compute assumptions')
    b, F = state.block_size, state.free_blocks
    reserve = sum(max(0, blocks(r.computed+horizon, b)-r.allocated)
                  for r in rows if r.request_id in required)
    if reserve > F:
        return dict(status='UNFUNDED_REQUIRED_SERVICE', required_blocks=reserve)
    remaining = F-reserve
    peers = [r for r in rows if r.request_id not in required]
    base = {r.request_id: min(horizon, b*r.allocated-r.computed) for r in peers}
    # For a request, marginal block gains are b until the final partial gain.
    # Sorting them preserves an attainable prefix (ties use block index/order).
    gains = []
    for order, r in enumerate(peers):
        slack = b*r.allocated-r.computed
        for k in range(max(0, blocks(horizon-slack, b))):
            gain = min(horizon, slack+b*(k+1))-min(horizon, slack+b*k)
            gains.append((gain, k, order, r.request_id))
    chosen = sorted(gains, key=lambda x: (-x[0], x[1], x[2]))[:remaining]
    upper = horizon*len(required)+sum(base.values())+sum(x[0] for x in chosen)
    zero = [rid for rid, opportunities in base.items() if opportunities == 0]
    return dict(status='BOUNDED_FIXED_SLICE', horizon_calls=horizon,
        required_ids=sorted(required), required_blocks=reserve,
        blocks_available_for_other_peers=remaining,
        peer_opportunities_without_new_blocks=base,
        zero_capacity_peer_ids=zero,
        minimum_zero_service_peers=max(0, len(zero)-remaining),
        maximum_total_output_opportunities=upper,
        minimum_total_withheld_opportunities=horizon*len(rows)-upper,
        volume_maximizing_additional_blocks=[x[3] for x in chosen])
