"""HORIZON's page arithmetic; no vLLM import or runtime mutation.

All extents are at one post-allocation boundary. Host LOAD hits are NOT free
GPU capacity. Native reservations and history deficits must be unioned, not
added twice. The runtime adapter remains responsible for KNOWN qualification.
"""
from dataclasses import dataclass


def ceildiv(n, d):
    return (n + d - 1) // d


@dataclass(frozen=True)
class HorizonConfig:
    H_a: int = 64
    H_r: int = 32
    lease_victim: bool = True
    stall_order: bool = True
    estimate_Hr: bool = False

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.H_a, self.H_r)):
            raise ValueError('Horizons must be nonnegative integer schedule steps')
        if any(type(x) is not bool for x in
               (self.lease_victim, self.stall_order, self.estimate_Hr)):
            raise ValueError('Feature switches must be bool')


def fluid_growth(n, horizon, block_size=16):
    """Requested fluid estimate; not a phase-independent page upper bound."""
    if (any(type(x) is not int or x < 0 for x in (n, horizon))
            or type(block_size) is not int or block_size <= 0):
        raise ValueError('Invalid counts')
    return ceildiv(n * horizon, block_size)


def phase_growth(residues, horizon):
    """Exact growth for a fixed cohort with ceil(history/b) held pages.

    A b-entry residue histogram supports O(b) = O(1) queries for fixed b=16.
    Maintaining it belongs in native membership/progress hooks, not rescans.
    No future completion is credited; at most one position per request/step.
    """
    if (not residues or any(type(x) is not int or x < 0 for x in residues)
            or type(horizon) is not int or horizon < 0):
        raise ValueError('Invalid residue histogram or horizon')
    b = len(residues)
    return sum(count * (ceildiv(r + horizon, b) - ceildiv(r, b))
               for r, count in enumerate(residues))


def known_capacity(snapshot, horizon, *, target_horizon=None):
    """Reference oracle for a qualified B KNOWN snapshot (O(requests)).

    This checks arithmetic against B's pure estimate, not runtime eligibility.
    Reserve complete remaining histories of running/inflight owners, then
    growth of running/inflight owners and the prospective target. Inflight
    LOADs may enter decoding within the horizon. Clip at model length.
    The target's reservation includes loaded destination pages, even if the
    transfer avoids recomputation. This function is not the O(1) online hook.
    """
    def unknown(reason):
        return dict(status='UNKNOWN', reason=reason, capacity_fit=None)
    if snapshot.get('status') != 'KNOWN':
        return unknown('UNQUALIFIED_SNAPSHOT')
    target_horizon = horizon if target_horizon is None else target_horizon
    if any(type(x) is not int or x < 0 for x in (horizon, target_horizon)):
        raise ValueError('Invalid horizon')
    try:
        b, limit = snapshot['block_size'], snapshot['max_model_len']
        free, native = snapshot['free_gpu_blocks'], snapshot['native_reserved_blocks']
        if any(type(x) is not int or x < 0 for x in (b, limit, free, native)) or not b or not limit:
            return unknown('INVALID_COUNTS')
        columns = snapshot['columns']
        if len(columns) != len(set(columns)):
            return unknown('DUPLICATE_COLUMNS')
        def row(values):
            if len(values) != len(columns):
                raise ValueError('row width')
            return dict(zip(columns, values))
        rows = [row(x) for x in snapshot['requests']]
        target = row(snapshot['target'])
        owners = {x['request']: x for x in rows}
        running, inflight = set(snapshot['running_ids']), set(snapshot['inflight_ids'])
        if (len(owners) != len(rows) or set(owners) != running | inflight
                or len(running) != len(snapshot['running_ids'])
                or len(inflight) != len(snapshot['inflight_ids'])
                or target['request'] in owners):
            return unknown('REQUEST_SET_MISMATCH')
        for r in [*rows, target]:
            values = [r[k] for k in ('history_tokens','held_gpu_blocks',
                       'computed_tokens','num_in_flight_tokens','scheduled_tokens_this_step')]
            if any(type(v) is not int or v < 0 for v in values):
                return unknown('INVALID_REQUEST_COUNTS')
            n, held, c, i, q = values
            if n > limit or i > c or c + q > n or c + q > held * b:
                return unknown('UNSUPPORTED_TOKEN_BOUNDARY')
            if r['request'] not in running and q:
                return unknown('NONRUNNING_SCHEDULED_TOKENS')
            if r['registry_identity'] is not True or r['plain_unshared_blocks'] is not True:
                return unknown('UNQUALIFIED_OWNER')
        if any(target[k] for k in ('held_gpu_blocks','computed_tokens',
                                  'num_in_flight_tokens','scheduled_tokens_this_step')):
            return unknown('TARGET_NOT_CLEAN')
        def deficit(r, h=0):
            return max(0, ceildiv(min(limit, r['history_tokens'] + h), b)
                       - r['held_gpu_blocks'])
        reserved = sum(deficit(owners[rid]) for rid in inflight)
        if reserved != native:
            return unknown('NATIVE_RESERVATION_MISMATCH')
        growth = sum(max(0, deficit(owners[rid], horizon)
                         - (deficit(owners[rid]) if rid in inflight else 0))
                     for rid in owners)
        target_pages = deficit(target, target_horizon)
    except (KeyError, TypeError, ValueError):
        return unknown('MISSING_OR_INVALID_SNAPSHOT')
    required = target_pages + reserved + growth
    return dict(status='KNOWN', capacity_fit=free >= required,
                target_total_blocks=target_pages,
                inflight_remaining_blocks=reserved, owner_increment_blocks=growth,
                required_free_blocks=required, margin_blocks=free-required)


def victim_index(request_ids, lease_expiry, step):
    """Return index/skips/fallback; caller must do native scheduled rollback.

    The requested reverse search is O(n) worst-case, unlike capacity queries.
    Leases expire at step == expiry (half-open lifetime).
    """
    if not request_ids:
        raise ValueError('Empty running set')
    skipped = 0
    for i in range(len(request_ids)-1, -1, -1):
        if lease_expiry.get(request_ids[i], step) <= step:
            return i, skipped, False
        skipped += 1
    return len(request_ids)-1, skipped, True
