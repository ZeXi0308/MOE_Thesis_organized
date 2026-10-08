"""Pure, conditional capacity accounting; no runtime hook or allocation.

Sources: pinned Scheduler._request_remaining_blocks/_inflight_prefill_reserved_blocks,
Scheduler.schedule/_update_after_schedule, KVCacheManager.allocate_slots and
FullAttentionManager.get_num_blocks_to_allocate. Caller must qualify one plain
FullAttention group, no APC/shared/null blocks, lookahead/speculation/encoder,
watermark or per-request admission cap, and provide one consistent boundary
snapshot. `qualified=True` asserts that external qualification, not a check here.

H is physically held blocks; N is current num_tokens. C already includes earlier
scheduled in-flight tokens I; current schedule's Q has allocated blocks but has
not yet advanced C. Never add I to C or add Q's blocks to H. Growth arguments are
explicit extra KV token positions beyond N, NOT scheduler rounds or a prediction
of outputs, EOS, completion, future capacity, or sustained execution. No freed
blocks are credited before they appear in free_gpu_blocks. Source-backed current
history accounting and assumed growth are reported separately.
"""

SOURCE_SHA256 = {
    'scheduler.py': '2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941',
    'kv_cache_manager.py': '3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf',
    'kv_cache_coordinator.py': '4c8fbb341f0bd3714eff54ce633f534c1475220decb17c0caed40cbd02b352a2',
    'single_type_kv_cache_manager.py': 'bcb27e38895332bf6a4c55608f2917eb9fd941ec5a629c14b88358f00aadeba8',
    'offloading_scheduler.py': '89ac26a80fbc29b9bcaa5a0daba88fb6309247f9bb053e48d2d61efa7d9d66f1',
}


def estimate(*, target, requests, running_ids, inflight_ids,
             native_reserved_blocks, free_gpu_blocks, block_size, max_model_len,
             growth_tokens, target_growth_tokens, qualified):
    """Return KNOWN conditional arithmetic, or UNKNOWN with no fit assertion.

    Each request row has request, history_tokens, held_gpu_blocks,
    computed_tokens, num_in_flight_tokens, scheduled_tokens_this_step. `requests`
    contains exactly the union of running_ids and inflight_ids, one row per RID;
    target is separate, clean and not already in either set. Use actual native
    reserved_blocks, not a second recomputation at a different boundary.
    """
    def unknown(reason):
        return dict(status='UNKNOWN', reason=reason, capacity_fit=None)

    if qualified is not True:
        return unknown('UNQUALIFIED_LAYOUT')
    counts = (native_reserved_blocks, free_gpu_blocks, block_size, max_model_len,
              growth_tokens, target_growth_tokens)
    if any(type(n) is not int or n < 0 for n in counts) or not block_size or not max_model_len:
        return unknown('INVALID_COUNTS')
    try:
        running, inflight = set(running_ids), set(inflight_ids)
        if len(running) != len(running_ids) or len(inflight) != len(inflight_ids):
            return unknown('DUPLICATE_MEMBERSHIP')
        rows = {row['request']: row for row in requests}
        if len(rows) != len(requests) or set(rows) != running | inflight:
            return unknown('REQUEST_SET_MISMATCH')
        if target['request'] in rows:
            return unknown('TARGET_ALREADY_RUNNING_OR_INFLIGHT')
        for row in [target, *rows.values()]:
            rid = row['request']
            n, h, c, i, q = (row[k] for k in (
                'history_tokens', 'held_gpu_blocks', 'computed_tokens',
                'num_in_flight_tokens', 'scheduled_tokens_this_step'))
            if not isinstance(rid, str) or not rid or any(type(v) is not int or v < 0 for v in (n,h,c,i,q)):
                return unknown('INVALID_REQUEST_COUNTS')
            if n > max_model_len or i > c or c + q > n or c + q > h * block_size:
                return unknown('UNSUPPORTED_COMPUTED_OR_HELD_BOUNDARY')
            if rid not in running and q:
                return unknown('NONRUNNING_SCHEDULED_TOKENS')
        if any(target[k] for k in ('held_gpu_blocks', 'computed_tokens', 'num_in_flight_tokens', 'scheduled_tokens_this_step')):
            return unknown('TARGET_NOT_CLEAN_RECOVERY')
    except (KeyError, TypeError):
        return unknown('MISSING_OR_INVALID_SNAPSHOT')

    def deficit(row, growth=0):
        extent = min(row['history_tokens'] + growth, max_model_len)
        return max((extent + block_size - 1) // block_size - row['held_gpu_blocks'], 0)

    remaining = {rid: deficit(rows[rid]) for rid in sorted(inflight)}
    reserved = sum(remaining.values())
    if reserved != native_reserved_blocks:
        return unknown('NATIVE_RESERVATION_MISMATCH')
    per_request = []
    running_increment = 0
    for rid in sorted(rows):
        r = remaining.get(rid, 0)
        d = deficit(rows[rid], growth_tokens) if rid in running else 0
        g = max(d - r, 0) if rid in running else 0
        running_increment += g
        per_request.append(dict(request=rid, is_running=rid in running, is_inflight=rid in inflight,
            inflight_remaining_blocks=r, running_interval_deficit_blocks=d,
            reservation_overlap_blocks=min(r, d), running_increment_blocks=g))
    current = deficit(target)
    target_total = deficit(target, target_growth_tokens)
    total = target_total + reserved + running_increment
    return dict(status='KNOWN', reason=None, capacity_fit=free_gpu_blocks >= total,
        target_current_history_blocks=current, target_growth_increment_blocks=target_total-current,
        target_total_blocks=target_total, inflight_remaining_blocks=reserved,
        running_increment_blocks=running_increment, required_free_blocks=total,
        free_gpu_blocks=free_gpu_blocks, margin_blocks=free_gpu_blocks-total,
        growth_tokens=growth_tokens, target_growth_tokens=target_growth_tokens,
        per_request=per_request,
        semantics='Conditional capacity only; no execution/EOS/latency or future-free guarantee. Zero growth is ordinary joint accounting.')
