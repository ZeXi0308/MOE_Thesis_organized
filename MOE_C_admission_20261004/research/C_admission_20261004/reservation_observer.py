"""Observe reservation and optionally probe one reservation or fixed-margin event.

Full/chunk plus reservation comparisons are not predictions
of an OOM, a preemption, or service benefit; local work is preemptible.
"""
import math
import time


def install(gate, probe_enabled=False, max_extra_s=.25, fixed_headroom_blocks=None):
    assert not gate.probe_enabled, 'The older pending-recovery probe must remain disabled'
    assert type(probe_enabled) is bool and math.isfinite(max_extra_s) and 0 <= max_extra_s <= .25
    assert fixed_headroom_blocks is None or type(fixed_headroom_blocks) is int and fixed_headroom_blocks >= 0
    signal_kind = 'known_prefill' if fixed_headroom_blocks is None else 'fixed_headroom'
    action_reason = 'reservation' if fixed_headroom_blocks is None else 'headroom'
    direct_flag = 'changed_by_reservation' if fixed_headroom_blocks is None else 'changed_by_headroom'
    fifo_flag = 'reservation_probe_fifo_held' if fixed_headroom_blocks is None else 'headroom_probe_fifo_held'
    original_before, original_report = gate.before_allocate, gate.report
    original_early, original_end = gate.early, gate.end
    scheduler = gate.s
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    stats = dict(observed_new_allocations=0, positive_reservation_evaluations=0,
        full_commit_opportunity_evaluations=0, chunk_commit_opportunity_evaluations=0,
        max_unallocated_blocks=0, first_full_commit_opportunity=None,
        geometry_checks=0, geometry_mismatches=0,
        semantics='Native existing _inflight_prefills only; excludes fully paused requests removed '
            'from that set. Remaining blocks are for the currently known sequence, minus actual '
            'allocated blocks. No future generated length, transfer-time sum, or extra allocation. '
            'A combined-footprint excess is a shadow diagnostic, not an unavoidable capacity failure.')
    probe = dict(probe_enabled=probe_enabled, max_extra_s=max_extra_s,
        signal_kind=signal_kind, fixed_headroom_blocks=fixed_headroom_blocks, first_opportunity=None,
        first_release=None, direct_action_evaluations=0, fifo_action_evaluations=0,
        semantics='Only the first full-commit conflict target can receive extra denial. '
            'Release is the first gate observation of no reservation, enough combined capacity, '
            'or the wall deadline; it is not actual prefill execution. FIFO holds have no native-fit '
            'counterfactual. The older pending-recovery first_opportunity is independent.')
    if fixed_headroom_blocks is not None:
        probe['semantics'] = ('Only the first native-fit/cap/KV-allowed new prefill with '
            'free minus full-required minus watermark blocks below the fixed headroom can be denied. '
            'Release uses margin reaching the fixed headroom or the same wall deadline, with no '
            'minimum hold. Recovery reservation never enters this trigger or release; its observation '
            'remains shadow-only. FIFO holds have no native-fit counterfactual. Release is a host '
            'gate observation, not actual prefill execution. Older pending-recovery events are separate.')
    fifo_ids = set()

    def release(now, reason, req):
        if probe['first_release'] is None:
            probe['first_release'] = dict(t=now-gate.origin, perf_s=now, reason=reason,
                observed_request_id=req.request_id)
        if gate.barrier == action_reason:
            gate.barrier = gate.barrier_deadline = None

    def active():
        return probe_enabled and probe['first_opportunity'] is not None and probe['first_release'] is None

    def before(req, **kwargs):
        check = original_before(req, **kwargs)
        row = check['row']
        if row is None:
            return check
        row.update(changed_by_reservation=False, reservation_probe_fifo_held=False,
            changed_by_headroom=False, headroom_probe_fifo_held=False)
        assert req not in scheduler._inflight_prefills
        assert kwargs['num_external_computed_tokens'] == 0 and not kwargs['delay_cache_blocks']
        assert check['fit']['native_reserved_blocks'] == 0
        members = []
        for old in scheduler._inflight_prefills:
            assert scheduler.requests.get(old.request_id) is old and not old.is_finished()
            allocated = len(manager.req_to_blocks.get(old.request_id, ()))
            remaining = scheduler._request_remaining_blocks(old)
            expected = max(0, math.ceil(min(old.num_tokens, scheduler.max_model_len)/manager.block_size)-allocated)
            stats['geometry_checks'] += 1
            stats['geometry_mismatches'] += int(remaining != expected)
            assert remaining == expected, 'Pinned no-prefix full-attention reservation geometry changed'
            members.append(dict(request_id=old.request_id, status=old.status.name,
                computed=old.num_computed_tokens, output=old.num_output_tokens,
                preemptions=old.num_preemptions, known_sequence_tokens=old.num_tokens,
                allocated_blocks=allocated, remaining_blocks=remaining))
        # The same native helper already used for asynchronous loads. The
        # member sum is retained to verify scope and avoid counting held pages twice.
        reserved = scheduler._inflight_prefill_reserved_blocks()
        assert reserved == sum(m['remaining_blocks'] for m in members)
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == row['free_blocks']
        full_total = row['full_required_blocks'] + row['native_watermark_blocks'] + reserved
        chunk_total = (row['chunk_required_blocks'] + row['native_watermark_blocks'] + reserved
                       if row['chunk_required_blocks'] is not None else None)
        eligible = row['base_allowed'] and kwargs['num_new_tokens'] > 0
        full_opportunity = bool(eligible and reserved > 0 and full_total > row['free_blocks'])
        chunk_opportunity = bool(eligible and reserved > 0 and chunk_total > row['free_blocks'])
        margin = row['free_blocks'] - row['full_required_blocks'] - row['native_watermark_blocks']
        headroom_opportunity = bool(eligible and fixed_headroom_blocks is not None and margin < fixed_headroom_blocks)
        signal_opportunity = full_opportunity if fixed_headroom_blocks is None else headroom_opportunity
        row.update(reservation_observed_t=time.perf_counter()-gate.origin,
            full_fit_margin_blocks=margin, fixed_headroom_opportunity=headroom_opportunity,
            known_prefill_unallocated_blocks=reserved,
            known_recovery_unallocated_blocks=sum(m['remaining_blocks'] for m in members if m['preemptions'] > 0),
            known_nonrecovery_unallocated_blocks=sum(m['remaining_blocks'] for m in members if m['preemptions'] == 0),
            full_plus_existing_unallocated_blocks=full_total,
            chunk_plus_existing_unallocated_blocks=chunk_total,
            full_commit_opportunity=full_opportunity, chunk_commit_opportunity=chunk_opportunity)
        stats['observed_new_allocations'] += 1
        stats['positive_reservation_evaluations'] += int(reserved > 0)
        stats['full_commit_opportunity_evaluations'] += int(full_opportunity)
        stats['chunk_commit_opportunity_evaluations'] += int(chunk_opportunity)
        stats['max_unallocated_blocks'] = max(stats['max_unallocated_blocks'], reserved)
        if full_opportunity and stats['first_full_commit_opportunity'] is None:
            # Keep the decision reference: after_allocate subsequently records
            # the actually executed native result, not a fabricated counterfactual.
            stats['first_full_commit_opportunity'] = dict(decision=row,
                existing_prefills=sorted(members, key=lambda m: m['request_id']))
        if signal_opportunity and probe['first_opportunity'] is None:
            # Both arms capture the same observation. Its anchor is independent
            # of Gate's older first-pending-recovery shadow event.
            event = dict(kind='action' if probe_enabled else 'shadow', target_request_id=req.request_id,
                signal_kind=signal_kind, fixed_headroom_blocks=fixed_headroom_blocks,
                t=row['reservation_observed_t'], state=dict(row), decision=row,
                token_budget=kwargs['token_budget'], scheduled_tokens=dict(gate.scheduled_tokens),
                existing_prefills=sorted(members, key=lambda m: m['request_id']),
                requests=[dict(request_id=r.request_id, status=r.status.name,
                    computed=r.num_computed_tokens, output=r.num_output_tokens,
                    preemptions=r.num_preemptions, num_tokens=r.num_tokens,
                    admitted=r.request_id in gate.admitted_ids)
                    for r in scheduler.requests.values() if not r.is_finished()])
            anchor = time.perf_counter()
            event.update(observation_anchor_perf_s=anchor,
                first_block_perf_s=anchor if probe_enabled else None, deadline_perf_s=anchor+max_extra_s)
            probe['first_opportunity'] = event
        event = probe['first_opportunity']
        if active() and req.request_id == event['target_request_id']:
            now = time.perf_counter()
            if now >= event['deadline_perf_s']:
                release(now, 'hard_deadline', req)
            elif fixed_headroom_blocks is not None and margin >= fixed_headroom_blocks:
                release(now, 'headroom_sufficient', req)
            elif fixed_headroom_blocks is None and reserved == 0:
                release(now, 'reservation_cleared', req)
            elif fixed_headroom_blocks is None and full_total <= row['free_blocks']:
                release(now, 'full_commit_fits', req)
            elif signal_opportunity:
                # Native fit/cap/KV already allowed this allocation. Never turn
                # a native failure into a controller hold that bypasses break.
                row.update(denied=True, reason=action_reason, **{direct_flag: True})
                gate.barrier, gate.barrier_deadline = action_reason, event['deadline_perf_s']
                probe['direct_action_evaluations'] += 1
        return check

    def early(req):
        if gate.barrier == action_reason:
            assert active()
            now = time.perf_counter()
            if now >= probe['first_opportunity']['deadline_perf_s']:
                gate.restore()  # Restore the older target before any later new request.
                release(now, 'hard_deadline', req)
                return 'restart'
        count = len(gate.decisions)
        result = original_early(req)
        if len(gate.decisions) > count:
            row = gate.decisions[-1]
            held = result == 'hold' and row['reason'] == 'fifo_'+action_reason
            # State observation inside the original early hook can itself cross
            # the deadline. Do not turn that last observation into a late hold.
            if held and time.perf_counter() >= probe['first_opportunity']['deadline_perf_s']:
                row.update(denied=False, reason=action_reason+'_deadline_restore',
                    changed_by_reservation=False, reservation_probe_fifo_held=False,
                    changed_by_headroom=False, headroom_probe_fifo_held=False)
                gate.restore()
                release(time.perf_counter(), 'hard_deadline', req)
                return 'restart'
            row.update(changed_by_reservation=False, reservation_probe_fifo_held=False,
                changed_by_headroom=False, headroom_probe_fifo_held=False)
            row[fifo_flag] = held
            if held:
                probe['fifo_action_evaluations'] += 1
                fifo_ids.add(req.request_id)
        return result

    def end(result):
        original_end(result)
        # A connector-only step can have no GPU work. Keep the bounded probe
        # from busy-spinning; Gate's existing timing includes the wait in service
        # time. No wait is added when native/baseline conditions already deny.
        if active() and gate.barrier == action_reason and not result.num_scheduled_tokens:
            now = time.perf_counter()
            delay = min(.001, probe['first_opportunity']['deadline_perf_s']-now)
            if delay > 0:
                time.sleep(delay)
                gate.idle_sleep_s += time.perf_counter()-now

    def report():
        return dict(original_report(), reservation_observation=stats,
            reservation_probe=dict(probe, fifo_held_request_ids=sorted(fifo_ids)))

    gate.before_allocate, gate.early, gate.end, gate.report = before, early, end, report
