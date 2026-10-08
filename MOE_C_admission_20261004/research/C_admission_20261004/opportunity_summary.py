#!/usr/bin/env python3
"""Describe native-fit/reservation scans or bounded-probe ABBA without GPU access.

Usage: python3 opportunity_summary.py ANALYZE_SUMMARY.json --output NEW.json
Run analyze.py first. Service accounting and paired metrics are reused, not
recomputed here. First-opportunity states from separate runs are not state forks.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from analyze import finite
from compare import compare, workload_difference


def counts(rows):
    return dict(evaluations=len(rows), unique_requests=len({r['request_id'] for r in rows}),
                request_ids=sorted({r['request_id'] for r in rows}))


def reservation_summary(admission, decisions, canonical, external, starts, requests):
    """Optional observer schema: absent is unavailable, incomplete is an error."""
    if 'reservation_observation' not in admission:
        return dict(available=False, status='RESERVATION_OBSERVATION_NOT_RECORDED')
    report = admission['reservation_observation']
    counters = ('observed_new_allocations', 'positive_reservation_evaluations',
        'full_commit_opportunity_evaluations', 'chunk_commit_opportunity_evaluations',
        'max_unallocated_blocks', 'geometry_checks', 'geometry_mismatches')
    for key in counters:
        if type(report.get(key)) is not int or report[key] < 0:
            raise ValueError('Missing or invalid reservation counter: ' + key)
    if 'first_full_commit_opportunity' not in report or not isinstance(report.get('semantics'), str):
        raise ValueError('Incomplete reservation observation report')
    fields = ('reservation_observed_t', 'known_prefill_unallocated_blocks',
        'known_recovery_unallocated_blocks', 'known_nonrecovery_unallocated_blocks',
        'full_plus_existing_unallocated_blocks', 'chunk_plus_existing_unallocated_blocks',
        'full_commit_opportunity', 'chunk_commit_opportunity')
    observed = [r for r in decisions if any(k in r for k in fields)]
    for row in observed:
        if any(k not in row for k in fields) or not finite(row['reservation_observed_t']):
            raise ValueError('Incomplete reservation decision row')
        for key in ('known_prefill_unallocated_blocks', 'known_recovery_unallocated_blocks',
                    'known_nonrecovery_unallocated_blocks', 'full_plus_existing_unallocated_blocks'):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError('Invalid reservation decision field: ' + key)
        for key in ('base_allowed', 'full_commit_opportunity', 'chunk_commit_opportunity'):
            if type(row.get(key)) is not bool:
                raise ValueError('Missing or nonboolean reservation decision field: ' + key)
        if ('native_allocation_result' not in row or
                row['native_allocation_result'] is not None and type(row['native_allocation_result']) is not bool):
            raise ValueError('Missing or invalid actual reservation allocation result')
        reserved = row['known_prefill_unallocated_blocks']
        if reserved != row['known_recovery_unallocated_blocks'] + row['known_nonrecovery_unallocated_blocks']:
            raise ValueError('Reservation recovery/nonrecovery sum disagrees')
        eligible = row['base_allowed'] and row['num_new_tokens'] > 0 and reserved > 0
        for kind in ('full', 'chunk'):
            required = row[kind + '_required_blocks']
            expected = required + row['native_watermark_blocks'] + reserved if required is not None else None
            total = row[kind + '_plus_existing_unallocated_blocks']
            if total != expected or eligible and total is None:
                raise ValueError('Inconsistent reservation combined footprint: ' + kind)
            if row[kind + '_commit_opportunity'] != bool(eligible and total > row['free_blocks']):
                raise ValueError('Inconsistent reservation opportunity flag: ' + kind)
    positive = [r for r in observed if r['known_prefill_unallocated_blocks'] > 0]
    full = [r for r in observed if r['full_commit_opportunity']]
    chunk = [r for r in observed if r['chunk_commit_opportunity']]
    derived = dict(observed_new_allocations=len(observed), positive_reservation_evaluations=len(positive),
        full_commit_opportunity_evaluations=len(full), chunk_commit_opportunity_evaluations=len(chunk),
        max_unallocated_blocks=max((r['known_prefill_unallocated_blocks'] for r in observed), default=0))
    if any(report[k] != v for k, v in derived.items()):
        raise ValueError('Reservation counters disagree with recorded gate rows')
    if report['geometry_mismatches'] > report['geometry_checks']:
        raise ValueError('Reservation geometry mismatches exceed checks')

    def allocation_results(rows):
        return dict(success=counts([r for r in rows if r['native_allocation_result'] is True]),
            failure=counts([r for r in rows if r['native_allocation_result'] is False]),
            no_recorded_result=counts([r for r in rows if r['native_allocation_result'] is None]))

    def first_event(rows):
        if not rows:
            return None
        row = rows[0]
        rid, t = row['request_id'], external(row['reservation_observed_t'])
        prefill, request = starts.get(rid), requests[rid]
        return dict(request_id=rid, external_time_s=t, decision_external_time_s=external(row['t']),
            arrival_s=request['arrival_s'], first_prefill_schedule_return_s=prefill,
            event_to_first_prefill_s=prefill-t if prefill is not None else None,
            arrival_to_first_prefill_s=prefill-request['arrival_s'] if prefill is not None else None,
            raw_status=request.get('status'), completion_s=request.get('completion_s'),
            native_allocation_result=row['native_allocation_result'], decision=row)

    first_full, first_chunk = first_event(full), first_event(chunk)
    recorded_first = report['first_full_commit_opportunity']
    if (recorded_first is None) != (first_full is None):
        raise ValueError('Reservation first-full snapshot presence disagrees')
    if recorded_first is not None:
        recorded_row = dict(recorded_first['decision'],
            request_id=canonical(recorded_first['decision']['request_id']))
        if recorded_row != full[0]:
            raise ValueError('Reservation first-full snapshot differs from first marked gate')
        members = [dict(m, request_id=canonical(m['request_id'])) for m in recorded_first['existing_prefills']]
        if sum(m['remaining_blocks'] for m in members) != full[0]['known_prefill_unallocated_blocks']:
            raise ValueError('Reservation first-full member sum disagrees')
        first_full['existing_prefills'] = members
    return dict(available=True, status='OBSERVATION_ONLY', reported_counters={k: report[k] for k in counters},
        counters_agree=True, observed_allocation_hooks=counts(observed),
        decision_rows_without_reservation_observation=counts([r for r in decisions if not any(k in r for k in fields)]),
        positive_reservation=counts(positive), actual_allocation_results=allocation_results(observed),
        full_commit_opportunities=dict(**counts(full), actual_allocation_results=allocation_results(full)),
        chunk_commit_opportunities=dict(**counts(chunk), actual_allocation_results=allocation_results(chunk)),
        first_full_commit_opportunity=first_full, first_chunk_commit_opportunity=first_chunk,
        geometry=dict(checks=report['geometry_checks'], mismatches=report['geometry_mismatches'],
            status='MISMATCH' if report['geometry_mismatches'] else
                'CHECKED_EXISTING_MEMBER_OBSERVATIONS' if report['geometry_checks'] else 'NO_MEMBER_CHECKS',
            scope='Runtime checks count existing-prefill member observations, not gate evaluations. '
                'Only the first full opportunity retains member geometry; aggregate checks cannot '
                'be independently reconstructed from every gate row.'),
        source_semantics=report['semantics'],
        semantics='Full/chunk flags are shadow combined-known-footprint opportunities at actual new-request '
            'allocation hooks, not controller actions or unavoidable capacity failures. Actual result is '
            'reported separately; no recorded result is not a failure or a counterfactual success. '
            'Counts cover all instrumented hook rows, not queue tails skipped by native breaks or FIFO '
            'barriers. Only native existing inflight-prefill members count; fully paused requests outside '
            'that set are excluded. First-event time uses reservation_observed_t aligned to the external '
            'arrival clock. First prefill is the host schedule-return boundary, not GPU completion; its '
            'difference from the event is observed latency, not added causal delay. No future output '
            'length, service benefit, or recovery-specific incremental information is inferred.')


def target_preemptions(raw, rid, event_s, first_token_s, canonical):
    """Sparse host preempt-method records, scoped to the reservation target."""
    if 'preemption_events' not in raw:
        return dict(available=False, status='PREEMPTION_EVENTS_NOT_RECORDED')
    rows = []
    for source in raw['preemption_events']:
        if canonical(source['request_id']) != rid:
            continue
        if any(type(source.get(k)) is not bool for k in
               ('original_preemption_called', 'original_preemption_returned')) or not finite(source.get('method_entered_s')):
            raise ValueError('Incomplete target preemption method record')
        returned, native_output = source['original_preemption_returned'], source['native_output_count_before']
        if type(native_output) is not int or native_output < 0 or returned and (
                not source['original_preemption_called'] or not finite(source.get('method_returned_s'))):
            raise ValueError('Invalid target preemption output count or successful return')
        entered = source['method_entered_s']
        rows.append(dict(source, request_id=rid, after_reservation_event=entered >= event_s,
            before_first_host_token=entered < first_token_s if finite(first_token_s) else None,
            native_output_zero_before=native_output == 0,
            time_since_reservation_event_s=entered-event_s,
            time_to_first_host_token_s=first_token_s-entered if finite(first_token_s) else None))
    actual = [r for r in rows if r['original_preemption_returned']]
    after = [r for r in actual if r['after_reservation_event']]
    return dict(available=True, attempted_calls=len(rows), actual_preemptions=len(actual),
        actual_preemptions_after_event=len(after),
        actual_preemptions_after_event_with_native_output_zero=sum(r['native_output_zero_before'] for r in after),
        actual_preemptions_after_event_before_first_output_both_signals=sum(
            r['native_output_zero_before'] and r['before_first_host_token'] is True for r in after),
        first_host_token_observed=finite(first_token_s), events=rows,
        semantics='Actual means the original native preempt method returned successfully. Post-event uses '
            'method entry on the external-arrival clock. Before-first-output classification requires both '
            'native_output_count_before==0 and method entry before the first host-received token; a missing '
            'host token is unresolved, not proof of this ordering. Native output count is scheduler state; '
            'entry/return and token receipt are host boundaries, not GPU/DMA completion. Counts do not infer '
            'why preemption happened or that reservation caused it.')


def reservation_probe_summary(admission, decisions, canonical, external, starts, requests, raw):
    """Shared namespace, with explicit known-prefill versus ordinary-headroom actions."""
    if 'reservation_probe' not in admission:
        return dict(available=False, status='RESERVATION_PROBE_NOT_RECORDED')
    report = admission['reservation_probe']
    signal_kind = report.get('signal_kind', 'known_prefill')
    if signal_kind not in ('known_prefill', 'fixed_headroom'):
        raise ValueError('Unknown allocation-probe signal_kind: ' + str(signal_kind))
    headroom = signal_kind == 'fixed_headroom'
    if headroom and (type(report.get('fixed_headroom_blocks')) is not int or report['fixed_headroom_blocks'] != 32):
        raise ValueError('The supported fixed-headroom profile requires exactly 32 pages')
    direct_flag = 'changed_by_headroom' if headroom else 'changed_by_reservation'
    fifo_flag = 'headroom_probe_fifo_held' if headroom else 'reservation_probe_fifo_held'
    reason = 'headroom' if headroom else 'reservation'
    trigger_flag = 'fixed_headroom_opportunity' if headroom else 'full_commit_opportunity'
    if type(report.get('probe_enabled')) is not bool or admission['probe_enabled'] is not False:
        raise ValueError('Reservation probe requires an explicit enable flag and disabled older probe')
    if not finite(report.get('max_extra_s')) or not 0 <= report['max_extra_s'] <= .25:
        raise ValueError('Invalid reservation probe deadline bound')
    for row in decisions:
        for key in set(('changed_by_reservation', 'reservation_probe_fifo_held', direct_flag, fifo_flag)):
            if type(row.get(key)) is not bool:
                raise ValueError('Missing or nonboolean allocation-probe field: ' + key)
        inactive = ('changed_by_reservation', 'reservation_probe_fifo_held') if headroom else (
                    'changed_by_headroom', 'headroom_probe_fifo_held')
        if any(key in row and row[key] is not False for key in inactive):
            raise ValueError('Actions from different allocation-probe signals are mixed')
        if headroom and type(row.get('native_fit')) is bool:
            margin = row['free_blocks']-row['full_required_blocks']-row['native_watermark_blocks']
            if row['native_reserved_blocks'] != 0 or row.get('full_fit_margin_blocks') != margin or (
                    type(row.get(trigger_flag)) is not bool or row[trigger_flag] != bool(
                        row['base_allowed'] and row['num_new_tokens'] > 0 and margin < 32)):
                raise ValueError('Fixed-headroom geometry/trigger disagrees with the native-checked row')
    direct = [r for r in decisions if r[direct_flag]]
    fifo = [r for r in decisions if r[fifo_flag]]
    if direct != [r for r in decisions if r['reason'] == reason and r['denied']]:
        raise ValueError('Allocation-probe direct flags disagree with actual refusal rows')
    if fifo != [r for r in decisions if r['reason'] == 'fifo_'+reason and r['denied']]:
        raise ValueError('Allocation-probe FIFO flags disagree with actual hold rows')
    for key, rows in (('direct_action_evaluations', direct), ('fifo_action_evaluations', fifo)):
        if type(report.get(key)) is not int or report[key] != len(rows):
            raise ValueError('Reservation probe counter disagrees: ' + key)
    if sorted(canonical(rid) for rid in report['fifo_held_request_ids']) != counts(fifo)['request_ids']:
        raise ValueError('Reservation FIFO unique request IDs disagree')
    if any(r['native_fit'] is not True or r['base_allowed'] is not True or
           r[trigger_flag] is not True or r['num_new_tokens'] <= 0 or
           r['native_allocation_result'] is not None for r in direct):
        raise ValueError('Reservation direct refusal lacks native-fit/base-allowed local opportunity')
    if any(r['native_fit'] is not None or r['base_allowed'] is not None or
           r['native_allocation_result'] is not None for r in fifo):
        raise ValueError('Unexpected native-fit evidence in reservation FIFO row')
    event, release = report['first_opportunity'], report['first_release']
    if not report['probe_enabled'] and (direct or fifo or release is not None):
        raise ValueError('Disabled reservation probe reports an actual action or release')
    opportunities = [r for r in decisions if r.get(trigger_flag)]
    if (event is None) != (not opportunities) or event is None and (direct or fifo or release is not None):
        raise ValueError('Selected signal trigger presence disagrees with opportunities/actions')
    event_summary, release_summary = None, None
    if event is not None:
        if event['kind'] != ('action' if report['probe_enabled'] else 'shadow'):
            raise ValueError('Reservation event kind disagrees with enable flag')
        if event.get('signal_kind', 'known_prefill') != signal_kind or (
                headroom and event.get('fixed_headroom_blocks') != 32):
            raise ValueError('Event signal configuration disagrees with allocation-probe report')
        rid, t = canonical(event['target_request_id']), external(event['t'])
        decision = dict(event['decision'], request_id=canonical(event['decision']['request_id']))
        if decision != opportunities[0] or rid != decision['request_id'] or any(r['request_id'] != rid for r in direct):
            raise ValueError('Allocation-probe event/direct target differs from first selected-signal opportunity')
        anchor, block, deadline = (event[k] for k in
            ('observation_anchor_perf_s', 'first_block_perf_s', 'deadline_perf_s'))
        if not finite(anchor) or not finite(deadline) or deadline < anchor or (
                report['probe_enabled'] and not finite(block)) or (not report['probe_enabled'] and block is not None):
            raise ValueError('Invalid reservation trigger clock fields')
        state = dict(event['state'], request_id=canonical(event['state']['request_id']))
        if state['request_id'] != rid or state['base_allowed'] is not True or state['native_fit'] is not True or (
                state[trigger_flag] is not True or state['denied'] or state[direct_flag]):
            raise ValueError('Reservation event state is not the allowed pre-action opportunity')
        state_rows = sorted((dict(r, request_id=canonical(r['request_id'])) for r in event['requests']),
                           key=lambda r: r['request_id'])
        pre = dict(recorded=state, token_budget=event['token_budget'],
            scheduled_tokens={canonical(k): v for k, v in event['scheduled_tokens'].items()},
            unfinished_requests=len(state_rows), admitted_requests=sum(r['admitted'] for r in state_rows),
            live_computed_positive_requests=sum(r['computed'] > 0 for r in state_rows),
            live_output_positive_requests=sum(r['output'] > 0 for r in state_rows),
            computed_tokens=sum(r['computed'] for r in state_rows), output_tokens=sum(r['output'] for r in state_rows),
            status_counts=dict(collections.Counter(r['status'] for r in state_rows)), requests=state_rows,
            existing_prefills=[dict(r, request_id=canonical(r['request_id'])) for r in event['existing_prefills']])
        prefill, req = starts.get(rid), requests[rid]
        first_token = req['token_times_s'][0] if req.get('token_times_s') else None
        absolute_external = lambda perf: external(perf-admission['origin_perf_s']) if finite(perf) else None
        event_summary = dict(kind=event['kind'], signal_kind=signal_kind,
            fixed_headroom_blocks=report.get('fixed_headroom_blocks'), request_id=rid, external_time_s=t,
            observation_anchor_external_s=absolute_external(anchor), first_block_external_s=absolute_external(block),
            deadline_external_s=absolute_external(deadline), deadline_after_observation_anchor_s=deadline-anchor,
            actual_probe_denial_observed=bool(direct), before_state=pre, final_trigger_decision=decision,
            first_prefill_schedule_return_s=prefill, event_to_first_prefill_s=prefill-t if prefill is not None else None,
            first_block_to_first_prefill_s=prefill-absolute_external(block)
                if prefill is not None and finite(block) else None,
            arrival_s=req['arrival_s'], arrival_to_first_prefill_s=prefill-req['arrival_s'] if prefill is not None else None,
            source_request_outcome=req.get('status'), completion_s=req.get('completion_s'),
            observed_output_tokens=len(req.get('output_token_ids', [])), stop_reason=req.get('stop_reason'),
            first_token_host_s=first_token, event_to_first_token_s=first_token-t if finite(first_token) else None,
            first_prefill_to_first_token_s=first_token-prefill
                if finite(first_token) and prefill is not None else None,
            preemptions=target_preemptions(raw, rid, t, first_token, canonical))
        if release is not None:
            release_reasons = ('headroom_sufficient', 'hard_deadline') if headroom else (
                              'reservation_cleared', 'full_commit_fits', 'hard_deadline')
            if release['reason'] not in release_reasons or not all(
                    finite(release.get(k)) for k in ('t', 'perf_s')):
                raise ValueError('Invalid reservation release observation')
            released = external(release['t'])
            release_summary = dict(release, observed_request_id=canonical(release['observed_request_id']),
                external_time_s=released, time_since_first_block_s=release['perf_s']-block,
                release_to_target_first_prefill_s=prefill-released if prefill is not None else None)
    affected = []
    for rid in sorted({r['request_id'] for r in direct+fifo}):
        dr, fr = ([r for r in rows if r['request_id'] == rid] for rows in (direct, fifo))
        first, prefill = min(external(r['t']) for r in dr+fr), starts.get(rid)
        affected.append(dict(request_id=rid, direct_evaluations=len(dr), fifo_evaluations=len(fr),
            first_denial_external_s=first, first_prefill_schedule_return_s=prefill,
            denial_to_first_prefill_s=prefill-first if prefill is not None else None,
            arrival_to_first_prefill_s=prefill-requests[rid]['arrival_s'] if prefill is not None else None,
            raw_status=requests[rid].get('status')))
    return dict(available=True, probe_enabled=report['probe_enabled'], signal_kind=signal_kind,
        fixed_headroom_blocks=report.get('fixed_headroom_blocks'), max_extra_s=report['max_extra_s'],
        signal_opportunities=counts(opportunities),
        first_opportunity=event_summary, first_release=release_summary, counters_agree=True,
        direct_action=counts(direct), direct_rows=direct, fifo_action=counts(fifo),
        directly_or_fifo_held=counts(direct+fifo), affected_requests=affected,
        direct_validation=dict(checked_evaluations=len(direct), status=
            'CHECKED_ACTUAL_DIRECT_REFUSAL_FLAGS' if direct else 'NO_DIRECT_REFUSALS'),
        deadline_restore_observations=counts([r for r in decisions if r['reason'] == reason+'_deadline_restore']),
        semantics='The signal_kind distinguishes known-prefill reservation from ordinary fixed 32-page headroom; '
            'fixed headroom uses current full-fit margin and does not require recovery. The R-full observation '
            'namespace retains its independent shadow meaning. Both are separate from the older pending-recovery probe. '
            'Direct counts require a native-fit, cap/KV-allowed local selected-signal opportunity actually denied '
            'before allocation. FIFO holds have no native-fit query and are not proven executable counterfactuals. '
            'A shadow or action-kind trigger alone is not a measured refusal. Release is a host gate observation, '
            'not actual prefill; a deadline bounds extra refusal, not total waiting. Event-to-prefill and affected '
            'request waiting are descriptive, not causal added delay. Missing release remains unobserved.')


def progress_release_summary(admission, decisions, canonical, external):
    """The explicit progress schema is strict; older reports stay distinguishable."""
    if 'release_mode' not in admission:
        return dict(available=False, status='LEGACY_RELEASE_DIAGNOSTICS_NOT_RECORDED')
    mode = admission['release_mode']
    if mode not in ('timer', 'output_progress'):
        raise ValueError('Unknown release_mode: ' + str(mode))
    for row in decisions:
        for key in ('timer_would_block', 'progress_would_block', 'changed_by_progress',
                    'progress_additional_opportunity'):
            if type(row.get(key)) is not bool:
                raise ValueError('Missing or nonboolean progress decision field: ' + key)
    event = admission['first_opportunity']
    block = event.get('first_block_perf_s') if event else None
    anchor = event['observation_anchor_perf_s'] if event else None

    def observation(row):
        if row is None:
            return None
        return dict(row, external_time_s=external(row['t']),
            time_since_first_block_s=row['perf_s']-block if finite(block) else None)

    target_progress = {}
    for rid, record in admission['recovery_progress'].items():
        target_progress[canonical(rid)] = dict(record,
            **{key: observation(record[key]) for key in
               ('first_scheduled', 'first_output', 'first_finished', 'first_missing')})
    changed = [r for r in decisions if r['changed_by_progress']]
    additional = [r for r in decisions if r['progress_additional_opportunity']]
    invalid = [r for r in changed if mode != 'output_progress' or r.get('base_allowed') is not True
               or r['timer_would_block'] or not r['progress_would_block'] or not r.get('denied')]
    return dict(available=True, release_mode=mode,
        actual_changed_by_progress=counts(changed), actual_changed_rows=changed,
        actual_changed_flags_consistent=not invalid,
        additional_eligible_observations=counts(additional),
        timer_shadow_additional_opportunities=counts(additional if mode == 'timer' else []),
        timer_shadow_rows=additional if mode == 'timer' else [],
        frozen_pending_targets=[dict(r, request_id=canonical(r['request_id']))
                                for r in event['pending_targets']] if event else [],
        observation_anchor_external_s=external(anchor-admission['origin_perf_s'])
            if finite(anchor) else None,
        minimum_progress_release_after_observation_anchor_s=event['progress_min_release_perf_s']-anchor
            if event else None,
        minimum_progress_release_after_first_block_s=event['progress_min_release_perf_s']-block
            if event and finite(block) else None,
        target_progress=target_progress,
        release_observations={key: observation(admission['release_observations'][key])
                              for key in ('timer', 'output_progress', 'actual')},
        semantics='Only changed_by_progress in an output_progress arm is an actual extra direct refusal '
            'after timer opening with native fit and KV/cap allowance. Timer-arm additional opportunities '
            'are shadow suggestions, not actions. FIFO window suggestions do not establish native fit. '
            'Targets are frozen at trigger; observations and release opening are host boundaries, not '
            'GPU completion or actual new-prefill permission. Scheduled work is distinct from output progress.')


def summarize(cell):
    path = Path(cell['cell'])
    payload = (path / 'raw.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != cell['raw_sha256']:
        raise ValueError('raw.json changed since analyze.py: ' + str(path))
    raw = json.loads(payload)
    del payload
    admission_path = path / 'admission.json'
    admission = json.loads(admission_path.read_text())
    for key in ('probe_enabled', 'first_opportunity', 'native_allocations', 'fit_checks', 'fit_mismatches'):
        if key not in admission:
            raise ValueError('Not an admission_probe report: ' + str(path))
    origin, gate_origin = raw.get('measurement_origin_perf_counter_s'), admission.get('origin_perf_s')
    if not finite(origin) or not finite(gate_origin):
        raise ValueError('Missing performance-clock origin: ' + str(path))
    requests = {r['request_id']: r for r in raw['requests']}
    identity = {r[k]: rid for rid, r in requests.items()
                for k in ('request_id', 'internal_request_id', 'external_request_id') if r.get(k)}

    def canonical(rid):
        if rid not in identity:
            raise ValueError('Unmapped request identity: ' + str(rid))
        return identity[rid]

    def external(t):
        return gate_origin + t - origin if finite(t) else None

    starts = {canonical(s['request_id']): s['first_prefill_perf_s'] - origin
              for s in admission['starts']}
    decisions = [dict(r, request_id=canonical(r['request_id'])) for r in admission['decisions']]
    direct = [r for r in decisions if r.get('reason') in ('probe', 'probe_progress') and r.get('denied')]
    fifo = [r for r in decisions if r.get('reason') == 'fifo_probe' and r.get('denied')]
    changed = [r for r in decisions if r.get('changed_by_recovery')]
    progress_release = progress_release_summary(admission, decisions, canonical, external)
    allocations = admission['native_allocations']
    mismatches = [r for r in allocations if r.get('predicted') != r.get('actual')]
    unknown_fit = sum(type(r.get('predicted')) is not bool or type(r.get('actual')) is not bool
                      for r in allocations)
    checks_agree = admission['fit_checks'] == len(allocations)
    mismatch_agree = admission['fit_mismatches'] == len(mismatches)
    validation = dict(reported_checks=admission['fit_checks'], recorded_allocations=len(allocations),
        reported_mismatches=admission['fit_mismatches'], observed_mismatches=len(mismatches),
        unknown_boolean_results=unknown_fit, counters_agree=checks_agree and mismatch_agree,
        status=('MISMATCH_OR_INCONSISTENT' if mismatches or unknown_fit or not checks_agree or not mismatch_agree
                else 'CHECKED_EXECUTED_ALLOCATIONS_ONLY' if allocations else 'NO_EXECUTED_CHECKS'),
        actual_successes=sum(r.get('actual') is True for r in allocations),
        actual_failures=sum(r.get('actual') is False for r in allocations),
        deferred_predicted_fit_without_allocation=counts([r for r in decisions
            if r.get('native_fit') is True and r.get('native_allocation_result') is None]),
        mismatch_examples=mismatches[:5], source_sha256=admission.get('source_sha256'),
        scope='Validation covers executed native allocations, including recovery. Deferred requests have '
              'no actual allocation result; their read-only fit prediction is not a measured counterfactual.')

    pending = [r for r in decisions if r.get('recovery_count', 0) > 0]
    combinations = collections.defaultdict(list)
    covered = {key: [] for key in ('cap', 'kv', 'native_capacity', 'native_fit_unknown', 'legal_local')}
    for row in pending:
        cap = row['active'] >= admission['cap']
        kv = row['free_blocks'] < admission['kv_floor'] and row['age_s'] < 10
        native = None if row.get('native_fit') is None else not row['native_fit']
        combinations[f'cap={cap},kv={kv},native_block={native}'].append(row)
        if cap:
            covered['cap'].append(row)
        if kv:
            covered['kv'].append(row)
        if native is True:
            covered['native_capacity'].append(row)
        if native is None:
            covered['native_fit_unknown'].append(row)
        if row.get('opportunity'):
            covered['legal_local'].append(row)
    coverage = dict(pending_recovery_evaluations=counts(pending),
        overlapping_conditions={key: counts(rows) for key, rows in covered.items()},
        condition_combinations={key: counts(rows) for key, rows in combinations.items()},
        actual_reasons=dict(collections.Counter(r['reason'] for r in pending)),
        semantics='Conditions overlap. FIFO-barrier rows have no native-fit query; unknown is not failure. '
                  'Only new requests actually reaching an instrumented hook are observed, not queue tails '
                  'behind native breaks. No model for unobserved or future opportunities is inferred.')

    event = admission['first_opportunity']
    event_summary = None
    if event is not None:
        rid, t = canonical(event['request_id']), external(event['t'])
        state_rows = [dict(r, request_id=canonical(r['request_id'])) for r in event['requests']]
        state_rows.sort(key=lambda r: r['request_id'])
        pre = dict(recorded=event['state'], fit=event['fit'], token_budget=event['token_budget'],
            scheduled_tokens={canonical(k): v for k, v in event['scheduled_tokens'].items()},
            unfinished_requests=len(state_rows),
            admitted_requests=sum(r['admitted'] for r in state_rows),
            live_computed_positive_requests=sum(r['computed'] > 0 for r in state_rows),
            live_output_positive_requests=sum(r['output'] > 0 for r in state_rows),
            computed_tokens=sum(r['computed'] for r in state_rows),
            output_tokens=sum(r['output'] for r in state_rows),
            status_counts=dict(collections.Counter(r['status'] for r in state_rows)),
            requests=state_rows)
        block = event.get('first_block_perf_s')
        prefill = starts.get(rid)
        req = requests[rid]
        event_summary = dict(kind=event['kind'], request_id=rid, external_time_s=t,
            actual_probe_denial_observed=bool(direct), before_state=pre,
            first_prefill_schedule_return_s=prefill,
            event_to_first_prefill_s=prefill-t if prefill is not None else None,
            first_block_to_first_prefill_s=prefill-(block-origin)
                if prefill is not None and finite(block) else None,
            requested_release_after_first_block_s=event.get('release_perf_s', block)-block
                if finite(block) else None,
            hard_deadline_after_first_block_s=event.get('hard_deadline_perf_s', block)-block
                if finite(block) else None,
            source_request_outcome=req.get('status'), completion_s=req.get('completion_s'),
            arrival_s=req['arrival_s'], observed_output_tokens=len(req.get('output_token_ids', [])),
            latest_logged_direct_denial_after_block_s=max(
                (gate_origin+r['t']-block for r in direct), default=None) if finite(block) else None,
            interpretation='This is the recorded pre-action host state and observed event-to-prefill time, '
                'not added causal delay. First prefill is the scheduler-return assignment boundary, '
                'not GPU completion. Current computed/output counts are host state, not complete history. '
                'requested_release_after_first_block_s is the nominal timer boundary; progress-release '
                'observations separately record when the selected gate was observed to open.')

    affected = []
    for rid in sorted({r['request_id'] for r in direct + fifo}):
        dr = [r for r in direct if r['request_id'] == rid]
        fr = [r for r in fifo if r['request_id'] == rid]
        first = min(external(r['t']) for r in dr + fr)
        affected.append(dict(request_id=rid, direct_probe_evaluations=len(dr), fifo_probe_evaluations=len(fr),
            first_denial_external_s=first, first_prefill_schedule_return_s=starts.get(rid),
            denial_to_first_prefill_s=starts[rid]-first if rid in starts else None,
            new_prefill_wait_s=starts[rid]-requests[rid]['arrival_s'] if rid in starts else None,
            raw_status=requests[rid].get('status')))
    summary = dict(cell=cell['cell'], probe_enabled=admission['probe_enabled'],
        admission_sha256=hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        configuration={k: admission.get(k) for k in ('cap', 'kv_floor', 'delay_s', 'max_extra_s',
            'active_definition', 'release_mode')},
        native_fit_validation=validation, first_opportunity=event_summary, recovery_opportunity_coverage=coverage,
        progress_release=progress_release,
        reservation_observation=reservation_summary(admission, decisions, canonical, external, starts, requests),
        reservation_probe=reservation_probe_summary(admission, decisions, canonical, external, starts, requests, raw),
        intervention=dict(direct_probe=counts(direct), fifo_probe=counts(fifo), changed_by_recovery=counts(changed),
            directly_or_fifo_held=counts(direct+fifo),
            changed_flag_agrees_with_direct_rows=(changed == direct), affected_requests=affected,
            other_fifo_reasons=dict(collections.Counter(r['reason'] for r in decisions
                if r.get('reason', '').startswith('fifo_') and r['reason'] != 'fifo_probe')),
            semantics='Direct probe rows are actual baseline-allowed extra refusals. FIFO-probe rows are '
                'actual propagation of the probe barrier, but native fit is unqueried, so they are not '
                'all proven extra refusals relative to native scheduling. Repeated evaluations are not runs.'),
        controller={k: v for k, v in admission.items() if k.startswith('controller_') or k == 'calls'},
        service={k: cell[k] for k in ('raw_sha256', 'workload_identity_sha256', 'run_status', 'error',
            'planned_requests', 'arrived_requests', 'outcomes', 'all_completed', 'actual_preemption_count',
            'arrival_window_s', 'service_denominator_s', 'completed_drain_s', 'observed_drain_s', 'drain_censored',
            'distributions', 'total_output_tokens', 'output_tokens_per_s', 'natural_stop_count',
            'length_stop_count', 'stop_reasons', 'joint_slo_grid', 'per_request_json', 'per_request_csv')})
    # Pairwise output comparison needs sequences but not the large per-token time trace.
    retained = {'requests': [{k: v for k, v in r.items() if k in (
        'request_id', 'arrival_s', 'prompt_token_ids_sha256', 'prompt_tokens', 'max_output_tokens',
        'output_token_ids', 'status', 'stop_reason', 'finish_reason')} for r in requests.values()]}
    if 'reservation_probe' in admission:
        # Keep compact same-request evidence for pairing after all four arms are
        # loaded. No per-token trace or unrelated raw engine-step data survives.
        first_gates = {kind: {} for kind in ('any', 'native_checked', 'legal', 'allowed')}
        for row in decisions:
            predicates = dict(any=True, native_checked=type(row.get('native_fit')) is bool,
                legal=row.get('native_fit') is True and row.get('base_allowed') is True,
                allowed=row.get('native_allocation_result') is True)
            for kind, include in predicates.items():
                if include:
                    first_gates[kind].setdefault(row['request_id'],
                        dict(row, external_time_s=external(row['t'])))
        preempt_raw = {'preemption_events': [dict(r, request_id=canonical(r['request_id']))
            for r in raw['preemption_events']]} if 'preemption_events' in raw else {}
        retained['_matched_target_context'] = dict(starts=starts, first_gates=first_gates,
            preempt_raw=preempt_raw, requests={rid: dict(
                request_id=rid, arrival_s=r['arrival_s'], prompt_tokens=r.get('prompt_tokens'),
                prompt_token_ids_sha256=r.get('prompt_token_ids_sha256'),
                max_output_tokens=r.get('max_output_tokens'), status=r.get('status'),
                stop_reason=r.get('stop_reason', r.get('finish_reason')),
                completion_s=r.get('completion_s'), observed_output_tokens=len(r.get('output_token_ids', [])),
                first_token_host_s=r['token_times_s'][0] if r.get('token_times_s') else None)
                for rid, r in requests.items()})
    return summary, retained


def matched_candidate_target(candidate, baseline, candidate_raw, baseline_raw):
    """Compare the candidate allocation-probe target with that same ID in its fixed pair."""
    event = candidate['reservation_probe']['first_opportunity']
    if event is None:
        return dict(available=False, status='NO_CANDIDATE_RESERVATION_TARGET')
    rid = event['request_id']

    def arm(summary, retained):
        context = retained['_matched_target_context']
        request = context['requests'][rid]
        first_token = request['first_token_host_s']
        # Reuse the native-return/entered-time classification, but count the
        # whole request lifecycle rather than either arm's different trigger.
        preemptions = target_preemptions(context['preempt_raw'], rid, request['arrival_s'],
            first_token, lambda request_id: request_id)
        if preemptions['available']:
            events = [{k: v for k, v in r.items() if k not in
                       ('after_reservation_event', 'time_since_reservation_event_s')}
                      for r in preemptions['events']]
            actual = [r for r in events if r['original_preemption_returned']]
            before = [r for r in actual if r['native_output_zero_before'] and
                      r['before_first_host_token'] is True]
            preemptions = dict(available=True, attempted_calls=preemptions['attempted_calls'],
                actual_preemptions=len(actual),
                actual_preemptions_before_first_output_both_signals=len(before)
                    if finite(first_token) else None,
                first_host_token_observed=finite(first_token), events=events,
                semantics='Full lifecycle count, not truncated at either reservation event. Actual means '
                    'the original native method returned; pre-output requires native output zero and '
                    'method entry before the first host token. Missing first token is unresolved.')
        gates = {kind: rows.get(rid) for kind, rows in context['first_gates'].items()}
        prefill, arrival, completion = context['starts'].get(rid), request['arrival_s'], request['completion_s']
        difference = lambda a, b: a-b if finite(a) and finite(b) else None
        return dict(request, cell=summary['cell'], raw_sha256=summary['service']['raw_sha256'],
            admission_sha256=summary['admission_sha256'],
            first_gate=gates['any'], first_native_fit_gate=gates['native_checked'],
            first_legal_gate=gates['legal'], first_allowed_gate=gates['allowed'],
            first_gate_external_s=gates['any']['external_time_s'] if gates['any'] else None,
            first_native_fit_gate_external_s=gates['native_checked']['external_time_s'] if gates['native_checked'] else None,
            first_legal_gate_external_s=gates['legal']['external_time_s'] if gates['legal'] else None,
            first_prefill_schedule_return_s=prefill,
            arrival_to_first_prefill_s=difference(prefill, arrival), ttft_s=difference(first_token, arrival),
            flow_s=difference(completion, arrival), first_prefill_to_first_token_s=difference(first_token, prefill),
            preemptions=preemptions)

    a, b = arm(candidate, candidate_raw), arm(baseline, baseline_raw)
    for key in ('arrival_s', 'prompt_tokens', 'prompt_token_ids_sha256', 'max_output_tokens'):
        if a[key] != b[key]:
            raise ValueError('Matched candidate target workload differs: ' + key)
    differences = {}
    for key in ('arrival_s', 'first_gate_external_s', 'first_native_fit_gate_external_s',
                'first_legal_gate_external_s', 'first_prefill_schedule_return_s', 'first_token_host_s',
                'completion_s', 'arrival_to_first_prefill_s', 'ttft_s', 'flow_s',
                'first_prefill_to_first_token_s', 'observed_output_tokens'):
        differences[key] = a[key]-b[key] if finite(a[key]) and finite(b[key]) else None
    for key in ('actual_preemptions', 'actual_preemptions_before_first_output_both_signals'):
        av, bv = a['preemptions'].get(key), b['preemptions'].get(key)
        differences[key] = av-bv if finite(av) and finite(bv) else None
    return dict(available=True, comparison_event_namespace='reservation_probe',
        signal_kind=candidate['reservation_probe'].get('signal_kind', 'known_prefill'), canonical_request_id=rid,
        candidate_event_external_s=event['external_time_s'],
        candidate_actual_direct_actions=candidate['reservation_probe']['direct_action']['evaluations'],
        candidate=a, baseline=b, candidate_minus_baseline=differences,
        same_stop_reason=a['stop_reason'] == b['stop_reason'], same_status=a['status'] == b['status'],
        semantics='The candidate first allocation-probe target selects the canonical ID for both fixed paired '
            'runs; the baseline own first target is not substituted. First gate includes any recorded FIFO '
            'observation, while native-fit and legal gate boundaries are separate. Missing gate data does '
            'not identify earlier native break reasons. Host gate/schedule/token/preemption times are not '
            'GPU completion. Separate-run initial state and prior waiting can differ; deltas do not '
            'establish a same-state intervention effect or imply service benefit from fewer pre-output preemptions.')


def event_similarity(candidate, baseline):
    a, b = candidate['first_opportunity'], baseline['first_opportunity']
    if a is None or b is None:
        return dict(status='NO_COMPARABLE_EVENT', candidate_has_event=a is not None, baseline_has_event=b is not None,
            target_identity='UNAVAILABLE', candidate_request=a['request_id'] if a else None,
            baseline_request=b['request_id'] if b else None)
    ap, bp = a['before_state'], b['before_state']
    states = {}
    for key in ('active', 'running', 'free_blocks', 'recovery_count', 'recovery_waiting_count',
                'recovery_remote_inflight_count', 'recovery_ready_count', 'recovery_running_recompute_count',
                'known_prefill_unallocated_blocks', 'known_recovery_unallocated_blocks',
                'known_nonrecovery_unallocated_blocks', 'full_plus_existing_unallocated_blocks',
                'chunk_plus_existing_unallocated_blocks', 'full_fit_margin_blocks'):
        x, y = ap['recorded'].get(key), bp['recorded'].get(key)
        states[key] = dict(candidate=x, baseline=y, delta=x-y if finite(x) and finite(y) else None)
    for key in ('live_computed_positive_requests', 'live_output_positive_requests', 'computed_tokens',
                'output_tokens', 'unfinished_requests', 'admitted_requests', 'token_budget'):
        x, y = ap[key], bp[key]
        states[key] = dict(candidate=x, baseline=y, delta=x-y)
    ar, br = ({r['request_id']: r for r in p['requests']} for p in (ap, bp))
    common = ar.keys() & br.keys()
    return dict(status='DESCRIPTIVE_SEPARATE_RUN_STATES', same_first_request=a['request_id'] == b['request_id'],
        target_identity='SAME_REQUEST_ID' if a['request_id'] == b['request_id'] else 'DIFFERENT_REQUEST_IDS',
        candidate_request=a['request_id'], baseline_request=b['request_id'],
        external_event_time_delta_s=a['external_time_s']-b['external_time_s'], state_differences=states,
        candidate_only_unfinished_ids=sorted(ar.keys()-br.keys()),
        baseline_only_unfinished_ids=sorted(br.keys()-ar.keys()), common_unfinished_requests=len(common),
        common_request_field_mismatch_counts={k: sum(ar[rid][k] != br[rid][k] for rid in common)
            for k in ('status', 'computed', 'output', 'preemptions', 'num_tokens', 'admitted')},
        same_scheduled_tokens=ap['scheduled_tokens'] == bp['scheduled_tokens'],
        interpretation='Identity and host-state similarity are descriptive. Even identical fields do not '
            'prove identical GPU/KV/transfer state, a shared randomness trajectory, or a strict causal state fork.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matched-targets-only', action='store_true',
        help='Write only reservation/headroom same-candidate-target pair diagnostics to the new output.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing output.')
    analysis = json.loads(args.analysis.read_text())
    cells = analysis['cells']
    if len(cells) not in (1, 4):
        parser.error('Supply only one scan cell or the four ABBA cells in execution order.')
    summaries, raws = [], {}
    for cell in cells:
        summary, raw = summarize(cell)
        summaries.append(summary)
        raws[cell['cell']] = raw
    order = [c['probe_enabled'] for c in summaries]
    release_order = [c['configuration']['release_mode'] for c in summaries]
    reservation_order = [c['reservation_probe'].get('probe_enabled') for c in summaries]
    signal_kinds = [c['reservation_probe'].get('signal_kind') for c in summaries]
    if order == [False]*4 and reservation_order == [False, True, True, False]:
        if signal_kinds == ['fixed_headroom']*4:
            design = 'BASELINE_HEADROOM_HEADROOM_BASELINE'
        elif signal_kinds == ['known_prefill']*4:
            design = 'BASELINE_RESERVATION_RESERVATION_BASELINE'
        else:
            parser.error('All four allocation-probe arms must use the same explicit signal kind.')
    elif order == [True]*4 and release_order == ['timer', 'output_progress', 'output_progress', 'timer']:
        design = 'TIMER_PROGRESS_PROGRESS_TIMER'
    elif order in ([False], [False, True, True, False]) and all(v in (None, 'timer') for v in release_order) and all(
            v in (None, False) for v in reservation_order):
        design = 'BASELINE_SCAN' if len(cells) == 1 else 'BASELINE_DELAY_DELAY_BASELINE'
    else:
        parser.error('Expected baseline scan, baseline/delay/delay/baseline, timer/progress/progress/timer, '
                     'or baseline/reservation-or-headroom/reservation-or-headroom/baseline.')
    allocation_probe = design in ('BASELINE_RESERVATION_RESERVATION_BASELINE', 'BASELINE_HEADROOM_HEADROOM_BASELINE')
    pairs = []
    for i, j in ((1, 0), (2, 3)) if len(cells) == 4 else ():
        reason = workload_difference(raws[cells[i]['cell']], raws[cells[j]['cell']])
        if reason:
            raise ValueError('Cannot pair different workloads: ' + reason)
        pair = compare(cells[i], cells[j], raws)
        pair['candidate_release_mode'] = release_order[i]
        pair['baseline_release_mode'] = release_order[j]
        if allocation_probe:
            pair['comparison_event_namespace'] = 'reservation_probe'
            pair['comparison_signal_kind'] = signal_kinds[i]
            pair['candidate_reservation_probe_enabled'] = reservation_order[i]
            pair['baseline_reservation_probe_enabled'] = reservation_order[j]
            pair['first_opportunity_similarity'] = event_similarity(
                summaries[i]['reservation_probe'], summaries[j]['reservation_probe'])
            pair['matched_candidate_target'] = matched_candidate_target(
                summaries[i], summaries[j], raws[cells[i]['cell']], raws[cells[j]['cell']])
        else:
            pair['first_opportunity_similarity'] = event_similarity(summaries[i], summaries[j])
        pairs.append(pair)
    result = dict(schema_version=1, source_analysis=str(args.analysis.resolve()), experiment_design=design,
        role='EXPLORATORY_NATIVE_FIT_OPPORTUNITY_AND_SINGLE_ACTION_DIAGNOSTIC', independent_unit='run',
        inference='One scan or two ABBA pairs; no request-level significance or strict same-state claim. '
            'An action-kind event without actual direct denials is not an intervention. Natural output '
            'differences are retained and cannot be treated as equal-work acceleration. Progress-specific '
            'intervention requires changed_by_progress, not merely the shared initial timer denials. '
            'Reservation/headroom ABBA pairs use reservation_probe first events with explicit signal_kind; '
            'known_prefill counts changed_by_reservation, while fixed_headroom counts changed_by_headroom. '
            'Fixed 32-page headroom is an ordinary capacity rule, not a recovery mechanism or novelty claim; '
            'the older pending-recovery event and changed_by_recovery are separate diagnostics.',
        service_semantics={k: analysis.get(k) for k in ('measurement_semantics', 'denominator_semantics',
            'timeout_semantics', 'slo_status')}, cells=summaries, abba_pairs=pairs)
    if args.matched_targets_only:
        if not allocation_probe:
            parser.error('--matched-targets-only requires the four reservation or headroom ABBA arms.')
        result = dict(schema_version=1, source_analysis=str(args.analysis.resolve()),
            source_analysis_sha256=hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
            experiment_design=design, independent_unit='run',
            abba_pairs=[{k: p[k] for k in ('candidate', 'baseline', 'matched_candidate_target')} for p in pairs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    for c in summaries:
        reservation = allocation_probe
        event = c['reservation_probe']['first_opportunity'] if reservation else c['first_opportunity']
        action = c['reservation_probe'] if reservation else c['intervention']
        print(f"{Path(c['cell']).name}: fit={c['native_fit_validation']['status']}; "
              f"event={event['kind'] if event else 'none'}; "
              f"direct/fifo unique={action['direct_action' if reservation else 'direct_probe']['unique_requests']}/"
              f"{action['fifo_action' if reservation else 'fifo_probe']['unique_requests']}; outcomes={c['service']['outcomes']}")
    print(args.output.resolve())


if __name__ == '__main__':
    main()
