"""CPU-only analysis of TERMINAL async/inline native Host groups.

Usage: python -B analyze_inline_host.py execution/GROUP --out execution/GROUP/inline_summary.json
The caller must first establish that the controller/worker exited and released
the shared lock. Status checks here prevent accidental live-group analysis but
are not a substitute for that resource check. Old completed groups are accepted.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from analyze import analyze_cell, compare_tokens, fingerprint, read_json, stats


def compact_event(event):
    fields = ('event', 'request_id', 'decision_s', 'allocation_s', 'known_tokens',
              'generated_tokens', 'host_hit_tokens', 'prefix_sha256', 'preemptions',
              'free_blocks', 'full_required_blocks', 'reserved_blocks', 'watermark_blocks',
              'joint_capacity', 'eligible', 'pending_load_jobs', 'pending_transfer_jobs',
              'running', 'waiting', 'recent_step_s', 'action', 'actual_action', 'fallback')
    return {key: event.get(key) for key in fields}


def union_intervals(intervals):
    """Do not multiply one batch wait by its number of inline requests."""
    merged = []
    for begin, end in sorted(set(intervals)):
        if end < begin:
            raise ValueError('negative inline wait interval')
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([begin, end])
    return merged


def step_index(raw, time_s):
    if time_s is None:
        return None
    ends = [step['end_s'] for step in raw.get('steps', [])]
    index = bisect_left(ends, time_s)
    return index if index < len(ends) else None


def output_step_sequences(raw):
    """Emission ordinal -> measured engine.step index, independent of wall drift."""
    ends = {step['end_s']: index for index, step in enumerate(raw.get('steps', []))}
    return {row['external_id']: [ends[time_s] for time_s in row['token_times_s']]
            for row in raw['requests']}


def successor(raw, request, decision_s):
    if request is None or decision_s is None:
        return None
    times = request['token_times_s']
    index = bisect_right(times, decision_s)
    next_s = times[index] if index < len(times) else None
    rid = request['request_id']
    preempts = [row for row in raw.get('preemptions', []) if row['request_id'] == rid
                and decision_s < row['time_s'] and (next_s is None or row['time_s'] < next_s)]
    return dict(output_token_index_zero_based=index if next_s is not None else None,
                next_output_s=next_s, output_step=step_index(raw, next_s),
                decision_to_next_output_s=next_s-decision_s if next_s is not None else None,
                preemptions_before_next_output=len(preempts),
                completion_s=request.get('completion_s'), censored=next_s is None)


def decision_state(raw, request, event):
    """Observed state at lookup; the previous completed step is separately labelled."""
    time_s = event['decision_s']
    index = step_index(raw, time_s)
    rows = raw['requests']
    preceding = raw['steps'][index-1] if index is not None and index > 0 else None
    return dict(event=compact_event(event), service_step=index,
                outputs_before_decision=sum(bisect_left(row['token_times_s'], time_s) for row in rows),
                completed_before_decision=sum(row.get('completion_s', float('inf')) < time_s for row in rows),
                target_output_prefix_sha256=fingerprint(request['output_token_ids'][:event['generated_tokens']]),
                previous_completed_step=preceding,
                previous_step_is_not_a_decision_aligned_state=True)


def normalized_schedule_prefix(raw, index):
    external = {row['request_id']: row['external_id'] for row in raw['requests']}
    return [[(external.get(entry['request_id'], entry['request_id']),
              entry['count'], entry['start_computed'], entry['end_computed'],
              entry['known_tokens'], entry['generated_tokens'])
             for entry in step.get('scheduled', [])]
            for step in raw.get('scheduler_steps', [])[:index]]


def inline_analysis(raw):
    inline = raw.get('inline_host', {})
    rows = inline.get('records', [])
    requests = {row['request_id']: row for row in raw['requests']}
    events = {(row['request_id'], row['event']): row for row in raw.get('decisions', [])}
    intervals = [(row['wait_start_s'], row['wait_end_s']) for row in rows if row.get('waited')]
    unique_intervals = sorted(set(intervals))
    merged = union_intervals(intervals)
    diagnostics = []
    for row in rows:
        if not row.get('requested'):
            continue
        request = requests.get(row['request_id'])
        event = events.get((row['request_id'], row.get('selector_event')))
        diagnostics.append(dict(row, external_id=request.get('external_id') if request else None,
                                selector_decision=compact_event(event) if event else None,
                                successor=successor(raw, request, event.get('decision_s') if event else row['time_s'])))
    actual = [row for row in diagnostics if row.get('committed') and row.get('waited')]
    warnings = []
    if any(row.get('filtered') and not row.get('waited') for row in rows):
        warnings.append('Receive notification filtered without recorded native wait.')
    if any(row.get('committed') and not (row.get('waited') and row.get('filtered')) for row in rows):
        warnings.append('Committed inline LOAD lacks wait or normal ACK/filter evidence; inspect terminal failure.')
    return dict(enabled=bool(inline.get('enabled')), lookup_records=len(rows),
                stage_counts={key: sum(bool(row.get(key)) for row in rows)
                              for key in ('requested', 'committed', 'waited', 'filtered')},
                actual_waited_commits=len(actual), distinct_actual_requests=len({row['request_id'] for row in actual}),
                fallback_lookup_count=sum(not row.get('requested') for row in rows),
                fallback_reasons=dict(Counter(row.get('reason', 'missing') for row in rows if not row.get('requested'))),
                requested_not_committed_reasons=dict(Counter(row.get('reason', 'missing') for row in rows
                                                            if row.get('requested') and not row.get('committed'))),
                unique_native_wait_calls=len(unique_intervals),
                unique_wait_call_duration_s=stats(end-begin for begin, end in unique_intervals),
                wait_wall_union_s=sum(end-begin for begin, end in merged),
                requested_records=diagnostics,
                first_requested=min(diagnostics, key=lambda row: row['time_s'], default=None),
                first_actual_waited_commit=min(actual, key=lambda row: row['time_s'], default=None),
                warnings=warnings)


def readiness_analysis(raw, resources):
    """Version-specific CPU-clock bounds; no fabricated GPU end timestamp.

    Native source order: submit LOAD after schedule returns; worker polls its
    CUDA end event before reporting stats; selector records stats immediately
    before connector update; base scheduler then acknowledges finished_recving.
    The measured engine step returns after this acknowledgement. Inline has
    conditional dispatch followed by a worker wait, not an async-ready ACK.
    """
    steps, schedules = raw['steps'], raw['scheduler_steps']
    if len(schedules) < len(steps) or any(not step['start_s'] <= schedules[i]['time_s'] <= step['end_s']
                                       for i, step in enumerate(steps)):
        return dict(status='UNSUPPORTED_STEP_ALIGNMENT')
    cfg, = resources['group_config']
    token_bytes = resources['gpu_bytes']//resources['gpu_blocks']//cfg['tokens_per_block']
    issued, reports = {}, {}
    for event in raw['commits']:
        if event['external_tokens'] > 0:
            issued.setdefault(step_index(raw, event['allocation_s']), []).append(event)
    for row in raw['transfers']:
        if row.get('load', {}).get('bytes', 0):
            reports.setdefault(step_index(raw, row['time_s']), []).append(row)
    inline = {row['selector_event']: row for row in raw.get('inline_host', {}).get('records', [])
              if row.get('committed') and row.get('waited')}
    requests = {row['request_id']: row for row in raw['requests']}
    rows, unsupported = [], []
    for i, events in issued.items():
        reported = reports.get(i, [])
        # Anonymous aggregate LOAD stats are attributable here only because no
        # LOAD crosses either step boundary and exactly one LOAD is issued and
        # reported, with matching bytes. Do not pair equal sizes arbitrarily.
        valid = (len(events) == 1 and len(reported) == 1 and
                 len(reported[0]['load']['sizes']) == 1 and
                 reported[0]['load']['sizes'][0] == events[0]['external_tokens']*token_bytes and
                 (i == 0 or steps[i-1]['pending_load_jobs'] == 0) and
                 steps[i]['pending_load_jobs'] == 0)
        if not valid:
            unsupported.append(dict(issue_step=i, reason='LOAD association not uniquely supported'))
            continue
        event, report = events[0], reported[0]
        rid = event['request_id']
        scheduled = [(j, entry) for j in range(i, len(steps))
                     for entry in schedules[j]['scheduled']
                     if entry['request_id'] == rid and entry['count'] > 0]
        if not scheduled:
            unsupported.append(dict(issue_step=i, reason='no subsequent target dispatch'))
            continue
        j, entry = scheduled[0]
        wait = inline.get(event['event'])
        issue_schedule, dispatch = schedules[i]['time_s'], schedules[j]['time_s']
        assert issue_schedule <= report['time_s'] <= steps[i]['end_s']
        data_upper = wait['wait_end_s'] if wait else report['time_s']
        execution_lower = max(dispatch, wait['wait_end_s']) if wait else dispatch
        row = dict(event=event['event'], external_id=requests[rid]['external_id'],
                   mode='inline' if wait else 'async', issue_step=i, execution_batch_step=j,
                   skipped_intervening_schedule_steps=max(0, j-i-1),
                   load_bytes=report['load']['bytes'], reported_cuda_duration_s=report['load']['time'],
                   data_complete_interval_s=[issue_schedule, data_upper],
                   data_complete_exact_s=None, completion_report_entry_s=report['time_s'],
                   schedule_return_s=dispatch, actual_gpu_start_exact_s=None,
                   actual_gpu_start_interval_s=[execution_lower, steps[j]['end_s']],
                   first_scheduled_work=entry,
                   next_output=successor(raw, requests[rid], event['decision_s']))
        if wait:
            assert j == i
            row.update(scheduler_async_ready_interval_s=None,
                       ready_semantics='conditional same-batch dispatch; native worker wait gates forward',
                       wait_return_s=wait['wait_end_s'],
                       worker_notification_filtered_s=wait.get('filtered_s'))
        else:
            assert j > i
            row.update(scheduler_async_ready_interval_s=[report['time_s'], steps[i]['end_s']],
                       ready_semantics='finished_recving ID acknowledged, then promoted in a later schedule call',
                       ack_to_schedule_return_interval_s=[max(0, dispatch-steps[i]['end_s']),
                                                          dispatch-report['time_s']],
                       ack_to_actual_gpu_start_interval_s=[max(0, dispatch-steps[i]['end_s']),
                                                          steps[j]['end_s']-report['time_s']])
        rows.append(row)

    # Describe the largest stall of any actually preempted request. This is an
    # observed precommit interval, not a pure capacity-wait measurement.
    preempted = {event['request_id'] for event in raw['preemptions']}
    gaps = [(times[k]-times[k-1], rid, k, times[k-1], times[k])
            for rid, req in requests.items() if rid in preempted
            for times in [req['token_times_s']] for k in range(1, len(times))]
    largest = None
    if gaps:
        gap, rid, ordinal, start, end = max(gaps, key=lambda item: item[0])
        ps = [x for x in raw['preemptions'] if x['request_id'] == rid and start <= x['time_s'] < end]
        cs = [x for x in raw['commits'] if x['request_id'] == rid and start <= x['decision_s'] < end]
        ds = [x for x in raw['decisions'] if x['request_id'] == rid and start <= x['decision_s'] < end]
        largest = dict(external_id=requests[rid]['external_id'], gap_s=gap,
                       output_ordinal=ordinal, first_preemption_s=ps[0]['time_s'] if ps else None,
                       commits=[compact_event(x) for x in cs],
                       lookup_fallback_counts=dict(Counter(x['fallback'] for x in ds)),
                       preemption_to_first_commit_s=cs[0]['decision_s']-ps[0]['time_s'] if ps and cs else None,
                       first_commit_to_output_s=end-cs[0]['decision_s'] if cs else None,
                       precommit_interval_is_not_pure_capacity_wait=True)
    async_rows = [row for row in rows if row['mode'] == 'async']
    return dict(status='EXISTING_TRACE_INTERVAL_DIAGNOSIS', load_commits=sum(map(len, issued.values())),
                associated_events=len(rows), unsupported=unsupported, events=rows,
                async_ack_to_schedule_return_upper_s=stats(row['ack_to_schedule_return_interval_s'][1]
                                                          for row in async_rows),
                async_skipped_schedule_steps=dict(Counter(row['skipped_intervening_schedule_steps']
                                                          for row in async_rows)),
                largest_preempted_request_gap=largest,
                limitations=[
                    'CUDA duration has no CPU-clock origin: allocation plus CUDA duration is not a completion timestamp.',
                    'Completion report precedes scheduler ACK; its interval is not an exact observed ACK timestamp.',
                    'Schedule return records dispatch, not actual target GPU execution; the latter is only bounded by step completion.',
                    'Inline has no asynchronous readiness ACK before execution. Its worker notification filter is not a ready timestamp.',
                    'Data-to-ACK and ACK-to-GPU-start contributions cannot be precisely apportioned from these intervals.',
                    'No new execution, performance comparison, exact capacity-wait attribution, or service-gain bound is produced.',
                ])


def cell_analysis(path, group):
    raw = read_json(path)
    base, requests = analyze_cell(path, group)
    config = base['config']
    inputs = read_json(path.parent/'inputs.json', [])
    planned = len(inputs) or config.get('requests', len(requests))
    completed = base['completed_requests']
    all_completed = completed == planned and len(requests) == planned
    statuses = Counter(row.get('finish_reason') or 'unfinished' for row in raw['requests'])
    # No engine rejection interface is used by this harness. Failures refer to
    # terminal cell failures; their outstanding requests are not labelled completed.
    terminal = read_json(path.parent/'status.json', {})
    metrics = base['request_metrics']
    by_work = {row['external_id']: row for row in base['scheduled_position_counts']['by_request']}
    output_steps = output_step_sequences(raw)
    for row in metrics:
        row['max_generation_gap_s'] = row['token_gap_s']['max']
        row['scheduled_work'] = by_work.get(row['external_id'])
        row['preemptions'] = base['preemptions_by_request'].get(row['request_id'], 0)
        row['completion_step_zero_based'] = step_index(raw, row['completion_s'])
        row['output_step_sequence_sha256'] = fingerprint(output_steps[row['external_id']])
    selected = ('cell', 'policy', 'config', 'status', 'output_tokens', 'output_work',
                'completion_latency_s', 'ttft_s', 'admission_lag_s', 'makespan_s',
                'all_complete_s', 'service_and_drain_s', 'output_tokens_per_s', 'steps_s',
                'preemptions', 'preempted_requests', 'scheduled_native_tokens',
                'scheduled_position_counts', 'transfers', 'decision_count', 'decision_actions',
                'decision_fallbacks', 'committed_count', 'committed_actions', 'committed_fallbacks',
                'eligible_commits', 'capacity_pressure', 'prefix_consistency_reported', 'warnings')
    result = {key: base[key] for key in selected}
    result.update(host_load_mode=config.get('host_load_mode', 'async'),
                  measured_service_steps=len(raw.get('steps', [])),
                  scheduler_steps_including_drain=len(raw.get('scheduler_steps', [])),
                  denominator=dict(planned_external_arrivals=planned, admitted=len(requests),
                                   completed=completed, unfinished_admitted=len(requests)-completed,
                                   not_admitted=planned-len(requests), rejected=0,
                                   cell_failed=terminal.get('status') != 'COMPLETE',
                                   cell_timeout='Timeout' in terminal.get('error', ''),
                                   finish_reasons=dict(statuses)),
                  all_external_arrivals_completed=all_completed,
                  primary_mean_external_completion_s=base['completion_latency_s']['mean'] if all_completed else None,
                  completion_statistics_are_completed_only=not all_completed,
                  request_max_generation_gap_s=stats(row['max_generation_gap_s'] for row in metrics),
                  last_arrival_to_last_completion_s=(max(row['completion_s'] for row in raw['requests'])-
                                                   max(row['arrival_s'] for row in raw['requests'])) if all_completed else None,
                  inline_host=inline_analysis(raw),
                  readiness=readiness_analysis(raw, read_json(path.parent/'resources.json')),
                  request_metrics=metrics)
    return result, base, raw, requests


def first_fork(left, right):
    """Match the first actual inline action, never choose a favorable event later."""
    record = right[0]['inline_host']['first_actual_waited_commit']
    if record is None:
        return dict(status='NO_ACTUAL_INLINE_WAITED_COMMIT')
    inline_raw, async_raw = right[2], left[2]
    inline_req = right[3][record['external_id']]
    async_req = left[3].get(record['external_id'])
    event = next((row for row in inline_raw.get('decisions', [])
                  if row['request_id'] == record['request_id'] and row['event'] == record['selector_event']), None)
    if event is None or async_req is None:
        return dict(status='MISSING_REQUEST_OR_SELECTOR_EVENT', inline_record=record)
    keys = ('known_tokens', 'generated_tokens', 'host_hit_tokens', 'preemptions', 'prefix_sha256')
    candidates = [row for row in async_raw.get('commits', [])
                  if row['request_id'] == async_req['request_id']
                  and all(row.get(key) == event.get(key) for key in keys)]
    result = dict(status='MATCHED_TARGET_STATE_ONLY' if len(candidates) == 1 else 'NO_UNIQUE_MATCHED_ASYNC_COMMIT',
                  matching_keys=list(keys), matched_async_commits=len(candidates),
                  inline_record=record, inline_prestate=decision_state(inline_raw, inline_req, event))
    if len(candidates) != 1:
        return result
    baseline = candidates[0]
    result['async_prestate'] = decision_state(async_raw, async_req, baseline)
    result['async_successor'] = successor(async_raw, async_req, baseline['decision_s'])
    ai, bi = result['async_prestate']['service_step'], result['inline_prestate']['service_step']
    a, b = normalized_schedule_prefix(async_raw, ai), normalized_schedule_prefix(inline_raw, bi)
    first_difference = next((index for index, pair in enumerate(zip(a, b)) if pair[0] != pair[1]),
                            min(len(a), len(b)) if len(a) != len(b) else None)
    result['prior_schedule_comparison'] = dict(async_steps=len(a), inline_steps=len(b), identical=a == b,
                                               first_different_step=first_difference,
                                               async_sha256=fingerprint(a), inline_sha256=fingerprint(b))
    result['decision_state_equal'] = {key: baseline.get(key) == event.get(key) for key in
                                    ('free_blocks', 'full_required_blocks', 'reserved_blocks',
                                     'watermark_blocks', 'pending_load_jobs', 'pending_transfer_jobs', 'running', 'waiting')}
    result['interpretation'] = ('Target and observed-state matching does not establish identical GPU/Host state. '
                                'Unmatched prefixes preclude a strict same-state causal interpretation.')
    return result


def completion_area_decomposition(left, right, first_fork_result):
    """Exact retrospective arithmetic, not an executable counterfactual.

    With t[-1]=0, sum completion timestamps = sum_j W[j]*(t[j]-t[j-1]),
    where W[j] counts requests whose completion step is >= j. External
    arrivals cancel between these paired arms. End-to-end intervals include
    inter-step client/harness time; raw engine-step durations alone do not.
    """
    a, b = left[0], right[0]
    if not a['all_external_arrivals_completed'] or not b['all_external_arrivals_completed']:
        return dict(status='UNSUPPORTED_INCOMPLETE_REQUESTS')
    ra = {r['external_id']: r for r in a['request_metrics']}
    rb = {r['external_id']: r for r in b['request_metrics']}
    if ra.keys() != rb.keys() or any(ra[k]['arrival_s'] != rb[k]['arrival_s'] for k in ra):
        return dict(status='UNSUPPORTED_DIFFERENT_EXTERNAL_ARRIVALS')
    ends_a = [step['end_s'] for step in left[2]['steps']]
    ends_b = [step['end_s'] for step in right[2]['steps']]
    if len(ends_a) != len(ends_b):
        return dict(status='UNSUPPORTED_DIFFERENT_STEP_COUNTS')
    def vectors(metrics, ends):
        completed = Counter(r['completion_step_zero_based'] for r in metrics.values())
        remaining, weights, dt, previous = len(metrics), [], [], 0.0
        for j, end in enumerate(ends):
            weights.append(remaining)
            dt.append(end-previous)
            previous = end
            remaining -= completed[j]
        assert remaining == 0
        actual = sum(r['completion_s'] for r in metrics.values())
        reconstructed = sum(w*d for w, d in zip(weights, dt))
        assert abs(actual-reconstructed) < 1e-6
        return weights, dt
    wa, da = vectors(ra, ends_a)
    wb, db = vectors(rb, ends_b)
    n = len(ra)
    timing_a = sum(w*(y-x) for w, x, y in zip(wa, da, db))/n
    timing_b = sum(w*(y-x) for w, x, y in zip(wb, da, db))/n
    progress_a = sum((v-w)*dt for v, w, dt in zip(wb, wa, da))/n
    progress_b = sum((v-w)*dt for v, w, dt in zip(wb, wa, db))/n
    observed = b['primary_mean_external_completion_s']-a['primary_mean_external_completion_s']
    assert abs(timing_a+progress_b-observed) < 1e-8
    assert abs(timing_b+progress_a-observed) < 1e-8
    timing = [(w+v)*.5*(y-x)/n for w, v, x, y in zip(wa, wb, da, db)]
    progress = [(v-w)*.5*(x+y)/n for w, v, x, y in zip(wa, wb, da, db)]
    first = b['inline_host']['first_actual_waited_commit']
    cut = first['step_index'] if first else None
    matched = first_fork_result.get('prior_schedule_comparison', {}).get('identical')
    return dict(status='EXACT_OBSERVED_ACCOUNTING_NOT_CAUSAL', requests=n, steps=len(da),
                observed_primary_delta_s=observed,
                timing_using_async_completion_steps_s=timing_a,
                timing_using_inline_completion_steps_s=timing_b,
                completion_step_change_using_async_clock_s=progress_a,
                completion_step_change_using_inline_clock_s=progress_b,
                symmetric_timing_component_s=sum(timing),
                symmetric_completion_step_component_s=sum(progress),
                first_actual_inline_step=cut, matched_prior_schedule=matched,
                symmetric_timing_before_first_actual_inline_step_s=sum(timing[:cut]) if cut is not None else None,
                symmetric_timing_from_first_actual_inline_step_s=sum(timing[cut:]) if cut is not None else None,
                limitations=[
                    'Crossing one arm clock with another completion-step assignment is algebra only, not a runnable policy or benefit bound.',
                    'Symmetric components average the two exact decomposition orders rather than selecting a favorable reference.',
                    'Timing includes contention, changed work, observation overhead and drift; this calculation does not identify their causal shares.',
                    'Before the first actual wait is not an untreated control: adapter overhead may already exist. No timing component is subtracted from reported performance.',
                    'This diagnoses the observed pair; a future policy may change other requests and cannot inherit its small completion-step component as an upper bound.',
                ])


def compare(left, right):
    a, b = left[0], right[0]
    token_check = compare_tokens(left[1], right[1], left[3], right[3])
    ignored = {'out', 'policy', 'host_load_mode'}
    fixed_a = {key: value for key, value in a['config'].items() if key not in ignored}
    fixed_b = {key: value for key, value in b['config'].items() if key not in ignored}
    fixed_differences = {key: [fixed_a.get(key), fixed_b.get(key)] for key in fixed_a.keys() | fixed_b.keys()
                         if fixed_a.get(key) != fixed_b.get(key)}
    # analyze.py predates host_load_mode. Its fixed-config mismatch is expected
    # for this action; report exact remaining differences rather than hide them.
    token_check['matching_metadata']['fixed_config'] = None
    affected = {row['external_id'] for row in b['inline_host']['requested_records'] if row.get('waited')}
    by_a = {row['external_id']: row for row in a['request_metrics']}
    steps_a, steps_b = output_step_sequences(left[2]), output_step_sequences(right[2])
    rows = []
    for row in b['request_metrics']:
        previous = by_a.get(row['external_id'])
        if previous is None:
            continue
        delta = lambda key: (row[key]-previous[key] if row.get(key) is not None and previous.get(key) is not None else None)
        sa, sb = steps_a[row['external_id']], steps_b[row['external_id']]
        changed = [(index, y-x) for index, (x, y) in enumerate(zip(sa, sb)) if x != y]
        rows.append(dict(external_id=row['external_id'], received_inline_wait=row['external_id'] in affected,
                         completion_delta_s=delta('completion_latency_s'), ttft_delta_s=delta('ttft_s'),
                         max_generation_gap_delta_s=delta('max_generation_gap_s'),
                         output_tokens_delta=row['output_tokens']-previous['output_tokens'],
                         async_completion_step=previous['completion_step_zero_based'],
                         inline_completion_step=row['completion_step_zero_based'],
                         completion_step_delta=delta('completion_step_zero_based'),
                         output_step_sequence_changed=sa != sb,
                         changed_output_ordinals=len(changed),
                         unmatched_output_ordinals=abs(len(sa)-len(sb)),
                         first_changed_output_ordinal=changed[0][0] if changed else None,
                         last_changed_output_ordinal=changed[-1][0] if changed else None,
                         changed_output_step_deltas=stats(value for _, value in changed),
                         repeated_positions_delta=(row['scheduled_work']['repeated_scheduled_positions']-
                                                   previous['scheduled_work']['repeated_scheduled_positions'])
                         if row['scheduled_work'] and previous['scheduled_work'] else None))
    def group_metrics(members):
        values = [row['completion_delta_s'] for row in members if row['completion_delta_s'] is not None]
        return dict(requests=len(members), completion_delta_s=stats(values),
                    completion_improved=sum(value < 0 for value in values),
                    completion_worsened=sum(value > 0 for value in values),
                    max_generation_gap_delta_s=stats(row['max_generation_gap_delta_s'] for row in members))
    def difference(key):
        return b[key]-a[key] if a[key] is not None and b[key] is not None else None
    changed_work = [dict(external_id=row['external_id'], received_inline_wait=row['received_inline_wait'],
                         repeated_positions_delta=row['repeated_positions_delta'])
                    for row in rows if row['repeated_positions_delta'] not in (None, 0)]
    trajectories = dict(
        measured_service_steps=[a['measured_service_steps'], b['measured_service_steps']],
        scheduler_steps_including_drain=[a['scheduler_steps_including_drain'], b['scheduler_steps_including_drain']],
        completion_step_changed_requests=sum(row['completion_step_delta'] not in (None, 0) for row in rows),
        output_step_sequence_changed_requests=sum(row['output_step_sequence_changed'] for row in rows),
        changed_peer_external_ids=[row['external_id'] for row in rows
                                   if not row['received_inline_wait'] and row['output_step_sequence_changed']],
        repeated_work_delta_total=sum(row['repeated_positions_delta'] or 0 for row in rows),
        repeated_work_delta_by_changed_request=changed_work,
        interpretation='Zero-based steps and output ordinals. Emission positions compare execution trajectories, not token content or quality. Unchanged peer steps can still have different wall latency.')
    fork = first_fork(left, right)
    return dict(async_cell=a['cell'], inline_cell=b['cell'], delta_convention='inline minus async',
                fixed_config_differences_excluding_assigned_action=fixed_differences,
                output_consistency=token_check,
                primary_mean_completion_delta_s=difference('primary_mean_external_completion_s'),
                makespan_delta_s=difference('makespan_s'), service_and_drain_delta_s=difference('service_and_drain_s'),
                full_service=group_metrics(rows),
                actual_inline_recipients=group_metrics([row for row in rows if row['received_inline_wait']]),
                other_requests=group_metrics([row for row in rows if not row['received_inline_wait']]),
                execution_trajectory=trajectories,
                completion_area=completion_area_decomposition(left, right, fork),
                per_request_deltas=rows, first_actual_fork=fork)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    group = args.group.resolve()
    status = read_json(group/'status.json', {})
    if status.get('status') not in ('COMPLETE', 'FAILED', 'ABORT_LOCK_BUSY'):
        parser.error('group has no recognized terminal status; refusing live/unknown group analysis')
    files = sorted(group.rglob('raw.json'))
    files += [path for path in sorted(group.rglob('partial_raw.json')) if not (path.parent/'raw.json').exists()]
    cells, errors = [], []
    for path in files:
        cell_status = read_json(path.parent/'status.json', {}).get('status')
        if cell_status not in ('COMPLETE', 'FAILED'):
            errors.append(dict(path=str(path), error='nonterminal cell status; not read'))
            continue
        try:
            cells.append(cell_analysis(path, group))
        except (KeyError, TypeError, ValueError, OSError) as error:
            errors.append(dict(path=str(path), error=repr(error)))
    comparisons = []
    for left, right in combinations(cells, 2):
        if left[0]['host_load_mode'] == 'inline':
            left, right = right, left
        if (left[0]['host_load_mode'], right[0]['host_load_mode']) == ('async', 'inline'):
            comparisons.append(compare(left, right))
    result = dict(group=str(group), group_status=status, cells=[cell[0] for cell in cells],
                  async_inline_comparisons=comparisons, analysis_errors=errors,
                  limitations=[
                      'Exploratory run-level results; request/token/event counts are not independent run replicates.',
                      'Primary mean uses every external arrival and is null if any is incomplete; no selected-success mean substitutes for it.',
                      'Missing or failed episodes remain visible in group status; no raw means no measured result.',
                      'Wait intervals are deduplicated/unioned, never summed per request. They are already included in service latency.',
                      'Reported CUDA transfer durations and scheduled token positions are not additive wall-clock costs.',
                      'First-fork matching is diagnostic, not a same-state oracle or an online performance gain.',
                      'Peer deltas include all subsequent trajectory changes; they are not causal effects of only the first intervention.',
                      'Fixed output counts and token consistency do not establish natural-task quality or equal compute.',
                  ])
    out = args.out or group/'inline_summary.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(out), cells=len(cells), comparisons=len(comparisons), errors=errors), ensure_ascii=False))
    return int(bool(errors) or not cells)


if __name__ == '__main__':
    raise SystemExit(main())
