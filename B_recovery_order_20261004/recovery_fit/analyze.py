#!/usr/bin/env python3
"""Reuse frozen request metrics; join one recovery-fit decision to native work."""
import argparse
import bisect
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analyze import read


def helpers():
    path = BASE/'normal_capacity/analyze_group.py'
    source = path.read_text(); old = '(native|age|flush_first)'
    if source.count(old) != 1:
        raise RuntimeError('Cell-name adapter boundary changed')
    namespace = dict(__name__='recovery_fit_group', __file__=str(path))
    exec(compile(source.replace(old, '(native|fit_once)'), str(path), 'exec'), namespace)
    modules = []
    for name in ('capacity_handoff/analyze_tail.py', 'tail_reservation/summarize_runs.py'):
        spec = importlib.util.spec_from_file_location('fit_'+Path(name).stem, BASE/name)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        modules.append(module)
    return namespace['analyze_group'], modules[0].analyze, modules[1].summarize_cell


def optional(directory, name):
    path = next((directory/(name+suffix) for suffix in ('.json', '.json.gz')
                 if (directory/(name+suffix)).exists()), None)
    return read(path) if path else None


def request_evidence(rid, stamp, raw, cell, allocations):
    """Within-run observations, bounded by the containing recovery episode."""
    origin = raw['measurement_origin_perf_counter_s']
    source = raw.get('internal_to_source', {}).get(rid)
    result = dict(internal_request=rid, source_request=source)
    episodes = [e for e in (allocations or {}).get('episodes', [])
                if e['request'] == rid and e['begin_host_perf_s'] <= stamp <= e['end_host_perf_s']]
    result['containing_recovery_episode_count'] = len(episodes) if allocations is not None else None
    episode = episodes[0] if len(episodes) == 1 else None
    result['allocation_observation'] = 'AVAILABLE' if episode else 'UNAVAILABLE_OR_AMBIGUOUS_EPISODE'
    result['episode'] = episode['episode'] if episode else None
    attempts = [a for a in episode['attempts'] if a['record']['begin_host_perf_s'] >= stamp] if episode else None
    rows = []
    for attempt in attempts or []:
        record = attempt['record']; arguments = record['arguments']
        rows.append(dict(record_index=attempt['record_index'], begin_s=attempt['begin_s'],
            end_s=attempt['end_s'], decision_to_begin_s=record['begin_host_perf_s']-stamp,
            outcome=attempt['outcome'], phase=attempt['phase'],
            delay_cache_blocks=arguments.get('delay_cache_blocks'),
            num_external_computed_tokens=arguments.get('num_external_computed_tokens'),
            num_new_tokens=arguments.get('num_new_tokens'), blocks=attempt['blocks']))
    successes = [a for a in rows if a['outcome'] == 'success']
    result.update(native_allocation_attempts=rows if attempts is not None else None,
        first_native_allocation_attempt=next(iter(rows), None),
        first_successful_native_allocation=next(iter(successes), None),
        first_successful_non_delayed_allocation=next((a for a in successes if a['delay_cache_blocks'] is False), None))
    stop = episode['end_s'] if episode else None
    jobs = [j for j in cell.get('jobs', []) if source is not None and j['request'] == source
            and j['is_store'] is False and j['ready_s'] is not None
            and stamp-origin <= j['ready_s'] <= stop] if stop is not None else None
    schedule = episode.get('resumed_schedule_record') if episode else None
    schedule_s = schedule['host_perf_s']-origin if schedule and schedule['host_perf_s'] >= stamp else None
    request = next((r for r in raw['requests'] if source is not None and r['request_id'] == source), None)
    outputs = request.get('token_times_s', []) if request else []
    index = bisect.bisect_right(outputs, stamp-origin)
    next_output = outputs[index] if index < len(outputs) else None
    recovery = episode.get('request_recovery') if episode else None
    result.update(load_jobs_after_decision=jobs,
        load_ready_count=len(jobs) if jobs is not None else None,
        load_submitted_count=sum(j['submit_begin_s'] is not None for j in jobs) if jobs is not None else None,
        load_host_completed_count=sum(j['job_completed_s'] is not None for j in jobs) if jobs is not None else None,
        load_acknowledged_count=sum(j['ack_retired_s'] is not None for j in jobs) if jobs is not None else None,
        first_resumed_schedule_plan_s=schedule_s,
        decision_to_schedule_plan_s=schedule_s-(stamp-origin) if schedule_s is not None else None,
        request_status=request.get('status') if request else 'missing',
        next_output_after_decision_s=next_output,
        decision_to_next_output_s=next_output-(stamp-origin) if next_output is not None else None,
        decision_to_output_wait_lower_bound_s=(next_output if next_output is not None else raw['observation_end_s'])-(stamp-origin),
        recovery_to_next_output=recovery,
        observed_request_metrics=next((r for r in cell.get('per_request', [])
                                      if source is not None and r['request'] == source), None))
    return result


def actions(data, raw, cell, allocations):
    if data is None or raw is None:
        return dict(status='UNAVAILABLE', missing=[name for name, value in
            (('recovery-fit', data), ('raw', raw)) if value is None], raw_action_record=data)
    origin = raw['measurement_origin_perf_counter_s']; mapping = raw.get('internal_to_source', {})
    events = data.get('events'); missing = []; failures = []; rows = []
    if not isinstance(events, list):
        events = []; missing.append('events')
    decisions = [e for e in events if e.get('kind') == 'first_fit_opportunity']
    if len(decisions) > 1: failures.append('more_than_one_decision')
    if data.get('mode') != cell['mode']: failures.append('mode_mismatch')
    if data.get('status') != 'UNINSTALLED': missing.append('uninstalled_observation')
    for event_index, event in enumerate(events):
        if event.get('kind') != 'first_fit_opportunity': continue
        required = ('host_perf_s', 'baseline_head', 'candidate_head', 'final_head',
                    'queue_changed', 'waiting_before', 'waiting_after')
        absent = [k for k in required if k not in event]
        missing.extend('event.'+k for k in absent)
        if absent: continue
        before, after = event['waiting_before'], event['waiting_after']
        if not isinstance(before, list) or not isinstance(after, list):
            missing.append('waiting_lists'); continue
        changed = before != after; declared = event['queue_changed'] is True
        if changed != declared: failures.append('queue_change_flag_mismatch')
        if Counter(before) != Counter(after): failures.append('waiting_members_changed')
        if (before[0] if before else None) != event['baseline_head']: failures.append('baseline_head_mismatch')
        if (after[0] if after else None) != event['final_head']: failures.append('final_head_mismatch')
        if cell['mode'] == 'native' and (changed or declared): failures.append('native_changed_queue')
        if changed and (not after or after[0] != event['candidate_head']): failures.append('candidate_not_final_head')
        candidate = event['candidate_head']
        passed = before[:before.index(candidate)] if candidate in before else None
        if passed is None: failures.append('candidate_missing_from_waiting')
        ids = list(dict.fromkeys([event['baseline_head'], candidate]+(passed or [])))
        stamp = event['host_perf_s']
        requests = [request_evidence(rid, stamp, raw, cell, allocations) for rid in ids]
        if any(r['source_request'] is None for r in requests): missing.append('request_source_mapping')
        rows.append(dict(event_index=event_index, decision_s=stamp-origin,
            queue_changed_reported=declared, queue_order_changed_observed=changed,
            actual_candidate_intervention=cell['mode'] == 'fit_once' and declared and changed,
            baseline_head=dict(internal_request=event['baseline_head'], source_request=mapping.get(event['baseline_head'])),
            candidate_head=dict(internal_request=candidate, source_request=mapping.get(candidate)),
            final_head=dict(internal_request=event['final_head'], source_request=mapping.get(event['final_head'])),
            passed_requests=[dict(internal_request=rid, source_request=mapping.get(rid)) for rid in passed] if passed is not None else None,
            action_host_duration_s=(event['return_host_perf_s']-event['action_begin_host_perf_s']
                if all(k in event for k in ('action_begin_host_perf_s', 'return_host_perf_s')) else None),
            request_evidence=requests, raw_decision=event))
    errors = [e for e in events if 'error' in e.get('kind', '').lower() or e.get('error') is not None]
    if errors or data.get('status') == 'ERROR': failures.append('policy_error')
    if data.get('action_count') is not None and data['action_count'] != sum(
            r['actual_candidate_intervention'] for r in rows):
        failures.append('action_count_mismatch')
    return dict(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        failed_checks=sorted(set(failures)), missing=sorted(set(missing)),
        policy_status=data.get('status'), policy_outcome=data.get('outcome'),
        suggested_decisions=len(decisions), native_shadow_decisions=len(decisions) if cell['mode'] == 'native' else 0,
        queue_change_reported_count=sum(e.get('queue_changed') is True for e in decisions),
        queue_order_change_observed_count=sum(r['queue_order_changed_observed'] for r in rows),
        actual_candidate_intervention_count=sum(r['actual_candidate_intervention'] for r in rows),
        callback_calls=data.get('callback_calls'), skip_counts=data.get('skip_counts'),
        rows=rows, error_events=errors, raw_action_record=data)


def analyze_session(session):
    session = session.resolve(); group, tail, summarize = helpers(); result = group(session)
    for cell in result['cells']:
        directory = Path(cell['directory'])
        raw, data, observer, recovery = [optional(directory, name) for name in
                                      ('raw', 'recovery-fit', 'capacity-handoff', 'recovery-order')]
        allocation = tail(raw, observer, recovery) if all(x is not None for x in (raw, observer, recovery)) else None
        if allocation is not None:
            origin = raw['measurement_origin_perf_counter_s']
            for episode in allocation['episodes']:
                preempt = episode['preempt_record']
                joined = [r for r in cell.get('recovery_events', []) if r['request'] == episode['source_request']
                    and preempt['begin_host_perf_s'] <= r['demand_host_s']+origin <= preempt['end_host_perf_s']]
                episode.update(request_recovery=joined[0] if len(joined) == 1 else None,
                               request_recovery_join_count=len(joined))
        cell['recovery_allocations'] = allocation
        cell['recovery_fit_actions'] = actions(data, raw, cell, allocation)
        compact = summarize(cell, session/'recovery-fit-metrics.json')
        cell['run_summary'] = {k: compact[k] for k in ('phase_times', 'fixed1024_contract', 'actual_outputs_known',
            'output_count_missing_requests', 'stop_reason_counts', 'actual_output_count_distribution',
            'missing_planned_request_rows', 'recovery', 'per_request')}
    by_path = {c['directory']: c for c in result['cells']}
    for comparison in result['comparisons']:
        pair = [by_path.get(comparison.get(k)) for k in ('candidate', 'native')]
        comparison['fixed1024_contracts'] = {k: c['run_summary']['fixed1024_contract'] if c else None
            for k, c in zip(('candidate', 'native'), pair)}
        comparison['copy_work_delta_candidate_minus_native'] = {
            direction: {key: pair[0]['copy_work'][direction][key]-pair[1]['copy_work'][direction][key]
                for key in ('jobs', 'completed', 'bytes', 'missing_bytes', 'gpu_elapsed_sum_s', 'missing_gpu_elapsed')}
            for direction in ('load', 'store')} if all(c and 'copy_work' in c for c in pair) else None
    modes = [c['mode'] for c in result['cells']]
    result['execution_layout'] = dict(cell_count=len(modes), modes=modes,
        expected_abba=['native', 'fit_once', 'fit_once', 'native'],
        complete_abba=modes == ['native', 'fit_once', 'fit_once', 'native'],
        comparison_count=len(result['comparisons']))
    result['analyzer_sources_sha256'] = {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest()
        for name in ('analyze.py', 'normal_capacity/analyze_group.py', 'capacity_handoff/analyze_tail.py',
                     'tail_reservation/summarize_runs.py', 'recovery_fit/analyze.py')}
    result['recovery_fit_semantics'] = (
        'Native first-fit suggestions are shadow observations, never interventions. A queue-order change is '
        'separate from subsequent native allocation, LOAD ready/submission/host completion/ACK, scheduler plan '
        'and client output. Request joins use internal-to-source IDs and the containing within-run recovery '
        'episode; later LOAD jobs are restricted to that episode. Allocation success may only allocate the '
        'asynchronous restore prefix; non-delayed allocation is reported separately. Neither success nor a '
        'scheduler plan is GPU execution. Decision-to-next-output can include a later recovery episode; the '
        'original preempt-to-next-output metric and censoring are retained separately. All timestamps are '
        'host perf_counter observations relative to raw origin; GPU copy durations are not absolute times. '
        'All-request metrics, failures, unfinished requests, fixed research SLO and copy work are inherited. '
        'Two ABBA run contrasts are descriptive; same request IDs do not establish matched runtime states.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = analyze_session(args.session)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), layout=result['execution_layout'],
        cells=[dict(directory=c['directory'], status=c['recovery_fit_actions']['status'],
                    interventions=c['recovery_fit_actions'].get('actual_candidate_intervention_count'))
               for c in result['cells']])))


if __name__ == '__main__':
    main()
