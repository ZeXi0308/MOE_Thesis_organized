"""Inspect the optional native-transfer startup observer on an already analyzed group.

Usage: python analyze_transfer_start.py COMPLETE_GROUP [--out PATH]
No engine imports, re-analysis of full service, or cross-run performance inference.
Missing/incomplete groups return UNRUN without opening raw episode files.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import analyze as base


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def distribution(values):
    # Do not present summed overlapping call durations as a wall-clock phase.
    result = base.stats(values)
    result.pop('sum')
    return result


def inspect_cell(raw, summary):
    steps = raw['steps']
    complete, drain_end = raw['all_complete_s'], raw['service_and_drain_s']
    if not steps or any(not (finite(s.get('start_s')) and finite(s.get('end_s'))
                            and 0 <= s['start_s'] < s['end_s']) for s in steps):
        raise ValueError('invalid formal step intervals')
    if any(a['end_s'] > b['start_s'] for a, b in zip(steps, steps[1:])):
        raise ValueError('overlapping formal steps')
    if not (finite(complete) and finite(drain_end) and steps[-1]['end_s'] <= complete <= drain_end):
        raise ValueError('invalid service/drain boundary')
    starts = [s['start_s'] for s in steps]

    def step_at(timestamp):
        index = bisect_right(starts, timestamp) - 1 if finite(timestamp) else -1
        return index if index >= 0 and timestamp <= steps[index]['end_s'] else None

    schedules, drain_schedules = {}, []
    for schedule in raw['scheduler_steps']:
        t = schedule['time_s']
        index = step_at(t)
        if index is not None:
            if index in schedules:
                raise ValueError('duplicate scheduler return in a formal step')
            schedules[index] = schedule
        elif finite(t) and complete <= t <= drain_end and schedule.get('scheduled') == []:
            drain_schedules.append(t)
        else:
            raise ValueError('scheduler return outside formal steps or empty drain')
    if len(schedules) != len(steps):
        raise ValueError('formal step missing scheduler return')

    calls, by_step, drain_calls = [], defaultdict(list), []
    for index, call in enumerate(raw.get('native_transfer_start_calls', [])):
        start, end = call.get('start_s'), call.get('end_s')
        if not (finite(start) and finite(end) and 0 <= start <= end):
            raise ValueError(f'call {index}: invalid interval')
        if call.get('returned_normally') is not True:
            raise ValueError(f'call {index}: native startup did not return normally')
        row = dict(call_index=index, **call, wall_s=end-start)
        step = call.get('step_index')
        if call.get('phase') == 'service':
            if type(step) is not int or not 0 <= step < len(steps):
                raise ValueError(f'call {index}: invalid service step label')
            if not steps[step]['start_s'] <= start <= end <= steps[step]['end_s']:
                raise ValueError(f'call {index}: outside its labeled service step')
            by_step[step].append(row)
        elif call.get('phase') == 'drain':
            if step is not None or not complete <= start <= end <= drain_end:
                raise ValueError(f'call {index}: invalid drain label or interval')
            drain_calls.append(row)
        else:
            raise ValueError(f'call {index}: unrecognized phase')
        calls.append(row)

    def step_detail(index):
        step, schedule = steps[index], schedules[index]
        return dict(step_index=index, start_s=step['start_s'], end_s=step['end_s'],
            wall_s=step['end_s']-step['start_s'], scheduler_return_s=schedule['time_s'],
            start_to_scheduler_return_s=schedule['time_s']-step['start_s'],
            scheduler_return_to_step_end_s=step['end_s']-schedule['time_s'],
            scheduled_requests=len(schedule['scheduled']),
            process_cpu_s=step.get('process_cpu_s'), driver_thread_cpu_s=step.get('driver_thread_cpu_s'),
            native_startup_calls=by_step[index],
            load_reports_in_step=[dict(report_time_s=t['time_s'], **t['load'])
                for t in raw.get('transfers', []) if t.get('load', {}).get('bytes', 0) > 0
                and step['start_s'] <= t['time_s'] <= step['end_s']])

    eligible = [c for c in summary['committed_event_diagnostics'] if c.get('eligible')]
    first = min(eligible, key=lambda c: c['decision_s']) if eligible else None
    first_result = dict(status='NO_ELIGIBLE_COMMIT')
    if first is not None:
        decision_step = step_at(first['decision_s'])
        output_step = step_at(first.get('next_output_s'))
        if decision_step is None:
            raise ValueError('first eligible commit decision lies outside formal steps')
        if not first.get('censored') and output_step is None:
            raise ValueError('first eligible commit next output lies outside formal steps')
        fields = ('event', 'external_id', 'request_id', 'actual_action', 'known_tokens',
                  'generated_tokens', 'host_hit_tokens', 'eligible', 'joint_capacity',
                  'fallback', 'decision_s', 'allocation_s', 'next_output_s',
                  'decision_to_next_output_s', 'next_output_token_index', 'censored',
                  'censor_reason', 'later_commit_before_next_output', 'commits_sharing_next_output')
        first_result = dict(status='OBSERVED' if output_step is not None else 'CENSORED',
            commit={k: first.get(k) for k in fields}, decision_step_index=decision_step,
            next_output_step_index=output_step,
            steps=[step_detail(i) for i in sorted({decision_step, output_step}-{None})])

    uncovered = [i for i in range(len(steps)) if not by_step[i]]
    uncovered_nonempty = [i for i in uncovered if schedules[i]['scheduled']]
    status = ('COVERAGE_MISSING' if not calls else
              'PARTIAL_COVERAGE' if uncovered_nonempty else 'VALIDATED_OBSERVATION')
    fields = ('requests', 'completed_requests', 'all_requests_completed', 'makespan_s',
              'all_complete_s', 'service_and_drain_s', 'completion_latency_s', 'ttft_s',
              'token_gap_s', 'output_tokens', 'output_tokens_per_s', 'scheduled_native_tokens',
              'eligible_commits', 'committed_actions', 'committed_fallback_count')
    return dict(cell=summary['cell'], status=status,
        native_transfer_probe_enabled=summary.get('config', {}).get('native_transfer_probe'),
        coverage=dict(formal_steps=len(steps), service_calls=sum(map(len, by_step.values())),
            formal_steps_with_calls=len(steps)-len(uncovered),
            call_count_by_formal_step=[len(by_step[i]) for i in range(len(steps))],
            uncovered_formal_steps=uncovered, uncovered_nonempty_steps=uncovered_nonempty,
            drain_calls=len(drain_calls), drain_scheduler_returns=len(drain_schedules),
            drain_start_s=complete, drain_end_s=drain_end,
            normal_returns=len(calls), all_call_intervals_match_labels=True),
        call_wall_s=dict(all=distribution(c['wall_s'] for c in calls),
            service=distribution(c['wall_s'] for c in calls if c['phase'] == 'service'),
            drain=distribution(c['wall_s'] for c in drain_calls)),
        largest_calls=sorted(calls, key=lambda c: c['wall_s'], reverse=True)[:5],
        first_eligible_commit=first_result,
        full_service_from_existing_summary={k: summary.get(k) for k in fields},
        per_request_max_generation_gap_s=distribution(
            r['token_gap_s']['max'] for r in summary.get('request_metrics', [])),
        whole_cell_load_cuda_reported=summary['transfers'].get('load'),
        full_service_pointer=dict(file='summary.json', cell=summary['cell']))


def analyze_group(group):
    group = Path(group).resolve()
    result = dict(group=str(group), status='UNRUN', cells=[], errors=[],
        interpretation='Single-observer diagnostic, not a policy comparison or benefit estimate. '
        'Native start_load_kv includes deferred STORE startup as well as LOAD startup; '
        'its CPU wall interval is not CUDA copy time. Scheduler-return splits are clock '
        'boundaries, not a complete compute/transfer decomposition. CUDA report timestamps '
        'identify reporting, not transfer execution. Overlapping durations are not added '
        'or subtracted as end-to-end savings. In particular, native handle_preemptions can '
        'submit deferred STOREs and wait for jobs outside start_load_kv; an outside-call '
        'delay is not thereby classified as model computation. '
        'No legacy Host-arm causal comparison is made.')
    status = base.read_json(group/'status.json', {})
    summary = base.read_json(group/'summary.json')
    if status.get('status') != 'COMPLETE' or not summary:
        return dict(result, reason='completed_group_and_existing_summary_required')
    cells = summary.get('cells', [])
    if ((summary.get('group_status') or {}).get('status') != 'COMPLETE' or not cells
            or any(summary.get(k) for k in ('analysis_errors', 'invalid_cells', 'incomplete_cells'))
            or any(not c.get('all_requests_completed') or c.get('status', {}).get('status') != 'COMPLETE'
                   for c in cells)):
        return dict(result, reason='existing_summary_not_complete_and_clean')
    # Resolve raw paths under this completed group, never follow stale absolute paths.
    paths = [(group/c['cell']/'raw.json').resolve() for c in cells]
    if any(not p.is_relative_to(group) or not p.is_file() for p in paths):
        return dict(result, reason='completed_episode_raw_missing_or_outside_group')
    for cell, path in zip(cells, paths):
        try:
            result['cells'].append(inspect_cell(base.read_json(path), cell))
        except (KeyError, TypeError, ValueError) as error:
            result['errors'].append(dict(cell=cell['cell'], error=str(error)))
    statuses = Counter(c['status'] for c in result['cells'])
    result['status'] = ('INVALID_OBSERVATION' if result['errors'] else
        'COVERAGE_MISSING' if statuses['COVERAGE_MISSING'] else
        'PARTIAL_COVERAGE' if statuses['PARTIAL_COVERAGE'] else 'VALIDATED_OBSERVATION')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = analyze_group(args.group)
    if args.out or args.group.is_dir():
        (args.out or args.group/'transfer_start_summary.json').write_text(
            json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], group=result['group'],
        cells=[dict(cell=c['cell'], status=c['status']) for c in result['cells']],
        errors=result['errors']), indent=2))
    return 0 if result['status'] == 'VALIDATED_OBSERVATION' else 1


if __name__ == '__main__':
    raise SystemExit(main())
