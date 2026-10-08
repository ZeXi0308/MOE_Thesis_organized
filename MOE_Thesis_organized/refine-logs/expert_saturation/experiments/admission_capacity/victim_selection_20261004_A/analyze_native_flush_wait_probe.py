#!/usr/bin/env python3
"""Reuse full-request mixed-budget analysis and join actual flush/victim job sets."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import analyze_native_mixed_budget_probe as mixed


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_flush_wait_probe_r01'
SEMANTICS = ('Observed host wall time of original native wait calls, grouped by exact engine_call_index '
    'and job-ID set intersection with successful preemptions. STORE waits include predecessor dependencies; '
    'overlap does not assign time to a victim, divide time among jobs, or estimate removable benefit. '
    'Non-overlap does not exclude indirect dependence on earlier STORE jobs. '
    'Observer overhead is not isolated. No-event UNKNOWN is not zero wait. Steps/events are not independent repeats.')


def job_set(value):
    if (not isinstance(value, list) or any(type(x) is not int for x in value)
            or len(value) != len(set(value))):
        return None
    return set(value)


def duration(row, prefix=''):
    start, end, wall = (row.get(prefix + key) for key in ('entered_s', 'returned_s', 'wall_s'))
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in (start, end, wall))
            or end < start or wall != end - start):
        return None
    return wall


def summarize_flush_wait(raw):
    observation = (raw or {}).get('flush_wait_observation')
    if not isinstance(observation, dict):
        return dict(status='UNKNOWN', reason='FLUSH_WAIT_OBSERVATION_MISSING',
                    measured_wait_total_s=None, wait_wall_s=None, steps=[], semantics=SEMANTICS)
    schema_errors, steps = [], {}
    def at(step):
        if step not in steps:
            steps[step] = dict(engine_call_index=step, actual_preemptions=[], unconfirmed_preemptions=[],
                known_victim_job_ids=set(), unknown_victim_requests=[], flush_job_ids=set(),
                wait_job_ids=set(), waits=[], flush_events=[])
        return steps[step]

    preemptions = (raw or {}).get('preemption_events')
    preempt_step_unknown = not isinstance(preemptions, list)
    if not isinstance(preemptions, list):
        schema_errors.append(dict(where='preemption_events', reason='MISSING_PREEMPTION_ARRAY'))
        preemptions = []
    for index, event in enumerate(preemptions):
        if not isinstance(event, dict):
            preempt_step_unknown = True
            schema_errors.append(dict(where='preemption', index=index, reason='INVALID_RECORD'))
            continue
        step = event.get('engine_call_index')
        if type(step) is not int or step < 0:
            preempt_step_unknown = True
            schema_errors.append(dict(where='preemption', index=index, reason='UNKNOWN_STEP'))
            continue
        row = at(step)
        rid = event.get('request_id')
        actual = event.get('original_preemption_called') is True and event.get('original_preemption_returned') is True
        if not actual:
            row['unconfirmed_preemptions'].append(rid)
            continue
        dependency = event.get('pending_native_store_jobs')
        dependency = dependency if isinstance(dependency, dict) else {}
        ids = job_set(dependency.get('job_ids')) if dependency.get('status') == 'KNOWN' else None
        row['actual_preemptions'].append(dict(request=rid, preemption_index=index,
            pending_store_status=dependency.get('status', 'UNKNOWN'),
            pending_store_job_ids=sorted(ids) if ids is not None else None))
        if ids is None:
            row['unknown_victim_requests'].append(rid)
        else:
            row['known_victim_job_ids'].update(ids)

    events = observation.get('events')
    if not isinstance(events, list):
        schema_errors.append(dict(where='flush_events', reason='MISSING_EVENT_ARRAY'))
        events = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            schema_errors.append(dict(where='flush_event', index=index, reason='INVALID_RECORD'))
            continue
        step, ids = event.get('engine_call_index'), job_set(event.get('job_ids'))
        if type(step) is not int or step < 0 or not ids:
            schema_errors.append(dict(where='flush_event', index=index, reason='UNKNOWN_STEP_OR_NONEMPTY_JOB_SET'))
            continue
        row = at(step)
        row['flush_job_ids'].update(ids)
        handle_wall = duration(event, 'handle_')
        calls = event.get('wait_calls')
        if (handle_wall is None or not isinstance(calls, list)
                or type(event.get('handle_completed')) is not bool
                or (event.get('handle_completed') and len(calls) != 1)):
            schema_errors.append(dict(where='flush_event', index=index, reason='INVALID_HANDLE_RECORD'))
        row['flush_events'].append(dict(event_index=index, job_ids=sorted(ids),
            handle_entered_s=event.get('handle_entered_s'), handle_returned_s=event.get('handle_returned_s'),
            handle_wall_s=handle_wall, handle_completed=event.get('handle_completed'),
            handle_error=event.get('handle_error')))
        for call_index, call in enumerate(calls if isinstance(calls, list) else []):
            if not isinstance(call, dict):
                schema_errors.append(dict(where='wait_call', index=index, call_index=call_index, reason='INVALID_RECORD'))
                continue
            wait_ids, wall = job_set(call.get('job_ids')), duration(call)
            valid = (wait_ids == ids and wall is not None and handle_wall is not None
                and event['handle_entered_s'] <= call['entered_s'] <= call['returned_s'] <= event['handle_returned_s']
                and type(call.get('completed')) is bool)
            if not valid:
                schema_errors.append(dict(where='wait_call', index=index, call_index=call_index,
                                          reason='INVALID_WAIT_IDS_OR_INTERVAL'))
            if wait_ids is not None:
                row['wait_job_ids'].update(wait_ids)
            known = row['known_victim_job_ids']
            unknown_victims = bool(preempt_step_unknown or row['unknown_victim_requests'] or row['unconfirmed_preemptions'])
            overlap = (wait_ids & known) if wait_ids is not None else set()
            association = ('UNKNOWN' if not valid else 'VICTIM_JOB_OVERLAP' if overlap
                else 'UNKNOWN' if unknown_victims or not isinstance((raw or {}).get('preemption_events'), list)
                else 'NO_VICTIM_JOB_OVERLAP')
            row['waits'].append(dict(event_index=index, call_index=call_index,
                job_ids=sorted(wait_ids) if wait_ids is not None else None,
                entered_s=call.get('entered_s'), returned_s=call.get('returned_s'),
                wall_s=wall if valid else None, completed=call.get('completed'), error=call.get('error'),
                association=association,
                association_completeness='PARTIAL' if unknown_victims else 'COMPLETE' if valid else 'UNKNOWN',
                known_victim_job_overlap=sorted(overlap)))

    all_waits, handles, serial_steps = [], [], []
    for step, row in sorted(steps.items()):
        row['known_victim_job_overlap'] = sorted(row['wait_job_ids'] & row['known_victim_job_ids'])
        row['other_or_unattributed_wait_job_ids'] = sorted(row['wait_job_ids'] - row['known_victim_job_ids'])
        row['known_victim_job_ids_without_recorded_wait'] = sorted(row['known_victim_job_ids'] - row['wait_job_ids'])
        for key in ('known_victim_job_ids', 'flush_job_ids', 'wait_job_ids'):
            row[key] = sorted(row[key])
        row['recorded_wait_wall_sum_s'] = sum(w['wall_s'] for w in row['waits'] if w['wall_s'] is not None)
        all_waits.extend(row['waits'])
        handles.extend(event['handle_wall_s'] for event in row['flush_events'])
        serial_steps.append(row)
    header_ok = (observation.get('status') == 'INSTALLED_AND_RESTORED'
        and observation.get('installed') is True and observation.get('methods_restored') is True
        and observation.get('errors') == [])
    native_calls_completed = (all(w['completed'] is True for w in all_waits)
        and all(event['handle_completed'] is True for row in serial_steps for event in row['flush_events']))
    verified = header_ok and not schema_errors and native_calls_completed
    grouped = defaultdict(list)
    for call in all_waits:
        grouped[call['association']].append(call['wall_s'])
    return dict(status='VERIFIED' if verified else 'UNKNOWN_OR_INVALID',
        observer_status=observation.get('status'), observer_installed=observation.get('installed'),
        methods_restored=observation.get('methods_restored'), observer_errors=observation.get('errors'),
        observer_overhead_s=observation.get('observer_overhead_s'), schema_errors=schema_errors,
        nonempty_flush_events=len(events), recorded_wait_calls=len(all_waits),
        native_calls_completed=native_calls_completed,
        wait_completion_counts=dict(Counter(str(w['completed']) for w in all_waits)),
        wait_wall_s=mixed.base.distribution([w['wall_s'] for w in all_waits]),
        handle_wall_s=mixed.base.distribution(handles),
        measured_wait_total_s=(sum(w['wall_s'] for w in all_waits) if verified else None),
        association_counts=dict(Counter(w['association'] for w in all_waits)),
        wait_wall_s_by_association={key: mixed.base.distribution(values) for key, values in grouped.items()},
        actual_preemptions=sum(len(row['actual_preemptions']) for row in serial_steps),
        unconfirmed_preemptions=sum(len(row['unconfirmed_preemptions']) for row in serial_steps),
        actual_preemptions_with_unknown_dependencies=sum(len(row['unknown_victim_requests']) for row in serial_steps),
        steps_with_known_victim_dependencies_without_recorded_wait=sum(
            bool(row['known_victim_job_ids_without_recorded_wait']) for row in serial_steps),
        steps=serial_steps, semantics=SEMANTICS)


def analyze(session, package=PACKAGE):
    result = mixed.analyze(session, package/'pkg/staged_store_rotation.py', package/'pkg/inputs/pro_high/config.json')
    directory = session / ('cell-00-' + result['plan']['cells'][0]['label'])
    archive = mixed.base.archive_dir(directory)
    path = archive/'raw.json'
    if not path.exists():
        path = archive/'raw.json.gz'
    raw = mixed.base.optional(path)
    result['flush_wait_analysis'] = summarize_flush_wait(raw)
    requested = result['cell'].get('config', {}).get('record_flush_wait') is True
    result['flush_wait_analysis']['runner_observer_requested'] = requested
    if (result['status'] == 'COMPLETE_CHARACTERIZATION' and requested
            and result['flush_wait_analysis']['status'] == 'VERIFIED'):
        result['status'] = 'COMPLETE_FLUSH_WAIT_CHARACTERIZATION'
    else:
        result['status'] = 'INCOMPLETE_OR_INVALID'
    result['analysis_code_sha256'][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['semantics'] += ' ' + SEMANTICS
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session.resolve(), args.package.resolve())
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    summary = result['flush_wait_analysis']
    print(json.dumps(dict(status=result['status'], output=str(args.output),
        observer_status=summary.get('observer_status'), wait_wall_s=summary.get('wait_wall_s'),
        association_counts=summary.get('association_counts'))))


if __name__ == '__main__':
    main()
