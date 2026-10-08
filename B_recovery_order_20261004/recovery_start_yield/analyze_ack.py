#!/usr/bin/env python3
"""Frozen complete-service analysis plus bounded repeated breaks and native ACK observations."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ONCE_ANALYZER_SHA = 'c35a6c6587b3707ca599ffd2412357abe791142764439403e81e01252bdc5732'


def load_once():
    path = ROOT/'analyze.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != ONCE_ANALYZER_SHA:
        raise RuntimeError('Frozen one-round analyzer changed')
    spec = importlib.util.spec_from_file_location('ack_frozen_start_yield_analysis', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def replace_once(text, old, new):
    if text.count(old) != 1: raise RuntimeError('Frozen ACK-analysis boundary changed: '+old)
    return text.replace(old, new)


def extend_actions(result, data, raw, number):
    if data is None or raw is None: return result
    origin = raw['measurement_origin_perf_counter_s']
    events = data.get('events') if isinstance(data.get('events'), list) else []
    failures = list(result.get('failed_checks', [])); missing = list(result.get('missing', []))
    requested = [event for event in events if event.get('kind') == 'actual_break_requested']
    executed = [event for event in events if event.get('kind') == 'actual_break_executed']
    if data.get('requested_breaks') != result.get('requested_break_count'):
        failures.append('top_requested_break_count_mismatch')
    if len(requested) != result.get('requested_break_count'): failures.append('requested_break_records_mismatch')
    if len(executed) != result.get('executed_break_count'): failures.append('executed_break_records_mismatch')
    known_targets = {row['deferred_request']['internal_request'] for row in result.get('rows', [])}
    related_kinds = ('actual_break_requested', 'actual_break_executed', 'ack_entry_observation',
                     'release', 'native_other_head_pass_through', 'continued_legal_boundary')
    for event in events:
        if event.get('kind') in related_kinds and event.get('deferred_request') not in known_targets:
            missing.append('event_selected_target_link')
    terminal_counts = {}
    for row in result.get('rows', []):
        target = row['deferred_request']['internal_request']; initial = row['raw_decision']; step = row['decision_step']
        selected = [(index, event) for index, event in enumerate(events)
                    if event.get('deferred_request') == target and event.get('kind') in related_kinds]
        def records(kind):
            return [dict(event_index=index, host_s=event['host_perf_s']-origin if number(event.get('host_perf_s')) else None,
                         step=event.get('step'), raw_record=event)
                    for index, event in selected if event.get('kind') == kind]
        req, done = records('actual_break_requested'), records('actual_break_executed')
        entries, releases = records('ack_entry_observation'), records('release')
        continuations, passes = records('continued_legal_boundary'), records('native_other_head_pass_through')
        for kind, records_list in (('requested', req), ('executed', done)):
            if [record['raw_record'].get('ordinal') for record in records_list] != list(range(1, len(records_list)+1)):
                failures.append(kind+'_break_ordinals')
            for record in records_list:
                if record['host_s'] is None: missing.append(kind+'_break_clock')
                elif record['host_s'] < row['decision_s']: failures.append(kind+'_break_before_decision')
                if record['step'] != step+record['raw_record'].get('ordinal', -100)-1:
                    failures.append(kind+'_break_outside_allowed_round')
        for record in done:
            ordinal = record['raw_record'].get('ordinal')
            matches = [candidate for candidate in req if candidate['raw_record'].get('ordinal') == ordinal]
            if len(matches) != 1: failures.append('executed_break_without_unique_request')
            elif matches[0]['event_index'] >= record['event_index']: failures.append('break_execution_precedes_request')
            if ordinal == 2 and not any(candidate['step'] == step+1 and candidate['event_index'] < record['event_index']
                                        for candidate in continuations):
                missing.append('second_break_legal_boundary')
        if done and done[0]['host_s'] != row.get('first_break_s'): failures.append('first_break_clock_mismatch')
        initial_jobs = initial.get('initial_jobs')
        expected_ids = {job_id for beneficiary in initial.get('beneficiaries', []) for job_id in beneficiary.get('job_ids', [])}
        if not isinstance(initial_jobs, list): initial_jobs = []; missing.append('initial_jobs')
        initial_ids = {job.get('job_id') for job in initial_jobs}
        if initial_ids != expected_ids: failures.append('initial_job_set_differs_from_beneficiaries')

        def check_jobs(jobs, label):
            if not isinstance(jobs, list): missing.append(label+'.jobs'); return
            if {job.get('job_id') for job in jobs} != initial_ids: failures.append(label+'.original_job_set_changed')
            for job in jobs:
                state = job.get('state')
                if state == 'ACK_RETIRED' and not (job.get('request_identity_match') is True
                        and job.get('job_identity_match') is False and job.get('request_transfer_contains') is False
                        and job.get('pending_count') is None): failures.append(label+'.inconsistent_ack_retirement')
                if state == 'PENDING' and not (job.get('request_identity_match') is True
                        and job.get('job_identity_match') is True and job.get('request_transfer_contains') is True
                        and type(job.get('pending_count')) is int and job['pending_count'] > 0):
                    failures.append(label+'.inconsistent_pending_state')
        check_jobs(initial_jobs, 'initial')
        for entry in entries:
            record = entry['raw_record']; check_jobs(record.get('jobs'), 'ack_entry')
            if record.get('initial_step') != step: failures.append('ack_entry_initial_step_mismatch')
            states = [job.get('state') for job in record.get('jobs', [])]
            all_acked = bool(states) and all(state == 'ACK_RETIRED' for state in states)
            if record.get('all_original_loads_ack_retired') is not all_acked:
                failures.append('ack_entry_all_retired_flag_mismatch')
            entry['state_counts'] = {state: states.count(state) for state in sorted({state for state in states if isinstance(state, str)})}
        for boundary in continuations: check_jobs(boundary['raw_record'].get('jobs'), 'continued_boundary')
        for passed in passes:
            if passed['raw_record'].get('native_head') == target: failures.append('pass_through_is_selected_target')
        if len(releases) > 1: failures.append('multiple_release_records')
        release = releases[0] if len(releases) == 1 else None
        reason = release['raw_record'].get('reason') if release else None
        if reason == 'ALL_ORIGINAL_LOADS_ACK_RETIRED': classification = 'ORIGINAL_LOAD_ACK_RETIREMENT'
        elif reason == 'TWO_ROUND_LIMIT': classification = 'TWO_ROUND_LIMIT'
        elif isinstance(reason, str) and reason.startswith('UNKNOWN'): classification = 'UNKNOWN_NATIVE_STATE_RELEASE'
        elif reason is not None: classification = 'NATIVE_SAFETY_OR_QUALIFICATION_RELEASE'
        elif row['actual_yield_executed']: classification = 'RELEASE_UNOBSERVED_AT_OBSERVATION_END'
        else: classification = 'SHADOW_RELEASE_UNOBSERVED'
        terminal_counts[classification] = terminal_counts.get(classification, 0)+1
        if release:
            record = release['raw_record']
            if record.get('requested_breaks') != row['requested_breaks'] or record.get('executed_breaks') != row['executed_breaks']:
                failures.append('release_break_count_mismatch')
            if any(item['event_index'] > release['event_index'] for item in done): failures.append('executed_break_after_release')
            if not number(release['host_s']): missing.append('release_clock')
            if type(release['step']) is not int: missing.append('release_step')
            elif release['step'] > step+2: failures.append('release_after_two_round_limit')
            if record.get('ack_jobs') is not None: check_jobs(record['ack_jobs'], 'release')
        last_entry = next((entry for entry in reversed(entries) if release and entry['event_index'] < release['event_index']), None)
        row.update(actual_break_requests=req, actual_break_executions=done,
            initial_native_job_states=initial_jobs, initial_stale_job_threshold=initial.get('initial_stale_job_threshold'),
            ack_entry_observations=entries, continued_legal_boundaries=continuations,
            native_other_head_pass_through=passes, release_records=releases,
            release_observation=dict(classification=classification, reason=reason,
                release_ack_observation_phase=release['raw_record'].get('ack_observation_phase') if release else None,
                release_s=release['host_s'] if release else None, release_step=release['step'] if release else None,
                observed_selection_to_release_s=release['host_s']-row['decision_s']
                    if release and number(release['host_s']) else None,
                latest_prior_ack_entry_observation=last_entry,
                semantics='Recorded release source and host lifetime; the latest prior ACK-entry snapshot '
                    'need not be at the release instant. Lifetime includes native work and gates, not '
                    'counterfactual added latency or savings. Releasing this rule does not guarantee allocation.'))
    result.update(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        failed_checks=sorted(set(failures)), missing=sorted(set(missing)),
        actual_break_request_record_count=len(requested), actual_break_execution_record_count=len(executed),
        terminal_classification_counts=terminal_counts,
        native_other_head_pass_through_count=sum(event.get('kind') == 'native_other_head_pass_through' for event in events))
    return result


def adapted_once():
    once = load_once(); namespace = dict(vars(once))
    source = inspect.getsource(once.actions).replace("'yield_once'", "'yield_ack'")
    source = replace_once(source, "('recovery-start-yield', data)", "('recovery-start-yield-ack', data)")
    source = replace_once(source, '0 <= executed <= requested <= 1:', '0 <= executed <= requested <= 2:')
    source = replace_once(source, "'invalid_one_break_counters'", "'invalid_two_break_counters'")
    source = replace_once(source, 'actual_count > 1 or executed_total > 1:', 'actual_count > 1 or executed_total > 2:')
    source = replace_once(source, "'one_break_budget_exceeded'", "'two_break_budget_exceeded'")
    exec(compile(source, '<ack-bounded-actions>', 'exec'), namespace)
    base_actions = namespace['actions']
    namespace['actions'] = lambda data, raw, cell, allocations, request_evidence, number: extend_actions(
        base_actions(data, raw, cell, allocations, request_evidence, number), data, raw, number)
    source = inspect.getsource(once.analyze_session).replace('yield_once', 'yield_ack')
    source = replace_once(source, "'recovery-start-yield'", "'recovery-start-yield-ack'")
    source = replace_once(source, "'B_recovery_start_yield'", "'B_recovery_start_yield_ack'")
    source = replace_once(source, 'ONE_ROUND_RECOMPUTE_START_YIELD', 'TWO_ROUND_ACK_BOUNDED_RECOMPUTE_START_YIELD')
    exec(compile(source, '<ack-bounded-session>', 'exec'), namespace)
    return namespace


def analyze_session(session):
    namespace = adapted_once(); result = namespace['analyze_session'](session)
    by_path = {cell['directory']: cell for cell in result['cells']}
    for pair in result['comparisons']:
        pair['ack_bounded_execution_observations'] = {key: {name: by_path[pair[key]]['recovery_start_yield_actions'].get(name)
            for name in ('status', 'actual_yield_count', 'requested_break_count', 'executed_break_count',
                         'actual_break_execution_record_count', 'terminal_classification_counts',
                         'native_other_head_pass_through_count')}
            if pair.get(key) in by_path else None for key in ('candidate', 'native')}
    result['start_yield_semantics'] = result['start_yield_semantics'].replace(
        'A next-entry marker only observes the following native scheduler entry: break-to-entry time is not counterfactual added latency, and native capacity waiting may continue.',
        'ACK-entry snapshots observe original native scheduler job state; observation intervals are not counterfactual added latency, and native capacity waiting may continue.')
    result['start_yield_ack_semantics'] = (
        'One selected recompute head may execute at most two breaks: at selection and optionally the following '
        'native round. Actual request/execution records retain ordinal, step and host time. The action count '
        'counts selected targets with an actual break, not break iterations. Initial pending LOAD job IDs are '
        'fixed; ACK_RETIRED records both native scheduler job-table and request-transfer membership retirement '
        'with matching request identity. PENDING and UNKNOWN states remain explicit. Release distinguishes '
        'original-LOAD retirement, the two-round limit, unknown state and native safety/qualification exits. '
        'A native shadow has zero breaks even when it records ACK states and release. Other-head pass-through '
        'does not prove allocation or output. Deferred-head and beneficiary paths, all-request metrics, copy '
        'work, output quantity/sequences, GC, process phases and external-arrival lag inherit the frozen '
        'analysis. No local interval or lifetime is an added-delay/saved-latency counterfactual; zero-action '
        'arms do not establish intervention benefit. Missing releases and unstarted cells are not fabricated.')
    result['analyzer_sources_sha256']['recovery_start_yield/analyze_ack.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = analyze_session(args.session)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), layout=result['execution_layout'], cells=[dict(
        directory=cell['directory'], **{key: cell['recovery_start_yield_actions'].get(key) for key in
        ('status', 'legal_opportunities', 'actual_yield_count', 'executed_break_count', 'terminal_classification_counts')})
        for cell in result['cells']])))


if __name__ == '__main__':
    main()
