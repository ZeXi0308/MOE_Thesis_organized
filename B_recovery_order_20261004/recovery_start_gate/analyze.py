#!/usr/bin/env python3
"""Complete-service outcomes and a single bounded asynchronous-recovery start gate."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PINS = {
    'recovery_retry_defer/analyze.py': '4bb4157ebf3b12686336aaa736aec25cebb2b480927f9204979a974598178953',
    'recovery_start_yield/analyze.py': 'c35a6c6587b3707ca599ffd2412357abe791142764439403e81e01252bdc5732',
}
ENTRY_LIMIT = 16


def load(name):
    path = BASE/name
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
        raise RuntimeError('Frozen start-gate analysis dependency changed: '+name)
    spec = importlib.util.spec_from_file_location('start_gate_'+path.parent.name+'_analysis', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def actions(data, raw, cell, allocations, request_evidence, number):
    if data is None or raw is None:
        return dict(status='UNAVAILABLE', missing=[name for name, value in
            (('recovery-start-gate', data), ('raw', raw)) if value is None], raw_action_record=data)
    events = data.get('events'); missing = []; failures = []; rows = []
    if not isinstance(events, list): events = []; missing.append('events')
    mode = cell['mode']; origin = raw['measurement_origin_perf_counter_s']; mapping = raw.get('internal_to_source', {})
    if mode not in ('native', 'wait_release') or data.get('mode') != mode: failures.append('mode_mismatch')
    if data.get('status') != 'UNINSTALLED': missing.append('uninstalled_observation')
    if data.get('block_epoch_entries') != ENTRY_LIMIT: failures.append('declared_entry_limit_mismatch')
    opportunities = [event for event in events if event.get('kind') == 'legal_start_gate_opportunity']
    if len(opportunities) > 1: failures.append('more_than_one_selected_target')
    requested_total = executed_total = actual_count = 0
    selected_ids = {event.get('request') for event in opportunities}
    kinds = ('actual_break_requested', 'actual_break_executed', 'entry_observation', 'release',
             'continued_legal_boundary', 'native_other_head_pass_through')
    for event in events:
        if event.get('kind') in kinds and event.get('request') not in selected_ids:
            missing.append('event_selected_request_link')
    for index, event in enumerate(events):
        if event.get('kind') != 'legal_start_gate_opportunity': continue
        required = ('request', 'step', 'host_perf_s', 'candidate', 'running_cohort', 'native_action',
                    'candidate_action', 'final_action', 'deadline_step', 'requested_breaks', 'executed_breaks')
        absent = [key for key in required if key not in event]
        missing.extend('event.'+key for key in absent)
        if absent: continue
        rid, initial, stamp = event['request'], event['step'], event['host_perf_s']
        requested, executed = event['requested_breaks'], event['executed_breaks']
        if type(requested) is not int or type(executed) is not int or not 0 <= executed <= requested <= ENTRY_LIMIT:
            failures.append('invalid_bounded_break_counters'); continue
        requested_total += requested; executed_total += executed; actual_count += executed > 0
        if mode == 'native' and (requested or executed): failures.append('native_requested_or_executed_gate')
        if mode == 'native' and event['final_action'] != 'SHADOW_ONLY': failures.append('native_final_action_mismatch')
        if executed and event['final_action'] != 'WAITING_LOOP_BREAK': failures.append('executed_final_action_mismatch')
        if mode == 'wait_release' and not requested: failures.append('selected_candidate_never_requested_break')
        if type(initial) is not int or not number(stamp): missing.append('valid_selection_step_clock'); continue
        if event['deadline_step'] != initial+ENTRY_LIMIT: failures.append('declared_deadline_step_mismatch')
        first = event.get('first_break_host_perf_s')
        if executed and not number(first): missing.append('first_executed_break_clock')
        elif executed and first < stamp: failures.append('first_break_precedes_selection')
        def records(kind):
            return [dict(event_index=i, step=entry.get('step'),
                host_s=entry['host_perf_s']-origin if number(entry.get('host_perf_s')) else None,
                raw_record=entry) for i, entry in enumerate(events)
                if entry.get('kind') == kind and entry.get('request') == rid]
        requests, executions = records('actual_break_requested'), records('actual_break_executed')
        entries, releases = records('entry_observation'), records('release')
        boundaries, passes = records('continued_legal_boundary'), records('native_other_head_pass_through')
        if any(record['raw_record'].get('native_head') == rid for record in passes):
            failures.append('pass_through_is_selected_target')
        if len(requests) != requested: failures.append('requested_break_record_count_mismatch')
        if len(executions) != executed: failures.append('executed_break_record_count_mismatch')
        for name, records_list in (('requested', requests), ('executed', executions)):
            if [record['raw_record'].get('ordinal') for record in records_list] != list(range(1, len(records_list)+1)):
                failures.append(name+'_break_ordinal_mismatch')
            steps = [record['step'] for record in records_list]
            if len(steps) != len(set(steps)): failures.append(name+'_multiple_breaks_in_same_entry')
            for record in records_list:
                if type(record['step']) is not int or not initial <= record['step'] < initial+ENTRY_LIMIT:
                    failures.append(name+'_break_outside_entry_limit')
                if not number(record['host_s']): missing.append(name+'_break_clock')
                elif record['host_s'] < stamp-origin: failures.append(name+'_break_precedes_selection')
        for record in executions:
            matching = [candidate for candidate in requests
                        if candidate['raw_record'].get('ordinal') == record['raw_record'].get('ordinal')]
            if len(matching) != 1 or matching[0]['event_index'] >= record['event_index']:
                failures.append('executed_break_without_prior_request')
            elif matching[0]['step'] != record['step']: failures.append('request_execution_step_mismatch')
        if executions and executions[0]['host_s'] != (first-origin if number(first) else None):
            failures.append('first_break_clock_mismatch')
        if len(releases) > 1: failures.append('multiple_release_records')
        if not releases: missing.append('selected_target_release_record')
        release = releases[0] if len(releases) == 1 else None
        release_s = release['host_s'] if release else None
        reason = release['raw_record'].get('reason') if release else event.get('release_reason')
        classification = ('COHORT_COMPLETION_EVENT' if reason == 'COHORT_REQUEST_FINISHED' else
            'ENTRY_LIMIT' if reason == 'BLOCK_EPOCH_LIMIT' else
            'UNINSTALL' if reason == 'UNINSTALL' else
            'NATIVE_SAFETY_OR_QUALIFICATION_RELEASE' if reason is not None else 'RELEASE_UNOBSERVED')
        if release:
            if any(record['event_index'] > release['event_index'] for record in executions):
                failures.append('break_executed_after_release')
            if type(release['step']) is not int: missing.append('release_step')
            elif release['step'] > initial+ENTRY_LIMIT: failures.append('release_after_entry_limit')
            if not number(release_s): missing.append('release_clock')
            elif release_s < stamp-origin: failures.append('release_precedes_selection')
            for key, count in (('requested_breaks', requested), ('executed_breaks', executed)):
                if release['raw_record'].get(key) != count: failures.append('release_'+key+'_mismatch')
        for entry in entries:
            if entry['raw_record'].get('initial_step') != initial: failures.append('entry_initial_step_mismatch')
        evidence = request_evidence(rid, stamp, raw, cell, allocations)
        if evidence.get('source_request') is None: missing.append('target_source_mapping')
        attempts = evidence.get('native_allocation_attempts')
        during = [attempt for attempt in attempts or [] if executed and number(first) and number(release_s)
                  and first-origin <= attempt['begin_s'] < release_s]
        after = next((attempt for attempt in attempts or [] if number(release_s) and attempt['begin_s'] >= release_s), None)
        rows.append(dict(event_index=index, decision_s=stamp-origin, initial_step=initial,
            release_before_or_at_entry=initial+ENTRY_LIMIT,
            target=dict(internal_request=rid, source_request=mapping.get(rid)),
            requested_breaks=requested, executed_breaks=executed, actual_gate_executed=bool(executed),
            first_break_s=first-origin if number(first) else None,
            actual_break_requests=requests, actual_break_executions=executions,
            recorded_running_cohort=event['running_cohort'], entry_observations=entries,
            continued_legal_boundaries=boundaries, native_other_head_pass_through=passes,
            release_records=releases, release_s=release_s, release_reason=reason,
            release_classification=classification,
            release_cohort_finished=release['raw_record'].get('cohort_finished') if release else None,
            selection_free_gpu_blocks=event['candidate'].get('free_gpu_blocks'),
            release_free_gpu_blocks=release['raw_record'].get('free_gpu_blocks') if release else None,
            observed_selection_to_release_s=release_s-(stamp-origin) if number(release_s) else None,
            observed_first_break_to_release_s=release_s-(first-origin) if number(release_s) and number(first) else None,
            observed_path='ACTUAL_GATED_RECOVERY_PATH' if executed else 'ACTUAL_UNMODIFIED_NATIVE_PATH',
            observed_target_evidence=evidence,
            native_target_allocation_attempts_during_gate=during if attempts is not None and executed else None,
            first_native_target_allocation_after_release=after,
            release_to_next_output_s=evidence['next_output_after_decision_s']-release_s
                if number(release_s) and number(evidence.get('next_output_after_decision_s')) else None,
            raw_decision=event))
    if data.get('action_count') != actual_count: failures.append('actual_gate_count_mismatch')
    if data.get('requested_breaks') != requested_total: failures.append('requested_break_total_mismatch')
    if data.get('executed_breaks') != executed_total: failures.append('executed_break_total_mismatch')
    if actual_count > 1 or executed_total > ENTRY_LIMIT: failures.append('gate_or_break_budget_exceeded')
    errors = [event for event in events if event.get('error') is not None or event.get('exception') is not None]
    if errors or data.get('status') == 'ERROR': failures.append('policy_error')
    release_counts = {}
    for row in rows:
        reason = row['release_reason'] or 'RELEASE_UNOBSERVED'
        release_counts[reason] = release_counts.get(reason, 0)+1
    return dict(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        failed_checks=sorted(set(failures)), missing=sorted(set(missing)), policy_status=data.get('status'),
        policy_outcome=data.get('outcome'), legal_opportunities=len(opportunities),
        native_shadow_opportunities=len(opportunities) if mode == 'native' else 0,
        actual_gate_count=actual_count, requested_break_count=requested_total, executed_break_count=executed_total,
        unexecuted_requested_break_count=requested_total-executed_total, entry_limit=ENTRY_LIMIT,
        release_reason_counts=release_counts, decision_calls=data.get('decision_calls'),
        qualification_skip_counts=data.get('skip_counts'), rows=rows, error_events=errors, raw_action_record=data)


def analyze_session(session):
    parent = load('recovery_retry_defer/analyze.py'); arrival_helper = load('recovery_start_yield/analyze.py')
    dependencies = {}
    def adapted_fit(repeat):
        fit = parent.load('recovery_fit/analyze.py')
        source = repeat.replace_once(inspect.getsource(fit.helpers), '(native|fit_once)', '(native|wait_release)')
        namespace = dict(vars(fit)); exec(compile(source, '<start-gate-group-helper>', 'exec'), namespace)
        fit.helpers = namespace['helpers']; optional = fit.optional
        fit.optional = lambda directory, name: optional(directory, 'recovery-start-gate' if name == 'recovery-fit' else name)
        fit.actions = lambda data, raw, cell, allocations: actions(data, raw, cell, allocations, repeat.request_evidence, parent.number)
        dependencies.update(optional=optional, summary=fit.read.__globals__['summary'])
        return fit, optional
    parent.adapted_fit = adapted_fit; result = parent.analyze_session(session)
    for cell in result['cells']:
        cell['recovery_start_gate_actions'] = cell.pop('retry_defer_actions')
        directory = Path(cell['directory']); optional = dependencies['optional']
        raw, config = optional(directory, 'raw'), optional(directory, 'config') or {}
        cell['external_arrival_observations'] = arrival_helper.arrivals(raw, dependencies['summary'], parent.number)
        cell['start_gate_policy_evidence'] = {key: value for key, value in config.items()
            if key == 'B_recovery_start_gate' or key.startswith('recovery_start_gate_')}
    by_path = {cell['directory']: cell for cell in result['cells']}
    for pair in result['comparisons']:
        pair['start_gate_execution_observations'] = pair.pop('retry_execution_observations')
        for key, values in pair['start_gate_execution_observations'].items():
            if values is not None: values['release_reason_counts'] = by_path[pair[key]]['recovery_start_gate_actions'].get('release_reason_counts')
    expected = ['native', 'wait_release', 'wait_release', 'native']; layout = result['execution_layout']
    layout.update(design='ONE_TARGET_ASYNC_RECOVERY_START_GATE', expected_abba=expected,
                  complete_abba=layout['modes'] == expected)
    for row in layout['planned_but_not_started']: row['mode'] = expected[row['cell_index']]
    result.pop('retry_defer_semantics')
    result['start_gate_semantics'] = (
        'One selected asynchronous-recovery request is observed at its native allocation boundary. '
        'The recorded running cohort is a release condition, not a second beneficiary/deferred role. '
        'Qualification, requested breaks and executed breaks are separate. One actual target can execute '
        'at most 16 breaks before selection entry + 16; skipped native decision entries need not execute '
        'breaks. Entry/cohort state and raw release reason remain recorded, without predicting capacity '
        'or assuming a fixed wall delay. Cohort completion itself does not prove freed GPU capacity; '
        'selection/release free-block observations and finished cohort held blocks remain separate. '
        'Gate lifetime includes native computation and waiting; it is not '
        'counterfactual added latency or saved latency. Native shadows retain actual unchanged native '
        'paths. The selected request is joined to its containing preemption episode, native allocations, '
        'LOAD ready/submission/host completion/scheduler ACK, resumed schedule, next client output and '
        'subsequent preemptions. Releasing this gate does not guarantee capacity, allocation or output. '
        'All arrived requests, failures, unfinished requests, fixed-output counts/sequences, frozen SLO '
        'denominators, copy work, GC callback wall observations, compact materialization, full process '
        'phases and external-arrival lag keep inherited calculations. Zero-action cells cannot establish '
        'a gate effect. Nearest-native run contrasts do not match runtime states; local intervals and GPU '
        'copy-duration sums are not added into full-service benefit. Unstarted cells are not fabricated.')
    for name in (*PINS, 'recovery_start_gate/analyze.py'):
        result['analyzer_sources_sha256'][name] = hashlib.sha256((BASE/name).read_bytes()).hexdigest()
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
        directory=cell['directory'], **{key: cell['recovery_start_gate_actions'].get(key) for key in
        ('status', 'legal_opportunities', 'actual_gate_count', 'executed_break_count', 'release_reason_counts')})
        for cell in result['cells']])))


if __name__ == '__main__':
    main()
