#!/usr/bin/env python3
"""Native/defer-once request outcomes, executed retry breaks, and unchanged GC/storage accounting."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[1]
PINS = {
    'recovery_fit/analyze.py': '2aa8681a9ce34a176d24663f28450303c61ad72b5a824507af07a09c0078232c',
    'recovery_repeat/analyze.py': '51ce8d545808df00ca0cb10628cc02abe37c9e9dc1b6e3a894d3303edf635982',
    'recovery_gc_diag/analyze.py': 'b7fe9cab001ac9da95dc8b1dc645258ac115223629a7fa5ca55ec05e1185ab64',
    'output_event_compact/analyze.py': 'f67ed0f901ec95804abdfac6862eb5ad9c6f98cb09f0c9e92c68ee6c3653c831',
}


def load(name):
    path = BASE/name
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
        raise RuntimeError('Frozen retry-analysis dependency changed: '+name)
    spec = importlib.util.spec_from_file_location('retry_'+path.parent.name+'_analysis', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def actions(data, raw, cell, allocations, request_evidence):
    if data is None or raw is None:
        return dict(status='UNAVAILABLE', missing=[name for name, value in
            (('recovery-retry-defer', data), ('raw', raw)) if value is None], raw_action_record=data)
    mode = cell['mode']; missing = []; failures = []; rows = []
    events = data.get('events')
    if not isinstance(events, list): events = []; missing.append('events')
    if mode not in ('native', 'defer_once') or data.get('mode') != mode: failures.append('mode_mismatch')
    if data.get('status') != 'UNINSTALLED': missing.append('uninstalled_observation')
    opportunities = [event for event in events if event.get('kind') == 'legal_retry_opportunity']
    releases = [event for event in events if event.get('kind') == 'release']
    if len(opportunities) > 1: failures.append('more_than_one_target_selected')
    origin = raw['measurement_origin_perf_counter_s']; mapping = raw.get('internal_to_source', {})
    requested_total = 0; executed_total = 0; executed_gates = 0
    for event_index, event in enumerate(events):
        if event.get('kind') != 'legal_retry_opportunity': continue
        required = ('host_perf_s', 'target', 'num_preemptions', 'previous_preemption',
            'baseline_action', 'candidate_action', 'final_action', 'waiting_order', 'running_cohort',
            'requested_breaks', 'executed_breaks', 'deadline_perf_s', 'release_reason', 'release_observed_perf_s')
        absent = [key for key in required if key not in event]
        missing.extend('event.'+key for key in absent)
        if absent: continue
        requested, executed = event['requested_breaks'], event['executed_breaks']
        if type(requested) is not int or type(executed) is not int or min(requested, executed) < 0:
            failures.append('invalid_break_counters'); continue
        requested_total += requested; executed_total += executed; executed_gates += executed > 0
        if executed > requested: failures.append('executed_exceeds_requested_breaks')
        if mode == 'native' and (requested or executed or event['final_action'] != 'SHADOW_ONLY'):
            failures.append('native_requested_or_executed_deferral')
        if executed and event['final_action'] != 'WAITING_LOOP_BREAK': failures.append('executed_final_action_mismatch')
        if mode == 'defer_once' and requested == 0: failures.append('selected_candidate_never_requested_break')
        stamp = event['host_perf_s']; deadline = event['deadline_perf_s']; release = event['release_observed_perf_s']
        first, last = event.get('first_break_host_perf_s'), event.get('last_break_host_perf_s')
        if not number(stamp) or not number(deadline): missing.append('valid_opportunity_and_deadline_clocks'); continue
        if not number(data.get('max_extra_gate_s')) or not math.isclose(deadline-stamp, data['max_extra_gate_s'], abs_tol=1e-8):
            failures.append('declared_gate_deadline_mismatch')
        if executed and not all(number(value) for value in (first, last)):
            missing.append('executed_break_clocks')
        elif executed and not stamp <= first <= last:
            failures.append('nonmonotone_executed_break_clocks')
        if executed and not number(release): missing.append('release_clock_for_executed_gate')
        if number(release) and (release < stamp or number(last) and release < last):
            failures.append('release_precedes_gate_or_last_break')
        if mode == 'native' and (release is not None or event['release_reason'] is not None):
            failures.append('native_has_gate_release')
        linked_releases = [row for row in releases if row.get('target') == event['target']]
        if mode == 'defer_once':
            if len(linked_releases) != 1: missing.append('single_release_record')
            elif linked_releases[0].get('reason') != event['release_reason'] or linked_releases[0].get('host_perf_s') != release:
                failures.append('release_record_mismatch')
        evidence = request_evidence(event['target'], stamp, raw, cell, allocations)
        if evidence.get('source_request') is None: missing.append('target_source_mapping')
        attempts = evidence.get('native_allocation_attempts')
        during_gate = [attempt for attempt in attempts or [] if executed and number(first) and number(release)
            and first-origin <= attempt['begin_s'] < release-origin]
        after_release = next((attempt for attempt in attempts or [] if number(release) and attempt['begin_s'] >= release-origin), None)
        rows.append(dict(event_index=event_index, decision_s=stamp-origin,
            target=dict(internal_request=event['target'], source_request=mapping.get(event['target'])),
            recorded_num_preemptions=event['num_preemptions'], previous_preemption=event['previous_preemption'],
            baseline_action=event['baseline_action'], requested_action=event['candidate_action'], final_action=event['final_action'],
            requested_breaks=requested, executed_breaks=executed, unexecuted_requested_breaks=requested-executed,
            actual_gate_executed=executed > 0,
            first_break_s=first-origin if number(first) else None,
            last_break_s=last-origin if number(last) else None,
            deadline_s=deadline-origin, release_s=release-origin if number(release) else None,
            release_reason=event['release_reason'], linked_release_records=linked_releases,
            selection_to_release_s=release-stamp if number(release) else None,
            first_break_to_release_s=release-first if number(release) and number(first) else None,
            deadline_observation_lateness_s=max(0., release-deadline) if number(release) else None,
            waiting_requests_at_selection=[dict(internal_request=rid, source_request=mapping.get(rid)) for rid in event['waiting_order']],
            running_cohort_at_selection=[dict(internal_request=rid, source_request=mapping.get(rid)) for rid in event['running_cohort']],
            observed_path='ACTUAL_DEFERRED_PATH' if executed else 'ACTUAL_UNMODIFIED_NATIVE_PATH',
            observed_target_evidence=evidence,
            native_target_allocation_attempts_during_gate=during_gate if attempts is not None and executed else None,
            first_native_target_allocation_after_release=after_release,
            release_to_next_output_s=evidence['next_output_after_decision_s']-(release-origin)
                if number(release) and number(evidence.get('next_output_after_decision_s')) else None,
            raw_decision=event))
    if data.get('executed_breaks') != executed_total: failures.append('executed_break_total_mismatch')
    if data.get('action_count') != executed_gates: failures.append('executed_gate_count_mismatch')
    if executed_gates > 1: failures.append('more_than_one_actual_gate')
    if mode == 'native' and releases: failures.append('native_release_events')
    errors = [event for event in events if event.get('error') is not None or event.get('exception') is not None]
    if errors or data.get('status') == 'ERROR': failures.append('policy_error')
    return dict(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        failed_checks=sorted(set(failures)), missing=sorted(set(missing)),
        policy_status=data.get('status'), policy_outcome=data.get('outcome'),
        legal_opportunities=len(opportunities), native_shadow_opportunities=len(opportunities) if mode == 'native' else 0,
        actual_gate_count=executed_gates, requested_break_count=requested_total, executed_break_count=executed_total,
        unexecuted_requested_break_count=requested_total-executed_total,
        decision_calls=data.get('decision_calls'), qualification_skip_counts=data.get('skip_counts'),
        rows=rows, release_records=releases, error_events=errors, raw_action_record=data)


def adapted_fit(repeat):
    fit = load('recovery_fit/analyze.py')
    source = inspect.getsource(fit.helpers)
    source = repeat.replace_once(source, '(native|fit_once)', '(native|defer_once)')
    namespace = dict(vars(fit)); exec(compile(source, '<retry-group-helper>', 'exec'), namespace)
    fit.helpers = namespace['helpers']; original_optional = fit.optional
    fit.optional = lambda directory, name: original_optional(directory, 'recovery-retry-defer' if name == 'recovery-fit' else name)
    fit.actions = lambda data, raw, cell, allocations: actions(data, raw, cell, allocations, repeat.request_evidence)
    return fit, original_optional


def analyze_session(session):
    repeat, gc, storage = (load(name) for name in ('recovery_repeat/analyze.py',
        'recovery_gc_diag/analyze.py', 'output_event_compact/analyze.py'))
    fit, optional = adapted_fit(repeat); result = fit.analyze_session(session)
    for cell in result['cells']:
        cell['retry_defer_actions'] = cell.pop('recovery_fit_actions')
        summaries = {row['request']: row for row in cell['run_summary']['per_request']}
        for row in cell['retry_defer_actions'].get('rows', []):
            evidence = row['observed_target_evidence']
            evidence['whole_request_recovery_summary'] = summaries.get(evidence.get('source_request'))
        directory = Path(cell['directory'])
        raw, probe, timing, config = (optional(directory, name) for name in ('raw', 'gc-observation', 'timing', 'config'))
        declared_storage = (config or {}).get('B_output_event_storage')
        if declared_storage in ('compact', 'legacy'):
            cell['output_event_storage'] = storage.storage_diagnostic(raw, timing or {}, declared_storage, number)
        else:
            cell['output_event_storage'] = dict(status='UNVERIFIED', missing=['config.B_output_event_storage'],
                record=(raw or {}).get('output_event_storage'))
        cell['expected_output_event_storage'] = declared_storage
        cell['gc_diagnostic'] = gc.gc_diagnostic(raw, probe, cell.get('planned', (config or {}).get('requests', 0)))
        cell['gc_generation2'] = storage.generation2(gc, cell['gc_diagnostic'])
        cell['run_summary']['phase_times'].update(request_observation_s=(raw or {}).get('observation_end_s'),
            output_event_materialization_s=((raw or {}).get('output_event_storage') or {}).get('materialization_s'))
        cell['gpu_copy_work_duration_s'] = {direction: cell.get('copy_work', {}).get(direction, {}).get('gpu_elapsed_sum_s')
                                            for direction in ('load', 'store')}
    by_path = {cell['directory']: cell for cell in result['cells']}
    for pair in result['comparisons']:
        members = {key: by_path.get(pair.get(key)) for key in ('candidate', 'native')}
        pair['retry_execution_observations'] = {key: {name: cell['retry_defer_actions'].get(name)
            for name in ('status', 'legal_opportunities', 'actual_gate_count', 'requested_break_count', 'executed_break_count')}
            if cell else None for key, cell in members.items()}
        pair['storage_observations'] = {key: dict(status=cell['output_event_storage']['status'],
            declared_mode=cell['expected_output_event_storage'],
            recorded_mode=(cell['output_event_storage'].get('record') or {}).get('mode'))
            if cell else None for key, cell in members.items()}
        candidate, native = members['candidate'], members['native']
        pair['phase_delta_candidate_minus_native_s'] = {}
        keys = set((candidate or {}).get('run_summary', {}).get('phase_times', {}))
        keys.update((native or {}).get('run_summary', {}).get('phase_times', {}))
        for key in sorted(keys):
            a = candidate['run_summary']['phase_times'].get(key) if candidate else None
            b = native['run_summary']['phase_times'].get(key) if native else None
            pair['phase_delta_candidate_minus_native_s'][key] = a-b if number(a) and number(b) else None
    modes = [cell['mode'] for cell in result['cells']]; expected = ['native', 'defer_once', 'defer_once', 'native']
    receipt = optional(session, 'receipt') or {}
    observed_indices = {int(match[1]) for cell in result['cells']
        if (match := re.search(r'cell-(\d+)-cap\d+-', cell['directory'])) is not None}
    result['execution_layout'] = dict(design='ONE_TARGET_RETRY_DEFERRAL', cell_count=len(modes), modes=modes,
        expected_abba=expected, complete_abba=modes == expected, comparison_count=len(result['comparisons']),
        controller_status=receipt.get('status'), controller_error=receipt.get('error'),
        controller_stop_reason=receipt.get('stop_reason'),
        planned_but_not_started=[dict(cell_index=index, mode=mode) for index, mode in enumerate(expected)
                                 if index not in observed_indices],
        unstarted_semantics='Planned arms without a cell directory were not executed. They are not '
            'fabricated cells or failed scientific requests. All arrived/failed/unfinished requests in '
            'actually started cells retain the original request accounting.')
    result.pop('recovery_fit_semantics')
    result['retry_defer_semantics'] = (
        'An opportunity and requested break are not an executed waiting-loop break. A single selected target '
        'may have multiple executed breaks; actual_gate_count counts selected targets with positive executed '
        'breaks, not loop iterations. Native has zero requested/executed deferrals. Every target path is '
        'the actual within-run observation; native shadow qualification is not a deferred counterfactual. '
        'The containing preemption episode, allocations, LOAD submission/host completion/ACK, resumed '
        'schedule plan, next client output, later preemptions and full-request recovery union are retained. '
        'Release removes only the extra gate, not native capacity waiting; deadline observation can be late. '
        'The gate also delays later waiting requests, whose complete-service costs remain in all-request '
        'outcomes. Original failures, unfinished requests, output work, copy work, SLO denominators and '
        'nearest-native pair calculations are unchanged. GC callback intervals are wall observations, '
        'not CPU time; compact post-observation materialization remains in capture-return/process costs. '
        'Overlapping stages, GC intervals and GPU copy-duration sums are not added into request savings.')
    for name in (*PINS, 'recovery_retry_defer/analyze.py'):
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
    print(json.dumps(dict(output=str(args.output), layout=result['execution_layout'],
        cells=[dict(directory=cell['directory'], **{key: cell['retry_defer_actions'].get(key) for key in
            ('status', 'actual_gate_count', 'requested_break_count', 'executed_break_count')}) for cell in result['cells']])))


if __name__ == '__main__':
    main()
