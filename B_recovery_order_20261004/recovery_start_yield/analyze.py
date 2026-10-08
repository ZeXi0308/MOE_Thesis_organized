#!/usr/bin/env python3
"""One-round recompute-start yield: unchanged full-service metrics and actual request/job paths."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PARENT_SHA = '4bb4157ebf3b12686336aaa736aec25cebb2b480927f9204979a974598178953'


def load_parent():
    path = BASE/'recovery_retry_defer/analyze.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen full-service retry analyzer changed')
    spec = importlib.util.spec_from_file_location('start_yield_frozen_analysis', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def beneficiary_evidence(record, stamp, raw, cell, allocations, request_evidence, number):
    evidence = request_evidence(record['request'], stamp, raw, cell, allocations)
    origin = raw['measurement_origin_perf_counter_s']; decision_s = stamp-origin
    ids = record.get('job_ids', []); batch_ids = record.get('current_batch_job_ids', [])
    jobs = {job['job_id']: job for job in cell.get('jobs', [])}
    joined = []
    for job_id in ids:
        job = jobs.get(job_id)
        joined.append(dict(job_id=job_id, registered_in_decision_batch=job_id in batch_ids,
            observation=job,
            host_stage_minus_decision_s={key: job[key]-decision_s if job and number(job.get(key)) else None
                for key in ('ready_s', 'submit_begin_s', 'submit_end_s', 'job_completed_s', 'ack_retired_s')}))
    acknowledgments = [row['observation'].get('ack_retired_s') for row in joined if row['observation']]
    all_acked = bool(joined) and len(acknowledgments) == len(joined) and all(number(value) for value in acknowledgments)
    latest_ack = max(acknowledgments) if all_acked else None
    allocation = evidence.get('first_successful_non_delayed_allocation')
    evidence.update(role='RECORDED_LOAD_BENEFICIARY', recorded_decision_state=record,
        pending_load_jobs_at_decision=joined,
        current_decision_batch_load_jobs=[row for row in joined if row['registered_in_decision_batch']],
        all_recorded_pending_load_acks_observed=all_acked,
        all_recorded_pending_loads_last_ack_s=latest_ack,
        last_pending_load_ack_to_non_delayed_allocation_s=allocation['begin_s']-latest_ack
            if allocation and latest_ack is not None else None,
        last_pending_load_ack_to_next_output_s=evidence['next_output_after_decision_s']-latest_ack
            if latest_ack is not None and number(evidence.get('next_output_after_decision_s')) else None)
    return evidence


def actions(data, raw, cell, allocations, request_evidence, number):
    if data is None or raw is None:
        return dict(status='UNAVAILABLE', missing=[name for name, value in
            (('recovery-start-yield', data), ('raw', raw)) if value is None], raw_action_record=data)
    mode = cell['mode']; events = data.get('events'); missing = []; failures = []; rows = []
    if not isinstance(events, list): events = []; missing.append('events')
    if mode not in ('native', 'yield_once') or data.get('mode') != mode: failures.append('mode_mismatch')
    if data.get('status') != 'UNINSTALLED': missing.append('uninstalled_observation')
    opportunities = [event for event in events if event.get('kind') == 'legal_start_yield_opportunity']
    next_entries = [event for event in events if event.get('kind') == 'next_schedule_entry_pass']
    if len(opportunities) > 1: failures.append('more_than_one_selected_opportunity')
    origin = raw['measurement_origin_perf_counter_s']; mapping = raw.get('internal_to_source', {})
    requested_total = executed_total = actual_count = 0
    for index, event in enumerate(events):
        if event.get('kind') != 'legal_start_yield_opportunity': continue
        required = ('host_perf_s', 'step', 'deferred_request', 'beneficiaries', 'candidate', 'native_locals',
                    'baseline_action', 'final_action', 'requested_breaks', 'executed_breaks')
        absent = [key for key in required if key not in event]
        missing.extend('event.'+key for key in absent)
        if absent: continue
        requested, executed = event['requested_breaks'], event['executed_breaks']
        if type(requested) is not int or type(executed) is not int or not 0 <= executed <= requested <= 1:
            failures.append('invalid_one_break_counters'); continue
        requested_total += requested; executed_total += executed; actual_count += executed > 0
        if mode == 'native' and (requested or executed or event['final_action'] != 'SHADOW_ONLY'):
            failures.append('native_requested_or_executed_yield')
        if executed and event['final_action'] != 'WAITING_LOOP_BREAK': failures.append('executed_final_action_mismatch')
        if mode == 'yield_once' and not requested: failures.append('selected_candidate_never_requested_break')
        stamp = event['host_perf_s']; first = event.get('first_break_host_perf_s')
        if not number(stamp): missing.append('valid_decision_clock'); continue
        if executed and not number(first): missing.append('executed_break_clock')
        if number(first) and first < stamp: failures.append('break_precedes_decision')
        deferred = event['deferred_request']; beneficiaries = event['beneficiaries']
        if not isinstance(beneficiaries, list) or not beneficiaries:
            missing.append('nonempty_beneficiary_records'); beneficiaries = []
        beneficiary_ids = [record.get('request') for record in beneficiaries]
        if deferred in beneficiary_ids: failures.append('deferred_request_is_beneficiary')
        if len(beneficiary_ids) != len(set(beneficiary_ids)): failures.append('duplicate_beneficiary')
        deferred_path = request_evidence(deferred, stamp, raw, cell, allocations)
        deferred_path['role'] = 'DEFERRED_RECOMPUTE_HEAD'
        beneficiary_paths = []
        for record in beneficiaries:
            absent = [key for key in ('request', 'locations', 'job_ids', 'current_batch_job_ids') if key not in record]
            missing.extend('beneficiary.'+key for key in absent)
            if absent: continue
            job_ids, batch_ids = record['job_ids'], record['current_batch_job_ids']
            if not isinstance(job_ids, list) or not job_ids or not isinstance(batch_ids, list):
                missing.append('beneficiary.pending_load_job_ids'); continue
            if not set(batch_ids) <= set(job_ids): failures.append('batch_job_not_in_pending_jobs')
            evidence = beneficiary_evidence(record, stamp, raw, cell, allocations, request_evidence, number)
            for joined in evidence['pending_load_jobs_at_decision']:
                job = joined['observation']
                if job is None: missing.append('beneficiary.load_job_observation'); continue
                if job['is_store'] is not False: failures.append('beneficiary_job_not_load')
                if evidence['source_request'] is not None and job['request'] != evidence['source_request']:
                    failures.append('beneficiary_load_request_mismatch')
                if number(job.get('ack_retired_s')) and job['ack_retired_s'] < stamp-origin:
                    failures.append('beneficiary_load_ack_precedes_decision')
            beneficiary_paths.append(evidence)
        if any(path.get('source_request') is None for path in [deferred_path, *beneficiary_paths]):
            missing.append('request_source_mapping')
        linked_entries = [entry for entry in next_entries if entry.get('deferred_request') == deferred
                          and entry.get('decision_step') == event['step']]
        if len(linked_entries) > 1: failures.append('multiple_next_entry_markers')
        next_entry = linked_entries[0] if len(linked_entries) == 1 else None
        if next_entry:
            if next_entry.get('previous_executed_breaks') != executed: failures.append('next_entry_break_count_mismatch')
            if next_entry.get('step') != event['step']+1: failures.append('next_entry_not_next_round')
            if not number(next_entry.get('host_perf_s')): missing.append('next_entry_clock')
            elif next_entry['host_perf_s'] < stamp: failures.append('next_entry_precedes_decision')
        next_s = next_entry['host_perf_s']-origin if next_entry and number(next_entry.get('host_perf_s')) else None
        rows.append(dict(event_index=index, decision_s=stamp-origin, decision_step=event['step'],
            deferred_request=dict(internal_request=deferred, source_request=mapping.get(deferred)),
            beneficiaries=[dict(internal_request=rid, source_request=mapping.get(rid)) for rid in beneficiary_ids],
            baseline_action=event['baseline_action'], final_action=event['final_action'],
            requested_breaks=requested, executed_breaks=executed, actual_yield_executed=bool(executed),
            first_break_s=first-origin if number(first) else None,
            observed_path='ACTUAL_YIELD_RUN_PATHS' if executed else 'ACTUAL_UNMODIFIED_NATIVE_PATHS',
            deferred_request_evidence=deferred_path, beneficiary_request_evidence=beneficiary_paths,
            observed_target_evidence=deferred_path,  # Temporary frozen-parent attachment alias, removed below.
            next_schedule_entry_pass=next_entry, next_schedule_entry_s=next_s,
            decision_to_next_schedule_entry_s=next_s-(stamp-origin) if next_s is not None else None,
            executed_break_to_next_schedule_entry_s=next_s-(first-origin)
                if next_s is not None and number(first) else None,
            raw_decision=event))
    if data.get('action_count') != actual_count: failures.append('actual_action_count_mismatch')
    if data.get('executed_breaks') != executed_total: failures.append('executed_break_total_mismatch')
    if actual_count > 1 or executed_total > 1: failures.append('one_break_budget_exceeded')
    errors = [event for event in events if event.get('error') is not None or event.get('exception') is not None]
    if errors or data.get('status') == 'ERROR': failures.append('policy_error')
    return dict(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        failed_checks=sorted(set(failures)), missing=sorted(set(missing)),
        policy_status=data.get('status'), policy_outcome=data.get('outcome'),
        legal_opportunities=len(opportunities), native_shadow_opportunities=len(opportunities) if mode == 'native' else 0,
        actual_yield_count=actual_count, actual_gate_count=actual_count,  # Temporary parent summary alias.
        requested_break_count=requested_total, executed_break_count=executed_total,
        unexecuted_requested_break_count=requested_total-executed_total,
        decision_calls=data.get('decision_calls'), qualification_skip_counts=data.get('skip_counts'),
        rows=rows, next_schedule_entry_records=next_entries, error_events=errors, raw_action_record=data)


def arrivals(raw, summary, number):
    if raw is None: return dict(status='UNAVAILABLE')
    requests = raw.get('requests', []); rows = []
    for request in requests:
        arrival = request.get('arrival_s'); admission = request.get('admission_s'); returned = request.get('engine_add_return_s')
        rows.append(dict(request=request['request_id'], arrival_s=arrival, admission_s=admission,
            engine_add_return_s=returned,
            arrival_to_admission_s=admission-arrival if number(admission) and number(arrival) else None,
            arrival_to_engine_add_return_s=returned-arrival if number(returned) and number(arrival) else None))
    last_arrival = max((row['arrival_s'] for row in rows if number(row['arrival_s'])), default=None)
    return dict(status='OBSERVED', request_rows=len(rows), per_request=rows,
        arrival_to_admission_s=summary([row['arrival_to_admission_s'] for row in rows]),
        arrival_to_engine_add_return_s=summary([row['arrival_to_engine_add_return_s'] for row in rows]),
        last_external_arrival_s=last_arrival,
        arrival_end_to_observation_end_s=raw['observation_end_s']-last_arrival if last_arrival is not None else None,
        semantics='Host submission lag is retained separately; complete-service TTFT and flow still start '
            'at the original external arrival. Missing requests and failures retain frozen full-service accounting.')


def analyze_session(session):
    parent = load_parent(); dependencies = {}

    def adapted_fit(repeat):
        fit = parent.load('recovery_fit/analyze.py')
        source = repeat.replace_once(inspect.getsource(fit.helpers), '(native|fit_once)', '(native|yield_once)')
        namespace = dict(vars(fit)); exec(compile(source, '<start-yield-group-helper>', 'exec'), namespace)
        fit.helpers = namespace['helpers']; optional = fit.optional
        fit.optional = lambda directory, name: optional(directory, 'recovery-start-yield' if name == 'recovery-fit' else name)
        fit.actions = lambda data, raw, cell, allocations: actions(data, raw, cell, allocations, repeat.request_evidence, parent.number)
        dependencies.update(optional=optional, summary=fit.read.__globals__['summary'])
        return fit, optional

    parent.adapted_fit = adapted_fit; result = parent.analyze_session(session)
    for cell in result['cells']:
        action = cell.pop('retry_defer_actions'); cell['recovery_start_yield_actions'] = action
        action.pop('actual_gate_count', None)
        summaries = {row['request']: row for row in cell['run_summary']['per_request']}
        for row in action.get('rows', []):
            row.pop('observed_target_evidence')
            for evidence in [row['deferred_request_evidence'], *row['beneficiary_request_evidence']]:
                evidence['whole_request_recovery_summary'] = summaries.get(evidence.get('source_request'))
        directory = Path(cell['directory']); optional = dependencies['optional']
        raw, config = optional(directory, 'raw'), optional(directory, 'config') or {}
        cell['external_arrival_observations'] = arrivals(raw, dependencies['summary'], parent.number)
        cell['start_yield_policy_evidence'] = {key: value for key, value in config.items()
                                             if key == 'B_recovery_start_yield' or key.startswith('recovery_start_yield_')}
    for pair in result['comparisons']:
        observations = pair.pop('retry_execution_observations')
        for value in observations.values():
            if value is not None: value['actual_yield_count'] = value.pop('actual_gate_count')
        pair['start_yield_execution_observations'] = observations
    expected = ['native', 'yield_once', 'yield_once', 'native']; layout = result['execution_layout']
    layout.update(design='ONE_ROUND_RECOMPUTE_START_YIELD', expected_abba=expected,
                  complete_abba=layout['modes'] == expected)
    for row in layout['planned_but_not_started']: row['mode'] = expected[row['cell_index']]
    result.pop('retry_defer_semantics')
    result['start_yield_semantics'] = (
        'A recorded legal opportunity is separate from a requested and an actually executed waiting-loop '
        'break. Native shadows contain only their actual unmodified paths, not yield counterfactuals. '
        'The deferred recompute head and recorded LOAD beneficiaries are separate requests and roles. '
        'Beneficiary pending LOADs are joined by recorded job IDs, including jobs ready before the decision; '
        'current_batch_job_ids identify jobs registered in the decision round, not an inferred submit/ACK round. '
        'Actual submit, host polling completion and scheduler ACK times are retained separately; GPU copy '
        'durations are not absolute GPU-finish times. Non-delayed allocation, schedule plans and next client '
        'outputs are actual within-run observations, not guarantees of benefit. A next-entry marker only '
        'observes the following native scheduler entry: break-to-entry time is not counterfactual added '
        'latency, and native capacity waiting may continue. Full request/preemption histories, failures, '
        'unfinished requests, output sequences/counts, fixed SLO, all copy work, compact materialization, '
        'GC wall intervals, and complete process phases retain frozen calculations. Local intervals cannot '
        'be summed as service savings. Zero executed actions cannot explain latency differences. Original '
        'nearest-native run contrasts are descriptive and do not match runtime states across runs.')
    result['analyzer_sources_sha256']['recovery_start_yield/analyze.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
        ('status', 'legal_opportunities', 'actual_yield_count', 'requested_break_count', 'executed_break_count')})
        for cell in result['cells']])))


if __name__ == '__main__':
    main()
