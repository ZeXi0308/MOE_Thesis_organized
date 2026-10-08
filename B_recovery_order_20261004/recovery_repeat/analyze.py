#!/usr/bin/env python3
"""Frozen full-service metrics plus actual multi-episode recovery bypass trajectories."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
FIT_SHA = '2aa8681a9ce34a176d24663f28450303c61ad72b5a824507af07a09c0078232c'
LIMITS = {'once': 1, 'repeat8': 8}


def load_fit():
    path = BASE/'recovery_fit/analyze.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != FIT_SHA:
        raise RuntimeError('Frozen recovery-fit analyzer changed')
    spec = importlib.util.spec_from_file_location('repeat_frozen_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIT = load_fit()
ORIGINAL_HELPERS = FIT.helpers
ORIGINAL_OPTIONAL = FIT.optional


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Frozen analysis adaptation boundary changed: '+old)
    return text.replace(old, new)


def helpers():
    group, tail, summarize = ORIGINAL_HELPERS()
    compare = group.__globals__['comparisons']
    source = inspect.getsource(compare)
    source = replace_once(source, "endswith('-native')", "endswith('-once')")
    source = replace_once(source, 'Nearest native cell in execution order', 'Nearest once cell in execution order')
    source = replace_once(source, 'Candidate or native raw unavailable', 'Candidate or once raw unavailable')
    compare_namespace = dict(compare.__globals__)
    exec(compile(source, str(BASE/'analyze.py')+'[once-reference]', 'exec'), compare_namespace)
    path = BASE/'normal_capacity/analyze_group.py'
    source = replace_once(path.read_text(), '(native|age|flush_first)', '(once|repeat8)')
    namespace = dict(__name__='repeat_analysis_group', __file__=str(path))
    exec(compile(source, str(path)+'[repeat-modes]', 'exec'), namespace)
    namespace['comparisons'] = compare_namespace['comparisons']
    return namespace['analyze_group'], tail, summarize


def request_evidence(rid, stamp, raw, cell, allocations):
    evidence = FIT.request_evidence(rid, stamp, raw, cell, allocations)
    origin = raw['measurement_origin_perf_counter_s']
    decision_s = stamp-origin
    episodes = [episode for episode in (allocations or {}).get('episodes', [])
                if episode['request'] == rid and episode['begin_host_perf_s'] <= stamp <= episode['end_host_perf_s']]
    evidence['action_preemption_episode'] = ({key: episodes[0][key] for key in (
        'episode', 'begin_s', 'end_s', 'end_reason', 'preempt_record_index', 'preempt_record')}
        if len(episodes) == 1 else None)
    records = raw.get('preemption_events')
    if not isinstance(records, list):
        evidence['later_preemption_observation'] = 'UNAVAILABLE'
        return evidence
    successful = [event for event in records if event.get('internal_request_id') == rid
                  and event.get('original_preemption_returned') is True]
    later = [event for event in successful if event['method_entered_s'] > decision_s]
    next_output = evidence.get('next_output_after_decision_s')
    after_output = [event for event in later if next_output is not None and event['method_entered_s'] >= next_output]
    evidence.update(later_preemption_observation='AVAILABLE',
        observed_successful_preemptions_through_decision=sum(event['method_returned_s'] <= decision_s for event in successful),
        later_successful_preemptions=later,
        later_preemptions_before_next_output=sum(event['method_entered_s'] < next_output for event in later)
            if next_output is not None else None,
        later_preemptions_after_next_output=len(after_output) if next_output is not None else None,
        first_preemption_after_next_output_s=after_output[0]['method_entered_s'] if after_output else None,
        next_output_to_next_preemption_s=after_output[0]['method_entered_s']-next_output if after_output else None)
    return evidence


def actions(data, raw, cell, allocations):
    if data is None or raw is None:
        return dict(status='UNAVAILABLE', missing=[key for key, value in
            (('recovery-repeat', data), ('raw', raw)) if value is None], raw_action_record=data)
    origin, mapping = raw['measurement_origin_perf_counter_s'], raw.get('internal_to_source', {})
    mode = cell['mode']; limit = LIMITS.get(mode)
    events = data.get('events'); missing = []; failures = []; rows = []; opportunities = []
    if not isinstance(events, list):
        events = []; missing.append('events')
    if data.get('mode') != mode: failures.append('mode_mismatch')
    if limit is None or data.get('action_limit') != limit: failures.append('action_limit_mismatch')
    if data.get('status') != 'UNINSTALLED': missing.append('uninstalled_observation')
    completed_count = 0; changed_count = 0; reported_count = 0; bypassed = set(); skip_counts = Counter()
    for event_index, event in enumerate(events):
        if event.get('kind') != 'fit_opportunity': continue
        required = ('host_perf_s', 'baseline_head', 'candidate_head', 'final_head', 'queue_changed',
                    'waiting_before', 'waiting_after', 'candidate_num_preemptions', 'action',
                    'action_count_before', 'action_count_after', 'action_limit', 'action_skip_reasons')
        absent = [key for key in required if key not in event]
        missing.extend('event.'+key for key in absent)
        if absent: continue
        before, after = event['waiting_before'], event['waiting_after']
        if not isinstance(before, list) or not isinstance(after, list):
            missing.append('waiting_lists'); continue
        changed = before != after; reported = event['queue_changed'] is True
        changed_count += changed; reported_count += reported
        if changed != reported: failures.append('queue_change_flag_mismatch')
        if Counter(before) != Counter(after): failures.append('waiting_members_changed')
        if (before[0] if before else None) != event['baseline_head']: failures.append('baseline_head_mismatch')
        if (after[0] if after else None) != event['final_head']: failures.append('final_head_mismatch')
        candidate = event['candidate_head']; episode = (candidate, event['candidate_num_preemptions'])
        passed = before[:before.index(candidate)] if candidate in before else None
        if passed is None: failures.append('candidate_missing_from_waiting')
        if changed and (not after or after[0] != candidate): failures.append('candidate_not_final_head')
        if changed and [rid for rid in before if rid != candidate] != after[1:]:
            failures.append('other_request_relative_order_changed')
        if event['action_count_before'] != completed_count: failures.append('action_count_before_mismatch')
        if event['action_limit'] != limit: failures.append('event_action_limit_mismatch')
        reasons = event['action_skip_reasons']
        if not isinstance(reasons, list):
            missing.append('action_skip_reasons_list'); reasons = []
        expected_reasons = ([] if episode not in bypassed else ['EPISODE_ALREADY_BYPASSED'])
        if limit is not None and completed_count >= limit: expected_reasons.append('BUDGET_EXHAUSTED')
        if reasons != expected_reasons: failures.append('episode_or_budget_skip_mismatch')
        skip_counts.update(reasons)
        completed = event['action'] == 'REORDERED' and changed and reported
        if completed:
            if episode in bypassed: failures.append('same_episode_bypassed_twice')
            bypassed.add(episode); completed_count += 1
        if event['action'] == 'SHADOW_ONLY' and (changed or reported): failures.append('shadow_changed_queue')
        if event['action_count_after'] != completed_count: failures.append('action_count_after_mismatch')
        opportunities.append(dict(event_index=event_index, decision_s=event['host_perf_s']-origin,
            candidate_request=candidate, candidate_source_request=mapping.get(candidate),
            candidate_num_preemptions=event['candidate_num_preemptions'], baseline_head=event['baseline_head'],
            action=event['action'], queue_order_changed_observed=changed, completed_bypass=completed,
            action_skip_reasons=reasons, action_count_before=event['action_count_before'],
            action_count_after=event['action_count_after']))
        # Shadow opportunities have no joined alternative outcome or counterfactual trajectory.
        if not changed: continue
        ids = list(dict.fromkeys([event['baseline_head'], candidate]+(passed or [])))
        requests = [request_evidence(rid, event['host_perf_s'], raw, cell, allocations) for rid in ids]
        if any(request['source_request'] is None for request in requests): missing.append('request_source_mapping')
        rows.append(dict(event_index=event_index, decision_s=event['host_perf_s']-origin,
            actual_queue_change=True, completed_bypass=completed,
            candidate_num_preemptions=event['candidate_num_preemptions'],
            baseline_head=dict(internal_request=event['baseline_head'], source_request=mapping.get(event['baseline_head'])),
            candidate_head=dict(internal_request=candidate, source_request=mapping.get(candidate)),
            final_head=dict(internal_request=event['final_head'], source_request=mapping.get(event['final_head'])),
            passed_requests=[dict(internal_request=rid, source_request=mapping.get(rid)) for rid in passed] if passed is not None else None,
            action_host_duration_s=event['return_host_perf_s']-event['action_begin_host_perf_s']
                if all(key in event for key in ('return_host_perf_s', 'action_begin_host_perf_s')) else None,
            request_evidence=requests, raw_decision=event))
    if data.get('action_count') != completed_count: failures.append('action_count_mismatch')
    if limit is not None and completed_count > limit: failures.append('action_budget_exceeded')
    if data.get('shadow_count') != len(opportunities): failures.append('qualifying_opportunity_count_mismatch')
    errors = [event for event in events if event.get('error') is not None or event.get('action') == 'ERROR']
    if errors or data.get('status') == 'ERROR': failures.append('policy_error')
    return dict(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        failed_checks=sorted(set(failures)), missing=sorted(set(missing)), policy_status=data.get('status'),
        policy_outcome=data.get('outcome'), action_limit=limit, qualifying_opportunities=len(opportunities),
        shadow_only_opportunities=sum(row['action'] == 'SHADOW_ONLY' for row in opportunities),
        queue_change_reported_count=reported_count, queue_order_change_observed_count=changed_count,
        actual_completed_bypass_count=completed_count, unique_bypassed_request_episodes=len(bypassed),
        unique_bypassed_requests=len({rid for rid, _ in bypassed}), action_skip_counts=dict(skip_counts),
        callback_calls=data.get('callback_calls'), qualification_and_action_skip_counts=data.get('skip_counts'),
        opportunities=opportunities, rows=rows, error_events=errors, raw_action_record=data)


def analyze_session(session):
    FIT.helpers = helpers
    FIT.optional = lambda directory, name: ORIGINAL_OPTIONAL(directory, 'recovery-repeat' if name == 'recovery-fit' else name)
    FIT.actions = actions
    result = FIT.analyze_session(session)
    for cell in result['cells']:
        action = cell.pop('recovery_fit_actions'); cell['recovery_repeat_actions'] = action
        requests = {row['request']: row for row in cell['run_summary']['per_request']}
        for row in action.get('rows', []):
            for evidence in row['request_evidence']:
                evidence['whole_request_recovery_summary'] = requests.get(evidence['source_request'])
    for pair in result['comparisons']:
        pair['once'] = pair.pop('native')
        for old, new in (('aggregate_delta_candidate_minus_native', 'aggregate_delta_candidate_minus_once'),
                         ('copy_work_delta_candidate_minus_native', 'copy_work_delta_candidate_minus_once')):
            if old in pair: pair[new] = pair.pop(old)
        if 'fixed1024_contracts' in pair:
            pair['fixed1024_contracts']['once'] = pair['fixed1024_contracts'].pop('native')
        for row in pair.get('per_request', []): row['once_status'] = row.pop('native_status')
    modes = [cell['mode'] for cell in result['cells']]
    expected = ['once', 'repeat8', 'repeat8', 'once']
    result['execution_layout'] = dict(cell_count=len(modes), modes=modes, expected_abba=expected,
        complete_abba=modes == expected, comparison_count=len(result['comparisons']))
    result['analyzer_sources_sha256']['recovery_repeat/analyze.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['recovery_repeat_semantics'] = (
        'once and repeat8 use the same frozen fit predicate, with total completed-bypass budgets 1 and 8. '
        'Every observed queue change has its own request/preemption-episode join and native allocation, '
        'LOAD submission/host completion/ACK, resumed schedule plan, next client output and later successful '
        'preemptions. Shadow opportunities report eligibility/budget/episode skips only, never alternative '
        'policy outcomes. Local next-output waits may overlap and must not be summed into request benefit; '
        'whole-request recovery union and all-request metrics retain subsequent preemptions. All timestamps '
        'are host observations; GPU copy durations are separate work diagnostics. Original all-request, '
        'failure, unfinished, fixed-output, SLO and copy-work calculations are unchanged. The same nearest '
        'reference rule now chooses once; two run-level contrasts are descriptive, not same-state effects.')
    result.pop('recovery_fit_semantics')
    result['semantics'] = result['semantics'].replace('native cells', 'once cells')
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
        cells=[dict(directory=cell['directory'], status=cell['recovery_repeat_actions']['status'],
            actual_changes=cell['recovery_repeat_actions'].get('queue_order_change_observed_count'),
            completed_bypasses=cell['recovery_repeat_actions'].get('actual_completed_bypass_count')) for cell in result['cells']])))


if __name__ == '__main__':
    main()
