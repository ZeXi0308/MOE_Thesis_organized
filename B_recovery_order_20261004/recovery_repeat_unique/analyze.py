#!/usr/bin/env python3
"""Complete-service outcomes: repeat8 episode reuse versus unique8 request reuse."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[1]
PINS = {
    'recovery_repeat/analyze.py': '51ce8d545808df00ca0cb10628cc02abe37c9e9dc1b6e3a894d3303edf635982',
    'recovery_retry_defer/analyze.py': '4bb4157ebf3b12686336aaa736aec25cebb2b480927f9204979a974598178953',
    'recovery_start_yield/analyze.py': 'c35a6c6587b3707ca599ffd2412357abe791142764439403e81e01252bdc5732',
}
EXPECTED = ['repeat8', 'unique8', 'unique8', 'repeat8']
ARTIFACT = 'recovery-repeat-unique'


def load(name):
    path = BASE/name
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
        raise RuntimeError('Frozen analysis dependency changed: '+name)
    spec = importlib.util.spec_from_file_location('unique_'+path.parent.name+'_analysis', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def group_helpers(repeat):
    source = inspect.getsource(repeat.helpers)
    for old, new in (("endswith('-once')", "endswith('-repeat8')"),
                     ('Nearest once cell', 'Nearest repeat8 cell'),
                     ('Candidate or once raw', 'Candidate or repeat8 raw'),
                     ('(once|repeat8)', '(repeat8|unique8)')):
        source = repeat.replace_once(source, old, new)
    namespace = dict(vars(repeat))
    exec(compile(source, '<unique-group-modes-and-reference>', 'exec'), namespace)
    return namespace['helpers']()


def inherited_actions(repeat, evidence=None):
    source = inspect.getsource(repeat.actions)
    source = repeat.replace_once(source, "('recovery-repeat', data)", "('recovery-repeat-unique', data)")
    source = repeat.replace_once(source, "if passed is None: failures.append('candidate_missing_from_waiting')",
        "if candidate is not None and passed is None: failures.append('candidate_missing_from_waiting')")
    source = repeat.replace_once(source,
        "expected_reasons = ([] if episode not in bypassed else ['EPISODE_ALREADY_BYPASSED'])",
        "expected_reasons = (['NO_UNUSED_REQUEST_IN_LEGAL_SET'] if candidate is None else "
        "[] if episode not in bypassed else ['EPISODE_ALREADY_BYPASSED'])")
    namespace = dict(vars(repeat), LIMITS={'repeat8': 8, 'unique8': 8})
    if evidence is not None: namespace['request_evidence'] = evidence
    exec(compile(source, '<unique-null-candidate-action-parser>', 'exec'), namespace)
    return namespace['actions']


def actions(data, raw, cell, allocations, base_actions):
    result = base_actions(data, raw, cell, allocations)
    if result['status'] == 'UNAVAILABLE': return result
    failures, missing = result['failed_checks'][:], result['missing'][:]
    used, episodes = set(), set()
    counts = Counter(); rows = {row['event_index']: row for row in result['rows']}
    mode = cell['mode']
    required = ('candidates', 'repeat8_suggestion', 'unique8_suggestion',
                'repeat8_suggestion_num_preemptions', 'repeat8_would_execute',
                'used_request_ids_before', 'used_request_ids_after')
    for opportunity in result['opportunities']:
        index = opportunity['event_index']; event = data['events'][index]
        absent = [key for key in required if key not in event]
        missing.extend('event.'+key for key in absent)
        if absent: continue
        candidates = event['candidates']
        if not isinstance(candidates, list) or any(not isinstance(row, dict) or 'request' not in row or 'eligible' not in row for row in candidates):
            missing.append('event.complete_legal_candidate_records'); continue
        legal = [row['request'] for row in candidates if row['eligible'] is True]
        if len(legal) != len(set(legal)): failures.append('duplicate_legal_candidate')
        before_used = sorted(used)
        if event['used_request_ids_before'] != before_used: failures.append('used_request_ids_before_mismatch')
        repeat_suggestion, unique_suggestion = event['repeat8_suggestion'], event['unique8_suggestion']
        unused = [rid for rid in legal if rid not in used]
        if repeat_suggestion not in legal: failures.append('repeat_suggestion_not_legal')
        if unique_suggestion is None:
            if unused: failures.append('null_unique_suggestion_with_unused_legal_request')
        elif unique_suggestion not in unused: failures.append('unique_suggestion_not_unused_legal_request')
        selected = event['candidate_head']
        if selected != (repeat_suggestion if mode == 'repeat8' else unique_suggestion):
            failures.append('mode_selected_suggestion_mismatch')
        if selected is None and (event['candidate_num_preemptions'] is not None or event['queue_changed'] or event['action'] != 'SHADOW_ONLY'):
            failures.append('invalid_null_candidate_action')
        repeat_episode = (repeat_suggestion, event['repeat8_suggestion_num_preemptions'])
        repeat_row = next((row for row in candidates if row['request'] == repeat_suggestion), {})
        if repeat_row.get('num_preemptions') is None:
            missing.append('repeat_suggestion_recorded_num_preemptions')
        elif repeat_row['num_preemptions'] != repeat_episode[1]:
            failures.append('repeat_suggestion_episode_mismatch')
        would_execute = event['action_count_before'] < 8 and repeat_episode not in episodes
        if type(event['repeat8_would_execute']) is not bool or event['repeat8_would_execute'] != would_execute:
            failures.append('repeat_would_execute_current_state_mismatch')
        completed = opportunity['completed_bypass']; changed = opportunity['queue_order_changed_observed']
        reused = selected in used if selected is not None else False
        if completed:
            if mode == 'unique8' and reused: failures.append('unique_request_bypassed_twice')
            counts['actual_repeated_request_bypasses'] += reused
            used.add(selected); episodes.add((selected, event['candidate_num_preemptions']))
        if event['used_request_ids_after'] != sorted(used): failures.append('used_request_ids_after_mismatch')
        suggestions_differ = repeat_suggestion != unique_suggestion
        suppression = (mode == 'unique8' and selected is None and would_execute
            and event['action'] == 'SHADOW_ONLY' and not changed
            and event['action_skip_reasons'] == ['NO_UNUSED_REQUEST_IN_LEGAL_SET'])
        different_reorder = mode == 'unique8' and completed and selected != repeat_suggestion
        same_state_final = repeat_suggestion if would_execute else event['baseline_head']
        counts['suggestion_disagreement_opportunities'] += suggestions_differ
        counts['suggestion_disagreements_repeat_would_execute'] += suggestions_differ and would_execute
        counts['unique_no_unused_legal_request_opportunities'] += unique_suggestion is None
        counts['actual_different_reorder_count'] += different_reorder
        counts['unique_legal_suppression_count'] += suppression
        detail = dict(legal_candidate_requests=legal, legal_candidate_records=candidates,
            unused_legal_candidate_requests=unused, repeat8_suggestion=repeat_suggestion,
            unique8_suggestion=unique_suggestion,
            repeat8_suggestion_num_preemptions=event['repeat8_suggestion_num_preemptions'],
            repeat8_would_execute=event['repeat8_would_execute'],
            used_request_ids_before=event['used_request_ids_before'], used_request_ids_after=event['used_request_ids_after'],
            selected_request_previously_used=reused, suggestions_differ=suggestions_differ,
            actual_different_reorder=different_reorder, actual_legal_suppression=suppression,
            same_state_repeat8_suggested_final_head=same_state_final,
            observed_final_head=event['final_head'])
        opportunity.update(detail)
        if index in rows: rows[index].update(detail)
    if data.get('used_request_ids') != sorted(used): failures.append('final_used_request_ids_mismatch')
    if len(used) != result['unique_bypassed_requests']: failures.append('unique_request_count_mismatch')
    result.update(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        failed_checks=sorted(set(failures)), missing=sorted(set(missing)), used_request_ids=sorted(used),
        recorded_used_request_ids=data.get('used_request_ids'),
        selection_observations={key: counts[key] for key in (
            'suggestion_disagreement_opportunities', 'suggestion_disagreements_repeat_would_execute',
            'unique_no_unused_legal_request_opportunities', 'actual_repeated_request_bypasses',
            'actual_different_reorder_count', 'unique_legal_suppression_count')},
        effective_unique_execution_difference_count=counts['actual_different_reorder_count']+counts['unique_legal_suppression_count'],
        completed_bypasses_below_budget=result['actual_completed_bypass_count'] < 8)
    return result


def gap_distribution(summary, values):
    result = summary(values); known = sorted(value for value in values if value is not None)
    if known:
        position = (len(known)-1)*.99; low = int(position); high = min(low+1, len(known)-1)
        result['p99'] = known[low]+(known[high]-known[low])*(position-low)
    else: result['p99'] = None
    return result


def analyze_session(session):
    repeat = load('recovery_repeat/analyze.py'); costs = load('recovery_retry_defer/analyze.py')
    arrival = load('recovery_start_yield/analyze.py')
    gc = costs.load('recovery_gc_diag/analyze.py'); storage = costs.load('output_event_compact/analyze.py')
    fit = repeat.FIT; optional = repeat.ORIGINAL_OPTIONAL
    base_actions = inherited_actions(repeat)
    fit.helpers = lambda: group_helpers(repeat)
    fit.optional = lambda directory, name: optional(directory, ARTIFACT if name == 'recovery-fit' else name)
    fit.actions = lambda data, raw, cell, allocations: actions(data, raw, cell, allocations, base_actions)
    result = fit.analyze_session(session); summary = fit.read.__globals__['summary']
    for cell in result['cells']:
        action = cell.pop('recovery_fit_actions'); cell['recovery_repeat_unique_actions'] = action
        summaries = {row['request']: row for row in cell['run_summary']['per_request']}
        for row in action.get('rows', []):
            for evidence in row['request_evidence']:
                evidence['whole_request_recovery_summary'] = summaries.get(evidence['source_request'])
        directory = Path(cell['directory'])
        raw, probe, timing, config = (optional(directory, name) for name in ('raw', 'gc-observation', 'timing', 'config'))
        declared_storage = (config or {}).get('B_output_event_storage')
        cell['expected_output_event_storage'] = declared_storage
        cell['output_event_storage'] = (storage.storage_diagnostic(raw, timing or {}, declared_storage, costs.number)
            if declared_storage in ('compact', 'legacy') else dict(status='UNVERIFIED', missing=['config.B_output_event_storage'],
                record=(raw or {}).get('output_event_storage')))
        cell['gc_diagnostic'] = gc.gc_diagnostic(raw, probe, cell.get('planned', (config or {}).get('requests', 0)))
        cell['gc_generation2'] = storage.generation2(gc, cell['gc_diagnostic'])
        cell['run_summary']['phase_times'].update(request_observation_s=(raw or {}).get('observation_end_s'),
            output_event_materialization_s=((raw or {}).get('output_event_storage') or {}).get('materialization_s'))
        cell['gpu_copy_work_duration_s'] = {direction: cell.get('copy_work', {}).get(direction, {}).get('gpu_elapsed_sum_s')
                                          for direction in ('load', 'store')}
        cell['external_arrival_observations'] = arrival.arrivals(raw, summary, costs.number)
        cell['repeat_unique_policy_evidence'] = {key: value for key, value in (config or {}).items()
            if key == 'B_recovery_repeat_unique' or key.startswith('recovery_repeat_unique_')}
        rows = cell.get('per_request', []); planned = (config or {}).get('requests', cell.get('planned'))
        cell['primary_maxgap'] = dict(planned_denominator=planned, recorded_requests=len(rows),
            missing_planned_request_rows=cell['run_summary'].get('missing_planned_request_rows'),
            exact_complete_distribution=type(planned) is int and len(rows) == planned
            and cell['run_summary'].get('missing_planned_request_rows') == 0 and all(
                row['status'] == 'completed' and row['maxgap_s'] is not None for row in rows),
            observed_closed_gaps=gap_distribution(summary, [row['maxgap_s'] for row in rows]),
            all_request_censored_lower_bounds=gap_distribution(summary, [row['maxgap_lower_bound_s'] for row in rows]),
            semantics='Primary mean/P99/max use all original requests; incomplete gaps remain lower bounds, not final maxima. Missing/no-output values are unknown, never zero. P99 uses the frozen inclusive linear interpolation.')
    by_path = {cell['directory']: cell for cell in result['cells']}
    for pair in result['comparisons']:
        candidate, reference = by_path[pair['candidate']], by_path.get(pair.get('native'))
        exact = pair['status'] == 'AVAILABLE' and reference and all(
            cell['primary_maxgap']['exact_complete_distribution'] for cell in (candidate, reference))
        pair['primary_maxgap_delta_unique8_minus_repeat8_s'] = {key:
            candidate['primary_maxgap']['observed_closed_gaps'][key]-reference['primary_maxgap']['observed_closed_gaps'][key]
            if exact else None for key in ('mean', 'p99', 'maximum')}
        a, b = (cell['external_arrival_observations'].get('arrival_end_to_observation_end_s') if cell else None
                for cell in (candidate, reference))
        pair['drain_delta_unique8_minus_repeat8_s'] = a-b if costs.number(a) and costs.number(b) else None
        pair['execution_observations'] = {role: {key: cell['recovery_repeat_unique_actions'].get(key) for key in (
            'status', 'actual_completed_bypass_count', 'unique_bypassed_requests', 'unique_bypassed_request_episodes',
            'selection_observations', 'effective_unique_execution_difference_count')}
            if cell else None for role, cell in (('unique8', candidate), ('repeat8', reference))}
        pair['storage_observations'] = {role: dict(status=cell['output_event_storage']['status'],
            declared_mode=cell['expected_output_event_storage'], recorded_mode=(cell['output_event_storage'].get('record') or {}).get('mode'))
            if cell else None for role, cell in (('unique8', candidate), ('repeat8', reference))}
        phases = set(candidate['run_summary']['phase_times']) | set((reference or {}).get('run_summary', {}).get('phase_times', {}))
        pair['phase_delta_unique8_minus_repeat8_s'] = {}
        for key in sorted(phases):
            a = candidate['run_summary']['phase_times'].get(key)
            b = reference['run_summary']['phase_times'].get(key) if reference else None
            pair['phase_delta_unique8_minus_repeat8_s'][key] = a-b if costs.number(a) and costs.number(b) else None
        pair['unique8'] = pair.pop('candidate'); pair['repeat8'] = pair.pop('native')
        for old, new in (('aggregate_delta_candidate_minus_native', 'aggregate_delta_unique8_minus_repeat8'),
                         ('copy_work_delta_candidate_minus_native', 'copy_work_delta_unique8_minus_repeat8')):
            if old in pair: pair[new] = pair.pop(old)
        if 'fixed1024_contracts' in pair:
            contracts = pair['fixed1024_contracts']
            contracts['unique8'] = contracts.pop('candidate'); contracts['repeat8'] = contracts.pop('native')
        for row in pair.get('per_request', []):
            row['unique8_status'] = row.pop('candidate_status'); row['repeat8_status'] = row.pop('native_status')
    modes = [cell['mode'] for cell in result['cells']]
    indices = []
    for cell in result['cells']:
        match = re.search(r'cell-(\d+)-cap\d+-', cell['directory'])
        if not match or int(match[1]) not in range(4) or cell['mode'] != EXPECTED[int(match[1])]:
            raise ValueError('Unexpected repeat8/unique8 ABBA cell')
        indices.append(int(match[1]))
    receipt = optional(session, 'receipt') or {}
    result['execution_layout'] = dict(design='REPEAT_EPISODE_VS_UNIQUE_REQUEST_BUDGET8', cell_count=len(modes), modes=modes,
        expected_abba=EXPECTED, complete_abba=modes == EXPECTED and indices == list(range(4)),
        observed_cell_indices=indices, comparison_count=len(result['comparisons']),
        controller_status=receipt.get('status'), controller_error=receipt.get('error'), controller_stop_reason=receipt.get('stop_reason'),
        planned_but_not_started=[dict(cell_index=i, mode=mode) for i, mode in enumerate(EXPECTED) if i not in indices],
        unstarted_semantics='Unstarted cells are not measured or failed scientific requests. Every started cell and its original request accounting remain.')
    result.pop('recovery_fit_semantics')
    result['semantics'] = result['semantics'].replace('native cells', 'repeat8 cells')
    result['recovery_repeat_unique_semantics'] = (
        'Both modes observe the same frozen legal candidate set and both suggestions. Repeat8 permits one bypass per request/preemption episode; '
        'unique8 permits one bypass per request ID. Both total budgets are eight actual bypasses, not a promise of equal executed work. '
        'Unique exhaustion and fewer than eight bypasses are actual execution/work differences, not missing samples. Different actual reorders '
        'and unique legal suppressions are separate. A legal suppression requires unchanged queue, no unused legal ID and repeat8_would_execute '
        'true under the current arm\'s actual budget/episode history. Budget or already-used-episode suggestion differences alone are not suppression. '
        'The alternative suggestion is a same-state decision diagnostic, never an alternate executed trajectory. Only actual queue changes have '
        'request/containing-preemption-episode joins to allocations, LOAD submission/host poll/ACK, schedule, next client output and later preemptions. '
        'All-request failures, unfinished counts, fixed-output contracts/sequences, copy work, external-arrival lag and recovery unions retain frozen '
        'calculations. Maxgap mean/P99/max are the prespecified primary outcomes; TTFT, flow, throughput and drain are costs. Historical joint SLO '
        'is diagnostic only. GC callbacks are wall observations, not CPU time; compact post-end materialization remains in phase/process costs. '
        'Local waits, GC intervals and GPU copy-duration sums are not added as service savings. Nearest-repeat8 run contrasts are descriptive, not '
        'matched-state causal effects; request rows are not independent experimental repetitions.')
    for name in (*PINS, 'recovery_gc_diag/analyze.py', 'output_event_compact/analyze.py', 'recovery_repeat_unique/analyze.py'):
        result['analyzer_sources_sha256'][name] = hashlib.sha256((BASE/name).read_bytes()).hexdigest()
    return result


def self_check():
    repeat = load('recovery_repeat/analyze.py')
    group, _, _ = group_helpers(repeat)
    cells = [dict(directory=f'/UNRUN/cell-{i:02d}-cap256-{mode}/output', status='RAW_UNAVAILABLE') for i, mode in enumerate(EXPECTED)]
    pairs = group.__globals__['comparisons'](cells)
    assert [(p['candidate'], p['native']) for p in pairs] == [(cells[1]['directory'], cells[0]['directory']), (cells[2]['directory'], cells[3]['directory'])]
    summary = repeat.FIT.read.__globals__['summary']
    assert gap_distribution(summary, [0., 1., 2., None])['p99'] == 1.98
    assert gap_distribution(summary, [None])['p99'] is None
    evidence_calls = []
    def evidence(rid, *args):
        evidence_calls.append(rid); return dict(source_request=rid)
    parser = inherited_actions(repeat, evidence)
    for mode in ('repeat8', 'unique8'):
        events = []; used = set(); count = 0
        for index in range(3):
            candidate = 'a' if mode == 'repeat8' else ('a', 'b', None)[index]
            episode = (1, 1, 2)[index] if mode == 'repeat8' else 1 if candidate else None
            changed = index != 1 if mode == 'repeat8' else index < 2
            reasons = ([] if changed else ['EPISODE_ALREADY_BYPASSED'] if mode == 'repeat8' else ['NO_UNUSED_REQUEST_IN_LEGAL_SET'])
            before = ['h', 'a', 'b']; after = [candidate]+[rid for rid in before if rid != candidate] if changed else before[:]
            row = dict(kind='fit_opportunity', host_perf_s=index+1., baseline_head='h', candidate_head=candidate,
                final_head=after[0], queue_changed=changed, waiting_before=before, waiting_after=after,
                candidate_num_preemptions=episode, action='REORDERED' if changed else 'SHADOW_ONLY',
                action_count_before=count, action_count_after=count+int(changed), action_limit=8,
                action_skip_reasons=reasons, repeat8_suggestion='a', unique8_suggestion='a' if index == 0 else
                'b' if mode == 'repeat8' or index == 1 else None,
                repeat8_suggestion_num_preemptions=(1, 1, 2)[index], repeat8_would_execute=index != 1,
                candidates=[dict(request=rid, eligible=rid != 'h', num_preemptions=(1, 1, 2)[index] if rid == 'a' else 1)
                            for rid in before], used_request_ids_before=sorted(used))
            if changed: used.add(candidate); count += 1
            row['used_request_ids_after'] = sorted(used); events.append(row)
        data = dict(mode=mode, status='UNINSTALLED', action_limit=8, action_count=count, shadow_count=3,
                    used_request_ids=sorted(used), events=events)
        raw = dict(measurement_origin_perf_counter_s=0., internal_to_source={rid: rid for rid in ('h', 'a', 'b')})
        result = actions(data, raw, dict(mode=mode), None, parser)
        assert result['status'] == 'ANALYZED', result['failed_checks']
        assert result['actual_completed_bypass_count'] == 2
        assert result['unique_bypassed_requests'] == (1 if mode == 'repeat8' else 2)
        assert len(result['rows']) == 2 and len(result['opportunities']) == 3 and None not in evidence_calls
        assert result['selection_observations']['unique_legal_suppression_count'] == (mode == 'unique8')
        assert result['selection_observations']['actual_different_reorder_count'] == (mode == 'unique8')
        if mode == 'unique8':
            last = data['events'][-1]
            last.update(repeat8_suggestion_num_preemptions=1, repeat8_would_execute=False)
            last['candidates'][1]['num_preemptions'] = 1
            blocked = actions(data, raw, dict(mode=mode), None, parser)
            assert blocked['status'] == 'ANALYZED' and blocked['selection_observations']['unique_legal_suppression_count'] == 0
        data['action_count'] += 1
        assert actions(data, raw, dict(mode=mode), None, parser)['status'] == 'FAIL'
    print('PASS: mode/reference mapping, canonical P99 interpolation, null candidate without request join, per-ID/episode counts, actual reorder versus eligible suppression, mismatched counter. CPU schema fixtures only; no scientific results.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check: self_check()
    if args.session is None:
        if args.self_check: return
        parser.error('--session and --output required')
    if args.output is None: parser.error('--output required')
    if args.output.exists(): raise FileExistsError(args.output)
    result = analyze_session(args.session)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), layout=result['execution_layout'], cells=[dict(
        directory=cell['directory'], **{key: cell['recovery_repeat_unique_actions'].get(key) for key in
        ('status', 'actual_completed_bypass_count', 'unique_bypassed_requests', 'selection_observations')}) for cell in result['cells']])))


if __name__ == '__main__':
    main()
