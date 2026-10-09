#!/usr/bin/env python3
"""All-request ABBA analysis for one original-arrival victim correction."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import analyze_native_group as base
import analyze_native_equal_held_group as equal
import analyze_native_mixed_budget_probe as mixed


KIND = 'PRO6000_NATIVE_ARRIVAL_ONCE'
RULES = ('tail', 'arrival_once', 'arrival_once', 'tail')
PACKAGE = Path(__file__).parent / 'candidate_native_arrival_once_r01'
POLICY_SOURCE = PACKAGE / 'pkg/staged_store_rotation.py'


def verify_choices(decisions, source, executed_sha, block_size=None):
    choose, source_sha = None, None
    if source is not None and source.is_file():
        payload = source.read_bytes()
        source_sha = hashlib.sha256(payload).hexdigest()
        if source_sha == executed_sha:
            function = next((node for node in ast.parse(payload).body
                if isinstance(node, ast.FunctionDef) and node.name == '_arrival_once_choice'), None)
            if function is not None:
                namespace = {'math': math}
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
                choose = namespace['_arrival_once_choice']
    records, consumed, state_known = [], False, True
    proposals = requests = 0
    for index, event in enumerate(decisions):
        item = dict(decision_index=index, step=event.get('step'), rule=event.get('rule'))
        rows = event.get('candidates', [])
        indices = [row.get('index') for row in rows]
        if choose is None:
            item['status'] = 'EXECUTED_POLICY_SOURCE_UNAVAILABLE'
        elif not state_known:
            item['status'] = 'PRIOR_PROPOSAL_STATE_UNVERIFIABLE'
        elif (not rows or any(type(i) is not int for i in indices)
                or indices != list(range(indices[0], indices[-1]+1))
                or type(event.get('unprocessed_suffix_start')) is not int
                or indices[0] != event['unprocessed_suffix_start']
                or rows[-1].get('request') != event.get('native_tail')
                or any(not isinstance(row.get('request'), str) for row in rows)
                or len({row['request'] for row in rows}) != len(rows)):
            item['status'], state_known = 'INVALID_SUFFIX_IDENTITY', False
        elif event.get('rule') not in ('tail', 'arrival_once'):
            item['status'], state_known = 'UNSUPPORTED_RULE', False
        elif (type(event.get('fallback_unknown')) is not bool
                or type(event.get('active_protection_or_phase')) is not bool
                or any(type(row.get('qualified')) is not bool for row in rows)):
            item['status'], state_known = 'REQUIRED_GUARDS_NOT_RECORDED', False
        elif event['fallback_unknown'] != any(not row['qualified'] for row in rows):
            item['status'], state_known = 'INCONSISTENT_RECORDED_QUALIFICATION', False
        else:
            proposed, reason, after = choose(rows, event['fallback_unknown'],
                event['active_protection_or_phase'], consumed)
            proposed_request = next(row['request'] for row in rows if row['index'] == proposed)
            active = event['rule'] == 'arrival_once'
            requested = active and proposed_request != event['native_tail']
            selected = proposed_request if active else event['native_tail']
            expected = dict(mode='ACTIVE' if active else 'SHADOW', proposed_request=proposed_request,
                action_requested=requested, fallback=reason, consumed_before=consumed, consumed_after=after)
            recorded = event.get('arrival_once')
            mismatches = []
            for key, value in expected.items():
                actual = recorded.get(key) if isinstance(recorded, dict) else None
                if (not isinstance(recorded, dict) or key not in recorded or actual != value
                        or type(value) is bool and type(actual) is not bool):
                    mismatches.append('arrival_once.'+key)
            for key, value in dict(selected=selected, unrestricted_selected=selected,
                    changed=requested, current_guard_applied=False, current_guard_eligible=False).items():
                actual = event.get(key)
                if actual != value or type(value) is bool and type(actual) is not bool:
                    mismatches.append(key)
            proposals += int(not consumed and after)
            requests += int(requested)
            consumed = after  # Advance independently recomputed state, never recorded state.
            item.update(status='MISMATCH' if mismatches else 'MATCH', expected_selected=selected,
                        expected_probe=expected, mismatches=mismatches)
        records.append(item)
    known = state_known and choose is not None
    return dict(source=str(source) if source else None, source_sha256=source_sha,
        executed_source_sha256=executed_sha, source_verified=choose is not None,
        status_counts=dict(Counter(row['status'] for row in records)), checks=records,
        recomputed_consumed=consumed if known else None,
        recomputed_proposal_count=proposals if known else None,
        recomputed_action_request_count=requests if known else None,
        recorded_action_request_count=sum(isinstance(e.get('arrival_once'), dict)
            and e['arrival_once'].get('action_requested') is True for e in decisions),
        semantics='SHA-matched helper replay in logged decision order from fresh measurement state. '
                  'Tail consumes the same once-only shadow; only arrival_once requests replacement. '
                  'Consumption is a proposal, including execution failure, not successful preemption. '
                  'No block-size, host-state, output-budget or equal-capacity condition is added.')


def proposal(decisions):
    rows = [(i, event) for i, event in enumerate(decisions)
        if isinstance(event.get('arrival_once'), dict)
        and event['arrival_once'].get('consumed_before') is False
        and event['arrival_once'].get('consumed_after') is True]
    return rows[0] if len(rows) == 1 else (None, None)


def intervention_prefix(reference, candidate, old_raw, new_raw):
    # Explicit field adaptation reuses the existing observed-prefix comparison.
    adapted, decisions = [], []
    for cell in (reference, candidate):
        observation = cell.get('native_victim_observation', {})
        events = observation.get('raw_decisions', [])
        index, _ = proposal(events)
        events = events[:index+1] if index is not None else events
        decisions.append(events)
        adapted.append(dict(cell, native_victim_observation=dict(observation,
            raw_decisions=[dict(e, equal_held_once=e.get('arrival_once')) for e in events])))
    result = equal.intervention_prefix(*adapted, old_raw, new_raw)
    result['proposal_field_mapping'] = 'arrival_once -> equal_held_once for shared prefix reader only'
    arrivals, preemptions, missing = [], [], []
    for events, raw in zip(decisions, (old_raw, new_raw)):
        _, event = proposal(events)
        if event is None:
            return result
        identity = raw.get('internal_to_source', {})
        order, acts, unknown = [], [], 0
        for decision in events:
            rows = decision.get('candidates', [])
            known = all(type(r.get('arrival_time')) in (int, float)
                        and math.isfinite(r['arrival_time']) and type(r.get('index')) is int
                        and identity.get(r.get('request')) is not None for r in rows)
            unknown += int(not known)
            order.append([decision.get('step'), identity.get(decision.get('failed_request')),
                [[identity[r['request']], r['index']] for r in sorted(rows,
                    key=lambda r: (r['arrival_time'], r['index']))] if known else None])
        cutoff = base.delta(raw.get('measurement_origin_perf_counter_s'), event.get('host_perf_counter_s'))
        for preempt in raw.get('preemption_events', []):
            entered = preempt.get('method_entered_s')
            if cutoff is None or not base.finite(entered):
                unknown += 1
            elif entered < cutoff:
                acts.append({key: preempt.get(key) for key in ('request_id', 'engine_call_index',
                    'native_output_count_before', 'original_preemption_called',
                    'original_preemption_returned', 'actual_released_blocks')})
        arrivals.append(order); preemptions.append(acts); missing.append(unknown)
    result['original_arrival_order_prefix'] = dict(counts=[len(x) for x in arrivals],
        exact_equal=arrivals[0] == arrivals[1], sha256=[base.digest(x) for x in arrivals],
        semantics='Actual candidate arrival_time/index rank, mapped to source IDs; epoch values remain in raw decisions.')
    result['actual_preemption_prefix'] = dict(counts=[len(x) for x in preemptions],
        exact_equal=preemptions[0] == preemptions[1], sha256=[base.digest(x) for x in preemptions],
        reference=preemptions[0], candidate=preemptions[1],
        semantics='Calls entered strictly before each own proposal clock, including earlier calls in the same engine step. No claim of equal hidden state.')
    result['additional_unknown_counts'] = missing
    if any(missing):
        result['status'] = 'INCOMPLETE_OBSERVATION'
    return result


def once_outcome(cell, raw, store):
    decisions = store.get('victim_decisions', [])
    index, event = proposal(decisions)
    if event is None or raw is None:
        return dict(status='NO_UNIQUE_PROPOSAL_OR_RAW')
    joins = [row for row in cell['native_victim_observation']['actual_release'].get('joins', [])
             if row.get('decision_index') == index]
    joined = joins[0] if len(joins) == 1 else {}
    requested = event['arrival_once'].get('action_requested') is True
    executed = (requested and event.get('selected') != event.get('native_tail')
                and joined.get('join_status') == 'UNIQUE')
    cutoff = joined.get('actual_preemption', {}).get('method_returned_s') if executed else None
    later, unknown = [], 0
    if executed:
        for preempt in raw.get('preemption_events', []):
            if preempt.get('original_preemption_called') is not True or preempt.get('original_preemption_returned') is not True:
                continue
            at = preempt.get('method_entered_s')
            if not base.finite(cutoff) or not base.finite(at):
                unknown += 1
            elif at >= cutoff and preempt.get('internal_request_id') == event.get('native_tail'):
                later.append(preempt)
    tail_event = dict(event, selected=event.get('native_tail'))
    tail_outcome = base.native_event_outcomes([tail_event], raw, store.get('residency_admissions'))
    return dict(status='ANALYZED', decision_index=index, step=event.get('step'),
        original_tail=event.get('native_tail'), proposed_request=event['arrival_once'].get('proposed_request'),
        action_requested=requested, action_execution_confirmed=executed,
        execution_join_status=joined.get('join_status', 'MISSING_OR_AMBIGUOUS'),
        actual_released_blocks=joined.get('actual_released_blocks') if executed else None,
        original_tail_later_successful_preemptions=later if executed else None,
        original_tail_later_preemption_count=len(later) if executed and unknown == 0 else None,
        unknown_successful_preemption_clocks=unknown if executed else None,
        original_tail_observed_outcome=tail_outcome.get('per_decision', [{}])[0].get('victim'),
        semantics='Execution uses the inherited unique step+internal-ID successful-preempt join. '
                  'Later original-tail preemption is observed after that method returned, including the same step. '
                  'Original-tail output/completion waits are measured from the proposal, even when it was not selected; '
                  'they are not a counterfactual recovery time. Other request effects and selected-victim waits remain '
                  'in inherited all-request comparisons/native_event_outcomes. Released capacity may differ substantially.')


def analyze(session, timeout_s=600, policy_source=POLICY_SOURCE, input_config=None):
    policy_source = Path(policy_source) if policy_source is not None else None
    input_config = Path(input_config) if input_config is not None else PACKAGE / 'pkg/inputs/pro_high/config.json'
    previous = base.KIND, base.RULES, base.verify_choices, base.read_native_cell, base.compare_cells
    original_read, original_compare = base.read_native_cell, base.compare_cells

    def read_cell(directory, spec, plan, timeout, source):
        cell, raw = original_read(directory, spec, plan, timeout, source)
        archive = base.archive_dir(directory)
        store = base.optional(archive / 'selective-store.json', {})
        choice = cell['executed_choice_check']
        checks = cell['configuration_check']['checks']
        expected = dict(mode='ACTIVE' if spec['native_victim_rule'] == 'arrival_once' else 'SHADOW',
            consumed=choice['recomputed_consumed'], proposal_count=choice['recomputed_proposal_count'],
            action_request_count=choice['recomputed_action_request_count'])
        recorded = store.get('arrival_once_probe') or {}
        for key, value in expected.items():
            actual = recorded.get(key)
            checks.append(dict(field='arrival_once_probe.'+key, actual=actual, expected=value,
                matches=value is not None and actual == value and type(actual) is type(value)))
        checks.append(dict(field='arrival_once_helper_source', actual=choice['source_verified'],
                           expected=True, matches=choice['source_verified']))
        try:
            executed_input = directory / 'candidate_native_oldest_strong_r01/pkg/inputs/pro_high/config.json'
            config_path = executed_input if executed_input.is_file() else input_config
            config, workload = base.read(config_path), base.read(config_path.with_name('workload.json'))
            budget = mixed.budget_check(config, workload, cell.get('config') or {}, raw)
            budget.update(input_config_path=str(config_path), input_config_sha256=mixed.sha(config_path),
                          input_workload_sha256=mixed.sha(config_path.with_name('workload.json')))
            cell['budget_consistency'] = budget
            checks.append(dict(field='mixed_budget_assignment', actual=budget['status'], expected='VERIFIED',
                               matches=budget['status'] == 'VERIFIED'))
            cell['arrival_once_outcome'] = once_outcome(cell, raw, store)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            cell['arrival_once_analysis_error'] = f'{type(error).__name__}: {error}'
            checks.append(dict(field='arrival_once_analysis', actual=str(error), expected='COMPLETE', matches=False))
        cell['configuration_check']['status'] = ('VERIFIED' if all(c['matches'] for c in checks)
                                                  else 'MISMATCH_OR_MISSING')
        return cell, raw

    def compare(reference, candidate, old_raw, new_raw):
        result = original_compare(reference, candidate, old_raw, new_raw)
        result['pre_intervention_prefix'] = intervention_prefix(reference, candidate, old_raw, new_raw)
        _, event = proposal(candidate.get('native_victim_observation', {}).get('raw_decisions', []))
        if event is not None:
            identity = new_raw.get('internal_to_source', {})
            participants = {name: identity.get(event.get(field)) for name, field in
                (('original_tail', 'native_tail'), ('selected_victim', 'selected'),
                 ('allocation_failed_request', 'failed_request'))}
            result['intervention_participants'] = dict(source_request_ids=participants,
                all_request_comparison_indices=[i for i, row in enumerate(result['per_request'])
                                                if row['request_id'] in participants.values()],
                semantics='Indices into inherited per_request differences, including who finishes later; roles are policy-dependent, not matched causal events.')
        return result

    try:
        base.KIND, base.RULES, base.verify_choices = KIND, RULES, verify_choices
        base.read_native_cell, base.compare_cells = read_cell, compare
        result = base.analyze(Path(session), timeout_s, policy_source)
    finally:
        base.KIND, base.RULES, base.verify_choices, base.read_native_cell, base.compare_cells = previous
    result['analysis_code_sha256'] = {Path(module.__file__).name: mixed.sha(Path(module.__file__))
        for module in (base, equal, mixed)}
    result['analysis_code_sha256'][Path(__file__).name] = mixed.sha(Path(__file__))
    result['semantics'] = dict(result['semantics'], intervention=
        'First legal original-arrival inversion only, then tail permanently; protection/unknown state falls back. '
        'Mixed assigned budgets are 40x128 and 280x1024 with natural EOS. This is not an equal-capacity '
        'or recovery-cost experiment: extra released pages are retained explicitly, and observed prefix '
        'differences prohibit assuming identical pre-action state.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=600)
    parser.add_argument('--policy-source', type=Path, default=POLICY_SOURCE)
    parser.add_argument('--input-config', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.session, args.timeout_s, args.policy_source, args.input_config)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
