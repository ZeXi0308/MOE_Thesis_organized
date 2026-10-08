"""Validate native budget observations and summarize formal E policy episodes.

Uses analyze.py for service, work, transfer and output-consistency accounting.
Rule replay on an observed trajectory predicts actions, never counterfactual cost.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from analyze import aggregate_policies, analyze_group, event_key, read_json, stats


EXPECTED_ARMS = 'host,recompute,length,budget,budget,length,recompute,host'


def integer(value):
    return type(value) is int and value >= 0


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_budget_episode(raw):
    observation = raw.get('target_selection', {}).get('budget_observation')
    if observation is None:
        return dict(status='UNRUN', errors=[], reason='budget_observation_missing')
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    if not isinstance(observation, dict) or not isinstance(observation.get('steps'), list):
        return dict(status='INVALID', errors=['malformed_budget_observation'])
    budgets, schedules, steps = observation['steps'], raw.get('scheduler_steps', []), raw.get('steps', [])
    if not all(isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
               for rows in (budgets, schedules, steps)):
        return dict(status='INVALID', errors=['malformed_budget_or_schedule_steps'])
    if any(not isinstance(row.get('scheduled'), list) or
           any(not isinstance(entry, dict) for entry in row['scheduled']) for row in schedules):
        return dict(status='INVALID', errors=['malformed_native_schedule_entries'])
    require(bool(budgets), 'empty_budget_observation')
    require(observation.get('signal') == 'native_remaining_scheduled_token_budget', 'unknown_budget_signal')
    require(len(budgets) == len(schedules), 'budget_and_schedule_count_differ')
    require(observation.get('checked_steps') == sum(x.get('checked') is True for x in budgets), 'checked_counter_mismatch')
    require(observation.get('error_steps') == sum(x.get('status') == 'ERROR' for x in budgets), 'error_counter_mismatch')
    require(observation.get('error_steps') == 0, 'observer_reported_errors')
    main_counts = [0] * len(steps)
    drain_count, total_tokens = 0, 0
    spans = []
    last_end = -math.inf
    for index, (budget, schedule) in enumerate(zip(budgets, schedules)):
        tag = f'step[{index}]'
        require(budget.get('step') == index, f'{tag}: noncontiguous_index')
        require(budget.get('checked') is True and budget.get('status') == 'CHECKED', f'{tag}: not_checked')
        entries = schedule.get('scheduled', [])
        counts = [entry.get('count') for entry in entries]
        require(all(integer(count) and count > 0 for count in counts), f'{tag}: invalid_native_counts')
        require(len({entry.get('request_id') for entry in entries}) == len(entries), f'{tag}: duplicate_native_request')
        total = sum(counts) if all(integer(count) for count in counts) else None
        for field in ('initial_budget_tokens', 'remaining_budget_tokens', 'allocated_compute_tokens',
                      'native_scheduled_tokens_total', 'positive_allocation_requests', 'allocation_calls',
                      'failed_allocations', 'zero_token_allocations'):
            require(integer(budget.get(field)), f'{tag}: invalid_{field}')
        require(budget.get('native_scheduled_tokens_total') == total == budget.get('allocated_compute_tokens'),
                f'{tag}: native_or_allocated_total_mismatch')
        require(budget.get('positive_allocation_requests') == len(entries), f'{tag}: positive_request_count_mismatch')
        if all(integer(budget.get(field)) for field in ('initial_budget_tokens', 'remaining_budget_tokens')) and total is not None:
            require(budget['initial_budget_tokens'] - total == budget['remaining_budget_tokens'], f'{tag}: remaining_identity_mismatch')
        fields = ('positive_allocation_requests', 'failed_allocations', 'zero_token_allocations', 'allocation_calls')
        if all(integer(budget.get(field)) for field in fields):
            require(sum(budget[field] for field in fields[:3]) == budget['allocation_calls'], f'{tag}: allocation_call_count_mismatch')
        if budget.get('pause_state') == 'PAUSED_ALL':
            require(budget.get('initial_budget_tokens') == 0, f'{tag}: paused_all_nonzero_budget')
        else:
            require(budget.get('pause_state') in ('UNPAUSED', 'PAUSED_NEW'), f'{tag}: unknown_pause_state')
        start, end, observed = budget.get('start_s'), budget.get('end_s'), schedule.get('time_s')
        if not all(finite(value) for value in (start, end, observed)):
            errors.append(f'{tag}: invalid_timestamps')
            continue
        require(last_end <= start <= end <= observed + 1e-9, f'{tag}: schedule_time_order')
        last_end = end
        spans.append(end - start)
        containing = [i for i, step in enumerate(steps)
                      if finite(step.get('start_s')) and finite(step.get('end_s'))
                      and step['start_s'] <= start and observed <= step['end_s'] + 1e-9]
        if containing:
            require(len(containing) == 1, f'{tag}: overlapping_main_step_intervals')
            main_counts[containing[0]] += 1
        else:
            complete = raw.get('all_complete_s')
            require(finite(complete) and start >= complete, f'{tag}: outside_main_steps_before_drain')
            require(total == 0, f'{tag}: drain_scheduled_compute')
            drain_count += 1
        total_tokens += total or 0
    require(all(count == 1 for count in main_counts), 'main_step_without_exactly_one_schedule')
    return dict(status='INVALID' if errors else 'VALID', errors=errors,
        observer_steps=len(budgets), scheduler_steps=len(schedules), main_steps=len(steps),
        main_step_schedule_counts=main_counts, drain_schedule_steps=drain_count,
        scheduled_native_tokens=total_tokens, native_schedule_plus_wrapper_span_s=stats(spans),
        interpretation='Offline checks align observer aggregates with each native schedule and main service step; per-ID allocation equality was asserted inside the runtime. Schedule spans include native schedule plus wrapper and do not isolate instrumentation overhead.')


def replay_decisions(raw, policy, length_threshold=1792):
    errors, events, commits = [], [], []
    observation = raw.get('target_selection', {}).get('budget_observation')
    if observation is None:
        return dict(status='UNRUN', errors=[], reason='budget_observation_missing', events=[], commits=[])
    if not isinstance(observation, dict) or not isinstance(observation.get('steps'), list) or any(
            not isinstance(step, dict) for step in observation['steps']):
        return dict(status='INVALID', errors=['malformed_budget_observation'], events=[], commits=[])
    if not all(isinstance(raw.get(field, []), list) and
               all(isinstance(row, dict) for row in raw.get(field, [])) for field in ('decisions', 'commits')):
        return dict(status='INVALID', errors=['malformed_decisions_or_commits'], events=[], commits=[])
    budgets = observation.get('steps', [])
    event_index = {}
    previous_signal = {}
    def replay(row, kind):
        if not row.get('eligible'):
            return None
        known, remaining, step = (row.get(key) for key in ('known_tokens', 'remaining_budget_tokens', 'budget_step'))
        if not (integer(known) and known > 0 and integer(remaining) and integer(step) and step < len(budgets)):
            errors.append(f'{kind} {event_key(row)!r}: invalid_decision_signal')
            return None
        budget = budgets[step]
        minimum, maximum = budget.get('remaining_budget_tokens'), budget.get('initial_budget_tokens')
        if not (integer(minimum) and integer(maximum) and minimum <= remaining <= maximum):
            errors.append(f'{kind} {event_key(row)!r}: signal_outside_step_budget_bounds')
        decision = row.get('decision_s')
        if not (finite(decision) and finite(budget.get('start_s')) and finite(budget.get('end_s'))
                and budget['start_s'] <= decision <= budget['end_s']):
            errors.append(f'{kind} {event_key(row)!r}: decision_outside_schedule')
        b = 'recompute' if known <= remaining else 'host'
        length = 'recompute' if known <= length_threshold else 'host'
        selected = row.get('action')
        if policy == 'budget' and selected != b:
            errors.append(f'{kind} {event_key(row)!r}: actual_budget_action_mismatch')
        if policy == 'length' and selected != length:
            errors.append(f'{kind} {event_key(row)!r}: actual_length_action_mismatch')
        if kind == 'commit' and row.get('actual_action') != selected:
            errors.append(f'{kind} {event_key(row)!r}: selected_vs_committed_action_mismatch')
        return dict(request_id=row.get('request_id'), event=row.get('event'), budget_step=step,
            decision_s=decision, known_tokens=known, remaining_budget_tokens=remaining,
            budget_prediction=b, length_prediction=length, predictions_disagree=b != length,
            selected_action=selected, actual_action=row.get('actual_action') if kind == 'commit' else None)
    for row in raw.get('decisions', []):
        key = event_key(row)
        if key in event_index:
            errors.append(f'duplicate_decision_key: {key!r}')
        event_index[key] = row
        item = replay(row, 'event')
        if item is not None:
            step, remaining = item['budget_step'], item['remaining_budget_tokens']
            if remaining > previous_signal.get(step, math.inf):
                errors.append(f'event {key!r}: remaining_budget_increased')
            previous_signal[step] = remaining
            events.append(item)
    committed_keys = set()
    for row in raw.get('commits', []):
        key = event_key(row)
        if key in committed_keys:
            errors.append(f'duplicate_commit_key: {key!r}')
        committed_keys.add(key)
        source = event_index.get(key)
        if source is None:
            errors.append(f'orphan_commit: {key!r}')
        elif any(source.get(field) != row.get(field) for field in
                 ('eligible', 'known_tokens', 'remaining_budget_tokens', 'budget_step', 'action', 'decision_s')):
            errors.append(f'commit_decision_state_mismatch: {key!r}')
        item = replay(row, 'commit')
        if item is not None:
            commits.append(item)
    def summarize(rows):
        return dict(eligible_records=len(rows), budget_vs_length_disagreements=sum(x['predictions_disagree'] for x in rows),
            budget_predictions=dict(Counter(x['budget_prediction'] for x in rows)),
            length_predictions=dict(Counter(x['length_prediction'] for x in rows)),
            actual_budget_vs_length_disagreements=sum(x['selected_action'] != x['length_prediction'] for x in rows) if policy == 'budget' else None,
            actual_length_vs_budget_disagreements=sum(x['selected_action'] != x['budget_prediction'] for x in rows) if policy == 'length' else None)
    return dict(status='INVALID' if errors else 'VALID', errors=errors, length_threshold=length_threshold,
        event_summary=summarize(events), commit_summary=summarize(commits), events=events, commits=commits,
        interpretation='Action replay uses this arm\'s decision-time state. Arms follow different trajectories; disagreement counts are not paired action outcomes or counterfactual performance.')


def service_metrics(cell):
    fields = ('makespan_s', 'service_and_drain_s', 'completion_latency_s', 'token_gap_s', 'ttft_s',
        'output_tokens_per_s', 'output_tokens', 'output_work', 'scheduled_native_tokens', 'transfers',
        'preemptions', 'eligible_decisions', 'eligible_commits', 'decision_to_next_output_s',
        'eligible_decision_to_next_output_s', 'recovery_by_action', 'censored_commits',
        'commits_sharing_output_count', 'committed_fallbacks', 'prefix_consistency_reported',
        'eligible_request_state_preserved_count', 'lookup_overhead_s', 'adapter_overhead_s')
    result = {key: cell[key] for key in fields}
    result['scheduled_position_totals'] = cell['scheduled_position_counts']['totals']
    return result


def analyze_budget_group(group, length_threshold=1792, expected_arms=EXPECTED_ARMS):
    group = Path(group).resolve()
    base = analyze_group(group)
    cells, valid, errors = [], [], []
    for cell in base['cells']:
        raw_path = Path(cell['raw_path'])
        raw = read_json(raw_path)
        warm = read_json(raw_path.parent / 'warmup.json')
        checks = validate_budget_episode(raw)
        warm_checks = validate_budget_episode(warm) if warm is not None else dict(status='UNRUN', errors=[], reason='warmup_missing')
        replay = replay_decisions(raw, cell['policy'], length_threshold)
        statuses = [checks['status'], warm_checks['status'], replay['status']]
        status = 'INVALID' if 'INVALID' in statuses else ('UNRUN' if 'UNRUN' in statuses else 'VALID')
        if replay['status'] != 'UNRUN' and cell['policy'] == 'length' and cell['threshold'] != length_threshold:
            status = 'INVALID'
            errors.append(f"{cell['cell']}: executed length threshold differs from requested replay threshold")
        if checks.get('scheduled_native_tokens') is not None and checks['scheduled_native_tokens'] != cell['scheduled_native_tokens']:
            status = 'INVALID'
            errors.append(f"{cell['cell']}: budget total differs from formal analyzer native work")
        item = dict(cell=cell['cell'], policy=cell['policy'], status=status,
            budget_validation=checks, warmup_budget_validation=warm_checks, action_replay=replay,
            formal_metrics=service_metrics(cell) if status == 'VALID' else None,
            fingerprints=cell['fingerprints'], warnings=cell['warnings'])
        cells.append(item)
        if status == 'VALID':
            valid.append(cell)
    valid_names = {x['cell'] for x in valid}
    expected = expected_arms.split(',') if expected_arms else []
    actual = [x['policy'] for x in base['cells']]
    order_matches = actual == expected if expected else None
    repeats = []
    if order_matches:
        for first in range(len(cells) // 2):
            last = len(cells) - 1 - first
            left, right = cells[first], cells[last]
            if left['policy'] != right['policy']:
                continue
            repeats.append(dict(policy=left['policy'], first_cell=left['cell'], reverse_repeat_cell=right['cell'],
                first_metrics=left['formal_metrics'], reverse_repeat_metrics=right['formal_metrics'],
                interpretation='Two observed runs in opposite order; descriptive repetition, not independent event trials.'))
    comparisons = [x for x in base['greedy_token_consistency']
                   if x['left_cell'] in valid_names and x['right_cell'] in valid_names]
    any_observation = any(x['budget_validation']['status'] != 'UNRUN' for x in cells)
    invalid = any(x['status'] == 'INVALID' for x in cells) or bool(errors or base['analysis_errors'] or base['invalid_cells'])
    complete = (base.get('group_status') or {}).get('status') == 'COMPLETE'
    status = 'INVALID' if invalid else ('UNRUN' if not any_observation else
             ('VALIDATED_COMPLETE' if complete and order_matches and len(valid) == len(expected) else 'INCOMPLETE'))
    return dict(schema='E.budget_analysis.v1', group=str(group), status=status,
        group_status=base['group_status'], expected_order=expected, observed_formal_order=actual,
        order_matches=order_matches, formal_cells=len(base['cells']), observer_validated_cells=len(valid),
        interpretation='No performance evidence for the budget policy is available.' if status == 'UNRUN' else
            'Descriptive formal service results; no causal event matching or adaptive-benefit conclusion is generated.',
        definitions=dict(base_accounting='Reuses analyze.py analyze_group; only COMPLETE cells with all requests completed are formal.',
            reporting='Formal service metrics are exposed only for cells whose measured and warmup budget observations validate.',
            timing='Observer start/end spans include native scheduling and wrapper. Existing adapter_seconds covers only the local lookup adapter; complete instrumentation overhead is not separated.',
            thresholds='Only length uses the declared length threshold. Budget has no fitted threshold even if the shared config carries the length threshold field.',
            work='STORE stays in full end-to-end accounting. LOAD/STORE CUDA durations may overlap; never add them to infer wall time.',
            quality='Output equality is a consistency diagnostic, not task quality. Different natural output lengths prevent an equal-output-work speedup claim.',
            replay='B and L predictions are evaluated only on current observed known prefix and remaining budget; cross-arm trajectories are not counterfactual outcomes.'),
        cells=cells, policy_summary=aggregate_policies(valid), reverse_repeats=repeats,
        greedy_token_consistency=comparisons, incomplete_cells=base['incomplete_cells'],
        invalid_cells=base['invalid_cells'], analysis_errors=base['analysis_errors'], budget_errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--length-threshold', type=int, default=1792)
    parser.add_argument('--expected-arms', default=EXPECTED_ARMS)
    args = parser.parse_args()
    if not args.group.is_dir() or args.length_threshold < 0:
        parser.error('existing group directory and nonnegative length threshold are required')
    result = analyze_budget_group(args.group, args.length_threshold, args.expected_arms)
    output = args.out or args.group / 'budget_summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], formal_cells=result['formal_cells'],
                         observer_validated_cells=result['observer_validated_cells'], output=str(output)), ensure_ascii=False))
    return 0 if result['status'] == 'VALIDATED_COMPLETE' else (1 if result['status'] == 'INVALID' else 2)


if __name__ == '__main__':
    raise SystemExit(main())
