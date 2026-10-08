#!/usr/bin/env python3
"""Compact unadjusted run summaries and original nearest-native contrasts for one/two groups."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def delta(candidate, native):
    return candidate-native if finite(candidate) and finite(native) else None


def summary(value):
    return {key: value.get(key) for key in ('n', 'missing', 'mean', 'p95', 'maximum')} if isinstance(value, dict) else None


def cell_name(directory):
    return next((part for part in reversed(Path(directory).parts) if part.startswith('cell-')), None)


def selected_request(row, role, per_request):
    identity = row.get(role, {})
    matches = [entry for entry in row.get('request_evidence', [])
               if entry.get('internal_request') == identity.get('internal_request')]
    if len(matches) != 1:
        return dict(status='UNAVAILABLE_OR_AMBIGUOUS', identity=identity)
    evidence = matches[0]
    request = per_request.get(evidence.get('source_request'), {})
    recovery = evidence.get('recovery_to_next_output') or {}
    allocation = evidence.get('first_native_allocation_attempt') or {}
    return dict(source_request=evidence.get('source_request'), request_status=evidence.get('request_status'),
        decision_to_next_output_s=evidence.get('decision_to_next_output_s'),
        decision_to_output_wait_lower_bound_s=evidence.get('decision_to_output_wait_lower_bound_s'),
        recovery_to_next_output_s=recovery.get('demand_to_next_output_s'),
        repeated_preempts_before_output=recovery.get('repeated_preempts_before_output'),
        first_native_allocation_outcome=allocation.get('outcome'),
        decision_to_first_allocation_s=allocation.get('decision_to_begin_s'),
        decision_to_schedule_plan_s=evidence.get('decision_to_schedule_plan_s'),
        load_submitted_count=evidence.get('load_submitted_count'),
        load_acknowledged_count=evidence.get('load_acknowledged_count'),
        recovery_union_lower_bound_s=request.get('cumulative_recovery_union_lower_bound_s'),
        recovery_union_exact=request.get('cumulative_recovery_union_exact'),
        total_preemptions=request.get('preemptions'), flow_s=request.get('flow_s'),
        maxgap_s=request.get('maxgap_s'), outputs=request.get('outputs'))


def compact_run(session, cell, timing):
    run = cell.get('run_summary', {})
    action = cell.get('recovery_fit_actions', {})
    per_request = {request['request']: request for request in run.get('per_request', [])}
    context = {key: value for key, value in (timing or {}).items() if key != 'per_request'}
    context['latest_first_output_minus_decision_s'] = delta(
        context.get('latest_first_output_s'), context.get('decision_s'))
    context['all_planned_first_outputs_precede_decision'] = (
        context.get('first_outputs_before_decision') == cell.get('planned')
        and finite(context.get('latest_first_output_s')) and finite(context.get('decision_s'))
        and context['latest_first_output_s'] < context['decision_s']) if timing else None
    decisions = []
    for row in action.get('rows', []):
        decisions.append(dict(decision_s=row.get('decision_s'),
            actual_candidate_intervention=row.get('actual_candidate_intervention'),
            successor=selected_request(row, 'candidate_head', per_request),
            original_head=selected_request(row, 'baseline_head', per_request)))
    recovery = run.get('recovery', {})
    return dict(run=session.name+'/'+cell_name(cell['directory']), mode=cell.get('mode'),
        status=cell.get('status'), error=cell.get('error'), planned=cell.get('planned'),
        completed=cell.get('completed'), failed=cell.get('failed'), unfinished=cell.get('unfinished'),
        statuses=cell.get('statuses'), missing_planned_rows=run.get('missing_planned_request_rows'),
        outputs=cell.get('outputs'), fixed1024_status=run.get('fixed1024_contract', {}).get('status'),
        duration_s=cell.get('duration_s'), token_throughput_per_s=cell.get('output_tokens_per_s'),
        request_throughput_per_s=cell.get('throughput_rps'),
        drain_from_last_external_arrival_s=context.get('drain_from_last_external_arrival_s'),
        observed={key: summary(cell.get('observed', {}).get(key)) for key in ('ttft_s', 'flow_s', 'maxgap_s')},
        all_request_lower_bounds={key: summary(value) for key, value in cell.get('all_request_lower_bounds', {}).items()},
        joint_slo=cell.get('joint_slo'), copies=cell.get('copy_work'),
        actions=dict(status=action.get('status'), suggested=action.get('suggested_decisions'),
            queue_changes_reported=action.get('queue_change_reported_count'),
            queue_changes_observed=action.get('queue_order_change_observed_count'),
            actual_changes=action.get('actual_candidate_intervention_count'), decisions=decisions),
        recovery=dict(episodes=recovery.get('episodes'), affected_requests=recovery.get('affected_requests'),
            repeated_requests=recovery.get('repeated_requests'),
            all_request_union_lower_bound_s=summary(recovery.get('all_request_cumulative_union_lower_bound_s'))),
        timing_context=context if timing else dict(status='UNAVAILABLE'))


def compare(sessions):
    result = dict(groups=[], runs=[], contrasts=[], semantics=(
        'All supplied runs retained, including the first r01 native run. No endpoints adjusted, '
        'no run excluded, no request-level significance or confidence interval. Original nearest-native '
        'pair mappings and their scalar differences are copied unchanged. Observed distributions and '
        'all-request lower bounds are distinct; failures/unfinished/missing remain in counts and SLO denominators. '
        'Closed pre-decision gaps exclude gaps crossing the decision. Different decision times are not '
        'matched causal prefixes; earlier TTFT/gap differences cannot be caused by the later queue change. '
        'Successor/head chains are within-run observations, not matched-state counterfactuals. GPU copy '
        'duration sums are work diagnostics, not request savings. These are same-load development groups.'))
    for session in sessions:
        metric_path, timing_path = session/'recovery-fit-metrics.json', session/'timing-context.json'
        metric_bytes, timing_bytes = metric_path.read_bytes(), timing_path.read_bytes()
        data, timing = json.loads(metric_bytes), json.loads(timing_bytes)
        metric_sha = hashlib.sha256(metric_bytes).hexdigest()
        if timing.get('input_sha256') != metric_sha:
            raise ValueError('Timing context was not derived from supplied metrics: '+str(session))
        contexts = {cell['cell']: cell for cell in timing['cells']}
        if len(contexts) != len(timing['cells']):
            raise ValueError('Duplicate timing cell identities: '+str(session))
        group_runs = [compact_run(session, cell, contexts.get(cell_name(cell['directory']))) for cell in data['cells']]
        by_directory = {cell['directory']: run for cell, run in zip(data['cells'], group_runs)}
        result['groups'].append(dict(session=str(session), execution_layout=data.get('execution_layout'),
            run_count=len(group_runs), original_contrast_count=len(data['comparisons']),
            metrics_sha256=metric_sha, timing_sha256=hashlib.sha256(timing_bytes).hexdigest()))
        result['runs'].extend(group_runs)
        for pair in data['comparisons']:
            candidate, native = [by_directory.get(pair.get(key)) for key in ('candidate', 'native')]
            row = {key: pair.get(key) for key in ('status', 'reference_rule', 'aggregate_delta_candidate_minus_native',
                'observed_summary_delta', 'copy_work_delta_candidate_minus_native', 'metric_counts', 'output_sequences')}
            row.update(candidate=candidate['run'] if candidate else pair.get('candidate'),
                       native=native['run'] if native else pair.get('native'))
            if candidate and native:
                row['drain_delta_candidate_minus_native_s'] = delta(
                    candidate['drain_from_last_external_arrival_s'], native['drain_from_last_external_arrival_s'])
                row['joint_slo_pass_delta_candidate_minus_native'] = delta(
                    (candidate.get('joint_slo') or {}).get('passes'), (native.get('joint_slo') or {}).get('passes'))
                cflow, nflow = [run['observed']['flow_s']['mean'] for run in (candidate, native)]
                row['flow_mean_delta_percent'] = 100*(cflow/nflow-1) if finite(cflow) and finite(nflow) and nflow else None
            result['contrasts'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, nargs='+', required=True, help='One or two completed analysis directories')
    parser.add_argument('--output', type=Path, help='Optional new JSON; stdout always gives a compact run/pair readout')
    args = parser.parse_args()
    sessions = [path.resolve() for path in args.session]
    if not 1 <= len(sessions) <= 2 or len(set(sessions)) != len(sessions):
        parser.error('Supply one or two distinct session directories')
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    result = compare(sessions)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    for run in result['runs']:
        print(json.dumps(dict(run=run['run'], actions=run['actions']['actual_changes'],
            completed=run['completed'], planned=run['planned'], observed=run['observed'],
            goodput=run['joint_slo'], token_throughput_per_s=run['token_throughput_per_s'],
            drain_s=run['drain_from_last_external_arrival_s'],
            pre_decision_gap_failures=run['timing_context'].get('requests_already_failing_fixed_gap_slo'),
            latest_first_output_minus_decision_s=run['timing_context'].get('latest_first_output_minus_decision_s')),
            allow_nan=False))
    for pair in result['contrasts']:
        print(json.dumps(dict(candidate=pair['candidate'], native=pair['native'],
            observed_summary_delta=pair['observed_summary_delta'],
            flow_mean_delta_percent=pair.get('flow_mean_delta_percent'),
            aggregate_delta=pair['aggregate_delta_candidate_minus_native']), allow_nan=False))
    print(json.dumps(dict(output=str(args.output) if args.output else None,
        runs=len(result['runs']), original_nearest_native_contrasts=len(result['contrasts']))))


if __name__ == '__main__':
    main()
