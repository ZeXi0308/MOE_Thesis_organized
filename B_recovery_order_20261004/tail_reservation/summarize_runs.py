#!/usr/bin/env python3
"""Compact descriptive summaries of immutable metrics and their recorded inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze import read, summary


def fixed_contract(config, raw):
    if config is None:
        return dict(status='UNAVAILABLE', reason='Configuration missing')
    if config.get('output_mode') != 'fixed':
        return dict(status='NOT_APPLICABLE', output_mode=config.get('output_mode'))
    if raw is None:
        return dict(status='UNAVAILABLE', reason='Fixed output declared, raw missing')
    requests = raw['requests']; expected = config.get('requests')
    bad = [r['request_id'] for r in requests if r.get('status') != 'completed'
        or len(r.get('output_token_ids', [])) != 1024 or len(r.get('token_times_s', [])) != 1024
        or r.get('max_output_tokens') != 1024 or r.get('stop_reason') != 'length']
    checks = dict(output_tokens=config.get('output_tokens') == 1024,
        max_output_tokens=config.get('max_output_tokens') == 1024,
        ignore_eos=config.get('ignore_eos') is True, min_tokens=config.get('min_tokens') == 0,
        expected_request_count=expected is not None and len(requests) == expected,
        unique_ids=len({r['request_id'] for r in requests}) == len(requests),
        all_requests_completed_length1024=not bad, raw_complete=raw.get('status') == 'COMPLETE')
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
        planned=expected, raw_requests=len(requests), actual_outputs=sum(len(r.get('output_token_ids', [])) for r in requests),
        expected_outputs=expected*1024 if expected is not None else None, violating_requests=bad)


def union_length(intervals):
    total = 0.; end = None
    for start, finish in sorted(intervals):
        total += max(0., finish-max(start, end if end is not None else start))
        end = max(finish, end if end is not None else finish)
    return total


def phases(timing):
    boundaries = dict(engine_init=('engine_init_start', 'engine_init_end'), warmup=('warmup_start', 'warmup_end'),
        measurement=('measurement_start', 'measurement_return'), post_request_drain=('measurement_return', 'post_request_drain_end'),
        shutdown=('shutdown_start', 'process_end'), process=('process_start', 'process_end'))
    return {name+'_s': timing[b+'_perf_s']-timing[a+'_perf_s']
        if a+'_perf_s' in timing and b+'_perf_s' in timing else None for name, (a, b) in boundaries.items()}


def summarize_cell(cell, metrics_path):
    referenced = Path(cell['directory'])
    candidates = [metrics_path.parent/referenced.parent.name/referenced.name, referenced, ROOT.parent/referenced]
    directory = next((p.resolve() for p in candidates if p.is_dir()), referenced)
    rawpath = next((directory/n for n in ('raw.json', 'raw.json.gz') if (directory/n).exists()), None)
    raw = read(rawpath) if rawpath else None
    config = read(directory/'config.json') if (directory/'config.json').exists() else None
    timing = read(directory/'timing.json') if (directory/'timing.json').exists() else {}
    rawrows = {r['request_id']: r for r in raw['requests']} if raw else {}
    canonical = {r['request']: r for r in cell.get('per_request', [])}
    ids = sorted(set(rawrows) | set(canonical)); recoveries = cell.get('recovery_events')
    planned = cell.get('planned', (config or {}).get('requests'))
    unidentified = max(0, planned-len(ids)) if planned is not None else None
    unknown_rows = [None]*(unidentified or 0)
    rows = []; unmapped_recoveries = []
    for event in recoveries or []:
        if event['request'] not in ids: unmapped_recoveries.append(event)
    for rid in ids:
        request = rawrows.get(rid, {}); metric = canonical.get(rid, {})
        events = sorted((e for e in recoveries or [] if e['request'] == rid), key=lambda e: e['demand_host_s'])
        intervals = []; unresolved = 0; missing_intervals = 0; repeated_after_output = 0
        for i, event in enumerate(events):
            start, finish = event['demand_host_s'], event['next_output_s']
            unresolved += finish is None
            if finish is None and raw: finish = raw['observation_end_s']
            if finish is None or finish < start: missing_intervals += 1
            else: intervals.append((start, finish))
            if i and events[i-1]['next_output_s'] is not None and start > events[i-1]['next_output_s']:
                repeated_after_output += 1
        observed_sum = sum(e['demand_to_next_output_s'] for e in events if e['demand_to_next_output_s'] is not None)
        rows.append(dict(request=rid, status=request.get('status', metric.get('status', 'missing')),
            outputs=len(request['output_token_ids']) if 'output_token_ids' in request else metric.get('outputs'),
            stop_reason=request.get('stop_reason'), raw_missing=rid not in rawrows,
            preemptions=len(events) if recoveries is not None else None,
            repeat_preemptions_after_prior_recovery_output=repeated_after_output if recoveries is not None else None,
            recovery_episodes_without_next_output=unresolved if recoveries is not None else None,
            recovery_interval_missing=missing_intervals if recoveries is not None else None,
            completed_episode_wait_sum_s=observed_sum if recoveries is not None else None,
            cumulative_recovery_union_lower_bound_s=union_length(intervals) if recoveries is not None and not missing_intervals else None,
            cumulative_recovery_union_exact=not unresolved and not missing_intervals if recoveries is not None else None,
            **{k: metric.get(k) for k in ('ttft_s', 'flow_s', 'maxgap_s', 'slo_pass')}))
    first_outputs = [r['token_times_s'][0] for r in rawrows.values() if r.get('token_times_s')]
    action_rows = cell.get('tail_actions', {}).get('rows')
    changes = [r for r in action_rows or [] if r['changed']]
    first_action = min((r['begin_s'] for r in changes), default=None)
    first_allocated = min((r['begin_s'] for r in changes if r['record']['outcome'] == 'allocated'), default=None)
    last_first = max(first_outputs, default=None)
    complete_firsts = raw is not None and len(first_outputs) == planned == len(rawrows)
    available_outputs = [r['outputs'] for r in rows if r['outputs'] is not None]
    result = {k: cell.get(k) for k in ('directory', 'mode', 'status', 'error', 'planned', 'statuses', 'completed', 'failed',
        'unfinished', 'duration_s', 'outputs', 'output_tokens_per_s', 'throughput_rps', 'joint_slo', 'observed',
        'all_request_lower_bounds', 'copy_work')}
    result.update(inputs=dict(raw=str(rawpath) if rawpath else None, config=str(directory/'config.json') if config else None,
        timing=str(directory/'timing.json') if timing else None), phase_times=phases(timing),
        fixed1024_contract=fixed_contract(config, raw), all_request_rows=len(rows),
        missing_planned_request_rows=unidentified,
        actual_outputs_known=sum(available_outputs), output_count_missing_requests=len(rows)-len(available_outputs)+(unidentified or 0),
        stop_reason_counts=dict(Counter(str(r['stop_reason']) if r['stop_reason'] is not None else 'MISSING' for r in rows)),
        actual_output_count_distribution=dict(Counter(str(r['outputs']) if r['outputs'] is not None else 'MISSING' for r in rows)),
        recovery=dict(episodes=len(recoveries) if recoveries is not None else None,
            affected_requests=sum(bool(r['preemptions']) for r in rows) if recoveries is not None else None,
            repeated_requests=sum((r['preemptions'] or 0)>1 for r in rows) if recoveries is not None else None,
            repeat_preemptions_after_prior_recovery_output=sum(r['repeat_preemptions_after_prior_recovery_output'] or 0 for r in rows)
                if recoveries is not None else None,
            all_request_cumulative_union_lower_bound_s=summary([r['cumulative_recovery_union_lower_bound_s'] for r in rows]+unknown_rows),
            completed_episode_wait_sum_s=summary([r['completed_episode_wait_sum_s'] for r in rows]+unknown_rows),
            unmapped_episodes=unmapped_recoveries),
        action_vs_first_output=dict(action_observation_available=action_rows is not None,
            changed_calls=len(changes) if action_rows is not None else None, first_changed_call_s=first_action,
            first_changed_successful_allocation_s=first_allocated, last_observed_first_output_s=last_first,
            requests_without_observed_first_output=(planned-len(first_outputs)) if raw is not None and planned is not None else None,
            all_planned_first_outputs_observed=complete_firsts,
            first_action_minus_last_observed_first_output_s=first_action-last_first if first_action is not None and last_first is not None else None,
            first_action_after_all_first_outputs=first_action>last_first if first_action is not None and complete_firsts else None),
        per_request=rows)
    return result


def summarize(metrics_path):
    metrics = read(metrics_path)
    return dict(metrics=str(metrics_path.resolve()), metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        cells=[summarize_cell(c, metrics_path.resolve()) for c in metrics['cells']],
        semantics='All known planned request IDs, failures and unfinished requests are retained; absent rows and fields are explicit. '
            'Recovery union merges each request preempt-to-next-output intervals to avoid double counting overlapping preemptions. '
            'Without next output the interval ends at observation end and is a lower bound, never a completed wait. '
            'Episode sums are reported separately and may overlap. Repeated-after-output counts adjacent preemptions whose prior '
            'recovery next output was already received. First-output/action ordering is within-run host evidence, not a cross-run '
            'causal claim. Phase durations share host perf_counter but are not additive GPU timings. Fixed1024 checks run only '
            'when executed config output_mode is fixed. Original metrics and raw files are unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = summarize(args.metrics)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), cells=len(result['cells']))))


if __name__ == '__main__':
    main()
