#!/usr/bin/env python3
"""Describe completion turnover at recorded legal pending-recovery admission gates.

Usage: python3 completion_turnover.py ANALYSIS.json --output NEW.json
Optional --design cadence-scan accepts only the explicit .2s single baseline.
No replay or online policy.
"""
import argparse
import bisect
import collections
import hashlib
import json
from pathlib import Path

from analyze import finite, quantile
from compare_quartet import ORDERS
from slo_failure_breakdown import read_hashed


def stats(values):
    values = sorted(values)
    if any(not finite(v) for v in values):
        raise ValueError('Nonfinite observed time')
    return dict(n=len(values), min=values[0] if values else None,
        mean=sum(values)/len(values) if values else None,
        p10=quantile(values, .1), p50=quantile(values, .5), p90=quantile(values, .9),
        p95=quantile(values, .95), max=values[-1] if values else None)


def opportunity_stats(rows):
    waits = [r['next_completion_wait_s'] for r in rows if r['next_completion_wait_s'] is not None]
    return dict(evaluations=len(rows), unique_requests=len({r['request_id'] for r in rows}),
        time_to_next_host_completion_s=stats(waits), no_later_completion=len(rows)-len(waits),
        next_completion_within_previous_hold_windows={str(t): sum(w <= t for w in waits)
            for t in (.1, .25)},
        actual_allocation_results=dict(collections.Counter(str(r['native_allocation_result']) for r in rows)),
        next_completion_unique_requests=len({rid for r in rows for rid in r['next_completion_request_ids']}))


def cadence_metadata(path, requests, admission):
    config, config_sha = read_hashed(path/'config.json')
    workload, workload_sha = read_hashed(path/'workload.json')
    expected = dict(requests=384, admission_mode='kv', cap=256, admission_cap=256,
        native_running_limit=256, engine_max_num_seqs=256, kv_floor=3277,
        native_admission_guard_matches_complete_cap=True, admission_count='complete_unique_unfinished',
        probe_enabled=False, reservation_observation=False, reservation_probe_enabled=False,
        record_gc=True, history_retained=False, arrival_gap_s=.2, arrival_span_s=.2*383,
        output_tokens=1024, max_output_tokens=1024, min_tokens=0, ignore_eos=False)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('Cadence-scan config mismatch: '+key)
    for key, value in dict(mode='kv', cap=256, kv_floor=3277, max_signal_wait_s=10, probe_enabled=False).items():
        if admission.get(key) != value:
            raise ValueError('Cadence-scan admission mismatch: '+key)
    if workload.get('schema') != 'olmoe-natural-cadence-pressure-02s-observation-v1':
        raise ValueError('Not the declared cadence-scan workload')
    trace = [.2*i for i in range(384)]
    contract = dict(max_output_tokens=1024, ignore_eos=False, min_tokens=0)
    if workload['arrival_traces_s'] != {'steady': trace} or workload['output_contract'] != contract:
        raise ValueError('Cadence scan must preserve exact .2*i arrivals and natural EOS contract')
    if hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() != config['workload_sha256']:
        raise ValueError('Configured workload hash mismatch')
    source = workload['source_requests']
    if len(requests) != 384 or len(source) != 384 or any(
            r['request_id'] != s['request_id'] or r['arrival_s'] != t or
            r['max_output_tokens'] != 1024 or s['max_output_tokens'] != 1024
            for r, s, t in zip(requests, source, trace)):
        raise ValueError('Observed population/order/arrivals/budgets differ from declared workload')
    return dict(config_sha256=config_sha, workload_file_sha256=workload_sha,
        verified_configuration=expected, verified_output_contract=contract,
        comparison='NONE_SINGLE_OBSERVATION_ONLY_BASELINE')


def local_decision_partition(admission, identity):
    labels = ('no_pending_recovery', 'pending_recovery_native_denied',
        'pending_recovery_native_fit_cap_or_KV_blocked', 'pending_recovery_base_allowed_age_lt10',
        'pending_recovery_base_allowed_age_ge10')
    groups = {key: [] for key in labels}
    local = [r for r in admission['decisions'] if type(r.get('native_fit')) is bool
        and r['num_new_tokens'] > 0 and not r['load_kv_async']]
    for row in local:
        if row['base_allowed'] != (row['native_fit'] and row['active'] < admission['cap'] and
                not (row['free_blocks'] < admission['kv_floor'] and row['age_s'] < 10)):
            raise ValueError('Gate-row base allowance disagrees with cap/KV/age')
        label = (labels[0] if row['recovery_count'] == 0 else
                 labels[1] if not row['native_fit'] else labels[2] if not row['base_allowed'] else
                 labels[3] if row['age_s'] < 10 else labels[4])
        groups[label].append(identity[row['request_id']])
    return dict(evaluations=len(local), unique_requests=len({identity[r['request_id']] for r in local}),
        groups={key: dict(evaluations=len(ids), unique_requests=len(set(ids)), request_ids=sorted(set(ids)))
            for key, ids in groups.items()},
        excluded_without_native_fit=dict(collections.Counter(r['reason'] for r in admission['decisions']
            if type(r.get('native_fit')) is not bool)),
        excluded_checked_nonlocal_evaluations=sum(type(r.get('native_fit')) is bool and
            not (r['num_new_tokens'] > 0 and not r['load_kv_async']) for r in admission['decisions']),
        semantics='Mutually exclusive actual new local decision evaluations; FIFO has no checked fit '
            'and is excluded. Unique IDs may recur across groups over time and are not additive. '
            'No pending recovery takes precedence irrespective of native fit; later groups partition '
            'positive pending recovery. Native-denied means the recorded fit was false, not a probe action.')


def analyze_cell(cell, cadence=False):
    path = Path(cell['cell'])
    requests, request_sha = read_hashed(Path(cell['per_request_json']))
    admission, admission_sha = read_hashed(path/'admission.json')
    by_id = {r['request_id']: r for r in requests}
    if len(by_id) != len(requests) or len(requests) != cell['planned_requests']:
        raise ValueError('Request population mismatch')
    if not cadence and any(r['outcome'] != 'completed' or not finite(r['completion_s']) for r in requests):
        raise ValueError('This diagnostic requires the complete quartet population')
    metadata = cadence_metadata(path, requests, admission) if cadence else None
    completed = sorted((r['completion_s'], r['request_id']) for r in requests
        if r['outcome'] == 'completed' and finite(r['completion_s']))
    times = [t for t, _ in completed]
    checked = [r for r in admission['decisions'] if type(r.get('native_fit')) is bool]
    opportunities = [r for r in checked if r['native_fit'] and r['base_allowed'] and
        r['recovery_count'] > 0 and r['num_new_tokens'] > 0 and not r['load_kv_async']]
    if opportunities != [r for r in checked if r['opportunity']]:
        raise ValueError('Recorded opportunity does not match exact gate-row predicate')
    raw = None
    if cadence or opportunities or cell['actual_preemption_count']:
        raw, raw_sha = read_hashed(path/'raw.json')
        if raw_sha != cell['raw_sha256']:
            raise ValueError('Raw file differs from existing analysis')
        identity = {r[k]: r['request_id'] for r in raw['requests']
            for k in ('request_id', 'internal_request_id', 'external_request_id') if r.get(k)}
        if {r['request_id'] for r in raw['requests']} != set(by_id):
            raise ValueError('Raw/per-request identities differ')
        for r in raw['requests']:
            if r['completion_s'] != by_id[r['request_id']]['completion_s']:
                raise ValueError('Raw/per-request completion differs')
    rows = []
    for row in opportunities:
        rid = identity[row['request_id']]
        t = admission['origin_perf_s'] + row['t'] - raw['measurement_origin_perf_counter_s']
        index = bisect.bisect_left(times, t)
        next_t = times[index] if index < len(times) else None
        age_bypass = row['signal_wait_limit_bypass']
        if age_bypass != (row['age_s'] >= admission['max_signal_wait_s']):
            raise ValueError('Age bypass flag mismatch')
        rows.append(dict(request_id=rid, external_time_s=t, gate_relative_time_s=row['t'],
            age_s=row['age_s'], age_limit_bypass=age_bypass,
            age_bypass_needed_for_KV_allowance=age_bypass and row['free_blocks'] < admission['kv_floor'],
            free_blocks=row['free_blocks'], active=row['active'], recovery_count=row['recovery_count'],
            native_fit=row['native_fit'], base_allowed=row['base_allowed'],
            native_allocation_result=row['native_allocation_result'], denied=row['denied'],
            next_completion_s=next_t, next_completion_wait_s=next_t-t if next_t is not None else None,
            next_completion_request_ids=[rid for ct, rid in completed if ct == next_t]))
    actual_preemptions = []
    if raw is not None:
        for event in raw['preemption_events']:
            if event['original_preemption_returned'] is not True:
                continue
            rid = identity[event['request_id']]
            first = by_id[rid]['first_token_s']
            before = event['method_entered_s'] < first if finite(first) else None
            zero = event['native_output_count_before'] == 0
            actual_preemptions.append(dict(request_id=rid,
                method_entered_s=event['method_entered_s'], native_output_count_before=event['native_output_count_before'],
                first_host_token_s=first, before_first_host_output=before, native_output_zero=zero,
                classification='first_host_output_unobserved' if before is None else
                    'before_first_output_both' if before and zero else
                    'after_first_output_both' if not before and not zero else 'host_native_order_disagreement'))
    if len(actual_preemptions) != cell['actual_preemption_count']:
        raise ValueError('Actual preemption count mismatch')
    classification = {}
    labels = ('before_first_output_both', 'after_first_output_both', 'host_native_order_disagreement')
    for label in labels + (('first_host_output_unobserved',) if cadence else ()):
        selected = [r for r in actual_preemptions if r['classification'] == label]
        classification[label] = dict(events=len(selected), unique_requests=len({r['request_id'] for r in selected}))
    terminal = [r for r in requests if r['outcome'] == 'completed']
    counts = dict(budget_1024=sum(r['max_output_tokens'] == 1024 for r in requests),
        actual_output_1024=sum(r['output_tokens'] == 1024 for r in terminal),
        actual_output_at_most_128=sum(r['output_tokens'] <= 128 for r in terminal))
    window = None
    if rows:
        lo, hi = min(r['external_time_s'] for r in rows), max(r['external_time_s'] for r in rows)
        local = [t for t in times if lo <= t <= hi]
        window = dict(first_opportunity_s=lo, last_opportunity_s=hi,
            completed_requests_in_span=len(local), distinct_host_completion_times=len(set(local)),
            completion_interarrival_s=stats([b-a for a, b in zip(local, local[1:])]))
    result = dict(cell=path.name, raw_sha256=cell['raw_sha256'], raw_reloaded=raw is not None,
        source_per_request_json=cell['per_request_json'], source_per_request_sha256=request_sha,
        admission_sha256=admission_sha, planned_requests=cell['planned_requests'], outcomes=cell['outcomes'],
        clock_conversion=dict(gate_origin_perf_s=admission['origin_perf_s'],
            measurement_origin_perf_s=raw['measurement_origin_perf_counter_s'] if raw else None,
            gate_to_external_offset_s=admission['origin_perf_s']-raw['measurement_origin_perf_counter_s'] if raw else None,
            formula='external_time_s = gate_origin_perf_s + decision.t - measurement_origin_perf_s',
            status='APPLIED' if raw else 'NOT_NEEDED_NO_OPPORTUNITIES'),
        output_counts=counts, output_fractions={k:v/len(requests) for k,v in counts.items()},
        max_output_tokens_counts=dict(collections.Counter(r['max_output_tokens'] for r in requests)),
        stop_reasons=dict(collections.Counter(r['stop_reason'] for r in requests)),
        completion_external_time_s=stats(times), completion_interarrival_s=stats([b-a for a,b in zip(times,times[1:])]),
        completed_by_last_external_arrival=sum(t <= cell['arrival_window_s'] for t in times),
        completion_timeline=[dict(request_id=rid, completion_s=t) for t,rid in completed],
        all_legal_pending_opportunities=opportunity_stats(rows),
        age_bypass_opportunities=opportunity_stats([r for r in rows if r['age_limit_bypass']]),
        age_fresh_opportunities=opportunity_stats([r for r in rows if not r['age_limit_bypass']]),
        age_bypass_needed_for_KV_allowance=sum(r['age_bypass_needed_for_KV_allowance'] for r in rows),
        opportunity_window=window, opportunities=rows, preemption_classification=classification,
        preemption_events=actual_preemptions)
    if cadence:
        result.update(cadence_scan=metadata, local_decision_partition=local_decision_partition(admission, identity),
            arrived_requests=cell['arrived_requests'], observation_end_s=cell['observation_end_s'],
            incomplete_requests=len(requests)-len(terminal),
            output_length_semantics='Actual lengths use completed requests only; fractions retain all384 '
                'as denominator. Incomplete observed prefixes are not counted as short outputs. '
                'Missing later completion remains null, not zero wait; full service outcomes are in source analysis.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--design', choices=('quartet', 'cadence-scan'), default='quartet')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite output')
    source, source_sha = read_hashed(args.analysis)
    cells = {Path(c['cell']).name:c for c in source['cells']}
    cadence = args.design == 'cadence-scan'
    order = ('probe-00-baseline',) if cadence else ORDERS['simple-recheck']
    if set(cells) != set(order) or len(source['cells']) != len(order):
        raise ValueError('Unexpected explicit cell names for '+args.design)
    if len({c['workload_identity_sha256'] for c in cells.values()}) != 1:
        raise ValueError('Workload identities differ')
    result = dict(source_analysis=str(args.analysis.resolve()), source_analysis_sha256=source_sha,
        independent_unit='run; repeated gates and requests are not independent experiments',
        cells=[analyze_cell(cells[name], cadence=cadence) for name in order],
        semantics='All pending-recovery opportunities use actual new-request decision rows: native_fit '
            'and base_allowed, positive local prefill, non-async load, recovery_count>0. No snapshots '
            'or skipped queue positions are imputed. Age>=10 bypass and whether it was needed for '
            'KV allowance are separate. Future host completion is an offline diagnostic, never an '
            'online condition or proof of KV/GPU release. The .1/.25s bins refer only to previous '
            'probe hold windows; intervals are not summed into removable latency. Whole-population '
            'output and completion distributions do not establish local slow turnover. Native preempt '
            'classification requires successful method return, native output count and first host '
            'token; host/native disagreement stays explicit. No policy benefit or workload-change '
            'necessity is inferred. A request may have events in both preemption categories, so their '
            'unique-request counts are not additive. Fixed arms with no legal pending gates and zero reported '
            'preemptions reuse their existing complete per-request analysis without rereading raw.')
    if cadence:
        result['design'] = 'cadence-scan: single baseline, no policy or cross-trace benefit comparison'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for cell in result['cells']:
        print(cell['cell'], cell['all_legal_pending_opportunities'])
    print(args.output.resolve())


if __name__ == '__main__':
    main()
