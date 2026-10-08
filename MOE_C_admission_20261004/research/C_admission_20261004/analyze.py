#!/usr/bin/env python3
"""Full-population analysis of one or more immutable admission experiment cells.

Usage: python analyze.py CELL [CELL ...] --output NEW_SUMMARY.json
Writes NEW_SUMMARY.json plus one sibling *.requests.json and *.requests.csv per cell.
No application SLO is assumed. The fixed 20-point grid is exploratory, not a selector.
Runs, rather than requests from the same run, are the independent repeat unit.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


TTFT_GRID_S = (2, 5, 10, 20, 40)
MAXGAP_GRID_S = (0.25, 0.5, 1, 2)
FLOW_LIMIT_S = 120


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def quantile(values, q):
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def distribution(values, denominator):
    """Finite-value summaries explicitly conditioned on observation, plus full-N CDF."""
    values = sorted(float(x) for x in values if finite(x))
    counts, cumulative, cdf = collections.Counter(values), 0, []
    for value, count in sorted(counts.items()):
        cumulative += count
        cdf.append([value, cumulative / denominator if denominator else None])
    return dict(denominator=denominator, observed_n=len(values), missing_n=denominator-len(values),
                observed_fraction=len(values)/denominator if denominator else None,
                observed_only=dict(mean=sum(values)/len(values) if values else None,
                    p50=quantile(values, .5), p90=quantile(values, .9),
                    p95=quantile(values, .95), p99=quantile(values, .99),
                    min=values[0] if values else None, max=values[-1] if values else None),
                full_population_cdf=cdf)


def request_rows(raw):
    end = raw.get('observation_end_s')
    if not finite(end) or end < 0:
        raise ValueError('observation_end_s must be finite and nonnegative')
    rows, ids = [], set()
    for source in raw['requests']:
        rid = source['request_id']
        if rid in ids:
            raise ValueError(f'duplicate request ID: {rid}')
        ids.add(rid)
        arrival = source['arrival_s']
        times = source.get('token_times_s', [])
        tokens = source.get('output_token_ids', [])
        if not finite(arrival) or arrival < 0 or len(times) != len(tokens):
            raise ValueError(f'invalid arrival or token/time alignment: {rid}')
        if any(not finite(t) or t < arrival or t > end + 1e-6 for t in times):
            raise ValueError(f'invalid token timestamp: {rid}')
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError(f'nonmonotonic token timestamps: {rid}')
        status = source.get('status', 'unfinished')
        completed = status == 'completed'
        completion = source.get('completion_s')
        if completed and (not finite(completion) or completion < arrival or completion > end + 1e-6):
            raise ValueError(f'invalid completion timestamp: {rid}')
        if completed and times and completion < times[-1]:
            raise ValueError(f'completion precedes final token: {rid}')
        arrived = arrival <= end
        explicit_timeout = status in ('timeout', 'timed_out') or source.get('timed_out', False)
        runtime_timeout = raw.get('error') == 'runtime_limit' and arrived and status == 'unfinished'
        outcome = ('timeout' if explicit_timeout or runtime_timeout else status)
        if outcome not in ('completed', 'failed', 'timeout', 'rejected'):
            outcome = 'unfinished'
        gaps = [b-a for a, b in zip(times, times[1:])]
        observed_gap = max(gaps, default=0.0) if times else None
        complete_gap = observed_gap if completed else None
        stop = source.get('stop_reason', source.get('finish_reason'))
        submit = source.get('admission_s')
        if submit is not None and (not finite(submit) or submit < arrival):
            raise ValueError(f'invalid engine submission timestamp: {rid}')
        rows.append(dict(request_id=rid, raw_status=status, outcome=outcome,
            arrived_at_observation_end=arrived, arrival_s=arrival,
            engine_submission_s=submit, submission_lag_s=submit-arrival if submit is not None else None,
            first_token_s=times[0] if times else None, completion_s=completion,
            ttft_s=times[0]-arrival if times else None,
            flow_s=completion-arrival if completed else None,
            tpot_s=(times[-1]-times[0])/(len(times)-1) if completed and len(times) >= 2 else None,
            max_generation_gap_s=complete_gap, observed_max_generation_gap_s=observed_gap,
            ttft_lower_bound_s=times[0]-arrival if times else max(0, end-arrival),
            flow_lower_bound_s=completion-arrival if completed else max(0, end-arrival),
            max_generation_gap_lower_bound_s=(max(observed_gap, end-times[-1])
                if times and not completed else observed_gap),
            prompt_tokens=source.get('prompt_tokens'), max_output_tokens=source.get('max_output_tokens'),
            output_tokens=len(tokens), stop_reason=stop,
            natural_stop=completed and stop == 'stop', length_stop=completed and stop == 'length',
            output_token_ids_sha256=hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
            error=source.get('error')))
    return rows


def controller_summary(admission):
    if not admission:
        return dict(available=False)
    decisions = admission.get('decisions', [])
    snapshots = admission.get('snapshots', [])
    changed = [r for r in decisions if r.get('changed_by_recovery')]
    denied = [r for r in decisions if r.get('denied')]
    eligible = [r for r in decisions if r.get('base_allowed', not r.get('denied') or r.get('changed_by_recovery'))]
    numeric = {}
    for field in ('free_blocks', 'running', 'active', 'recovery_count', 'oldest_recovery_wait_s'):
        records = snapshots or decisions
        numeric[field] = distribution([r.get(field) for r in records], len(records))
    return dict(available=True, decision_evaluations=len(decisions), snapshot_count=len(snapshots),
        configuration=dict({k: admission.get(k) for k in ('mode', 'cap', 'kv_floor', 'max_signal_wait_s',
            'probe_enabled', 'delay_s', 'max_extra_s', 'admission_count')},
            **{k: admission[k] for k in ('budget_blocks', 'block_size', 'budget_overrides') if k in admission}),
        denied_evaluations=len(denied), denied_request_ids=sorted({str(r.get('request_id')) for r in denied}),
        changed_by_recovery_evaluations=len(changed),
        changed_request_ids=sorted({str(r.get('request_id')) for r in changed}),
        wait_limit_bypass_evaluations=sum(bool(r.get('signal_wait_limit_bypass')) for r in decisions),
        wait_limit_bypass_request_ids=sorted({str(r.get('request_id')) for r in decisions
            if r.get('signal_wait_limit_bypass')}),
        changed_fraction_of_base_eligible=len(changed)/len(eligible) if eligible else None,
        reasons=dict(collections.Counter(r.get('reason', 'unspecified') for r in decisions)),
        observed_states=numeric,
        changed_decisions=changed,
        overhead={k: v for k, v in admission.items() if 'overhead' in k or 'timing' in k
                  or k in ('controller_cpu_s', 'controller_wall_s', 'calls')},
        interpretation='Event counts include repeated deferrals; unique IDs are also reported. '
            'Changed-by-recovery is an online same-state baseline comparison, not outcome causality.')


def analyze_cell(cell):
    raw_path = cell / 'raw.json'
    raw = json.loads(raw_path.read_text())
    admission_path = cell / 'admission.json'
    admission = json.loads(admission_path.read_text()) if admission_path.exists() else {}
    rows = request_rows(raw)
    starts = {r['request_id']: r for r in admission.get('starts', [])}
    source_by_id = {r['request_id']: r for r in raw['requests']}
    origin = raw.get('measurement_origin_perf_counter_s')
    for row in rows:
        source = source_by_id[row['request_id']]
        start = starts.get(source.get('internal_request_id'), {})
        stamp = start.get('first_prefill_perf_s')
        prefill = stamp-origin if finite(stamp) and finite(origin) else None
        row.update(first_prefill_schedule_return_s=prefill,
            new_prefill_wait_s=prefill-row['arrival_s'] if prefill is not None else None)
    n = len(rows)
    end = raw['observation_end_s']
    arrival_window = max([r['arrival_s'] for r in rows], default=0)
    last_completion = max([r['completion_s'] for r in rows if r['outcome'] == 'completed'], default=0)
    duration = max(arrival_window, end, last_completion)
    outcomes = {k: sum(r['outcome'] == k for r in rows)
                for k in ('completed', 'failed', 'unfinished', 'rejected', 'timeout')}
    output_total = sum(r['output_tokens'] for r in rows)
    workload_identity = [dict(request_id=r['request_id'], arrival_s=r['arrival_s'],
        prompt_sha256=r.get('prompt_token_ids_sha256'), prompt_tokens=r.get('prompt_tokens'),
        max_output_tokens=r.get('max_output_tokens')) for r in raw['requests']]
    grid = []
    for ttft in TTFT_GRID_S:
        for gap in MAXGAP_GRID_S:
            good = sum(r['outcome'] == 'completed' and r['ttft_s'] is not None and
                r['max_generation_gap_s'] is not None and r['ttft_s'] <= ttft and
                r['max_generation_gap_s'] <= gap and r['flow_s'] <= FLOW_LIMIT_S for r in rows)
            grid.append(dict(ttft_limit_s=ttft, maxgap_limit_s=gap, flow_limit_s=FLOW_LIMIT_S,
                good_requests=good, planned_requests=n, joint_attainment=good/n if n else None,
                goodput_requests_per_s=good/duration if duration else None))
    summary = dict(cell=str(cell.resolve()), raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        workload_identity_sha256=hashlib.sha256(json.dumps(workload_identity,
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        run_status=raw.get('status'), error=raw.get('error'), planned_requests=n,
        arrived_requests=sum(r['arrived_at_observation_end'] for r in rows), outcomes=outcomes,
        actual_preemption_count=raw.get('actual_preemption_count'),
        arrival_window_s=arrival_window, observation_end_s=end, last_completion_s=last_completion,
        service_denominator_s=duration, all_completed=outcomes['completed'] == n,
        observed_drain_s=max(0, end-arrival_window),
        completed_drain_s=max(0, last_completion-arrival_window) if outcomes['completed'] == n else None,
        drain_censored=outcomes['completed'] != n,
        distributions={field: distribution([r[field] for r in rows], n) for field in
            ('submission_lag_s', 'new_prefill_wait_s', 'ttft_s', 'flow_s', 'tpot_s', 'max_generation_gap_s',
             'observed_max_generation_gap_s', 'ttft_lower_bound_s', 'flow_lower_bound_s',
             'max_generation_gap_lower_bound_s', 'output_tokens')},
        total_output_tokens=output_total,
        output_tokens_per_s=output_total/duration if duration else None,
        completed_requests_per_s=outcomes['completed']/duration if duration else None,
        stop_reasons=dict(collections.Counter(str(r['stop_reason']) for r in rows)),
        natural_stop_count=sum(r['natural_stop'] for r in rows),
        length_stop_count=sum(r['length_stop'] for r in rows),
        host_chunk_diagnostics=raw.get('host_chunk_diagnostics'),
        joint_slo_grid=grid, admission=controller_summary(admission))
    overhead = summary['admission'].get('overhead', {})
    overhead['wall_fraction_of_service'] = (admission['controller_wall_s']/duration
        if finite(admission.get('controller_wall_s')) and duration else None)
    overhead['cpu_us_per_schedule_call'] = (admission['controller_cpu_s']*1e6/admission['calls']
        if finite(admission.get('controller_cpu_s')) and admission.get('calls') else None)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cells', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    targets = [args.output]
    for i, cell in enumerate(args.cells):
        base = args.output.with_name(f'{args.output.stem}.{i:02d}-{cell.name}.requests')
        targets.extend((Path(str(base)+'.json'), Path(str(base)+'.csv')))
    if any(p.exists() for p in targets):
        parser.error('Refusing to overwrite an existing output; choose a new --output path.')
    cells = [analyze_cell(cell) for cell in args.cells]
    result = dict(schema_version=1, independent_unit='run',
        inference='Descriptive run-level results only; no request-level significance tests or best-grid selection.',
        measurement_semantics='All TTFT and flow times start at external arrival_s. admission_s is '
            'engine submission, not prefill start. New prefill wait ends at the return of its first '
            'scheduler call assigning tokens, when that separate trace is available. '
            'Each planned request remains in the denominator. '
            'Missing/unfinished outcomes do not attain SLO. Max generation gap is between host-return '
            'token timestamps; unresolved same-chunk token timing is not reconstructed. Completed '
            'one-token requests have zero inter-token gaps. Unfinished latency lower bounds are '
            'descriptive observation bounds, not completed latency estimates. TPOT is (last token time '
            '- first token time)/(output count-1) for completed requests with at least two tokens; '
            'one-token and unfinished requests remain missing, with the full population denominator.',
        timeout_semantics='Explicit timeout statuses, or arrived unfinished requests at runtime_limit, '
            'are timeout outcomes. Unarrived planned requests remain unfinished.',
        slo_status='EXPLORATORY_FIXED_GRID_NO_APPLICATION_SLO',
        denominator_semantics='max(last scheduled external arrival, last completion, observation end), from origin 0; '
            'same formula for all arms. Attainment denominator is every planned request.',
        cells=[summary for summary, rows in cells])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, (summary, rows) in enumerate(cells):
        json_path, csv_path = targets[1+2*i:3+2*i]
        with json_path.open('x') as f:
            json.dump(rows, f, indent=2, allow_nan=False)
            f.write('\n')
        with csv_path.open('x', newline='') as f:
            fields = list(rows[0]) if rows else ['request_id']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summary['per_request_json'] = str(json_path.resolve())
        summary['per_request_csv'] = str(csv_path.resolve())
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
