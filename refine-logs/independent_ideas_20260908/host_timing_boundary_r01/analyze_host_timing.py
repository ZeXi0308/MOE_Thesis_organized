#!/usr/bin/env python3
"""Exact host-clock interval accounting on existing request traces; no replay."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics as stats

PHASES = ('frontend_gap', 'scheduler_prefix', 'poststamp_prefill', 'poststamp_decode')


def read(path):
    return json.loads(path.read_text())


def check(value, message):
    if not value:
        raise ValueError(message)


def summary(values):
    v = sorted(values)
    def percentile(p):
        rank = (len(v) - 1) * p
        lo = int(rank)
        return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (rank - lo)
    return dict(n=len(v), mean=stats.mean(v), p50=percentile(.5), p95=percentile(.95),
                max=max(v), min=min(v)) if v else None


def analyze_cell(path, config, workload):
    raw = read(path)
    check(raw['status'] == 'COMPLETE', 'episode incomplete')
    plan = raw['plan']
    check(plan == config['plans'][int(path.stem.split('-')[-1])], 'plan mismatch')
    check(read(path.with_name(path.name.replace('cell-', 'checks-')))['status'] == 'PASS',
          'GPU process check failed')
    check(raw['host_chunk_diagnostics']['token_level_itl_resolved'], 'unresolved host chunks')
    events = raw['output_events']
    receipt_values = [e['received_s'] for e in events]
    check(receipt_values == sorted(receipt_values), 'events out of temporal order')
    receipts = sorted(set(receipt_values))
    steps = raw['scheduler_steps']
    check(len(receipts) == len(steps), 'step/receipt count mismatch')
    by_request = defaultdict(list)
    for event in events:
        check(event['prefix_valid'] and event['chunk_size'] == 1, 'token event not resolved')
        check(not event['native_metrics']['is_corrupted'], 'corrupted native clock metrics')
        by_request[event['request_id']].append(event)
    intervals, first_schedule = [], {}
    previous = 0.0
    for i, (step, receipt) in enumerate(zip(steps, receipts)):
        start, end = step['start_s'], step['end_s']
        check(step['step'] == i and previous <= start <= end <= receipt, 'time boundary overlap')
        check(not step['preempted_request_ids'], 'unexpected preemption')
        prefill = sum(r['prefill_tokens'] for r in step['scheduled']) > 0
        intervals.extend([(previous, start, 'frontend_gap'), (start, end, 'scheduler_prefix'),
                          (end, receipt, 'poststamp_prefill' if prefill else 'poststamp_decode')])
        for scheduled in step['scheduled']:
            first_schedule.setdefault(scheduled['request_id'], start)
        previous = receipt
    check(previous <= raw['observation_end_s'], 'episode endpoint before final receipt')
    intervals.append((previous, raw['observation_end_s'], 'frontend_gap'))
    interval_ends = [end for _, end, _ in intervals]
    max_residual = 0.0

    def overlap(lo, hi):
        nonlocal max_residual
        check(0 <= lo <= hi <= raw['observation_end_s'], 'request outside episode')
        buckets = dict.fromkeys(PHASES, 0.0)
        for start, end, label in intervals[bisect_right(interval_ends, lo):]:
            if start >= hi:
                break
            buckets[label] += max(0.0, min(end, hi) - max(start, lo))
        residual = abs(sum(buckets.values()) - (hi - lo))
        max_residual = max(max_residual, residual)
        check(residual < 1e-9, 'non-conserving request accounting')
        return buckets

    expected = {s['request_id']: (s, a * plan['arrival_scale']) for s, a in
                zip(workload['source_requests'], workload['arrival_traces_s'][plan['regime']])}
    check(len(raw['requests']) == len(expected) == len(by_request), 'request population mismatch')
    check({r['request_id'] for r in raw['requests']} == set(expected) == set(by_request),
          'request identity mismatch')
    per_request = []
    for r in raw['requests']:
        rid = r['request_id']
        source, arrival = expected[rid]
        check(r['status'] == 'completed' and r['arrival_s'] == arrival, 'request status/arrival mismatch')
        check(all(r[k] == source[k] for k in ('document_id', 'prompt_token_ids_sha256')),
              'document/prompt mismatch')
        times, event_rows = r['token_times_s'], by_request[rid]
        check(len(times) == len(r['output_token_ids']) == config['output_tokens'] == len(event_rows),
              'token population mismatch')
        check(times == [e['received_s'] for e in event_rows], 'host event/request time mismatch')
        check(r['output_token_ids'] == [e['new_token_ids'][0] for e in event_rows], 'token identity mismatch')
        check(r['native_metrics'] == event_rows[-1]['native_metrics'], 'final native metrics mismatch')
        admitted, added, scheduled = r['admission_s'], r['engine_add_return_s'], first_schedule[rid]
        check(arrival <= admitted <= added <= scheduled <= times[0], 'admission timeline mismatch')
        ttft_buckets = overlap(arrival, times[0])
        decode_buckets = overlap(times[0], times[-1])
        submit_buckets = overlap(arrival, admitted)
        tpot_parts = {key: value / (len(times) - 1) for key, value in decode_buckets.items()}
        host_tpot = (times[-1] - times[0]) / (len(times) - 1)
        native_times = [e['native_metrics']['last_token_ts'] for e in event_rows]
        check(native_times == sorted(native_times), 'native token clock moved backwards')
        check(all(e['native_metrics']['first_token_ts'] == native_times[0] for e in event_rows),
              'native first token stamp changed')
        native_tpot = (native_times[-1] - native_times[0]) / (len(native_times) - 1)
        native_wait = r['native_metrics']['scheduled_ts'] - r['native_metrics']['queued_ts']
        check(native_wait >= 0, 'negative native queue interval')
        per_request.append(dict(request_id=rid, document_id=r['document_id'], ttft_s=times[0]-arrival,
            host_tpot_s=host_tpot, native_tpot_s=native_tpot,
            host_minus_native_tpot_s=host_tpot-native_tpot,
            native_host_tpot_classification_differs=(host_tpot <= config['tpot_slo_s']) !=
                                                    (native_tpot <= config['tpot_slo_s']),
            host_tpot_margin_s=host_tpot-config['tpot_slo_s'],
            ttft_buckets_s=ttft_buckets, tpot_buckets_s=tpot_parts,
            frontend_tpot_fraction=tpot_parts['frontend_gap']/host_tpot,
            submission_lag_s=admitted-arrival, submission_lag_buckets_s=submit_buckets,
            engine_add_duration_s=added-admitted, add_return_to_first_schedule_s=scheduled-added,
            first_schedule_to_first_token_s=times[0]-scheduled, native_queue_s=native_wait,
            queue_boundary_difference_s=(scheduled-added)-native_wait,
            host_native_itl_difference_s=summary([(times[i]-times[i-1])-(native_times[i]-native_times[i-1])
                                                for i in range(1,len(times))])))
    ttl = overlap(0.0, raw['observation_end_s'])
    # Means of mutually exclusive components sum exactly; medians generally do not.
    total_decode_span = sum(r['host_tpot_s'] for r in per_request)
    passed = sum(r['ttft_s'] <= config['ttft_slo_s'] and r['host_tpot_s'] <= config['tpot_slo_s']
                 for r in per_request)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), plan=plan,
        n_steps=len(steps), n_requests=len(per_request), episode_wall_s=raw['observation_end_s'],
        episode_buckets_s=ttl, n_slo_pass=passed, goodput_rps=passed/raw['observation_end_s'],
        mean_tpot_buckets_s={k:stats.mean(r['tpot_buckets_s'][k] for r in per_request) for k in PHASES},
        frontend_fraction_of_total_request_decode_span=sum(r['tpot_buckets_s']['frontend_gap'] for r in per_request)/total_decode_span,
        frontend_fraction_of_ttft=sum(r['ttft_buckets_s']['frontend_gap'] for r in per_request)/sum(r['ttft_s'] for r in per_request),
        native_host_tpot_classification_mismatches=sum(r['native_host_tpot_classification_differs'] for r in per_request),
        max_conservation_residual_s=max_residual,
        summaries={k:summary([r[k] for r in per_request]) for k in ['host_tpot_s','ttft_s','native_tpot_s',
            'host_minus_native_tpot_s','submission_lag_s','engine_add_duration_s',
            'add_return_to_first_schedule_s','native_queue_s','queue_boundary_difference_s','frontend_tpot_fraction']},
        requests=per_request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    check(not args.output_dir.exists(), 'output directory must be new')
    cells, issues = [], []
    for engine in ('forward', 'reverse'):
        root = args.source_dir / 'gpu_results' / engine
        config, workload = read(root/'config.json'), read(root/'workload.json')
        check(config == read(args.source_dir/'plans'/engine/'config.json'), 'engine config changed')
        check(workload == read(args.source_dir/'plans'/engine/'workload.json'), 'engine workload changed')
        for index in range(len(config['plans'])):
            path = root/f'cell-{index:03d}.json'
            try:
                cells.append(dict(analyze_cell(path, config, workload), engine=engine))
            except (OSError, KeyError, TypeError, ValueError) as error:
                issues.append(dict(path=str(path), error=str(error), status='INVALID_OR_MISSING'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(status='MEASUREMENT_ONLY' if not issues and len(cells)==32 else 'INCOMPLETE',
        evidence_type='REQUEST_LEVEL_OBSERVATIONAL_HOST_INTERVAL_ACCOUNTING', new_gpu_executions=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), source_dir=str(args.source_dir),
        cells=cells, issues=issues)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],cells=len(cells),issues=issues)))
    for c in cells:
        print(json.dumps(dict(engine=c['engine'],plan=c['plan'],
            frontend_tpot_pct=100*c['frontend_fraction_of_total_request_decode_span'],
            submit_p95_ms=1000*c['summaries']['submission_lag_s']['p95'],
            delivery_tpot_delta_max_us=1e6*c['summaries']['host_minus_native_tpot_s']['max'],
            classification_mismatches=c['native_host_tpot_classification_mismatches'])))


if __name__ == '__main__':
    main()
