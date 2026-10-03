"""Charge each observed output interval once using sparse preemption events.

Marked gaps include time on both sides of preemption, not pure recovery cost.
No diagnostic projection, inferred transfer starts, or counterfactual attribution.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def partition(raw):
    require(raw['diagnostics'] == 'SPARSE_PREEMPTION_EVENTS', 'Direct sparse events required')
    stop = raw['observation_end_s']
    require(math.isfinite(stop) and stop >= 0, 'Invalid observation boundary')
    requests = {r['request_id']: r for r in raw['requests']}
    require(len(requests) == len(raw['requests']), 'Duplicate request identity')
    grouped = defaultdict(list)
    counts = Counter()
    for index, event in enumerate(raw['preemption_events']):
        require(event['request_id'] in requests, 'Unknown preemption request')
        kind = ('method_failed' if not event['original_preemption_returned'] else
                'method_returned_call_returned' if event['engine_call_index'] < raw['engine_return_count']
                else 'method_returned_call_not_returned')
        counts[kind] += 1
        grouped[event['request_id']].append(dict(event_index=index, kind=kind, **event))
    fields = ('pre_output_wait_s', 'gap_with_preemption_s', 'other_generation_gap_s',
              'completed_tail_s', 'censored_tail_s', 'observed_request_s')
    rows = []
    for rid, r in requests.items():
        times = r['token_times_s']
        require(len(times) == len(r['output_token_ids']), 'Output/timestamp mismatch')
        complete = r['status'] == 'completed'
        end = r['completion_s'] if complete else stop
        arrival = r['arrival_s']
        arrived = arrival <= stop
        require(all(math.isfinite(t) for t in times) and times == sorted(times), 'Invalid output times')
        require(math.isfinite(end) and end <= stop, 'Invalid completion boundary')
        require(not times or arrival <= times[0] <= times[-1] <= end, 'Output outside request')
        if not arrived:
            require(not complete and not times and not grouped[rid], 'Future request has observed activity')
        marks = defaultdict(list)
        for event in grouped[rid]:
            k = event['last_returned_output_count']
            require(0 <= k <= len(times), 'Preemption output position out of range')
            require(event['last_new_output_s'] == (times[k-1] if k else None),
                    'Preemption last output differs from measured output')
            low, high = (times[k-1] if k else arrival), (times[k] if k < len(times) else end)
            require(low <= event['method_entered_s'] <= high, 'Preemption outside marked interval')
            if event['original_preemption_returned']:
                require(event['method_entered_s'] <= event['method_returned_s'] <= high,
                        'Preemption return outside marked interval')
                marks[k].append(event['event_index'])
        gaps = [dict(before_output_count=k, start_s=times[k-1], end_s=times[k],
                     duration_s=times[k]-times[k-1], successful_method_event_indices=marks[k])
                for k in range(1, len(times)) if marks[k]]
        marked = math.fsum(g['duration_s'] for g in gaps)
        other = math.fsum(b-a for k, (a, b) in enumerate(zip(times, times[1:]), 1) if not marks[k])
        pre = max(0, (times[0] if times else end)-arrival) if arrived else 0
        tail = end-times[-1] if times else 0
        observed = max(0, end-arrival) if arrived else 0
        values = dict(pre_output_wait_s=pre, gap_with_preemption_s=marked,
            other_generation_gap_s=other, completed_tail_s=tail if complete else 0,
            censored_tail_s=tail if not complete else 0, observed_request_s=observed)
        residual = observed-math.fsum(values[k] for k in fields[:-1])
        require(math.isclose(residual, 0, abs_tol=1e-9), 'Request-time conservation failed')
        rows.append(dict(request_id=rid, status=r['status'], arrived=arrived,
            outputs=len(times), first_output_observed=bool(times),
            completed_without_output=complete and not times, **values,
            accounting_residual_s=residual, marked_gap_count=len(gaps),
            successful_preemptions_in_generation_gaps=sum(len(g['successful_method_event_indices']) for g in gaps),
            max_marked_gap_s=max((g['duration_s'] for g in gaps), default=None),
            pre_output_successful_event_indices=marks[0],
            tail_successful_event_indices=marks[len(times)] if times else [],
            preemption_events=grouped[rid], marked_gaps=gaps))
    return dict(status=raw['status'], comparable_complete_service=raw['status']=='COMPLETE'
        and all(r['status']=='completed' for r in rows), source_run_id=raw.get('run_id'),
        annotation_basis='Actual sparse method events in this same measured run',
        totals={k: math.fsum(r[k] for r in rows) for k in fields},
        counts=dict(planned=len(rows), arrived=sum(r['arrived'] for r in rows),
            completed=sum(r['status']=='completed' for r in rows),
            marked_generation_gaps=sum(r['marked_gap_count'] for r in rows), **counts),
        requests=rows, semantics=[
            'Each generation interval is charged once even when it contains several preemptions.',
            'Only successful preemption methods mark intervals; failed attempts remain explicit records.',
            'Successful methods in unreturned engine calls remain recorded, with no inferred new output or completion.',
            'Pre-output wait becomes TTFT only if a first output is observed; zero-output completions and unfinished waits stay distinct.',
            'Incomplete tails are right-censored observed durations, not completed generation gaps or full request latency.',
            'Marked generation gaps contain preemption-adjacent execution, queueing and any restoration, not a pure recovery tax.',
            'Output times are engine returns; within a multi-token chunk internal token gaps remain unresolved.',
            'Sums are overlapping request-seconds, not episode wall time or additive transfer/compute cost.',
            'This partition does not infer load readiness, bytes, recomputation, native/forced cause, or policy benefit.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = partition(json.loads(args.raw.read_text()))
    result['source_raw'] = str(args.raw)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(counts=result['counts'], totals=result['totals']), indent=2))
