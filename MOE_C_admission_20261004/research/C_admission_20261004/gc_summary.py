#!/usr/bin/env python3
"""Align passive GC callback intervals with host output gaps; never subtract latency.

Usage: python gc_summary.py ANALYSIS.json --output NEW.json
The input is an existing analyze.py full-population summary. No GPU is used.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


GAP_THRESHOLD_S = 0.25
PHASES = {'ingress', 'engine_step', 'output_recording'}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def partition(intervals, lower, upper):
    """Disjoint duration buckets indexed by the set of active (generation, phase)."""
    boundaries = collections.defaultdict(list)
    event_indices = []
    for index, start, end, generation, phase in intervals:
        left, right = max(start, lower), min(end, upper)
        if right <= left:
            continue
        label = (generation, phase)
        boundaries[left].append((label, 1))
        boundaries[right].append((label, -1))
        event_indices.append(index)
    active, buckets = collections.Counter(), collections.Counter()
    previous = None
    for stamp in sorted(boundaries):
        labels = tuple(sorted(label for label, count in active.items() if count))
        if previous is not None and labels:
            buckets[labels] += stamp - previous
        for label, change in boundaries[stamp]:
            active[label] += change
            if active[label] < 0:
                raise ValueError('invalid GC interval partition')
        previous = stamp
    rows = [dict(active_labels=[dict(generation=g, start_phase=p) for g, p in labels],
                 union_s=duration) for labels, duration in sorted(buckets.items())]
    return dict(union_s=sum(buckets.values()), event_indices=event_indices,
                disjoint_by_generation_start_phase=rows)


def gc_intervals(raw):
    observation = raw.get('gc_observation')
    if observation is None:
        return None, dict(available=False, status='UNAVAILABLE',
                         reason='raw has no gc_observation; no zero GC time is inferred')
    origin = raw['measurement_origin_perf_counter_s']
    if observation['schema_version'] != 1 or observation['origin_perf_s'] != origin:
        raise ValueError('unsupported GC schema or mismatched perf-counter origin')
    intervals, unpaired = [], []
    for index, event in enumerate(observation['events']):
        start, end = event['start_perf_s'], event['end_perf_s']
        if start is None or end is None:
            unpaired.append(dict(event_index=index, **event))
            continue
        if not finite(start) or not finite(end) or end < start:
            raise ValueError('invalid paired GC interval')
        if event['generation'] not in (0, 1, 2) or event['phase'] not in PHASES:
            raise ValueError('unsupported GC generation or start phase')
        intervals.append((index, start, end, event['generation'], event['phase']))
    if (observation['unpaired_start_count'] != sum(e['end_perf_s'] is None for e in unpaired)
            or observation['unpaired_stop_count'] != sum(e['start_perf_s'] is None for e in unpaired)
            or observation['error_count'] != len(observation['errors'])):
        raise ValueError('GC diagnostic counters disagree with recorded events')
    installed, removed = observation['installed_perf_s'], observation['removed_perf_s']
    if not finite(installed) or not finite(removed) or removed < installed:
        raise ValueError('GC observer installation/removal boundary unavailable or invalid')
    if any(start < installed or end > removed for _, start, end, _, _ in intervals):
        raise ValueError('GC event outside observer lifetime')
    result = dict(available=True, schema_version=1,
        origin_perf_s=origin, installed_s=installed-origin, removed_s=removed-origin,
        enabled_at_install=observation['enabled_at_install'],
        thresholds_at_install=observation['thresholds_at_install'],
        event_count=len(observation['events']), paired_event_count=len(intervals),
        unpaired_start_count=observation['unpaired_start_count'],
        unpaired_stop_count=observation['unpaired_stop_count'], unpaired_events=unpaired,
        error_count=observation['error_count'], errors=observation['errors'],
        callback_count=observation['callback_count'],
        callback_wall_s=observation['callback_wall_s'],
        callback_cpu_s=observation['callback_cpu_s'], cpu_clock=observation['cpu_clock'],
        observed_union=partition(intervals, installed, removed),
        service_union=partition(intervals, origin, origin+raw['observation_end_s']))
    return intervals, result


def summarize_raw(raw):
    origin = raw['measurement_origin_perf_counter_s']
    if not finite(origin):
        raise ValueError('missing perf-counter origin')
    intervals, gc = gc_intervals(raw)
    requests = {r['request_id']: r for r in raw['requests']}
    if len(requests) != len(raw['requests']):
        raise ValueError('duplicate canonical request IDs')
    batches = {}
    for event in raw['output_events']:
        if event['request_id'] not in requests:
            raise ValueError('output event has unknown canonical request ID')
        if event['chunk_size'] != len(event['new_token_ids']):
            raise ValueError('output chunk size mismatch')
        if event['chunk_size'] == 0:
            continue  # A completion without a new token is not a token receipt.
        index, stamp = event['engine_call_index'], event['received_s']
        if type(index) is not int or index < 0 or not finite(stamp):
            raise ValueError('invalid host output event')
        if index not in batches:
            batches[index] = dict(t=stamp, request_ids=set())
        if batches[index]['t'] != stamp:
            raise ValueError('one engine call has multiple host receipt timestamps')
        batches[index]['request_ids'].add(event['request_id'])
    ordered = sorted(batches)
    maxima, interval_requests = collections.defaultdict(set), collections.defaultdict(set)
    for rid, request in requests.items():
        times = request['token_times_s']
        gaps = list(zip(times, times[1:]))
        maximum = max((right-left for left, right in gaps), default=0)
        for left, right in gaps:
            if right-left > GAP_THRESHOLD_S:
                interval_requests[(left, right)].add(rid)
                if right-left == maximum:
                    maxima[(left, right)].add(rid)
    rows, nonconsecutive = [], []
    long_gap_union = gc_union = 0.0
    all_shared, all_maximal = set(), set()
    for left_index, right_index in zip(ordered, ordered[1:]):
        left, right = batches[left_index], batches[right_index]
        start, end = left['t'], right['t']
        if end <= start:
            raise ValueError('non-increasing host receipt timestamps across engine calls')
        if end-start <= GAP_THRESHOLD_S:
            continue
        if right_index != left_index+1:
            nonconsecutive.append(dict(start_s=start, end_s=end, duration_s=end-start,
                engine_call_before=left_index, engine_call_after=right_index))
            continue
        shared = interval_requests[(start, end)]
        if shared != left['request_ids'] & right['request_ids']:
            raise ValueError('raw token timestamps and output-event membership disagree')
        maximal = maxima[(start, end)]
        overlap = (dict(available=False, union_s=None) if intervals is None else
                   dict(available=True, **partition(intervals, origin+start, origin+end)))
        if intervals is not None:
            overlap['fraction_of_host_gap'] = overlap['union_s']/(end-start)
            gc_union += overlap['union_s']
        rows.append(dict(start_s=start, end_s=end, duration_s=end-start,
            start_perf_s=origin+start, end_perf_s=origin+end,
            engine_call_before=left_index, engine_call_after=right_index,
            requests_output_before=len(left['request_ids']),
            requests_output_after=len(right['request_ids']),
            shared_interval_requests=len(shared), shared_request_ids=sorted(shared),
            requests_with_this_as_maxgap=len(maximal), maxgap_request_ids=sorted(maximal),
            gc_overlap=overlap))
        long_gap_union += end-start  # Adjacent receipt intervals have disjoint interiors.
        all_shared.update(shared)
        all_maximal.update(maximal)
    return dict(gc_observation=gc, public_gaps_over_250ms=rows,
        public_gap_count=len(rows), public_gap_union_s=long_gap_union,
        gc_intersection_with_public_gap_union_s=gc_union if intervals is not None else None,
        unique_shared_interval_requests=len(all_shared),
        unique_requests_with_public_gap_as_maxgap=len(all_maximal),
        nonconsecutive_call_gaps_over_250ms=nonconsecutive,
        nonconsecutive_call_gap_count=len(nonconsecutive),
        output_receiving_engine_calls=len(ordered))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite existing output; choose a new path.')
    payload = args.analysis.read_bytes()
    analysis = json.loads(payload)
    cells = []
    for cell in analysis['cells']:
        path = Path(cell['cell'])/'raw.json'
        raw_payload = path.read_bytes()
        if hashlib.sha256(raw_payload).hexdigest() != cell['raw_sha256']:
            raise ValueError('raw hash differs from source analysis: '+str(path))
        raw = json.loads(raw_payload)
        cells.append(dict(cell=cell['cell'], raw_sha256=cell['raw_sha256'],
            admission_configuration=cell.get('admission', {}).get('configuration'),
            service={k: cell[k] for k in ('planned_requests', 'arrived_requests', 'outcomes',
                'observation_end_s', 'service_denominator_s', 'total_output_tokens')},
            **summarize_raw(raw)))
        print(f"{path.parent.name}: gaps={cells[-1]['public_gap_count']}; "
              f"GC available={cells[-1]['gc_observation']['available']}")
    result = dict(schema_version=1, source_analysis=str(args.analysis.resolve()),
        source_analysis_sha256=hashlib.sha256(payload).hexdigest(),
        gap_threshold_s=GAP_THRESHOLD_S, independent_unit='run', cells=cells,
        semantics='Host token-receipt intervals between consecutive engine call indices. '
            'Calls without new output tokens are excluded and nonconsecutive-index gaps are '
            'listed separately. No host output occurs inside each listed interval. Shared '
            'request counts require a token at both endpoints; they differ from per-request '
            'maxgap counts. Equal maxima can match multiple intervals; unique request counts '
            'use sets. GC intervals use absolute perf_counter start/stop callback boundaries, '
            'not pure GC CPU time or GPU/copy completion. Unpaired events are unresolved and '
            'excluded from interval totals, not assigned zero duration. Union/intersection '
            'time is counted once. Breakdown rows partition time by the active set of '
            '(generation,start phase) labels: multiple labels share one row, so these rows '
            'sum to the union without double counting. Phase denotes host measurement phase '
            'at start callback, not attribution to a library or cause. Temporal overlap is '
            'descriptive, not a causal saving or an online signal. Callback overhead is '
            'reported as recorded; no GC, callback, or gap time is subtracted from service.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
