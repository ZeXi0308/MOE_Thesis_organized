#!/usr/bin/env python3
"""One unchanged-once cell: inherited service metrics and GC/gap timing overlap."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPEAT_SHA = '51ce8d545808df00ca0cb10628cc02abe37c9e9dc1b6e3a894d3303edf635982'


def merged(intervals):
    result = []
    for begin, end in sorted(intervals):
        if end <= begin:
            continue
        if result and begin <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([begin, end])
    return result


def length(intervals):
    return sum(end-begin for begin, end in merged(intervals))


def overlap(left, right):
    left, right = merged(left), merged(right)
    intersections, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        begin, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > begin:
            intersections.append([begin, end])
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    a, b, common = length(left), length(right), length(intersections)
    union = a+b-common
    return dict(shared_gap_union_s=a, retained_gc_union_s=b,
        intersection_intervals_s=intersections, intersection_s=common,
        fraction_of_shared_gap=common/a if a else None,
        fraction_of_retained_gc=common/b if b else None,
        interval_union_s=union, intersection_over_union=common/union if union else None)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def shared_gaps(raw, horizon):
    gaps, maxima, invalid, missing = defaultdict(set), {}, [], []
    seen = set()
    for row in raw.get('requests', []):
        rid, times = row.get('request_id'), row.get('token_times_s')
        if not isinstance(rid, str) or rid in seen:
            invalid.append(dict(request=rid, reason='MISSING_OR_DUPLICATE_ID'))
            continue
        seen.add(rid)
        if not isinstance(times, list):
            missing.append(rid)
            continue
        if (any(not number(t) or t < 0 or t > horizon for t in times)
                or any(b < a for a, b in zip(times, times[1:]))):
            invalid.append(dict(request=rid, reason='INVALID_TOKEN_TIME_ORDER_OR_BOUNDARY'))
            continue
        positive = [(a, b) for a, b in zip(times, times[1:]) if b > a]
        maxima[rid] = max((b-a for a, b in positive), default=None)
        for pair in positive:
            gaps[pair].add(rid)
    shared = [(a, b, ids) for (a, b), ids in gaps.items() if len(ids) >= 2]
    shared.sort(key=lambda x: (-(x[1]-x[0]), x[0], x[1]))
    top = [dict(begin_s=a, end_s=b, duration_s=b-a, request_count=len(ids),
        requests=sorted(ids), largest_observed_closed_gap_request_count=sum(maxima[rid] == b-a for rid in ids))
        for a, b, ids in shared[:10]]
    return dict(shared_gap_count=len(shared), top10=top,
        all_shared_intervals=merged([(a, b) for a, b, _ in shared]),
        raw_unique_request_count=len(seen), missing_token_times_requests=missing,
        invalid_requests=invalid, requests_with_closed_positive_gap=sum(v is not None for v in maxima.values()),
        requests_without_closed_positive_gap=sum(v is None for v in maxima.values()),
        definition='Exact consecutive token_times_s endpoints, no rounding/interpolation; >=2 distinct '
            'requests share the pair. Max-gap counts use largest observed closed token intervals, including ties; '
            'they exclude TTFT and unobserved/censored tails. Failures and unfinished remain in inherited metrics.')


def gc_diagnostic(raw, probe, planned):
    if raw is None or probe is None:
        return dict(status='UNVERIFIED', missing=[name for name, value in
            [('raw', raw), ('gc-observation', probe)] if value is None])
    origin, horizon = raw.get('measurement_origin_perf_counter_s'), raw.get('observation_end_s')
    if not number(origin) or not number(horizon) or horizon < 0:
        return dict(status='UNVERIFIED', missing=['valid_raw_measurement_origin_or_end'])
    if not isinstance(raw.get('requests'), list):
        return dict(status='UNVERIFIED', missing=['raw_request_rows'])
    issues, retained = [], []
    if probe.get('status') != 'UNINSTALLED':
        issues.append('GC_OBSERVER_NOT_UNINSTALLED')
    if any(probe.get('pairing_issues', {}).values()) or probe.get('pending_at_uninstall'):
        issues.append('INCOMPLETE_GC_PAIRS')
    events = probe.get('events')
    if not isinstance(events, list):
        events = []; issues.append('MISSING_RETAINED_GC_EVENTS')
    for index, event in enumerate(events):
        begin, end, gen = event.get('start_host_perf_s'), event.get('stop_host_perf_s'), event.get('generation')
        if (not number(begin) or not number(end) or end < begin
                or type(gen) is not int or gen not in (0, 1, 2)):
            issues.append('INVALID_GC_EVENT_'+str(index)); continue
        a, b = max(0.0, begin-origin), min(horizon, end-origin)
        if b <= a:
            continue
        retained.append(dict(event_index=index, generation=gen,
            start_host_perf_s=begin, stop_host_perf_s=end,
            original_begin_s=begin-origin, original_end_s=end-origin,
            original_duration_s=end-begin, clipped_begin_s=a, clipped_end_s=b,
            clipped_duration_s=b-a, clipped=(a != begin-origin or b != end-origin)))
    union = merged([(e['clipped_begin_s'], e['clipped_end_s']) for e in retained])
    generations = []
    for gen in range(3):
        rows = [e for e in retained if e['generation'] == gen]
        intervals = merged([(e['clipped_begin_s'], e['clipped_end_s']) for e in rows])
        generations.append(dict(generation=gen, retained_events=rows,
            retained_union_intervals_s=intervals, retained_union_s=length(intervals)))
    gaps = shared_gaps(raw, horizon)
    gaps['missing_planned_request_row_count'] = max(0, planned-gaps['raw_unique_request_count'])
    if gaps['invalid_requests'] or gaps['missing_token_times_requests']:
        issues.append('INCOMPLETE_OUTPUT_GAP_OBSERVATION')
    if gaps['missing_planned_request_row_count']:
        issues.append('MISSING_PLANNED_REQUEST_ROWS')
    for gap in gaps['top10']:
        gap['gc_overlap'] = overlap([(gap['begin_s'], gap['end_s'])], union)
    all_shared = gaps.pop('all_shared_intervals')
    top_intervals = [(g['begin_s'], g['end_s']) for g in gaps['top10']]
    return dict(status='UNVERIFIED' if issues else 'ANALYZED', issues=issues,
        all_request_denominator=planned,
        measurement=dict(origin_host_perf_s=origin, end_host_perf_s=origin+horizon, duration_s=horizon),
        installation=probe.get('installation'), uninstallation=probe.get('uninstallation'),
        retention=probe.get('retention'), pairing_issues=probe.get('pairing_issues'),
        pending_at_uninstall=probe.get('pending_at_uninstall'),
        per_generation=generations, retained_gc_union_intervals_s=union,
        retained_gc_union_s=length(union),
        longest_retained_event_by_measurement_overlap=max(retained, key=lambda e:e['clipped_duration_s'], default=None),
        longest_original_retained_event_intersecting_measurement=max(retained, key=lambda e:e['original_duration_s'], default=None),
        installation_window_aggregates=probe.get('per_generation'),
        installation_aggregate_semantics='Short-event counts/durations cannot be clipped to measurement; '
            'retained unions are not total GC time because short generation-0/1 events were compressed.',
        shared_output_gaps=gaps, top10_shared_gap_overlap=overlap(top_intervals, union),
        all_shared_gap_overlap=overlap(all_shared, union),
        semantics='All intervals use host perf_counter. GC callback boundaries are not precise CPU pause '
            'or GPU times; intersection is temporal association, not proof of causation. Overlapping GC '
            'events and shared gaps are unioned before totals and ratios; no request multiplicity weighting.')


def analyze_session(session):
    path = BASE/'recovery_repeat/analyze.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != REPEAT_SHA:
        raise RuntimeError('Frozen repeat analyzer changed')
    spec = importlib.util.spec_from_file_location('gc_frozen_repeat_analysis', path)
    repeat = importlib.util.module_from_spec(spec); spec.loader.exec_module(repeat)
    result = repeat.analyze_session(session)
    if [cell['mode'] for cell in result['cells']] != ['once']:
        raise ValueError('This diagnostic requires exactly one unchanged once cell')
    cell = result['cells'][0]
    directory = Path(cell['directory'])
    def optional(name):
        path = directory/(name+'.json')
        return json.loads(path.read_text()) if path.exists() else None
    cell['gc_diagnostic'] = gc_diagnostic(optional('raw'), optional('gc-observation'), cell['planned'])
    result['execution_layout'].update(design='SINGLE_UNCHANGED_ONCE_GC_DIAGNOSTIC',
        has_control=False, complete_abba=False, comparison_count=0)
    result['gc_diagnostic_semantics'] = 'One observed once cell; not an ABBA/control contrast and no GC optimization. '
    result['analyzer_sources_sha256']['recovery_gc_diag/analyze.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def self_test():
    assert merged([(0, 2), (1, 3), (3, 4)]) == [[0, 4]]
    assert overlap([(0, 2), (1, 3)], [(1, 2), (1.5, 4)])['intersection_s'] == 2
    raw = dict(measurement_origin_perf_counter_s=100, observation_end_s=5,
        requests=[dict(request_id=r, token_times_s=t) for r, t in
                  [('a', [0, 1, 4, 5]), ('b', [1, 4]), ('c', [1, 4]), ('unfinished', [])]])
    probe = dict(status='UNINSTALLED', pairing_issues={}, pending_at_uninstall=[], events=[
        dict(generation=0, start_host_perf_s=99, stop_host_perf_s=101),
        dict(generation=1, start_host_perf_s=100.5, stop_host_perf_s=102),
        dict(generation=2, start_host_perf_s=104, stop_host_perf_s=106),
        dict(generation=2, start_host_perf_s=98, stop_host_perf_s=100)])
    d = gc_diagnostic(raw, probe, 4)
    assert d['status'] == 'ANALYZED' and d['retained_gc_union_s'] == 3
    assert [g['retained_union_s'] for g in d['per_generation']] == [1, 1.5, 1]
    gap = d['shared_output_gaps']['top10'][0]
    assert (gap['begin_s'], gap['end_s'], gap['request_count']) == (1, 4, 3)
    assert gap['largest_observed_closed_gap_request_count'] == 3
    assert gap['gc_overlap']['intersection_s'] == 1
    assert gap['gc_overlap']['intersection_over_union'] == .2
    assert d['all_request_denominator'] == 4
    assert d['shared_output_gaps']['requests_without_closed_positive_gap'] == 1
    assert gc_diagnostic(raw, None, 4)['status'] == 'UNVERIFIED'
    print('PASS: clipping at raw boundaries; overlapping GC/gap unions; exact shared token endpoints; max-gap counts; unfinished denominator; missing is unverified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.session is None and args.output is None:
            return
    if args.session is None or args.output is None:
        parser.error('--session and --output are required for real analysis')
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze_session(args.session)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), layout=result['execution_layout'],
                         gc_status=result['cells'][0]['gc_diagnostic']['status'])))


if __name__ == '__main__':
    main()
