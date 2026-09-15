#!/usr/bin/env python3
"""Join native step geometry and inclusive host spans with CUDA activity union."""
import argparse
import gzip
import json
import math
from pathlib import Path
import re

GPU_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset'}
MARKER = re.compile(r'^moe_probe/on/step=(\d+)/engine\.step$')


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def union_us(intervals, lo, hi):
    clipped = sorted((max(lo, a), min(hi, b)) for a, b in intervals if a < hi and b > lo)
    merged = []
    for a, b in clipped:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b-a for a, b in merged), merged


def analyze(directory):
    raw = read(directory / 'cell-000.json')
    p = raw['profiling_probe']
    mode = p['mode']
    issues = []
    markers, activities = {}, []
    categories, streams, devices, association_fields = set(), set(), set(), set()
    if mode == 'on':
        path = directory / 'trace-000.json'
        if not path.exists():
            path = directory / 'trace-000.json.gz'
        if not path.exists():
            issues.append('TRACE_MISSING')
        else:
            trace = read(path)
            for event in trace['traceEvents']:
                if event.get('ph') != 'X':
                    continue
                cat = event.get('cat', '')
                categories.add(cat)
                start, duration = event['ts'], event['dur']
                if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
                    raise ValueError('invalid trace timestamp/duration')
                if cat in GPU_CATEGORIES:
                    activities.append((start, start+duration))
                    metadata = event.get('args', {})
                    devices.add(str(metadata.get('device', 'UNKNOWN')))
                    streams.add(str((metadata.get('device', 'UNKNOWN'), metadata.get('stream', event.get('tid')))))
                    association_fields.update(k for k in metadata if any(s in k.lower()
                        for s in ('correlation', 'external id', 'graph', 'node')))
                match = MARKER.fullmatch(event.get('name', ''))
                if match and cat != 'gpu_user_annotation':
                    step = int(match.group(1))
                    if step in markers:
                        raise ValueError('duplicate CPU engine marker')
                    markers[step] = (start, start+duration)
            if not activities:
                issues.append('NO_CUDA_ACTIVITIES_CANNOT_ATTRIBUTE_HOST')
            if len(devices) > 1 or 'UNKNOWN' in devices:
                issues.append('CUDA_DEVICE_SCOPE_UNRESOLVED')
    spans = p['host_spans']
    steps = []
    for step in raw['scheduler_steps']:
        index = step['step']
        host = [s for s in spans if s['step_id'] == index]
        engines = [s for s in host if s['stage'] == 'engine.step']
        if len(engines) != 1:
            raise ValueError('native step must have exactly one engine span')
        row = dict(step=index, prefill_tokens=sum(r['prefill_tokens'] for r in step['scheduled']),
            decode_tokens=sum(r['decode_tokens'] for r in step['scheduled']),
            request_ids=[r['request_id'] for r in step['scheduled']],
            actual_active=step['actual_active'], waiting=step['waiting_requests'],
            inclusive_host_ms={s['stage']: 1000*s['host_elapsed_s'] for s in host},
            physical_shapes=[s for s in p['physical_shapes'] if s['step_id'] == index])
        if mode == 'on' and index in markers:
            lo, hi = markers[index]
            total, intervals = union_us(activities, lo, hi)
            row.update(trace_engine_ms=(hi-lo)/1000, cuda_union_ms=total/1000,
                cuda_covered_fraction=total/(hi-lo) if hi > lo else None,
                cuda_first_to_last_ms=(intervals[-1][1]-intervals[0][0])/1000 if intervals else 0,
                uncovered_trace_interval_ms=(hi-lo-total)/1000,
                cuda_events_crossing_marker=sum(a < hi and b > lo and (a < lo or b > hi)
                    for a, b in activities))
        elif mode == 'on':
            issues.append(f'ENGINE_MARKER_MISSING_STEP_{index}')
        steps.append(row)
    if raw['status'] != 'COMPLETE' or p['status'] != 'CAPTURED':
        issues.append('EPISODE_OR_PROFILE_INCOMPLETE')
    if not p['target_step_ids']:
        issues.append('TARGET_4_DECODE_512_PREFILL_NOT_OBSERVED')
    return dict(mode=mode, issues=issues, status='DIAGNOSTIC' if not issues else 'UNQUALIFIED',
        requests_completed=sum(r['status'] == 'completed' for r in raw['requests']),
        target_step_ids=p['target_step_ids'], trace_categories=sorted(categories),
        cuda_devices=sorted(devices), cuda_streams=sorted(streams),
        available_association_fields=sorted(association_fields),
        cuda_activity_count=len(activities), steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for label in ('off0', 'on0', 'on1', 'off1'):
        path = args.results_dir / label
        results[label] = analyze(path) if (path / 'cell-000.json').exists() else dict(status='MISSING')
    report = dict(status='MEASUREMENT_ONLY' if all(v['status'] == 'DIAGNOSTIC' for v in results.values())
        else 'INCOMPLETE_OR_UNQUALIFIED', episodes=results,
        limits=['CPU spans are inclusive and nested; do not sum them.',
            'CUDA union describes trace activity coverage, not exclusive GPU critical-path causality.',
            'Uncovered time includes launch gaps, host work and potentially missing CUDA activity.',
            'CUDA graph/event coverage must be checked before interpreting low coverage as a host bottleneck.',
            'OFF and ON regenerate states and may have different identities/geometry; no paired action Oracle.',
            'Profiler perturbation prohibits treating ON/OFF differences as optimization gains.'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
