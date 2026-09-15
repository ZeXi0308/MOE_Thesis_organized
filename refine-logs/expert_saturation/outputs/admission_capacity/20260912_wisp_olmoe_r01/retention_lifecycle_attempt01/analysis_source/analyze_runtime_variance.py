"""Join passive GC/CPU observations to retained request/engine clocks; no subtraction."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def union_length(intervals):
    total, end = 0.0, float('-inf')
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def overlap(events, start, stop, generation=None):
    selected = [e for e in events if e['start'] < stop and e['stop'] > start
                and (generation is None or e['generation'] == generation)]
    return dict(events=len(selected), union_s=union_length(
        [(max(start, e['start']), min(stop, e['stop'])) for e in selected]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--phase-analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, phases = args.input_dir, read(args.phase_analysis)
    observed = read(root / 'runtime_events.json')
    events, issues = [], []
    for event in observed['events']:
        if event.get('start_perf_ns') is None or event.get('stop_perf_ns') is None:
            issues.append('unpaired GC event')
            continue
        assert event['duration_ns'] == event['stop_perf_ns'] - event['start_perf_ns'] >= 0
        events.append(dict(start=event['start_perf_ns']/1e9, stop=event['stop_perf_ns']/1e9,
            generation=event['generation'], collected=event['collected'], thread=event['thread_id']))
    results, every_call = {}, []
    for name, summary in phases['repeats'].items():
        raw = read(root / name / 'raw.json')
        outer = read(root / name / 'runtime_observation.json')
        origin = raw['measurement_origin_perf_counter_s']
        call_rows, snapshots = [], []
        assert raw['runtime_observer_enabled'] and raw['cpu_diagnostics']
        for call, phase_call in zip(raw['engine_calls'], summary['per_call'], strict=True):
            assert call['index'] == phase_call['index']
            before, after = call['runtime_before'], call['runtime_after']
            # Float conversion of a perf_counter origin can round by nanoseconds.
            tolerance = 1e-6
            valid = (origin+call['cpu_observation_start_s']-tolerance <= before['start_perf_ns']/1e9
                <= before['end_perf_ns']/1e9 <= origin+call['start_s']+tolerance
                <= origin+call['return_s']+tolerance
                and origin+call['return_s']-tolerance <= after['start_perf_ns']/1e9
                <= after['end_perf_ns']/1e9 <= origin+call['cpu_observation_end_s']+tolerance)
            if not valid:
                issues.append(f'{name}/{call["index"]}: observer clock envelope mismatch')
            snapshots.extend([before, after])
            start, stop = origin+call['start_s'], origin+call['return_s']
            stage = next(k for k, v in summary['retention']['stages'].items()
                         if call['index'] in v['engine_calls'])
            row = dict(repeat=name, index=call['index'], stage=stage,
                wall_s=stop-start, cpu_delta=call['cpu_delta'],
                observation_envelope_s=call['observation_envelope_s'],
                gc=overlap(events, start, stop),
                gc_by_generation={str(g): overlap(events, start, stop, g) for g in range(3)},
                cpu_before=before['cpu'], cpu_after=after['cpu'],
                rss_before=before['rss']['bytes'], rss_after=after['rss']['bytes'],
                load_section_ms=phase_call['load_section_ms'], bytes=phase_call['bytes'],
                groups=phase_call['groups'])
            call_rows.append(row)
        every_call.extend(call_rows)
        episode_gc = overlap(events, origin, origin+raw['observation_end_s'])
        stages = {}
        for stage in summary['retention']['stages']:
            rows = [r for r in call_rows if r['stage'] == stage]
            stages[stage] = dict(calls=[r['index'] for r in rows],
                **{key:sum(r[key] for r in rows) for key in ('wall_s','groups','bytes','load_section_ms')},
                gc_overlap_s=sum(r['gc']['union_s'] for r in rows))
        gc_in_calls = sum(r['gc']['union_s'] for r in call_rows)
        assert gc_in_calls <= episode_gc['union_s'] + 1e-6
        results[name] = dict(wall_s=raw['observation_end_s'], engine_wall_s=summary['engine_wall_s'],
            observation_envelope_s=summary['observation_envelope_s'],
            gc_episode=episode_gc, gc_in_calls_s=gc_in_calls,
            gc_outside_calls_s=max(0.0,episode_gc['union_s']-gc_in_calls),
            gc_enabled_samples=dict(Counter(str(s['gc']['enabled']) for s in snapshots)),
            gc_thresholds=sorted({tuple(s['gc']['threshold']) for s in snapshots}),
            cpu_core_samples=dict(Counter(str(s['cpu']['core_before']) for s in snapshots)),
            cpu_frequency_khz_samples=dict(Counter(str(s['cpu']['scaling_cur_freq_khz']) for s in snapshots)),
            snapshot_migrations=sum(s['cpu']['migrated'] is True for s in snapshots),
            cpu_observation_unknowns=sum(s['cpu']['status']=='unknown' for s in snapshots),
            retained_records_before=outer['before']['runtime_records_count'],
            retained_records_after=outer['after']['runtime_records_count'],
            rss_before=outer['before']['snapshot']['rss']['bytes'],
            rss_after=outer['after']['snapshot']['rss']['bytes'],
            stages=stages, calls=call_rows, requests=summary['requests'],
            equality_to_first=summary['equality_to_first'], cgroup_delta=summary['cgroup_delta'])
    output=dict(status='OBSERVATIONAL_DIAGNOSTIC', issues=issues, repeats=results,
        gc_total_events=len(events), gc_generations=dict(Counter(str(e['generation']) for e in events)),
        gc_total_envelope_s=union_length([(e['start'],e['stop']) for e in events]),
        longest_gc_events=sorted(events,key=lambda e:e['stop']-e['start'],reverse=True)[:10],
        longest_engine_calls=sorted(every_call,key=lambda r:r['wall_s'],reverse=True)[:12],
        runtime_records_final=observed['runtime_records_count'],
        semantics=['GC is a callback envelope overlapping host/CUDA progress, not subtractable pure GC tax.',
            'Per-step and whole-episode GC intersections are union lengths; nested windows are not summed.',
            'CPU frequency is a driver-reported endpoint sample, not the actual frequency throughout a step.',
            'In-capture observations remain in episode time; request clocks include them by occurrence. Outer snapshots/IO are outside this window; no full-cycle benefit or population noise bound.',
            'One engine per input; repeated documents and policy order are defined by config/episodes. Structural equality is checked separately, not assumed across policies.'])
    with args.out.open('x') as stream:
        json.dump(output,stream,indent=2);stream.write('\n')
    if issues:
        raise SystemExit('observation issues retained in output')


if __name__ == '__main__':
    main()
