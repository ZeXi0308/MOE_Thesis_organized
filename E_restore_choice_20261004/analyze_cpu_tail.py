"""Local CPU-only analysis of two Host repeats with passive CPU/GPU telemetry.

Usage: python analyze_cpu_tail.py LOCAL_GROUP [--out REPORT.json]
Default output: cpu_tail_summary.json. Exit 0 validates complete protocol/data;
matched_work and CPU observation availability are separate, not causal claims.
No GPU/remote access, counter interpolation, fitted speed threshold or correction.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path

import analyze as base
from analyze_headroom import container_checks
from analyze_one_event import difference
from analyze_telemetry_control import finite, step_observations, timing_summary, summarize_steps

OBSERVER_SHA = '49f41f15af24f10a7b51ff5eb68a52af528a8b3f43273c95bdb1fb90d7b9d8d7'
DRIVER_COUNTERS = ('voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches', 'utime_ticks', 'stime_ticks')
GROUP_COUNTERS = ('usage_usec', 'nr_periods', 'nr_throttled', 'throttled_usec',
                  'pressure_some_total_usec', 'pressure_full_total_usec')


def parsed_value(sample, key):
    value = sample.get(key, {})
    return value.get('values') if isinstance(value, dict) and value.get('error') is None else None


def cpu_samples(path, raw):
    """Preserve every line/error; align probe start (not earlier GPU query time)."""
    samples = []
    if not path.is_file():
        return [], ['missing_telemetry_log']
    origin = raw.get('clock_alignment', {}).get('episode_origin_monotonic_s')
    if not finite(origin):
        return [], ['missing_episode_monotonic_origin']
    steps = raw['steps']
    starts = [s['start_s'] for s in steps]
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        out = dict(line=line_no, relative_s=None, aligned_step=None, frequency_khz=None,
                   last_cpu=None, driver_identity=None, cgroup_identity=None, counters={}, errors=[])
        try:
            row = json.loads(line)
            h = row.get('host_cpu')
            if not isinstance(h, dict):
                raise ValueError('missing host_cpu object')
            out['host_cpu_record'] = h  # Keep unavailable fields, errors and provenance verbatim.
            out['gpu_queries'] = {k: row.get(k) for k in ('gpu', 'compute_processes')}
            for key in ('gpu', 'compute_processes'):
                if not isinstance(row.get(key), dict) or row[key].get('returncode') != 0:
                    out['errors'].append(f'{key}_query_unavailable_or_failed')
            if h.get('schema') != 'E.passive_host_cpu.v1' or not finite(h.get('monotonic_s')):
                raise ValueError('missing CPU probe schema or monotonic timestamp')
            t = h['monotonic_s']-origin
            out['relative_s'] = t
            i = bisect_right(starts, t)-1
            if i >= 0 and t <= steps[i]['end_s']:
                out['aligned_step'] = i
            out['alignment'] = ('service_step' if out['aligned_step'] is not None else
                'before_service' if steps and t < steps[0]['start_s'] else
                'post_service' if t > raw['all_complete_s'] else 'between_service_steps')
            stat, status = parsed_value(h, 'driver_task_stat'), parsed_value(h, 'driver_task_status')
            if (isinstance(stat, dict) and type(stat.get('pid')) is int and stat['pid'] > 0
                    and stat.get('pid') == h.get('driver_tid') == h.get('parent_pid')
                    and type(stat.get('starttime_ticks')) is int):
                out['driver_identity'] = [stat['pid'], stat['starttime_ticks']]
                out['last_cpu'] = stat.get('processor')
                for k in ('utime_ticks', 'stime_ticks'):
                    out['counters'][k] = stat.get(k)
                if isinstance(status, dict):
                    for k in ('voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches'):
                        out['counters'][k] = status.get(k)
                else:
                    out['errors'].append('driver_status_unavailable')
            else:
                out['errors'].append('driver_stat_missing_or_identity_mismatch')
            freq = h.get('current_cpu_frequency', {})
            if (isinstance(freq, dict) and freq.get('error') is None and finite(freq.get('khz'))
                    and freq['khz'] > 0 and type(out['last_cpu']) is int and out['last_cpu'] >= 0
                    and freq.get('cpu') == out['last_cpu']):
                out['frequency_khz'] = freq['khz']
                out['frequency_source_path'] = freq.get('source_path')
            else:
                out['errors'].append('frequency_missing_or_not_last_cpu')
            group = h.get('cgroup', {})
            prov = group.get('provenance', {})
            mapping = prov.get('mapping') or {}
            ns = group.get('parent_namespaces', {})
            obs_ns = prov.get('observer_namespaces', {})
            namespace_values = {k: ns.get(k, {}).get('value') for k in ('cgroup', 'mnt')}
            group_ok = (prov.get('error') is None and mapping.get('version') == 'v2'
                and group.get('current_membership') == prov.get('initial_membership')
                and all(namespace_values[k] and namespace_values[k] == obs_ns.get(k, {}).get('value')
                        for k in namespace_values))
            if group_ok:
                out['cgroup_identity'] = [mapping.get('directory'), group.get('current_membership'), namespace_values]
                counters, pressure = parsed_value(group, 'cpu_stat'), parsed_value(group, 'cpu_pressure')
                if isinstance(counters, dict):
                    out['counters'].update({k: counters.get(k) for k in GROUP_COUNTERS[:4]})
                else:
                    out['errors'].append('cgroup_cpu_stat_unavailable')
                if isinstance(pressure, dict):
                    for kind in ('some', 'full'):
                        out['counters'][f'pressure_{kind}_total_usec'] = pressure.get(kind, {}).get('total')
                else:
                    out['errors'].append('cgroup_cpu_pressure_unavailable')
            else:
                out['errors'].append('cgroup_v2_mapping_missing_changed_or_namespace_mismatch')
            missing = [k for k in DRIVER_COUNTERS+GROUP_COUNTERS
                       if type(out['counters'].get(k)) is not int or out['counters'][k] < 0]
            if missing:
                out['errors'].append('unavailable_counter_fields:'+','.join(missing))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            out['errors'].append(f'malformed_or_unavailable_sample:{type(exc).__name__}:{exc}')
        samples.append(out)
    return samples, [] if samples else ['empty_telemetry_log']


def intervals(samples, raw, times):
    result = []
    for a, b in zip(samples, samples[1:]):
        start, end = a['relative_s'], b['relative_s']
        record = dict(left_line=a['line'], right_line=b['line'], start_s=start, end_s=end,
                      elapsed_s=None, observed_last_cpu_changes=None, increments={}, errors=[])
        if not (finite(start) and finite(end) and end > start):
            record['errors'].append('missing_or_nonmonotonic_adjacent_sample_time')
            result.append(record)
            continue
        record['elapsed_s'] = end-start
        record['sampling_gap_over_1_5s'] = end-start > 1.5  # Disclosure only; not a performance threshold.
        same_driver = a['driver_identity'] is not None and a['driver_identity'] == b['driver_identity']
        same_group = a['cgroup_identity'] is not None and a['cgroup_identity'] == b['cgroup_identity']
        if same_driver and all(type(x['last_cpu']) is int and x['last_cpu'] >= 0 for x in (a, b)):
            record['observed_last_cpu_changes'] = int(a['last_cpu'] != b['last_cpu'])
        for key in DRIVER_COUNTERS + GROUP_COUNTERS:
            left, right = a['counters'].get(key), b['counters'].get(key)
            stable = same_driver if key in DRIVER_COUNTERS else same_group
            ok = stable and all(type(v) is int and v >= 0 for v in (left, right))
            record['increments'][key] = right-left if ok and right >= left else None
            if ok and right < left:
                record['errors'].append(f'counter_decreased_or_reset:{key}')
        # Fully enclosed steps only: do not apportion boundary steps or sample state to every token.
        enclosed = [i for i, step in enumerate(raw['steps']) if start <= step['start_s'] and step['end_s'] <= end]
        small = [times[i] for i in enclosed if 2 <= times[i]['scheduled_requests'] <= 8]
        record['small_batch_2_8_steps'] = [r['step'] for r in small]
        record['small_batch_2_8_timing'] = summarize_steps(small)
        record['endpoint_frequency_khz'] = [a['frequency_khz'], b['frequency_khz']]
        result.append(record)
    return result


def window_summary(samples, spans, start, end):
    selected = [s for s in samples if finite(s['relative_s']) and start <= s['relative_s'] <= end]
    covered = [s for s in spans if finite(s['elapsed_s']) and start <= s['start_s'] < s['end_s'] <= end]
    def totals(key, getter):
        valid = [s for s in covered if getter(s) is not None]
        return dict(sum=sum(getter(s) for s in valid) if valid else None, valid_intervals=len(valid),
                    unavailable_intervals=len(covered)-len(valid), covered_seconds=sum(s['elapsed_s'] for s in valid))
    return dict(start_s=start, end_s=end, samples=len(selected), adjacent_intervals=len(covered),
        first_sample_s=selected[0]['relative_s'] if selected else None,
        last_sample_s=selected[-1]['relative_s'] if selected else None,
        frequency_khz=base.stats(s['frequency_khz'] for s in selected),
        missing_frequency_samples=sum(s['frequency_khz'] is None for s in selected),
        sampled_last_cpus=dict(Counter(str(s['last_cpu']) for s in selected)),
        frequency_source_paths=dict(Counter(s.get('frequency_source_path') for s in selected)),
        observed_last_cpu_changes=totals('migration', lambda s:s['observed_last_cpu_changes']),
        counter_increments={k:totals(k, lambda s,k=k:s['increments'].get(k)) for k in DRIVER_COUNTERS+GROUP_COUNTERS},
        sampling_interval_s=base.stats(s['elapsed_s'] for s in covered),
        sampling_gaps_over_1_5s=sum(s['sampling_gap_over_1_5s'] for s in covered),
        interpretation='Only adjacent samples wholly inside this window; boundary portions and missing samples are not interpolated. Last-CPU changes are a lower bound on migrations. Frequencies are sampled last-CPU sysfs values, not effective per-step frequency; cgroup counters include other group members and do not exclude ancestor throttling.')


def cpu_diagnostic(path, raw, times):
    samples, errors = cpu_samples(path, raw)
    spans = intervals(samples, raw, times)
    tail = timing_summary(times)['small_batch_tail_start_step']
    start = raw['steps'][tail]['start_s'] if tail < len(times) else raw['all_complete_s']
    end = raw['steps'][-1]['end_s'] if times else raw['all_complete_s']
    runs, active = [], []
    for row in times + [dict(scheduled_requests=0)]:
        if 2 <= row['scheduled_requests'] <= 8:
            active.append(row)
        elif active:
            lo, hi = active[0]['step'], active[-1]['step']
            runs.append(dict(first_step=lo, last_step=hi, timing=summarize_steps(active),
                CPU=window_summary(samples, spans, raw['steps'][lo]['start_s'], raw['steps'][hi]['end_s'])))
            active = []
    field_availability = dict(
        frequency_khz=sum(s['frequency_khz'] is not None for s in samples),
        last_cpu=sum(type(s['last_cpu']) is int and s['last_cpu'] >= 0 for s in samples),
        **{k:sum(type(s['counters'].get(k)) is int and s['counters'][k] >= 0 for s in samples)
           for k in DRIVER_COUNTERS+GROUP_COUNTERS})
    available = any(field_availability.values())
    return dict(status='UNAVAILABLE' if not available else 'PARTIAL' if errors or any(s['errors'] for s in samples+spans) else 'AVAILABLE',
        errors=errors, sample_error_count=sum(bool(s['errors']) for s in samples),
        valid_samples_by_field=field_availability,
        tail=window_summary(samples, spans, start, end), small_batch_2_8_runs=runs,
        samples=samples, adjacent_sample_intervals=spans,
        interpretation='The recorded 1 Hz windows and contiguous 2–8-request runs expose actual fast/slow timing without fitting a cutoff. CPU probe start follows GPU queries and spans several reads. Clock-aligned correlation is not causation, kernel timing, or a basis for retrospective latency correction.')


def analyze_group(group):
    group = Path(group).resolve()
    service = base.analyze_group(group)
    checks = container_checks(group, sorted(group.rglob('raw.json')))
    cells, contexts, errors = [], {}, []
    for cell in service['cells']:
        path = Path(cell['raw_path']); config = cell['config']; name = cell['cell']
        try:
            expected = dict(policy='host', policies='host,host', gpu_telemetry_enabled=True,
                            gpu_telemetry_modes='on,on', timing_observer=True, target_spec=None)
            if not all(k in config and config[k] == v for k,v in expected.items()):
                raise ValueError('expected two Host/on,on repeats with per-step CPU timing and no target intervention')
            hash_path = next((p/'runtime_source_hashes.json' for p in (path.parent, path.parent.parent)
                              if (p/'runtime_source_hashes.json').is_file()), None)
            hashes = base.read_json(hash_path, {}) if hash_path is not None else {}
            if not config.get('telemetry_script') or hashes.get(config['telemetry_script']) != OBSERVER_SHA:
                raise ValueError('new observer identity missing or mismatched')
            raw = base.read_json(path)
            trace, times, drain = step_observations(raw, include_drain=True)
            diagnostic = cpu_diagnostic(path.parent/'timing_observer.jsonl', raw, times)
            errors.extend(f'{name}:{x}' for x in diagnostic['errors'])
            contexts[name] = dict(trace=trace, sources=base.fingerprint(hashes))
            cells.append(dict(cell=name, timing=timing_summary(times), drain=drain, CPU=diagnostic))
        except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
            errors.append(f'{name}:{type(exc).__name__}:{exc}')
    pair = None
    if len(service['cells']) == 2 and all(c['cell'] in contexts for c in service['cells']):
        a, b = service['cells']; ca, cb = contexts[a['cell']], contexts[b['cell']]
        token = service['greedy_token_consistency'][0]
        schedule = difference(ca['trace'], cb['trace'])
        metadata = dict(token['matching_metadata'], runtime_sources=ca['sources'] == cb['sources'])
        outputs = not (token['differences'] or token['completion_metadata_differences'] or token['missing_from_left'] or token['missing_from_right'])
        pair = dict(left_cell=a['cell'], right_cell=b['cell'], outputs_and_stop_equal=outputs,
                    normalized_service_schedule=schedule, matching_metadata=metadata,
                    matched_work=outputs and schedule['equal'] and all(v is True for v in metadata.values()))
    failed = bool(errors or service['analysis_errors'] or service['invalid_cells'] or any(c['status'] == 'FAILED' for c in checks))
    complete = len(cells) == 2 and all(c['status'] == 'COMPLETE' for c in checks) and not service['incomplete_cells']
    status = 'INVALID' if failed else 'UNRUN' if not service['cells'] else 'VALIDATED_COMPLETE' if complete else 'INCOMPLETE'
    return dict(schema='E.cpu_tail_analysis.v1', group=str(group), status=status,
        validation_passed=status == 'VALIDATED_COMPLETE', matched_work=status == 'VALIDATED_COMPLETE' and bool(pair and pair['matched_work']),
        expected_observer_sha256=OBSERVER_SHA, errors=errors, container_status_checks=checks,
        cells=cells, pair=pair, base_analysis=service,
        interpretation='Two fixed Host repeats; not an observer on/off experiment or an adaptive-policy test. Missing CPU fields remain unavailable, never zero. Cgroup-level increments cannot attribute stalls to this driver or exclude ancestor limits; sampled frequency and last-CPU changes do not identify effective CPU speed or total migrations. CPU time may include CUDA polling. No performance correction or causal attribution is performed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('group', type=Path); p.add_argument('--out', type=Path)
    args = p.parse_args(); report = analyze_group(args.group)
    out = args.out or args.group/'cpu_tail_summary.json'
    if out.parent.is_dir():
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    else:
        out = None
    print(json.dumps(dict(status=report['status'], matched_work=report['matched_work'], report=str(out) if out else None)))
    return 0 if report['validation_passed'] else 1 if report['status'] == 'INVALID' else 2


if __name__ == '__main__':
    raise SystemExit(main())
