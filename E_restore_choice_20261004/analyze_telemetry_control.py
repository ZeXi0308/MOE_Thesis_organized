"""Local CPU analysis of Host-only GPU telemetry on/off/off/on control.

Usage: python analyze_telemetry_control.py LOCAL_COMPLETE_GROUP [--out PATH]
Reuses analyze.py for service/cost/output metrics; never imports torch/vLLM.
Missing data is UNRUN, not a telemetry benefit. Exit 0 validates execution;
matched_work separately requires identical output, metadata and native schedule.
CPU time can include CUDA polling: it is not GPU kernel duration or CPU math.
"""
import argparse
from bisect import bisect_right
from itertools import combinations
import json
import math
from pathlib import Path

import analyze as base
from analyze_headroom import container_checks
from analyze_one_event import difference


EXPECTED_MODES = [True, False, False, True]
OLD_TELEMETRY_SHA = '4edd510cdb51346051b7993dc2f6d46aff456dc0d87ccf3ea0447e7f6a084ae3'


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def telemetry_log(path, enabled):
    errors, samples = [], []
    if not path.is_file():
        return dict(errors=['missing_telemetry_log'], samples=0, file_exists=False)
    data = path.read_text()
    if not enabled:
        return dict(errors=[] if not data.strip() else ['off_log_not_empty'],
                    samples=0, file_exists=True, bytes=path.stat().st_size)
    for i, line in enumerate(data.splitlines()):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
            assert isinstance(sample, dict)
            assert all(finite(sample.get(k)) for k in ('monotonic_s', 'unix_s', 'query_wall_s'))
            assert sample['query_wall_s'] >= 0
            assert 'host_cpu' not in sample, 'extended observer was not the intended factor'
            for query in ('gpu', 'compute_processes'):
                assert isinstance(sample[query], dict)
                assert sample[query].get('returncode') == 0 and not sample[query].get('error')
            assert sample['gpu']['stdout'].strip()
            samples.append(sample)
        except (KeyError, TypeError, ValueError, AssertionError) as exc:
            errors.append(f'line_{i + 1}:invalid_old_gpu_sample:{exc}')
    if not samples:
        errors.append('on_log_has_no_valid_samples')
    times = [s['monotonic_s'] for s in samples]
    if any(b <= a for a, b in zip(times, times[1:])):
        errors.append('telemetry_samples_not_strictly_ordered')
    return dict(errors=errors, samples=len(samples), file_exists=True, bytes=path.stat().st_size,
                first_sample_monotonic_s=times[0] if times else None,
                last_sample_monotonic_s=times[-1] if times else None,
                query_wall_s=base.stats(s['query_wall_s'] for s in samples),
                interval_s=base.stats(b-a for a, b in zip(times, times[1:])))


def step_observations(raw, include_drain=False):
    """Map formal steps strictly; separately retain bounded post-service empty drains.

    The default two-value return is preserved for existing callers.
    """
    rows = raw['requests']
    names = {r['request_id']: r['external_id'] for r in rows}
    if len(names) != len(rows) or len(set(names.values())) != len(rows):
        raise ValueError('duplicate request identities')
    steps = raw['steps']
    if any(not (finite(s.get('start_s')) and finite(s.get('end_s')) and s['start_s'] < s['end_s']) for s in steps):
        raise ValueError('invalid service step interval')
    if any(a['end_s'] > b['start_s'] for a, b in zip(steps, steps[1:])):
        raise ValueError('overlapping service steps')
    complete, drain_end = raw.get('all_complete_s'), raw.get('service_and_drain_s')
    if not (finite(complete) and finite(drain_end) and 0 <= complete <= drain_end
            and (not steps or steps[-1]['end_s'] <= complete)):
        raise ValueError('invalid service completion or drain boundary')
    empty_drains = []
    starts, by_step = [s['start_s'] for s in steps], {}
    for schedule in raw['scheduler_steps']:
        t = schedule['time_s']
        i = bisect_right(starts, t) - 1 if finite(t) else -1
        if i >= 0 and t <= steps[i]['end_s']:
            if i in by_step:
                raise ValueError('duplicate native schedule within service step')
            by_step[i] = schedule
        elif finite(t) and complete < t <= drain_end:
            if schedule.get('scheduled') != []:
                raise ValueError('post-service drain must have an empty scheduled list')
            empty_drains.append(dict(time_s=t, free_blocks_after_schedule=schedule['free_blocks_after_schedule']))
        else:
            raise ValueError('native schedule outside service steps and bounded drain interval')
    if len(by_step) != len(steps):
        raise ValueError('service step without exactly one native schedule')
    trace, observed = [], []
    for i, step in enumerate(steps):
        schedule = by_step[i]
        entries = schedule['scheduled']
        if len({e['request_id'] for e in entries}) != len(entries):
            raise ValueError('duplicate scheduled request')
        normalized = []
        for e in entries:
            if type(e['count']) is not int or e['count'] <= 0 or e['end_computed'] - e['start_computed'] != e['count']:
                raise ValueError('invalid native scheduled position count')
            normalized.append([names[e['request_id']]] + [e[k] for k in
                ('count', 'start_computed', 'end_computed', 'known_tokens', 'generated_tokens')])
        trace.append(dict(scheduled=normalized, free_blocks_after_schedule=schedule['free_blocks_after_schedule']))
        if any(not finite(step.get(k)) or step[k] < 0 for k in ('process_cpu_s', 'driver_thread_cpu_s')):
            raise ValueError('per-step CPU timing must remain enabled in every arm')
        wall = step['end_s'] - step['start_s']
        observed.append(dict(step=i, scheduled_requests=len(entries), scheduled_tokens=sum(e['count'] for e in entries),
            wall_s=wall, process_cpu_s=step['process_cpu_s'], driver_thread_cpu_s=step['driver_thread_cpu_s'],
            wall_minus_driver_cpu_s=wall-step['driver_thread_cpu_s']))
    drain = dict(empty_schedule_count=len(empty_drains), empty_schedules=empty_drains,
                 all_complete_s=complete, service_and_drain_s=drain_end, wall_s=drain_end-complete)
    return (trace, observed, drain) if include_drain else (trace, observed)


TIME_FIELDS = ('wall_s', 'process_cpu_s', 'driver_thread_cpu_s', 'wall_minus_driver_cpu_s')


def summarize_steps(rows):
    return dict(steps=len(rows), scheduled_tokens=sum(r['scheduled_tokens'] for r in rows),
                **{k: base.stats(r[k] for r in rows) for k in TIME_FIELDS})


def timing_summary(rows):
    # Frozen existing headroom timing definition; no timing-dependent segment selection.
    tail = 1 + max((r['step'] for r in rows if r['scheduled_requests'] > 32), default=-1)
    buckets = [(1, 1, '1'), (2, 8, '2-8'), (9, 32, '9-32'), (33, 128, '33-128'), (129, float('inf'), '129+')]
    return dict(all_steps=summarize_steps(rows), small_batch_tail_start_step=tail,
        small_batch_tail=summarize_steps(rows[tail:]),
        scheduled_request_buckets={label: summarize_steps([r for r in rows if lo <= r['scheduled_requests'] <= hi])
                                   for lo, hi, label in buckets},
        empty_schedule_steps=sum(r['scheduled_requests'] == 0 for r in rows))


def analyze_group(group):
    group = Path(group).resolve()
    service = base.analyze_group(group)
    raw_files = sorted(group.rglob('raw.json'))
    checks = container_checks(group, raw_files)
    cells, contexts, errors = [], {}, []
    for summary in service['cells']:
        path = Path(summary['raw_path'])
        cell = path.parent
        config = summary['config']
        failures = []
        try:
            if config.get('policy') != 'host' or config.get('policies') != 'host,host,host,host':
                failures.append('expected_four_Host_arms')
            enabled = config.get('gpu_telemetry_enabled')
            if type(enabled) is not bool or config.get('gpu_telemetry_modes') != 'on,off,off,on':
                failures.append('invalid_telemetry_mode_configuration')
            if config.get('timing_observer') is not True or config.get('target_spec') is not None:
                failures.append('per_step_CPU_timing_disabled_or_target_intervention')
            hash_path = next((p / 'runtime_source_hashes.json' for p in (cell, cell.parent)
                              if (p / 'runtime_source_hashes.json').is_file()), None)
            hashes = base.read_json(hash_path, {}) if hash_path is not None else {}
            if not isinstance(hashes, dict):
                raise ValueError('runtime source hashes must be an object')
            if not config.get('telemetry_script') or hashes.get(config['telemetry_script']) != OLD_TELEMETRY_SHA:
                failures.append('frozen_old_telemetry_script_hash_missing_or_mismatched')
            log = telemetry_log(cell / 'timing_observer.jsonl', enabled)
            failures.extend(log['errors'])
            raw = base.read_json(path)
            origin = raw.get('clock_alignment', {}).get('episode_origin_monotonic_s')
            if finite(origin) and log.get('first_sample_monotonic_s') is not None:
                log['first_sample_relative_to_episode_origin_s'] = log['first_sample_monotonic_s'] - origin
                log['last_sample_relative_to_episode_origin_s'] = log['last_sample_monotonic_s'] - origin
                log['completion_minus_last_sample_s'] = raw['all_complete_s'] - log['last_sample_relative_to_episode_origin_s']
            trace, times, drain = step_observations(raw, include_drain=True)
            controlled_config = {k: v for k, v in config.items() if k not in ('out', 'gpu_telemetry_enabled')}
            contexts[summary['cell']] = dict(trace=trace, times=times, drain=drain,
                fixed_config=base.fingerprint(controlled_config), runtime_sources=base.fingerprint(hashes))
            cells.append(dict(cell=summary['cell'], enabled=enabled, errors=failures,
                              telemetry_log=log, timing=timing_summary(times), steps=times, drain=drain))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            failures.append(f'malformed_cell:{type(exc).__name__}:{exc}')
            cells.append(dict(cell=summary['cell'], enabled=config.get('gpu_telemetry_enabled'), errors=failures))
        errors.extend(f"{summary['cell']}:{e}" for e in failures)
    pairs = []
    base_pairs = {(p['left_cell'], p['right_cell']): p for p in service['greedy_token_consistency']}
    for left, right in combinations(service['cells'], 2):
        pair = dict(left_cell=left['cell'], right_cell=right['cell'], matched_work=False)
        a, b = contexts.get(left['cell']), contexts.get(right['cell'])
        if a is not None and b is not None:
            token = base_pairs[(left['cell'], right['cell'])]
            metadata = {k: token['matching_metadata'][k] for k in
                        ('external_workload', 'prompt_token_ids', 'engine_args', 'resources', 'warmup_shape')}
            metadata.update(fixed_config_except_gpu_telemetry=a['fixed_config'] == b['fixed_config'],
                            runtime_sources=a['runtime_sources'] == b['runtime_sources'])
            schedule = difference(a['trace'], b['trace'])
            outputs_equal = not (token['differences'] or token['completion_metadata_differences'] or
                                 token['missing_from_left'] or token['missing_from_right'])
            pair.update(matching_metadata=metadata, normalized_schedule=schedule, outputs_equal=outputs_equal,
                matched_work=all(v is True for v in metadata.values()) and schedule['equal'] and outputs_equal,
                right_minus_left_makespan_s=right['makespan_s']-left['makespan_s'])
            drain_left, drain_right = a['drain'], b['drain']
            pair['drain_comparison'] = dict(
                left_empty_schedule_count=drain_left['empty_schedule_count'],
                right_empty_schedule_count=drain_right['empty_schedule_count'],
                empty_schedule_boundaries=difference(
                    [s['free_blocks_after_schedule'] for s in drain_left['empty_schedules']],
                    [s['free_blocks_after_schedule'] for s in drain_right['empty_schedules']]),
                right_minus_left_drain_wall_s=drain_right['wall_s']-drain_left['wall_s'],
                right_minus_left_service_and_drain_s=drain_right['service_and_drain_s']-drain_left['service_and_drain_s'],
                interpretation='Only validated empty schedules after all_complete_s and through service_and_drain_s. Count/boundary differences are reported but are not scheduled compute differences; drain wall time and base costs remain recorded.')
            if schedule['equal']:
                deltas = [dict(step=x['step'], scheduled_requests=x['scheduled_requests'], scheduled_tokens=x['scheduled_tokens'],
                               **{k: y[k]-x[k] for k in TIME_FIELDS}) for x, y in zip(a['times'], b['times'])]
                pair['right_minus_left_step_timing'] = timing_summary(deltas)
        pairs.append(pair)
    observed_modes = [c['enabled'] for c in cells]
    complete = bool(checks) and all(c['status'] == 'COMPLETE' for c in checks)
    failed = bool(errors or service['analysis_errors'] or service['invalid_cells'] or any(c['status'] == 'FAILED' for c in checks))
    if len(cells) == 4 and observed_modes != EXPECTED_MODES:
        errors.append('formal_mode_order_mismatch')
        failed = True
    status = ('INVALID' if failed else 'UNRUN' if not raw_files else 'VALIDATED_COMPLETE'
              if complete and len(cells) == 4 and observed_modes == EXPECTED_MODES and not service['incomplete_cells'] else 'INCOMPLETE')
    return dict(schema='E.telemetry_control_analysis.v1', group=str(group), status=status,
        validation_passed=status == 'VALIDATED_COMPLETE', matched_work=status == 'VALIDATED_COMPLETE' and len(pairs) == 6 and all(p['matched_work'] for p in pairs),
        expected_modes=EXPECTED_MODES, observed_modes=observed_modes, errors=errors, container_status_checks=checks,
        cells=cells, pairs=pairs, base_analysis=service,
        interpretation='Host-only on/off/off/on comparison of the frozen old 1 Hz GPU telemetry subprocess. Per-step CPU and native/output logging remain enabled. Formal service schedules remain one-to-one with measured steps; bounded post-service empty drains are reported separately, retained in complete costs, and not classified as scheduled compute differences. Tail starts after the last formal step with more than 32 scheduled requests. No timing-based segment selection, offline correction, GPU kernel-time inference, policy benefit or quality claim. Work differences invalidate equal-work attribution but remain descriptive; valid matched work alone does not prove telemetry caused every observed timing difference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = analyze_group(args.group)
    out = args.out or args.group / 'telemetry_control_summary.json'
    if out.parent.is_dir():
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    else:
        out = None  # An unrun/nonexistent group is not materialized as experiment data.
    print(json.dumps(dict(status=result['status'], validation_passed=result['validation_passed'],
                         matched_work=result['matched_work'], report=str(out) if out else None), ensure_ascii=False))
    return 0 if result['validation_passed'] else 1 if result['status'] == 'INVALID' else 2


if __name__ == '__main__':
    raise SystemExit(main())
