#!/usr/bin/env python3
"""Same-once storage contrast: inherited request/action metrics and passive GC joins."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PINS = {
    'recovery_repeat/analyze.py': '51ce8d545808df00ca0cb10628cc02abe37c9e9dc1b6e3a894d3303edf635982',
    'recovery_gc_diag/analyze.py': 'b7fe9cab001ac9da95dc8b1dc645258ac115223629a7fa5ca55ec05e1185ab64',
}


def load(name):
    path = BASE/name
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
        raise RuntimeError('Frozen analysis source changed: '+name)
    spec = importlib.util.spec_from_file_location('storage_'+path.parent.name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def adapted_repeat():
    repeat = load('recovery_repeat/analyze.py')
    # These aliases affect directory grouping and reference selection only.
    source = inspect.getsource(repeat.helpers)
    for old, new in (('(once|repeat8)', '(legacy|compact)'),
                     ("endswith('-once')", "endswith('-legacy')"),
                     ('Nearest once cell', 'Nearest legacy cell'),
                     ('Candidate or once raw unavailable', 'Candidate or legacy raw unavailable')):
        source = repeat.replace_once(source, old, new)
    namespace = dict(vars(repeat))
    exec(compile(source, str(BASE/'recovery_repeat/analyze.py')+'[storage-groups]', 'exec'), namespace)
    repeat.helpers = namespace['helpers']
    original_actions = repeat.actions

    def same_once(data, raw, cell, allocations):
        # Keep the recorded mode, counters and events intact for real validation.
        policy_cell = dict(cell, mode='once')
        return original_actions(data, raw, policy_cell, allocations)

    repeat.actions = same_once
    return repeat


def storage_diagnostic(raw, timing, mode, number):
    data = raw.get('output_event_storage') if raw else None
    if not isinstance(data, dict):
        return dict(status='UNVERIFIED', missing=['raw.output_event_storage'])
    fields = ('mode', 'event_count', 'materialization_s', 'materialization_applied',
              'materialization_start_host_perf_s', 'materialization_end_host_perf_s')
    missing = [name for name in fields if name not in data]; failures = []
    if data.get('mode') != mode: failures.append('STORAGE_MODE_MISMATCH')
    events = raw.get('output_events')
    if not isinstance(events, list): missing.append('raw.output_events')
    elif data.get('event_count') != len(events): failures.append('OUTPUT_EVENT_COUNT_MISMATCH')
    start, end = (data.get(key) for key in fields[-2:])
    duration = data.get('materialization_s')
    origin, horizon = raw.get('measurement_origin_perf_counter_s'), raw.get('observation_end_s')
    after_observation = None
    if mode == 'compact':
        if data.get('materialization_applied') is not True: failures.append('MATERIALIZATION_NOT_APPLIED')
        if not all(number(value) for value in (start, end, duration, origin, horizon)):
            missing.append('valid_materialization_and_observation_clocks')
        else:
            after_observation = start >= origin+horizon
            if not after_observation or end < start or not math.isclose(end-start, duration, abs_tol=1e-9):
                failures.append('INVALID_POST_OBSERVATION_MATERIALIZATION_INTERVAL')
            returned = timing.get('measurement_return_perf_s')
            if number(returned) and end > returned: failures.append('MATERIALIZATION_AFTER_CAPTURE_RETURN')
        if isinstance(events, list) and data.get('materialized_event_count') != len(events):
            failures.append('MATERIALIZED_EVENT_COUNT_MISMATCH')
    elif mode == 'legacy':
        if (data.get('materialization_applied') is not False or duration != 0
                or start is not None or end is not None):
            failures.append('UNEXPECTED_LEGACY_MATERIALIZATION')
    return dict(status='FAIL' if failures else 'UNVERIFIED' if missing else 'ANALYZED',
        missing=missing, failed_checks=failures, record=data,
        materialization_after_request_observation=after_observation,
        semantics='Legacy dictionaries are built during capture. Compact materialization is after raw '
            'observation_end_s and included in capture-return/process phases; it is not subtracted from '
            'those phases or added to client token timestamps. Metadata records are execution declarations.')


def generation2(gc, diagnostic):
    group = next((row for row in diagnostic.get('per_generation', []) if row['generation'] == 2), None)
    if group is None:
        return dict(status='UNVERIFIED', reason='Generation-2 interval observations unavailable')
    intervals = group['retained_union_intervals_s']; events = group['retained_events']
    gaps = diagnostic['shared_output_gaps']['top10']
    return dict(status=diagnostic['status'], retained_event_count=len(events),
        retained_union_intervals_s=intervals, retained_union_s=group['retained_union_s'],
        maximum_clipped_interval_s=max((row['clipped_duration_s'] for row in events), default=0.),
        maximum_original_interval_s=max((row['original_duration_s'] for row in events), default=0.),
        top10_shared_gap_overlap=gc.overlap([(row['begin_s'], row['end_s']) for row in gaps], intervals))


def analyze_session(session):
    repeat, gc = adapted_repeat(), load('recovery_gc_diag/analyze.py')
    result = repeat.analyze_session(session)
    for cell in result['cells']:
        directory = Path(cell['directory']); optional = repeat.ORIGINAL_OPTIONAL
        raw, probe, timing = (optional(directory, name) for name in ('raw', 'gc-observation', 'timing'))
        timing = timing or {}
        cell['expected_recovery_policy'] = 'once'
        cell['output_event_storage'] = storage_diagnostic(raw, timing, cell['mode'], gc.number)
        planned = cell.get('planned', (optional(directory, 'config') or {}).get('requests', 0))
        cell['gc_diagnostic'] = gc.gc_diagnostic(raw, probe, planned)
        cell['gc_generation2'] = generation2(gc, cell['gc_diagnostic'])
        storage = (raw or {}).get('output_event_storage', {})
        cell['run_summary']['phase_times'].update(
            request_observation_s=(raw or {}).get('observation_end_s'),
            output_event_materialization_s=storage.get('materialization_s'))
        cell['gpu_copy_work_duration_s'] = {direction: cell.get('copy_work', {}).get(direction, {}).get('gpu_elapsed_sum_s')
                                            for direction in ('load', 'store')}
    by_path = {cell['directory']: cell for cell in result['cells']}
    for pair in result['comparisons']:
        pair['legacy'] = pair.pop('once')
        for old, new in (('aggregate_delta_candidate_minus_once', 'aggregate_delta_compact_minus_legacy'),
                         ('copy_work_delta_candidate_minus_once', 'copy_work_delta_compact_minus_legacy')):
            if old in pair: pair[new] = pair.pop(old)
        if 'fixed1024_contracts' in pair:
            pair['fixed1024_contracts']['legacy'] = pair['fixed1024_contracts'].pop('once')
        for row in pair.get('per_request', []): row['legacy_status'] = row.pop('once_status')
        members = {key: by_path.get(pair.get(key)) for key in ('candidate', 'legacy')}
        pair['same_once_policy_observations'] = {key: dict(
            status=cell['recovery_repeat_actions']['status'],
            recorded_mode=(cell['recovery_repeat_actions'].get('raw_action_record') or {}).get('mode'),
            action_limit=cell['recovery_repeat_actions'].get('action_limit'),
            actual_bypasses=cell['recovery_repeat_actions'].get('actual_completed_bypass_count'))
            if cell else None for key, cell in members.items()}
        pair['storage_observation_status'] = {key: cell['output_event_storage']['status'] if cell else 'UNAVAILABLE'
                                              for key, cell in members.items()}
        candidate, legacy = members['candidate'], members['legacy']
        def delta(section, key):
            a = candidate.get(section, {}).get(key) if candidate else None
            b = legacy.get(section, {}).get(key) if legacy else None
            return a-b if gc.number(a) and gc.number(b) else None
        pair['gc_gen2_delta_compact_minus_legacy'] = {key: delta('gc_generation2', key)
            for key in ('retained_event_count', 'retained_union_s', 'maximum_clipped_interval_s')}
        phases = set((candidate or {}).get('run_summary', {}).get('phase_times', {}))
        phases.update((legacy or {}).get('run_summary', {}).get('phase_times', {}))
        pair['phase_delta_compact_minus_legacy_s'] = {}
        for key in sorted(phases):
            a = candidate['run_summary']['phase_times'].get(key) if candidate else None
            b = legacy['run_summary']['phase_times'].get(key) if legacy else None
            pair['phase_delta_compact_minus_legacy_s'][key] = a-b if gc.number(a) and gc.number(b) else None
    modes = [cell['mode'] for cell in result['cells']]; expected = ['legacy', 'compact', 'compact', 'legacy']
    result['execution_layout'] = dict(design='OUTPUT_EVENT_STORAGE_SAME_ONCE', cell_count=len(modes),
        modes=modes, expected_abba=expected, complete_abba=modes == expected,
        has_control='legacy' in modes and 'compact' in modes, comparison_count=len(result['comparisons']))
    result['recovery_repeat_semantics'] = ('Every arm uses the same once policy and action budget 1. '
        'Directory aliases only group output-event storage; the unmodified once action validator receives '
        'the original recovery-repeat record, including its real mode and counters. Actual recovery actions '
        'and request trajectories can differ between runs and remain explicit.')
    result['semantics'] = result['semantics'].replace('once cells', 'legacy cells')
    result['output_event_storage_semantics'] = ('Orthogonal output-event storage experiment, not a recovery-policy '
        'acceleration comparison. Nearest legacy reference preserves forward/reverse pairs, earlier on ties; '
        'missing controls remain unavailable. All original request, failure, unfinished, fixed-output, SLO, '
        'output-sequence and copy-work results are retained. Host phases and post-observation materialization '
        'are separate from GPU copy-duration sums; they are not added into request benefit. GC/shared-gap '
        'interval intersections are descriptive temporal association, not causal attribution. Historical '
        'runs are not controls; two run pairs are not request-level independent replicates.')
    for name in (*PINS, 'output_event_compact/analyze.py'):
        result['analyzer_sources_sha256'][name] = hashlib.sha256((BASE/name).read_bytes()).hexdigest()
    return result


def self_test():
    repeat = adapted_repeat(); group, _, _ = repeat.helpers()
    compare = group.__globals__['comparisons']
    cells = [dict(directory=f'/fixture/cell-{i:02d}-cap256-{mode}/output', status='RAW_UNAVAILABLE')
             for i, mode in enumerate(('legacy', 'compact', 'compact', 'legacy'))]
    pairs = compare(cells)
    assert [(pair['candidate'], pair['native']) for pair in pairs] == [
        (cells[1]['directory'], cells[0]['directory']), (cells[2]['directory'], cells[3]['directory'])]
    assert all(pair['status'] == 'UNAVAILABLE' for pair in pairs)
    missing = compare([cells[1]])
    assert len(missing) == 1 and missing[0]['native'] is None and missing[0]['status'] == 'UNAVAILABLE'
    record = dict(mode='once', action_limit=1, status='UNINSTALLED', events=[], action_count=0, shadow_count=0)
    raw = dict(measurement_origin_perf_counter_s=0, internal_to_source={})
    cell = dict(mode='compact')
    assert repeat.actions(record, raw, cell, None)['status'] == 'ANALYZED'
    assert cell['mode'] == 'compact' and record['mode'] == 'once'
    wrong = repeat.actions(dict(record, mode='compact'), raw, cell, None)
    assert wrong['status'] == 'FAIL' and 'mode_mismatch' in wrong['failed_checks']
    print('PASS: nearest legacy ABBA pairing; absent control unavailable; storage aliases preserve real once validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true'); args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.session is None and args.output is None: return
    if args.session is None or args.output is None:
        parser.error('--session and --output are required for real analysis')
    if args.output.exists(): raise FileExistsError(args.output)
    result = analyze_session(args.session)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), layout=result['execution_layout'])))


if __name__ == '__main__':
    main()
