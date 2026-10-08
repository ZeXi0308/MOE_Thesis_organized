#!/usr/bin/env python3
"""Reuse request metrics; check ready LOAD decisions against native submissions."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'e02600cd689c6a2337ecb7ff84abf4891fbd70f3eb0574e09afa96e5d6268b49'


def load_actions(directory, expected_mode, completion_mode):
    paths = [directory/name for name in ('load-order.json', 'recovery-order.json')]
    missing = [str(path) for path in paths if not path.exists()]
    if not any((directory/name).exists() for name in ('raw.json', 'raw.json.gz')):
        missing.append(str(directory/'raw.json[.gz]'))
    if missing:
        return dict(status='UNAVAILABLE', missing=missing)
    data, order = (json.loads(path.read_text()) for path in paths)
    if not isinstance(data.get('events'), list) or not isinstance(order.get('events'), list):
        return dict(status='UNAVAILABLE', reason='Missing LOAD decision or native event list')
    if completion_mode is None:
        return dict(status='UNAVAILABLE', reason='Native completion boundary unverified')
    errors, batches = [], []
    mode = data.get('mode')
    if mode != expected_mode or mode not in ('native', 'short_load'):
        errors.append('load_mode_mismatch')
    if completion_mode != 'native':
        errors.append('non_native_completion_boundary')
    submits = [event for event in order['events']
               if event.get('kind') == 'submit_begin' and event.get('is_store') is False]
    for index, event in enumerate(data['events']):
        before, after, sizes = event.get('before'), event.get('after'), event.get('bytes', {})
        failures = []
        if not isinstance(before, list) or not isinstance(after, list):
            batches.append(dict(index=index, recorded=event, errors=['missing_order_lists']))
            errors.append('batch_order_unavailable')
            continue
        same = collections.Counter(before) == collections.Counter(after)
        if len(before) < 2 or len(set(before)) != len(before) or not same:
            failures.append('invalid_multi_load_job_multiset')
        fallback = event.get('fallback')
        known_sizes = isinstance(sizes, dict) and all(type(sizes.get(str(jid))) is int
            and sizes[str(jid)] >= 0 for jid in before)
        expected = before
        if fallback is None:
            if data.get('layout_fallback') is not None or not isinstance(data.get('layout'), dict):
                failures.append('missing_valid_layout_declaration')
            if not known_sizes:
                failures.append('missing_valid_byte_estimates')
            elif mode == 'short_load':
                expected = sorted(before, key=lambda jid: sizes[str(jid)])
        if after != expected:
            failures.append('not_native_fallback_or_stable_size_order')
        matched = [row for row in submits if row.get('job_id') in set(before)]
        observed = [row['job_id'] for row in matched]
        if observed != after:
            failures.append('native_submission_order_or_multiset_mismatch')
        start, cost = event.get('host_perf_s'), event.get('decision_s')
        upper = data['events'][index+1].get('host_perf_s') if index+1 < len(data['events']) else None
        numeric = lambda value: type(value) in (int, float)
        if not numeric(start) or not numeric(cost) or cost < 0:
            failures.append('invalid_decision_clock')
        elif any(not numeric(row.get('host_perf_s')) or row['host_perf_s'] < start
                 or numeric(upper) and row['host_perf_s'] >= upper for row in matched):
            failures.append('submission_outside_recorded_batch_interval')
        batches.append(dict(index=index, before=before, after=after, bytes=sizes,
            fallback=fallback, host_perf_s=start, decision_s=cost,
            expected_policy_order=expected, same_job_multiset=same,
            observed_submit_begin_order=observed, submission_matches_after=observed == after,
            changed=before != after, errors=failures))
        errors.extend('batch_'+str(index)+':'+reason for reason in failures)
    costs = [event.get('decision_s') for event in data['events']]
    valid_costs = [value for value in costs if type(value) in (int, float) and value >= 0]
    return dict(status='ACTION_CHECK_FAILED' if errors else 'PASS_ACTION_CHECK',
        source=str(paths[0]), mode=mode, completion_mode=completion_mode,
        layout=data.get('layout'), layout_fallback=data.get('layout_fallback'),
        source_sha256=data.get('source_sha256'), errors=errors, batches=batches,
        multi_load_batches=len(batches), recorded_order_changes=sum(bool(row.get('changed')) for row in batches),
        verified_submission_changes=sum(bool(row.get('changed')) and not row['errors'] for row in batches),
        fallback_counts=dict(collections.Counter(str(event.get('fallback')) for event in data['events'])),
        decision_cost=dict(observed=len(valid_costs), missing=len(costs)-len(valid_costs),
            sum_s=sum(valid_costs), maximum_s=max(valid_costs, default=None)),
        semantics='Zero or single LOAD batches intentionally emit no decision event. Layout/alias checks are runtime declarations, not independent physical alias proof. submit_begin joins verify host submission order and job identity, not GPU completion or request benefit; decision cost excludes native submission.')


def analyze_group(session):
    path = ROOT.parent/'completion_handoff/analyze_group.py'; payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PARENT_SHA:
        raise RuntimeError('Completion analysis source changed')
    text = payload.decode(); old = '(native|after_sample)'
    if text.count(old) != 1:
        raise RuntimeError('LOAD analysis mode boundary changed')
    namespace = dict(__name__='load_order_parent_analysis', __file__=str(path))
    exec(compile(text.replace(old, '(native|short_load)'), str(path)+'[load_order]', 'exec'), namespace)
    result = namespace['analyze_group'](session)
    by_path = {cell['directory']: cell for cell in result['cells']}
    for cell in result['cells']:
        cell['load_order'] = load_actions(Path(cell['directory']), cell['mode'], cell['completion_finalize']['mode'])
    for contrast in result['comparisons']:
        statuses = {key: by_path[contrast[key]]['load_order']['status']
                    if contrast.get(key) in by_path else 'UNAVAILABLE' for key in ('native', 'candidate')}
        contrast['load_action_status'] = statuses
    result['semantics'] += ' Ready LOAD ordering is the only varied policy; all completion finalizers must remain native. Missing LOAD observations are UNAVAILABLE; latency contrasts alone do not establish an ordering action.'
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path); args = parser.parse_args()
    destination = args.output or args.session/'load-metrics.json'
    if destination.exists():
        raise FileExistsError(destination)
    result = analyze_group(args.session)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(destination)


if __name__ == '__main__':
    main()
