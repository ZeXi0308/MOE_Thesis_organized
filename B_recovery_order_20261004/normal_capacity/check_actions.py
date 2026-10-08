#!/usr/bin/env python3
"""Check recorded native STORE actions, lifecycle, and source identities."""
import argparse, collections, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import read


def check(directory):
    rawpath = next((directory/n for n in ('raw.json', 'raw.json.gz') if (directory/n).exists()), None)
    if rawpath is None or not (directory/'recovery-order.json').exists():
        return dict(status='UNAVAILABLE', reason='Raw or recovery-order trace missing', directory=str(directory))
    raw = read(rawpath); observer = read(directory/'recovery-order.json'); events = observer['events']
    phases = ('job_created', 'ready', 'submit_begin', 'submit_end', 'job_completed', 'ack_retired')
    jobs = collections.defaultdict(lambda: collections.defaultdict(list)); counts = collections.defaultdict(collections.Counter)
    for index, event in enumerate(events):
        if event.get('job_id') is not None:
            jid = event['job_id']; jobs[jid][event['kind']].append((index, event)); counts[event['kind']][jid] += 1
    expected = set(counts['job_created']); mapping = raw.get('internal_to_source', {})
    sources = {r['request_id'] for r in raw['requests']}; identity_errors = []; lifecycle_errors = []; source_by_job = {}
    phase_checks = {kind: dict(unique_jobs=len(counts[kind]), missing=sorted(expected-set(counts[kind])),
        extra=sorted(set(counts[kind])-expected), nonunit={str(j): n for j, n in counts[kind].items() if n != 1}) for kind in phases}
    for jid, stages in jobs.items():
        records = [record for stage in stages.values() for _, record in stage]
        requests = {r.get('request') for r in records}; directions = {r.get('is_store') for r in records}
        valid = len(requests) == len(directions) == 1 and None not in requests and None not in directions
        internal = next(iter(requests)) if len(requests) == 1 else None; source = mapping.get(internal)
        source_by_job[jid] = source
        if not valid or source not in sources:
            identity_errors.append(dict(job_id=jid, requests=sorted(str(r) for r in requests), source=source))
        if all(len(stages.get(kind, [])) == 1 for kind in phases):
            rows = [stages[kind][0] for kind in phases]
            if any(a[0] >= b[0] or a[1]['host_perf_s'] > b[1]['host_perf_s'] for a, b in zip(rows, rows[1:])):
                lifecycle_errors.append(jid)
    checks = []; mode = observer.get('mode'); alias_reasons = (None, 'unknown_destination', 'overlapping_destination')
    for index, event in enumerate(events):
        if event['kind'] != 'reorder':
            continue
        before, after, required = event['before'], event['after'], set(event['required'])
        stop = index+1; submits = []
        while stop < len(events) and events[stop]['kind'] in ('submit_begin', 'submit_end'):
            if events[stop]['kind'] == 'submit_begin':
                submits.append(events[stop]['job_id'])
            stop += 1
        def ready(jid):
            stages = jobs.get(jid, {}); rows = stages.get('ready', [])
            return (len(rows) == 1 and rows[0][0] < index and rows[0][1].get('is_store') is True
                and rows[0][1].get('required_flush') == (jid in required)
                and all(at > index for at, _ in stages.get('submit_begin', [])))
        fallback = event.get('fallback'); flush_fallback = event.get('flush_fallback')
        stable_flush = sorted(before, key=lambda jid: jid not in required)
        expected_order = before if fallback else (stable_flush if mode == 'flush_first' else sorted(before) if mode == 'age' else before)
        legal_flush = before if flush_fallback else stable_flush
        dependency_ok = all(any(at < index and row.get('is_store') is True for at, row in jobs.get(jid, {}).get('job_created', [])) for jid in required)
        wait_ok = not required or (stop < len(events) and events[stop]['kind'] == 'wait_begin' and set(events[stop]['required']) == required)
        checks.append(dict(t=event['host_perf_s']-raw['measurement_origin_perf_counter_s'], before=before,
            after=after, required=sorted(required), actual=submits, changed=before != after, exact=submits == after,
            same_multiset=collections.Counter(before) == collections.Counter(after), unique_batch=len(before) == len(set(before)),
            ready_unsubmitted=all(ready(jid) for jid in before), dependency_preserved=dependency_ok and wait_ok,
            legal_rule=(mode in ('native', 'age', 'flush_first') and after == expected_order),
            alias_fallback_consistent=(fallback in alias_reasons and flush_fallback in alias_reasons
                and event.get('legal_flush_order') == legal_flush and (not fallback or before == after)
                and (mode != 'flush_first' or fallback == flush_fallback)),
            changed_flag_consistent=event['changed'] == (before != after)))
    flags = ('exact', 'same_multiset', 'unique_batch', 'ready_unsubmitted', 'dependency_preserved',
             'legal_rule', 'alias_fallback_consistent', 'changed_flag_consistent')
    phase_ok = all(not c['missing'] and not c['extra'] and not c['nonunit'] for c in phase_checks.values())
    accepted = all(row.get('accepted') is True for stages in jobs.values() for _, row in stages.get('submit_end', []))
    errors = [r for r in checks if not all(r[k] for k in flags)]
    return dict(status='PASS' if phase_ok and accepted and not errors and not identity_errors and not lifecycle_errors else 'FAIL',
        directory=str(directory), mode=mode, phase_checks=phase_checks, identity_errors=identity_errors,
        lifecycle_order_errors=lifecycle_errors, source_request_by_job=source_by_job, submit_accepted=accepted,
        reorder_count=len(checks), changed=sum(r['changed'] for r in checks),
        all_actual_orders_match=all(r['exact'] for r in checks), all_multisets_match=all(r['same_multiset'] for r in checks),
        fallbacks=dict(collections.Counter(str(e.get('fallback')) for e in events if e['kind']=='reorder')),
        failed_checks=errors, reorder_checks=[r for r in checks if r['changed']],
        semantics='Six phases and identities are checked for all observed jobs, including post-request drain. '
            'Alias checks verify recorded fallback decisions; destination block IDs are absent, so physical aliasing '
            'is not independently recomputed. All times are host observations. PASS verifies actions, not request benefit.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = check(args.input)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result.get(k) for k in ('status', 'mode', 'reorder_count', 'changed', 'all_actual_orders_match', 'all_multisets_match')}))


if __name__ == '__main__':
    main()
