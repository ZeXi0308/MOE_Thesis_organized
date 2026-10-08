#!/usr/bin/env python3
"""Reuse all-request metrics and check the single Host-source reference handoff."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import json
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analyze import read

PINS = {
    'analyze.py': 'baeb1f2904a72039c948be3ff6e57fe6d585deaff4b4a5705032b41b6794d8a3',
    'normal_capacity/analyze_group.py': 'bab5488cdbc0d586645c5ba905c9d4661ea3e36fd5083613acc80825895f089c',
    'capacity_handoff/analyze_tail.py': '7bbb95c0b8fba105493685783233a642c71a87e342ea172784dc87cc374f04cc',
    'tail_reservation/summarize_runs.py': '2cfb5b49210be4a905824f3402e1db2901386953851122597628b809c5f87f02',
}


def helpers():
    for name, sha in PINS.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Frozen analysis source changed: '+name)
    path = BASE/'normal_capacity/analyze_group.py'; text = path.read_text()
    old = '(native|age|flush_first)'
    if text.count(old) != 1: raise RuntimeError('Cell-name adapter boundary changed')
    namespace = dict(__name__='source_group_adapter', __file__=str(path))
    exec(compile(text.replace(old, '(native|age|flush_first|early_pin)'), str(path), 'exec'), namespace)
    modules = []
    for name in ('capacity_handoff/analyze_tail.py', 'tail_reservation/summarize_runs.py'):
        spec = importlib.util.spec_from_file_location('source_'+Path(name).stem, BASE/name)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); modules.append(module)
    return namespace['analyze_group'], modules[0].analyze, modules[1].summarize_cell


def check_actions(data, raw, cell, allocations, block_size):
    checks = {}; missing = []; origin = raw['measurement_origin_perf_counter_s']
    def check(name, value):
        checks[name] = value
        if value is None: missing.append(name)
    events = data.get('events', [])
    if not isinstance(events, list): events = []; missing.append('events_list')
    for key in ('mode', 'status', 'selected_request', 'events', 'remaining_early_refs'):
        if key not in data: missing.append('data.'+key)
    known = [e for e in events if isinstance(e, dict) and 'kind' in e and isinstance(e.get('host_perf_s'), (int, float))]
    if len(known) != len(events): missing.append('event_kind_or_time')
    by_kind = {kind: [e for e in known if e['kind'] == kind] for kind in {e['kind'] for e in known}}
    selected = by_kind.get('selected', []); pins = by_kind.get('early_pin', [])
    handoffs = by_kind.get('native_handoff', []); releases = by_kind.get('release', [])
    errors = [e for e in known if 'error' in e['kind'] or e.get('error') is not None]
    required = dict(selected=('request', 'external_tokens', 'history_tokens', 'gpu_free', 'key_repr', 'before', 'valid', 'action'),
        early_pin=('before_refs', 'after', 'acquired_refs'),
        allocation=('request', 'num_new_tokens', 'external_tokens', 'success', 'gpu_held', 'gpu_free', 'before', 'after'),
        native_handoff=('external_tokens', 'load_jobs', 'before_early_release'),
        release=('reason', 'before', 'after', 'remaining_early_refs'))
    for event in known:
        missing.extend(event['kind']+'.'+key for key in required.get(event['kind'], ()) if key not in event)
    check('mode_matches', data.get('mode') == cell['mode'] if 'mode' in data else None)
    check('selected_at_most_one', len(selected) <= 1); check('pin_at_most_one', len(pins) <= 1)
    check('native_no_pin', not pins if cell['mode'] == 'native' else True)
    check('no_error_events', not errors)
    check('uninstalled', data.get('status') == 'UNINSTALLED' if 'status' in data else None)
    check('remaining_zero', data.get('remaining_early_refs') == 0 if 'remaining_early_refs' in data else None)
    check('known_event_kinds', all(e['kind'] in ('selected', 'early_pin', 'allocation', 'native_handoff', 'release') or e in errors for e in known))
    check('event_time_order', all(a['host_perf_s'] <= b['host_perf_s'] for a, b in zip(known, known[1:])))
    selection = selected[0] if len(selected) == 1 else None
    check('selection_field_matches', data.get('selected_request') == (selection.get('request') if selection else None)
        if len(selected) <= 1 and 'selected_request' in data else None)
    source = raw.get('internal_to_source', {}).get(data.get('selected_request'))
    observations = []; per_key = []; selected_episode = None; linked_jobs = []
    if selection:
        keys = selection.get('key_repr'); n = len(keys) if isinstance(keys, list) else 0
        if not n: missing.append('selected.key_repr')
        per_key = [dict(index=i, key_repr=key) for i, key in enumerate(keys or [])]
        for key in ('request', 'external_tokens', 'history_tokens', 'gpu_free', 'valid', 'action'):
            if key not in selection: missing.append('selected.'+key)
        check('selected_source_mapped', source in {r['request_id'] for r in raw['requests']} if source is not None else None)
        check('one_handoff_at_most', len(handoffs) <= 1)
        check('one_release', len(releases) == 1 if releases else None)
        check('unique_keys', len(set(keys)) == n if n else None)
        ext = selection.get('external_tokens')
        check('selected_key_count', n*block_size == ext if n and block_size and ext is not None else None)
        expected_pin = cell['mode'] == 'early_pin' and selection.get('valid') is True
        check('selected_action', selection.get('action') == ('early_pin' if expected_pin else 'no_early_pin')
            if 'valid' in selection and 'action' in selection else None)
        check('expected_pin', len(pins) == 1 if expected_pin and pins else None if expected_pin else not pins)

        def snapshot(event, field):
            rows = event.get(field)
            if not isinstance(rows, list) or len(rows) != n or any(not isinstance(r, dict)
                or not all(k in r for k in ('cached', 'ready', 'ref_cnt', 'block_id'))
                or type(r['cached']) is not bool or type(r['ready']) is not bool
                or r['cached'] and (not isinstance(r['ref_cnt'], int) or not isinstance(r['block_id'], int)) for r in rows):
                missing.append(event['kind']+'.'+field); return None
            return rows

        before = snapshot(selection, 'before')
        if selection.get('valid') is True:
            check('selected_ready', all(r['cached'] and r['ready'] and isinstance(r['ref_cnt'], int)
                and r['ref_cnt'] >= 0 for r in before) if before is not None else None)
        for event in known:
            if event is not selection and event['kind'] in ('allocation',) and event.get('request') != data.get('selected_request'):
                check('allocation_request_matches', False)
            for field in ('before', 'after', 'before_early_release'):
                if field not in event: continue
                rows = snapshot(event, field)
                if rows is None: continue
                lost = [i for i, r in enumerate(rows) if not r['cached']]
                unready = [i for i, r in enumerate(rows) if r['cached'] and not r['ready']]
                changed_ids = [i for i, r in enumerate(rows) if before is not None and r['cached']
                    and before[i]['cached'] and r['block_id'] != before[i]['block_id']]
                observations.append(dict(kind=event['kind'], field=field, host_s=event['host_perf_s']-origin,
                    missing_key_indices=lost, unready_key_indices=unready, changed_block_id_indices=changed_ids))
                for i, row in enumerate(rows):
                    per_key[i][event['kind']+'_'+field+'_ref'] = row['ref_cnt']
                    if not row['cached'] and 'first_loss_observed_s' not in per_key[i]:
                        per_key[i]['first_loss_observed_s'] = event['host_perf_s']-origin
                    if i in changed_ids and 'first_relocation_observed_s' not in per_key[i]:
                        per_key[i]['first_relocation_observed_s'] = event['host_perf_s']-origin
        pin = pins[0] if len(pins) == 1 else None
        if pin:
            after = snapshot(pin, 'after'); refs = pin.get('before_refs')
            refs_ok = isinstance(refs, list) and len(refs) == n and all(isinstance(r, int) for r in refs)
            if not refs_ok: missing.append('early_pin.before_refs')
            check('pin_after_selection', pin['host_perf_s'] >= selection['host_perf_s'])
            check('pin_acquired_each_key', pin.get('acquired_refs') == n if 'acquired_refs' in pin else None)
            check('pin_ref_plus_one', all(r['cached'] and r['ready'] and r['ref_cnt'] == v+1
                for v, r in zip(refs, after)) if refs_ok and after is not None else None)
            check('pin_keeps_selected_blocks', all(b['block_id'] == a['block_id'] for b, a in zip(before, after))
                if before is not None and after is not None else None)
            if refs_ok:
                for row, ref in zip(per_key, refs): row['early_pin_before_ref'] = ref
        for release in releases:
            rb, ra = snapshot(release, 'before'), snapshot(release, 'after')
            check('release_remaining_zero', release.get('remaining_early_refs') == 0 if 'remaining_early_refs' in release else None)
            check('release_after_pin', release['host_perf_s'] >= pin['host_perf_s'] if pin else True)
            check('release_ref_minus_one' if pin else 'native_release_no_ref_change',
                all(a['cached'] and b['cached'] and a['block_id'] == b['block_id']
                    and isinstance(a['ref_cnt'], int) and a['ref_cnt'] == b['ref_cnt']-1 for b, a in zip(rb, ra))
                if pin and rb is not None and ra is not None else rb == ra if rb is not None and ra is not None else None)
        for handoff in handoffs:
            hs = snapshot(handoff, 'before_early_release'); jids = handoff.get('load_jobs')
            if not isinstance(jids, list): missing.append('native_handoff.load_jobs'); jids = []
            if 'external_tokens' not in handoff: missing.append('native_handoff.external_tokens')
            count = (handoff['external_tokens']//block_size if isinstance(handoff.get('external_tokens'), int)
                and block_size and handoff['external_tokens'] >= 0 and handoff['external_tokens'] % block_size == 0 else None)
            check('handoff_load_presence', bool(jids) == (count > 0) if count is not None else None)
            if count is not None:
                for i, row in enumerate(per_key): row['native_load_expected'] = i < count
            matched = hs[:min(n, count)] if hs is not None and count is not None else None
            check('handoff_matched_ref_at_least_two' if pin else 'handoff_matched_ref_positive',
                all(r['cached'] and r['ready'] and r['ref_cnt'] >= (2 if pin else 1) for r in matched)
                if matched is not None else None)
            if pin:
                check('handoff_all_early_refs_positive', all(r['cached'] and r['ready'] and r['ref_cnt'] >= 1 for r in hs)
                    if hs is not None else None)
            for jid in jids:
                jobs = [j for j in cell.get('jobs', []) if j['job_id'] == jid]
                check('job_'+str(jid)+'_identity', jobs[0]['request'] == source and jobs[0]['is_store'] is False if len(jobs) == 1 else None)
                for job in jobs:
                    missing.extend('job_'+str(jid)+'.'+stage for stage in ('ready_s', 'submit_begin_s', 'submit_end_s', 'job_completed_s', 'ack_retired_s')
                        if job.get(stage) is None)
                linked_jobs.extend(jobs)
        if pin and len(releases) == 1:
            owned_observations = [r for r in observations if pin['host_perf_s']-origin <= r['host_s'] <= releases[0]['host_perf_s']-origin]
            check('pinned_source_survives', all(not r['missing_key_indices'] and not r['unready_key_indices']
                and not r['changed_block_id_indices'] for r in owned_observations))
        stamp = selection['host_perf_s']
        episodes = [e for e in (allocations or {}).get('episodes', []) if e['request'] == selection.get('request')
                    and e['begin_host_perf_s'] <= stamp <= e['end_host_perf_s']]
        check('selected_episode_unique', len(episodes) == 1 if allocations is not None else None)
        if len(episodes) == 1:
            selected_episode = episodes[0]
            check('selected_request_recovery_unique', selected_episode.get('request_recovery_join_count') == 1)
    else:
        check('no_unselected_actions', not known if not selected else True)
    failures = [k for k, value in checks.items() if value is False]
    losses = [r for r in observations if r['missing_key_indices']]
    unready = [r for r in observations if r['unready_key_indices']]
    relocations = [r for r in observations if r['changed_block_id_indices']]
    last_external = next((e.get('external_tokens') for e in reversed(known) if 'external_tokens' in e), None)
    return dict(status='FAIL' if failures else 'UNVERIFIED' if missing else 'PASS', checks=checks,
        failed_checks=failures, missing=missing, error_events=errors, raw_action_record=data,
        outcome='NO_SELECTION' if not selected else 'PINNED' if pins else 'SELECTED_WITHOUT_PIN',
        selected_count=len(selected), actual_pin_count=len(pins), selected_internal_request=data.get('selected_request'),
        selected_source_request=source, selected_recovery_episode=selected_episode,
        first_key_loss_observed_s=min((r['host_s'] for r in losses), default=None),
        first_unready_observed_s=min((r['host_s'] for r in unready), default=None), source_observations=observations,
        first_relocation_observed_s=min((r['host_s'] for r in relocations), default=None),
        per_key=per_key, last_observed_external_tokens=last_external,
        native_handoff_external_tokens=handoffs[0].get('external_tokens') if len(handoffs) == 1 else None,
        linked_load_jobs=linked_jobs)


def analyze_session(session):
    group, tail, summarize = helpers(); result = group(session)
    for cell in result['cells']:
        output = Path(cell['directory']); names = ('raw', 'source-handoff', 'capacity-handoff', 'recovery-order')
        paths = {name: output/(name+'.json') for name in names}
        absent = [name for name, path in paths.items() if not path.exists()]
        if absent:
            cell['source_actions'] = dict(status='UNVERIFIED', missing=absent,
                raw_action_record=read(paths['source-handoff']) if paths['source-handoff'].exists() else None)
        else:
            raw, data, observer, recovery = [read(paths[name]) for name in names]
            allocation = tail(raw, observer, recovery); origin = raw['measurement_origin_perf_counter_s']
            for episode in allocation['episodes']:
                preempt = episode['preempt_record']
                joined = [r for r in cell['recovery_events'] if r['request'] == episode['source_request']
                    and preempt['begin_host_perf_s'] <= r['demand_host_s']+origin <= preempt['end_host_perf_s']]
                episode.update(request_recovery=joined[0] if len(joined) == 1 else None, request_recovery_join_count=len(joined))
                next_output = joined[0]['next_output_s'] if len(joined) == 1 else None
                episode['ack_to_next_output'] = [dict(job_id=j['job_id'], ack_s=j['times']['ack_retired']-origin,
                    next_output_s=next_output, elapsed_s=next_output-(j['times']['ack_retired']-origin) if next_output is not None else None)
                    for j in episode['load_jobs'] if 'ack_retired' in j['times']]
            cell['recovery_allocations'] = allocation
            cell['source_actions'] = check_actions(data, raw, cell, allocation, observer.get('qualification', {}).get('block_size'))
        compact = summarize(cell, session/'source-metrics.json')
        cell['run_summary'] = {k: compact[k] for k in ('phase_times', 'fixed1024_contract', 'actual_outputs_known',
            'output_count_missing_requests', 'stop_reason_counts', 'actual_output_count_distribution',
            'missing_planned_request_rows', 'recovery', 'per_request')}
    result['analyzer_sources_sha256'] = PINS
    result['source_action_semantics'] = ('All native/early_pin runs, unselected runs, errors and missing observations are retained. '
        'Reference checks use selected key indices; native LOAD overlap is its external-token prefix, which may differ in length. '
        'No inferred absolute GPU finish or cross-run '
        'matched-state effect. First source loss/unready times are first host log observations, not exact eviction times; '
        'allocation.before is logged at the allocation return event and has no independent timestamp. Native release has '
        'no early reference to decrement. Candidate LOAD-matched keys must hold at least two refs before early release; '
        'unmatched pinned keys need only the early ref. Relocation means a cached key has a different observed block ID, '
        'not an exactly timed eviction. '
        'Recovery wait per request is an interval union, with unfinished intervals retained as lower bounds. Original '
        'all-request metrics, failure/SLO denominators and copy work are reused without overwriting frozen analyzers.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = analyze_session(args.session)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), cells=[dict(directory=c['directory'],
        status=c['source_actions']['status'], selected=c['source_actions'].get('selected_count'),
        pins=c['source_actions'].get('actual_pin_count')) for c in result['cells']])))


if __name__ == '__main__':
    main()
