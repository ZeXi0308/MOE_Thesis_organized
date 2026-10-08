#!/usr/bin/env python3
"""Check recorded full-tail actions and reuse frozen all-request/recovery metrics."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze import read, summary

PINS = {
    'analyze.py': 'baeb1f2904a72039c948be3ff6e57fe6d585deaff4b4a5705032b41b6794d8a3',
    'normal_capacity/analyze_group.py': 'bab5488cdbc0d586645c5ba905c9d4661ea3e36fd5083613acc80825895f089c',
    'capacity_handoff/analyze_tail.py': '7bbb95c0b8fba105493685783233a642c71a87e342ea172784dc87cc374f04cc',
}


def frozen_analyzers():
    for name, expected in PINS.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen analyzer changed: '+name)
    path = ROOT/'normal_capacity/analyze_group.py'; source = path.read_text()
    old = '(native|age|flush_first)'
    if source.count(old) != 1:
        raise RuntimeError('Cell-name adapter boundary changed')
    namespace = dict(__file__=str(path), __name__='tail_group_adapter')
    exec(compile(source.replace(old, '(native|age|flush_first|full_tail)'), str(path), 'exec'), namespace)
    spec = importlib.util.spec_from_file_location('frozen_tail_analysis', ROOT/'capacity_handoff/analyze_tail.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return namespace['analyze_group'], module.analyze


def check_actions(raw, policy, observer, recovery, config, mode):
    """Use nested host intervals, never event IDs or cross-run state matches."""
    origin = raw['measurement_origin_perf_counter_s']; mapping = raw['internal_to_source']
    size = observer['qualification']['block_size']; maxlen = config['max_model_len']
    page_bytes = config.get('actual_gpu_kv_bytes_per_block')
    allocations = [e for e in observer['events'] if e['kind'] == 'allocate']
    jobs = {}
    for event in recovery['events']:
        if event.get('is_store') is False and event.get('job_id') is not None:
            jobs.setdefault(event['job_id'], {})[event['kind']] = event
    rows = []; used_jobs = set()
    for index, event in enumerate(policy['events']):
        rid = event['request']; requested, effective = event['requested'], event['effective']
        changed = requested != effective; history = event['request_num_tokens']
        computed = (event['request_num_computed_tokens'] + requested['num_new_computed_tokens']
                    + requested['num_external_computed_tokens'])
        matches = [(i, a) for i, a in enumerate(allocations) if a['request'] == rid
            and event['host_perf_s'] <= a['begin_host_perf_s']
            and a['end_host_perf_s'] <= event['return_host_perf_s']]
        checks = dict(original_once=event['original_calls'] == 1, mode=policy['mode'] == mode,
            logged_scope=event['num_preemptions'] > 0 and requested['num_new_tokens'] == 0
                and requested['num_external_computed_tokens'] > 0,
            changed_flag=event['changed'] == changed,
            only_new_tokens_changed=all(effective.get(k) == v for k, v in requested.items() if k != 'num_new_tokens')
                and set(requested) == set(effective),
            native_unchanged=mode != 'native' or not changed, unique_nested_allocation=len(matches) == 1)
        if changed:
            checks['allowed_change'] = (mode == 'full_tail' and policy.get('qualification_fallback') is None
                and event.get('fallback') is None and requested['delay_cache_blocks'] is True
                and requested['full_sequence_must_fit'] is True
                and not any(requested[k] for k in ('num_lookahead_tokens', 'num_encoder_tokens', 'num_new_computed_tokens'))
                and 0 <= computed < history <= maxlen
                and event['tail_tokens'] == effective['num_new_tokens'] == history-computed)
        checks['expected_effective'] = effective['num_new_tokens'] == (
            event['tail_tokens'] if mode == 'full_tail' and event.get('fallback') is None else requested['num_new_tokens'])
        row = dict(index=index, source_request=mapping.get(rid), record=event, checks=checks,
            changed=changed, begin_s=event['host_perf_s']-origin, load=None)
        if len(matches) == 1:
            ai, allocation = matches[0]; before, after = allocation['before'], allocation['after']
            checks['effective_arguments'] = all(allocation['arguments'].get(k) == v for k, v in effective.items())
            checks['outcome_matches'] = event['outcome'] == ('exception' if allocation.get('exception') else
                'allocated' if allocation['success'] else 'allocation_failed')
            checks['snapshots_match'] = all(event[side]['held_blocks'] == snap['held_gpu_blocks'][0]
                and event[side]['free_blocks'] == snap['free_gpu_blocks'] for side, snap in (('before', before), ('after', after)))
            if changed:
                checks['exact_supported'] = (allocation['descriptor']['exact'] and before['num_in_flight_tokens'] == 0
                    and not any(allocation['arguments']['new_computed_block_counts']))
            hb, ha = before['held_gpu_blocks'][0], after['held_gpu_blocks'][0]
            fb, fa = before['free_gpu_blocks'], after['free_gpu_blocks']
            prefix = (computed+size-1)//size; target = (history+size-1)//size
            row.update(allocation_index=ai, held_before=hb, held_after=ha, free_before=fb, free_after=fa,
                held_delta=ha-hb, free_consumed=fb-fa, history_blocks=target,
                original_prefix_target_blocks=prefix, extra_held_above_original_prefix=max(0, ha-max(hb, prefix)))
            if allocation['descriptor']['exact']:
                checks['free_matches_held_delta'] = fb-fa == ha-hb
                if allocation['success'] and changed:
                    checks['full_history_held'] = ha == target
                    checks['returned_tail_blocks'] = event.get('returned_new_blocks') == max(0, target-max(hb, prefix))
                if not allocation['success'] and not allocation.get('exception'):
                    checks['failed_no_allocation'] = (hb, fb) == (ha, fa)
        if event['outcome'] == 'allocated':
            later = [e['host_perf_s'] for e in policy['events'][index+1:] if e['request'] == rid and e['outcome'] == 'allocated']
            stop = min(later, default=float('inf'))
            candidates = [(jid, stages) for jid, stages in jobs.items() if 'job_created' in stages
                and stages['job_created']['request'] == rid
                and event['return_host_perf_s'] <= stages['job_created']['host_perf_s'] < stop]
            checks['unique_load_job'] = len(candidates) == 1
            if len(candidates) == 1:
                jid, stages = candidates[0]; checks['load_job_not_reused'] = jid not in used_jobs; used_jobs.add(jid)
                expected = (requested['num_external_computed_tokens']//size*page_bytes if page_bytes
                    and computed == requested['num_external_computed_tokens'] and computed % size == 0 else None)
                actual = stages.get('job_completed', {}).get('bytes')
                row['load'] = dict(job_id=jid, expected_prefix_bytes=expected, observed_completed_bytes=actual,
                    prefix_work_matches=(actual == expected if actual is not None and expected is not None else None),
                    host_stages_s={k: e['host_perf_s']-origin for k, e in stages.items()})
                if row['load']['prefix_work_matches'] is not None:
                    checks['load_work_not_inflated'] = row['load']['prefix_work_matches']
        rows.append(row)
    errors = [dict(index=r['index'], failed=[k for k, v in r['checks'].items() if not v])
              for r in rows if not all(r['checks'].values())]
    unverified = [r['index'] for r in rows if r['record']['outcome'] == 'allocated'
                  and (r['load'] is None or r['load']['prefix_work_matches'] is None)]
    missing_jobs = sorted(set(jobs)-used_jobs)
    return dict(status='FAIL' if errors else 'UNVERIFIED' if unverified or missing_jobs else 'PASS',
        mode=mode, recorded_calls=len(rows), counts=dict(Counter(('changed/' if r['changed'] else 'unchanged/')
            +r['record']['outcome'] for r in rows)), errors=errors, unverified_load_calls=unverified,
        unmatched_load_jobs=missing_jobs, rows=rows, qualification_fallback=policy.get('qualification_fallback'))


def analyze_session(session):
    group_analyze, tail_analyze = frozen_analyzers(); result = group_analyze(session)
    for cell in result['cells']:
        output = Path(cell['directory']); names = ('raw', 'tail-reservation', 'capacity-handoff', 'recovery-order', 'config')
        missing = [n for n in names if not (output/(n+'.json')).exists()]
        if missing:
            cell['tail_actions'] = dict(status='UNAVAILABLE', missing=missing); continue
        raw, policy, observer, recovery, config = [read(output/(n+'.json')) for n in names]
        cell['tail_actions'] = check_actions(raw, policy, observer, recovery, config, cell['mode'])
        allocation = tail_analyze(raw, observer, recovery); origin = raw['measurement_origin_perf_counter_s']
        for episode in allocation['episodes']:
            preempt = episode['preempt_record']
            joined = [r for r in cell['recovery_events'] if r['request'] == episode['source_request']
                and preempt['begin_host_perf_s'] <= r['demand_host_s']+origin <= preempt['end_host_perf_s']]
            episode['request_recovery'] = joined[0] if len(joined) == 1 else None
            episode['request_recovery_join_count'] = len(joined)
            next_output = joined[0]['next_output_s'] if len(joined) == 1 else None
            episode['ack_to_next_output'] = [dict(job_id=j['job_id'], ack_s=j['times']['ack_retired']-origin,
                next_output_s=next_output, elapsed_s=next_output-(j['times']['ack_retired']-origin) if next_output is not None else None)
                for j in episode['load_jobs'] if 'ack_retired' in j['times']]
        cell['tail_allocations'] = allocation
        cell['tail_actions']['ack_to_next_output_s'] = summary([j['elapsed_s']
            for e in allocation['episodes'] for j in e['ack_to_next_output']])
    result['analyzer_sources_sha256'] = PINS
    result['action_semantics'] = ('Only recorded recovering async calls are checked. Nested host intervals join '
        'the unchanged observer; missing records are explicit. Full-tail block targets cover current history, '
        'not future outputs. LOAD checks use actual GPU page bytes and completed payload bytes, not destination '
        'identity or copy duration. Equal held counts do not prove block identity. All host phases are observations; '
        'no absolute GPU finish or cross-run matched-state causal effect is inferred. Frozen group analyzer is '
        'adapted only in memory to accept full_tail directory names; all request/failure/SLO denominators are retained.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = analyze_session(args.session)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), cells=[dict(directory=c['directory'],
        status=c['tail_actions']['status'], counts=c['tail_actions'].get('counts')) for c in result['cells']])))


if __name__ == '__main__':
    main()
