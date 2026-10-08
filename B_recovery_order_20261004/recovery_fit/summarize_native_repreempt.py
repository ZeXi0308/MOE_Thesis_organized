#!/usr/bin/env python3
"""Existing two r02 native cells: output-after-restore then re-preemption branches."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results, inputs = [], {}
    for cell in ('cell-01-cap256-native', 'cell-02-cap256-native'):
        directory = args.session/cell/'output'
        def read(name):
            path = directory/(name+'.json'); content = path.read_bytes()
            inputs[str(path)] = hashlib.sha256(content).hexdigest()
            return json.loads(content)
        raw, selective, trace = read('raw'), read('selective-store'), read('recovery-order')
        rows = {row['request_id']: row for row in raw['requests']}
        previous, jobs = {}, {}
        for event in trace['events']:
            if event.get('is_store') is False and event['kind'] in (
                    'job_created', 'ready', 'submit_begin', 'job_completed', 'ack_retired'):
                jobs.setdefault(event['job_id'], {}).setdefault(event['kind'], event['host_perf_s'])
        for event in raw['preemption_events']:
            if not event.get('original_preemption_returned'):
                continue
            rid = event['internal_request_id']; old = previous.get(rid); previous[rid] = event
            if old is None:
                continue
            outputs = [t for t in rows[event['request_id']]['token_times_s']
                       if old['method_returned_s'] <= t < event['method_entered_s']]
            if not outputs:
                continue
            matches = [v for v in selective['victim_decisions']
                       if v['selected'] == rid and v['step'] == event['engine_call_index']]
            if len(matches) != 1:
                raise ValueError('Native preemption has no unique actual decision: '+rid)
            decision = matches[0]; stamp = decision['host_perf_counter_s']
            candidates = {row['request']: row for row in decision['candidates']}
            def state(key):
                row = candidates[key]
                computed, held = row['bidkv']['computed_tokens'], row['held_blocks']
                return dict(held=held, computed=computed, residency_outputs=row['residency_outputs'],
                    next_token_block_gap=max(0, (computed+1+15)//16-held))
            pending = [key for key, job in jobs.items()
                       if job.get('job_created', float('inf')) <= stamp < job.get('ack_retired', float('inf'))]
            results.append(dict(cell=cell, target=event['request_id'], step=event['engine_call_index'],
                outputs_since_previous_preempt=len(outputs),
                failed_request=raw['internal_to_source'][decision['failed_request']],
                free_blocks=decision['free_blocks'], changed=decision['changed'],
                native_tail_selected=decision['native_tail'] == decision['selected'],
                target_state=state(rid), failed_state=state(decision['failed_request']),
                pending_created_loads=pending,
                all_load_lifecycles_have_ack=all('ack_retired' in job for job in jobs.values()),
                preempt_minus_logger_us=(raw['measurement_origin_perf_counter_s']+
                    event['method_entered_s']-stamp)*1e6))
    result = dict(rows=results, inputs_sha256=inputs,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        semantics='Only the two frozen fit-r02 native cells. Same request and engine step joins actual '
            'FCFS allocation-failure decisions; no strategy modification or alternative trajectory. '
            'Block arithmetic assumes the already-qualified16-token block and a next pure-decode token, '
            'not a recorded full allocate_slots argument. Job acknowledgement is native host confirmation, '
            'not an absolute GPU completion timestamp. No pending established LOAD does not cover future tasks.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(matched=len(results), all_other_requests=all(
        row['target'] != row['failed_request'] for row in results), pending_load_decisions=sum(
        bool(row['pending_created_loads']) for row in results))))


if __name__ == '__main__':
    main()
