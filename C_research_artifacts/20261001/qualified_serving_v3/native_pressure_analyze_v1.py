#!/usr/bin/env python3
"""Read-only structural summary of c-native-pressure-observer-v1 receipts.

No quality, OOM, performance, causal-benefit or paper-GO verdict is produced.
The output is created exclusively; neither input nor an existing output changes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count_nonnegative(value, label):
    require(type(value) is int and value >= 0, label + ' must be a nonnegative integer')
    return value


def analyze(pressure, runtime):
    require(pressure['schema'] == 'c-native-pressure-observer-v1', 'unsupported observer schema')
    require(pressure['status'] in ('COMPLETE', 'INCOMPLETE'), 'unknown observer status')
    calls, attempts = pressure['scheduler_calls'], pressure['allocation_attempts']
    require(isinstance(calls, list) and isinstance(attempts, list), 'event lists required')
    require([c['call_id'] for c in calls] == list(range(len(calls))), 'call IDs differ')
    require([e['attempt_id'] for e in attempts] == list(range(len(attempts))), 'attempt IDs differ')
    for name in ('total_blocks', 'usable_blocks', 'block_size', 'kv_cache_memory_bytes',
                 'max_num_running_reqs', 'max_num_scheduled_tokens'):
        count_nonnegative(runtime[name], name)
    total, usable = runtime['total_blocks'], runtime['usable_blocks']
    require(total >= 1 and usable == total - 1, 'runtime null-block geometry differs')
    config = pressure['configuration']
    count_nonnegative(config['watermark_blocks'], 'watermark_blocks')
    require(config['scheduler_reserve_full_isl'] == runtime['scheduler_reserve_full_isl'],
            'full-current-sequence reservation receipts differ')
    free_samples, state_samples, people = [], [], {}
    linked = {c['call_id']: [] for c in calls}
    none_events, allocation_errors, retry_events, preemptions, waiting = [], [], [], [], []

    def person(request_id):
        require(isinstance(request_id, str) and bool(request_id), 'missing native request ID')
        return people.setdefault(request_id, dict(request_id=request_id,
            allocation_attempt_ids=[], allocate_none_attempt_ids=[], successful_attempt_ids=[],
            allocation_error_attempt_ids=[], retry_after_none_attempt_ids=[],
            status_before_counts=Counter(), preempted_call_ids=[], scheduled_call_ids=[],
            total_scheduled_tokens=0, observed_queue_heads=[]))

    def free(value, label):
        count_nonnegative(value, label)
        require(value <= usable, label + ' exceeds usable pool')
        free_samples.append(value)

    last_attempt = {}
    for event in attempts:
        aid, cid, rid = event['attempt_id'], event['call_id'], event['request_id']
        require(cid is None or (type(cid) is int and cid in linked), 'allocation call link differs')
        if cid is not None:
            linked[cid].append(aid)
        request = person(rid)
        request['allocation_attempt_ids'].append(aid)
        require(isinstance(event['status_before'], str), 'allocation status must be a string')
        request['status_before_counts'][event['status_before']] += 1
        for name in ('num_prompt_tokens', 'num_tokens', 'num_computed_tokens'):
            count_nonnegative(event[name], name)
        require(isinstance(event['arguments'], dict), 'allocation arguments missing')
        for name in ('free_blocks_before', 'free_blocks_after'):
            free(event[name], name)
        previous = last_attempt.get(rid)
        if previous is not None and previous['returned_none'] is True:
            request['retry_after_none_attempt_ids'].append(aid)
            retry_events.append(dict(request_id=rid, attempt_id=aid, call_id=cid,
                previous_none_attempt_id=previous['attempt_id'],
                status_before=event['status_before'], returned_none=event['returned_none']))
        last_attempt[rid] = event
        if event['returned_none'] is None:
            require('error' in event and event['succeeded'] is None, 'unfinished allocation lacks error')
            request['allocation_error_attempt_ids'].append(aid)
            allocation_errors.append(event)
        else:
            require(type(event['returned_none']) is bool and
                    event['succeeded'] is (not event['returned_none']), 'allocation outcome differs')
            if event['returned_none']:
                request['allocate_none_attempt_ids'].append(aid)
                none_events.append(event)
            else:
                request['successful_attempt_ids'].append(aid)

    labels = Counter()
    for call in calls:
        cid = call['call_id']
        require(call['attempt_ids'] == linked[cid], 'call-to-attempt inventory differs')
        expected_waiting_none = [i for i in linked[cid] if attempts[i]['returned_none'] is True
                                and attempts[i]['status_before'] in ('WAITING', 'PREEMPTED')]
        require(call['waiting_rejection_attempt_ids'] == expected_waiting_none,
                'waiting rejection inventory differs')
        require(isinstance(call['classifications'], list), 'exit classifications missing')
        labels.update(call['classifications'])
        for boundary in ('before', 'after'):
            state = call[boundary]
            require(state['total_blocks'] == total and state['usable_blocks'] == usable,
                    'snapshot pool geometry differs')
            require(state['seq_limit'] == runtime['max_num_running_reqs'] and
                    state['token_budget_initial'] == runtime['max_num_scheduled_tokens'],
                    'snapshot scheduler limits differ')
            free(state['free_blocks'], 'snapshot free_blocks')
            for name in ('running', 'waiting', 'skipped_waiting', 'effective_running'):
                count_nonnegative(state[name], name)
            state_samples.append(state)
            for queue in ('waiting', 'skipped_waiting'):
                head = state[queue + '_head']
                require((head is None) == (state[queue] == 0), 'queue/head inventory differs')
                if head is not None:
                    person(head['request_id'])['observed_queue_heads'].append(dict(
                        call_id=cid, boundary=boundary, queue=queue, status=head['status']))
        for rid, tokens in call.get('scheduled_tokens', {}).items():
            request = person(rid)
            request['scheduled_call_ids'].append(cid)
            request['total_scheduled_tokens'] += count_nonnegative(tokens, 'scheduled tokens')
        for rid in call.get('preempted_request_ids', []):
            person(rid)['preempted_call_ids'].append(cid)
            preemptions.append(dict(call_id=cid, request_id=rid))
        after = call['after']
        if after['waiting'] + after['skipped_waiting']:
            waiting.append(dict(call_id=cid, waiting=after['waiting'],
                skipped_waiting=after['skipped_waiting'], waiting_head=after['waiting_head'],
                skipped_waiting_head=after['skipped_waiting_head'],
                actual_waiting_none_attempt_ids=expected_waiting_none,
                actual_waiting_none_request_ids=[attempts[i]['request_id'] for i in expected_waiting_none],
                exit_boundary_classifications=call['classifications'],
                remaining_token_budget=call.get('remaining_token_budget'),
                seq_limit_reached_at_return=call['seq_limit_reached_at_return'],
                pause_state=after['pause_state'],
                per_request_waiting_causes='UNATTRIBUTED; heads/counts are not a full queue inventory'))
    for name, actual in (('schedule_calls', len(calls)), ('allocation_calls', len(attempts)),
                         ('allocation_none_returns', len(none_events))):
        require(pressure[name] == actual, 'observer aggregate differs: ' + name)
    for request in people.values():
        request['status_before_counts'] = dict(request['status_before_counts'])
    return dict(schema='c-native-pressure-structural-summary-v1',
        scope='Structural observations only; no quality, OOM, policy-benefit or paper-GO verdict.',
        source_observer_status=pressure['status'], source_observer_error=pressure.get('error'),
        observation_end_s=pressure.get('observation_end_s'),
        capacity=dict(resolved_runtime=runtime, observer_configuration=config,
            effective_pool_capacity_blocks=usable,
            capacity_definition='Physical pool total minus one null block; watermark reported separately.'),
        observations=dict(free_block_sample_count=len(free_samples),
            minimum_observed_free_blocks=min(free_samples) if free_samples else None,
            peak_observed_used_blocks=usable - min(free_samples) if free_samples else None,
            used_blocks_definition='usable_blocks - free_blocks at schedule/allocation boundaries',
            zero_free_block_observations=sum(v == 0 for v in free_samples),
            peak_running=max((s['running'] for s in state_samples), default=None),
            peak_waiting_and_skipped=max((s['waiting'] + s['skipped_waiting']
                                          for s in state_samples), default=None)),
        counts=dict(schedule_calls=len(calls), allocation_calls=len(attempts),
            allocation_calls_outside_schedule=sum(e['call_id'] is None for e in attempts),
            allocation_none_returns=len(none_events),
            allocation_none_unique_requests=len({e['request_id'] for e in none_events}),
            allocation_none_by_status=dict(Counter(e['status_before'] for e in none_events)),
            allocation_exceptions=len(allocation_errors),
            schedule_exceptions=sum('error' in c for c in calls),
            retry_after_none_attempts=len(retry_events), preemption_events=len(preemptions),
            unique_preempted_requests=len({e['request_id'] for e in preemptions}),
            observed_request_ids=len(people), calls_with_waiting_at_return=len(waiting),
            waiting_exit_calls_without_waiting_none=sum(not w['actual_waiting_none_attempt_ids']
                                                      for w in waiting),
            exit_boundary_classifications=dict(labels)),
        allocation_none_events=none_events, allocation_exception_events=allocation_errors,
        retry_events=retry_events, preemption_events=preemptions,
        per_request=[people[rid] for rid in sorted(people)], waiting_observations=waiting,
        limitations=[
            'Native request IDs remain exact; no external-ID normalization or inferred merging.',
            'Retry means the next observed allocation attempt for the same ID after a None return; '
            'ordinary later decode allocations are not all counted as retries.',
            'None is native allocation rejection, not CUDA OOM; full-current-sequence reservation '
            'and watermark checks can reject with positive free blocks.',
            'FIFO successors without allocation attempts are not counted as KV rejections. '
            'Exit budget/sequence labels do not assign causes to every waiting request.',
            'Waiting counts are repeated snapshots, not unique requests or waiting durations. '
            'Unseen queue members are absent from the observed-ID inventory.',
            'Snapshots miss internal transients and ownership/sharing; no required-block '
            'counterfactual or universal watermark-adjusted capacity is inferred.',
            'Observer COMPLETE is not generation, drain, EOS or task-quality qualification. '
            'Host observation time is instrumented scope, not a performance comparison.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pressure', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw_pressure, raw_runtime = args.pressure.read_bytes(), args.runtime.read_bytes()
    result = analyze(json.loads(raw_pressure), json.loads(raw_runtime))
    result['input_sha256'] = dict(measured_pressure=hashlib.sha256(raw_pressure).hexdigest(),
                                  resolved_runtime=hashlib.sha256(raw_runtime).hexdigest())
    result['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
