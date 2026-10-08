"""Check saved MC cohorts against observed output calls, without GPU replay.

This does not observe physical block frees or estimate policy effects. The two
complete first-eligible snapshots provide the directly recorded old cohorts;
all-request output continuity is a separate, broader screening observation.
"""
import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def continuity(events):
    holes = []
    for previous, current in zip(events, events[1:]):
        missing = current['engine_call_index'] - previous['engine_call_index'] - 1
        if missing:
            holes.append(dict(after_call=previous['engine_call_index'],
                before_call=current['engine_call_index'], missing_calls=missing,
                host_gap_s=current['received_s']-previous['received_s']))
    unusual = [e['engine_call_index'] for e in events
        if e['chunk_size'] != 1 and not
        (e['finished'] and e['finish_reason'] == 'stop' and e['chunk_size'] == 0)]
    return dict(missing_calls=sum(e['missing_calls'] for e in holes),
        holes=holes, unexpected_chunk_calls=unusual)


def inspect_arm(path):
    gate, gate_sha = read(path/'admission.json')
    raw, raw_sha = read(path/'raw.json')
    assert raw['status'] == 'COMPLETE' and raw['error'] is None
    assert raw['engine_call_count'] == raw['engine_return_count'] == gate['calls']
    by_id = defaultdict(list)
    times_by_call = {}
    for event in raw['output_events']:
        assert event['prefix_valid']
        by_id[event['request_id']].append(event)
        index, now = event['engine_call_index'], event['received_s']
        assert index not in times_by_call or times_by_call[index] == now
        times_by_call[index] = now
    for events in by_id.values():
        assert all(a['engine_call_index'] < b['engine_call_index']
                   for a, b in zip(events, events[1:]))
        assert events[-1]['finished'] and not any(e['finished'] for e in events[:-1])
    requests = {r['request_id']: r for r in raw['requests']}
    assert requests.keys() == by_id.keys()
    recorded = gate['mc_budget']['first_eligible']
    decision = gate['decisions'][recorded['decision_index']]
    assert decision['request_id'] == recorded['request_id']
    assert decision['t'] == recorded['t'] and decision['mc_eligible']
    assert decision['changed_by_mc_budget'] and decision['native_allocation_result']
    offset = gate['origin_perf_s']-raw['measurement_origin_perf_counter_s']
    trigger = recorded['t']+offset
    ordered = sorted(times_by_call.items())
    position = bisect_left([value for _, value in ordered], trigger)
    assert 0 < position < len(ordered)
    previous_call, previous_time = ordered[position-1]
    current_call, current_time = ordered[position]
    # Adjacent indices bracket this decision in a synchronous engine.step call.
    # No index is inferred across an unobserved empty-output call.
    assert current_call == previous_call+1 and previous_time < trigger < current_time
    start = next(s for s in gate['starts'] if s['request_id'] == recorded['request_id'])
    prefill = start['first_prefill_perf_s']-raw['measurement_origin_perf_counter_s']
    assert trigger <= prefill <= current_time
    cohort = []
    for old in recorded['old_rows']:
        rid = raw['internal_to_source'][old['request_id']]
        request, events = requests[rid], by_id[rid]
        prior = [e for e in events if e['engine_call_index'] < current_call]
        later = [e for e in events if e['engine_call_index'] >= current_call]
        assert prior and later and not prior[-1]['finished']
        observed_before = prior[-1]['cumulative_tokens']
        assert old['n'] == request['prompt_tokens']+observed_before
        assert old['r'] == request['max_output_tokens']-observed_before
        assert old['allocated_blocks'] == (old['n']+15)//16
        expected_finish = current_call+old['r']-1
        finished = later[-1]
        progress = continuity([prior[-1], *later])
        cohort.append(dict(request_id=rid, old_n=old['n'], remaining_bound=old['r'],
            observed_output_before=observed_before,
            prior_output_call=prior[-1]['engine_call_index'],
            expected_finish_call_at_latest=expected_finish,
            actual_finish_call=finished['engine_call_index'],
            finish_call_lateness=finished['engine_call_index']-expected_finish,
            actual_remaining_calls=finished['engine_call_index']-current_call+1,
            completion_after_trigger_s=finished['received_s']-trigger,
            stop_reason=finished['finish_reason'],
            missing_calls=progress['missing_calls'],
            unexpected_chunk_calls=progress['unexpected_chunk_calls'], holes=progress['holes']))
    first_output_index = {rid: next(e['engine_call_index'] for e in events if e['chunk_size'])
                          for rid, events in by_id.items()}
    all_continuity = {rid: continuity([e for e in by_id[rid]
                         if e['engine_call_index'] >= first_output_index[rid]])
                      for rid in by_id}
    violations = [row for row in cohort if row['finish_call_lateness'] > 0
                  or row['missing_calls'] or row['unexpected_chunk_calls']]
    return dict(cell=str(path), admission_sha256=gate_sha, raw_sha256=raw_sha,
        mc_successful_relaxations=gate['mc_budget']['successful_relaxations'],
        directly_recorded_complete_cohorts=1,
        trigger=dict(request_id=recorded['request_id'], external_s=trigger,
            previous_output_call=previous_call, previous_output_return_s=previous_time,
            current_output_call=current_call, current_output_return_s=current_time,
            first_prefill_schedule_return_s=prefill,
            recorded_remaining_bound_min=min(r['remaining_bound'] for r in cohort),
            recorded_remaining_bound_max=max(r['remaining_bound'] for r in cohort)),
        cohort_summary=dict(requests=len(cohort),
            completed_by_declared_bound=sum(r['finish_call_lateness'] <= 0 for r in cohort),
            completed_at_bound=sum(r['finish_call_lateness'] == 0 for r in cohort),
            completed_before_bound=sum(r['finish_call_lateness'] < 0 for r in cohort),
            requests_with_missing_calls=sum(bool(r['missing_calls']) for r in cohort),
            total_missing_calls=sum(r['missing_calls'] for r in cohort),
            requests_with_unexpected_chunks=sum(bool(r['unexpected_chunk_calls']) for r in cohort),
            observed_progress_violations=len(violations),
            maximum_finish_call_lateness=max(r['finish_call_lateness'] for r in cohort),
            max_completion_after_trigger_s=max(r['completion_after_trigger_s'] for r in cohort)),
        all_request_decode_continuity=dict(requests=len(all_continuity),
            requests_with_missing_calls=sum(bool(v['missing_calls']) for v in all_continuity.values()),
            total_missing_calls=sum(v['missing_calls'] for v in all_continuity.values()),
            requests_with_unexpected_chunks=sum(bool(v['unexpected_chunk_calls']) for v in all_continuity.values()),
            exceptions={k:v for k,v in all_continuity.items()
                        if v['missing_calls'] or v['unexpected_chunk_calls']}),
        cohort=cohort)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must not exist; preserved analyses are never overwritten.')
    result = dict(schema_version=1, new_gpu_runs=0, new_online_actions=0,
        question='Did the two recorded first-MC cohorts lose output progress used for future release?',
        inference='Observed native host-output call conformance, not policy benefit or physical free timing.',
        round_semantics='Trigger call k0 is h=0. A remaining declared output bound r must finish by '
            'return k0+r-1; its release is credited starting next schedule h=r.',
        limitations=[
            'Only the first eligible full old-row snapshot per MC arm is directly recorded.',
            'All-request post-first-output continuity is a broader screen, not reconstruction of every physical MC state.',
            'Output call indices are synchronous host evidence, not GPU timestamps or direct block-free events.',
            'Completion and observed progress do not guarantee wall-clock latency, task quality or global SLOs.',
            'No counterfactual strategy, per-request independent inference or subtraction of overhead.'],
        arms=[inspect_arm(args.group/name) for name in ('probe-01-mcbudget', 'probe-02-mcbudget')])
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps([{k:arm[k] for k in ('cell','cohort_summary','all_request_decode_continuity')}
                      for arm in result['arms']], indent=2))


if __name__ == '__main__':
    main()
