"""Describe observed preemption/recovery costs and tails; separate trajectories, no causal attribution."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def dist(values):
    xs = sorted(values)
    if not xs:
        return dict(n=0)
    def q(p):
        i = (len(xs)-1)*p
        lo = int(i)
        return xs[lo] + (xs[min(lo+1, len(xs)-1)]-xs[lo])*(i-lo)
    return dict(n=len(xs), min=xs[0], median=q(.5), p90=q(.9), p95=q(.95), max=xs[-1])


def load(session, arm):
    archive = next(session.glob(f'*_{arm}/archive'))
    return json.loads((archive/'raw.json').read_text()), json.loads((archive/'selective-store.json').read_text())


def arm_result(raw, store):
    reqs = {r['internal_request_id']: r for r in raw['requests']}
    events = store['events']
    commits = {(e['step'], e['victim']): e for e in events
               if e['event'] == 'commit_check' and e['reason'] == 'READY'}
    capacity = {(e['step'], e['new_victim']): e for e in events if e['event'] == 'capacity_victim_commit'}
    preemptions, delays, previous, repeat_outputs = [], [], {}, []
    for p in raw['preemption_events']:
        rid, step, entered = p['internal_request_id'], p['engine_call_index'], p['method_entered_s']
        times = reqs[rid]['token_times_s']
        idx = bisect_right(times, entered)
        next_output = times[idx] if idx < len(times) else None
        if next_output is not None:
            delays.append(next_output-entered)
        if rid in previous:
            repeat_outputs.append(p['last_returned_output_count']-previous[rid])
        previous[rid] = p['last_returned_output_count']
        c = commits.get((step, rid))
        preemptions.append(dict(request_id=p['request_id'], internal_request_id=rid, step=step,
            entered_s=entered, output_count=p['last_returned_output_count'],
            matched_forced_commit=c is not None, capacity_replacement=(step, rid) in capacity,
            recovery_target=reqs[c['target']]['request_id'] if c else None,
            first_later_output_s=next_output, to_first_later_output_s=next_output-entered if next_output else None))
    by_source = defaultdict(list)
    for e in raw['output_events']:
        if e['chunk_size']:
            by_source[e['request_id']].append(e)
    gaps = []
    for rid, r in reqs.items():
        times = r['token_times_s']
        for i in range(1, len(times)):
            gaps.append((times[i]-times[i-1], rid, times[i-1], times[i]))
    top = []
    for gap, rid, start, end in sorted(gaps, reverse=True)[:3]:
        own = [p for p in preemptions if p['internal_request_id'] == rid and start < p['entered_s'] < end]
        recovering = [p for p in preemptions if p['recovery_target'] == reqs[rid]['request_id']
                      and start < p['entered_s'] < end]
        outs = by_source[reqs[rid]['request_id']]
        start_call = next(e['engine_call_index'] for e in outs if e['received_s'] == start)
        end_call = next(e['engine_call_index'] for e in outs if e['received_s'] == end)
        chain = [dict(kind='last_output', time_s=start, engine_call_index=start_call)]
        chain += [dict(kind='request_preempted', **p) for p in own]
        chain += [dict(kind='forced_recovery_commit_for_request', **p) for p in recovering]
        chain.append(dict(kind='next_output', time_s=end, engine_call_index=end_call))
        chain.sort(key=lambda e: e.get('time_s', e.get('entered_s')))
        top.append(dict(request_id=reqs[rid]['request_id'], gap_s=gap,
            capacity_new_victim_in_this_gap=any(p['capacity_replacement'] for p in own), chain=chain,
            global_unattributed_prepare_rejections_in_step_interval=sum(e['event']=='prepare_rejected'
                and start_call < e['step'] <= end_call for e in events)))
    summary = dict(engine_calls=raw['engine_call_count'], total_preemptions=len(preemptions),
        matched_forced_preemptions=sum(p['matched_forced_commit'] for p in preemptions),
        unmatched_preemptions=sum(not p['matched_forced_commit'] for p in preemptions),
        prepare_count=sum(e['event']=='prepare' for e in events),
        prepare_rejected_reasons=dict(Counter(e['reason'] for e in events if e['event']=='prepare_rejected')),
        commit_check_reasons=dict(Counter(e['reason'] for e in events if e['event']=='commit_check')),
        distinct_preempted_requests=len({p['request_id'] for p in preemptions}),
        preemptions_without_later_output=sum(p['first_later_output_s'] is None for p in preemptions),
        preemption_to_first_later_output_s=dist(delays), consecutive_preemption_pairs=len(repeat_outputs),
        consecutive_preemptions_with_no_new_output=sum(x==0 for x in repeat_outputs),
        consecutive_preemptions_with_one_or_two_new_outputs=sum(1 <= x <= 2 for x in repeat_outputs),
        output_tokens_between_consecutive_preemptions=dist(repeat_outputs),
        generation_gap_counts={str(t):sum(gap >= t for gap, *_ in gaps) for t in (1.0, 2.0, 3.0)},
        three_largest_generation_gaps=top)
    return summary, preemptions


def analyze(session, pair_path):
    pair = json.loads(pair_path.read_text())
    summaries, raws, stores, preemptions = {}, {}, {}, {}
    for arm in ('off','on'):
        raws[arm], stores[arm] = load(session, arm)
        summaries[arm], preemptions[arm] = arm_result(raws[arm], stores[arm])
    actions = pair['capacity_victim']['actions']
    roles = {role: {a['request_outcomes'][role]['on']['request_id'] for a in actions}
             for role in ('target', 'new_victim', 'old_victim')}
    unique_new = {}
    occurrences = Counter(a['request_outcomes']['new_victim']['on']['request_id'] for a in actions)
    preempt_counts = {arm: Counter(p['request_id'] for p in preemptions[arm]) for arm in ('off','on')}
    for a in actions:
        outcome = a['request_outcomes']['new_victim']
        off, on = outcome['off'], outcome['on']
        rid = on['request_id']
        unique_new[rid] = dict(request_id=rid, capacity_replacement_count=occurrences[rid],
            preemptions_off=preempt_counts['off'][rid], preemptions_on=preempt_counts['on'][rid],
            flow_off_s=off['flow_s'], flow_on_s=on['flow_s'], flow_difference_s=on['flow_s']-off['flow_s'],
            max_gap_off_s=off['max_observed_generation_gap_s'], max_gap_on_s=on['max_observed_generation_gap_s'],
            max_gap_difference_s=on['max_observed_generation_gap_s']-off['max_observed_generation_gap_s'],
            output_tokens_off=off['output_tokens'], output_tokens_on=on['output_tokens'],
            output_sequence_differs=off['output_token_ids_sha256'] != on['output_token_ids_sha256'])
    victims = list(unique_new.values())
    comparisons = {}
    for name in ('flow','max_gap'):
        key = name+'_difference_s'
        comparisons[name] = dict(improved=sum(v[key]<0 for v in victims), worsened=sum(v[key]>0 for v in victims),
            equal=sum(v[key]==0 for v in victims), difference_s=dist([v[key] for v in victims]),
            three_largest_on_minus_off=sorted(victims, key=lambda v:v[key], reverse=True)[:3])
    return dict(session=str(session), evidence_level='OBSERVED_NATIVE_REQUEST_TRAJECTORIES', arms=summaries,
        capacity_committed_role_counts={role:len(ids) for role,ids in roles.items()},
        capacity_role_overlap=dict(new_victim_and_target=len(roles['new_victim'] & roles['target']),
            new_victim_and_old_victim=len(roles['new_victim'] & roles['old_victim'])),
        cancelled_choices=pair['capacity_victim']['cancelled_choices'],
        new_victim_request_comparison=dict(unique_requests=len(victims), action_occurrences=sum(occurrences.values()),
            paired_output_sequence_differences=sum(v['output_sequence_differs'] for v in victims),
            requests_with_more_preemptions_on=sum(v['preemptions_on']>v['preemptions_off'] for v in victims),
            comparisons=comparisons),
        scope=['Both arms use independent policy trajectories; role subsets are defined by on-arm actions.',
            'Preemption-to-next-output includes waiting, recovery and host execution; it is not recomputation or transfer time.',
            'These per-preemption waits can overlap and must not be summed as episode time.',
            'Sparse records do not expose recomputed tokens, transfer bytes or per-stage hardware costs.',
            'Prepare rejections lack actor IDs; counts inside gap windows are global, not attributed to that request.',
            'More frequent successful service rotations can coexist with shorter maximum waits; this pair alone does not establish causality or repeatability.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=Path, default=HERE/'moe-a-capacity-victim-session-r01-20261001')
    parser.add_argument('--pair', type=Path, default=HERE/'A_CAPACITY_VICTIM_PAIR_RESULT_R01_20261001.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session, args.pair)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('arms','cancelled_choices')}, indent=2))
