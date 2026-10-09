"""Diagnose recovery progress and longest gaps using existing terminal traces.

Usage: python -B cpu_preparation/analyze_recovery_progress.py GROUP... --out JSON
No service comparison, new logging, runtime import, or performance upper bound.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path


def brief(e):
    return {k:e.get(k) for k in ('event','decision_s','allocation_s','known_tokens',
        'generated_tokens','host_hit_tokens','eligible','action','actual_action','fallback')}


def capacity_release_context(raw):
    """Observed completion/resource overlap, never a counterfactual bound.

    steps.waiting omits native skipped_waiting. Count all submitted unfinished
    requests and subtract native running instead; retain both quantities.
    This boundary count says nothing about which resource blocked a request,
    nor whether every RUNNING request received GPU work in that iteration.
    """
    requests = {r['request_id']: r for r in raw['requests']}
    if not all(r.get('completed') for r in requests.values()):
        return dict(status='UNSUPPORTED_INCOMPLETE_REQUESTS')
    steps, schedules = raw['steps'], raw['scheduler_steps']
    if len(steps) != len(schedules):
        return dict(status='UNSUPPORTED_STEP_ALIGNMENT')
    ends = [s['end_s'] for s in steps]
    admissions = sorted(r['admitted_s'] for r in requests.values())
    completions = sorted(r['completion_s'] for r in requests.values())
    assert all(r['completion_s'] in ends for r in requests.values())
    outstanding = [bisect_right(admissions, t)-bisect_right(completions, t) for t in ends]
    nonrunning = [n-s['running'] for n, s in zip(outstanding, steps)]
    # A negative value or a waiting count larger than this residual would
    # invalidate the accounting used here, rather than imply spare capacity.
    assert all(n >= s['waiting'] >= 0 for n, s in zip(nonrunning, steps))
    last_nonrunning = max((i for i, n in enumerate(nonrunning) if n), default=None)
    capacity_failures = [e for e in raw['decisions']
                         if e.get('fallback') == 'full_capacity_not_jointly_available_native']
    last_capacity = max((e['decision_s'] for e in capacity_failures), default=None)
    eligible = [e for e in raw['commits'] if e.get('eligible')]
    rows = []
    for rid in sorted({e['request_id'] for e in eligible}):
        r = requests[rid]
        j = bisect_left(ends, r['completion_s'])
        rows.append(dict(external_id=r['external_id'], completion_step=j,
            completion_s=r['completion_s'], jointly_eligible_commits=sum(e['request_id']==rid for e in eligible),
            submitted_unfinished_at_step_end=outstanding[j], running_at_step_end=steps[j]['running'],
            nonrunning_submitted_at_step_end=nonrunning[j], ordinary_waiting_at_step_end=steps[j]['waiting'],
            nonrunning_submitted_after_completion_max=max(nonrunning[j:], default=0),
            external_arrivals_after_completion=sum(q['arrival_s']>r['completion_s'] for q in requests.values()),
            free_blocks_after_schedule=schedules[j]['free_blocks_after_schedule'],
            free_blocks_after_output=steps[j]['free_blocks'],
            completion_step_free_count_increase=steps[j]['free_blocks']-schedules[j]['free_blocks_after_schedule'],
            completion_steps_after_last_nonrunning=j-last_nonrunning if last_nonrunning is not None else None,
            last_nonrunning_to_completion_s=r['completion_s']-ends[last_nonrunning] if last_nonrunning is not None else None))
    return dict(status='OBSERVED_COMPLETION_CAPACITY_CONTEXT',
        sampled_steps=len(steps), all_requests_completed=True,
        jointly_eligible_commits=len(eligible), jointly_eligible_requests=len(rows),
        last_sample_with_nonrunning_submitted_step=last_nonrunning,
        last_sample_with_nonrunning_submitted_s=ends[last_nonrunning] if last_nonrunning is not None else None,
        last_capacity_failed_lookup_step=bisect_left(ends, last_capacity) if last_capacity is not None else None,
        last_capacity_failed_lookup_s=last_capacity,
        jointly_eligible_requests_completing_with_nonrunning_submitted=sum(r['nonrunning_submitted_at_step_end']>0 for r in rows),
        by_request=rows,
        limitations=[
            'Post-engine-step boundary accounting; not a continuous-time queue or pure capacity-wait measurement.',
            'RUNNING membership does not guarantee execution in this step; token budget and GPU work still matter.',
            'Free blocks are allocatable counts; native reuse can still require flushing a STORE dependency.',
            'The schedule-to-output free-count change is net across all requests in that step, not a per-request allocation trace.',
            'Actual completions and future arrivals are retrospective diagnostics, never online selector inputs or a bound for different trajectories.',
        ])


def extract(path, capacity_only=False):
    raw = json.loads(path.read_text())
    if capacity_only:
        return dict(cell=str(path.parent), capacity_release=capacity_release_context(raw))
    requests = {r['request_id']:r for r in raw['requests']}
    commits, preempts, decisions = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in raw['commits']:
        commits[e['request_id']].append(e)
    for e in raw['preemptions']:
        preempts[e['request_id']].append(e)
    for e in raw['decisions']:
        decisions[e['request_id']].append(e)
    gaps = {}
    for rid,r in requests.items():
        times = r['token_times_s']
        index = max(range(1,len(times)), key=lambda i:times[i]-times[i-1])
        gaps[rid] = dict(output_index=index, start_s=times[index-1], end_s=times[index],
                         gap_s=times[index]-times[index-1])
    chains = []
    re_preempt, adjacent_before_output, adjacent_with_preempt = [], [], []
    for rid,events in commits.items():
        r = requests[rid]
        grouped = defaultdict(list)
        for i,e in enumerate(events):
            output_index = bisect_right(r['token_times_s'],e['decision_s'])
            next_output = r['token_times_s'][output_index]
            grouped[output_index].append(e)
            between = [p for p in preempts[rid] if e['allocation_s'] < p['time_s'] < next_output]
            if between:
                re_preempt.append(dict(external_id=r['external_id'],event=e['event']))
            if i+1 < len(events) and events[i+1]['allocation_s'] < next_output:
                adjacent_before_output.append(dict(external_id=r['external_id'],event=e['event']))
                if any(p['time_s'] < events[i+1]['allocation_s'] for p in between):
                    adjacent_with_preempt.append(dict(external_id=r['external_id'],event=e['event']))
        for index,sequence in grouped.items():
            first = sequence[0]
            end = r['token_times_s'][index]
            preceding = [p for p in preempts[rid] if p['time_s'] <= first['decision_s']]
            last_preempt = preceding[-1] if preceding else None
            gap = gaps[rid]
            chains.append(dict(external_id=r['external_id'], output_index=index,
                committed_recoveries=len(sequence), sequence=[brief(e) for e in sequence],
                chain_start_s=first['decision_s'], next_output_s=end,
                choice_to_first_new_output_elapsed_s=end-first['decision_s'],
                preemptions_during_chain=sum(first['allocation_s'] < p['time_s'] < end for p in preempts[rid]),
                preceding_preemption_s=last_preempt['time_s'] if last_preempt else None,
                preceding_preemption_to_first_choice_s=first['decision_s']-last_preempt['time_s']
                    if last_preempt else None,
                is_request_longest_gap_endpoint=index==gap['output_index'],
                request_longest_gap=gap,
                chain_fraction_of_request_longest_gap=(end-first['decision_s'])/gap['gap_s']
                    if index==gap['output_index'] else None))
    longest = []
    for rid in commits:
        gap = gaps[rid]
        inside = [c for c in commits[rid] if gap['start_s'] <= c['decision_s'] < gap['end_s']]
        attempts = [e for e in decisions[rid] if gap['start_s'] <= e['decision_s'] < gap['end_s']]
        ps = [p for p in preempts[rid] if gap['start_s'] <= p['time_s'] < gap['end_s']]
        first_choice = inside[0]['decision_s'] if inside else None
        longest.append(dict(external_id=requests[rid]['external_id'], **gap,
            committed_recoveries=[brief(c) for c in inside],
            recovery_lookup_fallbacks=dict(Counter(e['fallback'] for e in attempts)),
            preemptions_in_gap=len(ps), first_preemption_s=ps[0]['time_s'] if ps else None,
            gap_start_to_first_preemption_s=ps[0]['time_s']-gap['start_s'] if ps else None,
            first_preemption_to_first_committed_choice_s=first_choice-ps[0]['time_s']
                if ps and first_choice is not None else None,
            first_committed_choice_to_gap_end_s=gap['end_s']-first_choice
                if first_choice is not None else None))
    longest.sort(key=lambda g:g['gap_s'],reverse=True)
    maxgap_actions = Counter()
    for g in longest:
        key = g['committed_recoveries'][-1]['fallback'] if g['committed_recoveries'] else 'no_commit_inside_gap'
        maxgap_actions[key] += 1
    chains.sort(key=lambda c:c['choice_to_first_new_output_elapsed_s'],reverse=True)
    return dict(cell=str(path.parent), committed_recoveries=len(raw['commits']),
        recovery_requests=len(commits),
        commits_repreempted_before_next_output=len(re_preempt),
        commits_repreempted_before_next_output_by_request=dict(Counter(e['external_id'] for e in re_preempt)),
        adjacent_commit_pairs_before_next_output=len(adjacent_before_output),
        adjacent_commit_pairs_with_repreemption_before_next_output=len(adjacent_with_preempt),
        longest_choice_to_output_chains=chains[:3],
        multiple_commit_no_output_chains=[c for c in chains if c['committed_recoveries']>1],
        recovery_requests_longest_gap_final_commit_fallback=dict(maxgap_actions),
        longest_recovery_request_gaps=longest[:3],
        recovery_requests_longest_gap_sum_s=sum(g['gap_s'] for g in longest),
        same_gaps_committed_choice_tail_sum_s=sum(g['first_committed_choice_to_gap_end_s'] or 0 for g in longest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups',type=Path,nargs='+')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--capacity-only',action='store_true',
                        help='Diagnose completion/resource overlap without repeating prior gap analysis')
    args = parser.parse_args()
    cells = []
    for group in args.groups:
        assert json.loads((group/'status.json').read_text())['status']=='COMPLETE'
        cells.extend(extract(path, args.capacity_only) for path in sorted(group.rglob('raw.json')))
    result = dict(status='CPU_CAPACITY_RELEASE_CONTEXT' if args.capacity_only else 'CPU_RECOVERY_PROGRESS_DIAGNOSIS', definitions=dict(
        successful_commit='Observed native allocation callback with actual_action; not a claim transfer or full recovery has already completed.',
        no_output_chain='Consecutive same-request committed recoveries whose first strictly later output is the same token index; interval runs once from first decision to that output.',
        repreemption='Native preemption timestamp strictly after allocation and strictly before the matched next output.',
        adjacent_pair='The next same-request committed recovery occurs before the earlier commit\'s next output; preemption presence counted separately.',
        longest_gap='Largest observed consecutive-token timestamp gap for a recovered request; earliest gap selected in a tie.',
        waiting='Time from first preemption in that gap to first committed choice is observed recovery waiting, not a pure capacity-wait duration.',
        sums='Sums across requests may overlap in wall time; reported only to describe coverage, never as throughput benefit or a strict benefit upper bound.',
        causal_scope='Within-run progress diagnosis; different runs are not matched-state interventions.'),cells=cells)
    if args.capacity_only:
        result['definitions'] = dict(
            scope='Completed synchronous episodes with every submitted request and final output accounted for; not a general asynchronous accounting rule.',
            nonrunning_submitted='At step end: admitted requests minus completed requests minus native running; includes skipped_waiting absent from steps.waiting.',
            sampling='scheduler_steps.free_blocks_after_schedule precedes output processing; steps.running/waiting/free_blocks follow engine.step return.',
            action_scope='Every request with at least one jointly eligible committed recovery; no target selected by favorable completion outcome.',
            interpretation='Overlap of observed completion with observed nonrunning work, not the effect of a hypothetical earlier completion or an online signal.')
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if args.capacity_only:
        print(json.dumps(dict(output=str(args.out), cells=len(cells),
            eligible_request_completions=sum(c['capacity_release'].get('jointly_eligible_requests',0) for c in cells),
            completions_with_nonrunning_submitted=sum(c['capacity_release'].get('jointly_eligible_requests_completing_with_nonrunning_submitted',0) for c in cells)),indent=2))
        return
    print(json.dumps(dict(output=str(args.out),cells=[{k:c[k] for k in (
        'cell','committed_recoveries','recovery_requests','commits_repreempted_before_next_output',
        'adjacent_commit_pairs_before_next_output','recovery_requests_longest_gap_final_commit_fallback',
        'recovery_requests_longest_gap_sum_s','same_gaps_committed_choice_tail_sum_s')} for c in cells]),indent=2))


if __name__ == '__main__':
    main()
