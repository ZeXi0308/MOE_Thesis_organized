#!/usr/bin/env python3
"""Small recovery replay for a saved native run; no model or policy simulation.

Use math_scale_analyze_v1.py for workload integrity and quality qualification.
This program exclusively creates --output and never modifies input artifacts.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from health_analyze_v2 import native_id_mapping


def add_interval(intervals, start, end):
    repeated = sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)
    merged = []
    for a, b in sorted(intervals + [(start, end)]):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged, repeated


def ranges(values):
    result = []
    for value in values:
        if result and value == result[-1][1] + 1:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def analyze(run):
    hashes = {}
    def read(name):
        data = (run / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    source = read('source-input.json')
    requests = source['requests'] if isinstance(source, dict) else source
    expected_ids = {'measured/' + row['request_id'] for row in requests}
    expected = len(requests)
    if not expected or len(expected_ids) != expected:
        raise ValueError('source request inventory is empty or contains duplicate IDs')
    mapping, errors = native_id_mapping(read('measured-native-sampling.json'), expected_ids)
    if errors:
        raise ValueError('native ID mapping failed: ' + '; '.join(errors[:10]))
    pressure = read('measured-pressure.json')
    calls = pressure['scheduler_calls']
    attempts = {a['attempt_id']: a for a in pressure['allocation_attempts']}
    if not calls or pressure['status'] != 'COMPLETE':
        raise ValueError('a complete nonempty pressure observation is required')
    first = calls[0]['before']
    if first['running'] or first['waiting'] + first['skipped_waiting'] != expected:
        raise ValueError('new-waiting counts require the entire source burst queued at first schedule')

    def external(rid):
        if rid not in mapping:
            raise ValueError('pressure ID absent from qualified native mapping: ' + str(rid))
        return mapping[rid]
    def head(value):
        return None if value is None else dict(request_id=external(value['request_id']), status=value['status'])

    admitted, pending, active_episode = set(), {}, {}
    episodes, intervals = [], defaultdict(list)
    repeated, preempt_counts, counts = Counter(), Counter(), Counter()
    pending_hist, new_hist, joint_hist = Counter(), Counter(), Counter()
    pending_calls, competition_calls, multi_calls = [], [], []
    head_fail_calls, competition_none_calls = set(), set()
    sample_states = []
    for c in calls:
        cid, scheduled = c['call_id'], c['scheduled_tokens']
        new_waiting, n_pending = expected - len(admitted), len(pending)
        competing = bool(n_pending and new_waiting)
        before_head = c['before']['waiting_head']
        if n_pending:
            pending_calls.append(cid)
            pending_hist[n_pending] += 1
            new_hist[new_waiting] += 1
            joint_hist[(n_pending, new_waiting)] += 1
            counts['pending_regular_waiting_head_is_preempted'] += bool(before_head and before_head['status'] == 'PREEMPTED')
            if len(sample_states) < 24:
                sample_states.append(dict(call_id=cid, pending_restore=n_pending,
                    never_admitted_new_waiting=new_waiting, waiting_head=head(before_head)))
        if competing:
            competition_calls.append(cid)
        if n_pending >= 2:
            multi_calls.append(cid)
        successes = {}
        for aid in c['attempt_ids']:
            a = attempts[aid]
            rid, status = a['request_id'], a['status_before']
            external(rid)
            if status == 'WAITING' and pending:
                counts['new_allocation_attempts_while_restore_pending'] += 1
            if status == 'PREEMPTED':
                if rid not in pending:
                    raise ValueError('restore attempt has no preceding observed preemption: ' + rid)
                episode = pending[rid]
                hit = a['arguments']['num_new_computed_tokens']
                if episode['restore_attempts'] == 0:
                    episode.update(first_restore_attempt_call=cid, first_restore_attempt_s=a['host_s'],
                                   first_attempt_cache_hit_tokens=hit)
                episode['restore_attempts'] += 1
                episode['restore_none_returns'] += a['returned_none'] is True
                episode['last_attempt_cache_hit_tokens'] = hit
                if a['returned_none']:
                    if competing:
                        competition_none_calls.add(cid)
                    if before_head and rid == before_head['request_id']:
                        head_fail_calls.add(cid)
            if a['succeeded'] and rid in scheduled:
                successes[rid] = a
                if status == 'WAITING':
                    counts['new_admissions_while_restore_pending'] += bool(pending)
                    admitted.add(rid)
                elif status == 'PREEMPTED':
                    episode = pending.pop(rid)
                    episode.update(resume_call=cid, resume_allocation_s=a['host_s'],
                        preempt_to_resume_allocation_s=a['host_s'] - episode['preempt_schedule_start_s'],
                        resume_cache_hit_tokens=a['arguments']['num_new_computed_tokens'],
                        cache_hit_loss_first_attempt_to_resume_tokens=episode['first_attempt_cache_hit_tokens'] - a['arguments']['num_new_computed_tokens'],
                        resume_scheduled_tokens=scheduled[rid],
                        prompt_tokens=a['num_prompt_tokens'], tokens_at_resume=a['num_tokens'],
                        new_waiting_at_resume=expected-len(admitted),
                        other_pending_after_resume=len(pending))
                    counts['successful_resumes'] += 1
        for rid, n in scheduled.items():
            external(rid)
            if rid not in successes:
                raise ValueError('scheduled request has no successful allocation in this call: ' + rid)
            a = successes[rid]
            args = a['arguments']
            start = a['num_computed_tokens'] + args['num_new_computed_tokens'] + args['num_external_computed_tokens']
            intervals[rid], overlap = add_interval(intervals[rid], start, start + n)
            repeated[rid] += overlap
            counts['total_emitted_scheduled_tokens'] += n
            if overlap:
                counts['calls_with_repeated_positions_request_pairs'] += 1
                if rid in active_episode:
                    active_episode[rid]['repeated_scheduled_token_positions'] += overlap
                else:
                    counts['repeated_positions_without_observed_preemption'] += overlap
        for rid in c['preempted_request_ids']:
            if rid in pending:
                raise ValueError('request preempted again before its observed resume: ' + rid)
            preempt_counts[rid] += 1
            episode = dict(request_id=external(rid), cycle=preempt_counts[rid], preempt_call=cid,
                preempt_schedule_start_s=c['start_s'], running_before=c['before']['running'],
                free_blocks_before=c['before']['free_blocks'],
                new_waiting_after_schedule=expected-len(admitted),
                waiting_head_before=head(before_head), waiting_head_after=head(c['after']['waiting_head']),
                restore_attempts=0, restore_none_returns=0, resume_call=None,
                repeated_scheduled_token_positions=0)
            episodes.append(episode)
            pending[rid] = active_episode[rid] = episode
        counts['maximum_pending_after_schedule'] = max(counts['maximum_pending_after_schedule'], len(pending))

    none_counts = Counter(a['status_before'] for a in attempts.values() if a['returned_none'])
    resumed = [e for e in episodes if e['resume_call'] is not None]
    waits = [e['preempt_to_resume_allocation_s'] for e in resumed]
    return dict(schema='math-recovery-evidence-v1', run=str(run), raw_sha256=hashes,
        scope='Observed scheduler replay only; workload integrity and quality are delegated to math_scale_analyze_v1.py.',
        expected_requests=expected, first_admitted_requests=len(admitted), scheduler_calls=len(calls),
        observation_status='COMPLETE', recovery_status='NO_PREEMPTION_OBSERVED' if not episodes else ('ALL_OBSERVED_PREEMPTIONS_RESUMED' if not pending else 'PENDING_AT_OBSERVATION_END'),
        preemptions=len(episodes), distinct_preempted_requests=len(preempt_counts),
        max_preemptions_per_request=max(preempt_counts.values(), default=0),
        requests_with_repeated_preemptions=sum(n > 1 for n in preempt_counts.values()),
        successful_resumes=counts['successful_resumes'], pending_at_end=[external(rid) for rid in pending],
        allocation_none_by_status={s:none_counts[s] for s in ('WAITING','PREEMPTED','RUNNING')},
        other_allocation_none_statuses={s:n for s,n in none_counts.items() if s not in ('WAITING','PREEMPTED','RUNNING')},
        repeated_scheduled_token_positions=sum(repeated.values()),
        repeated_positions_by_request={external(rid):n for rid,n in repeated.items() if n},
        total_emitted_scheduled_tokens=counts['total_emitted_scheduled_tokens'],
        repeated_positions_without_observed_preemption=counts['repeated_positions_without_observed_preemption'],
        recovery_wait_s=dict(count=len(waits), sum=sum(waits), maximum=max(waits, default=0)),
        cache_hit_loss_first_attempt_to_resume_tokens=sum(e['cache_hit_loss_first_attempt_to_resume_tokens'] for e in resumed),
        competition=dict(pending_restore_schedule_starts=len(pending_calls),
            restore_and_new_waiting_schedule_starts=len(competition_calls),
            at_least_two_pending_schedule_starts=len(multi_calls),
            maximum_pending=max(max(pending_hist,default=0), counts['maximum_pending_after_schedule']),
            pending_count_histogram=dict(pending_hist), new_waiting_count_histogram_while_pending=dict(new_hist),
            pending_and_new_count_histogram=[dict(pending=p,new_waiting=n,schedule_starts=k) for (p,n),k in sorted(joint_hist.items())],
            pending_regular_waiting_head_is_preempted=counts['pending_regular_waiting_head_is_preempted'],
            new_allocation_attempts_while_restore_pending=counts['new_allocation_attempts_while_restore_pending'],
            new_admissions_while_restore_pending=counts['new_admissions_while_restore_pending'],
            competition_calls_with_restore_none=len(competition_none_calls),
            pending_calls_with_starting_regular_head_none=len(head_fail_calls),
            pending_call_ranges=ranges(pending_calls), competition_call_ranges=ranges(competition_calls),
            at_least_two_pending_call_ranges=ranges(multi_calls), first_24_pending_states=sample_states),
        recovery_episodes=episodes, limitations=[
            'New-waiting counts require the checked initial full-source burst and no cancellation or replacement; full workload qualification is separate.',
            'Allocation None retries execute no model tokens. Repeated positions count only emitted successful schedules overlapping earlier schedules of the same request, not device FLOPs or isolated GPU time.',
            'Shared cache positions never previously executed by this request do not count as repeated request execution.',
            'Pending counts come from observed preemption and successful emitted resume transitions; snapshots expose regular queue heads, not full alternative feasibility.',
            'Cache-hit loss is first attempted restore minus successful restore; a negative value means an observed gain. It is not a causal benefit estimate for a different policy.',
            'Zero preemptions is a valid observed outcome, with zero recovery/recomputation evidence; it does not invalidate the workload.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must not already exist')
    report = analyze(args.run.resolve())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:report[k] for k in ('expected_requests','recovery_status','preemptions',
        'successful_resumes','repeated_scheduled_token_positions')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
