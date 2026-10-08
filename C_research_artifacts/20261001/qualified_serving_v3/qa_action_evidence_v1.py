"""Read-only replay of four saved QA cells; writes only its sibling JSON.

No policy simulation: allocation retries are not recomputation, and rejected
queue successors have no measured counterfactual feasibility.
"""
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
CELLS = {
    'native128_first': 'docqa128_raw_v1_20261002/qwen7b-native-docqa128-v1/native',
    'fixed64_first': 'docqa_concurrency64_raw_v1_20261002/qwen7b-docqa-concurrency64-v1/native',
    'native128_confirmation': 'docqa_concurrency_confirmation_raw_v1_20261002/qwen7b-native-docqa128-confirm-v1/native',
    'fixed64_confirmation': 'docqa_concurrency_confirmation_raw_v1_20261002/qwen7b-docqa-concurrency64-confirm-v1/native',
}


def load(path):
    return json.loads(path.read_text())


def short_id(request_id):
    return request_id.removeprefix('measured/').rsplit('-', 1)[0]


def stats(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    def percentile(p):
        x = (len(values) - 1) * p
        a = int(x)
        return values[a] + (values[min(a + 1, len(values) - 1)] - values[a]) * (x - a)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=percentile(.95), minimum=values[0], maximum=values[-1])


def add_interval(intervals, start, end):
    repeated = sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)
    merged = []
    for a, b in sorted(intervals + [(start, end)]):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged, repeated


def analyze_cell(relative):
    path = ROOT / relative
    pressure, outputs, steps, runtime = [load(path / f) for f in
        ('measured-pressure.json', 'measured-outputs.json', 'measured-steps.json', 'runtime.json')]
    calls, attempts = pressure['scheduler_calls'], pressure['allocation_attempts']
    assert pressure['status'] == 'COMPLETE' and len(outputs) == 128
    assert all(r['finished'] and r['arrival_s'] == 0 for r in outputs)
    assert len(calls) == len(steps['steps'])
    rows = {r['request_id']: r for r in outputs}
    by_request = collections.defaultdict(list)
    for a in attempts:
        by_request[a['request_id']].append(a)
    pending, admitted = set(), set()
    count, concurrency, head_failures = collections.Counter(), collections.Counter(), set()
    actual_head_failures, multiple_head_failures = set(), set()
    intervals, repeated = collections.defaultdict(list), collections.Counter()
    competition_calls, multiple_calls, preemptions = [], [], []
    for c in calls:
        new_waiting = len(outputs) - len(admitted)
        competing = bool(pending and new_waiting)
        if pending:
            count['pending_restore_schedule_starts'] += 1
            count['pending_head_is_preempted'] += c['before']['waiting_head']['status'] == 'PREEMPTED'
        if competing:
            competition_calls.append(c['call_id'])
            concurrency[len(pending)] += 1
            if len(pending) >= 2:
                multiple_calls.append(c['call_id'])
        successes = {}
        for aid in c['attempt_ids']:
            a = attempts[aid]
            rid = a['request_id']
            if a['status_before'] == 'WAITING' and pending:
                count['new_attempt_while_restore_pending'] += 1
            if a['status_before'] == 'PREEMPTED' and a['returned_none']:
                if competing:
                    head_failures.add(c['call_id'])
                    if rid == c['before']['waiting_head']['request_id']:
                        actual_head_failures.add(c['call_id'])
                        if c['call_id'] in multiple_calls:
                            multiple_head_failures.add(c['call_id'])
            if a['succeeded'] and rid in c['scheduled_tokens']:
                successes[rid] = a
                if a['status_before'] == 'PREEMPTED':
                    assert rid in pending
                    pending.remove(rid)
                    count['successful_resumes'] += 1
                elif a['status_before'] == 'WAITING':
                    count['new_admit_while_restore_pending'] += bool(pending)
                    admitted.add(rid)
        for rid, n in c['scheduled_tokens'].items():
            a = successes[rid]
            args = a['arguments']
            start = a['num_computed_tokens'] + args['num_new_computed_tokens'] + args['num_external_computed_tokens']
            intervals[rid], overlap = add_interval(intervals[rid], start, start + n)
            repeated[rid] += overlap
        for rid in c['preempted_request_ids']:
            preemptions.append((c, rid))
        pending.update(c['preempted_request_ids'])
    assert not pending and len(admitted) == 128
    events = []
    for c, rid in preemptions:
        trials = [a for a in by_request[rid] if a['status_before'] == 'PREEMPTED' and a['call_id'] > c['call_id']]
        resumed = next(a for a in trials if a['succeeded'])
        trials = [a for a in trials if a['attempt_id'] <= resumed['attempt_id']]
        row = rows[short_id(rid)]
        ts = sorted(set(h['return_s'] for h in row['host_returns'] if h['delta_token_ids']))
        events.append(dict(request_id=row['request_id'], preempt_call=c['call_id'],
            resume_call=resumed['call_id'], preempt_to_resume_allocation_s=resumed['host_s'] - c['start_s'],
            running_before=c['before']['running'], waiting_before=c['before']['waiting'],
            free_blocks_before=c['before']['free_blocks'],
            waiting_head_before=c['before']['waiting_head'], waiting_head_after=c['after']['waiting_head'],
            restore_attempts=len(trials), restore_none_returns=sum(a['returned_none'] for a in trials),
            prompt_tokens=resumed['num_prompt_tokens'], tokens_at_resume=resumed['num_tokens'],
            first_attempt_cache_hit_tokens=trials[0]['arguments']['num_new_computed_tokens'],
            resume_cache_hit_tokens=resumed['arguments']['num_new_computed_tokens'],
            resume_scheduled_tokens=calls[resumed['call_id']]['scheduled_tokens'][rid],
            repeated_scheduled_token_positions=repeated[rid],
            completed_flow_s=row['host_elapsed_s'] - row['arrival_s'],
            maximum_host_return_gap_s=max((b-a for a,b in zip(ts,ts[1:])), default=0)))
    pcounts = collections.Counter(r for c, r in preemptions)
    return dict(raw_directory=relative, raw_sha256={f: hashlib.sha256((path/f).read_bytes()).hexdigest()
                    for f in ('measured-pressure.json', 'measured-outputs.json', 'measured-steps.json', 'runtime.json')},
        runtime=dict(vllm=runtime['vllm'], scheduler_source_sha256=runtime['vllm_source_sha256']['v1/core/sched/scheduler.py']),
        completed=len(outputs), natural_eos=sum(r['finish_reason']=='stop' for r in outputs),
        output_tokens=sum(len(r['output_token_ids']) for r in outputs),
        episode_last_step_return_s=steps['steps'][-1]['return_s'],
        completed_request_flow_s=stats([r['host_elapsed_s']-r['arrival_s'] for r in outputs]),
        scheduler_calls=len(calls), gate_counts=dict(collections.Counter(x for c in calls for x in c['classifications'])),
        allocation_none_by_status=dict(collections.Counter(a['status_before'] for a in attempts if a['returned_none'])),
        preemptions=len(preemptions), distinct_preempted_requests=len(pcounts),
        max_preemptions_per_request=max(pcounts.values(), default=0),
        repeated_preempt_restore_cycles_observed=sum(v>1 for v in pcounts.values()),
        repeated_scheduled_token_positions=sum(repeated.values()),
        total_scheduled_tokens=sum(sum(c['scheduled_tokens'].values()) for c in calls),
        recovery_competition=dict(**{k:count[k] for k in ('pending_restore_schedule_starts',
            'pending_head_is_preempted','successful_resumes','new_attempt_while_restore_pending',
            'new_admit_while_restore_pending')},
            restore_and_never_admitted_new_schedule_starts=len(competition_calls),
            concurrent_preempted_histogram=dict(concurrency),
            maximum_concurrent_preempted=max(concurrency, default=0),
            at_least_two_preempted_schedule_starts=len(multiple_calls),
            at_least_two_preempted_call_ids=multiple_calls,
            competition_calls_with_restore_allocation_none=len(head_failures),
            competition_calls_with_starting_head_allocation_none=len(actual_head_failures),
            multiple_preempted_calls_with_starting_head_allocation_none=len(multiple_head_failures),
            competition_call_ids=competition_calls), preempted_requests=events), rows


def comparison(native, fixed, native_summary, fixed_summary):
    all_rows = []
    for rid, a in native.items():
        b = fixed[rid]
        all_rows.append(dict(request_id=rid, fixed_minus_native_flow_s=b['host_elapsed_s']-a['host_elapsed_s'],
            same_output_tokens=a['output_token_ids']==b['output_token_ids']))
    def group(values):
        delta = [r['fixed_minus_native_flow_s'] for r in values]
        return dict(delta_flow_s=stats(delta), fixed_faster=sum(x<0 for x in delta),
            fixed_slower=sum(x>0 for x in delta), fixed_faster_by_over_1s=sum(x < -1 for x in delta),
            fixed_slower_by_over_1s=sum(x>1 for x in delta))
    preempted = {r['request_id'] for r in native_summary['preempted_requests']}
    return dict(all_128=group(all_rows), exact_output_subset=group([r for r in all_rows if r['same_output_tokens']]),
        changed_output_subset=group([r for r in all_rows if not r['same_output_tokens']]),
        fixed_minus_native_episode_s=fixed_summary['episode_last_step_return_s']-native_summary['episode_last_step_return_s'],
        preempted_request_deltas=[r for r in all_rows if r['request_id'] in preempted],
        largest_improvements=sorted(all_rows,key=lambda r:r['fixed_minus_native_flow_s'])[:3],
        largest_regressions=sorted(all_rows,key=lambda r:r['fixed_minus_native_flow_s'],reverse=True)[:3])


def main():
    summaries, outputs = {}, {}
    for name, path in CELLS.items():
        summaries[name], outputs[name] = analyze_cell(path)
    pairs = {}
    for suffix in ('first', 'confirmation'):
        a,b = 'native128_'+suffix,'fixed64_'+suffix
        pairs[suffix] = comparison(outputs[a],outputs[b],summaries[a],summaries[b])
    report = dict(schema='qa-action-evidence-v1', scope='Four existing QA runs only; no GPU, SSH, counterfactual execution, or mathematical-workload extrapolation.',
        runs=summaries, paired_descriptive_comparisons=pairs,
        interpretation=[
            'Native already prioritizes PREEMPTED ahead of never-admitted requests in every observed recovery competition; no new-request bypass was observed.',
            'Repeated allocation None retries do not execute recomputation. Repeated token positions count only emitted scheduled tokens overlapping earlier scheduled positions of the same request.',
            'Each native victim is preempted and resumed once. Both fixed64 runs remove all observed preemption and repeated-token-position costs; these costs are already covered by simple throttling.',
            'Cache-hit decline during rejected recovery attempts is observed; it motivates measuring cache survival cost but does not establish a new recovery-policy benefit.',
            'Full queue contents and alternative required-block checks were not logged. Head allocation None blocks unattempted successors, but no successor feasibility or beneficial reordering is established.',
            'All four cells complete 128 requests. Cross-arm latency differences are descriptive: 42 outputs change, and admission, batching, cache behavior, and generation can all contribute.',
            'Host return gaps and preemption-to-allocation intervals are wall-clock observations, not device ITL or isolated recomputation runtime.',
        ])
    target = ROOT/'qa_action_evidence_v1.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(target),'bytes':target.stat().st_size,
        'native_competition':summaries['native128_first']['recovery_competition'],
        'pairs':{k:v['all_128'] for k,v in pairs.items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()
