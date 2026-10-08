"""Observed preemption/pause alignment from three raw arrays; no causal replay."""
from collections import Counter, defaultdict

def diagnose(raw):
    steps, calls = raw['scheduler_steps'], raw['engine_calls']
    events = [e for e in raw['output_events'] if e.get('chunk_size', 0) > 0]
    by_request, engines, prompts, prefill = defaultdict(list), {}, {}, Counter()
    for call in calls:
        assert call['returned'] and call['scheduler_step_stop'] == call['scheduler_step_start'] + 1
        engines[call['scheduler_step_start']] = call
    for event in sorted(events, key=lambda e: e['received_s']):
        by_request[event['request_id']].append(event)
    for step in steps:
        for row in step['scheduled']:
            prompts[row['request_id']] = row['prompt_tokens']
            prefill[row['request_id']] += row['prefill_tokens']
    def other_output(request, start, stop):
        counts = Counter()
        if start is not None and stop is not None:
            for event in events:
                if event['request_id'] != request and start < event['received_s'] <= stop:
                    counts[event['request_id']] += event['chunk_size']
        return dict(tokens=sum(counts.values()), by_request=dict(counts))
    preemptions = []
    for position, step in enumerate(steps):
        for request in step['preempted_request_ids']:
            resume = next((s for s in steps[position:] if any(r['request_id'] == request for r in s['scheduled'])), None)
            prior = [e for e in by_request[request] if e['received_s'] <= step['start_s']]
            last = prior[-1]['received_s'] if prior else None
            first = next((e['received_s'] for e in by_request[request]
                          if resume and e['received_s'] >= engines[resume['step']]['return_s']), None)
            absence_end = resume['start_s'] if resume else max(c['return_s'] for c in calls)
            absence_end = max(step['end_s'], absence_end)
            absence_steps = [s['step'] for s in steps[position:]
                             if resume is None or s['step'] < resume['step']]
            replay = [r for s in steps[position:] for r in s['scheduled'] if r['request_id'] == request
                      and first is not None and engines[s['step']]['return_s'] <= first]
            preemptions.append(dict(request_id=request, preemption_step=step['step'], preemption_s=step['start_s'],
                resume_step=resume['step'] if resume else None, last_token_before_s=last,
                first_new_token_after_resume_s=first, token_gap_s=first - last if first is not None and last is not None else None,
                absence_start_s=step['end_s'], absence_end_s=absence_end, absence_s=absence_end - step['end_s'],
                absent_steps=absence_steps, resume_censored=resume is None, next_token_censored=first is None,
                other_output_during_gap=other_output(request, last, first),
                other_output_during_absence=other_output(request, step['end_s'], absence_end),
                scheduled_prefill_until_next_token=sum(r['prefill_tokens'] for r in replay),
                scheduled_tokens_until_next_token=sum(r['scheduled_tokens'] for r in replay)))
    gaps = []
    for request, output in by_request.items():
        for left, right in zip(output, output[1:]):
            start, stop = left['received_s'], right['received_s']
            matching = [i for i, p in enumerate(preemptions)
                        if p['request_id'] == request and start < p['preemption_s'] <= stop]
            gaps.append(dict(request_id=request, start_s=start, stop_s=stop, gap_s=stop - start,
                matching_preemption_indices=matching,
                matching_preemption_steps=[preemptions[i]['preemption_step'] for i in matching]))
    gaps.sort(key=lambda g: g['gap_s'], reverse=True)
    long = [g for g in gaps if g['gap_s'] >= 1.0]
    unmatched = next((g for g in gaps if not g['matching_preemption_indices']), None)
    for gap in [*long, gaps[0] if gaps else None, unmatched]:
        if gap:
            gap['other_output'] = other_output(gap['request_id'], gap['start_s'], gap['stop_s'])
    return dict(preemptions=preemptions, preemption_events=len(preemptions),
        longest_gap=gaps[0] if gaps else None,
        longest_gap_without_matching_preemption=unmatched,
        second_scale_gaps=long, second_scale_gap_count=len(long),
        second_scale_with_preemption=sum(bool(g['matching_preemption_indices']) for g in long),
        token_level_itl_resolved=all(e['chunk_size'] <= 1 for e in events),
        scheduled_prefill=dict(expected_unique_prompt_tokens=sum(prompts.values()), actual_tokens=sum(prefill.values()),
            excess_tokens=sum(prefill.values()) - sum(prompts.values()),
            excess_by_request={r: prefill[r] - n for r, n in prompts.items()}),
        scope=['Only raw scheduler_steps/output_events/engine_calls; source IDs are already recorded by capture.',
               'Absence spans preemption scheduler return to first rescheduling start; censored at final engine return if unresolved.',
               'Token gaps use positive-output host receipts; other outputs use (start, stop]. No pre-first-token gap is invented.',
               'Several preemptions may match one token gap; do not sum their overlapping gaps or interpret overlap as causal attribution.',
               'Prefill excess is actually scheduled original-prompt work beyond one pass; replayed generated tokens are not included.'])
