"""Action/position-work diagnosis only, for a terminal two-burst H/R group.

Usage: python -B cpu_preparation/analyze_two_burst_actions.py GROUP
No service-metric recalculation, new instrumentation, GPU, or online oracle.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path


def extract(path):
    raw = json.loads(path.read_text())
    rows = {r['request_id']:r for r in raw['requests']}
    names = {k:r['external_id'] for k,r in rows.items()}
    ends = [s['end_s'] for s in raw['steps']]
    seen = defaultdict(set)
    work = defaultdict(list)
    for step, schedule in enumerate(raw['scheduler_steps']):
        for e in schedule['scheduled']:
            rid = e['request_id']
            positions = set(range(e['start_computed'], e['end_computed']))
            repeat = len(positions & seen[rid])
            work[rid].append(dict(step=step, scheduled=e['count'], repeated=repeat))
            seen[rid].update(positions)
    commits = defaultdict(list)
    for e in raw['commits']:
        commits[e['request_id']].append(e)
    recovery = []
    fields = ('event','preemptions','decision_s','allocation_s','known_tokens',
              'generated_tokens','host_hit_tokens','free_blocks','full_required_blocks',
              'eligible','action','actual_action','fallback','external_tokens')
    for rid, events in commits.items():
        sequence = []
        for i,e in enumerate(events):
            step = bisect_right(ends, e['allocation_s'])
            next_step = bisect_right(ends, events[i+1]['allocation_s']) if i+1 < len(events) else len(ends)
            window = [w for w in work[rid] if step <= w['step'] < next_step]
            index = bisect_right(rows[rid]['token_times_s'], e['decision_s'])
            sequence.append(dict(**{k:e.get(k) for k in fields}, local_step=step,
                next_output_index=index,
                next_output_step=bisect_right(ends, rows[rid]['token_times_s'][index])-1,
                next_output_s=rows[rid]['token_times_s'][index],
                subsequent_commit_step=next_step if i+1 < len(events) else None,
                native_scheduled_positions_before_next_commit_or_finish=sum(w['scheduled'] for w in window),
                native_repeated_positions_before_next_commit_or_finish=sum(w['repeated'] for w in window)))
        recovery.append(dict(external_id=names[rid], commits=len(events), sequence=sequence))
    counts = lambda events,field: dict(Counter(str(e.get(field)) for e in events))
    eligible = [e for e in raw['commits'] if e.get('eligible')]
    first_preempt = raw['preemptions'][0] if raw['preemptions'] else None
    first = None if first_preempt is None else dict(external_id=names[first_preempt['request_id']],
        **{k:first_preempt[k] for k in ('time_s','known_tokens','generated_tokens','computed_tokens')},
        local_step=bisect_right(ends, first_preempt['time_s']))
    per_request = {names[rid]:dict(scheduled=sum(w['scheduled'] for w in records),
        repeated=sum(w['repeated'] for w in records), unique=len(seen[rid])) for rid,records in work.items()}
    return dict(cell=path.parent.name, raw_path=str(path), formal_steps=len(ends),
        decision_count=len(raw['decisions']), decision_fallbacks=counts(raw['decisions'],'fallback'),
        committed_count=len(raw['commits']), actual_actions=counts(raw['commits'],'actual_action'),
        jointly_eligible_decisions=sum(bool(e.get('eligible')) for e in raw['decisions']),
        jointly_eligible_commits=len(eligible), jointly_eligible_actual_actions=counts(eligible,'actual_action'),
        jointly_eligible_requests=dict(Counter(names[e['request_id']] for e in eligible)),
        committed_fallbacks=counts(raw['commits'],'fallback'),
        preemptions=len(raw['preemptions']), distinct_preempted_requests=len({e['request_id'] for e in raw['preemptions']}),
        first_preemption=first, recovery_sequences=recovery,
        position_work_totals={k:sum(w[k] for w in per_request.values()) for k in ('scheduled','unique','repeated')},
        position_work_by_request=per_request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    args = parser.parse_args()
    assert json.loads((args.group/'status.json').read_text())['status'] == 'COMPLETE'
    cells = [extract(p) for p in sorted(args.group.rglob('raw.json'))]
    assert len(cells) == 2
    left,right = cells
    delta = []
    for key in sorted(left['position_work_by_request']):
        a,b = left['position_work_by_request'][key],right['position_work_by_request'][key]
        if a != b:
            delta.append(dict(external_id=key, left=a, right=b,
                right_minus_left={k:b[k]-a[k] for k in a}))
    right_targets = set(right['jointly_eligible_requests'])
    result = dict(status='DESCRIPTIVE_EXECUTED_ACTION_AND_SUCCESSOR_DIAGNOSIS',
        group=str(args.group), cells=cells,
        changed_native_work_requests=delta,
        right_minus_left_position_work={k:right['position_work_totals'][k]-left['position_work_totals'][k]
            for k in ('scheduled','unique','repeated')},
        repeat_delta_on_right_jointly_eligible_targets=sum(
            r['right_minus_left']['repeated'] for r in delta if r['external_id'] in right_targets),
        repeat_delta_on_other_requests=sum(
            r['right_minus_left']['repeated'] for r in delta if r['external_id'] not in right_targets),
        current_judgment=[
            'Actual legal action choices occurred, but neither the opportunity count nor pre-action trajectories are matched across these full-policy runs.',
            'Within-arm successor sequences show repeated recovery work; cross-arm work differences alone do not establish that the recovery choice caused the different opportunity history.',
            'Future Host coverage changes are observations; this trace without matched intervention does not identify their cause.'],
        notes=[
            'Joint eligibility is within each observed state; events across policies are not automatically matched causal trials.',
            'Pre-action trajectories may differ because fixed wall-time arrivals land on different engine steps.',
            'Per-commit work includes the allocation step and ends before the next commit step, or at request completion.',
            'Scheduled/repeated positions are logical native work, never additive GPU wall time or an online reward.',
            'Fallback counters are lookup/commit counts, not distinct requests or additional output tokens.'])
    out = args.group/'action_space.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),cells=[{k:c[k] for k in (
        'cell','committed_count','jointly_eligible_commits','jointly_eligible_actual_actions',
        'jointly_eligible_requests','committed_fallbacks','position_work_totals','first_preemption')} for c in cells],
        right_minus_left_position_work=result['right_minus_left_position_work'],changed_native_work_requests=delta),indent=2))


if __name__ == '__main__':
    main()
