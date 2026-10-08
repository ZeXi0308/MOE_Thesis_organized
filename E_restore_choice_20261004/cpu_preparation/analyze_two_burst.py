"""Extract completed two-burst service results; no runtime or GPU imports.

Usage: python -B cpu_preparation/analyze_two_burst.py GROUP [--out JSON]
Reuses analyze.py for all existing timing/work/consistency definitions. Adds
arrival cohorts and explicitly distinguishes episode duration from drain tail.
An incomplete arm is retained as a failure record, never filled with zeros.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze import analyze_group, counts, read_json, stats


def metrics(rows):
    done = [r for r in rows if r['completed']]
    arrivals = [r['arrival_s'] for r in rows]
    last = max((r['completion_s'] for r in done), default=None)
    return dict(requests=len(rows), completed_requests=len(done),
        unfinished_requests=len(rows)-len(done),
        output_tokens=sum(r['output_tokens'] for r in rows),
        completion_latency_s=stats(r['completion_latency_s'] for r in rows),
        ttft_s=stats(r['ttft_s'] for r in rows),
        per_request_max_generation_gap_s=stats(r['token_gap_s']['max'] for r in rows),
        dispatch_start_lag_s=stats(r['admitted_s']-r['arrival_s'] for r in rows
                                 if r.get('admitted_s') is not None),
        first_dispatch_s=min((r['admitted_s'] for r in rows
                              if r.get('admitted_s') is not None), default=None),
        last_dispatch_s=max((r['admitted_s'] for r in rows
                             if r.get('admitted_s') is not None), default=None),
        final_completion_s=last,
        final_completion_after_last_arrival_s=last-max(arrivals)
            if last is not None and arrivals and len(done) == len(rows) else None,
        finish_reasons=counts(rows, 'finish_reason'), stop_reasons=counts(rows, 'stop_reason'))


def actions(commits):
    eligible = [c for c in commits if c.get('eligible')]
    return dict(commits=len(commits), actual_actions=counts(commits, 'actual_action'),
        jointly_eligible_commits=len(eligible),
        jointly_eligible_actual_actions=counts(eligible, 'actual_action'),
        fallback_count=sum(c.get('fallback') not in (None, 'none') for c in commits),
        fallbacks=counts(commits, 'fallback', include_none=False))


def overlap_snapshots(cell, boundary):
    if boundary is None:
        return []
    raw = read_json(Path(cell['raw_path']))
    first = [r for r in raw['requests'] if r['arrival_s'] < boundary]
    second = [r for r in raw['requests'] if r['arrival_s'] == boundary]
    samples = [('second_external_arrival', boundary),
               ('second_first_dispatch_start', min(r['admitted_s'] for r in second)),
               ('second_last_dispatch_start', max(r['admitted_s'] for r in second))]
    result = []
    for name, timestamp in samples:
        emitted = [bisect_right(r['token_times_s'], timestamp) for r in first]
        completed = sum(r.get('completed') and r['completion_s'] <= timestamp for r in first)
        result.append(dict(boundary=name, time_s=timestamp,
            first_cohort_requests=len(first), first_cohort_completed=completed,
            first_cohort_remaining=len(first)-completed,
            first_cohort_requests_without_output=sum(n == 0 for n in emitted),
            first_cohort_output_tokens_emitted=sum(emitted),
            first_cohort_requested_output_tokens=sum(r['max_output_tokens'] for r in first),
            first_cohort_emitted_tokens_per_request=stats(emitted),
            first_cohort_output_cap_fraction_per_request=stats(
                n/r['max_output_tokens'] for n, r in zip(emitted, first))))
    return result


def extract_cell(cell):
    rows = cell['request_metrics']
    arrivals = sorted({r['arrival_s'] for r in rows})
    cohorts = []
    work = cell['scheduled_position_counts']
    commits = cell['committed_event_diagnostics']
    for arrival in arrivals:
        members = [r for r in rows if r['arrival_s'] == arrival]
        ids = {r['request_id'] for r in members}
        cohort_work = [r for r in work['by_request'] if r['request_id'] in ids]
        cohorts.append(dict(arrival_s=arrival, **metrics(members),
            recovery=actions([c for c in commits if c['request_id'] in ids]),
            scheduled_position_work={k: sum(r[k] for r in cohort_work) for k in work['totals']},
            preemptions=sum(cell['preemptions_by_request'].get(rid, 0) for rid in ids)))
    boundary = arrivals[1] if len(arrivals) == 2 else None
    earlier = [r for r in rows if boundary is not None and r['arrival_s'] < boundary]
    full = metrics(rows)
    full.update(makespan_s=cell['makespan_s'], output_tokens_per_s=cell['output_tokens_per_s'],
        service_and_drain_from_episode_origin_s=cell['service_and_drain_s'],
        service_and_drain_after_last_arrival_s=cell['service_and_drain_s']-max(arrivals)
            if cell['service_and_drain_s'] is not None and arrivals else None,
        post_completion_connector_drain_and_bookkeeping_s=
            cell['service_and_drain_s']-cell['all_complete_s']
            if cell['service_and_drain_s'] is not None and cell['all_complete_s'] is not None else None)
    return dict(cell=cell['cell'], policy=cell['policy'], status=cell['status'],
        expected_requests=cell['config'].get('requests'), full_service=full, cohorts=cohorts,
        first_cohort_unfinished_at_second_external_arrival=sum(
            not r['completed'] or r['completion_s'] > boundary for r in earlier)
            if boundary is not None else None,
        second_burst_overlap_snapshots=overlap_snapshots(cell, boundary),
        recovery=actions(commits), decision_actions=cell['decision_actions'],
        decision_fallbacks=cell['decision_fallbacks'], preemptions=cell['preemptions'],
        preempted_requests=cell['preempted_requests'],
        scheduled_position_work=work['totals'], transfers=cell['transfers'],
        capacity_pressure=cell['capacity_pressure'],
        eligible_decision_to_next_output_s=cell['eligible_decision_to_next_output_s'],
        warnings=cell['warnings'])


def pair_delta(left, right):
    a = {r['external_id']: r for r in left['request_metrics']}
    b = {r['external_id']: r for r in right['request_metrics']}
    if a.keys() != b.keys():
        return dict(status='REQUEST_SET_MISMATCH')
    rows = [dict(external_id=k, arrival_s=a[k]['arrival_s'],
                 completion_latency_delta_s=b[k]['completion_latency_s']-a[k]['completion_latency_s'],
                 max_gap_delta_s=b[k]['token_gap_s']['max']-a[k]['token_gap_s']['max'])
            for k in sorted(a)]
    latency = [r['completion_latency_delta_s'] for r in rows]
    return dict(left_cell=left['cell'], right_cell=right['cell'],
        direction='right minus left; descriptive paired requests, not independent run repetitions',
        per_request_completion_delta_s=stats(latency),
        earlier_requests=sum(x < 0 for x in latency), later_requests=sum(x > 0 for x in latency),
        equal_requests=sum(x == 0 for x in latency),
        largest_completion_benefits=sorted(rows, key=lambda r:r['completion_latency_delta_s'])[:5],
        largest_completion_harms=sorted(rows, key=lambda r:r['completion_latency_delta_s'], reverse=True)[:5])


def analyze(group):
    group_status = read_json(Path(group)/'status.json', {})
    if group_status.get('status') not in ('COMPLETE', 'FAILED', 'ABORT_LOCK_BUSY'):
        raise ValueError('Refusing raw reads until whole group is terminal.')
    base = analyze_group(group)
    result = dict(group=base['group'], group_status=base['group_status'],
        status='EXTRACTION_ONLY',
        definitions=dict(primary='Mean completion latency from fixed external arrival, all requests.',
            cohorts='Grouped by predeclared external arrival_s, not observed admission.',
            dispatch_start_lag='Timestamp immediately before add_request minus external arrival; enqueue call completion is not instrumented.',
            drain='Report service duration from origin, duration after final external arrival, and post-completion connector drain separately.',
            quality='Equal fixed output counts do not establish equal native compute, token content, or task quality.',
            uncertainty='One arm is one run; request differences are descriptive and do not supply independent run-level samples.'),
        cells=[extract_cell(c) for c in base['cells']],
        token_consistency=base['greedy_token_consistency'],
        incomplete_cells=base['incomplete_cells'], invalid_cells=base['invalid_cells'],
        analysis_errors=base['analysis_errors'])
    # Preserve invalid/unrun arms from the reused analyzer. Do not claim a failed
    # arm has zero failures merely because only complete cells are summarised.
    result['comparison_complete'] = bool(len(base['cells']) == 2 and
        base['group_status'] and base['group_status'].get('status') == 'COMPLETE' and
        not any(base[k] for k in ('incomplete_cells', 'invalid_cells', 'analysis_errors')))
    result['arrival_contract'] = [dict(cell=c['cell'],
        counts=dict(Counter(r['arrival_s'] for r in c['request_metrics'])),
        matches_160_at_0_and_160_at_20=Counter(r['arrival_s'] for r in c['request_metrics']) == {0.:160,20.:160},
        requested_output_tokens=c['output_work']['total_max_output_tokens'],
        actual_output_tokens=c['output_tokens']) for c in base['cells']]
    result['two_burst_full_policy_contract'] = bool(result['comparison_complete'] and
        {c['policy'] for c in base['cells']} == {'host', 'recompute'} and
        all(c['target_selection'].get('selection_mode') == 'full_policy' for c in base['cells']) and
        all(c['matches_160_at_0_and_160_at_20'] and c['requested_output_tokens'] == 278528
            and c['actual_output_tokens'] == 278528 for c in result['arrival_contract']))
    if result['two_burst_full_policy_contract']:
        result['status'] = 'EXPLORATORY_TWO_BURST_FULL_POLICY_COMPARISON'
    if len(base['cells']) == 2:
        result['request_differences'] = pair_delta(*base['cells'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = analyze(args.group)
    output = args.out or args.group / 'two_burst_summary.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), comparison_complete=result['comparison_complete'],
        cells=[dict(cell=c['cell'], policy=c['policy'], full_service=c['full_service'],
                    recovery=c['recovery']) for c in result['cells']]), indent=2))
    return 0 if result['comparison_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
