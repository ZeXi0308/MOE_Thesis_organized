#!/usr/bin/env python3
"""Thin v7 paired comparison restricted to the same frozen two-burst scenario."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import compare_math_runs_v7 as base


def arrival_summary(path):
    raw = base.read(path)
    return {key:raw.get(key) for key in ('schema', 'arrival_contract', 'metrics_definitions', 'cohorts', 'per_request')}


def compare(left_analysis, right_analysis, left_run=None, right_run=None,
            left_label='left', right_label='right', allowed=('max_num_seqs',), allow_policy_change=False):
    la, ra = arrival_summary(left_analysis), arrival_summary(right_analysis)
    if la['schema'] != 'c-math-arrival-analysis-v1' or ra['schema'] != la['schema']:
        raise ValueError('Only two arrival analyses may be paired; old zero-arrival results are a different scenario')
    if not la['arrival_contract'] or la['arrival_contract'] != ra['arrival_contract']:
        raise ValueError('Offered arrival contracts differ')
    if not la['metrics_definitions'] or la['metrics_definitions'] != ra['metrics_definitions']:
        raise ValueError('Arrival metric definitions differ')
    result = base.compare(left_analysis, right_analysis, left_run, right_run,
        left_label, right_label, allowed, allow_policy_change)
    left, right = ({r['request_id']:r for r in arm['per_request']} for arm in (la, ra))
    for row in result['per_request']:
        l, r = left[row['request_id']], right[row['request_id']]
        for metric in ('arrival_s', 'completion_at_s', 'submission_lag_s', 'submission_return_lag_s'):
            row[metric] = base.change(l.get(metric), r.get(metric))
    tails = {'gsm8k-test-0236'}
    for arm in (left, right):
        for metric in ('completion_at_s', 'completion_s'):
            available = [r for r in arm.values() if base.numeric(r.get(metric))]
            if available: tails.add(max(available, key=lambda r:r[metric])['request_id'])
    result['output_tail_sensitivity'] = [dict(request_id=rid,
        **{metric:base.change(left[rid].get(metric), right[rid].get(metric)) for metric in
           ('arrival_s', 'output_tokens', 'completion_s', 'completion_at_s')},
        left_correct=left[rid].get('correct'), right_correct=right[rid].get('correct'),
        left_finish_reason=left[rid].get('finish_reason'), right_finish_reason=right[rid].get('finish_reason'),
        left_truncated=left[rid].get('truncated'), right_truncated=right[rid].get('truncated'))
        for rid in sorted(tails & set(left) & set(right))]
    result['output_tail_selection'] = 'Union of maximum completion_at_s and maximum flow in either arm, plus historical request 0236.'
    result['cohort_changes'] = []
    for lc, rc in zip(la['cohorts'], ra['cohorts']):
        if lc['arrival_s'] != rc['arrival_s']: raise ValueError('Cohort order differs')
        result['cohort_changes'].append(dict(arrival_s=lc['arrival_s'],
            count_changes={k:base.change(lc['counts'][k],rc['counts'][k]) for k in lc['counts']},
            timing_changes={metric:{stat:base.change(lc['timing'][metric][stat],rc['timing'][metric][stat])
                for stat in ('mean','max','p50','p90','p95','observed','missing')} for metric in lc['timing']}))
    result.update(schema='c-math-arrival-run-pair-v1', arrival_contract=la['arrival_contract'],
        metrics_definitions=la['metrics_definitions'], comparator_sha256=base.digest(__file__),
        base_comparator_sha256=base.digest(base.__file__))
    result['limitations'].append('completion_s is offered-arrival flow including submission lag; completion_at_s locates the episode tail. Cross-arrival-scenario speed claims are excluded.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('left-analysis', 'right-analysis', 'output'): parser.add_argument('--'+key, type=Path, required=True)
    for key in ('left-run', 'right-run'): parser.add_argument('--'+key, type=Path)
    for key in ('left-label', 'right-label'): parser.add_argument('--'+key, default=key.split('-')[0])
    parser.add_argument('--allow-engine-difference', action='append')
    parser.add_argument('--allow-policy-change', action='store_true')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = compare(args.left_analysis, args.right_analysis, args.left_run, args.right_run,
        args.left_label, args.right_label, args.allow_engine_difference or ('max_num_seqs',), args.allow_policy_change)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key:result[key] for key in ('comparison_integrity','issues','episode_changes')}))
    sys.exit(0 if result['comparison_integrity'] != 'MISMATCH' else 2)


if __name__ == '__main__':
    main()
