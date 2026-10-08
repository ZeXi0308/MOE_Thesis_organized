#!/usr/bin/env python3
"""Describe two fixed-cap development pairs without legacy performance gates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

METRICS = ('joint_slo_requests_per_s', 'output_tokens_per_s', 'prompt_plus_output_tokens_per_s',
    'ttft_mean_s', 'ttft_p95_s', 'ttft_max_s', 'completion_flow_mean_s',
    'completion_flow_p95_s', 'completion_flow_max_s', 'request_max_gap_mean_s',
    'request_max_gap_p95_s', 'request_max_gap_max_s', 'elapsed_s')


def relative(a, b): return 100*(a/b-1) if a is not None and b not in (None, 0) else None
def delta(a, b): return a-b if a is not None and b is not None else None
def quantile(v, q):
    if not v: return None
    v = sorted(v); x = (len(v)-1)*q; i = int(x)
    return v[i]+(v[min(i+1, len(v)-1)]-v[i])*(x-i)


def arm(row):
    s = row['summary']
    return dict(cell=row['cell'], policy=row['policy'], valid=row['valid'],
        qualified_numerator=s['joint_slo_qualified_requests'], full_elapsed_s=s['elapsed_s'],
        U=s['joint_slo_requests_per_s'], request_count=s['request_count'],
        finished_count=s['finished_count'], output_tokens=s['output_tokens'])


def pair_description(pair, rows):
    c, r = rows.get(pair.get('small_cell')), rows.get(pair.get('large_cell'))
    direction = pair.get('directions', {}).get('small_over_large')
    if not c or not r or 'summary' not in c or 'summary' not in r or direction is None:
        return dict(candidate_cell=pair.get('small_cell'), reference_cell=pair.get('large_cell'), status='UNAVAILABLE')
    cs, rs = c['summary'], r['summary']
    flow = {rid: -v if v is not None else None for rid, v in pair.get('flow_large_minus_small_s_by_request', {}).items()}
    values = [v for v in flow.values() if v is not None]; missing = len(flow)-len(values)
    return dict(status='DESCRIBED', candidate=arm(c), reference=arm(r),
        delta_U_req_per_s=delta(cs['joint_slo_requests_per_s'], rs['joint_slo_requests_per_s']),
        delta_U_percent=relative(cs['joint_slo_requests_per_s'], rs['joint_slo_requests_per_s']),
        metrics={k: dict(candidate=cs.get(k), reference=rs.get(k),
            delta=delta(cs.get(k), rs.get(k)), relative_percent=relative(cs.get(k), rs.get(k))) for k in METRICS},
        qualified_added_ids=direction['qualified_added_ids'], qualified_lost_ids=direction['qualified_lost_ids'],
        additional_100ms_gap_failure_ids=direction['additional_100ms_gap_failure_ids'],
        failed_by_slo_field=dict(candidate=c.get('failed_by_slo_field'), reference=r.get('failed_by_slo_field')),
        flow_candidate_minus_reference_s=dict(requests=len(flow), observed=len(values), missing=missing,
            harmed=sum(v > 0 for v in values), improved=sum(v < 0 for v in values), unchanged=sum(v == 0 for v in values),
            mean=statistics.fmean(values) if values and not missing else None,
            median=statistics.median(values) if values and not missing else None,
            p95=quantile(values, .95) if not missing else None,
            min=min(values, default=None), max=max(values, default=None)),
        output_lengths_match=pair.get('output_lengths_match'),
        output_token_ids_different_count=pair.get('output_token_ids_different_count'),
        different_output_LCP_tokens=pair.get('different_output_LCP_tokens'),
        timing=dict(candidate=c.get('timing'), reference=r.get('timing')))


def repeat_description(arms):
    if len(arms) != 2: return dict(status='UNAVAILABLE', observed_episodes=len(arms))
    us = [a['U'] for a in arms]
    return dict(episodes=arms, second_minus_first_U_req_per_s=delta(us[1], us[0]),
        second_vs_first_U_percent=relative(us[1], us[0]), U_range_req_per_s=max(us)-min(us),
        note='Two episode observations, not a noise distribution, confidence interval or acceptance threshold.')


def summarize(source):
    rows = {r['cell']: r for r in source.get('runs', [])}
    pairs = [pair_description(p, rows) for p in source.get('pairs', [])]
    described = [p for p in pairs if p['status'] == 'DESCRIBED']
    coverage = source.get('warm_actual_cap_coverage', {})
    expected_warms = source.get('frozen_design', {}).get('warm_policies', [])
    warm_ready = bool(expected_warms) and set(coverage) == set(expected_warms) and all(
        v.get('warm_valid_complete') is True and v.get('missing_caps') == [] for v in coverage.values())
    ready = (source.get('observed_formal_runs') == 4 and source.get('valid_complete_formal_runs') == 4
        and source.get('driver_status', {}).get('status') == 'COMPLETE' and warm_ready
        and not source.get('missing_formal_cells') and not source.get('duplicate_cells')
        and not source.get('unexpected_raw_paths') and len(described) == len(pairs) == 2
        and all(p[side]['valid'] for p in described for side in ('candidate', 'reference')))
    no_data = not source.get('observed_formal_runs', 0) and not source.get('runs')
    status = 'COMPLETE_EXPLORATORY' if ready else 'UNRUN' if no_data and source.get('driver_status', {}).get('status') not in ('FAILED', 'COMPLETE', 'UNREADABLE') else 'INCOMPLETE_OR_INVALID'
    repeats = {side: repeat_description([p[side] for p in described]) for side in ('candidate', 'reference')}
    pooled = None
    if ready:
        pooled = {}
        for side in ('candidate', 'reference'):
            n = sum(p[side]['qualified_numerator'] for p in described)
            t = sum(p[side]['full_elapsed_s'] for p in described)
            pooled[side] = dict(sum_qualified=n, sum_full_elapsed_s=t, U=n/t)
        pooled['delta_U_req_per_s'] = pooled['candidate']['U']-pooled['reference']['U']
        pooled['delta_U_percent'] = relative(pooled['candidate']['U'], pooled['reference']['U'])
    return dict(status=status, original_legacy_status=source.get('status'),
        research_contract_slo=source.get('frozen_design', {}).get('primary_slo'),
        pair_effect_signs=['positive' if p['delta_U_req_per_s'] > 0 else 'negative' if p['delta_U_req_per_s'] < 0 else 'zero' for p in described],
        pairs=pairs, within_policy_repeat_U=repeats, pooled_description=pooled,
        completeness=dict(observed_formal_runs=source.get('observed_formal_runs'),
            valid_complete_formal_runs=source.get('valid_complete_formal_runs'), driver_status=source.get('driver_status'),
            missing_formal_cells=source.get('missing_formal_cells'), duplicate_cells=source.get('duplicate_cells'),
            unexpected_raw_paths=source.get('unexpected_raw_paths'), warm_coverage_complete=warm_ready,
            warm_actual_cap_coverage=coverage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixed_results', type=Path); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); output = args.output or args.fixed_results.parent/'study_results.json'
    if output.resolve() == args.fixed_results.resolve(): parser.error('Do not overwrite source analysis')
    result = dict(status='UNRUN')
    if args.fixed_results.exists():
        blob = args.fixed_results.read_bytes()
        try: result = summarize(json.loads(blob))
        except (ValueError, KeyError, TypeError) as exc: result = dict(status='INCOMPLETE_OR_INVALID', error=repr(exc))
        result['source_sha256'] = hashlib.sha256(blob).hexdigest()
    result.update(schema='d-fixed-tradeoff-development-v1', source_file=str(args.fixed_results),
        scope=[
            'Candidate is the smaller fixed cap in each original small_over_large pair. Each pair is retained; pooled counts/time cannot hide opposite signs.',
            '4s TTFT / 0.1s maximum generation gap / 20s completion is a research contract without established production justification. U is qualified requests divided by full elapsed, including all arrivals and drain.',
            'Complete status requires existing valid complete episodes, COMPLETE driver and recorded warm cap coverage; it does not indicate a scientific win or independent confirmation.',
            'Original 3% benefit and 5% cost gates remain legacy fields in fixed_results.json; they do not define this exploratory status. Throughput/latency costs are reported without requiring simultaneous improvement.',
            'Episodes are the repeated units. Per-request changes explain consequences, not independent experimental repetitions. No automatic significance, winner, novelty or continuation decision is made.',
            'Output sequence differences are retained; equal fixed output lengths do not establish equal content, routing, computation or quality. No raw data, source analysis or measured denominator is modified.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(f'{result["status"]}; {output}')


if __name__ == '__main__': main()
