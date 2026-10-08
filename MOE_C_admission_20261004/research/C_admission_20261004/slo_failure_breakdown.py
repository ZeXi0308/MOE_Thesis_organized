#!/usr/bin/env python3
"""Describe the frozen 20-point SLO failures and paired crossings, without replay.

Usage: python slo_failure_breakdown.py ANALYSIS.json --output NEW.json
Reads the four per-request JSON files named by an existing simple-recheck analysis.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from analyze import TTFT_GRID_S, MAXGAP_GRID_S, FLOW_LIMIT_S
from compare_quartet import ORDERS


MASK_LABELS = {'0': 'success', '1': 'TTFT only', '2': 'gap only', '3': 'TTFT + gap',
               '4': 'flow only', '5': 'TTFT + flow', '6': 'gap + flow', '7': 'TTFT + gap + flow'}
OUTCOMES = ('completed', 'failed', 'unfinished', 'rejected', 'timeout')


def read_hashed(path):
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def failure_mask(row, ttft, gap, flow):
    if row['outcome'] != 'completed':
        return None
    values = [row[key] for key in ('ttft_s', 'max_generation_gap_s', 'flow_s')]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError('Completed request has an unresolved latency metric: '+row['request_id'])
    return sum(bit for bit, value, limit in zip((1, 2, 4), values, (ttft, gap, flow)) if value > limit)


def analyze(source, source_path, source_sha):
    order = ORDERS['simple-recheck']
    cells = {Path(cell['cell']).name: cell for cell in source['cells']}
    if set(cells) != set(order) or len(source['cells']) != 4:
        raise ValueError('Expected the explicit fixed192/KV256/KV256/fixed192 quartet')
    if len({cell['workload_identity_sha256'] for cell in cells.values()}) != 1:
        raise ValueError('Source workload identities differ')
    grid = [(t, g, FLOW_LIMIT_S) for t in TTFT_GRID_S for g in MAXGAP_GRID_S]
    population, masks, arms = {}, {}, []
    for name in order:
        cell = cells[name]
        path = Path(cell['per_request_json'])
        rows, sha = read_hashed(path)
        by_id = {row['request_id']: row for row in rows}
        if len(by_id) != len(rows) or len(rows) != cell['planned_requests']:
            raise ValueError('Duplicate/missing planned request rows: '+name)
        observed_outcomes = collections.Counter(row['outcome'] for row in rows)
        if set(observed_outcomes)-set(OUTCOMES):
            raise ValueError('Unsupported request outcome')
        outcomes = {outcome: observed_outcomes[outcome] for outcome in OUTCOMES}
        if outcomes != cell['outcomes']:
            raise ValueError('Request outcomes differ from source summary')
        reference = {(point['ttft_limit_s'], point['maxgap_limit_s'], point['flow_limit_s']): point
                     for point in cell['joint_slo_grid']}
        if set(reference) != set(grid):
            raise ValueError('Source analysis does not contain exactly the frozen 20 points')
        population[name] = by_id
        results = []
        for limits in grid:
            decisions = {rid: failure_mask(row, *limits) for rid, row in by_id.items()}
            masks[name, limits] = decisions
            counted = collections.Counter(decisions.values())
            counts = {str(mask): counted[mask] for mask in range(8)}
            noncompleted = counted[None]
            if sum(counts.values())+noncompleted != len(rows):
                raise ValueError('SLO population accounting does not conserve requests')
            if counts['0'] != reference[limits]['good_requests']:
                raise ValueError('Recomputed SLO attainment differs from original analysis')
            results.append(dict(ttft_limit_s=limits[0], maxgap_limit_s=limits[1], flow_limit_s=limits[2],
                planned_requests=len(rows), counts_by_failure_mask=counts,
                noncompleted_requests=noncompleted, good_requests=counts['0'],
                only_ttft_failed=counts['1'], only_gap_failed=counts['2'], only_flow_failed=counts['4'],
                multiple_metrics_failed=sum(counts[key] for key in ('3', '5', '6', '7')),
                joint_attainment=counts['0']/len(rows)))
        arms.append(dict(cell=name, source_per_request_json=str(path.resolve()),
            source_per_request_sha256=sha, source_raw_sha256=cell['raw_sha256'],
            workload_identity_sha256=cell['workload_identity_sha256'],
            planned_requests=len(rows), arrived_requests=cell['arrived_requests'], outcomes=outcomes,
            all20=results))
    pairs = []
    for candidate, baseline in ((order[1], order[0]), (order[2], order[3])):
        a, b = population[candidate], population[baseline]
        if a.keys() != b.keys():
            raise ValueError('Paired request IDs differ')
        for rid in a:
            if any(a[rid][key] != b[rid][key] for key in ('arrival_s', 'prompt_tokens', 'max_output_tokens')):
                raise ValueError('Paired request input metadata differs')
        points = []
        for limits in grid:
            am, bm = masks[candidate, limits], masks[baseline, limits]
            improved = [rid for rid in a if bm[rid] != 0 and am[rid] == 0]
            worsened = [rid for rid in a if bm[rid] == 0 and am[rid] != 0]
            detail = lambda ids: dict(requests=len(ids), output_count_differs=sum(
                a[rid]['output_tokens'] != b[rid]['output_tokens'] for rid in ids),
                output_sequence_differs=sum(a[rid]['output_token_ids_sha256'] !=
                                           b[rid]['output_token_ids_sha256'] for rid in ids))
            points.append(dict(ttft_limit_s=limits[0], maxgap_limit_s=limits[1], flow_limit_s=limits[2],
                paired_requests=len(a), baseline_failure_to_candidate_success=detail(improved),
                baseline_success_to_candidate_failure=detail(worsened),
                both_success=sum(am[rid] == bm[rid] == 0 for rid in a),
                both_failure=sum(am[rid] != 0 and bm[rid] != 0 for rid in a),
                net_success_change=len(improved)-len(worsened)))
        pairs.append(dict(candidate=candidate, baseline=baseline, all20=points))
    return dict(schema_version=1, source_analysis=str(source_path.resolve()),
        source_analysis_sha256=source_sha, failure_bits=dict(TTFT=1, gap=2, flow=4),
        mask_labels=MASK_LABELS, independent_unit='run', arms=arms, pairs=pairs,
        semantics='Every planned arrival remains in the denominator. Completed requests receive '
            'a three-bit mask using strict failure > the frozen limit (equality passes); zero means '
            'joint success. Noncompleted outcomes are separately retained as SLO failures, with '
            'unresolved latency masks, not imputed metric violations. Mask counts plus noncompleted '
            'counts conserve the full population. All 20 original TTFT/gap/flow threshold points '
            'are retained and their success counts are verified against the source analysis. '
            'Crossings pair the same canonical request IDs, with fixed pairs01/00 and02/03. '
            'Output changes among crossings are descriptive. These static outcome masks are not '
            'a policy counterfactual, an online signal, a reachable-benefit bound, or independent '
            'request-level repeats. No latency or failure category is removed from the service result.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite existing output; choose a new path.')
    source, sha = read_hashed(args.analysis)
    result = analyze(source, args.analysis, sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
