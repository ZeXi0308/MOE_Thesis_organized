#!/usr/bin/env python3
"""Apply the predeclared four-point rule to two complete, pinned block audits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = '597bedb4573531d49c4e173fc8dc3b8c6c937a05fdc9e8272c0562b5fbc43c78'

def read_pinned(path, expected):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f'Audit drift: {path}')
    return json.loads(payload)

def select(t30, t200):
    rows = []
    identity = None
    for block, audit, expected_status, arms in (
        ('T30', t30, 'TWO_OF_FOUR_POINTS_NO_SELECTION', ('ltr_t30_q1', 'ltr_t30_q10')),
        ('T200', t200, 'T200_BLOCK_COMPLETE_SELECTION_REQUIRES_T30', ('ltr_t200_q1', 'ltr_t200_q10')),
    ):
        if audit['status'] != expected_status or audit['package_manifest_sha256'] != MANIFEST:
            raise ValueError(f'{block}: wrong audit identity or incomplete block')
        metrics = audit['metrics']
        if set(metrics) != {'eager', *arms}:
            raise ValueError(f'{block}: unexpected arms')
        for arm, m in metrics.items():
            if (m['completed'], m['failed'], m['unfinished'], m['expected_requests']) != (64, 0, 0, 64):
                raise ValueError(f'{block}/{arm}: incomplete cohort')
            cohort = sorted((r['request_id'], r['document_id'], r['prompt_sha256'], r['arrival_s'], r['max_output']) for r in m['requests'])
            if identity is None:
                identity = cohort
            if cohort != identity:
                raise ValueError(f'{block}/{arm}: cross-block cohort differs')
        eager = metrics['eager']
        for arm in arms:
            m = metrics[arm]
            rate_ratio = m['actual_output_tokens_s'] / eager['actual_output_tokens_s']
            flow_ratio = m['mean_flow_with_incomplete_penalty_s'] / eager['mean_flow_with_incomplete_penalty_s']
            eligible = rate_ratio >= .97 and flow_ratio <= 1.05
            if eligible != audit['comparisons'][arm + '_vs_eager']['within_frozen_97pct_output_105pct_mean_flow_budget']:
                raise ValueError(f'{arm}: budget disagrees with audit')
            rows.append(dict(block=block, arm=arm, eligible=eligible,
                actual_output_tokens_s=m['actual_output_tokens_s'],
                mean_flow_s=m['mean_flow_with_incomplete_penalty_s'],
                max_gap_s=m['max_gap_request_max_s'],
                eager_actual_output_tokens_s=eager['actual_output_tokens_s'],
                eager_mean_flow_s=eager['mean_flow_with_incomplete_penalty_s'],
                eager_max_gap_s=eager['max_gap_request_max_s'],
                output_rate_ratio=rate_ratio, mean_flow_ratio=flow_ratio,
                gap_ratio=m['max_gap_request_max_s']/eager['max_gap_request_max_s']))
    eligible = [r for r in rows if r['eligible']]
    ranked = sorted(eligible, key=lambda r: (r['max_gap_s'], -r['actual_output_tokens_s'], r['mean_flow_s'], int(r['arm'].split('_')[1][1:]), int(r['arm'].split('_')[2][1:])))
    winners = [ranked[0]['arm']] if ranked else []
    return {'schema_version': 1,
        'status': 'FOUR_POINTS_COMPLETE_NO_ELIGIBLE_POINT' if not winners else 'FOUR_POINTS_COMPLETE_DEVELOPMENT_SELECTION',
        'rule': 'Within its own block: token rate >=97% of eager and mean flow <=105%; minimum absolute maximum generation gap among eligible points. Frozen H transfer contract tie-break: higher token rate, lower mean flow, lower T, lower Q. No grid expansion.',
        'package_manifest_sha256': MANIFEST, 'points': rows,
        'eligible_arms': [r['arm'] for r in eligible],
        'selected_arm': winners[0] if len(winners)==1 else None,
        'tied_best_arms': winners if len(winners)>1 else [],
        'evidence_ceiling': 'Development calibration on seen G64. Single ordered block per point, natural outputs can differ. No statistical stability, blind validation, full-LTR reproduction, H128 transfer, or method contribution established.'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--t30', type=Path, required=True)
    p.add_argument('--t30-sha256', required=True)
    p.add_argument('--t200', type=Path, required=True)
    p.add_argument('--t200-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = select(read_pinned(args.t30,args.t30_sha256), read_pinned(args.t200,args.t200_sha256))
    result['audit_sources'] = [{'path':str(args.t30),'sha256':args.t30_sha256}, {'path':str(args.t200),'sha256':args.t200_sha256}]
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({'status':result['status'],'selected_arm':result['selected_arm'],'eligible_arms':result['eligible_arms']}))

if __name__=='__main__':
    main()
