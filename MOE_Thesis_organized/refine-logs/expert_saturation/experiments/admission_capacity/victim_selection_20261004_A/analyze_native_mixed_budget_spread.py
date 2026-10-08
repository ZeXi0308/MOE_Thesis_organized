#!/usr/bin/env python3
"""Reuse the mixed-budget characterization and verify the changed arrival input."""
import argparse
import hashlib
import json
from pathlib import Path

import analyze_native_mixed_budget_probe as mixed


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / 'candidate_native_mixed_budget_spread_r01'


def analyze(session, package=PACKAGE):
    result = mixed.analyze(session, package / 'pkg/staged_store_rotation.py',
                           package / 'pkg/inputs/pro_high/config.json')
    if result['plan'].get('arrival_diagnostic') != 'SAME_MIXED_BUDGET_SPREAD_0_2_SECONDS':
        raise ValueError('Expected the frozen spread-arrival diagnostic plan')
    source = Path(result['budget_consistency']['input_config_path']).with_name('workload.json')
    workload = json.loads(source.read_text())
    ids = [r['request_id'] for r in workload['source_requests']]
    expected = {rid: i * 0.2 for i, rid in enumerate(ids)}
    observed = result['cell']['metrics']['requests']
    errors = [dict(request=r['request_id'], actual=r.get('arrival_s'),
                   expected=expected.get(r['request_id'])) for r in observed
              if r.get('arrival_s') != expected.get(r['request_id'])]
    checks = dict(input_trace=workload['arrival_traces_s']['steady'] == list(expected.values()),
                  observed_ids=len(observed) == len(expected) == 320
                      and set(r['request_id'] for r in observed) == set(expected),
                  observed_arrivals=not errors,
                  workload_hash=hashlib.sha256(source.read_bytes()).hexdigest()
                      == result['plan']['input_files_sha256']['pkg/inputs/pro_high/workload.json'])
    result['arrival_consistency'] = dict(status='VERIFIED' if all(checks.values()) else 'MISMATCH',
        checks=checks, errors=errors, gap_s=0.2, final_external_arrival_s=63.8,
        semantics='Arrival source is never paused by the policy; all latency uses these external times. '
                  'This single changed-arrival workload is not a victim-policy performance contrast.')
    result['analysis_code_sha256'][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not all(checks.values()):
        result['status'] = 'INCOMPLETE_OR_INVALID'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session.resolve())
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(args.output))))


if __name__ == '__main__':
    main()
