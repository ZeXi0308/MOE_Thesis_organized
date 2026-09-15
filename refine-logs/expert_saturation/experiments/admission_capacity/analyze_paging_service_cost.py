"""Attach existing executed retention costs to complete request intervals.

Only engine-call and outside-call host intervals form an additive partition.
CUDA ensure envelopes, host apply and control counters are separate observations.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    source = a.input_dir / 'analysis.json'
    analysis = json.loads(source.read_text())
    assert not analysis['issues']
    episodes = []
    for engine in analysis['engines']:
        for row in engine['rows']:
            raw_path = a.input_dir / 'results' / engine['label'] / row['name'] / 'raw.json'
            raw = json.loads(raw_path.read_text())
            calls = raw['engine_calls']
            assert raw['status'] == 'COMPLETE' and len(calls) == 24
            assert all(c['returned'] and c['return_s'] >= c['start_s'] for c in calls)
            assert all(x['return_s'] <= y['start_s'] for x, y in zip(calls, calls[1:]))
            requests = list(row['requests'].values())
            assert len(requests) == 3 and all(r['status'] == 'completed' for r in requests)
            parts = []
            weighted_calls = []
            for c in calls:
                overlap = sum(max(0., min(c['return_s'], r['arrival_s'] + r['completion_latency_s'])
                                  - max(c['start_s'], r['arrival_s'])) for r in requests) / 3
                weighted_calls.append(dict(step=c['index'], engine_wall_s=c['return_s']-c['start_s'],
                                           mean_completion_contribution_s=overlap))
            for r in requests:
                begin, end = r['arrival_s'], r['arrival_s'] + r['completion_latency_s']
                inside = sum(max(0., min(c['return_s'], end) - max(c['start_s'], begin)) for c in calls)
                outside = end-begin-inside
                assert outside >= -1e-10
                parts.append(dict(arrival_s=begin, completion_s=end, latency_s=end-begin,
                                  engine_intersection_s=inside, outside_engine_s=outside))
            mean_latency = sum(r['latency_s'] for r in parts)/3
            mean_engine = sum(c['mean_completion_contribution_s'] for c in weighted_calls)
            mean_outside = sum(r['outside_engine_s'] for r in parts)/3
            assert abs(mean_latency - mean_engine - mean_outside) < 1e-10
            episodes.append(dict(engine=engine['label'], repeat=row['repeat'], arm=row['arm'],
                raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                capture_wall_s=row['capture_wall_s'], cycle_wall_s=row['cycle_wall_s'],
                mean_completion_s=mean_latency, mean_engine_s=mean_engine,
                mean_outside_engine_s=mean_outside, calls=weighted_calls,
                payload_bytes=row['payload_bytes'], groups=row['groups'],
                ensure_cuda_envelope_ms=row['load_section_ms'],
                warmup_capture_s=row['warmup_capture_wall_s'], flush_wall_s=row['flush_wall_s'],
                pure_h2d_ms=None, exposed_h2d_wait_ms=None, isolated_policy_control_ms=None))
    pairs = [dict(engine=p['engine'], block=p['block'], baseline=p['baseline'], treatment=p['treatment'],
                  delta=p['delta'], delta_pct=p['delta_pct']) for p in analysis['pairs']
             if p['baseline'] == 'none_early']
    assert len(episodes) == 12 and len(pairs) == 8
    result = dict(status='OBSERVED_REQUEST_INTERVAL_ACCOUNTING',
        input_analysis_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), episodes=episodes, pairs=pairs,
        scope=['No new policy execution or causal timing estimate.',
               'Mean request latency = engine intersections + outside-engine intervals, including arrival-relative waits.',
               'CUDA ensure envelopes include copies, map updates and host submission gaps. They are not an additive sub-bucket or removable-time bound.',
               'No isolated transfer/control measurement is imputed; unavailable fields remain null.',
               'The five-warmup repeat cycle is retained separately; no unmeasured steady-service amortization.'])
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
