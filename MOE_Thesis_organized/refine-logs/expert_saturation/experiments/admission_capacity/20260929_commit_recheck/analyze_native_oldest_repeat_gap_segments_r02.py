#!/usr/bin/env python3
"""Describe observed gap segments; no counterfactual or gate attribution."""
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
SESSION = ROOT / 'moe-a-native-oldest-repeat-session-r02-20261002'
CANONICAL = ROOT / 'A_NATIVE_OLDEST_REPEAT_TRIPLET_RESULT_R02_20261002.json'
OUT = ROOT / 'A_NATIVE_OLDEST_REPEAT_GAP_SEGMENTS_R02_20261002.json'


def stats(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    def percentile(p):
        x = (len(values) - 1) * p
        lo = int(x)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return dict(n=len(values), minimum=values[0], median=median(values),
                p90=percentile(.90), p95=percentile(.95), maximum=values[-1])


def main():
    canonical = json.loads(CANONICAL.read_text())
    archive = SESSION / 'cell-00-queue_fund' / 'archive'
    raw = json.loads((archive / 'raw.json').read_text())
    store = json.loads((archive / 'selective-store.json').read_text())
    origin = raw['measurement_origin_perf_counter_s']
    anchors = [r for r in store['events'] if r['event'] == 'oldest_anchor']
    gaps = []
    for request in raw['requests']:
        times = request['token_times_s']
        index = max(range(1, len(times)), key=lambda i: times[i] - times[i-1])
        lo, hi = times[index-1:index+1]
        rid = request['internal_request_id']
        actual = [p for p in raw['preemption_events']
                  if p.get('internal_request_id') == rid
                  and p.get('original_preemption_called') is True
                  and p.get('original_preemption_returned') is True
                  and lo <= p['method_entered_s'] <= hi]
        selected = [a for a in anchors if a['target'] == rid
                    and lo <= a['host_perf_counter_s'] - origin <= hi]
        admissions = [a for a in store['residency_admissions']
                      if a['request'] == rid
                      and lo <= a['host_perf_counter_s'] - origin <= hi]
        gaps.append(dict(source=request['request_id'], gap_s=hi-lo,
            last_output_s=lo, next_output_s=hi, output_count_before_gap=index,
            preemptions=[dict(step=p['engine_call_index'],
                time_s=p['method_entered_s']) for p in actual],
            anchors=[dict(episode_id=a['episode_id'],
                time_s=a['host_perf_counter_s']-origin,
                last_output_to_anchor_s=a['host_perf_counter_s']-origin-lo,
                anchor_to_next_output_s=hi-(a['host_perf_counter_s']-origin))
                for a in selected],
            admissions=[dict(step=a['step'],
                time_s=a['host_perf_counter_s']-origin,
                admission_to_next_output_s=hi-(a['host_perf_counter_s']-origin))
                for a in admissions]))
    episodes = []
    requests = {r['internal_request_id']: r for r in raw['requests']}
    for anchor in anchors:
        when = anchor['host_perf_counter_s'] - origin
        times = requests[anchor['target']]['token_times_s']
        previous = max(t for t in times if t <= when)
        following = min(t for t in times if t > when)
        episodes.append(dict(episode_id=anchor['episode_id'],
            prior_output_to_anchor_s=when-previous,
            time_beyond_1s_trigger_at_anchor_s=when-previous-1.0,
            anchor_to_first_output_s=following-when,
            complete_gap_s=following-previous))
    gaps.sort(key=lambda r: r['gap_s'], reverse=True)
    by_arm = {arm: {r['request_id']: r for r in a['metrics']['requests']}
              for arm, a in canonical['arms'].items()}
    comparisons = []
    for source, funded in by_arm['queue_fund'].items():
        native = by_arm['native'][source]
        comparisons.append(dict(source=source, native_gap_s=native['max_gap_s'],
            fund_gap_s=funded['max_gap_s'],
            gap_delta_s=funded['max_gap_s']-native['max_gap_s']))
    worse = [r for r in comparisons if r['gap_delta_s'] > 0]
    better = [r for r in comparisons if r['gap_delta_s'] < 0]
    result = dict(status='COMPLETE_DESCRIPTIVE_LOCAL_ANALYSIS',
        inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (CANONICAL, archive/'raw.json', archive/'selective-store.json')},
        episode_segments={key: stats([r[key] for r in episodes]) for key in
            ('prior_output_to_anchor_s', 'time_beyond_1s_trigger_at_anchor_s',
             'anchor_to_first_output_s', 'complete_gap_s')},
        long_request_maxima={str(threshold): dict(
            count=sum(r['gap_s'] > threshold for r in gaps),
            with_actual_preemption=sum(r['gap_s'] > threshold and bool(r['preemptions']) for r in gaps),
            with_funded_anchor=sum(r['gap_s'] > threshold and bool(r['anchors']) for r in gaps))
            for threshold in (.1, 1.0, 2.0)},
        fund_vs_native=dict(improved=len(better), worsened=len(worse),
            worsening_size_s=stats([r['gap_delta_s'] for r in worse]),
            improvement_size_s=stats([-r['gap_delta_s'] for r in better]),
            worsened_by_more_than_100ms=sum(r['gap_delta_s'] > .1 for r in worse),
            worsened_by_more_than_1s=sum(r['gap_delta_s'] > 1 for r in worse)),
        aggregate_gate_counts=store.get('oldest_gate_counts'),
        request_maxima=gaps, episodes=episodes, request_comparisons=comparisons,
        limitation='Observed same-process perf-counter segments only. Waiting before an anchor includes multiple possible gates; aggregate gate counts cannot attribute any particular delay. Admission-to-output is not device LOAD duration. Separate arm trajectories and changed natural outputs do not identify per-request causal cost. Request and episode observations are correlated.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ('request_maxima','episodes','request_comparisons')},indent=2))
    print('TOP3', json.dumps(gaps[:3], indent=2))


if __name__ == '__main__':
    main()
