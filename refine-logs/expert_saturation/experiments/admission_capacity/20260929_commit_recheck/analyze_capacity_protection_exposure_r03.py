"""Describe actual protection spans and independent trajectory divergence.

Output within a protection span demonstrates concurrent service, not the causal
effect of protection. The sparse performance capture has no waiting-gate events.
"""
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
from analyze_capacity_identical_phase_r01 import timeline, first_sequence_divergence

ROOT = Path(__file__).resolve().parent
SESSION = ROOT / 'moe-a-capacity-protection-session-r03-20261001'


def main():
    result = json.loads((ROOT / 'A_CAPACITY_PROTECTION_PAIR_RESULT_R03_20261001.json').read_text())
    raws = {}
    arms = {}
    for i, arm in enumerate(('q1', 'q10')):
        raw = json.loads((SESSION / f'cell-{i:02d}-capacity_protection_{arm}/archive/raw.json').read_text())
        raws[arm] = raw
        episodes = result['arms'][arm]['episodes']
        spans = []
        token_times = sorted(t for r in raw['requests'] for t in r['token_times_s'])
        for ep in episodes:
            if ep['extend'] is None:
                continue
            start, end = ep['extend']['relative_time_s'], ep['release']['relative_time_s']
            own = next(r['token_times_s'] for r in raw['requests'] if r['request_id'] == ep['request_id'])
            own_outputs = bisect_left(own, end) - bisect_left(own, start)
            all_outputs = bisect_left(token_times, end) - bisect_left(token_times, start)
            spans.append(dict(request_id=ep['request_id'], start_s=start, end_s=end,
                duration_s=end-start, own_outputs_after_extension=own_outputs,
                other_outputs_during_extension=all_outputs-own_outputs,
                start_free_blocks=ep['extend']['free_blocks'],
                start_future_growth_blocks=ep['extend']['future_growth_blocks'],
                next_preemption_observation=ep['after_release']['next_preemption_observation']))
        ordered = sorted(spans, key=lambda x:x['start_s'])
        assert all(x['end_s'] <= y['start_s'] for x,y in zip(ordered, ordered[1:]))
        cumulative, preempts = timeline(raw)
        arms[arm] = dict(extension_spans=spans, first_5s=cumulative(5),
            first_native_preemption_s=preempts[0],
            extended_unique_targets=len({s['request_id'] for s in spans}),
            extension_total_host_s=sum(s['duration_s'] for s in spans),
            extension_peer_outputs=sum(s['other_outputs_during_extension'] for s in spans),
            extend_growth_counts=dict(Counter(str(s['start_future_growth_blocks']) for s in spans)),
            extended_after_release_preemption=dict(Counter(s['next_preemption_observation'] for s in spans)))
    divergence, _ = first_sequence_divergence(raws['q1'], raws['q10'],
        arms['q1']['first_native_preemption_s'], arms['q10']['first_native_preemption_s'])
    output = dict(arms=arms, first_sequence_divergence=divergence, scope=[
        'Disjoint host intervals between extension and release; not exclusive GPU occupation or additive lost service.',
        'Peer output counts include concurrently running requests. Sparse captures do not reveal blocked waiting-loop opportunities.',
        'Complete paired trajectories differ before protection. No per-action causal effect or equal-work comparison is identified.'])
    path = ROOT / 'A_CAPACITY_PROTECTION_EXPOSURE_R03_20261001.json'
    with path.open('x') as handle:
        json.dump(output, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({k:{n:v for n,v in a.items() if n!='extension_spans'} for k,a in arms.items()}))
    print(json.dumps(divergence))


if __name__ == '__main__':
    main()
