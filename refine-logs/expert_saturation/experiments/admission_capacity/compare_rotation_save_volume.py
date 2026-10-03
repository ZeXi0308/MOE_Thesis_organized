"""Observed-trajectory work/byte inventory; no counterfactual time saving."""
import json
from pathlib import Path
from analyze_call_progress import analyze as call_progress

O=Path(__file__).resolve().parents[2]/'outputs/admission_capacity'


def main():
    out=O/'20260914_rotation_save_volume_r01';out.mkdir(exist_ok=False)
    old=json.loads((O/'20260914_kv_roundtrip_feasibility_r01/budget.json').read_text())
    rows=[]
    for block in (0,1):
        for arm in ('native','headroom','least_progress','most_output'):
            path=O/f'20260914_d6_strong_baselines_r01/readback/results/block{block}-d6-{arm}/raw.json'
            raw=json.loads(path.read_text());events=raw['preemption_events']
            assert len(events)==raw['actual_preemption_count']
            progress=call_progress(path)
            full_bytes=valid_bytes=0;largest_prefix={};prefixes=[]
            for event in events:
                s=event['victim_state'];computed=s['computed_tokens'];blocks=computed//16
                assert s['block_counts'][0]>=blocks
                rid=raw['internal_to_source'][event['victim_internal_request_id']]
                full_bytes+=blocks*2097152;valid_bytes+=computed*131072
                largest_prefix[rid]=max(largest_prefix.get(rid,0),blocks*2097152)
                prefixes.append(dict(step=event['attempted_step'],request=rid,tokens=blocks*16,
                                     computed_tail= computed-blocks*16))
            if arm=='least_progress':
                prior=next(x for x in old['results'] if x['block']==block)
                assert valid_bytes==prior['total_oneway_valid_bytes'] and len(events)==len(prior['events'])
            recompute=progress['phases'].get('recompute',dict(calls=0,returned_tokens=0,engine_seconds=0))
            rows.append(dict(block=block,arm=arm,observed_preemptions=len(events),
                             observed_recompute_tokens=sum(s['recompute_tokens'] for s in raw['scheduler_steps']),
                             observed_total_calls=len(raw['engine_steps']),observed_total_outputs=progress['returned_tokens'],
                             observed_recompute_calls=recompute['calls'],observed_outputs_during_recompute=recompute['returned_tokens'],
                             observed_recompute_call_seconds=recompute['engine_seconds'],
                             hypothetical_full_snapshot_store_bytes=full_bytes,
                             hypothetical_one_load_per_snapshot_bytes=full_bytes,
                             all_request_longest_logical_prefix_bytes=sum(largest_prefix.values()),
                             scope='Bytes are a workload inventory at observed preemption states, not executed offload. '
                                   'Full snapshot at each event; second byte field assumes one later load per snapshot. '
                                   'Longest-prefix sum assumes reusable per-request prefix chunks and no eviction; '
                                   'excludes metadata/staging and does not guarantee native cache behavior.',
                             prefixes=prefixes))
    result=dict(status='OBSERVED_TRAJECTORY_VOLUME_ONLY',cells=rows,
                reused_least_inventory='20260914_kv_roundtrip_feasibility_r01/budget.json: both39-event byte totals rechecked',
                interpretation='No subtraction of recompute spans, no multiplied microbenchmark speedup, '
                               'no future-route or action counterfactual. New policy changes later states and these bytes.')
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:print({k:v for k,v in r.items() if k not in ('prefixes','scope')})


if __name__=='__main__':main()
