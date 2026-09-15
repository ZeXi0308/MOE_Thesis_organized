"""All retained repeat pairs; complete-request denominator, no drift correction."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def cell(path):
    if json.loads((path/'status.json').read_text())['status'] != 'COMPLETE':
        raise ValueError('Incomplete cell: '+str(path))
    raw=json.loads((path/'raw.json').read_text())
    action=json.loads((path/'selective-store.json').read_text())
    transfer=json.loads((path/'offload-events.json').read_text())
    reqs=raw['requests']
    assert len(reqs)==32 and all(q['status']=='completed' and len(q['output_token_ids'])==1024 for q in reqs)
    cut=next(c['start_s'] for c in raw['engine_steps'] if c['scheduler_step_start']==328)
    source=raw['internal_to_source'][action['selected']]
    victim=next(q for q in reqs if q['request_id']==source)
    pre=next(e for e in action['events'] if e['event']=='preempt')
    assert pre['step']==329
    selected=next(e for e in action['events'] if e['event']=='select')
    assert selected['step']==328 and selected['computed']==3305 and selected['save_cap']==3306
    metrics=dict(mean_completion_s=statistics.mean(q['completion_s']-q['arrival_s'] for q in reqs),
                 wall_s=max(q['completion_s'] for q in reqs)-min(q['arrival_s'] for q in reqs),
                 max_itl_s=max(b-a for q in reqs for a,b in zip(q['token_times_s'],q['token_times_s'][1:])),
                 victim_gap_s=victim['token_times_s'][pre['output']]-victim['token_times_s'][pre['output']-1],
                 pre_action_start_s=cut,
                 post_action_mean_s=statistics.mean(q['completion_s']-cut for q in reqs),
                 recompute_tokens=sum(s['recompute_tokens'] for s in raw['scheduler_steps']),
                 calls=len(raw['engine_steps']),
                 save_enabled=action['save'],
                 bytes={k:sum(t[k]['bytes'] for t in transfer['transfers']) for k in ('store','load')})
    jobs=[j for event in transfer['completed_jobs'] for j in event['jobs']]
    if action['save']:
        assert metrics['bytes']['store']==432013312 and metrics['bytes']['load']>=432013312
        assert any(j['request']==action['selected'] and j['is_store'] for j in jobs)
        assert any(j['request']==action['selected'] and not j['is_store'] for j in jobs)
    else:
        assert metrics['bytes']==dict(store=0,load=0)
    signature=[[(q['request_id'],q['scheduled_start_computed'],q['scheduled_tokens'],q['output_tokens_before'])
                for q in s['scheduled']] for s in raw['scheduler_steps']]
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    metrics['schedule_sha256']=digest(signature)
    metrics['output_sha256']=digest({q['request_id']:q['output_token_ids'] for q in reqs})
    durations=[c['returned_s']-c['start_s'] for c in raw['engine_steps']]
    metrics['call_timing']=dict(median_s=statistics.median(durations),mean_s=statistics.mean(durations),
                               sum_s=sum(durations),p95_s=sorted(durations)[int(.95*(len(durations)-1))],
                               max_s=max(durations),calls_over_100ms=sum(t>.1 for t in durations))
    return metrics,raw,signature[:328],source


def pair(off_path,on_path):
    off,a,pa,va=cell(off_path);on,b,pb,vb=cell(on_path)
    assert not off['save_enabled'] and on['save_enabled'], 'Arm identity mismatch'
    assert pa==pb and va==vb, 'Pre-action state/workload differs; do not silently pool'
    qa={q['request_id']:q for q in a['requests']};qb={q['request_id']:q for q in b['requests']}
    assert set(qa)==set(qb)
    assert all(qa[k]['arrival_s']==qb[k]['arrival_s'] and qa[k]['prompt_token_ids_sha256']==qb[k]['prompt_token_ids_sha256'] for k in qa)
    delta={k:qb[k]['completion_s']-qa[k]['completion_s'] for k in qa}
    return dict(off=off,on=on,
                delta_pct={k:100*(on[k]/off[k]-1) for k in ('mean_completion_s','wall_s','max_itl_s','victim_gap_s')},
                pre_action_drift_s=on['pre_action_start_s']-off['pre_action_start_s'],
                post_action_mean_difference_s=on['post_action_mean_s']-off['post_action_mean_s'],
                completion_difference_s=delta,slower_requests=sum(v>0 for v in delta.values()),
                equal_output_sequences=sum(qa[k]['output_token_ids']==qb[k]['output_token_ids'] for k in qa),
                pre_action_schedule_equal=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);args=p.parse_args()
    print(json.dumps(dict(blocks={str(i):pair(args.results/f'block{i}-off',args.results/f'block{i}-on') for i in range(2)},
                          scope='Two retained pairs, same fixed workload. No drift-corrected causal effect, significance or full-method claim.'),indent=2))
