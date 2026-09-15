"""Analyze actual repeated native execution; never infer results for UNRUN cells."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def analyze_arm(path):
    read=lambda n:json.loads((path/n).read_text())
    status=read('status.json')
    if status['status']!='COMPLETE':raise ValueError(f'Incomplete arm {path}: {status}')
    raw=read('raw.json');action=read('selective-store.json');transfer=read('offload-events.json')
    requests=raw['requests'];assert len(requests)==32 and len({r['request_id'] for r in requests})==32
    assert all(r['status']=='completed' and len(r['output_token_ids'])==1024 for r in requests)
    events=action['events'];prepares={e['step']:e for e in events if e['event']=='prepare'}
    commits=[e for e in events if e['event']=='commit_check' and e['reason']=='READY']
    assert action['applied_rotations']==len(commits)>=2 and action['status']=='DRAINED'
    for c in commits:
        p=prepares[c['step']-1];assert (p['victim'],p['target'])==(c['victim'],c['target'])
        actual=[e for e in raw['preemption_events'] if e['attempted_step']==c['step'] and e['victim_internal_request_id']==c['victim']]
        assert len(actual)==1
    releases=[e for e in events if e['event']=='target_new_output']
    assert len(releases)==len(commits)
    ids=raw['internal_to_source']
    # Directly validate that the protected request actually produced new output
    # before the wrapper released it; release is observed at next schedule start.
    for c,e in zip(commits,releases):
        assert c['target']==e['request'] and e['step']>c['step']
        before=raw['memory_trace'][c['step']]['before']['requests'][c['target']]['output_tokens']
        r=next(r for r in requests if r['request_id']==ids[c['target']])
        first=r['token_times_s'][before]
        call=next(x for x in raw['engine_steps'] if x['returned_s']==first)
        assert call['scheduler_step_start']<e['step']
    wall=max(r['completion_s'] for r in requests)-min(r['arrival_s'] for r in requests)
    phases={};last_end=None
    for call in raw['engine_steps']:
        if last_end is not None:assert call['start_s']>=last_end
        last_end=call['returned_s'];elapsed=last_end-call['start_s'];assert elapsed>=0
        sched=raw['scheduler_steps'][call['scheduler_step_start']:call['scheduler_step_end']]
        q=[q for s in sched for q in s['scheduled']]
        phase='prefill' if any(x['prefill_tokens'] for x in q) else 'recompute' if any(x['recompute_tokens'] for x in q) else 'decode' if q else 'no_scheduled_tokens'
        row=phases.setdefault(phase,dict(calls=0,engine_seconds=0.0));row['calls']+=1;row['engine_seconds']+=elapsed
    engine=sum(x['engine_seconds'] for x in phases.values());outside=wall-engine
    assert outside>=-1e-8
    byte_counts={k:sum(t[k]['bytes'] for t in transfer['transfers']) for k in ('store','load')}
    jobs=[j for e in transfer['completed_jobs'] for j in e['jobs']]
    if action['save']:assert byte_counts['store']>0 and byte_counts['load']>0
    else:assert byte_counts==dict(store=0,load=0) and not jobs
    return dict(mean_completion_s=statistics.mean(r['completion_s']-r['arrival_s'] for r in requests),wall_s=wall,
        max_itl_s=max(t-s for r in requests for s,t in zip(r['token_times_s'],r['token_times_s'][1:])),
        engine_seconds=engine,outside_engine_seconds=outside,phases=phases,applied_rotations=len(commits),
        cancellations=dict(Counter(e['reason'] for e in events if e['event']=='commit_check' and e['reason']!='READY')),
        completed_store_jobs=sum(j['is_store'] for j in jobs),completed_load_jobs=sum(not j['is_store'] for j in jobs),
        transfers_bytes=byte_counts,recomputed_tokens=sum(s['recompute_tokens'] for s in raw['scheduler_steps']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    rows={arm:analyze_arm(a.results/f'save-{arm}') for arm in ('off','on')}
    result=dict(status='REPEATED_EXECUTION_QUALIFIED',arms=rows,delta_pct={k:100*(rows['on'][k]/rows['off'][k]-1) for k in ['mean_completion_s','wall_s','max_itl_s']},
        scope='Actual repeated prepare/preempt/new-output events and all completed requests. n=1/arm fixed order; no stable timing benefit, bitwise KV/quality or strongest-immediate-baseline acceptance. Engine phases partition engine time; transfer device statistics are not additional subtractable exposed time.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
