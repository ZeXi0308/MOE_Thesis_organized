"""Qualify actual staged event chains and summarize retained full-request cost."""
import argparse
import json
import statistics
from pathlib import Path


def analyze(root):
    arms={};traces={}
    for arm in ('off','on'):
        p=root/f'save-{arm}'
        status=json.loads((p/'status.json').read_text())
        if status['status']!='COMPLETE':raise ValueError(f'{arm}: incomplete {status}')
        raw=json.loads((p/'raw.json').read_text())
        action=json.loads((p/'selective-store.json').read_text())
        trans=json.loads((p/'offload-events.json').read_text())
        reqs=raw['requests'];ids=raw['internal_to_source']
        assert len(reqs)==32 and len({q['request_id'] for q in reqs})==32
        assert all(q['status']=='completed' and len(q['output_token_ids'])==1024 and len(q['token_times_s'])==1024 for q in reqs)
        plans=[e for e in action['events'] if e['event']=='prepare']
        commits=[e for e in action['events'] if e['event']=='commit_check']
        assert len(plans)==len(commits)==1 and commits[0]['reason']=='READY' and action['preempted']
        plan=plans[0];step=commits[0]['step'];assert step==plan['step']+1
        assert action['save']==(arm=='on')
        pre=[e for e in raw['preemption_events'] if e['attempted_step']==step and e['victim_internal_request_id']==plan['victim']]
        assert len(pre)==1
        timeline={}
        for role in ('target','victim'):
            internal=plan[role];rid=ids[internal]
            q=next(q for q in reqs if q['request_id']==rid)
            before=raw['memory_trace'][step]['before']['requests'][internal]
            n=before['output_tokens'];first=q['token_times_s'][n]
            call=next(c for c in raw['engine_steps'] if c['returned_s']==first)
            timeline[role]=dict(request=rid,output_at_action=n,first_new_step=call['scheduler_step_start'],gap_s=first-q['token_times_s'][n-1],first_new_s=first)
        target_events=[e for e in action['events'] if e['event']=='target_new_output']
        assert len(target_events)==1 and target_events[0]['request']==plan['target']
        completed=[j for e in trans['completed_jobs'] for j in e['jobs']]
        victim_jobs=[j for j in completed if j['request']==plan['victim']]
        stores=[j for j in victim_jobs if j['is_store']];loads=[j for j in victim_jobs if not j['is_store']]
        byte_counts={k:sum(t[k]['bytes'] for t in trans['transfers']) for k in ('store','load')}
        metas=[e for e in action['events'] if e['event']=='metadata']
        if arm=='on':
            assert stores and loads and byte_counts['store']>0 and byte_counts['load']>0
            prepare_meta=next(e for e in metas if e['step']==plan['step'])
            commit_meta=next(e for e in metas if e['step']==step)
            assert prepare_meta['stores'] and set(prepare_meta['stores'])<=set(commit_meta['flush'])
        else:assert not stores and not loads and byte_counts==dict(store=0,load=0)
        arms[arm]=dict(timeline=timeline,bytes=byte_counts,completed_store_jobs=len(stores),completed_load_jobs=len(loads),
            mean_completion_s=statistics.mean(q['completion_s']-q['arrival_s'] for q in reqs),
            wall_s=max(q['completion_s'] for q in reqs)-min(q['arrival_s'] for q in reqs),
            max_itl_s=max(b-a for q in reqs for a,b in zip(q['token_times_s'],q['token_times_s'][1:])),
            calls=len(raw['engine_steps']),recompute_tokens=sum(s['recompute_tokens'] for s in raw['scheduler_steps']))
        traces[arm]=[[(q['request_id'],q['scheduled_start_computed'],q['scheduled_tokens'],q['output_tokens_before']) for q in s['scheduled']] for s in raw['scheduler_steps'][:plan['step']]]
    return dict(status='SINGLE_EVENT_EXECUTION_QUALIFIED',arms=arms,preparation_prefix_equal=traces['off']==traces['on'],
        delta_pct={k:100*(arms['on'][k]/arms['off'][k]-1) for k in ('mean_completion_s','wall_s','max_itl_s')},
        scope='One event per arm on new GPU; actual completed jobs and request outcomes. No bitwise restored-KV proof, direct physical fence timing, repeated net benefit or full repeated-rotation method claim.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=analyze(a.results)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
