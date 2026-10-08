"""Evaluate conditional futures after matched actual KV/request-state checks."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();load=lambda p:json.loads(p.read_text())
    specs=load(a.bundle/'STATUS.json');rows=[];requests={}
    for spec in specs['cells']:
        label=spec['label'];r=a.bundle/'readback/results'/label
        status=load(r/'status.json');state=load(r/'action-state.json')
        assert status['status']=='COMPLETE' and status['requests_completed']==32
        assert state['reference_match']==dict(logical_state=True,effective_kv=True)
        env=load(r/'environment.json')
        for name,h in env['source_sha256'].items():assert h==specs['files'][name]
        raw=load(r/'raw.json');ds=load(r/'headroom-decisions.json')
        d=next(x for x in ds if x['step']==329)
        cutoff=d['branch_action_start_perf_counter']-raw['host_clock_origin_perf_counter']
        forced=[x for x in ds if x['forced_preempted']]
        deferred=[x for x in ds if x.get('deferred_first_action')]
        action=spec['action'];expected=330 if action=='defer' else 329
        assert forced[0]['step']==expected
        assert len(deferred)==(1 if action=='defer' else 0)
        if deferred:assert deferred[0]['step']==329
        orders=[x['effective_victim_order'] for x in forced]
        assert orders==(['most_output']+['least_progress']*(len(orders)-1) if action=='most' else ['least_progress']*len(orders))
        target=state['state']['waiting'][0]
        events=[e for e in raw['output_events'] if e['external_request_id']==target and e['received_s']>=cutoff and e['chunk_size']>0]
        call=next(c for c in raw['engine_steps'] if c['returned_s']==events[0]['received_s'])
        assert call['scheduler_step_start']>=329 and target in call['output_request_ids']
        per={q['request_id']:q['completion_s']-cutoff for q in raw['requests']}
        assert len(per)==32 and all(v>0 for v in per.values())
        assert all(q['status']=='completed' and len(q['output_token_ids'])==1024 for q in raw['requests'])
        silence=[]
        for q in raw['requests']:
            times=[cutoff]+[t for t in q['token_times_s'] if t>=cutoff]
            silence.extend(v-u for u,v in zip(times,times[1:]))
        requests[label]=per
        rows.append(dict(label=label,block=spec['block'],action=action,cutoff_s=cutoff,
            mean_remaining_completion_s=statistics.mean(per.values()),last_remaining_completion_s=max(per.values()),
            max_post_action_silence_s=max(silence),
            target_first_token_wait_s=events[0]['received_s']-cutoff,
            target_first_token_calls=call['scheduler_step_end']-329,predicted_calls=5 if action=='defer' else 4,
            first_forced_step=forced[0]['step'],forced_exchanges=len(forced),total_calls=len(raw['engine_steps']),
            diagnostic_seconds=state['diagnostic_seconds'],raw_sha256=hashlib.sha256((r/'raw.json').read_bytes()).hexdigest()))
    by={r['label']:r for r in rows};pairs=[]
    for b in (0,1):
        n=by[f'block{b}-least']
        for action in ('most','defer'):
            r=by[f'block{b}-{action}']
            differences={rid:requests[r['label']][rid]-requests[n['label']][rid] for rid in requests[n['label']]}
            pairs.append(dict(block=b,target=action,baseline='least',
                mean_remaining_delta_s=r['mean_remaining_completion_s']-n['mean_remaining_completion_s'],
                mean_remaining_delta_pct=100*(r['mean_remaining_completion_s']/n['mean_remaining_completion_s']-1),
                last_remaining_delta_s=r['last_remaining_completion_s']-n['last_remaining_completion_s'],
                slower_requests=sum(v>0 for v in differences.values()),per_request_delta_s=differences))
    result=dict(status='MEASUREMENT_ONLY',cells=rows,pairs=pairs,
        scope='Conditional future after matched recorded requests and valid KV bytes; diagnostic instrumented, '
              'not uninstrumented end-to-end performance, full engine checkpoint, significance, Oracle or method GO.')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    for row in rows:print(row)
    for pair in pairs:print({k:v for k,v in pair.items() if k!='per_request_delta_s'})


if __name__=='__main__':main()
