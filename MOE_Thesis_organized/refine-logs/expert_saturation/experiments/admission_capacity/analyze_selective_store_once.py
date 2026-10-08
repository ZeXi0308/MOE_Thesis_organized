"""Actual single-victim store/load event chain and full retained-request cost."""
import json,statistics
from pathlib import Path

BASE=Path('refine-logs/expert_saturation/outputs/admission_capacity/20260914_selective_store_once_r01')
rows={};raws={}
for label in ['selective-off','selective-on']:
    p=BASE/'readback/results'/label
    assert json.loads((p/'status.json').read_text())['status']=='COMPLETE'
    r=json.loads((p/'raw.json').read_text());raws[label]=r
    action=json.loads((p/'selective-store.json').read_text())
    transfer=json.loads((p/'offload-events.json').read_text())
    victim=r['internal_to_source'][action['selected']]
    req=next(q for q in r['requests'] if q['request_id']==victim)
    assert len(r['requests'])==32 and all(q['status']=='completed' for q in r['requests'])
    pre=next(e for e in action['events'] if e['event']=='preempt')
    first_new=req['token_times_s'][pre['output']]
    call=next(c for c in r['engine_steps'] if c['returned_s']==first_new)
    scheduled=[dict(step=s['step'],computed=x['scheduled_start_computed'],tokens=x['scheduled_tokens'])
               for s in r['scheduler_steps'][329:call['scheduler_step_end']]
               for x in s['scheduled'] if x['request_id']==victim]
    event_time=next(c['start_s'] for c in r['engine_steps'] if c['scheduler_step_start']==329)
    rows[label]=dict(victim=victim,first_new_step=call['scheduler_step_start'],
        preempt_to_first_new_s=first_new-event_time,
        victim_gap_s=first_new-req['token_times_s'][pre['output']-1],
        recovery_calls=scheduled,bytes={k:sum(t[k]['bytes'] for t in transfer['transfers']) for k in ['load','store']},
        completed_jobs=transfer['completed_jobs'],
        mean_completion_s=statistics.mean(q['completion_s']-q['arrival_s'] for q in r['requests']),
        wall_s=max(q['completion_s'] for q in r['requests'])-min(q['arrival_s'] for q in r['requests']),
        max_itl_s=max(b-a for q in r['requests'] for a,b in zip(q['token_times_s'],q['token_times_s'][1:])),
        recompute_tokens=sum(s['recompute_tokens'] for s in r['scheduler_steps']),
        total_calls=len(r['engine_steps']),preemptions=r['actual_preemption_count'])
outputs=lambda r:{q['request_id']:q['output_token_ids'] for q in r['requests']}
same=outputs(raws['selective-off'])==outputs(raws['selective-on'])
def prefix(r):
    return [[(q['request_id'],q['scheduled_start_computed'],q['scheduled_tokens'],q['output_tokens_before'])
             for q in s['scheduled']] for s in r['scheduler_steps'][:329]]
result=dict(status='SINGLE_EVENT_NATIVE_INTERFACE_EVIDENCE',arms=rows,
    output_sequences_equal=same,pre_action_schedule_equal=prefix(raws['selective-off'])==prefix(raws['selective-on']),
    delta_pct={k:100*(rows['selective-on'][k]/rows['selective-off'][k]-1)
               for k in ['mean_completion_s','wall_s','max_itl_s','victim_gap_s']},
    scope='One run per arm; actual transfer and recovery, no repeated performance effect or bitwise KV comparison. Repeated loads of saved prefix included.')
(BASE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='arms'})
print({label:{k:v for k,v in arm.items() if k!='completed_jobs'} for label,arm in rows.items()})
