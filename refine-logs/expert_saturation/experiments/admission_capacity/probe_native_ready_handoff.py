"""Actual load notifications to first compute; never a transfer latency oracle."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dist(xs):return dict(min=min(xs),median=statistics.median(xs),max=max(xs))


def analyze(folder):
    raw,off,policy=(read(folder/n) for n in ('raw.json','offload-events.json','selective-store.json'))
    assert raw['status']=='COMPLETE' and off['diagnostic'] and policy['diagnostic']
    origin=raw['measurement_origin_perf_counter_s']; aliases=raw['internal_to_source']
    steps=raw['scheduler_steps']; calls=raw['engine_steps']; snapshots={s['step']:s for s in policy['eligibility_snapshots']}
    requests={r['request_id']:r for r in raw['requests']}
    loads=[e for e in off['dispatch'] if not e['is_store']]
    assert len({e['job_id'] for e in loads})==len(loads)
    rows=[]
    for event in loads:
        jid,rid=event['job_id'],event['request']; assert event['accepted'] is True
        found=[(entry,job) for entry in off['completed_jobs'] for job in entry['jobs'] if job['job_id']==jid]
        assert len(found)==1
        entry,job=found[0]; assert not job['is_store'] and job['request']==rid and job['count']==1
        notified=entry['time_s']-origin
        host_calls=[c for c in calls if c['start_s']<=notified<=c['returned_s']]
        assert len(host_calls)==1 and host_calls[0]['completed']
        callback_call=host_calls[0]; next_step=callback_call['scheduler_step_end']
        assert callback_call['scheduler_step_start']+1==next_step
        first=next(s for s in steps if s['start_s']>notified and any(q['internal_request_id']==rid for q in s['scheduled']))
        q=next(q for q in first['scheduled'] if q['internal_request_id']==rid)
        before=snapshots[first['step']]['requests'][rid]
        request=requests[aliases[rid]]; timestamp=request['token_times_s'][q['output_tokens_before']]
        compute_call=next(c for c in calls if c['scheduler_step_start']<=first['step']<c['scheduler_step_end'])
        pending=q['prompt_tokens']+q['output_tokens_before']-q['scheduled_start_computed']
        rows.append(dict(job_id=jid,request_id=aliases[rid],dispatch_s=event['before_perf_s']-origin,
            notification_callback_s=notified,notification_call=callback_call['call_index'],
            first_legal_next_step=next_step,first_compute_step=first['step'],
            missed_scheduler_opportunities=first['step']-next_step,
            begin_status=before['status'],begin_computed=before['computed'],begin_allocated_blocks=before['held_blocks'],
            protected_at_begin=snapshots[first['step']]['protected_id']==rid,
            actual_scheduled_start_computed=q['scheduled_start_computed'],actual_pending=pending,
            target_positions=q['scheduled_tokens'],new_output_same_compute_call=timestamp==compute_call['returned_s'],
            notification_to_schedule_start_ms=(first['start_s']-notified)*1000,
            dispatch_to_notification_ms=(notified-(event['before_perf_s']-origin))*1000))
    return dict(status='MEASUREMENT_ONLY',load_jobs=len(rows),
        jobs_next_schedule=sum(r['missed_scheduler_opportunities']==0 for r in rows),
        missed_scheduler_opportunities=dict(Counter(r['missed_scheduler_opportunities'] for r in rows)),
        begin_statuses=dict(Counter(r['begin_status'] for r in rows)),
        jobs_new_output_same_compute_call=sum(r['new_output_same_compute_call'] for r in rows),
        protected_jobs=sum(r['protected_at_begin'] for r in rows),
        pending_positions=dist([r['actual_pending'] for r in rows]),
        notification_to_schedule_start_ms=dist([r['notification_to_schedule_start_ms'] for r in rows]),
        dispatch_to_notification_ms=dist([r['dispatch_to_notification_ms'] for r in rows]),rows=rows,
        sources={str(folder/n):sha(folder/n) for n in ('raw.json','offload-events.json','selective-store.json')},
        source_sha256=sha(Path(__file__)),
        semantics='Completion is observed on entry to connector.update_connector_output, before its original body and '
            'scheduler finished-receiving insertion. It is not the physical CUDA completion timestamp. '
            'The next synchronous schedule is the first opportunity after this output processing. '
            'Begin WAIT_REMOTE can coexist with a finished notification; it is not proof of an unfinished transfer. '
            'All actual loaded jobs retained, including naturally resumed requests outside protected commits. '
            'Detailed diagnostic callback/snapshot costs remain; these milliseconds are not pure waiting or savings. '
            'No fixed future used for a counterfactual, no alternative action or performance claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cell',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(); assert not a.output.exists()
    result=analyze(a.cell);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','sources','semantics')}))


if __name__=='__main__':main()
