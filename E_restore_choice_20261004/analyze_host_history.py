"""Read-only CPU analysis of a completed one-event Host-history observation.

Usage: python3 -B analyze_host_history.py GROUP [--stdout]
Normally writes GROUP/history_summary.json. Missing prerequisites/observations
are UNRUN; --stdout permits checking that gate without creating an artifact.
No GPU, cache query, timing fit, or re-execution of the base/one-event audits.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path


def read_source(path, hashes):
    data=path.read_bytes()
    hashes[str(path)]=hashlib.sha256(data).hexdigest()
    return json.loads(data)


def lookup_view(event, ends):
    if event is None:
        return None
    return dict(**{k:event.get(k) for k in ('event','decision_s','preemptions','known_tokens',
        'generated_tokens','host_hit_tokens','action','eligible','fallback')},
        step=bisect_left(ends,event['decision_s']))


def allocation_link(records, recovery, builder, before):
    """Follow the observer's latest allocation, including native continuation calls."""
    root=recovery['record_index']; stop=builder['record_index']
    request_id=recovery['before']['request_id']; preemptions=recovery['before']['preemptions']
    candidates=[r for r in records if r.get('operation')=='allocation' and
                r.get('before',{}).get('request_id')==request_id and root<=r['record_index']<stop]
    candidates.sort(key=lambda r:r['record_index'])
    issues=[]; linked=before.get('last_allocation')
    if before.get('request_id')!=request_id or before.get('preemptions')!=preemptions:
        issues.append('builder_request_or_preemption_mismatch')
    if not candidates or candidates[0]['record_index']!=root:
        issues.append('recovery_allocation_missing_from_chain')
    if len({r['record_index'] for r in candidates})!=len(candidates):
        issues.append('duplicate_allocation_record_index')
    for i,record in enumerate(candidates):
        if record.get('status')!='returned':
            issues.append('allocation_not_returned')
        if any(record.get(side,{}).get('request_id')!=request_id or
               record.get(side,{}).get('preemptions')!=preemptions for side in ('before','after')):
            issues.append('allocation_request_or_preemption_mismatch')
        if i and (record.get('lookup') is not None or record.get('external_tokens')!=0):
            issues.append('intermediate_allocation_not_native_continuation')
    if candidates:
        latest=candidates[-1]
        expected=dict(record_index=latest['record_index'],time_s=latest['time_s'],
                      preemptions=latest['before']['preemptions'],
                      external_tokens=latest.get('external_tokens'),lookup=latest.get('lookup'))
        if not isinstance(linked,dict) or not expected.keys()<=linked.keys() or any(linked[k]!=v for k,v in expected.items()):
            issues.append('last_allocation_not_latest_record_or_metadata_mismatch')
        times=[r['time_s'] for r in candidates]+[builder['time_s']]
        if times!=sorted(times):
            issues.append('allocation_builder_time_order_mismatch')
    else:
        latest=None
    return dict(valid=not issues,recovery_allocation_record_index=root,
                linked_allocation_record_index=linked.get('record_index') if isinstance(linked,dict) else None,
                allocation_record_indices=[r['record_index'] for r in candidates],
                intermediate_allocation_count=max(0,len(candidates)-1),issues=sorted(set(issues)))


def completion_view(records, job, creation, request_id, next_lookup):
    reports=[]
    for record in records:
        if record.get('operation')!='store_completion':
            continue
        for before in record.get('before',[]):
            if (before.get('job_id'),before.get('creation_record_index'),before.get('request_id')) != (
                    job['job_id'],creation,request_id):
                continue
            after=next((x for x in record.get('after',[]) if x['job_id']==job['job_id']),None)
            acknowledged=record.get('status')=='returned' and after is not None and (
                after.get('removed_by_native') is True or after.get('pending_count_after')==0)
            # time_s is callback ENTRY. An earlier completed service step supplies
            # ordering via the synchronous outer engine.step loop; same-step entry
            # times alone do not establish when the native update returned.
            earlier_step=next_lookup is not None and record.get('phase')=='service' and (
                isinstance(record.get('step_index'),int) and
                record['step_index']<next_lookup['step'] and record['time_s']<next_lookup['decision_s'])
            reports.append(dict(record_index=record['record_index'],phase=record.get('phase'),
                step=record.get('step_index'),entry_time_s=record['time_s'],status=record.get('status'),
                reported_count=before.get('reported_count'),present_before=before.get('present_before'),
                pending_count_before=before.get('pending_count_before'),after=after,
                native_zero_or_removed_observed=acknowledged,
                acknowledged_before_next_lookup=True if acknowledged and earlier_step else None))
    confirmed=any(x['acknowledged_before_next_lookup'] is True for x in reports)
    return dict(reports=reports,
        acknowledged_before_next_lookup=True if confirmed else None,
        assessment=('ACKNOWLEDGED_IN_EARLIER_SERVICE_STEP' if confirmed else
                    'NO_NEXT_LOOKUP' if next_lookup is None else
                    'ACKNOWLEDGED_BUT_ORDER_NOT_ESTABLISHED' if any(x['native_zero_or_removed_observed'] for x in reports) else
                    'PARTIAL_OR_INCOMPLETE_REPORTS' if reports else 'NO_COMPLETION_REPORT_OBSERVED'))


def analyze_arm(raw, external_id):
    observer=raw.get('host_history_observer')
    result=dict(enabled=observer.get('enabled') if isinstance(observer,dict) else None,
                observer_errors=observer.get('errors',[]) if isinstance(observer,dict) else [],
                issues=[],recoveries=[])
    if not isinstance(observer,dict) or observer.get('enabled') is not True:
        result.update(status='UNRUN',issues=['host_history_observer_missing_or_disabled'])
        return result
    records=observer.get('records',[]); issues=result['issues']
    if result['observer_errors']:
        issues.append('observer_errors_make_absence_and_coverage_incomplete')
    if not records:
        result.update(status='INCOMPLETE_OBSERVATION',issues=issues+['enabled_observer_has_no_records'])
        return result
    if any(r.get('status')!='returned' for r in records):
        issues.append('observer_has_nonreturned_records')
    if [r.get('record_index') for r in records]!=list(range(len(records))):
        issues.append('observer_record_indices_not_contiguous')
    requests=[r for r in raw['requests'] if r['external_id']==external_id]
    if len(requests)!=1:
        result.update(status='INCOMPLETE_OBSERVATION',issues=issues+['target_identity_not_unique'])
        return result
    request=requests[0]; request_id=request['request_id']
    ends=[s['end_s'] for s in raw['steps']]
    commits=sorted((c for c in raw['commits'] if c['request_id']==request_id),key=lambda c:c['allocation_s'])
    decisions=sorted((d for d in raw['decisions'] if d['request_id']==request_id),key=lambda d:d['decision_s'])
    allocations=[r for r in records if r.get('operation')=='allocation' and
                 r.get('before',{}).get('request_id')==request_id]
    result.update(target_request_id=request_id,total_successful_target_recoveries=len(commits),
                  observer_record_count=len(records))
    if len(commits)<3:
        issues.append('fewer_than_three_successful_target_recoveries')
    # Position accounting for this target only; no token contents or future-length model.
    native=[]; seen=set()
    for step,schedule in enumerate(raw['scheduler_steps']):
        for row in schedule['scheduled']:
            if row['request_id']!=request_id:
                continue
            start,end=row['start_computed'],row['end_computed']
            repeated=sum(p in seen for p in range(start,end));seen.update(range(start,end))
            native.append(dict(step=step,time_s=schedule['time_s'],start_computed=start,
                               end_computed=end,count=row['count'],repeated_positions=repeated))
    for i,commit in enumerate(commits[:3]):
        matches=[r for r in allocations if (r.get('lookup') or {}).get('event')==commit['event']]
        if len(matches)!=1:
            issues.append(f"event_{commit['event']}:allocation_record_not_unique")
            continue
        allocation=matches[0]
        if allocation.get('status')!='returned' or 'after' not in allocation:
            issues.append(f"event_{commit['event']}:allocation_not_returned")
            continue
        for key in ('preemptions','known_tokens','generated_tokens','host_hit_tokens','action'):
            if allocation['lookup'].get(key)!=commit.get(key):
                issues.append(f"event_{commit['event']}:lookup_{key}_mismatch")
        for key in ('preemptions','known_tokens','generated_tokens'):
            if allocation['before'].get(key)!=commit.get(key):
                issues.append(f"event_{commit['event']}:allocation_state_{key}_mismatch")
        if allocation.get('external_tokens')!=commit.get('external_tokens'):
            issues.append(f"event_{commit['event']}:external_tokens_mismatch")
        next_commit=commits[i+1] if i+1<len(commits) else None
        next_decision=next((d for d in decisions if d['decision_s']>commit['allocation_s']),None)
        next_lookup=lookup_view(next_decision,ends)
        stop_time=next_commit['allocation_s'] if next_commit else float('inf')
        next_allocations=([r for r in allocations if (r.get('lookup') or {}).get('event')==next_commit['event']]
                          if next_commit else [])
        stop_record=next_allocations[0]['record_index'] if len(next_allocations)==1 else len(records)
        if next_commit and len(next_allocations)!=1:
            issues.append(f"event_{commit['event']}:next_allocation_boundary_missing_or_ambiguous")
        groups=allocation['before']['groups']; after_groups=allocation['after']['groups']
        if len(groups)!=1 or len(after_groups)!=1 or groups[0]['group_index']!=after_groups[0]['group_index']:
            issues.append(f"event_{commit['event']}:unsupported_KV_group_layout")
            continue
        chunk_size=groups[0]['tokens_per_chunk']
        if not isinstance(chunk_size,int) or chunk_size<=0 or after_groups[0]['tokens_per_chunk']!=chunk_size:
            issues.append(f"event_{commit['event']}:invalid_chunk_size")
            continue
        before_cursor=groups[0]['next_stored_chunk_idx'];after_cursor=after_groups[0]['next_stored_chunk_idx']
        builders=[]; jobs=[]; no_jobs=[]
        for record in records:
            if record.get('operation')!='store_builder' or not (
                    allocation['record_index']<record['record_index']<stop_record and record['time_s']<stop_time):
                continue
            before=next((x for x in record.get('before',[]) if x['request_id']==request_id),None)
            if before is None:
                continue
            after=next((x for x in record.get('after',[]) if x['request_id']==request_id),None)
            view=dict(record_index=record['record_index'],step=record.get('step_index'),
                phase=record.get('phase'),entry_time_s=record['time_s'],status=record.get('status'),
                before=before,after=after,
                allocation_link=allocation_link(records,allocation,record,before))
            builders.append(view)
            if not view['allocation_link']['valid']:
                issues.extend(f"record_{record['record_index']}:{reason}" for reason in view['allocation_link']['issues'])
            if record.get('status')!='returned' or 'jobs' not in record:
                issues.append(f"record_{record['record_index']}:store_builder_incomplete")
                continue
            created=[j for j in record['jobs'] if j['request_id']==request_id]
            if not created:
                no_jobs.append(view)
            for job in created:
                valid=job.get('unmatched_key_count')==0 and len(job['groups'])==1
                ranges=[]; covered=set()
                for group in job['groups']:
                    valid=valid and group['group_index']==groups[0]['group_index']
                    indices=[]
                    for a,b in group['chunk_intervals']:
                        valid=valid and isinstance(a,int) and isinstance(b,int) and 0<=a<b
                        indices.extend(range(a,b));ranges.append([a,b])
                    valid=valid and len(indices)==group['chunk_count'] and len(set(indices))==len(indices)
                    covered.update(indices)
                valid=valid and len(covered)==job['key_count']
                if not valid:
                    issues.append(f"record_{record['record_index']}:job_{job['job_id']}_chunk_mapping_incomplete")
                jobs.append(dict(job_id=job['job_id'],creation_record_index=record['record_index'],
                    creation_step=record.get('step_index'),creation_entry_time_s=record['time_s'],
                    actual_chunk_intervals=ranges,actual_token_intervals=[[a*chunk_size,b*chunk_size] for a,b in ranges],
                    key_count=job['key_count'],unmatched_key_count=job.get('unmatched_key_count'),
                    mapping_complete=valid,builder_state=view,
                    completion=completion_view(records,job,record['record_index'],request_id,next_lookup)))
        times=request['token_times_s'];output_index=bisect_right(times,commit['decision_s'])
        next_output=times[output_index] if output_index<len(times) else None
        work=[n for n in native if commit['allocation_s']<=n['time_s']<stop_time]
        if work and not builders:
            issues.append(f"event_{commit['event']}:scheduled_work_without_store_builder_observation")
        result['recoveries'].append(dict(successful_recovery_ordinal=i+1,
            commit=dict(**lookup_view(commit,ends),actual_action=commit['actual_action'],
                        allocation_s=commit['allocation_s'],external_tokens=commit['external_tokens']),
            allocation_record_index=allocation['record_index'],allocation_before=allocation['before'],
            allocation_after=allocation['after'],tokens_per_chunk=chunk_size,
            cursor_before=before_cursor,cursor_after=after_cursor,
            rewound_chunk_interval=[after_cursor,before_cursor] if after_cursor<before_cursor else None,
            next_lookup=next_lookup,next_successful_lookup=lookup_view(next_commit,ends),
            next_allocation_s=next_commit['allocation_s'] if next_commit else None,
            store_builder_calls=len(builders),store_builder_status_counts=dict(Counter(x['status'] for x in builders)),
            returned_no_job_calls=len(no_jobs),first_no_job_builder=no_jobs[0] if no_jobs else None,
            last_no_job_builder=no_jobs[-1] if no_jobs else None,created_store_jobs=jobs,
            creation_observation=('JOBS_CREATED' if jobs else 'NO_JOBS_IN_OBSERVED_RETURNED_BUILDERS' if no_jobs else 'NO_COMPLETE_BUILDER_OBSERVATION'),
            next_output_index_zero_based=output_index if next_output is not None else None,
            next_output_step=bisect_left(ends,next_output) if next_output is not None else None,
            next_output_before_next_allocation=next_output is not None and next_output<stop_time,
            native_compute_through_next_output=[n for n in work if next_output is None or n['time_s']<=next_output],
            native_positions_between_allocations=sum(n['count'] for n in work),
            repeated_positions_between_allocations=sum(n['repeated_positions'] for n in work)))
    result['issues']=sorted(set(issues))
    result['status']='INCOMPLETE_OBSERVATION' if issues else 'COMPLETE_HISTORY_OBSERVATION'
    return result


def analyze(group):
    group=Path(group).resolve(); hashes={}
    result=dict(schema='E.host_history_analysis.v1',group=str(group),status='UNRUN',
        source_sha256=hashes,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),arms=[],
        limitations=['Reuses existing base and one-event verdicts; does not repeat their full audits.',
                     'Allocation before-state is after selector lookup advice; actual Host hit comes from the existing commit.',
                     'STORE builders may link to later zero-external-token native continuation allocations in the same request/preemption, rather than directly to the recovery commit allocation.',
                     'STORE windows end at the next target allocation; next lookup includes failed/uncommitted attempts.',
                     'Completion times are wrapper entries. Only returned acknowledgements in earlier service steps establish ordering via the synchronous engine.step loop; same-step ordering is unresolved.',
                     'Native job removal or zero pending count does not establish continued Host residency or eviction cause.',
                     'No cache query, inferred suppressive mechanism, latency correction, fitted cost, or performance benefit.'])
    required=[group/name for name in ('status.json','summary.json','one_event_summary.json')]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:
        result['missing_prerequisites']=missing;return result
    status,base,one=[read_source(p,hashes) for p in required]
    if status.get('status')!='COMPLETE':
        result['missing_prerequisites']=['group_not_COMPLETE'];return result
    if (base.get('group_status',{}).get('status')!='COMPLETE' or any(base.get(k) for k in
            ('invalid_cells','incomplete_cells','analysis_errors')) or one.get('validation_passed') is not True):
        result.update(status='INVALID_PREREQUISITES',issues=['base_or_one_event_not_passed']);return result
    if Path(base['group']).resolve()!=group or Path(one['group']).resolve()!=group:
        raise ValueError('base or one-event report belongs to a different group')
    target=one['frozen_target_spec']['target']['external_id'];result['target_external_id']=target
    audits={a['cell']:a for a in one['target_audits']}
    for service in base['cells']:
        name=service['cell'];folder=(group/name).resolve()
        if not folder.is_relative_to(group) or audits.get(name,{}).get('target_valid') is not True:
            raise ValueError('cell is outside group or missing accepted one-event audit')
        raw_path=folder/'raw.json';raw=read_source(raw_path,hashes)
        arm=analyze_arm(raw,target)
        provenance=folder.parent/'runtime_source_hashes.json'
        if provenance.is_file():
            sources=read_source(provenance,hashes)
            arm['executed_observer_source_hashes']={k:v for k,v in sources.items() if Path(k).name=='host_history_observer.py'}
        else:
            arm['executed_observer_source_hashes']={}
        if arm['enabled'] is True and not arm['executed_observer_source_hashes']:
            arm['issues'].append('executed_observer_source_hash_missing');arm['status']='INCOMPLETE_OBSERVATION'
        arm.update(cell=name,policy=service['policy'],raw_path=str(raw_path),raw_sha256=hashes[str(raw_path)])
        result['arms'].append(arm)
    states={a['status'] for a in result['arms']}
    result['status']=('COMPLETE_HISTORY_OBSERVATION' if states=={'COMPLETE_HISTORY_OBSERVATION'} else
                      'UNRUN' if not states or states=={'UNRUN'} else 'INCOMPLETE_OBSERVATION')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group',type=Path);parser.add_argument('--stdout',action='store_true')
    args=parser.parse_args()
    try:
        result=analyze(args.group)
    except (KeyError,TypeError,ValueError,OSError,IndexError) as exc:
        result=dict(schema='E.host_history_analysis.v1',status='INVALID_INPUTS',error=f'{type(exc).__name__}: {exc}')
    if args.stdout or not args.group.is_dir():
        print(json.dumps(result,indent=2,ensure_ascii=False))
    else:
        out=args.group/'history_summary.json';out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(dict(status=result['status'],out=str(out))))
    return 0 if result['status']=='COMPLETE_HISTORY_OBSERVATION' else 2 if result['status']=='UNRUN' else 1


if __name__=='__main__':
    raise SystemExit(main())
