#!/usr/bin/env python3
"""Describe realized spare-followup actor costs and the longest complete-request gaps."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from analyze_capacity_protection_pair_r01 import distribution


ROLES=('target','primary_target','primary_victim','queue_head','displaced_oldest')
COST_ROLES=('primary_target','primary_victim','queue_head','displaced_oldest')


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def metric_map(result,arm):
    rows=result['arms'][arm]['metrics']['requests']
    return {row['request_id']:row for row in rows}


def role_actor(action,role):
    if role=='target':
        return action['choice']['target'],action['final'],action['first_new_output_s'],None
    actor=action['actors'][role]
    return actor['internal_id'],actor['final'],actor['next_output_s'],actor['actual_later_preemptions']


def actor_summary(actions,role,off_metrics,on_metrics):
    rows=[]
    for action in actions:
        internal,final,next_output,preemptions=role_actor(action,role)
        if internal is None:continue
        rid=final['request_id'] if final else None
        choice=action['choice_s']
        rows.append(dict(step=action['choice']['step'],internal_id=internal,request_id=rid,
            next_output_after_choice_s=next_output-choice if next_output is not None else None,
            completion_after_choice_s=final['completion_s']-choice
                if final and final['completion_s'] is not None else None,
            final_status=final['status'] if final else None,
            later_actual_preemptions=preemptions,
            full_request_gap_s=on_metrics[rid]['max_gap_s'] if rid else None,
            full_request_flow_s=on_metrics[rid]['flow_s'] if rid else None,
            gap_difference_vs_off_s=(on_metrics[rid]['max_gap_s']-off_metrics[rid]['max_gap_s']
                if rid and on_metrics[rid]['max_gap_s'] is not None
                and off_metrics[rid]['max_gap_s'] is not None else None),
            flow_difference_vs_off_s=(on_metrics[rid]['flow_s']-off_metrics[rid]['flow_s'])
                if rid else None))
    unique={row['request_id'] for row in rows if row['request_id']}
    return dict(action_occurrences=len(rows),distinct_requests=len(unique),
        completed_occurrences=sum(row['final_status']=='completed' for row in rows),
        next_output_after_choice_s=distribution([row['next_output_after_choice_s'] for row in rows]),
        completion_after_choice_s=distribution([row['completion_after_choice_s'] for row in rows]),
        action_occurrences_with_later_preemption=sum(bool(row['later_actual_preemptions']) for row in rows
            if row['later_actual_preemptions'] is not None),
        later_actual_preemptions_sum=sum(len(row['later_actual_preemptions']) for row in rows
            if row['later_actual_preemptions'] is not None),
        unique_requests_with_worse_gap_vs_off=sum(on_metrics[rid]['max_gap_s'] is not None
            and off_metrics[rid]['max_gap_s'] is not None
            and on_metrics[rid]['max_gap_s']>off_metrics[rid]['max_gap_s'] for rid in unique),
        unique_requests_with_worse_flow_vs_off=sum(on_metrics[rid]['flow_s']>off_metrics[rid]['flow_s']
            for rid in unique),rows=rows)


def worst_gaps(raw,store,actions,off_metrics,on_metrics,limit=6):
    origin=raw['measurement_origin_perf_counter_s']
    ranked=[]
    for request in raw['requests']:
        times=request['token_times_s']
        if len(times)<2:continue
        gap,start,end,index=max((b-a,a,b,i+1) for i,(a,b) in enumerate(zip(times,times[1:])))
        ranked.append((gap,start,end,index,request))
    ranked.sort(key=lambda item:(item[0],item[4]['request_id']),reverse=True)
    events=store['events'];preempts=raw.get('preemption_events',[])
    rows=[]
    for gap,start,end,index,request in ranked[:limit]:
        internal=request['internal_request_id'];rid=request['request_id']
        observed_preempts=[dict(step=p['engine_call_index'],time_s=p['method_entered_s'],
            output_count=p.get('last_returned_output_count')) for p in preempts
            if p.get('internal_request_id')==internal and p.get('original_preemption_called') is True
            and p.get('original_preemption_returned') is True and start-.01<=p['method_entered_s']<=end]
        admission_events=[dict(event=e['event'],step=e['step'],time_s=e['host_perf_counter_s']-origin,
            kind=e.get('native_admission')) for e in events
            if ((e.get('event')=='recovery_commit_admitted' and e.get('target')==internal)
                or (e.get('event')=='spare_followup_admission' and e.get('target')==internal))
            and isinstance(e.get('host_perf_counter_s'),(int,float))
            and start<=e['host_perf_counter_s']-origin<=end]
        admission_events.sort(key=lambda e:e['time_s'])
        first_preempt=observed_preempts[0] if len(observed_preempts)==1 else None
        first_admission=next((e for e in admission_events if first_preempt is None
            or e['time_s']>=first_preempt['time_s']),None)
        forced=bool(first_preempt and any(e.get('event')=='commit_check'
            and e.get('step')==first_preempt['step'] and e.get('reason')=='READY'
            and e.get('victim')==internal for e in events))
        related=[]
        for action in actions:
            for role in ROLES:
                actor,_,_,_=role_actor(action,role)
                if actor==internal:
                    related.append(dict(role=role,choice_step=action['choice']['step'],
                        choice_time_s=action['choice_s']))
        pre_to_admission=(first_admission['time_s']-first_preempt['time_s']
            if first_preempt and first_admission else None)
        rows.append(dict(request_id=rid,internal_id=internal,gap_s=gap,
            gap_start_s=start,gap_end_s=end,token_index_after_gap=index,
            off_max_gap_s=off_metrics[rid]['max_gap_s'],on_max_gap_s=on_metrics[rid]['max_gap_s'],
            off_flow_s=off_metrics[rid]['flow_s'],on_flow_s=on_metrics[rid]['flow_s'],
            observed_actual_preemptions=observed_preempts,
            matching_regular_rotation_victim=forced,
            recorded_native_admissions=admission_events,
            preemption_to_first_recorded_admission_s=pre_to_admission,
            first_recorded_admission_to_output_s=end-first_admission['time_s']
                if first_admission else None,
            preemption_to_admission_fraction_of_gap=pre_to_admission/gap
                if pre_to_admission is not None else None,
            followup_choices_within_gap=sum(start<=a['choice_s']<=end for a in actions),
            own_followup_roles=related))
    if not rows:return dict(requests=[],union_window_s=None)
    lo=min(row['gap_start_s'] for row in rows);hi=max(row['gap_end_s'] for row in rows)
    return dict(requests=rows,union_window_s=[lo,hi],
        union_followup_choices=sum(lo<=a['choice_s']<=hi for a in actions),
        union_regular_protection_starts=sum(e.get('event')=='protection_start'
            and e.get('protection_origin')=='REGULAR_COMMIT'
            and isinstance(e.get('host_perf_counter_s'),(int,float))
            and lo<=e['host_perf_counter_s']-origin<=hi for e in events),
        union_regular_protection_releases=sum(e.get('event')=='protection_release'
            and e.get('protection_origin')=='REGULAR_COMMIT'
            and isinstance(e.get('host_perf_counter_s'),(int,float))
            and lo<=e['host_perf_counter_s']-origin<=hi for e in events))


def analyze(session,main_path):
    main=json.loads(main_path.read_text())
    archives=list(session.glob('cell-*-spare_followup_on/archive'))
    if len(archives)!=1:
        raise ValueError(f'Expected one on-arm archive, found {len(archives)}')
    archive=archives[0]
    raw=json.loads((archive/'raw.json').read_text())
    store=json.loads((archive/'selective-store.json').read_text())
    actions=main['followups']['actions']
    off=metric_map(main,'off');on=metric_map(main,'on')
    if len(off)!=128 or len(on)!=128 or set(off)!=set(on):
        raise ValueError('Complete-cohort request map differs')
    roles={role:actor_summary(actions,role,off,on) for role in COST_ROLES}
    old=main['arms']['off']['metrics'];new=main['arms']['on']['metrics']
    return dict(status='SOURCE_LOCALIZATION_COMPLETE' if main['status']=='COMPLETE_PAIR'
        else 'INCOMPLETE_PAIR',source=dict(main_result=str(main_path),main_result_sha256=sha256(main_path),
        on_raw=str(archive/'raw.json'),on_raw_sha256=sha256(archive/'raw.json'),
        on_selective_store=str(archive/'selective-store.json'),
        on_selective_store_sha256=sha256(archive/'selective-store.json')),
        full_cohort=dict(completed_off=old['completed'],completed_on=new['completed'],
            output_rate_ratio=new['actual_output_tokens_s']/old['actual_output_tokens_s'],
            mean_flow_ratio=new['mean_flow_with_incomplete_penalty_s']/old['mean_flow_with_incomplete_penalty_s'],
            max_gap_off_s=old['max_gap_request_max_s'],max_gap_on_s=new['max_gap_request_max_s'],
            predeclared_criteria=main['predeclared_exploration_criteria'],
            all_criteria_met=main['all_exploration_criteria_met']),
        followup=dict(choices=main['followups']['choices'],
            actual_output_completion_chains=main['followups']['actual_output_completion_chains'],
            admission_kinds=main['followups']['admission_kinds'],
            roles=roles),worst_on_gaps=worst_gaps(raw,store,actions,off,on),
        interpretation=[
            'All actor results use complete requests; action-defined groups overlap and are not additive causal costs.',
            'Recorded preemption-to-native-admission time is observable, but sparse logs do not prove its entire duration is queue waiting.',
            'A missing native admission receipt is unknown, not evidence of no native service. This ordered seen-input pair is not a same-state counterfactual.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,
        default=Path('moe-a-spare-followup-session-r01-20261001'))
    parser.add_argument('--result','--pair-result',dest='pair_result',type=Path,
        default=Path('A_SPARE_FOLLOWUP_PAIR_RESULT_R01_20261001.json'))
    parser.add_argument('--output',type=Path,
        default=Path('A_SPARE_FOLLOWUP_TARGET_PEER_R01_20261001.json'))
    args=parser.parse_args()
    report=analyze(args.session,args.pair_result)
    with args.output.open('x') as output:
        json.dump(report,output,indent=2,allow_nan=False);output.write('\n')
    print(json.dumps(dict(status=report['status'],cohort=report['full_cohort'],
        choices=report['followup']['choices'])))


if __name__=='__main__':main()
