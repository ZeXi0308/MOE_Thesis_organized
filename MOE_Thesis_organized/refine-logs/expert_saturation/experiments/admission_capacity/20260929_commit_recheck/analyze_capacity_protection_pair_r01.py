"""Analyze actual q1/q10 archives, including service after protection ends.

No GPU result is produced until real archive paths are supplied. Missing event
evidence remains unknown; complete-request metrics never use just protected windows.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
from evaluate_goodput import pair, summarize, quantile


def distribution(values):
    values=[x for x in values if x is not None]
    return dict(n=len(values),median=quantile(values,.5),p95=quantile(values,.95),max=max(values,default=None))


def analyze_arm(archive,quantity):
    raw=json.loads((archive/'raw.json').read_text())
    store=json.loads((archive/'selective-store.json').read_text())
    if store.get('recovery_min_outputs') not in (None,quantity):
        raise ValueError('Archive recovery_min_outputs differs from specified arm')
    for key,expected in [('capacity_victim',True),('fit_first_resume',False),('commit_recheck',False)]:
        if key in store and store[key] is not expected:
            raise ValueError(f'Unexpected {key} policy')
    requests={r['internal_request_id']:r for r in raw['requests']}
    if len(requests)!=len(raw['requests']):
        raise ValueError('Duplicate internal request identity')
    metrics=summarize(raw,128,180)
    origin=raw.get('measurement_origin_perf_counter_s')
    events=store.get('events')
    preemption_logging=isinstance(raw.get('preemption_events'),list)
    records=raw.get('preemption_events',[])
    actual=[p for p in records if p.get('original_preemption_called') is True
            and p.get('original_preemption_returned') is True]
    by_request=defaultdict(list)
    by_key={(p['engine_call_index'],p['internal_request_id']):p for p in actual}
    for p in sorted(actual,key=lambda x:x['method_entered_s']):
        by_request[p['internal_request_id']].append(p)
    forced={(e['step'],e['victim']):e for e in events or []
            if e.get('event')=='commit_check' and e.get('reason')=='READY'}
    capacity={(e['step'],e['new_victim']) for e in events or [] if e.get('event')=='capacity_victim_commit'}
    def stamp(event):
        t=event.get('host_perf_counter_s')
        return t-origin if isinstance(t,(int,float)) and isinstance(origin,(int,float)) else None
    def event_row(event):
        return dict(event,relative_time_s=stamp(event))
    def next_output(rid,t):
        if t is None or rid not in requests:
            return None
        times=requests[rid]['token_times_s'];i=bisect_right(times,t)
        return times[i] if i<len(times) else None
    def preemption_row(p):
        key=(p['engine_call_index'],p['internal_request_id'])
        t=next_output(p['internal_request_id'],p['method_entered_s'])
        return dict(step=p['engine_call_index'],time_s=p['method_entered_s'],
            output_count=p['last_returned_output_count'],
            cause=('capacity_replacement' if key in capacity else 'ordinary_forced') if key in forced
                else 'native_no_adapter_commit' if events is not None else 'unknown',
            first_later_output_s=t,preemption_to_later_output_s=t-p['method_entered_s'] if t is not None else None)
    episodes,pending,unmatched=[],{},[]
    kinds={'protection_start','protection_extend','protection_release','target_new_output',
           'target_terminal','recovery_commit_admitted'}
    for event in events or []:
        kind=event.get('event')
        if kind not in kinds:
            continue
        rid=event.get('request',event.get('target'))
        if kind=='protection_start':
            if rid in pending:
                pending[rid]['unclosed_reason']='ANOTHER_START_BEFORE_RELEASE'
            episode=dict(target_internal_id=rid,request_id=requests.get(rid,{}).get('request_id'),
                start=event_row(event),extend=None,first_output_event=None,terminal=None,
                admission=None,release=None,unknown_fields=[])
            episodes.append(episode);pending[rid]=episode
            continue
        episode=pending.get(rid)
        if episode is None:
            unmatched.append(event_row(event));continue
        field={'protection_extend':'extend','target_new_output':'first_output_event',
               'target_terminal':'terminal','recovery_commit_admitted':'admission','protection_release':'release'}[kind]
        if episode[field] is not None:
            episode['unknown_fields'].append('duplicate_'+field)
        episode[field]=event_row(event)
        if kind=='protection_release':
            pending.pop(rid)
    for episode in episodes:
        rid=episode['target_internal_id'];request=requests.get(rid)
        if request is None:
            episode['unknown_fields'].append('request_identity');continue
        start=episode['start'];admission=episode['admission'];release=episode['release']
        count=start.get('output_count');times=request['token_times_s']
        episode['first_actual_new_output_s']=times[count] if type(count) is int and 0<=count<len(times) else None
        receipt_ok=None
        if admission is not None:
            receipt_ok=(admission.get('native_admission')=='SCHEDULED_TOKENS'
                        and admission.get('scheduled_tokens',0)>0) or (
                admission.get('native_admission')=='ASYNC_LOAD_ADMITTED' and bool(admission.get('load_job_ids')))
        episode['native_admission_receipt_valid']=receipt_ok
        episode['matching_native_victim_preemption']=((admission['step'],admission.get('planned_victim')) in by_key
            if admission is not None and preemption_logging else None)
        episode['final_request']=dict(status=request['status'],completion_s=request.get('completion_s'),
            stop_reason=request.get('stop_reason'),output_tokens=len(request['output_token_ids']))
        if admission is None:
            episode['unknown_fields'].append('native_admission_receipt')
        if release is None or release['relative_time_s'] is None:
            episode['unknown_fields'].append('release_or_release_time')
            episode['after_release']=None;continue
        rt=release['relative_time_s']
        later=next((p for p in by_request[rid] if p['method_entered_s']>rt),None)
        completion=request.get('completion_s')
        gaps=[b-a for a,b in zip(times,times[1:]) if b>rt]
        episode['after_release']=dict(next_preemption=preemption_row(later) if later else None,
            next_preemption_observation='UNKNOWN' if not preemption_logging else 'OBSERVED' if later else
                'NONE_BEFORE_COMPLETION' if request['status']=='completed' else 'NONE_BEFORE_CAPTURE_END',
            all_later_preemptions=sum(p['method_entered_s']>rt for p in by_request[rid]) if preemption_logging else None,
            next_output_s=next_output(rid,rt),max_generation_gap_ending_after_release_s=max(gaps,default=None),
            completion_minus_release_s=completion-rt if completion is not None else None)
    short,all_intervals=[],[]
    for rid,preemptions in by_request.items():
        for before,after in zip(preemptions,preemptions[1:]):
            delta=after['last_returned_output_count']-before['last_returned_output_count']
            all_intervals.append(delta)
            if delta not in (1,2):
                continue
            first=next_output(rid,before['method_entered_s'])
            recoveries=[(key,e) for key,e in forced.items() if e['target']==rid and key in by_key
                and first is not None and before['method_entered_s']<by_key[key]['method_entered_s']<first]
            recovery=('capacity_replacement' if recoveries[0][0] in capacity else 'ordinary_rotation') if len(recoveries)==1 else (
                'no_adapter_recovery_observed' if not recoveries and events is not None else 'unknown')
            releases=[ep['release'] for ep in episodes if ep['target_internal_id']==rid and ep['release'] is not None
                and ep['release']['relative_time_s'] is not None and first is not None
                and first<=ep['release']['relative_time_s']<after['method_entered_s']]
            short.append(dict(request_id=requests[rid]['request_id'],new_outputs_between_preemptions=delta,
                earlier_preemption_step=before['engine_call_index'],later_preemption=preemption_row(after),
                recovery_kind=recovery,first_output_after_earlier_preemption_s=first,
                protection_releases_before_later_preemption=releases,
                final_status=requests[rid]['status'],final_completion_s=requests[rid].get('completion_s')))
    quantities=Counter(ep['release'].get('new_output_tokens') for ep in episodes if ep['release'])
    return dict(archive=str(archive),actual_quantity=store.get('recovery_min_outputs'),
        capture_status=raw.get('status'),capture_error=raw.get('error'),store_status=store.get('status'),
        metrics=metrics,protection_summary=dict(start_count=len(episodes),
            extended_count=sum(ep['extend'] is not None for ep in episodes),
            release_reasons=dict(Counter(ep['release'].get('reason','unknown') for ep in episodes if ep['release'])),
            release_actual_new_output_counts={str(k):v for k,v in quantities.items()},
            terminal_statuses=dict(Counter(ep['terminal'].get('status','unknown') for ep in episodes if ep['terminal'])),
            admission_kinds=dict(Counter(ep['admission'].get('native_admission','unknown') for ep in episodes if ep['admission'])),
            unclosed_episode_count=sum(ep['release'] is None for ep in episodes),
            logging_support='KNOWN' if events is not None and store.get('recovery_min_outputs')==quantity else 'UNKNOWN'),
        preemption_summary=dict(recorded_actual_count=raw.get('actual_preemption_count'),
            actual_successful_records=len(actual) if preemption_logging else None,
            records_with_unknown_success=sum('original_preemption_returned' not in p or 'original_preemption_called' not in p for p in records),
            consecutive_pairs=len(all_intervals) if preemption_logging else None,
            zero_output_pairs=sum(x==0 for x in all_intervals) if preemption_logging else None,
            short_one_or_two_output_pairs=len(short) if preemption_logging else None,
            distinct_short_pair_requests=len({r['request_id'] for r in short}) if preemption_logging else None,
            short_pair_later_causes=dict(Counter(r['later_preemption']['cause'] for r in short)),
            all_preemption_to_next_output_s=distribution([next_output(p['internal_request_id'],p['method_entered_s'])-p['method_entered_s']
                for p in actual if next_output(p['internal_request_id'],p['method_entered_s']) is not None])),
        episodes=episodes,short_service_pairs=short,unmatched_protection_events=unmatched),raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q1',type=Path,required=True,help='q1 archive directory')
    parser.add_argument('--q10',type=Path,required=True,help='q10 archive directory')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    arms,raws={},{}
    for label,path,quantity in [('q1',args.q1,1),('q10',args.q10,10)]:
        arms[label],raws[label]=analyze_arm(path,quantity)
    comparison=pair(arms['q1']['metrics'],arms['q10']['metrics'])
    requests={label:{r['request_id']:r for r in raw['requests']} for label,raw in raws.items()}
    differing=[rid for rid,r in requests['q1'].items() if r['output_token_ids']!=requests['q10'][rid]['output_token_ids']]
    stop_differing=[rid for rid,r in requests['q1'].items()
                   if r.get('stop_reason')!=requests['q10'][rid].get('stop_reason')]
    result=dict(status='COMPLETE_PAIR' if all(a['capture_status']=='COMPLETE' and a['capture_error'] is None
        and a['metrics']['completed']==128 for a in arms.values()) else 'INCOMPLETE_PAIR',arms=arms,
        full_cohort_comparison=comparison,output_sequence_difference_requests=differing,
        stop_reason_difference_requests=stop_differing,
        stop_reason_counts={label:dict(Counter(str(r.get('stop_reason','unknown'))
                            for r in rows.values())) for label,rows in requests.items()},
        scope=['All 128 requests, all preemptions and final completion are included; protection windows are not the performance denominator.',
            'Post-release followups can overlap across episodes and must not be summed as independent cost.',
            'FINISHED_STOPPED means EOS or another native stop; sparse records do not distinguish the exact EOS token.',
            'Missing event fields remain null/unknown; native async admission does not prove transfer completion.',
            'Independent trajectories do not identify action-level causal benefit or statistical repeatability.'])
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(status=result['status'],completed={k:v['metrics']['completed'] for k,v in arms.items()})))


if __name__=='__main__':
    main()
