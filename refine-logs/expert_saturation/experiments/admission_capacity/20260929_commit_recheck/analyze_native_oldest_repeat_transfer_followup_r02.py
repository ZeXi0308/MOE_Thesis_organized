#!/usr/bin/env python3
"""Describe observable recovery intervals in the completed repeated fund arm."""
import argparse
import json
import statistics
from pathlib import Path

from analyze_printed_transfer_intervals_r01 import analyze_log


def distribution(values):
    values=sorted(values)
    assert values
    def quantile(p):
        index=(len(values)-1)*p
        low=int(index);high=min(low+1,len(values)-1)
        return values[low]+(values[high]-values[low])*(index-low)
    return dict(n=len(values),min=values[0],median=statistics.median(values),
                p90=quantile(.90),p95=quantile(.95),max=values[-1])


def analyze(session):
    archive=session/'cell-00-queue_fund'/'archive'
    raw=json.loads((archive/'raw.json').read_text())
    store=json.loads((archive/'selective-store.json').read_text())
    assert raw['status']=='COMPLETE' and store['status']=='DRAINED'
    assert store['oldest_episode_count']==58 and store['oldest_retired_count']==58
    requests={r['internal_request_id']:r for r in raw['requests']}
    events={}
    for event in store['events']:
        if event.get('episode_id') and event['event'].startswith('oldest_'):
            events.setdefault(event['episode_id'],{})[event['event']]=event
    origin=raw['measurement_origin_perf_counter_s']
    stages={key:[] for key in ('anchor_to_native_admission_s',
        'native_admission_to_running_s','running_to_first_output_observed_s',
        'native_admission_to_first_output_observed_s')}
    victim_pauses=[];rows=[]
    for episode_id,record in sorted(events.items()):
        anchor=record['oldest_anchor'];admission=record['oldest_native_admission']
        output=record['oldest_target_new_output'];commit=record['oldest_commit']
        assert admission['target']==output['target']==anchor['target']
        assert record['oldest_retire']['reason']=='FIRST_NEW_OUTPUT'
        assert admission['native_admission']=='ASYNC_LOAD_ADMITTED'
        running=[entry for entry in store['residency_admissions']
                 if entry['request']==anchor['target']
                 and admission['host_perf_counter_s']<=entry['host_perf_counter_s']
                 <=output['host_perf_counter_s']]
        assert len(running)==1,(episode_id,len(running))
        running=running[0]
        intervals=dict(anchor_to_native_admission_s=
            admission['host_perf_counter_s']-anchor['host_perf_counter_s'],
            native_admission_to_running_s=
            running['host_perf_counter_s']-admission['host_perf_counter_s'],
            running_to_first_output_observed_s=
            output['host_perf_counter_s']-running['host_perf_counter_s'],
            native_admission_to_first_output_observed_s=
            output['host_perf_counter_s']-admission['host_perf_counter_s'])
        assert all(value>=0 for value in intervals.values())
        for key,value in intervals.items():stages[key].append(value)
        target=requests[anchor['target']]
        first_count=output['output_count']
        assert first_count>anchor['target_output_count']
        row=dict(episode_id=episode_id,anchor_step=anchor['step'],
            target_source_id=target['request_id'],forced=commit['forced'],
            native_admission_kind=admission['native_admission'],
            native_admission_scheduled_tokens=admission['scheduled_tokens'],
            load_job_ids=admission['load_job_ids'],
            anchor_host_s=anchor['host_perf_counter_s']-origin,
            native_admission_host_s=admission['host_perf_counter_s']-origin,
            running_host_s=running['host_perf_counter_s']-origin,
            first_output_observed_host_s=output['host_perf_counter_s']-origin,
            first_new_output_raw_client_s=target['token_times_s'][first_count-1],
            host_intervals=intervals)
        if commit['forced']:
            victim_id=commit['victim'];victim=requests[victim_id]
            preemptions=[p for p in raw['preemption_events']
                if p['internal_request_id']==victim_id
                and p['engine_call_index']==commit['step']]
            assert len(preemptions)==1,(episode_id,len(preemptions))
            preempt=preemptions[0];count=preempt['last_returned_output_count']
            next_output=victim['token_times_s'][count]
            pause=next_output-preempt['last_new_output_s']
            victim_pauses.append(pause)
            row['victim']=dict(source_id=victim['request_id'],
                raw_preemption_s=preempt['method_entered_s'],
                last_output_before_preemption_s=preempt['last_new_output_s'],
                first_output_after_preemption_s=next_output,
                last_to_next_output_gap_s=pause,
                completed=victim['status']=='completed')
        rows.append(row)
    transfer=analyze_log(session/'cell-00-queue_fund'/'launch.log')
    return dict(status='OBSERVED_TRANSFER_FOLLOWUP',session=str(session),
        arm='queue_fund',episodes=len(rows),forced_episodes=len(victim_pauses),
        async_load_admitted=sum(row['native_admission_kind']=='ASYNC_LOAD_ADMITTED'
                                for row in rows),
        same_step_scheduled_tokens=sum(row['native_admission_scheduled_tokens']
                                       for row in rows),
        host_stage_distributions={key:distribution(values) for key,values in stages.items()},
        raw_victim_last_to_next_output_gap_s=distribution(victim_pauses),
        episodes_detail=rows,
        printed_transfer_intervals=transfer,
        known_limits=[
            'Host stage durations subtract only time.perf_counter values in selective-store; raw victim output gaps subtract only raw measurement-clock values.',
            'ASYNC_LOAD_ADMITTED records native job acceptance, not DMA completion; residency_admissions marks the later running admission after asynchronous load processing.',
            'The admission-to-running interval includes load completion, scheduler waiting and possible recomputation. Per-job copied bytes, completed transfer duration, cache coverage and per-request recomputed tokens were not recorded.',
            'Printed KV bytes and CUDA event times are reset interval sums across all requests. The first possibly warmup-mixed print is excluded and the unprinted tail is unknown; these sums cannot be assigned to the 58 episodes.',
            'CUDA event copy time starts after preceding stream waits and is not request-visible pause. No overlap fraction or causal incremental victim cost is identifiable from these records.'
        ],
        minimum_future_observation='For per-episode transfer attribution, record load job_id, request_id, copied bytes, completed host timestamp, native cached-prefix tokens and recomputed scheduled tokens; retain host timestamps for first resumed running and output. Avoid extra GPU experiments solely to fill this gap.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--session',type=Path,default=Path('moe-a-native-oldest-repeat-session-r02-20261002'))
    parser.add_argument('--output',type=Path,default=Path('A_NATIVE_OLDEST_REPEAT_TRANSFER_FOLLOWUP_R02_20261002.json'))
    args=parser.parse_args()
    result=analyze(args.session)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','episodes','forced_episodes','async_load_admitted')}))
