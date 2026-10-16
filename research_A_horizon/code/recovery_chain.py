"""Join host observations by request and LOAD job, preserving missing boundaries."""
from bisect import bisect_right
from collections import defaultdict, Counter


def reconstruct(raw, horizon, offload):
    origin=raw['measurement_origin_perf_counter_s']
    end=origin+raw['observation_end_s']
    requests={r['internal_request_id']:r for r in raw['requests'] if 'internal_request_id' in r}
    preempts=defaultdict(list)
    for event in raw.get('preemption_events',[]):
        if event['original_preemption_returned']:
            preempts[event['internal_request_id']].append(origin+event['method_returned_s'])
    events=defaultdict(list)
    for event in horizon['events']:
        rid=event.get('request') or event.get('request_id_value')
        events[rid].append(event)
    loads=defaultdict(list);completed=defaultdict(list)
    for row in (offload or {}).get('dispatch',[]):
        if row['is_store'] is False and row['accepted'] is True:
            loads[row['request']].append(row)
    for row in (offload or {}).get('completed_jobs',[]):
        for job in row['jobs']:
            if job['is_store'] is False:
                completed[(job['request'],job['job_id'])].append(row['time_s'])
    episodes=[]
    for rid,times in preempts.items():
        times.sort()
        for index,start in enumerate(times):
            stop=times[index+1] if index+1<len(times) else end
            within=lambda t:start<=t<stop
            relevant=[e for e in events[rid] if within(e['host_perf_s'])]
            def first(kind):
                return next((e for e in relevant if e['kind']==kind),None)
            alloc=first('recovery_allocated')
            load_rows=[row for row in loads[rid] if within(row['before_perf_s'])]
            jobs=[]
            for row in load_rows:
                completions=[t for t in completed[(rid,row['job_id'])]
                             if row['after_perf_s']<=t<stop]
                jobs.append(dict(job_id=row['job_id'],submit_s=row['before_perf_s']-origin,
                    submitted_s=row['after_perf_s']-origin,
                    host_completion_s=min(completions)-origin if completions else None))
            ack=first('recovery_ack_applied');reentry=first('recovery_reentry_allocated')
            ts=requests[rid]['token_times_s']
            candidate=bisect_right(ts,(alloc['host_perf_s'] if alloc else start)-origin)
            output=ts[candidate] if candidate<len(ts) and ts[candidate]+origin<stop else None
            nodes=dict(preempt_s=start-origin,allocation_s=alloc['host_perf_s']-origin if alloc else None,
                load_submit_s=min((j['submit_s'] for j in jobs),default=None),
                load_host_complete_s=max((j['host_completion_s'] for j in jobs),default=None)
                    if jobs and all(j['host_completion_s'] is not None for j in jobs) else None,
                ack_applied_s=ack['host_perf_s']-origin if ack else None,
                reentry_allocation_s=reentry['host_perf_s']-origin if reentry else None,
                first_output_s=output)
            path='unallocated' if alloc is None else 'async_LOAD' if alloc['asynchronous'] else 'recompute'
            durations={}
            for a,b in [('preempt_s','allocation_s'),('allocation_s','load_submit_s'),
                        ('load_submit_s','load_host_complete_s'),('load_host_complete_s','ack_applied_s'),
                        ('ack_applied_s','reentry_allocation_s'),('reentry_allocation_s','first_output_s'),
                        ('allocation_s','first_output_s'),('preempt_s','first_output_s')]:
                if nodes[a] is not None and nodes[b] is not None:
                    assert nodes[b]>=nodes[a], (rid,a,b,nodes)
                    durations[a.removesuffix('_s')+'_to_'+b.removesuffix('_s')+'_s']=nodes[b]-nodes[a]
            episodes.append(dict(request=rid,index=index,path=path,nodes=nodes,load_jobs=jobs,
                durations=durations,first_output_observed=output is not None,
                next_preemption_s=stop-origin if index+1<len(times) else None,
                missing_boundaries=[k for k,v in nodes.items() if v is None]))
    return dict(episodes=episodes,paths=dict(Counter(e['path'] for e in episodes)),
        semantics='Single-process host perf_counter clock. Demand starts at observed preemption return, '
        'not unobserved allocation pressure. GPU allocation precedes LOAD submission; ACK may lead '
        'to a second allocation boundary. Job completion is host notification, not GPU finish. '
        'Missing ACK/reentry timestamps in older traces remain null. Aggregate durations overlap '
        'subintervals and must not be summed with them. Repeated preemptions bound separate episodes.')
