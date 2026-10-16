#!/usr/bin/env python3
"""Full-population metrics for HORIZON cells; incomplete cells stay explicit."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

B=Path(__file__).resolve().parents[2]/'B_recovery_order_20261004/pkg/metrics.py'


def distribution(values):
    values=sorted(values)
    def q(p):
        if not values:return None
        x=(len(values)-1)*p; lo=math.floor(x);hi=math.ceil(x)
        return values[lo]+(values[hi]-values[lo])*(x-lo)
    return dict(n=len(values),mean=sum(values)/len(values) if values else None,
                p50=q(.5),p95=q(.95),p99=q(.99),max=max(values) if values else None)


def analyze(cell):
    raw_path=cell/'raw.json'
    status=json.loads((cell/'status.json').read_text())
    if not raw_path.exists():return dict(cell=str(cell),status=status,performance=None)
    raw=json.loads(raw_path.read_text());end=raw['observation_end_s']
    rows=[r for r in raw['requests'] if r['arrival_s']<=end]
    if len({r['request_id'] for r in rows})!=len(rows):raise ValueError('Duplicate request IDs')
    horizon=json.loads((cell/'horizon.json').read_text())
    offload=json.loads((cell/'offload.json').read_text()) if (cell/'offload.json').exists() else None
    preempts=[e for e in raw.get('preemption_events',[]) if e['original_preemption_returned']]
    counts=Counter(r['status'] for r in rows)
    ttft=[];flow=[];gaps=[];per_request=[];joint=stalled=unknown=single=repeat=delivered=0
    for r in rows:
        ts=r['token_times_s'];tokens=r['output_token_ids']
        if len(ts)!=len(tokens) or ts!=sorted(ts):raise ValueError('Invalid token timeline')
        tt=ts[0]-r['arrival_s'] if ts else None
        f=r['completion_s']-r['arrival_s'] if r['completion_s'] is not None else None
        gap=max((b-a for a,b in zip(ts,ts[1:])),default=0.) if ts else None
        if tt is not None:ttft.append(tt)
        if f is not None:flow.append(f)
        if gap is not None:gaps.append(gap)
        single+=len(ts)==1
        unknown+=gap is None or r['status']!='completed'
        stalled+=gap is not None and gap>.2
        passed=r['status']=='completed' and tt is not None and tt<=5 and gap is not None and gap<=.2
        joint+=passed;delivered+=len(tokens)
        repetitive=len(tokens)>=512 and any(all(tokens[-512+i]==tokens[-512+i%p] for i in range(512)) for p in range(1,33))
        repeat+=repetitive
        per_request.append(dict(request_id=r['request_id'],status=r['status'],ttft_s=tt,flow_s=f,
            maxgap_s=gap,maxgap_is_lower_bound=r['status']!='completed',joint_pass=passed,
            output_tokens=len(tokens),finish_reason=r.get('stop_reason'),periodic_last512=repetitive))
    origin=raw['measurement_origin_perf_counter_s']
    by_id={r['internal_request_id']:r for r in rows if 'internal_request_id' in r}
    allocations={};thrash=0;episodes=[]
    for event in horizon['events']:
        rid=event.get('request')
        if event['kind']=='recovery_allocated':allocations[rid]=event
        if event['kind']=='preempted' and rid in allocations:
            alloc=allocations.pop(rid);ts=by_id[rid]['token_times_s']
            emitted=bisect_right(ts,event['host_perf_s']-origin)-bisect_right(ts,alloc['host_perf_s']-origin)
            thrash+=emitted<=16
            episodes.append(dict(request=rid,allocate_step=alloc['step'],repreempt_step=event['step'],
                                 outputs_before_repreempt=emitted,thrash=emitted<=16))
    n=len(rows);steps=horizon['counts'].get('schedule_calls',0)
    init=json.loads((cell/'init.json').read_text())
    result=dict(cell=str(cell),status=status,complete=raw['status']=='COMPLETE',
        n_planned=len(raw['requests']),n_arrived=n,n_not_arrived=len(raw['requests'])-n,
        completed=counts['completed'],failed=counts['failed'],unfinished=counts['unfinished'],
        ttft_s=distribution(ttft),flow_s=distribution(flow),maxgap_s=distribution(gaps),
        stall_rate=stalled/n if n else None,stall_count=stalled,unknown_or_censored_gap_count=unknown,
        single_token_vacuous_gap_count=single,joint_pass=joint,joint_goodput_rps=joint/end if end>0 else None,
        delivered_output_tokens=delivered,output_token_s=delivered/end if end>0 else None,
        observation_s=end,preemption_count=len(preempts),
        preempted_requests=len({e['request_id'] for e in preempts}),
        preempted_request_fraction=len({e['request_id'] for e in preempts})/n if n else None,
        thrash_count=thrash,thrash_episodes=episodes,
        periodic_last512_count=repeat,periodic_last512_fraction=repeat/n if n else None,
        finish_reasons=dict(Counter(r.get('stop_reason','unfinished') for r in rows)),
        scheduler_us_per_step=horizon['scheduler_seconds']*1e6/steps if steps else None,
        controller_seconds=horizon['controller_seconds'],ledger_seconds=horizon['ledger_seconds'],
        peer_committed_ratio=distribution([r['exact_peer_demand']/init['num_gpu_blocks'] for r in horizon['steps']]),
        peer_horizon_growth_ratio=distribution([(r['exact_peer_demand']-r['R']-r['I'])/init['num_gpu_blocks'] for r in horizon['steps']]),
        reservation_sampling='KNOWN waiting-gate attempts, not uniform scheduler steps; committed includes R/I, growth excludes them',
        actions=horizon['counts'],requests=per_request,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (raw_path,cell/'horizon.json',cell/'engine_args.json',cell/'inputs.json')},
        semantics='Linear-interpolated quantiles as B metrics; all arrived requests retained. Gaps exclude TTFT. Incomplete gaps are lower bounds; one-token completed requests have vacuous zero gap. Host receipt and host completion observations, not device completion.')
    if B.exists():
        spec=importlib.util.spec_from_file_location('B_metrics',B);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        result['inherited_B_metrics']=m.summarize_episode_requests(raw['requests'],observation_end_s=end,ttft_slo_s=5,tpot_slo_s=.2)
        assert result['ttft_s']['p95']==result['inherited_B_metrics']['latency_s']['ttft']['p95']
    from recovery_chain import reconstruct
    result['recovery_chains']=reconstruct(raw,horizon,offload)
    if offload is not None:
        result['offload_completed_job_records']=len(offload['completed_jobs'])
        result['offload_dispatch_count']=len(offload['dispatch'])
    if 'baseline' in horizon:
        baseline=horizon['baseline']
        result['baseline_fit_checks']=baseline['fit_checks']
        result['baseline_fit_mismatches']=baseline['fit_mismatches']
        if 'new_load_adaptation' in baseline:
            result['mc_load_activity']=baseline['new_load_adaptation']
        if 'mc_budget' in baseline:
            result['mc_activity']={k:baseline['mc_budget'][k] for k in
                ('attempted_evaluations','eligible_evaluations','successful_relaxations','guard_fallback_counts')}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True)
    args=p.parse_args();result=[analyze(c) for c in sorted(args.session.glob('cell-*')) if c.is_dir()]
    (args.session/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for r in result:
        print(json.dumps({k:r.get(k) for k in ('cell','complete','n_arrived','preempted_request_fraction','maxgap_s','ttft_s','output_token_s')}))
