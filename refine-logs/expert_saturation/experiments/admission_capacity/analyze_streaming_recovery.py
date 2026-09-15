"""Descriptive EOS/open-population accounting; no threshold-selected frontier."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import analyze_effective_recovery_service as life

LABELS = ['stream-block0-native', 'stream-block0-most_output', 'stream-block1-most_output', 'stream-block1-native']
check, read = life.check, life.read


def mean(values):
    return statistics.mean(values) if values else None


def request_metrics(raw, workload, eos):
    expected = {r['request_id']: a for r, a in zip(workload['source_requests'], workload['arrival_traces_s']['steady'])}
    sources = {r['request_id']: r for r in workload['source_requests']}
    return_times = {c['returned_s'] for c in raw['engine_steps']}
    requests = {r['request_id']: r for r in raw['requests']}
    check(len(requests) == len(raw['requests']) == len(expected) and requests.keys() == expected.keys(), 'request identities differ')
    observed, terminals = {rid: [] for rid in requests}, {}
    for e in raw['output_events']:
        rid = e['request_id']; check(rid in observed and rid not in terminals and e['prefix_valid'], 'invalid output identity/prefix/terminal')
        check(e['received_s'] in return_times, 'output event lacks synchronous engine return')
        check(e['chunk_size'] == len(e['new_token_ids']) and e['cumulative_tokens'] == len(observed[rid])+e['chunk_size'], 'output volume does not conserve')
        observed[rid].extend([e['received_s']]*e['chunk_size'])
        if e['finished']: terminals[rid] = e
    rows = []
    for rid, q in requests.items():
        ts, ids = q['token_times_s'], q['output_token_ids']; n = len(ids); event = terminals.get(rid)
        cap = q['max_output_tokens']; end = q['completion_s']; reason = q['stop_reason']
        check(cap == sources[rid]['max_output_tokens'] and q['prompt_tokens'] == sources[rid]['prompt_token_count']
            and q['prompt_token_ids_sha256'] == sources[rid]['prompt_token_ids_sha256'], 'prompt identity or output cap differs')
        check(q['status'] == 'completed' and event is not None and q['arrival_s'] == expected[rid], 'missing terminal/arrival')
        check(0 <= n <= cap and len(ts) == n and ts == sorted(ts) == observed[rid], 'output times/volume differ')
        check(event['received_s'] == end and event['finish_reason'] == reason and event['cumulative_token_ids'] == ids, 'terminal event differs')
        check(reason in ('stop', 'length') and (reason != 'length' or n == cap), 'EOS/max-length contract violated')
        check(expected[rid] <= q['admission_s'] <= end <= raw['observation_end_s'] and all(q['admission_s'] <= t <= end for t in ts), 'clock order differs')
        rows.append(dict(request_id=rid, arrival_s=q['arrival_s'], terminal_s=end, output_tokens=n,
            completion_latency_s=end-q['arrival_s'], ttft_s=ts[0]-q['arrival_s'] if ts else None,
            max_engine_return_gap_s=max((b-a for a,b in zip(ts,ts[1:])), default=None),
            last_output_s=ts[-1] if ts else None, terminal_after_last_output_s=end-ts[-1] if ts else None,
            finish_reason=reason, native_stop_reason=q.get('native_stop_reason'),
            output_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
    wall=raw['observation_end_s'];check(wall>0,'nonpositive wall')
    eos_ids=[]
    for value in (eos.get('hf_eos_token_id'), (eos.get('generation_config') or {}).get('eos_token_id'),
                  (eos.get('generation_overrides') or {}).get('eos_token_id')):
        eos_ids.extend(value if isinstance(value,list) else [] if value is None else [value])
    stops=[r for r in rows if r['finish_reason']=='stop'];gaps=[r['max_engine_return_gap_s'] for r in rows if r['max_engine_return_gap_s'] is not None]
    return dict(requests=len(rows),completed_requests=len(rows),full_wall_s=wall,
        requests_per_s=len(rows)/wall,output_tokens=sum(r['output_tokens'] for r in rows),
        output_tokens_per_s=sum(r['output_tokens'] for r in rows)/wall,
        mean_completion_s=mean([r['completion_latency_s'] for r in rows]),
        mean_ttft_s=mean([r['ttft_s'] for r in rows if r['ttft_s'] is not None]),
        ttft_defined_requests=sum(r['ttft_s'] is not None for r in rows),
        max_engine_return_gap_s=max(gaps,default=None),gap_defined_requests=len(gaps),
        finish_reason_counts=dict(Counter(r['finish_reason'] for r in rows)),
        output_count_histogram=dict(Counter(str(r['output_tokens']) for r in rows)),
        eos_token_reason_confirmed=sum(isinstance(r['native_stop_reason'],int) and r['native_stop_reason'] in eos_ids for r in stops),
        stop_without_explicit_token_reason=sum(r['native_stop_reason'] is None for r in stops),
        eos_boundary='stop counts are observed; only explicit native token reasons matching model EOS IDs are token-confirmed. No-token-reason stops are not silently relabeled.',
        arrival_span_s=max(expected.values())-min(expected.values()),
        arrivals_while_prior_request_unfinished=sum(any(p['arrival_s']<r['arrival_s']<p['terminal_s'] for p in rows) for r in rows),
        arrivals_inside_engine_calls=sum(any(c['start_s']<r['arrival_s']<c['returned_s'] for c in raw['engine_steps']) for r in rows),
        per_request=rows)


def adapter_coverage(raw, decisions, variant):
    steps=raw['scheduler_steps'];check(len(decisions)==len(steps),'adapter decision coverage differs')
    counts=Counter();receipts=Counter((e['attempted_step'],e['victim_internal_request_id']) for e in raw['preemption_events'])
    recorded=Counter()
    for k,(d,s) in enumerate(zip(decisions,steps)):
        check(d['step']==s['step']==k and d['status']=='APPLIED' and d['active'] is True and d['population_mode']=='open','open adapter bypassed or incomplete')
        check(d['mode']==('native' if variant=='native' else 'rotate'),'adapter mode differs')
        check(d['actual_scheduled']=={x['internal_request_id']:x['scheduled_tokens'] for x in s['scheduled']},'actual dispatch receipt differs')
        combined=d['forced_preempted']+d['natural_preempted'];check(len(combined)==len(set(combined)) and set(combined)==set(d['preempted']),'preemption class overlap')
        recorded.update((k,rid) for rid in combined)
        counts.update(active_steps=1,active_below32=int(s['running_before']<32),forced_preemptions=len(d['forced_preempted']),natural_preemptions=len(d['natural_preempted']))
        if d.get('proposal'): counts['proposal_'+d['proposal']['action']]+=1
        if d.get('recovery_released'): counts['release_'+d['recovery_released']['reason']]+=1
    check(recorded==receipts,'adapter and native preemption receipts differ')
    if variant=='native':check(not counts['forced_preemptions'],'native arm forced an eviction')
    return dict(counts)


def analyze(raw, workload, decisions, eos, variant):
    check(raw['status']=='COMPLETE' and raw['error'] is None and raw.get('output_mode')=='eos','complete EOS capture required')
    metrics=request_metrics(raw,workload,eos);coverage=adapter_coverage(raw,decisions,variant)
    lifecycle=life.analyze(raw)
    # A terminal EOS without a new output is complete, never a censored interval.
    for r in lifecycle['residencies']:
        q=next(q for q in metrics['per_request'] if q['request_id']==r['request_id'])
        r['terminal_s']=q['terminal_s'] if r['end_reason']=='completed' else None
        if r['resumed'] and r['new_outputs']==0 and r['end_reason']=='completed':
            r['outcome_bucket']='completed_without_new_output'
            for suffix,amount in [('_residencies',1),('_recompute_positions',r['recompute_positions'])]:
                lifecycle['summary']['censored_no_output'+suffix]-=amount
                key='completed_without_new_output'+suffix;lifecycle['summary'][key]=lifecycle['summary'].get(key,0)+amount
    lifecycle['service_classes']=dict(Counter(('repreempted_' if r['end_reason']=='repreempted' else 'terminal_')+
        ('zero' if r['new_outputs']==0 else 'one_two' if r['new_outputs']<=2 else 'three_plus') for r in lifecycle['residencies']))
    scheduler=sum(s['end_s']-s['start_s'] for s in raw['scheduler_steps']);engine=sum(c['returned_s']-c['start_s'] for c in raw['engine_steps'])
    check(0<=scheduler<=engine<=metrics['full_wall_s']+1e-6,'wall buckets do not conserve')
    return dict(status='MEASUREMENT_ONLY',metrics=metrics,adapter_coverage=coverage,lifecycle=lifecycle,
        cost_buckets_s=dict(scheduler_inclusive=scheduler,engine_excluding_scheduler=engine-scheduler,outside_engine=metrics['full_wall_s']-engine),resolved_eos=eos)


def compare(a,b):
    x={r['request_id']:r for r in a['metrics']['per_request']};y={r['request_id']:r for r in b['metrics']['per_request']}
    check(x.keys()==y.keys(),'pair identities differ')
    return dict(left=a['label'],right=b['label'],same_full_output=sum(x[r]['output_sha256']==y[r]['output_sha256'] for r in x),
        per_request=[dict(request_id=r,output_token_delta=y[r]['output_tokens']-x[r]['output_tokens'],
            completion_delta_s=y[r]['completion_latency_s']-x[r]['completion_latency_s'],
            left_gap_s=x[r]['max_engine_return_gap_s'],right_gap_s=y[r]['max_engine_return_gap_s']) for r in x],
        boundary='Different EOS/lengths and token trajectories remain outcomes, not output-quality or equal-work speedup claims.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();check(not args.output_dir.exists(),'retain prior analysis')
    workload=read(args.inputs/'workload.json');check(len(workload['source_requests'])==64,'frozen64 inputs required')
    cells=[]
    for label in LABELS:
        folder=args.results_root/label;row=dict(label=label,status='UNRUN');raw=None
        try:
            row['runtime_status']=read(folder/'status.json')
            path=folder/'raw.json';path=path if path.exists() else folder/'raw.json.gz';raw=read(path)
            cfg=read(folder/'config.json');check(cfg['population_mode']=='open' and cfg['output_mode']=='eos','runner mode differs')
            row.update(analyze(raw,workload,read(folder/'headroom-decisions.json'),read(folder/'resolved-eos.json'),cfg['variant']))
            row['raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            check(row['runtime_status']['status']=='COMPLETE','runner completion failed after capture')
        except (OSError,ValueError,KeyError,TypeError,AssertionError) as error:
            row.update(status='INCOMPLETE' if raw is not None or folder.exists() else 'UNRUN',error=f'{type(error).__name__}: {error}')
            if raw: row['retained_partial_counts']=dict(requests=len(raw.get('requests',[])),output_tokens=sum(len(q.get('output_token_ids',[])) for q in raw.get('requests',[])),raw_status=raw.get('status'))
        cells.append(row)
    complete=all(r['status']=='MEASUREMENT_ONLY' for r in cells)
    result=dict(status='MEASUREMENT_ONLY' if complete else 'INCOMPLETE',cells=cells,
        comparisons=[compare(cells[0],cells[1]),compare(cells[3],cells[2])] if complete else [],
        boundary='Native in-process engine-return timestamps; terminal completion differs from last new output. No Q(g), selected SLO threshold, client-receipt, quality or method-GO claim.',
        lifecycle_source_sha256=hashlib.sha256(Path(life.__file__).read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True);(args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],cells=[dict(label=r['label'],status=r['status'],error=r.get('error')) for r in cells])))


if __name__=='__main__':main()
