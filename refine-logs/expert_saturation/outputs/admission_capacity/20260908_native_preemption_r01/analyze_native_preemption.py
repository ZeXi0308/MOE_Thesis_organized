#!/usr/bin/env python3
"""Full native-preemption versus safe29; scheduled interval accounting, not receipts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
SAFE_PATH=HERE.parent/'20260908_kv_safe_static_r02/analyze_safe_static.py'
spec=importlib.util.spec_from_file_location('safe_analysis_helpers',SAFE_PATH)
safe=importlib.util.module_from_spec(spec);spec.loader.exec_module(safe)
base, read, require=safe.base,safe.read,safe.require
distribution=sys.modules['metrics']._distribution
LABELS=['repeat0-native32','repeat0-safe','repeat1-safe','repeat1-native32']


def overlap(interval, covered):
    start,end=interval
    return sum(max(0,min(end,b)-max(start,a)) for a,b in covered)


def extend(covered,interval):
    result=[]
    for start,end in sorted([*covered,list(interval)]):
        if result and start<=result[-1][1]:
            result[-1][1]=max(result[-1][1],end)
        else:
            result.append([start,end])
    return result


def execution_accounting(raw):
    steps,calls=raw['scheduler_steps'],raw['engine_steps']
    completed,call_for_step={},{}
    previous=0
    receipt_events=defaultdict(list)
    for event in raw['output_events']:
        receipt_events[event['received_s']].append(event)
    for index,call in enumerate(calls):
        require(call['call_index']==index and call['scheduler_step_start']==previous,'engine call/step identity mismatch')
        end=call['scheduler_step_end'];require(previous<=end<=len(steps),'invalid engine call step interval')
        require(call['start_s']<=call['returned_s'],'engine call time reversed')
        for k in range(previous,end):
            require(steps[k]['step']==k and call['start_s']<=steps[k]['start_s']<=steps[k]['end_s']<=call['returned_s'],
                    'scheduler interval is outside engine call')
            call_for_step[k]=call
            if call['completed']:
                completed[k]=call
        previous=end
        if call['completed']:
            require(sum(e['chunk_size'] for e in receipt_events[call['returned_s']])==call['new_output_tokens'],
                    'engine call new output count mismatch')
    require(previous==len(steps),'scheduler steps not covered by engine calls')
    covers=defaultdict(list);per_request=defaultdict(lambda:Counter())
    records=[];totals=Counter();attempt_tokens=0
    for index,step in enumerate(steps):
        require(sum(r['scheduled_tokens'] for r in step['scheduled'])==step['total_scheduled_tokens'],'scheduled step total mismatch')
        if index not in completed:
            attempt_tokens+=step['total_scheduled_tokens'];continue
        for r in step['scheduled']:
            rid=r['request_id'];start,end=r['scheduled_start_computed'],r['computed_after'];amount=end-start
            require(start>=0 and amount==r['scheduled_tokens']>0,'invalid actual scheduled range')
            repeated=overlap((start,end),covers[rid])
            prompt_end=min(end,r['prompt_tokens'])
            new_prefill=max(0,prompt_end-start)-overlap((start,max(start,prompt_end)),covers[rid])
            new_decode=amount-repeated-new_prefill
            require((repeated,new_prefill,new_decode)==(r['recompute_tokens'],r['prefill_tokens'],r['decode_tokens']),
                    'recorded mutually exclusive work differs from prior executed interval union')
            high=max((b for a,b in covers[rid]),default=0)
            require(r['executed_high_water_before']==high,'recorded high-water differs from previous completed work')
            work=dict(scheduled_tokens=amount,recomputed_tokens=repeated,new_prefill_tokens=new_prefill,new_decode_tokens=new_decode)
            totals.update(work);per_request[rid].update(work)
            if repeated:
                records.append(dict(step=index,request_id=rid,start=start,end=end,**work,
                    call_start_s=completed[index]['start_s'],call_returned_s=completed[index]['returned_s']))
            covers[rid]=extend(covers[rid],(start,end))
    require(totals['scheduled_tokens']==totals['recomputed_tokens']+totals['new_prefill_tokens']+totals['new_decode_tokens'],
            'global scheduled work conservation failed')
    return dict(totals=dict(totals),per_request={k:dict(v) for k,v in per_request.items()},recomputed_intervals=records,
        unique_intervals=dict(covers),completed_engine_calls=sum(c['completed'] for c in calls),
        failed_engine_calls=sum(not c['completed'] for c in calls),attempted_scheduled_tokens=attempt_tokens,
        completed_calls_without_receipts=sum(c['completed'] and not c['output_request_ids'] for c in calls),
        completed_calls_without_new_tokens=sum(c['completed'] and c['new_output_tokens']==0 for c in calls),
        semantics='Work counts use only successful engine calls and overlap with previously executed request-position intervals. '
                  'Recomputed generated history is not new decoding. Work is already inside full host request time.'),call_for_step


def preemptions(raw,call_for_step):
    indexed={r['request_id']:r for r in raw['requests']};events=[]
    for event in raw['preemption_events']:
        e=dict(event);rid=raw['internal_to_source'][e['victim_internal_request_id']];e['request_id']=rid
        k=e['attempted_step'];call=call_for_step.get(k)
        e['host_call_interval_s']=[call['start_s'],call['returned_s']] if call else None
        if e['original_preemption_returned']:
            before,after=e['victim_state'],e['victim_state_after']
            require(e['original_preemption_called'] and after['computed_tokens']==0,'native preemption did not reset computed state')
            require(after['num_preemptions']==before['num_preemptions']+1,'preemption counter mismatch')
            prefix=e['output_token_ids_before']
            require(prefix==e['output_token_ids_after'] and len(prefix)==before['output_tokens']==after['output_tokens'],
                    'preemption changed already generated output prefix')
            require(indexed[rid]['output_token_ids'][:len(prefix)]==prefix,'final output lost preemption-time prefix')
            e['computed_reset_and_prefix_preserved']=True
        events.append(e)
    require(sum(e['original_preemption_returned'] for e in events)==raw['actual_preemption_count'],'native preemption count mismatch')
    return events


def request_effects(raw,events,work):
    victims={e['request_id'] for e in events if e['original_preemption_returned']}
    rows=[];completion=[]
    for r in raw['requests']:
        gaps=[dict(start_s=a,end_s=b,itl_s=b-a,output_index=i+1) for i,(a,b) in enumerate(zip(r['token_times_s'],r['token_times_s'][1:]))]
        gap=max(gaps,key=lambda x:x['itl_s']) if gaps else None
        if gap:
            inside=[e for e in events if e['request_id']==r['request_id'] and e['host_call_interval_s'] and
                    gap['start_s']<=e['host_call_interval_s'][0] and e['host_call_interval_s'][1]<=gap['end_s']]
            own=[x for x in work['recomputed_intervals'] if x['request_id']==r['request_id'] and
                 gap['start_s']<=x['call_start_s'] and x['call_returned_s']<=gap['end_s']]
            gap.update(own_preemption_steps=[e['attempted_step'] for e in inside],
                own_recomputed_tokens=sum(x['recomputed_tokens'] for x in own),
                own_recompute_steps=[x['step'] for x in own])
        latency=r['completion_s']-r['arrival_s'] if r['status']=='completed' else None
        if latency is not None:completion.append(latency)
        rows.append(dict(request_id=r['request_id'],document_id=r['document_id'],status=r['status'],
            is_preemption_victim=r['request_id'] in victims,completion_latency_s=latency,
            completed_output_tokens=len(r['output_token_ids']),longest_itl=gap,
            recomputed_tokens=work['per_request'].get(r['request_id'],{}).get('recomputed_tokens',0)))
    return dict(per_request=rows,completion_latency_s=distribution(completion),
        request_max_itl_s=distribution([r['longest_itl']['itl_s'] for r in rows if r['longest_itl']]),
        longest_itl_requests=sorted([r for r in rows if r['longest_itl']],key=lambda x:x['longest_itl']['itl_s'],reverse=True)[:8],
        victim_request_ids=sorted(victims),victim_requests=[r for r in rows if r['is_preemption_victim']],
        limitation='Temporal association and per-arm identity matching do not isolate a counterfactual cost of one preemption event.')


def inspect(directory,label):
    repeat,arm=label.split('-')
    out=dict(label=label,repeat=int(repeat[-1]),arm=arm,status='MISSING',full_episode_comparison_eligible=False,raw_path=str(directory/'raw.json'))
    terminal=read(directory/'status.json') if (directory/'status.json').exists() else None
    q=read(directory/'safe-cap-qualification.json') if (directory/'safe-cap-qualification.json').exists() else None
    out.update(terminal_status=terminal,qualification=q)
    if not (directory/'raw.json').exists():
        out['status']='UNRUN' if terminal and terminal['status']=='UNRUN' else 'INCOMPLETE' if terminal else 'MISSING';return out
    try:
        raw,config=read(directory/'raw.json'),read(directory/'config.json');cap=29 if arm=='safe' else 32
        out.update(raw_status=raw['status'],cap=config['cap'])
        require(q['status']=='QUALIFIED' and q['safe_cap']==29,'frozen safe29 qualification not met')
        reserved=(config['prompt_tokens']+config['output_tokens']+q['block_size']-1)//q['block_size']
        require(reserved==q['per_request_reserved_blocks'] and min(32,q['usable_blocks']//reserved)==29,'safe formula differs')
        require(config['requested_arm']==arm and config['cap']==cap,'arm/cap mismatch')
        mode='stop_before_preemption' if arm=='safe' else 'native_recompute'
        require(raw['preemption_mode']==config['preemption_mode']==mode,'preemption mode mismatch')
        identity=base.validate_identity(raw,config,HERE/'inputs_preparation/prepared','long',cap)
        metrics=base.summarize_episode_requests(raw['requests'],observation_end_s=raw['observation_end_s'],ttft_slo_s=5.0,tpot_slo_s=0.2)
        saved=read(directory/'metrics.json');require(all(saved.get(k)==v for k,v in metrics.items()),'saved request metrics mismatch')
        work,calls=execution_accounting(raw);events=preemptions(raw,calls)
        if arm=='safe':require(raw['actual_preemption_count']==0,'safe arm executed native preemption')
        pools=[t[k]['pool'] for t in raw['memory_trace'] for k in ['before','after'] if t[k] is not None]
        require(all(0<=p['used_blocks']<=p['usable_blocks'] and p['used_blocks']+p['free_blocks']==p['usable_blocks']==q['usable_blocks'] for p in pools),'invalid pool accounting')
        all_pools=pools+[e[k] for e in raw['preemption_events'] for k in ['pool','pool_after'] if k in e]
        all_pools += [a[k] for t in raw['memory_trace'] for a in t['allocation_failures'] for k in ['before','after']]
        expected='CAPACITY_BOUNDARY_STOP' if raw['capacity_boundary'] else raw['status']
        status=expected if terminal['status']==expected else 'INCOMPLETE'
        if status=='COMPLETE' and not (identity['all_requests_completed'] and metrics['n_completed']==32):status='INCOMPLETE'
        out.update(status=status,identity_check=identity,full_episode_comparison_eligible=status=='COMPLETE',metrics=metrics,
            work=work,preemption_events=events,actual_preemption_count=raw['actual_preemption_count'],effects=request_effects(raw,events,work),
            max_scheduled_active=max(s['actual_active'] for s in raw['scheduler_steps']),
            max_decode_active=max(s['decode_requests'] for s in raw['scheduler_steps']),max_waiting=max(s['waiting_requests'] for s in raw['scheduler_steps']),
            decode_width_step_counts=dict(sorted(Counter(s['decode_requests'] for s in raw['scheduler_steps']).items())),
            allocation_failures=sum(len(t['allocation_failures']) for t in raw['memory_trace']),
            temporarily_unscheduled_decode_ids=sum(len(t['existing_decode_not_scheduled_ids']) for t in raw['memory_trace']),
            pool_ranges={k:[min(p[k] for p in pools),max(p[k] for p in pools)] for k in pools[0]},
            pool_ranges_scope='Scheduler before/after snapshots only; transient allocation/preemption points excluded.',
            max_used_fraction=max(p['used_blocks']/p['usable_blocks'] for p in pools),minimum_free_blocks=min(p['free_blocks'] for p in pools),
            all_observed_pool_ranges={k:[min(p[k] for p in all_pools),max(p[k] for p in all_pools)] for k in all_pools[0]},
            all_observed_pool_scope='Scheduler before/after plus allocation-failure before/after and preemption before/after.',
            observed_zero_free=any(p['free_blocks']==0 for p in all_pools),
            capacity_boundary=raw['capacity_boundary'],engine_args=read(directory/'engine_args.json'),
            memory_before=read(directory/'memory-before.json'),memory_after=read(directory/'memory-after.json'))
    except (KeyError,ValueError,TypeError,OSError) as exc:
        out.update(status='INVALID',error=str(exc),full_episode_comparison_eligible=False)
    return out


def comparisons(rows):
    result=[]
    for repeat in [0,1]:
        native=next(r for r in rows if r['repeat']==repeat and r['arm']=='native32')
        guarded=next(r for r in rows if r['repeat']==repeat and r['arm']=='safe')
        p=dict(repeat=repeat,base=native['label'],action=guarded['label'],status='UNRUN_OR_UNQUALIFIED')
        if native['full_episode_comparison_eligible'] and guarded['full_episode_comparison_eligible']:
            a,b=native['metrics'],guarded['metrics']
            p.update(status='DESCRIPTIVE_COMPLETE_EPISODE_COMPARISON',safe_throughput_relative_change=b['throughput_rps']/a['throughput_rps']-1,
                safe_duration_relative_change=b['observation_duration_s']/a['observation_duration_s']-1,
                safe_minus_native={f'{metric}_{q}_s':b['latency_s'][metric][q]-a['latency_s'][metric][q]
                                  for metric in ['ttft','tpot','itl'] for q in ['p50','p99']})
            x={r['request_id']:r for r in native['effects']['per_request']};y={r['request_id']:r for r in guarded['effects']['per_request']}
            p['matched_request_differences']=[dict(request_id=rid,native_preemption_victim=x[rid]['is_preemption_victim'],
                safe_minus_native_completion_latency_s=y[rid]['completion_latency_s']-x[rid]['completion_latency_s'],
                native_max_itl_s=x[rid]['longest_itl']['itl_s'],safe_max_itl_s=y[rid]['longest_itl']['itl_s']) for rid in x]
        result.append(p)
    return result


def readable(result):
    lines=['# Native preemption versus safe29: full request accounting','',result['status'],'',
        '| Cell | Status | Completed | Full duration s | Throughput req/s | TTFT p50/p99 s | TPOT p50/p99 ms | Completion p99 s | Max ITL s | Native preemptions | Recomputed tokens |',
        '|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|']
    for r in result['cells']:
        if not r['full_episode_comparison_eligible']:
            lines.append(f"| {r['label']} | {r['status']} | — | — | — | — | — | — | — | — | — |");continue
        m=r['metrics'];t=m['latency_s'];e=r['effects']
        lines.append(f"| {r['label']} | {r['status']} | {m['n_completed']}/32 | {m['observation_duration_s']:.5f} | {m['throughput_rps']:.5f} | "
            f"{t['ttft']['p50']:.5f}/{t['ttft']['p99']:.5f} | {t['tpot']['p50']*1000:.5f}/{t['tpot']['p99']*1000:.5f} | "
            f"{e['completion_latency_s']['p99']:.5f} | {e['longest_itl_requests'][0]['longest_itl']['itl_s']:.5f} | {r['actual_preemption_count']} | {r['work']['totals']['recomputed_tokens']} |")
    lines+=['','## Victims, work and no-output calls','']
    for r in result['cells']:
        if 'work' not in r:continue
        lines.append(f"- {r['label']}: active/decode/wait maxima {r['max_scheduled_active']}/{r['max_decode_active']}/{r['max_waiting']}; "
            f"scheduler-boundary KV peak {r['max_used_fraction']*100:.3f}%, minimum free {r['minimum_free_blocks']}; "
            f"any observed allocation/preemption/scheduler point reaches zero free blocks: {r['observed_zero_free']}; successful calls without new tokens "
            f"{r['work']['completed_calls_without_new_tokens']}; work {r['work']['totals']}.")
        lines.append(f"  - Pooled token ITL p99 {r['metrics']['latency_s']['itl']['p99']:.6f} s versus request-max-ITL p99 "
            f"{r['effects']['request_max_itl_s']['p99']:.6f} s. These have different populations; sparse victim pauses can disappear in pooled p99.")
        for victim in r['effects']['victim_requests']:
            lines.append(f"  - Victim {victim['request_id']}: recomputed {victim['recomputed_tokens']} positions; longest ITL {victim['longest_itl']}.")
    v=result['execution_verification']
    lines+=['','## Boundaries','',
        f"Engine arguments equal: {result['engine_args_equal']}; five executed source hashes match: {all(v['per_cell_source_match'].values())}; software equal: {all(v['software_fields_equal'].values())}; warmup raw count: {v['warmup_count']}.",
        'All main metrics use complete host request histories. Waiting/recomputation/telemetry already reside in that denominator; no stage is added again.',
        'Recomputation is interval overlap in successful scheduled work. Token receipt count is not a work counter; successful no-output calls are retained.',
        'Native victims may stop decoding while waiting. That is legal preemption, not a nonpreemptive-invariant failure.',
        'Reference 5 s / 200 ms SLO is secondary. Request-level p99 uses 32 repeated inputs and is descriptive; output quality and expert reclaim are unmeasured.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results-dir',type=Path,default=HERE/'gpu_results')
    parser.add_argument('--output-dir',type=Path,default=HERE/'analysis');parser.add_argument('--execution-archive',type=Path,default=HERE/'execution.tar.gz')
    args=parser.parse_args();require(not args.output_dir.exists(),'output directory must be new')
    rows=[inspect(args.results_dir/label,label) for label in LABELS]
    engines=[r['engine_args'] for r in rows if 'engine_args' in r];equal=bool(engines) and all(e==engines[0] for e in engines)
    require(equal or not engines,'engine arguments differ')
    result=dict(status='MEASUREMENT_ONLY' if all(r['full_episode_comparison_eligible'] for r in rows) else 'INCOMPLETE_OR_UNRUN',
        cells=rows,comparisons=comparisons(rows),engine_args_equal=equal,
        execution_verification=safe.verification(rows,args.results_dir,args.execution_archive),
        dependencies=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in [Path(__file__).resolve(),SAFE_PATH,safe.PREVIOUS,Path(sys.modules['metrics'].__file__)]])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'report.md').write_text(readable(result))
    print(json.dumps({k:result[k] for k in ['status','engine_args_equal']},indent=2))


if __name__=='__main__':main()
