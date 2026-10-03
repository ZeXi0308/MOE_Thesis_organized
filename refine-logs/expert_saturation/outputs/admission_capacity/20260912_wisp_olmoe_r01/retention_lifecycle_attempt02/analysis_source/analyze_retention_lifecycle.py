"""Describe paired none/early versus frequency/late episodes, including trace costs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from analyze_native_pager_transfer import requests
from analyze_runtime_variance import overlap

FIELDS = ('capture_wall_s', 'cycle_wall_s', 'engine_wall_s', 'payload_bytes', 'groups', 'load_section_ms',
          'flush_wall_s', 'old_ttft_s', 'old_max_itl_s', 'old_boundary_itl_s', 'old_post_action_max_itl_s',
          'old_completion_s', 'new_ttft_s', 'new_max_itl_s', 'new_completion_s', 'snapshot_wall_s')
TOTALS = ('miss', 'evict', 'weight_copy_bytes', 'group_count', 'load_cuda_span_ms')
WARMUPS = ['warmup.json'] + [f'warmup_injection_{s}_release8.json' for s in
                           ('none_early', 'frequency_early', 'frequency_late', 'decode_late')]


def read(path): return json.loads(path.read_text())
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def check(ok, message, issues):
    if not ok: issues.append(message)


def describe(rows):
    return {k: dict(values=[r[k] for r in rows], min=min(r[k] for r in rows),
        max=max(r[k] for r in rows), median=statistics.median(r[k] for r in rows)) for k in FIELDS} if rows else {}


def trace_signature(records, aliases):
    return [dict(step=r['context']['step_id'], layer=r['layer_name'],
        rows=[(aliases[x['internal_request_id']], x['computed_position'], x['prompt_tokens']) for x in r['context']['rows']],
        topk=r['row_topk_experts'], active=r['active_experts'], entry=r['entry_resident_experts'],
        protected=r['retention']['chosen_protected_experts'], final=r['retention']['final_resident_experts'],
        groups=[{k:g[k] for k in ('required_experts','ensure_experts','protected_after','loaded_experts',
                                 'reloaded_experts','evicted_experts','weight_copy_bytes')} for g in r['groups']]) for r in records]


def engine_result(root, cell, protocol):
    label=cell['label']; path=root/'results'/label; issues=[]; rows=[]
    phase_path=next(p for p in (root/f'{label}_phases.json',root/f'{label}_analysis.json') if p.exists())
    phases=read(phase_path); observed=read(root/f'{label}_observations.json')
    pager=read(path/'pager_summary.json'); cycle=read(path/'cycle_complete.json'); episodes=read(path/'episodes.json')
    check(cell['status']=='COMPLETE' and cell['returncode']==0 and read(path/'status.json')['status']=='COMPLETE','incomplete engine',issues)
    check(cycle['trace_retention']=='episode' and len(episodes)==len(cycle['repeats'])==8,'lifecycle/repeat count mismatch',issues)
    check(read(path/'environment.json')['sources']==protocol['source_sha256'],'source hashes mismatch',issues)
    check(phases['phase_sum_matches_global'] and not observed['issues'],'phase/observation reader issues',issues)
    checks=[json.loads(line) for line in (path/'gpu_checks.jsonl').read_text().splitlines()]
    check(len(checks)>=19 and all(c['decision']=='PASS' and not c['foreign_pids'] and not c['query_errors'] for c in checks),'GPU boundaries unavailable/failed',issues)
    measured=defaultdict(list); totals={k:Counter() for k in ('all','measurement')}; count=failures=0
    with (path/'pager/calls.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line); check(r['call_id']==count,f'call_id discontinuity at {count}',issues); count+=1
            failures+=r['status']!='complete'
            for k in ('all','measurement') if r['measurement'] else ('all',): totals[k].update({f:r[f] for f in TOTALS})
            if r['measurement']: measured[r['context']['phase']].append(r)
    check(count==pager['all_calls']==observed['runtime_records_final'] and failures==pager['failed_calls']==0,'global calls/failures mismatch',issues)
    check(set(measured)=={e['phase'] for e in episodes} and sum(map(len,measured.values()))==pager['measurement_calls'],'measurement phase coverage mismatch',issues)
    check(all(abs(totals[k][f]-pager[k][f])<1e-6 for k in totals for f in TOTALS),'pager totals differ from records',issues)
    flushes=cycle['trace_flushes']; written=0
    for f in flushes:
        check(f['status']=='complete' and f['flushed_calls_before']==written and f['held_records_after']==f['held_events_after']==0,'flush status/reference/count mismatch',issues)
        written+=f['records_written']; check(f['flushed_calls_after']==written,'flush cumulative count mismatch',issues)
    check(written==count and sum(f['bytes_written'] for f in flushes)==(path/'pager/calls.jsonl').stat().st_size,'flush file byte/count mismatch',issues)
    events=[]
    for e in read(path/'runtime_events.json')['events']:
        if e.get('start_perf_ns') is None or e.get('stop_perf_ns') is None:
            issues.append('unpaired GC event'); continue
        check(e['duration_ns']==e['stop_perf_ns']-e['start_perf_ns']>=0,'GC clock mismatch',issues)
        events.append(dict(start=e['start_perf_ns']/1e9,stop=e['stop_perf_ns']/1e9,generation=e['generation']))
    cycles={r['name']:r for r in cycle['repeats']}; workload=read(path/'workload.json')
    for i,e in enumerate(episodes):
        phase=e['phase']; name=phase.split('/')[0]; d=path/name; row=dict(repeat=i,name=name,phase=phase,status=e['status']); rows.append(row)
        try:
            raw=read(d/'raw.json'); p=phases['repeats'][name]; rr=measured[phase]; c=cycles[name]
            res=read(d/'measurement_resources.json'); arm=res['group_retention']+'_'+res['group_retention_order']; row['arm']=arm
            check(raw['status']=='COMPLETE' and not p['issues'] and p['phase_bytes_valid'] and p['extra_bytes']==0 and p['cpu_window_order_valid'],name+': incomplete/phase checks',issues)
            check(p['retention']['invariants_valid'] and p['retention']['raw_row_routes_present'] and p['retention']['engine_call_join_valid'],name+': retention checks',issues)
            req,errors=requests(raw,workload); issues.extend(name+': '+x for x in errors)
            old=[r for r in req.values() if r['arrival_s']==0]; new=[r for r in req.values() if r['arrival_s']!=0]
            check(len(old)==2 and len(new)==1 and all(r['status']=='completed' for r in req.values()) and sorted(len(r['output_token_ids']) for r in req.values())==[8,16,16],name+': request coverage',issues)
            check(all(p['equality_to_first'][k] for k in ('cache','alloc','logical','pre_action_signature','retention_resources')),name+': common preaction/resource mismatch',issues)
            warmup_wall=0.0; warmup_counts=dict(episodes=0,requests=0,tokens=0)
            check(sorted(f.name for f in d.glob('warmup*.json'))==sorted(WARMUPS),name+': common warmup files',issues)
            for f in WARMUPS:
                warm=read(d/f); check(warm['status']=='COMPLETE' and all(r['status']=='completed' and len(r['output_token_ids'])==r['max_output_tokens'] for r in warm['requests']),name+'/'+f+': incomplete',issues)
                warmup_wall+=warm['observation_end_s']; warmup_counts['episodes']+=1; warmup_counts['requests']+=len(warm['requests']); warmup_counts['tokens']+=sum(len(r['output_token_ids']) for r in warm['requests'])
            stages={}; joined=[]
            for stage,s in p['retention']['stages'].items():
                calls=[raw['engine_calls'][j] for j in s['engine_calls']]
                records=[r for call in calls for r in rr if call['scheduler_step_start']<=r['context']['step_id']<call['scheduler_step_stop']]; joined.extend(r['call_id'] for r in records)
                stages[stage]=dict(engine_calls=s['engine_calls'],layer_calls=len(records),groups=sum(len(r['groups']) for r in records),
                    bytes=sum(r['weight_copy_bytes'] for r in records),wall_s=sum(x['return_s']-x['start_s'] for x in calls),
                    load_section_ms=sum(r['load_cuda_span_ms'] for r in records),host_apply_ms=sum(r['host_apply_ms'] for r in records))
                check(stages[stage]['bytes']==s['bytes'] and stages[stage]['groups']==s['groups'],name+': stage accounting mismatch',issues)
            check(sorted(joined)==sorted(r['call_id'] for r in rr),name+': actual call join mismatch',issues)
            f=c['trace_flush']; start,end=c['start_perf_ns']/1e9,c['end_perf_ns']/1e9; origin=raw['measurement_origin_perf_counter_s']
            check(abs(c['capture_wall_s']-raw['observation_end_s'])<1e-8 and start<=origin<=origin+raw['observation_end_s']<=f['start_perf_ns']/1e9<=f['end_perf_ns']/1e9<=end,name+': cycle clock mismatch',issues)
            outer=read(d/'runtime_observation.json'); sig=trace_signature(rr,raw['internal_to_source'])
            row.update(status=raw['status'],requests=req,call_id_first=rr[0]['call_id'],call_id_last=rr[-1]['call_id'],layer_calls=len(rr),
                trace_sha256=digest(sig),output_sha256=digest({k:r['output_token_ids'] for k,r in req.items()}),
                capture_wall_s=raw['observation_end_s'],cycle_wall_s=end-start,engine_wall_s=sum(x['wall_s'] for x in stages.values()),
                payload_bytes=sum(r['weight_copy_bytes'] for r in rr),groups=sum(len(r['groups']) for r in rr),load_section_ms=sum(r['load_cuda_span_ms'] for r in rr),
                flush_wall_s=(f['end_perf_ns']-f['start_perf_ns'])/1e9,flush=f,warmup_capture_wall_s=warmup_wall,warmup_counts=warmup_counts,
                old_ttft_s=max(r['ttft_s'] for r in old),old_max_itl_s=max(r['max_itl_s'] for r in old),old_completion_s=max(r['completion_latency_s'] for r in old),
                old_boundary_itl_s=max(p['old_boundary_itl_s'].values()),old_post_action_max_itl_s=max(p['post_action_old_max_itl_s'].values()),
                new_ttft_s=new[0]['ttft_s'],new_max_itl_s=new[0]['max_itl_s'],new_completion_s=new[0]['completion_latency_s'],
                snapshot_wall_s=p['snapshot_wall_s'],observation_envelope_s=p['observation_envelope_s'],stages=stages,
                fixed_resources=p['fixed_resources'],cuda_memory=p['memory'],equality_to_first=p['equality_to_first'],
                held_before=outer['before']['runtime_records_count'],held_after=outer['after']['runtime_records_count'],
                gc_capture=overlap(events,origin,origin+raw['observation_end_s']),gc_cycle=overlap(events,start,end),gc_flush=overlap(events,f['start_perf_ns']/1e9,f['end_perf_ns']/1e9))
        except (OSError,KeyError,ValueError,TypeError,IndexError,StopIteration) as exc: issues.append(name+': '+repr(exc))
    return dict(label=label,status=cell['status'],issues=issues,rows=rows,all_calls=count,measurement_calls=pager['measurement_calls'],
        totals=totals,source_sha256=read(path/'environment.json')['sources'],gpu_checks=[{k:c[k] for k in ('stage','decision','caller_pid','allowed_pids')} for c in checks],
        process_wall_s=cell['process_wall_s'],post_init_cycle_wall_s=cycle['post_init_cycle_wall_s'],trace_flushes=flushes,
        gc_cycle=overlap(events,cycle['start_perf_ns']/1e9,cycle['end_perf_ns']/1e9),
        per_arm={arm:describe([r for r in rows if r.get('arm')==arm and 'cycle_wall_s' in r]) for arm in ('none_early','frequency_late')})


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    issues=[]; engines=[]; pairs=[]
    try:
        protocol=read(args.input_dir/'protocol.json'); execution=read(args.input_dir/'results/execution.json')
        check(execution['status']=='COMPLETE' and len(execution['cells'])==len({c['label'] for c in execution['cells']})==2,'campaign engine count/status mismatch',issues)
        for cell in execution['cells']:
            try:
                engine=engine_result(args.input_dir,cell,protocol); engines.append(engine); issues.extend(cell['label']+': '+s for s in engine['issues'])
                rows=engine['rows']; check([r.get('arm') for r in rows]==protocol['engine_sequences'][cell['label']],cell['label']+': declared arm sequence mismatch',issues)
                check(Counter(r.get('arm') for r in rows)==Counter(none_early=4,frequency_late=4),cell['label']+': arm counts mismatch',issues)
                for i in (0,2,4,6):
                    group={r['arm']:r for r in rows[i:i+2]}; a,b=group['none_early'],group['frequency_late']
                    pairs.append(dict(engine=cell['label'],positions=[i,i+1],none=a['name'],frequency=b['name'],
                        delta_frequency_minus_none={k:b[k]-a[k] for k in FIELDS},delta_pct={k:100*(b[k]/a[k]-1) if a[k] else None for k in FIELDS},
                        request_deltas={rid:{k:b['requests'][rid][k]-a['requests'][rid][k] for k in ('ttft_s','tpot_s','max_itl_s','completion_latency_s')} for rid in a['requests']}))
            except (OSError,KeyError,ValueError,TypeError,IndexError,StopIteration) as exc:
                issues.append(cell['label']+': '+repr(exc))
                if not any(e['label']==cell['label'] for e in engines):
                    engines.append(dict(label=cell['label'],status=cell.get('status'),rows=[],issues=[repr(exc)]))
    except (OSError,KeyError,ValueError,TypeError) as exc: issues.append('campaign: '+repr(exc))
    all_rows=[r for e in engines for r in e['rows'] if 'trace_sha256' in r]
    check(len(all_rows)==16 and len(pairs)==8,'incomplete usable repeats/pairs',issues)
    check(len({digest(r['fixed_resources']) for r in all_rows})==1,'fixed KV/scratch resources differ',issues)
    consistent={arm:{field:len({r[field] for r in all_rows if r['arm']==arm})==1 for field in ('trace_sha256','output_sha256')} for arm in ('none_early','frequency_late')}
    output=dict(status='ISSUES' if issues else 'DESCRIPTIVE_RETENTION_LIFECYCLE_COMPARISON',issues=issues,engines=engines,pairs=pairs,
        within_mode_equal=consistent,cross_mode_output_equal=len({r['output_sha256'] for r in all_rows})==1,
        scope=['Two engines; eight dependent episodes per arm from three reused documents. No significance or population noise floor.',
            'Trace/output equality is descriptive within each actual policy; different cross-policy routes and caches are allowed.',
            'Capture/request clocks include observations; repeat cycle adds reset, five warmups, IO and flush, but excludes its own cycle_progress write and shared tail. Engine cycle/process retain those costs; process wall also retains setup/teardown and is not assigned to individual arms.',
            'Stage wall is the sum of disjoint actual engine-call intervals. Host/CUDA/GC spans overlap and are never added to wall or subtracted as benefit.',
            'GC observer closes before its export and engine shutdown; GC intersections cover only observed intervals, not all process time.',
            'GPU checks are retained boundary snapshots; allocation equality does not establish identical physical KV contents.'])
    with args.out.open('x') as f: json.dump(output,f,indent=2,allow_nan=False); f.write('\n')
    if issues: raise SystemExit('issues retained in output')


if __name__=='__main__': main()
