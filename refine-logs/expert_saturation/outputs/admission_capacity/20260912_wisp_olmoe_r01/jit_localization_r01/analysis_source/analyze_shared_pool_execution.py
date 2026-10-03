"""Describe declared actual shared-pool engines; own-trace checks are not counterfactuals."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from analyze_layer_budget import LRU
from analyze_layer_budget_execution import difference, ERRORS
from analyze_native_phases import retention_stats
from analyze_native_pager_transfer import requests, call_accounting
from analyze_retention_lifecycle import FIELDS, TOTALS, WARMUPS, check, read, digest
from analyze_runtime_variance import overlap
from shared_pool_plan import plan_shared_pool
from wisp_expert_groups import partition_experts

METRICS = FIELDS + ('d2d_bytes', 'canonical_groups', 'host_plan_ms', 'host_apply_ms', 'stage_d2d_bytes', 'writeback_d2d_bytes')
MODES = {'uniform', 'selected', 'split', 'oneshot', 'fullstage'}
PAIRS = [('uniform','oneshot'), ('selected','oneshot'), ('split','oneshot'),
         ('uniform','split'), ('uniform','selected'), ('selected','split'), ('fullstage','oneshot')]


def amounts(records):
    d2d=sum(r.get('d2d_copy_bytes',0) for r in records)
    writeback=sum(len(r.get('full_stage_plan',{}).get('writeback',[]))*12582912 for r in records)
    return dict(layer_calls=len(records), payload_bytes=sum(r['weight_copy_bytes'] for r in records),
        d2d_bytes=d2d,stage_d2d_bytes=d2d-writeback,writeback_d2d_bytes=writeback,groups=sum(len(r['groups']) for r in records),
        canonical_groups=sum(r.get('canonical_group_count',len(r['groups'])) for r in records),
        evictions=sum(r['evict'] for r in records),canonical_evictions=sum(r.get('canonical_evict',r['evict']) for r in records),
        load_section_ms=sum(r['load_cuda_span_ms'] for r in records),
        host_plan_ms=sum(r.get('host_plan_ms',0) for r in records),host_apply_ms=sum(r['host_apply_ms'] for r in records))


def trace_check(records, caps, sizes, initial, raw=None):
    """Rebuild exact own-policy measurement state; initial is an actual cache snapshot."""
    issues=[]; states={}; routes=[]; signatures=[]; steps=Counter()
    for name,cap in caps.items():
        s=initial[name]; state=LRU(cap); state.slots=list(s['slot_to_expert']); state.ticks=list(s['lru_tick'])
        state.mapping={int(e):slot for e,slot in s.get('expert_to_slot',{e:i for i,e in enumerate(state.slots) if e!=-1}).items()}; state.clock=s['lru_clock']; states[name]=state
        check(all(state.snapshot()[k]==v for k,v in s.items()),name+': initial metadata mismatch',issues)
    for r in records:
        name=r['layer_name']; state=states[name]; active=set(r['active_experts']); groups=r['groups']; context=r['context']
        tag=f"call {r['call_id']}"; steps[(context['step_id'],name)]+=1
        check(r['status']=='complete' and len(context['rows'])==len(r['row_topk_experts'])==r['rows'] and
              set(e for row in r['row_topk_experts'] for e in row)==active,tag+': row/active mismatch',issues)
        check(sorted(state.mapping)==r['entry_resident_experts'],tag+': sequential entry mismatch',issues)
        if 'shared_plan' in r or 'full_stage_plan' in r:
            fullstage='full_stage_plan' in r
            if fullstage:
                from full_stage_plan import plan_full_stage
            key='full_stage_plan' if fullstage else 'shared_plan'; planner=plan_full_stage if fullstage else plan_shared_pool
            expected=planner(active,state.snapshot(),int(name.split('.')[2]))
            copies=expected['hit_stage']+expected['writeback'] if fullstage else expected['d2d']
            check(r[key]==expected and len(groups)==int(bool(active)) and not ('shared_plan' in r and fullstage)
                  and r['execution']=='shared_pool_'+('fullstage' if fullstage else 'oneshot'),tag+': staged plan/path mismatch',issues)
            for g in groups:
                check(g['required_experts']==sorted(active) and g['loaded_experts']==[c['expert'] for c in expected['h2d']]
                    and g['d2d_experts']==[c['expert'] for c in copies] and
                    g['evicted_experts']==expected['entry_evicted_experts'] and g['evict']==expected['entry_evict'],tag+': actual copy/entry eviction mismatch',issues)
            check(r.get('canonical_group_count')==expected['canonical_group_count'] and r.get('canonical_evict')==expected['canonical_evict']
                and r['d2d_copy_bytes']==len(copies)*sizes[name],tag+': canonical/D2D accounting mismatch',issues)
            for group in expected['canonical_groups']: state.ensure(group)
            final=r['final_resident_experts']
        else:
            check([g['required_experts'] for g in groups]==partition_experts(active,state.mapping,state.cap),tag+': ordinary partition mismatch',issues)
            for g in groups:
                missing,victims=state.ensure(g['ensure_experts'])
                check(g['loaded_experts']==missing and g['evicted_experts']==victims,tag+': ordinary LRU mismatch',issues)
            check(retention_stats([r],caps).get('invariants_valid') and r['retention']['mode']=='none',tag+': ordinary retention mismatch',issues)
            final=r['retention']['final_resident_experts']
        check(sorted(state.mapping)==final and sorted(e for g in groups for e in g['required_experts'])==sorted(active),tag+': final/coverage mismatch',issues)
        check(r.get('d2d_copy_bytes',0)==sum(g.get('d2d_copy_bytes',0) for g in groups),tag+': D2D sum mismatch',issues)
        if raw:
            step=raw['scheduler_steps'][context['step_id']]
            expected_rows=[(s['internal_request_id'],p,s['prompt_tokens']) for s in step['scheduled']
                           for p in range(s['scheduled_start_computed'],s['scheduled_start_computed']+s['scheduled_tokens'])]
            check(context['row_request_order_verified'] and Counter((x['internal_request_id'],x['computed_position'],x['prompt_tokens']) for x in context['rows'])==Counter(expected_rows),tag+': scheduler row alignment mismatch',issues)
        aliases=raw['internal_to_source'] if raw else {}
        route=dict(layer=name,step=context['step_id'],topk=r['row_topk_experts'],
            rows=[(aliases.get(x['internal_request_id'],x['internal_request_id']),x['computed_position'],x['prompt_tokens']) for x in context['rows']])
        routes.append(route); signatures.append(dict(**route,entry=r['entry_resident_experts'],final=final,
            groups=[{k:g.get(k) for k in ('required_experts','ensure_experts','loaded_experts','reloaded_experts','evicted_experts','d2d_experts')} for g in groups],plan=r.get('shared_plan',r.get('full_stage_plan'))))
    if raw:
        expected_steps=Counter({(s['step'],name):1 for s in raw['scheduler_steps'] if s['total_scheduled_tokens'] for name in caps})
        check(steps==expected_steps,'trace step/layer coverage mismatch',issues)
    check(all(c['byte_count_valid'] for c in call_accounting(records,sizes)),'H2D group accounting mismatch',issues)
    return dict(issues=issues,route_sha256=digest(routes),trace_sha256=digest(signatures),final_cache_sha256=digest({n:s.snapshot() for n,s in states.items()}))


def engine_result(root, cell, declared, protocol):
    label=cell['label']; path=root/'results'/label; mode=declared['mode']; issues=[]; rows=[]
    repeats=protocol['workload']['episodes_per_fresh_engine']; private_cap=20 if mode=='fullstage' else 21
    pager=read(path/'pager_summary.json'); cycle=read(path/'cycle_complete.json'); episodes=read(path/'episodes.json')
    workload=read(path/'workload.json'); config=read(path/'config.json'); caps={r['layer_name']:r['cap'] for r in pager['layers']}
    sizes={r['layer_name']:r['pinned_bytes']//r['num_experts'] for r in pager['layers']}; limits=protocol['resources']
    check(cell['status']=='COMPLETE' and cell['returncode']==0 and read(path/'status.json')['status']=='COMPLETE',label+': incomplete engine',issues)
    check(len(caps)==16 and set(sizes.values())=={12582912} and pager['expert_scratch_bytes']==limits['expert_weight_bytes'],label+': total scratch mismatch',issues)
    expected_caps=read(root/'allocations/selected.json') if mode=='selected' else {n:private_cap if mode in ('split','oneshot','fullstage') else 24 for n in caps}
    check(caps==expected_caps,label+': actual allocation mismatch',issues)
    check(config['same_engine_runtime_diagnostic'] and config['trace_retention']=='episode' and not config['verify_kernel'] and
          config['requests']==3 and config['injection_chunk']==8 and config['token_budget']==160,label+': runtime config mismatch',issues)
    prepared_path=root/'prepared'/declared['prepared']/'workload.json'; prepared=read(prepared_path)
    block=protocol['inputs'][declared['block']]
    check(hashlib.sha256(prepared_path.read_bytes()).hexdigest()==block['prepared_sha256'] and
          [r['document_id'] for r in workload['source_requests']]==block['documents'],label+': block document mismatch',issues)
    check(workload['source_requests']==prepared['source_requests'] and workload['actual_prompt_token_ids']==
          [ids[:n] for ids,n in zip(prepared['actual_prompt_token_ids'],protocol['workload']['prompts'])],label+': prepared prefix mismatch',issues)
    source=read(path/'environment.json')['sources']; frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source').glob('*.py')}
    check(bool(source) and all(frozen.get(n)==h for n,h in source.items()),label+': source mismatch',issues)
    shared=read(path/'shared_pool.json') if mode in ('split','oneshot','fullstage') else None
    if shared:
        check(shared['status']=='COMPLETE' and shared['mode']==mode and not shared['qualification'] and not shared['validation'] and
              all(frozen.get(n)==h for n,h in shared['sources'].items()),label+': shared mode/source/qualification mismatch',issues)
    trace=[json.loads(s) for s in (path/'pager/calls.jsonl').read_text().splitlines()]; measured=defaultdict(list); phases=defaultdict(list)
    for i,r in enumerate(trace):
        check(r['call_id']==i and r['status']=='complete',label+': call continuity/status mismatch',issues); phases[r['context']['phase']].append(r)
        check(r['group_count']==len(r['groups']) and r.get('d2d_copy_bytes',0)==sum(len(g.get('d2d_experts',[]))*sizes[r['layer_name']] for g in r['groups']),label+': actual groups/D2D mismatch',issues)
        if r['measurement']: measured[r['context']['phase']].append(r)
    check(len(trace)==pager['all_calls'] and pager['failed_calls']==0 and sum(map(len,measured.values()))==pager['measurement_calls'],label+': total calls mismatch',issues)
    check(all(c['byte_count_valid'] for c in call_accounting(trace,sizes)),label+': all-phase H2D mismatch',issues)
    for kind,rr in [('all',trace),('measurement',[r for r in trace if r['measurement']])]:
        check(all(abs(sum(r[k] for r in rr)-pager[kind][k])<1e-6 for k in TOTALS),label+': pager counters mismatch',issues)
    flushes=cycle['trace_flushes']; written=0
    for f in flushes:
        check(f['status']=='complete' and f['flushed_calls_before']==written and f['held_records_after']==f['held_events_after']==0,label+': flush mismatch',issues)
        written+=f['records_written']; check(f['flushed_calls_after']==written,label+': flush count mismatch',issues)
    check(written==len(trace) and sum(f['bytes_written'] for f in flushes)==(path/'pager/calls.jsonl').stat().st_size,label+': flush bytes mismatch',issues)
    checks=[json.loads(s) for s in (path/'gpu_checks.jsonl').read_text().splitlines()]
    check(len(checks)==2*repeats+3 and all(c['decision']=='PASS' and not c['foreign_pids'] and not c['query_errors'] for c in checks),label+': GPU boundary mismatch',issues)
    events=[]; observer=read(path/'runtime_events.json')
    check(observer['runtime_records_count']==observer['flushed_records_count']==len(trace) and observer['held_records_count']==0 and observer['final_snapshot_after_close'],label+': observer retention mismatch',issues)
    for e in observer['events']:
        if e.get('start_perf_ns') is None or e.get('stop_perf_ns') is None: issues.append(label+': unpaired GC'); continue
        check(e['duration_ns']==e['stop_perf_ns']-e['start_perf_ns']>=0,label+': GC clock mismatch',issues)
        events.append(dict(start=e['start_perf_ns']/1e9,stop=e['stop_perf_ns']/1e9))
    check(cycle['trace_retention']=='episode' and len(episodes)==len(cycle['repeats'])==repeats and set(measured)=={e['phase'] for e in episodes},label+': episode coverage mismatch',issues)
    cycles={c['name']:c for c in cycle['repeats']}
    for i,e in enumerate(episodes):
        phase=e['phase']; name=phase.split('/')[0]; d=path/name; raw=read(d/'raw.json'); res=read(d/'measurement_resources.json'); rr=measured[phase]; c=cycles[name]
        row=dict(repeat=i,name=name,phase=phase,status=raw['status']); rows.append(row)
        check(len(rr)==384,name+': declared workload layer-call count mismatch',issues)
        req,errors=requests(raw,workload); issues.extend(name+': '+s for s in errors); old=[r for r in req.values() if r['arrival_s']==0]; new=[r for r in req.values() if r['arrival_s']!=0]
        check(raw['status']=='COMPLETE' and len(old)==2 and len(new)==1 and all(r['status']=='completed' for r in req.values()) and sorted(len(r['output_token_ids']) for r in req.values())==[8,16,16],name+': request completion mismatch',issues)
        check([len(req[r['request_id']]['output_token_ids']) for r in workload['source_requests']]==protocol['workload']['outputs'],name+': per-request output budget mismatch',issues)
        check(res['layer_caps']==caps and res['expert_slots_total']==384 and res['expert_scratch_bytes']==limits['expert_weight_bytes'] and
              res['actual_unique_kv_storage_bytes']==limits['kv_bytes'] and res['cpu_affinity']==limits['CPU'] and res['scheduler_requests']==0,name+': resources mismatch',issues)
        check(res['group_retention']=='none' and res['group_retention_order']=='early' and res['group_retention_guard']=='none',name+': unexpected protection',issues)
        if shared:
            check(res['private_slots_total']==16*private_cap and res['shared_slots']==384-16*private_cap and res['shared_pool_mode']==mode and
                  len({(s['device'],s['pointer']) for s in res['expert_scratch_unique_storages']})==2 and sum(s['bytes'] for s in res['expert_scratch_unique_storages'])==limits['expert_weight_bytes'],name+': unique shared storage mismatch',issues)
        verified=trace_check(rr,caps,sizes,res['cache'],raw); issues.extend(name+': '+s for s in verified.pop('issues'))
        check(all(('shared_plan' in r)==(mode=='oneshot') and ('full_stage_plan' in r)==(mode=='fullstage') for r in rr),name+': execution path mismatch',issues)
        calls=raw['engine_calls']; action=raw['event_actions'][0]; origin=raw['measurement_origin_perf_counter_s']; start,end=c['start_perf_ns']/1e9,c['end_perf_ns']/1e9; f=c['trace_flush']
        check(abs(c['capture_wall_s']-raw['observation_end_s'])<1e-8 and start<=origin<=origin+raw['observation_end_s']<=f['start_perf_ns']/1e9<=f['end_perf_ns']/1e9<=end,name+': cycle time mismatch',issues)
        check(cycle['start_perf_ns']/1e9<=start<=end<=cycle['end_perf_ns']/1e9 and c['held_records']==c['held_events']==0,name+': repeat retention/time mismatch',issues)
        check(all(x['cpu_observation_start_s']<=x['start_s']<=x['return_s']<=x['cpu_observation_end_s'] for x in calls),name+': CPU clock mismatch',issues)
        warm_counts=Counter(); warm_wall=0; check(sorted(p.name for p in d.glob('warmup*.json'))==sorted(WARMUPS),name+': warmup files mismatch',issues)
        for filename in WARMUPS:
            warm=read(d/filename); check(warm['status']=='COMPLETE' and all(r['status']=='completed' and len(r['output_token_ids'])==r['max_output_tokens'] for r in warm['requests']),name+': incomplete warmup',issues)
            warm_counts.update(episodes=1,requests=len(warm['requests']),tokens=sum(len(r['output_token_ids']) for r in warm['requests'])); warm_wall+=warm['observation_end_s']
        outer=read(d/'runtime_observation.json'); old_done=max(r['arrival_s']+r['completion_latency_s'] for r in old); stages=defaultdict(list)
        check(outer['after']['runtime_records_count']-outer['before']['runtime_records_count']==len(rr),name+': outer observation count mismatch',issues)
        for call in calls:
            key='before_injection' if call['index']<action['engine_call'] else 'injection' if call['index']==action['engine_call'] else 'until_old_complete' if call['return_s']<=old_done else 'after_old_complete'
            stages[key].append(call)
        stage_stats={k:dict(engine_calls=[x['index'] for x in cc],wall_s=sum(x['return_s']-x['start_s'] for x in cc),
            **amounts([r for x in cc for r in rr if x['scheduler_step_start']<=r['context']['step_id']<x['scheduler_step_stop']])) for k,cc in stages.items()}
        check(sum(s['layer_calls'] for s in stage_stats.values())==len(rr),name+': engine/trace join mismatch',issues)
        row.update(**verified,**amounts(rr),requests=req,output_sha256=digest({k:r['output_token_ids'] for k,r in req.items()}),
            capture_wall_s=raw['observation_end_s'],cycle_wall_s=end-start,engine_wall_s=sum(x['return_s']-x['start_s'] for x in calls),
            flush_wall_s=(f['end_perf_ns']-f['start_perf_ns'])/1e9,flush=f,warmup_counts=dict(warm_counts),warmup_capture_wall_s=warm_wall,
            old_ttft_s=max(r['ttft_s'] for r in old),old_max_itl_s=max(r['max_itl_s'] for r in old),old_completion_s=max(r['completion_latency_s'] for r in old),
            old_boundary_itl_s=max(r['token_times_s'][4]-r['token_times_s'][3] for r in old),old_post_action_max_itl_s=max(max(b-a for a,b in zip(r['token_times_s'][3:],r['token_times_s'][4:])) for r in old),
            new_ttft_s=new[0]['ttft_s'],new_max_itl_s=new[0]['max_itl_s'],new_completion_s=new[0]['completion_latency_s'],snapshot_wall_s=action['snapshot_end_s']-action['snapshot_start_s'],
            stages=stage_stats,legacy_private_scratch_bytes=pager['scratch_bytes'],expert_scratch_bytes=res['expert_scratch_bytes'],cuda_memory=read(d/'cuda_memory.json'),
            held_before=outer['before']['runtime_records_count'],held_after=outer['after']['runtime_records_count'],
            gc_capture=overlap(events,origin,origin+raw['observation_end_s']),gc_cycle=overlap(events,start,end),gc_flush=overlap(events,f['start_perf_ns']/1e9,f['end_perf_ns']/1e9),
            gc_old_boundary=[overlap(events,origin+r['token_times_s'][3],origin+r['token_times_s'][4]) for r in old])
    repeat_sum=sum(r['cycle_wall_s'] for r in rows); whole=cycle['post_init_cycle_wall_s']
    check(abs(whole-(cycle['end_perf_ns']-cycle['start_perf_ns'])/1e9)<1e-8 and repeat_sum<=whole<=cell['process_wall_s'],label+': whole/process accounting mismatch',issues)
    return dict(label=label,mode=mode,block=declared['block'],status=cell['status'],issues=issues,rows=rows,source_sha256=source,shared_pool=shared,gc_observer={k:v for k,v in observer.items() if k!='events'},
        prompt_sha256=digest(workload['actual_prompt_token_ids']),gpu_checks=checks,trace_flushes=flushes,all_calls=len(trace),by_phase={p:amounts(rr) for p,rr in phases.items()},
        per_layer_measurement={n:amounts([r for r in trace if r['measurement'] and r['layer_name']==n]) for n in caps},
        process_wall_s=cell['process_wall_s'],post_init_cycle_wall_s=cycle['post_init_cycle_wall_s'],repeat_cycle_sum_s=sum(r['cycle_wall_s'] for r in rows),measurement_capture_sum_s=sum(r['capture_wall_s'] for r in rows),
        post_init_nonrepeat_wall_s=whole-repeat_sum,process_outside_cycle_wall_s=cell['process_wall_s']-whole,
        means={k:statistics.mean(r[k] for r in rows) for k in METRICS},description={k:dict(min=min(r[k] for r in rows),max=max(r[k] for r in rows),median=statistics.median(r[k] for r in rows)) for k in METRICS},
        warmup_counts={k:sum(r['warmup_counts'][k] for r in rows) for k in ('episodes','requests','tokens')})


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    root=args.input_dir; issues=[]; engines=[]; comparisons=[]; pairs=[]; execution=None; expected_rows=expected_pairs=expected_comparisons=0
    try:
        protocol=read(root/'protocol.json'); declared=read(root/'run_cells.json'); execution=read(root/'results/execution.json')
        check(declared==protocol['sequence'] and bool(declared) and all(c['mode'] in MODES for c in declared)
              and len({c['label'] for c in declared})==len(declared),'frozen matrix mismatch',issues)
        blocks=list(dict.fromkeys(c['block'] for c in declared)); repeats=protocol['workload']['episodes_per_fresh_engine']
        check(type(repeats) is int and repeats>0,'invalid frozen repeat count',issues); expected_rows=len(declared)*repeats
        selected_pairs=protocol.get('comparisons',PAIRS)
        check(len({tuple(pair) for pair in selected_pairs})==len(selected_pairs) and all(len(pair)==2 and pair[0]!=pair[1] and set(pair)<=MODES for pair in selected_pairs),'invalid comparison pairs',issues)
        for block in blocks:
            modes=[c['mode'] for c in declared if c['block']==block]
            check(len(set(modes))==len(modes),'duplicate mode within document block',issues)
            expected_comparisons+=sum(a in modes and b in modes for a,b in selected_pairs)
        expected_pairs=expected_comparisons*repeats
        check(execution['status']=='COMPLETE' and [c['label'] for c in execution['cells']]==[c['label'] for c in declared],'declared engine completion/order mismatch',issues)
        for cell in execution['cells']:
            try:
                d=next(d for d in declared if d['label']==cell['label']); engine=engine_result(root,cell,d,protocol); engines.append(engine); issues.extend(engine['issues'])
            except ERRORS as exc: issues.append(cell['label']+': '+repr(exc)); engines.append(dict(label=cell['label'],status=cell.get('status'),rows=[],issues=[repr(exc)]))
        for block in blocks:
            group={e['mode']:e for e in engines if e.get('block')==block}
            check(len({e['prompt_sha256'] for e in group.values()})==1,'block prompt mismatch',issues)
            for a,b in selected_pairs:
                if a not in group or b not in group: continue
                ea,eb=group[a],group[b]; comparisons.append(dict(block=block,a=a,b=b,**difference(ea['means'],eb['means'],METRICS),engine_totals=difference(ea,eb,('process_wall_s','post_init_cycle_wall_s','repeat_cycle_sum_s','measurement_capture_sum_s'))))
                for i,(ra,rb) in enumerate(zip(ea['rows'],eb['rows'])):
                    pairs.append(dict(block=block,ordinal=i,a=a,b=b,**difference(ra,rb,METRICS),
                        equality={k:ra[k]==rb[k] for k in ('route_sha256','trace_sha256','output_sha256','final_cache_sha256')},
                        request_deltas={rid:{k:rb['requests'][rid][k]-ra['requests'][rid][k] for k in ('ttft_s','tpot_s','max_itl_s','completion_latency_s')} for rid in ra['requests']}))
    except ERRORS as exc: issues.append('campaign: '+repr(exc))
    rows=[r for e in engines for r in e['rows']]; warmups=sum(e.get('warmup_counts',{}).get('episodes',0) for e in engines)
    check(len(rows)==expected_rows and warmups==expected_rows*len(WARMUPS) and len(pairs)==expected_pairs and len(comparisons)==expected_comparisons and expected_comparisons>0,'incomplete measurements/warmups/comparisons',issues)
    output=dict(status='ISSUES' if issues else 'DESCRIPTIVE_NATIVE_SHARED_POOL_EXECUTION',issues=issues,execution=execution,engines=engines,repeat_pairs=pairs,engine_mean_comparisons=comparisons,
        counts=dict(measurements=len(rows),warmups=warmups,requests=sum(len(r.get('requests',{})) for r in rows),tokens=sum(len(q['output_token_ids']) for r in rows for q in r.get('requests',{}).values())),
        scope=['Fresh engines and dependent repeats follow the frozen matrix; compare only within each document block. Block and arm order may be confounded. No CI, significance or population noise floor.',
            'Each arm executes its own future state; sequential LRU/copy checks use only that arm actual rows. Neither cross-arm output equality nor quality is assumed.',
            'H2D and D2D are separate. Fullstage D2D is entry-hit staging then private writeback; the latter is split using verified copy counts at12MiB/expert. Canonical groups simulate own current ordinary20 or21 only.',
            'Unique expert storage is384 slots: split/oneshot private336+stage48, fullstage private320+stage64. Fullstage uses all64 staged weights with global64/mapNone; no previous-stage cache or reused future route.',
            'Capture includes observations; repeat cycle includes reset/five warmups/writes/flush. Its own cycle_progress write and shared tail remain in whole-cycle/process totals. Retain all slow/failed rows.',
            'GC/load/host intervals overlap wall; never subtract or sum them as benefit. GC observer ends before export/shutdown; GPU checks are boundary snapshots. Qualification overhead is disabled.'])
    with args.out.open('x') as f: json.dump(output,f,indent=2,allow_nan=False); f.write('\n')
    if issues: raise SystemExit('issues retained in output')


if __name__=='__main__': main()
