"""Recompute retained native event episodes; never overwrite raw or select repeats."""
import argparse
from collections import Counter
import json
from pathlib import Path
from analyze_native_pager_transfer import read, requests


SUM_KEYS = ('actual_bytes', 'unique_bytes', 'extra_bytes', 'groups', 'load_section_ms',
            'host_apply_ms', 'route_to_host_ms', 'engine_wall_s')


def totals(rows):
    return {key: sum(r[key] for r in rows) for key in SUM_KEYS}


def relative_lru(state):
    return {**state, 'cache': {name: {**s, 'lru_clock': 0,
        'lru_tick': [t-s['lru_clock'] for t in s['lru_tick']]}
        for name, s in state['cache'].items()}}


def cell(path, prediction):
    raw, workload, pager = (read(path/name) for name in ('raw.json', 'workload.json', 'pager_summary.json'))
    config = read(path/'config.json'); issues = []
    check = lambda ok, message: issues.append(message) if not ok else None
    check(raw['status']=='COMPLETE', 'episode incomplete: '+str(raw.get('error')))
    actions, rows = raw['event_actions'], {r['request_id']:r for r in raw['requests']}
    if len(actions)!=1: raise ValueError('exactly one retained event action required')
    action=actions[0]; old=list(raw['event_arrival']['after_output_tokens']); new=action['request_id']
    aliases=raw['internal_to_source']; before=action['before']
    state={**before, 'running':[aliases[r] for r in before['running']],
        'waiting':[aliases[r] for r in before['waiting']],
        'requests':{aliases[r]:v for r,v in before['requests'].items()}}
    served=[i for i,s in enumerate(workload['source_requests']) if rows[s['request_id']]['status']!='not_injected']
    active_work={**workload, 'source_requests':[workload['source_requests'][i] for i in served],
        'actual_prompt_token_ids':[workload['actual_prompt_token_ids'][i] for i in served]}
    metrics, reqissues=requests({**raw, 'requests':[r for r in raw['requests'] if r['status']!='not_injected']},active_work)
    issues.extend(reqissues)
    for rid in rows:
        if rid not in metrics:
            metrics[rid]=dict(status=rows[rid]['status'],arrival_s=None,ttft_s=None,tpot_s=None,
                max_itl_s=None,completion_latency_s=None,output_token_ids=[],token_times_s=[])
    check(len(old)==2 and len(rows)==3 and [rows[r]['prompt_tokens'] for r in old+[new]]==[32,32,128], 'request identity/length')
    check(all(len(rows[r]['output_token_ids'])==16 and rows[r]['status']=='completed' for r in old), 'old completion lengths')
    check(len(rows[new]['output_token_ids'])==(8 if action['inject'] else 0), 'new output length')
    check(rows[new]['status']==('completed' if action['inject'] else 'not_injected'), 'new completion/hold status')
    check(all(before['old_output_tokens'][r]==rows[r]['output_token_ids'][:4] and len(before['old_output_tokens'][r])==4 for r in old), 'old fourth-token prefix')
    release=max(rows[r]['token_times_s'][3] for r in old)
    check(action['release_s']==release and rows[new]['arrival_s']==(release if action['inject'] else None), 'event arrival clock')
    check(release<=action['detected_s']<=action['snapshot_start_s']<=action['snapshot_end_s']<=action['threshold_applied_s']<=action['before_add_end_s'], 'snapshot ordering')
    if action['inject']:
        check(action['before_add_end_s']<=rows[new]['admission_s']<=rows[new]['engine_add_return_s'], 'event add ordering')
    else:
        check(rows[new]['admission_s'] is None and rows[new]['completion_s'] is None, 'hold marker was admitted/completed')
    trace=[json.loads(s) for s in (path/'pager/calls.jsonl').read_text().splitlines()]
    measured=[r for r in trace if r['context']['phase']=='measurement']
    sizes={r['layer_name']:r['pinned_bytes']//r['num_experts'] for r in pager['layers']}
    cache={name:set(s['slot_to_expert'])-{-1} for name,s in pager['measurement_initial_cache'].items()}
    bystep={}; signature=[]
    for r in measured:
        ctx=r['context']; groups=r['groups']; required=[e for g in groups for e in g['required_experts']]
        loaded=[e for g in groups for e in g['loaded_experts']]; resident=set(r['entry_resident_experts'])
        missing=set(required)-resident; size=sizes[r['layer_name']]
        check(r['grouping_axis']=='expert' and r['measurement'] and r['status']=='complete', 'invalid expert measurement call')
        check(len(required)==len(set(required)) and set(required)==set(r['active_experts']), 'required disjoint coverage')
        check(Counter(loaded)==Counter(missing) and set(r['missing_experts'])==missing, 'loads differ from unique entry misses')
        check(resident==cache[r['layer_name']], 'entry residency transition')
        for g in groups:
            need=set(g['required_experts']); loads=set(g['loaded_experts']); victims=set(g['evicted_experts'])
            check(g['start']==0 and g['stop']==r['rows'] and len(need)==g['unique_experts']<=pager['cap'], 'partial rows/cap')
            check(loads==need-resident and victims<=resident and not victims&need, 'group miss/victim state')
            check(len(loads)==len(g['loaded_experts'])==g['miss'] and g['evict']==len(victims) and g['weight_copy_bytes']==g['miss']*size, 'group counters/bytes')
            resident=(resident-victims)|loads
            check(need<=resident and len(resident)<=pager['cap'], 'post-group residency')
        cache[r['layer_name']]=resident
        check(r['weight_copy_bytes']==len(loaded)*size, 'call bytes')
        bystep.setdefault(ctx['step_id'],[]).append(r)
        signature.append(dict(step=ctx['step_id'],layer=r['layer_name'],rows=r['rows'],physical_rows=[(aliases[x['internal_request_id']],x['computed_position']) for x in ctx['rows']],active=r['active_experts'],
            entry=r['entry_resident_experts'],groups=[{k:g[k] for k in ('required_experts','loaded_experts','evicted_experts')} for g in groups]))
    steps=raw['scheduler_steps']; engines=raw['engine_calls']; calls=[]; seen=[]
    for e in engines:
        selected=steps[e['scheduler_step_start']:e['scheduler_step_stop']]; seen.extend(s['step'] for s in selected)
        check(e['index']==len(calls) and e['returned'] and e['start_s']<=e['return_s'], 'engine call index/status/time')
        rs=[r for s in selected for r in bystep.get(s['step'],[])]
        for s in selected:
            wanted=Counter((r['internal_request_id'],pos) for r in s['scheduled'] for pos in range(r['scheduled_start_computed'],r['computed_after']))
            matched=bystep.get(s['step'],[])
            check(e['start_s']<=s['start_s']<=s['end_s']<=e['return_s'], 'scheduler/engine time nesting')
            check(len(matched)==len(sizes) and {r['layer_name'] for r in matched}==set(sizes), 'step layer coverage')
            check(all(Counter((x['internal_request_id'],x['computed_position']) for x in r['context']['rows'])==wanted and
                r['context']['row_request_order_verified'] and r['context']['valid_row_start']==0 and
                r['context']['valid_row_stop']==r['context']['expected_rows']==sum(wanted.values())<=r['rows'] for r in matched), 'physical row alignment')
            check(not s['preempted_request_ids'] and not s['recomputed_tokens'] and all(r['computed_adjustment']==0 for r in s['scheduled']), 'preemption/recompute')
            live=[rid for rid in old if rows[rid]['token_times_s'][0]<=e['start_s']<rows[rid]['completion_s']]
            check(all(sum(r['decode_tokens'] for r in s['scheduled'] if r['request_id']==rid)==1 for rid in live), 'live old decode must advance one')
        scheduled=[r for s in selected for r in s['scheduled']]
        actual=sum(r['weight_copy_bytes'] for r in rs)
        unique=sum(len(set(r['active_experts'])-set(r['entry_resident_experts']))*sizes[r['layer_name']] for r in rs)
        calls.append(dict(index=e['index'],start_s=e['start_s'],return_s=e['return_s'],steps=[s['step'] for s in selected],
            scheduled_tokens=sum(r['scheduled_tokens'] for r in scheduled),prefill_rows=sum(r['prefill_tokens'] for r in scheduled),
            old_decode_rows=sum(r['decode_tokens'] for r in scheduled if r['request_id'] in old),new_prefill_rows=sum(r['prefill_tokens'] for r in scheduled if r['request_id']==new),
            new_decode_rows=sum(r['decode_tokens'] for r in scheduled if r['request_id']==new),actual_bytes=actual,unique_bytes=unique,extra_bytes=actual-unique,
            groups=sum(len(r['groups']) for r in rs),load_section_ms=sum(g['load_cuda_span_ms'] for r in rs for g in r['groups']),
            host_apply_ms=sum(r['host_apply_ms'] for r in rs),route_to_host_ms=sum(r['route_to_host_ms'] for r in rs),engine_wall_s=e['return_s']-e['start_s']))
    check(seen==[s['step'] for s in steps] and set(bystep)==set(seen), 'engine/step/call coverage')
    check(sum(c['actual_bytes'] for c in calls)==pager['measurement']['weight_copy_bytes'], 'pager summary bytes')
    for output in raw['output_events']:
        matching=[e for e in engines if e['return_s']==output['received_s'] and output['external_request_id'] in e['output_request_ids']]
        check(len(matching)==1 and output['prefix_valid'] and output['chunk_size']==1, 'receipt/engine or token prefix/chunk')
    first=calls[action['engine_call']]; expected=2+config['injection_chunk']
    check(first['scheduled_tokens']==expected and first['old_decode_rows']==2 and first['new_prefill_rows']==config['injection_chunk'], 'first action shape')
    check(action['before_add_end_s']<=first['start_s'] and (not action['inject'] or rows[new]['engine_add_return_s']<=first['start_s']), 'snapshot/add/first-call ordering')
    last_prefill=max((c['index'] for c in calls if c['index']>=first['index'] and c['new_prefill_rows']),default=first['index']-1)
    old_done=max(rows[r]['completion_s'] for r in old)
    buckets={'preaction':[c for c in calls if c['index']<first['index']],
        'action_through_new_prefill':[c for c in calls if first['index']<=c['index']<=last_prefill],
        'post_prefill_decode':[c for c in calls if c['index']>last_prefill and c['index']>=first['index']]}
    key='chunk'+str(config['injection_chunk']) if action['inject'] else 'hold'
    predicted=prediction['predictions'][key]['predicted_unique_copy_bytes_all_layers']
    first_layers={r['layer_name']:dict(active=len(r['active_experts']),entry_hits=len(set(r['active_experts'])&set(r['entry_resident_experts'])),missing=len(r['missing_experts']),actual_bytes=r['weight_copy_bytes']) for r in measured if r['context']['step_id'] in first['steps']}
    resource=read(path/'measurement_resources.json'); memory=read(path/'cuda_memory.json')
    warmed={(r['layer_name'],r['rows']) for r in trace if r['context']['phase'].startswith('warmup')}
    result=dict(path=str(path),status=raw['status'],issues=issues,config=config,requests=metrics,
        first_action_call=first,all_calls=calls,all_calls_totals=totals(calls),decomposition={k:totals(v) for k,v in buckets.items()},
        calls_after_old_completion=[c['index'] for c in calls if c['start_s']>=old_done],
        after_old_completion_totals=totals([c for c in calls if c['start_s']>=old_done]),
        old_cross_action_itl_s={r:rows[r]['token_times_s'][4]-rows[r]['token_times_s'][3] for r in old},
        old_postaction_max_itl_s={r:max(b-a for a,b in zip(rows[r]['token_times_s'][3:],rows[r]['token_times_s'][4:])) for r in old},
        whole_wall_s=raw['observation_end_s'],postaction_wall_s=raw['observation_end_s']-release,new_ttft_s=metrics[new]['ttft_s'],event_release_s=release,
        snapshot_wall_s=action['snapshot_end_s']-action['snapshot_start_s'],release_to_first_call_start_s=first['start_s']-release,
        preaction_old_outputs=before['old_output_tokens'],completed_requests=sum(r['status']=='completed' for r in rows.values()),generated_tokens=sum(len(r['output_token_ids']) for r in rows.values()),
        prediction=dict(frozen=prediction['predictions'][key],predicted_bytes=predicted,actual_bytes=first['actual_bytes'],relative_error_actual_over_predicted_pct=100*(first['actual_bytes']/predicted-1),observed_first_layers=first_layers),
        resources=dict(**resource,scratch_bytes=pager['scratch_bytes'],pinned_master_bytes=pager['pinned_bytes'],cuda_memory=memory),
        all_measured_layer_shapes_seen_in_warmup=all((r['layer_name'],r['rows']) in warmed for r in measured),
        final_resident_sets={k:sorted(v) for k,v in cache.items()},final_full_slot_map_lru=None)
    prefix_steps={s for c in calls if c['index']<first['index'] for s in c['steps']}
    return result,dict(state=state,initial=pager['measurement_initial_cache'],signature=signature,prefix_signature=[r for r in signature if r['step'] in prefix_steps],workload=workload,
        sources=read(path/'environment.json')['sources'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--prediction',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--update-derived',action='store_true');args=p.parse_args()
    if args.update_derived and (args.out.name!='analysis.json' or args.out.parent.resolve()!=args.input_dir.parent.resolve()):
        p.error('--update-derived only updates analysis.json beside the raw results directory')
    execution=read(args.input_dir/'execution.json');prediction=read(args.prediction);cells={};norms={}
    for entry in execution['cells']:
        name=entry['label']
        try: cells[name],norms[name]=cell(args.input_dir/name,prediction)
        except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
            status=args.input_dir/name/'status.json'
            cells[name]=dict(status='ANALYSIS_UNAVAILABLE',retained_execution_status=read(status) if status.exists() else None,issues=[str(exc)])
        cells[name]['execution']=entry
    comparisons=[];names=list(norms)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            x,y=cells[a],cells[b];nx,ny=norms[a],norms[b]
            token_diff={r:next((i+1 for i,(u,v) in enumerate(zip(x['requests'][r]['output_token_ids'],y['requests'][r]['output_token_ids'])) if u!=v),None) for r in x['requests']}
            comparisons.append(dict(a=a,b=b,same_workload=nx['workload']==ny['workload'],same_source_hashes=nx['sources']==ny['sources'],
                same_measurement_initial_slots_lru=nx['initial']==ny['initial'],same_preaction_full_logical_state=nx['state']==ny['state'],
                same_preaction_state_with_relative_lru=relative_lru(nx['state'])==relative_lru(ny['state']),
                same_fixed_resources=all(x['resources'][k]==y['resources'][k] for k in ('actual_unique_kv_storage_bytes','scratch_bytes','pinned_master_bytes','expert_cap','pool_num_gpu_blocks')),
                same_preaction_recorded_routes_groups=nx['prefix_signature']==ny['prefix_signature'],same_measured_recorded_routes_groups=nx['signature']==ny['signature'],outputs_equal={r:x['requests'][r]['output_token_ids']==y['requests'][r]['output_token_ids'] for r in x['requests']},first_different_served_token_1based=token_diff,
                same_task=x['config']['injection_chunk']!=0 and y['config']['injection_chunk']!=0,
                whole_wall_delta_b_minus_a_s=y['whole_wall_s']-x['whole_wall_s'],whole_wall_ratio_b_over_a=y['whole_wall_s']/x['whole_wall_s'],postaction_wall_delta_b_minus_a_s=y['postaction_wall_s']-x['postaction_wall_s'],new_ttft_delta_b_minus_a_s=y['new_ttft_s']-x['new_ttft_s'] if x['new_ttft_s'] is not None and y['new_ttft_s'] is not None else None))
    output=dict(schema_version=1,prediction_source=str(args.prediction),prediction_precedes_all_runs=all(prediction['created_unix_s']<e['started_unix_s'] for e in execution['cells']),cells=cells,comparisons=comparisons,
        boundaries=['Same logical KV block/state metadata does not establish tensor bitwise equality.',
            'Hold has two completed requests; its wall is not same-task throughput.',
            'Every repeat and failed analysis is retained. Prediction error tests the uniform/independence assumptions, not the problem family.',
            'Old post-action maxITL includes the fourth-to-fifth-token boundary. All request metrics subtract actual event arrival.',
            'Engine wall, host_apply and CUDA load sections overlap; never add them. Snapshot costs remain in request and whole wall.',
            'Traffic is shared across old/new rows. Final resident sets are reconstructed; final full slot/map/LRU was not dumped.',
            'Per-token route IDs/weights are not retained; same routes/groups means recorded expert unions/groups only.',
            'Interpret output identity, first-boundary latency, later latency and complete work independently; none substitutes for another.'])
    with args.out.open('w' if args.update_derived else 'x') as f: json.dump(output,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
