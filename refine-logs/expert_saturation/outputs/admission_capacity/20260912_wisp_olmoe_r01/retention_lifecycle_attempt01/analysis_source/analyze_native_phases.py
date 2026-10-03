"""Small phase reader for retained same-engine repeats and release-baseline ABBA."""
import argparse, json
from pathlib import Path
from collections import Counter
from analyze_native_pager_transfer import requests

def read(p): return json.loads(p.read_text())
def stat(text): return {k:int(v) for k,v in (s.split() for s in text.splitlines())} if text else {}

def retention_stats(records, cap):
    """Count layer-expert occurrences; ensure touches are not demand hits."""
    if not records or any('retention' not in r for r in records):
        return dict(status='UNAVAILABLE', groups=None, eligible=None, applied=None)
    counts=Counter();hist={k:Counter() for k in ('protected','decode_overlap','entry_overlap')};valid=True
    for r in records:
        t=r['retention'];active=set(r['active_experts']);entry=set(r['entry_resident_experts'])
        chosen=set(t['chosen_protected_experts']);decode=set(t['real_decode_experts'] or [])
        counts.update(layer_calls=1,groups=len(r['groups']),eligible=int(t['eligible']),applied=int(t['applied']),
            protected=len(chosen),decode_overlap=len(chosen&decode),entry_overlap=len(chosen&entry),
            entry_active_hits=len(active&entry))
        for key,value in (('protected',chosen),('decode_overlap',chosen&decode),('entry_overlap',chosen&entry)):
            hist[key][len(value)]+=1
        valid &= chosen<=active and set(t['protected_decode_intersection'])==chosen&decode and set(t['protected_entry_resident'])==chosen&entry
        valid &= not chosen or (t['eligible'] and len(chosen)==len(decode)<cap)
        if t['mode']=='matched_hash' and chosen:valid &= len(chosen&entry)==len(decode&entry)
        if t['mode']=='decode' and chosen:valid &= chosen==decode
        if 'row_topk_experts' in r:
            rows=r['row_topk_experts'];metadata=r['context']['rows']
            valid &= set(e for row in rows for e in row)==active and len(rows)==len(metadata)
            if t['eligible']:
                observed_decode=set(e for row,m in zip(rows,metadata) if m['computed_position']>=m['prompt_tokens'] for e in row)
                valid &= observed_decode==decode
                if t['mode']=='frequency':
                    frequencies=Counter(e for row in rows for e in row)
                    valid &= chosen==set(sorted(active,key=lambda e:(-frequencies[e],e))[:len(decode)])
        current=set(entry);held=chosen&entry if t.get('order','early')=='late' else set()
        for g in r['groups']:
            execute=set(g['required_experts']);ensure=set(g['ensure_experts']);loads=set(g['loaded_experts'])
            held |= chosen&execute
            valid &= ensure==execute|held and len(ensure)<=cap and set(g['protected_after'])==held
            valid &= loads==ensure-current and g['miss']==len(loads)
            current=(current-set(g['evicted_experts']))|loads
            valid &= ensure<=current and len(current)<=cap
            counts['repeated_ensure_touches']+=len(ensure-execute)
        valid &= current==set(t['final_resident_experts']) and chosen<=current
    return dict(status='OBSERVED',**counts,invariants_valid=bool(valid),count_histograms={k:dict(v) for k,v in hist.items()},
        fallback_reasons=dict(Counter(r['retention']['fallback_reason'] for r in records if r['retention']['fallback_reason'])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--update-derived',action='store_true');a=p.parse_args()
    if a.update_derived and (a.out.name!='analysis.json' or a.out.parent.resolve()!=a.input_dir.resolve().parents[1]):p.error('update is restricted to analysis.json beside results/')
    entries=read(a.input_dir/'episodes.json');pager=read(a.input_dir/'pager_summary.json');workload=read(a.input_dir/'workload.json')
    trace=[json.loads(s) for s in (a.input_dir/'pager/calls.jsonl').read_text().splitlines()]
    sizes={r['layer_name']:r['pinned_bytes']//r['num_experts'] for r in pager['layers']}
    result={};norm={}
    for entry in entries:
        phase=entry['phase'];name=phase.split('/')[0];root=a.input_dir/name
        raw=read(root/'raw.json');res=read(root/'measurement_resources.json');reset=read(root/'reset.json')
        records=[r for r in trace if r['context']['phase']==phase];aliases=raw['internal_to_source'];action=raw['event_actions'][0]
        before=action['before'];old=list(before['old_output_tokens']);rows={r['request_id']:r for r in raw['requests']};new=action['request_id']
        metrics,issues=requests(raw,workload);old_done=max(rows[r]['completion_s'] for r in old)
        logical={k:v for k,v in before.items() if k not in ('requests','running','waiting')}
        logical.update(requests={aliases[k]:{n:v for n,v in r.items() if n!='kv_block_ids'} for k,r in before['requests'].items()},
            running=[aliases[k] for k in before['running']],waiting=[aliases[k] for k in before['waiting']])
        blocks={aliases[k]:r['kv_block_ids'] for k,r in before['requests'].items()}
        calls=raw['engine_calls'];cpu_keys=list(calls[0]['cpu_delta']);cpu={k:sum(c['cpu_delta'][k] for c in calls) if all(c['cpu_delta'][k] is not None for c in calls) else None for k in cpu_keys}
        env0=res['environment_before'];env1=read(root/'environment_after.json')['cpu']
        cg0=stat(env0['proc']['/sys/fs/cgroup/cpu.stat']);cg1=stat(env1['proc']['/sys/fs/cgroup/cpu.stat'])
        cgroup={k:cg1[k]-v for k,v in cg0.items() if k in cg1}
        valid_cpu=all(c['cpu_observation_start_s']<=c['start_s']<=c['return_s']<=c['cpu_observation_end_s'] for c in calls)
        checked=[];signature=[]
        for r in records:
            req=[e for g in r['groups'] for e in g['required_experts']];loads=[e for g in r['groups'] for e in g['loaded_experts']]
            missing=set(r['active_experts'])-set(r['entry_resident_experts'])
            checked.append(r['measurement'] and r['status']=='complete' and set(req)==set(r['active_experts']) and len(req)==len(set(req)) and
                Counter(loads)==Counter(missing) and len(loads)*sizes[r['layer_name']]==r['weight_copy_bytes'])
            signature.append(dict(step=r['context']['step_id'],layer=r['layer_name'],rows=[(aliases[x['internal_request_id']],x['computed_position']) for x in r['context']['rows']],
                active=r['active_experts'],entry=r['entry_resident_experts'],groups=[{k:g[k] for k in ('required_experts','loaded_experts','evicted_experts','weight_copy_bytes')} for g in r['groups']]))
            if 'retention' in r:
                signature[-1].update(ensure_groups=[g['ensure_experts'] for g in r['groups']],
                    protected_groups=[g['protected_after'] for g in r['groups']],final_entry=r['retention']['final_resident_experts'],
                    prompt_tokens_by_row=[x['prompt_tokens'] for x in r['context']['rows']])
        actual=sum(r['weight_copy_bytes'] for r in records);unique=sum(len(set(r['active_experts'])-set(r['entry_resident_experts']))*sizes[r['layer_name']] for r in records)
        percall=[dict(index=c['index'],start_s=c['start_s'],return_s=c['return_s'],wall_s=c['return_s']-c['start_s'],observation_envelope_s=c['observation_envelope_s'],cpu_delta=c['cpu_delta'],
            scheduled_tokens=sum(s['total_scheduled_tokens'] for s in raw['scheduler_steps'][c['scheduler_step_start']:c['scheduler_step_stop']]),
            new_prefill_tokens=sum(r['prefill_tokens'] for s in raw['scheduler_steps'][c['scheduler_step_start']:c['scheduler_step_stop']] for r in s['scheduled'] if r['request_id']==new),
            bytes=sum(r['weight_copy_bytes'] for r in records if c['scheduler_step_start']<=r['context']['step_id']<c['scheduler_step_stop'])) for c in calls]
        prefix_steps={s['step'] for c in calls[:16] for s in raw['scheduler_steps'][c['scheduler_step_start']:c['scheduler_step_stop']]}
        post_old=[c for c in percall if c['start_s']>=old_done];post_old_prefill=[c for c in post_old if c['new_prefill_tokens']]
        completed_call=next(c for c in calls if c['return_s']==old_done)
        computed=min(rows[new]['prompt_tokens'],max(r['computed_after'] for s in raw['scheduler_steps'][:completed_call['scheduler_step_stop']] for r in s['scheduled'] if r['request_id']==new))
        release=raw.get('prefill_release_actions',[]);variant=entry.get('variant')
        if variant:
            release_expected=variant=='release8' or variant.startswith('retention_')
            valid=completed_call['index']==15 and computed==96 and all(rows[r]['status']=='completed' and len(rows[r]['output_token_ids'])==16 for r in old)
            valid=valid and len(release)==int(release_expected) and [c['new_prefill_tokens'] for c in post_old_prefill]==([32] if release_expected else [8]*4)
            if release:
                t=release[0];valid=valid and t['engine_call']==16 and (t['computed_tokens'],t['prompt_tokens'],t['remaining_prefill_tokens'],t['threshold_before'],t['threshold_after'])==(96,128,32,8,0)
                valid=valid and all(r['output_tokens']==16 and r['status']=='completed' and not r['present_in_scheduler'] for r in t['old_requests'].values()) and t['eligible_s']==old_done<=t['start_s']<=t['applied_s']<=t['end_s']<=post_old[0]['start_s']
            if not valid:issues.append('old-completion release boundary/effect mismatch')
        norm[name]=dict(logical=logical,blocks=blocks,cache=read(root/'measurement_initial_cache.json'),alloc=res['allocations'],signature=signature,
            outputs={k:r['output_token_ids'] for k,r in rows.items()},free_queue=reset['free_kv_block_ids_in_order'],
            prefix_signature=[r for r in signature if r['step'] in prefix_steps],prefix_old_outputs={r:rows[r]['output_token_ids'] for r in old},
            future_signature=[r for r in signature if r['step'] not in prefix_steps],
            decode_signature={str(r['rows'][0][1])+':'+r['layer']:r for r in signature if len(r['rows'])==1 and r['rows'][0][0]==new and r['rows'][0][1]>=rows[new]['prompt_tokens']})
        result[name]=dict(status=raw['status'],phase=phase,variant=variant,issues=issues,requests=metrics,calls=len(calls),layer_calls=len(records),phase_bytes_valid=all(checked),
            whole_wall_s=raw['observation_end_s'],engine_wall_s=sum(c['return_s']-c['start_s'] for c in calls),
            observation_envelope_s=sum(c['observation_envelope_s'] for c in calls),cpu_window_order_valid=valid_cpu,cpu=cpu,
            runqueue_stat_enabled=env0['proc']['/proc/sys/kernel/sched_schedstats'],cgroup_delta=cgroup,cpu_max=env0['proc']['/sys/fs/cgroup/cpu.max'],
            cpu_environment_before=env0,cpu_environment_after=env1,per_call=percall,
            old_boundary_itl_s={rid:rows[rid]['token_times_s'][4]-rows[rid]['token_times_s'][3] for rid in old},
            actual_bytes=actual,unique_bytes=unique,extra_bytes=actual-unique,load_section_ms=sum(g['load_cuda_span_ms'] for r in records for g in r['groups']),
            host_apply_ms=sum(r['host_apply_ms'] for r in records),reset_allocations_unchanged=reset['allocations_unchanged'],kv_free_queue_modified=reset['kv_free_queue_modified'],
            snapshot_wall_s=action['snapshot_end_s']-action['snapshot_start_s'],memory=read(root/'cuda_memory.json'),
            fixed_resources=dict(actual_kv_bytes=res['actual_unique_kv_storage_bytes'],scratch_bytes=pager['scratch_bytes'],pinned_bytes=pager['pinned_bytes'],expert_cap=res['expert_cap']),
            gpu_boundaries=dict(before_reset=reset['gpu'],after_measurement=read(root/'environment_after.json')['gpu']),
            old_done_s=old_done,new_prompt_computed_at_old_done=computed,release_actions=release,
            release_action_wall_s=sum(t['end_s']-t['start_s'] for t in release),
            prefix_calls_0_through_15=dict(bytes=sum(c['bytes'] for c in percall[:16]),engine_wall_s=sum(c['wall_s'] for c in percall[:16])),
            post_old=dict(calls=[c['index'] for c in post_old],bytes=sum(c['bytes'] for c in post_old),engine_wall_s=sum(c['wall_s'] for c in post_old),
                wall_to_observation_end_s=raw['observation_end_s']-old_done,prefill_steps=[c['index'] for c in post_old_prefill],
                prefill_tokens=[c['new_prefill_tokens'] for c in post_old_prefill],prefill_bytes=sum(c['bytes'] for c in post_old_prefill),prefill_engine_wall_s=sum(c['wall_s'] for c in post_old_prefill)))
        if variant and variant.startswith('retention_'):
            mode=res.get('group_retention',variant.removeprefix('retention_'));order=res.get('group_retention_order','early')
            joined={};stages={k:[] for k in ('pre_action','first_action','later_until_old_complete','post_old_prefill','post_old_decode')}
            for c,pc in zip(calls,percall):
                rr=[r for r in records if c['scheduler_step_start']<=r['context']['step_id']<c['scheduler_step_stop']]
                joined[c['index']]=rr;pc['retention']=retention_stats(rr,res['expert_cap'])
                pc['unique_bytes']=sum(len(set(r['active_experts'])-set(r['entry_resident_experts']))*sizes[r['layer_name']] for r in rr)
                pc['extra_bytes']=pc['bytes']-pc['unique_bytes'];pc['groups']=sum(len(r['groups']) for r in rr)
                pc['load_section_ms']=sum(g['load_cuda_span_ms'] for r in rr for g in r['groups'])
                key=('pre_action' if c['index']<action['engine_call'] else 'first_action' if c['index']==action['engine_call']
                     else 'later_until_old_complete' if c['index']<=completed_call['index'] else 'post_old_prefill' if pc['new_prefill_tokens'] else 'post_old_decode')
                stages[key].append(pc)
            summary=retention_stats(records,res['expert_cap'])
            if not summary.get('invariants_valid') or any(r['retention']['mode']!=mode for r in records):issues.append('retention fields/invariants unavailable or inconsistent')
            if any(r['retention'].get('order','early')!=order for r in records):issues.append('retention order differs from measured policy')
            join_valid=sum(map(len,joined.values()))==len(records) and all(len(joined[c['index']])==(len(pager['layers']) if c['scheduled_tokens'] else 0) for c in percall)
            if not join_valid:issues.append('retention trace does not cover actual engine calls exactly')
            result[name]['retention']=dict(mode=mode,**summary,stages={k:dict(engine_calls=[c['index'] for c in cc],
                **{field:sum(c[field] for c in cc) for field in ('bytes','unique_bytes','extra_bytes','groups','wall_s','load_section_ms')},
                retention=retention_stats([r for c in cc for r in joined[c['index']]],res['expert_cap'])) for k,cc in stages.items()},
                engine_call_join_valid=join_valid,layer_details=[dict(engine_call=c,step=r['context']['step_id'],layer=r['layer_name'],
                    actual_bytes=r['weight_copy_bytes'],groups=len(r['groups']),**r['retention']) for c,rr in joined.items() for r in rr])
            if all('row_topk_experts' in r for r in records):
                result[name]['retention'].update(order=order,raw_row_routes_present=True)
            first_step=calls[action['engine_call']]['scheduler_step_start']
            norm[name]['pre_action_signature']=[r for r in signature if r['step']<first_step]
            norm[name]['retention_resources']={k:res[k] for k in ('actual_unique_kv_storage_bytes','pool_num_gpu_blocks','free_blocks','expert_cap','cpu_affinity')}
            result[name]['post_action_old_max_itl_s']={rid:max(b-a for a,b in zip(rows[rid]['token_times_s'][3:],rows[rid]['token_times_s'][4:])) for rid in old}
    first=next(iter(norm))
    for name,n in norm.items(): result[name]['equality_to_first']={k:n[k]==norm[first][k] for k in n}
    comparisons=[];names=list(norm)
    for i,n in enumerate(names):
        for m in names[i+1:]:
            x,y=result[n],result[m];dx,dy=norm[n]['decode_signature'],norm[m]['decode_signature'];matched=set(dx)&set(dy)
            differences=[dict(position_layer=k,a=dx[k]['active'],b=dy[k]['active']) for k in sorted(matched) if dx[k]['active']!=dy[k]['active']]
            comparisons.append(dict(a=n,b=m,equality={k:norm[n][k]==norm[m][k] for k in norm[n]},
                whole_wall_delta_b_minus_a_s=y['whole_wall_s']-x['whole_wall_s'],post_old_wall_delta_b_minus_a_s=y['post_old']['wall_to_observation_end_s']-x['post_old']['wall_to_observation_end_s'],
                post_old_bytes_delta_b_minus_a=y['post_old']['bytes']-x['post_old']['bytes'],
                matched_decode_layers=len(matched),decode_keys_equal=set(dx)==set(dy),decode_route_set_differences=differences,
                decode_entry_cache_set_difference_count=sum(dx[k]['entry']!=dy[k]['entry'] for k in matched),
                request_deltas_b_minus_a={r:{k:y['requests'][r][k]-v[k] for k in ('ttft_s','max_itl_s','completion_latency_s')} for r,v in x['requests'].items()}))
    out=dict(repeats=result,comparisons=comparisons,global_measurement_bytes=pager['measurement']['weight_copy_bytes'],phase_sum_matches_global=sum(r['actual_bytes'] for r in result.values())==pager['measurement']['weight_copy_bytes'],
        gpu_boundaries=dict(initial=read(a.input_dir/'environment.json')['gpu_before'],final=read(a.input_dir/'status.json')['gpu_after']),
        scopes=['Per-repeat initial cache comes from its own snapshot; the global summary initial cache is last-repeat only.',
            'CPU/thread/process/sched CPU and CUDA/host spans overlap; do not add them. CPU deltas include observation code.',
            'runqueue_wait_ns is not evidence of zero wait when sched_schedstats is disabled; raw values and enable flag are retained.',
            'Cgroup counters cover the cgroup observation interval, not exclusively engine.step.',
            'Logical state, allocation pointers and physical KV block IDs/free-queue order are separate comparisons. KV contents were not compared.',
            'Prefix equality compares all recorded rows, active/entry cache sets, groups, victims and bytes; full slot/map/LRU was not captured at call16.',
            'Thread/process CPU may include runtime polling or busy-wait; these counters are not pure CPU computation.',
            'GPU process observations are boundary snapshots. Stability of this group does not establish the cause of past timing drift.'])
    if any('retention' in r for r in result.values()):out['scopes'] += [
        'Retention stages partition actual engine calls; their wall_s excludes gaps and snapshots, which remain in whole_wall_s. Do not add load/host spans to wall.',
        'Protected/overlap counts are layer-call expert occurrences. Repeated ensure touches are bookkeeping and LRU refreshes, not independent demand-hit savings.',
            ('Raw row top-k routes are present; current decode union and frequency selection are independently recomputed.'
             if all(r.get('retention',{}).get('raw_row_routes_present') for r in result.values()) else
             'Raw exposes chosen/decode expert sets, not per-row routes or frequency histograms; frequency ranking and decode-union derivation cannot be independently reconstructed.'),
        'Call0..15 equality includes policy execution after injection; pre_action_signature separately compares only calls before the actual action.']
    with a.out.open('w' if a.update_derived else 'x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':main()
