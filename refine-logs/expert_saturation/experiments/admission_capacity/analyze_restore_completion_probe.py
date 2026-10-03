#!/usr/bin/env python3
"""Complete-request ablation of an independently replayed first-output obligation."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import analyze_ltr_component_probe as old
import analyze_ltr_packing_probe as packing

base,read,require,sha,digest,module=old.base,old.read,old.require,old.sha,old.digest,old.module
quantum_epochs=old.quantum_epochs
KV,BLOCKS=old.KV,old.BLOCKS
ROLES=['off','on','on','off']; LABELS=[f'block{i//2}-d6-restore-{role}' for i,role in enumerate(ROLES)]
DEFAULT=Path(__file__).resolve().parents[2]/'outputs/admission_capacity/20260914_restore_completion_r01'
SCOPE=('Same rank-prefix/FCFS/LTR200/10/native-recompute backend; only an actual resumed PREEMPTED request with '
       'past output and pending recovery can create a first-output obligation. No future output is used for decisions. '
       'Status is reconstructed from captured running membership/preemption count under the frozen three-state install scope. '
       'Actual completed requests alone release completion obligations. The independent helper replay supplies real priorities, '
       'including -2; original LTR quantum, actual ordering, victims, ownership and full costs remain checked. '
       'No SLO, Oracle, quality, full-LTR, significance or method-GO claim.')


def replay_obligations(raw,decisions,components,helper,enabled):
    requests={q['internal_request_id']:q for q in raw['requests']}
    tracker=helper.RestoreObligations(); counters=components.LTRCounters(200,10)
    previous=set(); maps=[]; max_required=0; active_steps=0; overlay_selections=0
    for k,(d,s,m) in enumerate(zip(decisions,raw['scheduler_steps'],raw['memory_trace'])):
        before,after=m['before'],m['after']; live=before['requests']
        visible={rid:helper.RequestView('RUNNING' if rid in before['running_ids'] else ('PREEMPTED' if v['num_preemptions'] else 'WAITING'),
            v['output_tokens'],v['prompt_tokens']+v['output_tokens']-v['computed_tokens']) for rid,v in live.items()}
        completed=previous-set(live)
        require(all(requests[r]['status']=='completed' and requests[r]['completion_s']<=s['start_s'] for r in completed),'obligation used unobserved completion')
        begin=tracker.begin_schedule(k,visible,completed)
        require(begin==d['obligation_begin'] and begin['outstanding']==d['obligations_before'],'obligation begin/release differs')
        original=counters.begin_schedule(live); effective=tracker.effective_priorities(original,enabled=enabled)
        require(d['counter_state_before']=={r:vars(v) for r,v in counters.states.items()} and d['priorities']==effective,'independent counter/priority replay differs')
        actual={x['internal_request_id']:x['scheduled_tokens'] for x in s['scheduled']}
        require(actual==d['tokens']==d['actual_scheduled'],'obligation replay lacks actual scheduled match')
        resumed=[rid for rid in actual if visible[rid].status=='PREEMPTED']
        remaining={rid:max(0,(v['prompt_tokens']+v['output_tokens']+15)//16-v['block_counts'][0]) for rid,v in after['requests'].items()}
        receipt=tracker.after_schedule(k,visible,resumed,actual,d['victims'],remaining,after['pool']['free_blocks'],enabled=enabled,effective_priorities=effective)
        require(receipt==d['obligation_after'],'obligation start/interruption/reservation differs')
        max_required=max(max_required,receipt['remaining_history_blocks']); active_steps+=bool(begin['outstanding'])
        overlay_selections+=sum(effective[r]==-2 for r in actual)
        counters.after_schedule(actual); previous=set(live); maps.append(effective)
    require(len(maps)==len(decisions)==len(raw['scheduler_steps']),'obligation replay coverage differs')
    completed=[r for r,q in requests.items() if q['status']=='completed' and q['completion_s']<=raw['observation_end_s']]
    require(len(completed)==len(requests)==32,'finalize has incomplete requests')
    tracker.finalize(len(decisions),{},completed); snapshot=tracker.snapshot()
    return maps,dict(snapshot=snapshot,replayed_steps=len(maps),active_begin_steps=active_steps,overlay_selected_calls=overlay_selections,
                     maximum_aggregate_remaining_blocks=max_required,releases=dict(Counter(e['reason'] for e in snapshot['events'] if e['type']=='release')))


# Source clone: only the expected-priority expression below differs from the audited
# old.component_accounting; its original counters and all actual-action checks remain.
def component_accounting(raw, decisions, boost, counters, expected_priorities):
    aliases, totals, held_counts, path = raw['internal_to_source'], Counter(), Counter(), []
    events = {}
    for e in raw['preemption_events']:
        events.setdefault(e['attempted_step'], []).append(e)
    require(len(decisions) == len(raw['scheduler_steps']), 'component/scheduler count differs')
    requests = {q['request_id']:q for q in raw['requests']}
    for k, (d, s, m) in enumerate(zip(decisions, raw['scheduler_steps'], raw['memory_trace'])):
        before, after = m['before'], m['after']; states, post = before['requests'], after['requests']
        require(d['step'] == k and d['status'] == 'APPLIED' and d['boost'] is boost
                and m['schedule_completed'] and not m['allocation_failures'], 'failed or unaligned component action')
        require(set(states) == set(post) and set(states) <= aliases.keys(), 'live identity changed inside schedule')
        priorities = counters.begin_schedule(states)
        require(d['counter_state_before'] == {r:vars(v) for r,v in counters.states.items()}
                and d['priorities'] == expected_priorities[k], 'counter history/priority differs')
        actual = {x['internal_request_id']:x['scheduled_tokens'] for x in s['scheduled']}
        require(d['tokens'] == d['actual_scheduled'] == actual and 0 <= sum(actual.values()) <= 1024,
                'planned versus actual token allocation differs')
        victims = d['victims']; selected = set(actual); running = set(before['running_ids'])
        require(len(set(victims)) == len(victims) and set(victims) <= running and not selected.intersection(victims), 'invalid victims')
        require(victims == [e['victim_internal_request_id'] for e in events.get(k, [])]
                and set(s['preempted_request_ids']) == {aliases[r] for r in victims}, 'plan/native preemptions differ')
        order = lambda r: (d['priorities'][r], requests[aliases[r]]['arrival_s'], r)
        require(list(d['tokens']) == sorted(selected, key=order), 'plan ordering is not past-only priority/FCFS')
        require(all(any(order(r) < order(v) for r in selected) for v in victims), 'victim has no higher-ranked selected beneficiary')
        for snapshot in (before, after):
            p = snapshot['pool']; owned = [v['block_counts'] for v in snapshot['requests'].values()]
            require(p['total_blocks'] == BLOCKS+1 and p['usable_blocks'] == BLOCKS
                    and 0 <= p['free_blocks'] <= BLOCKS and p['used_blocks']+p['free_blocks'] == BLOCKS
                    and all(len(v) == 1 and v[0] >= 0 for v in owned)
                    and sum(v[0] for v in owned) == p['used_blocks'], 'APC-off pool/owned conservation differs')
        released = 0
        for e in events.get(k, []):
            n = e['victim_state']['block_counts'][0]
            require(e['victim_state'] == states[e['victim_internal_request_id']]
                    and e['pool_after']['free_blocks']-e['pool']['free_blocks'] == n
                    and e['victim_state_after']['block_counts'] == [0], 'actual free delta differs from released ownership')
            released += n
        need = lambda r: max(0, (states[r]['prompt_tokens']+states[r]['output_tokens']+15)//16-states[r]['block_counts'][0])
        reserved = sum(need(r) for r in selected)
        remaining = sum(max(0, (states[r]['prompt_tokens']+states[r]['output_tokens']+15)//16-post[r]['block_counts'][0]) for r in selected)
        require(d['free_before'] == before['pool']['free_blocks'] and d['free_after'] == after['pool']['free_blocks']
                and d['free_after_reservation'] == d['free_before']+released-reserved >= 0
                and d['remaining_reserved_blocks'] == remaining <= d['free_after']
                and d['free_after']-remaining == d['free_after_reservation'], 'history reservation/free accounting differs')
        held = running-selected-set(victims)
        require(set(after['running_ids']) == running.union(selected)-set(victims) and len(after['running_ids']) <= 32,
                'resident/held set differs')
        for rid, v in states.items():
            q = requests[aliases[rid]]
            require(v['output_tokens'] == bisect_right(q['token_times_s'], s['start_s'])
                    and post[rid]['output_tokens'] == v['output_tokens'], 'pre-action output state uses unavailable output')
            expected = 0 if rid in victims else v['computed_tokens']+actual.get(rid, 0)
            require(post[rid]['computed_tokens'] == expected, 'unexpected computed-token mutation')
            if rid in selected:
                require(0 < actual[rid] <= v['prompt_tokens']+v['output_tokens']-v['computed_tokens'], 'scheduled beyond current history')
            if rid in held:
                require(post[rid] == v, 'held resident changed state')
                held_counts[aliases[rid]] += 1
        spent = [r for r in selected if priorities[r] == -1]
        totals.update(selected_request_calls=len(selected), boosted_counter_calls=len(spent),
                      effective_boost_calls=len(spent) if boost else 0,
                      boosted_recompute_calls=sum(x['recompute_tokens'] > 0 and x['internal_request_id'] in spent for x in s['scheduled']),
                      held_resident_calls=len(held), released_blocks=released, victim_events=len(victims))
        counters.after_schedule(actual)
        elapsed = s['end_s']-s['start_s']
        require(all(math.isfinite(d[t]) for t in ('decision_seconds','wrapped_schedule_seconds'))
                and 0 <= d['decision_seconds'] <= d['wrapped_schedule_seconds'] <= elapsed+1e-6, 'nested scheduler cost differs')
        totals.update(decision_s=d['decision_seconds'], wrapped_scheduler_s=d['wrapped_schedule_seconds'], scheduler_s=elapsed)
        path.append(dict(selected=[[aliases[r],n] for r,n in actual.items()], victims=[aliases[r] for r in victims], held=sorted(aliases[r] for r in held)))
    return dict(totals=dict(totals), held_calls_by_request=dict(held_counts), schedule_path_sha256=digest(path),
                cost_relation='decision <= wrapped component scheduler <= captured scheduler <= complete measured wall; do not add nested costs')

def inspect(results, spec, source, meta, workload, config, metrics, components, reference):
    folder = results/spec['label']
    row = dict(label=spec['label'], policy='restore_completion_'+spec['complete_restores'], status='UNRUN', eligible=False)
    try:
        raw_path = next((p for p in (folder/'raw.json',folder/'raw.json.gz') if p.exists()), None)
        if raw_path is None:
            row['status'] = 'INCOMPLETE' if folder.exists() and any(folder.iterdir()) else 'UNRUN'
            return row
        raw = read(raw_path)
        row['retained_requests'] = [dict(request_id=r['request_id'],status=r['status'],outputs=len(r['output_token_ids'])) for r in raw['requests']]
        cfg, engine, q, memory, env, terminal, saved = [read(folder/n) for n in
            ('config.json','engine_args.json','safe-cap-qualification.json','memory-before.json','environment.json','status.json','metrics.json')]
        require(raw['status'] == terminal['status'] == 'COMPLETE' and raw['error'] is None
                and raw['capacity_boundary'] is None and raw['preemption_mode'] == 'native_recompute', 'measurement failed/incomplete')
        require(engine == reference['engine'] and env['vllm'] == '0.26.0'
                and env['vllm_source_sha256'] == reference['runtime'], 'fixed engine/runtime differs')
        inventory = {n:h for n,h in meta['files_sha256'].items() if n.endswith('.py')}
        require(env['source_sha256'] == inventory and all(sha(source/n) == h for n,h in inventory.items()), 'executed source inventory differs')
        require(all(cfg[k] == v for k,v in config.items()) and cfg['cap'] == raw['target_cap'] == 32
                and cfg['completion_policy'] == 'restore_completion_'+spec['complete_restores']
                and cfg['component'] == dict(boost=spec['boost']=='on',threshold=200,quantum=10,base_order='FCFS',
                    backend='priority packing / current-history reservation / native recompute',packing='rank_prefix',complete_restores=spec['complete_restores']=='on'), 'workload/component configuration differs')
        require(q['status'] == 'QUALIFIED' and q['usable_blocks'] == BLOCKS and q['block_size'] == 16
                and q['prefix_caching'] is False and q['observed_scheduler_reserve_full_isl'] is True
                and q['coordinator_type'] == 'KVCacheCoordinatorNoPrefixCache' and q['group_count'] == 1
                and q['full_cap_reserved_blocks'] == 32*256 and q['full_reservation_sufficient'] is False
                and engine['kv_cache_memory_bytes'] == memory['kv_storage_bytes'] == KV, 'actual KV qualification differs')
        for index, count in enumerate((32,32,2)):
            warm = read(folder/f'warmup-{index}.json')
            require(warm['status'] == 'COMPLETE' and len(warm['requests']) == count
                    and all(r['status'] == 'completed' and len(r['output_token_ids']) == 16 for r in warm['requests']), 'warmup incomplete')
        require(env['gpu_before']['compute_processes'] == '' and len(raw['gpu_before']['compute_processes'].splitlines()) <= 1,
                'GPU pre-initialization/measurement not isolated under frozen own-PID exclusion')
        requests = {r['request_id']:r for r in raw['requests']}
        require(len(requests) == len(raw['requests']) == 32 and len(set(raw['internal_to_source'].values())) == 32, 'request identity/count differs')
        for expected, ids, arrival in zip(workload['source_requests'],workload['actual_prompt_token_ids'],workload['arrival_traces_s']['steady']):
            r = requests[expected['request_id']]
            require(r['document_id'] == expected['document_id'] and r['prompt_tokens'] == len(ids) == 3072
                    and r['prompt_token_ids_sha256'] == expected['prompt_token_ids_sha256'] == digest(ids)
                    and r['arrival_s'] == arrival and r['status'] == 'completed' and len(r['output_token_ids']) == 1024
                    and raw['internal_to_source'][r['internal_request_id']] == r['request_id'], 'request input/output identity differs')
            require(r['arrival_s'] <= r['admission_s'] <= r['engine_add_return_s'] <= r['token_times_s'][0]
                    and r['completion_s'] == r['token_times_s'][-1] and r['stop_reason'] == 'length', 'request clocks differ')
        recalculated = metrics.summarize_episode_requests(raw['requests'],observation_end_s=raw['observation_end_s'],ttft_slo_s=5.,tpot_slo_s=.2)
        require(all(saved.get(k) == v for k,v in recalculated.items()), 'saved metrics differ from raw')
        work = base.work_accounting(raw,requests)
        require(not work['cache']['successful_positive_adjustments'], 'APC-off unexpected computed jump')
        decisions = read(folder/'component-decisions.json')
        enabled=spec['complete_restores']=='on'
        require(all(d['packing']=='rank_prefix' and d['boost'] is True and d['complete_restores'] is enabled for d in decisions), 'executed obligation arm differs')
        helper=module('frozen_restore_obligation',source/'restore_obligation.py')
        expected,obligations=replay_obligations(raw,decisions,components,helper,enabled)
        require(obligations['snapshot']==read(folder/'restore-obligations.json'),'final obligation ledger differs')
        component = component_accounting(raw,decisions,True,components.LTRCounters(200,10),expected)
        component['quantum_epochs'] = quantum_epochs(raw,decisions,spec['boost']=='on')
        require(sum(e['selected_calls'] for e in component['quantum_epochs']['epochs']) == component['totals'].get('boosted_counter_calls',0),
                'quantum epochs do not conserve actual selected calls')
        for pause in work['pauses']:
            served = [x for s in raw['scheduler_steps'][pause['first_service_step']:] if s['end_s'] <= pause['next_token_s'] for x in s['scheduled'] if x['request_id'] == pause['request_id']]
            pause.update(scheduled_calls_through_new_output=len(served), recompute_calls_through_new_output=sum(x['recompute_tokens'] > 0 for x in served))
        per = [dict(request_id=r['request_id'],ttft_s=r['token_times_s'][0]-r['arrival_s'],completion_s=r['completion_s']-r['arrival_s'],
                    mean_tpot_s=(r['token_times_s'][-1]-r['token_times_s'][0])/(len(r['token_times_s'])-1),
                    max_itl_s=max(b-a for a,b in zip(r['token_times_s'],r['token_times_s'][1:])),output_sha256=digest(r['output_token_ids'])) for r in raw['requests']]
        require(component['totals']['scheduler_s'] <= recalculated['observation_duration_s'], 'scheduler exceeds full wall')
        require((folder/'resolved-scheduler-config.json').exists(),'resolved scheduler config missing')
        row.update(status='COMPLETE',eligible=True,config=cfg,engine=engine,work=work,component=component,per_request=per,
            recovery_residencies=packing.residencies(raw),planner_replay=packing.replay(folder,source,raw,decisions,'rank_prefix'),
            restore_obligations=obligations,obligation_ledger_sha256=sha(folder/'restore-obligations.json'),
            wall_s=recalculated['observation_duration_s'],throughput_rps=recalculated['throughput_rps'],
            output_throughput_tokens_s=32768/recalculated['observation_duration_s'],output_tokens=32768,
            mean_completion_s=sum(r['completion_s'] for r in per)/32,ttft_s=metrics._distribution([r['ttft_s'] for r in per]),
            mean_tpot_s=metrics._distribution([r['mean_tpot_s'] for r in per]),
            completion_s=metrics._distribution([r['completion_s'] for r in per]),request_max_itl_s=metrics._distribution([r['max_itl_s'] for r in per]),
            max_itl_s=max(r['max_itl_s'] for r in per),software={k:env[k] for k in ('python','torch','cuda','vllm','transformers')},
            gpu_uuid=re.search(r'GPU-[\w-]+',env['gpu_before']['device']).group(),raw_path=str(raw_path),raw_sha256=sha(raw_path),
            decisions_sha256=sha(folder/'component-decisions.json'),_outputs={r:q['output_token_ids'] for r,q in requests.items()})
        if enabled and obligations['overlay_selected_calls']==0:
            row.update(status='INVALID_NO_ACTION',eligible=False,error='no actual selected active obligation priority -2')
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,StopIteration,RuntimeError) as error:
        row.update(status='INCOMPLETE',eligible=False,error=str(error))
    return row


def finish(result,args):
    for row in result['cells']: row.pop('_outputs',None)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'REPORT.md').write_text('# First-output restore obligation\n\n'+result['status']+'\n\n'+SCOPE+'\n\n'+
        '\n'.join(f"- {r['label']}: {r['status']}" for r in result['cells'])+'\n\n'+result.get('error','Full metrics, raw-derived residency counts, actual priority/ledger replay and per-request changes are retained in analysis.json.')+'\n')
    print(json.dumps(dict(status=result['status'],preparation_checks=result.get('preparation_checks'),error=result.get('error'),
        cells=[dict(label=r['label'],status=r['status'],error=r.get('error')) for r in result['cells']])))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir',type=Path,default=DEFAULT); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--results-dir',type=Path); p.add_argument('--source-dir',type=Path); p.add_argument('--preparation-dir',type=Path)
    p.add_argument('--expected-metadata-sha256',required=True); args=p.parse_args()
    require(not args.output_dir.exists(),'output exists; refuse overwrite')
    bundle=args.run_dir.parent if args.run_dir.name.startswith('execution') else args.run_dir
    run=args.run_dir if args.run_dir.name.startswith('execution') else bundle/'execution'
    preparation=args.preparation_dir or bundle/'preparation'; source=args.source_dir or (run/'readback/pkg' if (run/'readback/pkg').is_dir() else preparation/'pkg')
    results=args.results_dir or run/'readback/results'; meta_path=preparation/'preparation.json'
    result=dict(status='UNRUN',cells=[dict(label=n,policy='restore_completion_'+r,status='UNRUN',eligible=False) for n,r in zip(LABELS,ROLES)],comparisons=[],
        scope=SCOPE,source=str(source),results_dir=str(results),reused_accounting_sha256=sha(Path(old.__file__)),reused_packing_sha256=sha(Path(packing.__file__)))
    try:
        if not meta_path.exists():
            require(not results.exists() or not any(results.iterdir()),'retained attempt lacks metadata');result['preparation_checks']='WAITING_FOR_FROZEN_PACKAGE';finish(result,args);return
        require(sha(meta_path)==args.expected_metadata_sha256,'frozen metadata changed');meta=read(meta_path)
        for package in {source,preparation/'pkg'}:
            require(all(sha(package/n)==h for n,h in meta['files_sha256'].items()),'frozen source differs')
            sums=dict((n.lstrip('*'),h) for h,n in (line.split(maxsplit=1) for line in (package/'SHA256SUMS').read_text().splitlines()))
            require(sums==meta['files_sha256'],'source inventory differs')
        campaign=read(source/'campaign.json'); cells=campaign['cells']
        require(campaign['comparison']==meta['comparison']=='restore_completion' and cells==meta['cells'] and [c['label'] for c in cells]==LABELS
            and [c['complete_restores'] for c in cells]==ROLES and all(c['boost']=='on' and c['packing']=='rank_prefix' and c['usable_blocks']==BLOCKS and c['kv_cache_bytes']==KV for c in cells),'frozen arms differ')
        inp=source/'inputs_preparation/prepared/long'; workload=read(inp/'workload.json'); config=read(inp/'config.json')
        require(hashlib.sha256(json.dumps(workload,sort_keys=True).encode()).hexdigest()==config['workload_sha256']
            and len(workload['source_requests'])==len(workload['actual_prompt_token_ids'])==len(workload['arrival_traces_s']['steady'])==32,'frozen inputs differ')
        ref=Path(meta['reference_cell']);require(ref.is_absolute() and sha(ref/'engine_args.json')==meta['reference_engine_args_sha256'],'reference engine differs')
        reference=dict(engine=read(ref/'engine_args.json'),runtime=read(ref/'environment.json')['vllm_source_sha256'])
        require(len(reference['runtime'])==7 and all(reference['runtime'][n]==h for n,h in meta['expected_runtime_sources'].items()),'pinned runtime differs')
        metrics=module('restore_frozen_metrics',source/'metrics.py');components=module('restore_frozen_counters',source/'recovery_service_components.py')
        rows=[inspect(results,c,source,meta,workload,config,metrics,components,reference) for c in cells];result['cells']=rows
        result.update(status='UNRUN' if all(r['status']=='UNRUN' for r in rows) else 'INCOMPLETE',preparation_checks='PASS',metadata_sha256=args.expected_metadata_sha256)
        if all(r['eligible'] for r in rows):
            pairs=[]
            for i,j,kind in [(0,1,'block0_off_on'),(3,2,'block1_off_on'),(0,3,'off_repeat'),(1,2,'on_repeat')]:
                a,b=rows[i],rows[j]
                normalize=lambda row:dict(row,config=dict(row['config'],completion_policy='common_restore',component=dict(row['config']['component'],complete_restores=False)))
                pair=base.compare(normalize(a),normalize(b),kind)
                left={q['request_id']:q for q in a['per_request']}; right={q['request_id']:q for q in b['per_request']}
                for change in pair['per_request_changes']:
                    rid=change['request_id'];change['mean_tpot_s']=right[rid]['mean_tpot_s']-left[rid]['mean_tpot_s']
                pair.update(baseline_policy=a['policy'],action_policy=b['policy'],schedule_path_equal=a['component']['schedule_path_sha256']==b['component']['schedule_path_sha256'])
                pair['per_request_change_counts']={metric:dict(improved=sum(q[metric]<0 for q in pair['per_request_changes']),harmed=sum(q[metric]>0 for q in pair['per_request_changes']),equal=sum(q[metric]==0 for q in pair['per_request_changes'])) for metric in ('ttft_s','completion_s','max_itl_s','mean_tpot_s')}
                pairs.append(pair)
            result.update(status='MEASUREMENT_ONLY',comparisons=pairs)
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,RuntimeError) as error:
        result.update(status='INCOMPLETE',error=str(error),comparisons=[])
    finish(result,args)


if __name__=='__main__':main()
