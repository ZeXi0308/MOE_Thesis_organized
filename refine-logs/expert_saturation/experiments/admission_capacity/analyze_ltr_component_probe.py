#!/usr/bin/env python3
"""Retained four-cell LTR counter transplant: CPU checks and descriptive metrics."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import sys

import analyze_prefix_cache_baseline as base

read, require, sha, digest = base.read, base.require, base.sha, base.digest
LABELS = ['block0-d6-boost-off', 'block0-d6-boost-on', 'block1-d6-boost-on', 'block1-d6-boost-off']
META_SHA = '1c7944de68f18252ed13228b3dd1235c5e5faef0455f26ebdd6e5fe6eea73728'
KV, BLOCKS = 13960740864, 6656
SCOPE = ('A scheduling-counter component on a shared custom priority-packing/current-history-reservation '
         'backend using native recompute; boost-off is not native vLLM and neither arm is full LTR. '
         'Measured wall includes decision, instrumentation and runtime costs without subtraction. '
         'No SLO, Oracle, quality, significance or method GO. Same-input repeats are correlated. '
         'Recovery spans overlap and cannot be summed into wall. Resource checks use qualified APC-off '
         'full-attention ownership counts and actual releases, not an independent physical block-ID replay. '
         'No full planner replay: the resolved long-prefill chunk threshold is not recorded.')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


def quantum_epochs(raw, decisions, boost):
    """Post-run outcomes only; begin-state counters and selected calls define epochs."""
    requests = {q['request_id']:q for q in raw['requests']}; aliases = raw['internal_to_source']
    calls = {k:c for c in raw['engine_steps'] for k in range(c['scheduler_step_start'],c['scheduler_step_end'])}
    active, previous, epochs = {}, {}, []
    def close(rid, end, reason):
        e = active.pop(rid); q = requests[aliases[rid]]; times = q['token_times_s']; start = bisect_right(times,e['start_s'])
        require(e['start_s'] <= end <= raw['observation_end_s'], 'quantum epoch clock differs')
        first = times[start] if start < len(times) else None; exhausted = e.get('exhaustion_call_return_s')
        require(0 <= e['selected_calls'] <= 10, 'observed quantum exceeds frozen allowance')
        e.update(end_s=end,end_reason=reason,incomplete=reason=='capture_end',new_outputs=bisect_right(times,end)-start,
                 quantum_exhausted=exhausted is not None,unused_quantum=10-e['selected_calls'],
                 first_new_output_after_start_s=first,exhaustion_preceded_first_new_output=None if exhausted is None or first is None else exhausted < first,
                 new_outputs_through_exhausting_call=None if exhausted is None else bisect_right(times,exhausted)-start)
        e['exhaustion_call_finished_without_new_output'] = None if exhausted is None else e['new_outputs_through_exhausting_call'] == 0
        epochs.append(e)
    for k,(d,s) in enumerate(zip(decisions,raw['scheduler_steps'])):
        states = d['counter_state_before']; actual = {x['internal_request_id']:x for x in s['scheduled']}
        require(calls[k]['completed'] and d['status']=='APPLIED', 'unconfirmed service cannot spend an observed quantum')
        for rid in set(active)-states.keys():
            q=requests[aliases[rid]]; require(q['status']=='completed' and q['completion_s'] <= s['start_s'], 'active quantum disappeared before completion')
            close(rid,q['completion_s'],'request_completion')
        for rid,v in states.items():
            reset = previous.get(rid,{}).get('idle',0) >= 200
            if rid in active and (reset or v['priority'] != -1):
                close(rid,s['start_s'],'starvation_trigger_reset' if reset else 'normal_demotion')
            if v['priority'] == -1 and rid not in active:
                require(v['quantum_remaining']==10, 'quantum begins without full frozen allowance')
                active[rid]=dict(request_id=aliases[rid],start_step=k,start_s=s['start_s'],selected_calls=0,recompute_only_calls=0,
                                 policy_effective=boost,start_reason='starvation_trigger',selected_steps=[])
            if rid in active and rid in actual:
                e=active[rid]; x=actual[rid]; e['selected_calls']+=1; e['selected_steps'].append(k)
                e['recompute_only_calls']+=int(x['recompute_tokens']==x['scheduled_tokens'])
                if v['quantum_remaining']==1:
                    e.update(exhaustion_step=k,exhaustion_schedule_end_s=s['end_s'],exhaustion_call_return_s=calls[k]['returned_s'])
        previous={rid:dict(v,idle=0 if rid in actual else v['idle']+1,
                          quantum_remaining=v['quantum_remaining']-int(rid in actual and v['priority']==-1)) for rid,v in states.items()}
    for rid in list(active):
        q=requests[aliases[rid]]; completed=q['status']=='completed' and q['completion_s'] <= raw['observation_end_s']
        close(rid,q['completion_s'] if completed else raw['observation_end_s'],'request_completion' if completed else 'capture_end')
    return dict(epochs=epochs,incomplete_epochs=sum(e['incomplete'] for e in epochs),
                end_reasons=dict(Counter(e['end_reason'] for e in epochs)),
                exhausted_without_new_output=sum(e['exhaustion_call_finished_without_new_output'] is True for e in epochs),
                boundary='Observed epoch is (begin_schedule enclosing host step start, end] in host seconds. '
                'Demotion/reset ends at the next enclosing step start; completion includes the EOS/length receipt; capture end is censored. '
                'Counter exhaustion occurs during scheduling (reported schedule end is an upper boundary); first-output comparison uses the '
                'exhausting successful engine call return, so output returned by that same call counts as service. '
                'Future token times are post-run outcomes only. Boost-off counter epochs are diagnostic, not applied priority protection.')


def component_accounting(raw, decisions, boost, counters):
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
                and d['priorities'] == {r:p if boost else 0 for r,p in priorities.items()}, 'counter history/priority differs')
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
    row = dict(label=spec['label'], policy='custom_ltr_component_boost_'+spec['boost'], status='UNRUN', eligible=False)
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
                and cfg['completion_policy'] == 'ltr_component_'+spec['boost']
                and cfg['component'] == dict(boost=spec['boost']=='on',threshold=200,quantum=10,base_order='FCFS',
                    backend='priority packing / current-history reservation / native recompute'), 'workload/component configuration differs')
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
        component = component_accounting(raw,decisions,spec['boost']=='on',components.LTRCounters(200,10))
        component['quantum_epochs'] = quantum_epochs(raw,decisions,spec['boost']=='on')
        require(sum(e['selected_calls'] for e in component['quantum_epochs']['epochs']) == component['totals'].get('boosted_counter_calls',0),
                'quantum epochs do not conserve actual selected calls')
        for pause in work['pauses']:
            served = [x for s in raw['scheduler_steps'][pause['first_service_step']:] if s['end_s'] <= pause['next_token_s'] for x in s['scheduled'] if x['request_id'] == pause['request_id']]
            pause.update(scheduled_calls_through_new_output=len(served), recompute_calls_through_new_output=sum(x['recompute_tokens'] > 0 for x in served))
        per = [dict(request_id=r['request_id'],ttft_s=r['token_times_s'][0]-r['arrival_s'],completion_s=r['completion_s']-r['arrival_s'],
                    max_itl_s=max(b-a for a,b in zip(r['token_times_s'],r['token_times_s'][1:])),output_sha256=digest(r['output_token_ids'])) for r in raw['requests']]
        require(component['totals']['scheduler_s'] <= recalculated['observation_duration_s'], 'scheduler exceeds full wall')
        row.update(status='COMPLETE',eligible=True,config=cfg,engine=engine,work=work,component=component,per_request=per,
            wall_s=recalculated['observation_duration_s'],throughput_rps=recalculated['throughput_rps'],
            output_throughput_tokens_s=32768/recalculated['observation_duration_s'],output_tokens=32768,
            mean_completion_s=sum(r['completion_s'] for r in per)/32,ttft_s=metrics._distribution([r['ttft_s'] for r in per]),
            completion_s=metrics._distribution([r['completion_s'] for r in per]),request_max_itl_s=metrics._distribution([r['max_itl_s'] for r in per]),
            max_itl_s=max(r['max_itl_s'] for r in per),software={k:env[k] for k in ('python','torch','cuda','vllm','transformers')},
            gpu_uuid=re.search(r'GPU-[\w-]+',env['gpu_before']['device']).group(),raw_path=str(raw_path),raw_sha256=sha(raw_path),
            decisions_sha256=sha(folder/'component-decisions.json'),_outputs={r:q['output_token_ids'] for r,q in requests.items()})
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,StopIteration) as error:
        row.update(status='INCOMPLETE',eligible=False,error=str(error))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--results-dir',type=Path); p.add_argument('--source-dir',type=Path); p.add_argument('--preparation-dir',type=Path)
    p.add_argument('--execution-owner',default='root long-task session'); args=p.parse_args()
    require(not args.output_dir.exists(), 'output exists; refuse overwrite')
    bundle = args.run_dir.parent if args.run_dir.name == 'execution' else args.run_dir
    preparation = args.preparation_dir or bundle/'preparation'
    run = args.run_dir if args.run_dir.name.startswith('execution') else bundle/'execution'
    source = args.source_dir or (run/'readback/pkg' if (run/'readback/pkg').is_dir() else preparation/'pkg')
    results = args.results_dir or run/'readback/results'
    result = dict(status='UNRUN',cells=[],comparisons=[],scope=SCOPE,source=str(source),
                  execution_owner=args.execution_owner,results_dir=str(results),shared_accounting_sha256=sha(Path(base.__file__)),
                  checks=['past-only counter state','request/step/output alignment','full wall and resource accounting','common backend/source/independent trajectories'])
    try:
        require(sha(preparation/'preparation.json') == META_SHA, 'original preparation metadata changed')
        meta = read(preparation/'preparation.json')
        for package in {source,preparation/'pkg'}:
            require(all(sha(package/n) == h for n,h in meta['files_sha256'].items()), 'frozen package content differs')
            sums = dict((name.lstrip('*'),h) for h,name in (line.split(maxsplit=1) for line in (package/'SHA256SUMS').read_text().splitlines()))
            require(sums == meta['files_sha256'], 'package SHA256SUMS inventory differs')
        campaign = read(source/'campaign.json')['cells']
        require(campaign == meta['cells'] and [x['label'] for x in campaign] == LABELS
                and [x['boost'] for x in campaign] == ['off','on','on','off']
                and all(x['usable_blocks'] == BLOCKS and x['kv_cache_bytes'] == KV for x in campaign), 'frozen four-cell campaign differs')
        inp=source/'inputs_preparation/prepared/long'; workload=read(inp/'workload.json'); config=read(inp/'config.json')
        require(hashlib.sha256(json.dumps(workload,sort_keys=True).encode()).hexdigest() == config['workload_sha256']
                and len(workload['source_requests']) == len(workload['actual_prompt_token_ids']) == len(workload['arrival_traces_s']['steady']) == 32, 'frozen workload differs')
        command=shlex.split(meta['command']); ref=Path(command[command.index('--reference-cell')+1])
        require(sha(ref/'engine_args.json') == meta['reference_engine_args_sha256'], 'fixed reference engine changed')
        reference=dict(engine=read(ref/'engine_args.json'),runtime=read(ref/'environment.json')['vllm_source_sha256'])
        require(len(reference['runtime']) == 7 and all(reference['runtime'][n] == h for n,h in meta['expected_runtime_sources'].items()), 'pinned runtime reference differs')
        metrics=module('ltr_frozen_metrics',source/'metrics.py'); components=module('ltr_frozen_counters',source/'recovery_service_components.py')
        rows=[inspect(results,c,source,meta,workload,config,metrics,components,reference) for c in campaign]; result['cells']=rows
        result.update(status='UNRUN' if all(r['status']=='UNRUN' for r in rows) else 'INCOMPLETE',
                      preparation_checks='PASS',cap32_full_history_margin_blocks=BLOCKS-32*256,metrics_sha256=sha(source/'metrics.py'))
        if all(r['eligible'] for r in rows):
            pairs=[]
            for i,j,kind in [(0,1,'block0_off_on'),(3,2,'block1_off_on'),(0,3,'off_repeat'),(1,2,'on_repeat')]:
                a,b=rows[i],rows[j]
                normalize=lambda row: dict(row,config=dict(row['config'],completion_policy='common_component',component=dict(row['config']['component'],boost=False)))
                pair=base.compare(normalize(a),normalize(b),kind)
                pair.update(baseline_policy=a['policy'],action_policy=b['policy'],schedule_path_equal=a['component']['schedule_path_sha256']==b['component']['schedule_path_sha256'])
                pairs.append(pair)
            result.update(status='MEASUREMENT_ONLY',comparisons=pairs)
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:
        result.update(status='INCOMPLETE',error=str(error),comparisons=[])
    for row in result['cells']: row.pop('_outputs',None)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'REPORT.md').write_text('# LTR counter component probe\n\n'+result['status']+'\n\n'+SCOPE+'\n\n'+
        '\n'.join(f"- {r['label']}: {r['status']}" for r in result['cells'])+'\n\n'+
        (result.get('error','Full per-request changes, output differences, actual recovery spans, resource and nested costs are in analysis.json.'))+'\n')
    print(json.dumps(dict(status=result['status'],error=result.get('error'),cells=[dict(label=r['label'],status=r['status'],error=r.get('error')) for r in result['cells']])))


if __name__ == '__main__':
    main()
