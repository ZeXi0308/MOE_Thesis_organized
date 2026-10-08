#!/usr/bin/env python3
"""Four LTR200/10-on packing cells; preserve actual recovery and full-request cost."""
import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import analyze_ltr_component_probe as old

base, read, require, sha, digest, module = old.base, old.read, old.require, old.sha, old.digest, old.module
component_accounting, quantum_epochs = old.component_accounting, old.quantum_epochs
KV, BLOCKS = old.KV, old.BLOCKS
ROLES = ['fit_scan','rank_prefix','rank_prefix','fit_scan']
LABELS = [f'block{i//2}-d6-packing-{role}' for i,role in enumerate(ROLES)]
DEFAULT = Path(__file__).resolve().parents[2]/'outputs/admission_capacity/20260914_ltr_packing_r01'
SCOPE = ('Both arms are the same custom FCFS/current-history-reservation/native-recompute backend with LTR200/10 on; '
         'only fit-scan versus ranked-prefix capacity stop changes. Not full LTR or a holdout. '
         'All wall, nested decision/scheduler cost, request losses and outputs remain. No SLO/Oracle/quality/method GO. '
         'Recovery ends at the next actual preemption of that request; completion endings are separate. '
         'Counts of 0/1/2 new outputs apply only to actually resumed, subsequently re-preempted residencies. '
         'Future outputs are outcomes only; exact planner replay requires recorded resolved chunk threshold.')


def residencies(raw):
    requests={q['request_id']:q for q in raw['requests']}; aliases=raw['internal_to_source']; events=defaultdict(list)
    for e in raw['preemption_events']: events[aliases[e['victim_internal_request_id']]].append(e)
    rows=[]
    for rid,evs in events.items():
        q=requests[rid]
        for i,e in enumerate(evs):
            following=evs[i+1] if i+1<len(evs) else None; end_step=following['attempted_step'] if following else len(raw['scheduler_steps'])
            selected=[(s['step'],x) for s in raw['scheduler_steps'][e['attempted_step']:end_step] for x in s['scheduled'] if x['request_id']==rid]
            n=e['victim_state']['output_tokens']; end_n=following['victim_state']['output_tokens'] if following else len(q['output_token_ids'])
            require(end_n>=n and (bool(selected) or end_n==n), 'residency output without confirmed service')
            end_reason='actual_repreemption' if following else ('request_completion' if q['status']=='completed' else 'capture_end')
            rows.append(dict(request_id=rid,preempt_step=e['attempted_step'],first_service_step=selected[0][0] if selected else None,
                end_step=following['attempted_step'] if following else None,end_reason=end_reason,resumed=bool(selected),
                outputs_before=n,outputs_at_end=end_n,new_outputs=end_n-n,selected_calls=len(selected),
                recompute_only_calls=sum(x['recompute_tokens']==x['scheduled_tokens'] for _,x in selected),
                recompute_tokens=sum(x['recompute_tokens'] for _,x in selected),selected_steps=[k for k,_ in selected],
                first_new_output_s=q['token_times_s'][n] if end_n>n else None,
                next_preemption_host_perf_counter_s=following['host_perf_counter_s'] if following else None))
    closed=[r for r in rows if r['resumed'] and r['end_reason']=='actual_repreemption']
    return dict(rows=rows,repreempted_residencies=len(closed),new_output_histogram=dict(Counter(str(r['new_outputs']) for r in closed)),
                short_new_output_counts={str(n):sum(r['new_outputs']==n for r in closed) for n in (0,1,2)},
                unresumed=sum(not r['resumed'] for r in rows),end_reasons=dict(Counter(r['end_reason'] for r in rows)),
                boundary='Output-count difference between actual preemption receipts; next re-preemption closes before its scheduler call can return outputs. No residency is extended to the next quantum or eventual completion.')


def replay(folder,source,raw,decisions,packing):
    path=folder/'resolved-scheduler-config.json'
    if not path.exists(): return dict(status='LIMITED_ACTUAL_PLAN_AND_RESOURCE_RECEIPTS',reason='resolved long_prefill_token_threshold not recorded',replayed_steps=0)
    require('resolved-scheduler-config.json' in (source/'run_probe.py').read_text(), 'resolved config lacks frozen capture provenance')
    resolved=read(path); chunk=resolved['long_prefill_token_threshold']
    require(type(chunk) is int and chunk>=0, 'invalid resolved chunk threshold')
    tree=ast.parse((source/'ltr_recompute_native.py').read_text()); namespace=dict(dataclass=dataclass,__name__=__name__)
    nodes=[n for n in tree.body if getattr(n,'name',None) in ('Candidate','Plan','plan')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source/'ltr_recompute_native.py'),'exec'),namespace)
    requests={q['request_id']:q for q in raw['requests']}; aliases=raw['internal_to_source']
    for d,m in zip(decisions,raw['memory_trace']):
        before=m['before']; live=before['requests']
        rows=[namespace['Candidate'](rid,d['priorities'][rid],requests[aliases[rid]]['arrival_s'],rid in before['running_ids'],
              v['block_counts'][0],(v['prompt_tokens']+v['output_tokens']+15)//16,v['prompt_tokens']+v['output_tokens']-v['computed_tokens']) for rid,v in live.items()]
        plan=namespace['plan'](rows,before['pool']['free_blocks'],1024,32,chunk,packing=packing)
        require(plan.tokens==d['tokens'] and plan.victims==d['victims'] and plan.free_after_reservation==d['free_after_reservation'], 'exact packing replay differs')
    return dict(status='EXACT_FROZEN_PLAN_REPLAY',replayed_steps=len(decisions),resolved_chunk_threshold=chunk,receipt_sha256=sha(path))


def inspect(results, spec, source, meta, workload, config, metrics, components, reference):
    folder = results/spec['label']
    row = dict(label=spec['label'], policy='ltr_packing_'+spec['packing'], status='UNRUN', eligible=False)
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
                and cfg['completion_policy'] == 'ltr_packing_'+spec['packing']
                and cfg['component'] == dict(boost=spec['boost']=='on',threshold=200,quantum=10,base_order='FCFS',
                    backend='priority packing / current-history reservation / native recompute',packing=spec['packing']), 'workload/component configuration differs')
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
        require(all(d['packing']==spec['packing'] and d['boost'] is True for d in decisions), 'executed packing differs')
        component = component_accounting(raw,decisions,spec['boost']=='on',components.LTRCounters(200,10))
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
        row.update(status='COMPLETE',eligible=True,config=cfg,engine=engine,work=work,component=component,per_request=per,
            recovery_residencies=residencies(raw),planner_replay=replay(folder,source,raw,decisions,spec['packing']),
            wall_s=recalculated['observation_duration_s'],throughput_rps=recalculated['throughput_rps'],
            output_throughput_tokens_s=32768/recalculated['observation_duration_s'],output_tokens=32768,
            mean_completion_s=sum(r['completion_s'] for r in per)/32,ttft_s=metrics._distribution([r['ttft_s'] for r in per]),
            mean_tpot_s=metrics._distribution([r['mean_tpot_s'] for r in per]),
            completion_s=metrics._distribution([r['completion_s'] for r in per]),request_max_itl_s=metrics._distribution([r['max_itl_s'] for r in per]),
            max_itl_s=max(r['max_itl_s'] for r in per),software={k:env[k] for k in ('python','torch','cuda','vllm','transformers')},
            gpu_uuid=re.search(r'GPU-[\w-]+',env['gpu_before']['device']).group(),raw_path=str(raw_path),raw_sha256=sha(raw_path),
            decisions_sha256=sha(folder/'component-decisions.json'),_outputs={r:q['output_token_ids'] for r,q in requests.items()})
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,StopIteration) as error:
        row.update(status='INCOMPLETE',eligible=False,error=str(error))
    return row


def finish(result,args):
    for row in result['cells']: row.pop('_outputs',None)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'REPORT.md').write_text('# LTR packing baseline\n\n'+result['status']+'\n\n'+SCOPE+'\n\n'+
        '\n'.join(f"- {r['label']}: {r['status']}" for r in result['cells'])+'\n\n'+
        result.get('error','Per-request changes, output differences, exact preemption-bounded recovery counts, quantum epochs and full cost are retained in analysis.json.')+'\n')
    print(json.dumps(dict(status=result['status'],preparation_checks=result.get('preparation_checks'),error=result.get('error'),
                         cells=[dict(label=r['label'],status=r['status'],error=r.get('error')) for r in result['cells']])))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir',type=Path,default=DEFAULT); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--results-dir',type=Path); p.add_argument('--source-dir',type=Path); p.add_argument('--preparation-dir',type=Path)
    p.add_argument('--expected-metadata-sha256'); p.add_argument('--receipt',type=Path,help='Root freeze receipt JSON containing preparation_sha256')
    p.add_argument('--execution-owner',default='root long-task session'); args=p.parse_args()
    require(not args.output_dir.exists(),'output exists; refuse overwrite')
    bundle=args.run_dir.parent if args.run_dir.name.startswith('execution') else args.run_dir
    run=args.run_dir if args.run_dir.name.startswith('execution') else bundle/'execution'
    preparation=args.preparation_dir or bundle/'preparation'; source=args.source_dir or (run/'readback/pkg' if (run/'readback/pkg').is_dir() else preparation/'pkg')
    results=args.results_dir or run/'readback/results'; meta_path=preparation/'preparation.json'
    result=dict(status='UNRUN',cells=[dict(label=n,policy='ltr_packing_'+role,status='UNRUN',eligible=False) for n,role in zip(LABELS,ROLES)],comparisons=[],
        scope=SCOPE,execution_owner=args.execution_owner,source=str(source),results_dir=str(results),
        reused_accounting_sha256=sha(Path(old.__file__)),reused_work_accounting_sha256=sha(Path(base.__file__)))
    try:
        expected=args.expected_metadata_sha256 or (read(args.receipt)['preparation_sha256'] if args.receipt else None)
        if not meta_path.exists() or expected is None:
            require(not results.exists() or not any(results.iterdir()),'retained attempt lacks frozen metadata identity')
            result['preparation_checks']='WAITING_FOR_FROZEN_METADATA_AND_EXPECTED_SHA'; finish(result,args); return
        require(sha(meta_path)==expected,'frozen preparation metadata changed'); meta=read(meta_path)
        for package in {source,preparation/'pkg'}:
            require(all(sha(package/n)==h for n,h in meta['files_sha256'].items()),'frozen package content differs')
            sums=dict((name.lstrip('*'),h) for h,name in (line.split(maxsplit=1) for line in (package/'SHA256SUMS').read_text().splitlines()))
            require(sums==meta['files_sha256'],'package inventory differs')
        campaign=read(source/'campaign.json'); cells=campaign['cells']
        require(campaign['comparison']==meta['comparison']=='packing' and cells==meta['cells'] and [c['label'] for c in cells]==LABELS
                and [c['packing'] for c in cells]==ROLES and all(c['boost']=='on' and c['usable_blocks']==BLOCKS and c['kv_cache_bytes']==KV for c in cells),'frozen campaign differs')
        inp=source/'inputs_preparation/prepared/long'; workload=read(inp/'workload.json'); config=read(inp/'config.json')
        require(hashlib.sha256(json.dumps(workload,sort_keys=True).encode()).hexdigest()==config['workload_sha256']
                and len(workload['source_requests'])==len(workload['actual_prompt_token_ids'])==len(workload['arrival_traces_s']['steady'])==32,'frozen workload differs')
        command=shlex.split(meta['command']); ref=Path(command[command.index('--reference-cell')+1])
        require(sha(ref/'engine_args.json')==meta['reference_engine_args_sha256'],'fixed reference engine changed')
        reference=dict(engine=read(ref/'engine_args.json'),runtime=read(ref/'environment.json')['vllm_source_sha256'])
        require(len(reference['runtime'])==7 and all(reference['runtime'][n]==h for n,h in meta['expected_runtime_sources'].items()),'pinned runtime reference differs')
        metrics=module('packing_frozen_metrics',source/'metrics.py'); components=module('packing_frozen_counters',source/'recovery_service_components.py')
        rows=[inspect(results,c,source,meta,workload,config,metrics,components,reference) for c in cells]; result['cells']=rows
        result.update(status='UNRUN' if all(r['status']=='UNRUN' for r in rows) else 'INCOMPLETE',preparation_checks='PASS',
                      preparation_sha256=expected,archive_sha256=meta['archive_sha256'],cap32_full_history_margin_blocks=BLOCKS-32*256)
        if all(r['eligible'] for r in rows):
            pairs=[]
            for i,j,kind in [(0,1,'block0_fit_prefix'),(3,2,'block1_fit_prefix'),(0,3,'fit_repeat'),(1,2,'prefix_repeat')]:
                a,b=rows[i],rows[j]
                normalize=lambda row: dict(row,config=dict(row['config'],completion_policy='common_packing',component=dict(row['config']['component'],packing='common')))
                pair=base.compare(normalize(a),normalize(b),kind)
                for change,x,y in zip(pair['per_request_changes'],a['per_request'],b['per_request']): change['mean_tpot_s']=y['mean_tpot_s']-x['mean_tpot_s']
                pair.update(baseline_policy=a['policy'],action_policy=b['policy'],schedule_path_equal=a['component']['schedule_path_sha256']==b['component']['schedule_path_sha256'])
                pair['per_request_change_counts']={metric:dict(improved=sum(q[metric]<0 for q in pair['per_request_changes']),harmed=sum(q[metric]>0 for q in pair['per_request_changes']),equal=sum(q[metric]==0 for q in pair['per_request_changes'])) for metric in ('ttft_s','completion_s','max_itl_s','mean_tpot_s')}
                pairs.append(pair)
            result.update(status='MEASUREMENT_ONLY',comparisons=pairs)
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError) as error:
        result.update(status='INCOMPLETE',error=str(error),comparisons=[])
    finish(result,args)


if __name__=='__main__':
    main()
