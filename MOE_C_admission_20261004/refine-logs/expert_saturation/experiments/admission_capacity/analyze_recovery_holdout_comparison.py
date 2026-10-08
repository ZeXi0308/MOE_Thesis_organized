#!/usr/bin/env python3
"""Eight actual cohort3 native/rotation/component runs; no presumed winner."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import inspect as inspect_source
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import analyze_restore_token_reservation as token
import analyze_apc_rotation as rotation
import analyze_completion_headroom as headroom

prior=token.prior
base,read,require,sha,digest,module=token.base,token.read,token.require,token.sha,token.digest,token.module
DEFAULT=Path(__file__).resolve().parents[2]/'outputs/admission_capacity/20260914_recovery_holdout_comparison_r01'
INPUT_SHA='4775c84082d6afb6067b8060cda336e23fdb2a7d776fe70f72be4d09315c6c18'
WORKLOAD='34aeca96e0537736234aa8b4cba835dcbc54fec60869959b87ab391f0dbce554'
TOKEN_SHA='f5def0228345f7fabdbc883776fa22b923412e36130cbbfedbb793aaf7e28342'
ROTATION_SHA='1c728ec89fef52c3e908ba17ce0ee1e51bcdd5e4094793d7b06436f1076b5074'
ORDER=['native','most_output','fit_scan','guard_residual'];VARIANTS=ORDER+ORDER[::-1]
LABELS=[f'cohort3-block{i//4}-{v}' for i,v in enumerate(VARIANTS)]
ARMS={'native':('native','native','native','least_progress','native',False,False,False),
      'most_output':('rotation','strong_simple_rotation','rotate','most_output','native',False,False,False),
      'fit_scan':('component','ltr_component','restore_token_reservation_fit_scan','least_progress','fit_scan',True,False,False),
      'guard_residual':('component','ltr_component','restore_token_reservation_guard_residual','least_progress','rank_prefix',True,True,True)}
FIELDS=('adapter','policy_family','completion_policy','victim_order','packing','boost','complete_restores','reserve_ready_tokens')
SCOPE=('One frozen document-disjoint cohort from the same source distribution; two reverse-order blocks. '
    'All eight actual runs must qualify before comparison. Native/most use the measured d6 6656-block '
    'resource reference; components retain original LTR200/10 and first-output ledger. Full wall and '
    'nested decision/scheduler cost, all request losses, repeats and output differences remain. '
    'No SLO, quality, Oracle, independence-of-eight-workloads, significance or method-GO claim.')


def spec_config(cfg,spec,config):
    valid=(all(cfg[k]==v for k,v in config.items()) and cfg['variant']==spec['variant']
        and cfg['policy_family']==spec['policy_family'] and cfg['completion_policy']==spec['completion_policy']
        and cfg['rotation_victim_order']==spec['victim_order'] and cfg['rotation_config']==rotation.ROTATION
        and cfg['headroom_observer']=='fast' and cfg['reservation_policy']=='full'
        and cfg['fixed_kv_cache_memory_bytes']==token.KV and cfg['preemption_mode']=='native_recompute')
    expected=dict(boost=True,threshold=200,quantum=10,base_order='FCFS',
        backend='priority packing / current-history reservation / native recompute',packing=spec['packing'],
        complete_restores=spec['complete_restores'],reserve_ready_tokens=spec['reserve_ready_tokens'])
    return valid and (cfg.get('component')==expected if spec['adapter']=='component' else 'component' not in cfg)


def old_release(d,event):
    released=event['pool_after']['free_blocks']-event['pool']['free_blocks']
    require(event['original_preemption_called'] and event['original_preemption_returned']
        and event['pool']['free_blocks']==d['free_before']
        and released==d['candidate_released_blocks']==event['victim_state']['block_counts'][0]
        and event['victim_state_after']['block_counts']==[0], 'd6 native release/free delta differs')
    return released


def make_rotation_replay():
    require(sha(Path(rotation.__file__))==ROTATION_SHA,'reused rotation replay changed')
    source=inspect_source.getsource(rotation.replay)
    # APC-off d6 has no later APC adapter ownership-receipt fields. Keep the causal
    # tracker, full-history funding, actual release, held/resume and timing checks.
    lines=[line for line in source.splitlines() if not any(s in line for s in
        ("q=d['pre_exchange_ownership']","q['live_owned_blocks']","qualification_receipts+=1"))]
    ns=dict(rotation.replay.__globals__,check_release=old_release)
    exec(compile('\n'.join(lines),__file__+':d6-rotation-replay','exec'),ns)
    return ns['replay']


rotation_replay=make_rotation_replay()
require(sha(Path(token.__file__))==TOKEN_SHA,'reused token analyzer changed')
replay_source=token.replace_once(inspect_source.getsource(token.replay),"flag = spec['reserve_ready_tokens']=='on'","flag = spec['reserve_ready_tokens']")
replay_ns=dict(token.replay.__globals__)
exec(compile(replay_source,__file__+':boolean-spec-replay','exec'),replay_ns)
component_replay=replay_ns['replay']


def native_accounting(raw,decisions,spec,selector,metrics):
    policy=spec['completion_policy'];held=headroom.held_accounting(raw,decisions,policy)
    aliases=raw['internal_to_source'];requests={r['request_id']:r for r in raw['requests']}
    calls={k:c for c in raw['engine_steps'] for k in range(c['scheduler_step_start'],c['scheduler_step_end'])}
    events=[dict(e,request_id=aliases[e['victim_internal_request_id']],
        host_call_interval_s=[calls[e['attempted_step']]['start_s'],calls[e['attempted_step']]['returned_s']]) for e in raw['preemption_events']]
    counts=headroom.preemption_accounting(raw,decisions,events,policy,metrics._distribution)
    totals=Counter();path=[]
    for d,s,m in zip(decisions,raw['scheduler_steps'],raw['memory_trace']):
        before,after=m['before'],m['after'];states,post=before['requests'],after['requests']
        actual=d['actual_scheduled'];victims=set(d['preempted']);running=set(before['running_ids'])
        require(set(states)==set(post) and sum(actual.values())<=1024 and m['schedule_completed'],'native schedule/live state differs')
        for snap in (before,after):
            pool=snap['pool'];owned=[v['block_counts'] for v in snap['requests'].values()]
            require(pool['total_blocks']==6657 and pool['usable_blocks']==6656
                and 0<=pool['free_blocks']<=6656 and pool['used_blocks']+pool['free_blocks']==6656
                and all(len(v)==1 and v[0]>=0 for v in owned) and sum(v[0] for v in owned)==pool['used_blocks'],'APC-off owned/free conservation differs')
        for rid,v in states.items():
            require(v['output_tokens']==bisect_right(requests[aliases[rid]]['token_times_s'],s['start_s'])
                and post[rid]['output_tokens']==v['output_tokens'],'native snapshot has unavailable output')
            require(post[rid]['computed_tokens']==(0 if rid in victims else v['computed_tokens']+actual.get(rid,0)),
                'unexpected native token-state mutation')
            if rid in running-set(actual)-victims:require(post[rid]==v,'unscheduled resident changed state')
        elapsed=s['end_s']-s['start_s'];decision=d['decision_seconds']
        require(math.isfinite(decision) and 0<=decision<=elapsed+1e-7,'native decision cost outside scheduler')
        if 'wrapped_schedule_seconds' in d:
            require(decision<=d['wrapped_schedule_seconds']<=elapsed+1e-7,'rotation wrapped cost not nested')
            totals.update(wrapped_scheduler_s=d['wrapped_schedule_seconds'])
        totals.update(decision_s=decision,scheduler_s=elapsed,victim_events=len(victims),
            selected_request_calls=len(actual),held_resident_calls=len(running-set(actual)-victims))
        path.append(dict(selected=[[aliases[r],n] for r,n in actual.items()],victims=[aliases[r] for r in d['preempted']],
            held=sorted(aliases[r] for r in running-set(actual)-victims)))
    for e in raw['preemption_events']:
        released=e['pool_after']['free_blocks']-e['pool']['free_blocks']
        require(released==e['victim_state']['block_counts'][0] and e['victim_state_after']['block_counts']==[0], 'native actual block release differs')
        totals.update(released_blocks=released)
    replay=rotation_replay(raw,decisions,selector) if spec['adapter']=='rotation' else dict(status='NATIVE_OBSERVER_ONLY')
    replay.pop('ownership_receipts',None);replay.pop('qualification_scope',None)
    return dict(component=dict(totals=dict(totals),schedule_path_sha256=digest(path),held=held,
            cost_relation='decision and any recorded wrapped scheduler are nested in total scheduler; native wrapper total is not separately recorded'),
        native_preemption_accounting=counts,rotation_replay=replay,action_eligible=True)


def action_accounting(folder,source,raw,decisions,spec,components,metrics):
    if spec['adapter']!='component':
        selector=module('holdout_rotation_selector',source/'absence_rotation.py')
        return native_accounting(raw,decisions,spec,selector,metrics)
    enabled=spec['complete_restores']
    require(all(d['boost'] is True and d['packing']==spec['packing'] and d['complete_restores'] is enabled
        and d['reserve_ready_tokens'] is spec['reserve_ready_tokens'] for d in decisions),'actual component arm differs')
    helper=module('holdout_restore_helper',source/'restore_obligation.py')
    expected,obligations=prior.replay_obligations(raw,decisions,components,helper,enabled)
    require(obligations['snapshot']==read(folder/'restore-obligations.json'),'actual obligation ledger differs')
    component=prior.component_accounting(raw,decisions,True,components.LTRCounters(200,10),expected)
    component['quantum_epochs']=prior.quantum_epochs(raw,decisions,True)
    require(sum(e['selected_calls'] for e in component['quantum_epochs']['epochs'])==component['totals'].get('boosted_counter_calls',0),'quantum selected accounting differs')
    replay=component_replay(folder,source,raw,decisions,spec)
    eligible=not enabled or (obligations['overlay_selected_calls']>0 and replay['token_budget_action_counts'].get('protected_changed_steps',0)>0)
    return dict(component=component,restore_obligations=obligations,planner_replay=replay,action_eligible=eligible,
        obligation_ledger_sha256=sha(folder/'restore-obligations.json'))


def make_inspector():
    require(sha(Path(prior.__file__))==token.PRIOR_SHA,'reused common inspector changed')
    source=inspect_source.getsource(prior.inspect);replace=token.replace_once
    start=source.index('        require(all(cfg[k] == v for k,v in config.items())');end=source.index("        require(q['status']",start)
    source=replace(source,source[start:end],"        require(spec_config(cfg,spec,config) and cfg['cap']==raw['target_cap']==32,'canonical variant/input config differs')\n")
    source=replace(source,"policy='restore_completion_'+spec['complete_restores']","policy=spec['completion_policy'],variant=spec['variant'],policy_family=spec['policy_family']")
    start=source.index("        decisions = read(folder/'component-decisions.json')");end=source.index('        for pause in work[',start)
    source=replace(source,source[start:end],"        decision_path=folder/('component-decisions.json' if spec['adapter']=='component' else 'headroom-decisions.json')\n        decisions=read(decision_path)\n        details=action_accounting(folder,source,raw,decisions,spec,components,metrics)\n        component=details['component']\n")
    start=source.index('recovery_residencies=packing.residencies(raw),planner_replay=');end=source.index('            wall_s=',start)
    source=replace(source,source[start:end],"recovery_residencies=packing.residencies(raw),\n")
    source=replace(source,"decisions_sha256=sha(folder/'component-decisions.json')","decisions_sha256=sha(decision_path)")
    source=replace(source,"        if enabled and obligations['overlay_selected_calls']==0:","        row.update({k:v for k,v in details.items() if k!='component'})\n        if not details['action_eligible']:")
    source=replace(source,"error='no actual selected active obligation priority -2'","error='no actual protected token-reservation action'")
    ns=dict(prior.inspect.__globals__,spec_config=spec_config,action_accounting=action_accounting)
    exec(compile(source,__file__+':shared-request-inspector','exec'),ns)
    return ns['inspect']


inspect_cell=make_inspector()


def pair_settings(a,b):
    excluded={'variant','policy_family','completion_policy','rotation_victim_order','component'}
    return a['engine']==b['engine'] and {k:v for k,v in a['config'].items() if k not in excluded}=={k:v for k,v in b['config'].items() if k not in excluded}


source=inspect_source.getsource(token.compare).replace("a['config']['component']","a['config'].get('component')").replace("b['config']['component']","b['config'].get('component')")
ns=dict(token.compare.__globals__,pair_settings=pair_settings)
exec(compile(source,__file__+':shared-metrics-compare','exec'),ns);compare=ns['compare']


def finish(result,args):
    for row in result['cells']:row.pop('_outputs',None)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'REPORT.md').write_text('# Cohort3 recovery comparison\n\n'+result['status']+'\n\n'+SCOPE+'\n\n'+
        '\n'.join(f"- {r['label']}: {r['status']}" for r in result['cells'])+'\n\n'+result.get('error','All raw-derived request losses, output differences, full costs and repeat comparisons remain in analysis.json.')+'\n')
    print(json.dumps(dict(status=result['status'],preparation_checks=result.get('preparation_checks'),error=result.get('error'),
        cells=[dict(label=r['label'],status=r['status'],error=r.get('error')) for r in result['cells']])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,default=DEFAULT);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--results-dir',type=Path);p.add_argument('--source-dir',type=Path);p.add_argument('--preparation-dir',type=Path)
    p.add_argument('--expected-metadata-sha256',required=True);args=p.parse_args();require(not args.output_dir.exists(),'refuse output overwrite')
    bundle=args.run_dir.parent if args.run_dir.name.startswith('execution') else args.run_dir
    run=args.run_dir if args.run_dir.name.startswith('execution') else bundle/'execution';prep=args.preparation_dir or bundle/'preparation'
    source=args.source_dir or (run/'readback/pkg' if (run/'readback/pkg').is_dir() else prep/'pkg');results=args.results_dir or run/'readback/results'
    result=dict(status='UNRUN',cells=[dict(label=n,variant=v,status='UNRUN',eligible=False) for n,v in zip(LABELS,VARIANTS)],comparisons=[],scope=SCOPE,
        reused_token_analyzer_sha256=TOKEN_SHA,reused_rotation_analyzer_sha256=ROTATION_SHA,source=str(source),results_dir=str(results))
    try:
        meta_path=prep/'preparation.json'
        if not meta_path.exists():
            require(not results.exists() or not any(results.iterdir()),'retained attempt lacks metadata')
            result['preparation_checks']='WAITING_FOR_FROZEN_PACKAGE';finish(result,args);return
        require(sha(meta_path)==args.expected_metadata_sha256,'frozen metadata changed');meta=read(meta_path)
        for package in {source,prep/'pkg'}:
            require(all(sha(package/n)==h for n,h in meta['files_sha256'].items()),'frozen source differs')
            sums=dict((n.lstrip('*'),h) for h,n in (line.split(maxsplit=1) for line in (package/'SHA256SUMS').read_text().splitlines()))
            require(sums==meta['files_sha256'],'frozen inventory differs')
        campaign=read(source/'campaign.json');cells=campaign['cells']
        expected=[dict(label=n,cohort_id='cohort3',block=i//4,variant=v,kv_cache_bytes=token.KV,usable_blocks=6656,cap=32,
            **dict(zip(FIELDS,ARMS[v]))) for i,(n,v) in enumerate(zip(LABELS,VARIANTS))]
        require(cells==meta['cells']==expected and campaign['comparison']==meta['comparison']=='native_vs_most_output_vs_fit_scan_vs_guard_residual','canonical eight arms differ')
        receipts=source/'source_receipts';require(sha(receipts/'holdout_PREPARATION.json')==meta['holdout_preparation_sha256']==campaign['input_preparation_sha256']==INPUT_SHA,'holdout receipt differs')
        inp=source/'inputs_preparation/prepared/long';workload=read(inp/'workload.json');config=read(inp/'config.json')
        require(hashlib.sha256(json.dumps(workload,sort_keys=True).encode()).hexdigest()==config['workload_sha256']==meta['holdout_workload_sha256']==WORKLOAD,'cohort3 workload differs')
        inputs=read(receipts/'holdout_inputs_report.json');prior_rows=inputs['excluded_prior_requests'];all_rows=prior_rows+workload['source_requests']
        require(len(prior_rows)==128 and len(workload['source_requests'])==32 and all(len({r[k] for r in all_rows})==160 for k in ('document_id','document_sha256','prompt_token_ids_sha256')),'holdout overlaps prior identities')
        require(workload['arrival_traces_s']['steady']==[round(i*.05,10) for i in range(32)] and all(len(t)==3072 for t in workload['actual_prompt_token_ids']),'holdout lengths/arrival identity differs')
        ref=DEFAULT.parent/'20260914_d6_strong_baselines_r01/readback/results/block0-d6-most_output'
        reference=dict(engine=read(ref/'engine_args.json'),runtime=read(ref/'environment.json')['vllm_source_sha256'])
        require(reference['engine']['kv_cache_memory_bytes']==token.KV and read(ref/'safe-cap-qualification.json')['usable_blocks']==6656
            and len(reference['engine'])==18 and len(reference['runtime'])==7
            and all(reference['runtime'][n]==h for n,h in meta['expected_runtime_sources'].items()),'actual d6 reference differs')
        token_source=DEFAULT.parent/'20260914_restore_token_reservation_r01/preparation/pkg'
        require(all(sha(source/n)==sha(token_source/n) for n in ('restore_obligation.py','recovery_service_components.py','ltr_recompute_native.py','rotation_native.py','absence_rotation.py')),'frozen action/helper code changed')
        metrics=module('holdout_comparison_metrics',source/'metrics.py');components=module('holdout_comparison_counters',source/'recovery_service_components.py')
        rows=[inspect_cell(results,c,source,meta,workload,config,metrics,components,reference) for c in cells]
        result.update(cells=rows,status='UNRUN' if all(r['status']=='UNRUN' for r in rows) else 'INCOMPLETE',preparation_checks='PASS',metadata_sha256=args.expected_metadata_sha256,
            actual_resource_reference=str(ref),actual_reference_engine_sha256=sha(ref/'engine_args.json'))
        if all(r['eligible'] for r in rows):
            pairs=[(b+a,b+c,f'block{block}_{VARIANTS[b+a]}_to_{VARIANTS[b+c]}') for block,b in ((0,0),(1,4)) for a in range(4) for c in range(a+1,4)]
            # Reverse block order is an execution order, not baseline identity.
            pairs=[(j,i,k) if ORDER.index(rows[i]['variant'])>ORDER.index(rows[j]['variant']) else (i,j,k) for i,j,k in pairs]
            pairs=[(i,j,f"block{i//4}_{rows[i]['variant']}_to_{rows[j]['variant']}") for i,j,_ in pairs]
            pairs += [(i,7-i,ORDER[i]+'_repeat') for i in range(4)]
            result.update(status='MEASUREMENT_ONLY',comparisons=[compare(rows[i],rows[j],kind) for i,j,kind in pairs])
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,RuntimeError) as error:
        result.update(status='INCOMPLETE',error=str(error),comparisons=[])
    finish(result,args)


if __name__=='__main__':main()
