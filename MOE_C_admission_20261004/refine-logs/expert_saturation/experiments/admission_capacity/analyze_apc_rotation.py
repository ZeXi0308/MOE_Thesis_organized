#!/usr/bin/env python3
"""Native/most APC-on comparison; each policy keeps its own observed trajectory."""
import argparse
import ast
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import analyze_prefix_cache_baseline as base

require, read, sha, digest = base.require, base.read, base.sha, base.digest
ROOT = Path(__file__).resolve().parents[2]/'outputs/admission_capacity/20260914_apc_rotation_r01'
LABELS = ['cohort2-block0-native_apc','cohort2-block0-most_apc','cohort2-block1-most_apc','cohort2-block1-native_apc']
ROTATION = dict(min_absence_steps=30,min_steps_between_swaps=20,free_block_slack=0,min_residency_steps=30,protect_progress_fraction=.9,max_absences_per_request=8,enabled=True)
INVENTORY = {'run_probe.py','native_capture.py','memory_telemetry.py','metrics.py','safe_static.py','rotation_native.py','absence_rotation.py'}
SCOPE = base.SCOPE + (' Installation and exclusive-live-block qualifications rely on executed frozen '
    'checks and retained receipts; this is not an independent offline replay of physical block IDs/refcounts. '
    'Qualification, instrumentation and decision costs remain inside full wall time. No new GPU data is '
    'created by this analyzer. Replaying decisions uses only each step before-state and prior actual events.')


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value; spec.loader.exec_module(value); return value


def normalize(value, aliases):
    if isinstance(value,dict):
        return {aliases.get(k,k):normalize(v,aliases) for k,v in value.items()
                if k not in ('start_s','end_s','decision_seconds','wrapped_schedule_seconds','waiting_before','waiting_requests')}
    if isinstance(value,list): return [normalize(v,aliases) for v in value]
    return aliases.get(value,value) if isinstance(value,str) else value


def check_release(d, event):
    require(event['original_preemption_called'] and event['original_preemption_returned'], 'native release failed')
    observed=event['pool_after']['free_blocks']-event['pool']['free_blocks']
    require(event['pool']['free_blocks']==d['free_before'], 'release does not start at decision pool')
    require(observed == d['actual_released_blocks'] == d['expected_released_blocks'] == d['candidate_released_blocks'], 'actual free delta differs')
    require(event['victim_state']['block_counts'] == [observed] and event['victim_state_after']['block_counts'] == [0], 'exclusive victim block count differs')
    return observed


def replay(raw, decisions, selector):
    require(len(decisions)==len(raw['scheduler_steps'])==len(raw['memory_trace']), 'decision/step alignment differs')
    tracker=selector.AbsenceRotation(selector.RotationConfig(**ROTATION),victim_order='most_output')
    events={(e['attempted_step'],e['victim_internal_request_id']):e for e in raw['preemption_events']}
    protected, output_start, active, forced, held, count, qualification_receipts = None,None,False,0,0,0,0
    for k,(d,s,m) in enumerate(zip(decisions,raw['scheduler_steps'],raw['memory_trace'])):
        require(d['step']==s['step']==m['attempted_step']==k and d['status']=='APPLIED' and d['mode']=='rotate', 'decision identity/status differs')
        before,after=m['before'],m['after']; states=before['requests']; running=before['running_ids']
        owned=lambda rid: states[rid]['block_counts'][0]
        need=lambda rid: max(0,(states[rid]['prompt_tokens']+states[rid]['output_tokens']+15)//16-owned(rid))
        pure=lambda rid: states[rid]['output_tokens']>0 and states[rid]['computed_tokens']==states[rid]['prompt_tokens']+states[rid]['output_tokens']-1
        require(d['free_before']==before['pool']['free_blocks'] and d['free_after']==after['pool']['free_blocks'], 'decision pool differs')
        require(0 <= d['decision_seconds'] <= d['wrapped_schedule_seconds'] <= s['end_s']-s['start_s']+1e-7, 'decision time is not nested in scheduler')
        active=active or (len(running)==32 and all(pure(r) and not states[r]['num_preemptions'] for r in running) and before['waiting_count']==0)
        require(d['active']==active and d['effective_victim_order']=='most_output' and d['applied_rotations_before']==forced, 'policy activation/config differs')
        if d.get('recovery_completed'):
            require(protected==d['recovery_completed'] and protected in states and states[protected]['output_tokens']>output_start, 'recovery completed without prior output')
            protected,output_start=None,None
        proposal=d.get('proposal'); expected_call=active and protected is None and all(pure(r) for r in running); funded=False
        require(bool(proposal)==expected_call, 'selector invocation differs from pre-state')
        if proposal:
            waiting=[r for r in states if r not in running and states[r]['num_preemptions']>0]
            views=[selector.RequestView(r,states[r]['computed_tokens'],states[r]['prompt_tokens'],4096,states[r]['output_tokens']) for r in running]
            previous=tracker.last_swap_step
            expected=tracker.decide(k,views,waiting,before['pool']['free_blocks'],{r:need(r) for r in waiting})
            require(asdict(expected)==proposal, 'decision differs from causal past-state replay'); count+=1
            if expected.action=='rotate':
                victim,target=expected.victim_id,expected.resume_id
                require(d['candidate_required_blocks']==need(target) and d['candidate_released_blocks']==owned(victim), 'candidate funding differs')
                q=d['pre_exchange_ownership']
                require(q['live_owned_blocks']==before['pool']['used_blocks']==sum(owned(r) for r in states), 'pre-exchange ownership receipt differs')
                qualification_receipts+=1
                if d['free_before']+d['candidate_released_blocks'] < need(target):
                    tracker.last_swap_step=previous
                    require(d.get('not_applied_reason')=='victim cannot fund complete recovery history','unfunded rejection missing')
                else:
                    funded=True
                    protected,output_start=target,states[target]['output_tokens']
                    require(d['forced_preempted']==[victim], 'funded victim not actually preempted')
                    check_release(d,events[k,victim]); forced+=1
                    q=d['post_exchange_ownership']; require(q['live_owned_blocks']==after['pool']['used_blocks'], 'post-exchange ownership receipt differs')
                    qualification_receipts+=1
        require(d['recovery_target']==protected, 'recovery target changed')
        require(funded or not d['forced_preempted'], 'forced preemption has no funded proposal')
        require(d['actual_scheduled']=={x['internal_request_id']:x['scheduled_tokens'] for x in s['scheduled']}, 'actual schedule differs')
        resumed=[x['internal_request_id'] for x in s['scheduled'] if x['internal_request_id'] not in running
                 and states[x['internal_request_id']]['num_preemptions']>0]
        require(sorted(d['resumed'])==sorted(resumed),'resumed IDs differ from prior waiting and actual scheduled state')
        require(set(d['preempted'])==set(d['forced_preempted'])|set(d['natural_preempted'])
                and sorted(raw['internal_to_source'][r] for r in d['preempted'])==sorted(s['preempted_request_ids']), 'preemption lists differ')
        if protected:
            require(not d['natural_preempted'] and d['actual_scheduled'].get(protected,0)>0 and d['free_after']>=d['recovery_remaining_blocks_after'], 'recovery reserve not maintained')
        for rid in d['held']:
            require(rid in running and rid in after['running_ids'] and rid not in d['actual_scheduled']
                    and states[rid]['computed_tokens']==after['requests'][rid]['computed_tokens']
                    and states[rid]['block_counts']==after['requests'][rid]['block_counts'], 'held state changed')
            held+=1
        tracker.note_preempted(k,d['preempted']); tracker.note_resumed(k,d['resumed'])
        if d['forced_preempted']: tracker.note_rotation_applied()
    require(forced>0, 'INVALID_NO_ACTION')
    return dict(causal_proposals_checked=count,forced_preemptions=forced,held_request_steps=held,ownership_receipts=qualification_receipts,
                qualification_scope='Live reference-count checks in pinned adapter plus offline aggregate/free-delta receipts; no offline per-block refcount replay.',
                decision_s=sum(d['decision_seconds'] for d in decisions),wrapped_schedule_s=sum(d['wrapped_schedule_seconds'] for d in decisions))


def inspect(run,spec,source,workload,input_config,metrics,selector,rotation):
    directory=run/'gpu_results'/spec['label']; row=dict(label=spec['label'],role=spec['role'],policy=spec['completion_policy'],block=spec['block'],status='UNRUN',eligible=False)
    try:
        raw_path=next((p for p in (directory/'raw.json',directory/'raw.json.gz') if p.exists()),None)
        terminal=read(directory/'status.json') if (directory/'status.json').exists() else {}
        if raw_path is None:
            row.update(status='INCOMPLETE' if directory.exists() and any(directory.iterdir()) else 'UNRUN',terminal=terminal); return row
        raw=read(raw_path); row['retained_requests']=[dict(request_id=r['request_id'],status=r['status'],outputs=len(r['output_token_ids'])) for r in raw['requests']]
        require((run/'frozen').is_dir(),'measured run lacks frozen source')
        cfg,engine,q,memory,env,reset,saved,decisions=[read(directory/n) for n in ('config.json','engine_args.json','safe-cap-qualification.json','memory-before.json','environment.json','prefix-cache-reset.json','metrics.json','headroom-decisions.json')]
        require(cfg['enable_prefix_caching'] is engine['enable_prefix_caching'] is q['prefix_caching'] is True, 'both arms require APC on')
        require(cfg['completion_policy']==spec['completion_policy'] and cfg['rotation_victim_order']==spec['victim_order']
                and cfg['rotation_config']==ROTATION and cfg['headroom_observer']=='not_installed', 'frozen policy differs')
        require(cfg['model']==input_config['model'] and cfg['workload_sha256']==base.WORKLOAD and cfg['prompt_tokens']==3072
                and cfg['output_tokens']==1024 and cfg['cap']==raw['target_cap']==32, 'workload changed')
        expected=dict(model='allenai/OLMoE-1B-7B-0924',revision=base.REVISION,tokenizer_revision=base.REVISION,dtype='bfloat16',seed=input_config['seed'],max_model_len=4096,max_num_seqs=32,max_num_batched_tokens=1024,gpu_memory_utilization=.9,enable_chunked_prefill=True,enable_prefix_caching=True,scheduling_policy='fcfs',async_scheduling=False,kv_cache_memory_bytes=base.KV,scheduler_reserve_full_isl=True,stream_interval=1,enforce_eager=False,enable_return_routed_experts=False)
        require(engine==expected and memory['kv_storage_bytes']==base.KV and q['status']=='QUALIFIED' and q['usable_blocks']==base.BLOCKS
                and q['block_size']==16 and q['observed_scheduler_reserve_full_isl'] is True and q['coordinator_type']=='UnitaryKVCacheCoordinator', 'engine/live KV qualification differs')
        require(reset['returned'] is True and reset['after']['free_blocks']==reset['after']['usable_blocks']==base.BLOCKS
                and all(reset['after'][k]==0 for k in ('nonzero_refcount_blocks','negative_refcount_blocks','hashed_blocks','pending_requests')) and reset['after']['counts']==[0,0], 'cold measurement reset failed')
        require(set(env['source_sha256'])==INVENTORY and all(sha(source/n)==h for n,h in env['source_sha256'].items())
                and env['vllm_source_sha256']==rotation.VLLM_SOURCE_SHA256 and env['vllm']=='0.26.0', 'executed source inventory differs')
        for index,count in enumerate((32,32,2)):
            warm=read(directory/f'warmup-{index}.json'); require(warm['status']=='COMPLETE' and len(warm['requests'])==count and all(r['status']=='completed' and len(r['output_token_ids'])==16 for r in warm['requests']), 'common warmup missing/failed')
        require(raw['status']==terminal['status']=='COMPLETE' and not raw['error'] and raw['capacity_boundary'] is None and raw['preemption_mode']=='native_recompute', 'measurement incomplete/failed')
        requests={r['request_id']:r for r in raw['requests']}; require(len(requests)==len(raw['requests'])==32, 'request count/uniqueness differs')
        for w,ids,arrival in zip(workload['source_requests'],workload['actual_prompt_token_ids'],workload['arrival_traces_s']['steady']):
            r=requests[w['request_id']]
            require(r['document_id']==w['document_id'] and r['prompt_tokens']==len(ids)==3072 and r['prompt_token_ids_sha256']==w['prompt_token_ids_sha256']==digest(ids)
                    and r['arrival_s']==arrival and r['status']=='completed' and len(r['output_token_ids'])==1024 and raw['internal_to_source'][r['internal_request_id']]==r['request_id'], 'input/request identity differs')
            require(r['arrival_s']<=r['admission_s']<=r['engine_add_return_s']<=r['token_times_s'][0] and r['completion_s']==r['token_times_s'][-1] and r['stop_reason']=='length', 'request time/order differs')
        computed=metrics.summarize_episode_requests(raw['requests'],observation_end_s=raw['observation_end_s'],ttft_slo_s=5.,tpot_slo_s=.2)
        require(all(saved.get(k)==v for k,v in computed.items()), 'raw and saved metrics differ')
        work=base.work_accounting(raw,requests); cache=read(directory/'cache-accounting.json')
        require(all(cache.get(k)==v for k,v in work['cache'].items()), 'cache jump accounting differs')
        pools=[m[k]['pool'] for m in raw['memory_trace'] for k in ('before','after')]
        require(pools and all(p['usable_blocks']==base.BLOCKS and 0<=p['used_blocks']<=base.BLOCKS and p['used_blocks']+p['free_blocks']==base.BLOCKS for p in pools), 'global pool does not close')
        policy=replay(raw,decisions,selector) if spec['completion_policy']=='rotate' else dict(decision_s=0.,forced_preemptions=0)
        require(spec['completion_policy']=='rotate' or decisions==[], 'native arm installed controller')
        scheduler=sum(s['end_s']-s['start_s'] for s in raw['scheduler_steps']); engine_s=sum(c['returned_s']-c['start_s'] for c in raw['engine_steps'])
        wall=computed['observation_duration_s']; require(0<=policy['decision_s']<=scheduler<=engine_s<=wall+1e-7, 'host timing buckets overlap or reverse')
        per_request=[dict(request_id=r['request_id'],ttft_s=r['token_times_s'][0]-r['arrival_s'],completion_s=r['completion_s']-r['arrival_s'],max_itl_s=max(b-a for a,b in zip(r['token_times_s'],r['token_times_s'][1:])),output_sha256=digest(r['output_token_ids'])) for r in raw['requests']]
        aliases=raw['internal_to_source']; outputs={r:q['output_token_ids'] for r,q in requests.items()}
        row.update(status='COMPLETE',eligible=True,config=cfg,engine=engine,software={k:env[k] for k in ('python','torch','cuda','vllm','transformers')},gpu_uuid=re.search(r'GPU-[\w-]+',env['gpu_before']['device']).group(),
                   usable_kv_blocks=base.BLOCKS,cap32_declared_reservation_blocks=32*256,cap32_declared_reservation_margin_blocks=base.BLOCKS-32*256,
                   per_request=per_request,wall_s=wall,throughput_rps=computed['throughput_rps'],mean_completion_s=sum(r['completion_s'] for r in per_request)/32,max_itl_s=max(r['max_itl_s'] for r in per_request),
                   ttft_s=metrics._distribution([r['ttft_s'] for r in per_request]),request_max_itl_s=metrics._distribution([r['max_itl_s'] for r in per_request]),work=work,policy_checks=policy,
                   timing=dict(scheduler_inclusive_s=scheduler,decision_subset_s=policy['decision_s'],engine_non_scheduler_s=engine_s-scheduler,outside_engine_s=wall-engine_s),
                   execution_path_sha256=digest(normalize(raw['scheduler_steps'],aliases)),decision_path_sha256=digest(normalize(decisions,aliases)),output_sha256=digest(outputs),
                   raw_path=str(raw_path),raw_sha256=sha(raw_path),_outputs=outputs)
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,RuntimeError,StopIteration) as exc:
        row.update(status='INVALID_OR_INCOMPLETE',error=str(exc),eligible=False)
    return row


def compare(a,b,kind):
    allowed={'completion_policy','rotation_victim_order'}
    trimmed=lambda row: dict(row,config={k:v for k,v in row['config'].items() if k not in allowed})
    result=base.compare(trimmed(a),trimmed(b),kind)
    result.update(baseline_policy=a['policy'],action_policy=b['policy'],same_execution_path=a['execution_path_sha256']==b['execution_path_sha256'],same_decision_path=a['decision_path_sha256']==b['decision_path_sha256'],same_outputs=a['output_sha256']==b['output_sha256'])
    return result


def cpu_checks(selector):
    tracker=selector.AbsenceRotation(selector.RotationConfig(**ROTATION),victim_order='most_output'); tracker.note_preempted(0,['waiting'])
    rows=[selector.RequestView('a',3500,3072,4096,429),selector.RequestView('b',3700,3072,4096,629)]
    p=tracker.decide(30,rows,['waiting'],5,{'waiting':230}); require(p.victim_id=='b' and p.resume_id=='waiting','past-state selector fixture failed')
    independent=selector.AbsenceRotation(selector.RotationConfig(**ROTATION),victim_order='most_output')
    no_history=independent.decide(30,rows,['waiting'],5,{'waiting':230})
    require(no_history.action=='noop' and not independent.absent_since and tracker.absent_since=={'waiting':0}, 'independent policy inherited another policy history')
    good=dict(free_before=1,actual_released_blocks=2,expected_released_blocks=2,candidate_released_blocks=2)
    receipt=dict(original_preemption_called=True,original_preemption_returned=True,pool={'free_blocks':1},pool_after={'free_blocks':3},victim_state={'block_counts':[2]},victim_state_after={'block_counts':[0]})
    require(check_release(good,receipt)==2,'valid measured-release fixture failed')
    negative=[]
    for name,fn in [('identity_alignment',lambda:base.work_accounting({'scheduler_steps':[],'engine_steps':[],'memory_trace':[{}]},{})),
                    ('actual_free_delta',lambda:check_release(good,dict(receipt,pool_after={'free_blocks':2})))]:
        try: fn()
        except (ValueError,KeyError): negative.append(name)
    require(len(negative)==2,'negative control was accepted')
    return dict(status='PASS_CPU_ONLY',identity_alignment_rejected=True,causal_before_state_selector=True,
                accounting_good_and_bad_release_checked=True,independent_policy_history_checked=True,rejections=negative,
                scope='Four bounded implementation checks only; no actual GPU state regeneration, performance or APC execution.')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); args=p.parse_args()
    require(not args.output_dir.exists(),'output directory exists; refusing overwrite')
    source=args.run_dir/'frozen'
    if not source.is_dir(): source=ROOT/'preparation/source'
    selector=module('absence_rotation',source/'absence_rotation.py'); rotation=module('apc_rotation_native',source/'rotation_native.py'); metrics=module('apc_rotation_metrics',source/'metrics.py')
    tree=ast.parse((source/'rotation_native.py').read_text())
    writes=[target.attr for node in ast.walk(tree) if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign))
            for target in getattr(node,'targets',[getattr(node,'target',None)]) if isinstance(target,ast.Attribute)
            and target.attr in {'num_computed_tokens','num_output_tokens','output_token_ids','num_tokens','_all_token_ids'}]
    require(not writes,'adapter directly assigns native token state')
    require(vars(selector.RotationConfig())==ROTATION and len(rotation.VLLM_SOURCE_SHA256)==7, 'frozen method/source contract changed')
    campaign=read(source/'campaign.json'); require([c['label'] for c in campaign['cells']]==LABELS,'campaign order differs')
    require(all(c['cohort_id']=='cohort2' and c['prefix_caching']=='on' and c['cap']==32 and c['role']==c['label'].split('-')[-1]
                and c['completion_policy']==('rotate' if c['role']=='most_apc' else 'native') and c['victim_order']==('most_output' if c['role']=='most_apc' else 'least_progress') for c in campaign['cells']), 'campaign arm changed')
    inp=source/'cohorts/cohort2/inputs_preparation/prepared/long'; workload=read(inp/'workload.json'); config=read(inp/'config.json')
    require(base.hashlib.sha256(json.dumps(workload,sort_keys=True).encode()).hexdigest()==config['workload_sha256']==base.WORKLOAD,'frozen input hash differs')
    rows=[inspect(args.run_dir,c,source,workload,config,metrics,selector,rotation) for c in campaign['cells']]
    result=dict(status='UNRUN' if all(r['status']=='UNRUN' for r in rows) else 'INCOMPLETE_CAMPAIGN',cells=rows,comparisons=[],cpu_checks=cpu_checks(selector),scope=SCOPE,
                dependencies={str(p):sha(p) for p in (Path(__file__),Path(base.__file__),source/'metrics.py',source/'absence_rotation.py',source/'rotation_native.py')})
    if all(r['eligible'] for r in rows):
        try:
            result['comparisons']=[compare(rows[i],rows[j],kind) for i,j,kind in [(0,1,'block0_native_most'),(3,2,'block1_native_most'),(0,3,'native_repeat'),(1,2,'most_repeat')]]; result['status']='MEASUREMENT_ONLY'
        except ValueError as exc: result.update(status='INVALID_PAIRING',error=str(exc),comparisons=[])
    for row in rows: row.pop('_outputs',None)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'REPORT.md').write_text('# APC-on native versus most_output\n\n'+result['status']+'\n\n'+SCOPE+'\n\n'+'\n'.join(f"- {r['label']}: {r['status']}" for r in rows)+'\n\nFull request losses/gains, timing, successful work, recovery spans and fingerprints are in analysis.json.\n')
    print(json.dumps(dict(status=result['status'],cells=[dict(label=r['label'],status=r['status']) for r in rows],cpu_checks=result['cpu_checks'])))


if __name__=='__main__': main()
