#!/usr/bin/env python3
"""Read eight complete victim-order cells, normalize IDs, extract actual paths."""
import argparse, collections, json
from pathlib import Path


def main(root, dest):
    manifest=json.loads((root/'preparation/source/campaign.json').read_text())
    execution=json.loads((root/'execution/execution.json').read_text())
    assert execution['status']=='COMPLETE', 'campaign must be COMPLETE before analysis'
    assert len(execution['cells'])==8 and all(c['status']=='READ_BACK' for c in execution['cells'])
    cells={}
    for cell in manifest['cells']:
        label=cell['label']; path=root/'execution/gpu_results'/label
        raw=json.loads((path/'raw.json').read_text()); decisions=json.loads((path/'headroom-decisions.json').read_text())
        assert raw['status']=='COMPLETE' and raw['error'] is None
        assert len(raw['requests'])==32 and all(r['status']=='completed' for r in raw['requests'])
        ids=raw['internal_to_source']; source=lambda x: ids[x] if x is not None else None
        traces={s['attempted_step']:s for s in raw['memory_trace']}
        steps={s['step']:s for s in raw['scheduler_steps']}
        engines={}
        for e in raw['engine_steps']:
            assert e['completed'] and e['scheduler_step_end']==e['scheduler_step_start']+1
            engines[e['scheduler_step_start']]=e
        assert len(decisions)==len(traces)==len(steps)==len(engines)
        bytime={e['returned_s']:s for s,e in engines.items()}
        requests={r['request_id']:r for r in raw['requests']}
        events=collections.defaultdict(list)
        for e in raw['output_events']:
            assert e['prefix_valid'] and e['chunk_size']==1
            events[e['request_id']].append(e)
        preempt={(p['attempted_step'],p['victim_internal_request_id']):p for p in raw['preemption_events']}
        absence={}; resident={}; count=collections.Counter(); legal=[]; rotations=[]; proposals=[]; recovery=[]
        for d in decisions:
            s=d['step']; before=traces[s]['before']; states=before['requests']; p=d['proposal']
            assert d['status']=='APPLIED' and traces[s]['schedule_completed']
            if p and p['action']=='rotate':
                eligible=[]
                for rid in before['running_ids']:
                    x=states[rid]; progress=max(0,x['computed_tokens']-x['prompt_tokens'])/1024
                    if progress<0.9 and count[rid]<8 and s-resident.get(rid,-10**9)>=30:
                        eligible.append(dict(internal_id=rid,request_id=source(rid),computed_tokens=x['computed_tokens'],
                            output_tokens=x['output_tokens'],computed_progress=progress,block_count=sum(x['block_counts']),
                            absences_before=count[rid],residency_steps=s-resident[rid] if rid in resident else None))
                key=(lambda x: (-x['output_tokens'],x['internal_id'])) if cell['role']=='most_output' else (lambda x:(x['computed_progress'],x['internal_id']))
                eligible.sort(key=key)
                victim=p['victim_id']; target=p['resume_id']; free=before['pool']['free_blocks']
                released=sum(states[victim]['block_counts'])
                need=(states[target]['prompt_tokens']+states[target]['output_tokens']+15)//16-sum(states[target]['block_counts'])
                tracked_wait=[rid for rid in states if rid not in before['running_ids'] and rid in absence]
                expected_target=max(tracked_wait,key=lambda rid:(s-absence[rid],rid))
                applied=bool(d['forced_preempted'])
                checks=dict(selected_order=eligible[0]['internal_id']==victim,
                    longest_absence=target==expected_target,absence_threshold=s-absence[target]>=30,
                    output_not_recompute=all(states[rid]['computed_tokens']==states[rid]['prompt_tokens']+states[rid]['output_tokens']-1 for rid in before['running_ids']),
                    released_matches=released==d['candidate_released_blocks'],needed_matches=need==d['candidate_required_blocks'],
                    funding_gate=applied==(free+released>=need),
                    native_forced_payload=(d['forced_preempted']==[victim] and victim in d['preempted'] and (s,victim) in preempt) if applied else True)
                legal.append(dict(step=s,checks=checks))
                compact=dict(step=s,resume=source(target),victim=source(victim),resume_output_tokens=states[target]['output_tokens'],
                    resume_absence_steps=s-absence[target],victim_output_tokens=states[victim]['output_tokens'],
                    victim_computed_tokens=states[victim]['computed_tokens'],victim_computed_progress=max(0,states[victim]['computed_tokens']-3072)/1024,
                    free_blocks=free,released_blocks=released,required_blocks=need,funding_margin=free+released-need,
                    applied=applied,eligible=[{k:v for k,v in x.items() if k!='internal_id'} for x in eligible],
                    not_applied_reason=d.get('not_applied_reason'))
                proposals.append(compact)
                if applied: rotations.append(compact)
            if d['recovery_target']:
                target=d['recovery_target']
                legal.append(dict(step=s,checks=dict(recovery_target_scheduled=d['actual_scheduled'].get(target,0)>0,
                    no_natural_preempt_during_protection=not d['natural_preempted'],
                    remaining_history_funded=d['free_after']>=d['recovery_remaining_blocks_after'],
                    held_not_scheduled=not any(x in d['actual_scheduled'] for x in d['held']),
                    held_preserves_kv_and_computed=all(states[x]['block_counts']==traces[s]['after']['requests'][x]['block_counts'] and states[x]['computed_tokens']==traces[s]['after']['requests'][x]['computed_tokens'] for x in d['held']))))
            for rid in d['preempted']:
                absence.setdefault(rid,s);count[rid]+=1;resident.pop(rid,None)
            for rid in d['resumed']:
                absence.pop(rid,None);resident[rid]=s
        for (s,rid),p in sorted(preempt.items()):
            req=source(rid); produced=p['victim_state']['output_tokens']; prior=[x for x in events[req] if x['cumulative_tokens']==produced]
            assert len(prior)==1
            nxt=next(x for x in events[req] if x['cumulative_tokens']>produced)
            outstep=bytime[nxt['received_s']]
            resume=next((d['step'] for d in decisions[s:] if rid in d['resumed']),None)
            assert resume is not None and resume<=outstep
            recompute=sum(x['recompute_tokens'] for k in range(resume,outstep+1) for x in steps[k]['scheduled'] if x['internal_request_id']==rid)
            row=dict(preempt_step=s,request_id=req,kind='forced' if rid in decisions[s]['forced_preempted'] else 'natural',
                produced_output_tokens_at_preempt=produced,computed_tokens_discarded=p['victim_state']['computed_tokens'],
                output_count_after_preempt=p['victim_state_after']['output_tokens'],computed_tokens_after_preempt=p['victim_state_after']['computed_tokens'],
                output_prefix_preserved=p['output_token_ids_before']==p['output_token_ids_after'],
                resume_step=resume,first_new_output_step=outstep,next_output_tokens=nxt['cumulative_tokens'],
                recomputed_tokens_to_first_new_output=recompute,scheduled_recovery_steps=outstep-resume+1,
                absence_steps=resume-s,pause_s=nxt['received_s']-prior[0]['received_s'],
                previous_output_step=bytime[prior[0]['received_s']],last_output_to_preempt_engine_start_s=engines[s]['start_s']-prior[0]['received_s'],
                preempt_to_resume_engine_start_s=engines[resume]['start_s']-engines[s]['start_s'],
                resume_engine_start_to_first_new_output_s=nxt['received_s']-engines[resume]['start_s'],
                resumed_under_protection=decisions[resume]['recovery_target']==rid,
                resume_forced_victim=[source(x) for x in decisions[resume]['forced_preempted']],
                held_request_steps_during_recovery=sum(len(decisions[k]['held']) for k in range(resume,outstep+1)),
                after_recovery_protection_completed_step=next((d['step'] for d in decisions[outstep+1:] if d.get('recovery_completed')==rid),None))
            assert abs(row['pause_s']-sum(row[k] for k in ('last_output_to_preempt_engine_start_s','preempt_to_resume_engine_start_s','resume_engine_start_to_first_new_output_s')))<1e-10
            recovery.append(row)
        pauses=[]
        for req,r in requests.items():
            times=r['token_times_s']; i=max(range(1,len(times)),key=lambda i: times[i]-times[i-1])
            pauses.append(dict(request_id=req,gap_s=times[i]-times[i-1],after_output_token=i,before_output_token=i+1,
                previous_output_step=bytime[times[i-1]],next_output_step=bytime[times[i]],
                preemption_paths=[x for x in recovery if x['request_id']==req and x['produced_output_tokens_at_preempt']==i]))
        pauses.sort(key=lambda x:-x['gap_s'])
        for pause in pauses[:5]:
            lo,hi=pause['previous_output_step']+1,pause['next_output_step']
            interval=decisions[lo:hi+1]
            pause['interval_decision_reasons']=dict(collections.Counter(d.get('not_applied_reason') or (d['proposal']['reason'] if d.get('proposal') else 'protected recovery') for d in interval))
            pause['interval_rotations']=[x for x in rotations if lo<=x['step']<=hi]
            pause['interval_natural_preemptions']=[dict(step=d['step'],request_ids=[source(x) for x in d['natural_preempted']]) for d in interval if d['natural_preempted']]
            pause['interval_recovery_target_steps']=dict(collections.Counter(source(d['recovery_target']) for d in interval if d['recovery_target']))
        failed=[dict(step=x['step'],check=k) for x in legal for k,v in x['checks'].items() if not v]
        first=rotations[0]
        b=traces[first['step']]['before']
        normalized_prestate=dict(running_ids=sorted(source(x) for x in b['running_ids']),free_blocks=b['pool']['free_blocks'],
            requests={source(rid):x for rid,x in b['requests'].items()})
        cellresult=dict(label=label,role=cell['role'],cohort=cell['cohort_id'],block=cell['block'],
            raw_path=str(path/'raw.json'),decisions_path=str(path/'headroom-decisions.json'),
            preemption_count=len(preempt),forced_count=sum(len(d['forced_preempted']) for d in decisions),
            natural_count=sum(len(d['natural_preempted']) for d in decisions),
            resume_count=sum(len(d['resumed']) for d in decisions),
            completed_protected_recoveries=sum(bool(d.get('recovery_completed')) for d in decisions),
            held_steps=sum(bool(d['held']) for d in decisions),held_request_steps=sum(len(d['held']) for d in decisions),
            insufficient_funding_proposals=[p for p in proposals if not p['applied']],
            not_applied_reasons=dict(collections.Counter(d.get('not_applied_reason') for d in decisions if d.get('not_applied_reason'))),
            noop_reasons=dict(collections.Counter(d['proposal']['reason'] for d in decisions if d.get('proposal') and d['proposal']['action']=='noop')),
            legal_check_count=sum(len(x['checks']) for x in legal),legal_failures=failed,
            first_forced=first,first_forced_prestate=normalized_prestate,rotations=rotations,recovery_paths=recovery,
            longest_pause=pauses[0],top_pauses=pauses[:5],total_recomputed_tokens=sum(x['recompute_tokens'] for x in steps.values()))
        cells[label]=cellresult
    pairs=[]
    for cohort in ('cohort0','cohort1'):
        for block in (0,1):
            a=cells[f'{cohort}-block{block}-least_progress'];b=cells[f'{cohort}-block{block}-most_output']
            aseq=[(r['step'],r['victim'],r['resume']) for r in a['rotations']];bseq=[(r['step'],r['victim'],r['resume']) for r in b['rotations']]
            firstidx=next((i for i,(x,y) in enumerate(zip(aseq,bseq)) if x!=y),None)
            pairs.append(dict(cohort=cohort,block=block,first_forced_sequence_difference_index=firstidx,
                A_first_forced=a['first_forced'],B_first_forced=b['first_forced'],
                normalized_prestate_equal=a['first_forced_prestate']==b['first_forced_prestate'],
                first_actual_preemption_victim_difference=next((dict(index=i,A=x,B=y) for i,(x,y) in enumerate(zip(a['recovery_paths'],b['recovery_paths'])) if x['request_id']!=y['request_id']),None),
                common_victim_prefix=[x['request_id'] for i,x in enumerate(a['recovery_paths']) if i<len(b['recovery_paths']) and all(xx['request_id']==yy['request_id'] for xx,yy in zip(a['recovery_paths'][:i+1],b['recovery_paths'][:i+1]))],
                A_longest_pause=a['longest_pause'],B_longest_pause=b['longest_pause']))
    result=dict(evidence_type='NATIVE_SERVING_INPROCESS_HOST_CAPTURE_ACTUAL_ACTION_PATH',
        boundary='This is an action-path diagnosis. Paired states after the first treatment action are allowed to diverge; output counts are actual engine-produced outputs, while recomputed tokens are disjoint executed-history work. No semantic quality or causal performance attribution is established.',
        pause_accounting='Max host-received ITL = previous output receipt to preempt engine start + preempt engine start to resume engine start + resume engine start to first new output receipt. The middle bucket includes the preemption step; these are host engine boundaries, not pure GPU kernel times.',
        cells=cells,pairs=pairs)
    dest.mkdir(parents=True,exist_ok=True);(dest/'paths.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='DONE',cells=len(cells),failed_legal_checks=sum(len(c['legal_failures']) for c in cells.values()),out=str(dest/'paths.json')),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();main(args.root,args.out)
