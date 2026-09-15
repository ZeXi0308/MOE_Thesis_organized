"""Fixed-length, closed-cohort resource simulation. No model execution or timing.

Predictive mode uses initial state and pre-cutoff controller history. Optional
load-completion replay is explicitly observational, not a future predictor.
Generated token identities are not modeled. Each candidate has its own KV.
"""
from copy import deepcopy
from math import ceil


def simulate(initial, history, action, selector_class, view_class, max_steps=5000, start_step=329, first_victim=None, override_step=None, decision_log=None, skip_step=None, stop_step=None, native_preemptions=None, saved_prefixes=None, restore_delay_steps=2, observed_load_completions=None, native_resume_first=None, staged_recovery_targets=None):
    if saved_prefixes and (action!='native' or restore_delay_steps<1):
        raise ValueError('Saved-prefix simulation requires native policy and positive load delay')
    if staged_recovery_targets and (action!='native' or not native_preemptions
            or not set(staged_recovery_targets)<=set(native_preemptions)):
        raise ValueError('Staged recovery requires matching explicit native events')
    repeated_staging=action in ('staged_most_off','staged_most_on')
    save_repeated=action=='staged_most_on'
    if repeated_staging and (saved_prefixes or observed_load_completions is not None or restore_delay_steps<1):
        raise ValueError('Repeated candidates use only their own generated prefixes and assumed load delay')
    if save_repeated:saved_prefixes={}
    pending_plan=None;staging_events=[];stored_tokens=0;peak_host_tokens=0
    pending_loads={}
    if observed_load_completions is not None and not saved_prefixes:
        raise ValueError('Completion replay requires saved-prefix state')
    states=deepcopy(initial['requests'])
    for r in states.values():r['allocated']=len(r.pop('blocks'))
    running=list(initial['running']);waiting=list(initial['waiting']);free=initial['free']
    order={'most':'first_most_then_least','continuous_most':'most_output','staged_most_off':'most_output','staged_most_on':'most_output'}.get(action,'least_progress')
    tracker=selector_class(victim_order=order)
    tracker.config.enabled=action!='native'
    capacity=free+sum(r['allocated'] for r in states.values())
    for h in history:
        tracker.note_preempted(h['step'],h['preempted'])
        tracker.note_resumed(h['step'],h['resumed'])
    protected=None;output_at_start=None;deferred=False;trace=[];completed={}
    def total(r):return r['prompt']+r['output']
    def need(r):return max(0,ceil(total(r)/16)-r['allocated'])
    def pure(r):return r['output']>0 and r['computed']==total(r)-1
    for step in range(start_step,max_steps):
        if not states:break
        if stop_step is not None and step>=stop_step:tracker.config.enabled=False
        pre_free=free;preempted=[];resumed=[];forced=None;scheduled={};budget=1024;loads=[]
        def preempt(rid):
            nonlocal free,budget
            if rid in scheduled:budget+=scheduled.pop(rid)['tokens']
            running.remove(rid);r=states[rid];free+=r['allocated']
            r['allocated']=0;r['computed']=0;r['preemptions']+=1;r['status']='PREEMPTED'
            waiting.insert(0,rid);preempted.append(rid)
        # Explicit candidate actions use the native notification path: unlike a
        # rotation, they do not allow waiting admission in the same call.
        if native_preemptions and step in native_preemptions:
            if action!='native':raise ValueError('Explicit native events require native policy')
            victim=native_preemptions[step]
            target=staged_recovery_targets.get(step) if staged_recovery_targets else None
            if target is not None:
                if target not in waiting or target==victim:
                    raise ValueError('Staged target must already be absent')
                if free+states[victim]['allocated']<need(states[target]):
                    raise ValueError('Staged victim cannot fund recovery')
                protected=target;output_at_start=states[target]['output'];forced=victim
            preempt(victim)
            if target is not None:
                waiting.remove(target);waiting.insert(0,target)
        if native_resume_first and step in native_resume_first:
            if action!='native':raise ValueError('Explicit waiting order requires native policy')
            target=native_resume_first[step]
            if target not in waiting:raise ValueError('Promotion target is not waiting')
            waiting.remove(target);waiting.insert(0,target)
        if protected is not None and states[protected]['output']>output_at_start:
            protected=None;output_at_start=None
        committed_this_step=False
        if pending_plan is not None:
            p=pending_plan;pending_plan=None;victim,target=p['victim'],p['target']
            valid=(victim in running and target in waiting and target not in pending_loads
                   and pure(states[victim]) and states[victim]['computed']==p['computed']+1
                   and states[victim]['output']==p['output']+1
                   and states[target]==p['target_state']
                   and free+states[victim]['allocated']>=need(states[target]))
            staging_events.append(dict(step=step,event='commit' if valid else 'cancel',victim=victim,target=target))
            if valid:
                protected=target;output_at_start=states[target]['output'];forced=victim
                preempt(victim);waiting.remove(target);waiting.insert(0,target)
                committed_this_step=True
        # Native adapter waits until skipped async loads have rejoined scheduling.
        if protected is None and not committed_this_step and (not repeated_staging or not pending_loads) and all(pure(states[r]) for r in running):
            rows=[view_class(r,states[r]['computed'],states[r]['prompt'],states[r]['prompt']+states[r]['max_tokens'],states[r]['output']) for r in running]
            old_swap=tracker.last_swap_step
            proposal=tracker.decide(step,rows,waiting,free,{r:need(states[r]) for r in waiting})
            if proposal.action=='rotate':
                target,victim=proposal.resume_id,proposal.victim_id
                eligible=[r.request_id for r in rows if r.progress<tracker.config.protect_progress_fraction
                    and tracker.absence_count.get(r.request_id,0)<tracker.config.max_absences_per_request
                    and step-tracker.resident_since.get(r.request_id,-10**9)>=tracker.config.min_residency_steps]
                fundable=[rid for rid in eligible if free+states[rid]['allocated']>=need(states[target])]
                if decision_log is not None:
                    decision_log.append(dict(step=step,target=target,default_victim=victim,
                        fundable=fundable,outputs={rid:states[rid]['output'] for rid in running},
                        free=free,target_need=need(states[target])))
                apply_override=(step==override_step if override_step is not None else tracker.applied_rotations==0)
                if first_victim is not None and apply_override:
                    if override_step is None and step!=start_step:
                        raise ValueError('explicit first-victim branch must apply at input state')
                    if first_victim not in fundable:raise ValueError('ineligible or unfundable victim')
                    victim=first_victim
                if free+states[victim]['allocated']<need(states[target]):tracker.last_swap_step=old_swap
                elif step==skip_step:
                    pass  # suppress this swap; consume its normal cooldown
                elif action=='defer' and not deferred:
                    tracker.last_swap_step=old_swap;deferred=True
                elif repeated_staging:
                    if target in pending_loads or states[victim]['output']+1>=states[victim]['max_tokens']:
                        tracker.last_swap_step=old_swap
                    else:
                        pending_plan=dict(step=step,victim=victim,target=target,
                            computed=states[victim]['computed'],output=states[victim]['output'],
                            target_state=deepcopy(states[target]),prefix=states[victim]['computed']//16*16)
                        staging_events.append(dict(step=step,event='prepare',victim=victim,target=target,prefix=pending_plan['prefix']))
                else:
                    protected=target;output_at_start=states[target]['output'];forced=victim
                    preempt(victim)
                    waiting.remove(target);waiting.insert(0,target)
        index=0
        while index<len(running) and budget>0:
            rid=running[index];r=states[rid]
            tokens=min(total(r)-r['computed'],budget)
            cost=max(0,ceil((r['computed']+tokens)/16)-r['allocated'])
            if protected is not None and rid!=protected:
                one_cost=max(0,ceil((r['computed']+1)/16)-r['allocated'])
                if one_cost>free-need(states[protected]):index+=1;continue
            while cost>free and running:
                victim=running[-1];preempt(victim)
                if victim==rid:break
            if rid not in running:break
            free-=cost;r['allocated']+=cost
            scheduled[rid]=dict(computed=r['computed'],tokens=tokens,output=r['output'])
            budget-=tokens;index+=1
        if len(preempted)==(1 if forced else 0):
            deferred_waiting=[]
            while waiting and budget>0:
                rid=waiting[0];r=states[rid]
                if rid in pending_loads:
                    if pending_loads[rid] is None or step<pending_loads[rid]:
                        deferred_waiting.append(waiting.pop(0));continue
                    del pending_loads[rid]
                if protected is not None and rid!=protected:break
                if free<need(r):break  # full-history admission check
                if saved_prefixes and rid in saved_prefixes and r['computed']==0:
                    prefix=saved_prefixes[rid]
                    if prefix<=0 or prefix%16 or prefix>total(r)-1:
                        raise ValueError('Invalid saved full-block prefix')
                    cost=prefix//16-r['allocated']
                    if cost<0:raise ValueError('Unexpected pre-load allocation')
                    free-=cost;r['allocated']+=cost;r['computed']=prefix
                    pending_loads[rid]=(step+restore_delay_steps if observed_load_completions is None else None)
                    loads.append(dict(request=rid,tokens=prefix,ready_step=pending_loads[rid]))
                    deferred_waiting.append(waiting.pop(0));continue
                tokens=min(total(r)-r['computed'],budget)
                cost=max(0,ceil((r['computed']+tokens)/16)-r['allocated'])
                free-=cost;r['allocated']+=cost
                waiting.pop(0);running.append(rid);r['status']='RUNNING';resumed.append(rid)
                scheduled[rid]=dict(computed=r['computed'],tokens=tokens,output=r['output'])
                budget-=tokens
            waiting=deferred_waiting+waiting
        if save_repeated and pending_plan is not None:
            p=pending_plan;rid=p['victim'];entry=scheduled.get(rid)
            if entry is not None and entry['tokens']==1 and entry['computed']==p['computed']:
                old=saved_prefixes.get(rid,0)
                stored_tokens+=max(0,p['prefix']-old)
                saved_prefixes[rid]=max(old,p['prefix'])
                peak_host_tokens=max(peak_host_tokens,sum(saved_prefixes.values()))
        after_schedule=free
        tracker.note_preempted(step,preempted);tracker.note_resumed(step,resumed)
        if forced:tracker.note_rotation_applied()
        outputs=[];finished=[]
        for rid,s in scheduled.items():
            r=states[rid];r['computed']+=s['tokens']
            if r['computed']==total(r):
                r['output']+=1;outputs.append(rid)
                if r['output']==r['max_tokens']:
                    free+=r['allocated'];running.remove(rid);finished.append(rid)
                    completed[rid]=step;del states[rid]
                    if save_repeated:saved_prefixes.pop(rid,None)
        # Worker notifications belong to the END of the engine call. Making a
        # completed load schedulable in the same call would use future data.
        if observed_load_completions is not None:
            for rid in observed_load_completions.get(step,[]):
                if rid not in pending_loads or pending_loads[rid] is not None:
                    raise ValueError('Completion without a unique pending load')
                pending_loads[rid]=step+1
        trace.append(dict(step=step,free_before=pre_free,free_after_schedule=after_schedule,
                          scheduled=scheduled,outputs=outputs,completed=finished,
                          preempted=preempted,resumed=resumed,forced=forced))
        if saved_prefixes:trace[-1]['loads']=loads
        if free<0 or free+sum(r['allocated'] for r in states.values())!=capacity:
            raise ValueError('KV conservation failed')
    scope='Structural token/block model only; no wall-clock, route, quality or serving gain prediction.'
    if observed_load_completions is not None:
        scope+=' Observed completion events are external replay inputs; this is not a predictive counterfactual.'
    result=dict(status='COMPLETE' if not states else 'STEP_LIMIT',trace=trace,completed=completed,
                last_step=trace[-1]['step'],scope=scope)
    if repeated_staging:
        result.update(staging_events=staging_events,stored_tokens=stored_tokens,peak_host_tokens=peak_host_tokens)
        result['scope']+=' Repeated staged actions assume successful full-prefix host allocation and store fence, no host eviction, zero store-induced wall cost, and fixed load delay in steps. Not an end-to-end performance prediction or an Oracle bound.'
    return result
