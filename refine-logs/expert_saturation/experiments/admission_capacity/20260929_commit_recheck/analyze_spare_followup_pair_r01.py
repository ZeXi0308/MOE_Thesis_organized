"""Analyze complete native Q1 off/on cohorts and actual bounded follow-up chains."""
import argparse
from collections import Counter
import json
from pathlib import Path
from analyze_protection_yield_triplet_r01 import (read_cell, compare, request_result,
    next_output, actual_preemptions_after, require)
from analyze_capacity_protection_pair_r01 import distribution


def followups(cell):
    raw,store=cell['raw'],cell['store']
    origin=raw['measurement_origin_perf_counter_s']
    reqs={r['internal_request_id']:r for r in raw['requests']}
    events=store['events']
    choices=[e for e in events if e.get('event')=='spare_followup_choice']
    admissions=[e for e in events if e.get('event')=='spare_followup_admission']
    receipts={(e['step'],e['target']):e for e in admissions}
    require(len(receipts)==len(admissions),'Duplicate follow-up admission receipt')
    starts={(e['step'],e['request']):e for e in events if e.get('event')=='protection_start'}
    primary_keys=[(e['primary_start_step'],e['primary_target']) for e in choices]
    require(len(primary_keys)==len(set(primary_keys)),'More than one follow-up for a primary recovery')
    rows=[]
    for choice in choices:
        target=choice['target'];primary=choice['primary_target']
        require(target in reqs and primary in reqs,'Actor missing from full cohort')
        admission=receipts.get((choice['step'],target))
        when=choice['host_perf_counter_s']-origin
        count=choice['output_tokens_at_choice']
        times=reqs[target]['token_times_s']
        output=times[count] if type(count) is int and 0<=count<len(times) else None
        kind=admission.get('native_admission') if admission else None
        admitted=bool(admission and (kind=='SCHEDULED_TOKENS' and admission.get('scheduled_tokens',0)>0
            or kind=='ASYNC_LOAD_ADMITTED' and admission.get('load_job_ids')))
        at=admission['host_perf_counter_s']-origin if admission else None
        release=next((e for e in events if e.get('event')=='protection_release'
            and e['request']==primary and e['step']==choice['step']
            and e.get('reason')=='OUTPUT_GOAL_REACHED'),None)
        primary_released_after_output=bool(release and release.get('new_output_tokens',0)>=1)
        start=starts.get((choice['primary_start_step'],primary))
        primary_times=reqs[primary]['token_times_s']
        primary_count=start.get('output_count') if start else None
        primary_output_before_choice=bool(type(primary_count) is int and 0<=primary_count<len(primary_times)
            and primary_times[primary_count]<when)
        regular_origin=bool(start and start.get('protection_origin')=='REGULAR_COMMIT')
        contemporaneous_preempts=[p for p in raw.get('preemption_events',[])
            if p['engine_call_index']==choice['step'] and p.get('original_preemption_called') is True
            and p.get('original_preemption_returned') is True]
        actors={}
        for role in ('primary_target','primary_victim','displaced_oldest','queue_head'):
            rid=choice.get(role);request=reqs.get(rid)
            actors[role]=dict(internal_id=rid,final=request_result(request),
                next_output_s=next_output(request,when),
                actual_later_preemptions=actual_preemptions_after(raw,rid,when) if rid else None)
        rows.append(dict(choice=choice,native_receipt=admission,
            native_admitted=admitted,choice_s=when,admission_s=at,
            first_new_output_s=output,
            admission_to_output_s=output-at if admitted and at is not None and output is not None and output>at else None,
            primary_release=release,primary_released_after_output=primary_released_after_output,
            primary_raw_output_before_choice=primary_output_before_choice,
            primary_origin_regular_commit=regular_origin,
            actual_preemptions_same_step=contemporaneous_preempts,
            final=request_result(reqs[target]),actors=actors))
    chains=[r for r in rows if r['native_admitted'] and r['admission_to_output_s'] is not None
        and r['final']['status']=='completed']
    return dict(choices=len(choices),receipts=len(admissions),native_admissions=sum(r['native_admitted'] for r in rows),
        actual_output_completion_chains=len(chains),
        all_after_primary_output=all(r['primary_released_after_output'] for r in rows),
        all_after_actual_primary_output=all(r['primary_raw_output_before_choice'] for r in rows),
        all_origins_regular=all(r['primary_origin_regular_commit'] for r in rows),
        all_without_same_step_preemption=all(not r['actual_preemptions_same_step'] for r in rows),
        admission_kinds=dict(Counter(r['native_receipt'].get('native_admission') if r['native_receipt'] else 'MISSING' for r in rows)),
        admission_to_output_s=distribution([r['admission_to_output_s'] for r in rows]),actions=rows,
        scope='Later actor outcomes overlap across actions and are not causal or additive costs. Async receipt is admission, not transfer completion.')


def analyze(session,order=('off','on')):
    require(tuple(order) in (('off','on'),('on','off')),'Unknown pair order')
    cells={label:read_cell(session,i,'spare_followup_'+label,1) for i,label in enumerate(order)}
    for label,c in cells.items():
        enabled=label=='on'
        require(c['config'].get('spare_followup') is enabled and c['store'].get('spare_followup') is enabled,
            'Executed follow-up flag differs from cell')
    require(not any(e.get('event') in ('spare_followup_choice','spare_followup_admission')
        for e in cells['off']['store']['events']),'Control contains a follow-up action')
    chain=followups(cells['on']);comparison=compare(cells['off'],cells['on'])
    old=cells['off']['arm']['metrics'];new=cells['on']['arm']['metrics']
    criteria=dict(full_cohort_complete=all(c['complete'] for c in cells.values()),
        actual_admission_output_completion=chain['actual_output_completion_chains']>0,
        original_target_output_precedes_followup=chain['all_after_primary_output'],
        raw_primary_output_precedes_followup=chain['all_after_actual_primary_output'],
        no_recursive_followups=chain['all_origins_regular'],
        no_extra_preemption_at_followup=chain['all_without_same_step_preemption'],
        rate_at_least_97pct_q1=new['actual_output_tokens_s']>=.97*old['actual_output_tokens_s'],
        mean_flow_at_most_105pct_q1=new['mean_flow_with_incomplete_penalty_s']<=1.05*old['mean_flow_with_incomplete_penalty_s'],
        lower_max_request_gap=new['max_gap_request_max_s']<old['max_gap_request_max_s'])
    return dict(status='COMPLETE_PAIR' if criteria['full_cohort_complete'] else 'INCOMPLETE_PAIR',
        session=str(session),arms={label:dict(source_sha256=c['source_sha256'],metrics=c['arm']['metrics'],
            preemption_summary=c['arm']['preemption_summary'],actual_preemption_count=c['raw'].get('actual_preemption_count'),
            forced_rotations=c['status'].get('forced_rotations'),
            stop_reason_counts=dict(Counter(r.get('stop_reason') for r in c['raw']['requests']))) for label,c in cells.items()},
        comparison=comparison,followups=chain,predeclared_exploration_criteria=criteria,
        all_exploration_criteria_met=all(criteria.values()),
        limitations=['Seen-input ordered exploration, not confirmation or equal-work acceleration.',
            'Original targets, victims and bypassed oldest waiters remain in the complete 128-request denominator.',
            'No statistical or action-level causal effect follows from one independent-trajectory pair.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--order',choices=('off-on','on-off'),default='off-on')
    args=parser.parse_args();result=analyze(args.session,tuple(args.order.split('-')))
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],criteria=result['predeclared_exploration_criteria'])))
