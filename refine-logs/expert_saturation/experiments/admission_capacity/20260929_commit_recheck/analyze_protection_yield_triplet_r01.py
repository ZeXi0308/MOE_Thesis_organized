#!/usr/bin/env python3
"""Complete-cohort Q1 / Q10-yield / plain-Q10 triplet and actual yield chains."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from analyze_capacity_protection_pair_r01 import analyze_arm, distribution
from evaluate_goodput import pair


ARMS=(('q1','protection_q1',1),('yield','protection_q10_yield',10),
      ('plain_q10','protection_q10_plain',10))


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(test,message):
    if not test:raise ValueError(message)


def read_cell(session,index,name,quantity):
    archive=session/f'cell-{index:02d}-{name}'/'archive'
    arm,raw=analyze_arm(archive,quantity)
    store=json.loads((archive/'selective-store.json').read_text())
    status=json.loads((archive/'status.json').read_text())
    config=json.loads((archive/'config.json').read_text())
    cohort=arm['metrics']
    complete=(arm['capture_status']=='COMPLETE' and arm['capture_error'] is None
        and arm['store_status']=='DRAINED' and status.get('status')=='COMPLETE'
        and status.get('capture_status')=='COMPLETE' and status.get('error') is None
        and len(raw['requests'])==cohort['completed']==status.get('requests_completed')==128
        and cohort['failed']==cohort['unfinished']==0)
    return dict(archive=str(archive),arm=arm,raw=raw,store=store,status=status,
        config=config,complete=complete,
        source_sha256=dict(raw=sha256(archive/'raw.json'),
            selective_store=sha256(archive/'selective-store.json')))


def compare(reference,candidate):
    a=reference['arm']['metrics'];b=candidate['arm']['metrics']
    paired=pair(a,b)
    old={r['request_id']:r for r in reference['raw']['requests']}
    new={r['request_id']:r for r in candidate['raw']['requests']}
    require(len(old)==len(new)==128 and set(old)==set(new),'Triplet request identities differ')
    return dict(full_cohort_pair=paired,
        output_sequence_difference_requests=sorted(rid for rid in old
            if old[rid]['output_token_ids']!=new[rid]['output_token_ids']),
        stop_reason_difference_requests=sorted(rid for rid in old
            if old[rid].get('stop_reason')!=new[rid].get('stop_reason')),
        maximum_request_gap_s=dict(reference=a['max_gap_request_max_s'],
            candidate=b['max_gap_request_max_s']))


def request_result(request):
    if request is None:return None
    return dict(request_id=request['request_id'],status=request['status'],
        completion_s=request.get('completion_s'),stop_reason=request.get('stop_reason'),
        output_tokens=len(request['output_token_ids']))


def next_output(request,after_s):
    if request is None or after_s is None:return None
    times=request['token_times_s'];position=bisect_right(times,after_s)
    return times[position] if position<len(times) else None


def actual_preemptions_after(raw,internal_id,after_s):
    records=raw.get('preemption_events')
    if not isinstance(records,list):return None
    return [dict(step=event['engine_call_index'],time_s=event['method_entered_s'],
        output_count=event.get('last_returned_output_count'))
        for event in records if event.get('internal_request_id')==internal_id
        and event.get('original_preemption_called') is True
        and event.get('original_preemption_returned') is True
        and event.get('method_entered_s',-1)>after_s]


def yield_chains(cell):
    raw=cell['raw'];store=cell['store']
    origin=raw['measurement_origin_perf_counter_s']
    requests={r['internal_request_id']:r for r in raw['requests']}
    events=store.get('events',[])
    choices=[e for e in events if e.get('event')=='protection_yield_to_ready_head']
    admissions=[e for e in events if e.get('event')=='yield_head_admission']
    releases=[e for e in events if e.get('event')=='protection_release'
        and e.get('reason')=='YIELD_TO_READY_HEAD']
    def action_key(event):return event['step'],event['head'],event['protected']
    admit_by_key={action_key(e):e for e in admissions}
    release_by_key={(e['step'],e['request']):e for e in releases}
    require(len(admit_by_key)==len(admissions),'Duplicate yield admission receipt')
    require(len(release_by_key)==len(releases),'Duplicate yield release receipt')
    rows=[]
    for choice in choices:
        key=action_key(choice);step,head_id,original_id=key
        admission=admit_by_key.get(key)
        release=release_by_key.get((step,original_id))
        head=requests.get(head_id);original=requests.get(original_id)
        require(head is not None and original is not None,'Yield actor absent from 128 requests')
        start=choice['head_output_start']
        require(type(start) is int and start>=0,'Invalid yield head output count')
        yield_s=choice['host_perf_counter_s']-origin
        head_new_output_s=(head['token_times_s'][start]
            if start<len(head['token_times_s']) else None)
        if head_new_output_s is not None and head_new_output_s<=yield_s:
            head_new_output_s=None
        original_next_output_s=next_output(original,yield_s)
        admission_s=(admission['host_perf_counter_s']-origin if admission else None)
        kind=admission.get('native_admission') if admission else None
        admitted=(kind=='SCHEDULED_TOKENS' and admission.get('scheduled_tokens',0)>0
            or kind=='ASYNC_LOAD_ADMITTED' and bool(admission.get('load_job_ids')))
        rows.append(dict(step=step,waiting_head=head_id,released_original_target=original_id,
            yield_time_s=yield_s,head_output_count_at_yield=start,
            original_output_count_at_yield=choice['protected_output_count'],
            head_full_history_tokens=choice['head_full_history_tokens'],
            head_need_blocks=choice['head_need_blocks'],
            free_blocks=choice['free_blocks'],
            protected_future_growth_blocks=choice['future_growth_blocks'],
            native_inflight_reserved_blocks=choice['native_inflight_reserved_blocks'],
            release_recorded=release is not None,
            release_time_s=release['host_perf_counter_s']-origin if release else None,
            native_admission=dict(kind=kind,recorded=admission is not None,
                admitted=admitted,scheduled_tokens=admission.get('scheduled_tokens') if admission else None,
                load_job_ids=admission.get('load_job_ids') if admission else None,
                head_status=admission.get('head_status') if admission else None,
                held_blocks=admission.get('held_blocks') if admission else None,
                original_target_scheduled_tokens=admission.get('original_target_scheduled_tokens') if admission else None,
                time_s=admission_s),
            head_new_output_s=head_new_output_s,
            admission_to_head_new_output_s=(head_new_output_s-admission_s
                if admitted and admission_s is not None and head_new_output_s is not None
                and head_new_output_s>admission_s else None),
            yield_to_head_new_output_s=(head_new_output_s-yield_s
                if head_new_output_s is not None else None),
            head_final=request_result(head),
            head_actual_preemptions_after_yield=actual_preemptions_after(raw,head_id,yield_s),
            original_next_output_s=original_next_output_s,
            original_final=request_result(original),
            original_actual_preemptions_after_yield=actual_preemptions_after(raw,original_id,yield_s)))
    require(len(rows)==len({(r['step'],r['waiting_head'],r['released_original_target']) for r in rows}),
        'Duplicate yield action key')
    admission_kinds=Counter(r['native_admission']['kind'] for r in rows)
    actual_admitted=[r for r in rows if r['native_admission']['admitted']]
    chains=[r for r in actual_admitted if r['head_new_output_s'] is not None
        and r['native_admission']['time_s'] is not None
        and r['head_new_output_s']>r['native_admission']['time_s']
        and r['head_final']['status']=='completed']
    return dict(count=len(rows),release_count=len(releases),
        postnative_admission_receipt_count=len(admissions),
        counter_yield_releases=store.get('yield_releases'),
        gate_counts=store.get('yield_gate_counts'),
        admission_kinds=dict(admission_kinds),
        actual_native_admissions=len(actual_admitted),
        admission_to_new_output_and_completion_chains=len(chains),
        released_original_targets_completed=sum(r['original_final']['status']=='completed' for r in rows),
        released_original_targets_preempted_again=sum(bool(r['original_actual_preemptions_after_yield'])
            for r in rows if r['original_actual_preemptions_after_yield'] is not None),
        original_actual_later_preemptions_sum=sum(len(r['original_actual_preemptions_after_yield'])
            for r in rows if r['original_actual_preemptions_after_yield'] is not None),
        admission_to_head_new_output_s=distribution(
            [r['admission_to_head_new_output_s'] for r in rows]),
        yield_to_head_new_output_s=distribution([r['yield_to_head_new_output_s'] for r in rows]),
        consistent_receipt_counts=(len(rows)==len(releases)==len(admissions)==store.get('yield_releases')),
        actions=rows)


def analyze(session):
    cells={}
    for index,(label,name,quantity) in enumerate(ARMS):
        cells[label]=read_cell(session,index,name,quantity)
    comparisons=dict(yield_vs_q1=compare(cells['q1'],cells['yield']),
        yield_vs_plain_q10=compare(cells['plain_q10'],cells['yield']))
    for label,cell in cells.items():
        expected=label=='yield'
        require(cell['store'].get('yield_to_ready_head') is expected
            and cell['config'].get('yield_to_ready_head') is expected,
            f'{label}: executed yield flag differs')
    chain=yield_chains(cells['yield'])
    require(not any(e.get('event') in ('protection_yield_to_ready_head','yield_head_admission')
        for label in ('q1','plain_q10') for e in cells[label]['store'].get('events',[])),
        'Control arm contains yield action')
    complete=all(cell['complete'] for cell in cells.values())
    q1=cells['q1']['arm']['metrics'];candidate=cells['yield']['arm']['metrics']
    performance_criteria=dict(
        full_cohort_complete=complete,
        actual_native_yield_admission=chain['actual_native_admissions']>0,
        actual_admission_to_output_and_completion=chain['admission_to_new_output_and_completion_chains']>0,
        output_rate_at_least_97pct_q1=(candidate['actual_output_tokens_s']>=
            .97*q1['actual_output_tokens_s']),
        mean_flow_at_most_105pct_q1=(candidate['mean_flow_with_incomplete_penalty_s']<=
            1.05*q1['mean_flow_with_incomplete_penalty_s']),
        lower_max_request_gap_than_q1=(candidate['max_gap_request_max_s'] is not None
            and q1['max_gap_request_max_s'] is not None
            and candidate['max_gap_request_max_s']<q1['max_gap_request_max_s']))
    return dict(status='COMPLETE_TRIPLET' if complete else 'INCOMPLETE_TRIPLET',
        session=str(session),arms={label:dict(archive=cell['archive'],
            source_sha256=cell['source_sha256'],capture_status=cell['arm']['capture_status'],
            capture_error=cell['arm']['capture_error'],store_status=cell['arm']['store_status'],
            cell_status=cell['status'].get('status'),
            metrics=cell['arm']['metrics'],
            protection_summary=cell['arm']['protection_summary'],
            preemption_summary=cell['arm']['preemption_summary'],
            actual_preemption_count=cell['raw'].get('actual_preemption_count'),
            forced_rotations=cell['status'].get('forced_rotations'),
            capacity_victim_commits=cell['status'].get('capacity_victim_commits'),
            stop_reason_counts=cell['status'].get('finish_reason_counts'))
            for label,cell in cells.items()},
        comparisons=comparisons,yield_actions=chain,
        predeclared_exploration_criteria=performance_criteria,
        all_exploration_criteria_met=all(performance_criteria.values())
            and chain['consistent_receipt_counts'],
        interpretation=[
            'All comparisons use complete 128-request cohorts. Protection windows and yielded heads are post-action descriptions, not independent performance denominators.',
            'SCHEDULED_TOKENS and ASYNC_LOAD_ADMITTED are native admission receipts; async admission alone does not prove transfer completion. Head new output is the next host-return token after the recorded head output count.',
            'NO_NATIVE_ADMISSION, missing receipt, or missing next output does not form an admission-to-output chain. A later ordinary admission outside the recorded yield step may still occur.',
            'All later original-target preemptions are observed native preemption calls after yield, not a causal cost estimate. Independent arm trajectories are not action-level counterfactuals.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.session)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(status=result['status'],yield_actions=result['yield_actions']['count'])))


if __name__=='__main__':main()
