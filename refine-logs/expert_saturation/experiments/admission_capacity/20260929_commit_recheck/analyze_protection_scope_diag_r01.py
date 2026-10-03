#!/usr/bin/env python3
"""Localize actual Q10 protection gates in one unchanged-policy archive.

Numerical headroom is necessary, not proof of native admission. Performance
comparisons and unobserved ordinary native admissions are outside this report.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from analyze_capacity_protection_pair_r01 import analyze_arm, distribution


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            sha.update(chunk)
    return sha.hexdigest()


def count(values):
    return dict(sorted((str(key),number) for key,number in Counter(values).items()))


def analyze(session):
    archives=list(session.glob('cell-*-protection_scope_q10_diagnostic/archive'))
    if len(archives)!=1:
        raise ValueError(f'Expected one Q10 scope archive, found {len(archives)}')
    archive=archives[0]
    arm,raw=analyze_arm(archive,10)
    store=json.loads((archive/'selective-store.json').read_text())
    status=json.loads((archive/'status.json').read_text())
    if 'protection_scope_steps' not in store:
        raise ValueError('Missing new protection_scope_steps telemetry')
    steps=store['protection_scope_steps']
    episodes=[ep for ep in arm['episodes'] if ep['extend'] is not None]
    requests={r['internal_request_id']:r for r in raw['requests']}
    origin=raw['measurement_origin_perf_counter_s']
    admissions=defaultdict(list)
    for event in store['events']:
        if event.get('event') in ('recovery_commit_admitted','direct_commit','fit_first_admitted'):
            stamp=event.get('host_perf_counter_s')
            if isinstance(stamp,(int,float)):
                admissions[event['target']].append(event)
    for rows in admissions.values():
        rows.sort(key=lambda event:event['host_perf_counter_s'])

    def episode_for(row):
        matches=[(index,ep) for index,ep in enumerate(episodes)
            if ep['target_internal_id']==row['protected']
            and ep['extend']['step']<=row['step']
            and (ep['release'] is None or row['step']<ep['release']['step'])]
        if len(matches)!=1:
            raise ValueError(f'Extended step {row["step"]} does not map to one episode')
        return matches[0]

    episode_rows=defaultdict(list)
    break_rows=[];necessary=[]
    for row in steps:
        index,episode=episode_for(row)
        episode_rows[index].append(row)
        head=row['waiting_break']
        if head is None:
            if row['waiting_break_count']!=0:
                raise ValueError('Waiting break count lacks its first-head record')
            continue
        if row['waiting_break_count']!=1:
            raise ValueError('Native waiting break occurred more than once in one step')
        token_budget=head['remaining_token_budget']
        reserved=head['native_inflight_reserved_blocks']
        fit=(token_budget>0 and reserved==0 and head['free_blocks']>=
            head['incremental_full_history_need_blocks']+head['protected_future_growth_blocks'])
        if fit!=head['necessary_numeric_headroom']:
            raise ValueError('Logged numerical headroom differs from recomputed condition')
        break_rows.append((row,head,index,fit))
        if fit:
            request=requests.get(head['head'])
            stamp=head['host_perf_counter_s'];relative=stamp-origin
            output_s=None
            if request is not None:
                times=request['token_times_s']
                output_index=bisect_right(times,relative)
                if output_index<len(times):output_s=times[output_index]
            after=next((event for event in admissions.get(head['head'],())
                if event['host_perf_counter_s']>=stamp),None)
            admission_s=after['host_perf_counter_s']-origin if after else None
            necessary.append(dict(step=row['step'],episode_index=index,
                protected=row['protected'],waiting_head=head['head'],head_status=head['status'],
                break_time_s=relative,remaining_token_budget=token_budget,
                free_blocks=head['free_blocks'],incremental_full_history_need_blocks=
                    head['incremental_full_history_need_blocks'],
                protected_future_growth_blocks=head['protected_future_growth_blocks'],
                margin_blocks=head['numeric_margin_blocks'],
                head_transfer_jobs_count=head['head_transfer_jobs_count'],
                registered_transfer_jobs_count=head['registered_transfer_jobs_count'],
                pending_push_work=head['pending_push_work'],
                next_head_output_s=output_s,
                break_to_next_head_output_s=(output_s-relative if output_s is not None else None),
                next_recorded_adapter_admission=(dict(event=after['event'],step=after['step'],
                    native_admission=after.get('native_admission'),time_s=admission_s,
                    break_to_admission_s=admission_s-relative,
                    before_next_output=(admission_s<=output_s if output_s is not None else None))
                    if after else None),
                head_final_status=request['status'] if request else None))

    episode_summary=[]
    for index,episode in enumerate(episodes):
        rows=episode_rows[index]
        first=episode['first_output_event'];release=episode['release']
        first_to_release=(release['relative_time_s']-first['relative_time_s']
            if first and release and first['relative_time_s'] is not None
            and release['relative_time_s'] is not None else None)
        episode_summary.append(dict(index=index,target=episode['target_internal_id'],
            extend_step=episode['extend']['step'],release_step=release['step'] if release else None,
            release_reason=release.get('reason') if release else None,
            first_output_observed_to_release_s=first_to_release,
            extended_steps=len(rows),waiting_break_steps=sum(r['waiting_break'] is not None for r in rows),
            necessary_headroom_steps=sum(r['waiting_break'] is not None and
                r['waiting_break']['necessary_numeric_headroom'] for r in rows),
            held_peer_steps=sum(r['held_peer_count']>0 for r in rows),
            peer_scheduled_tokens=sum(r.get('peer_scheduled_tokens_sum',0) for r in rows)))

    heads=[head for _,head,_,_ in break_rows]
    metrics=arm['metrics']
    complete=(raw.get('status')=='COMPLETE' and raw.get('error') is None
        and status.get('status')=='COMPLETE' and store.get('status')=='DRAINED'
        and len(raw['requests'])==metrics['completed']==status.get('requests_completed')==128
        and metrics['failed']==metrics['unfinished']==0)
    return dict(status='SOURCE_LOCALIZATION_COMPLETE' if complete else 'INCOMPLETE_OR_FAILED',
        session=str(session),archive=str(archive),source_sha256=dict(
            raw=digest(archive/'raw.json'),selective_store=digest(archive/'selective-store.json')),
        cohort=dict(expected=128,observed=len(raw['requests']),completed=metrics['completed'],
            failed=metrics['failed'],unfinished=metrics['unfinished'],
            raw_status=raw.get('status'),raw_error=raw.get('error'),
            cell_status=status.get('status'),cell_error=status.get('error'),
            store_status=store.get('status'),request_statuses=count(r['status'] for r in raw['requests'])),
        scope=dict(extended_episodes=len(episodes),extended_steps=len(steps),
            actual_waiting_break_steps=len(break_rows),
            actual_waiting_break_calls=sum(row['waiting_break_count'] for row in steps),
            distinct_waiting_heads=len({head['head'] for head in heads}),
            episodes_with_waiting_break=len({index for _,_,index,_ in break_rows}),
            waiting_head_statuses=count(head['status'] for head in heads),
            native_reservation_dispositions=count(head['native_reservation_disposition'] for head in heads),
            waiting_head_transfer_jobs_count=count(head['head_transfer_jobs_count'] for head in heads),
            pending_push_work=count(head['pending_push_work'] for head in heads),
            numeric_margin_blocks=distribution([head['numeric_margin_blocks'] for head in heads]),
            known_zero_reservation_steps=sum(head['native_inflight_reserved_blocks']==0 for head in heads),
            necessary_numeric_headroom_steps=len(necessary),
            necessary_preempted_head_steps=sum(row['head_status']=='PREEMPTED' for row in necessary),
            necessary_distinct_heads=len({row['waiting_head'] for row in necessary}),
            necessary_episodes=len({row['episode_index'] for row in necessary}),
            necessary_margin_blocks=distribution([row['margin_blocks'] for row in necessary]),
            held_peer_steps=sum(row['held_peer_count']>0 for row in steps),
            held_peer_instances_sum=sum(row['held_peer_count'] for row in steps),
            held_peer_growth_blocks_sum=sum(row['held_peer_growth_blocks_sum'] for row in steps),
            held_peer_count_per_step=distribution([row['held_peer_count'] for row in steps]),
            peer_scheduled_steps=sum(row.get('peer_scheduled_request_count',0)>0 for row in steps),
            peer_scheduled_request_instances_sum=sum(row.get('peer_scheduled_request_count',0) for row in steps),
            peer_scheduled_tokens_sum=sum(row.get('peer_scheduled_tokens_sum',0) for row in steps),
            first_output_observed_to_release_s=distribution(
                [ep['first_output_observed_to_release_s'] for ep in episode_summary])),
        extended_episodes=episode_summary,necessary_fit_breaks=necessary,
        interpretation=[
            'A reached waiting break proves this native loop selected a non-target head with token budget remaining; it does not prove that head would be admitted without the break.',
            'Necessary numerical headroom requires known native reservation R=0 and free blocks at the break at least incremental head full-history need plus protected future growth. Status, transfer state, and other native gates remain separate.',
            'Next head output is a host-return observation. Adapter admission receipts cover only recorded adapter admissions; ordinary native admission and transfer completion may be unobserved.',
            'This one same-policy diagnostic is source localization, not a performance comparison or action-level counterfactual.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.session)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(status=result['status'],cohort=result['cohort'],scope=result['scope'])))


if __name__=='__main__':main()
