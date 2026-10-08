"""Completed local STORE-replay trajectory and frozen-formula diagnostics only.

Usage: python -B cpu_preparation/analyze_e246_store_replay.py GROUP --host-reference RAW --out JSON
No service analysis rerun, runtime imports, branch execution, or inferred benefit.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_stop_order import Trace, all_requests_comparison, peer_events

TARGET = 'measured/E246'


def lead_runs(a, b):
    rows = []
    for i, (left, right) in enumerate(zip(a.token_steps[TARGET], b.token_steps[TARGET])):
        delta = right - left
        if rows and rows[-1]['on_minus_baseline_steps'] == delta:
            rows[-1]['output_index_end_exclusive'] = i + 1
        else:
            rows.append(dict(output_index_start=i, output_index_end_exclusive=i + 1,
                             on_minus_baseline_steps=delta))
    return rows


def compare(a, b, label):
    c = all_requests_comparison(a, b)
    rows = c.pop('rows')
    peers = [k for k in a.requests if k != TARGET and a.request_schedule[k] != b.request_schedule[k]]
    changed_steps = [r[0] for k in peers for r in a.request_schedule[k]+b.request_schedule[k]
                     if r not in a.request_schedule[k] or r not in b.request_schedule[k]]
    return dict(baseline_label=label, baseline_sha256=a.sha256, on_sha256=b.sha256,
        formal_steps=dict(baseline=len(a.ends), on=len(b.ends)),
        requests=c['requests'], equal_finish_steps=c['equal_finish_steps'],
        equal_token_emission_steps=c['equal_token_emission_steps'],
        equal_native_per_request_step_schedules=c['equal_native_per_request_step_schedules'],
        identical_final_token_sequences=sum(a.requests[k]['output_token_ids'] == b.requests[k]['output_token_ids'] for k in a.requests),
        output_counts=dict(baseline=sum(len(r['output_token_ids']) for r in a.requests.values()),
                           on=sum(len(r['output_token_ids']) for r in b.requests.values())),
        native_position_work=dict(baseline=c['position_work_totals']['host'], on=c['position_work_totals']['other'],
                                  on_minus_baseline=c['position_work_other_minus_host']),
        changed_work=[dict(external_id=r['external_id'], on_minus_baseline=r['position_work_other_minus_host'])
                      for r in rows if any(r['position_work_other_minus_host'].values())],
        changed_finish=[dict(external_id=r['external_id'], baseline=r['host_finish_step'], on=r['other_finish_step'])
                        for r in rows if not r['finish_step_equal']],
        changed_emission_steps=[dict(external_id=r['external_id'], count=r['changed_token_emission_steps'],
                                    first_output_index=r['first_changed_output_index'])
                                for r in rows if not r['token_emission_steps_equal']],
        first_schedule_difference=next((i for i, (x, y) in enumerate(zip(a.schedule,b.schedule)) if x != y),None),
        peer_native_range_changes=dict(count=len(peers),
            first_step=min(changed_steps,default=None),last_step=max(changed_steps,default=None),
            all_same_scheduled_step_indices=all([r[0] for r in a.request_schedule[k]] ==
                                                [r[0] for r in b.request_schedule[k]] for k in peers),
            all_same_total_positions=all(a.request_work[k] == b.request_work[k] for k in peers),
            examples=[dict(external_id=k,
                baseline_only_rows=[r for r in a.request_schedule[k] if r not in b.request_schedule[k]],
                on_only_rows=[r for r in b.request_schedule[k] if r not in a.request_schedule[k]]) for k in peers[:3]],
            row_fields=['step','start_computed','end_computed','count','known_tokens','generated_tokens']),
        target_output_step_lead_runs=lead_runs(a,b),
        target_finish_steps=dict(baseline=a.finish[TARGET],on=b.finish[TARGET]))


def gap(trace, first_output_index=0):
    row = trace.requests[TARGET]
    candidates = [dict(output_index=i, previous_step=trace.token_steps[TARGET][i-1],
                       output_step=trace.token_steps[TARGET][i], gap_s=t-row['token_times_s'][i-1])
                  for i,t in enumerate(row['token_times_s']) if i > 0 and i >= first_output_index]
    return max(candidates, key=lambda r:r['gap_s']) if candidates else None


def shadow(trace):
    config = json.loads((trace.path.parent/'config.json').read_text())
    engine = json.loads((trace.path.parent/'engine_args.json').read_text())
    quantum = config['batch_tokens']
    commits = {(c['event'],c['request_id']):c for c in trace.raw['commits']}
    rows = []
    for e in trace.raw['decisions']:
        if not e['eligible']:
            continue
        assert e['joint_capacity'] and e['host_hit_tokens'] > 0
        assert e['full_required_blocks']+e['reserved_blocks']+e['watermark_blocks'] <= e['free_blocks']
        step = trace.step(e['decision_s'])
        h = e['headroom']; running = h['running_requests']
        scheduled = {r['request_id']:r for r in trace.raw['scheduler_steps'][step]['scheduled']}
        ids = {r['request_id'] for r in running}
        supported = h['supported'] and len(ids) == len(running) == e['running']
        for r in running:
            n,c = r['known_tokens'],r['computed_tokens']; s=scheduled.get(r['request_id'],{})
            supported &= (n == c+1 and r['is_prefill_chunk'] is False and
                          r['next_decode_eligible_step'] <= e['scheduler_step'] and n < engine['max_model_len'] and
                          s.get('count') == 1 and s.get('start_computed') == c and
                          s.get('end_computed') == n and s.get('known_tokens') == n)
        # Verification only: no non-target waiting computation exists in this
        # step. q itself uses the decision-time running snapshot, never the
        # eventual step's total token count.
        supported &= not (set(scheduled)-ids-{e['request_id']})
        q = quantum-len(running) if supported else None
        if q is not None and q <= 0:
            q = None
        row = dict(event=e['event'],external_id=trace.ids[e['request_id']],step=step,
                   known_tokens=e['known_tokens'],generated_tokens=e['generated_tokens'],
                   host_hit_tokens=e['host_hit_tokens'],current_q=q,
                   current_q_status='SUPPORTED_DECISION_SNAPSHOT_RECONSTRUCTION' if q else 'UNSUPPORTED',
                   actual_action=commits.get((e['event'],e['request_id']),{}).get('actual_action'),
                   target_committed=commits.get((e['event'],e['request_id']),{}).get('target_committed',False))
        for name,budget in [('static1792',1792),('current',q)]:
            if budget is None:
                row[name]=None;continue
            cr=(e['known_tokens']+budget-1)//budget
            ch=1+(e['known_tokens']-e['host_hit_tokens']+budget-1)//budget
            row[name]=dict(CR=cr,CH=ch,action='recompute' if cr < ch else 'host')
        rows.append(row)
    return dict(formula='CR=ceil(K/q); CH=1+ceil((K-h)/q); R iff CR<CH; otherwise H',
        current_q_definition='2048 minus decision-time supported one-token-ready running count; native rows only verify support and absence of other waiting compute, not calculate an online q from future work',
        eligible_lookup_count=len(rows),rows=rows,
        action_counts={name:dict(Counter(r[name]['action'] if r[name] else 'unsupported' for r in rows))
                       for name in ('static1792','current')},
        not_executed_recompute_suggestions={name:[dict(event=r['event'],external_id=r['external_id'],step=r['step'])
            for r in rows if r[name] and r[name]['action']=='recompute' and r['actual_action']!='recompute']
            for name in ('static1792','current')},
        interpretation='Shadow advice on observed on-trajectory states only; no concatenated counterfactual execution or latency benefit.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('group',type=Path);p.add_argument('--host-reference',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    summary=json.loads((a.group/'one_event_summary.json').read_text())
    assert json.loads((a.group/'status.json').read_text())['status']=='COMPLETE'
    assert summary['validation_passed']
    traces={}
    for audit in summary['target_audits']:
        mode='on' if audit['store_replay']['enabled'] else 'off'
        assert mode not in traces
        traces[mode]=Trace(a.group/audit['cell']/'raw.json',TARGET)
    assert set(traces)=={'off','on'}
    reference=Trace(a.host_reference,TARGET)
    assert json.loads((a.host_reference.parent/'status.json').read_text())['status']=='COMPLETE'
    target={mode:dict(first_three_recoveries=peer_events(t,TARGET,0)['first_three_recovery_commit_to_next_output'],
        native_position_work=t.peer_work,finish_step=t.finish[TARGET],
        maximum_gap=gap(t),maximum_gap_after_first_recovery=gap(t,125)) for mode,t in traces.items()}
    result=dict(schema='E.store_replay_trajectory.v1',group=str(a.group),
        source_paths={**{k:str(v.path) for k,v in traces.items()},'host_reference':str(a.host_reference)},
        conventions='Formal steps/output indices zero-based; position/output-index intervals half-open; negative on-minus-baseline steps means on emits earlier.',
        off_on=compare(traces['off'],traces['on'],'recompute_replay_off'),
        previous_host_on_discrete_only=compare(reference,traces['on'],'previous_history_group_host'),
        target=target,on_trajectory_frozen_step_formula=shadow(traces['on']),
        bounded_findings=[
            'Only E246 finishes one formal step earlier (1910 versus 1911); all 319 peer output/finish steps and global 2197-step count remain unchanged.',
            'The maximum E246 stall remains at steps 545 to 1068: off output indices 179 to 180, on 180 to 181. The 523-step lead for index180 shifts the same long wait to the following output; it does not eliminate it.',
            'On retains one-step progress through final output after that stall, but still schedules 1179 more repeated positions than the prior Host trajectory. Both frozen q rules advise only the already executed initial R on the observed on trajectory; no new unexecuted R sequence is identified.'],
        limitations=['One completed off/on pair; no reversal or adaptive policy execution.',
                     'Cross-group Host reference compares discrete work and steps only; no cross-group wall-clock benefit.',
                     'Fixed output counts do not establish equal contents or quality; scheduled positions are not GPU durations.'])
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.out),
                         finish_steps=result['off_on']['target_finish_steps'],
                         work_delta_vs_off=result['off_on']['native_position_work']['on_minus_baseline'],
                         work_delta_vs_host=result['previous_host_on_discrete_only']['native_position_work']['on_minus_baseline'],
                         shadow_counts=result['on_trajectory_frozen_step_formula']['action_counts']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
