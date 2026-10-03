"""Cost context of every action changed by A's published commit recheck.

This joins factual episodes; it does not delete events or predict intervention
performance from a frozen trajectory.
"""
import argparse
import json
from pathlib import Path


def analyze(outputs):
    candidate = outputs/'20260915_natural_native_full_gate_r01/commit_recheck'
    rows = []
    for arm, group in [('selected','20260915_natural_saved_recovery_gate_r01'),
                       ('full','20260915_natural_native_full_gate_r01')]:
        decisions = json.loads((candidate/f'{arm}.json').read_text())
        src = outputs/group/'execution_weste_26862/analysis.json'
        episode = json.loads(src.read_text())
        cell_name = 'diagnostic-current' if arm=='selected' else 'diagnostic-native-full'
        selective = json.loads((src.parent/'readback/results'/cell_name/'selective-store.json').read_text())
        assert episode['status'] == 'COMPLETE' and not episode['errors']
        segs = episode['diagnostic']['segments']
        largest = max((s for s in segs if s['L_s'] is not None and s['F_s'] is not None),
                      key=lambda s:s['F_s']-s['L_s'])
        changes = [d for d in decisions['rows'] if d.get('decision')=='RESUME_WITHOUT_VICTIM']
        assert len(changes) == decisions['changed']
        for d in changes:
            commit_time = next(s['host_perf_counter_s'] for s in selective['eligibility_snapshots']
                               if s['step']==d['step'])
            prepare_step = d['step']-1
            meta = next(e for e in selective['events'] if e['event']=='metadata' and e['step']==prepare_step)
            prep_job_ids = meta['stores']
            offload = episode['diagnostic']['offload']
            done = {j['job_id']:e['time_s'] for e in offload['completed_jobs'] for j in e['jobs']}
            dispatched = {j['job_id']:j for j in offload['dispatch'] if j['accepted']}
            prep_jobs = [dict(job_id=j, dispatch_s=dispatched[j]['before_perf_s'],
                completion_report_s=done[j], submitted_before_commit=dispatched[j]['before_perf_s']<commit_time,
                completion_reported_before_commit=done[j]<commit_time) for j in prep_job_ids]
            delta = next((e for e in selective['events'] if e['event']=='store_delta' and e['step']==prepare_step), None)
            historical = [j for j in episode.get('native_full_scope',{}).get('observed_store_jobs',[])
                          if j['request']==d['victim'] and j['completed_perf_s']<commit_time]
            matches = [s for s in segs if s['internal_request_id']==d['victim'] and s['preempt_step']==d['step']]
            assert len(matches)==1
            victim = matches[0]
            target = [s for s in segs if s['internal_request_id']==d['target']
                      and s['preempt_step']<d['step']<s['executed_steps'][0]['step']]
            assert len(target)==1 and len(victim['actual_load_dispatches'])==1
            load = victim['actual_load_dispatches'][0]
            completed = {j['job_id'] for e in episode['diagnostic']['offload']['completed_jobs']
                         for j in e['jobs'] if not j['is_store']}
            assert load['accepted'] and load['job_id'] in completed
            prefix = victim['executed_steps'][0]['scheduled_start_computed']
            assert prefix%16==0 and victim['executed_steps'][0]['computed_adjustment']==0
            rows.append(dict(arm=arm, source=str(src), candidate_commits_evaluated=decisions['checks'],
                commit_step=d['step'], target=d['target'], victim=d['victim'], free_blocks=d['free'],
                commit_snapshot_perf_s=commit_time, prepare_store_jobs=prep_jobs,
                prepare_new_store_blocks=delta['new_store_blocks'] if delta else
                    sum(len(j['logical_chunk_indices']) for j in episode['native_full_scope']['observed_store_jobs']
                        if j['request']==d['victim'] and j['metadata_step']==prepare_step),
                historical_full_store_completed_jobs=len(historical) if arm=='full' else None,
                historical_full_store_completed_blocks=sum(len(j['logical_chunk_indices']) for j in historical)
                    if arm=='full' else None,
                target_required_blocks=d['target_remaining_blocks'], retained_victim_blocks=d['retained_victim_blocks'],
                victim_load_job_id=load['job_id'], victim_actual_loaded_prefix_tokens=prefix,
                victim_reconstructed_load_bytes=prefix//16*episode['config']['kv_block_bytes'],
                victim_confirmed_recompute_tokens=victim['confirmed_recompute_tokens'],
                victim_factual_output_gap_s=victim['F_s']-victim['L_s'],
                victim_factual_subsequent_outputs=victim['useful_outputs'],
                victim_factual_segment_end=victim['end'],
                target_factual_output_gap_s=target[0]['F_s']-target[0]['L_s'],
                largest_factual_recovery_gap=dict(request=largest['request_id'], preempt_step=largest['preempt_step'],
                                                  gap_s=largest['F_s']-largest['L_s']),
                victim_event_is_largest=largest is victim,
                prepared_store_state_is_undone_by_candidate=False,
                counterfactual_output_gap_s=None, net_load_bytes_saved=None))
    return dict(status='FACTUAL_COST_CONTEXT_NOT_ACTION_VALUE', rows=rows,
        semantics=[
            'Every published changed action is included; the original 25 commits per episode remain the candidate denominator.',
            'Loaded prefix bytes are reconstructed on the same fixed native no-APC full-attention path and linked to accepted/completed jobs.',
            'One potential avoided load is not a net saving guarantee: retained capacity and batches may induce other future work.',
            'Prepare state already exists at commit. The candidate does not cancel it; this does not imply all associated D2H has already executed.',
            'Factual output gaps include ordinary execution/queue/recovery; they are not removable delay estimates.',
            'The longest factual gap does not involve this victim event; indirect improvements or harms remain untested.',
            'These are diagnostic contexts, not the lightweight four-cell trajectory, an Oracle upper bound, or new samples.'])


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--outputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=analyze(args.outputs)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
        f.write('\n')
    print(json.dumps(result['rows'],indent=2))
