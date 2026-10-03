#!/usr/bin/env python3
"""Read-only actual packing divergence and interrupted-recovery localization."""
from bisect import bisect_right
import json
from pathlib import Path
import sys

BUNDLE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BUNDLE.parents[2]/'experiments/admission_capacity'))
from analyze_prefix_cache_baseline import read, sha, digest, require
from analyze_ltr_packing_probe import residencies
from analyze_ltr_gap_localization import gap


def action(step):
    return dict(tokens={x['request_id']:x['scheduled_tokens'] for x in step['scheduled']},victims=sorted(step['preempted_request_ids']))


def boundary(raw,decisions,k):
    s=raw['scheduler_steps'][k]; q={r['request_id']:r for r in raw['requests']}; aliases=raw['internal_to_source']
    counts={rid:bisect_right(r['token_times_s'],s['start_s']) for rid,r in q.items()}
    prior=next((c['returned_s'] for c in raw['engine_steps'] if c['scheduler_step_end']==k),0.)
    m=raw['memory_trace'][k]['before']; d=decisions[k]
    order=sorted(m['requests'],key=lambda rid:(d['priorities'][rid],q[aliases[rid]]['arrival_s'],rid))
    candidates=[]
    for iid in order:
        r=m['requests'][iid]; history=r['prompt_tokens']+r['output_tokens']; owned=r['block_counts'][0]
        candidates.append(dict(request_id=aliases[iid],priority=d['priorities'][iid],resident=iid in m['running_ids'],
            computed_tokens=r['computed_tokens'],output_tokens=r['output_tokens'],owned_blocks=owned,
            current_history_blocks=(history+15)//16,additional_history_blocks=max(0,(history+15)//16-owned),
            pending_tokens=history-r['computed_tokens'],scheduled_tokens=d['actual_scheduled'].get(iid,0)))
    visible=dict(requests={aliases[i]:v for i,v in m['requests'].items()},pool=m['pool'],
                 running=[aliases[i] for i in m['running_ids']],counters={aliases[i]:v for i,v in d['counter_state_before'].items()})
    return dict(common_prefix_steps=k,elapsed_through_last_common_call_s=prior,next_schedule_start_s=s['start_s'],
        returned_output_tokens=sum(counts.values()),output_counts_by_request=counts,
        returned_prefix_sha256=digest({rid:q[rid]['output_token_ids'][:counts[rid]] for rid in sorted(q)}),
        visible_prestate_sha256=digest(visible),free_blocks=m['pool']['free_blocks'],ranked_candidates=candidates,
        actual_action=action(s),actual_request_order=[x['request_id'] for x in s['scheduled']])


def inspect_pair(left,right):
    raw=[]; decisions=[]
    for row in (left,right):
        path=Path(row['raw_path']); d=path.parent/'component-decisions.json'
        require(sha(path)==row['raw_sha256'] and sha(d)==row['decisions_sha256'],'qualified raw changed')
        raw.append(read(path)); decisions.append(read(d))
    a,b=raw; steps=min(len(a['scheduler_steps']),len(b['scheduler_steps']))
    k=next(i for i in range(steps) if action(a['scheduler_steps'][i])!=action(b['scheduler_steps'][i]))
    ba,bb=boundary(a,decisions[0],k),boundary(b,decisions[1],k)
    common_order=all([x['request_id'] for x in a['scheduler_steps'][i]['scheduled']]==[x['request_id'] for x in b['scheduler_steps'][i]['scheduled']] for i in range(k))
    target=max(right['per_request'],key=lambda r:r['max_itl_s']); q=next(q for q in b['requests'] if q['request_id']==target['request_id'])
    index=max(range(len(q['token_times_s'])-1),key=lambda j:q['token_times_s'][j+1]-q['token_times_s'][j])
    maximum=gap(b,decisions[1],q,index); require(maximum['gap_s']==right['max_itl_s'],'qualified maximum differs')
    resident=residencies(b); require(resident==right['recovery_residencies'],'independent residency recount differs')
    short=[r for r in resident['rows'] if r['resumed'] and r['end_reason']=='actual_repreemption' and r['new_outputs']<=2]
    return dict(fit_label=left['label'],prefix_label=right['label'],first_action_divergence_step=k,
        common_actual_request_order_equal=common_order,fit_boundary=ba,prefix_boundary=bb,
        available_output_prefix_equal=ba['returned_prefix_sha256']==bb['returned_prefix_sha256'],
        recorded_prestate_equal=ba['visible_prestate_sha256']==bb['visible_prestate_sha256'],
        rank_prefix_maximum_gap=maximum,rank_prefix_short_repreempted_residencies=short,
        fit_short_counts=left['recovery_residencies']['short_new_output_counts'],prefix_short_counts=resident['short_new_output_counts'],
        full_metrics={r['label']:{key:r[key] for key in ('wall_s','mean_completion_s','max_itl_s','throughput_rps')} for r in (left,right)},
        source_raw_sha256={r['label']:r['raw_sha256'] for r in (left,right)})


def main():
    source=BUNDLE/'analysis/analysis.json'; out=Path(__file__).parent
    require(not (out/'analysis.json').exists(),'refuse localization overwrite')
    measured=read(source); require(measured['status']=='MEASUREMENT_ONLY' and all(c['eligible'] for c in measured['cells']),'requires four qualified cells')
    cells={r['label']:r for r in measured['cells']}
    pairs=[inspect_pair(cells[f'block{i}-d6-packing-fit_scan'],cells[f'block{i}-d6-packing-rank_prefix']) for i in (0,1)]
    result=dict(status='MEASUREMENT_ONLY_ACTUAL_TRAJECTORY_LOCALIZATION',source_analysis_sha256=sha(source),pairs=pairs,
        scope='All measurements are retained policy-specific executions. Common actual token allocations and returned prefixes do not establish full-engine or KV-byte checkpoint equality. '
        'Recorded prestate includes counters, block counts and pool state only. No future trace is reused as a counterfactual and timing differences are not wholly attributed to packing.')
    lines=['# Packing first-divergence and recovery localization','',result['status'],'',result['scope'],'',
        '| pair | first action difference | common returned outputs fit/prefix | elapsed fit/prefix s | prefix max gap s | 0/1/2 outputs fit → prefix |',
        '|---|---:|---|---|---:|---|']
    for p in pairs:
        a,b=p['fit_boundary'],p['prefix_boundary']; g=p['rank_prefix_maximum_gap']
        lines.append(f"| {p['fit_label']} / {p['prefix_label']} | {p['first_action_divergence_step']} | {a['returned_output_tokens']} / {b['returned_output_tokens']} | {a['elapsed_through_last_common_call_s']:.6f} / {b['elapsed_through_last_common_call_s']:.6f} | {g['gap_s']:.6f} | {p['fit_short_counts']} → {p['prefix_short_counts']} |")
        changed={rid:[n,b['actual_action']['tokens'].get(rid,0)] for rid,n in a['actual_action']['tokens'].items() if b['actual_action']['tokens'].get(rid)!=n}
        lines.extend(['',f"First differing actual token allocations fit/prefix: {changed}; victims: {a['actual_action']['victims']} / {b['actual_action']['victims']}. Both begin with {a['free_blocks']} free blocks. "])
        lines.extend(['',f"{p['prefix_label']}: longest gap request {g['request_id']}, output {g['previous_output_count']}→{g['previous_output_count']+1}, "
            f"steps {g['first_step']}–{g['last_step']}, {g['scheduling_calls']} calls, max consecutive nonselected {g['maximum_contiguous_nonselected']}. "
            f"Partial-recompute selections: {[(s['step'],s['computed_before'],s['computed_after'],s['new_outputs']) for s in g['target_selections']]}. "
            f"Recorded common prestate equality={p['recorded_prestate_equal']}, available output-prefix equality={p['available_output_prefix_equal']}. "])
    lines.extend(['','At the first difference, the shared30 decode selections reserve3 growth blocks, leaving145 of the initial148 free. Waiting request3571 needs212 blocks; even releasing later resident3640 owned63 gives208, four blocks short. Resident3640 itself needs142 more blocks. Fit-scan continues its recovery with994 tokens; rank-prefix stops after the shared30 decode tokens. These are observed matching prestate counts and actual actions, not a reused future trace.',
        'Rank-prefix longest gaps split into133 nonselected calls, one997-position partial restore, then200 nonselected calls, then four boosted restore calls. The partial restore resets idle133→0, is held40 calls, and is preempted again before producing a new output. The later boosted restore succeeds.',
        'Rank-prefix removes the measured 1/2-output re-preemption segments but leaves six 0-output re-preemption segments per run, versus two in fit-scan. It therefore does not eliminate unfinished-restore churn in this common backend.',
        'Elapsed time already differs before the first differing action: prefix minus fit is +0.290346s in block0 and -0.252724s in block1. These offsets are retained; later wall/completion differences cannot all be assigned to the differing packing actions.',
        'The direct residual supports a bounded next causal test at an observed zero-output recovery: preserve a selected recovery only until its first new output, then release the obligation, independently execute the future, and count costs to every other request. '
        'This is an untested action hypothesis, not evidence of feasibility, net benefit or full-LTR failure. The earliest post-start starvation/blocking state should be checked before allocating another run.',
        'The complete action boundaries, ranked candidates, both prefix longest-gap timelines, and every retained 0/1/2-output segment are in analysis.json.'])
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n'); (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],pairs=[dict(label=p['prefix_label'],first_divergence=p['first_action_divergence_step'],common_outputs=p['prefix_boundary']['returned_output_tokens'],maxgap=p['rank_prefix_maximum_gap']['gap_s'],short=p['prefix_short_counts']) for p in pairs])))


if __name__=='__main__': main()
