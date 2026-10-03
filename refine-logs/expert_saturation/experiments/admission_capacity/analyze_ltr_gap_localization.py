#!/usr/bin/env python3
"""CPU localization of retained LTR-component output gaps; no policy replay."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
from analyze_prefix_cache_baseline import read, require, sha, digest


def gap(raw, decisions, request, index):
    start,end=request['token_times_s'][index:index+2]; rid=request['request_id']; iid=request['internal_request_id']
    requests={q['request_id']:q for q in raw['requests']}; aliases=raw['internal_to_source']
    calls={k:c for c in raw['engine_steps'] for k in range(c['scheduler_step_start'],c['scheduler_step_end'])}
    steps=[s for s in raw['scheduler_steps'] if start <= s['start_s'] < end]
    timeline, selections, recompute, runs, open_run, signatures = [], [], [], [], [], []
    for s in steps:
        k=s['step']; d=decisions[k]; call=calls[k]; v=d['counter_state_before'][iid]
        require(call['completed'] and call['scheduler_step_end']-call['scheduler_step_start']==1, 'requires confirmed single-schedule calls')
        output=lambda q: bisect_right(q['token_times_s'],call['returned_s'])-bisect_right(q['token_times_s'],call['start_s'])
        selected=next((x for x in s['scheduled'] if x['request_id']==rid),None)
        row=dict(step=k,call_start_s=call['start_s'],call_return_s=call['returned_s'],idle_before=v['idle'],priority=v['priority'],
                 selected_tokens=0 if selected is None else selected['scheduled_tokens'],new_outputs=output(request),
                 held=iid in raw['memory_trace'][k]['before']['running_ids'] and selected is None and iid not in d['victims'])
        timeline.append(row)
        if selected is None:
            open_run.append(row)
        else:
            if open_run: runs.append(open_run); open_run=[]
            following=decisions[k+1]['counter_state_before'].get(iid) if k+1 < len(decisions) else None
            event=dict(row,recompute_tokens=selected['recompute_tokens'],computed_before=selected['computed_before'],
                       computed_after=selected['computed_after'],idle_after_selection=0,next_idle_before=None if following is None else following['idle'],
                       reset_without_new_output=output(request)==0)
            selections.append(event)
        for x in s['scheduled']:
            if x['recompute_tokens']:
                recompute.append(dict(step=k,request_id=x['request_id'],scheduled_tokens=x['scheduled_tokens'],
                                      recompute_tokens=x['recompute_tokens'],new_outputs=output(requests[x['request_id']])))
        signatures.append(dict(step=k,idle=v['idle'],priority=v['priority'],tokens=d['tokens'].get(iid,0),
                               victims=[aliases[v] for v in d['victims']],recompute=[[x['request_id'],x['recompute_tokens']] for x in s['scheduled'] if x['recompute_tokens']]))
    if open_run: runs.append(open_run)
    span_steps={s['step'] for s in steps}; victims=[]
    for e in raw['preemption_events']:
        if e['attempted_step'] in span_steps:
            victims.append(dict(step=e['attempted_step'],request_id=aliases[e['victim_internal_request_id']],
                computed_tokens=e['victim_state']['computed_tokens'],outputs=e['victim_state']['output_tokens'],
                released_blocks=e['pool_after']['free_blocks']-e['pool']['free_blocks']))
    engine_s=sum(max(0,min(end,calls[s['step']]['returned_s'])-max(start,calls[s['step']]['start_s'])) for s in steps)
    require(0 <= engine_s <= end-start+1e-8, 'gap engine accounting overlaps')
    other={q['request_id']:bisect_right(q['token_times_s'],end)-bisect_right(q['token_times_s'],start) for q in raw['requests'] if q['request_id']!=rid}
    return dict(request_id=rid,previous_output_count=index+1,start_s=start,end_s=end,gap_s=end-start,
        first_step=steps[0]['step'],last_step=steps[-1]['step'],scheduling_calls=len(steps),
        maximum_contiguous_nonselected=max((len(r) for r in runs),default=0),maximum_idle_before=max(r['idle_before'] for r in timeline),
        nonselected_runs=[dict(first_step=r[0]['step'],last_step=r[-1]['step'],calls=len(r),
                             span_s=r[-1]['call_return_s']-r[0]['call_start_s']) for r in runs],
        target_selections=selections,target_idle_resets_without_new_output=sum(r['reset_without_new_output'] for r in selections),
        target_recompute_tokens=sum(r['recompute_tokens'] for r in selections),held_calls=sum(r['held'] for r in timeline),
        target_boosted_calls=sum(r['priority']==-1 and r['selected_tokens']>0 for r in timeline),
        victim_events=victims,recompute_selections=recompute,other_output_counts=other,other_new_outputs=sum(other.values()),
        engine_call_s=engine_s,host_outside_engine_s=end-start-engine_s,
        mean_engine_call_s=engine_s/len(steps),max_engine_call_s=max(calls[s['step']]['returned_s']-calls[s['step']]['start_s'] for s in steps),
        scheduler_s=sum(s['end_s']-s['start_s'] for s in steps),decision_s=sum(decisions[s['step']]['decision_seconds'] for s in steps),
        step_path_sha256=digest(signatures),timeline=timeline)


def inspect(row, matched):
    raw_path=Path(row['raw_path']); decisions_path=raw_path.parent/'component-decisions.json'
    require(sha(raw_path)==row['raw_sha256'] and sha(decisions_path)==row['decisions_sha256'], 'retained source changed after main analysis')
    raw=read(raw_path); decisions=read(decisions_path); requests={q['request_id']:q for q in raw['requests']}
    target=max(row['per_request'],key=lambda q:q['max_itl_s']); request=requests[target['request_id']]
    worst=lambda q: max(range(len(q['token_times_s'])-1),key=lambda i:q['token_times_s'][i+1]-q['token_times_s'][i])
    maximum=gap(raw,decisions,request,worst(request)); require(maximum['gap_s']==row['max_itl_s'], 'maximum gap differs from qualified metrics')
    epochs=row['component']['quantum_epochs']['epochs']; require(epochs and all(e['policy_effective'] for e in epochs), 'missing effective boost epoch')
    epoch=min(epochs,key=lambda e:e['start_step']); boosted=requests[epoch['request_id']]; t=epoch['first_new_output_after_start_s']
    i=boosted['token_times_s'].index(t); require(i>0, 'first boost output has no prior-token gap')
    boost_gap=gap(raw,decisions,boosted,i-1); k=epoch['selected_steps'][0]; action=decisions[k]; aliases=raw['internal_to_source']
    first=dict(step=k,start_s=raw['scheduler_steps'][k]['start_s'],boosted_request=epoch['request_id'],
               selected={aliases[r]:n for r,n in action['actual_scheduled'].items()},
               victims=[aliases[r] for r in action['victims']],free_before=action['free_before'],free_after=action['free_after'],
               counter_before=action['counter_state_before'][boosted['internal_request_id']])
    relevant={maximum['request_id'],epoch['request_id'],*first['victims']}
    return dict(label=row['label'],raw_sha256=row['raw_sha256'],decisions_sha256=row['decisions_sha256'],
                maximum_gap=maximum,boosted_request_gap=boost_gap,boosted_request_max_itl_s=max(b-a for a,b in zip(boosted['token_times_s'],boosted['token_times_s'][1:])),
                first_effective_boost=first,quantum_epoch=epoch,quantum_exhausted_without_new_output=row['component']['quantum_epochs']['exhausted_without_new_output'],
                matched_off_on_changes=[x for x in matched['per_request_changes'] if x['request_id'] in relevant])


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--analysis-json',type=Path,required=True); parser.add_argument('--output-dir',type=Path,required=True); args=parser.parse_args()
    require(not args.output_dir.exists(),'refuse output overwrite'); analysis=read(args.analysis_json)
    require(analysis['status']=='MEASUREMENT_ONLY' and len(analysis['cells'])==4 and all(r['eligible'] for r in analysis['cells']), 'requires qualified complete four-cell analysis')
    rows=[]
    for row in analysis['cells']:
        if row['policy']=='custom_ltr_component_boost_on':
            matched=next(p for p in analysis['comparisons'] if p['action']==row['label'] and p['kind'].endswith('off_on'))
            rows.append(inspect(row,matched))
    require(len(rows)==2,'expected two retained on cells')
    result=dict(status='MEASUREMENT_ONLY_EVENT_LOCALIZATION',source_analysis=str(args.analysis_json),source_analysis_sha256=sha(args.analysis_json),cells=rows,
        maximum_gap_same_step_path=rows[0]['maximum_gap']['step_path_sha256']==rows[1]['maximum_gap']['step_path_sha256'],
        scope='Observed same-input native-recompute component trajectories only; no counterfactual, new scheduler, full LTR or method failure. '
        'Gap output window is (previous receipt,next receipt]; scheduler steps and completed engine calls are joined by step index. '
        'Scheduler/decision times are nested in engine time, never added or subtracted as savings. Future receipts are outcomes only.')
    lines=['# LTR component longest-gap localization','',result['status'],'',result['scope'],'',
           '| cell | max-gap request | gap s | calls | max consecutive unselected | idle resets without output | boosted calls in max gap |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        g=r['maximum_gap']; lines.append(f"| {r['label']} | {g['request_id']} | {g['gap_s']:.6f} | {g['scheduling_calls']} | {g['maximum_contiguous_nonselected']} | {g['target_idle_resets_without_new_output']} | {g['target_boosted_calls']} |")
    for r in rows:
        g=r['maximum_gap']; e=r['quantum_epoch']; f=r['first_effective_boost']; early=[x for x in g['target_selections'] if x['reset_without_new_output']]
        lines.extend(['',f"{r['label']}: output {g['previous_output_count']}→{g['previous_output_count']+1}, steps {g['first_step']}–{g['last_step']}. "
            f"Target partial-service steps {[x['step'] for x in early]} assign idle=0 without a new token; positive idle values reset are {[x['idle_before'] for x in early if x['idle_before']>0]}. Total target recomputation {g['target_recompute_tokens']} positions; "
            f"held calls {g['held_calls']}; other requests return {g['other_new_outputs']} tokens in the same gap. "
            f"First boost at step {f['step']} selects {f['boosted_request']} and evicts {f['victims']}; its 10-call epoch returns {e['new_outputs']} new outputs. "
            f"Quantum exhaustion without new output: {r['quantum_exhausted_without_new_output']}. "
            f"Gap engine calls {g['engine_call_s']:.6f}s, outside-engine host intervals {g['host_outside_engine_s']:.6f}s; nested scheduler {g['scheduler_s']:.6f}s."])
    lines.extend(['','Both max-gap step paths equal: '+str(result['maximum_gap_same_step_path'])+'.',
        'The measured residual is an output gap split by partial recompute service: selected-call idleness can reset while output absence continues. '
        'The successful quantum serves a different request; it does not establish a bound for every request. '
        'Matched request costs and timing may change across repeats; full engine/compiler/GPU cause and a better policy remain untested.',
        'Next smallest step: check whether the existing funded-resume proposal preserves this interrupted recovery through its first new output; benefits remain unmeasured until an independently executed comparison.'])
    args.output_dir.mkdir(parents=True); (args.output_dir/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],same_path=result['maximum_gap_same_step_path'],cells=[dict(label=r['label'],gap_s=r['maximum_gap']['gap_s'],first_step=r['maximum_gap']['first_step'],last_step=r['maximum_gap']['last_step'],max_nonselected=r['maximum_gap']['maximum_contiguous_nonselected'],idle_resets_without_new_output=r['maximum_gap']['target_idle_resets_without_new_output']) for r in rows])))


if __name__=='__main__':
    main()
