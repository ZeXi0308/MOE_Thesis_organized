#!/usr/bin/env python3
"""Real ordered-dispatch divergence, ready service, and residual output pauses."""
import ast
from bisect import bisect_right
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
BUNDLE=Path(__file__).resolve().parents[2]
E=BUNDLE.parents[2]/'experiments/admission_capacity'
sys.path.insert(0,str(E))
from analyze_prefix_cache_baseline import read,sha,require
OLD=BUNDLE.parent/'20260914_restore_completion_r01/analysis/localization/localize.py'
spec=importlib.util.spec_from_file_location('prior_localization',OLD)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def ordered_action(step):
    return (old.action(step),[x['request_id'] for x in step['scheduled']])

def ready_service(raw,decisions,k,request_ids=None):
    m=raw['memory_trace'][k]['before'];d=decisions[k];aliases=raw['internal_to_source']
    qs={q['request_id']:q for q in raw['requests']}
    order=sorted(m['requests'],key=lambda r:(d['priorities'][r],qs[aliases[r]]['arrival_s'],r))
    active={r for r in order if d['priorities'][r]==-2}
    ready=[r for i,r in enumerate(order) if r in m['running_ids'] and r not in d['victims']
        and m['requests'][r]['prompt_tokens']+m['requests'][r]['output_tokens']-m['requests'][r]['computed_tokens']==1
        and (aliases[r] in request_ids if request_ids is not None else any(p in active for p in order[:i]))]
    call=next(c for c in raw['engine_steps'] if c['scheduler_step_start']==k)
    require(call['scheduler_step_end']==k+1 and call['completed'],'requires one confirmed schedule per call')
    rows=[]
    for r in ready:
        q=qs[aliases[r]];before=m['requests'][r]['output_tokens']
        count=bisect_right(q['token_times_s'],call['returned_s'])-before
        rows.append(dict(request_id=aliases[r],outputs_before=before,selected_tokens=d['actual_scheduled'].get(r,0),
            actual_new_outputs=count,first_new_output_s=q['token_times_s'][before] if count else None))
    return dict(step=k,candidate_scope='matched action request IDs' if request_ids is not None else 'later than actual active obligation',ready_candidates=len(ready),already_outputting=sum(r['outputs_before']>0 for r in rows),
        actually_selected=sum(r['selected_tokens']>0 for r in rows),new_outputs=sum(r['actual_new_outputs'] for r in rows),rows=rows)

def reservation_tax(raw,decisions):
    path=BUNDLE/'preparation/pkg/ltr_recompute_native.py'
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if getattr(n,'name',None) in ('Candidate','Plan','plan')]
    source=ast.unparse(ast.Module(body=nodes,type_ignores=[]))
    amount=next(line for line in source.splitlines() if line.strip().startswith('amount = min('))
    require(source.count(amount)==1,'planner amount observer boundary differs')
    observer="\n        observations.append(dict(request_id=row.request_id, ready=ready, before_budget=token_budget, amount=amount, withheld=max(0,min(row.pending_tokens,token_budget,chunk_limit or token_budget)-amount), candidates=[c.request_id for c in ordered[i+1:] if c.resident and c.pending_tokens==1 and c.request_id not in victims and c.request_id not in trial]))"
    source=source.replace(amount,amount+observer)
    stop="if packing == 'rank_prefix':\n                break"
    require(source.count(stop)==1,'rank-prefix stop observer boundary differs')
    source=source.replace(stop,"if packing == 'rank_prefix':\n                stops.append(dict(request_id=row.request_id,need=need,available=free_blocks+released))\n                break")
    ns=dict(dataclass=dataclass,__name__=__name__,observations=[],stops=[])
    exec(compile(source,str(path)+':observers','exec'),ns)
    aliases=raw['internal_to_source'];qs={q['request_id']:q for q in raw['requests']};events=[]
    chunk=read(Path(raw['_folder'])/'resolved-scheduler-config.json')['long_prefill_token_threshold']
    for d,m in zip(decisions,raw['memory_trace']):
        ns['observations'].clear();ns['stops'].clear();before=m['before']
        rows=[ns['Candidate'](r,d['priorities'][r],qs[aliases[r]]['arrival_s'],r in before['running_ids'],
            v['block_counts'][0],(v['prompt_tokens']+v['output_tokens']+15)//16,
            v['prompt_tokens']+v['output_tokens']-v['computed_tokens']) for r,v in before['requests'].items()]
        p=ns['plan'](rows,before['pool']['free_blocks'],1024,32,chunk,packing='rank_prefix',reserve_ready_tokens=True)
        require(p.tokens==d['actual_scheduled'] and p.victims==d['victims'] and p.free_after_reservation==d['free_after_reservation'],'observer changed actual plan')
        for e in ns['observations']:
            if e['ready'] and e['withheld']:
                selected=[r for r in e['candidates'] if r in p.tokens]
                unserved=[r for r in e['candidates'] if r not in p.tokens and r not in p.victims]
                events.append(dict(step=d['step'],protected_request=aliases[e['request_id']],ready_count=e['ready'],
                    withheld_tokens=e['withheld'],actual_later_selected=len(selected),actual_unserved=[aliases[r] for r in unserved],
                    actual_later_victims=[aliases[r] for r in e['candidates'] if r in p.victims],
                    unused_step_token_budget=1024-sum(p.tokens.values()),
                    rank_prefix_stops=[dict(request_id=aliases[x['request_id']],need=x['need'],available=x['available']) for x in ns['stops']]))
    return dict(events=events,withholding_events=len(events),unserved_candidate_events=sum(bool(e['actual_unserved']) for e in events),
        unserved_with_rank_prefix_stop_events=sum(bool(e['actual_unserved']) and bool(e['rank_prefix_stops']) for e in events),
        boundary='Observer-only frozen pure planner replay; candidates/withheld amounts can repeat across protected allocations and are not a net latency saving. Unused budget and unserved candidates are actual plan outcomes.')

def pair(a,da,ra,b,db,rb,matched):
    limit=min(len(a['scheduler_steps']),len(b['scheduler_steps']))
    token=next((k for k in range(limit) if old.action(a['scheduler_steps'][k])!=old.action(b['scheduler_steps'][k])),None)
    order=next((k for k in range(limit) if ordered_action(a['scheduler_steps'][k])!=ordered_action(b['scheduler_steps'][k])),None)
    if order is None:return dict(baseline=ra['label'],action=rb['label'],first_ordered_dispatch=None,first_tokens_victims=token,matched=matched)
    ba,bb=old.boundary(a,da,order),old.boundary(b,db,order)
    changes=matched['per_request_changes']
    extrema=[min(changes,key=lambda q:q['max_itl_s']),max(changes,key=lambda q:q['max_itl_s'])]
    details=[dict(change=q,baseline_gap=old.request_gap(a,da,q['request_id']),action_gap=old.request_gap(b,db,q['request_id'])) for q in extrema]
    token_boundary=None if token==order or token is None else dict(baseline=old.boundary(a,da,token),action=old.boundary(b,db,token))
    served=ready_service(b,db,order)
    return dict(baseline=ra['label'],action=rb['label'],first_ordered_dispatch=order,first_tokens_victims=token,
        baseline_boundary=ba,action_boundary=bb,later_token_boundary=token_boundary,
        recorded_visible_prestate_equal=ba['visible_prestate_sha256']==bb['visible_prestate_sha256'],
        returned_output_prefix_equal=ba['returned_prefix_sha256']==bb['returned_prefix_sha256'],
        elapsed_delta_before_first_ordered_dispatch_s=bb['elapsed_through_last_common_call_s']-ba['elapsed_through_last_common_call_s'],
        action_ready_service=served,baseline_ready_service=ready_service(a,da,order,{r['request_id'] for r in served['rows']}),
        largest_itl_gain_and_harm=details,matched=matched)

def main():
    output=Path(__file__).parent;require(not (output/'analysis.json').exists(),'preserve existing localization')
    main_path=BUNDLE/'analysis/analysis.json';main=read(main_path)
    require(main['status']=='MEASUREMENT_ONLY' and len(main['cells'])==6 and all(r['eligible'] for r in main['cells']),'requires six qualified real cells')
    indexed={r['label']:r for r in main['cells']};derived=[];pairs=[]
    def qualified(row):
        raw,dec,detail=old.load_cell(row);raw['_folder']=str(Path(row['raw_path']).parent)
        detail.update(work=row['work']['totals'],cost=row['component']['totals'],
            metrics={k:row[k] for k in ('wall_s','throughput_rps','mean_completion_s','max_itl_s')})
        return raw,dec,detail
    for block in (0,1):
        b,db,rb=qualified(indexed[f'block{block}-guard_residual'])
        rb['reservation_tax']=reservation_tax(b,db);derived.append(rb)
        for baseline in ('guard_all','fit_scan'):
            a,da,ra=qualified(indexed[f'block{block}-{baseline}']);derived.append(ra)
            matched=next(p for p in main['comparisons'] if p['baseline']==ra['label'] and p['action']==rb['label'])
            pairs.append(pair(a,da,ra,b,db,rb,matched))
            del a,da
        del b,db
    result=dict(status='MEASUREMENT_ONLY_DISPATCH_AND_GAP_LOCALIZATION',source_analysis_sha256=sha(main_path),cells=derived,pairs=pairs,
        scope='Actual independent trajectories only. Common state means recorded metadata/counters and returned output prefix, never KV tensor equality. Ordered dispatch can diverge before token/victim maps. Each gap is the interval between two real output receipts; nested costs are not added. Observer-only planning identifies unused reservations, not alternate-run latency.')
    lines=['# Restore token reservation: actual dispatch and remaining gaps','',result['status'],'',result['scope'],'',
        '| Pair | First ordered dispatch | First token/victim map | Visible/output prefix equal | Prior elapsed delta s | Ready selected/eligible |',
        '|---|---:|---:|---|---:|---|']
    for p in pairs:
        lines.append(f"| {p['baseline']} → {p['action']} | {p['first_ordered_dispatch']} | {p['first_tokens_victims']} | {p.get('recorded_visible_prestate_equal')}/{p.get('returned_output_prefix_equal')} | {p.get('elapsed_delta_before_first_ordered_dispatch_s',0):+.6f} | {p.get('action_ready_service',{}).get('actually_selected')}/{p.get('action_ready_service',{}).get('ready_candidates')} |")
    lines.extend(['','| Cell | Max gap request / output | Gap s | Steps | Longest unselected run | Held calls | Target recompute |',
        '|---|---|---:|---|---:|---:|---:|'])
    for c in derived:
        g=c['maximum_gap']
        lines.append(f"| {c['label']} | {g['request_id']} / {g['previous_output_count']} | {g['gap_s']:.6f} | {g['first_step']}–{g['last_step']} | {g['maximum_contiguous_nonselected']} | {g['held_calls']} | {g['target_recompute_tokens']} |")
        if 'reservation_tax' in c:
            t=c['reservation_tax'];lines.append(f"\n{c['label']}: withholding events {t['withholding_events']}; events with counted but actually unserved surviving ready candidates {t['unserved_candidate_events']}; those accompanied by an actual rank-prefix capacity stop {t['unserved_with_rank_prefix_stop_events']}.\n")
    lines.append('\nComplete per-request comparisons, maximal-gap victim/service timelines, actual ledger delivery, short residencies, full work/cost and reservation events are in analysis.json. No new controller or unexecuted latency claim.')
    (output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],pairs=len(pairs),cells=len(derived))))

if __name__=='__main__':main()
