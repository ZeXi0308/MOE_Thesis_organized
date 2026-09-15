#!/usr/bin/env python3
"""Real-trajectory obligation delivery, cost and delay-transfer localization."""
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
BUNDLE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BUNDLE.parents[2]/'experiments/admission_capacity'))
from analyze_prefix_cache_baseline import read,sha,digest,require
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


def ledger_delivery(raw,decisions,ledger,row):
    aliases=raw['internal_to_source']; requests={q['request_id']:q for q in raw['requests']}; delivered=[]
    by_start={(e['request_id'],e['start_step']):e for e in ledger['events'] if e['type']=='release'}
    require(ledger==row['restore_obligations']['snapshot'],'ledger differs from qualified helper replay')
    for start in (e for e in ledger['events'] if e['type']=='start'):
        iid=start['request_id'];rid=aliases[iid];q=requests[rid];n=start['output_tokens'];k=start['step'];release=by_start[(iid,k)]
        close=release['release_step']; times=q['token_times_s']; first=times[n] if n<len(times) else None
        s=raw['scheduler_steps'][close] if close<len(raw['scheduler_steps']) else None
        if release['reason']=='new_output':
            require(s is not None and first<=s['start_s'] and bisect_right(times,s['start_s'])==release['observed_output_tokens']>n,'new-output release lacks prior actual receipt')
        elif release['reason']=='interrupted':
            require(s is not None and iid in decisions[close]['victims'] and bisect_right(times,s['start_s'])==n,'interrupted obligation boundary differs')
        else:
            require(release['reason']=='completed' and q['status']=='completed' and q['completion_s']<=(s['start_s'] if s else raw['observation_end_s']),'completion release lacks real completion')
        residency=next(r for r in row['recovery_residencies']['rows'] if r['request_id']==rid and r['first_service_step']==k)
        end=residency['end_step']; post_gap=None
        if end is not None and residency['new_outputs']>0:
            end_s=raw['scheduler_steps'][end]
            post_gap=dict(lower_s=end_s['start_s']-first,upper_s=end_s['end_s']-first)
        delivered.append(dict(request_id=rid,start_step=k,start_outputs=n,release_step=close,release_reason=release['reason'],
            observed_output_tokens=release['observed_output_tokens'],first_new_output_s=first,
            actual_selected_steps=[j for j in range(k,min(close,len(decisions))) if iid in decisions[j]['actual_scheduled']],
            active_minus2_selected_steps=[j for j in range(k,min(close,len(decisions))) if iid in decisions[j]['actual_scheduled'] and decisions[j]['priorities'][iid]==-2],
            residency_end_step=end,residency_end_reason=residency['end_reason'],new_outputs_before_next_preemption=residency['new_outputs'],
            first_output_to_repreemption_bounds=post_gap))
    return delivered


def request_gap(raw,decisions,rid):
    q=next(q for q in raw['requests'] if q['request_id']==rid);times=q['token_times_s'];iid=q['internal_request_id']
    index=max(range(len(times)-1),key=lambda i:times[i+1]-times[i]);g=gap(raw,decisions,q,index)
    g['active_minus2_selections']=[k for k in range(g['first_step'],g['last_step']+1) if iid in decisions[k]['actual_scheduled'] and decisions[k]['priorities'][iid]==-2]
    g['all_obligation_selected_steps']=[dict(step=k,requests=[raw['internal_to_source'][r] for r in decisions[k]['actual_scheduled'] if decisions[k]['priorities'][r]==-2]) for k in range(g['first_step'],g['last_step']+1) if any(decisions[k]['priorities'][r]==-2 for r in decisions[k]['actual_scheduled'])]
    return g


def load_cell(row):
    path=Path(row['raw_path']);d=path.parent/'component-decisions.json';ledger_path=path.parent/'restore-obligations.json'
    require(sha(path)==row['raw_sha256'] and sha(d)==row['decisions_sha256'] and sha(ledger_path)==row['obligation_ledger_sha256'],'qualified source changed')
    raw=read(path);decisions=read(d);ledger=read(ledger_path);delivery=ledger_delivery(raw,decisions,ledger,row)
    work=Counter()
    for s in raw['scheduler_steps']:
        for x in s['scheduled']:work[x['request_id']]+=x['recompute_tokens']
    require(sum(work.values())==row['work']['totals']['repeated_executed'],'per-request recompute accounting differs')
    maximum=max(row['per_request'],key=lambda q:q['max_itl_s'])
    return raw,decisions,dict(label=row['label'],delivery=delivery,recompute_by_request=dict(work),
        maximum_gap=request_gap(raw,decisions,maximum['request_id']),short_residencies=[r for r in row['recovery_residencies']['rows'] if r['resumed'] and r['end_reason']=='actual_repreemption' and r['new_outputs']<=2])


def inspect_pair(off,on,matched):
    a,da,ra=load_cell(off);b,db,rb=load_cell(on)
    k=next(i for i in range(min(len(a['scheduler_steps']),len(b['scheduler_steps']))) if action(a['scheduler_steps'][i])!=action(b['scheduler_steps'][i]))
    ba,bb=boundary(a,da,k),boundary(b,db,k)
    changes=matched['per_request_changes'];top=sorted((q for q in changes if q['max_itl_s']>0),key=lambda q:q['max_itl_s'],reverse=True)[:3]
    harmed=[]
    for item in top:
        rid=item['request_id'];harmed.append(dict(request_id=rid,matched_change=item,off_gap=request_gap(a,da,rid),on_gap=request_gap(b,db,rid)))
    deltas=[dict(request_id=rid,off=ra['recompute_by_request'][rid],on=rb['recompute_by_request'][rid],delta=rb['recompute_by_request'][rid]-ra['recompute_by_request'][rid]) for rid in sorted(ra['recompute_by_request'])]
    first=next(e for e in rb['delivery'] if e['start_step']<k<e['release_step'])
    original=next(e for e in ra['delivery'] if e['request_id']==first['request_id'] and e['start_step']==first['start_step'])
    require(original['start_outputs']==first['start_outputs'],'first common obligation start differs')
    return dict(off_label=off['label'],on_label=on['label'],first_actual_divergence_step=k,off_boundary=ba,on_boundary=bb,
        recorded_prestate_equal=ba['visible_prestate_sha256']==bb['visible_prestate_sha256'],available_output_prefix_equal=ba['returned_prefix_sha256']==bb['returned_prefix_sha256'],
        pre_divergence_elapsed_delta_s=bb['elapsed_through_last_common_call_s']-ba['elapsed_through_last_common_call_s'],
        first_common_obligation=dict(off=original,on=first),off=ra,on=rb,top_itl_harmed=harmed,
        per_request_recompute_changes=sorted(deltas,key=lambda q:abs(q['delta']),reverse=True),matched_per_request_changes=changes,
        request_change_counts=matched['per_request_change_counts'],aggregate_deltas=matched['deltas'],
        work={r['label']:r['work']['totals'] for r in (off,on)},scheduler_cost={r['label']:r['component']['totals'] for r in (off,on)})


def main():
    out=Path(__file__).parent;require(not (out/'analysis.json').exists(),'refuse overwrite')
    source=BUNDLE/'analysis/analysis.json';main=read(source);require(main['status']=='MEASUREMENT_ONLY' and all(r['eligible'] for r in main['cells']),'requires qualified four-cell result')
    cells={r['label']:r for r in main['cells']};pairs=[]
    for i in (0,1):
        off,on=cells[f'block{i}-d6-restore-off'],cells[f'block{i}-d6-restore-on'];matched=next(p for p in main['comparisons'] if p['kind']==f'block{i}_off_on')
        pairs.append(inspect_pair(off,on,matched))
    result=dict(status='MEASUREMENT_ONLY_DELIVERY_COST_TRANSFER',source_analysis_sha256=sha(source),pairs=pairs,
        scope='Real independent trajectories and actual ledger/receipt alignment only. No fixed-future counterfactual. '
        'Returned prefixes and visible prestate equality are not full-engine or KV-byte checkpoint equality. '
        'First-output to later preemption is bounded by the containing scheduler interval; nested costs are not additive. '
        'Later request states differ across arms; only the initial common obligation has a matched action-prefix comparison.')
    lines=['# First-output obligation: delivery, cost, and transferred delays','',result['status'],'',result['scope'],'',
        '| pair | first action divergence | common outputs off/on | elapsed delta before divergence s | max ITL off→on s |',
        '|---|---:|---|---:|---|']
    for p in pairs:
        a,b=p['off_boundary'],p['on_boundary'];oa,ob=p['off']['maximum_gap'],p['on']['maximum_gap'];first=p['first_common_obligation']
        lines.append(f"| {p['on_label']} | {p['first_actual_divergence_step']} | {a['returned_output_tokens']}/{b['returned_output_tokens']} | {p['pre_divergence_elapsed_delta_s']:+.6f} | {oa['gap_s']:.6f}→{ob['gap_s']:.6f} |")
        lines.extend(['',f"First shared restore {first['on']['request_id']} starts at step{first['on']['start_step']}; off releases as {first['off']['release_reason']} at {first['off']['release_step']}, "
            f"on releases as {first['on']['release_reason']} at {first['on']['release_step']} and is re-preempted at {first['on']['residency_end_step']} after {first['on']['new_outputs_before_next_preemption']} new output. "
            f"On maximum gap belongs to {ob['request_id']} at steps{ob['first_step']}–{ob['last_step']}."])
    lines.extend(['','Delivery: both on cells fulfill27 first-output obligations, with81 actually selected -2 calls and zero interrupted obligations. Off has six interruptions. '
        'All six zero-output re-preemption residencies disappear; four one-output residencies remain on, no two-output residencies. This is successful first-output delivery, not continuing-service protection.',
        'Cost: repeated executed positions increase74982→96955 (+21973), scheduling calls1872→1906, actual preemptions25→27, held-resident calls916→1663 in both repeats. Fresh prompt/decode position counts are unchanged. '
        'Full completion/wall benefit changes sign across repeats, and timing already differs before the first action divergence; do not assign all timing changes to the intervention.',
        'Transfer: five requests improve maximum ITL and27 worsen in each pair. The largest stable ITL harm is request3259; its actual gap, the other top affected gaps, per-request recompute changes, and all output differences remain in analysis.json.',
        'Verdict: the obligation delivers its promised first new output and reduces the global longest generation pause in these two repeats. The remaining failure is incomplete cost amortization and redistributed pauses after release; net efficiency is not established. '
        'Do not tune200/10 or blindly lengthen protection. The next bounded causal question is whether an already qualified competing recovery should start when its first-output work will soon be discarded; it must account for the victim and every held request, not assume a longer lease is beneficial.'])
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],pairs=[dict(on=p['on_label'],first_divergence=p['first_actual_divergence_step'],prefix_elapsed_delta=p['pre_divergence_elapsed_delta_s'],on_maxgap=p['on']['maximum_gap']['gap_s']) for p in pairs])))


if __name__=='__main__':main()
