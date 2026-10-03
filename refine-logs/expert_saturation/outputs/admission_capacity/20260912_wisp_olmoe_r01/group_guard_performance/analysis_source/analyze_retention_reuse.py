"""Observed cache lifetimes after mixed calls; all future labels are post-hoc."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read(p): return json.loads(p.read_text())


def reuse(records, index, expert, resident):
    """Ensure happens before compute; an ensure-only touch is not a use."""
    evicted = False
    loads = 0
    for distance, r in enumerate(records[index+1:], 1):
        for group in r['groups']:
            if expert in group['evicted_experts']:
                evicted = True
            loads += expert in group['loaded_experts']
            if expert in group['required_experts']:
                assert loads in (0, 1)
                return dict(distance=distance, step=r['context']['step_id'],
                    survived=resident and not evicted, loads=loads,
                    old_decode=expert in r['_old_decode'],
                    entry_hit=expert in r['entry_resident_experts'])
    return dict(distance=None, evicted_without_observed_reuse=evicted)


def add(stats, value):
    stats['origins'] += 1
    if value['distance'] is None:
        stats['right_censored_no_observed_reuse'] += 1
        stats['censored_evicted'] += int(value['evicted_without_observed_reuse'])
        return
    stats['observed_reuse'] += 1
    stats['reuse_with_original_residency'] += int(value['survived'])
    stats['reload_before_reuse'] += value['loads']
    stats['next_layer_call_use'] += value['distance'] == 1
    stats['next_layer_call_entry_hit'] += value['distance'] == 1 and value['entry_hit']
    stats['first_reuse_has_old_decode'] += value['old_decode']
    stats['first_reuse_old_decode_hit'] += value['old_decode'] and value['survived']
    stats['distance_'+str(value['distance'])] += 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); root=args.input_dir; source=read(root/'analysis.json')
    assert not source['issues'] and len(source['engines'])==2
    episodes=[]; pairs=[]; hashes={}
    for engine in source['engines']:
        data=root/'results'/engine['label']; path=data/'pager/calls.jsonl'
        hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        phases=defaultdict(list)
        with path.open() as f:
            for line in f:
                r=json.loads(line)
                if r['measurement']: phases[r['context']['phase']].append(r)
        indexed={}
        for row in engine['rows']:
            raw=read(data/row['name']/'raw.json'); old={r['request_id'] for r in raw['requests'] if r['arrival_s']==0}
            records=phases[row['phase']]; assert len(records)==384
            by_layer=defaultdict(list); current={}; touches=Counter()
            for r in records:
                ids=[(raw['internal_to_source'][m['internal_request_id']],m['computed_position'],m['prompt_tokens']) for m in r['context']['rows']]
                assert len(ids)==len(r['row_topk_experts'])
                r['_identity']=ids
                r['_old_decode']={expert for meta,route in zip(ids,r['row_topk_experts']) if meta[0] in old and meta[1]>=meta[2] for expert in route}
                state=set(r['entry_resident_experts']); name=r['layer_name']
                if name in current: assert current[name]==state
                before=set(state); computed=[]; loaded=[]
                for g in r['groups']:
                    evict,load,ensure,use=map(set,(g['evicted_experts'],g['loaded_experts'],g['ensure_experts'],g['required_experts']))
                    assert evict<=state and not load&state and not evict&ensure
                    state=(state-evict)|load
                    assert use<=ensure<=state and len(state)<=24
                    computed.extend(use);loaded.extend(load)
                    touches['ensure_only_occurrences']+=len(ensure-use)
                assert set(computed)==set(r['active_experts']) and len(computed)==len(set(computed))
                assert set(loaded)==set(r['active_experts'])-before and len(loaded)==len(set(loaded))
                assert state==set(r['retention']['final_resident_experts'])
                current[name]=state;by_layer[name].append(r)
            assert len(by_layer)==16 and all(len(rr)==24 for rr in by_layer.values())
            stats=defaultdict(Counter); origin_calls=Counter();next_misses=Counter()
            for layer, rr in by_layer.items():
                for i,r in enumerate(rr):
                    if not r['retention']['candidate_eligible']:continue
                    assert i+1<len(rr)
                    A,C,F=map(set,(r['active_experts'],r['entry_resident_experts'],r['retention']['final_resident_experts']))
                    P=set(r['retention']['chosen_protected_experts']) if r['retention']['applied'] else set()
                    assert P<=F
                    categories={'retained_protected':P,'retained_other':F-P,
                        'evicted_entry':C-F,'computed_not_retained':A-F}
                    origin_calls['mixed']+=1;origin_calls['applied']+=r['retention']['applied']
                    origin_calls['rejected']+=r['retention']['guard_rejected']
                    nxt=rr[i+1]; N=set(nxt['active_experts']); NC=set(nxt['entry_resident_experts'])
                    assert F==NC
                    next_misses['calls']+=1;next_misses['active']+=len(N);next_misses['hits']+=len(N&NC);next_misses['misses']+=len(N-NC)
                    for label,experts in categories.items():
                        for expert in experts:
                            v=reuse(rr,i,expert,expert in F);add(stats[label],v)
            assert origin_calls['mixed']==192
            episodes.append(dict(engine=engine['label'],repeat=row['repeat'],arm=row['arm'],
                origin_calls=dict(origin_calls),categories={k:dict(v) for k,v in stats.items()},next_calls=dict(next_misses),touches=dict(touches)))
            indexed[row['arm'],row['repeat']]={(r['context']['step_id'],r['layer_name']):r for r in records}
        for start in (0,3):
            block={r['arm']:r for r in engine['rows'][start:start+3]};base=block['none_early'];left=indexed[base['arm'],base['repeat']]
            for arm in ('frequency_late','frequency_late_no_extra_groups'):
                row=block[arm];right=indexed[arm,row['repeat']];account=Counter();stages=defaultdict(Counter)
                assert left.keys()==right.keys()
                for key,n in left.items():
                    t=right[key];assert n['_identity']==t['_identity']
                    A,B,C,D=map(set,(n['active_experts'],t['active_experts'],n['entry_resident_experts'],t['entry_resident_experts']))
                    values=dict(common_demand_saved=len((A&B)&(D-C)),common_demand_added=len((A&B)&(C-D)),
                        treatment_only_misses=len((B-A)-D),baseline_only_misses=len((A-B)-C),
                        miss_delta=len(B-D)-len(A-C))
                    assert values['miss_delta']==values['common_demand_added']-values['common_demand_saved']+values['treatment_only_misses']-values['baseline_only_misses']
                    account.update(values)
                    stage='pre_action' if key[0]<4 else 'old_active_mixed' if key[0]<16 else 'post_old'
                    stages[stage].update(values)
                pairs.append(dict(engine=engine['label'],block=start//3,arm=arm,account=dict(account),stages={k:dict(v) for k,v in stages.items()}))
    assert len(episodes)==12 and len(pairs)==8
    result=dict(status='POST_HOC_ACTUAL_RETENTION_LIFETIMES',episodes=episodes,pairs=pairs,input_sha256=hashes,
        scope=['All 12 executed performance episodes retained; same-mode repeats are dependent, not independent lifetime samples.',
            'Origins are expert occurrences at the end of eligible mixed layer calls. Retained categories partition final residents; evicted_entry and computed_not_retained overlap and must not be summed.',
            'Protected means chosen in an actually applied call; inert chosen sets count as retained_other. Ensure-only touches are not compute uses.',
            'Future labels stop at the first actual compute use in this policy; no observed use before episode end is right-censored, never proof of no future need.',
            'Cross-policy accounting uses common actual demand plus each policy unique demand, not a shared future trace or causal cache-only speedup. No latency or deployment claim.'])
    with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()
