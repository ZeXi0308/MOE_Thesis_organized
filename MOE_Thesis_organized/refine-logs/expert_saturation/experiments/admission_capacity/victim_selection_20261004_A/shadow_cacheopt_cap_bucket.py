#!/usr/bin/env python3
"""Read-only CacheOPT-inspired component proxy; stdout only, no outcome model.

All SLOs share a tier. Declared remaining cap is bucketed by ceil(r/128),
larger first; allocated KV tokens by ceil(held_blocks*block_size/128), smaller
first; equal buckets choose the latest native suffix index. Bucket zero is
reserved for zero, and positive bucket k is ((k-1)*128, k*128]. These boundaries
and the final tie are this diagnostic's choices, not authors' defaults. This
is a cap-proxy adaptation of the paper's example span, not complete CacheOPT.
No EOS/outcome, old pure/private/pending qualification, or threshold search.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

SPAN=128


def read(path):
    return json.loads(path.read_text())


def integer(value, minimum=0):
    return type(value) is int and value>=minimum


def bucket_choice(event, block_size):
    tail=event.get('native_tail');rows=event.get('candidates')
    if not isinstance(rows,list) or not rows or any(not isinstance(r,dict) for r in rows):
        return tail,'UNKNOWN_SUFFIX'
    indices=[r.get('index') for r in rows]
    ids=[r.get('request') for r in rows]
    if (any(not integer(i) for i in indices)
            or indices!=list(range(indices[0],indices[-1]+1))
            or not integer(event.get('unprocessed_suffix_start'))
            or indices[0]!=event.get('unprocessed_suffix_start')
            or any(not isinstance(rid,str) for rid in ids) or len(set(ids))!=len(ids)
            or ids[0]!=event.get('failed_request') or ids[-1]!=tail):
        return tail,'UNKNOWN_SUFFIX'
    active=event.get('active_protection_or_phase')
    if type(active) is not bool:return tail,'UNKNOWN_PROTECTION'
    if active:return tail,'ACTIVE_PROTECTION_OR_PHASE'
    if not integer(block_size,1):return tail,'UNKNOWN_BLOCK_SIZE'
    for row in rows:
        cap=row.get('max_tokens');output=row.get('output_tokens');held=row.get('held_blocks')
        if (not integer(cap,1) or not integer(output) or output>cap
                or not integer(held) or row.get('request_status')!='RUNNING'):
            return tail,'UNKNOWN_BUDGET_CAPACITY_OR_NATIVE_STATUS'
    chosen=max(rows,key=lambda r:((r['max_tokens']-r['output_tokens']+SPAN-1)//SPAN,
        -((r['held_blocks']*block_size+SPAN-1)//SPAN),r['index']))
    return chosen['request'],None


def remaining(row):
    cap=row.get('max_tokens');output=row.get('output_tokens')
    return cap-output if integer(cap,1) and integer(output) and output<=cap else None


def delta_summary(values):
    if not values:return 'known=0'
    return f'known={len(values)} min={min(values)} median={median(values):g} max={max(values)} sum={sum(values)}'


def analyze_cell(cell):
    archive=cell/'archive';raw=read(archive/'raw.json');store=read(archive/'selective-store.json')
    profile=archive/'normal_capacity_profile.json'
    block_size=read(profile).get('block_size_tokens') if profile.is_file() else None
    events=store.get('victim_decisions',[]);mapping=raw.get('internal_to_source',{})
    records=[];fallbacks=Counter()
    for index,event in enumerate(events):
        selected,reason=bucket_choice(event,block_size)
        if reason:fallbacks[reason]+=1
        candidates=event.get('candidates')
        rows={r.get('request'):r for r in (candidates if isinstance(candidates,list) else []) if isinstance(r,dict)}
        budget=event.get('remaining_budget');release=event.get('max_release_shadow')
        choices=dict(actual=event.get('selected'),
            remaining_budget=budget.get('proposed_request') if isinstance(budget,dict) else None,
            max_release=release.get('proposed_request') if isinstance(release,dict) else None)
        records.append(dict(index=index,event=event,rows=rows,proxy=selected,reason=reason,choices=choices))
    print(f'\n{cell.name}: rule={store.get("native_victim_rule")} raw_status={raw.get("status")} '
          f'requests={len(raw.get("requests",[]))} decisions={len(events)} block_size={block_size}')
    print(f'  proxy fallbacks={dict(fallbacks)} UNKNOWN={sum(n for k,n in fallbacks.items() if k.startswith("UNKNOWN"))}')
    for label in ('actual','remaining_budget','max_release'):
        known=[r for r in records if r['proxy'] in r['rows'] and r['choices'][label] in r['rows']]
        different=[r for r in known if r['proxy']!=r['choices'][label]]
        release=[];left=[];held=[];same=0;release_unknown=0
        for record in different:
            a=record['rows'][record['proxy']];b=record['rows'][record['choices'][label]]
            x=a.get('immediate_releasable_blocks');y=b.get('immediate_releasable_blocks')
            if integer(x) and integer(y):release.append(x-y);same+=x==y
            else:release_unknown+=1
            x=remaining(a);y=remaining(b)
            if x is not None and y is not None:left.append(x-y)
            x=a.get('held_blocks');y=b.get('held_blocks')
            if integer(x) and integer(y):held.append(x-y)
        print(f'  vs {label}: different={len(different)}/{len(known)} comparator_unknown={len(records)-len(known)} '
              f'different_id_same_release={same} different_id_release_unknown={release_unknown}')
        print(f'    changed-only proxy-minus-comparator release_pages: {delta_summary(release)}; '
              f'held_pages: {delta_summary(held)}; remaining_tokens: {delta_summary(left)}')
    outside=[r for r in records if r['reason'] is None and all(r['choices'][k] in r['rows']
             and r['proxy']!=r['choices'][k] for k in ('remaining_budget','max_release'))]
    print(f'  differs_from_both_endpoint_shadows={len(outside)}/{len(records)}')
    # Fixed chronological examples: outside-both first, then other actual differences.
    examples=outside+[r for r in records if r not in outside and r['reason'] is None
        and r['proxy']!=r['choices']['actual']]
    for record in examples[:4]:
        event=record['event']
        print(f'  example decision={record["index"]} step={event.get("step")} '
              f'outside_both={record in outside}')
        for label,rid in dict(proxy=record['proxy'],**record['choices']).items():
            row=record['rows'].get(rid,{})
            left=remaining(row);held=row.get('held_blocks')
            rb=(left+SPAN-1)//SPAN if left is not None else None
            kb=(held*block_size+SPAN-1)//SPAN if integer(held) and integer(block_size,1) else None
            print(f'    {label}: {mapping.get(rid,rid)} index={row.get("index")} '
                  f'release={row.get("immediate_releasable_blocks")} held={held} '
                  f'remaining={left} buckets=({rb},{kb}) output={row.get("output_tokens")} '
                  f'cap={row.get("max_tokens")} pending={row.get("pending_native_store_dependencies")}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    args=parser.parse_args()
    cells=sorted(p for p in args.session.glob('cell-*') if (p/'archive/raw.json').is_file())
    if not cells:parser.error('No cell archives containing raw.json')
    print(__doc__.strip())
    print('Release comparisons use action-before candidate refcount observations; shadow choices were not executed. '
          'Each trajectory is reported separately; event counts are not independent repeats. '
          'Delta sums count decision observations, not cumulative counterfactual capacity. '
          'Recorded endpoint shadows are compared on these states, not replayed future trajectories.')
    for cell in cells:analyze_cell(cell)


if __name__=='__main__':main()
