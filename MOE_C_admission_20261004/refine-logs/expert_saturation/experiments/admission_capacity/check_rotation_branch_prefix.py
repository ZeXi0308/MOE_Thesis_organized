#!/usr/bin/env python3
"""Verify observed pre-action prefixes, without claiming hidden/KV equivalence."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def inspect(path,cutoff):
    raw=json.loads(path.read_text())
    ids=raw['internal_to_source']
    normalize=lambda rid:ids[rid]
    trace=next(m for m in raw['memory_trace'] if m['attempted_step']==cutoff)['before']
    state=dict(pool=trace['pool'],running=[normalize(r) for r in trace['running_ids']],
               waiting_count=trace['waiting_count'],
               requests={normalize(k):v for k,v in trace['requests'].items()})
    requests={r['request_id']:r for r in raw['requests']}
    prefixes={rid:requests[rid]['output_token_ids'][:s['output_tokens']]
              for rid,s in state['requests'].items()}
    timeline=[]
    for step in raw['scheduler_steps']:
        if step['step']>=cutoff:break
        rows=[{k:v for k,v in r.items() if k!='internal_request_id'} for r in step['scheduled']]
        timeline.append(dict(step=step['step'],rows=rows,running=step['running_before'],waiting=step['waiting_before']))
    assert len(timeline)==cutoff
    calls=[c for c in raw['engine_steps'] if c['scheduler_step_end']<=cutoff]
    assert len(calls)==cutoff and all(c['completed'] for c in calls)
    boundary=calls[-1]['returned_s']
    emitted={rid:[] for rid in prefixes}
    for event in raw['output_events']:
        if event['received_s']<=boundary:
            emitted[event['request_id']].extend(event['new_token_ids'])
    assert emitted==prefixes, 'output prefix does not align to pre-action boundary'
    return dict(path=str(path),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                state=state,prefixes=prefixes,timeline=timeline,
                executed=[dict(step=s["step"],rows=s["rows"]) for s in timeline],
                prefix_tokens=sum(map(len,prefixes.values())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--cutoff',type=int,default=329)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reference=None;rows=[]
    for block in (0,1):
        for arm in ('least_progress','most_output'):
            label=f'block{block}-d6-{arm}'
            item=inspect(a.results/label/'raw.json',a.cutoff)
            if reference is None:reference=item
            equal={k:item[k]==reference[k] for k in ('state','prefixes','timeline','executed')}
            first_difference=next((i for i,(x,y) in enumerate(zip(reference['timeline'],item['timeline'])) if x!=y),None)
            rows.append(dict(label=label,raw_sha256=item['raw_sha256'],prefix_tokens=item['prefix_tokens'],
                             hashes={k:digest(item[k]) for k in equal},equal_to_reference=equal,
                             first_schedule_difference=first_difference))
    out=dict(status='OBSERVED_PREFIX_CHECK',cutoff=a.cutoff,cells=rows,
             state_at_cutoff=reference['state'],
             scope='Necessary observable alignment only. No KV tensor contents, hidden states, '
                   'RNG state, allocator order or counterfactual outcome verified. No GPU run.')
    with a.output.open('x') as f:json.dump(out,f,indent=2)
    for row in rows:print(row['label'],row['prefix_tokens'],row['equal_to_reference'],row['first_schedule_difference'])


if __name__=='__main__':main()
