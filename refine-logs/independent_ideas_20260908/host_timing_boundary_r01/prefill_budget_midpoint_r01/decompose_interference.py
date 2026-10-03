"""Intersect actual request spans with mutually exclusive host-time phases."""
import hashlib, json, statistics
from pathlib import Path
root=Path(__file__).resolve().parent
phases=('interstep_host','scheduler_prefix','poststamp_prefill','poststamp_decode')
output=[]
for path in sorted((root/'readback/results').glob('*/cell-*-raw.json')):
    raw=json.loads(path.read_text()); steps=raw['scheduler_steps']; intervals=[]; previous=0.0
    assert raw['status']=='COMPLETE' and len(raw['engine_step_returns'])==len(steps)
    for i,(s,ret) in enumerate(zip(steps,raw['engine_step_returns'])):
        start,end,receipt=s['start_s'],s['end_s'],ret['received_s']
        assert ret['scheduler_begin']==i and ret['scheduler_end']==i+1 and previous<=start<=end<=receipt
        label='poststamp_prefill' if any(r['prefill_tokens'] for r in s['scheduled']) else 'poststamp_decode'
        intervals.extend(((previous,start,phases[0]),(start,end,phases[1]),(end,receipt,label))); previous=receipt
    assert previous<=raw['observation_end_s']
    intervals.append((previous,raw['observation_end_s'],phases[0])); rows=[]; residuals=[]
    for r in raw['requests']:
        times=r['token_times_s']; assert len(times)==128 and times==sorted(times)
        parts={}
        for name,lo,hi,denom in [('ttft_s',r['arrival_s'],times[0],1),('tpot_s',times[0],times[-1],127)]:
            buckets={label:sum(max(0,min(end,hi)-max(start,lo)) for start,end,p in intervals if p==label)/denom for label in phases}
            residual=abs(sum(buckets.values())-(hi-lo)/denom);assert residual<1e-9;residuals.append(residual);parts[name]=buckets
        rows.append(dict(request_id=r['request_id'],prompt_tokens=r['prompt_tokens'],**parts))
    groups={}
    for group,length in [('all',None),('short',128),('long',2048)]:
        selected=[r for r in rows if length is None or r['prompt_tokens']==length]
        groups[group]={name:{p:statistics.mean(r[name][p] for r in selected) for p in phases} for name in ('ttft_s','tpot_s')} if selected else None
    output.append(dict(block=path.parent.name,plan=raw['plan'],source=str(path.relative_to(root)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),max_residual_s=max(residuals),groups=groups,requests=rows))
result=dict(semantics='Exact host-time accounting, not removable overhead or pure GPU time. Means add; quantiles do not. Poststamp includes scheduler wrapper tail and engine return. TTFT phases describe elapsed waiting intervals, not per-request exclusive resource use.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cells=output)
with (root/'analysis/interference.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
for c in output: print(json.dumps(dict(block=c['block'],plan=c['plan'],mean_tpot_parts_ms={k:1000*v for k,v in c['groups']['all']['tpot_s'].items()},max_residual_s=c['max_residual_s'])))
