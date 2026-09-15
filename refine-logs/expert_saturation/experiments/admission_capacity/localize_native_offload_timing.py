"""Observed common-prefix and same-arm timing localization; no counterfactuals."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity';p=o/'20260914_native_offload_baseline_r01';load=lambda p:json.loads(p.read_text())
raw={name:load(p/'readback/results'/name/'raw.json') for name in ['block0-off','block0-on','block1-on','block1-off']}
def sig(s):return [(r['request_id'],r['scheduled_start_computed'],r['scheduled_tokens'],r['output_tokens_before']) for r in s['scheduled']]
def timing(c,s):return {'engine':c['returned_s']-c['start_s'],'scheduler':s['end_s']-s['start_s']}
rows=[]
for a,b in [('block0-off','block0-on'),('block1-off','block1-on'),('block0-on','block1-on'),('block0-off','block1-off')]:
 x,y=raw[a],raw[b];xs,ys=x['scheduler_steps'],y['scheduler_steps'];xc,yc=x['engine_steps'],y['engine_steps']
 assert len(xc)==len(xs) and len(yc)==len(ys)
 first=next((i for i,(u,v) in enumerate(zip(xs,ys)) if sig(u)!=sig(v)),min(len(xs),len(ys)))
 common=[i for i in range(min(len(xs),len(ys))) if sig(xs[i])==sig(ys[i])]
 diffs=[]
 for i in range(min(len(xs),len(ys))):
  tx,ty=timing(xc[i],xs[i]),timing(yc[i],ys[i]);diffs.append(dict(step=i,signature_equal=i in common,engine_delta_s=ty['engine']-tx['engine'],scheduler_delta_s=ty['scheduler']-tx['scheduler']))
 token_counts={}
 for s in xs[:first]:
  for r in s['scheduled']:token_counts[r['request_id']]=max(token_counts.get(r['request_id'],0),r['output_tokens_before'])
 xr={r['request_id']:r for r in x['requests']};yr={r['request_id']:r for r in y['requests']}
 full_equal=sum(xr[r]['output_token_ids']==yr[r]['output_token_ids'] for r in xr)
 prefix_equal=all(xr[r]['output_token_ids'][:n]==yr[r]['output_token_ids'][:n] for r,n in token_counts.items())
 buckets={}
 for name,indices in [('common_prefix',range(first)),('same_signature',common),('post_prefix',range(first,len(diffs)))]:
  buckets[name]=dict(calls=len(indices),engine_delta_s=sum(diffs[i]['engine_delta_s'] for i in indices),scheduler_delta_s=sum(diffs[i]['scheduler_delta_s'] for i in indices))
 rows.append(dict(a=a,b=b,first_schedule_difference=first if first<max(len(xs),len(ys)) else None,full_output_sequences_equal=full_equal,prefix_output_equal=prefix_equal,prefix_tokens_checked=sum(token_counts.values()),buckets=buckets,largest_absolute_call_deltas=sorted(diffs,key=lambda r:abs(r['engine_delta_s']),reverse=True)[:12]))
(p/'prefix_timing.json').write_text(json.dumps(dict(status='OBSERVED_LOCALIZATION',comparisons=rows,scope='Same schedule signature is logical executed work, not equal physical KV layout, kernels or hidden state. Same-signature buckets can be disjoint; only common_prefix precedes schedule divergence.'),indent=2)+'\n')
for r in rows:print({k:v for k,v in r.items() if k!='largest_absolute_call_deltas'});print('largest',r['largest_absolute_call_deltas'][:3])
