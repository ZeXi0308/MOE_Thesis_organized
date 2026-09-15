"""Finite native offload profile aggregation with per-step conservation."""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();load=lambda p:json.loads(p.read_text())
rows=[];raws={}
for name in ['block0-profile-off','block0-profile-on','block1-profile-on','block1-profile-off']:
 folder=a.results/name
 if not (folder/'status.json').exists():rows.append(dict(label=name,status='UNRUN'));continue
 status=load(folder/'status.json')['status'];row=dict(label=name,status=status);rows.append(row)
 if status!='COMPLETE':continue
 raw=load(folder/'raw.json');raws[name]=raw;requests=raw['requests'];assert len(requests)==32 and all(r['status']=='completed' and len(r['output_token_ids'])==1024 for r in requests)
 ev=load(folder/'offload-events.json');sp=load(folder/'cost-spans.json');q=load(folder/'safe-cap-qualification.json');assert q['usable_blocks']==6656 and load(folder/'warmup-cache-reset.json')['success']
 row.update(wall_s=raw['observation_end_s'],mean_completion_s=statistics.mean(r['completion_s']-r['arrival_s'] for r in requests),calls=len(raw['engine_steps']),load_bytes=sum(r['load']['bytes'] for r in ev['transfers']),store_bytes=sum(r['store']['bytes'] for r in ev['transfers']),recomputed_positions=sum(s['recompute_tokens'] for s in raw['scheduler_steps']))
 if sp['enabled']:
  grouped=defaultdict(list);totals=defaultdict(lambda:defaultdict(float))
  for s in sp['rows']:
   grouped[s['step']].append(s)
   for k in ['wall_s','thread_s','exclusive_wall_s','exclusive_thread_s']:totals[s['label']][k]+=s[k]
   totals[s['label']]['count']+=1
  errors=[]
  for step,ss in grouped.items():
   roots=[s for s in ss if s['label']=='engine.step'];assert len(roots)==1
   for k in ['wall_s','thread_s']:
    error=sum(s['exclusive_'+k] for s in ss)-roots[0][k];errors.append(abs(error));assert abs(error)<1e-7
  assert len(grouped)==len(raw['engine_steps'])+load(folder/'post-request-drain.json')['calls']
  row.update(profile_totals=totals,max_conservation_error_s=max(errors),profile_rows=len(sp['rows']))
comparisons=[]
def signature(raw):return [[(r['request_id'],r['scheduled_start_computed'],r['scheduled_tokens'],r['output_tokens_before']) for r in s['scheduled']] for s in raw['scheduler_steps']]
for left,right in [('block0-profile-off','block0-profile-on'),('block1-profile-off','block1-profile-on'),('block0-profile-on','block1-profile-on')]:
 if left not in raws or right not in raws:continue
 l,r=raws[left],raws[right];ls,rs=signature(l),signature(r)
 lr={x['request_id']:x for x in l['requests']};rr={x['request_id']:x for x in r['requests']}
 comparisons.append(dict(left=left,right=right,schedule_equal=ls==rs,first_schedule_difference=next((i for i,(x,y) in enumerate(zip(ls,rs)) if x!=y),None),equal_output_sequences=sum(lr[k]['output_token_ids']==rr[k]['output_token_ids'] for k in lr),wall_delta_pct=100*(r['observation_end_s']/l['observation_end_s']-1)))
a.output.write_text(json.dumps(dict(status='MEASUREMENT_ONLY',rows=rows,comparisons=comparisons,scope='Exclusive main-thread timings only; thread CPU is not all threads and engine residual is not necessarily GPU. Profile overhead and path equality must be considered.'),indent=2)+'\n')
for r in rows:print(r)
print(comparisons)
