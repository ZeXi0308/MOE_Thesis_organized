"""Retain incomplete cells; compare only fully completed off/on pairs."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
load=lambda p:json.loads(p.read_text());rows=[]
for label in ['block0-off','block0-on','block1-on','block1-off']:
 folder=a.results/label
 if not (folder/'status.json').exists():rows.append(dict(label=label,status='UNRUN'));continue
 status=load(folder/'status.json');row=dict(label=label,status=status['status']);rows.append(row)
 if row['status']!='COMPLETE':row['detail']=status;continue
 raw=load(folder/'raw.json');requests=raw['requests'];cfg=load(folder/'config.json');q=load(folder/'safe-cap-qualification.json');con=load(folder/'connector.json');events=load(folder/'offload-events.json')
 assert raw['status']=='COMPLETE' and len(requests)==32 and all(r['status']=='completed' and len(r['output_token_ids'])==1024 for r in requests)
 assert q['usable_blocks']==6656 and load(folder/'warmup-cache-reset.json')['success']
 assert con['type']==('OffloadingConnector' if cfg['offload_gib'] else 'NoneType')
 row.update(mean_completion_s=statistics.mean(r['completion_s']-r['arrival_s'] for r in requests),wall_s=raw['observation_end_s'],
   max_itl_s=max(t2-t1 for r in requests for t1,t2 in zip(r['token_times_s'],r['token_times_s'][1:])),
   recompute_positions=sum(s['recompute_tokens'] for s in raw['scheduler_steps']),
   positive_lookup_calls=sum((x['matched'] or 0)>0 for x in events['lookup']),
   lookup_matched_sum=sum(x['matched'] or 0 for x in events['lookup']),
   load_bytes=sum(x['load']['bytes'] for x in events['transfers']),store_bytes=sum(x['store']['bytes'] for x in events['transfers']),
   post_request_drain=load(folder/'post-request-drain.json'))
 row['external_load_observed']=row['load_bytes']>0 and row['positive_lookup_calls']>0
pairs=[]
for block in [0,1]:
 off=next(r for r in rows if r['label']==f'block{block}-off');on=next(r for r in rows if r['label']==f'block{block}-on')
 if off['status']==on['status']=='COMPLETE':pairs.append(dict(block=block,delta_pct={k:100*(on[k]/off[k]-1) for k in ['mean_completion_s','wall_s','max_itl_s','recompute_positions'] if off[k]},external_load_observed=on['external_load_observed']))
a.output.write_text(json.dumps(dict(status='MEASUREMENT_ONLY',rows=rows,pairs=pairs,scope='Lookup sums may include repeated queries; not unique loaded token count. Completed transfer bytes are actual reports. Incomplete pairs excluded from effects but retained as rows.'),indent=2)+'\n')
print(json.dumps(rows,indent=2))
