import bisect, hashlib, json, math
from pathlib import Path
root=Path('/Users/leandrozhao/Desktop/、++++++++/refine-logs/independent_ideas_20260908/host_timing_boundary_r01/next_prefill_budget')
saved=json.loads((root/'analysis/interference.json').read_text())
phases=('interstep_host','scheduler_prefix','poststamp_prefill','poststamp_decode')
paths=sorted((root/'readback/results').glob('*/cell-*-raw.json'))
diffs={'request':[],'group':[],'max_residual':[]}; metadata=[]; percell=[]
assert len(paths)==len(saved['cells'])==8
for path,cell in zip(paths,saved['cells']):
 raw=json.loads(path.read_text()); phase_events=[(0.0,phases[0])]
 for i,step in enumerate(raw['scheduler_steps']):
  receipt=[r for r in raw['engine_step_returns'] if r['scheduler_begin']==i]
  assert len(receipt)==1 and receipt[0]['scheduler_end']==i+1
  rec=receipt[0]['received_s']
  assert phase_events[-1][0]<=step['start_s']<=step['end_s']<=rec
  phase_events.extend([(step['start_s'],phases[1]),(step['end_s'],phases[2] if sum(r['prefill_tokens'] for r in step['scheduled']) else phases[3]),(rec,phases[0])])
 assert phase_events[-1][0]<=raw['observation_end_s']
 boundaries=[v[0] for v in phase_events]
 name=str(path.relative_to(root)); rebuilt=[]; residuals=[]; ordinary_residuals=[]
 for field,val in [('block',path.parent.name),('plan',raw['plan']),('source',name),('source_sha256',hashlib.sha256(path.read_bytes()).hexdigest())]:
  if cell[field]!=val: metadata.append({'source':name,'field':field,'saved':cell[field],'rebuilt':val})
 assert len(raw['requests'])==len(cell['requests'])==16
 for request,stored in zip(raw['requests'],cell['requests']):
  rid=request['request_id']; parts={}
  for field in ('request_id','prompt_tokens'):
   if request[field]!=stored[field]: metadata.append({'source':name,'request':rid,'field':field})
  ts=request['token_times_s']; assert len(ts)==128 and ts==sorted(ts)
  for metric,lo,hi,den in [('ttft_s',request['arrival_s'],ts[0],1),('tpot_s',ts[0],ts[-1],len(ts)-1)]:
   cuts=sorted(set([lo,hi]+[t for t in boundaries if lo<t<hi]))
   lengths={p:[] for p in phases}
   for a,b in zip(cuts,cuts[1:]):
    phase=phase_events[bisect.bisect_right(boundaries,a)-1][1]
    lengths[phase].append(b-a)
   parts[metric]={p:math.fsum(lengths[p])/den for p in phases}
   residuals.append(abs(math.fsum(parts[metric].values())-(hi-lo)/den))
   ordinary_residuals.append(abs(sum(parts[metric].values())-(hi-lo)/den))
   for phase,value in parts[metric].items():
    old=stored[metric][phase]
    if value!=old: diffs['request'].append({'field':f'{name}/requests/{rid}/{metric}/{phase}','saved':old,'rebuilt':value,'absolute_difference':abs(value-old)})
  rebuilt.append(dict(request_id=rid,prompt_tokens=request['prompt_tokens'],**parts))
 for group,length in [('all',None),('short',128),('long',2048)]:
  selected=[r for r in rebuilt if length is None or r['prompt_tokens']==length]
  if not selected:
   assert cell['groups'][group] is None
   continue
  for metric in ('ttft_s','tpot_s'):
   for phase in phases:
    value=math.fsum(r[metric][phase] for r in selected)/len(selected);old=cell['groups'][group][metric][phase]
    if value!=old: diffs['group'].append({'field':f'{name}/groups/{group}/{metric}/{phase}','saved':old,'rebuilt':value,'absolute_difference':abs(value-old)})
 value=max(residuals);old=cell['max_residual_s']
 if value!=old: diffs['max_residual'].append({'field':f'{name}/max_residual_s','saved':old,'rebuilt':value,'ordinary_sum_rebuilt':max(ordinary_residuals),'absolute_difference':abs(value-old)})
 percell.append({'source':name,'saved_max_residual_s':old,'independent_fsum_max_residual_s':value,'independent_ordinary_max_residual_s':max(ordinary_residuals)})
current_sha=hashlib.sha256((root/'decompose_interference.py').read_bytes()).hexdigest()
if saved['script_sha256']!=current_sha:metadata.append({'field':'script_sha256','saved':saved['script_sha256'],'current':current_sha})
report={'cells':len(paths),'requests':len(paths)*16,'request_numeric_fields':len(paths)*16*8,'metadata_differences':metadata,'source_cell_order_matches':all(c['source']==str(p.relative_to(root)) for p,c in zip(paths,saved['cells'])),'differences':{key:{'count':len(items),'max_absolute_difference':max((i['absolute_difference'] for i in items),default=0),'largest_examples':sorted(items,key=lambda i:i['absolute_difference'],reverse=True)[:6]} for key,items in diffs.items()},'per_cell_residuals':percell}
print(json.dumps(report,indent=2))
