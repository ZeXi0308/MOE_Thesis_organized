"""Observed completion area by nonoverlapping call interval class, not causal tax."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity';p=o/'20260914_recovery_stop_r01'
load=lambda p:json.loads(p.read_text());inputs=load(o/'20260914_recovery_progress_model_r01/pressure_inputs.json');rows=[]
for point in ['d2','d4','d6']:
 for block in [0,1]:
  for action in ['native','rotate']:
   path=o/f'20260913_pressure_sweep_r01/execution_review_westc_r02/readback/results/block{block}-{point}-{action}/raw.json'
   raw=load(path);steps={s['step']:s for s in raw['scheduler_steps']};calls=raw['engine_steps'];completion=[r['completion_s'] for r in raw['requests']]
   cut=next(c['start_s'] for c in calls if c['scheduler_step_start']==inputs[point]['cutoff'])
   area=dict(with_recompute=0.,without_recompute=0.);counts=dict(with_recompute=0,without_recompute=0);recomputed=0
   for i,c in enumerate(calls):
    if c['start_s']<cut:continue
    end=calls[i+1]['start_s'] if i+1<len(calls) else max(completion)
    rec=sum(r['recompute_tokens'] for r in steps[c['scheduler_step_start']]['scheduled']);recomputed+=rec
    key='with_recompute' if rec else 'without_recompute';counts[key]+=1
    area[key]+=sum(max(0.,min(end,t)-c['start_s']) for t in completion)/len(completion)
   mean=sum(t-cut for t in completion)/len(completion)
   assert abs(sum(area.values())-mean)<1e-8
   rows.append(dict(point=point,block=block,action=action,mean=mean,completion_area_s=area,call_counts=counts,recomputed_positions=recomputed,source=str(path)))
(p/'observed_completion_area.json').write_text(json.dumps(dict(status='OBSERVED_ACCOUNTING_ONLY',rows=rows,scope='Start-to-next-start intervals include host gaps and co-scheduled productive work. With-recompute bucket is not pure recompute cost or removable causal saving; all request completion areas exactly conserved.'),indent=2)+'\n')
for point in ['d2','d4','d6']:
 for block in [0,1]:
  n=next(r for r in rows if r['point']==point and r['block']==block and r['action']=='native');r=next(r for r in rows if r['point']==point and r['block']==block and r['action']=='rotate')
  print(point,block,'mean delta',r['mean']-n['mean'],'with-recompute delta',r['completion_area_s']['with_recompute']-n['completion_area_s']['with_recompute'],'other delta',r['completion_area_s']['without_recompute']-n['completion_area_s']['without_recompute'],'recompute',n['recomputed_positions'],r['recomputed_positions'])
