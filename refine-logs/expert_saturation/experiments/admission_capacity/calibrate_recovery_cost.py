"""Three-coefficient resource cost baseline, calibrated only on native traces."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from recovery_progress_model import simulate
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity'
p=o/'20260914_recovery_progress_model_r01';load=lambda p:json.loads(p.read_text())
source=o/'20260914_d6_action_branches_r01/pkg/absence_rotation.py'
spec=importlib.util.spec_from_file_location('frozen_rotation',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
pressure=o/'20260913_pressure_sweep_r01/execution_review_westc_r02/readback/results'
train_paths=[pressure/f'block0-{d}-native/raw.json' for d in ('d2','d6')]
X=[];Y=[]
for path in train_paths:
 raw=load(path);steps={s['step']:s for s in raw['scheduler_steps']};calls=raw['engine_steps']
 for i,c in enumerate(calls):
  s=steps[c['scheduler_step_start']]
  if any(r['prefill_tokens'] for r in s['scheduled']):continue
  duration=(calls[i+1]['start_s'] if i+1<len(calls) else c['returned_s'])-c['start_s']
  X.append([1,len(s['scheduled']),sum(r['scheduled_tokens'] for r in s['scheduled'])]);Y.append(duration)
X=np.array(X,float);Y=np.array(Y,float)
# Exact active-set search for this three-variable nonnegative least squares fit.
best=None
for mask in range(1,8):
 active=[j for j in range(3) if mask & (1<<j)];beta=np.zeros(3)
 beta[active]=np.linalg.lstsq(X[:,active],Y,rcond=None)[0]
 if min(beta)<0:continue
 loss=float(np.sum((X@beta-Y)**2))
 if best is None or loss<best[0]:best=(loss,beta)
assert best is not None
beta=best[1]
inputs=load(p/'pressure_inputs.json');tests=[]
for point in ('d2','d4','d6'):
 inp=inputs[point]
 for action in ('native','least','continuous_most'):
  if action=='continuous_most' and point!='d6':continue
  prediction=simulate(inp['state'],inp['history'],action,m.AbsenceRotation,m.RequestView,start_step=inp['cutoff'])
  clock=0.;done={}
  for row in prediction['trace']:
   feature=np.array([1,len(row['scheduled']),sum(r['tokens'] for r in row['scheduled'].values())])
   clock+=float(feature@beta)
   for rid in row['completed']:done[rid]=clock
  assert len(done)==32
  for block in (0,1):
   if action=='continuous_most':path=o/f'20260914_d6_strong_baselines_r01/readback/results/block{block}-d6-most_output/raw.json'
   else:path=pressure/f'block{block}-{point}-{"native" if action=="native" else "rotate"}/raw.json'
   raw=load(path);cut=next(c['start_s'] for c in raw['engine_steps'] if c['scheduler_step_start']==inp['cutoff'])
   remaining=[r['completion_s']-cut for r in raw['requests']]
   mean=float(np.mean(list(done.values())));actual_mean=float(np.mean(remaining));last=max(remaining)
   tests.append(dict(point=point,action=action,block=block,in_calibration=path in train_paths,
       predicted_mean_remaining_s=mean,actual_mean_remaining_s=actual_mean,mean_error_pct=100*(mean/actual_mean-1),
       predicted_last_remaining_s=clock,actual_last_remaining_s=last,last_error_pct=100*(clock/last-1),source=str(path)))
result=dict(status='SIMPLE_COST_BASELINE_RETROSPECTIVE',features=['constant','scheduled_requests','scheduled_tokens'],
 beta_seconds=beta.tolist(),training_paths=[str(p) for p in train_paths],training_calls=len(Y),training_rmse_s=float(np.sqrt(best[0]/len(Y))),
 tests=tests,scope='Native block0 d2+d6 only fit; all finite call durations retained, including outliers. '
 'Same old workload, retrospective timing transfer, no new holdout or online gain. Cost includes per-call host gap. '
 'Final-call host-gap approximation is retained; no future test timing used in predictions.')
(p/'cost_baseline.json').write_text(json.dumps(result,indent=2)+'\n')
print('beta',beta,'train RMSE',result['training_rmse_s'])
for t in tests:print(t['point'],t['action'],t['block'],'train',t['in_calibration'],'mean_error',round(t['mean_error_pct'],2),'last_error',round(t['last_error_pct'],2))
