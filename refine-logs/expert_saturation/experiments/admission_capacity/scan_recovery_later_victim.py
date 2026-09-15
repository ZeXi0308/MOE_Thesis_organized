"""Retrospective model-only single-event replacement; independent full futures."""
import importlib.util
import json
from pathlib import Path
import sys
import time
from recovery_progress_model import simulate
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity'
source=o/'20260914_recovery_progress_model_r01';dest=o/'20260914_recovery_later_victim_r01'
dest.mkdir(exist_ok=False)
load=lambda p:json.loads(p.read_text())
inp=load(source/'input.json');beta=load(source/'cost_baseline.json')['beta_seconds']
spec=importlib.util.spec_from_file_location('frozen_rotation',o/'20260914_d6_action_branches_r01/pkg/absence_rotation.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def run(**kw):return simulate(inp['state'],inp['history'],'least',m.AbsenceRotation,m.RequestView,**kw)
def metrics(result):
 assert result['status']=='COMPLETE' and len(result['completed'])==len(inp['state']['requests'])
 clock=0.;done={};last={rid:0. for rid in inp['state']['requests']};gap=0.
 for step in result['trace']:
  clock+=beta[0]+beta[1]*len(step['scheduled'])+beta[2]*sum(r['tokens'] for r in step['scheduled'].values())
  for rid in step['outputs']:gap=max(gap,clock-last[rid]);last[rid]=clock
  for rid in step['completed']:done[rid]=clock
 return dict(mean=sum(done.values())/len(done),last=clock,gap=gap,calls=len(result['trace']))
decisions=[];base=run(decision_log=decisions);bm=metrics(base)
old=load(source/'predictions.json')
print('old prediction keys',list(old),flush=True)
rows=[];started=time.perf_counter()
for d in decisions:
 step=d['step'];prefix=[r for r in base['trace'] if r['step']<step]
 for victim in d['fundable']:
  if victim==d['default_victim']:continue
  r=run(first_victim=victim,override_step=step)
  assert r['trace'][:len(prefix)]==prefix
  assert r['trace'][len(prefix)]['forced']==victim
  mm=metrics(r)
  rows.append(dict(step=step,victim=victim,default_victim=d['default_victim'],metrics=mm,
      delta_pct={k:100*(mm[k]/bm[k]-1) for k in ('mean','last','gap')}))
 print('step',step,'candidates',len(d['fundable'])-1,flush=True)
out=dict(status='MODEL_ONLY_EXPLORATORY',baseline=bm,decisions=decisions,rows=rows,
 elapsed_cpu_wall_s=time.perf_counter()-started,
 scope='Single victim replacement at one baseline event; all subsequent actions least. Same old fixed-length cohort. No GPU execution, no Oracle or deployment guarantee.')
(dest/'surface.json').write_text(json.dumps(out,indent=2)+'\n')
(dest/'model_source.py').write_text(Path(__file__).with_name('recovery_progress_model.py').read_text())
print('rows',len(rows),'baseline',bm,flush=True)
for r in sorted(rows,key=lambda r:r['metrics']['mean'])[:6]:print(r,flush=True)
