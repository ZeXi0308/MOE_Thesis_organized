"""Stop proactive swaps permanently from one baseline event; model diagnostic."""
import importlib.util,json,sys
from pathlib import Path
from recovery_progress_model import simulate
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity'
events=o/'20260914_recovery_later_victim_r01'
p=o/'20260914_recovery_stop_r01';source=o/'20260914_recovery_progress_model_r01'
load=lambda p:json.loads(p.read_text());inp=load(source/'input.json');b=load(source/'cost_baseline.json')['beta_seconds']
spec=importlib.util.spec_from_file_location('frozen_rotation',o/'20260914_d6_action_branches_r01/pkg/absence_rotation.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def run(**kw):return simulate(inp['state'],inp['history'],'least',m.AbsenceRotation,m.RequestView,**kw)
def metrics(r):
 assert r['status']=='COMPLETE' and len(r['completed'])==32
 clock=0.;done=[];last={rid:0. for rid in inp['state']['requests']};gap=0.
 for s in r['trace']:
  clock+=b[0]+b[1]*len(s['scheduled'])+b[2]*sum(v['tokens'] for v in s['scheduled'].values())
  for rid in s['outputs']:gap=max(gap,clock-last[rid]);last[rid]=clock
  done.extend([clock]*len(s['completed']))
 return dict(mean=sum(done)/len(done),last=clock,gap=gap,calls=len(r['trace']))
all_rows={}
for point,config in load(source/'pressure_inputs.json').items():
 if point=='d0':continue
 inp=config
 def run(**kw):return simulate(inp['state'],inp['history'],'least',m.AbsenceRotation,m.RequestView,start_step=inp['cutoff'],**kw)
 decisions=[];base=run(decision_log=decisions);bm=metrics(base);rows=[]
 native=simulate(inp['state'],inp['history'],'native',m.AbsenceRotation,m.RequestView,start_step=inp['cutoff'])
 for d in decisions:
  step=d['step'];r=run(stop_step=step);idx=step-inp['cutoff']
  assert r['trace'][:idx]==base['trace'][:idx] and all(t['forced'] is None for t in r['trace'][idx:])
  if step==inp['cutoff']:assert r==native
  mm=metrics(r);rows.append(dict(step=step,metrics=mm,delta_pct={k:100*(mm[k]/bm[k]-1) for k in ['mean','last','gap']}))
 all_rows[point]=dict(baseline=bm,native=metrics(native),rows=rows)
 print(point,'baseline',bm,'events',len(rows),'both improve',sum(r['delta_pct']['mean']<0 and r['delta_pct']['gap']<=0 for r in rows))
 print('last stop',rows[-1])
(p/'pressure_surface.json').write_text(json.dumps(dict(status='MODEL_ONLY_RETROSPECTIVE',points=all_rows),indent=2)+'\n')
