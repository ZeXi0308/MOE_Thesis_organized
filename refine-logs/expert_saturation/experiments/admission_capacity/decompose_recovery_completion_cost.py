"""Exact completion-area accounting under the frozen additive cost model."""
import importlib.util,json,sys
from pathlib import Path
from recovery_progress_model import simulate
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity';p=o/'20260914_recovery_stop_r01';source=o/'20260914_recovery_progress_model_r01'
load=lambda p:json.loads(p.read_text());b=load(source/'cost_baseline.json')['beta_seconds']
spec=importlib.util.spec_from_file_location('frozen_rotation',o/'20260914_d6_action_branches_r01/pkg/absence_rotation.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
rows=[]
for point,inp in load(source/'pressure_inputs.json').items():
 if point=='d0':continue
 for action in ['native','least','continuous_most']:
  r=simulate(inp['state'],inp['history'],action,m.AbsenceRotation,m.RequestView,start_step=inp['cutoff']);assert r['status']=='COMPLETE'
  n=len(inp['state']['requests']);unfinished=n;clock=0.;done=[];first=None
  areas=dict(call_constant=0.,scheduled_requests=0.,new_output_tokens=0.,recompute_tokens=0.)
  tokens=outputs=0
  for s in r['trace']:
   t=sum(v['tokens'] for v in s['scheduled'].values());out=len(s['outputs']);tokens+=t;outputs+=out
   terms=dict(call_constant=b[0],scheduled_requests=b[1]*len(s['scheduled']),new_output_tokens=b[2]*out,recompute_tokens=b[2]*(t-out))
   for k,v in terms.items():areas[k]+=unfinished/n*v
   clock+=sum(terms.values())
   if s['completed'] and first is None:first=clock
   done.extend([clock]*len(s['completed']));unfinished-=len(s['completed'])
  mean=sum(done)/n;assert abs(sum(areas.values())-mean)<1e-9
  rows.append(dict(point=point,action=action,mean=mean,last=clock,first=first,calls=len(r['trace']),tokens=tokens,outputs=outputs,recomputed_positions=tokens-outputs,mean_cost_terms_s=areas))
(p/'completion_cost.json').write_text(json.dumps(dict(status='MODEL_COST_ACCOUNTING',rows=rows,scope='Mutually exclusive terms of fitted additive cost, weighted by unfinished requests; not measured physical stage decomposition or causal intervention savings.'),indent=2)+'\n')
for r in rows:print(r)
