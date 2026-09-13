"""Model-only first-victim tradeoff surface; does not execute GPU actions."""
import importlib.util
import json
from pathlib import Path
import sys
import time
from recovery_progress_model import simulate
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity';p=o/'20260914_recovery_progress_model_r01'
load=lambda p:json.loads(p.read_text());inp=load(p/'input.json');beta=load(p/'cost_baseline.json')['beta_seconds']
spec=importlib.util.spec_from_file_location('frozen_rotation',o/'20260914_d6_action_branches_r01/pkg/absence_rotation.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
rows=[]
for victim in [None]+inp['state']['running']:
 start=time.perf_counter();result=simulate(inp['state'],inp['history'],'least',m.AbsenceRotation,m.RequestView,first_victim=victim)
 clock=0.;done={};last={rid:0. for rid in inp['state']['requests']};gap=0.
 for step in result['trace']:
  clock+=beta[0]+beta[1]*len(step['scheduled'])+beta[2]*sum(r['tokens'] for r in step['scheduled'].values())
  for rid in step['outputs']:gap=max(gap,clock-last[rid]);last[rid]=clock
  for rid in step['completed']:done[rid]=clock
 rows.append(dict(victim=victim or 'baseline_least',predicted_mean_remaining_s=sum(done.values())/32,
    predicted_last_remaining_s=clock,predicted_max_post_action_silence_s=gap,last_step=result['last_step'],cpu_seconds=time.perf_counter()-start))
baseline=rows[0]
for r in rows:
 r['predicted_mean_delta_pct']=100*(r['predicted_mean_remaining_s']/baseline['predicted_mean_remaining_s']-1)
 r['predicted_silence_delta_pct']=100*(r['predicted_max_post_action_silence_s']/baseline['predicted_max_post_action_silence_s']-1)
 r['pareto_nondominated']=not any(q['predicted_mean_remaining_s']<=r['predicted_mean_remaining_s'] and q['predicted_max_post_action_silence_s']<=r['predicted_max_post_action_silence_s'] and (q['predicted_mean_remaining_s']<r['predicted_mean_remaining_s'] or q['predicted_max_post_action_silence_s']<r['predicted_max_post_action_silence_s']) for q in rows)
out=dict(status='MODEL_ONLY_UNVERIFIED_CANDIDATES',rows=rows,scope='Existing observed state, fixed lengths, native-calibrated cost. Explicit first victim then least. No GPU candidate validation, no Oracle, no statistical tail guarantee or online deployment claim.')
(p/'candidate_surface.json').write_text(json.dumps(out,indent=2)+'\n')
for r in sorted(rows,key=lambda r:r['predicted_mean_remaining_s'])[:7]:print(r)
print('baseline',baseline)
