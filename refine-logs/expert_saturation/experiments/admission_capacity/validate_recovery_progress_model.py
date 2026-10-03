"""Export pre-action-only input, simulate, then compare with actual futures."""
import importlib.util
import json
from pathlib import Path
import sys
from recovery_progress_model import simulate

root=Path(__file__).resolve().parents[2];p=root/'outputs/admission_capacity/20260914_d6_action_branches_r01'
out=root/'outputs/admission_capacity/20260914_recovery_progress_model_r01'
out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('frozen_rotation',p/'pkg/absence_rotation.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
load=lambda f:json.loads(f.read_text())
initial=load(p/'pkg/reference_action_state.json')['state']
ds=load(p/'readback/results/block0-least/headroom-decisions.json')
canon=lambda rid:rid.rsplit('-',1)[0]
history=[dict(step=d['step'],preempted=[canon(r) for r in d.get('preempted',[])],resumed=[canon(r) for r in d.get('resumed',[])]) for d in ds if d['step']<329]
(out/'input.json').write_text(json.dumps(dict(state=initial,history=history),indent=2)+'\n')
# No future rows are passed to the simulator.
results={a:simulate(initial,history,a,mod.AbsenceRotation,mod.RequestView) for a in ('least','most','defer')}
(out/'predictions.json').write_text(json.dumps(results,indent=2)+'\n')
checks=[]
for action,pred in results.items():
 raw=load(p/f'readback/results/block0-{action}/raw.json')
 steps={s['step']:s for s in raw['scheduler_steps']};mem={s['attempted_step']:s for s in raw['memory_trace']}
 mismatches=[]
 for row in pred['trace']:
  actual=steps.get(row['step'])
  if actual is None:mismatches.append(dict(step=row['step'],reason='actual trace ended'));break
  target={f"measured/{r['request_id']}":dict(computed=r['scheduled_start_computed'],tokens=r['scheduled_tokens'],output=r['output_tokens_before']) for r in actual['scheduled']}
  if list(row['scheduled'].items())!=list(target.items()):
   mismatches.append(dict(step=row['step'],kind='schedule',predicted=row['scheduled'],actual=target));break
  free=mem[row['step']]['after']['pool']['free_blocks']
  if row['free_after_schedule']!=free:
   mismatches.append(dict(step=row['step'],kind='allocation',predicted=row['free_after_schedule'],actual=free));break
 checks.append(dict(action=action,predicted_last_step=pred['last_step'],actual_last_step=raw['scheduler_steps'][-1]['step'],
                    predicted_completed=len(pred['completed']),first_mismatch=mismatches[:1]))
(out/'validation.json').write_text(json.dumps(dict(status='CALIBRATION_ONLY',checks=checks),indent=2)+'\n')
for c in checks:print(c['action'],c['predicted_last_step'],c['actual_last_step'],str(c['first_mismatch'])[:1800])
