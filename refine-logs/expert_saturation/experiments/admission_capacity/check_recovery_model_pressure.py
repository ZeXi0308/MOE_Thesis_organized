"""Retrospective pressure transfer of the state-only resource model."""
import importlib.util
import json
from pathlib import Path
import sys
from recovery_progress_model import simulate
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity'
b=o/'20260913_pressure_sweep_r01/execution_review_westc_r02/readback/results'
p=o/'20260914_d6_action_branches_r01';out=o/'20260914_recovery_progress_model_r01'
load=lambda p:json.loads(p.read_text())
spec=importlib.util.spec_from_file_location('frozen_rotation',p/'pkg/absence_rotation.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
canon=lambda rid:rid.rsplit('-',1)[0]
def state_at(raw,cutoff):
 s=next(x for x in raw['memory_trace'] if x['attempted_step']==cutoff)['before'];running=s['running_ids']
 requests={canon(rid):dict(computed=r['computed_tokens'],prompt=r['prompt_tokens'],output=r['output_tokens'],
            max_tokens=1024,preemptions=r['num_preemptions'],status='RUNNING' if rid in running else 'PREEMPTED',
            blocks=list(range(r['block_counts'][0]))) for rid,r in s['requests'].items()}
 waiting=[canon(r) for r in s['requests'] if r not in running]
 assert len(waiting)<=1 and len(requests)==32
 return dict(requests=requests,running=[canon(r) for r in running],waiting=waiting,free=s['pool']['free_blocks'])
checks=[];inputs={}
for point in ('d0','d2','d4','d6'):
 folder=b/f'block0-{point}-rotate';decisions=load(folder/'headroom-decisions.json')
 cutoff=next((x['step'] for x in decisions if x.get('forced_preempted')),329)
 raw=load(folder/'raw.json');state=state_at(raw,cutoff)
 history=[dict(step=d['step'],preempted=[canon(r) for r in d.get('preempted',[])],resumed=[canon(r) for r in d.get('resumed',[])]) for d in decisions if d['step']<cutoff]
 inputs[point]=dict(state=state,history=history,cutoff=cutoff)
 predictions={action:simulate(state,history,action,m.AbsenceRotation,m.RequestView,start_step=cutoff) for action in ('least','native')}
 for action,pred in predictions.items():
  role='rotate' if action=='least' else 'native'
  for block in (0,1):
   label=f'block{block}-{point}-{role}';raw=load(b/label/'raw.json')
   actual_state=state_at(raw,cutoff);initial_equal=actual_state==state
   steps={s['step']:s for s in raw['scheduler_steps']};memory={s['attempted_step']:s for s in raw['memory_trace']}
   completion={q['external_request_id']:next(c['scheduler_step_start'] for c in raw['engine_steps'] if c['returned_s']==q['completion_s']) for q in raw['requests']}
   mismatch=None
   for row in pred['trace']:
    s=steps.get(row['step'])
    if s is None:mismatch=dict(step=row['step'],reason='actual_ended');break
    actual={f"measured/{r['request_id']}":dict(computed=r['scheduled_start_computed'],tokens=r['scheduled_tokens'],output=r['output_tokens_before']) for r in s['scheduled']}
    if list(row['scheduled'].items())!=list(actual.items()) or row['free_after_schedule']!=memory[row['step']]['after']['pool']['free_blocks']:
     mismatch=dict(step=row['step'],reason='schedule_or_allocation');break
   checks.append(dict(label=label,cutoff=cutoff,initial_state_equal=initial_equal,predicted_last_step=pred['last_step'],actual_last_step=max(steps),
                       completion_steps_equal=pred['completed']==completion,first_mismatch=mismatch))
(out/'pressure_inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
(out/'pressure_validation.json').write_text(json.dumps(dict(status='RETROSPECTIVE_STRUCTURAL_TRANSFER',checks=checks,
    scope='Existing same fixed-length cohort under four initialized KV capacities. No GPU loading feasibility, new holdout, timing, EOS or online gain claim.'),indent=2)+'\n')
for c in checks:print(c)
