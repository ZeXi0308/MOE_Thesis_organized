"""Additional output/release checks and existing same-state baseline transfer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from recovery_progress_model import simulate
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity'
p=o/'20260914_d6_action_branches_r01';out=o/'20260914_recovery_progress_model_r01'
load=lambda p:json.loads(p.read_text())
spec=importlib.util.spec_from_file_location('frozen_rotation',p/'pkg/absence_rotation.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
inp=load(out/'input.json');state,hist=inp['state'],inp['history']
checks=[];timings=[]
for action in ('least','most','defer','native','continuous_most'):
 start=time.perf_counter();prediction=simulate(state,hist,action,m.AbsenceRotation,m.RequestView);timings.append(dict(action=action,seconds=time.perf_counter()-start))
 if action in ('native','continuous_most'):
  role='native' if action=='native' else 'most_output'
  paths=[o/f'20260914_d6_strong_baselines_r01/readback/results/block{b}-d6-{role}/raw.json' for b in (0,1)]
 else:paths=[p/f'readback/results/block{b}-{action}/raw.json' for b in (0,1)]
 for path in paths:
  raw=load(path);actual={s['step']:s for s in raw['scheduler_steps']};memory={s['attempted_step']:s for s in raw['memory_trace']};calls={c['scheduler_step_start']:c for c in raw['engine_steps']}
  # Actual completed output events are validation labels, not simulator inputs.
  emitted={c['returned_s']:[] for c in raw['engine_steps']}
  for e in raw['output_events']:
   if e['chunk_size']>0:emitted[e['received_s']].append(e['external_request_id'])
  completions={q['external_request_id']:next(c['scheduler_step_start'] for c in raw['engine_steps'] if c['returned_s']==q['completion_s']) for q in raw['requests']}
  mismatches=[]
  for row in prediction['trace']:
   step=row['step'];s=actual.get(step)
   if s is None:mismatches.append(dict(step=step,kind='actual_ended'));break
   expected={f"measured/{r['request_id']}":dict(computed=r['scheduled_start_computed'],tokens=r['scheduled_tokens'],output=r['output_tokens_before']) for r in s['scheduled']}
   tests=dict(schedule=list(row['scheduled'].items())==list(expected.items()),
       free=row['free_after_schedule']==memory[step]['after']['pool']['free_blocks'],
       outputs=row['outputs']==emitted[calls[step]['returned_s']])
   if not all(tests.values()):mismatches.append(dict(step=step,checks=tests));break
  checks.append(dict(action=action,path=str(path),predicted_last_step=prediction['last_step'],actual_last_step=max(actual),
                     completion_steps_equal=prediction['completed']==completions,first_mismatch=mismatches[:1]))
result=dict(status='EXISTING_DATA_STRUCTURAL_VALIDATION',checks=checks,local_cpu_timings=timings,
 scope='Same old fixed-length workload and initial state; no new holdout, no timing model, no GPU or online overhead measurement. Real futures are validation labels only.')
(out/'transfer_validation.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'model_source.py').write_text(Path(__file__).with_name('recovery_progress_model.py').read_text())
for c in checks:print(c['action'],Path(c['path']).parent.name,c['predicted_last_step'],c['actual_last_step'],c['completion_steps_equal'],c['first_mismatch'])
print(timings)
