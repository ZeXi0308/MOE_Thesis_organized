import json,sys,runpy
from pathlib import Path
from bisect import bisect_right
root=Path('/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation')
e=root/'experiments/admission_capacity';o=root/'outputs/admission_capacity';sys.path.insert(0,str(e))
from absence_rotation import AbsenceRotation,RequestView
f=o/'20260914_repeated_staged_model_r01';b=o/'20260915_repeated_staged_probe_r02/execution_weste_26862';r=json.loads((b/'readback/results/save-on/raw.json').read_text());t=json.loads((b/'readback/results/save-on/offload-events.json').read_text());i=json.loads((f/'input.json').read_text())
p=runpy.run_path(str(f/'model_source.py'))['simulate'](i['state'],i['history'],'staged_most_on',AbsenceRotation,RequestView,restore_delay_steps=2)
loads=[dict(step=s['step'],**x) for s in p['trace'] for x in s.get('loads',[]) if 1100<=s['step']<=1105]
starts=[s['host_start_perf_counter_s'] for s in r['memory_trace']]
obs=[]
for event in t['completed_jobs']:
 n=bisect_right(starts,event['time_s'])-1
 if 1100<=n<=1105:
  for j in event['jobs']:
   if not j['is_store']:obs.append(dict(completed_call=n,eligible_call=n+1,**j,canonical='measured/'+r['internal_to_source'][j['request']]))
sched={str(n):[q for q in r['scheduler_steps'][n]['scheduled'] if q['request_id'].endswith('0000507')] for n in range(1102,1106)}
a=json.loads((b/'readback/results/save-on/selective-store.json').read_text())
result=dict(model_actions=[x for x in p['staging_events'] if 1101<=x['step']<=1105],actual_actions=[x for x in a['events'] if 1101<=x.get('step',-1)<=1105],status='OBSERVED_FIRST_DIVERGENCE_LOCALIZATION',model_loads=loads,actual_completion_notifications=obs,actual_target_scheduled=sched,scope='Uses observed worker notifications only to explain mismatch; no future replay or predictive claim.')
(b/'prediction_check/ready_localization.json').write_text(json.dumps(result,indent=2))
(b/'prediction_check/localize.py').write_text(Path(__file__).read_text())
print(json.dumps(result,indent=2))
