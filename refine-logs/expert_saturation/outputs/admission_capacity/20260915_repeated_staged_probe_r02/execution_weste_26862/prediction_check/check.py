import sys,json,runpy,hashlib
from pathlib import Path
root=Path('/Users/leandrozhao/Desktop/、++++++++')
e=root/'refine-logs/expert_saturation/experiments/admission_capacity'
o=root/'refine-logs/expert_saturation/outputs/admission_capacity'
sys.path.insert(0,str(e))
from absence_rotation import AbsenceRotation,RequestView
frozen=o/'20260914_repeated_staged_model_r01'
simulate=runpy.run_path(str(frozen/'model_source.py'))['simulate']
inp=json.loads((frozen/'input.json').read_text())
summary=json.loads((frozen/'summary_corrected.json').read_text())
base=o/'20260915_repeated_staged_probe_r02/execution_weste_26862'
analysis=json.loads((base/'analysis.json').read_text())
rows={}
for arm in ('off','on'):
 raw=json.loads((base/f'readback/results/save-{arm}/raw.json').read_text())
 canonical=lambda rid:'measured/'+raw['internal_to_source'][rid]
 before=raw['memory_trace'][329]['before']; initial=inp['state']
 assert before['pool']['free_blocks']==initial['free']
 assert [canonical(r) for r in before['running_ids']]==initial['running']
 assert before['waiting_count']==len(initial['waiting'])==1
 assert set(map(canonical,before['requests']))-set(initial['running'])==set(initial['waiting'])
 assert set(map(canonical,before['requests']))==set(initial['requests'])
 for rid,r in before['requests'].items():
  q=initial['requests'][canonical(rid)]
  assert (r['computed_tokens'],r['prompt_tokens'],r['output_tokens'],r['block_counts'])==(q['computed'],q['prompt'],q['output'],[len(q['blocks'])])
 for delay in ([2] if arm=='off' else [1,2,4,8]):
  name='staged_off' if arm=='off' else f'staged_on_delay{delay}'
  pred=simulate(inp['state'],inp['history'],'staged_most_'+arm,AbsenceRotation,RequestView,restore_delay_steps=delay)
  frozen_row=summary['rows'][name]
  assert pred['last_step']+1==frozen_row['total_calls']
  assert pred.get('stored_tokens',0)==frozen_row['stored_tokens']
  first=None
  for row in pred['trace']:
   n=row['step']
   if n>=len(raw['scheduler_steps']):first={'step':n,'reason':'extra_model_step'};break
   actual={'measured/'+q['request_id']:dict(computed=q['scheduled_start_computed'],tokens=q['scheduled_tokens'],output=q['output_tokens_before']) for q in raw['scheduler_steps'][n]['scheduled']}
   checks=dict(schedule=list(row['scheduled'].items())==list(actual.items()),free=row['free_after_schedule']==raw['memory_trace'][n]['after']['pool']['free_blocks'],outputs=len(row['outputs'])==raw['engine_steps'][n]['new_output_tokens'])
   if not all(checks.values()):
    first=dict(step=n,checks=checks,predicted_schedule=row['scheduled'],actual_schedule=actual,predicted_free=row['free_after_schedule'],actual_free=raw['memory_trace'][n]['after']['pool']['free_blocks'],predicted_loads=row.get('loads',[]));break
  rows[name]=dict(predicted_calls=frozen_row['total_calls'],actual_calls=len(raw['engine_steps']),matched_prefix_steps=len(pred['trace']) if first is None else first['step']-329,first_mismatch=first,predicted_store_bytes=frozen_row['stored_tokens']*131072,actual_store_bytes=analysis['arms'][arm]['transfers_bytes']['store'],predicted_load_bytes=frozen_row['loaded_tokens']*131072,actual_load_bytes=analysis['arms'][arm]['transfers_bytes']['load'])
out=base/'prediction_check';out.mkdir(exist_ok=False)
result=dict(status='FROZEN_STRUCTURAL_PREDICTION_CHECK',initial_state_matched=True,model_sha256=hashlib.sha256((frozen/'model_source.py').read_bytes()).hexdigest(),rows=rows,scope='Frozen model and pre-action state; no calibration or future completion labels. Delay cases remain predeclared sensitivities. Aggregate closeness does not validate a mismatching trajectory. Performance table reused without recalculation.')
(out/'analysis.json').write_text(json.dumps(result,indent=2))
(out/'check.py').write_text(Path(__file__).read_text())
print(json.dumps({k:{n:v for n,v in r.items() if n!='first_mismatch'}|{'first_mismatch_step':r['first_mismatch']['step'] if r['first_mismatch'] else None} for k,r in rows.items()},indent=2))
