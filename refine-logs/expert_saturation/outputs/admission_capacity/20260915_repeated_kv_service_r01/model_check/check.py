import hashlib,json,runpy,sys
from pathlib import Path
root=Path('/Users/leandrozhao/Desktop/、++++++++/refine-logs/expert_saturation');o=root/'outputs/admission_capacity'
sys.path.insert(0,str(root/'experiments/admission_capacity'))
from absence_rotation import AbsenceRotation,RequestView
model=o/'20260915_repeated_staged_probe_r02/execution_weste_26862/prediction_check/model_skipped_guard.py'
simulate=runpy.run_path(str(model))['simulate'];i=json.loads((o/'20260914_repeated_staged_model_r01/input.json').read_text());rows={}
base=o/'20260915_repeated_kv_service_r01/execution_weste_26862/readback/results'
for arm in ('off','on'):
 path=base/f'diag-{arm}/raw.json';blob=path.read_bytes();r=json.loads(blob);initial=i['state'];before=r['memory_trace'][329]['before'];canonical=lambda rid:'measured/'+r['internal_to_source'][rid]
 assert before['pool']['free_blocks']==initial['free']
 assert [canonical(x) for x in before['running_ids']]==initial['running']
 assert before['waiting_count']==len(initial['waiting'])==1
 assert set(map(canonical,before['requests']))==set(initial['requests'])
 for rid,q in before['requests'].items():
  s=initial['requests'][canonical(rid)];assert (q['computed_tokens'],q['prompt_tokens'],q['output_tokens'],q['block_counts'])==(s['computed'],s['prompt'],s['output'],[len(s['blocks'])])
 pred=simulate(i['state'],i['history'],'staged_most_'+arm,AbsenceRotation,RequestView,restore_delay_steps=2);first=None
 for row in pred['trace']:
  n=row['step']
  if n>=len(r['scheduler_steps']):first=dict(step=n,reason='extra_model_step');break
  actual={'measured/'+q['request_id']:dict(computed=q['scheduled_start_computed'],tokens=q['scheduled_tokens'],output=q['output_tokens_before']) for q in r['scheduler_steps'][n]['scheduled']}
  checks=dict(schedule=list(row['scheduled'].items())==list(actual.items()),free=row['free_after_schedule']==r['memory_trace'][n]['after']['pool']['free_blocks'],outputs=len(row['outputs'])==r['engine_steps'][n]['new_output_tokens'])
  if not all(checks.values()):first=dict(step=n,checks=checks);break
 rows[arm]=dict(raw_sha256=hashlib.sha256(blob).hexdigest(),initial_state_matched=True,model_calls=pred['last_step']+1,actual_calls=len(r['engine_steps']),first_mismatch=first,matched_steps=len(pred['trace']) if first is None else first['step']-329)
out=o/'20260915_repeated_kv_service_r01/model_check';out.mkdir(exist_ok=False)
result=dict(status='NEW_EXECUTION_STATE_CHECK',model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),arms=rows,scope='New diagnostic executions; same requests and fixed lengths, not independent workload holdout. Guard correction fixed using prior r02. Delay2 unchanged, no observed completion labels supplied. Does not validate timing or quality. Source is measurement owners temporary readback; hashes allow canonical join.')
(out/'analysis.json').write_text(json.dumps(result,indent=2));(out/'check.py').write_text(Path(__file__).read_text());print(json.dumps(result,indent=2))
