"""KV bytes and recompute alignment; no assumed bidirectional bandwidth."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity';p=o/'20260914_kv_roundtrip_feasibility_r01'
load=lambda p:json.loads(p.read_text());snap=load(o/'20260914_d6_action_state_r01/readback/results/repeat0/action-state.json')
bpt=sum(next(r['bytes']/r['tokens'] for r in layer['requests'].values() if r['tokens']) for layer in snap['layers'].values());assert bpt==131072
beta=load(o/'20260914_recovery_progress_model_r01/cost_baseline.json')['beta_seconds'][2];results=[]
for block in [0,1]:
 raw=load(o/f'20260914_d6_strong_baselines_r01/readback/results/block{block}-d6-least_progress/raw.json');steps=raw['scheduler_steps'];calls={c['scheduler_step_start']:c for c in raw['engine_steps']};rows=[];lifetime=[]
 for e in raw['preemption_events']:
  rid=e['victim_internal_request_id'];start=e['attempted_step'];state=e['victim_state'];positions=state['computed_tokens'];saved=int(positions*bpt)
  next_eviction=min([q['attempted_step'] for q in raw['preemption_events'] if q['victim_internal_request_id']==rid and q['attempted_step']>start]+[10**9])
  sched=[(s['step'],r) for s in steps if start<=s['step']<next_eviction for r in s['scheduled'] if r['internal_request_id']==rid]
  assert sched
  restored=next((st for st,r in sched if r['computed_after']>=state['prompt_tokens']+state['output_tokens']),None)
  assert restored is not None
  first=sched[0][0];window=calls[restored]['returned_s']-calls[first]['start_s']
  rows.append(dict(step=start,request=rid,computed=positions,valid_bytes=saved,padded_bytes=sum(state['block_counts'])*16*int(bpt),first_recovery_step=first,first_new_output_step=restored,mixed_recovery_window_s=window,token_cost_model_s=positions*beta))
  lifetime.extend([(start,1,saved),(restored,0,-saved)])
 host=peak=0
 for _,_,change in sorted(lifetime):host+=change;peak=max(peak,host)
 assert host==0
 results.append(dict(block=block,events=rows,total_oneway_valid_bytes=sum(r['valid_bytes'] for r in rows),observed_timeline_host_peak_bytes=peak))
result=dict(status='CPU_FEASIBILITY_ONLY',bytes_per_valid_token=bpt,model_equal_direction_break_even_GB_s=2*bpt/beta/1e9,results=results,scope='Counterfactual host occupancy follows observed recompute lifetimes and is only sizing. Mixed recovery windows include other requests. Cost coefficient is empirical, not removable physical tax. No D2H or native swap measurement.')
(p/'budget.json').write_text(json.dumps(result,indent=2)+'\n')
print('bpt',bpt,'model break-even GB/s',result['model_equal_direction_break_even_GB_s'])
for r in results:print('block',r['block'],'events',len(r['events']),'oneway GB',r['total_oneway_valid_bytes']/1e9,'peak GB',r['observed_timeline_host_peak_bytes']/1e9,'first',r['events'][0])
