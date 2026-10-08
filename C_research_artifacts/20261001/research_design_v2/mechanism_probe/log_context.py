#!/usr/bin/env python3
"""Existing-log context only; does not alter any policy or performance record."""
import bisect,json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
root=P.parents[1]
traces={}
for line in (P/'state_comparison.jsonl').read_text().splitlines():
 r=json.loads(line);traces.setdefault(r['cell'],{})[r['call']]=r
result={}
for label in ['retirement_1','retirement_2']:
 cell=root/'c-native-retirement-fresh-v1'/label/'native_retirement'
 raw=json.loads((cell/'raw.json').read_text());g=json.loads((cell/'retirement-envelope.json').read_text())
 ss=raw['scheduler_steps'];gs=g['steps'];mixed=[];age=[];prefix_ok=0;protected_ok=0;protected=set();waiting_dt=mixed_dt=0
 internal={r['internal_request_id']:r['request_id'] for r in raw['requests']}
 reqs={r['request_id']:r for r in raw['requests']}
 first={}
 for i,s in enumerate(ss):
  for r in s['scheduled']:first.setdefault(r['request_id'],i)
 eligible=[i for i,x in enumerate(gs) if x['eligible_all_decode']]
 for i,(s,x,mem) in enumerate(zip(ss,gs,raw['memory_trace'])):
  rids=[internal[k] for k in mem['before']['running_ids']];protected.intersection_update(rids)
  if x['eligible_all_decode']:protected=set(rids)
  dt=(ss[i+1]['start_s'] if i+1<len(ss) else raw['observation_end_s'])-s['start_s']
  if x['waiting_before']:waiting_dt+=dt
  record=traces[label][i+1]
  if record.get('mixed_head_fits'):
   mixed_dt+=dt
   row_by_id={r['request_id']:r for r in s['scheduled']}
   D=[k for k in rids if row_by_id[k]['output_tokens_before']>0 and row_by_id[k]['computed_before']==reqs[k]['prompt_tokens']+row_by_id[k]['output_tokens_before']-1]
   prefix_ok+=rids[:len(D)]==D;protected_ok+=protected.issubset(D)
   nxt=eligible[bisect.bisect_right(eligible,i)]
   mixed.append(ss[nxt]['start_s']-s['start_s'])
   waiting=sorted((r for r in reqs.values() if r['engine_add_return_s']<=s['start_s'] and first[r['request_id']]>=i),key=lambda r:r['engine_add_return_s'])
   age.append(s['start_s']-waiting[0]['arrival_s'])
 def pct(xs,q):
  v=sorted(xs);idx=(len(v)-1)*q;a=int(idx);b=min(a+1,len(v)-1);return v[a]*(b-idx)+v[b]*(idx-a) if a!=b else v[a]
 d=[x['decision_s'] for x in gs]
 result[label]=dict(mixed_fit_D_prefix_calls=prefix_ok,mixed_fit_prior_obligations_retained_calls=protected_ok,
   waiting_schedule_interval_seconds=waiting_dt,mixed_fit_schedule_interval_seconds=mixed_dt,
   mixed_fit_fraction_of_waiting_schedule_intervals=mixed_dt/waiting_dt,
   next_observed_all_decode_delay_median_s=statistics.median(mixed),next_observed_all_decode_delay_max_s=max(mixed),
   mixed_head_observed_age_median_s=statistics.median(age),mixed_head_observed_age_max_s=max(age),
   decision_pre_schedule_p50_s=pct(d,.5),decision_pre_schedule_p95_s=pct(d,.95),decision_pre_schedule_p99_s=pct(d,.99),
   decision_pre_schedule_total_s=sum(d),episode_s=raw['observation_end_s'])
with (P/'log_context.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
