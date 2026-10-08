import bisect,collections,json,pathlib,statistics
root=pathlib.Path('/Users/zhaozhenyu/Desktop/毕业设计/B_recovery_order_20261004/session-r03')
def dist(xs):
 xs=sorted(xs)
 def q(x):
  v=(len(xs)-1)*x;lo=int(v);hi=min(lo+1,len(xs)-1);return xs[lo]+(v-lo)*(xs[hi]-xs[lo])
 return dict(n=len(xs),min=min(xs),median=q(.5),p95=q(.95),max=max(xs),sum=sum(xs)) if xs else None
result={}
for cell in ['cell-00-native','cell-01-flush_first']:
 p=root/cell/'output';raw=json.load(open(p/'raw.json'));es=json.load(open(p/'recovery-order.json'))['events'];origin=raw['measurement_origin_perf_counter_s'];byj=collections.defaultdict(dict);byr=collections.defaultdict(list);counts=collections.defaultdict(collections.Counter)
 for e in es:
  if 'job_id' in e:byj[e['job_id']][e['kind']]=e;counts[e['kind']][e['job_id']]+=1
  if 'request' in e:byr[e['request']].append(e)
 phases=['job_created','ready','submit_begin','submit_end','job_completed','ack_retired'];expected=set(counts['job_created']);phase_checks={k:dict(unique_jobs=len(counts[k]),missing=sorted(expected-set(counts[k])),extra=sorted(set(counts[k])-expected),nonunit={str(j):n for j,n in counts[k].items() if n!=1}) for k in phases}
 reorder_checks=[];waits=[]
 for i,e in enumerate(es):
  if e['kind']=='reorder' and e['changed']:
   submits=[]
   for nxt in es[i+1:]:
    if nxt['kind'] in ('reorder','wait_begin'):break
    if nxt['kind']=='submit_begin':submits.append(nxt['job_id'])
   reorder_checks.append(dict(t=e['host_perf_s']-origin,before=e['before'],after=e['after'],required=e['required'],actual=submits,exact=submits==e['after'],same_multiset=collections.Counter(e['before'])==collections.Counter(e['after'])))
  if e['kind']=='wait_begin':
   nxt=next(v for v in es[i+1:] if v['kind']=='wait_end');waits.append(dict(t=e['host_perf_s']-origin,ms=1000*(nxt['host_perf_s']-e['host_perf_s']),required=e['required']))
 loads={j:v for j,v in byj.items() if not v['job_created']['is_store']};rows=[];mapping=raw['internal_to_source'];times={r['request_id']:r['token_times_s'] for r in raw['requests']}
 for pre in (e for e in es if e['kind']=='preempt'):
  rid=pre['request'];t=pre['host_perf_s'];src=mapping[rid];ts=times[src];idx=bisect.bisect_right(ts,t-origin);end=ts[idx]+origin
  candidates=[(j,v) for j,v in loads.items() if v['ready']['request']==rid and t<=v['ready']['host_perf_s']<=end];assert len(candidates)==1
  jid,j=candidates[0];rel=[e for e in byr[rid] if t<=e['host_perf_s']<=end];row=dict(request=src,jid=jid,preempt=t-origin,ready=j['ready']['host_perf_s']-origin,next_output=end-origin,capacity_wait=any(e['kind']=='capacity_wait' for e in rel))
  scheduled=next(e for e in rel if e['kind']=='scheduled');row.update(pre_ready_ms=1000*(j['ready']['host_perf_s']-t),ready_submit_ms=1000*(j['submit_begin']['host_perf_s']-j['ready']['host_perf_s']),submit_done_ms=1000*(j['job_completed']['host_perf_s']-j['submit_begin']['host_perf_s']),done_ack_ms=1000*(j['ack_retired']['host_perf_s']-j['job_completed']['host_perf_s']),ack_schedule_ms=1000*(scheduled['host_perf_s']-j['ack_retired']['host_perf_s']),schedule_output_ms=1000*(end-scheduled['host_perf_s']),total_ms=1000*(end-t));rows.append(row)
 rr=[e for e in es if e['kind']=='reorder'];out=dict(duration=raw['observation_end_s'],phase_checks=phase_checks,reorder_count=len(rr),required_count=sum(bool(e['required']) for e in rr),changed=len(reorder_checks),fallbacks=dict(collections.Counter(str(e['fallback']) for e in rr)),all_actual_orders_match=all(e['exact'] for e in reorder_checks),all_multisets_match=all(e['same_multiset'] for e in reorder_checks),reorder_checks=reorder_checks,wait_stats_ms=dist([w['ms'] for w in waits]),waits=waits,recovery_count=len(rows),capacity_wait_count=sum(r['capacity_wait'] for r in rows),recovery_stats={k:dist([r[k] for r in rows]) for k in rows[0] if k.endswith('_ms')},pre_ready_share_of_event_wait=sum(r['pre_ready_ms'] for r in rows)/sum(r['total_ms'] for r in rows),rows=rows,submit_accepted=all(byj[j]['submit_end']['accepted'] for j in expected))
 print(cell,json.dumps({k:v for k,v in out.items() if k not in ('rows','waits','reorder_checks')}));print('topwaits',sorted(waits,key=lambda w:-w['ms'])[:4]);print('first_actions',reorder_checks[:3]);result[cell]=out
json.dump(result,open('/private/tmp/b_trace_probe/flush_action_check.json','w'),indent=2)
