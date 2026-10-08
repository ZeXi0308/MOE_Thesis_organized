import argparse,bisect,collections,json,pathlib,statistics
parser=argparse.ArgumentParser();parser.add_argument("--input",type=pathlib.Path,required=True);parser.add_argument("--output",type=pathlib.Path,required=True);args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
p=args.input;raw=json.load(open(p/'raw.json'));obs=json.load(open(p/'recovery-order.json'));events=obs['events'];origin=raw['measurement_origin_perf_counter_s'];mapping=raw['internal_to_source'];times={r['request_id']:r['token_times_s'] for r in raw['requests']}
byj=collections.defaultdict(dict);byr=collections.defaultdict(list)
for e in events:
 if e.get('job_id') is not None:byj[e['job_id']][e['kind']]=e
 if e.get('request'):byr[e['request']].append(e)
def dist(xs):
 xs=sorted(xs)
 def q(x):
  v=(len(xs)-1)*x;lo=int(v);hi=min(lo+1,len(xs)-1);return xs[lo]+(v-lo)*(xs[hi]-xs[lo])
 return dict(n=len(xs),min=min(xs),median=q(.5),p95=q(.95),max=max(xs),sum=sum(xs)) if xs else None
loads={j:v for j,v in byj.items() if not v['job_created']['is_store']};rows=[]
for n,pre in enumerate(e for e in events if e['kind']=='preempt'):
 rid=pre['request'];t=pre['host_perf_s'];src=mapping[rid];ts=times[src];idx=bisect.bisect_right(ts,t-origin);nextout=ts[idx]+origin if idx<len(ts) else None
 candidates=[(j,v) for j,v in loads.items() if v['ready']['request']==rid and t<=v['ready']['host_perf_s']<=nextout]
 assert len(candidates)==1,(n,rid,candidates)
 jid,j=candidates[0];r=dict(n=n+1,request=rid,source=src,load=jid,preempt_s=t-origin,next_output_s=nextout-origin)
 for k in ('job_created','ready','submit_begin','submit_end','job_completed','ack_retired'):r[k+'_s']=j[k]['host_perf_s']-origin
 rel=[e for e in byr[rid] if t<=e['host_perf_s']<=nextout];r['related']=[{k:v for k,v in e.items() if k!='request'} for e in rel if e['kind'] in ('lookup','capacity_wait','allocation_ok','scheduled')]
 for kind in ('lookup','capacity_wait','allocation_ok','scheduled'):
  ss=[e['host_perf_s']-origin for e in rel if e['kind']==kind];r[kind+'_s']=ss[0] if ss else None
 r['bytes']=j['job_completed']['bytes'];r['gpu_elapsed_ms']=1000*j['job_completed']['gpu_elapsed_s']
 for name,a,b in [('preempt_to_lookup','preempt','lookup'),('lookup_to_ready','lookup','ready'),('preempt_to_ready','preempt','ready'),('ready_to_submit','ready','submit_begin'),('submit_to_hostdone','submit_begin','job_completed'),('hostdone_to_ack','job_completed','ack_retired'),('ack_to_scheduled','ack_retired','scheduled'),('scheduled_to_output','scheduled','next_output'),('total','preempt','next_output')]:r[name+'_ms']=1000*(r[b+'_s']-r[a+'_s']) if r[a+'_s'] is not None and r[b+'_s'] is not None else None
 rows.append(r)
summary={k:dist([r[k] for r in rows if r[k] is not None]) for k in rows[0] if k.endswith('_ms')}
print('SUMMARY',json.dumps(summary))
for n in [i for i in [1,4,20,21,39,47] if i<=len(rows)]:
 r=rows[n-1];print('EVENT',json.dumps({k:v for k,v in r.items() if k!='related'}))
print('TOP TOTAL',[(r['n'],r['load'],r['source'],r['total_ms'],r['preempt_to_ready_ms']) for r in sorted(rows,key=lambda r:-r['total_ms'])[:5]])
# All ready/load submits until each wait/reorder/start point remain in observed event order; measure concurrent unsubmitted LOADs.
pending=[];active=[];snapshots=[];load_batches=[]
for e in events:
 if e['kind']=='ready' and not e['is_store']:pending.append(e['job_id']);snapshots.append((e['host_perf_s']-origin,list(pending),list(active)))
 elif e['kind']=='submit_begin' and not e['is_store']:pending.remove(e['job_id']);active.append(e['job_id'])
 elif e['kind']=='job_completed' and not e['is_store']:active.remove(e['job_id'])
print('LOAD_READY_UNSUBMITTED_MAX',max(len(x[1]) for x in snapshots),'HOST_OBSERVED_INFLIGHT_MAX',max(len(x[2])+1 for x in snapshots))
print('LOAD_READY_MULTI',[(t,p,a) for t,p,a in snapshots if len(p)>1])
print('LOAD_READY_WHILE_INFLIGHT',[(t,p,a) for t,p,a in snapshots if a])
json.dump(dict(summary=summary,rows=rows,ready_snapshots=snapshots),args.output.open('x'),indent=2)
