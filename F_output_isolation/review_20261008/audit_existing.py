"""Existing strong-baseline event decomposition; no new run or simulated benefit."""
import hashlib,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def q(xs,p):
 xs=sorted(xs);z=(len(xs)-1)*p;i=int(z);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(z-i)
def describe(xs):return {'count':len(xs),'p50':q(xs,.5),'p99':q(xs,.99),'max':max(xs)}
report={'evidence':'Existing runs are exploratory; runs, not events, are independent units','runs':{},'sha256':{}}
for load in ['medium','high']:
 tr=json.loads((R/'inputs'/f'{load}.json').read_text());req={x['request_id']:x for x in tr['requests']};index={}
 for bi,b in enumerate(tr['batches']):
  h=0
  for rank,x in enumerate(b['outputs']):
   index[(x['request_id'],x['position'])]=(bi,rank,h,len(b['outputs']));h+=req[x['request_id']]['heavy']
 for name in ['01_fixed1','06_fixed1']:
  folder=R/f'formal_{load}'/name
  s=json.loads((folder/'server.json').read_text());c=json.loads((folder/'client.json').read_text());prod=json.loads((folder/'producer.json').read_text())
  recv=dict(s['source_received']);em=[x[2] for x in prod['events']];event={};light=[];gaps=[];readygaps=[]
  for rid,spec in req.items():
   if not spec['heavy']:
    readygaps.extend((b-a)*1000 for a,b in zip(spec['token_times_s'],spec['token_times_s'][1:]));v=c[rid]['visible_ns'];gaps.extend((b-a)/1e6 for a,b in zip(v,v[1:]))
   pos=0
   for record,v in zip(s['ledger'][rid],c[rid]['visible_ns']):
    for j in range(record['token_count']):
     bi,rank,h,n=index[(rid,pos+j)];t=record['consumed_ns'];parts=[(recv[bi]-em[bi])/1e6,(t-recv[bi])/1e6,(v-t)/1e6]
     event[(rid,pos+j)]=(t,parts,(v-em[bi])/1e6)
     if not spec['heavy']:light.append((parts,(v-em[bi])/1e6))
    pos+=record['token_count']
   assert pos==len(spec['output_token_ids'])
  adjacency={k:[] for k in ['00','01','10','11']};spans=[];fraction=[];first=[]
  for bi,b in enumerate(tr['batches']):
   items=b['outputs'];times=[event[(x['request_id'],x['position'])][0] for x in items]
   first.append((times[0]-recv[bi])/1e6);spans.append((times[-1]-times[0])/1e6)
   if bi+1<len(tr['batches']):fraction.append((times[-1]-recv[bi])/1e9/(tr['batches'][bi+1]['time_s']-b['time_s']))
   for x,y,t,u in zip(items,items[1:],times,times[1:]):
    key=str(int(req[x['request_id']]['heavy']))+str(int(req[y['request_id']]['heavy']))
    adjacency[key].append((u-t)/1000)
  threshold=q([x[1] for x in light],.99);tail=[x for x in light if x[1]>=threshold]
  report['runs'][f'{load}/{name}']={'light_delivery_ms':describe([x[1] for x in light]),'segment_p99_ms':[q([x[0][i] for x in light],.99) for i in range(3)],'tail_conditional_mean_segments_ms':[statistics.mean(x[0][i] for x in tail) for i in range(3)],'segment_order':['emit_to_receive','receive_to_collector','collector_to_visible'],'engine_gap_ms':describe(readygaps),'client_gap_ms':describe(gaps),'adjacent_collector_us_by_previous_current_heavy':{k:describe(v) for k,v in adjacency.items()},'batch_collector_span_ms':describe(spans),'drain_to_next_interval_fraction':describe(fraction),'first_collector_delay_ms':describe(first),'first_batch_delay_ms':first[0],'note':'Adjacent deltas include previous chat/serialization plus next output processing and scheduling; NOT request CPU. No P99 sums, no oracle claims.'}
  for p in [folder/'server.json',folder/'client.json',folder/'producer.json',R/'inputs'/f'{load}.json']:report['sha256'][str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
out=Path(__file__).with_name('existing_evidence.json');out.write_text(json.dumps(report,indent=2)+'\n');print(out)
