"""Paired CPU replay driver: one server CPU, one parsing client, one clock source.
Source timestamp and batch size unchanged. No GPU model or CUDA execution.
"""
import argparse, asyncio, hashlib, json, os, resource, socket, struct, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def producer(tracepath,sockpath,outpath,core):
 os.sched_setaffinity(0,{int(core)})
 d=json.loads(Path(tracepath).read_text());epoch=time.perf_counter_ns()+300_000_000
 s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect(sockpath);s.sendall(struct.pack('!Q',epoch));rows=[]
 for i,b in enumerate(d['batches']):
  target=epoch+int(b['time_s']*1e9)
  delay=(target-time.perf_counter_ns())/1e9
  if delay>0:time.sleep(delay)
  emitted=time.perf_counter_ns();s.sendall(struct.pack('!I',i));rows.append([i,target,emitted,time.perf_counter_ns()])
 s.sendall(struct.pack('!I',0xffffffff));Path(outpath).write_text(json.dumps({'epoch_ns':epoch,'events':rows,'affinity':sorted(os.sched_getaffinity(0))})+'\n');s.close()

def quantile(xs,q):
 if not xs:return None
 ys=sorted(xs);pos=(len(ys)-1)*q;i=int(pos);f=pos-i
 return ys[i]*(1-f)+ys[min(i+1,len(ys)-1)]*f

async def one(a,arm,index):
 import aiohttp
 d=json.loads(a.trace.read_text());out=a.output/f'{index:02d}_{arm}';out.mkdir(parents=True,exist_ok=False)
 log=(out/'server.log').open('w')
 cmd=[sys.executable,str(ROOT/'replay_server.py'),'--trace',str(a.trace),'--out',str(out),'--arm',arm,'--port',str(a.port),'--core',str(a.server_core)]
 if a.profile:cmd.append('--profile')
 server=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 client_start_cpu=None
 try:
  timeout=aiohttp.ClientTimeout(total=180)
  async with aiohttp.ClientSession(timeout=timeout,connector=aiohttp.TCPConnector(limit=0),read_bufsize=16*1024*1024) as session:
   base=f'http://127.0.0.1:{a.port}'
   for _ in range(1200):
    if server.poll() is not None:raise RuntimeError(f'Server failed: {out}/server.log')
    try:
     async with session.get(base+'/ready') as resp:ready=await resp.json()
     break
    except aiohttp.ClientError:await asyncio.sleep(.1)
   else:raise TimeoutError('server initialization')
   received={};seen_headers=set()
   async def consume(r):
    rid=r['request_id'];events=[];chunks=0;byte_count=0;done=False;objects=[];usage=None;finish=None;start=time.perf_counter_ns()
    async with session.get(base+'/stream/'+rid) as resp:
     assert resp.status==200;seen_headers.add(rid)
     async for line in resp.content:
      byte_count+=len(line)
      if not line.startswith(b'data: '):continue
      payload=line[6:].strip()
      if payload==b'[DONE]':done=True;break
      obj=json.loads(payload)
      parsed_ns=time.perf_counter_ns()
      if 'error' in obj:raise RuntimeError(obj)
      if obj.get('choices')==[]:
       usage=obj['usage'];continue
      assert len(obj.get('choices',[]))==1,obj
      ch=obj['choices'][0]
      if 'role' in ch['delta']:continue
      objects.append(payload)
      if ch.get('finish_reason') is not None:finish=ch['finish_reason']
      chunks+=1;events.append(parsed_ns)
     finished=time.perf_counter_ns()
    assert done,f'{rid} missing DONE'
    assert finish=='length' and usage['completion_tokens']==len(r['output_token_ids'])
    received[rid]={'visible_ns':events,'complete_ns':finished,'http_start_ns':start,'bytes':byte_count,'chunks':chunks,'objects':objects,'usage':usage,'finish_reason':finish}
   tasks=[asyncio.create_task(consume(r)) for r in d['requests']]
   for _ in range(500):
    if any(t.done() and t.exception() for t in tasks):await asyncio.gather(*tasks)
    if len(seen_headers)==len(tasks):break
    await asyncio.sleep(.01)
   else:raise TimeoutError('stream connections barrier')
   client_start_cpu=time.process_time()
   prod=subprocess.Popen([sys.executable,str(ROOT/'run_replay.py'),'producer',str(a.trace),str(out/'source.sock'),str(out/'producer.json'),str(a.producer_core)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   await asyncio.wait_for(asyncio.gather(*tasks),timeout=d['elapsed_s']+60)
   client_cpu=time.process_time()-client_start_cpu
   assert prod.wait(timeout=10)==0,prod.communicate()
   async with session.get(base+'/stats') as resp:stats=await resp.json()
   # Full semantic validation is deliberately outside every timed stream.
   for v in received.values():
    content=[];digest=hashlib.sha256();lp_count=0
    for payload in v.pop('objects'):
     ch=json.loads(payload)['choices'][0]
     content.append(ch['delta'].get('content') or '')
     if ch.get('logprobs'):
      for token in ch['logprobs']['content']:
       digest.update(json.dumps(token,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode());digest.update(b'\n');lp_count+=1
    v.update(text_sha256=hashlib.sha256(''.join(content).encode()).hexdigest(),logprobs_sha256=digest.hexdigest(),logprobs_count=lp_count)
   source=json.loads((out/'producer.json').read_text());epoch=source['epoch_ns']
   emitted={b['time_s']:source['events'][i][2] for i,b in enumerate(d['batches'])}
   delays={'light':[],'heavy':[]};planned_delays={'light':[],'heavy':[]};gaps={'light':[],'heavy':[]};complete={'light':[],'heavy':[]};post_engine={'light':[],'heavy':[]};sema={}
   for r in d['requests']:
    rid=r['request_id'];v=received[rid];ledger=stats['ledger'][rid];kind='heavy' if r['heavy'] else 'light'
    counts=[x['token_count'] if isinstance(x,dict) else x for x in ledger]
    assert len(counts)==len(v['visible_ns']),(rid,len(counts),len(v['visible_ns']))
    assert sum(counts)==len(r['output_token_ids']),(rid,sum(counts),len(r['output_token_ids']))
    if r['heavy']:assert v['logprobs_count']==len(r['output_token_ids'])
    cursor=0
    for count,ns in zip(counts,v['visible_ns']):
     for t in r['token_times_s'][cursor:cursor+count]:
      delays[kind].append((ns-emitted[t])/1e6);planned_delays[kind].append((ns-epoch-int(t*1e9))/1e6)
     cursor+=count
    gaps[kind].extend((y-x)/1e6 for x,y in zip(v['visible_ns'],v['visible_ns'][1:]))
    complete[kind].append((v['complete_ns']-epoch)/1e9-r['arrival_s'])
    post_engine[kind].append((v['complete_ns']-emitted[r['token_times_s'][-1]])/1e6)
    sema[rid]={k:v[k] for k in ('text_sha256','logprobs_sha256','logprobs_count')}
   elapsed=(max(x['complete_ns'] for x in received.values())-epoch)/1e9
   result={'arm':arm,'trace':d['name'],'requests':len(received),'tokens':sum(len(r['output_token_ids']) for r in d['requests']),'elapsed_s':elapsed,'req_s':len(received)/elapsed,'tokens_s':sum(len(r['output_token_ids']) for r in d['requests'])/elapsed,'all_completed':len(received)==len(d['requests']),'server_cpu_s':stats['cpu_seconds'],'client_cpu_s':client_cpu,'server_cpu_pct':100*stats['cpu_seconds']/elapsed,'client_cpu_pct':100*client_cpu/elapsed,'server_peak_rss_mib':stats['peak_rss_kib']/1024,'client_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'bytes':sum(r['bytes'] for r in received.values()),'semantics':sema,'output_processor':stats['output_processor'],'producer_lateness_p99_ms':quantile([(x[2]-x[1])/1e6 for x in source['events']],.99),'profile':a.profile}
   for k in ('light','heavy'):
    for q in (.5,.95,.99):result[f'{k}_delivery_p{int(q*100)}_ms']=quantile(delays[k],q)
    result[f'{k}_planned_delivery_p99_ms']=quantile(planned_delays[k],.99)
    result[f'{k}_gap_p99_ms']=quantile(gaps[k],.99)
    result[f'{k}_completion_p99_s']=quantile(complete[k],.99)
    result[f'{k}_drain_max_ms']=max(post_engine[k])
   (out/'client.json').write_text(json.dumps(received)+'\n');(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
   print(json.dumps({k:v for k,v in result.items() if k not in ('semantics',)}),flush=True)
   return result
 finally:
  server.terminate()
  try:server.wait(timeout=10)
  except subprocess.TimeoutExpired:server.kill();server.wait()
  log.close()

async def main(a):
 os.sched_setaffinity(0,{a.client_core});a.output.mkdir(parents=True,exist_ok=False)
 (a.output/'command.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
 results=[]
 for i,arm in enumerate(a.arms.split(',')):
  r=await one(a,arm,i)
  if results:assert r['semantics']==results[0]['semantics'],'Full output content/logprob mismatch'
  results.append(r)
 (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='producer':producer(*sys.argv[2:]);sys.exit(0)
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--arms',default='native,fixed1,fixed8,fixed8,fixed1,native');p.add_argument('--port',type=int,default=18761);p.add_argument('--server-core',type=int,default=2);p.add_argument('--client-core',type=int,default=4);p.add_argument('--producer-core',type=int,default=6);p.add_argument('--profile',action='store_true');a=p.parse_args();a.trace=a.trace.resolve();a.output=a.output.resolve();asyncio.run(main(a))
