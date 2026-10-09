"""Native vLLM output path, with engine execution replaced by timed batch indices.
CPU-only; no shared package modification. HTTP uses FastAPI/StreamingResponse.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_TARGET_DEVICE='cpu')
import argparse, asyncio, hashlib, importlib.metadata, json, resource, struct, time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=['native1','native2','native4','token4','bytes1024','class_static','lazy_static'],required=True);p.add_argument('--port',type=int,default=18761);p.add_argument('--core',type=int,default=2);p.add_argument('--profile',action='store_true');a=p.parse_args()
os.sched_setaffinity(0,{a.core})
from native_adapter import NativeAdapter
from fastapi import FastAPI
from starlette.responses import StreamingResponse
import uvicorn
from vllm.v1.engine.async_llm import AsyncLLM
from vllm import envs
from vllm.utils.gc_utils import freeze_gc_heap
trace=json.loads(a.trace.read_text());a.out.mkdir(exist_ok=True,parents=True)
adapter=NativeAdapter(trace['requests'])
from gate import install
materializer=install(adapter,a.arm)
batches=[adapter.build_batch(b) for b in trace['batches']]
app=FastAPI(); state={'connected':set(),'received':[],'cpu_start':None,'cpu_end':None};sockpath=str(a.out/'source.sock')
original=adapter.processor.process_outputs
stage={'calls':0,'cpu_ns':0,'wall_ns':0,'max_call_ns':0,'units':0,'max_units':0}
def measured(outputs,*args,**kwargs):
 t=time.perf_counter_ns();c=time.thread_time_ns();r=original(outputs,*args,**kwargs);dt=time.perf_counter_ns()-t
 stage['calls']+=1;stage['cpu_ns']+=time.thread_time_ns()-c;stage['wall_ns']+=dt;stage['max_call_ns']=max(stage['max_call_ns'],dt);stage['units']+=len(outputs)
 stage['max_units']=max(stage['max_units'],len(outputs))
 return r
adapter.processor.process_outputs=measured
# Call-level profiling is enabled only for the one diagnostic run.
profiler=None
if a.profile:
 import cProfile
 profiler=cProfile.Profile()
class Engine:
 async def get_output_async(self):
  index=struct.unpack('!I',await self.reader.readexactly(4))[0]
  if index==0xffffffff:
   state['cpu_end']=time.process_time();state['end_ns']=time.perf_counter_ns()
   if profiler: profiler.disable();profiler.dump_stats(str(a.out/'profile.pstats'))
   await asyncio.Future()
  state['received'].append([index,time.perf_counter_ns()])
  batch=batches[index]
  return batch
 async def abort_requests_async(self,r):
  assert not r, f'Unexpected stop detected: {r}'
engine=Engine()
async def input_connection(reader,writer):
 engine.reader=reader
 state['epoch_ns']=struct.unpack('!Q',await reader.readexactly(8))[0]
 state['cpu_start']=time.process_time()
 if profiler:profiler.enable()
 shim=SimpleNamespace(engine_core=engine,output_processor=adapter.processor,output_handler=None,log_stats=False,logger_manager=None,renderer=None)
 os.environ['VLLM_V1_OUTPUT_PROC_CHUNK_SIZE']='1'
 AsyncLLM._run_output_handler(shim)
 state['handler']=shim.output_handler
 try:await shim.output_handler
 finally:writer.close()
@app.on_event('startup')
async def start():
 if os.path.exists(sockpath):os.unlink(sockpath)
 state['socket']=await asyncio.start_unix_server(input_connection,path=sockpath)
 # Native API lifespan performs this after app initialization, before yield.
 # Omitting it makes replay import/prebuilt-data GC pauses look like HOL.
 freeze_gc_heap()
@app.get('/ready')
async def ready():return {'connected':len(state['connected']),'total':len(trace['requests']),'socket':sockpath}
@app.get('/stream/{rid}')
async def stream(rid:str):
 state['connected'].add(rid)
 return StreamingResponse(adapter.stream(rid),media_type='text/event-stream')
@app.get('/stats')
async def stats():
 now=time.process_time()
 data={'arm':a.arm,'cpu_seconds':now-state['cpu_start'] if state['cpu_start'] is not None else None,
 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'output_processor':stage,
 'source_received':state['received'],'ledger':adapter.ledger,'epoch_ns':state.get('epoch_ns'),
 'versions':{m:importlib.metadata.version(m) for m in ['vllm','torch','transformers','tokenizers','pydantic','fastapi','uvicorn']},
 'affinity':sorted(os.sched_getaffinity(0)),'native_default_chunk_size':envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE,
 'materialization':materializer.report()}
 (a.out/'server.json').write_text(json.dumps(data)+'\n');return data
if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=a.port,access_log=False,log_level='warning',loop='uvloop')
