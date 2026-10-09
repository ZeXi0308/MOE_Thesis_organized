"""One short native-model top-20 capture, retaining actual output batch order."""
import argparse, hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))

PROMPTS=[
 'Explain why a computer needs both memory and storage. Give concrete examples.\nAnswer:',
 'A streaming server handles many users at once. Describe how its response path works.\nAnswer:',
 'Write a short Python function that merges two sorted lists, then explain its complexity.\nAnswer:',
 'Compare a hash table and a balanced search tree using examples.\nAnswer:',
 '请用中文解释计算机缓存为什么可以提高性能，并举例说明。\n回答：',
 'Continue this technical note: UTF-8 encodes symbols such as café, 中文, and 🙂. A byte stream',
 'List the steps needed to test a JSON encoder, including escaped strings and numbers.\nAnswer:',
 'A request produces probabilities for alternative tokens. Explain how a client might use them.\nAnswer:',
]

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 from bootstrap import apply
 repair=apply()
 import numpy as np, torch
 from vllm import SamplingParams
 from vllm.sampling_params import RequestOutputKind
 from vllm.engine.arg_utils import EngineArgs
 from vllm.v1.engine.llm_engine import LLMEngine
 from vllm.tokenizers import get_tokenizer
 torch.set_num_threads(8)
 model='/root/autodl-tmp/moe-research-20261002/model'
 tokenizer=get_tokenizer(model,trust_remote_code=False)
 work=[dict(request_id=f'real-{i:03d}',arrival_s=0.0 if i<32 else .25,
            heavy=i%4==0,max_tokens=96 if i%2==0 else 192,
            prompt_token_ids=tokenizer.encode(PROMPTS[i%8]),prompt_index=i%8)
       for i in range(64)]
 # Heavy and light both contain short/long caps; no cost-derived service priority.
 for r in work:r['max_tokens']=96 if (int(r['request_id'][-3:])//4)%2==0 else 192
 (a.output/'workload.json').write_text(json.dumps(work,ensure_ascii=False,indent=2)+'\n')
 kwargs=dict(model=model,dtype='bfloat16',seed=20261009,max_model_len=4096,
             max_num_seqs=64,max_num_batched_tokens=4096,enable_chunked_prefill=True,
             enable_prefix_caching=False,scheduling_policy='fcfs',async_scheduling=False,
             stream_interval=1,enforce_eager=False,gpu_memory_utilization=.8,
             enable_return_routed_experts=False)
 (a.output/'engine_args.json').write_text(json.dumps(kwargs,indent=2)+'\n')
 init=time.perf_counter(); engine=LLMEngine.from_engine_args(EngineArgs(**kwargs),enable_multiprocessing=False)
 initialized=time.perf_counter()-init
 # One native warmup, excluded from captured workload and saved separately.
 engine.add_request('warmup',{'prompt_token_ids':work[0]['prompt_token_ids']},
                    SamplingParams(temperature=0,max_tokens=4,logprobs=20))
 while engine.has_unfinished_requests():engine.step()
 rows={r['request_id']:dict(r,output_token_ids=[],token_times_s=[],finished=False) for r in work}
 batches=[]; arrays={};steps=[];overhead_ns=0
 original=engine.output_processor.process_outputs
 origin=time.perf_counter_ns(); epoch=time.time()
 def now():return (time.perf_counter_ns()-origin)/1e9
 def capture(outputs,*args,**kw):
  nonlocal overhead_ns
  t=time.perf_counter_ns(); items=[]; idx=len(batches)
  for out in outputs:
   state=engine.output_processor.request_states.get(out.request_id)
   if state is None:continue
   rid=state.external_req_id; r=rows[rid]; start=len(r['output_token_ids'])
   ids=list(out.new_token_ids); r['output_token_ids'].extend(ids)
   r['token_times_s'].extend([(t-origin)/1e9]*len(ids))
   item={'request_id':rid,'token_start':start,'token_end':start+len(ids),
         'new_token_ids':ids,'finished':out.finish_reason is not None,
         'finish_reason':str(out.finish_reason) if out.finish_reason is not None else None,
         'stop_reason':out.stop_reason}
   lp=out.new_logprobs
   if lp is not None:
    key=f'b{idx:04d}_{rid}'
    arrays[key+'_ids']=lp.logprob_token_ids.copy()
    arrays[key+'_values']=lp.logprobs.copy()
    arrays[key+'_ranks']=lp.sampled_token_ranks.copy()
    item['logprobs_key']=key
   items.append(item)
  batches.append({'time_s':(t-origin)/1e9,'outputs':items})
  overhead_ns+=time.perf_counter_ns()-t
  return original(outputs,*args,**kw)
 engine.output_processor.process_outputs=capture
 cursor=0
 try:
  while cursor<len(work) or engine.has_unfinished_requests():
   if now()>90:raise TimeoutError('capture exceeds 90 s')
   while cursor<len(work) and work[cursor]['arrival_s']<=now():
    r=work[cursor]
    params=SamplingParams(temperature=0,max_tokens=r['max_tokens'],logprobs=20 if r['heavy'] else None,
                          output_kind=RequestOutputKind.DELTA)
    engine.add_request(r['request_id'],{'prompt_token_ids':r['prompt_token_ids']},params,
                       arrival_time=epoch+r['arrival_s'])
    rows[r['request_id']]['admitted_s']=now();cursor+=1
   if not engine.has_unfinished_requests():time.sleep(.001);continue
   start=now(); out=engine.step();end=now();steps.append({'start_s':start,'end_s':end})
   for response in out:
    if response.finished:
     rows[response.request_id].update(finished=True,completed_s=end,
       finish_reason=response.outputs[0].finish_reason,stop_reason=response.outputs[0].stop_reason)
 finally:
  engine.output_processor.process_outputs=original
  trace={'name':'real_top20_20261009','elapsed_s':now(),'requests':list(rows.values()),
         'batches':batches,'steps':steps,'capture_cpu_wall_overhead_ns':overhead_ns,
         'note':'Observed native batch timing includes capture overhead; no original-order reconstruction.'}
  (a.output/'trace.json').write_text(json.dumps(trace,ensure_ascii=False)+'\n')
  np.savez(a.output/'candidates.npz',**arrays)
 assert all(r['finished'] for r in rows.values())
 meta={'versions':{x:importlib.metadata.version(x) for x in ['vllm','torch','numpy','transformers','tokenizers','pydantic']},
       'gpu':torch.cuda.get_device_name(),'initialization_s':initialized,'bootstrap':repair,
       'requests':len(rows),'tokens':sum(len(r['output_token_ids']) for r in rows.values()),
       'heavy_tokens':sum(len(r['output_token_ids']) for r in rows.values() if r['heavy']),
       'batches':len(batches),'capture_overhead_ms':overhead_ns/1e6,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (a.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
