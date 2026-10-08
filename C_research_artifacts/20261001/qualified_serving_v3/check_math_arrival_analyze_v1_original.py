from pathlib import Path
from types import SimpleNamespace as NS
import collections,copy,hashlib,json,sys,tempfile
P=Path('/Users/zhaozhenyu/Desktop/毕业设计/C_research_artifacts/20261001/qualified_serving_v3')
sys.path.insert(0,str(P));sys.dont_write_bytecode=True
import math_arrival_analyze_v1 as analyze
import compare_math_arrival_runs_v1 as compare
import arrival_generate_v1 as generate
from check_arrival_generate_v1 import Clock,Params
from native_pressure_observer_v1 import observe_native_pressure
class Queue(list):
 def peek_request(self):return self[0]
class Engine:
 def __init__(self,clock):
  self.clock=clock;self.active={}
  pool=NS(num_gpu_blocks=101,get_num_free_blocks=lambda:100)
  self.s=NS(requests={},current_step=32,running=[],waiting=Queue(),skipped_waiting=Queue(),
    max_model_len=4096,scheduler_reserve_full_isl=True,num_waiting_for_streaming_input=0,
    max_num_running_reqs=1024,max_num_scheduled_tokens=4096,_pause_state='UNPAUSED')
  def allocate(request,num_new_tokens,num_new_computed_tokens=0,new_computed_blocks=None,
      num_lookahead_tokens=0,num_external_computed_tokens=0,delay_cache_blocks=False,num_encoder_tokens=0,
      full_sequence_must_fit=False,reserved_blocks=0,has_scheduled_reqs=False):return []
  self.s.kv_cache_manager=NS(block_pool=pool,watermark_blocks=0,allocate_slots=allocate)
  def schedule():
   self.s.current_step+=1;scheduled={}
   for r in list(self.s.requests.values()):
    self.s.kv_cache_manager.allocate_slots(r,1)
    if r.status=='WAITING':self.s.waiting.remove(r);self.s.running.append(r);r.status='RUNNING'
    scheduled[r.request_id]=1
   return NS(num_scheduled_tokens=scheduled,preempted_req_ids=[])
  self.s.schedule=schedule;self.engine_core=NS(engine_core=NS(scheduler=self.s))
 def add_request(self,rid,prompt,sampling,arrival_time):
  index=int(rid.rsplit('-',1)[1]);offered=0. if index<512 else 4.
  assert self.clock.t-self.clock.origin>=offered and arrival_time==self.clock.epoch+offered
  native=rid+'-abcdef01';r=NS(request_id=native,sampling_params=sampling,status='WAITING',
                            num_tokens=1,num_computed_tokens=0,num_prompt_tokens=1)
  self.s.requests[native]=r;self.s.waiting.append(r);self.active[rid]=(native,0)
  self.clock.t+=.001
 def has_unfinished_requests(self):return bool(self.active)
 def step(self):
  self.s.schedule();self.clock.t+=2.5;out=[]
  for rid,(native,n) in list(self.active.items()):
   n+=1;done=n==3;r=self.s.requests[native];r.num_computed_tokens+=1;r.num_tokens+=1
   completion=NS(token_ids=[10,11,12][:n],text='fixture 0',finish_reason='stop' if done else None,stop_reason=None)
   out.append(NS(request_id=rid,outputs=[completion],finished=done))
   if done:del self.active[rid];del self.s.requests[native];self.s.running.remove(r)
   else:self.active[rid]=(native,n)
  return out
old_time=generate.time;saved={k:sys.modules.get(k) for k in ('vllm','vllm.sampling_params')}
sys.modules['vllm']=NS(SamplingParams=Params);sys.modules['vllm.sampling_params']=NS(RequestOutputKind=NS(CUMULATIVE='cumulative'))
try:
 with tempfile.TemporaryDirectory(prefix='arrival-analysis-fixture-',dir='/private/tmp') as tmp:
  run=Path(tmp)/'native';run.mkdir();source=json.loads((P/'math_twoburst_inputs1024_v1.json').read_text())
  rendered=[dict(r,prompt='fixture Answer:',prompt_token_ids=[1],prompt_tokens=1,
      prompt_token_ids_sha256=hashlib.sha256(b'[1]').hexdigest()) for r in source['requests']]
  def dump(name,obj): (run/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
  clock=Clock();generate.time=clock;engine=Engine(clock)
  with observe_native_pressure(engine,run/'measured-pressure.json'):
   status=generate.generate(engine,rendered,1024,'measured',run,source['arrival_traces_s'])
  status['expected_requests']=1024;dump('status.json',status)
  dump('source-input.json',source);dump('rendered-inputs.json',rendered)
  args=dict(enable_prefix_caching=True,async_scheduling=False,max_model_len=4096,max_num_seqs=1024,
            max_num_batched_tokens=4096,kv_cache_memory_bytes=8589934592,scheduler_reserve_full_isl=True)
  dump('config.json',dict(engine_args=args,sampling=dict(temperature=0.,min_tokens=0,max_tokens=1024,ignore_eos=False,stop=[]),
    arrival='fixed-two-burst: first512 at0s; last512 at4s',policy='native',fixed_margin_blocks=None))
  dump('resolved-runtime.json',dict(total_blocks=101,usable_blocks=100,block_size=16,kv_cache_memory_bytes=8589934592,
    max_num_running_reqs=1024,max_num_scheduled_tokens=4096,scheduler_reserve_full_isl=True))
  dump('tokenizer.json',dict(chat_template='synthetic fixture only',answer_cue='Answer:'))
  dump('resolved-eos.json',dict(hf_eos_token_id=2,generation_config={},sampling_defaults=dict(repetition_penalty=1.05)))
  dump('prefix-cache-reset.json',dict(reset_succeeded=True,cached_hash_keys_after=0,before=dict(status='QUALIFIED'),after=dict(status='QUALIFIED')))
  dump('native-drain.json',dict(status='QUALIFIED',unfinished=False,requests=0,running=0,waiting=0,total_blocks=101,free_blocks=100))
  (run/'cell-source.py').write_text('# synthetic CPU fixture, no GPU\n')
  dump('provenance.json',dict(input_sha256=analyze.FROZEN_INPUT_SHA,answer_keys_loaded=False,
      cell_sha256=hashlib.sha256((run/'cell-source.py').read_bytes()).hexdigest()))
  result=analyze.analyze(run,P/'math_answers1024_v1.json')
  assert result['integrity']=='VALID',result['issues'][:20]
  raws=json.loads((run/'measured-outputs.json').read_text());byid={r['request_id']:r for r in raws}
  for r in result['per_request']:
   raw=byid[r['request_id']]
   assert r['ttft_s']==raw['token_times_s'][0]-raw['arrival_s']
   assert r['completion_s']==raw['host_elapsed_s']-raw['arrival_s'] and r['completion_at_s']==raw['host_elapsed_s']
   assert r['submission_lag_s']==raw['add_request_s']-raw['arrival_s']
   assert r['max_host_gap_s']==max(b-a for a,b in zip(raw['token_times_s'],raw['token_times_s'][1:]))
  assert [len(s['new_native_request_ids']) for s in json.loads((run/'measured-arrival-receipt.json').read_text())['sampling_scans']]==[512,512]
  assert result['counts']['planned']==1024 and len(result['per_request'])==1024
  assert [c['counts']['planned'] for c in result['cohorts']]==[512,512]
  assert result['attribution']['totals']['scheduled_tokens']==3072
  assert result['attribution']['totals']['first_prefill_tokens']==1024
  assert result['attribution']['totals']['decode_tokens']==2048
  assert result['attribution']['totals']['recompute_tokens']==0
  analysis=Path(tmp)/'analysis.json';analysis.write_text(json.dumps(result))
  pair=compare.compare(analysis,analysis)
  assert pair['comparison_integrity']=='MATCHED',pair['issues']
  assert len(pair['cohort_changes'])==2 and all(x['completion_s']['right_minus_left']==0 for x in pair['per_request'])
  old=copy.deepcopy(result);old['schema']='c-math-scale-analysis-v1';oldpath=Path(tmp)/'old.json';oldpath.write_text(json.dumps(old))
  try:compare.compare(analysis,oldpath)
  except ValueError:pass
  else:raise AssertionError('old schema accepted')
  # Shift only one second-cohort raw output/journal clock: pressure/step attribution must reject it.
  original_outputs=(run/'measured-outputs.json').read_text();original_journal=(run/'measured-host-returns.jsonl').read_text()
  target=raws[512];target['token_times_s']=[t-4. for t in target['token_times_s']];target['host_elapsed_s']-=4.
  for e in target['host_returns']:e['return_s']-=4.
  dump('measured-outputs.json',raws)
  with (run/'measured-host-returns.jsonl').open('w') as f:
   for r in raws:
    for e in r['host_returns']:f.write(json.dumps(e)+'\n')
  invalid=analyze.analyze(run,P/'math_answers1024_v1.json')
  assert invalid['integrity']=='INVALID' and 'host_events_without_matching_engine_step' in invalid['issues']
  (run/'measured-outputs.json').write_text(original_outputs);(run/'measured-host-returns.jsonl').write_text(original_journal)
  receipt=json.loads((run/'measured-arrival-receipt.json').read_text())
  receipt['sampling_scans'][1]['new_native_request_ids'][0]=receipt['sampling_scans'][0]['new_native_request_ids'][0]
  dump('measured-arrival-receipt.json',receipt)
  invalid=analyze.analyze(run,P/'math_answers1024_v1.json')
  assert invalid['integrity']=='INVALID' and 'sampling_scan_native_id_inventory' in invalid['issues']
  print(json.dumps(dict(status='PASS',scope='synthetic CPU 1024 rows; actual generator and pressure observer; no GPU',
      checks=['full VALID analysis','TTFT/flow subtraction and completion_at retained','submission lag counted',
      'two native-ID batches 512+512','unchanged raw-clock attribution 1024 firstprefill +2048decode',
      'shifted raw clocks rejected','duplicate second-batch native ID rejected','paired wrapper MATCHED and old-schema rejected'],
      cohort_mean_flow=[c['timing']['completion_s']['mean'] for c in result['cohorts']],
      cohort_submission_lag=[c['timing']['submission_lag_s']['mean'] for c in result['cohorts']],denominator=result['counts']['planned'])))
finally:
 generate.time=old_time
 for k,v in saved.items():
  if v is None:sys.modules.pop(k,None)
  else:sys.modules[k]=v
