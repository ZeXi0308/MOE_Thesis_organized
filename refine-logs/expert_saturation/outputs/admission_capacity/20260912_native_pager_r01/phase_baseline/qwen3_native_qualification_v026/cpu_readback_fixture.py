"""Synthetic accounting fixture only: no Qwen weights, inference, or GPU results."""
import ast,copy,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace as N,ModuleType
from analyze_qualification import analyze,read,digest
B=Path(__file__).parent.resolve();S=B/'source';P=B.parent;F=B/'cpu_readback_fixture_r01';F.mkdir(exist_ok=False)
EP=F/'repeat_0_qwen_qualification';EP.mkdir();(F/'pager').mkdir()
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def jl(path,values):path.write_text(''.join(json.dumps(v)+'\n' for v in values))
sys.path.insert(0,str(S))
# Reuse the small preexisting CPU scheduler classes, without executing their tests.
classes=[n for n in ast.parse(Path('/private/tmp/check_admission_width.py').read_text()).body if isinstance(n,ast.ClassDef)];ns={'N':N};exec(compile(ast.Module(body=classes,type_ignores=[]),'cpu_engine_fixture','exec'),ns)
Req,Scheduler,Engine=[ns[k] for k in ('Req','Scheduler','Engine')];Req.num_output_placeholders=0;Req.spec_token_ids=[];Scheduler.num_spec_tokens=0;Scheduler.num_lookahead_tokens=0
v=ModuleType('vllm');v.SamplingParams=lambda **kw:N(**kw);sp=ModuleType('vllm.sampling_params');sp.RequestOutputKind=N(CUMULATIVE='c');sys.modules.update({'vllm':v,'vllm.sampling_params':sp})
cn={};exec(compile((S/'native_capture.py').read_text(),'native_capture_cpu_fixture','exec'),cn);clock=[0.]
cn['time']=N(perf_counter=lambda:clock[0],time=lambda:1000.+clock[0],sleep=lambda d:clock.__setitem__(0,clock[0]+d),thread_time_ns=time.thread_time_ns,process_time_ns=time.process_time_ns)
original=Engine.step
def timed(self):
 values=original(self);clock[0]+=.007;return values
Engine.step=timed
def capture(work,outputs):
 e=Engine();e.s.max_num_running_reqs=4;e.s.scheduler_config.max_num_seqs=4;clock[0]=0.
 return cn['capture_episode'](e,work,dict(output_tokens=outputs,cap=4,policy='static',allow_preemption=False,phase_prefill_mode='static32'),regime='steady',arrival_scale=1,run_id='fixture',cpu_diagnostics=True)
work=dict(tokenizer_identity=dict(repository='Qwen/Qwen3-30B-A3B',revision='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'),source_requests=[dict(request_id=f'fixture{i}',document_id=f'fixture_doc{i}') for i in range(4)],actual_prompt_token_ids=[[100+i]*64 for i in range(4)],arrival_traces_s={'steady':[i*.05 for i in range(4)]})
raw=capture(work,8);assert raw['status']=='COMPLETE',raw['error']
warm=capture(dict(work,source_requests=work['source_requests'][:1],actual_prompt_token_ids=work['actual_prompt_token_ids'][:1],arrival_traces_s={'steady':[0.]}),2)
phase='repeat_0_qwen_qualification/measurement';layer_names=[f'model.layers.{i}.mlp.experts.routed_experts' for i in range(48)]
layers=[dict(layer_name=n,cap=48,num_experts=128,pinned_bytes=128*9437184,scratch_bytes=48*9437184) for n in layer_names]
trace=[];references=[];loaded_once=set()
for step in raw['scheduler_steps']:
 context=dict(phase=phase,step_id=step['step'],row_request_order_verified=True,rows=[dict(internal_request_id=r['internal_request_id'],computed_position=p) for r in step['scheduled'] for p in range(r['scheduled_start_computed'],r['computed_after'])])
 for name in layer_names:
  loaded=[] if name in loaded_once else list(range(8));n=step['total_scheduled_tokens']
  g=dict(required_experts=list(range(8)),ensure_experts=list(range(8)),unique_experts=8,loaded_experts=loaded,evicted_experts=[],miss=len(loaded),evict=0,weight_copy_bytes=len(loaded)*9437184,start=0,stop=n,load_cuda_span_ms=0.)
  r=dict(call_id=len(trace),layer_name=name,context=copy.deepcopy(context),grouping_axis='expert',execution='expert',entry_resident_experts=list(range(8)) if name in loaded_once else [],active_experts=list(range(8)),row_topk_experts=[list(range(8)) for _ in range(n)],rows=n,retention=dict(mode='none'),groups=[g],group_count=1,miss=len(loaded),evict=0,weight_copy_bytes=g['weight_copy_bytes'],status='complete',measurement=False,validation_run=True)
  if name not in loaded_once:
   references.append(dict(layer_name=name,call_id=r['call_id'],context=r['context'],rows=n,actual_expert_groups=[list(range(8))],scope='all pre-call rows; CPU fixture',allfinite=True,maxabs=0.,relative_l2=0.,rtol=.01,atol=.01,allclose=True))
  loaded_once.add(name);trace.append(r)
pager=dict(all_calls=len(trace),cap=48,layers=layers,all={k:sum(r[k] for r in trace) for k in ('miss','evict','weight_copy_bytes','group_count')})
dump(F/'status.json',dict(status='COMPLETE',fixture_only=True));dump(F/'config.json',dict(qwen_qualification=True,verify_kernel=True,trace_retention='episode',requests=4,output_tokens=8,expert_cap=48,prefill_limit=32,arrival_interval=.05))
dump(F/'engine_args.json',dict(max_num_seqs=4,max_num_batched_tokens=64,kv_cache_memory_bytes=536870912,load_format='qwen_bf16_serial_v026',dtype='bfloat16',async_scheduling=False,enable_prefix_caching=False))
dump(F/'environment.json',dict(sources=read(B/'source_manifest.json')['files']))
dump(F/'episodes.json',[dict(phase=phase)]);dump(F/'workload.json',work);dump(F/'pager_summary.json',pager);dump(EP/'raw.json',raw);dump(EP/'warmup.json',warm)
dump(EP/'measurement_resources.json',dict(actual_unique_kv_storage_bytes=536870912,expert_cap=48,scheduler_requests=0,cache={name:dict(expert_to_slot={}) for name in layer_names}))
jl(F/'pager/calls.jsonl',trace)
dump(F/'qualification_validation.json',dict(layer_validation=references,validation_calls=48,unique_validated_layers=48,reference_full_weight_copy_bytes=54*2**30))
dump(EP/'map_optimization.json',dict(validation=dict(status='PASS',layers=48),statistics=dict(totals=dict(misses=sum(r['miss'] for r in trace)))))
dump(F/'memory_observer.json',dict(values_equal=True,warmup_mode='nested',measurement_mode='nested'))
jl(F/'gpu_checks.jsonl',[dict(decision='PASS',foreign_pids=[],query_errors=[],fixture_only=True)])
# Reuse real CPU loader event fixtures; combine with its independent strict-target fixture.
R=P/'qwen3_streaming_readiness'
for dst,src in [('load.jsonl','cpu_fixture_r01/success.jsonl'),('manifest.json','cpu_fixture_r01/manifest.json'),('index.json','cpu_fixture_r01/index.json'),('load.jsonl.targets.json','cpu_loader_fixture_r01/not_downloaded.jsonl.targets.json')]:
 (F/dst).write_bytes((R/src).read_bytes())
args=(F,F/'load.jsonl',F/'manifest.json',F/'index.json',S)
result=analyze(*args);assert result['computed_positions']==284 and result['measurement_output_tokens']==32
failures=[]
def rejected(label,path,change):
 prior=read(path);changed=copy.deepcopy(prior);change(changed);dump(path,changed)
 try:analyze(*args)
 except (ValueError,KeyError):failures.append(label)
 else:raise AssertionError('invalid fixture accepted: '+label)
 finally:dump(path,prior)
rejected('missing_reference_layer',F/'qualification_validation.json',lambda v:v['layer_validation'].pop())
rejected('ready_decode_not_advanced',EP/'raw.json',lambda v:next(s for s in v['scheduler_steps'] if s['existing_decode_request_ids']).__setitem__('existing_decode_all_scheduled',False))
rejected('reference_bytes_mixed',F/'qualification_validation.json',lambda v:v.__setitem__('reference_full_weight_copy_bytes',v['reference_full_weight_copy_bytes']+9437184))
rejected('missing_target',F/'load.jsonl.targets.json',lambda v:v['expected_named_parameters'].append('missing.weight'))
prior=trace[0]['context']['rows'][0]['computed_position'];trace[0]['context']['rows'][0]['computed_position']=999;jl(F/'pager/calls.jsonl',trace)
try:analyze(*args)
except ValueError:failures.append('physical_position_misalignment')
else:raise AssertionError('bad position accepted')
trace[0]['context']['rows'][0]['computed_position']=prior;jl(F/'pager/calls.jsonl',trace)
old=(F/'load.jsonl').read_text();events=[json.loads(s) for s in old.splitlines()];events[-1]['event']='FAILED';jl(F/'load.jsonl',events)
try:analyze(*args)
except ValueError:failures.append('loader_incomplete_despite_target_PASS')
else:raise AssertionError('failed stream accepted')
(F/'load.jsonl').write_text(old)
summary=dict(status='PASS',scope='CPU synthetic readback/accounting only; no Qwen inference, hardware validation, or performance evidence',no_gpu=True,source_files=len(read(B/'source_manifest.json')['files']),positions=284,layer_row_joins=284*48,measurement_outputs=32,warmup_outputs=2,negative_checks=failures,analyzer_sha256=hashlib.sha256((B/'analyze_qualification.py').read_bytes()).hexdigest())
dump(B/'cpu_readback_checks.json',summary);print(json.dumps(summary,indent=2))
