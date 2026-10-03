import hashlib,json,os,signal,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;protocol=json.loads((p/'protocol.json').read_text())
for name,sha in protocol['source_sha256'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
source=Path(protocol['prepared_source']['remote']);assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['prepared_source']['sha256']
installed=Path('/root/autodl-tmp/expert-saturation/vllm-0.26/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py');assert hashlib.sha256(installed.read_bytes()).hexdigest()==protocol['native_scheduler_sha256']
out=Path('/root/autodl-tmp/continuous-admission-v026-r01');out.mkdir(exist_ok=False)
python='/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
model='/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5'
env=dict(os.environ,OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',WISP_PLUGIN_DISABLE='1',WISP_PREFETCH='0',WISP_DYNAMIC='0',PYTHONPATH='/root/autodl-tmp/wisp-runtime-0112-r01/source/src:'+str(p))
launch=dict(status='RUNNING',start_unix_s=time.time(),results=str(out),cells=[])
def save(): (p/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
save()
import pynvml as nv
nv.nvmlInit();gpu=nv.nvmlDeviceGetHandleByIndex(0);exit_code=0;process=None
try:
 for cell in protocol['cells']:
  stage=p/cell['name'];stage.mkdir(exist_ok=False)
  with (stage/'preflight.jsonl').open('x') as log:
   idle=0
   for tick in range(121):
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
    if query.returncode:raise RuntimeError('GPU query failed: '+query.stderr)
    launchers=[]
    for entry in Path('/proc').iterdir():
     if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
     try:argv=(entry/'cmdline').read_bytes().split(b'\0')
     except OSError:continue
     if len(argv)>1 and b'python' in Path(argv[0].decode(errors='replace')).name.encode() and any(Path(x.decode(errors='replace')).name.startswith(('run_native_pager','run_wisp','run_session','run_expert_union')) and x.endswith(b'.py') for x in argv[1:]):launchers.append(dict(pid=int(entry.name),argv=[x.decode(errors='replace') for x in argv if x]))
    idle=idle+1 if not query.stdout.strip() and not launchers else 0
    state=dict(unix_s=time.time(),gpu=query.stdout.strip(),launchers=launchers,consecutive_idle=idle)
    log.write(json.dumps(state)+'\n');log.flush()
    if tick%6==0 or idle==3:print(json.dumps(dict(cell=cell['name'],preflight=state)),flush=True)
    if idle==3:break
    time.sleep(5)
   else:raise RuntimeError('GPU remained occupied; no cell initialization attempted')
  command=['taskset','-c','0-7','timeout','-s','TERM','600',python,'-u',str(p/'run_native_pager.py'),'--output',str(out/cell['name']),'--prepared',str(source.parent),'--model',model,'--continuous-admission','--nested-memory-observer','--expert-cap','16','--requests','8','--token-budget','64','--kv-bytes','536870912','--prompt-tokens','128','--output-tokens','64','--trace-retention','episode','--execution','expert','--admission-limit',str(cell['admission_limit']),'--phase-policy',cell['phase_policy'],'--warmup-source-indices','0','1','2','--warmup-prompt-tokens','128']
  for flag,key in [('--source-indices','source_indices'),('--prompt-lengths','prompt_tokens'),('--output-lengths','output_tokens'),('--arrival-times','arrival_times_s')]:command += [flag]+[str(v) for v in protocol[key]]
  record=dict(**cell,status='RUNNING',start_unix_s=time.time(),command=command,results=str(out/cell['name']));launch['cells'].append(record);save()
  with (stage/'run.log').open('x') as log,(stage/'hardware.jsonl').open('x') as hw:
   process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record['child_pid']=process.pid
   (stage/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
   while process.poll() is None:
    info=nv.nvmlDeviceGetMemoryInfo(gpu)
    hw.write(json.dumps(dict(unix_s=time.time(),processes=[dict(pid=x.pid,used_bytes=x.usedGpuMemory) for x in nv.nvmlDeviceGetComputeRunningProcesses(gpu)],used_bytes=info.used,temperature_c=nv.nvmlDeviceGetTemperature(gpu,nv.NVML_TEMPERATURE_GPU),power_mw=nv.nvmlDeviceGetPowerUsage(gpu)))+'\n');hw.flush();time.sleep(.5)
  record.update(status='EXITED',exit_code=process.returncode,end_unix_s=time.time());record['process_wall_s']=record['end_unix_s']-record['start_unix_s'];(stage/'launch.json').write_text(json.dumps(record,indent=2)+'\n');save();print(json.dumps(record),flush=True)
  if process.returncode:exit_code=process.returncode;break
except BaseException as exc:
 if process is not None and process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=20)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGKILL);process.wait()
 launch['error']=f'{type(exc).__name__}: {exc}';exit_code=1;raise
finally:
 nv.nvmlShutdown();launch.update(status='EXITED',exit_code=exit_code,end_unix_s=time.time());launch['parent_wall_s']=launch['end_unix_s']-launch['start_unix_s'];save();print(json.dumps(launch),flush=True)
raise SystemExit(exit_code)
