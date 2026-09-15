import hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent
protocol=json.loads((p/'protocol.json').read_text())
for name,sha in protocol['source_sha256'].items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha, name
source=Path(protocol['prepared_source']['remote'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['prepared_source']['sha256']
scheduler_source=Path('/root/autodl-tmp/expert-saturation/vllm-0.26/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py')
assert hashlib.sha256(scheduler_source.read_bytes()).hexdigest()==protocol['native_scheduler_sha256']
out=Path('/root/autodl-tmp/admission-memory-observer-v026-r01')
assert not out.exists()

with (p/'preflight.jsonl').open('x') as gate_log:
 idle=0
 for tick in range(121):
  query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
  if query.returncode:raise RuntimeError('GPU query failed: '+query.stderr)
  launchers=[]
  for entry in Path('/proc').iterdir():
   if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
   try:argv=(entry/'cmdline').read_bytes().split(b'\0')
   except OSError:continue
   if len(argv)>1 and b'python' in Path(argv[0].decode(errors='replace')).name.encode() and any(Path(x.decode(errors='replace')).name.startswith(('run_native_pager','run_wisp','run_session','run_expert_union')) and x.endswith(b'.py') for x in argv[1:]):
    launchers.append(dict(pid=int(entry.name),argv=[x.decode(errors='replace') for x in argv if x]))
  idle=idle+1 if not query.stdout.strip() and not launchers else 0
  state=dict(unix_s=time.time(),gpu=query.stdout.strip(),launchers=launchers,consecutive_idle=idle)
  gate_log.write(json.dumps(state)+'\n');gate_log.flush()
  if tick%6==0 or idle==3:print(json.dumps(dict(preflight=state)),flush=True)
  if idle==3:break
  time.sleep(5)
 else:raise RuntimeError('GPU remained occupied; no initialization attempted')

model='/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5'
python='/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',WISP_PLUGIN_DISABLE='1',WISP_PREFETCH='0',WISP_DYNAMIC='0',PYTHONPATH='/root/autodl-tmp/wisp-runtime-0112-r01/source/src:'+str(p))
command=['taskset','-c','0-7','timeout','-s','TERM','600',python,'-u',str(p/'run_native_pager.py'),'--output',str(out),'--prepared',str(source.parent),'--model',model,'--same-engine-admission-memory-observer','--expert-cap','16','--requests','3','--token-budget','64','--kv-bytes','536870912','--injection-chunk','16','--prompt-tokens','128','--trace-retention','episode','--execution','token']
launch=dict(status='RUNNING',start_unix_s=time.time(),command=command,results=str(out))
(p/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
import pynvml as nv
nv.nvmlInit();gpu=nv.nvmlDeviceGetHandleByIndex(0)
with (p/'run.log').open('x') as log,(p/'hardware.jsonl').open('x') as hw:
 process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 launch['child_pid']=process.pid
 print(json.dumps(launch),flush=True)
 while process.poll() is None:
  info=nv.nvmlDeviceGetMemoryInfo(gpu)
  sample=dict(unix_s=time.time(),processes=[dict(pid=x.pid,used_bytes=x.usedGpuMemory) for x in nv.nvmlDeviceGetComputeRunningProcesses(gpu)],used_bytes=info.used,temperature_c=nv.nvmlDeviceGetTemperature(gpu,nv.NVML_TEMPERATURE_GPU),power_mw=nv.nvmlDeviceGetPowerUsage(gpu))
  hw.write(json.dumps(sample)+'\n');hw.flush();time.sleep(.5)
nv.nvmlShutdown()
launch.update(status='EXITED',exit_code=process.returncode,end_unix_s=time.time())
launch['process_wall_s']=launch['end_unix_s']-launch['start_unix_s']
(p/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps(launch),flush=True)
if (out/'status.json').exists():print((out/'status.json').read_text(),flush=True)
raise SystemExit(process.returncode)
