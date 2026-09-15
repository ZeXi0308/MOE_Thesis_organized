"""Eight fresh processes, one shared GPU lease, no background waiting or retries."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;result=root/'results';status=root/'group-status.json'
if result.exists() or status.exists():raise SystemExit('refuse existing execution')
result.mkdir();data=dict(status='STARTING',pid=os.getpid(),cells=[],started=time.time())
lock=open('/root/autodl-tmp/moe-research-gpu.lock','a')
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(os.environ,HF_HOME='/root/autodl-tmp/hf-cache',HF_HUB_OFFLINE='1',VLLM_USE_FLASHINFER_SAMPLER='0')
 for label,gib,mode in [('b0-off-original',0,'original'),('b0-off-direct',0,'direct'),('b0-on-original',16,'original'),('b0-on-direct',16,'direct'),('b1-on-direct',16,'direct'),('b1-on-original',16,'original'),('b1-off-direct',0,'direct'),('b1-off-original',0,'original')]:
  env['KV_COUNT_OBSERVER']=mode
  check=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
  cell=dict(label=label,gib=gib,observer_mode=mode,gpu_before=check.stdout,started=time.time());data['cells'].append(cell)
  if check.stdout.strip():raise RuntimeError('ABORT_GPU_BUSY')
  data['status']='RUNNING';status.write_text(json.dumps(data,indent=2)+'\n')
  command=[sys.executable,'-u','run_probe.py','--domain','long','--reservation-policy','full','--completion-policy','native','--cap','32','--gpu-memory-utilization','0.9','--kv-cache-bytes','13960740864','--offload-gib',str(gib),'--output-dir',str(result/label)]
  with (result/(label+'.log')).open('w') as log:
   run=subprocess.run(command,cwd=root/'pkg',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  cell.update(exit_code=run.returncode,finished=time.time())
  status.write_text(json.dumps(data,indent=2)+'\n')
  if run.returncode:raise RuntimeError('cell failed; stop remaining cells')
  time.sleep(2)
 data['status']='COMPLETE'
except Exception as e:
 data.update(status='ABORTED_OR_FAILED',error=repr(e));raise
finally:
 data['finished']=time.time();status.write_text(json.dumps(data,indent=2)+'\n')
