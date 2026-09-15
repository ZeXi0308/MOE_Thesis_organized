"""Bounded read-only GPU wait, then exactly one authorized staged execution."""
import json,pathlib,shlex,subprocess,sys,time,os
root=pathlib.Path(__file__).resolve().parent
record=json.loads((root/'execution/execution.json').read_text())
assert record['status']=='STAGED' and not record['cells']
assert record['host']=='root@connect.westc.seetacloud.com' and record['port']=='53036'
assert record['archive_sha256']=='fd1342a6aea8f2cc886a0a6957d4059dcc2d3ffad1797b0428fb23d8be162b4c'
state=dict(status='WAITING_GPU',pid=os.getpid(),started_unix_s=time.time(),maximum_wait_s=900,observations=0)
status=root/'wait_then_resume_status.json'
def save():status.write_text(json.dumps(state,indent=2)+'\n')
ssh=['ssh','-p','53036','-S','/private/tmp/moe-westc-53036.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','root@connect.westc.seetacloud.com']
try:
 while time.time()-state['started_unix_s']<state['maximum_wait_s']:
  result=subprocess.run(ssh+['nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'],text=True,capture_output=True,timeout=30)
  observation=dict(unix_s=time.time(),returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
  with (root/'gpu_wait_observations.jsonl').open('a') as f:f.write(json.dumps(observation)+'\n')
  state['observations']+=1;state['latest_observation']=observation;save();print(json.dumps(observation),flush=True)
  if result.returncode:raise RuntimeError('GPU query failed; no execution was requested')
  if not observation['stdout']:break
  time.sleep(30)
 else:
  state['status']='BLOCKED_GPU';save();raise SystemExit(2)
 state['status']='STARTING_STAGED_DRIVER';save()
 driver=root.parents[2]/'experiments/admission_capacity/run_frozen_kv_remote.py'
 command=[sys.executable,'-B',str(driver),'--resume-staged','--source',record['source'],'--output',str(root/'execution'),'--host',record['host'],'--port',record['port'],'--control-path','/private/tmp/moe-westc-53036.sock','--gpu-uuid',record['gpu_uuid'],'--remote-dir',record['remote_dir'],'--labels',*record['labels']]
 print(shlex.join(command),flush=True)
 state['driver_returncode']=subprocess.run(command).returncode
 state['status']='DRIVER_COMPLETE' if state['driver_returncode']==0 else 'DRIVER_STOPPED';state['finished_unix_s']=time.time();save()
 raise SystemExit(state['driver_returncode'])
except Exception as error:
 state.update(status='STOPPED',error=str(error),finished_unix_s=time.time());save();raise
