from pathlib import Path
import json,os,time
p=Path('/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r01');out=Path('/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-r01')
def read(q):
 try:return json.loads(q.read_text()) if q.exists() else None
 except (OSError,ValueError):return None
def tail(q):
 if not q.exists():return None
 with q.open('rb') as f:
  f.seek(max(0,q.stat().st_size-18000));rows=f.read().decode(errors='replace').splitlines()
 try:return json.loads(rows[-1]) if rows else None
 except ValueError:return None
def live(pid):
 if not pid:return False
 q=Path('/proc')/str(pid)
 try:return q.exists() and (q/'stat').read_text().split()[2]!='Z' and str(p).encode() in (q/'cmdline').read_bytes()
 except OSError:return False
while True:
 launch=read(p/'launch.json') or {};dirs=list((p/'detached_receipts').glob('detached-*'));last=tail(dirs[0]/'receipt.jsonl') if dirs else None
 hw=tail(p/'hardware.jsonl');loader=tail(out/'loader_receipt.jsonl')
 r=dict(unix_s=time.time(),parent_state=launch.get('status'),monitor_receipt=last,worker_pid=launch.get('child_pid'),worker_live=live(launch.get('child_pid')),parent_live=live(last.get('child_pid')) if last and last.get('status')=='CHILD_STARTED' else False,elapsed_s=time.time()-launch.get('child_start_unix_s',time.time()),loader_latest=loader,qualification=read(out/'comparison/numerical_qualification/attribution_gate.json'),result=read(out/'comparison/status.json'))
 if hw:r['resources']=dict(gpu_used_bytes=hw['used_bytes'],host_current_bytes=hw['memory']['current'],host_anon_bytes=hw['memory']['stat']['anon'],foreign=hw['foreign'],sample_unix_s=hw['unix_s'])
 r['temporary_shards']=[dict(name=q.name,bytes=q.stat().st_size) for q in (out/'shard_workspace').rglob('*.safetensors')]
 if launch.get('status')=='EXITED':r['launch_final']=launch
 print(json.dumps(r),flush=True)
 if launch.get('status')=='EXITED' or (last and last.get('status') in ('MONITOR_ERROR','CHILD_EXITED')):break
 time.sleep(30)
