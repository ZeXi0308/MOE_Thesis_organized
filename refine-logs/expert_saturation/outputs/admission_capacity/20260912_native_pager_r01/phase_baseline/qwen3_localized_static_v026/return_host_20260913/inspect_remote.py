import json,os,shutil,subprocess,time
from pathlib import Path
stage=Path('/root/autodl-tmp/qwen3-localized-static-v026-launch-r01');out=Path('/root/autodl-tmp/qwen3-localized-static-v026-r01')
r=dict(unix_s=time.time(),paths={str(p):p.exists() for p in [stage,out,Path('/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python')]})
for name,p in [('launch',stage/'launch.json'),('result',out/'comparison/status.json')]:r[name]=json.loads(p.read_text()) if p.exists() else None
for name,p in [('loader_tail',out/'loader_receipt.jsonl'),('run_tail',stage/'run.log'),('hardware_tail',stage/'hardware.jsonl')]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-16000));r[name]=f.read().decode(errors='replace').splitlines()[-3:]
r['own_processes']=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes().split(b'\0')
  if any(str(stage).encode()+b'/' in a and a.endswith(b'.py') for a in argv):r['own_processes'].append(dict(pid=int(p.name),argv=[a.decode(errors='replace') for a in argv if a]))
 except OSError:pass
for name,cmd in [('gpu',['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version','--format=csv,noheader']),('gpu_processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])]:
 a=subprocess.run(cmd,capture_output=True,text=True,timeout=15);r[name]=dict(rc=a.returncode,stdout=a.stdout,stderr=a.stderr)
r['cgroup']={name:(Path('/sys/fs/cgroup')/name).read_text() for name in ['memory.max','memory.current','memory.events'] if (Path('/sys/fs/cgroup')/name).exists()}
r['disks']={str(p):shutil.disk_usage(p)._asdict() for p in [Path('/root'),Path('/root/autodl-tmp')] if p.exists()}
print(json.dumps(r))
