import json,os,shutil,subprocess,time
from pathlib import Path
os.sched_setaffinity(0,{8})
worker=5127;prefix='/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-r02/shard_workspace/'
maps=[line.rsplit(None,1)[-1] if '(deleted)' not in line else line[line.index('/root/'): ] for line in Path('/proc',str(worker),'maps').read_text().splitlines() if prefix in line]
fds=[]
for p in Path('/proc',str(worker),'fd').iterdir():
 try:
  target=os.readlink(p)
  if prefix in target:fds.append(dict(fd=p.name,target=target))
 except OSError:pass
sizes=[]
for p in sorted(Path('/root/autodl-tmp').iterdir()):
 r=subprocess.run(['du','-sk',str(p)],capture_output=True,text=True,timeout=15)
 sizes.append(dict(name=p.name,returncode=r.returncode,kib=int(r.stdout.split()[0]) if r.returncode==0 else None))
print(json.dumps(dict(unix_s=time.time(),own_deleted_or_mapped_shard_paths=sorted(set(maps)),own_shard_fds=fds,top_level_data_kib=sizes,disks={str(p):shutil.disk_usage(p)._asdict() for p in [Path('/root'),Path('/root/autodl-tmp')]},scope='Read-only disk sizes and only this worker shard mappings; no file deletion or modification.')))
