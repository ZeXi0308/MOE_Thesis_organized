import json,os,shutil,time
from pathlib import Path
r=dict(unix_s=time.time(),mounts=[],storage_roots={})
for line in Path('/proc/mounts').read_text().splitlines():
 fields=line.split()
 if len(fields)>2 and ('autodl' in fields[1] or fields[1] in ['/root','/']):r['mounts'].append(dict(target=fields[1],filesystem=fields[2]))
for p in sorted(Path('/root').iterdir()):
 if not p.name.startswith(('autodl-','qwen3-compressed-cache')):continue
 v=dict(is_dir=p.is_dir())
 if p.is_dir():
  try:v.update(disk=shutil.disk_usage(p)._asdict(),entries=sorted(q.name for q in p.iterdir())[:120])
  except OSError as e:v['error']=str(e)
 r['storage_roots'][str(p)]=v
print(json.dumps(r))
