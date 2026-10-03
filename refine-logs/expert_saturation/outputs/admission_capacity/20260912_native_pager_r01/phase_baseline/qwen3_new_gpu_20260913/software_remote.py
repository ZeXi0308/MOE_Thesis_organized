from pathlib import Path
import json,subprocess,time
r={'unix_s':time.time()}
for pid in [2049]:
 q=Path('/proc')/str(pid)
 if q.exists():
  argv=(q/'cmdline').read_bytes().split(b'\0');r[str(pid)]={'argv_safe':[x.decode(errors='replace') if b'://' not in x and b'password' not in x.lower() and b'token' not in x.lower() else '<redacted>' for x in argv], 'cwd':str((q/'cwd').resolve()),'stdout':str((q/'fd/1').resolve())}
 else:r[str(pid)]={'exists':False}
r['entries']={}
for name in ['/root/autodl-tmp/expert-saturation','/root/autodl-tmp/expert-saturation/vllm-0.26','/root/autodl-tmp/.autodl']:
 q=Path(name);r['entries'][name]=sorted(x.name for x in q.iterdir()) if q.exists() else None
r['cpu_model']=next((s for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),None)
code="import importlib.metadata as m,json,sys;names=['pip','vllm','torch','transformers','triton','nvidia-ml-py'];d=dict(python=sys.executable);\nfor n in names:\n try:d[n]=m.version(n)\n except m.PackageNotFoundError:d[n]=None\nprint(json.dumps(d))"
a=subprocess.run(['/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python','-c',code],capture_output=True,text=True,timeout=15);r['venv']={'exit_code':a.returncode,'stdout':a.stdout,'stderr':a.stderr}
print(json.dumps(r))
