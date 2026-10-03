import importlib.metadata as md,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path
r={'unix_s':time.time(),'python':sys.version,'executable':sys.executable,'affinity':sorted(os.sched_getaffinity(0)),'memlock':resource.getrlimit(resource.RLIMIT_MEMLOCK)}
r['packages']={}
for name in ['torch','vllm','transformers','huggingface-hub','pip','uv','nvidia-ml-py','triton']:
 try:r['packages'][name]=md.version(name)
 except md.PackageNotFoundError:r['packages'][name]=None
r['cgroup']={}
for name in ['memory.max','memory.current','memory.events','memory.stat']:
 q=Path('/sys/fs/cgroup')/name;r['cgroup'][name]=q.read_text() if q.exists() else None
r['disk']={str(q):shutil.disk_usage(q)._asdict() for q in [Path('/root'),Path('/root/autodl-tmp')] if q.exists()}
r['paths']={}
for name in ['/root/autodl-tmp','/root/autodl-tmp/expert-saturation','/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python','/root/autodl-tmp/wisp-runtime-0112-r01/source/src','/root/autodl-tmp/hf-cache/hub','/root/.cache/huggingface/hub']:
 q=Path(name);r['paths'][name]={'exists':q.exists(),'entries':sorted(x.name for x in q.iterdir())[:40] if q.is_dir() else []}
r['processes']=[]
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  argv=(q/'cmdline').read_bytes().split(b'\0');exe=Path(argv[0].decode(errors='replace')).name
  if any(k in exe for k in ('python','pip','uv','run_native','run_wisp')):r['processes'].append({'pid':int(q.name),'exe':exe,'entry':Path(argv[1].decode(errors='replace')).name if len(argv)>1 else None})
 except OSError:pass
for label,args in [('gpu',['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version','--format=csv,noheader']),('gpu_processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])]:
 a=subprocess.run(args,capture_output=True,text=True,timeout=15);r[label]={'returncode':a.returncode,'stdout':a.stdout,'stderr':a.stderr}
print(json.dumps(r))
