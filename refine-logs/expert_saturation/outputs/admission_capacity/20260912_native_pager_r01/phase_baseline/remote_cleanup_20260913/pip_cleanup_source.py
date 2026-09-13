import json,os,shutil,subprocess,time
from pathlib import Path
cache=Path('/root/.cache/pip/http-v2')
receipt=dict(started_unix_s=time.time(),path=str(cache),scope='Only rebuildable pip HTTP download cache; model, venv, compiler/kernel caches and experiment data untouched')
def query(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=10);return dict(command=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
receipt['gpu_before']=query(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
receipt['disk_before']=query(['df','-B1','/root','/root/autodl-tmp'])
users=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(Path(x).name in ('pip','pip3','uv') for x in args[:3]):users.append(dict(pid=int(proc.name),reason='installer'))
  for fd in (proc/'fd').iterdir():
   try:target=os.readlink(fd)
   except OSError:continue
   if target==str(cache) or target.startswith(str(cache)+'/'):users.append(dict(pid=int(proc.name),reason='open_cache_file'));break
 except OSError:continue
receipt['active_cache_users']=users
if receipt['gpu_before']['exit_code'] or receipt['gpu_before']['stdout'].strip() or users:
 receipt['status']='DEFERRED_BUSY';receipt['deleted']=False
elif not cache.exists():receipt.update(status='ALREADY_ABSENT',deleted=False)
else:
 assert cache.resolve()==cache and not cache.is_symlink()
 receipt['allocated_before']=query(['du','-s','-B1',str(cache)])
 receipt['logical_bytes']=sum(f.stat().st_size for f in cache.rglob('*') if f.is_file() and not f.is_symlink())
 shutil.rmtree(cache);receipt.update(status='COMPLETE',deleted=True,verified_absent=not cache.exists())
receipt.update(finished_unix_s=time.time(),disk_after=query(['df','-B1','/root','/root/autodl-tmp']))
Path('/root/autodl-tmp/cleanup-20260913-01a0953f-pip-cache.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
