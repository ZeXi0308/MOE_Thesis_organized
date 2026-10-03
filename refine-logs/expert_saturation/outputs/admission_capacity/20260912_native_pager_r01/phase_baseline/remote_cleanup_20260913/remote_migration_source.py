import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
manifest=json.load(sys.stdin)
roots={'/root/autodl-tmp/moe-rotation-holdout-20260913-r02','/root/autodl-tmp/moe-rotation-fourarm-20260913-r01','/root/autodl-tmp/moe-rotation-victim-order-20260913-r01'}
receipt_path=Path('/root/autodl-tmp/cleanup-20260913-01a0953f-migration-r01.json')
assert not receipt_path.exists()
def command(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=10);return dict(command=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
def ancestors():
 seen=set();pid=os.getpid()
 while pid>0 and pid not in seen:
  seen.add(pid)
  try:pid=int(next(s.split()[1] for s in Path(f'/proc/{pid}/status').read_text().splitlines() if s.startswith('PPid:')))
  except (OSError,StopIteration):break
 return seen
excluded=ancestors()
def active_users():
 active=[];installers=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name) in excluded:continue
  try:
   args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0');cwd=os.readlink(proc/'cwd')
   if any(x in ('pip','pip3','uv') or Path(x).name in ('pip','pip3') for x in args[:3]):installers.append(int(proc.name))
   reasons=[]
   if any(cwd==r or cwd.startswith(r+'/') for r in roots):reasons.append('cwd')
   if any(r in arg for r in roots for arg in args):reasons.append('argument')
   for fd in (proc/'fd').iterdir():
    try:target=os.readlink(fd)
    except OSError:continue
    if any(target.startswith(r+'/') for r in roots):reasons.append('open_file');break
   if reasons:active.append(dict(pid=int(proc.name),reasons=reasons))
  except (OSError,PermissionError):continue
 return active,installers
receipt=dict(status='VERIFYING_REVERSIBLE_MIGRATION',started_unix_s=time.time(),verified=[],deleted=[],migrated=[],disk_before=command(['df','-B1','/root/autodl-tmp','/root']))
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
try:
 active,installers=active_users();receipt['active_before']=active;receipt['installer_pids']=installers
 if active:raise RuntimeError('Old campaign files are in use; no deletion')
 assert len(manifest['files'])==36
 for item in manifest['files']:
  root=Path(item['remote_root']);path=Path(item['remote_path'])
  assert str(root) in roots and item['relative_path'].split('/')[0]=='results' and path.name=='raw.json'
  assert len(Path(item['relative_path']).parts)==3 and path==root/item['relative_path']
  assert path.resolve()==path and path.is_file() and not path.is_symlink()
  before=path.stat();assert before.st_size==item['size_bytes'] and before.st_mtime_ns==item['remote_inventory_mtime_ns']
  h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
  after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  assert h.hexdigest()==item['sha256'],str(path)
  receipt['verified'].append(dict(path=str(path),sha256=h.hexdigest(),bytes=after.st_size,inode=after.st_ino,mtime_ns=after.st_mtime_ns))
 active,installers=active_users();receipt['active_after_hash']=active
 if active:raise RuntimeError('Old campaign became active; no deletion')
 receipt['status']='ALL_36_MATCH_LOCAL_ARCHIVES';receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
 target_root=Path('/root/moe-retained-raw-20260913-01a0953f')
 assert not target_root.exists()
 expected_bytes=sum(x['bytes'] for x in receipt['verified'])
 assert shutil.disk_usage('/root').free > expected_bytes + 2*1024**3
 target_root.mkdir()
 receipt['data_preserved']=True
 receipt['scope']='Relocate verified byte-identical raw to available system disk; original paths remain readable through symlinks. No research data or pip cache discarded.'
 for item in receipt['verified']:
  path=Path(item['path']);s=path.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(item['inode'],item['bytes'],item['mtime_ns'])
  relative=path.relative_to('/root/autodl-tmp');target=target_root/relative
  target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
  shutil.copy2(path,target)
  h=hashlib.sha256()
  with target.open('rb') as f:
   for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
  assert h.hexdigest()==item['sha256'] and target.stat().st_size==item['bytes']
  s=path.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(item['inode'],item['bytes'],item['mtime_ns'])
  link=path.with_name('raw.json.migration-link-01a0953f');assert not link.exists()
  link.symlink_to(target);os.replace(link,path)
  assert path.is_symlink() and path.resolve()==target and path.stat().st_size==item['bytes']
  receipt['migrated'].append(dict(original_path=str(path),retained_path=str(target),sha256=item['sha256'],bytes=item['bytes'],original_path_readable=True))
 receipt.update(status='COMPLETE',migrated_raw_bytes=sum(x['bytes'] for x in receipt['migrated']),deleted_raw_files=0,pip_http_cache_deleted=False,finished_unix_s=time.time(),disk_after=command(['df','-B1','/root/autodl-tmp','/root']),gpu_after=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']))
except BaseException as exc:
 receipt.update(status='FAILED_OR_PARTIAL',error=f'{type(exc).__name__}: {exc}',finished_unix_s=time.time())
finally:
 receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if receipt['status']!='COMPLETE':raise SystemExit(1)
