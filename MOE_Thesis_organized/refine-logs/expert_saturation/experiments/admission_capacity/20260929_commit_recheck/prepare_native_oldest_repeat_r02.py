from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,tarfile,time
lock_path=Path('/root/autodl-tmp/moe-research-gpu.lock')
stage=Path('/root/moe-a-native-oldest-repeat-stage-r02-20261002')
archive=Path(str(stage)+'.tar.gz')
receipt_path=Path('/root/moe-a-completed-source-dedup-oldest-repeat-r02-20261002.json')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
with lock_path.open('r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 st=os.fstat(lock.fileno());assert f'{st.st_dev}:{st.st_ino}'=='2304:29005388732'
 assert not stage.exists() and not receipt_path.exists()
 aborted=json.loads(Path('/root/moe-a-native-oldest-repeat-session-r01-20261002/receipt.json').read_text())
 assert aborted['status']=='ABORTED' and not aborted['cells']
 assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
 roots=set()
 for f in Path('/root').glob('moe-a-*-session-*/receipt.json'):
  try:r=json.loads(f.read_text())
  except (OSError,ValueError):continue
  if r.get('status')!='CELLS_COMPLETE' or not r.get('cells') or not all(c.get('exit_code')==0 and c.get('archive_status')=='VERIFIED' for c in r['cells']):continue
  for c in r['cells']:
   argv=c.get('argv',[])
   if argv and argv[0].startswith('/root/moe-a-') and argv[0].endswith('/pkg/run.sh'):
    p=Path(argv[0]).parent.parent
    if (p/'manifest.json').is_file() and not p.is_symlink():roots.add(p)
 before=shutil.disk_usage('/root').free
 seen={};hash_cache={};rows=[];skipped=[]
 def verified_digest(p):
  s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
  if key not in hash_cache:hash_cache[key]=sha(p)
  return hash_cache[key]
 for package in sorted(roots):
  manifest=json.loads((package/'manifest.json').read_text());payload=[]
  for name,digest in manifest.items():
   relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
   p=package/name
   if not p.is_file() or p.is_symlink() or verified_digest(p)!=digest:
    skipped.append(dict(package=str(package),name=name,reason='PAYLOAD_NOT_IMMUTABLE_MANIFEST_MATCH'));break
   payload.append((p,digest))
  else:
   for p,digest in payload:
    st=p.stat();key=(digest,st.st_size,st.st_mode,st.st_dev)
    original=seen.setdefault(key,p);src=original.stat()
    if src.st_ino==st.st_ino:continue
    assert verified_digest(original)==digest
    tmp=p.with_name(p.name+'.a-source-dedup-temp');assert not tmp.exists()
    os.link(original,tmp);os.replace(tmp,p)
    rows.append(dict(source=str(original),path=str(p),bytes=st.st_size,sha256=digest))
 result=dict(status='COMPLETE',completed_unix_s=time.time(),free_before=before,
   free_after=shutil.disk_usage('/root').free,completed_package_roots=len(roots),
   files=rows,skipped=skipped,note='Only byte-identical manifest-listed immutable payloads of completed own A packages. All paths retained; no models, caches, scientific raw, other lines or mutable launch markers modified.')
 receipt_path.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('files','skipped')})+' files='+str(len(rows))+' skipped='+str(len(skipped)),flush=True)
 assert sha(archive)=='058db882a98084d859c7f6dfa11cc9a0eafbf23c2923097333b47a1cd03089a2'
 with tarfile.open(archive) as tar:
  assert all(m.name.startswith(stage.name+'/') or m.name==stage.name for m in tar.getmembers())
  assert all('..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in tar.getmembers())
  tar.extractall('/root',filter='data')
 assert sha(stage/'NATIVE_OLDEST_REPEAT_TRIPLET_PLAN_R02_20261002.json')=='5b3adf5dc8854261c4840697d4bcfe302dd6a488ca892dc5d0074752a47ecfa2'
 print('STAGED',str(stage),'free_bytes',shutil.disk_usage('/root').free,flush=True)
os.chdir(stage)
log=os.open(str(stage/'controller.log'),os.O_CREAT|os.O_WRONLY|os.O_EXCL,0o600)
os.dup2(log,1);os.dup2(log,2);os.close(log)
python='/root/autodl-tmp/moe-a-runtime-20260930/venv/bin/python'
os.execv(python,[python,'-B','run_native_oldest_repeat_triplet_r02.py','NATIVE_OLDEST_REPEAT_TRIPLET_PLAN_R02_20261002.json','--expected-plan-sha256','5b3adf5dc8854261c4840697d4bcfe302dd6a488ca892dc5d0074752a47ecfa2'])
