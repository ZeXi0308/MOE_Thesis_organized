from pathlib import Path
import fcntl,json,tarfile,hashlib,shutil,os
p=Path('/root/moe-a-native-oldest-admission-session-r01-20261002')
out=Path('/root/moe-a-native-oldest-admission-r01-20261002-readback.tar.gz')
with open('/root/autodl-tmp/moe-research-gpu.lock','r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 st=os.fstat(lock.fileno());assert f'{st.st_dev}:{st.st_ino}'=='2304:29005388732'
 r=json.loads((p/'receipt.json').read_text())
 assert r['status']=='CELLS_COMPLETE',r['status']
 assert not out.exists()
 metadata=p/'completed-storage-before-launch';assert not metadata.exists();metadata.mkdir()
 for name in ('moe-a-completed-victim-size-dedup-r02-20261002.json','moe-a-verified-victim-size-raw-compaction-r02-20261002.json'):
  src=Path('/root')/name;d=json.loads(src.read_text());assert d['status']=='COMPLETE';shutil.copy2(src,metadata/name)
 files=[f for f in p.rglob('*') if f.is_file() and not any(x.startswith('runtime-cache') for x in f.relative_to(p).parts[:-1])]
 assert not any(f.is_symlink() for f in files)
 with tarfile.open(out,'w:gz') as t:
  for f in files:t.add(f,arcname=str(Path(p.name)/f.relative_to(p)),recursive=False)
 print(json.dumps(dict(status=r['status'],files=len(files),raw_bytes=sum(f.stat().st_size for f in files),archive=str(out),archive_bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),receipt=r)),flush=True)
