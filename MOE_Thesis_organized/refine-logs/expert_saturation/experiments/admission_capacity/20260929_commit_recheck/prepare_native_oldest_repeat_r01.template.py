from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,tarfile,time
lock_path=Path('/root/autodl-tmp/moe-research-gpu.lock')
session=Path('/root/moe-a-native-oldest-admission-session-r03-20261002')
seed=Path('/root/moe-a-native-backfill-only-primary-session-r01-20261001/runtime-cache-first')
out=Path('/root/moe-a-completed-oldest-admission-dedup-r03-20261002.json')
archive=Path('/root/moe-a-native-oldest-repeat-stage-r01-20261002.tar.gz')
stage=Path('/root/moe-a-native-oldest-repeat-stage-r01-20261002')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
with lock_path.open('r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 st=os.fstat(lock.fileno());assert f'{st.st_dev}:{st.st_ino}'=='2304:29005388732'
 assert not stage.exists() and not out.exists()
 receipt=json.loads((session/'receipt.json').read_text())
 assert receipt['status']=='CELLS_COMPLETE' and all(c['archive_status']=='VERIFIED' for c in receipt['cells'])
 assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
 before=shutil.disk_usage('/root').free;rows=[]
 def merge(original,duplicate,kind,expected=None,mtime=False):
  assert original.is_file() and duplicate.is_file() and not original.is_symlink() and not duplicate.is_symlink()
  a,b=original.stat(),duplicate.stat()
  if a.st_ino==b.st_ino and a.st_dev==b.st_dev:return
  if a.st_dev!=b.st_dev or a.st_size!=b.st_size or (mtime and a.st_mtime_ns!=b.st_mtime_ns):return
  digest=sha(original)
  if sha(duplicate)!=digest:return
  if expected is not None:assert expected==digest
  tmp=duplicate.with_name(duplicate.name+'.a-dedup-temp');assert not tmp.exists()
  os.link(original,tmp);os.replace(tmp,duplicate)
  rows.append(dict(source=str(original),path=str(duplicate),bytes=a.st_size,sha256=digest,kind=kind))
 # Frozen completed source snapshots: retain every path, hard-link equal payload copies.
 source_dedup=[]
 for completed_name,stage_name,package_name in (
  ('moe-a-native-oldest-admission-session-r03-20261002','moe-a-native-oldest-admission-stage-r02-20261002','candidate_native_oldest_admission_r02'),):
  done=json.loads((Path('/root')/completed_name/'receipt.json').read_text())
  assert done['status']=='CELLS_COMPLETE' and all(c['exit_code']==0 and c['archive_status']=='VERIFIED' for c in done['cells'])
  frozen=Path('/root')/stage_name
  first=frozen/'first'/package_name
  manifest=json.loads((first/'manifest.json').read_text())
  for label in ('second','third'):
   other=frozen/label/package_name
   assert sha(first/'manifest.json')==sha(other/'manifest.json')
   for name,digest in manifest.items():
    assert '..' not in Path(name).parts and not Path(name).is_absolute()
    original,duplicate=first/name,other/name
    assert sha(original)==sha(duplicate)==digest
    merge(original,duplicate,'completed_source_payload',digest)
 for label in ('first','second','third'):
  cache=session/('runtime-cache-'+label)
  for duplicate in cache.rglob('*'):
   original=seed/duplicate.relative_to(cache)
   if duplicate.is_file() and not duplicate.is_symlink() and original.is_file() and not original.is_symlink():
    merge(original,duplicate,'completed_cache',mtime=True)
 for i,cell in enumerate(receipt['cells']):
  cell_dir=session/f"cell-{i:02d}-{cell['arm']}"
  hashes=json.loads((cell_dir/'output_sha256.json').read_text())
  output=Path(cell['output_dir'])
  assert str(output).startswith('/root/moe-a-native-oldest-admission-output-')
  for name,digest in hashes.items():merge(cell_dir/'archive'/name,output/name,'completed_output',digest)
 compact_path=Path('/root/moe-a-verified-oldest-admission-raw-compaction-r03-20261002.json')
 assert not compact_path.exists()
 compact_files=[];compact_dirs=[];compact_units=[]
 for old_name,tar_name,tar_sha in (
  ('moe-a-native-oldest-admission-session-r03-20261002','moe-a-native-oldest-admission-r03-20261002-readback.tar.gz','5d8b90d43f6c7253c7707353d0c0c7237efeb8c335dfd32367f0317ace5f65d3'),):
  old_dir=Path('/root')/old_name;tar_path=Path('/root')/tar_name
  done=json.loads((old_dir/'receipt.json').read_text())
  assert done['status']=='CELLS_COMPLETE' and all(c['exit_code']==0 and c['archive_status']=='VERIFIED' for c in done['cells'])
  assert tar_path.is_file() and not tar_path.is_symlink() and sha(tar_path)==tar_sha
  with tarfile.open(tar_path) as old_tar:tar_members={m.name for m in old_tar.getmembers() if m.isfile()}
  for i,cell in enumerate(done['cells']):
   cell_dir=old_dir/f"cell-{i:02d}-{cell['arm']}"
   saved=cell_dir/'archive';output=Path(cell['output_dir'])
   assert str(output).startswith('/root/moe-a-native-') and not output.is_symlink() and not saved.is_symlink()
   hashes=json.loads((cell_dir/'output_sha256.json').read_text())
   for name,digest in hashes.items():
    assert '..' not in Path(name).parts and not Path(name).is_absolute()
    assert str(Path(old_name)/saved.relative_to(old_dir)/name) in tar_members
    for f in (saved/name,output/name):
     assert f.is_file() and not f.is_symlink() and sha(f)==digest
     compact_files.append(dict(path=str(f),sha256=digest,bytes=f.stat().st_size,tar=str(tar_path)))
   compact_dirs.extend((saved,output))
  compact_units.append(dict(session=str(old_dir),retained_archive=str(tar_path),sha256=tar_sha,local_copy='Verified complete raw session in A_DIR before this operation'))
 compact_info=dict(status='PREPARED',units=compact_units,files=compact_files,free_before=shutil.disk_usage('/root').free,note='Only completed own A uncompressed redundant outputs; original raw contents retained in verified remote archives AND complete verified local copies. Runtime seed/caches and metadata retained.')
 compact_path.write_text(json.dumps(compact_info,indent=2)+'\n')
 for item in compact_files:Path(item['path']).unlink()
 for directory in compact_dirs:
  marker=directory/'REMOTE_RAW_IN_VERIFIED_ARCHIVE.json'
  assert not marker.exists()
  marker.write_text(json.dumps(dict(compaction_receipt=str(compact_path),units=compact_units,note='Original contents preserved; extract named verified archive to a new directory to read them.'),indent=2)+'\n')
 compact_info.update(status='COMPLETE',completed_unix_s=time.time(),free_after=shutil.disk_usage('/root').free)
 compact_path.write_text(json.dumps(compact_info,indent=2)+'\n')
 print('VERIFIED_RAW_COMPACTED',len(compact_files),compact_info['free_before'],compact_info['free_after'],flush=True)
 result=dict(status='COMPLETE',completed_unix_s=time.time(),free_before=before,free_after=shutil.disk_usage('/root').free,files=rows,note='Byte-identical completed own A copies only; all paths retained. Never rerun completed controllers.')
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='files'}),'files',len(rows),flush=True)
 assert sha(archive)=='ARCHIVE_SHA_PENDING'
 with tarfile.open(archive) as t:
  assert all(x.name.startswith(stage.name+'/') or x.name==stage.name for x in t.getmembers())
  assert all('..' not in Path(x.name).parts and not x.issym() and not x.islnk() for x in t.getmembers())
  t.extractall('/root',filter='data')
 assert sha(stage/'NATIVE_OLDEST_REPEAT_TRIPLET_PLAN_R01_20261002.json')=='PLAN_SHA_PENDING'
 # Immutable new payload copies can share bytes; run markers, caches and logs remain private.
 first=stage/'first'/'candidate_native_oldest_repeat_r01'
 manifest=json.loads((first/'manifest.json').read_text())
 previous=Path('/root/moe-a-native-oldest-admission-stage-r02-20261002/first/candidate_native_oldest_admission_r02')
 for name,digest in manifest.items():
  assert sha(first/name)==digest
  source=previous/name
  if source.is_file() and sha(source)==digest:
   merge(source,first/name,'frozen_shared_unchanged_payload',digest)
 for label in ('second','third'):
  other=stage/label/'candidate_native_oldest_repeat_r01'
  assert sha(first/'manifest.json')==sha(other/'manifest.json')
  for name,digest in manifest.items():
   assert sha(first/name)==sha(other/name)==digest
   merge(first/name,other/name,'frozen_new_source_payload',digest)
 # Complete source remains extracted, and exact upload is retained locally.
 uploaded_sha=sha(archive);assert uploaded_sha=='ARCHIVE_SHA_PENDING'
 archive.unlink()
 result.update(files=rows,retired_duplicate_upload=dict(path=str(archive),sha256=uploaded_sha,retention='Complete extracted stage plus verified original in local /private/tmp; no scientific raw retired'),free_after_stage=shutil.disk_usage('/root').free)
 out.write_text(json.dumps(result,indent=2)+'\n')
 print('STAGED',str(stage),'free_bytes',shutil.disk_usage('/root').free,flush=True)
os.chdir(stage)
log=os.open(str(stage/'controller.log'),os.O_CREAT|os.O_WRONLY|os.O_EXCL,0o600)
os.dup2(log,1);os.dup2(log,2);os.close(log)
python='/root/autodl-tmp/moe-a-runtime-20260930/venv/bin/python'
os.execv(python,[python,'-B','run_native_oldest_repeat_triplet_r01.py','NATIVE_OLDEST_REPEAT_TRIPLET_PLAN_R01_20261002.json','--expected-plan-sha256','PLAN_SHA_PENDING'])

