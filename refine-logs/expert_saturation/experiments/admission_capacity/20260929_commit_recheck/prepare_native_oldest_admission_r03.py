from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,tarfile,time
stage=Path('/root/moe-a-native-oldest-admission-stage-r03-20261002')
archive=Path(str(stage)+'.tar.gz')
receipt=Path('/root/moe-a-oldest-admission-source-dedup-r03-20261002.json')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
with open('/root/autodl-tmp/moe-research-gpu.lock','r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 st=os.fstat(lock.fileno());assert f'{st.st_dev}:{st.st_ino}'=='2304:29005388732'
 old=Path('/root/moe-a-native-oldest-admission-session-r02-20261002')
 done=json.loads((old/'receipt.json').read_text());assert done['status']=='ABORTED' and not done['cells']
 assert not stage.exists() and not receipt.exists()
 assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
 assert sha(archive)=='fd3363f7dac1c4fde03333b507d071e4128f0c8792a45cfb5fcd246092471bf0'
 before=shutil.disk_usage('/root').free;keepers={};rows=[]
 units=[
 ('moe-a-native-bidkv-full-running-session-r01-20261002','moe-a-native-bidkv-full-running-stage-r01-20261002','candidate_native_bidkv_full_running_r01',3),
 ('moe-a-native-completion-deferral-session-r01-20261002','moe-a-native-completion-deferral-stage-r01-20261002','candidate_native_completion_deferral_r01',3),
 ('moe-a-native-victim-size-session-r02-20261002','moe-a-native-victim-size-stage-r01-20261002','candidate_native_victim_size_r01',3),
 ('moe-a-native-current-guard-session-r02-20261002','moe-a-native-current-guard-stage-r01-20261002','candidate_native_current_guard_r01',3),
 ('moe-a-native-self-preempt-continue-session-r01-20261002','moe-a-native-self-preempt-continue-stage-r01-20261002','candidate_native_self_preempt_continue_r01',3),
 ('moe-a-native-current-guard-fresh-b1-session-r02-20261002','moe-a-native-current-guard-fresh-b1-stage-r01-20261002','candidate_native_current_guard_fresh_r01',4),
 ('moe-a-native-current-guard-fresh-b2-session-r02-20261002','moe-a-native-current-guard-fresh-b2-stage-r02-20261002','candidate_native_current_guard_fresh_r01',4),
 ('moe-a-native-oldest-admission-session-r01-20261002','moe-a-native-oldest-admission-stage-r01-20261002','candidate_native_oldest_admission_r01',3),
 ('moe-a-native-oldest-admission-session-r02-20261002','moe-a-native-oldest-admission-stage-r02-20261002','candidate_native_oldest_admission_r02',3)]
 for session_name,stage_name,package_name,count in units:
  prior=json.loads((Path('/root')/session_name/'receipt.json').read_text())
  frozen_unrun=session_name==old.name
  assert (prior['status']=='ABORTED' and not prior['cells']) if frozen_unrun else (prior['status']=='CELLS_COMPLETE' and all(c['exit_code']==0 and c['archive_status']=='VERIFIED' for c in prior['cells']))
  for label in ('first','second','third','fourth')[:count]:
   package=Path('/root')/stage_name/label/package_name
   for name,digest in json.loads((package/'manifest.json').read_text()).items():
    assert '..' not in Path(name).parts and not Path(name).is_absolute()
    duplicate=package/name;assert duplicate.is_file() and not duplicate.is_symlink() and sha(duplicate)==digest
    info=duplicate.stat();key=(digest,info.st_size,info.st_mode)
    source=keepers.setdefault(key,duplicate);si=source.stat()
    if si.st_dev==info.st_dev and si.st_ino!=info.st_ino:
     temp=duplicate.with_name(duplicate.name+'.a-dedup-temp');assert not temp.exists()
     os.link(source,temp);os.replace(temp,duplicate)
     rows.append(dict(source=str(source),path=str(duplicate),sha256=digest,bytes=info.st_size,frozen_unrun_payload=frozen_unrun))
 items=[
 ('moe-a-native-oldest-admission-stage-r01-20261002.tar.gz','453fdb44b98d872c1bcdf8a5aea4b5c9d20114b22caa95d89d7fa2cf199d9497'),
 ('moe-a-native-oldest-admission-stage-r02-20261002.tar.gz','24aaa27de1aacd110d836e7758281b9ca7332a76fdf4608fbf25de2381df68fe')]
 for name,digest in items:
  p=Path('/root')/name;assert p.is_file() and not p.is_symlink() and sha(p)==digest
  p.unlink()
 result=dict(status='COMPLETE',free_before=before,free_after=shutil.disk_usage('/root').free,files=rows,retired_transport_archives=items,completed_unix_s=time.time(),note='Only hash-identical immutable source payloads: all paths retained; no raw/model/runtime cache altered. R02 source manifest payloads are immutable across cells; launch markers, outputs and mutable caches remain separate. Removed redundant upload tars retained verified locally and extracted remotely.')
 receipt.write_text(json.dumps(result,indent=2)+'\n')
 with tarfile.open(archive) as t:
  assert all((m.name==stage.name or m.name.startswith(stage.name+'/')) and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in t.getmembers())
  t.extractall('/root',filter='data')
 assert sha(stage/'NATIVE_OLDEST_ADMISSION_TRIPLET_PLAN_R03_20261002.json')=='bddddc32bc50d678febbe0372be888555d47bf20594ef3cac5e0e9b4990ab08e'
 print(json.dumps(dict(status='R03_STAGED',dedup_paths=len(rows),free_before=before,free_after=shutil.disk_usage('/root').free)),flush=True)
os.chdir(stage)
log=os.open(str(stage/'controller.log'),os.O_CREAT|os.O_WRONLY|os.O_EXCL,0o600)
os.dup2(log,1);os.dup2(log,2);os.close(log)
python='/root/autodl-tmp/moe-a-runtime-20260930/venv/bin/python'
os.execv(python,[python,'-B','run_native_oldest_admission_triplet_r03.py','NATIVE_OLDEST_ADMISSION_TRIPLET_PLAN_R03_20261002.json','--expected-plan-sha256','bddddc32bc50d678febbe0372be888555d47bf20594ef3cac5e0e9b4990ab08e'])

