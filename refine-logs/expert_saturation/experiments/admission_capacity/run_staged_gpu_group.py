"""Run one untouched staged group under the shared remote advisory flock.

Stage with run_frozen_kv_remote.py first. No retries or automatic recovery.
The remote child inherits the lock descriptor across controller interruption.
"""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

REMOTE = '''import fcntl,hashlib,json,os,pathlib,subprocess,sys,tarfile,time
root=pathlib.Path.cwd()
record=dict(status="STARTING",pid=os.getpid(),started_unix_s=time.time(),cells=[])
def save():
 p=root/"group-execution.json.tmp";p.write_text(json.dumps(record,indent=2)+"\\n");p.replace(root/"group-execution.json")
with (root/"group-once.lock").open("x") as once:
 once.write(str(os.getpid()))
with pathlib.Path("/root/autodl-tmp/moe-research-gpu.lock").open("a+") as lock:
 try:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  record["shared_lock_acquired_unix_s"]=time.time();save()
  assert not (root/"results").exists() or not any((root/"results").iterdir()),"existing results; no rerun"
  with tarfile.open(root/"execution.tar.gz") as archive:
   assert all((root/m.name).read_bytes()==archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()),"source changed"
  cells=json.loads((root/"campaign.json").read_text())["cells"]
  record["status"]="RUNNING";save()
  for cell in cells:
   gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
   assert gpu==sys.argv[1],"GPU UUID changed"
   processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
   assert not processes,"GPU busy; abort group"
   row=dict(label=cell["label"],status="RUNNING",started_unix_s=time.time(),gpu_uuid=gpu,gpu_processes=processes)
   record["cells"].append(row);save()
   rc=subprocess.run([sys.executable,"-u","run_one_cell.py",cell["label"]],pass_fds=(lock.fileno(),),start_new_session=True).returncode
   terminal_path=root/"results"/cell["label"]/"status.json"
   row.update(returncode=rc,finished_unix_s=time.time(),terminal=json.loads(terminal_path.read_text()) if terminal_path.exists() else {})
   row["status"]="COMPLETE" if rc==0 and row["terminal"].get("status")=="COMPLETE" else "INCOMPLETE";save()
   assert row["status"]=="COMPLETE","cell failed; preserve and stop"
  record["status"]="COMPLETE"
 except BaseException as exc:
  record.update(status="STOPPED",error=f"{type(exc).__name__}: {exc}");raise
 finally:
  record["finished_unix_s"]=time.time();save()
'''


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: run_staged_gpu_group.py EXECUTION_DIR SSH_CONTROL_PATH')
    out, control = Path(sys.argv[1]).resolve(), sys.argv[2]
    record = json.loads((out/'execution.json').read_text())
    assert record['status'] == 'STAGED' and not record['cells'], 'untouched STAGED group required'
    assert not any((out/'gpu_results').iterdir()), 'local results exist'
    ssh = ['ssh', '-p', str(record['port']), '-S', control, '-o', 'BatchMode=yes',
           '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4', record['host']]
    scp = ['scp', '-P', str(record['port']), '-o', 'ControlPath='+control, '-o', 'BatchMode=yes']
    remote, py = record['remote_dir'], '/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
    def query(code):
        return json.loads(subprocess.check_output(ssh+[shlex.join([py, '-c', code])], text=True))
    def save():
        temp = out/'execution.json.tmp'
        temp.write_text(json.dumps(record, indent=2)+'\n'); temp.replace(out/'execution.json')
    local_lock = out/'.driver.lock'
    with local_lock.open('x') as f:
        f.write(str(Path(__file__).resolve()))
    try:
        digest = hashlib.sha256((out/'execution.tar.gz').read_bytes()).hexdigest()
        assert digest == record['archive_sha256'], 'local archive changed'
        with tarfile.open(out/'execution.tar.gz') as archive:
            assert all((out/'frozen'/m.name).read_bytes() == archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()), 'local frozen source changed'
            assert record['labels'] == [c['label'] for c in json.load(archive.extractfile('campaign.json'))['cells']], 'label order changed'
        verify = query(f'''import hashlib,json,pathlib,subprocess
r=pathlib.Path({remote!r});print(json.dumps(dict(sha256=hashlib.sha256((r/'execution.tar.gz').read_bytes()).hexdigest(),started=(r/'group-once.lock').exists(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())))''')
        assert verify['sha256'] == digest and not verify['started'] and not verify['processes'], 'staged mismatch, activity or GPU busy'
        record.update(group_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      remote_group_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),group_preflight=verify)
        payload = base64.b64encode(REMOTE.encode()).decode()
        uploaded = query(f'''import base64,hashlib,json,pathlib
p=pathlib.Path({remote!r})/'run_gpu_group.py';b=base64.b64decode({payload!r});p.write_bytes(b) if not p.exists() else None;assert p.read_bytes()==b;print(json.dumps(hashlib.sha256(b).hexdigest()))''')
        assert uploaded == record['remote_group_sha256']
        record['status'] = 'RUNNING'; save()
        with (out/'group.ssh.log').open('x') as log:
            rc = subprocess.run(ssh+['cd '+shlex.quote(remote)+' && '+shlex.join([py, '-u', 'run_gpu_group.py', record['gpu_uuid']])], stdout=log, stderr=subprocess.STDOUT).returncode
        record['group_ssh_returncode'] = rc
        if rc == 255:
            raise RuntimeError('SSH disconnected; inspect original group before any recovery')
        manifest = query(f'''import hashlib,json,pathlib,tarfile
r=pathlib.Path({remote!r});p=r/'group-readback.tar.gz'
assert not p.exists(),'readback already exists'
with tarfile.open(p,'w:gz') as a:
 for n in ['group-execution.json','run_gpu_group.py','results']:
  if (r/n).exists():a.add(r/n,arcname=n)
print(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),group=json.loads((r/'group-execution.json').read_text()))))''')
        archive = out/'group-readback.tar.gz'
        subprocess.run(scp+[record['host']+':'+remote+'/group-readback.tar.gz', str(archive)], check=True)
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['sha256'], 'readback hash mismatch'
        with tarfile.open(archive) as bundle:
            bundle.extractall(out/'group_readback', filter='data')
        retained_results = out/'group_readback'/'results'
        for item in retained_results.iterdir() if retained_results.exists() else ():
            (out/'gpu_results'/item.name).symlink_to(Path('../group_readback/results')/item.name, target_is_directory=item.is_dir())
        record.update(group_readback_sha256=manifest['sha256'], cells=manifest['group']['cells'],
                      status='COMPLETE' if rc == 0 and manifest['group']['status'] == 'COMPLETE' else 'STOPPED')
        print(json.dumps(dict(status=record['status'], cells=record['cells'])), flush=True)
    except BaseException as exc:
        record.update(status='STOPPED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['updated_unix_s'] = time.time(); save(); local_lock.unlink()


if __name__ == '__main__':
    main()
