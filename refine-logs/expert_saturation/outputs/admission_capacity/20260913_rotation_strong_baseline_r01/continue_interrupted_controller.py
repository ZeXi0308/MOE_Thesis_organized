#!/usr/bin/env python3
"""Recover completed readback and run only the untouched suffix of this campaign."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'execution02_westc_53036'
REMOTE = '/root/autodl-tmp/moe-rotation-strong-baseline-20260913-r01'
HOST = 'root@connect.westc.seetacloud.com'
SOCKET = '/private/tmp/moe-westc-53036.sock'
PYTHON = '/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
SSH = ['ssh', '-p', '53036', '-S', SOCKET, '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4', HOST]
SCP = ['scp', '-P', '53036', '-o', 'ControlPath='+SOCKET, '-o', 'BatchMode=yes']


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def command(argv, **kw): return subprocess.run(argv, check=True, **kw)
def query(code): return json.loads(subprocess.check_output(SSH+[shlex.join([PYTHON, '-c', code])], text=True))
def write(p, obj):
    tmp = p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(p)


def main():
    if not __debug__: raise RuntimeError('Optimized Python disables recovery assertions; refuse execution')
    # This recovery is intentionally restricted to the observed three-cell prefix.
    process = subprocess.run(['ps', '-p', '42366', '-o', 'pid=,command='], capture_output=True, text=True)
    assert process.returncode == 1 and not process.stdout.strip(), 'original controller PID still exists or query failed'
    original = (RUN/'execution.json').read_bytes(); state = json.loads(original)
    labels = state['labels']; frozen = RUN/'frozen'
    manifest = read(frozen/'campaign.json')
    assert state['host'] == HOST and state['port'] == '53036' and state['remote_dir'] == REMOTE
    assert state['status'] == 'RUNNING' and len(state['cells']) == 3
    assert [x['label'] for x in state['cells']] == labels[:3] == [x['label'] for x in manifest['cells'][:3]]
    assert [x['status'] for x in state['cells']] == ['READ_BACK', 'READ_BACK', 'RUNNING']
    assert labels == [x['label'] for x in manifest['cells']] and len(labels) == 8
    assert sha(ROOT/'preparation/execution.tar.gz') == sha(RUN/'execution.tar.gz') == state['archive_sha256']
    with tarfile.open(RUN/'execution.tar.gz') as archive:
        assert all((frozen/m.name).read_bytes() == archive.extractfile(m).read() for m in archive.getmembers() if m.isfile())
    for cell in state['cells'][:2]:
        assert sha(RUN/(cell['label']+'.tar.gz')) == cell['archive_sha256']
        assert read(RUN/'gpu_results'/cell['label']/'status.json')['status'] == 'COMPLETE'
    check = query(f'''import json,pathlib,tarfile,hashlib,subprocess
r=pathlib.Path({REMOTE!r}); rows=[]
for p in sorted((r/'results').glob('*-execution.json')):
 d=json.loads(p.read_text()); d['any_pid_live']=any((pathlib.Path('/proc')/str(d[k])).exists() for k in ['launcher_pid','child_pid']);rows.append(d)
with tarfile.open(r/'execution.tar.gz') as a:
 same=all((r/m.name).read_bytes()==a.extractfile(m).read() for m in a.getmembers() if m.isfile())
print(json.dumps(dict(rows=rows,source_matches=same,archive_sha256=hashlib.sha256((r/'execution.tar.gz').read_bytes()).hexdigest(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),result_names=sorted(p.name for p in (r/'results').iterdir()),archive_names=sorted(p.name for p in r.glob('*.tar.gz')))))''')
    assert check['source_matches'] and check['archive_sha256'] == state['archive_sha256']
    assert check['gpu'] == state['gpu_uuid'] and not check['processes'], 'GPU identity differs or busy'
    rows = {c['label']:c for c in check['rows']}
    assert set(rows) == set(labels[:3])
    assert all(c['status']=='EXITED' and c['returncode']==0 and not c['any_pid_live'] for c in rows.values())
    assert all(not any(name == label or name.startswith(label+'.') or name.startswith(label+'-') for name in check['result_names']) for label in labels[3:])
    assert all(not (RUN/'gpu_results'/label).exists() and not (RUN/(label+'.ssh.log')).exists() for label in labels[3:])
    assert not (RUN/'gpu_results'/labels[2]).exists(), 'third local readback already exists'
    assert all(label+'.tar.gz' not in check['archive_names'] and not (RUN/(label+'.tar.gz')).exists() for label in labels[2:]), 'orphan archive in pending readback/suffix; inspect before recovery'
    recovery = RUN/'controller_recovery'/str(time.time_ns()); recovery.mkdir(parents=True)
    (recovery/'execution_before.json').write_bytes(original)
    old_lock = RUN/'.driver.lock'
    lock_contents = old_lock.read_text()
    old_attempt = Path(lock_contents.strip())/'attempt.json'
    assert old_attempt.parent.parent == RUN/'attempts'
    (recovery/'driver_lock_before.txt').write_text(lock_contents)
    (recovery/'driver_attempt_before.json').write_bytes(old_attempt.read_bytes())
    write(recovery/'remote_precheck.json', check)
    receipt = dict(status='RECOVERING', pid=os.getpid(), started_unix_s=time.time(), script_sha256=sha(Path(__file__)), original_controller_pid=42366, recover_label=labels[2], untouched_suffix=labels[3:], reason='User turn interruption ended local controller; third remote cell completed successfully. No cell is rerun.')
    write(recovery/'recovery.json', receipt)
    with (RUN/'.recovery.lock').open('x') as stream: stream.write(str(os.getpid())+'\n')
    try:
        assert (RUN/'execution.json').read_bytes() == original, 'another writer changed execution state'
        old_lock.rename(recovery/'orphaned_driver.lock')
        with old_lock.open('x') as stream: stream.write(str(recovery)+'\n')
        state['controller_recovery'] = str(recovery.relative_to(RUN))
        write(RUN/'execution.json', state)
        for index in range(2, 8):
            label = labels[index]
            if index == 2:
                cell = state['cells'][index]
                cell.update(returncode=0, finished_unix_s=rows[label]['finished_unix_s'], recovered_from_remote_terminal=True)
            else:
                # Refuse any competing or partially started cell; no automatic retry.
                gate = query(f'''import pathlib,json,subprocess
r=pathlib.Path({REMOTE!r})/'results'; label={label!r}
print(json.dumps(dict(activity=any(p.name==label or p.name.startswith(label+'.') or p.name.startswith(label+'-') for p in r.iterdir()),archive_exists=(r.parent/(label+'.tar.gz')).exists(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())))''')
                write(recovery/(label+'-boundary.json'), gate)
                assert not gate['activity'] and not gate['archive_exists'] and not (RUN/(label+'.tar.gz')).exists() and not gate['processes'], 'cell/archive already exists or GPU busy'
                cell = dict(label=label, status='RUNNING', started_unix_s=time.time())
                state['cells'].append(cell); write(RUN/'execution.json', state)
                remote_command = 'cd '+shlex.quote(REMOTE)+' && '+shlex.join([PYTHON, '-u', 'run_one_cell.py', label])
                with (RUN/(label+'.ssh.log')).open('x') as log:
                    result = subprocess.run(SSH+[remote_command], stdout=log, stderr=subprocess.STDOUT)
                cell.update(returncode=result.returncode, finished_unix_s=time.time()); write(RUN/'execution.json', state)
                if result.returncode == 255: raise RuntimeError('SSH observation lost; inspect original remote PID before any recovery')
            filename = label+'.tar.gz'
            remote_archive_exists = query(f"import json,pathlib;print(json.dumps(pathlib.Path({(REMOTE+'/'+filename)!r}).exists()))")
            local = RUN/filename
            assert not remote_archive_exists and not local.exists(), 'refusing archive reuse or overwrite'
            command(SSH+[shlex.join(['tar', '-czf', REMOTE+'/'+filename, '-C', REMOTE+'/results', label, label+'-execution.json', label+'.stdout.log', label+'.stderr.log'])])
            expected = query(f"import json,pathlib,hashlib;print(json.dumps(hashlib.sha256(pathlib.Path({(REMOTE+'/'+filename)!r}).read_bytes()).hexdigest()))")
            command(SCP+[HOST+':'+REMOTE+'/'+filename, str(local)])
            assert sha(local) == expected
            assert not (RUN/'gpu_results'/label).exists(), 'refusing readback overwrite'
            with tarfile.open(local) as archive: archive.extractall(RUN/'gpu_results', filter='data')
            terminal = read(RUN/'gpu_results'/label/'status.json')
            cell.update(status='READ_BACK', archive_sha256=expected, terminal=terminal); write(RUN/'execution.json', state)
            print(label, terminal, flush=True)
            if cell['returncode'] or terminal['status'] != 'COMPLETE': raise RuntimeError('cell failed; preserved results, suffix not launched')
        state['status'] = receipt['status'] = 'COMPLETE'
    except BaseException as error:
        state.update(status='STOPPED', recovery_error=repr(error)); receipt.update(status='STOPPED', error=repr(error)); raise
    finally:
        state['updated_unix_s'] = receipt['finished_unix_s'] = time.time()
        write(RUN/'execution.json', state); write(recovery/'recovery.json', receipt)
        if old_lock.exists() and old_lock.read_text().strip()==str(recovery): old_lock.unlink()
        (RUN/'.recovery.lock').unlink()


if __name__ == '__main__': main()
