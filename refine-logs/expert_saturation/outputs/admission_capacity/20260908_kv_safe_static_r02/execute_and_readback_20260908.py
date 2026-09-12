"""Run the frozen four cells serially, downloading each before the next launch."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parent
REMOTE = '/root/autodl-tmp/moe-kv-safe-static-20260908-r02'
PYTHON = '/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python'
HOST = 'root@connect.westd.seetacloud.com'
SSH = ['ssh', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10', '-o',
       'NumberOfPasswordPrompts=1', '-o', 'StrictHostKeyChecking=yes', '-p', '37116', HOST]
CELLS = ['repeat0-baseline16', 'repeat0-safe', 'repeat1-safe', 'repeat1-baseline16']


def remote(code):
    command = shlex.join([PYTHON, '-c', code])
    return json.loads(subprocess.check_output([*SSH, command], text=True, timeout=60))


def save(value):
    path = ROOT / 'execution-state-20260908.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    state_path = ROOT / 'execution-state-20260908.json'
    with state_path.open('x') as output:
        json.dump({'status': 'STARTING', 'local_driver_pid': os.getpid()}, output)
    state = dict(status='RUNNING', local_driver_pid=os.getpid(), started_unix_s=time.time(),
                 remote_root=REMOTE, retained=[], current=None, remote_originals_retained=True)
    local_results = ROOT / 'gpu_results'
    local_results.mkdir(exist_ok=False)
    try:
        for label in CELLS:
            state.update(current=label, stage='LAUNCH_REQUESTED')
            save(state)
            launch = remote(f'''import json,pathlib,subprocess
root=pathlib.Path({REMOTE!r}); label={label!r}
assert not (root/'results'/f'{{label}}-execution.json').exists()
with (root/f'{{label}}-launcher.log').open('x') as log:
 p=subprocess.Popen([{PYTHON!r},'-u','run_one_cell.py',label],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(launcher_pid=p.pid)))''')
            state.update(stage='REMOTE_RUNNING', **launch)
            save(state)
            print(json.dumps(dict(event='LAUNCHED', cell=label, **launch)), flush=True)
            while True:
                time.sleep(15)
                observation = remote(f'''import json,os,pathlib
root=pathlib.Path({REMOTE!r}); label={label!r}; meta=root/'results'/f'{{label}}-execution.json'
m=json.loads(meta.read_text()) if meta.exists() else {{'status':'METADATA_PENDING'}}
try: os.kill({launch['launcher_pid']},0); alive=True
except ProcessLookupError: alive=False
cell=root/'results'/label; status=cell/'status.json'
print(json.dumps(dict(metadata=m,launcher_alive=alive,cell_status=json.loads(status.read_text()) if status.exists() else None,warmups=len(list(cell.glob('warmup-*.json'))))))''')
                print(json.dumps(dict(event='OBSERVED', cell=label, launcher_alive=observation['launcher_alive'], launcher_status=observation['metadata']['status'], cell_status=observation['cell_status'], warmups=observation['warmups'])), flush=True)
                if observation['metadata'].get('status') in ['EXITED', 'LAUNCH_OR_WAIT_FAILED']:
                    break
                if not observation['launcher_alive']:
                    raise RuntimeError(f'{label}: launcher missing without terminal metadata; inspect before continuing')
            state.update(stage='ARCHIVING', last_observation=observation)
            save(state)
            archive_name = f'readback-{label}.tar.gz'
            archive_info = remote(f'''import hashlib,json,pathlib,tarfile
root=pathlib.Path({REMOTE!r}); label={label!r}; archive=root/{archive_name!r}
with tarfile.open(archive,'x:gz') as tar:
 for path in [root/'results'/label,root/'results'/f'{{label}}-execution.json',root/'results'/f'{{label}}.stdout.log',root/'results'/f'{{label}}.stderr.log',root/f'{{label}}-launcher.log']:
  if path.exists():tar.add(path,arcname=path.name)
print(json.dumps(dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size)))''')
            archive = ROOT / archive_name
            if archive.exists():
                raise FileExistsError(archive)
            subprocess.run(['scp', '-q', '-o', 'ConnectTimeout=15', '-o', 'NumberOfPasswordPrompts=1',
                            '-o', 'StrictHostKeyChecking=yes', '-P', '37116',
                            f'{HOST}:{REMOTE}/{archive_name}', str(archive)], check=True, timeout=60)
            assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive_info['sha256']
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    target = (local_results / member.name).resolve()
                    assert target.is_relative_to(local_results.resolve())
                    assert member.isfile() or member.isdir()
                    assert not target.exists()
                tar.extractall(local_results)
            cell_dir = local_results / label
            cell_status = json.loads((cell_dir / 'status.json').read_text())
            warmups = [json.loads(p.read_text()) for p in sorted(cell_dir.glob('warmup-*.json'))]
            raw = json.loads((cell_dir / 'raw.json').read_text()) if (cell_dir / 'raw.json').exists() else None
            record = dict(cell=label, archive=archive_name, **archive_info,
                          status=cell_status, warmup_raw_count=len(warmups),
                          raw_requests=len(raw['requests']) if raw else 0,
                          execution=observation['metadata'])
            state['retained'].append(record)
            state['stage'] = 'LOCAL_READBACK_VERIFIED'
            save(state)
            print(json.dumps(dict(event='LOCAL_READBACK_VERIFIED', **record)), flush=True)
            if observation['metadata'].get('returncode') != 0 or cell_status['status'] != 'COMPLETE':
                raise RuntimeError(f'{label}: unexpected failure archived; stop for diagnosis')
            assert raw is not None and len(raw['requests']) == 32 and len(warmups) == 3
        state.update(status='COMPLETE_LOCAL_READBACK', current=None, finished_unix_s=time.time())
        save(state)
    except BaseException as error:
        state.update(status='PAUSED_FOR_DIAGNOSIS', error=repr(error), observed_unix_s=time.time())
        save(state)
        raise


if __name__ == '__main__':
    main()
