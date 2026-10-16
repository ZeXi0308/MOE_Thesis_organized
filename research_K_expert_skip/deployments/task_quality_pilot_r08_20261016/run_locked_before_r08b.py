#!/usr/bin/env python3
"""Run a bounded job with the workspace's existing shared GPU lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--wait-seconds', type=int, default=0)
    p.add_argument('--root', default='/root/autodl-tmp/research_K_expert_skip')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not 0 <= a.wait_seconds <= 3600:
        raise ValueError('wait-seconds must be 0..3600')
    if not a.name.replace('_', '').replace('-', '').isalnum() or not 1 <= a.timeout <= 14400:
        raise ValueError('invalid run name or timeout')
    root = Path(a.root)
    out = root / 'runs' / a.name
    out.mkdir(parents=True, exist_ok=False)
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not cmd:
        raise ValueError('command is required')
    state = dict(name=a.name, command=cmd, controller_pid=os.getpid(),
                 created_unix=time.time(), timeout_s=a.timeout, state='PREPARING',
                 budget_s=15*3600, lock_path='/root/autodl-tmp/moe-research-gpu.lock')
    def save():
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(out / 'status.json')
    save()
    try:
        elapsed = 0.
        for path in (root / 'runs').glob('*/status.json'):
            row = json.loads(path.read_text())
            if path != out/'status.json' and row.get('state') in ('RUNNING', 'WAITING'):
                raise RuntimeError('Prior RUNNING record requires process check before another submission: '+str(path))
            elapsed += row.get('lock_held_s', 0.)
        if elapsed + a.timeout + 30 > state['budget_s']:
            raise RuntimeError('Requested timeout exceeds remaining 15 hour GPU budget')
        with open(state['lock_path'], 'a') as lock:
            deadline=time.monotonic()+a.wait_seconds
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic()>=deadline:raise
                    state['state']='WAITING';save();time.sleep(.5)
            started = time.monotonic()
            try:
                import pynvml as nv
                nv.nvmlInit()
                gpu = nv.nvmlDeviceGetHandleByIndex(0)
                procs = nv.nvmlDeviceGetComputeRunningProcesses(gpu) + nv.nvmlDeviceGetGraphicsRunningProcesses(gpu)
                if procs:
                    raise RuntimeError('GPU occupied by PIDs '+str(sorted({x.pid for x in procs})))
                state.update(gpu_uuid=nv.nvmlDeviceGetUUID(gpu), gpu_name=nv.nvmlDeviceGetName(gpu),
                             started_unix=time.time(), prior_gpu_seconds=elapsed, state='RUNNING')
                nv.nvmlShutdown()
                shutil.copytree(root/'code', out/'submitted_code', ignore=shutil.ignore_patterns('__pycache__','._*'))
                shutil.copytree(root/'configs', out/'submitted_configs')
                save()
                env = os.environ.copy()
                env['PYTHONPATH'] = str(root/'code') + os.pathsep + env.get('PYTHONPATH', '')
                env['LD_LIBRARY_PATH'] = '/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib' + os.pathsep + env.get('LD_LIBRARY_PATH', '')
                env['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
                env['TOKENIZERS_PARALLELISM'] = 'false'
                env['OMP_NUM_THREADS'] = '4'
                with (out/'stdout.log').open('w') as log:
                    child = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True, pass_fds=(lock.fileno(),))
                    state['worker_pid'] = child.pid
                    save()
                    try:
                        rc = child.wait(timeout=a.timeout)
                    except subprocess.TimeoutExpired:
                        state['timed_out'] = True
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            rc = child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            rc = child.wait()
                state.update(exit_code=rc, state='COMPLETE' if rc == 0 else 'FAILED')
            finally:
                state.update(lock_held_s=time.monotonic()-started, ended_unix=time.time())
                save()
    except BlockingIOError:
        state.update(state='LOCK_BUSY', ended_unix=time.time())
        save()
    except Exception as exc:
        state.update(state='FAILED', error=f'{type(exc).__name__}: {exc}', ended_unix=time.time())
        save()
    hashes = {}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'SHA256.json':
            h=hashlib.sha256()
            with path.open('rb') as f:
                for part in iter(lambda:f.read(1024*1024), b''):h.update(part)
            hashes[str(path.relative_to(out))]=h.hexdigest()
    (out/'SHA256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(state, indent=2), flush=True)
    raise SystemExit(0 if state['state']=='COMPLETE' else 1)


if __name__=='__main__':main()
