#!/usr/bin/env python3
"""One bounded job, using the existing shared GPU lock; retain failed runs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--root', default='/root/autodl-tmp/research_C_format_sched')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    root = Path(a.root)
    out = root / 'runs' / a.name
    out.mkdir(parents=True, exist_ok=False)
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    record = {'name': a.name, 'command': cmd, 'controller_pid': os.getpid(),
              'created_utc_s': time.time(), 'timeout_s': a.timeout, 'state': 'PREPARING'}
    def save():
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(out / 'status.json')
    save()
    try:
        elapsed = 0.
        for f in (root / 'runs').glob('*/status.json'):
            d = json.loads(f.read_text())
            elapsed += d.get('lock_held_s', 0.)
        if elapsed + a.timeout > 20 * 3600:
            raise RuntimeError('Would exceed 20 hour GPU budget with this job timeout')
        with open('/root/autodl-tmp/moe-research-gpu.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            import pynvml as n
            n.nvmlInit()
            h = n.nvmlDeviceGetHandleByIndex(0)
            procs = n.nvmlDeviceGetComputeRunningProcesses(h) + n.nvmlDeviceGetGraphicsRunningProcesses(h)
            if procs:
                raise RuntimeError(f'GPU processes exist: {[x.pid for x in procs]}')
            record['gpu_uuid'] = n.nvmlDeviceGetUUID(h)
            record['started_utc_s'] = time.time()
            record['prior_gpu_budget_seconds'] = elapsed
            record['state'] = 'RUNNING'
            shutil.copytree(root / 'code', out / 'submitted_code', ignore=shutil.ignore_patterns('__pycache__'))
            save()
            start = time.perf_counter()
            try:
                with (out / 'stdout.log').open('w') as log:
                    process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=30', str(a.timeout), *cmd],
                                               stdout=log, stderr=subprocess.STDOUT, cwd=root)
                    record['worker_pid'] = process.pid
                    save()
                    rc = process.wait()
                record.update(exit_code=rc, state='COMPLETE' if rc == 0 else 'FAILED')
            finally:
                record['lock_held_s'] = time.perf_counter() - start
                record['ended_utc_s'] = time.time()
                save()
    except Exception as e:
        record.update(state='FAILED', error=f'{type(e).__name__}: {e}', ended_utc_s=time.time())
        save()
    entries = {}
    for f in sorted(out.rglob('*')):
        if f.is_file() and f.name != 'SHA256.json':
            entries[str(f.relative_to(out))] = hashlib.sha256(f.read_bytes()).hexdigest()
    (out / 'SHA256.json').write_text(json.dumps(entries, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record['state'] == 'COMPLETE' else 1)


if __name__ == '__main__':
    main()
