#!/usr/bin/env python3
"""Q1 only after the frozen Q0 health screen; reuses verified model bytes."""
import fcntl
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

import health_launch as resource
from health_analyze_v2 import analyze

BASE = Path(__file__).resolve().parent
OUT = BASE / 'qwen-summary-serial-diagnostic-v1'


def main():
    freeze = json.loads((BASE / 'summary_serial_freeze_v1.json').read_text())
    for name, digest in freeze['files_sha256'].items():
        resource.require(resource.sha(BASE / name) == digest, 'frozen file differs: ' + name)
    resource.require(not OUT.exists(), 'existing Q1 output; no automatic retry')
    previous = json.loads((BASE / 'qwen-health-v1/launcher-receipt.json').read_text())
    resource.require(previous['status'] == 'COMPLETE', 'Q0 lifecycle not complete')
    gate = analyze(BASE / 'qwen-health-v1/native', BASE / 'workload.json')
    resource.require(gate['validity'] == 'COMPLETE' and
                     gate['screening'] == 'PASS_DEVELOPMENT_SCREEN', 'Q0 health screen did not pass')
    n32 = json.loads((BASE / 'qwen_q1_quality_result_v1.json').read_text())
    resource.require(n32['validity'] == 'COMPLETE' and n32['qualification'] == 'FAIL_DEVELOPMENT_SCREEN', 'requires retained valid N32 failure')
    fd = os.open(resource.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        resource.require(stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino) == resource.LOCK_ID,
                         'shared lock identity differs')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir()
        resource.OUT = OUT  # Only this process's log destination; no source mutation.
        receipt = {'status': 'RUNNING', 'start_unix_s': time.time(), 'launcher_pid': os.getpid(),
                   'freeze_sha256': resource.sha(BASE / 'summary_serial_freeze_v1.json'),
                   'lock_inode': resource.LOCK_ID}
        try:
            receipt['gpu_before'] = resource.gpu_idle()
            resource.save(OUT / 'launcher-receipt.json', receipt)
            resource.save(OUT / 'q0-health-gate.json', gate)
            manifest = json.loads((BASE / 'model_manifest.json').read_text())
            checked = []
            resource.require(resource.STAGE.is_dir() and not resource.STAGE.is_symlink(), 'model stage missing')
            for item in manifest['files']:
                p = resource.STAGE / item['filename']
                resource.require(not p.is_symlink() and p.stat().st_size == item['size'] and
                                 resource.sha(p) == item['sha256'], 'staged model differs: ' + item['filename'])
                checked.append(item)
            resource.save(OUT / 'model-reverified.json', dict(status='VERIFIED', files=checked,
                model_id=manifest['model_id'], revision=manifest['revision']))
            resource.gpu_idle()
            resource.bounded([sys.executable, str(BASE / 'health_native_summary_serial_v1.py'),
                '--inputs', str(BASE / 'summary_prepare_v1/summary_inputs_v1.json'),
                '--output', str(OUT / 'native'), '--model-dir', str(resource.STAGE),
                '--expected-gpu-uuid', resource.GPU], 'native.log', 1200, fd)
            result = json.loads((OUT / 'native/status.json').read_text())
            resource.require(result['status'] == 'COMPLETE', 'Q1 native run incomplete')
            receipt.update(status='COMPLETE', gpu_after=resource.gpu_idle())
        except BaseException as exc:
            receipt.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            receipt.update(end_unix_s=time.time(), scope='N1 concurrency quality diagnostic; does not replace N32 failure',
                           model_stage_retained=str(resource.STAGE))
            resource.save(OUT / 'launcher-receipt.json', receipt)
    finally:
        os.close(fd)


def interrupted(signum, _frame):
    if resource.SPAWNING:
        resource.PENDING_SIGNAL = signum
        return
    raise InterruptedError('summary launcher received signal ' + str(signum))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
