#!/usr/bin/env python3
"""One bounded native health run; uses an existing shared lock, never recreates it."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
GPU = 'GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e'
LOCK = Path('/root/autodl-tmp/moe-research-gpu.lock')
LOCK_ID = (2304, 25841682495)
STAGE = Path('/dev/shm/c-qwen-qualified-20261001-v1')
OUT = BASE / 'qwen-health-v1'
GIB = 1024 ** 3
SPAWNING, PENDING_SIGNAL = False, None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def gpu_idle():
    args = ['nvidia-smi', '-i', GPU, '--format=csv,noheader,nounits']
    info = subprocess.check_output(args + ['--query-gpu=uuid,memory.used'], text=True, timeout=15).strip()
    procs = subprocess.check_output(args + ['--query-compute-apps=pid,used_gpu_memory'], text=True, timeout=15).strip()
    require(info.split(',')[0].strip() == GPU and int(info.split(',')[1]) <= 64 and not procs,
            'GPU unavailable; never terminate another task')
    return {'gpu': info, 'compute_processes': procs}


def bounded(command, log_name, seconds, lock_fd):
    global SPAWNING
    record = {'command': command, 'timeout_s': seconds, 'start_unix_s': time.time()}
    with (OUT / log_name).open('x') as log:
        child = None
        try:
            SPAWNING = True
            try:
                child = subprocess.Popen(command, cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(lock_fd,))
            finally:
                SPAWNING = False
            record['pid'] = child.pid
            if PENDING_SIGNAL is not None:
                raise InterruptedError('signal during child spawn: ' + str(PENDING_SIGNAL))
            save(OUT / (log_name + '.lifecycle.json'), record)
            record['returncode'] = child.wait(timeout=seconds)
        except BaseException:
            # This process group was created here and contains only this run.
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=20)
            record['interrupted'] = True
            raise
        finally:
            record.update(end_unix_s=time.time(), returncode=child.poll() if child else None)
            save(OUT / (log_name + '.lifecycle.json'), record)
    require(record['returncode'] == 0, log_name + ' failed')


def stage_model():
    from range_download_v2 import download_ranged, memory_headroom
    m = json.loads((BASE / 'model_manifest.json').read_text())
    require(m['model_id'] == 'Qwen/Qwen2.5-1.5B-Instruct' and
            m['revision'] == '989aa7980e4cf806f80c7fef2b1adb7bc71aa306', 'model identity differs')
    total = sum(r['size'] for r in m['files'])
    require(shutil.disk_usage(STAGE.parent).free > total + GIB and
            memory_headroom() > total + 24 * GIB, 'insufficient RAM headroom')
    STAGE.mkdir(mode=0o700, exist_ok=False)
    receipt = {'status': 'DOWNLOADING', 'files': [], 'model_id': m['model_id'], 'revision': m['revision']}
    save(OUT / 'model-download.json', receipt)
    deadline = time.monotonic() + 900
    for item in m['files']:
        name = item['filename']
        require(Path(name).name == name, 'invalid model filename')
        partial = STAGE / (name + '.part')
        if name.endswith('.safetensors'):
            url = 'https://hf-mirror.com/' + m['model_id'] + '/resolve/' + m['revision'] + '/' + name
            progress = 0
            def on_chunk(count):
                nonlocal progress
                progress += count
                print(json.dumps({'filename': name, 'received_bytes': progress, 'expected_bytes': item['size']}), flush=True)
            result = download_ranged(url, partial, item['size'], item['sha256'], deadline,
                                     progress_callback=on_chunk)
        else:
            src = BASE / 'qwen_metadata' / name
            require(src.stat().st_size == item['size'] and sha(src) == item['sha256'], 'metadata differs')
            with src.open('rb') as source, partial.open('xb') as target:
                shutil.copyfileobj(source, target)
            result = {'received_bytes': partial.stat().st_size, 'sha256': sha(partial)}
        require(result['received_bytes'] == item['size'] and result['sha256'] == item['sha256'], 'model file differs')
        partial.rename(STAGE / name)
        receipt['files'].append(dict(item, status='VERIFIED'))
        save(OUT / 'model-download.json', receipt)
    receipt['status'] = 'VERIFIED'
    save(OUT / 'model-download.json', receipt)


def main():
    frozen = json.loads((BASE / 'freeze.json').read_text())
    for name, digest in frozen['files_sha256'].items():
        require(sha(BASE / name) == digest, 'frozen file changed: ' + name)
    require(not OUT.exists() and not STAGE.exists(), 'existing output/stage; no automatic retry')
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        require(stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino) == LOCK_ID, 'shared lock identity differs')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir()
        receipt = {'status': 'RUNNING', 'start_unix_s': time.time(), 'launcher_pid': os.getpid(),
                   'freeze_sha256': sha(BASE / 'freeze.json'), 'shared_lock_inode': LOCK_ID}
        try:
            receipt['gpu_before'] = gpu_idle()
            save(OUT / 'launcher-receipt.json', receipt)
            require(shutil.disk_usage(BASE).free > 2 * GIB, 'insufficient root disk reserve')
            bounded([sys.executable, str(Path(__file__)), '--stage'], 'download.log', 960, fd)
            gpu_idle()
            bounded([sys.executable, str(BASE / 'health_native.py'), '--model-dir', str(STAGE),
                     '--inputs', str(BASE / 'workload.json'), '--output', str(OUT / 'native'),
                     '--expected-gpu-uuid', GPU, '--kv-bytes', '8589934592', '--max-seqs', '32',
                     '--batch-tokens', '1024', '--max-model-len', '4096'], 'native.log', 1200, fd)
            status = json.loads((OUT / 'native' / 'status.json').read_text())
            require(status['status'] == 'COMPLETE', 'native run incomplete')
            receipt['gpu_after'] = gpu_idle()
            receipt['status'] = 'COMPLETE'
        except BaseException as exc:
            receipt.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            receipt.update(end_unix_s=time.time(), stage_retained=str(STAGE),
                           scope='16-request native quality screen only; no policy comparison')
            save(OUT / 'launcher-receipt.json', receipt)
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, _frame):
        global PENDING_SIGNAL
        if SPAWNING:
            PENDING_SIGNAL = signum
            return
        raise InterruptedError('launcher received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    stage_model() if args.stage else main()
