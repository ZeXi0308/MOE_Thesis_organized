#!/usr/bin/env python3
"""Enter the shared kernel flock wait queue with the existing finite deadline."""
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent
PARENT_SHA256 = '6846e19f8582f7b0d1a6fa83f7c8a3cf84aab6daf9ee69437de07d8b754fa6cf'


def acquire_gpu_lock(fd, wait_seconds, receipt, receipt_path, write):
    started = time.monotonic()
    waiting = False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if wait_seconds == 0:
            raise
        waiting = True
        receipt.update(status='WAIT_GPU', pid=os.getpid(), lock_wait=dict(
            started_unix_s=time.time(), timeout_seconds=wait_seconds))
        write(receipt_path, receipt)
        remaining = wait_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f'GPU lock wait exceeded {wait_seconds:g} seconds')
        previous = signal.getsignal(signal.SIGALRM)

        def timed_out(signum, frame):
            raise TimeoutError(f'GPU lock wait exceeded {wait_seconds:g} seconds')

        signal.signal(signal.SIGALRM, timed_out)
        try:
            signal.setitimer(signal.ITIMER_REAL, remaining)
            fcntl.flock(fd, fcntl.LOCK_EX)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    receipt.update(status='RUNNING', acquired_unix_s=time.time())
    if waiting:
        receipt['lock_wait'].update(acquired_unix_s=time.time(),
            elapsed_seconds=time.monotonic()-started)
    write(receipt_path, receipt)


def main():
    path = ROOT/'run_fixed_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA256:
        raise RuntimeError('Frozen fixed-output controller source changed')
    spec = importlib.util.spec_from_file_location('frozen_fixed_controller', path)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    namespace = dict(__name__='fixed_completion_wait_controller',
                     __file__=str(ROOT/'run_group.py'))
    exec(compile(parent.adapted_source(), str(ROOT/'run_group.py')+'[fixed-inputs]', 'exec'), namespace)
    namespace['acquire_gpu_lock'] = lambda fd, seconds, receipt, receipt_path: acquire_gpu_lock(
        fd, seconds, receipt, receipt_path, namespace['write'])
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
