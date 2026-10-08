"""One foreground, 3600-second wait on the existing whole-group flock.

The frozen experiment and its group deadline are unchanged. No retries or
CUDA initialization occur while waiting; normal run_native_host_group handles release.
"""
import fcntl
import os
import signal
import time

import run_native_host_group


original_flock = fcntl.flock


def acquire_once(fd, operation):
    if operation != (fcntl.LOCK_EX | fcntl.LOCK_NB):
        return original_flock(fd, operation)
    started = time.monotonic()
    print(f'WAIT_SHARED_GPU_LOCK controller_pid={os.getpid()} limit_seconds=3600', flush=True)
    signal.alarm(3600)
    try:
        original_flock(fd, fcntl.LOCK_EX)
    except TimeoutError:
        print('WAIT_SHARED_GPU_LOCK_TIMEOUT no_retry=true', flush=True)
        raise
    finally:
        signal.alarm(0)
    print(f'ACQUIRED_SHARED_GPU_LOCK waited_seconds={time.monotonic()-started:.3f}', flush=True)


if __name__ == '__main__':
    fcntl.flock = acquire_once
    raise SystemExit(run_native_host_group.main())
