#!/usr/bin/env python3
"""Wait only for observed shard-preparation PID 4632, then attempt serial diagnostic once."""
import json
import os
from pathlib import Path
import runpy
import time

BASE = Path(__file__).resolve().parent
STATUS = BASE / 'summary-serial-queue-v1.json'
PID, START_TICKS = 4632, 1204797297


def same_live_process():
    try:
        raw = Path(f'/proc/{PID}/stat').read_text()
    except FileNotFoundError:
        return False
    fields = raw[raw.rfind(')') + 2:].split()
    return int(fields[19]) == START_TICKS and fields[0] not in ('Z', 'X')


def save(row):
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2) + '\n')
    tmp.replace(STATUS)


if __name__ == '__main__':
    if STATUS.exists():
        raise RuntimeError('queue already exists; do not restart it')
    row = dict(status='WAITING_KNOWN_PROCESS', queue_pid=os.getpid(),
               waited_pid=PID, waited_start_ticks=START_TICKS,
               started_unix_s=time.time(), wait_timeout_s=1200,
               scope='No lock held while waiting; one nonblocking serial diagnostic attempt after observed prep ends')
    save(row)
    deadline = time.monotonic() + 1200
    try:
        while same_live_process():
            if time.monotonic() >= deadline:
                raise TimeoutError('known prep still alive; serial diagnostic not launched')
            time.sleep(15)
        row.update(status='STARTING_SERIAL_DIAGNOSTIC_ONCE', wait_ended_unix_s=time.time())
        save(row)
        # The frozen launcher revalidates Q0, model bytes, lock and GPU itself.
        # It fails if another task acquired the lock; this queue never retries.
        runpy.run_path(str(BASE / 'summary_serial_launch_v1.py'), run_name='__main__')
        row['status'] = 'SERIAL_DIAGNOSTIC_RETURNED_SUCCESSFULLY'
    except BaseException as exc:
        row.update(status='STOPPED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        row['ended_unix_s'] = time.time()
        save(row)
