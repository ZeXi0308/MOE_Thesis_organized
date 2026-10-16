#!/usr/bin/env python3
"""One-shot serial continuation; each predecessor must be locally archived.

This submits existing plans unchanged to frozen code-r12. It does not retry,
clean data, or modify another process. Local archive acknowledgments are written
only after downloading and checking the archive's SHA256 and byte count.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/research_A_horizon')
NAMES = ['olmoe48-w2', 'olmoe24-w1', 'olmoe48-w1', 'olmoe24-w3', 'olmoe48-w3']
PREDECESSOR = 'olmoe24-w2-corrected-frontier'
PREDECESSOR_PID = 30031
PREDECESSOR_START_TICKS = '2655066999'
BUDGET_SECONDS = 12 * 3600


def read(path):
    return json.loads(path.read_text())


def same_process(pid, ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] == ticks and fields[0] != 'Z'
    except FileNotFoundError:
        return False


def archive_barrier(name):
    session = ROOT / 'runs' / name
    receipt = read(session / 'receipt.json')
    if receipt['status'] != 'COMPLETE':
        raise RuntimeError(f'{name}: predecessor is {receipt["status"]}')
    archive = read(session / 'archive.json')
    print(f'WAITING_LOCAL_ARCHIVE {name} {archive["sha256"]}', flush=True)
    ack = session / 'local-archive-verified.json'
    while not ack.exists():
        time.sleep(15)
    verified = read(ack)
    if any(verified[k] != archive[k] for k in ('bytes', 'sha256')):
        raise RuntimeError(f'{name}: local archive acknowledgment mismatch')


def spent_seconds():
    total = 0.0
    for path in (ROOT / 'runs').glob('*/receipt.json'):
        r = read(path)
        if 'acquired_unix_s' not in r:
            continue
        if 'end_unix_s' not in r:
            raise RuntimeError(f'Other HORIZON group still active: {path.parent.name}')
        total += r['end_unix_s'] - r['acquired_unix_s']
    return total


def main():
    # Exclusive claim prevents accidentally launching this continuation twice.
    with (ROOT / 'runs/olmoe-matrix-continuation.claim').open('x') as f:
        import os
        json.dump({'pid': os.getpid(), 'plans': NAMES}, f)
    print(f'WAITING_PROCESS {PREDECESSOR_PID}', flush=True)
    while same_process(PREDECESSOR_PID, PREDECESSOR_START_TICKS):
        time.sleep(15)
    archive_barrier(PREDECESSOR)
    for name in NAMES:
        plan_path = ROOT / 'configs' / (name + '-corrected-frontier.json')
        plan = read(plan_path)
        actual_hash = hashlib.sha256(Path(plan['inputs']).read_bytes()).hexdigest()
        if actual_hash != plan['provenance']['inputs_sha256']:
            raise RuntimeError(f'{name}: input SHA mismatch')
        session = Path(plan['session'])
        if session.exists():
            raise RuntimeError(f'{name}: session already exists; no automatic retry')
        # Reserve each cell's timeout plus termination grace, not its best time.
        worst_case = len(plan['cells']) * (plan.get('per_cell_seconds', 450) + 20)
        spent = spent_seconds()
        if spent + worst_case > BUDGET_SECONDS:
            raise RuntimeError(f'GPU budget insufficient: spent={spent}, reserve={worst_case}')
        print(f'START {name} spent={spent:.3f} reserve={worst_case}', flush=True)
        subprocess.run([sys.executable, '-B', str(ROOT / 'code-r12/run_group.py'),
                        '--plan', str(plan_path), '--wait-lock-seconds', '1800'], check=True)
        archive_barrier(session.name)
    print('COMPLETE OLMoE matrix continuation', flush=True)


if __name__ == '__main__':
    main()
