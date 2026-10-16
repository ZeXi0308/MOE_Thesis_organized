#!/usr/bin/env python3
"""Continue six unstarted frontiers after the existing native Qwen group.

No Qwen resubmission, retries, or modification of historical sessions.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from continue_olmoe_matrix import (
    ROOT, archive_barrier, spent_seconds, BUDGET_SECONDS, same_process,
)


def verify_snapshot(manifest):
    for name, digest in manifest['code']['files'].items():
        path = ROOT / 'code-r13' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen source mismatch: {name}')


def main():
    manifest = json.loads((ROOT/'configs/matrix-r13-sha.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == manifest['controller_sha256']
    assert not same_process(34979, '2655225332'), 'Old coordinator must be terminal'
    verify_snapshot(manifest)
    with (ROOT/'runs/matrix-r13.claim').open('x') as f:
        json.dump(dict(pid=os.getpid(), plans=list(manifest['plans'])), f)
    print('WAITING_EXISTING_QWEN 34980', flush=True)
    while same_process(34980, '2655225335'):
        time.sleep(15)
    archive_barrier('qwen-calibration-r11b')
    for name, hashes in manifest['plans'].items():
        verify_snapshot(manifest)
        plan_path = ROOT/'configs'/f'{name}.json'
        assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == hashes['plan_sha256']
        plan = json.loads(plan_path.read_text())
        assert hashlib.sha256(Path(plan['inputs']).read_bytes()).hexdigest() == hashes['inputs_sha256']
        session = Path(plan['session'])
        assert not session.exists(), f'{name}: existing session, no automatic retry'
        reserve = len(plan['cells']) * (plan.get('per_cell_seconds', 450) + 20)
        spent = spent_seconds()
        assert spent + reserve <= BUDGET_SECONDS, (spent, reserve)
        print(f'START {name} code-r13 spent={spent:.3f} reserve={reserve}', flush=True)
        subprocess.run([sys.executable, '-B', str(ROOT/'code-r13/run_group.py'),
                        '--plan', str(plan_path), '--wait-lock-seconds', '21600'], check=True)
        archive_barrier(session.name)
    print('COMPLETE matrix continuation code-r13', flush=True)


if __name__ == '__main__':
    main()
