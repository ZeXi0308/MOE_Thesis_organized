#!/usr/bin/env python3
"""Resume the measured matrix after r11's confirmed zero-cell lock timeout.

Same frozen experiments; six-hour lock waits, no retries. Each completed group
must be archived locally before the next can request the public lock.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from continue_olmoe_matrix import ROOT, archive_barrier, spent_seconds, BUDGET_SECONDS


def main():
    old = json.loads((ROOT/'runs/qwen-calibration-r11/receipt.json').read_text())
    assert old['status'] == 'STOPPED' and old['cells'] == []
    assert 'Public GPU lock wait expired' in old['error']
    assert 'acquired_unix_s' not in old
    manifest = json.loads((ROOT/'configs/matrix-r11b-sha.json').read_text())
    with (ROOT/'runs/matrix-r11b.claim').open('x') as f:
        json.dump(dict(pid=os.getpid(), plans=list(manifest)), f)
    for name, hashes in manifest.items():
        plan_path = ROOT/'configs'/f'{name}.json'
        assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == hashes['plan_sha256']
        plan = json.loads(plan_path.read_text())
        assert hashlib.sha256(Path(plan['inputs']).read_bytes()).hexdigest() == hashes['inputs_sha256']
        session = Path(plan['session'])
        assert not session.exists(), f'{name}: existing session, no automatic retry'
        reserve = len(plan['cells']) * (plan.get('per_cell_seconds', 450) + 20)
        spent = spent_seconds()
        assert spent + reserve <= BUDGET_SECONDS, (spent, reserve)
        print(f'START {name} spent={spent:.3f} reserve={reserve}', flush=True)
        subprocess.run([sys.executable, '-B', str(ROOT/'code-r12/run_group.py'),
                        '--plan', str(plan_path), '--wait-lock-seconds', '21600'], check=True)
        archive_barrier(session.name)
    print('COMPLETE matrix continuation', flush=True)


if __name__ == '__main__':
    main()
