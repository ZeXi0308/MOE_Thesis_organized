#!/usr/bin/env python3
"""Start-gate action probe over the frozen native/compact/passive-GC cell."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = '0daa6c1f0017eec9d08d62827a34815ec8382117076a49153fdb3c58eb541621'
POLICY_SHA = 'aa6a3a427d44668f41c29af3bb1b2d176be112b81e68dc6275fa8632d94b3d2b'


def adapted_source():
    path = BASE/'recovery_start_yield/run_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen native/compact cell changed')
    policy = ROOT/'start_gate.py'
    if POLICY_SHA is None or not policy.exists() or hashlib.sha256(policy.read_bytes()).hexdigest() != POLICY_SHA:
        raise RuntimeError('Recovery-start-gate policy is not frozen or differs from its pin')
    text = path.read_text()
    changes = {
        "POLICY_SHA = '250b25a4260a7f59cbe7e183dab8d5e6fe96d8dfe1d6f6f9f441b609087d5fef'": 'POLICY_SHA = '+repr(POLICY_SHA),
        "ROOT/'yield_once.py'": "ROOT/'start_gate.py'",
        'from yield_once import': 'from start_gate import',
        'B_RECOVERY_START_YIELD': 'B_RECOVERY_START_GATE',
        'recovery_start_yield': 'recovery_start_gate',
        'recovery-start-yield': 'recovery-start-gate',
        'Recovery-start-yield': 'Recovery-start-gate',
        "'yield_once'": "'wait_release'",
        'must be native or yield_once': 'must be native or wait_release',
    }
    for old, new in changes.items():
        if not text.count(old):
            raise RuntimeError('Start-gate cell boundary changed: '+old)
        text = text.replace(old, new)
    return text


def main(source_transform=None):
    namespace = dict(__name__='recovery_start_gate_cell', __file__=str(__file__))
    exec(compile(adapted_source(), str(__file__)+'[frozen-cell-parent]', 'exec'), namespace)
    return namespace['main'](source_transform=source_transform)


if __name__ == '__main__':
    raise SystemExit(main())
