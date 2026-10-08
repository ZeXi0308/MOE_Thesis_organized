#!/usr/bin/env python3
"""ACK-bounded yield over the frozen native/compact/passive-GC cell."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '0daa6c1f0017eec9d08d62827a34815ec8382117076a49153fdb3c58eb541621'
POLICY_SHA = 'cc49e44011195688350dcf458b2c52289665e3076f794e6f520068ee8c8b6450'


def adapted_source():
    path = ROOT/'run_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen start-yield cell changed')
    policy = ROOT/'yield_until_ack.py'
    if POLICY_SHA is None or not policy.exists() or hashlib.sha256(policy.read_bytes()).hexdigest() != POLICY_SHA:
        raise RuntimeError('ACK-bounded yield policy is not frozen or differs from its pin')
    text = path.read_text()
    changes = {
        "POLICY_SHA = '250b25a4260a7f59cbe7e183dab8d5e6fe96d8dfe1d6f6f9f441b609087d5fef'": 'POLICY_SHA = '+repr(POLICY_SHA),
        "ROOT/'yield_once.py'": "ROOT/'yield_until_ack.py'",
        'from yield_once import': 'from yield_until_ack import',
        'B_RECOVERY_START_YIELD': 'B_RECOVERY_START_YIELD_ACK',
        'recovery_start_yield': 'recovery_start_yield_ack',
        'recovery-start-yield.json': 'recovery-start-yield-ack.json',
        "'yield_once'": "'yield_ack'",
        'must be native or yield_once': 'must be native or yield_ack',
    }
    for old, new in changes.items():
        if not text.count(old):
            raise RuntimeError('ACK-bounded yield cell boundary changed: '+old)
        text = text.replace(old, new)
    return text


def main(source_transform=None):
    namespace = dict(__name__='recovery_start_yield_ack_cell', __file__=str(__file__))
    exec(compile(adapted_source(), str(__file__)+'[frozen-cell-parent]', 'exec'), namespace)
    return namespace['main'](source_transform=source_transform)


if __name__ == '__main__':
    raise SystemExit(main())
