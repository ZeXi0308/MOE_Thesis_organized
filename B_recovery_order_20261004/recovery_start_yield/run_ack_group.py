#!/usr/bin/env python3
"""ACK-bounded yield ABBA with the frozen lock/resource/no-action controller."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'bead73f04948612f366196d1b8d626c0dfde309b9daaca0b2a13e8e9d4bb61ea'


def load_parent():
    path = ROOT/'run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen start-yield controller changed')
    spec = importlib.util.spec_from_file_location('ack_yield_group_parent', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def adapt_source(text, sources):
    changes = {
        "str(ROOT/'run_fixed_cell.py'), '--inputs'": "str(ROOT/'run_ack_fixed_cell.py'), '--inputs'",
        'yield_once': 'yield_ack',
        'B_RECOVERY_START_YIELD': 'B_RECOVERY_START_YIELD_ACK',
        'recovery_start_yield': 'recovery_start_yield_ack',
        'recovery-start-yield': 'recovery-start-yield-ack',
    }
    for old, new in changes.items():
        if not text.count(old):
            raise RuntimeError('ACK-bounded yield controller boundary changed: '+old)
        text = text.replace(old, new)
    return text, [*sources, ROOT/'run_group.py']


def adapted_source():
    return adapt_source(*load_parent().adapted_source())


def main():
    parent = load_parent(); original = parent.adapted_source
    parent.adapted_source = lambda: adapt_source(*original())
    parent.__file__ = str(__file__)
    return parent.main()


if __name__ == '__main__':
    raise SystemExit(main())
