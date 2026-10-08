#!/usr/bin/env python3
"""Retain fixed1024 termination and select the ACK-bounded yield cell."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'e259db08626db66104749c1b11c66fa8eafef15975f3d74f809315905364f38d'


def termination_source(text):
    path = ROOT/'run_fixed_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen start-yield fixed-output cell changed')
    spec = importlib.util.spec_from_file_location('ack_yield_fixed_parent', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    text = parent.termination_source(text)
    old = "        config['recovery_lease_mode']=lease_mode"
    if text.count(old) != 1:
        raise RuntimeError('ACK-bounded yield fixed-output provenance boundary changed')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return text.replace(old, f"        config['recovery_start_yield_ack_fixed_cell_sha256']={digest!r}\n"+old)


def main():
    spec = importlib.util.spec_from_file_location('ack_yield_cell', ROOT/'run_ack_cell.py')
    cell = importlib.util.module_from_spec(spec); spec.loader.exec_module(cell)
    return cell.main(source_transform=termination_source)


if __name__ == '__main__':
    raise SystemExit(main())
