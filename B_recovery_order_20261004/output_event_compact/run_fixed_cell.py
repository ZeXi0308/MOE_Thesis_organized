#!/usr/bin/env python3
"""Same fixed1024 termination transform in both event-storage arms."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '83867d5d521b76da8629d3c21e71e8240e8cb427ae2169ba368fbfa77c6036fd'


def termination_source(text):
    path = ROOT.parent/'recovery_gc_diag/run_fixed_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen GC fixed-output transform changed')
    spec = importlib.util.spec_from_file_location('storage_gc_fixed', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent.termination_source(text)


def main():
    spec = importlib.util.spec_from_file_location('storage_cell', ROOT/'run_cell.py')
    cell = importlib.util.module_from_spec(spec); spec.loader.exec_module(cell)
    return cell.main(source_transform=termination_source)


if __name__ == '__main__':
    raise SystemExit(main())
