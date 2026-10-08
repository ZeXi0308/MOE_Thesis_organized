#!/usr/bin/env python3
"""Retain the frozen fixed256x1024 termination for both start-gate arms."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '69b03e618595bc89b29874ef846e83a5524a0fece805d932ca70b50814af2829'


def termination_source(text):
    path = ROOT.parent/'source_handoff/run_fixed_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen native fixed-output transform changed')
    spec = importlib.util.spec_from_file_location('start_gate_fixed_parent', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    text = parent.termination_source(text)
    old = "        config['recovery_lease_mode']=lease_mode"
    if text.count(old) != 1:
        raise RuntimeError('Start-gate fixed-output provenance boundary changed')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return text.replace(old, f"        config['recovery_start_gate_fixed_cell_sha256']={digest!r}\n"+old)


def main():
    spec = importlib.util.spec_from_file_location('start_gate_cell', ROOT/'run_cell.py')
    cell = importlib.util.module_from_spec(spec); spec.loader.exec_module(cell)
    return cell.main(source_transform=termination_source)


if __name__ == '__main__':
    raise SystemExit(main())
