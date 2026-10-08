#!/usr/bin/env python3
"""Fixed256x1024 once diagnostic retaining the frozen termination transform."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = 'd79e26b390b099a2cd30767ec47329b900e2e06e6e63c2d1969025ff9014a712'


def termination_source(text):
    path = ROOT.parent/'recovery_repeat/run_fixed_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen recovery-repeat fixed cell changed')
    spec = importlib.util.spec_from_file_location('gc_diag_fixed_termination', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    text = parent.termination_source(text)
    old = "        config['recovery_lease_mode']=lease_mode"
    if text.count(old) != 1:
        raise RuntimeError('Fixed GC diagnostic provenance boundary changed')
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return text.replace(old, f"        config['gc_observation_fixed_cell_sha256']={sha!r}\n"+old)


def main():
    path = ROOT/'run_cell.py'
    spec = importlib.util.spec_from_file_location('gc_diag_cell', path)
    cell = importlib.util.module_from_spec(spec); spec.loader.exec_module(cell)
    return cell.main(source_transform=termination_source)


if __name__ == '__main__':
    raise SystemExit(main())
