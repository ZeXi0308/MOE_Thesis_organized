#!/usr/bin/env python3
"""Fixed256x1024 recovery-fit cell using the frozen termination transform."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXED_SHA = '69b03e618595bc89b29874ef846e83a5524a0fece805d932ca70b50814af2829'


def termination_source(text):
    path = ROOT.parent/'source_handoff/run_fixed_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != FIXED_SHA:
        raise RuntimeError('Frozen fixed cell transform changed')
    spec = importlib.util.spec_from_file_location('recovery_fit_fixed_termination', path)
    fixed = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixed)
    text = fixed.termination_source(text)
    old = "        config['recovery_lease_mode']=lease_mode"
    if text.count(old) != 1:
        raise RuntimeError('Fixed recovery-fit provenance boundary changed')
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return text.replace(old, f"        config['recovery_fit_fixed_cell_sha256']={sha!r}\n"+old)


def main():
    path = ROOT/'run_cell.py'
    spec = importlib.util.spec_from_file_location('recovery_fit_cell', path)
    cell = importlib.util.module_from_spec(spec); spec.loader.exec_module(cell)
    return cell.main(source_transform=termination_source)


if __name__ == '__main__':
    raise SystemExit(main())
