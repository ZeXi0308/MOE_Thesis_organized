#!/usr/bin/env python3
"""Fixed1024 source-handoff cell using the original frozen termination transform."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXED_SHA = '6846e19f8582f7b0d1a6fa83f7c8a3cf84aab6daf9ee69437de07d8b754fa6cf'


def load_fixed():
    path = ROOT.parent/'completion_handoff/run_fixed_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != FIXED_SHA:
        raise RuntimeError('Frozen fixed-output transform changed')
    spec = importlib.util.spec_from_file_location('source_handoff_fixed_termination', path)
    fixed = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixed)
    fixed.verify_sources()
    return fixed


def termination_source(text):
    fixed = load_fixed()
    text = fixed.fixed_termination_source(text)
    text = fixed.replace_once(text,
        "shlex.join([sys.executable, sys.argv[0], '--cell', *sys.argv[1:]])",
        'shlex.join([sys.executable, *sys.argv])')
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return fixed.replace_once(text, "        config['recovery_lease_mode']=lease_mode",
        f"        config['fixed_output_cell_sha256']={sha!r}\n"
        "        config['recovery_lease_mode']=lease_mode")


def main():
    path = ROOT/'run_cell.py'
    spec = importlib.util.spec_from_file_location('source_handoff_cell', path)
    cell = importlib.util.module_from_spec(spec); spec.loader.exec_module(cell)
    return cell.main(source_transform=termination_source)


if __name__ == '__main__':
    raise SystemExit(main())
