#!/usr/bin/env python3
"""Native/fit-once probe over the existing native source-observation runner."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PINS = {
    'source_handoff/run_cell.py': '4a0e65178540b005fe0101d4f8c0c4260eef2e1ac281d8bbaa311a2bc6228df3',
    'source_handoff/source_handoff.py': '0c952554e87b83c2cd31b9875bdfcb6469a73ce0eefcd089a2f5f28af7d43da7',
}


def load_parent():
    for name, expected in PINS.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen native observation source changed: '+name)
    path = BASE/'source_handoff/run_cell.py'
    spec = importlib.util.spec_from_file_location('recovery_fit_source_cell', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def adapt_source(text):
    def replace_once(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Recovery-fit adaptation boundary changed: '+old[:90])
        text = text.replace(old, new)
    old = "        source_data, uninstall_source = install_source_handoff(scheduler, 'native')"
    extra = "\n        from fit_once import install as install_recovery_fit"
    extra += "\n        fit_data, uninstall_fit = install_recovery_fit(scheduler, os.environ.get('B_RECOVERY_FIT', 'native'))"
    extra += "\n        config['B_recovery_fit'] = os.environ.get('B_RECOVERY_FIT', 'native')"
    extra += f"\n        config['recovery_fit_policy_sha256'] = {hashlib.sha256((ROOT/'fit_once.py').read_bytes()).hexdigest()!r}"
    extra += f"\n        config['recovery_fit_runner_sha256'] = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}"
    replace_once(old, old+extra)
    old = "                dump(out/'source-handoff.json', uninstall_source())"
    replace_once(old, "                dump(out/'recovery-fit.json', uninstall_fit())\n"+old)
    return text


def main(source_transform=None):
    if os.environ.setdefault('B_SOURCE_HANDOFF', 'native') != 'native':
        raise ValueError('Recovery-fit cells require B_SOURCE_HANDOFF=native')
    if os.environ.get('B_RECOVERY_FIT', 'native') not in ('native', 'fit_once'):
        raise ValueError('B_RECOVERY_FIT must be native or fit_once')
    sys.path.insert(0, str(ROOT))
    parent = load_parent(); original = parent.adapt_source
    parent.adapt_source = lambda text, mode: adapt_source(original(text, mode))
    return parent.main(source_transform=source_transform)


if __name__ == '__main__':
    raise SystemExit(main())
