#!/usr/bin/env python3
"""Unchanged once cell with passive GC timing at the measurement boundary."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = 'b59d9ec45edaf0d0771deae622f68fa26893c2c80f0ced1658dac453cdf9966c'


def load_parent():
    path = BASE/'recovery_repeat/run_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen recovery-repeat cell changed')
    spec = importlib.util.spec_from_file_location('gc_diag_repeat_cell', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def adapt_source(text):
    def replace_once(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('GC observation adaptation boundary changed: '+old[:90])
        text = text.replace(old, new)
    old = "        fit_data, uninstall_fit = install_recovery_repeat(scheduler, os.environ.get('B_RECOVERY_REPEAT', 'once'))"
    extra = "\n        from gc_probe import install as install_gc_probe"
    extra += "\n        gc_data, uninstall_gc = install_gc_probe()"
    extra += "\n        config['B_gc_observation'] = 'passive_callbacks'"
    extra += f"\n        config['gc_probe_sha256'] = {hashlib.sha256((ROOT/'gc_probe.py').read_bytes()).hexdigest()!r}"
    extra += f"\n        config['gc_observation_runner_sha256'] = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}"
    replace_once(old, old+extra)
    old = "                dump(out/'recovery-repeat.json', uninstall_fit())"
    replace_once(old, "                dump(out/'gc-observation.json', uninstall_gc())\n"+old)
    return text


def main(source_transform=None):
    if os.environ.setdefault('B_RECOVERY_REPEAT', 'once') != 'once':
        raise ValueError('GC diagnostic retains B_RECOVERY_REPEAT=once')
    sys.path.insert(0, str(ROOT))
    parent = load_parent(); original = parent.adapt_source
    parent.adapt_source = lambda text: adapt_source(original(text))
    return parent.main(source_transform=source_transform)


if __name__ == '__main__':
    raise SystemExit(main())
