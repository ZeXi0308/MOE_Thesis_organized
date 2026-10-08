#!/usr/bin/env python3
"""Native cell with waiting-decision observation outside existing observers."""
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
    spec = importlib.util.spec_from_file_location('recovery_queue_source_cell', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def adapt_source(text):
    def replace_once(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Waiting observer boundary changed: '+old[:90])
        text = text.replace(old, new)
    old = "        source_data, uninstall_source = install_source_handoff(scheduler, 'native')"
    extra = "\n        from observe_waiting import install as install_waiting_observer"
    extra += "\n        waiting_data, uninstall_waiting = install_waiting_observer(scheduler, source_data)"
    extra += "\n        config['B_recovery_queue'] = 'observe_only'"
    extra += f"\n        config['waiting_observer_sha256'] = {hashlib.sha256((ROOT/'observe_waiting.py').read_bytes()).hexdigest()!r}"
    extra += f"\n        config['recovery_queue_runner_sha256'] = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}"
    replace_once(old, old+extra)
    old = "                dump(out/'source-handoff.json', uninstall_source())"
    replace_once(old, "                dump(out/'waiting-decisions.json', uninstall_waiting())\n"+old)
    return text


def main(source_transform=None):
    if os.environ.setdefault('B_SOURCE_HANDOFF', 'native') != 'native':
        raise ValueError('Waiting observation requires B_SOURCE_HANDOFF=native')
    sys.path.insert(0, str(ROOT))
    parent = load_parent(); original = parent.adapt_source
    parent.adapt_source = lambda text, mode: adapt_source(original(text, mode))
    return parent.main(source_transform=source_transform)


if __name__ == '__main__':
    raise SystemExit(main())
