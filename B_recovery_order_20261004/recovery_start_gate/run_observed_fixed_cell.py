#!/usr/bin/env python3
"""Frozen start-gate actions and fixed output, with common boundary snapshots."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINS = {
    'run_cell.py': '34bbb12db5bdd70c788046bce7fa1505e624a06e48d95d4ece795a36aac46ec7',
    'run_fixed_cell.py': '7cabc4b26941d30dd72ffdd99908e63dad66a873dab77c3479acbee39a44255a',
}
PARENT_POLICY_SHA = 'aa6a3a427d44668f41c29af3bb1b2d176be112b81e68dc6275fa8632d94b3d2b'
OBSERVER_SHA = '0aa050605513cca39f79ad6aea132787724fe90901ced1b679c170f6881fb6f1'


def load(name):
    path = ROOT/name
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
        raise RuntimeError('Frozen start-gate runner changed: '+name)
    spec = importlib.util.spec_from_file_location('observed_'+path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def adapted_source():
    observer = ROOT/'observe_progress.py'
    if OBSERVER_SHA is None or hashlib.sha256(observer.read_bytes()).hexdigest() != OBSERVER_SHA:
        raise RuntimeError('Boundary observer is not frozen or has changed')
    text = load('run_cell.py').adapted_source()
    for old, new in (
        ('POLICY_SHA = '+repr(PARENT_POLICY_SHA), 'POLICY_SHA = '+repr(OBSERVER_SHA)),
        ("ROOT/'start_gate.py'", "ROOT/'observe_progress.py'"),
        ('from start_gate import', 'from observe_progress import'),
    ):
        if text.count(old) != 1:
            raise RuntimeError('Observed cell adaptation boundary changed: '+old)
        text = text.replace(old, new)
    return text


def main():
    namespace = dict(__name__='recovery_start_gate_observed_cell', __file__=str(__file__))
    exec(compile(adapted_source(), str(__file__)+'[frozen-cell]', 'exec'), namespace)
    return namespace['main'](source_transform=load('run_fixed_cell.py').termination_source)


if __name__ == '__main__':
    raise SystemExit(main())
