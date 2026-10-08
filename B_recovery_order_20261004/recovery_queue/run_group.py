#!/usr/bin/env python3
"""One native waiting-decision observation under the existing whole-GPU lock."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = 'de88fae812d97cfe49ca27789b7346f0c3d5013f5062458e02e158c1225a76e2'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapted_source():
    path = BASE/'source_handoff/run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen parent controller changed')
    parent = load('waiting_parent_group', path)
    text, sources = parent.adapted_source()
    replacements = {
        "p['sequence'] != ['native', 'early_pin', 'early_pin', 'native']": "p['sequence'] != ['native']",
        'Requires the fixed normal-capacity source-handoff ABBA plan': 'Requires one fixed normal-capacity native waiting-observation cell',
        "receipt['cells'][-1]['source_handoff_mode'] = mode": "receipt['cells'][-1]['waiting_observation_mode'] = 'native_waiting_decision'",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError('Frozen controller adaptation changed: '+old)
        text = text.replace(old, new)
    sources += [path, BASE/'source_handoff/run_cell.py',
                BASE/'source_handoff/run_fixed_cell.py', BASE/'source_handoff/source_handoff.py']
    return text, sources


def main():
    text, sources = adapted_source()
    waiter = load('waiting_lock', BASE/'completion_handoff/run_fixed_group_wait.py')
    settler = load('waiting_settle', BASE/'completion_handoff/run_fixed_group_settle.py')
    namespace = dict(__name__='waiting_controller', __file__=str(__file__),
                     EXTRA_SOURCE_PATHS=sources)
    exec(compile(text, str(__file__)+'[native-parent]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        settler.install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
