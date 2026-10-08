#!/usr/bin/env python3
"""One passive-GC once diagnostic under the existing whole-group GPU lock."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = 'b22b09cde3baccb77b2a5bf575a5ff9258069993485d5436c469abc8a4e1edda'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapted_source():
    path = BASE/'recovery_repeat/run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen recovery-repeat controller changed')
    parent = load('gc_diag_repeat_group', path)
    text, sources = parent.adapted_source()
    changes = {
        "p['sequence'] != ['once', 'repeat8', 'repeat8', 'once']": "p['sequence'] != ['once']",
        'Requires fixed normal-capacity recovery-repeat action-budget ABBA': 'Requires one fixed normal-capacity once GC-observation cell',
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('GC diagnostic controller boundary changed: '+old)
        text = text.replace(old, new)
    sources += [path, BASE/'recovery_repeat/run_cell.py', BASE/'recovery_repeat/run_fixed_cell.py',
                BASE/'recovery_repeat/repeat_fit.py']
    return text, sources


def main():
    text, sources = adapted_source()
    waiter = load('gc_diag_lock', BASE/'completion_handoff/run_fixed_group_wait.py')
    settler = load('gc_diag_settle', BASE/'completion_handoff/run_fixed_group_settle.py')
    namespace = dict(__name__='gc_diag_controller', __file__=str(__file__), EXTRA_SOURCE_PATHS=sources)
    exec(compile(text, str(__file__)+'[repeat-parent]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        settler.install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
