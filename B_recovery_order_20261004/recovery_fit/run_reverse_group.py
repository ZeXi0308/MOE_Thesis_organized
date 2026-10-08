#!/usr/bin/env python3
"""One development BAAB repeat; frozen policy, cells, inputs and shared GPU lock."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = '3495162500e7bdd06a616bdd69cf763f7d00272cf6155003667b2c163065bd7c'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapted_source():
    path = ROOT/'run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen recovery-fit parent controller changed')
    parent = load('recovery_fit_reverse_parent', path)
    text, sources = parent.adapted_source()
    changes = {
        "p['sequence'] != ['native', 'fit_once', 'fit_once', 'native']":
            "p['sequence'] != ['fit_once', 'native', 'native', 'fit_once']",
        'Requires fixed normal-capacity recovery-fit-once ABBA':
            'Requires one fixed normal-capacity recovery-fit-once BAAB development repeat',
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('Frozen reverse-order adaptation boundary changed: '+old)
        text = text.replace(old, new)
    return text, [*sources, path]


def main():
    text, sources = adapted_source()
    waiter = load('fit_reverse_lock', BASE/'completion_handoff/run_fixed_group_wait.py')
    settler = load('fit_reverse_settle', BASE/'completion_handoff/run_fixed_group_settle.py')
    namespace = dict(__name__='fit_reverse_controller', __file__=str(__file__), EXTRA_SOURCE_PATHS=sources)
    exec(compile(text, str(__file__)+'[frozen-parent]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        settler.install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
