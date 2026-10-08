#!/usr/bin/env python3
"""One native allocation-observation cell using the existing whole-GPU runner."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
GROUP_SHA = '7ba62d4abeaa38725075019dbe0f9fe46a0b336319ff0ce6c98f2b469c7ecc9c'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    path = BASE/'load_order/run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != GROUP_SHA:
        raise RuntimeError('Frozen parent controller changed')
    parent = load('tail_observation_parent', path)
    text = parent.adapted_source()
    replacements = {
        "p['sequence'] != ['native', 'short_load', 'short_load', 'native']": "p['sequence'] != ['native']",
        'Requires the fixed normal-capacity LOAD-order ABBA plan': 'Requires a single native normal-capacity observation cell',
        "receipt['cells'][-1]['load_order_mode'] = mode": "receipt['cells'][-1]['observation'] = 'recovery_allocation_tail'",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError('Frozen controller adaptation changed: '+old)
        text = text.replace(old, new)
    waiter = load('tail_observation_wait', BASE/'completion_handoff/run_fixed_group_wait.py')
    settler = load('tail_observation_settle', BASE/'completion_handoff/run_fixed_group_settle.py')
    namespace = dict(__name__='tail_observation_controller', __file__=str(__file__),
        EXTRA_SOURCE_PATHS=[path, *[BASE/name for name in parent.PINS]])
    exec(compile(text, str(__file__)+'[native-parent]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        settler.install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
