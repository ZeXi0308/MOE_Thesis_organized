#!/usr/bin/env python3
"""Start-gate ABBA using the frozen r05 lock,2.5GiB floor and zero-action stop."""
import hashlib
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = '43dcf2541200b35bc0122d71a69edf97f1e7bc9db433474261a295f91b61c4f8'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_parent():
    path = BASE/'recovery_start_yield/run_ack_space_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen r05 resource/controller adapter changed')
    return load('start_gate_controller_parent', path)


def adapted_source():
    text, sources = load_parent().adapted_source()
    changes = {
        "str(ROOT/'run_ack_fixed_cell.py'), '--inputs'": "str(ROOT/'run_fixed_cell.py'), '--inputs'",
        'B_RECOVERY_START_YIELD_ACK': 'B_RECOVERY_START_GATE',
        'recovery_start_yield_ack': 'recovery_start_gate',
        'recovery-start-yield-ack': 'recovery-start-gate',
        'yield_ack': 'wait_release',
        'had no executed yield': 'had no executed start-gate break',
    }
    for old, new in changes.items():
        if not text.count(old):
            raise RuntimeError('Start-gate controller boundary changed: '+old)
        text = text.replace(old, new)
    sources += [BASE/'recovery_start_yield'/name for name in
        ('run_cell.py', 'run_group.py', 'run_ack_group.py',
         'run_ack_host_group.py', 'run_ack_space_group.py')]
    return text, sources


def main():
    parent = load_parent()
    resource = parent.load_parent().load_parent().load_parent().load_parent()
    text, sources = adapted_source()
    waiter = load('start_gate_lock', BASE/'completion_handoff/run_fixed_group_wait.py')
    settler = load('start_gate_settle', BASE/'completion_handoff/run_fixed_group_settle.py')
    namespace = dict(__name__='recovery_start_gate_controller', __file__=str(__file__), EXTRA_SOURCE_PATHS=sources)
    exec(compile(text, str(__file__)+'[frozen-r05-controller]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        identity = os.fstat(fd)
        if (receipt['plan']['lock_path'] != resource.LOCK_PATH
                or (identity.st_dev, identity.st_ino) != resource.LOCK_IDENTITY):
            raise RuntimeError('Shared GPU lock identity differs; lock file is never replaced')
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        resource.check_space(Path(receipt_path).parent, receipt, namespace['write'])
        settler.install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
