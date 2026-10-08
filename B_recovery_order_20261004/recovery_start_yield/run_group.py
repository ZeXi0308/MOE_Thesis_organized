#!/usr/bin/env python3
"""Native/yield-once ABBA with the frozen group lock, 3GiB floor and no-action stop."""
import hashlib
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = '39bfe256430e3af2d2681ed99e8560f732942aade7cb00f7e0d97d06bc01bab6'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_parent():
    path = BASE/'recovery_retry_defer/run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen lock/resource controller changed')
    return load('start_yield_controller_parent', path)


def adapted_source():
    parent = load_parent(); text, sources = parent.adapted_source()
    changes = {
        "p['sequence'] != ['native', 'defer_once', 'defer_once', 'native']": "p['sequence'] != ['native', 'yield_once', 'yield_once', 'native']",
        'Requires fixed normal-capacity native/retry-defer ABBA': 'Requires fixed normal-capacity native/recovery-start-yield ABBA',
        'B_RECOVERY_RETRY_DEFER=mode,': 'B_RECOVERY_START_YIELD=mode,',
        "['recovery_retry_defer_mode']": "['recovery_start_yield_mode']",
        "cell/'output/recovery-retry-defer.json'": "cell/'output/recovery-start-yield.json'",
        "['retry_defer_action_counts']": "['recovery_start_yield_action_counts']",
        'First defer_once action evidence missing; no remainder started': 'First yield_once action evidence missing; no remainder started',
        'First defer_once had no executed deferral; reverse cells not started': 'First yield_once had no executed yield; reverse cells not started',
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('Recovery-start-yield controller boundary changed: '+old)
        text = text.replace(old, new)
    return text, [*sources, BASE/'recovery_retry_defer/run_group.py']


def main():
    parent = load_parent(); text, sources = adapted_source()
    waiter = load('start_yield_lock', BASE/'completion_handoff/run_fixed_group_wait.py')
    settler = load('start_yield_settle', BASE/'completion_handoff/run_fixed_group_settle.py')
    namespace = dict(__name__='recovery_start_yield_controller', __file__=str(__file__), EXTRA_SOURCE_PATHS=sources)
    exec(compile(text, str(__file__)+'[frozen-group-parent]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        identity = os.fstat(fd)
        if (receipt['plan']['lock_path'] != parent.LOCK_PATH
                or (identity.st_dev, identity.st_ino) != parent.LOCK_IDENTITY):
            raise RuntimeError('Shared GPU lock identity differs; lock file is never replaced')
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        parent.check_space(Path(receipt_path).parent, receipt, namespace['write'])
        settler.install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
