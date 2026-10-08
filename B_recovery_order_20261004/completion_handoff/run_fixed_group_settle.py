#!/usr/bin/env python3
"""Allow only the initial locked GPU handoff to settle, preserving every sample."""
import hashlib
import importlib.util
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
WAIT_SHA256 = 'e50957ef2393f977c8d790ef4ace5989b2375d46b7bea34f3f6ac056672c1beb'


def install_initial_settle(base, receipt, receipt_path, write):
    original = base.checked_boundary

    def first_boundary(uuid):
        base.checked_boundary = original
        started = time.monotonic()
        report = dict(started_unix_s=time.time(), timeout_seconds=30., observations=[])
        receipt['initial_handoff_settle'] = report
        state = original(uuid)
        report['observations'].extend(state.get('observations', [state]))
        while True:
            elapsed = time.monotonic() - started
            report.update(status='EMPTY' if state['empty'] else 'TIMEOUT' if elapsed >= 30. else 'WAITING',
                          elapsed_seconds=elapsed)
            write(receipt_path, receipt)
            if state['empty'] or elapsed >= 30.:
                return dict(state, observations=list(report['observations']))
            time.sleep(min(1., 30. - elapsed))
            state = base.boundary(uuid)
            report['observations'].append(state)

    base.checked_boundary = first_boundary


def load_pinned(name, path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen controller source changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    waiter = load_pinned('frozen_fixed_wait', ROOT/'run_fixed_group_wait.py', WAIT_SHA256)
    fixed = load_pinned('frozen_fixed_controller', ROOT/'run_fixed_group.py', waiter.PARENT_SHA256)
    namespace = dict(__name__='fixed_completion_settle_controller',
                     __file__=str(ROOT/'run_group.py'))
    exec(compile(fixed.adapted_source(), str(ROOT/'run_group.py')+'[fixed-inputs]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
