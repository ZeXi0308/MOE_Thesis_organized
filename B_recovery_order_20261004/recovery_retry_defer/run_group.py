#!/usr/bin/env python3
"""Native/defer-once ABBA using the existing whole-group lock and executor."""
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = 'de88fae812d97cfe49ca27789b7346f0c3d5013f5062458e02e158c1225a76e2'
LOCK_PATH = '/root/autodl-tmp/moe-research-gpu.lock'
LOCK_IDENTITY = (2304, 15049831297)
MINIMUM_FREE_BYTES = 3 * 1024**3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def check_space(session, receipt, write):
    free = shutil.disk_usage(session).free
    receipt.setdefault('disk_checks', []).append(dict(path=str(session), free_bytes=free,
        required_free_bytes=MINIMUM_FREE_BYTES, passed=free >= MINIMUM_FREE_BYTES))
    write(session/'receipt.json', receipt)
    if free < MINIMUM_FREE_BYTES:
        raise RuntimeError('Session/cache filesystem has less than 3GiB free; no cell started')


def adapted_source():
    path = BASE/'source_handoff/run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen native-source controller changed')
    text, sources = load('retry_defer_source_group', path).adapted_source()
    changes = {
        "p['sequence'] != ['native', 'early_pin', 'early_pin', 'native']": "p['sequence'] != ['native', 'defer_once', 'defer_once', 'native']",
        'Requires the fixed normal-capacity source-handoff ABBA plan': 'Requires fixed normal-capacity native/retry-defer ABBA',
        'B_SOURCE_HANDOFF=mode,': "B_SOURCE_HANDOFF='native', B_RECOVERY_RETRY_DEFER=mode, B_OUTPUT_EVENT_STORAGE='compact',",
        "receipt['cells'][-1]['source_handoff_mode'] = mode": "receipt['cells'][-1]['source_handoff_mode'] = 'native'\n"
            "            receipt['cells'][-1]['recovery_retry_defer_mode'] = mode\n"
            "            receipt['cells'][-1]['output_event_storage_mode'] = 'compact'\n"
            "            action = json.loads((cell/'output/recovery-retry-defer.json').read_text())\n"
            "            counts = {key: action.get(key) for key in ('action_count', 'executed_breaks')}\n"
            "            receipt['cells'][-1]['retry_defer_action_counts'] = counts\n"
            "            if i == 1:\n"
            "                if any(type(value) is not int for value in counts.values()):\n"
            "                    raise RuntimeError('First defer_once action evidence missing; no remainder started')\n"
            "                if counts['action_count'] == 0 or counts['executed_breaks'] == 0:\n"
            "                    receipt.update(status='STOP_NO_ACTION', stop_reason='First defer_once had no executed deferral; reverse cells not started')\n"
            "                    break",
        "return 0 if receipt['status'].startswith(('COMPLETE', 'BASELINES_COMPLETE')) else 1":
            "return 0 if receipt['status'].startswith(('COMPLETE', 'BASELINES_COMPLETE')) or receipt['status'] == 'STOP_NO_ACTION' else 1",
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('Retry-defer controller boundary changed: '+old)
        text = text.replace(old, new)
    sources += [path, BASE/'source_handoff/run_cell.py', BASE/'source_handoff/run_fixed_cell.py',
                BASE/'source_handoff/source_handoff.py', BASE/'output_event_compact/compact_capture.py',
                BASE/'recovery_gc_diag/gc_probe.py', BASE/'recovery_fit/fit_once.py',
                BASE/'recovery_queue/observe_waiting.py']
    return text, sources


def main():
    text, sources = adapted_source()
    waiter = load('retry_defer_lock', BASE/'completion_handoff/run_fixed_group_wait.py')
    settler = load('retry_defer_settle', BASE/'completion_handoff/run_fixed_group_settle.py')
    namespace = dict(__name__='retry_defer_controller', __file__=str(__file__), EXTRA_SOURCE_PATHS=sources)
    exec(compile(text, str(__file__)+'[native-source-parent]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        identity = os.fstat(fd)
        if (receipt['plan']['lock_path'] != LOCK_PATH
                or (identity.st_dev, identity.st_ino) != LOCK_IDENTITY):
            raise RuntimeError('Shared GPU lock identity differs; lock file is never replaced')
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        check_space(Path(receipt_path).parent, receipt, namespace['write'])
        settler.install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
