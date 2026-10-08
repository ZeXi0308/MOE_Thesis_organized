#!/usr/bin/env python3
"""One legacy/compact ABBA under the unchanged whole-group GPU lock."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = '8efaf8bebb4ebc1997cd6fa73c734e7bc31eee28c210393741065d67ba8555c3'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def adapted_source():
    path = BASE/'recovery_gc_diag/run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen GC diagnostic controller changed')
    text, sources = load('storage_gc_group', path).adapted_source()
    changes = {
        "p['sequence'] != ['once']": "p['sequence'] != ['legacy', 'compact', 'compact', 'legacy']",
        'Requires one fixed normal-capacity once GC-observation cell': 'Requires fixed once-policy output-event-storage ABBA',
        'B_RECOVERY_REPEAT=mode,': "B_RECOVERY_REPEAT='once', B_OUTPUT_EVENT_STORAGE=mode,",
        "receipt['cells'][-1]['recovery_repeat_mode'] = mode": "receipt['cells'][-1]['recovery_repeat_mode'] = 'once'\n            receipt['cells'][-1]['output_event_storage_mode'] = mode",
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('Storage controller boundary changed: '+old)
        text = text.replace(old, new)
    sources += [path, BASE/'recovery_gc_diag/run_cell.py', BASE/'recovery_gc_diag/run_fixed_cell.py',
                BASE/'recovery_gc_diag/gc_probe.py', BASE/'pkg/request_measurement.py']
    return text, sources


def main():
    text, sources = adapted_source()
    waiter = load('storage_lock', BASE/'completion_handoff/run_fixed_group_wait.py')
    settler = load('storage_settle', BASE/'completion_handoff/run_fixed_group_settle.py')
    namespace = dict(__name__='storage_controller', __file__=str(__file__), EXTRA_SOURCE_PATHS=sources)
    exec(compile(text, str(__file__)+'[gc-parent]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        settler.install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
