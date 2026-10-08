#!/usr/bin/env python3
"""One native/early-pin ABBA, fixed1024, using the existing whole-GPU lock."""
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


def adapted_source():
    path = BASE/'load_order/run_group.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != GROUP_SHA:
        raise RuntimeError('Frozen parent controller changed')
    parent = load('source_handoff_parent', path)
    text = parent.adapted_source()
    replacements = {
        "p['sequence'] != ['native', 'short_load', 'short_load', 'native']": "p['sequence'] != ['native', 'early_pin', 'early_pin', 'native']",
        'Requires the fixed normal-capacity LOAD-order ABBA plan': 'Requires the fixed normal-capacity source-handoff ABBA plan',
        "receipt['cells'][-1]['load_order_mode'] = mode": "receipt['cells'][-1]['source_handoff_mode'] = mode",
        "B_LOAD_ORDER=mode,": "B_LOAD_ORDER='native', B_TAIL_RESERVATION='native', B_SOURCE_HANDOFF=mode,",
        "str(ROOT.parent/'normal_capacity/inputs'/f'cap{cap}')": "str(Path(p['inputs_dir'])/f'cap{cap}')",
        "str(ROOT/'run_cell.py'), '--inputs'": "str(ROOT/'run_fixed_cell.py'), '--inputs'",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError('Frozen controller adaptation changed: '+old)
        text = text.replace(old, new)
    return text, [path, BASE/'capacity_handoff/observe_tail.py',
                  BASE/'tail_reservation/reserve_tail.py',
                  BASE/'completion_handoff/run_fixed_group.py',
                  *[BASE/name for name in parent.PINS]]


def main():
    text, sources = adapted_source()
    waiter = load('source_handoff_wait', BASE/'completion_handoff/run_fixed_group_wait.py')
    settler = load('source_handoff_settle', BASE/'completion_handoff/run_fixed_group_settle.py')
    namespace = dict(__name__='source_handoff_controller', __file__=str(__file__),
                     EXTRA_SOURCE_PATHS=sources)
    exec(compile(text, str(__file__)+'[native-parent]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        settler.install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
