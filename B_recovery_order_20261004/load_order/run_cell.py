#!/usr/bin/env python3
"""Reuse the frozen native cell, with only ready LOAD ordering varied."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
NORMAL_SHA = '9f92617d44bbf8cda6992261129c35ff844a14a569dc8f259ed2bdedaa2388bb'
COMPLETION_SHA = '6ec52ce72cef642abb9ec63fe0ca5a549160b70d9d70559659222a53bd1c3e49'
CORE_SHA = '51b3db810d13c74a124cf489ace3f8af5ecd3981183a3cb6fba9bec1ec048e6e'


def main():
    mode = os.environ.get('B_LOAD_ORDER', 'native')
    if mode not in ('native', 'short_load'):
        raise ValueError('B_LOAD_ORDER must be native or short_load')
    for key in ('B_RECOVERY_ORDER', 'B_COMPLETION_FINALIZE'):
        if os.environ.setdefault(key, 'native') != 'native':
            raise ValueError(key+' must retain native ordering/boundary')
    normal_path = BASE/'normal_capacity/run_cell.py'
    completion_path = BASE/'completion_handoff/completion_finalize.py'
    for path, expected in ((normal_path, NORMAL_SHA), (completion_path, COMPLETION_SHA),
                           (ROOT/'load_order.py', CORE_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen execution source changed: '+str(path))
    sys.path.insert(0, str(completion_path.parent))
    spec = importlib.util.spec_from_file_location('load_order_normal_cell', normal_path)
    normal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normal)
    original = normal.adapted_source

    def source():
        text = original()
        old = "        order_data, uninstall_order = install_recovery_order(scheduler, os.environ['B_RECOVERY_ORDER'])"
        if text.count(old) != 1:
            raise RuntimeError('Native observation installation boundary changed')
        replacement = old + "\n        from completion_finalize import install as install_completion_finalize"
        replacement += "\n        completion_data, uninstall_completion = install_completion_finalize(engine, 'native')"
        replacement += "\n        from load_order import install as install_load_order"
        replacement += f"\n        load_order_data, uninstall_load_order = install_load_order(scheduler, {mode!r})"
        replacement += f"\n        config['B_load_order'] = {mode!r}"
        replacement += "\n        config['B_completion_finalize'] = 'native'"
        replacement += f"\n        config['load_order_runner_sha256'] = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}"
        replacement += f"\n        config['load_order_core_sha256'] = {hashlib.sha256((ROOT/'load_order.py').read_bytes()).hexdigest()!r}"
        text = text.replace(old, replacement)
        old = "                dump(out/'recovery-order.json', uninstall_order())"
        if text.count(old) != 1:
            raise RuntimeError('Native observation teardown boundary changed')
        return text.replace(old, "                dump(out/'load-order.json', uninstall_load_order())\n"
            "                dump(out/'completion-finalize.json', uninstall_completion())\n" + old)

    normal.adapted_source = source
    normal.main()


if __name__ == '__main__':
    main()
