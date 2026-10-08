#!/usr/bin/env python3
"""Reuse the normal-capacity cell; only relocate native completion finalization."""
import hashlib
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    mode = os.environ.get('B_COMPLETION_FINALIZE', 'native')
    if mode not in ('native', 'after_sample'):
        raise ValueError('B_COMPLETION_FINALIZE must be native or after_sample')
    if os.environ.setdefault('B_RECOVERY_ORDER', 'native') != 'native':
        raise ValueError('Completion comparison fixes STORE ordering to native')
    spec = importlib.util.spec_from_file_location('normal_capacity_reused', ROOT.parent/'normal_capacity/run_cell.py')
    normal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normal)
    original = normal.adapted_source

    def source():
        text = original()
        old = "        order_data, uninstall_order = install_recovery_order(scheduler, os.environ['B_RECOVERY_ORDER'])"
        if text.count(old) != 1:
            raise RuntimeError('Normal-capacity installation boundary changed')
        replacement = old + "\n        from completion_finalize import install as install_completion_finalize"
        replacement += f"\n        completion_data, uninstall_completion = install_completion_finalize(engine, {mode!r})"
        replacement += f"\n        config['B_completion_finalize'] = {mode!r}"
        replacement += f"\n        config['completion_handoff_runner_sha256'] = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}"
        replacement += f"\n        config['completion_handoff_core_sha256'] = {hashlib.sha256((ROOT/'completion_finalize.py').read_bytes()).hexdigest()!r}"
        text = text.replace(old, replacement)
        old = "                dump(out/'recovery-order.json', uninstall_order())"
        if text.count(old) != 1:
            raise RuntimeError('Normal-capacity teardown boundary changed')
        return text.replace(old, "                dump(out/'completion-finalize.json', uninstall_completion())\n" + old)

    normal.adapted_source = source
    normal.main()


if __name__ == '__main__':
    main()
