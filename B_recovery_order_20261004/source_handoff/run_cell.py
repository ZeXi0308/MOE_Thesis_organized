#!/usr/bin/env python3
"""Native/early-pin cell over the frozen normal-capacity executor."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PINS = {
    'normal_capacity/run_cell.py': '9f92617d44bbf8cda6992261129c35ff844a14a569dc8f259ed2bdedaa2388bb',
    'completion_handoff/completion_finalize.py': '6ec52ce72cef642abb9ec63fe0ca5a549160b70d9d70559659222a53bd1c3e49',
    'capacity_handoff/observe_tail.py': 'b0e59962645ba95f76bb380bec2f9c3675ceac62d7b57debb24a184cbb9c5c8f',
}


def load_normal():
    for name, expected in PINS.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen execution source changed: '+name)
    path = BASE/'normal_capacity/run_cell.py'
    spec = importlib.util.spec_from_file_location('source_handoff_normal_cell', path)
    normal = importlib.util.module_from_spec(spec); spec.loader.exec_module(normal)
    return normal


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Source-handoff adaptation boundary changed: '+old[:90])
    return text.replace(old, new)


def adapt_source(text, mode):
    if mode not in ('native', 'early_pin'):
        raise ValueError(mode)
    old = "        order_data, uninstall_order = install_recovery_order(scheduler, os.environ['B_RECOVERY_ORDER'])"
    extra = "\n        from completion_finalize import install as install_completion_finalize"
    extra += "\n        completion_data, uninstall_completion = install_completion_finalize(engine, 'native')"
    extra += "\n        from observe_tail import install as install_tail_observer"
    extra += "\n        tail_data, uninstall_tail = install_tail_observer(scheduler)"
    extra += "\n        from source_handoff import install as install_source_handoff"
    extra += f"\n        source_data, uninstall_source = install_source_handoff(scheduler, {mode!r})"
    extra += f"\n        config['B_source_handoff'] = {mode!r}"
    extra += f"\n        config['source_handoff_policy_sha256'] = {hashlib.sha256((ROOT/'source_handoff.py').read_bytes()).hexdigest()!r}"
    extra += f"\n        config['source_handoff_runner_sha256'] = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}"
    extra += "\n        config['B_completion_finalize'] = 'native'"
    extra += "\n        config['B_capacity_handoff'] = 'observe_only'"
    extra += f"\n        config['capacity_handoff_observer_sha256'] = {PINS['capacity_handoff/observe_tail.py']!r}"
    text = replace_once(text, old, old+extra)
    old = "                dump(out/'recovery-order.json', uninstall_order())"
    return replace_once(text, old, "                dump(out/'source-handoff.json', uninstall_source())\n"
        "                dump(out/'capacity-handoff.json', uninstall_tail())\n"
        "                dump(out/'completion-finalize.json', uninstall_completion())\n"+old)


def main(source_transform=None):
    for key in ('B_RECOVERY_ORDER', 'B_COMPLETION_FINALIZE', 'B_LOAD_ORDER', 'B_TAIL_RESERVATION'):
        if os.environ.setdefault(key, 'native') != 'native':
            raise ValueError(key+' must retain native behavior')
    mode = os.environ.get('B_SOURCE_HANDOFF', 'native')
    if mode not in ('native', 'early_pin'):
        raise ValueError(mode)
    for directory in (ROOT, BASE/'completion_handoff', BASE/'capacity_handoff'):
        sys.path.insert(0, str(directory))
    normal = load_normal(); original = normal.adapted_source

    def source():
        text = original()
        return adapt_source(source_transform(text) if source_transform else text, mode)

    normal.adapted_source = source
    return normal.main()


if __name__ == '__main__':
    raise SystemExit(main())
