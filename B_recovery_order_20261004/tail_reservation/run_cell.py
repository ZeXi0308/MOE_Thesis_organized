#!/usr/bin/env python3
"""Native/full-tail cell over the frozen normal-capacity executor."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
NORMAL_SHA = '9f92617d44bbf8cda6992261129c35ff844a14a569dc8f259ed2bdedaa2388bb'
COMPLETION_SHA = '6ec52ce72cef642abb9ec63fe0ca5a549160b70d9d70559659222a53bd1c3e49'


def main():
    for key in ('B_RECOVERY_ORDER', 'B_COMPLETION_FINALIZE', 'B_LOAD_ORDER'):
        if os.environ.setdefault(key, 'native') != 'native':
            raise ValueError(key+' must retain native behavior')
    normal_path = BASE/'normal_capacity/run_cell.py'
    completion_path = BASE/'completion_handoff/completion_finalize.py'
    for path, expected in ((normal_path, NORMAL_SHA), (completion_path, COMPLETION_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen execution source changed: '+str(path))
    sys.path.insert(0, str(completion_path.parent))
    observer_path = BASE/'capacity_handoff/observe_tail.py'
    if hashlib.sha256(observer_path.read_bytes()).hexdigest() != 'b0e59962645ba95f76bb380bec2f9c3675ceac62d7b57debb24a184cbb9c5c8f':
        raise RuntimeError('Frozen tail observer changed')
    sys.path.insert(0, str(observer_path.parent))
    mode = os.environ.get('B_TAIL_RESERVATION', 'native')
    if mode not in ('native', 'full_tail'):
        raise ValueError(mode)
    spec = importlib.util.spec_from_file_location('tail_reservation_normal_cell', normal_path)
    normal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normal)
    original = normal.adapted_source

    def source():
        text = original()
        old = "        order_data, uninstall_order = install_recovery_order(scheduler, os.environ['B_RECOVERY_ORDER'])"
        if text.count(old) != 1:
            raise RuntimeError('Native observation installation boundary changed')
        extra = "\n        from completion_finalize import install as install_completion_finalize"
        extra += "\n        completion_data, uninstall_completion = install_completion_finalize(engine, 'native')"
        extra += "\n        from observe_tail import install as install_tail_observer"
        extra += "\n        tail_data, uninstall_tail = install_tail_observer(scheduler)"
        extra += "\n        from reserve_tail import install as install_tail_reservation"
        extra += f"\n        reservation_data, uninstall_reservation = install_tail_reservation(scheduler, {mode!r})"
        extra += f"\n        config['B_tail_reservation'] = {mode!r}"
        extra += f"\n        config['tail_reservation_policy_sha256'] = {hashlib.sha256((ROOT/'reserve_tail.py').read_bytes()).hexdigest()!r}"
        extra += "\n        config['B_completion_finalize'] = 'native'"
        extra += "\n        config['B_capacity_handoff'] = 'observe_only'"
        extra += f"\n        config['tail_reservation_runner_sha256'] = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}"
        extra += f"\n        config['capacity_handoff_observer_sha256'] = {hashlib.sha256(observer_path.read_bytes()).hexdigest()!r}"
        text = text.replace(old, old+extra)
        old = "                dump(out/'recovery-order.json', uninstall_order())"
        if text.count(old) != 1:
            raise RuntimeError('Native observation teardown boundary changed')
        return text.replace(old, "                dump(out/'tail-reservation.json', uninstall_reservation())\n"
            "                dump(out/'capacity-handoff.json', uninstall_tail())\n"
            "                dump(out/'completion-finalize.json', uninstall_completion())\n"+old)

    normal.adapted_source = source
    normal.main()


if __name__ == '__main__':
    main()
