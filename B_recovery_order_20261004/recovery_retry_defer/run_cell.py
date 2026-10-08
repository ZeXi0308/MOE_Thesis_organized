#!/usr/bin/env python3
"""Native/defer-once cell with common compact capture and passive GC observation."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PINS = {
    'recovery_retry_defer/retry_defer.py': '81e1e3d1a1a4d13336188242876c6c29ab2e298da62c9ef3735af7104c0917f0',
    'source_handoff/run_cell.py': '4a0e65178540b005fe0101d4f8c0c4260eef2e1ac281d8bbaa311a2bc6228df3',
    'source_handoff/source_handoff.py': '0c952554e87b83c2cd31b9875bdfcb6469a73ce0eefcd089a2f5f28af7d43da7',
    'recovery_gc_diag/gc_probe.py': '9a753ff4eeec483ab21f982b10b303b4e7c3e9919d0197aa4b4afa57a5a9975b',
    'output_event_compact/compact_capture.py': '778a62f5b6325e33541489c815c6a324c9d6c7f5723a9b26b93956b319a17683',
}


def load_parent():
    for name, expected in PINS.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen retry-defer dependency changed: '+name)
    path = BASE/'source_handoff/run_cell.py'
    spec = importlib.util.spec_from_file_location('retry_defer_native_cell', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


def adapt_source(text):
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Retry-defer adaptation boundary changed: '+old[:90])
        text = text.replace(old, new)
    replace('        from request_measurement import measure_episode',
        "        from compact_capture import get_capture, annotate\n"
        "        measure_episode = get_capture('compact')")
    old = "        source_data, uninstall_source = install_source_handoff(scheduler, 'native')"
    extra = "\n        if config.get('ordinary_backfill') is not False or selective.get('ordinary_backfill') is not False:"
    extra += "\n            raise RuntimeError('Retry-defer requires the original native ordinary_backfill=False')"
    extra += "\n        from retry_defer import install as install_retry_defer"
    extra += "\n        retry_data, uninstall_retry = install_retry_defer(scheduler, os.environ['B_RECOVERY_RETRY_DEFER'], selective=selective)"
    extra += "\n        from gc_probe import install as install_gc_probe"
    extra += "\n        gc_data, uninstall_gc = install_gc_probe()"
    extra += "\n        config['B_recovery_retry_defer'] = os.environ['B_RECOVERY_RETRY_DEFER']"
    extra += f"\n        config['recovery_retry_defer_policy_sha256'] = sha(Path({str(ROOT/'retry_defer.py')!r}))"
    extra += f"\n        config['recovery_retry_defer_runner_sha256'] = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}"
    extra += "\n        config['B_gc_observation'] = 'passive_callbacks'"
    extra += f"\n        config['gc_probe_sha256'] = {PINS['recovery_gc_diag/gc_probe.py']!r}"
    extra += "\n        config['B_output_event_storage'] = 'compact'"
    extra += f"\n        config['output_event_storage_sha256'] = {PINS['output_event_compact/compact_capture.py']!r}"
    replace(old, old+extra)
    old = "                timing['measurement_return_perf_s']=time.perf_counter()"
    replace(old, old+"\n                annotate(raw, 'compact')")
    old = "                dump(out/'source-handoff.json', uninstall_source())"
    replace(old, "                dump(out/'gc-observation.json', uninstall_gc())\n"
        "                dump(out/'recovery-retry-defer.json', uninstall_retry())\n"+old)
    return text


def main(source_transform=None):
    for key, value in (('B_SOURCE_HANDOFF', 'native'), ('B_OUTPUT_EVENT_STORAGE', 'compact')):
        if os.environ.setdefault(key, value) != value:
            raise ValueError('Retry-defer requires '+key+'='+value)
    if os.environ.setdefault('B_RECOVERY_RETRY_DEFER', 'native') not in ('native', 'defer_once'):
        raise ValueError('B_RECOVERY_RETRY_DEFER must be native or defer_once')
    if any(key in os.environ for key in ('B_RECOVERY_REPEAT', 'B_RECOVERY_FIT')):
        raise ValueError('Retry-defer does not install fit_once or repeat policies')
    for directory in (ROOT, BASE/'output_event_compact', BASE/'recovery_gc_diag'):
        sys.path.insert(0, str(directory))
    parent = load_parent(); original = parent.adapt_source
    parent.adapt_source = lambda text, mode: adapt_source(original(text, mode))
    return parent.main(source_transform=source_transform)


if __name__ == '__main__':
    raise SystemExit(main())
