#!/usr/bin/env python3
"""One native cell with per-request output caps; reuse the frozen normal runner."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PINS = {
    'normal_capacity/run_group.py': '85278d8c2e90f9ccd7d37b8003781258d0d6f226ad3397285c3658dc3b98f101',
    'normal_capacity/run_cell.py': '9f92617d44bbf8cda6992261129c35ff844a14a569dc8f259ed2bdedaa2388bb',
    'normal_capacity/safe_static.py': '128b65ca4d306880d3397a16b0cd382ddd893402075fb214d9e1ab18baff3fd7',
    'completion_handoff/run_fixed_group_wait.py': 'e50957ef2393f977c8d790ef4ace5989b2375d46b7bea34f3f6ac056672c1beb',
    'completion_handoff/run_fixed_group_settle.py': '4bf68b1f90dfb2b4474969b75d613f99271e2063ee9ed50ac81118f32c359372',
    'run_group.py': '4d8091f43a75aa466e75f6ba2c20be9c31d1bfb3d7c5a25a252fa2d7d607f8b3',
    'pkg/request_measurement.py': '1b322be02505381dedb0aaf47f58513dfef61d60bf98e8d9590e18e3012534e4',
}


def verify_sources():
    for name, expected in PINS.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen source changed: '+name)


def adapted_source():
    verify_sources()
    text = (BASE/'normal_capacity/run_group.py').read_text()

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Mixed native adaptation boundary changed: '+old[:80])
        text = text.replace(old, new)

    replace("        tasks = [(cap, 'native') for cap in p['baseline_caps']]\n        if p['baseline_caps'] != [64, 192, 256] or p['engine_max_num_seqs'] != 256:\n            raise ValueError('Only the three predeclared operating points are allowed')\n        kv_bytes = 'auto'",
        "        if p['sequence'] != ['native'] or p['cap'] != 256 or p['engine_max_num_seqs'] != 256 or p['fixed_gpu_kv_bytes'] != 77242302464:\n            raise ValueError('Requires the single native mixed-output cap256 plan')\n        tasks = [(256, 'native')]\n        kv_bytes = p['fixed_gpu_kv_bytes']\n        receipt['fixed_gpu_kv_bytes'] = kv_bytes")
    replace("str(ROOT/'inputs'/f'cap{cap}')", "str(Path(p['inputs_dir'])/f'cap{cap}')")
    replace("for f in [*sorted(ROOT.glob('*.py')), *sorted((ROOT.parent/'pkg').glob('*.py'))]",
        "for f in [*sorted(ROOT.glob('*.py')), *sorted((ROOT.parent/'pkg').glob('*.py')), *EXTRA_SOURCE_PATHS]")
    start = "            if i == 2:\n"
    end = "                receipt['simple_age_merged_with_native'] = age_equal"
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError('Original conditional contrast boundary changed')
    replace(text[text.index(start):text.index(end)+len(end)], '')
    return text


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    text = adapted_source()
    waiter = load('mixed_frozen_wait', BASE/'completion_handoff/run_fixed_group_wait.py')
    settler = load('mixed_frozen_settle', BASE/'completion_handoff/run_fixed_group_settle.py')
    namespace = dict(__name__='mixed_native_controller', __file__=str(BASE/'normal_capacity/run_group.py'),
        EXTRA_SOURCE_PATHS=[Path(__file__), BASE/'run_group.py',
            BASE/'completion_handoff/run_fixed_group_wait.py', BASE/'completion_handoff/run_fixed_group_settle.py'])
    exec(compile(text, str(BASE/'normal_capacity/run_group.py')+'[mixed-native]', 'exec'), namespace)

    def acquire(fd, seconds, receipt, receipt_path):
        waiter.acquire_gpu_lock(fd, seconds, receipt, receipt_path, namespace['write'])
        settler.install_initial_settle(namespace['base'], receipt, receipt_path, namespace['write'])

    namespace['acquire_gpu_lock'] = acquire
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
