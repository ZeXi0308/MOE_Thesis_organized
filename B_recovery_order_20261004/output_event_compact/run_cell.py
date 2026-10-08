#!/usr/bin/env python3
"""Only output-event storage differs; recovery policy and GC observation stay fixed."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT_SHA = '4d780917ad3e8b1cfd5fbc0758ec09b58180726b5c5751e27ced625303bcd00a'


def adapt_source(text):
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Output storage adapter boundary changed: '+old[:90])
        text = text.replace(old, new)
    replace('        from request_measurement import measure_episode',
        "        from compact_capture import get_capture, annotate\n"
        "        measure_episode = get_capture(os.environ['B_OUTPUT_EVENT_STORAGE'])")
    old = "        config['B_gc_observation'] = 'passive_callbacks'"
    replace(old, old+"\n        config['B_output_event_storage'] = os.environ['B_OUTPUT_EVENT_STORAGE']"
        +f"\n        config['output_event_storage_sha256'] = {hashlib.sha256((ROOT/'compact_capture.py').read_bytes()).hexdigest()!r}"
        +f"\n        config['output_event_storage_runner_sha256'] = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!r}")
    old = "                timing['measurement_return_perf_s']=time.perf_counter()"
    replace(old, old+"\n                annotate(raw, os.environ['B_OUTPUT_EVENT_STORAGE'])")
    return text


def main(source_transform=None):
    if os.environ.get('B_OUTPUT_EVENT_STORAGE') not in ('legacy', 'compact'):
        raise ValueError('B_OUTPUT_EVENT_STORAGE must be legacy or compact')
    path = BASE/'recovery_gc_diag/run_cell.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen passive-GC cell changed')
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('storage_gc_cell', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    original = parent.adapt_source
    parent.adapt_source = lambda text: adapt_source(original(text))
    return parent.main(source_transform=source_transform)


if __name__ == '__main__':
    raise SystemExit(main())
