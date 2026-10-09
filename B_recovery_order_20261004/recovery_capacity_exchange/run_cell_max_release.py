#!/usr/bin/env python3
"""Maximum-release sibling of the frozen after-failure fixed-workload child."""
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '165232c69efe3e9021b6e386d42d1d97f2a03a19ae77cedbb9229b267d310628'
POLICY_SHA = 'e97d14a7be79e421cdcde0189bc3b73fa6accfd4e7e41e8a391bc9dc308ede20'
path = ROOT/'run_cell_after_failure.py'
if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
    raise RuntimeError('Frozen after-failure child changed')
spec = importlib.util.spec_from_file_location('max_release_child_parent', path)
PARENT = importlib.util.module_from_spec(spec); spec.loader.exec_module(PARENT)


def adapt_source(text):
    text = PARENT.adapt_source(text)
    for before, after, count in (
        ('from exchange_after_failure import', 'from exchange_max_release import', 2),
        (PARENT.POLICY_SHA, POLICY_SHA, 1),
        (PARENT_SHA, hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 1)):
        if text.count(before) != count:
            raise RuntimeError('Maximum-release child boundary changed: '+before)
        text = text.replace(before, after)
    return text


def main(capture_source=None):
    source = inspect.getsource(PARENT.main)
    before = "'exchange_after_failure.py'"
    if source.count(before) != 1:
        raise RuntimeError('Expected one frozen after-failure policy path')
    source = source.replace(before, "'exchange_max_release.py'")
    namespace = dict(PARENT.__dict__, POLICY_SHA=POLICY_SHA,
                     adapt_source=adapt_source, __file__=str(__file__))
    exec(compile(source, str(__file__)+'[frozen-after-failure-main]', 'exec'), namespace)
    return namespace['main'](capture_source)


def self_check():
    class Captured(Exception): pass
    captured = []
    def capture(text):
        compile(text, '<complete-max-release-child>', 'exec')
        captured.append(text)
        raise Captured
    old = dict(os.environ)
    try:
        for mode in ('stall8', 'exchange_once'):
            os.environ['B_RECOVERY_CAPACITY_EXCHANGE'] = mode
            try: main(capture)
            except Captured: pass
            else: raise AssertionError('Capture did not stop native execution')
            current = captured[-1]
            try: PARENT.main(capture)
            except Captured: pass
            else: raise AssertionError('Parent capture did not stop native execution')
            expected = captured[-1].replace('from exchange_after_failure import', 'from exchange_max_release import')
            expected = expected.replace(PARENT.POLICY_SHA, POLICY_SHA).replace(
                PARENT_SHA, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            assert current == expected  # No termination/arrival/observer edits.
            assert current.count('from exchange_max_release import') == 2
            assert current.count('allocation_observation=tail_data') == 1
            assert current.count('install_gc_probe()') == 1
            assert 'measure_episode = make_capture(compact_capture, last_receipts)' in current
            assert 'ignore_eos=True, min_tokens=0,' in current
            assert 'output_tokens=16, output_tokens_by_request={}' in current
    finally:
        os.environ.clear(); os.environ.update(old)
    print('PASS: both complete child modes differ only by policy import/pin and runner provenance; compact receipt, passive GC, fixed1024 and warmup unchanged. CPU only.')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']: self_check()
    else: raise SystemExit(main())
