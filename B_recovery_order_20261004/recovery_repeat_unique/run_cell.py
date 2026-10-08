#!/usr/bin/env python3
"""Fixed1024 repeat8/unique8 cell with common compact capture and passive GC."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
POLICY_SHA = '960e5f98ce5390aa3e274cfc779329e6b2fad4d1fadf56dd6548e27279f3afae'
PINS = {
    'output_event_compact/run_cell.py': 'dc8ca7cc29c0e1f5152080eea5803d2bae5ef96490ace33c4564bb219c4c38a0',
    'source_handoff/run_fixed_cell.py': '69b03e618595bc89b29874ef846e83a5524a0fece805d932ca70b50814af2829',
    'output_event_compact/compact_capture.py': '778a62f5b6325e33541489c815c6a324c9d6c7f5723a9b26b93956b319a17683',
    'recovery_gc_diag/gc_probe.py': '9a753ff4eeec483ab21f982b10b303b4e7c3e9919d0197aa4b4afa57a5a9975b',
}


def load(relative, name):
    path = BASE/relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[relative]:
        raise RuntimeError('Frozen unique-repeat child dependency changed: '+relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def adapt_source(text):
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Unique-repeat installation boundary changed: '+old[:90])
        text = text.replace(old, new)
    replace('from repeat_fit import install as install_recovery_repeat',
            'from repeat_unique import install as install_recovery_repeat_unique')
    replace("install_recovery_repeat(scheduler, os.environ.get('B_RECOVERY_REPEAT', 'once'))",
            "install_recovery_repeat_unique(scheduler, os.environ['B_RECOVERY_REPEAT_UNIQUE'])")
    replace("config['B_recovery_repeat'] = os.environ.get('B_RECOVERY_REPEAT', 'once')",
            "config['B_recovery_repeat_unique'] = os.environ['B_RECOVERY_REPEAT_UNIQUE']")
    old = "        config['recovery_repeat_policy_sha256'] = "
    lines = [line for line in text.splitlines() if line.startswith(old)]
    if len(lines) != 1: raise RuntimeError('Expected one inherited policy provenance')
    replace(lines[0], f"        config['recovery_repeat_unique_policy_sha256'] = {POLICY_SHA!r}")
    old = "        config['recovery_repeat_runner_sha256'] = "
    lines = [line for line in text.splitlines() if line.startswith(old)]
    if len(lines) != 1: raise RuntimeError('Expected one inherited runner provenance')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    replace(lines[0], f"        config['recovery_repeat_unique_runner_sha256'] = {digest!r}")
    replace("dump(out/'recovery-repeat.json', uninstall_fit())",
            "dump(out/'recovery-repeat-unique.json', uninstall_fit())")
    return text


def main(capture_source=None):
    mode = os.environ.setdefault('B_RECOVERY_REPEAT_UNIQUE', 'repeat8')
    if mode not in ('repeat8', 'unique8'):
        raise ValueError('B_RECOVERY_REPEAT_UNIQUE must be repeat8 or unique8')
    # This inherited value satisfies the outer GC runner guard only. The final
    # compiled installation and config read the new effective mode exclusively.
    for key, value in (('B_RECOVERY_REPEAT', 'once'), ('B_OUTPUT_EVENT_STORAGE', 'compact')):
        if os.environ.setdefault(key, value) != value:
            raise ValueError('Unique-repeat child requires '+key+'='+value)
    if hashlib.sha256((ROOT/'repeat_unique.py').read_bytes()).hexdigest() != POLICY_SHA:
        raise RuntimeError('Unique-repeat policy changed')
    for relative in PINS:
        if hashlib.sha256((BASE/relative).read_bytes()).hexdigest() != PINS[relative]:
            raise RuntimeError('Frozen common observation changed: '+relative)
    sys.path.insert(0, str(ROOT))
    parent = load('output_event_compact/run_cell.py', 'unique_compact_child')
    fixed = load('source_handoff/run_fixed_cell.py', 'unique_fixed_termination')
    original = parent.adapt_source
    def adapted(text):
        text = adapt_source(original(text))
        if capture_source is not None: capture_source(text)
        return text
    parent.adapt_source = adapted
    return parent.main(source_transform=fixed.termination_source)


def self_check():
    class Captured(Exception): pass
    snapshots = []
    def capture(text):
        compile(text, '<full-unique-cell>', 'exec')
        snapshots.append(text)
        raise Captured()
    old = dict(os.environ)
    try:
        for mode in ('repeat8', 'unique8'):
            os.environ['B_RECOVERY_REPEAT_UNIQUE'] = mode
            try: main(capture)
            except Captured: pass
            else: raise AssertionError('Did not stop before native execution')
            text = snapshots[-1]
            assert text.count("install_recovery_repeat_unique(scheduler, os.environ['B_RECOVERY_REPEAT_UNIQUE'])") == 1
            assert 'from repeat_fit import' not in text and 'install_recovery_repeat(' not in text
            assert "config['B_recovery_repeat']" not in text
            assert "config['B_recovery_repeat_unique'] = os.environ['B_RECOVERY_REPEAT_UNIQUE']" in text
            assert text.count("get_capture(os.environ['B_OUTPUT_EVENT_STORAGE'])") == 1
            assert os.environ['B_OUTPUT_EVENT_STORAGE'] == 'compact'
            assert text.count('install_gc_probe()') == 1
            assert text.count("dump(out/'recovery-repeat-unique.json', uninstall_fit())") == 1
            assert text.index('uninstall_gc())') < text.index('uninstall_fit())') < text.index('uninstall_source())')
            assert "output_tokens=config['output_tokens'], output_mode='fixed'," in text
            assert "config['requests'] != 256 or config['output_tokens'] != 1024" in text
            calls = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'measure_episode']
            assert len(calls) == 1
            assert any(isinstance(arg, ast.Name) and arg.id=='config' for arg in calls[0].args)
            updates = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and ast.unparse(n.func)=='config.update'
                       and any(k.arg=='ignore_eos' for k in n.keywords)]
            assert len(updates)==1
            kw={k.arg:ast.literal_eval(k.value) for k in updates[0].keywords if k.arg in ('ignore_eos','min_tokens')}
            assert kw=={'ignore_eos':True,'min_tokens':0}
            assert 'output_tokens=16, output_tokens_by_request={}' in text
    finally:
        os.environ.clear(); os.environ.update(old)
    print('PASS: both effective modes compile; exactly one unique-repeat install; common compact/passiveGC; fixed256x1024/min0 and original16-token warmups; teardown retained. CPU only.')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']: self_check()
    else: raise SystemExit(main())
