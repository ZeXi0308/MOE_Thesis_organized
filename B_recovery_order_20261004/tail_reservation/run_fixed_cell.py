#!/usr/bin/env python3
"""Controlled fixed1024 cell; reuse the frozen full-tail and termination adapters."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PINS = {
    'tail_reservation/run_cell.py': 'e68c6502289d6e41ccfd9ea319758b557cec8a2687c4bbf0cd00266be2d6f0f7',
    'tail_reservation/run_group.py': 'd8adf5bc2e509fd178b829b505811301f29a347a154bd66e62384cd2f0bebfe1',
    'tail_reservation/reserve_tail.py': 'fd663e48fbe4291878c16319ffb6294ef63e4444e79f7f84789e13d2aad0ef76',
    'capacity_handoff/observe_tail.py': 'b0e59962645ba95f76bb380bec2f9c3675ceac62d7b57debb24a184cbb9c5c8f',
    'completion_handoff/run_fixed_group.py': '6846e19f8582f7b0d1a6fa83f7c8a3cf84aab6daf9ee69437de07d8b754fa6cf',
}


def load_fixed():
    for name, expected in PINS.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen fixed-tail execution source changed: '+name)
    path = BASE/'completion_handoff/run_fixed_group.py'
    spec = importlib.util.spec_from_file_location('fixed_tail_termination', path)
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    fixed.verify_sources()
    return fixed


def main():
    fixed = load_fixed()

    def termination_source(text):
        text = fixed.fixed_termination_source(text)
        # This is a standalone cell entry, unlike the parent's --cell dispatcher.
        text = fixed.replace_once(text,
            "shlex.join([sys.executable, sys.argv[0], '--cell', *sys.argv[1:]])",
            'shlex.join([sys.executable, *sys.argv])')
        sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return fixed.replace_once(text, "        config['recovery_lease_mode']=lease_mode",
            f"        config['fixed_output_cell_sha256']={sha!r}\n"
            "        config['recovery_lease_mode']=lease_mode")

    path = ROOT/'run_cell.py'
    text = fixed.replace_once(path.read_text(), '        text = original()',
        '        text = termination_source(original())')
    namespace = dict(__name__='fixed_tail_cell', __file__=str(path),
                     termination_source=termination_source)
    exec(compile(text, str(path)+'[fixed1024]', 'exec'), namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
