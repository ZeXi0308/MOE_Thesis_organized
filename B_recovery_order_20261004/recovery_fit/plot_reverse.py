#!/usr/bin/env python3
"""BAAB labels/order around the frozen recovery-fit plotter; metric semantics unchanged."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '6f6a4ef8d1358b1a7012020798508efe051420d228722edc4493ad39b185766e'


def adapted_source():
    path = ROOT/'plot_results.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen recovery-fit plotter changed')
    text = path.read_text()
    changes = {
        "ARMS = ('N1 native', 'C1 fit_once', 'C2 fit_once', 'N2 native')":
            "ARMS = ('C1 fit_once', 'N1 native', 'N2 native', 'C2 fit_once')",
        "COLORS = ('#245A81', '#CE722B', '#A94420', '#508CB2')":
            "COLORS = ('#CE722B', '#245A81', '#508CB2', '#A94420')",
        'Expected recovery-fit ABBA cell indices 0 through 3':
            'Expected recovery-fit BAAB cell indices 0 through 3',
        "('fit_once' if index in (1, 2) else 'native')":
            "('native' if index in (1, 2) else 'fit_once')",
        'Cell mode does not match native/fit_once/fit_once/native':
            'Cell mode does not match fit_once/native/native/fit_once',
        'Execution order: native / fit_once / fit_once / native':
            'Development repeat order: fit_once / native / native / fit_once',
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('Frozen reverse plot presentation boundary changed: '+old)
        text = text.replace(old, new)
    return text


def main():
    namespace = dict(__name__='recovery_fit_reverse_plot', __file__=str(__file__))
    exec(compile(adapted_source(), str(__file__)+'[frozen-plot]', 'exec'), namespace)
    return namespace['main']()


if __name__ == '__main__':
    main()
