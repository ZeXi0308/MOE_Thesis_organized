#!/usr/bin/env python3
"""Frozen full-service figure adapted to native/yield_ack and the two-round bound."""
import hashlib
import importlib.util
import inspect
from pathlib import Path

PARENT_SHA = '4b25eac56761688769be559f089a67ab888f071f1175ea08ecee2f915d3f2fc7'


def replace_once(text, old, new):
    if text.count(old) != 1: raise RuntimeError('Frozen ACK figure boundary changed: '+old)
    return text.replace(old, new)


def adapted_namespace():
    path = Path(__file__).resolve().with_name('plot_results.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen recovery-start figure source changed')
    spec = importlib.util.spec_from_file_location('ack_yield_frozen_plot', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    namespace = dict(vars(parent))
    namespace['LABELS'] = ('N1 native', 'Y1 yield_ack', 'Y2 yield_ack', 'N2 native')
    source = inspect.getsource(parent.started_cells).replace('yield_once', 'yield_ack')
    exec(compile(source, '<ack-yield-figure-arms>', 'exec'), namespace)
    source = inspect.getsource(parent.main)
    source = replace_once(source, "cell['mode'] == 'yield_once'", "cell['mode'] == 'yield_ack'")
    source = replace_once(source,
        'One-round recovery-start yield: full-service outcomes and actual execution',
        'Recovery-start yield until ACK: full-service outcomes and actual execution')
    source = replace_once(source, '    notes = []',
        "    notes = ['At most two native rounds / two breaks; original-LOAD scheduler ACK, round limit, or native safety exit releases the rule.']")
    source = replace_once(source,
        'A one-round break does not fix the delay or guarantee next-round allocation.',
        'Release does not guarantee immediate allocation; rule lifetime is not counterfactual added latency or savings.')
    exec(compile(source, '<ack-yield-figure>', 'exec'), namespace)
    return namespace


def main():
    adapted_namespace()['main']()


if __name__ == '__main__':
    main()
