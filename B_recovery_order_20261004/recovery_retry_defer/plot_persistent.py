#!/usr/bin/env python3
"""Frozen retry figure with an explicit zero-action-arm presentation note."""
import hashlib
import importlib.util
import inspect
from pathlib import Path

PARENT_SHA = '8864aef7922107316fc639e38f7fdb321a4075e41844b4ef92917628f27be9e7'


def main():
    path = Path(__file__).resolve().with_name('plot_results.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen retry figure source changed')
    spec = importlib.util.spec_from_file_location('persistent_retry_frozen_plot', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    source = inspect.getsource(parent.main)
    anchor = '    fig.text(.055, .035,\n'
    note = """    zero_action = [LABELS[arm] for arm, cell in cells if cell.get('mode') == 'defer_once'
                   and cell.get('retry_defer_actions', {}).get('actual_gate_count') == 0]
    if zero_action:
        fig.text(.055, .896 if unstarted else .914, 'Zero executed actions: '+', '.join(zero_action)+
                 '. Outcome differences in these arms are not evidence of a deferral effect.', fontsize=8, color='#8B3E32')
"""
    if source.count(anchor) != 1:
        raise RuntimeError('Frozen retry figure note position changed')
    namespace = dict(vars(parent))
    exec(compile(source.replace(anchor, note+anchor), '<persistent-retry-figure>', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
