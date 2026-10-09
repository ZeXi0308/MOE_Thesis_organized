#!/usr/bin/env python3
"""Ordinary service-age baselines: four raw age8/stall8 development runs."""
import hashlib
import importlib.util
import inspect
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PARENT_SHA = '2e024a45a7a6756d3e98c84514c579e71f8308a82c9fc66944241310a9f78d6d'
EXPECTED = ('age8', 'stall8', 'stall8', 'age8')
LABELS = ('age8-1', 'stall8-1', 'stall8-2', 'age8-2')


def load_parent():
    path = BASE/'recovery_repeat_unique/plot_results.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen repeat/unique figure source changed')
    spec = importlib.util.spec_from_file_location('service_age_frozen_plot', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def adapted_namespace():
    parent = load_parent()
    namespace = dict(vars(parent), __doc__=__doc__, __file__=__file__,
                     EXPECTED=EXPECTED, LABELS=LABELS)
    source = inspect.getsource(parent.started_cells).replace('repeat8', 'age8').replace('unique8', 'stall8')
    exec(compile(source, str(__file__)+'[service-age-cell-order]', 'exec'), namespace)
    source = inspect.getsource(parent.adapted_namespace)
    source = source.replace('repeat8', 'age8').replace('unique8', 'stall8')
    replacements = (
        ('Canonical repeat-unique-metrics.json', 'Canonical service-age-metrics.json'),
        ('recovery_repeat_unique_actions', 'recovery_service_age_actions'),
        ('unique_legal_suppression_count', 'stall_legal_suppression_count'),
        ('Recovery bypass reuse: age8 versus stall8', 'Ordinary service-age baseline: age8 versus stall8'),
        ('Suggestions are not alternative executed trajectories. Fewer stall8 bypasses are a real work difference. This is development evidence, not independent confirmation.',
         r'Suggestions are not alternative executed trajectories. Any bypass-count difference is actual work; this is development evidence, not independent confirmation.\\n'
         'Age8 uses original arrival; stall8 uses last client receipt, falling back to age8 if any legal candidate receipt is unknown.'),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('Frozen service-age plot adaptation boundary changed: '+old[:80])
        source = source.replace(old, new)
    exec(compile(source, str(__file__)+'[service-age-labels-and-actions]', 'exec'), namespace)
    return namespace['adapted_namespace']()


if __name__ == '__main__':
    adapted_namespace()['main']()
