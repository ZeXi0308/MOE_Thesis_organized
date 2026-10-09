#!/usr/bin/env python3
"""Native versus stall8: ordinary service-age baseline on a new arrival permutation."""
import hashlib
import importlib.util
import inspect
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PARENT_SHA = '8b6ad098280210691c3c1da65bbee1ba54ab9ed526f6cbe52d02b1b8fbf68c9f'
EXPECTED = ('native', 'stall8', 'stall8', 'native')
LABELS = ('native-1', 'stall8-1', 'stall8-2', 'native-2')


def load_parent():
    path = BASE/'recovery_service_age/plot_results.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen service-age figure source changed')
    spec = importlib.util.spec_from_file_location('native_service_age_frozen_plot', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def adapted_namespace():
    parent = load_parent()
    source = inspect.getsource(parent.adapted_namespace)

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('Frozen native plot adaptation boundary changed: '+old[:90])
        source = source.replace(old, new)

    # Only the cell/reference role becomes native. Age8 remains the selection
    # diagnostic and the unknown-receipt fallback, never the run-level reference.
    replace("inspect.getsource(parent.started_cells).replace('repeat8', 'age8')",
            "inspect.getsource(parent.started_cells).replace('repeat8', 'native')")
    replace("'Canonical service-age-metrics.json'", "'Canonical service-age-native-metrics.json'")
    replace("'recovery_service_age_actions'", "'recovery_service_age_native_actions'")
    replace("'Ordinary service-age baseline: age8 versus stall8'",
            "'Native versus stall8: ordinary service-age baseline'")
    extra = (
        ('actual_completed_bypass_count', 'actual_mutations_versus_native'),
        ('Bypass / unique IDs', 'Actual bypass / IDs'),
        ('Δreorder / suppressions', 'Δage / suppressions'),
        ('Development ABBA: planned 2 runs per arm. Independent run points; no pooled estimate or error bars.',
         'Development ABBA: same texts, new arrival permutation; planned 2 runs/arm. Raw run points, no pooled estimate or error bars.'),
        ('Bypass / IDs count executed reorders / distinct selected requests. Δreorder / suppression report actual stall8 differences from the same-state age8 suggestion.',
         'Actual bypass / IDs: mutations versus native / distinct target IDs. Δage / suppressions compare only against the same-state age8 suggestion.'),
        ('Age8 uses original arrival; stall8 uses last client receipt, falling back to age8 if any legal candidate receipt is unknown.',
         'Native executes no bypasses. Stall8 uses last client receipt and falls back to arrival order (not native) when a legal receipt is unknown.'),
        ('    namespace = dict(vars(PARENT),',
         '    replace('+repr('xytext=(6, 7 if index < 2 else -13),')+', '
         +repr("xytext=(-6 if index == 0 else 6, 7 if index < 2 else -13), ha='right' if index == 0 else 'left',")
         +')\n    namespace = dict(vars(PARENT),'),
    )
    replace('    for old, new in replacements:',
            '    replacements += '+repr(extra)+'\n    for old, new in replacements:')
    namespace = dict(vars(parent), __doc__=__doc__, __file__=__file__, EXPECTED=EXPECTED, LABELS=LABELS)
    exec(compile(source, str(__file__)+'[native-roles-and-age-diagnostic]', 'exec'), namespace)
    return namespace['adapted_namespace']()


if __name__ == '__main__':
    adapted_namespace()['main']()
