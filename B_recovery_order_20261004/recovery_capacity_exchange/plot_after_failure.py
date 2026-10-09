#!/usr/bin/env python3
"""Reuse the capacity-exchange figure with explicit zero-action explanations.

Usage is identical to plot_results.py. Only panel/footer wording changes; all
completion checks, canonical values, pair identities and rendering are reused.
"""
import importlib.util
import inspect
from pathlib import Path
import sys


def main():
    path = Path(__file__).with_name('plot_results.py')
    spec = importlib.util.spec_from_file_location('after_failure_base_plot', path)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    source = inspect.getsource(parent.render)
    missing = 'NO VERIFIED COMPLETE TARGET / DONOR CONTRAST'
    footer = ('Development action probe, not independent confirmation. LOAD bytes do not imply '
              'wall-clock savings; complete flow and throughput retain execution costs.')
    zero_text = ('ZERO EXECUTED EXCHANGES ({})\nNo selected target / donor contrast.\n'
                 'Latency variation is not a mechanism effect.')
    replacements = (
        (repr(missing), '(' + repr(zero_text) + '.format(LABELS[candidate]) if '
         "cells[candidate].get('capacity_exchange_actions', {}).get('status') == 'ANALYZED' and "
         "cells[candidate].get('capacity_exchange_actions', {}).get('actual_donor_preemptions_verified') == 0 "
         'else ' + repr(missing) + ')'),
        (repr(footer), repr(footer + '\nZero-exchange candidate runs have no target/donor contrast; their latency variation is not an executed exchange effect.')),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('Base figure wording boundary changed: ' + old[:80])
        source = source.replace(old, new)
    exec(compile(source, str(__file__) + '[wording-only]', 'exec'), vars(parent))
    return parent.main()


if __name__ == '__main__':
    sys.exit(main())
