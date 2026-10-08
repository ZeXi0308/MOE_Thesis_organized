#!/usr/bin/env python3
"""Frozen gate figure in BAAB order G1/N1/N2/G2; original nearest-native comparisons."""
import hashlib
import importlib.util
from pathlib import Path
import re

PARENT_SHA = '82fbff5ed5665169f4d85e4a713101e71fc5c8f9f4d22e34e31182f090480682'
EXPECTED = ('wait_release', 'native', 'native', 'wait_release')


def started_cells(data):
    indexed = {}
    for cell in data.get('cells', []):
        match = re.search(r'cell-(\d+)-cap\d+-', cell.get('directory', ''))
        if not match or not 0 <= int(match[1]) < 4:
            raise ValueError('Expected BAAB cell index 0..3')
        index = int(match[1])
        if index in indexed or cell.get('mode') != EXPECTED[index]:
            raise ValueError('Duplicate arm or mode inconsistent with BAAB')
        indexed[index] = cell
    layout = data.get('execution_layout', {})
    if layout.get('expected_abba') is not None and tuple(layout['expected_abba']) != EXPECTED:
        raise ValueError('Canonical planned order is not BAAB')
    for row in layout.get('planned_but_not_started', []):
        index = row.get('cell_index')
        if type(index) is not int or not 0 <= index < 4 or row.get('mode') != EXPECTED[index] or index in indexed:
            raise ValueError('Unstarted-arm metadata is inconsistent with observed BAAB cells')
    cells = sorted(indexed.items()); positions = {cell['directory']: i for i, (_, cell) in enumerate(cells)}
    natives = [i for i, (_, cell) in enumerate(cells) if cell['mode'] == 'native']
    for pair in data.get('comparisons', []):
        candidate = positions.get(pair.get('candidate')); native = positions.get(pair.get('native'))
        if candidate is not None and native is not None and natives:
            nearest = min(natives, key=lambda i: (abs(i-candidate), i))
            if cells[candidate][1]['mode'] != 'wait_release' or native != nearest:
                raise ValueError('Canonical comparison differs from original nearest-native rule')
    return cells


def adapted_namespace():
    path = Path(__file__).resolve().with_name('plot_results.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen ABBA gate figure source changed')
    spec = importlib.util.spec_from_file_location('reverse_gate_frozen_plot', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    namespace = parent.adapted_namespace()
    namespace.update(__doc__=__doc__, started_cells=started_cells,
        LABELS=('G1 wait_release', 'N1 native', 'N2 native', 'G2 wait_release'),
        COLORS=('#CE722B', '#245A81', '#508CB2', '#A94420'))
    return namespace


def main():
    adapted_namespace()['main']()


if __name__ == '__main__':
    main()
