#!/usr/bin/env python3
"""Four development-run outcomes: repeat8 versus unique8 recovery bypasses."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[1]
PARENT_SHA = '016937cae32fc51348e8d971c039bc6b6d01006824799615805b1662f8118c1e'
EXPECTED = ('repeat8', 'unique8', 'unique8', 'repeat8')
LABELS = ('repeat8-1', 'unique8-1', 'unique8-2', 'repeat8-2')


def load_parent():
    path = BASE/'recovery_start_gate/simple_cap/plot_results.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen simple-cap figure source changed')
    spec = importlib.util.spec_from_file_location('unique_frozen_scatter_plot', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


PARENT = load_parent()


def started_cells(data):
    result = {}
    for cell in data.get('cells', []):
        match = re.search(r'cell-(\d+)-cap(\d+)-(repeat8|unique8)(?:/|$)', cell.get('directory', ''))
        if not match:
            raise ValueError('Expected repeat8/unique8 cell directory')
        index, cap = int(match[1]), int(match[2]); mode = match[3]
        if index not in range(4) or index in result or mode != EXPECTED[index]:
            raise ValueError('Expected unique repeat8/unique8/unique8/repeat8 execution positions')
        if cell.get('mode') not in (None, mode) or cell.get('declared_cap') not in (None, cap):
            raise ValueError('Canonical mode/cap differs from cell directory')
        result[index] = cell
    layout = data.get('execution_layout', {})
    if layout.get('expected_abba', list(EXPECTED)) != list(EXPECTED):
        raise ValueError('Canonical mode order differs from this figure')
    for item in layout.get('planned_but_not_started', []):
        index = item['cell_index']
        if index not in range(4) or index in result or item.get('mode') != EXPECTED[index]:
            raise ValueError('Invalid planned-but-unstarted cell metadata')
    return sorted(result.items())


def values(cell):
    result = PARENT.values(cell)
    result['gap_max'] = (cell.get('primary_maxgap', {}).get('observed_closed_gaps', {}).get('maximum')
                         if result['exact_gap'] else None)
    return result


def adapted_namespace():
    source = inspect.getsource(PARENT.main)

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('Frozen scatter figure adaptation boundary changed: '+old[:90])
        source = source.replace(old, new)

    replace('Canonical simple-cap-metrics.json', 'Canonical repeat-unique-metrics.json')
    replace("('ttft_mean', 'gap_mean', '(a) Mean gap vs TTFT cost', 'TTFT mean (s)', 'All-request maximum gap: mean (s)')",
            "('ttft_mean', 'gap_p99', '(a) Tail gap vs TTFT cost', 'TTFT mean (s)', 'Per-request maximum gap: P99 (s)')")
    replace("('ttft_p95', 'gap_p99', '(b) Tail gap vs TTFT cost', 'TTFT P95 (s)', 'All-request maximum gap: P99 (s)')",
            "('ttft_p95', 'gap_max', '(b) Worst gap vs TTFT cost', 'TTFT P95 (s)', 'Worst per-request maximum gap (s)')")
    replace("gap = number(row['gap_mean'])+' / '+number(row['gap_p99']) if row['exact_gap'] else 'Not exact'",
            "gap = ' / '.join(number(row[key]) for key in ('gap_mean', 'gap_p99', 'gap_max')) if row['exact_gap'] else 'Not exact'")
    replace("native = {True: 'yes', False: 'NO'}.get(cell.get('no_B_intervention_verified'), '?')",
            "action = cell.get('recovery_repeat_unique_actions', {})\n"
            "        selection = action.get('selection_observations', {})\n"
            "        action_text = count(action.get('actual_completed_bypass_count'))+' / '+count(action.get('unique_bypassed_requests'))\n"
            "        action_text += '\\n'+count(selection.get('actual_different_reorder_count'))+' / '+count(selection.get('unique_legal_suppression_count'))\n"
            "        if action.get('status') != 'ANALYZED': action_text += ' (unverified)'")
    replace('output, native])', 'output, action_text])')
    replace("'Maxgap mean / P99\\n(s; exact only)'", "'Maxgap mean / P99 / max\\n(s; exact only)'")
    replace("'No B action\\nverified'", "'Bypass / unique IDs\\nΔreorder / suppressions'")
    replace('colWidths=[.135, .14, .14, .14, .09, .165, .12, .07]',
            'colWidths=[.12, .125, .15, .125, .08, .155, .105, .14]')
    replace('Native fixed-cap baseline: TTFT cost versus generation gaps',
            'Recovery bypass reuse: repeat8 versus unique8')
    replace(' | Exploratory same-workload independent runs; four raw run points, no pooled estimate.',
            ' | Development ABBA: planned 2 runs per arm. Independent run points; no pooled estimate or error bars.')
    replace('All-request maxgap mean/P99 are shown only when canonical marks the distribution exact.',
            'All-request maxgap mean/P99/max are shown only when canonical marks the distribution exact.')
    replace('Both arms use native recovery. This is an ordinary fixed-concurrency baseline, not a recovery mechanism; historical SLOs are not retuned or used to select a winner.',
            'Bypass / IDs count executed reorders / distinct selected requests. Δreorder / suppression report actual unique8 differences from the same-state repeat8 suggestion.\\n'
            'Suppression counts are decision occurrences, not distinct requests or preemption episodes.\\n'
            'Suggestions are not alternative executed trajectories. Fewer unique8 bypasses are a real work difference. This is development evidence, not independent confirmation.')
    namespace = dict(vars(PARENT), __doc__=__doc__, LABELS=LABELS, started_cells=started_cells, values=values)
    exec(compile(source, str(__file__)+'[frozen-scatter-layout]', 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    adapted_namespace()['main']()
