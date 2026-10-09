#!/usr/bin/env python3
"""Complete capacity-exchange ABBA runs and two separate target/donor contrasts.

Example:
  python3 -B recovery_capacity_exchange/plot_results.py \
    --metrics SESSION/capacity-exchange-metrics.json --output SESSION/results.png

Incomplete groups are refused, including truncated request gaps or throughput.
This script only reads canonical metrics; it does not reanalyze raw observations.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
EXPECTED = ['stall8', 'exchange_once', 'exchange_once', 'stall8']
LABELS = ('S1', 'E1', 'E2', 'S2')
PAIRS = ((0, 1), (3, 2))


def plot_helpers():
    """Reuse the frozen native/service-age scalar readers and lightweight style."""
    path = BASE / 'recovery_service_age_native/plot_results.py'
    spec = importlib.util.spec_from_file_location('exchange_native_plot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.adapted_namespace()


HELPERS = plot_helpers()
finite, number, count = (HELPERS[key] for key in ('finite', 'number', 'count'))
COLORS, MARKERS = (HELPERS[key] for key in ('COLORS', 'MARKERS'))


def complete_runs(data):
    """Honor canonical completion flags; never turn r01 lower bounds into wins."""
    layout = data.get('execution_layout', {})
    cells = data.get('cells', [])
    if not cells:
        raise ValueError('UNRUN: no measured cells; no figure written')
    if layout.get('expected_abba') != EXPECTED:
        raise ValueError('INCOMPLETE: expected stall8/exchange/exchange/stall8 ABBA')
    if layout.get('complete_abba') is not True or len(cells) != 4:
        raise ValueError('INCOMPLETE: all four planned runs are required; no figure written')
    # The analyzer returns cells in execution order; retain both forward/reverse pairs.
    if [cell.get('mode') for cell in cells] != EXPECTED:
        raise ValueError('INCOMPLETE: canonical cell order does not match ABBA')
    for index, cell in enumerate(cells):
        summary = cell.get('run_summary', {})
        flags = (
            cell.get('status') == 'COMPLETE',
            cell.get('primary_maxgap', {}).get('exact_complete_distribution') is True,
            summary.get('fixed1024_contract', {}).get('status') == 'PASS',
            cell.get('input_identity', {}).get('status') == 'MATCH',
            cell.get('observed', {}).get('flow_s', {}).get('missing') == 0,
        )
        load = cell.get('copy_work', {}).get('load', {})
        if not all(flags) or load.get('missing_bytes') != 0 or load.get('completed') != load.get('jobs'):
            raise ValueError(f'INCOMPLETE: {LABELS[index]} lacks complete service/input/copy evidence; no figure written')
    comparisons = []
    for reference, candidate in PAIRS:
        pair = [row for row in data.get('comparisons', [])
                if row.get('stall8') == cells[reference]['directory']
                and row.get('exchange_once') == cells[candidate]['directory']]
        if len(pair) != 1 or pair[0].get('input_identity', {}).get('status') != 'MATCH':
            raise ValueError('INCOMPLETE: corresponding canonical pair is unavailable')
        comparisons.append(pair[0])
    return cells, comparisons


def scalars(cell):
    result = HELPERS['values'](cell)
    load = cell['copy_work']['load']
    result['load_mib'] = load['bytes'] / 2**20 if finite(load.get('bytes')) else None
    required = ('gap_mean', 'gap_p99', 'gap_max', 'flow_mean', 'flow_p95', 'throughput', 'load_mib')
    if not all(finite(result[key]) for key in required):
        raise ValueError('INCOMPLETE: a required canonical run scalar is unavailable')
    return result


def selected_requests(pair, reference, candidate):
    """Only observed selections; source-ID joins are descriptive, not causal matches."""
    actions = candidate.get('capacity_exchange_actions', {})
    if actions.get('status') != 'ANALYZED' or actions.get('actual_donor_preemptions_verified') != 1:
        return []
    result = []
    for role in ('target', 'donor'):
        selected = [row for row in pair.get('selected_request_descriptive_contrasts', []) if row.get('role') == role]
        if len(selected) != 1:
            return []
        selected = selected[0]
        source = selected.get('source_request')
        change = selected.get('paired_whole_request_change') or {}
        outcomes = [next((row for row in cell.get('per_request', []) if row.get('request') == source), {})
                    for cell in (reference, candidate)]
        if (any(row.get('status') != 'completed' or not finite(row.get('maxgap_s')) for row in outcomes)
                or not finite(change.get('maxgap_s_delta'))):
            return []
        result.append((role, source, outcomes[0]['maxgap_s'], outcomes[1]['maxgap_s'], change['maxgap_s_delta']))
    return result


def render(data, cells, comparisons, rows, destination, metrics_path, sha):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(4, 6, height_ratios=(1.65, 1.65, 1.7, .9), hspace=.73, wspace=.75)
    panels = (
        ('gap_mean', '(a) Mean of per-request maximum gaps', 'Seconds'),
        ('gap_p99', '(b) P99 of per-request maximum gaps', 'Seconds'),
        ('gap_max', '(c) Worst per-request maximum gap', 'Seconds'),
        ('flow_mean', '(d) Complete arrival-to-completion flow', 'Seconds'),
        ('throughput', '(e) Complete-run output throughput', 'Output tokens / second'),
        ('load_mib', '(f) Complete LOAD work', 'MiB'),
    )
    for panel, (key, title, ylabel) in enumerate(panels):
        row_index, col = divmod(panel, 3)
        ax = fig.add_subplot(grid[row_index, 2*col:2*col+2])
        for index, row in enumerate(rows):
            ax.scatter(index, row[key], color=COLORS[index], marker=MARKERS[index], s=46, zorder=3)
            ax.annotate(number(row[key], 3 if key != 'throughput' else 1), (index, row[key]),
                        xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
            if key == 'flow_mean':
                ax.scatter(index, row['flow_p95'], facecolors='none', edgecolors=COLORS[index],
                           marker=MARKERS[index], s=46, zorder=3)
                ax.annotate(number(row['flow_p95']), (index, row['flow_p95']), xytext=(0, 7),
                            textcoords='offset points', ha='center', fontsize=8)
        ax.set(title=title, ylabel=ylabel, xticks=range(4), xticklabels=LABELS)
        ax.set_xlim(-.5, 3.5)
        ax.margins(y=.3)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        if key == 'flow_mean':
            ax.text(.02, .02, 'Filled: mean; hollow: P95', transform=ax.transAxes, fontsize=7.5)

    for pair_index, ((reference, candidate), pair) in enumerate(zip(PAIRS, comparisons)):
        ax = fig.add_subplot(grid[2, 3*pair_index:3*pair_index+3])
        selected = selected_requests(pair, cells[reference], cells[candidate])
        ax.set_title(f'({"gh"[pair_index]}) Pair {pair_index+1}: {LABELS[candidate]} minus {LABELS[reference]} — selected whole requests')
        if not selected:
            ax.text(.5, .5, 'NO VERIFIED COMPLETE TARGET / DONOR CONTRAST',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.axis('off')
            continue
        deltas = [row[4] for row in selected]
        labels = []
        for index, (role, source, baseline, exchange, delta) in enumerate(selected):
            labels.append(f'{role}\n{source.removeprefix("b-normal-")}')
            ax.barh(index, delta, height=.35, color=COLORS[candidate], alpha=.85)
            ax.annotate(f'{delta:+.3f}s', (delta, index), xytext=(5 if delta >= 0 else -5, 0),
                        textcoords='offset points', ha='left' if delta >= 0 else 'right', va='center', fontsize=8.5)
            ax.text(.02, index + .31, f'{LABELS[reference]} {baseline:.3f}s → {LABELS[candidate]} {exchange:.3f}s',
                    transform=ax.get_yaxis_transform(), va='center', fontsize=8)
        # Common zero makes sign interpretable; these two roles are not repetitions.
        limit = max(abs(value) for value in deltas) or .001
        ax.set_xlim(min(0, min(deltas)) - .25*limit, max(0, max(deltas)) + .25*limit)
        ax.set(yticks=range(2), yticklabels=labels, xlabel='Complete maxgap change (s); negative = smaller gap')
        ax.set_ylim(-.4, 1.6)
        ax.invert_yaxis()
        ax.axvline(0, color='#59646C', lw=.8)
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right', 'left']].set_visible(False)

    table_rows = []
    for index, (cell, row) in enumerate(zip(cells, rows)):
        summary = cell['run_summary']
        actions = cell.get('capacity_exchange_actions', {})
        table_rows.append([LABELS[index], count(cell.get('completed'))+'/'+count(cell.get('planned')),
                          number(row['ttft_mean'])+' / '+number(row['ttft_p95']),
                          count(summary.get('actual_outputs_known')), count(cell['copy_work']['load'].get('jobs')),
                          count(actions.get('actual_donor_preemptions_verified')),
                          count(cell.get('baseline_stall8_actions', {}).get('actual_completed_bypass_count'))])
    table_ax = fig.add_subplot(grid[3, :]); table_ax.axis('off')
    table = table_ax.table(cellText=table_rows, colLabels=['Run', 'Completed / planned', 'TTFT mean / P95 (s)',
                          'Output tokens', 'LOAD jobs', 'Verified donor preempts', 'Stall8 bypasses'],
                          loc='center', cellLoc='center', colWidths=[.065, .16, .2, .14, .105, .18, .15])
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.45)
    for (row, _), item in table.get_celld().items():
        item.set_edgecolor('#DFE4E8')
        if row == 0:
            item.set_facecolor('#EEF2F5'); item.set_text_props(weight='bold')
    fig.suptitle('One native capacity exchange versus stall8: complete service outcomes',
                 x=.055, ha='left', y=.98, fontsize=14, weight='bold')
    fig.text(.055, .95, metrics_path.parent.name+' | S = stall8; E = exchange_once. ABBA order: S1 / E1 / E2 / S2.', fontsize=9)
    fig.text(.055, .055,
             'Four raw run points; two independent runs per arm. No pooled estimate, error bars or request-level significance.\n'
             'Pair 1 is E1 − S1; pair 2 is E2 − S2. Selected target/donor identities may differ. Complete maxgaps include all later stalls.\n'
             'Selected-request contrasts are descriptive across runs, not matched runtime states or the full influence set. Full-run panels retain all requests.\n'
             'Development action probe, not independent confirmation. LOAD bytes do not imply wall-clock savings; complete flow and throughput retain execution costs.',
             fontsize=7.5, linespacing=1.35)
    fig.text(.985, .012, 'Input SHA256 '+sha, ha='right', fontsize=6.2, color='#59646C')
    fig.subplots_adjust(left=.085, right=.975, top=.90, bottom=.16)
    with destination.open('xb') as stream:
        fig.savefig(stream, format=destination.suffix[1:].lower(), dpi=220, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True, help='Canonical capacity-exchange-metrics.json')
    parser.add_argument('--output', type=Path, help='New .png or .pdf; never overwrite')
    parser.add_argument('--check-only', action='store_true', help='Check real canonical completion without rendering')
    args = parser.parse_args()
    if not args.check_only and (args.output is None or args.output.suffix.lower() not in ('.png', '.pdf')):
        parser.error('--output must end in .png or .pdf')
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    if not args.metrics.exists():
        print('UNRUN: metrics file does not exist; no figure written', file=sys.stderr)
        return 2
    payload = args.metrics.read_bytes()
    data = json.loads(payload)
    try:
        cells, comparisons = complete_runs(data)
        rows = [scalars(cell) for cell in cells]
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    sha = hashlib.sha256(payload).hexdigest()
    if not args.check_only:
        render(data, cells, comparisons, rows, args.output, args.metrics, sha)
    print(json.dumps(dict(status='COMPLETE', output=str(args.output) if not args.check_only else None,
                          metrics_sha256=sha, independent_runs_per_arm=2, separate_pairs=2)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
