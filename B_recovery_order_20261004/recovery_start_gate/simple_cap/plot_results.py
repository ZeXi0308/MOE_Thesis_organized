#!/usr/bin/env python3
"""Native cap224/cap256: four raw run points and complete-service costs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

CAPS = (256, 224, 224, 256)
LABELS = ('cap256-1', 'cap224-1', 'cap224-2', 'cap256-2')
COLORS = ('#245A81', '#CE722B', '#A94420', '#508CB2')
MARKERS = ('o', 's', 'D', '^')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value, digits=3):
    return f'{value:.{digits}f}' if finite(value) else '?'


def count(value):
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else '?'


def started_cells(data):
    result = {}
    for cell in data.get('cells', []):
        match = re.search(r'cell-(\d+)-cap(\d+)-native(?:/|$)', cell.get('directory', ''))
        if not match:
            raise ValueError('Expected static-cap native cell directory')
        index, cap = map(int, match.groups())
        if index not in range(4) or index in result or cap != CAPS[index]:
            raise ValueError('Expected unique cap256/cap224/cap224/cap256 execution order')
        if cell.get('mode') not in (None, 'native') or cell.get('declared_cap') not in (None, cap):
            raise ValueError('Cell mode / declared cap differs from static baseline design')
        result[index] = cell
    layout = data.get('execution_layout', {})
    if layout.get('expected_caps', list(CAPS)) != list(CAPS):
        raise ValueError('Canonical expected cap order differs from this figure')
    for item in layout.get('planned_but_not_started', []):
        index = item['cell_index']
        if index not in range(4) or index in result or item.get('cap') != CAPS[index] or item.get('mode') != 'native':
            raise ValueError('Invalid planned-but-unstarted cell metadata')
    return sorted(result.items())


def values(cell):
    """Read canonical scalars only; never impute unknowns or recompute quantiles."""
    observed = cell.get('observed', {})
    gap = cell.get('primary_maxgap', {})
    exact = gap.get('exact_complete_distribution') is True
    gaps = gap.get('observed_closed_gaps', {}) if exact else {}
    return dict(ttft_mean=observed.get('ttft_s', {}).get('mean'),
        ttft_p95=observed.get('ttft_s', {}).get('p95'),
        gap_mean=gaps.get('mean'), gap_p99=gaps.get('p99'), exact_gap=exact,
        flow_mean=observed.get('flow_s', {}).get('mean'),
        flow_p95=observed.get('flow_s', {}).get('p95'),
        throughput=cell.get('output_tokens_per_s'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True, help='Canonical simple-cap-metrics.json')
    parser.add_argument('--output', type=Path, required=True, help='New .png or .pdf; never overwrite')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suffix = args.output.suffix.lower().lstrip('.')
    if suffix not in ('png', 'pdf'):
        parser.error('--output must end in .png or .pdf')
    payload = args.metrics.read_bytes()
    data = json.loads(payload)
    cells = started_cells(data)
    scalars = {index: values(cell) for index, cell in cells}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig = plt.figure(figsize=(14.5, 7.6))
    grid = fig.add_gridspec(2, 3, height_ratios=(2.8, 1.35), hspace=.43)
    panels = (
        ('ttft_mean', 'gap_mean', '(a) Mean gap vs TTFT cost', 'TTFT mean (s)', 'All-request maximum gap: mean (s)'),
        ('ttft_p95', 'gap_p99', '(b) Tail gap vs TTFT cost', 'TTFT P95 (s)', 'All-request maximum gap: P99 (s)'),
        ('flow_mean', 'throughput', '(c) Completion cost and throughput', 'Arrival to completion: mean (s)', 'Output tokens / second'),
    )
    for panel, (xkey, ykey, title, xlabel, ylabel) in enumerate(panels):
        ax = fig.add_subplot(grid[0, panel])
        missing = []
        for index, cell in cells:
            row = scalars[index]; x, y = row[xkey], row[ykey]
            if not finite(x) or not finite(y):
                missing.append(LABELS[index]); continue
            ax.scatter(x, y, marker=MARKERS[index], color=COLORS[index], s=58, zorder=3,
                       facecolors=COLORS[index] if row['exact_gap'] else 'none')
            ax.annotate(LABELS[index], (x, y), xytext=(6, 7 if index < 2 else -13),
                        textcoords='offset points', fontsize=8, color=COLORS[index])
        if missing:
            ax.text(.02, .98, 'Unavailable: '+', '.join(missing), transform=ax.transAxes,
                    va='top', fontsize=7.3, color='#8B3E32', wrap=True)
        if not cells:
            ax.text(.5, .5, 'No started cells', ha='center', transform=ax.transAxes)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.margins(x=.30, y=.25); ax.ticklabel_format(style='plain', useOffset=False)
        ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)

    table_rows = []
    for index, cell in cells:
        row = scalars[index]; summary = cell.get('run_summary', {})
        output = count(summary.get('actual_outputs_known'))
        if summary.get('output_count_missing_requests') != 0:
            output += ' (partial)'
        failures = '/'.join(count(cell.get(key)) for key in ('failed', 'unfinished'))
        failures += '/'+count(summary.get('missing_planned_request_rows'))
        gap = number(row['gap_mean'])+' / '+number(row['gap_p99']) if row['exact_gap'] else 'Not exact'
        native = {True: 'yes', False: 'NO'}.get(cell.get('no_B_intervention_verified'), '?')
        table_rows.append([LABELS[index]+'\n'+str(cell.get('status', 'UNKNOWN')),
            number(row['ttft_mean'])+' / '+number(row['ttft_p95']), gap,
            number(row['flow_mean'])+' / '+number(row['flow_p95']), number(row['throughput'], 1),
            count(cell.get('completed'))+'/'+count(cell.get('planned'))+'\n'+failures,
            output, native])
    table_ax = fig.add_subplot(grid[1, :]); table_ax.axis('off')
    if table_rows:
        table = table_ax.table(cellText=table_rows, colLabels=[
            'Run / status', 'TTFT mean / P95\n(s)', 'Maxgap mean / P99\n(s; exact only)',
            'Flow mean / P95\n(s)', 'Output tok/s', 'Completed / N\nfailed / unfin. / missing',
            'Known output\ntokens', 'No B action\nverified'],
            colWidths=[.135, .14, .14, .14, .09, .165, .12, .07], loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1, 2.15)
        for (row, _), item in table.get_celld().items():
            item.set_edgecolor('#DFE4E8')
            if row == 0:
                item.set_facecolor('#EEF2F5'); item.set_text_props(weight='bold')
    else:
        table_ax.text(.5, .5, 'No measured request counts or outcomes.', ha='center')

    fig.suptitle('Native fixed-cap baseline: TTFT cost versus generation gaps',
                 x=.055, ha='left', y=.98, fontsize=14.5, weight='bold')
    fig.text(.055, .938, args.metrics.parent.name+' | Exploratory same-workload independent runs; four raw run points, no pooled estimate.', fontsize=9)
    fig.legend(handles=[Line2D([], [], ls='', marker=MARKERS[index], color=COLORS[index],
                              label=LABELS[index]) for index, _ in cells],
               loc='upper center', bbox_to_anchor=(.51, .912), ncol=4, frameon=False, fontsize=8.5)
    unstarted = [LABELS[index] for index in range(4) if index not in dict(cells)]
    if unstarted:
        fig.text(.055, .855, 'Not started: '+', '.join(unstarted)+'; not failed scientific requests.', fontsize=8, color='#8B3E32')
    fig.text(.055, .065,
        'Each point is one observed run; identical request IDs across runs do not imply matched runtime states. No cross-run causal matching or request-level significance.\n'
        'All-request maxgap mean/P99 are shown only when canonical marks the distribution exact. Missing or incomplete maxgap is unavailable, never zero.\n'
        'TTFT/flow are canonical observed summaries; with unfinished requests they are partial (hollow completion/throughput point). All counts and output work remain visible.\n'
        'Both arms use native recovery. This is an ordinary fixed-concurrency baseline, not a recovery mechanism; historical SLOs are not retuned or used to select a winner.',
        fontsize=7.3, linespacing=1.35)
    sha = hashlib.sha256(payload).hexdigest()
    fig.text(.985, .012, 'Input SHA256 '+sha, ha='right', fontsize=6.2, color='#59646C')
    fig.subplots_adjust(left=.065, right=.975, top=.815, bottom=.23, wspace=.40)
    with args.output.open('xb') as stream:
        fig.savefig(stream, format=suffix, dpi=220, facecolor='white')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), metrics_sha256=sha,
                          started_runs=len(cells), planned_unstarted=unstarted)))


if __name__ == '__main__':
    main()
