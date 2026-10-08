#!/usr/bin/env python3
"""Output-event storage intervention with unchanged once policy; no recovery contribution claim."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
FROZEN_PLOT_SHA = '6f6a4ef8d1358b1a7012020798508efe051420d228722edc4493ad39b185766e'
ARMS = ('L1 legacy', 'C1 compact', 'C2 compact', 'L2 legacy')
COLORS = ('#245A81', '#CE722B', '#A94420', '#508CB2')


def load_helpers():
    path = ROOT.parent/'recovery_fit/plot_results.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_PLOT_SHA:
        raise RuntimeError('Frozen complete-request plotting helpers changed')
    spec = importlib.util.spec_from_file_location('storage_frozen_plot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = load_helpers()


def number(value, digits=3):
    return f'{value:.{digits}f}' if HELPERS.finite(value) else '?'


def cells_by_order(data):
    result = [None]*4
    for cell in data.get('cells', []):
        match = re.search(r'cell-(\d+)-cap\d+-', cell.get('directory', ''))
        if not match or not 0 <= int(match[1]) < 4:
            raise ValueError('Expected four storage-ABBA cell indices 0 through 3')
        index = int(match[1]); expected = 'compact' if index in (1, 2) else 'legacy'
        if result[index] is not None or cell.get('mode') != expected:
            raise ValueError('Duplicate index or mode inconsistent with legacy/compact/compact/legacy')
        result[index] = cell
    return result


def table(ax, rows, headers, widths, size=8):
    ax.axis('off')
    result = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                      loc='center', cellLoc='center')
    result.auto_set_font_size(False); result.set_fontsize(size); result.scale(1, 1.6)
    for (row, _), item in result.get_celld().items():
        item.set_edgecolor('#DFE4E8')
        if row == 0:
            item.set_facecolor('#EEF2F5'); item.set_text_props(weight='bold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True, help='Real storage-metrics.json')
    parser.add_argument('--output', type=Path, required=True, help='New .png or .pdf; never overwrite')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    suffix = args.output.suffix.lower().lstrip('.')
    if suffix not in ('png', 'pdf'): parser.error('--output must end in .png or .pdf')
    payload = args.metrics.read_bytes(); cells = cells_by_order(json.loads(payload))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig = plt.figure(figsize=(13.6, 10.4))
    grid = fig.add_gridspec(3, 2, height_ratios=(2.8, 1.35, 2.4), hspace=.55, wspace=.3)

    gc_ax = fig.add_subplot(grid[0, 0]); labels = []; gc_max = 0
    for i, cell in enumerate(cells):
        gc = (cell or {}).get('gc_generation2', {})
        labels.append(ARMS[i]+'; n='+HELPERS.count(gc.get('retained_event_count')))
        values = [gc.get(key) for key in ('retained_union_s', 'maximum_clipped_interval_s')]
        if gc.get('status') != 'ANALYZED' or not all(HELPERS.finite(value) for value in values):
            gc_ax.text(.01, i, 'GC observations unavailable / unverified', transform=gc_ax.get_yaxis_transform(), va='center', fontsize=8)
            continue
        for value, offset, color, name in zip(values, (-.16, .16), ('#477A9C', '#CE722B'),
                ('Retained gen-2 wall union', 'Maximum interval within measurement')):
            value *= 1000; gc_max = max(gc_max, value)
            gc_ax.barh(i+offset, value, height=.27, color=color, label=name if i == 0 else None)
            gc_ax.text(value, i+offset, '  '+number(value, 1), va='center', fontsize=7.5)
    gc_ax.set(yticks=range(4), yticklabels=labels, ylim=(3.55, -.55),
        xlim=(0, gc_max*1.22 if gc_max > 0 else 1), xlabel='Retained GC callback wall intervals (ms)',
        title='(a) Generation-2 GC during request observation')
    if gc_ax.patches: gc_ax.legend(loc='center right', bbox_to_anchor=(.985, .5), frameon=False, fontsize=7.5)
    gc_ax.grid(axis='x', alpha=.2); gc_ax.set_axisbelow(True)

    gap_ax = fig.add_subplot(grid[0, 1]); points_seen = []; missing = []
    for i, cell in enumerate(cells):
        values = HELPERS.distribution(cell, 'maxgap_s')
        if values is None or not values[0]:
            missing.append(ARMS[i]+(': unavailable' if values is None else ': no completed metric'))
            continue
        points, denominator = values
        gap_ax.step([points[0], *points], [0, *[(j+1)/denominator for j in range(len(points))]],
            where='post', color=COLORS[i], ls='--' if i in (2, 3) else '-', lw=1.7, label=ARMS[i])
        points_seen.extend(points)
    if points_seen and min(points_seen) > 0: gap_ax.set_xscale('log')
    if missing: gap_ax.text(.02, .97, '\n'.join(missing), transform=gap_ax.transAxes, va='top', fontsize=8)
    gap_ax.set(xlabel='Per-request maximum generation gap (s)', ylabel='Fraction of all planned requests',
        ylim=(0, 1.025), title='(b) Complete-request maximum-gap CDF')
    gap_ax.grid(alpha=.2)
    if gap_ax.lines: gap_ax.legend(loc='lower right', frameon=False, fontsize=8)

    service_rows = []
    for i, cell in enumerate(cells):
        if cell is None:
            service_rows.append([ARMS[i]]+['?']*8); continue
        summary = cell.get('run_summary', {}); flow = cell.get('observed', {}).get('flow_s', {})
        slo = cell.get('joint_slo', {}); output = HELPERS.count(summary.get('actual_outputs_known'))
        if summary.get('output_count_missing_requests') != 0: output += ' (partial)'
        counts = '/'.join(HELPERS.count(cell.get(key)) for key in ('failed', 'unfinished'))
        counts += '/'+HELPERS.count(summary.get('missing_planned_request_rows'))
        service_rows.append([ARMS[i], HELPERS.count(cell.get('completed'))+'/'+HELPERS.count(cell.get('planned')),
            counts, output, number(flow.get('mean'))+' / '+number(flow.get('p95')),
            number(cell.get('duration_s')), number(cell.get('output_tokens_per_s'), 1),
            HELPERS.count(slo.get('passes'))+'/'+HELPERS.count(slo.get('denominator')), number(slo.get('goodput_rps'))])
    service_ax = fig.add_subplot(grid[1, :])
    table(service_ax, service_rows,
        ['Arm', 'Complete / N', 'Failed / unfin. / missing', 'Output tokens', 'Flow mean / P95 (s)',
         'Observation (s)', 'Output tokens/s', 'SLO passes / N', 'Goodput (req/s)'],
        [.095, .10, .16, .105, .15, .10, .10, .10, .09], size=7.7)
    service_ax.set_title('(c) External-arrival request outcomes; original fixed SLO and all-request denominator', pad=12)

    cost_ax = fig.add_subplot(grid[2, 0]); material_max = 0
    for i, cell in enumerate(cells):
        storage = (cell or {}).get('output_event_storage', {})
        value = (cell or {}).get('run_summary', {}).get('phase_times', {}).get('output_event_materialization_s')
        if storage.get('status') != 'ANALYZED' or not HELPERS.finite(value):
            cost_ax.text(.01, i, 'Materialization unavailable / unverified', transform=cost_ax.get_yaxis_transform(), va='center', fontsize=8)
            continue
        value *= 1000; material_max = max(material_max, value)
        cost_ax.barh(i, value, height=.5, color=COLORS[i])
        cost_ax.text(value, i, '  '+number(value, 2), va='center', fontsize=8)
    cost_ax.set(yticks=range(4), yticklabels=ARMS, ylim=(3.6, -.6),
        xlim=(0, material_max*1.23 if material_max > 0 else 1),
        xlabel='Post-observation materialization (host ms)',
        title='(d) Deferred output-record construction cost')
    cost_ax.grid(axis='x', alpha=.2); cost_ax.set_axisbelow(True)

    phase_rows = []
    for i, cell in enumerate(cells):
        phase = (cell or {}).get('run_summary', {}).get('phase_times', {})
        phase_rows.append([ARMS[i]]+[number(phase.get(key), 3 if key == 'post_request_drain_s' else 2) for key in (
            'engine_init_s', 'warmup_s', 'measurement_s', 'post_request_drain_s', 'shutdown_s', 'process_s')])
    phase_ax = fig.add_subplot(grid[2, 1])
    table(phase_ax, phase_rows, ['Arm', 'Init', 'Warmup', 'Capture\nreturn', 'Post\ndrain', 'Shutdown', 'Process'],
        [.20, .12, .13, .15, .12, .14, .14], size=7.2)
    phase_ax.set_title('(e) Reported host phases (seconds)', pad=14)
    phase_ax.text(0, .05, 'Materialization is included in capture return and process time.\n'
                  'Values are reported separately; no stage sums imply request savings.',
                  transform=phase_ax.transAxes, fontsize=7.6, va='bottom')
    for axis in (gc_ax, gap_ax, cost_ax): axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Output-event storage intervention; same recovery policy in every arm',
                 x=.055, ha='left', y=.975, fontsize=15, weight='bold')
    fig.text(.055, .94, args.metrics.parent.name+
             '  |  legacy / compact / compact / legacy; all arms use once (budget 1)', fontsize=9)
    fig.text(.055, .055,
        'Observation-storage experiment, not a recovery-scheduling contribution. CDF numerator uses completed finite metrics; all planned requests remain in N.\n'
        'GC callback spans are host wall observations, not CPU consumption or exact pause times. Retained gen-2 counts/union/max cover the recorded measurement window.\n'
        'Compact materialization happens after request observation and remains in capture-return/process costs. Legacy record construction occurs during capture.\n'
        'GC intervals, transfer durations and host phases are not added together or converted into request savings; no historical run is used as a control.',
        fontsize=7.4, linespacing=1.35)
    sha = hashlib.sha256(payload).hexdigest()
    fig.text(.985, .015, 'Input SHA256 '+sha, ha='right', fontsize=6.3, color='#59646C')
    fig.subplots_adjust(left=.12, right=.975, top=.875, bottom=.21)
    with args.output.open('xb') as stream: fig.savefig(stream, format=suffix, dpi=220, facecolor='white')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), metrics_sha256=sha,
                          arms_present=sum(cell is not None for cell in cells))))


if __name__ == '__main__':
    main()
