#!/usr/bin/env python3
"""One-page figure from source_handoff/analyze.py output; never invent missing observations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ARMS = ('N1 native', 'C1 early_pin', 'C2 early_pin', 'N2 native')
COLORS = ('#245A81', '#CE722B', '#A94420', '#508CB2')
PHASES = {
    'Demand → allocation return': '#CDD5DD',
    'Allocation → LOAD submit': '#779DBD',
    'Submit → host completion observation': '#64A7AF',
    'Host observation → scheduler ACK': '#B4CEE0',
    'ACK → first resumed schedule': '#E4B15F',
    'Allocation → first resumed schedule': '#CAB4DC',
    'First resumed schedule → next output': '#83B89B',
}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def arms(data):
    result = [None] * 4
    for cell in data.get('cells', []):
        match = re.search(r'cell-(\d+)-cap\d+-', str(cell.get('directory', '')))
        if not match or not 0 <= int(match[1]) < 4:
            raise ValueError('Expected the four source-handoff ABBA cell directories')
        index = int(match[1])
        if result[index] is not None:
            raise ValueError('Duplicate cell index')
        expected = 'early_pin' if index in (1, 2) else 'native'
        if cell.get('mode') != expected:
            raise ValueError('Cell mode does not match native/early_pin/early_pin/native')
        result[index] = cell
    return result


def distribution(cell, key):
    if cell is None or type(cell.get('planned')) is not int or cell['planned'] <= 0:
        return None
    rows = cell.get('run_summary', {}).get('per_request')
    if not isinstance(rows, list):
        return None
    ids = [r.get('request') for r in rows]
    if None in ids or len(set(ids)) != len(ids) or len(rows) > cell['planned']:
        return None
    # Failed, unfinished and missing requests remain in N, never become completed latencies.
    values = sorted(r[key] for r in rows
                    if r.get('status') == 'completed' and finite(r.get(key)) and r[key] >= 0)
    return values, cell['planned']


def recovery_chain(cell):
    if cell is None:
        return dict(note='UNVERIFIED: cell missing')
    action = cell.get('source_actions', {})
    if action.get('status') != 'PASS':
        return dict(note='UNVERIFIED: action checks '+str(action.get('status', 'missing')))
    if action.get('outcome') == 'NO_SELECTION':
        return dict(note='NO_SELECTION: no qualifying recovery was selected')
    episode = action.get('selected_recovery_episode') or {}
    recovery = episode.get('request_recovery') or {}
    start, finish = recovery.get('demand_host_s'), recovery.get('next_output_s')
    attempts = [a for a in episode.get('attempts', []) if a.get('outcome') == 'success']
    allocated = min((a['end_s'] for a in attempts if finite(a.get('end_s'))), default=None)
    scheduled = recovery.get('first_scheduled_plan_s')
    if not all(finite(t) for t in (start, finish, allocated, scheduled)):
        lower = recovery.get('censored_wait_lower_bound_s')
        note = 'UNVERIFIED: complete recovery boundaries unavailable'
        if finish is None and finite(lower):
            note += f'; no next output, observed wait ≥ {lower:.3f} s'
        return dict(note=note)
    jobs = action.get('linked_load_jobs')
    if not isinstance(jobs, list) or len(jobs) > 1:
        return dict(note='UNVERIFIED: missing or multiple linked LOADs')
    stages = [('Demand → allocation return', start, allocated)]
    if jobs:
        job = jobs[0]
        submit, observed, ack = [job.get(k) for k in ('submit_begin_s', 'job_completed_s', 'ack_retired_s')]
        if not all(finite(t) for t in (submit, observed, ack)):
            return dict(note='UNVERIFIED: LOAD submit / host observation / ACK missing')
        stages.extend([
            ('Allocation → LOAD submit', allocated, submit),
            ('Submit → host completion observation', submit, observed),
            ('Host observation → scheduler ACK', observed, ack),
            ('ACK → first resumed schedule', ack, scheduled)])
    else:
        stages.append(('Allocation → first resumed schedule', allocated, scheduled))
    stages.append(('First resumed schedule → next output', scheduled, finish))
    if any(hi < lo for _, lo, hi in stages):
        return dict(note='UNVERIFIED: nonmonotone host boundaries')
    marker = next((r.get('host_s') for r in action.get('source_observations', [])
                   if r.get('kind') == 'selected' and r.get('field') == 'before'), None)
    pin = next((r.get('host_s') for r in action.get('source_observations', [])
                if r.get('kind') == 'early_pin' and r.get('field') == 'after'), None)
    note = 'LOAD' if jobs else 'No LOAD; native recompute'
    repeated = recovery.get('repeated_preempts_before_output')
    if repeated:
        note += f'; {repeated} further preemption(s) before output'
    return dict(stages=[(name, lo-start, hi-start) for name, lo, hi in stages],
        selected=marker-start if finite(marker) else None,
        pin=pin-start if finite(pin) else None, total=finish-start,
        request=action.get('selected_source_request', 'UNVERIFIED'), note=note)


def count(value):
    return f'{value:,}' if type(value) is int else 'UNVERIFIED'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True, help='source-metrics.json from the original raw-backed analyzer')
    parser.add_argument('--output', type=Path, required=True, help='New .pdf or .png file; existing files are never overwritten')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suffix = args.output.suffix.lower().lstrip('.')
    if suffix not in ('pdf', 'png'):
        parser.error('--output must end in .pdf or .png')
    payload = args.metrics.read_bytes()
    cells = arms(json.loads(payload))
    if not any(cell is not None for cell in cells):
        raise ValueError('No observed cells; no figure generated')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig = plt.figure(figsize=(13.2, 9.3))
    grid = fig.add_gridspec(3, 2, height_ratios=(2.5, 1.05, 2.6), hspace=.52)
    for index, (key, title, xlabel) in enumerate((
        ('flow_s', '(a) Complete-request latency', 'Arrival → completion (s)'),
        ('maxgap_s', '(b) Per-request maximum generation gap', 'Maximum output gap (s)'))):
        ax = fig.add_subplot(grid[0, index]); all_values = []
        missing = []
        for i, cell in enumerate(cells):
            values = distribution(cell, key)
            if values is None or not values[0]:
                missing.append(ARMS[i]+': UNVERIFIED' if values is None else ARMS[i]+': no finite completed values')
                continue
            points, denominator = values
            ax.step([points[0], *points], [0, *[(j+1)/denominator for j in range(len(points))]],
                    where='post', color=COLORS[i], ls='--' if i in (2, 3) else '-', lw=1.7, label=ARMS[i])
            all_values.extend(points)
        if missing:
            ax.text(.02, .98, '\n'.join(missing), transform=ax.transAxes, va='top', fontsize=8, color='#8B3E32')
        if key == 'maxgap_s' and all_values and min(all_values) > 0:
            ax.set_xscale('log')
        ax.set(title=title, xlabel=xlabel, ylabel='Fraction of all planned requests', ylim=(0, 1.025))
        ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
        if ax.lines:
            ax.legend(loc='lower right', frameon=False, fontsize=8)

    table_ax = fig.add_subplot(grid[1, :]); table_ax.axis('off')
    table_rows = []
    for i, cell in enumerate(cells):
        if cell is None:
            table_rows.append([ARMS[i], 'UNVERIFIED', 'UNVERIFIED', 'UNVERIFIED', 'UNVERIFIED', 'UNVERIFIED'])
            continue
        summary, action = cell.get('run_summary', {}), cell.get('source_actions', {})
        outputs = count(summary.get('actual_outputs_known'))
        if summary.get('output_count_missing_requests') != 0:
            outputs = 'UNVERIFIED; known '+outputs
        failures = '/'.join(count(cell.get(k)) for k in ('failed', 'unfinished'))
        failures += '/'+count(summary.get('missing_planned_request_rows'))
        table_rows.append([ARMS[i], count(cell.get('completed'))+'/'+count(cell.get('planned')), failures,
            outputs, count(action.get('actual_pin_count'))+' / '+(action.get('status') or 'UNVERIFIED'),
            summary.get('fixed1024_contract', {}).get('status') or 'UNVERIFIED'])
    table = table_ax.table(cellText=table_rows,
        colLabels=['Arm', 'Completed / planned', 'Failed / unfinished / missing', 'Known output tokens', 'Actual pins / checks', 'Fixed1024 checks'],
        colWidths=[.13, .15, .22, .18, .16, .16], loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(8.3); table.scale(1, 1.45)
    for (row, col), item in table.get_celld().items():
        item.set_edgecolor('#DFE4E8')
        if row == 0:
            item.set_facecolor('#EEF2F5'); item.set_text_props(weight='bold')

    ax = fig.add_subplot(grid[2, :]); chains = [recovery_chain(c) for c in cells]
    xmax = max((c.get('total', 0) for c in chains), default=0)
    xmax = xmax * 1.18 if xmax > 0 else 1  # Empty display extent, never a measured zero.
    for i, chain in enumerate(chains):
        if 'stages' not in chain:
            ax.text(.01, i, chain['note'], transform=ax.get_yaxis_transform(), va='center', fontsize=8, color='#8B3E32')
            continue
        for name, lo, hi in chain['stages']:
            ax.barh(i, hi-lo, left=lo, height=.35, color=PHASES[name], edgecolor='white', linewidth=.5)
        for key, marker in (('selected', 'v'), ('pin', '*')):
            if finite(chain.get(key)):
                ax.scatter(chain[key], i-.23, marker=marker, s=35 if key=='selected' else 65,
                           c='#222222', zorder=4, clip_on=False)
        ax.text(chain['total']+.01*xmax, i, f"{chain['total']:.3f} s", va='center', fontsize=8)
        ax.text(.005, i+.3, chain['request']+' | '+chain['note'], transform=ax.get_yaxis_transform(),
                fontsize=7.5, color='#424B52', va='center')
    ax.set(yticks=range(4), yticklabels=ARMS, ylim=(3.65, -.55), xlim=(0, xmax),
           xlabel='Host time since selected recovery demand (s)',
           title='(c) Selected recovery → next output: observed host intervals, independently selected in each arm')
    ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
    handles = [Patch(facecolor=color, label=label) for label, color in PHASES.items()]
    handles += [Line2D([], [], color='#222222', marker='v', ls='', label='Qualifying allocation failure'),
                Line2D([], [], color='#222222', marker='*', ls='', label='Early Host read-reference acquired')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .074), ncol=3, frameon=False, fontsize=7.5)
    fig.suptitle('Host-source handoff: complete-request outcomes and recovery path', x=.06, ha='left', y=.98, fontsize=16, weight='bold')
    fig.text(.06, .941, args.metrics.parent.name+'  |  Execution order: native / early_pin / early_pin / native', fontsize=9)
    fig.text(.06, .035,
        'CDF numerator: finite completed-request metrics; denominator: all planned requests, including failures, unfinished and missing.\n'
        'Completion observation and ACK are host timestamps, not GPU finish instants. Stages include computation and observation delays.\n'
        'Selected chains are not matched-state counterfactuals; their differences are not end-to-end savings. Missing boundaries are UNVERIFIED.',
        fontsize=7.5, linespacing=1.4)
    sha = hashlib.sha256(payload).hexdigest()
    fig.text(.99, .008, 'Input SHA256 '+sha, ha='right', fontsize=6.4, color='#59646C')
    fig.subplots_adjust(left=.065, right=.985, top=.89, bottom=.245)
    with args.output.open('xb') as stream:
        fig.savefig(stream, format=suffix, dpi=220, facecolor='white')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), metrics_sha256=sha, arms_present=sum(c is not None for c in cells))))


if __name__ == '__main__':
    main()
