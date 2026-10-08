#!/usr/bin/env python3
"""Plot raw-backed recovery-fit metrics; missing observations stay explicitly missing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ARMS = ('N1 native', 'C1 fit_once', 'C2 fit_once', 'N2 native')
COLORS = ('#245A81', '#CE722B', '#A94420', '#508CB2')
MARKERS = {
    'Allocation return (async prefix)': ('o', False),
    'Allocation return (non-delayed)': ('o', True),
    'LOAD submit (host)': ('^', True),
    'Completion observed (host)': ('x', True),
    'Scheduler ACK (host)': ('+', True),
    'Resumed schedule plan': ('s', True),
    'Next client output': ('D', True),
}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def arms(data):
    result = [None] * 4
    for cell in data.get('cells', []):
        match = re.search(r'cell-(\d+)-cap\d+-', str(cell.get('directory', '')))
        if not match or not 0 <= int(match[1]) < 4:
            raise ValueError('Expected recovery-fit ABBA cell indices 0 through 3')
        index = int(match[1])
        if result[index] is not None:
            raise ValueError('Duplicate cell index')
        if cell.get('mode') != ('fit_once' if index in (1, 2) else 'native'):
            raise ValueError('Cell mode does not match native/fit_once/fit_once/native')
        result[index] = cell
    return result


def distribution(cell, key):
    if cell is None or type(cell.get('planned')) is not int or cell['planned'] <= 0:
        return None
    rows = cell.get('run_summary', {}).get('per_request')
    if not isinstance(rows, list):
        return None
    ids = [row.get('request') for row in rows]
    if None in ids or len(set(ids)) != len(ids) or len(rows) > cell['planned']:
        return None
    values = sorted(row[key] for row in rows if row.get('status') == 'completed'
                    and finite(row.get(key)) and row[key] >= 0)
    return values, cell['planned']


def chains(cell):
    if cell is None:
        return [dict(note='Cell unavailable')] * 2
    action = cell.get('recovery_fit_actions', {})
    if action.get('status') != 'ANALYZED':
        return [dict(note='UNVERIFIED action record: '+str(action.get('status', 'missing')))] * 2
    rows = action.get('rows', [])
    if not rows:
        return [dict(note='No qualifying fit opportunity observed; no intervention')] * 2
    if len(rows) != 1:
        return [dict(note='UNVERIFIED: multiple decision rows')] * 2
    row = rows[0]
    decision = row.get('decision_s')
    if not finite(decision):
        return [dict(note='UNVERIFIED: decision timestamp missing')] * 2
    result = []
    for role in ('candidate_head', 'baseline_head'):
        identity = row.get(role, {})
        matches = [entry for entry in row.get('request_evidence', [])
                   if entry.get('internal_request') == identity.get('internal_request')]
        if len(matches) != 1:
            result.append(dict(note='UNVERIFIED: request join unavailable or ambiguous'))
            continue
        entry = matches[0]
        finish = entry.get('decision_to_next_output_s')
        lower = entry.get('decision_to_output_wait_lower_bound_s')
        points = []
        for key, label in (
            ('first_successful_native_allocation', 'Allocation return (async prefix)'),
            ('first_successful_non_delayed_allocation', 'Allocation return (non-delayed)')):
            allocation = entry.get(key) or {}
            # The first success may itself be non-delayed; draw it once.
            if key == 'first_successful_native_allocation' and allocation.get('delay_cache_blocks') is not True:
                continue
            if finite(allocation.get('end_s')):
                points.append((label, allocation['end_s']-decision))
        for job in entry.get('load_jobs_after_decision') or []:
            for key, label in (('submit_begin_s', 'LOAD submit (host)'),
                               ('job_completed_s', 'Completion observed (host)'),
                               ('ack_retired_s', 'Scheduler ACK (host)')):
                if finite(job.get(key)):
                    points.append((label, job[key]-decision))
        schedule = entry.get('decision_to_schedule_plan_s')
        if finite(schedule):
            points.append(('Resumed schedule plan', schedule))
        if finite(finish):
            points.append(('Next client output', finish))
        # Never extrapolate missing allocation, LOAD or schedule boundaries.
        if any(stamp < 0 for _, stamp in points) or (finite(lower) and lower < 0):
            result.append(dict(note='UNVERIFIED: event precedes decision'))
            continue
        intervened = row.get('actual_candidate_intervention') is True
        role_label = ('Chosen successor' if intervened else 'Fit successor (shadow)') if role == 'candidate_head' else (
            'Bypassed head' if intervened else 'Native head')
        note = entry.get('source_request') or 'UNMAPPED REQUEST'
        if entry.get('allocation_observation') != 'AVAILABLE':
            note += ' | allocation chain unavailable'
        result.append(dict(role=role_label, points=points, finish=finish, lower=lower, note=note,
                           intervened=intervened))
    return result


def count(value):
    return f'{value:,}' if type(value) is int else '?'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New .png or .pdf; never overwrite')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suffix = args.output.suffix.lower().lstrip('.')
    if suffix not in ('png', 'pdf'):
        parser.error('--output must end in .png or .pdf')
    payload = args.metrics.read_bytes()
    cells = arms(json.loads(payload))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig = plt.figure(figsize=(13.2, 10.8))
    grid = fig.add_gridspec(3, 2, height_ratios=(2.5, 1, 3.4), hspace=.48)
    for index, (key, title, xlabel) in enumerate((
        ('flow_s', '(a) Complete-request latency', 'External arrival → completion (s)'),
        ('maxgap_s', '(b) Per-request maximum generation gap', 'Maximum output gap (s)'))):
        ax = fig.add_subplot(grid[0, index]); all_values = []; missing = []
        for i, cell in enumerate(cells):
            values = distribution(cell, key)
            if values is None or not values[0]:
                missing.append(ARMS[i]+(': unavailable' if values is None else ': no finite completed values'))
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

    table_ax = fig.add_subplot(grid[1, :]); table_ax.axis('off'); table_rows = []
    for i, cell in enumerate(cells):
        if cell is None:
            table_rows.append([ARMS[i], 'unavailable', '?', '?', '?', '?'])
            continue
        summary, action = cell.get('run_summary', {}), cell.get('recovery_fit_actions', {})
        outputs = count(summary.get('actual_outputs_known'))
        if summary.get('output_count_missing_requests') != 0:
            outputs += ' (incomplete count)'
        failures = '/'.join(count(cell.get(key)) for key in ('failed', 'unfinished'))
        failures += '/'+count(summary.get('missing_planned_request_rows'))
        table_rows.append([ARMS[i], count(cell.get('completed'))+'/'+count(cell.get('planned')), failures,
            outputs, count(action.get('actual_candidate_intervention_count'))+' / '+str(action.get('status', '?')),
            summary.get('fixed1024_contract', {}).get('status', '?')])
    table = table_ax.table(cellText=table_rows,
        colLabels=['Arm', 'Completed / planned', 'Failed / unfinished / missing', 'Known output tokens', 'Queue changes / checks', 'Fixed1024 checks'],
        colWidths=[.13, .15, .22, .18, .18, .14], loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.45)
    for (row, _), item in table.get_celld().items():
        item.set_edgecolor('#DFE4E8')
        if row == 0:
            item.set_facecolor('#EEF2F5'); item.set_text_props(weight='bold')

    ax = fig.add_subplot(grid[2, :]); observed = [chain for cell in cells for chain in chains(cell)]
    extents = [stamp for chain in observed for _, stamp in chain.get('points', [])]
    extents += [chain['lower'] for chain in observed if finite(chain.get('lower'))]
    xmax = max(extents, default=0)
    xmax = xmax*1.16 if xmax > 0 else 1  # Display extent only; never a measured zero.
    labels = []
    for i, chain in enumerate(observed):
        arm_index = i//2
        labels.append(ARMS[arm_index]+'\n'+chain.get('role', 'Successor' if i%2 == 0 else 'Head'))
        if 'points' not in chain:
            ax.text(.012, i, chain['note'], transform=ax.get_yaxis_transform(), va='center', fontsize=8, color='#8B3E32')
            continue
        finish, lower = chain.get('finish'), chain.get('lower')
        extent = finish if finite(finish) else lower
        if finite(extent):
            ax.hlines(i, 0, extent, color=COLORS[arm_index], lw=1.7, ls='-' if finite(finish) else '--')
            if not finite(finish):
                ax.scatter(extent, i, marker='>', color=COLORS[arm_index], s=25, zorder=4)
            ax.text(extent+.012*xmax, i, ('' if finite(finish) else '≥ ')+f'{extent:.3f} s', va='center', fontsize=8)
        else:
            ax.text(.012, i, 'Next output and observation cutoff unavailable', transform=ax.get_yaxis_transform(), fontsize=8)
        for label, stamp in chain['points']:
            marker, filled = MARKERS[label]
            style = dict(color=COLORS[arm_index]) if filled else dict(facecolors='white', edgecolors=COLORS[arm_index])
            ax.scatter(stamp, i, marker=marker, s=32, zorder=4, **style)
        ax.text(.004, i+.29, chain['note'], transform=ax.get_yaxis_transform(), fontsize=7.2, va='center', color='#424B52')
    ax.axvline(0, color='#333333', lw=.7)
    ax.set(yticks=range(8), yticklabels=labels, ylim=(7.55, -.55), xlim=(-.01*xmax, xmax),
           xlabel='Host time since within-run queue decision (s)',
           title='(c) Successor and original head: observed queue decision → next output')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], color='#444444', marker=marker, markerfacecolor='#444444' if filled else 'white',
                      ls='', label=label) for label, (marker, filled) in MARKERS.items()]
    handles.append(Line2D([], [], color='#444444', marker='>', ls='--', label='No next output; observed lower bound'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .085), ncol=3, frameon=False, fontsize=7.5)
    fig.suptitle('Recovery-fit probe: complete-request outcomes and observed execution', x=.06, ha='left', y=.98, fontsize=15, weight='bold')
    fig.text(.06, .948, args.metrics.parent.name+'  |  Execution order: native / fit_once / fit_once / native', fontsize=9)
    fig.text(.06, .032,
        'CDF numerator: finite completed-request metrics; denominator: all planned requests, including failed, unfinished and missing.\n'
        'Native decisions are shadow observations. Queue changes, allocation returns and schedule plans are distinct from GPU execution.\n'
        'All timeline markers are host observations; no GPU finish instants are inferred. Across-arm chains are not matched-state counterfactuals.\n'
        'Intervals include compute and observation delays; local timing differences are not summed or claimed as full-service savings.',
        fontsize=7.5, linespacing=1.35)
    sha = hashlib.sha256(payload).hexdigest()
    fig.text(.99, .008, 'Input SHA256 '+sha, ha='right', fontsize=6.4, color='#59646C')
    fig.subplots_adjust(left=.145, right=.985, top=.904, bottom=.235)
    with args.output.open('xb') as stream:
        fig.savefig(stream, format=suffix, dpi=220, facecolor='white')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), metrics_sha256=sha, arms_present=sum(c is not None for c in cells))))


if __name__ == '__main__':
    main()
