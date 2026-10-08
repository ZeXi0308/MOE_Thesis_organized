#!/usr/bin/env python3
"""Reuse frozen full-request CDFs; show only observed once/repeat8 actions and outcomes."""
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '6f6a4ef8d1358b1a7012020798508efe051420d228722edc4493ad39b185766e'
ARMS = ('O1 once', 'R1 repeat8', 'R2 repeat8', 'O2 once')
COLORS = ('#245A81', '#CE722B', '#A94420', '#508CB2')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def number(value, digits=3):
    return f'{value:.{digits}f}' if finite(value) else '?'


def count(value):
    return f'{value:,}' if type(value) is int else '?'


def actual_rows(cell):
    return (cell or {}).get('recovery_repeat_actions', {}).get('rows', [])


def row_count(cells):
    return sum(max(1, len(actual_rows(cell))) if (cell or {}).get('recovery_repeat_actions', {}).get('status') == 'ANALYZED'
               else 1 for cell in cells)


def table(ax, rows, labels, widths):
    ax.axis('off')
    result = ax.table(cellText=rows, colLabels=labels, colWidths=widths,
                      loc='center', cellLoc='center')
    result.auto_set_font_size(False); result.set_fontsize(7.8); result.scale(1, 1.45)
    for (row, _), item in result.get_celld().items():
        item.set_edgecolor('#DFE4E8')
        if row == 0:
            item.set_facecolor('#EEF2F5'); item.set_text_props(weight='bold')


def draw_tables(fig, grid, cells):
    counts, performance = [], []
    for i, cell in enumerate(cells):
        if cell is None:
            counts.append([ARMS[i], 'unavailable', '?', '?', '?', '?'])
            performance.append([ARMS[i], '?', '?', '?', '?', '?', '?'])
            continue
        summary = cell.get('run_summary', {}); actions = cell.get('recovery_repeat_actions', {})
        outputs = count(summary.get('actual_outputs_known'))
        if summary.get('output_count_missing_requests') != 0: outputs += ' (partial)'
        failures = '/'.join(count(cell.get(key)) for key in ('failed', 'unfinished'))
        failures += '/'+count(summary.get('missing_planned_request_rows'))
        counts.append([ARMS[i], count(cell.get('completed'))+'/'+count(cell.get('planned')), failures,
            outputs, count(actions.get('queue_order_change_observed_count'))+' / '+str(actions.get('status', '?')),
            summary.get('fixed1024_contract', {}).get('status', '?')])
        slo = cell.get('joint_slo', {}); work = cell.get('copy_work', {})
        copies = []
        for direction in ('load', 'store'):
            copy = work.get(direction, {}); size = copy.get('bytes')
            label = count(copy.get('jobs'))+' / '+number(size/2**20 if finite(size) else None, 1)
            if copy.get('missing_bytes') != 0: label += ' (partial)'
            copies.append(label)
        performance.append([ARMS[i], number(cell.get('output_tokens_per_s'), 1),
            number(cell.get('duration_s')), count(slo.get('passes'))+'/'+count(slo.get('denominator')),
            number(slo.get('goodput_rps')), *copies])
    table(fig.add_subplot(grid[1, :]), counts,
        ['Arm', 'Completed / planned', 'Failed / unfinished / missing', 'Known output tokens', 'Queue changes / checks', 'Fixed1024 checks'],
        [.12, .15, .22, .17, .20, .14])
    table(fig.add_subplot(grid[2, :]), performance,
        ['Arm', 'Output tokens/s', 'Observation (s)', 'Joint SLO passes / N', 'Goodput (req/s)', 'LOAD jobs / MiB', 'STORE jobs / MiB'],
        [.12, .12, .13, .17, .14, .16, .16])


def draw_actions(fig, grid, cells):
    from matplotlib.lines import Line2D
    ax = fig.add_subplot(grid[3, :]); labels = []; y = 0; extent = 0; starts = []
    for i, cell in enumerate(cells):
        action = (cell or {}).get('recovery_repeat_actions', {})
        rows = actual_rows(cell)
        if not rows or action.get('status') != 'ANALYZED':
            labels.append(ARMS[i])
            note = 'No actual queue change observed' if action.get('status') == 'ANALYZED' else 'Action observations unavailable or unverified'
            ax.text(.01, y, note, transform=ax.get_yaxis_transform(), va='center', fontsize=8, color='#8B3E32')
            y += 1
            continue
        for n, row in enumerate(rows, 1):
            labels.append(ARMS[i]+f' #{n}'); decision = row.get('decision_s')
            if not finite(decision):
                ax.text(.01, y, 'Decision timestamp unavailable', transform=ax.get_yaxis_transform(), va='center', fontsize=8)
                y += 1
                continue
            extent = max(extent, decision)
            starts.append(decision)
            ax.plot([decision, decision], [y-.22, y+.22], color='#222222', lw=.8, zorder=3)
            for role, offset, marker, style in (('candidate_head', -.13, 'o', '-'), ('baseline_head', .13, 's', '--')):
                rid = row.get(role, {}).get('internal_request')
                matches = [e for e in row.get('request_evidence', []) if e.get('internal_request') == rid]
                if len(matches) != 1:
                    ax.text(decision, y+offset, '  request join unavailable', fontsize=7)
                    continue
                evidence = matches[0]; output = evidence.get('next_output_after_decision_s')
                lower = evidence.get('decision_to_output_wait_lower_bound_s')
                end = output if finite(output) else decision+lower if finite(lower) else None
                if not finite(end) or end < decision:
                    ax.text(decision, y+offset, '  next output/cutoff unavailable', fontsize=7)
                    continue
                extent = max(extent, end)
                ax.plot([decision, end], [y+offset]*2, color=COLORS[i], ls=style, lw=1.5)
                ax.plot(end, y+offset, marker=marker if finite(output) else '>', color=COLORS[i], ls='', ms=4)
            y += 1
    left = max(0, min(starts)-.5) if starts else 0
    right = extent+max(.5, .03*(extent-left)) if extent > 0 else 1
    ax.set(yticks=range(y), yticklabels=labels, ylim=(y-.45, -.55), xlim=(left, right),
        xlabel='Host time since external measurement origin (s)',
        title='(c) Actual queue decisions → next client output, one row per executed change')
    ax.tick_params(axis='y', labelsize=7.8)
    ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
    fig.legend(handles=[Line2D([], [], color='#222222', marker='|', ls='', label='Actual queue decision'),
        Line2D([], [], color='#444444', marker='o', ls='-', label='Chosen successor → next output'),
        Line2D([], [], color='#444444', marker='s', ls='--', label='Original head → next output'),
        Line2D([], [], color='#444444', marker='>', ls='', label='No next output; observed lower bound')],
        loc='lower center', bbox_to_anchor=(.54, .092), ncol=2, frameon=False, fontsize=7.5)


def adapted_source():
    path = ROOT.parent/'recovery_fit/plot_results.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen recovery-fit plotter changed')
    text = path.read_text()
    changes = {
        "ARMS = ('N1 native', 'C1 fit_once', 'C2 fit_once', 'N2 native')": "ARMS = ('O1 once', 'R1 repeat8', 'R2 repeat8', 'O2 once')",
        "('fit_once' if index in (1, 2) else 'native')": "('repeat8' if index in (1, 2) else 'once')",
        'Cell mode does not match native/fit_once/fit_once/native': 'Cell mode does not match once/repeat8/repeat8/once',
        'Expected recovery-fit ABBA cell indices 0 through 3': 'Expected recovery-repeat ABBA cell indices 0 through 3',
        'fig = plt.figure(figsize=(13.2, 10.8))': 'fig = plt.figure(figsize=(13.2, 11.8+.27*max(0, row_count(cells)-4)))',
        'grid = fig.add_gridspec(3, 2, height_ratios=(2.5, 1, 3.4), hspace=.48)':
            'grid = fig.add_gridspec(4, 2, height_ratios=(2.5, 1, 1, max(2, .30*row_count(cells))), hspace=.48)',
        'Recovery-fit probe: complete-request outcomes and observed execution': 'Repeated recovery bypass: complete-request outcomes and actual actions',
        'Execution order: native / fit_once / fit_once / native': 'Development order: once / repeat8 / repeat8 / once',
        'Native decisions are shadow observations. Queue changes, allocation returns and schedule plans are distinct from GPU execution.':
            'Only actual queue changes have timelines; shadow opportunities have no alternative-outcome curve. Joint SLO retains the frozen thresholds.',
        'All timeline markers are host observations; no GPU finish instants are inferred. Across-arm chains are not matched-state counterfactuals.':
            'All timeline markers are host observations, not GPU finish instants. Across-arm chains are not matched-state counterfactuals.',
        'Intervals include compute and observation delays; local timing differences are not summed or claimed as full-service savings.':
            'Decision-to-output intervals can overlap and include repeated preemptions; never sum them as service savings. CDFs retain whole requests.',
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('Frozen repeat plot adaptation boundary changed: '+old)
        text = text.replace(old, new)
    start = '    table_ax = fig.add_subplot(grid[1, :]); table_ax.axis(\'off\'); table_rows = []'
    end = '    fig.suptitle('
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError('Frozen table/action panel boundary changed')
    left, right = text.index(start), text.index(end)
    return text[:left]+'    draw_tables(fig, grid, cells)\n    draw_actions(fig, grid, cells)\n'+text[right:]


def main():
    namespace = dict(__name__='repeat_actual_plot', __file__=str(__file__),
                     draw_tables=draw_tables, draw_actions=draw_actions, row_count=row_count)
    exec(compile(adapted_source(), str(__file__)+'[frozen-cdf]', 'exec'), namespace)
    return namespace['main']()


if __name__ == '__main__':
    main()
