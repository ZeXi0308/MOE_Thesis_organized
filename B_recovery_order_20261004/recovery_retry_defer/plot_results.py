#!/usr/bin/env python3
"""Started retry-deferral arms: full-service CDFs and each run's own target timeline."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
HELPER_SHA = '6f6a4ef8d1358b1a7012020798508efe051420d228722edc4493ad39b185766e'
LABELS = ('N1 native', 'D1 defer_once', 'D2 defer_once', 'N2 native')
COLORS = ('#245A81', '#CE722B', '#A94420', '#508CB2')


def load_helpers():
    path = ROOT.parent/'recovery_fit/plot_results.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError('Frozen complete-request plotting helpers changed')
    spec = importlib.util.spec_from_file_location('retry_frozen_plot', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


HELPERS = load_helpers()


def number(value, digits=3):
    return f'{value:.{digits}f}' if HELPERS.finite(value) else '?'


def started_cells(data):
    indexed = {}
    for cell in data.get('cells', []):
        match = re.search(r'cell-(\d+)-cap\d+-', cell.get('directory', ''))
        if not match or not 0 <= int(match[1]) < 4:
            raise ValueError('Expected native/defer-once ABBA cell index 0 through 3')
        index = int(match[1]); expected = 'defer_once' if index in (1, 2) else 'native'
        if index in indexed or cell.get('mode') != expected:
            raise ValueError('Duplicate cell or mode inconsistent with planned order')
        indexed[index] = cell
    return sorted(indexed.items())


def target_chain(cell):
    action = cell.get('retry_defer_actions', {})
    if action.get('status') != 'ANALYZED':
        return dict(note='Action observation unavailable or unverified')
    rows = action.get('rows', [])
    if not rows:
        return dict(note='No legal retry opportunity observed; zero deferral')
    if len(rows) != 1:
        return dict(note='UNVERIFIED: multiple target selections')
    row = rows[0]; evidence = row.get('observed_target_evidence', {})
    recovery = evidence.get('recovery_to_next_output') or {}
    episode = evidence.get('action_preemption_episode') or {}
    start = recovery.get('demand_host_s', episode.get('begin_s'))
    decision = row.get('decision_s'); output = evidence.get('next_output_after_decision_s')
    lower = evidence.get('decision_to_output_wait_lower_bound_s')
    if not all(HELPERS.finite(value) for value in (start, decision)):
        return dict(note='Preemption / decision join unavailable')
    finish = output if HELPERS.finite(output) else decision+lower if HELPERS.finite(lower) else None
    if not HELPERS.finite(finish) or not start <= decision <= finish:
        return dict(note='Full preemption-to-output boundary unavailable or nonmonotone')
    first, release = row.get('first_break_s'), row.get('release_s')
    gate = (first-start, release-start) if row.get('actual_gate_executed') is True and all(
        HELPERS.finite(value) for value in (first, release)) and start <= first <= release else None
    whole = evidence.get('whole_request_recovery_summary') or {}
    return dict(start=start, decision=decision-start, finish=finish-start,
        next_output_observed=HELPERS.finite(output), actual_gate=gate,
        selected_target=row.get('target', {}).get('source_request') or 'UNMAPPED TARGET',
        gate_executed=row.get('actual_gate_executed'), release_reason=row.get('release_reason'),
        requested_breaks=row.get('requested_breaks'), executed_breaks=row.get('executed_breaks'),
        whole_request_recovery_union=whole.get('cumulative_recovery_union_lower_bound_s'),
        later_preemptions_after_next_output=evidence.get('later_preemptions_after_next_output'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New .png or .pdf; never overwrite')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    suffix = args.output.suffix.lower().lstrip('.')
    if suffix not in ('png', 'pdf'): parser.error('--output must end in .png or .pdf')
    payload = args.metrics.read_bytes(); data = json.loads(payload); cells = started_cells(data)
    if not cells: raise ValueError('No started cells; no measured figure generated')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig = plt.figure(figsize=(13.3, 9.8))
    grid = fig.add_gridspec(3, 2, height_ratios=(2.5, 1.15, 2.5), hspace=.55)
    for index, (key, title, xlabel) in enumerate((
        ('flow_s', '(a) Complete-request latency', 'External arrival → completion (s)'),
        ('maxgap_s', '(b) Per-request maximum generation gap', 'Maximum output gap (s)'))):
        ax = fig.add_subplot(grid[0, index]); seen = []; missing = []
        for arm, cell in cells:
            observed = HELPERS.distribution(cell, key)
            if observed is None or not observed[0]:
                missing.append(LABELS[arm]+': completed metric unavailable'); continue
            values, denominator = observed
            ax.step([values[0], *values], [0, *[(j+1)/denominator for j in range(len(values))]],
                where='post', color=COLORS[arm], ls='--' if arm in (2, 3) else '-', lw=1.7, label=LABELS[arm])
            seen.extend(values)
        if key == 'maxgap_s' and seen and min(seen) > 0: ax.set_xscale('log')
        if missing: ax.text(.02, .98, '\n'.join(missing), transform=ax.transAxes, va='top', fontsize=8)
        ax.set(title=title, xlabel=xlabel, ylabel='Fraction of all planned requests in this arm', ylim=(0, 1.025))
        ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
        if ax.lines: ax.legend(loc='lower right', frameon=False, fontsize=8)

    table_rows = []
    for arm, cell in cells:
        summary = cell.get('run_summary', {}); actions = cell.get('retry_defer_actions', {})
        completed = HELPERS.count(cell.get('completed'))+'/'+HELPERS.count(cell.get('planned'))
        failure = '/'.join(HELPERS.count(cell.get(key)) for key in ('failed', 'unfinished'))
        failure += '/'+HELPERS.count(summary.get('missing_planned_request_rows'))
        outputs = HELPERS.count(summary.get('actual_outputs_known'))
        if summary.get('output_count_missing_requests') != 0: outputs += ' (partial)'
        slo = cell.get('joint_slo', {})
        table_rows.append([LABELS[arm], completed+'\n'+failure, outputs,
            number(cell.get('output_tokens_per_s'), 1),
            HELPERS.count(slo.get('passes'))+'/'+HELPERS.count(slo.get('denominator')),
            number(slo.get('goodput_rps')), HELPERS.count(actions.get('actual_gate_count')),
            HELPERS.count(actions.get('requested_break_count'))+' / '+HELPERS.count(actions.get('executed_break_count'))])
    table_ax = fig.add_subplot(grid[1, :]); table_ax.axis('off')
    table = table_ax.table(cellText=table_rows,
        colLabels=['Arm', 'Completed / N\nfailed / unfin. / missing', 'Known output tokens', 'Output tokens/s',
                   'Joint SLO passes / N', 'Goodput (req/s)', 'Actual\ngates', 'Requested / executed\nbreaks'],
        colWidths=[.12, .19, .14, .10, .13, .10, .08, .14], loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(7.7); table.scale(1, 2.0)
    for (row, _), item in table.get_celld().items():
        item.set_edgecolor('#DFE4E8')
        if row == 0: item.set_facecolor('#EEF2F5'); item.set_text_props(weight='bold')

    ax = fig.add_subplot(grid[2, :]); chains = [target_chain(cell) for _, cell in cells]
    extent = max([chain.get('finish', 0) for chain in chains]+[
        chain['actual_gate'][1] for chain in chains if chain.get('actual_gate')], default=0)
    extent = extent*1.17 if extent > 0 else 1
    for y, ((arm, cell), chain) in enumerate(zip(cells, chains)):
        if 'finish' not in chain:
            ax.text(.01, y, chain['note'], transform=ax.get_yaxis_transform(), va='center', fontsize=8, color='#8B3E32')
            continue
        ax.barh(y-.09, chain['finish'], height=.16, color='#D5DCE3')
        ax.plot([chain['decision'], chain['finish']], [y+.1]*2, color=COLORS[arm], lw=1.6)
        ax.scatter(chain['decision'], y+.1, marker='v', s=32, color=COLORS[arm], zorder=4)
        ax.scatter(chain['finish'], y+.1, marker='D' if chain['next_output_observed'] else '>',
                   s=30, color=COLORS[arm], zorder=4)
        if chain.get('actual_gate'):
            begin, end = chain['actual_gate']
            ax.barh(y+.1, end-begin, left=begin, height=.2, color='#E2B056', edgecolor='#8C681F', linewidth=.5, zorder=3)
            ax.plot([end, end], [y-.03, y+.23], color='#222222', lw=1)
        label = ('' if chain['next_output_observed'] else '≥ ')+number(chain['finish'])+' s'
        ax.text(chain['finish']+.012*extent, y+.1, label, va='center', fontsize=8)
        note = chain['selected_target']+' | preempt t='+number(chain['start'])+' s'
        note += ' | '+(str(chain['release_reason']) if chain['gate_executed'] else 'native path; no executed gate')
        note += ' | later preempts='+HELPERS.count(chain['later_preemptions_after_next_output'])
        ax.text(.002, y+.34, note, transform=ax.get_yaxis_transform(), va='center', fontsize=7.5, color='#424B52')
    ax.set(yticks=range(len(cells)), yticklabels=[LABELS[arm] for arm, _ in cells],
        ylim=(len(cells)-.4, -.55), xlim=(0, extent), xlabel='Host time since this target’s preceding preemption (s)',
        title='(c) Each run’s selected target: full preemption interval, decision, executed gate, and next output')
    ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
    fig.legend(handles=[Patch(facecolor='#D5DCE3', label='Preempt → next output'),
        Line2D([], [], color='#444444', marker='v', label='Observed decision → next output'),
        Patch(facecolor='#E2B056', edgecolor='#8C681F', label='Actual gate: first executed break → release'),
        Line2D([], [], color='#222222', marker='|', ls='', label='Extra gate released')],
        loc='lower center', bbox_to_anchor=(.53, .099), ncol=2, frameon=False, fontsize=8)
    layout = data.get('execution_layout', {}); unstarted = layout.get('planned_but_not_started', [])
    fig.suptitle('Bounded recovery-retry deferral: full-service outcomes and actual execution',
                 x=.055, ha='left', y=.975, fontsize=15, weight='bold')
    fig.text(.055, .938, args.metrics.parent.name+'  |  Started arms: '+', '.join(LABELS[arm] for arm, _ in cells), fontsize=9)
    if unstarted:
        fig.text(.055, .914, 'Planned but not started: '+', '.join(f"cell {row['cell_index']}: {row['mode']}" for row in unstarted)+
                 '; these are not failed scientific requests.', fontsize=8, color='#8B3E32')
    fig.text(.055, .035,
        'CDF numerators use finite completed metrics; all planned requests within each started arm remain in the denominator, including failed, unfinished and missing.\n'
        'SLO thresholds are inherited unchanged. A selected gate may execute several loop breaks; requested breaks are reported separately from executed breaks.\n'
        'Each target is selected within its own run; local chains are not matched-state cross-run counterfactuals. Overlapping intervals are not added as savings.\n'
        'Release removes only the extra gate, not native capacity waits. Host observations are not GPU finish instants; full requests retain later preemptions.',
        fontsize=7.4, linespacing=1.35)
    sha = hashlib.sha256(payload).hexdigest()
    fig.text(.985, .009, 'Input SHA256 '+sha, ha='right', fontsize=6.3, color='#59646C')
    fig.subplots_adjust(left=.11, right=.98, top=.855, bottom=.265)
    with args.output.open('xb') as stream: fig.savefig(stream, format=suffix, dpi=220, facecolor='white')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), metrics_sha256=sha, started_arms=len(cells), planned_unstarted=len(unstarted))))


if __name__ == '__main__':
    main()
