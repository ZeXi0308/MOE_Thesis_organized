#!/usr/bin/env python3
"""Full-service yield outcomes and separate deferred-head / pending-LOAD request paths."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[1]
PARENT_SHA = '8864aef7922107316fc639e38f7fdb321a4075e41844b4ef92917628f27be9e7'
LABELS = ('N1 native', 'Y1 yield_once', 'Y2 yield_once', 'N2 native')
COLORS = ('#245A81', '#CE722B', '#A94420', '#508CB2')


def load_parent():
    path = BASE/'recovery_retry_defer/plot_results.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen full-service figure source changed')
    spec = importlib.util.spec_from_file_location('start_yield_frozen_plot', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    return parent


PARENT = load_parent()
HELPERS = PARENT.HELPERS
number = PARENT.number


def started_cells(data):
    cells = {}
    for cell in data.get('cells', []):
        match = re.search(r'cell-(\d+)-cap\d+-', cell.get('directory', ''))
        if not match or not 0 <= int(match[1]) < 4: raise ValueError('Expected native/yield_once arm index 0..3')
        index = int(match[1]); expected = 'yield_once' if index in (1, 2) else 'native'
        if index in cells or cell.get('mode') != expected: raise ValueError('Duplicate arm or unexpected mode')
        cells[index] = cell
    return sorted(cells.items())


def request_path(arm, action, evidence, role):
    decision = action.get('decision_s'); output = evidence.get('next_output_after_decision_s')
    lower = evidence.get('decision_to_output_wait_lower_bound_s')
    wait = output-decision if all(HELPERS.finite(value) for value in (decision, output)) else lower
    if not HELPERS.finite(decision) or not HELPERS.finite(wait) or wait < 0:
        return dict(arm=arm, role=role, note='Decision / next-output join unavailable')
    loads = []
    for joined in evidence.get('pending_load_jobs_at_decision', []):
        job = joined.get('observation') or {}
        loads.append(dict(job_id=joined.get('job_id'), current_batch=joined.get('registered_in_decision_batch'),
            submit=job['submit_begin_s']-decision if HELPERS.finite(job.get('submit_begin_s')) else None,
            ack=job['ack_retired_s']-decision if HELPERS.finite(job.get('ack_retired_s')) else None))
    allocation = evidence.get('first_successful_non_delayed_allocation') or {}
    schedule = evidence.get('first_resumed_schedule_plan_s')
    return dict(arm=arm, role=role, decision=decision, wait=wait,
        next_output_observed=HELPERS.finite(output), source=evidence.get('source_request') or 'UNMAPPED REQUEST',
        actual_break=action.get('actual_yield_executed'), loads=loads,
        allocation=allocation['begin_s']-decision if HELPERS.finite(allocation.get('begin_s')) else None,
        schedule=schedule-decision if HELPERS.finite(schedule) else None,
        later_preempts=evidence.get('later_preemptions_after_next_output'),
        current_batch_count=sum(job['current_batch'] is True for job in loads))


def local_paths(cells):
    result = []
    for arm, cell in cells:
        actions = cell.get('recovery_start_yield_actions', {})
        if actions.get('status') != 'ANALYZED':
            result.append(dict(arm=arm, role='observation', note='Action records unavailable or unverified')); continue
        rows = actions.get('rows', [])
        if not rows:
            result.append(dict(arm=arm, role='no action', note='No legal opportunity observed; zero executed yield')); continue
        for action in rows:
            head_role = 'deferred head' if action.get('actual_yield_executed') else 'native head (shadow)'
            result.append(request_path(arm, action, action.get('deferred_request_evidence', {}), head_role))
            for index, evidence in enumerate(action.get('beneficiary_request_evidence', []), 1):
                result.append(request_path(arm, action, evidence, 'pending LOAD '+str(index)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New .png or .pdf; never overwrite')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    suffix = args.output.suffix.lower().lstrip('.')
    if suffix not in ('png', 'pdf'): parser.error('--output must end in .png or .pdf')
    payload = args.metrics.read_bytes(); data = json.loads(payload)
    cells = started_cells(data); paths = local_paths(cells)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    extra_height = max(0, len(paths)-6)*.38
    fig = plt.figure(figsize=(13.6, 10.2+extra_height))
    grid = fig.add_gridspec(3, 2, height_ratios=(2.5, 1.15, 2.6+extra_height), hspace=.55)
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
        if not cells: missing.append('No started arms; no measured distribution')
        if missing: ax.text(.02, .98, '\n'.join(missing), transform=ax.transAxes, va='top', fontsize=8)
        ax.set(title=title, xlabel=xlabel, ylabel='Fraction of all planned requests in this arm', ylim=(0, 1.025))
        ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
        if ax.lines: ax.legend(loc='lower right', frameon=False, fontsize=8)
    table_rows = []
    for arm, cell in cells:
        summary = cell.get('run_summary', {}); actions = cell.get('recovery_start_yield_actions', {})
        completed = HELPERS.count(cell.get('completed'))+'/'+HELPERS.count(cell.get('planned'))
        failure = '/'.join(HELPERS.count(cell.get(key)) for key in ('failed', 'unfinished'))
        failure += '/'+HELPERS.count(summary.get('missing_planned_request_rows'))
        outputs = HELPERS.count(summary.get('actual_outputs_known'))
        if summary.get('output_count_missing_requests') != 0: outputs += ' (partial)'
        slo = cell.get('joint_slo', {})
        table_rows.append([LABELS[arm], completed+'\n'+failure, outputs, number(cell.get('output_tokens_per_s'), 1),
            HELPERS.count(slo.get('passes'))+'/'+HELPERS.count(slo.get('denominator')), number(slo.get('goodput_rps')),
            HELPERS.count(actions.get('legal_opportunities')),
            HELPERS.count(actions.get('requested_break_count'))+' / '+HELPERS.count(actions.get('executed_break_count'))])
    table_ax = fig.add_subplot(grid[1, :]); table_ax.axis('off')
    if table_rows:
        table = table_ax.table(cellText=table_rows,
            colLabels=['Arm', 'Completed / N\nfailed / unfin. / missing', 'Known output tokens', 'Output tokens/s',
                       'Joint SLO passes / N', 'Goodput (req/s)', 'Legal\nopportunities', 'Requested / executed\nbreaks'],
            colWidths=[.12, .19, .14, .10, .13, .10, .09, .13], loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(7.6); table.scale(1, 2.0)
        for (row, _), item in table.get_celld().items():
            item.set_edgecolor('#DFE4E8')
            if row == 0: item.set_facecolor('#EEF2F5'); item.set_text_props(weight='bold')
    else: table_ax.text(.5, .5, 'No executed cells; no request counts or outcomes invented.', ha='center')
    ax = fig.add_subplot(grid[2, :]); stamps = [0.]
    for path in paths:
        if 'wait' not in path: continue
        stamps.extend(value for value in (path['wait'], path['allocation'], path['schedule']) if HELPERS.finite(value))
        stamps.extend(value for job in path['loads'] for value in (job['submit'], job['ack']) if HELPERS.finite(value))
    lower, upper = min(stamps), max(stamps); span = max(upper-lower, .01)
    for y, path in enumerate(paths):
        if 'wait' not in path:
            ax.text(.005, y, path['note'], transform=ax.get_yaxis_transform(), va='center', fontsize=8, color='#8B3E32'); continue
        color = COLORS[path['arm']]
        ax.plot([0, path['wait']], [y, y], color=color, lw=2, ls='-' if path['actual_break'] else '--')
        ax.scatter(0, y, marker='v', color=color, s=26, zorder=4)
        ax.scatter(path['wait'], y, marker='D' if path['next_output_observed'] else '>', color=color, s=29, zorder=4)
        for value, marker, ink in ((path['allocation'], '^', '#32694B'), (path['schedule'], '|', '#6F537F')):
            if HELPERS.finite(value): ax.scatter(value, y, marker=marker, color=ink, s=38, zorder=5)
        for job in path['loads']:
            for value, marker in ((job['submit'], 's'), (job['ack'], 'o')):
                if HELPERS.finite(value): ax.scatter(value, y, marker=marker, facecolors='white', edgecolors='#373737', s=31, zorder=6)
        label = ('' if path['next_output_observed'] else '≥ ')+number(path['wait'])+' s'
        ax.text(path['wait']+.012*span, y, label, va='center', fontsize=8)
        note = path['source']+' | decision t='+number(path['decision'])+' s'
        if path['loads']: note += ' | LOAD jobs='+str(len(path['loads']))+' (decision batch '+str(path['current_batch_count'])+')'
        note += ' | later preempts='+HELPERS.count(path['later_preempts'])
        ax.text(.002, y+.33, note, transform=ax.get_yaxis_transform(), va='center', fontsize=7.3, color='#424B52')
    if paths:
        ax.set(yticks=range(len(paths)), yticklabels=[LABELS[path['arm']].split()[0]+' '+path['role'] for path in paths],
            ylim=(len(paths)-.35, -.55), xlim=(lower-.025*span, upper+.17*span))
    else:
        ax.text(.5, .5, 'No local observations from started arms.', ha='center', transform=ax.transAxes); ax.set_yticks([])
    ax.set(xlabel='Host time relative to this run’s observed decision (s)',
           title='(c) Separate request paths: recompute head and each recorded pending-LOAD beneficiary')
    ax.axvline(0, color='#B7BEC4', lw=.8); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.legend(handles=[
        Line2D([], [], color='#555555', marker='v', label='Observed decision'),
        Line2D([], [], color='#555555', marker='D', label='Next client output'),
        Line2D([], [], ls='', marker='s', markerfacecolor='white', color='#373737', label='LOAD submit begins'),
        Line2D([], [], ls='', marker='o', markerfacecolor='white', color='#373737', label='Scheduler ACK'),
        Line2D([], [], ls='', marker='^', color='#32694B', label='Non-delayed allocation'),
        Line2D([], [], ls='', marker='|', color='#6F537F', label='Resumed schedule plan')],
        loc='lower center', bbox_to_anchor=(.54, .104), ncol=3, frameon=False, fontsize=8)
    layout = data.get('execution_layout', {}); unstarted = layout.get('planned_but_not_started', [])
    fig.suptitle('One-round recovery-start yield: full-service outcomes and actual execution',
                 x=.055, ha='left', y=.975, fontsize=14.5, weight='bold')
    fig.text(.055, .941, args.metrics.parent.name+' | Started arms: '+(', '.join(LABELS[arm] for arm, _ in cells) or 'none'), fontsize=9)
    notes = []
    if unstarted: notes.append('Planned but not started: '+', '.join(f"cell {row['cell_index']} {row['mode']}" for row in unstarted)+'; not failed scientific requests.')
    zero = [LABELS[arm] for arm, cell in cells if cell['mode'] == 'yield_once'
            and cell.get('recovery_start_yield_actions', {}).get('executed_break_count') == 0]
    if zero: notes.append('Zero executed breaks: '+', '.join(zero)+'. These outcome differences cannot be attributed to yield.')
    if notes: fig.text(.055, .905, '\n'.join(notes), fontsize=7.8, color='#8B3E32', linespacing=1.3)
    fig.text(.055, .031,
        'All planned requests within each started arm remain in the CDF/SLO denominator; finite completed metrics form CDF numerators. Fixed SLO thresholds are unchanged.\n'
        'Local paths are actual within-run observations, not matched-state cross-run counterfactuals. Native shadow lines show the unchanged native path.\n'
        'A pending-LOAD beneficiary label records the policy’s candidate role, not a proven beneficiary or earlier arrival. The recompute head is a separate request.\n'
        'Earlier ACK or a shorter local interval alone is not full-service gain; overlapping paths are not added as savings. Host times are not GPU completion instants.\n'
        'A one-round break does not fix the delay or guarantee next-round allocation. Hollow LOAD markers cover recorded pending job IDs, including earlier-ready jobs.',
        fontsize=7.1, linespacing=1.3)
    sha = hashlib.sha256(payload).hexdigest()
    fig.text(.985, .009, 'Input SHA256 '+sha, ha='right', fontsize=6.2, color='#59646C')
    fig.subplots_adjust(left=.17, right=.98, top=.855, bottom=.27)
    with args.output.open('xb') as stream: fig.savefig(stream, format=suffix, dpi=220, facecolor='white')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), metrics_sha256=sha, started_arms=len(cells),
                          local_path_rows=len(paths), planned_unstarted=len(unstarted))))


if __name__ == '__main__':
    main()
