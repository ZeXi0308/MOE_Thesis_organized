#!/usr/bin/env python3
"""Plot all recovery episodes from one completed, existing four-cell group."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--metrics', type=Path, default=root/'session-20261007-r01/completion-metrics.json')
    parser.add_argument('--output', type=Path, default=root/'recovery-phase-decomposition.png')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cells = read(args.metrics)['cells']
    if len(cells) != 4:
        raise ValueError('This figure is scoped to the existing four-cell group')
    labels = ['Demand → LOAD ready', 'Ready → submit', 'Submit → host completion observation',
              'Host observation → scheduler ack', 'Ack → first scheduled', 'Scheduled → next output']
    colors = ['#B6C5CF', '#AA66AA', '#0072B2', '#56B4E9', '#D55E00', '#009E73']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.7), sharex=True)
    total, loads_total, recompute_total = 0, 0, 0
    max_elapsed = max(e['demand_to_next_output_s'] for c in cells for e in c['recovery_events'])
    for panel, (ax, cell) in enumerate(zip(axes.flat, cells)):
        cell_dir = args.metrics.parent/Path(cell['directory']).parent.name
        action = read(cell_dir/'completion-action-check.json')
        if action['status'] != 'PASS_ACTION_CHECK':
            raise ValueError('Existing action check did not pass')
        jobs = {j['job_id']: j for j in action['load_chains']}
        raw = read(cell_dir/'output/raw.json')
        events = read(cell_dir/'output/recovery-order.json')['events']
        origin, mapping = raw['measurement_origin_perf_counter_s'], raw['internal_to_source']
        episodes = sorted(cell['recovery_events'], key=lambda e: e['demand_host_s'])
        repetitions, occurrences, names = Counter(e['request'] for e in episodes), Counter(), []
        nload = nrecompute = 0
        for y, episode in enumerate(episodes):
            demand, scheduled, output = (episode[k] for k in ('demand_host_s', 'first_scheduled_plan_s', 'next_output_s'))
            if None in (demand, scheduled, output) or episode['repeated_preempts_before_output']:
                raise ValueError('Ambiguous/censored episode requires a separately labelled figure')
            request = episode['request']; occurrences[request] += 1
            label = request.replace('b-normal-', '').replace('-long', ' L').replace('-short', ' S')
            if repetitions[request] > 1:
                label += f' ({occurrences[request]})'
            if episode['load_job_ids']:
                if len(episode['load_job_ids']) != 1:
                    raise ValueError('Multiple LOAD boundaries cannot be split uniquely by this plot')
                job = jobs[episode['load_job_ids'][0]]
                bounds = [demand] + [job[k] for k in ('ready_s', 'submit_begin_s', 'job_completed_s', 'ack_retired_s')] + [scheduled, output]
                if any(v is None for v in bounds) or any(a > b for a, b in zip(bounds, bounds[1:])):
                    raise ValueError('Missing/nonmonotone host boundary')
                for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
                    ax.barh(y, b-a, left=a-demand, color=colors[i], height=.64, linewidth=0)
                failures = [t-demand for t in episode['request_host_boundaries'].get('capacity_wait', []) if job['ack_retired_s'] <= t <= scheduled]
                ax.plot(failures, [y]*len(failures), 'x', color='#222222', ms=5, mew=1.1)
                nload += 1
            else:
                lookups = [e for e in events if e['kind'] == 'lookup' and mapping.get(e.get('request')) == request
                           and demand <= e['host_perf_s']-origin <= scheduled]
                zero_hit = bool(lookups and lookups[-1]['result'] == [0, False])
                label += ' [R]' if zero_hit else ' [no LOAD]'
                ax.barh(y, scheduled-demand, color='#E1E1E1', edgecolor='#888888', hatch='///', linewidth=.35, height=.64)
                ax.barh(y, output-scheduled, left=scheduled-demand, color=colors[-1], linewidth=0, height=.64)
                nrecompute += int(zero_hit)
            ax.text(output-demand+.07, y, f'{output-demand:.2f}', va='center', fontsize=7.4, color='#444444')
            names.append(label)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_ylim(len(names)-.45, -.75)
        ax.set_xlim(0, max_elapsed+.6)
        ax.set_xticks(range(0, int(max_elapsed)+1))
        ax.grid(axis='x', color='#E7E7E7', linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        mode = 'native' if cell['mode'] == 'native' else 'after_sample'
        ax.set_title(f'({chr(97+panel)}) Cell {panel:02d}: {mode}\n{len(episodes)} episodes · {nload} LOAD / {nrecompute} zero-hit recompute', loc='left', fontsize=10, pad=10)
        ax.set_xlabel('Host elapsed time since recovery demand (s)')
        ax.tick_params(axis='x', labelbottom=True)
        total += len(episodes); loads_total += nload; recompute_total += nrecompute
    handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
    handles += [Patch(facecolor='#E1E1E1', edgecolor='#888888', hatch='///', label='Zero-hit: demand → scheduled (unsplit)'),
                Line2D([], [], marker='x', color='#222222', linestyle='none', label='Allocation failure observed after LOAD ack')]
    fig.suptitle('Observed recovery paths under natural EOS', x=.055, y=.985, ha='left', fontsize=16, fontweight='bold')
    fig.text(.055, .951, f'All {total} recovery episodes: {loads_total} LOAD, {recompute_total} zero-hit recompute. Each panel is a separate run; rows are not matched states.', fontsize=10)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .064), ncol=2, frameon=False, fontsize=8.5)
    fig.text(.055, .025, 'Host observation is a native poll, not the GPU completion instant. Submit → observation includes unobserved execution and polling delay.\nDemand → ready and ack → scheduled include capacity/retry/scheduling; no further split is claimed. Ready → submit is <0.5 ms and nearly invisible.\n[R] marks a recorded zero-hit lookup with no LOAD; its pre-scheduled interval is intentionally unsplit. No speedup is inferred from these rows.', fontsize=8, linespacing=1.5)
    fig.subplots_adjust(left=.12, right=.975, top=.89, bottom=.23, hspace=.34, wspace=.27)
    with args.output.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=240, facecolor='white')
    plt.close(fig)
    print(json.dumps({'output': str(args.output), 'episodes': total, 'LOAD': loads_total, 'zero_hit_recompute': recompute_total}))


if __name__ == '__main__':
    main()
