#!/usr/bin/env python3
"""Plot one descriptive matched repeat from complete host-receipt captures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=root / 'gpu_results')
    parser.add_argument('--output-dir', type=Path, default=root / 'figures')
    args = parser.parse_args()
    captures = [json.loads((args.results_dir / label / 'raw.json').read_text())
                for label in ('repeat0-native32', 'repeat0-safe')]
    ordered = [sorted(raw['requests'], key=lambda r: (r['arrival_s'], r['request_id'])) for raw in captures]
    if any(raw['status'] != 'COMPLETE' or len(rows) != 32 or
           any(r['status'] != 'completed' for r in rows) for raw, rows in zip(captures, ordered)):
        raise ValueError('Plot requires both complete 32-request episodes')
    if [(r['request_id'], r['arrival_s']) for r in ordered[0]] != [
            (r['request_id'], r['arrival_s']) for r in ordered[1]]:
        raise ValueError('Request identities or planned arrivals differ')
    colors = dict(wait='#B9C2CC', generation='#356FA0', gap='#B93737')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 9), sharex=True, sharey=True)
    maximum = max(raw['observation_end_s'] for raw in captures)
    for ax, raw, rows, title in zip(axes, captures, ordered,
            ('Native cap32: preemption allowed', 'Safe cap29: guard enabled')):
        victims = {raw['internal_to_source'][e['victim_internal_request_id']]
                   for e in raw.get('preemption_events', []) if e['original_preemption_returned']}
        for rank, request in enumerate(rows):
            times = request['token_times_s']
            arrival, first, completion = request['arrival_s'], times[0], request['completion_s']
            ax.barh(rank, first - arrival, left=arrival, color=colors['wait'], height=.58)
            ax.barh(rank, completion - first, left=first, color=colors['generation'], height=.58)
            ax.plot(completion, rank, '.', color=colors['generation'], markersize=3)
            if request['request_id'] in victims:
                begin, end = max(zip(times, times[1:]), key=lambda pair: pair[1] - pair[0])
                ax.barh(rank, end - begin, left=begin, color=colors['gap'], height=.68)
                ax.annotate(f'{end - begin:.2f} s', ((begin + end) / 2, rank),
                            ha='center', va='center', color='white', fontsize=7,
                            fontweight='bold')
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xlim(0, maximum * 1.025)
        ax.set_ylim(31.8, -.8)
        ax.set_xlabel('Seconds since episode start')
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E5E8EC', linewidth=.6)
    axes[0].set_ylabel('Request rank by planned arrival')
    axes[0].set_yticks(range(0, 32, 2), [str(i + 1) for i in range(0, 32, 2)])
    fig.suptitle('Host-observed request timelines', fontsize=16, y=.975)
    fig.legend(handles=[Patch(color=colors['wait'], label='Arrival to first token receipt'),
                        Patch(color=colors['generation'], label='First to completion receipt'),
                        Patch(color=colors['gap'], label='Longest ITL in a preempted request')],
               loc='lower center', bbox_to_anchor=(.5, .088), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .038, 'One matched repeat; descriptive. Both panels use the same time scale.\n'
             'All spans use host-observed token receipt times. The generation span includes scheduling,\n'
             'recomputation and delivery gaps; it is not a measurement of pure GPU execution time.',
             ha='center', va='center', fontsize=9, color='#40464E')
    fig.subplots_adjust(left=.07, right=.985, bottom=.185, top=.92, wspace=.08)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / f'repeat0_request_timelines.{suffix}' for suffix in ('png', 'pdf')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Figure outputs already exist; use a new output directory')
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(json.dumps({'outputs': [str(path) for path in paths], 'evidence': 'one matched repeat; descriptive'}))


if __name__ == '__main__':
    main()
