#!/usr/bin/env python3
"""Plot one observed GC diagnostic; callback wall intervals are not CPU times."""
import argparse
import hashlib
import json
from pathlib import Path

COLORS = {0: '#477A9C', 1: '#41988F', 2: '#CE722B'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New .png or .pdf; never overwrite')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    suffix = args.output.suffix.lower().lstrip('.')
    if suffix not in ('png', 'pdf'): parser.error('--output must end in .png or .pdf')
    payload = args.metrics.read_bytes(); data = json.loads(payload)
    if len(data.get('cells', [])) != 1:
        raise ValueError('This figure requires exactly one observed diagnostic cell')
    cell = data['cells'][0]; diagnostic = cell.get('gc_diagnostic', {})
    if diagnostic.get('status') != 'ANALYZED':
        raise ValueError('GC diagnostic is unavailable or unverified')
    ranked = list(enumerate(diagnostic.get('shared_output_gaps', {}).get('top10', []), 1))
    ranked.sort(key=lambda pair: pair[1]['begin_s'])
    generations = {row['generation']: row for row in diagnostic.get('per_generation', [])}
    gen2 = sorted(generations.get(2, {}).get('retained_events', []), key=lambda row: row['original_begin_s'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11.8, 9), gridspec_kw={'height_ratios': (2.1, 1)})
    labels = []; longest = max((gap['duration_s']*1000 for _, gap in ranked), default=1)
    for y, (rank, gap) in enumerate(ranked):
        begin, end = gap['begin_s'], gap['end_s']; duration_ms = gap['duration_s']*1000
        labels.append(f"#{rank:02d}  t={begin:.6f} s  n={gap['request_count']}")
        top.barh(y, duration_ms, left=0, height=.56, color='#E3E8ED', edgecolor='#A9B3BC', linewidth=.6)
        for generation, observed in generations.items():
            for lo, hi in observed.get('retained_union_intervals_s', []):
                left, right = max(begin, lo), min(end, hi)
                if right > left:
                    top.barh(y, (right-left)*1000, left=(left-begin)*1000,
                             height=.42, color=COLORS[generation], linewidth=0)
        top.text(duration_ms+longest*.012, y, f'{duration_ms:.1f} ms', va='center', fontsize=8)
    top.set(yticks=range(len(labels)), yticklabels=labels,
        ylim=(len(labels)-.4, -.6), xlim=(0, longest*1.17),
        xlabel='Host time relative to each output-gap start (ms)',
        title='(a) Ten longest shared output gaps, ordered by their measured start time')
    if not ranked:
        top.text(.5, .5, 'No shared output-gap observations', transform=top.transAxes, ha='center')
    top.tick_params(axis='y', labelsize=8)
    handles = [Patch(facecolor='#E3E8ED', edgecolor='#A9B3BC', label='Shared client-output interval')]
    handles += [Patch(facecolor=COLORS[generation], label=f'Retained GC overlap: gen {generation}') for generation in (0, 1, 2)]
    top.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -.19), ncol=2,
               fontsize=8, frameon=False, borderaxespad=0)

    if gen2:
        times = [event['original_begin_s'] for event in gen2]
        durations = [event['original_duration_s']*1000 for event in gen2]
        bottom.scatter(times, durations, s=36, color=COLORS[2], zorder=3)
        for event, time, duration in zip(gen2, times, durations):
            if event.get('clipped'):
                bottom.scatter([time], [duration], marker='s', s=65, facecolors='none', edgecolors='#222222', zorder=4)
        peak = max(range(len(gen2)), key=lambda index: durations[index])
        bottom.annotate(f'{durations[peak]:.1f} ms', (times[peak], durations[peak]),
                        xytext=(-8, 8), textcoords='offset points', ha='right', fontsize=8)
        bottom.set_ylim(0, max(durations)*1.22)
    else:
        bottom.text(.5, .5, 'No retained generation-2 callback observations', transform=bottom.transAxes, ha='center')
    bottom.set(xlim=(0, diagnostic.get('measurement', {}).get('duration_s', cell['duration_s'])),
        xlabel='GC callback start since measurement origin (s)', ylabel='Callback wall interval (ms)',
        title=f'(b) Generation-2 callback intervals during measurement (n={len(gen2)})')
    for axis in (top, bottom):
        axis.grid(axis='x' if axis is top else 'both', alpha=.2)
        axis.set_axisbelow(True); axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('One-run GC timing diagnostic', x=.055, ha='left', y=.975, fontsize=16, weight='bold')
    fig.text(.055, .94, args.metrics.parent.name+
             f"  |  once; {cell['completed']}/{cell['planned']} completed; {cell['outputs']:,} output tokens", fontsize=9)
    fig.text(.055, .061,
        'Top-panel labels: duration rank, measured gap start t, and number n of requests sharing that exact interval.\n'
        'Colored segments are intersections with retained GC callback intervals; short gen-0/1 events were compressed and are not all shown.\n'
        'Callback start–stop spans measure host wall time, not CPU consumption, exact stop-the-world pauses, or GPU execution.\n'
        'This single run shows temporal association only: no control, speedup claim, or attribution of stalls from earlier runs.',
        fontsize=7.6, linespacing=1.35)
    sha = hashlib.sha256(payload).hexdigest()
    fig.text(.985, .015, 'Input SHA256 '+sha, ha='right', fontsize=6.3, color='#59646C')
    fig.subplots_adjust(left=.255, right=.97, top=.875, bottom=.185, hspace=.65)
    with args.output.open('xb') as stream:
        fig.savefig(stream, format=suffix, dpi=220, facecolor='white')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), metrics_sha256=sha,
                          plotted_shared_gaps=len(ranked), generation2_events=len(gen2))))


if __name__ == '__main__':
    main()
