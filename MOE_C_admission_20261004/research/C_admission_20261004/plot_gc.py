#!/usr/bin/env python3
"""Plot gc_summary.py host-gap/GC interval overlap without overwriting artifacts."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'c-admission-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


QUARTET = ('probe-00-retained', 'probe-01-released',
           'probe-02-released', 'probe-03-retained')
TOTAL_COLOR = '#C9CDD2'
GC_COLOR = '#007C91'


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def prepare(source):
    if source['schema_version'] != 1 or source['gap_threshold_s'] != .25:
        raise ValueError('Expected gc_summary schema 1 with a fixed 250 ms threshold.')
    cells = source['cells']
    if not cells:
        raise ValueError('No cells to plot.')
    names = [Path(cell['cell']).name for cell in cells]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate cell labels.')
    quartet = set(names) == set(QUARTET)
    if quartet:
        cells = [cells[names.index(name)] for name in QUARTET]
    for cell in cells:
        config = cell.get('admission_configuration') or {}
        if quartet and (config.get('mode') not in ('fixed', 'kv') or
                        config.get('cap') != 192 or config.get('kv_floor') != 0 or
                        config.get('probe_enabled') is not False):
            raise ValueError('Retained/released labels require cap192 / KV floor0 / probe_enabled=False; '
                             'the shared Gate may report kv mode with a zero water-level floor.')
        available = cell['gc_observation']['available']
        end = cell['service']['observation_end_s']
        if type(available) is not bool or not finite(end) or end <= 0:
            raise ValueError('Invalid GC availability or observation horizon.')
        gaps = cell['public_gaps_over_250ms']
        if cell['public_gap_count'] != len(gaps):
            raise ValueError('Public-gap count does not match rows.')
        previous_end = -1
        for gap in gaps:
            start, stop, duration = (gap[key] for key in ('start_s', 'end_s', 'duration_s'))
            if not all(finite(value) for value in (start, stop, duration)):
                raise ValueError('Nonfinite gap geometry.')
            if (start < previous_end or start < 0 or stop > end+1e-6 or duration <= .25
                    or not math.isclose(stop-start, duration, abs_tol=1e-9)):
                raise ValueError('Invalid or overlapping public-gap geometry.')
            overlap = gap['gc_overlap']
            if overlap['available'] != available:
                raise ValueError('Per-gap GC availability conflicts with the cell.')
            if available and (not finite(overlap['union_s']) or
                              not 0 <= overlap['union_s'] <= duration+1e-6):
                raise ValueError('GC union intersection exceeds the host gap.')
            if not available and overlap.get('union_s') is not None:
                raise ValueError('Unavailable GC must not be filled with a zero duration.')
            previous_end = stop
    return cells, quartet


def render(source, output, png=False):
    targets = [output] + ([output.with_suffix('.png')] if png else [])
    if output.suffix.lower() != '.svg' or any(path.exists() for path in targets):
        raise ValueError('Choose a new SVG path; existing SVG/PNG outputs are never overwritten.')
    cells, quartet = prepare(source)
    columns = 2 if len(cells) <= 4 else 3
    rows = math.ceil(len(cells)/columns)
    horizon = max(cell['service']['observation_end_s'] for cell in cells)
    maximum_ms = max([1000*gap['duration_s'] for cell in cells
                      for gap in cell['public_gaps_over_250ms']] + [250.])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 9, 'axes.titlesize': 10, 'axes.linewidth': .8,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 10,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(rows, columns, figsize=(11.6, 3.15*rows+2.0),
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=.085, right=.985, top=.81, bottom=.25,
                        wspace=.18, hspace=.73)
    fig.suptitle('Public host output gaps and GC interval overlap', y=.975,
                 fontsize=15, fontweight='semibold')
    fig.text(.5, .931, 'Retained / released / released / retained history; fixed cap192, KV floor0'
             if quartet else 'One panel per recorded run; configurations shown in panel titles',
             ha='center', fontsize=10, color='#555555')
    fig.legend(handles=[Patch(facecolor=TOTAL_COLOR, edgecolor='#70757D', label='Total host gap'),
                        Patch(facecolor=GC_COLOR, label='GC union overlap within that gap')],
               loc='upper center', bbox_to_anchor=(.5, .899), ncol=2, frameon=False)
    width = max(horizon*.009, .2)
    for index, (cell, ax) in enumerate(zip(cells, axes.flat)):
        name = Path(cell['cell']).name
        config = cell.get('admission_configuration') or {}
        if quartet:
            title = f'{name.split("-")[1]}  '+('Retained history' if name.endswith('retained')
                                             else 'Released history')+' | cap192'
        else:
            title = name+(f' | cap{config["cap"]}' if config.get('cap') is not None else '')
        gaps = cell['public_gaps_over_250ms']
        available = cell['gc_observation']['available']
        x = [gap['start_s'] for gap in gaps]
        ax.bar(x, [1000*gap['duration_s'] for gap in gaps], width=width,
               color=TOTAL_COLOR, edgecolor='#70757D', linewidth=.6, zorder=2)
        if available:
            ax.bar(x, [1000*gap['gc_overlap']['union_s'] for gap in gaps], width=width*.62,
                   color=GC_COLOR, linewidth=0, zorder=3)
        annotation = f'{len(gaps)} '+('gap' if len(gaps) == 1 else 'gaps')+' >250 ms'
        if not available:
            annotation += '\nGC observation unavailable'
        elif cell['gc_observation'].get('unpaired_start_count', 0) or cell['gc_observation'].get('unpaired_stop_count', 0):
            annotation += '\nUnpaired GC events excluded'
        ax.text(.02, .95, annotation, transform=ax.transAxes, va='top', fontsize=8.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
        ax.axhline(250, color='#8B9098', linestyle=(0, (3, 3)), linewidth=.7, zorder=1)
        ax.set_title(f'{chr(97+index)}   {title}', loc='left', pad=9)
        ax.set_xlim(-horizon*.015, horizon*1.025)
        ax.set_ylim(0, maximum_ms*1.22)
        ax.set_xlabel('External-arrival clock: interval start (s)')
        if index % columns == 0:
            ax.set_ylabel('Interval duration (ms)')
        ax.tick_params(labelbottom=True)
        ax.grid(axis='y', color='#E6E8EB', linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        service = cell['service']
        completed, planned = service['outcomes']['completed'], service['planned_requests']
        ax.text(0, -.32, f'Completed: {completed}/{planned}'+
                (' (all)' if completed == planned else '')+
                f'    Output: {service["total_output_tokens"]:,} tokens',
                transform=ax.transAxes, va='top', fontsize=8.5, color='#444444')
    for ax in list(axes.flat)[len(cells):]:
        ax.set_visible(False)
    caption = [
        'Bars sit at interval starts; heights compare durations, not the location of GC within an interval.',
        'Only >250 ms gaps between consecutive engine calls with host token receipts are shown; axes are shared.',
        'External-arrival timing; natural EOS can change output amounts. Each panel is one run.',
        'Temporal overlap only: no time is deducted from full service; overlap is not removable latency or GPU/copy saving.']
    for y, text in zip((.135, .108, .081, .054), caption):
        fig.text(.085, y, text, va='top', fontsize=8.6, color='#555555')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        fig.savefig(stream, format='svg', metadata={'Date': None,
            'Description': 'Passive GC overlap with public host output intervals. Shared axes; '
            'missing GC is unavailable, not zero. No latency subtraction or removable-time claim.'})
    if png:
        with output.with_suffix('.png').open('xb') as stream:
            fig.savefig(stream, format='png', dpi=180)
    plt.close(fig)
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True, help='gc_summary.py JSON.')
    parser.add_argument('--output', type=Path, required=True, help='New SVG path.')
    parser.add_argument('--png', action='store_true')
    args = parser.parse_args()
    for path in render(json.loads(args.analysis.read_text()), args.output, args.png):
        print(path.resolve())


if __name__ == '__main__':
    main()
