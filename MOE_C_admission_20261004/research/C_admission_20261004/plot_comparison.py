#!/usr/bin/env python3
"""Plot six real admission cells from analyze.py; never overwrite artifacts."""
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
from matplotlib.lines import Line2D


POLICIES = (
    ('fixed', 'Fixed cap', '#0072B2', '-'),
    ('kv', 'KV water level', '#009E73', (0, (5, 2.5))),
    ('recovery', 'Recovery aware', '#D55E00', (0, (3, 1.5, 1, 1.5))),
)
BLOCKS = (('test-00-fixed', 'test-01-kv', 'test-02-recovery'),
          ('test-05-fixed', 'test-04-kv', 'test-03-recovery'))
FIELDS = (('ttft_s', 'External arrival to first token', 'TTFT (s)'),
          ('flow_s', 'External arrival to completion', 'Completion time / flow (s)'),
          ('max_generation_gap_s', 'Maximum generation gap', 'Max generation gap (s)'))


def observed_cdf(cell, field):
    """Use the supplied full-population CDF without filling missing outcomes."""
    n = cell['planned_requests']
    if not isinstance(n, int) or n <= 0:
        raise ValueError('Each arm must have a positive planned-request denominator.')
    distribution = cell.get('distributions', {}).get(field)
    if distribution is None:
        return [], f'not reported; n={n}'
    if distribution['denominator'] != n:
        raise ValueError(f'{cell["cell"]}: {field} denominator differs from planned_requests')
    observed = distribution['observed_n']
    points = distribution['full_population_cdf']
    if not isinstance(observed, int) or not 0 <= observed <= n:
        raise ValueError(f'{field}: invalid observed count')
    previous_x, previous_y = -1., 0.
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y) and x >= 0 and x > previous_x
                and previous_y < y <= 1):
            raise ValueError(f'{field}: invalid full-population CDF')
        previous_x, previous_y = x, y
    if bool(points) != bool(observed) or not math.isclose(previous_y, observed/n, abs_tol=1e-10):
        raise ValueError(f'{field}: CDF endpoint does not match observed/planned')
    note = f'observed {observed}/{n}' if observed else f'no observations; n={n}'
    return points, note


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True, help='Real six-cell analyze.py JSON.')
    parser.add_argument('--output', type=Path, required=True, help='New SVG path.')
    parser.add_argument('--png', action='store_true', help='Also save a matching 180 dpi PNG.')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.svg':
        parser.error('--output must name an SVG file.')
    outputs = [args.output] + ([args.output.with_suffix('.png')] if args.png else [])
    if any(path.exists() for path in outputs):
        parser.error('Output already exists; choose a new --output path.')
    source = json.loads(args.analysis.read_text())
    cells = {}
    for cell in source['cells']:
        name = Path(cell['cell']).name
        if name in cells:
            raise ValueError(f'Duplicate cell name: {name}')
        cells[name] = cell
    missing = [name for block in BLOCKS for name in block if name not in cells]
    if missing:
        parser.error('Missing comparison cells: '+', '.join(missing))
    selected = [cells[name] for block in BLOCKS for name in block]
    if len({cell['workload_identity_sha256'] for cell in selected}) != 1:
        raise ValueError('The six cells must share the same workload and external arrival trace.')
    cdfs = {(name, field): observed_cdf(cells[name], field)
            for block in BLOCKS for name in block for field, _, _ in FIELDS}
    xmax = {field: max([point[0] for block in BLOCKS for name in block
            for point in cdfs[name, field][0]], default=0.) for field, _, _ in FIELDS}
    xmax = {field: value*1.05 if value > 0 else 1. for field, value in xmax.items()}

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 10, 'axes.titlesize': 11, 'axes.linewidth': .8,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 10,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.2), sharex='col', sharey=True)
    fig.subplots_adjust(left=.075, right=.982, top=.842, bottom=.305, wspace=.16, hspace=.38)
    fig.suptitle('Admission policies: full-population latency CDFs',
                 y=.974, fontsize=15, fontweight='semibold')
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=2, label=label)
               for _, label, color, style in POLICIES]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .940),
               ncol=3, frameon=False, columnspacing=2.6, handlelength=3.2)
    for row, block in enumerate(BLOCKS):
        for col, (field, title, xlabel) in enumerate(FIELDS):
            ax = axes[row, col]
            for policy_index, (name, (_, label, color, style)) in enumerate(zip(block, POLICIES)):
                points, note = cdfs[name, field]
                if points:
                    # Extend only the observed plateau: never append a fabricated 1.
                    xs = [0.] + [p[0] for p in points] + [xmax[field]]
                    ys = [0.] + [p[1] for p in points] + [points[-1][1]]
                    ax.step(xs, ys, where='post', color=color, linestyle=style, linewidth=1.9)
                ax.text(.97, .065+.072*(2-policy_index), f'{label}: {note}', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=7.6, color=color,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.82, pad=1.1))
            ax.set_title(f'{"abcdef"[row*3+col]}   {title}', loc='left', pad=10)
            ax.set_xlim(0, xmax[field])
            ax.set_ylim(0, 1.025)
            ax.set_xlabel(xlabel, labelpad=6)
            ax.set_yticks([0, .2, .4, .6, .8, 1])
            ax.tick_params(axis='x', labelbottom=True)
            ax.grid(color='#E5E7EB', linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        axes[row, 0].set_ylabel(('Forward block 0' if row == 0 else 'Reverse block 1')+
                                '\nFraction of all planned arrivals', labelpad=10)
    n_lines = ['Block '+str(row)+': '+ ';  '.join(
        f'{name}: n={cells[name]["planned_requests"]}' for name in block)
        for row, block in enumerate(BLOCKS)]
    caption = n_lines + [
        'Two run-level repeats per policy: execution 00 / 01 / 02, then 03 / 04 / 05. No request-level confidence intervals.',
        'Natural EOS output amounts may vary across arms; these curves do not establish equal-work speedups.',
        'Every planned request remains in the denominator. Missing or unobserved metrics never raise the CDF to 1.',
        'Max generation gap uses completed requests and host-return token timestamps; absent observations are labelled.',
        'Descriptive exploratory comparison; no independent-test-set claim.',
    ]
    for y, line in zip((.239, .214, .174, .147, .120, .093, .055), caption):
        fig.text(.075, y, line, fontsize=9.1, color='#555555', va='top')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        fig.savefig(handle, format='svg', metadata={'Date': None,
            'Description': 'Six real full-population CDFs per metric in two run-level order blocks. '
                'Missing outcomes are not renormalized; natural output lengths may differ. '
                'No request-level confidence intervals or independent-test-set claim.'})
    if args.png:
        with args.output.with_suffix('.png').open('xb') as handle:
            fig.savefig(handle, format='png', dpi=180)
    plt.close(fig)
    for path in outputs:
        print(path.resolve())


if __name__ == '__main__':
    main()
