#!/usr/bin/env python3
"""Render the four comparable pro-dev latency CDFs; never overwrite an artifact."""
import argparse
import json
import os
from pathlib import Path
import tempfile

# Keep the Matplotlib cache inside the permitted temporary directory.
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'c-admission-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ARMS = (
    ('dev-native256', 'native256', '#202020', '-', 2.7),
    ('dev-cap96', 'fixed96', '#0072B2', (0, (5, 2.5)), 1.9),
    ('dev-cap128', 'fixed128', '#D55E00', '-', 2.5),
    ('dev-cap192', 'fixed192', '#009E73', (0, (3, 1.5, 1, 1.5)), 1.6),
)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=root/'analysis/pro-dev-r01.json')
    parser.add_argument('--output', type=Path, default=root/'analysis/pro-dev-cdf.svg')
    parser.add_argument('--png', action='store_true', help='Also save a matching 180 dpi PNG.')
    args = parser.parse_args()
    outputs = [args.output] + ([args.output.with_suffix('.png')] if args.png else [])
    if any(path.exists() for path in outputs):
        parser.error('Output already exists; choose a new --output path.')
    source = json.loads(args.analysis.read_text())
    cells = {Path(cell['cell']).name: cell for cell in source['cells']}
    selected = [cells[name] for name, *_ in ARMS]
    if len({c['workload_identity_sha256'] for c in selected}) != 1:
        raise ValueError('The selected arms do not share the same workload and external arrival trace.')
    if any(c['planned_requests'] != 192 or c['outcomes']['completed'] != 192 for c in selected):
        raise ValueError('This figure is for the four complete 192-request development runs only.')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 11, 'axes.titlesize': 12, 'axes.linewidth': .8,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 11,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.35), sharey=True)
    fig.subplots_adjust(left=.075, right=.982, top=.785, bottom=.285, wspace=.16)
    fig.suptitle('Admission concurrency: full-request latency distributions',
                 x=.53, y=.965, fontsize=14, fontweight='semibold')
    for ax, field, title, xmax in zip(axes, ('ttft_s', 'flow_s'),
            ('a   External arrival to first token', 'b   External arrival to completion'), (50, 80)):
        for name, label, color, linestyle, linewidth in ARMS:
            d = cells[name]['distributions'][field]
            if d['denominator'] != 192 or d['observed_n'] != 192:
                raise ValueError('CDF denominator mismatch')
            points = d['full_population_cdf']
            xs = [0] + [p[0] for p in points] + [xmax]
            ys = [0] + [p[1] for p in points] + [1]
            ax.step(xs, ys, where='post', label=label, color=color,
                    linestyle=linestyle, linewidth=linewidth, alpha=.95)
        ax.set_title(title, loc='left', pad=12)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.025)
        ax.set_xlabel('TTFT (s)' if field == 'ttft_s' else 'Completion time / flow (s)', labelpad=8)
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.grid(axis='both', color='#E5E7EB', linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(.95, color='#9CA3AF', linestyle=(0, (2, 3)), linewidth=.8, zorder=0)
        for name, color in (('dev-native256', '#202020'), ('dev-cap128', '#D55E00')):
            p95 = cells[name]['distributions'][field]['observed_only']['p95']
            ax.plot(p95, .95, marker='o', markersize=4, color=color, markeredgecolor='white', markeredgewidth=.5)
    axes[0].set_ylabel('Fraction of all 192 arrivals', labelpad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53, .913),
               ncol=4, frameon=False, columnspacing=2.0, handlelength=3.0)
    nat, cap = cells['dev-native256']['distributions'], cells['dev-cap128']['distributions']
    left = (f"TTFT p95: native {nat['ttft_s']['observed_only']['p95']:.2f} s"
            f"  →  fixed128 {cap['ttft_s']['observed_only']['p95']:.2f} s")
    right = (f"Mean flow: native {nat['flow_s']['observed_only']['mean']:.2f} s"
             f"  →  fixed128 {cap['flow_s']['observed_only']['mean']:.2f} s\n"
             f"Flow p95: native {nat['flow_s']['observed_only']['p95']:.2f} s"
             f"  →  fixed128 {cap['flow_s']['observed_only']['p95']:.2f} s")
    fig.text(.075, .178, left, fontsize=10.1, color='#333333', va='top')
    fig.text(.565, .178, right, fontsize=10.1, color='#333333', va='top', linespacing=1.6)
    fig.text(.075, .070,
        'n = 192 per policy; one run per policy; identical external arrivals; all requests completed.',
        fontsize=9.4, color='#555555')
    fig.text(.075, .033,
        'Natural EOS (maximum 1,024 tokens); output work differs. No confidence intervals or independent-request replication.',
        fontsize=9.4, color='#555555')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format='svg', metadata={'Date': None,
        'Description': 'Two full-population empirical latency CDFs for four single development runs. '
            'Natural output lengths differ; no statistical inference or equal-work speedup is claimed.'})
    if args.png:
        fig.savefig(args.output.with_suffix('.png'), dpi=180)
    plt.close(fig)
    for path in outputs:
        print(path.resolve())


if __name__ == '__main__':
    main()
