#!/usr/bin/env python3
"""Plot only the four matched async fixed177 / full-declaration-budget runs."""
import argparse
import json
from pathlib import Path

from plot_comparison import FIELDS, Line2D, observed_cdf, plt


ORDER = ('probe-00-fixed177', 'probe-01-declaredbudget',
         'probe-02-declaredbudget', 'probe-03-fixed177')
BLOCKS = ((ORDER[0], ORDER[1]), (ORDER[3], ORDER[2]))
POLICIES = (('Fixed complete cap 177', 'Fixed177', '#0072B2', '-'),
            ('Full declaration budget (32768 pages; cap 256)', 'Full budget', '#D55E00', (0, (4, 2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True, help='Four-cell analyze.py JSON in ABBA order.')
    parser.add_argument('--cell-root', type=Path, help='Optional relocated parent directory of the four raw cells.')
    parser.add_argument('--output', type=Path, required=True, help='New SVG path; no existing artifact is overwritten.')
    parser.add_argument('--png', action='store_true')
    args = parser.parse_args()
    outputs = [args.output] + ([args.output.with_suffix('.png')] if args.png else [])
    if args.output.suffix.lower() != '.svg' or any(path.exists() for path in outputs):
        parser.error('--output must be a new SVG path, with a new PNG path if requested.')
    source = json.loads(args.analysis.read_text())['cells']
    if tuple(Path(c['cell']).name for c in source) != ORDER:
        parser.error('Require exactly fixed177 / declaredbudget / declaredbudget / fixed177; no additional arm.')
    cells = dict(zip(ORDER, source))
    if len({c['workload_identity_sha256'] for c in source}) != 1:
        parser.error('All four arms must share the workload and external arrival trace.')
    for name, cell in cells.items():
        path = (args.cell_root/name if args.cell_root else Path(cell['cell']))/'config.json'
        config = json.loads(path.read_text())
        fixed = name.endswith('-fixed177')
        expected = dict(async_scheduling=True, policy_intervention=True, ignore_eos=False,
            admission_mode='fixed' if fixed else 'declared_budget', admission_cap=177 if fixed else 256,
            native_running_limit=256, engine_max_num_seqs=256, kv_floor=0,
            admission_count='complete_unique_unfinished', budget_blocks=None if fixed else 32768)
        if any(k not in config or config[k] != value for k, value in expected.items()):
            parser.error(f'{name}: raw configuration conflicts with the async simple-rule labels.')
    cdfs = {(name, field): observed_cdf(cells[name], field)
            for name in ORDER for field, _, _ in FIELDS}
    limits = {}
    for field, _, _ in FIELDS:
        values = [x for name in ORDER for x, _ in cdfs[name, field][0]]
        log_axis = field == 'max_generation_gap_s' and values and min(values) > 0
        limits[field] = (min(values)*.8 if log_axis else 0.,
                        max(values)*1.08 if values and max(values) > 0 else 1., bool(log_axis))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
        'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.7), sharex='col', sharey=True)
    fig.subplots_adjust(left=.09, right=.983, top=.825, bottom=.32, wspace=.18, hspace=.43)
    fig.suptitle('Async simple admission rules: full-population latency CDFs',
                 y=.975, fontsize=15, fontweight='semibold')
    fig.legend(handles=[Line2D([], [], label=label, color=color, linestyle=style, linewidth=2)
                        for label, _, color, style in POLICIES],
               loc='upper center', bbox_to_anchor=(.535, .939), ncol=2, frameon=False, handlelength=3)
    for row, block in enumerate(BLOCKS):
        for col, (field, title, xlabel) in enumerate(FIELDS):
            ax = axes[row, col]
            xmin, xmax, log_axis = limits[field]
            for index, (name, (_, short, color, style)) in enumerate(zip(block, POLICIES)):
                points, note = cdfs[name, field]
                if points:
                    ax.step([xmin]+[x for x, _ in points]+[xmax],
                            [0.]+[y for _, y in points]+[points[-1][1]], where='post',
                            color=color, linestyle=style, linewidth=1.8)
                ax.text(.97, .065+.085*(1-index), f'{short}: {note}', transform=ax.transAxes,
                        ha='right', fontsize=7.5, color=color,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=1))
            if log_axis:
                ax.set_xscale('log')
                xlabel += '; log scale'
            ax.set(title=f'{"abcdef"[row*3+col]}   {title}', xlabel=xlabel,
                   xlim=(xmin, xmax), ylim=(0, 1.025))
            ax.set_yticks([0, .2, .4, .6, .8, 1])
            ax.tick_params(axis='x', labelbottom=True)
            ax.grid(color='#E5E7EB', linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        axes[row, 0].set_ylabel(('AB pair: 00 / 01' if row == 0 else 'BA pair: 03 / 02')+
                                '\nFraction of all planned arrivals')
    counts = ';  '.join(f'{name.removeprefix("probe-")}: {cells[name]["outcomes"]["completed"]}/'
                        f'{cells[name]["planned_requests"]} completed' for name in ORDER)
    captions = [counts,
        'ABBA execution: 00 fixed177, 01 full budget, 02 full budget, 03 fixed177. Two run-level repeats per rule.',
        'Both use native async / compiled maximum 256. Complete-inflight caps differ: 177 versus 256; KV floor 0.',
        'Full budget counts physical use plus each live declaration\'s unallocated remainder and the new full declaration.',
        'TTFT / completion include all waiting from external arrival. Missing outcomes retain the full population denominator.',
        'Natural EOS; output amounts may differ. Generation gaps use host returns; same-chunk timing is not reconstructed.',
        'Exploratory matched runs only: no native-v23 or MC comparison, request-level CI, independent confirmation, or quality claim.']
    for y, line in zip((.252, .216, .180, .144, .108, .072, .036), captions):
        fig.text(.09, y, line, fontsize=8.5, color='#555555', va='top')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        with path.open('xb') as stream:
            fig.savefig(stream, format=path.suffix[1:], dpi=180,
                        metadata={'Description': 'Four matched native async simple-rule runs in ABBA order. '
                            'Full-population latency CDFs; natural EOS. No additional historical or MC arm.'})
        print(path.resolve())
    plt.close(fig)


if __name__ == '__main__':
    main()
