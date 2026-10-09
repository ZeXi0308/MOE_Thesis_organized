#!/usr/bin/env python3
"""Plot the exploratory async MC-component ABBA using full-population CDFs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze import distribution
from plot_async_simple import FIELDS, Line2D, observed_cdf, plt


ORDER = ('probe-00-declaredbudget', 'probe-01-asyncmc',
         'probe-02-asyncmc', 'probe-03-declaredbudget')
BLOCKS = ((ORDER[0], ORDER[1]), (ORDER[3], ORDER[2]))
POLICIES = (('Full declaration budget', 'Full budget', '#0072B2', '-'),
            ('MC component adapted to async', 'Async MC', '#D55E00', (0, (4, 2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True, help='Four-cell analyze.py JSON in ABBA order.')
    parser.add_argument('--output', type=Path, required=True, help='New SVG path.')
    parser.add_argument('--png', action='store_true', help='Also save a 180 dpi PNG.')
    args = parser.parse_args()
    outputs = [args.output] + ([args.output.with_suffix('.png')] if args.png else [])
    if args.output.suffix.lower() != '.svg' or any(path.exists() for path in outputs):
        parser.error('Choose a new SVG path, with a new PNG path if requested.')
    source = json.loads(args.analysis.read_text())['cells']
    if tuple(Path(cell['cell']).name for cell in source) != ORDER:
        parser.error('Require exactly declaredbudget / asyncmc / asyncmc / declaredbudget.')
    if len({cell['workload_identity_sha256'] for cell in source}) != 1:
        parser.error('Workload or external arrivals differ across arms.')
    cells = dict(zip(ORDER, source))
    for name, cell in cells.items():
        config = cell['admission']['configuration']
        expected = dict(mode='async_mc' if name.endswith('-asyncmc') else 'declared_budget',
                        cap=256, budget_blocks=32768, admission_count='complete_unique_unfinished')
        if any(config.get(key) != value for key, value in expected.items()):
            parser.error(f'{name}: configuration conflicts with the MC-component comparison labels.')
        requests_path = Path(cell['per_request_json'])
        if not requests_path.is_file():
            requests_path = args.analysis.parent/requests_path.name
        rows = json.loads(requests_path.read_text())
        if len(rows) != cell['planned_requests'] or any(
                Counter(r['outcome'] for r in rows)[key] != value for key, value in cell['outcomes'].items()):
            parser.error(f'{name}: per-request population or outcomes disagree with the analysis.')
        for field, _, _ in FIELDS:
            expected_cdf = distribution([r[field] for r in rows], len(rows))['full_population_cdf']
            if expected_cdf != cell['distributions'][field]['full_population_cdf']:
                parser.error(f'{name}: {field} CDF disagrees with its request records.')
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
    fig.suptitle('Exploratory ABBA: async MC component adaptation',
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
    totals = ';  '.join(f'{name.removeprefix("probe-")}: {cells[name]["total_output_tokens"]}' for name in ORDER)
    captions = [counts,
        'ABBA execution: 00 full budget, 01 async MC, 02 async MC, 03 full budget. Two run-level repeats per rule.',
        'Common complete-inflight cap 256 and physical budget 32768 pages. Existing MC component adaptation; not a new method.',
        'TTFT / completion include all waiting from external arrival. Missing outcomes retain the full population denominator.',
        'Natural-EOS output tokens — '+totals+'.',
        'Generation gaps use host-return timestamps; same-chunk timing is not reconstructed. No request-level confidence intervals.',
        'Exploratory comparison only; no independent confirmation, equal-output speedup, or task-quality claim.']
    for y, line in zip((.252, .216, .180, .144, .108, .072, .036), captions):
        fig.text(.09, y, line, fontsize=8.5, color='#555555', va='top')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        with path.open('xb') as stream:
            fig.savefig(stream, format=path.suffix[1:], dpi=180,
                        metadata={'Description': 'Exploratory async MC-component ABBA; full-population latency CDFs. '
                            'Natural EOS output differs. No independent-confirmation or equal-output claim.'})
        print(path.resolve())
    plt.close(fig)


if __name__ == '__main__':
    main()
