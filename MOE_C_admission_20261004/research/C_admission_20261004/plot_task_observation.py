#!/usr/bin/env python3
"""Plot one real baseline from analyze.py and a paragraph or native async summary."""
import argparse
import json
from pathlib import Path

from plot_comparison import observed_cdf, plt


FIELDS = (('output_tokens', 'Generated workload', 'Output tokens (including stop tokens)'),
          ('ttft_s', 'External arrival to first token', 'Token TTFT (s)'),
          ('flow_s', 'External arrival to completion', 'Completion time / flow (s)'),
          ('max_generation_gap_s', 'Maximum generation gap', 'Maximum generation gap (s)'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True, help='Single-cell analyze.py JSON.')
    summaries = parser.add_mutually_exclusive_group(required=True)
    summaries.add_argument('--paragraph', type=Path, help='Matching paragraph_summary.py JSON.')
    summaries.add_argument('--async-summary', type=Path, help='Matching passive async_summary.py JSON.')
    parser.add_argument('--output', type=Path, required=True, help='New SVG path; existing files are refused.')
    parser.add_argument('--png', action='store_true', help='Also write a matching 180 dpi PNG.')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.svg':
        parser.error('--output must name an SVG file.')
    outputs = [args.output] + ([args.output.with_suffix('.png')] if args.png else [])
    if any(path.exists() for path in outputs):
        parser.error('Output already exists; choose a new path.')
    analysis = json.loads(args.analysis.read_text())
    summary = json.loads((args.paragraph or args.async_summary).read_text())
    if len(analysis['cells']) != 1:
        parser.error('Exactly one analyzed cell is required; this is not a policy comparison.')
    cell = analysis['cells'][0]
    outcomes = cell['outcomes']
    if args.paragraph:
        if (cell['raw_sha256'] != summary['sources_sha256']['raw'] or
                cell['planned_requests'] != summary['planned_requests']):
            parser.error('The two summaries must describe the same raw requests and denominator.')
        if any(outcomes.get(k, 0) != summary['outcomes'].get(k, 0)
               for k in outcomes.keys() | summary['outcomes'].keys()):
            parser.error('Outcome counts disagree between summaries.')
        plotted = dict(cell, distributions=dict(cell['distributions'], output_tokens=summary['output_tokens']))
    else:
        if (summary.get('evidence') != 'SINGLE_NATIVE_ASYNC_DEPLOYMENT_OBSERVATION_NOT_MC_OR_POLICY_COMPARISON'
                or Path(summary['cell']).resolve() != Path(cell['cell']).resolve()
                or cell['admission']['configuration']['mode'] != 'passive_native_async'
                or summary.get('new_policy_requested_actions') != 0
                or summary.get('new_policy_executed_actions') != 0):
            parser.error('Expected matching single native async observations without new policy actions.')
        plotted = cell
    cdfs = {field: observed_cdf(plotted, field) for field, _, _ in FIELDS}

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.3), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.84, bottom=.28, wspace=.18, hspace=.43)
    fig.suptitle(('Task baseline' if args.paragraph else 'Native async')+
                 ': workload and full-service distributions', y=.977,
                 fontsize=15, fontweight='semibold')
    if args.paragraph:
        config = summary['configuration']
        description = (f'configured stop: {json.dumps(config["stop_strings"])}  |  '
                       f'EOS {"ignored" if config["ignore_eos"] else "enabled"}')
    else:
        description = 'single native async arm  |  new policy actions: 0 requested / 0 executed'
    fig.text(.535, .925, f'{Path(cell["cell"]).name}  |  status: {cell["run_status"]}  |  '+description,
             ha='center', fontsize=9.5, color='#555555')
    for index, (ax, (field, title, xlabel)) in enumerate(zip(axes.flat, FIELDS)):
        points, note = cdfs[field]
        log_axis = (field == 'max_generation_gap_s' and points and points[0][0] > 0
                    and points[-1][0] / points[0][0] > 100)
        xmin = points[0][0]*.8 if log_axis else 0.
        xmax = points[-1][0]*1.06 if points and points[-1][0] > 0 else 1.
        if points:
            ax.step([xmin]+[x for x, _ in points]+[xmax],
                    [0.]+[y for _, y in points]+[points[-1][1]], where='post',
                    color='#0072B2', linewidth=1.9)
        if log_axis:
            ax.set_xscale('log')
            xlabel += '; log scale'
        ax.set(title=f'{"abcd"[index]}   {title}', xlabel=xlabel, xlim=(xmin, xmax), ylim=(0, 1.025))
        ax.text(.97, .055, note, transform=ax.transAxes, ha='right', color='#0072B2', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.grid(color='#E5E7EB', linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        if index % 2 == 0:
            ax.set_ylabel('Fraction of all planned requests')
    outcome_text = '; '.join(f'{k}: {v}' for k, v in outcomes.items())
    captions = [f'All outcomes — {outcome_text}. Arrived: {cell["arrived_requests"]}/{cell["planned_requests"]}.']
    if args.paragraph:
        termination_text = '; '.join(f'{k.replace("_", " ")}: {v}' for k, v in summary['termination'].items())
        captions += [f'Termination — {termination_text}.',
            'TTFT / flow include waiting from external arrival. Token TTFT differs from visible text; gaps use host-return times.',
            'Output counts include stop tokens and observed unfinished output. Missing metrics retain the full denominator.',
            'One exploratory run; no cross-task speed comparison, task-quality assessment, or statistical-significance claim.']
    else:
        chunks = cell.get('host_chunk_diagnostics') or {}
        resolved = {True: 'resolved', False: 'unresolved'}.get(chunks.get('token_level_itl_resolved'), 'not reported')
        termination_text = '; '.join(f'{k}: {v}' for k, v in cell['stop_reasons'].items())
        captions += [f'Output: {cell["total_output_tokens"]} tokens (including observed unfinished output). '
            f'Natural stops: {cell["natural_stop_count"]}; length stops: {cell["length_stop_count"]}.',
            f'Finish reasons — {termination_text}. TTFT / flow start at external arrival; missing metrics keep all requests.',
            f'Host-return gaps: multi-token chunks {chunks.get("multi_token_chunks", "not reported")}; '
            f'maximum chunk {chunks.get("max_chunk_size", "not reported")}; intra-chunk timing {resolved}. No GPU-token timing claim.',
            'One native async arm; no MC or policy comparison, task-quality assessment, significance, or speedup claim.']
    for y, line in zip((.21, .173, .128, .09, .052), captions):
        fig.text(.09, y, line, fontsize=8.5, color='#555555', va='top')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        with path.open('xb') as stream:
            fig.savefig(stream, format=path.suffix[1:], dpi=180,
                        metadata={'Description': 'Single real '+('task baseline' if args.paragraph else 'native async observation')+
                                  '; full-population CDFs. No comparative or quality claim.'})
        print(path.resolve())
    plt.close(fig)


if __name__ == '__main__':
    main()
