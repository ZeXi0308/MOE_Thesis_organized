#!/usr/bin/env python3
"""Plot the one native ABBA analysis artifact; no resummarization of raw data."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'moe-a-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def draw(data, output, candidate_label='Host-near'):
    targets = [output.with_suffix(ext) for ext in ('.pdf', '.png')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Refusing to overwrite an existing figure')
    cells = list(data['cells'].items())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    colors = ['#2266aa', '#b64b16', '#b64b16', '#2266aa']
    styles = ['-', '-', '--', '--']
    metrics = [('actual_completion_flow_s', 'External arrival to completion (s)'),
               ('ttft_s', 'External arrival to first output (s)'),
               ('max_gap_s', 'Per-request maximum generation gap (s)')]
    notes = []
    for index, (name, cell) in enumerate(cells):
        spec = cell.get('plan_cell', {})
        label = spec.get('label', name)
        m = cell.get('metrics', {})
        denominator = spec.get('requests', cell.get('expected_requests'))
        rows = m.get('requests', [])
        for axis, (key, title) in zip(axes[0], metrics):
            values = sorted(r[key] for r in rows if finite(r.get(key)))
            if values and denominator:
                # Denominator is all planned arrivals, including missing observations.
                axis.step([values[0], *values], [0, *[(j+1)/denominator for j in range(len(values))]],
                          where='post', label=label, color=colors[index % 4], linestyle=styles[index % 4])
            axis.set(xlabel=title, ylabel='Fraction of all planned requests', ylim=(0, 1.02))
            axis.grid(alpha=.2)
        preempt = cell.get('native_preemptions', {})
        observation = cell.get('native_victim_observation', {})
        notes.append(f"{label}: {cell.get('status', 'UNKNOWN')}\n"
                     f"  completed {m.get('completed', '?')}/{denominator}; "
                     f"failed {m.get('failed', '?')}; unfinished {m.get('unfinished', '?')}\n"
                     f"  output tokens {m.get('total_output_tokens', '?')}; "
                     f"preemptions {preempt.get('successful_calls', '?')}")
    axes[0, 0].legend(fontsize=8, loc='lower right')
    for axis, (label, comparison) in zip(axes[1, :2], data['comparisons'].items()):
        values = sorted(row['actual_completion_flow_difference_s']
                        for row in comparison.get('per_request', [])
                        if finite(row.get('actual_completion_flow_difference_s')))
        axis.axhline(0, color='black', linewidth=.8)
        axis.plot(range(1, len(values)+1), values, color='#773e92', linewidth=1.1)
        axis.set(xlabel='Matched request rank (sorted by difference)',
                 ylabel=f'{candidate_label} minus tail completion flow (s)',
                 title=f"{label}: {comparison.get('status', 'UNKNOWN')}\n"
                       f"{sum(x < 0 for x in values)} earlier / {sum(x > 0 for x in values)} later")
        axis.grid(alpha=.2)
    axes[1, 2].axis('off')
    axes[1, 2].text(0, 1, '\n\n'.join(notes), va='top', fontsize=8.5, family='monospace')
    fig.suptitle(f'Native victim selection: tail / {candidate_label} / {candidate_label} / tail', fontsize=14)
    fig.supxlabel('Exploratory run-level comparisons; independent trajectories, not same-state effects. '
                  'Natural outputs may differ. Missing observations are not imputed.', fontsize=9)
    for path in targets:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Figure basename; writes PDF and PNG')
    args = parser.parse_args()
    for path in draw(json.loads(args.analysis.read_text()), args.output):
        print(path)
