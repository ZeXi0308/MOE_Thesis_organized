"""Render the six corrected frontiers for one model (F2 or Qwen F6).

All six ten-arm panels must be measured and use the requested code snapshot.
Incomplete matrices produce a JSON status only, never an interpolated figure.
"""
import argparse
import json
from pathlib import Path

from plot_corrected import read_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', type=Path, nargs=6, required=True)
    parser.add_argument('--code-snapshot', default='code-r12')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output basename for JSON/PDF/PNG')
    args = parser.parse_args()
    panels = [read_panel(p, args.code_snapshot) for p in args.plans]
    keys = [(p['kv_gib'], p['workload']) for p in panels]
    expected = {(b, w) for b in (24, 48) for w in ('W1', 'W2', 'W3')}
    if len(set(keys)) != 6 or set(keys) != expected:
        raise ValueError('Exactly two KV budgets by three workloads required')
    if len({p['model_revision'] for p in panels}) != 1:
        raise ValueError('Do not mix models or revisions in a six-panel figure')
    if len({p['model_path'] for p in panels}) != 1:
        raise ValueError('Model paths differ')
    for budget in (24, 48):
        if len({p['arrival_rate'] for p in panels if p['kv_gib'] == budget}) != 1:
            raise ValueError('W1/W3 must transfer the W2 calibration rate')
    by_key = dict(zip(keys, panels))
    ordered = [by_key[b, w] for b in (48, 24) for w in ('W1', 'W2', 'W3')]
    missing = {p['session']: p['missing'] for p in ordered if p['missing']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        status='INCOMPLETE' if missing else 'COMPLETE',
        code_snapshot=args.code_snapshot, panels=ordered, missing=missing), indent=2)+'\n')
    if missing:
        print(json.dumps(dict(status='INCOMPLETE',missing=missing)))
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import math
    colors = {'native':'#333333', 'cap':'#2876a8', 'full-budget':'#8c5c29',
              'mc-load':'#7952a2', 'mc':'#7952a2', 'horizon':'#d65335'}
    points = [point for panel in ordered for point in panel['points']]
    if not all(math.isfinite(p['ttft_p95_s']) and p['ttft_p95_s'] >= 0
               and math.isfinite(p['gap_p99_s']) and p['gap_p99_s'] > 0 for p in points):
        raise ValueError('Finite nonnegative TTFT and positive maxgap required')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharey=True,
                             layout='constrained')
    for ax, panel in zip(axes.flat, ordered):
        markers = ['o', 's', '^', 'v', 'D', 'P', 'o', '^', 's', 'v']
        for index, point in enumerate(panel['points']):
            cfg = point['config']
            family = 'cap' if point['label'].startswith('cap') else cfg['policy']
            # Caps and horizons retain their actual values in each legend.
            ax.scatter(point['ttft_p95_s'], point['gap_p99_s'],
                       color=colors[family], marker=markers[index],
                       s=34, label=point['label'], zorder=3)
        curve = panel['points'][6:]
        ax.plot([p['ttft_p95_s'] for p in curve], [p['gap_p99_s'] for p in curve],
                color=colors['horizon'], linewidth=1, zorder=2)
        ax.set(yscale='log', xlabel='TTFT P95 (s)',
               title=f"{panel['workload']} · {panel['kv_gib']} GiB · λ={panel['arrival_rate']}")
        ax.grid(which='both', alpha=.2)
        ax.legend(fontsize=7, ncol=2, loc='best')
    for ax in axes[:, 0]:
        ax.set_ylabel('Request maxgap P99 (s)')
    model = 'Qwen3-8B' if 'Qwen3-8B' in ordered[0]['model_path'] else 'OLMoE-Instruct'
    fig.suptitle(f'{model}: corrected-ledger frontiers ({args.code_snapshot})\n'
                 'One development run per arm; natural generation; shared logarithmic y-axis',
                 fontsize=12)
    fig.savefig(args.output.with_suffix('.pdf'))
    fig.savefig(args.output.with_suffix('.png'), dpi=180)
    plt.close(fig)
    print(json.dumps(dict(status='RENDERED', panels=6, points=len(points))))


if __name__ == '__main__':
    main()
