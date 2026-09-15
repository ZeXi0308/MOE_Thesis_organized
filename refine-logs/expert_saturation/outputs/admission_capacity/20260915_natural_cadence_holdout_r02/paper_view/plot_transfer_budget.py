"""Derived view of sole-executor analyses; no raw recount or new samples."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/moe-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
groups = {
    'G': here.parent.parent / '20260915_natural_recovery_cadence_r01',
    'H': here.parent,
}
points = []
for group, folder in groups.items():
    data = json.loads((folder / 'execution_weste_26862/analysis.json').read_text())
    for pair in data['performance_comparisons']:
        assert pair['status'] == 'COMPLETE', 'Do not rank incomplete comparisons'
        metrics = pair['eager_relative_percent']
        points.append((group, pair['block'], metrics['output_token_rate_s'],
                       metrics['mean_completion_s'], metrics['max_engine_return_gap_s']))

fig, ax = plt.subplots(figsize=(7.5, 4.8), layout='constrained')
xlo, xhi = min(-4, min(p[2] for p in points) - 1.5), max(2, max(p[2] for p in points) + 1.5)
ylo, yhi = min(-2, min(p[3] for p in points) - 2), max(6, max(p[3] for p in points) + 2)
ax.fill_between([-3, xhi], ylo, 5, color='#d6eddf', alpha=.7,
                label='H cost budget (declared before H)')
ax.axvline(-3, color='#55846a', ls='--', lw=1)
ax.axhline(5, color='#55846a', ls='--', lw=1)
ax.axvline(0, color='#b8bdc4', lw=.7)
ax.axhline(0, color='#b8bdc4', lw=.7)
for group, block, rate, completion, gap in points:
    ax.scatter(rate, completion, s=65, marker=('o', '^')[block],
               color={'G': '#747e89', 'H': '#216cb0'}[group], zorder=3)
    offset = {('G', 0): (8, -17), ('G', 1): (8, 12),
              ('H', 0): (-10, 10), ('H', 1): (8, 10)}[group, block]
    ax.annotate(f'{group}{block}: max gap {gap:+.1f}%', (rate, completion),
                xytext=offset, textcoords='offset points', fontsize=9,
                ha='right' if (group, block) == ('H', 0) else 'left')
ax.set(xlim=(xlo, xhi), ylim=(ylo, yhi),
       xlabel='Actual output throughput change (%) — higher is better',
       ylabel='Mean completion time change (%) — lower is better',
       title='Eager vs current: service costs and maximum generation gap')
ax.grid(alpha=.15)
ax.legend(loc='upper right', frameon=False, fontsize=9)
fig.supxlabel('G = selection data; H = new-document, longer finite-arrival validation.\n'
              'Both H pairs must meet the cost budget and lower maximum gap. No confidence intervals or business SLO.\n'
              'Same inputs within each pair; generated output sequences may differ.',
              fontsize=8)
for suffix in ('pdf', 'svg', 'png'):
    fig.savefig(here / f'transfer_budget.{suffix}', dpi=180)
print(here / 'transfer_budget.png')
