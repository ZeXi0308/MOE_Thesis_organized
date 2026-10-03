"""Plot the existing primary analysis; no new samples or fitted thresholds."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/moe-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root.parent / 'execution_weste_26862/analysis.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
colors = {'selected': '#2563a6', 'native_full': '#c45b25'}
labels = {'selected': 'Selected saving', 'native_full': 'Full native saving'}
for block, ax in enumerate(axes[:2]):
    for scope in colors:
        c = data['cells'][f'block{block}-{scope}']
        assert c['comparable'] and c['requests']['request_count'] == 64
        gaps = sorted(r['max_engine_return_gap_s'] for r in c['requests']['requests'])
        ax.step([0, *gaps], [0, *[(i + 1) / len(gaps) for i in range(len(gaps))]],
                where='post', color=colors[scope], label=labels[scope], lw=1.8)
    ax.set(title=f'Pair {block}: all 64 requests', xlabel='Per-request maximum output gap (s)',
           xlim=(0, 4.05), ylim=(0, 1.02))
    ax.grid(alpha=.2)
axes[0].set_ylabel('Fraction of requests')
axes[0].legend(loc='lower right', frameon=False, fontsize=8)
ax = axes[2]
for block in (0, 1):
    points = []
    for scope in colors:
        r = data['cells'][f'block{block}-{scope}']['requests']
        point = (r['output_token_rate_s'], r['max_engine_return_gap_s'])
        points.append(point)
        ax.scatter(*point, color=colors[scope], marker=('o', '^')[block], s=55, zorder=3)
        tag = ('S' if scope == 'selected' else 'F') + str(block)
        offset = {'S0': (-69, -4), 'S1': (-5, -20), 'F0': (-75, 8)}.get(tag, (6, 6))
        ax.annotate(f'{tag}: {r["mean_completion_s"]:.2f}s mean', point,
                    xytext=offset, textcoords='offset points', fontsize=8)
    ax.annotate('', xy=points[1], xytext=points[0],
                arrowprops=dict(arrowstyle='->', color='#8b939e', lw=1, alpha=.8))
ax.set(title='Complete-service tradeoff', xlabel='Actual output throughput (tokens/s)',
       ylabel='Maximum output gap (s)', xlim=(1555, 1663), ylim=(2.05, 4.12))
ax.grid(alpha=.2)
fig.suptitle('Saving scope at fixed resources and current scheduling', fontsize=12)
fig.supxlabel('S/F = selected/full; circle/triangle = pair 0/1. '
         'Arrows: selected to full. Same inputs; generated outputs differ. No confidence intervals.',
         fontsize=8, color='#50565e')
for suffix in ('svg', 'pdf', 'png'):
    fig.savefig(root / f'saving_tradeoff.{suffix}', dpi=180)
print(root / 'saving_tradeoff.png')
