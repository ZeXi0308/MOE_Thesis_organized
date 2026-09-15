"""Plot every eligible holdout cell; no smoothing, pooling, or fitted frontier."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
source = ROOT / 'analysis.json'
data = json.loads(source.read_text())
cells = data['cells']
assert data['status'] == 'MEASUREMENT_ONLY'
assert len(cells) == 8 and all(c['eligible'] for c in cells)
styles = {
    'native': ('Native', '#34495e'),
    'most_output': ('Most output', '#0072b2'),
    'fit_scan': ('Fit scan', '#e69f00'),
    'guard_residual': ('Guard + residual budget', '#cc79a7'),
}
plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.5), sharey=True)
for ax, field, title, xlabel in zip(
    axes,
    ['throughput_rps', 'mean_completion_s'],
    ['Frozen primary objectives', 'Completion-time cost'],
    ['Completed requests / second (higher is better)',
     'Mean request completion latency, s (lower is better)'],
):
    for cell in cells:
        label, color = styles[cell['variant']]
        block = 0 if '-block0-' in cell['label'] else 1
        ax.scatter(cell[field], cell['max_itl_s'], color=color,
                   marker=['o', '^'][block], s=65, edgecolor='white',
                   linewidth=0.7, zorder=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 16)
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Maximum request ITL, s (lower is better)')
handles = [Line2D([], [], marker='o', color=color, linestyle='none',
                  label=label) for label, color in styles.values()]
handles += [Line2D([], [], marker=marker, color='gray', linestyle='none',
                   label=f'Block {i}') for i, marker in enumerate(['o', '^'])]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.065))
fig.suptitle('OLMoE / native vLLM: same KV budget, observed request tradeoffs',
             fontsize=12)
fig.text(0.5, 0.025,
         'One 32-document cohort; 3072 input / 1024 output tokens; '
         '6656 usable KV blocks. All eight cells shown; no confidence intervals.',
         ha='center', fontsize=8)
fig.tight_layout(rect=(0, 0.2, 1, 0.94))
for suffix in ['svg', 'png']:
    target = ROOT / f'tradeoffs.{suffix}'
    assert not target.exists(), f'Refuse overwrite: {target}'
    fig.savefig(target, dpi=180)
plt.close(fig)
print(json.dumps({'cells': len(cells), 'source_sha256':
                  hashlib.sha256(source.read_bytes()).hexdigest()}))
