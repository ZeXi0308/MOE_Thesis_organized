"""Full-request CDFs for native full-save ordinary-only control."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
result = json.loads((root / 'A_NATIVE_BACKFILL_LAZY_PERF_PAIR_RESULT_R01_20261002.json').read_text())
assert result['status'] == 'COMPLETE_PAIR'
arms = [('Native full-save', 'native_full', '#276e9b'),
        ('Native full-save + optimized ordinary', 'ordinary', '#598950')]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
for label, key, color in arms:
    metrics = result['arms'][key]['metrics']
    assert metrics['completed'] == 128
    for ax, field in zip(axes, ['max_gap_s', 'flow_s', 'ttft_s']):
        values = sorted(row[field] for row in metrics['requests'])
        ax.step(values, [(i + 1) / len(values) for i in range(len(values))],
                where='post', label=label, color=color)
for ax, xlabel in zip(axes, ['Max generation gap per request (s)',
                            'Arrival-to-completion time (s)', 'Time to first token (s)']):
    ax.set(xlabel=xlabel, ylim=(0, 1.02))
    ax.grid(alpha=.18)
axes[0].set_ylabel('Cumulative request fraction')
axes[0].legend(frameon=False, fontsize=7, loc='lower right')
fig.suptitle('Untimed optimized ordinary/native pair: 128 complete seen requests per arm', fontsize=11)
for ext in ('png', 'pdf'):
    path = root / ('A_NATIVE_BACKFILL_LAZY_PERF_FIGURE_R01_20261002.' + ext)
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=180)
plt.close(fig)
