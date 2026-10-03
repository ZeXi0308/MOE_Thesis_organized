"""Plot every request in the completed native spare-followup pair."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
result = json.loads((root / 'A_SPARE_FOLLOWUP_PAIR_RESULT_R01_20261001.json').read_text())
assert result['status'] == 'COMPLETE_PAIR'
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout='constrained')
for arm, label, color in [('off', 'Q1 reference', '#276e9b'),
                          ('on', 'Q1 + one spare follow-up', '#cf7c28')]:
    rows = result['arms'][arm]['metrics']['requests']
    for ax, key in zip(axes, ['max_gap_s', 'flow_s']):
        values = sorted(row[key] for row in rows)
        ax.step(values, [(i + 1) / len(values) for i in range(len(values))],
                where='post', label=label, color=color)
for ax in axes:
    ax.set(ylabel='Cumulative request fraction', ylim=(0, 1.02))
    ax.grid(alpha=.18)
axes[0].set_xlabel('Maximum generation gap per request (s)')
axes[1].set_xlabel('Arrival-to-completion time (s)')
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Seen H128: one ordered native pair; all 256 requests completed', fontsize=11)
for ext in ('png', 'pdf'):
    path = root / f'A_SPARE_FOLLOWUP_PAIR_FIGURE_R01_20261001.{ext}'
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=180)
plt.close(fig)
