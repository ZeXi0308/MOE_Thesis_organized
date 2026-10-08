"""Full-request distributions for the frozen new-input three-arm episode."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
result = json.loads((root / 'A_SPARE_FOLLOWUP_FRESH_RESULT_R01_20261001.json').read_text())
assert result['status'] == 'COMPLETE_FRESH_TRIPLET'
arms = [
    ('Q1 reference', result['arms']['off']['metrics'], '#276e9b'),
    ('Q1 + one follow-up', result['arms']['on']['metrics'], '#cf7c28'),
    ('Native full-save', result['native_reference']['metrics'], '#598950'),
]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
for label, metrics, color in arms:
    assert metrics['completed'] == 128
    for ax, key in zip(axes, ['max_gap_s', 'flow_s', 'ttft_s']):
        values = sorted(row[key] for row in metrics['requests'])
        ax.step(values, [(j + 1) / len(values) for j in range(len(values))],
                where='post', label=label, color=color)
for ax, xlabel in zip(axes, ['Max generation gap per request (s)',
                            'Arrival-to-completion time (s)', 'Time to first token (s)']):
    ax.set(xlabel=xlabel, ylim=(0, 1.02))
    ax.grid(alpha=.18)
axes[0].set_ylabel('Cumulative request fraction')
axes[0].legend(frameon=False, fontsize=8, loc='lower right')
fig.suptitle('Frozen new inputs: 128 complete requests per arm', fontsize=11)
for ext in ('png', 'pdf'):
    path = root / f'A_SPARE_FOLLOWUP_FRESH_FIGURE_R01_20261001.{ext}'
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=180)
plt.close(fig)
