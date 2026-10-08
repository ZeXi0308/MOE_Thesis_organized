"""Keep the two completed blocks separate in full-request distributions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
for i, order in enumerate(('off then on', 'on then off')):
    result = json.loads((root / f'A_SPARE_FOLLOWUP_PAIR_RESULT_R0{i+1}_20261001.json').read_text())
    assert result['status'] == 'COMPLETE_PAIR'
    for arm, label, color in [('off', 'Q1 reference', '#276e9b'),
                              ('on', 'Q1 + one spare follow-up', '#cf7c28')]:
        rows = result['arms'][arm]['metrics']['requests']
        for ax, key in zip(axes[i], ['max_gap_s', 'flow_s']):
            values = sorted(row[key] for row in rows)
            ax.step(values, [(j+1)/len(values) for j in range(len(values))],
                    where='post', label=label, color=color)
    for ax in axes[i]:
        ax.set(ylabel='Cumulative request fraction', ylim=(0, 1.02))
        ax.grid(alpha=.18)
    axes[i, 0].set_title(f'Block {i+1}: {order}', fontsize=10)
    axes[i, 0].set_xlabel('Maximum generation gap per request (s)')
    axes[i, 1].set_xlabel('Arrival-to-completion time (s)')
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle('Seen H128: two ordered blocks, each 128/128 complete per arm', fontsize=11)
for ext in ('png', 'pdf'):
    path = root / f'A_SPARE_FOLLOWUP_TWO_PAIRS_FIGURE_R01_20261001.{ext}'
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=180)
plt.close(fig)
