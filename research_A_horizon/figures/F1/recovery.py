"""Current Instruct recovery intervals; policy-selected episodes, not repetitions."""
import hashlib
import json
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/w2-48-calibration-policy-r03/metrics.json'
PHASES = [('preempt_to_allocation_s', 'Preempt → allocation'),
          ('load_submit_to_load_host_complete_s', 'LOAD → host completion'),
          ('allocation_to_first_output_s', 'Allocation → first output')]


def main():
    rows = json.loads(SOURCE.read_text())
    assert len(rows) == 2 and all(r['complete'] for r in rows)
    result = dict(source=str(SOURCE.relative_to(ROOT)),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), arms=[],
                  semantics='Policy-selected recovery episodes within one run per arm; '
                  'not independent system repetitions or a matched causal comparison. '
                  'Intervals overlap and cannot be summed. Host completion is not device completion. '
                  'Missing ACK and reentry observations remain missing.')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained', sharex=True, sharey=True)
    for ax, row, label, color in zip(axes, rows, ['Native', 'HORIZON 64/32'], ['#2876a8', '#d65335']):
        episodes = row['recovery_chains']['episodes']
        arm = dict(label=label, cell=Path(row['cell']).name, episodes=len(episodes), phases={})
        for i, (key, title) in enumerate(PHASES):
            samples = [dict(request=e['request'], episode=e['index'], seconds=e['durations'][key])
                       for e in episodes if key in e['durations']]
            values = [s['seconds'] for s in samples]
            assert all(v > 0 for v in values), 'Log plot requires positive observed intervals'
            arm['phases'][key] = dict(samples=samples, missing=len(episodes)-len(samples),
                                      median_s=median(values) if values else None)
            # Deterministic offsets expose overlapping episode observations.
            y = [i + .18 * (2 * j / max(1, len(values)-1) - 1) for j in range(len(values))]
            ax.scatter(values, y, s=18, color=color, alpha=.65)
            if values:
                ax.plot([median(values)]*2, [i-.24, i+.24], color='black', linewidth=2)
        result['arms'].append(arm)
        ax.set_title(f'{label}: {len(episodes)} recovery episodes')
        ax.set_xscale('log')
        ax.set_xlabel('Observed host interval (s, log scale)')
        ax.set_yticks(range(3), [p[1] for p in PHASES])
        ax.set_ylim(-.5, 2.5)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('OLMoE-Instruct · W2 · 48 GiB · λ=20\nDots: episodes; black ticks: medians; intervals overlap', fontsize=11)
    (OUT/'recovery-data.json').write_text(json.dumps(result, indent=2)+'\n')
    fig.savefig(OUT/'recovery.pdf')
    fig.savefig(OUT/'recovery.png', dpi=170)
    plt.close(fig)
    print(json.dumps([{k:v for k,v in arm.items() if k!='phases'} for arm in result['arms']]))


if __name__ == '__main__':
    main()
