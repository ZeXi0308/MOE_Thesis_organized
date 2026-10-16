"""Render the preselected ablation group only after complete measurements exist."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SESSION = 'w2-48-ablations-r10'
LABELS = ['Full 64/32', 'No recovery protection', 'No admission',
          'No stall ordering', 'Hr=16', 'Hr=64', 'EWMA Hr', 'MC LOAD']


def collect():
    directory = ROOT / 'runs' / SESSION
    planned = json.loads((ROOT / 'configs' / (SESSION + '.json')).read_text())
    path = directory / 'metrics.json'
    if not path.exists():
        return [], LABELS
    receipt_path = directory / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['plan'] == planned, 'Executed plan differs from the selected group'
    points = []
    for metric in json.loads(path.read_text()):
        cell = Path(metric['cell']).name
        index = int(cell.split('-')[1])
        assert 0 <= index < len(LABELS)
        inputs = json.loads((directory / cell / 'inputs.json').read_text())
        executed = next(c for c in receipt['cells'] if c['index'] == index)
        runner = next(arg for arg in executed['command'] if arg.endswith('/run_cell.py'))
        assert Path(runner).parent.name == 'code-r12', 'Ablations require the corrected LOAD ledger'
        points.append(dict(index=index, label=LABELS[index], cell=cell,
            code_snapshot=Path(runner).parent.name,
            config=planned['cells'][index],
            model_revision=inputs['config']['model_revision'],
            input_tokens_sha256=hashlib.sha256(json.dumps(
                inputs['workload']['actual_prompt_token_ids']).encode()).hexdigest(),
            metrics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            metrics={k: v for k, v in metric.items()
                     if k not in ('requests', 'inherited_B_metrics', 'thrash_episodes')}))
    assert len({p['index'] for p in points}) == len(points)
    assert len({p['input_tokens_sha256'] for p in points}) <= 1
    assert len({p['model_revision'] for p in points}) <= 1
    missing = [label for i, label in enumerate(LABELS)
               if not any(p['index'] == i and p['metrics']['complete'] for p in points)]
    return sorted(points, key=lambda p: p['index']), missing


def main():
    points, missing = collect()
    (OUT / 'data.json').write_text(json.dumps(dict(points=points, missing=missing,
        semantics='One run per setting; no confidence intervals or additive attribution. '
        'No recovery protection disables both the recovery gate (Hr=0) and lease victim selection. '
        'No admission retains recovery protection. MC LOAD is a separate component baseline. '
        'Ha=64 is the event-rich development setting; it is not the F3 selected Ha=256.'), indent=2) + '\n')
    if missing:
        print(json.dumps(dict(status='UNMEASURED_OR_INCOMPLETE', missing=missing)))
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    specs = [('maxgap_s', 'p99', 'Maxgap P99 (s)', 1),
             ('stall_rate', None, 'Stalled requests (%)', 100),
             ('ttft_s', 'p95', 'TTFT P95 (s)', 1),
             ('thrash_count', None, 'Thrash episodes', 1)]
    # Primary component ablations occupy the first four cells. Sensitivity
    # and MC measurements stay in data.json/T1 rather than mixing contrasts.
    for ax, (key, quantile, title, scale) in zip(axes.flat, specs):
        values = []
        for point in points[:4]:
            value = point['metrics'][key]
            values.append(scale * (value[quantile] if quantile else value))
        ax.bar(range(4), values, color=['#2876a8', '#d65335', '#d69e36', '#668f59'])
        ax.set_xticks(range(4), ['Full', '− recovery', '− admission', '− ordering'])
        ax.set_ylabel(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('OLMoE-Instruct · W2 · 48 GiB · Ha=64 / Hr=32\nSingle-run component ablations')
    fig.savefig(OUT / 'F4.pdf')
    fig.savefig(OUT / 'F4.png', dpi=170)
    plt.close(fig)
    print(json.dumps(dict(status='RENDERED', points=len(points))))


if __name__ == '__main__':
    main()
