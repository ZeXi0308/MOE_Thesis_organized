"""Render fixed-setting transfer from an explicit W2 selection and four runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'figures' / 'F2'))
from plot_corrected import read_panel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(w1_path, w2_path, high_path, w3_path, snapshot):
    panels = [read_panel(p, snapshot) for p in (w1_path, w2_path, w3_path)]
    w1, w2, w3 = panels
    assert [p['workload'] for p in panels] == ['W1', 'W2', 'W3']
    for key in ('model_revision', 'model_path', 'kv_gib', 'arrival_rate'):
        assert len({p[key] for p in panels}) == 1, key
    high = json.loads(high_path.read_text())
    selection = high['provenance']['transfer_selection']
    assert selection['code_snapshot'] == snapshot
    assert selection['source_session'] == w2['session']
    assert selection['plan_sha256'] == w2['plan_sha256']
    for key in ('metrics_sha256', 'receipt_sha256'):
        if key in w2:
            assert selection[key] == w2[key], 'Selection source changed'
    chosen = selection['selected_points']
    assert len(chosen) == 2
    assert chosen[0]['config']['policy'] == 'native'
    assert chosen[1]['config']['policy'] == 'horizon'
    source_plan = json.loads(w2_path.read_text())
    for key in ('inputs', 'runtime_overlay', 'library_path', 'hf_home', 'per_cell_seconds'):
        assert high.get(key) == source_plan.get(key), key
    assert len(high['cells']) == 2
    for cfg, point in zip(high['cells'], chosen):
        expected = dict(point['config'], arrival_rate=2 * w2['arrival_rate'])
        assert cfg == expected, 'Transfer changed more than arrival rate'
    result = dict(code_snapshot=snapshot, model_path=w2['model_path'],
                  kv_gib=w2['kv_gib'], selection=selection, points=[], missing=[],
                  source_panels=panels, high_plan_sha256=sha(high_path))
    for panel in panels:
        if not panel['missing']:
            receipt = json.loads((ROOT / 'runs' / panel['session'] / 'receipt.json').read_text())
            assert receipt['status'] == 'COMPLETE'
        elif panel['points']:
            result['missing'].append([panel['workload'], 'incomplete frontier'])
    for scenario, panel in (('W1', w1), ('W2 λ', w2), ('W3', w3)):
        for selected in chosen:
            matches = [p for p in panel['points'] if p['config'] == selected['config']]
            assert len(matches) <= 1
            if not matches:
                result['missing'].append([scenario, selected['label']])
                continue
            if scenario == 'W2 λ':
                assert matches[0] == selected, 'Selection point changed'
            result['points'].append(dict(matches[0], scenario=scenario, session=panel['session']))
    session = ROOT / 'runs' / Path(high['session']).name
    metrics_path = session / 'metrics.json'
    if not metrics_path.exists():
        result['missing'].extend([['W2 2λ', p['label']] for p in chosen])
    else:
        receipt_path = session / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        assert receipt['plan'] == high and receipt['status'] == 'COMPLETE'
        metrics = json.loads(metrics_path.read_text())
        by_name = {Path(m['cell']).name: m for m in metrics}
        assert len(by_name) == len(metrics)
        expected_names = {f'cell-{i:02d}-{c["policy"]}' for i, c in enumerate(high['cells'])}
        assert not set(by_name) - expected_names
        expected_inputs = json.loads((ROOT / 'configs' / Path(high['inputs']).name).read_text())
        for i, (cfg, selected) in enumerate(zip(high['cells'], chosen)):
            name = f'cell-{i:02d}-{cfg["policy"]}'
            m = by_name.get(name)
            if m is None or not m.get('complete'):
                result['missing'].append(['W2 2λ', selected['label']])
                continue
            executed = next(c for c in receipt['cells'] if c['index'] == i)
            runner = next(v for v in executed['command'] if v.endswith('/run_cell.py'))
            assert Path(runner).parent.name == snapshot
            assert executed['exit_code'] == 0 and not m['failed'] and not m['unfinished']
            inp = json.loads((session / name / 'inputs.json').read_text())
            for field in ('model_revision', 'workload', 'sampling', 'ignore_eos', 'output_tokens'):
                assert inp['config'].get(field) == expected_inputs['config'].get(field), field
            assert inp['workload']['actual_prompt_token_ids'] == expected_inputs['workload']['actual_prompt_token_ids']
            result['points'].append(dict(cell=name, label=selected['label'], config=cfg,
                scenario='W2 2λ', session=session.name, stall_rate=m['stall_rate'],
                ttft_p95_s=m['ttft_s']['p95'], gap_p99_s=m['maxgap_s']['p99'],
                finish_reasons=m['finish_reasons']))
        result.update(high_metrics_sha256=sha(metrics_path), high_receipt_sha256=sha(receipt_path))
    for p in result['points']:
        assert 0 <= p['stall_rate'] <= 1
        assert all(math.isfinite(p[k]) and p[k] >= 0 for k in ('ttft_p95_s', 'gap_p99_s'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('w1-plan', 'w2-plan', 'high-plan', 'w3-plan'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--code-snapshot', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output basename')
    args = parser.parse_args()
    result = collect(args.w1_plan, args.w2_plan, args.high_plan, args.w3_plan, args.code_snapshot)
    # Do not leave stale figures when a caller reuses an old output basename.
    assert not any(args.output.with_suffix(s).exists() for s in ('.json', '.png', '.pdf')), 'Use a new output basename'
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if result['missing']:
        print(json.dumps(dict(status='UNMEASURED_OR_INCOMPLETE', missing=result['missing'])))
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    scenarios = ['W1', 'W2 λ', 'W2 2λ', 'W3']
    for selected, color, marker in zip(result['selection']['selected_points'], ['#2876a8', '#d65335'], ['s', 'o']):
        rows = {p['scenario']: p for p in result['points'] if p['label'] == selected['label']}
        for ax, metric, scale in zip(axes, ['stall_rate', 'ttft_p95_s'], [100, 1]):
            ax.plot(range(4), [scale * rows[s][metric] for s in scenarios],
                    color=color, marker=marker, label=selected['label'])
    for ax in axes:
        ax.set_xticks(range(4), scenarios)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel('Requests with maxgap > 200 ms (%)')
    axes[1].set_ylabel('TTFT P95 (s)')
    model = 'Qwen3-8B' if 'Qwen3-8B' in result['model_path'] else 'OLMoE-Instruct'
    fig.suptitle(f"{model} · {result['kv_gib']} GiB · {args.code_snapshot}\nW2-selected settings; single-run exploratory transfer", fontsize=11)
    fig.savefig(args.output.with_suffix('.pdf'))
    fig.savefig(args.output.with_suffix('.png'), dpi=170)
    plt.close(fig)
    print(json.dumps(dict(status='RENDERED', points=len(result['points']))))


if __name__ == '__main__':
    main()
