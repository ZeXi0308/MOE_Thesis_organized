"""Recorded reservation samples and whole-scheduler time, without cost subtraction."""
import hashlib
import json
import argparse
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--code-snapshot',default='code-r12')
    args=parser.parse_args()
    panel=None
    if args.plan:
        sys.path.insert(0,str(ROOT/'figures/F2'))
        from plot_corrected import read_panel
        panel=read_panel(args.plan,args.code_snapshot)
        base=OUT/(panel['session']+'-cost')
        if panel['missing']:
            base.with_suffix('.json').write_text(json.dumps(panel,indent=2)+'\n')
            print(json.dumps(dict(status='UNMEASURED_OR_INCOMPLETE',missing=panel['missing'])))
            return
        source=ROOT/'runs'/panel['session']/'metrics.json'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==panel['metrics_sha256']
        by_cell={Path(m['cell']).name:m for m in json.loads(source.read_text())}
        points=[dict(p,metrics=by_cell[p['cell']]) for p in panel['points']]
        title_model='Qwen3-8B' if 'Qwen3-8B' in panel['model_path'] else 'OLMoE-Instruct'
        title=f"{title_model} · {panel['workload']} · {panel['kv_gib']} GiB KV · {panel['arrival_rate']} requests/s"
        labels=[p['label'] for p in points]
        data_path=base.with_suffix('.json')
    else:
        source = ROOT / 'figures/F2/data.json'
        points = json.loads(source.read_text())['points']
        base=OUT/'F5'
        data_path=OUT/'data.json'
        title='OLMoE-Instruct · W2 · 48 GiB KV · 20 requests/s'
        labels = ['native', 'cap128', 'cap160', 'cap192', 'full-budget', 'mc',
                  'H=16', 'H=64', 'H=256', 'H=1024']
    horizons = sorted((p for p in points if p['config']['policy'] == 'horizon'),
                      key=lambda p: p['config']['ha'])
    record = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), points=points,
        semantics='Reservation quantiles sample KNOWN waiting-gate attempts, not uniform steps. '
        'P50/P95 describe within-run states, not uncertainty. Scheduler times are whole-run '
        'per-step means including instrumentation; unequal policy trajectories prevent '
        'interpreting differences as isolated hook overhead. Warmup profile follows the recorded config.')
    if panel is not None:
        record['panel']=panel
    # Missing measurements must not appear as zero-cost policy points.
    for point in points:
        value=point['metrics']['scheduler_us_per_step']
        assert isinstance(value,(int,float)) and math.isfinite(value) and value>=0
    for point in horizons:
        for metric in ('peer_horizon_growth_ratio','peer_committed_ratio'):
            for quantile in ('p50','p95'):
                value=point['metrics'][metric][quantile]
                assert isinstance(value,(int,float)) and math.isfinite(value) and value>=0
    data_path.write_text(json.dumps(record, indent=2) + '\n')
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    x = [p['config']['ha'] for p in horizons]
    for metric, color, label in [('peer_horizon_growth_ratio', '#d65335', 'Growth only'),
                                ('peer_committed_ratio', '#2876a8', 'Including R/I')]:
        for quantile, style in [('p50', '-'), ('p95', '--')]:
            left.plot(x, [100*p['metrics'][metric][quantile] for p in horizons],
                      marker='o', ls=style, color=color, label=label+' '+quantile.upper())
    left.set(xscale='log', xlabel='Admission horizon (tokens)', ylabel='GPU KV pages (%)',
             title='(a) KNOWN gate samples')
    left.set_xticks(x, [str(h) for h in x])
    left.legend(fontsize=8)
    for i, label in enumerate(labels):
        values = [p['metrics']['scheduler_us_per_step'] for p in points if p['label'] == label]
        right.scatter([i]*len(values), values, s=25,
                      color='#d65335' if label.startswith('H=') else '#2876a8')
    right.set_xticks(range(len(labels)), labels, rotation=55, ha='right')
    right.set(ylabel='Whole scheduler (µs/step)', title='(b) Each point is one run')
    right.set_ylim(bottom=0)
    for ax in (left, right): ax.grid(alpha=.2)
    fig.suptitle(title+(('\n'+panel['code_snapshot']) if panel else ''))
    fig.savefig(base.with_suffix('.pdf'));fig.savefig(base.with_suffix('.png'), dpi=170)
    plt.close(fig)
    print(json.dumps(dict(runs=len(points), horizons=x)))


if __name__ == '__main__': main()
