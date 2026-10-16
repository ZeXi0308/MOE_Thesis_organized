"""Plot one complete ten-arm frontier without merging historical code versions."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def read_panel(plan_path,code_snapshot):
    plan=json.loads(plan_path.read_text());cells=plan['cells']
    assert len(cells)==10,'Ten-arm frontier required'
    assert [c['policy'] for c in cells]==['native']*4+['full-budget',plan['provenance']['mc_policy']]+['horizon']*4
    assert cells[0]['cap']==256 and len({c['cap'] for c in cells[:4]})==4
    assert [c['ha'] for c in cells[6:]]==[16,64,256,1024]
    assert all(c['hr']==32 for c in cells[6:])
    assert len({(c['kv_gib'],c['arrival_rate'],c.get('warmup_profile','legacy')) for c in cells})==1
    inputs_path=ROOT/'configs'/Path(plan['inputs']).name
    assert hashlib.sha256(inputs_path.read_bytes()).hexdigest()==plan['provenance']['inputs_sha256']
    expected_inputs=json.loads(inputs_path.read_text())
    session=ROOT/'runs'/Path(plan['session']).name
    result=dict(session=session.name,code_snapshot=code_snapshot,points=[],missing=[],
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        model_revision=expected_inputs['config']['model_revision'],
        model_path=expected_inputs['config']['model_path'],
        workload=expected_inputs['config']['workload'],kv_gib=cells[0]['kv_gib'],
        arrival_rate=cells[0]['arrival_rate'],
        semantics='One run per arm; no pooling across snapshots, no uncertainty estimate. '
        'Natural generation may differ; these are development comparisons.')
    metrics_path=session/'metrics.json'
    if not metrics_path.exists():
        result['missing']=[f'cell-{i:02d}-{c["policy"]}' for i,c in enumerate(cells)]
        return result
    receipt_path=session/'receipt.json';receipt=json.loads(receipt_path.read_text())
    assert receipt['plan']==plan,'Executed plan differs'
    metrics=json.loads(metrics_path.read_text())
    by_name={Path(m['cell']).name:m for m in metrics}
    assert len(by_name)==len(metrics)
    expected={f'cell-{i:02d}-{c["policy"]}' for i,c in enumerate(cells)}
    assert not set(by_name)-expected
    for i,cfg in enumerate(cells):
        name=f'cell-{i:02d}-{cfg["policy"]}';m=by_name.get(name)
        if m is None or not m.get('complete'):
            result['missing'].append(name);continue
        executed=next(c for c in receipt['cells'] if c['index']==i)
        runner=next(v for v in executed['command'] if v.endswith('/run_cell.py'))
        assert Path(runner).parent.name==code_snapshot,'Mixed code snapshots'
        assert executed['exit_code']==0 and not m['failed'] and not m['unfinished']
        inp=json.loads((session/name/'inputs.json').read_text())
        for field in ('model_revision','workload','sampling','ignore_eos','output_tokens'):
            assert inp['config'].get(field)==expected_inputs['config'].get(field),field
        assert inp['workload']['actual_prompt_token_ids']==expected_inputs['workload']['actual_prompt_token_ids']
        label=(f"H={cfg['ha']}" if cfg['policy']=='horizon' else
               f"cap{cfg['cap']}" if cfg['policy']=='native' and cfg['cap']!=256 else cfg['policy'])
        result['points'].append(dict(cell=name,label=label,config=cfg,
            ttft_p95_s=m['ttft_s']['p95'],gap_p99_s=m['maxgap_s']['p99'],
            stall_rate=m['stall_rate'],finish_reasons=m['finish_reasons']))
    result.update(metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                  receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--code-snapshot',default='code-r12')
    args=parser.parse_args();panel=read_panel(args.plan,args.code_snapshot)
    base=OUT/panel['session']
    base.with_suffix('.json').write_text(json.dumps(panel,indent=2)+'\n')
    if panel['missing']:
        print(json.dumps(dict(status='UNMEASURED_OR_INCOMPLETE',missing=panel['missing'])))
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,4.5),layout='constrained')
    colors={'native':'#333333','cap':'#2876a8','full-budget':'#8c5c29','mc-load':'#7952a2','mc':'#7952a2','horizon':'#d65335'}
    markers=['o','s','^','v','D','P','o','^','s','v']
    for index,point in enumerate(panel['points']):
        cfg=point['config'];family='cap' if point['label'].startswith('cap') else cfg['policy']
        ax.scatter(point['ttft_p95_s'],point['gap_p99_s'],color=colors[family],
                   marker=markers[index],s=38,label=point['label'])
    curve=panel['points'][6:]
    ax.plot([p['ttft_p95_s'] for p in curve],[p['gap_p99_s'] for p in curve],color=colors['horizon'],linewidth=1)
    model='Qwen3-8B' if 'Qwen3-8B' in panel['model_path'] else 'OLMoE-Instruct'
    ax.set(xlabel='TTFT P95 (s)',ylabel='Request maxgap P99 (s)',yscale='log',
        title=f"{model} · {panel['workload']} · {panel['kv_gib']} GiB · λ={panel['arrival_rate']}")
    ax.grid(which='both',alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Corrected-ledger development frontier · '+panel['code_snapshot'],fontsize=11)
    fig.savefig(base.with_suffix('.pdf'));fig.savefig(base.with_suffix('.png'),dpi=170)
    plt.close(fig);print(json.dumps(dict(status='RENDERED',points=len(panel['points']))))


if __name__=='__main__':main()
