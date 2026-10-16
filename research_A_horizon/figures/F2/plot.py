"""Rebuild the first W2 frontier from complete local cells; never fill missing arms."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SESSIONS=('w2-48-calibration-policy-r03','w2-48-native-aa-r04',
          'w2-48-baselines-r05','w2-48-frontier-r06','w2-48-frontier-r06b')
OUT=Path(__file__).resolve().parent

def collect():
    points=[];excluded=[]
    for name in SESSIONS:
        directory=ROOT/'runs'/name
        if not (directory/'metrics.json').exists():
            excluded.append(dict(session=name,reason='metrics not available'));continue
        receipt=json.loads((directory/'receipt.json').read_text())
        for metrics in json.loads((directory/'metrics.json').read_text()):
            cell=Path(metrics['cell']).name
            index=int(cell.split('-')[1]);cfg=receipt['plan']['cells'][index]
            if not metrics.get('complete') or metrics.get('unfinished') or metrics.get('failed'):
                excluded.append(dict(session=name,cell=cell,reason='not fully completed'));continue
            assert cfg['kv_gib']==48 and cfg['arrival_rate']==20
            assert cfg.get('warmup_profile','legacy')=='legacy'
            payload=json.loads((directory/cell/'inputs.json').read_text())
            assert payload['config']['workload']=='W2'
            assert payload['config']['model_revision']=='7f1c97f440f06ce36705e4f2b843edb5925f4498'
            command=next(c['command'] for c in receipt['cells'] if c['index']==index)
            runner=next(arg for arg in command if arg.endswith('/run_cell.py'))
            code_directory=Path(runner).parent.name
            assert code_directory in ('code','code-r05'), 'Keep corrected-ledger runs separate'
            policy=cfg['policy']
            label=('H='+str(cfg['ha']) if policy=='horizon' else
                   'cap'+str(cfg['cap']) if policy=='native' and cfg['cap']!=256 else policy)
            points.append(dict(session=name,cell=cell,label=label,config=cfg,
                code_directory=code_directory,ledger_version='original_before_inflight_growth_fix',
                model_revision=payload['config']['model_revision'],
                input_tokens_sha256=hashlib.sha256(json.dumps(payload['workload']['actual_prompt_token_ids']).encode()).hexdigest(),
                metrics={k:v for k,v in metrics.items() if k not in
                         ('requests','inherited_B_metrics','thrash_episodes')},
                metrics_file_sha256=hashlib.sha256((directory/'metrics.json').read_bytes()).hexdigest()))
    assert len({p['input_tokens_sha256'] for p in points})<=1, 'Frontier input mismatch'
    return points,excluded

def main():
    points,excluded=collect()
    expected={'native','cap128','cap160','cap192','full-budget','mc','H=16','H=64','H=256','H=1024'}
    missing=sorted(expected-{p['label'] for p in points})
    (OUT/'data.json').write_text(json.dumps(dict(points=points,excluded=excluded,missing_arms=missing,
        semantics='Original-ledger frontier, before the inflight LOAD growth correction. '
        'Legacy warmup only, identical prompt token inputs and model revision. '
        'Raw run-level points. Single development runs except native repetitions; no confidence intervals or dominance claim.'),indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7.4,4.6),layout='constrained')
    styles={'native':('o','#333333'),'cap':('s','#2876a8'),'full-budget':('D','#8c5c29'),
            'mc':('^','#7952a2'),'horizon':('o','#d65335')}
    for label in sorted({p['label'] for p in points}):
        rows=[p for p in points if p['label']==label]
        family='horizon' if label.startswith('H=') else 'cap' if label.startswith('cap') else label
        marker,color=styles[family]
        x=[p['metrics']['ttft_s']['p95'] for p in rows]
        y=[p['metrics']['maxgap_s']['p99'] for p in rows]
        display=label
        if label=='mc' and all(p['metrics'].get('mc_activity',{}).get('successful_relaxations')==0 for p in rows):
            display='MC (0 relaxations)'
        ax.scatter(x,y,marker=marker,c=color,s=36,label=display,alpha=.8)
        if family in ('cap','horizon'):
            for xx,yy in zip(x,y):
                ax.annotate(label,(xx,yy),xytext=(-5 if label=='H=1024' else 5,7 if family=='horizon' else -12),
                            ha='right' if label=='H=1024' else 'left',
                            textcoords='offset points',fontsize=8,color=color)
    curve=sorted([p for p in points if p['config']['policy']=='horizon'],key=lambda p:p['config']['ha'])
    if curve:
        ax.plot([p['metrics']['ttft_s']['p95'] for p in curve],
                [p['metrics']['maxgap_s']['p99'] for p in curve],c=styles['horizon'][1],lw=1,alpha=.6)
    ax.set(xlabel='TTFT P95 (s)',ylabel='Request maxgap P99 (s)',yscale='log',
           title='OLMoE-Instruct · W2 · 48 GiB KV · 20 requests/s')
    ax.grid(True,which='both',alpha=.18)
    ax.legend(fontsize=8,ncol=2,loc='best')
    fig.suptitle('Development frontier'+(' — incomplete arms' if missing else ''),fontsize=12)
    fig.savefig(OUT/'F2.pdf');fig.savefig(OUT/'F2.png',dpi=170);plt.close(fig)
    header=['run','arm','TTFT_P50_s','TTFT_P95_s','maxgap_P99_s','stall_rate','output_token_s',
            'flow_mean_s','flow_P95_s','joint_goodput_rps','preemptions','thrash','scheduler_us_per_step','stop','length']
    lines=['\t'.join(header)]
    for p in points:
        m=p['metrics'];values=[p['session']+'/'+p['cell'],p['label'],m['ttft_s']['p50'],m['ttft_s']['p95'],
            m['maxgap_s']['p99'],m['stall_rate'],m['output_token_s'],m['flow_s']['mean'],m['flow_s']['p95'],
            m['joint_goodput_rps'],m['preemption_count'],m['thrash_count'],m['scheduler_us_per_step'],
            m['finish_reasons'].get('stop',0),m['finish_reasons'].get('length',0)]
        lines.append('\t'.join(str(x) for x in values))
    (OUT/'T1-runs.tsv').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(points=len(points),missing_arms=missing)))

if __name__=='__main__':main()
