"""Fixed-setting transfer across workloads; missing runs remain missing."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SCENARIOS=[('W1',20),('W2',20),('W2',40),('W3',20)]
SESSIONS=['w2-48-frontier-r06','w2-48-frontier-r06b','load-transfer-r08']


def collect():
    points=[]
    for session in SESSIONS:
        directory=ROOT/'runs'/session
        file=directory/'metrics.json'
        if not file.exists(): continue
        receipt=json.loads((directory/'receipt.json').read_text())
        for metrics in json.loads(file.read_text()):
            name=Path(metrics['cell']).name
            index=int(name.split('-')[1]);cfg=receipt['plan']['cells'][index]
            if cfg['kv_gib']!=48 or cfg.get('warmup_profile','legacy')!='legacy': continue
            if not metrics['complete'] or metrics['failed'] or metrics['unfinished']: continue
            if cfg['policy']=='native' and cfg['cap']==160: arm='cap160'
            elif cfg['policy']=='horizon' and cfg['ha']==256 and cfg['hr']==32: arm='H=256/R=32'
            else: continue
            inp=directory/name/'inputs.json';payload=json.loads(inp.read_text())
            family=payload['config']['workload'];scenario=(family,cfg['arrival_rate'])
            if scenario not in SCENARIOS: continue
            points.append(dict(session=session,cell=name,arm=arm,scenario=list(scenario),
                config=cfg,model_revision=payload['config']['model_revision'],
                input_tokens_sha256=hashlib.sha256(json.dumps(payload['workload']['actual_prompt_token_ids']).encode()).hexdigest(),
                metrics_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                metrics={k:v for k,v in metrics.items() if k not in ('requests','inherited_B_metrics','thrash_episodes')}))
    for scenario in SCENARIOS:
        group=[p for p in points if tuple(p['scenario'])==scenario]
        assert len({p['input_tokens_sha256'] for p in group})<=1
        assert len({p['model_revision'] for p in group})<=1
        assert len({p['arm'] for p in group})==len(group), 'Duplicate run: choose explicitly'
    return points


def main():
    points=collect();arms=['cap160','H=256/R=32']
    missing=[dict(scenario=s,arm=a) for s in SCENARIOS for a in arms
             if not any(tuple(p['scenario'])==s and p['arm']==a for p in points)]
    (OUT/'data.json').write_text(json.dumps(dict(points=points,missing=missing,
        selection='W2 lambda20 development: cap160; H=256 is first scanned horizon eliminating preemption. Fixed for transfer.',
        semantics='One run per point, no significance claim. W2 lambda20 reused selection data; other conditions are transfer tests. Legacy warmup, natural EOS.'),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for arm,color,marker in zip(arms,['#2876a8','#d65335'],['s','o']):
        rows={tuple(p['scenario']):p for p in points if p['arm']==arm}
        for ax,metric in zip(axes,['stall_rate','ttft_s']):
            y=[]
            for scenario in SCENARIOS:
                if scenario not in rows:y.append(float('nan'));continue
                m=rows[scenario]['metrics'];y.append(100*m[metric] if metric=='stall_rate' else m[metric]['p95'])
            ax.plot(range(4),y,marker=marker,color=color,label=arm)
    for ax in axes:
        ax.set_xticks(range(4),['W1','W2 λ','W2 2λ','W3'])
        ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Requests with maxgap > 200 ms (%)')
    axes[1].set_ylabel('TTFT P95 (s)')
    fig.suptitle('Fixed settings · OLMoE-Instruct · 48 GiB'+(' · incomplete' if missing else ''))
    fig.savefig(OUT/'F3.pdf');fig.savefig(OUT/'F3.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(points=len(points),missing=len(missing))))


if __name__=='__main__':main()
