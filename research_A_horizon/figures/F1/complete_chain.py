"""Nonoverlapping host recovery intervals from the ACK-instrumented r09 run."""
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/w2-24-calibration-r09/metrics.json'
PHASES=[('preempt_to_allocation_s','Preempt → allocation'),
        ('allocation_to_load_submit_s','Allocation → LOAD submit'),
        ('load_submit_to_load_host_complete_s','LOAD → host completion'),
        ('load_host_complete_to_ack_applied_s','Host completion → ACK applied'),
        ('ack_applied_to_reentry_allocation_s','ACK → reentry allocation'),
        ('reentry_allocation_to_first_output_s','Reentry → first output')]


def main():
    rows=json.loads(SOURCE.read_text());assert len(rows)==1 and rows[0]['complete']
    episodes=rows[0]['recovery_chains']['episodes']
    assert episodes and all(e['path']=='async_LOAD' and len(e['load_jobs'])==1 for e in episodes)
    for e in episodes:
        values=[e['durations'][k] for k,_ in PHASES]
        assert all(v>0 for v in values)
        assert abs(sum(values)-e['durations']['preempt_to_first_output_s'])<1e-8
    colors=['#31688e','#8a6fad','#35a685','#e8aa37','#cc6677','#5a9dc2']
    fig,axes=plt.subplots(1,2,figsize=(11,4),width_ratios=[1,2],layout='constrained')
    bottom=0.;phases=[]
    for i,((key,label),color) in enumerate(zip(PHASES,colors)):
        samples=[dict(request=e['request'],episode=e['index'],seconds=e['durations'][key]) for e in episodes]
        values=[s['seconds'] for s in samples];avg=mean(values)
        phases.append(dict(key=key,label=label,mean_s=avg,median_s=median(values),samples=samples))
        axes[0].bar(0,avg,bottom=bottom,color=color,width=.45);bottom+=avg
        y=[i+.15*(2*j/max(1,len(values)-1)-1) for j in range(len(values))]
        axes[1].scatter(values,y,s=15,alpha=.65,color=color)
        axes[1].plot([median(values)]*2,[i-.22,i+.22],color='black',linewidth=2)
    axes[0].set_xlim(-.6,.6);axes[0].set_ylim(0,bottom*1.13)
    axes[0].set_xticks([0],['All 28 episodes'])
    axes[0].set_ylabel('Mean recovery duration (s)')
    axes[0].text(0,bottom+.015,f'{bottom:.3f} s',ha='center')
    axes[0].set_title('Adjacent intervals sum to total',fontsize=10)
    axes[1].set_yticks(range(len(PHASES)),[p[1] for p in PHASES]);axes[1].invert_yaxis()
    axes[1].set_xscale('log');axes[1].grid(axis='x',alpha=.2)
    axes[1].set_xlabel('Host interval (s, log scale)')
    axes[1].set_title('Episode dots; black ticks are medians',fontsize=10)
    fig.suptitle('Native OLMoE-Instruct · W2 · 24 GiB · λ=5',fontsize=12)
    result=dict(source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        episodes=len(episodes),phases=phases,mean_total_s=bottom,
        semantics='One native development run; episodes are not independent system repetitions. '
        'All episodes have exactly one accepted LOAD and all seven host boundaries. '
        'Adjacent intervals telescope to preemption-return through first-output duration; '
        'these six means may be summed, unlike the overlapping intervals in recovery.pdf. '
        'Host completion is notification, not device completion. JIT costs remain included.')
    (OUT/'complete-chain-data.json').write_text(json.dumps(result,indent=2)+'\n')
    fig.savefig(OUT/'complete-chain.pdf');fig.savefig(OUT/'complete-chain.png',dpi=170)
    plt.close(fig)
    print(json.dumps(dict(episodes=len(episodes),mean_total_s=bottom)))


if __name__=='__main__':main()
