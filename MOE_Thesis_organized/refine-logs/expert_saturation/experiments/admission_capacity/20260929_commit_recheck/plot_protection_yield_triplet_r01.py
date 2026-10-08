"""Complete-request gap distribution and flow for the retained native triplet."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
r=json.loads((root/'A_PROTECTION_YIELD_TRIPLET_RESULT_R01_20261001.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
for arm,label,color in [('q1','Q1','#276e9b'),('yield','Q10 with early release','#cf7c28'),('plain_q10','Plain Q10','#5d9568')]:
    rows=r['arms'][arm]['metrics']['requests']
    for ax,key in zip(axes,['max_gap_s','flow_s']):
        values=sorted(x[key] for x in rows)
        ax.step(values,[(i+1)/len(values) for i in range(len(values))],where='post',label=label,color=color)
for ax in axes:
    ax.set(ylabel='Cumulative request fraction',ylim=(0,1.02))
    ax.grid(alpha=.18)
axes[0].set_xlabel('Maximum generation gap per request (s)')
axes[1].set_xlabel('Arrival-to-completion time (s)')
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Seen H128: one ordered native triplet; all 384 requests completed',fontsize=11)
for ext in ('png','pdf'):
    p=root/f'A_PROTECTION_YIELD_TRIPLET_FIGURE_R01_20261001.{ext}'
    if p.exists():raise FileExistsError(p)
    fig.savefig(p,dpi=180)
plt.close(fig)
