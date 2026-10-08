"""Plot all 128 request outcomes from the retained Q1/Q10 pilot."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root/'A_CAPACITY_PROTECTION_PAIR_RESULT_R03_20261001.json').read_text())
rows = {arm:{r['request_id']:r for r in obj['metrics']['requests']}
        for arm,obj in data['arms'].items()}
fig, axes = plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
for arm, label in [('q1','Q1'),('q10','Q10')]:
    values = sorted(r['max_gap_s'] for r in rows[arm].values())
    axes[0].step(values,[(i+1)/len(values) for i in range(len(values))],where='post',label=label)
axes[0].set(xlabel='Maximum generation gap per request (s)',ylabel='Cumulative request fraction',ylim=(0,1.02))
axes[0].legend(frameon=False)
changed = set(data['output_sequence_difference_requests'])
for different,label,color in [(False,'Same output sequence','#276e9b'),(True,'Different output sequence','#cf7c28')]:
    ids = [rid for rid in rows['q1'] if (rid in changed)==different]
    axes[1].scatter([rows['q1'][rid]['flow_s'] for rid in ids],
                    [rows['q10'][rid]['flow_s'] for rid in ids],s=15,alpha=.8,color=color,label=label)
maximum = max(r['flow_s'] for arm in rows.values() for r in arm.values()) * 1.05
axes[1].plot([0,maximum],[0,maximum],color='gray',linewidth=.8,linestyle='--')
axes[1].set(xlabel='Q1 arrival-to-completion time (s)',ylabel='Q10 arrival-to-completion time (s)',
            xlim=(0,maximum),ylim=(0,maximum))
axes[1].legend(frameon=False,fontsize=8)
for ax in axes:
    ax.grid(alpha=.18)
fig.suptitle('Seen H128: one ordered native pilot; all requests completed',fontsize=11)
for ext in ('png','pdf'):
    output=root/f'A_CAPACITY_PROTECTION_PAIR_FIGURE_R03_20261001.{ext}'
    if output.exists():
        raise FileExistsError(output)
    fig.savefig(output,dpi=180)
plt.close(fig)
