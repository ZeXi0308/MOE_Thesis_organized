#!/usr/bin/env python3
"""Two complete native-budget blocks; point estimates, not confidence intervals."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
report = json.loads((HERE/'qmsum_chunk_repeat_v1.json').read_text())
out = HERE/'qmsum_chunk_repeat_plot_v1'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
fields = [('mean_completion_s',1,'Mean request completion (s)'),
          ('p95_completion_s',1,'p95 request completion (s)'),
          ('mean_max_host_gap_s',1000,'Mean per-request largest host gap (ms)')]
for ax, (field,scale,label) in zip(axes,fields):
    for pair,color,marker,label_pair in [('forward','#2467a9','o','Run order: 512 → 1024'),
                                         ('reverse','#df7020','s','Run order: 1024 → 512')]:
        values=[report['arms'][pair][b]['metrics'][field]*scale for b in ('1024','512')]
        ax.plot([0,1],values,color=color,marker=marker,markersize=7,linewidth=2,label=label_pair)
    ax.set_xticks([0,1],['1024','512'])
    ax.set_xlim(-.15,1.15)
    ax.set_xlabel('Native batch-token budget')
    ax.set_ylabel(label)
    ax.grid(axis='y',alpha=.18)
    for spine in ('top','right'):ax.spines[spine].set_visible(False)
axes[0].legend(frameon=False,fontsize=9,loc='upper left')
fig.suptitle('Native chunking repeats the gap–completion tradeoff',x=.06,ha='left',fontsize=16,weight='bold')
fig.text(.06,.865,'QMSum200 per cell · same host, prompts, density order and resources · lower is better',fontsize=10,color='#555555')
fig.text(.06,.03,'Two viewed development blocks; no confidence intervals. Output IDs repeat within each budget, but differ for 177/200 requests between budgets.',fontsize=9,color='#555555')
fig.subplots_adjust(left=.07,right=.98,bottom=.22,top=.75,wspace=.38)
for ext in ('png','svg'):fig.savefig(out/f'qmsum_chunk_repeat_v1.{ext}',dpi=200)
plt.close(fig)
print(out)
