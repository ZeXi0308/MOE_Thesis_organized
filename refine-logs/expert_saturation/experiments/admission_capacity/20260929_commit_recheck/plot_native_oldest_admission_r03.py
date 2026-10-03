#!/usr/bin/env python3
"""Complete target/victim progress in three independent R03 executions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
session=root/'moe-a-native-oldest-admission-session-r03-20261002'
specs=[('native','Native','#4c78a8'),('queue_only','Queue priority','#f58518'),
       ('queue_fund','Capacity funding','#228b65')]
peers=[('memory-train-article-0042224','Selected paused request'),
       ('memory-train-article-0042067','Planned / actual victim')]
fig,axes=plt.subplots(1,2,figsize=(9.6,3.5),sharex=True,sharey=True)
for i,(mode,label,color) in enumerate(specs):
    archive=session/f'cell-{i:02d}-{mode}'/'archive'
    raw=json.loads((archive/'raw.json').read_text())
    store=json.loads((archive/'selective-store.json').read_text())
    origin=raw['measurement_origin_perf_counter_s']
    anchor=next(e for e in store['events'] if e['event']=='oldest_anchor')
    anchor_s=anchor['host_perf_counter_s']-origin
    for ax,(rid,title) in zip(axes,peers):
        req=next(r for r in raw['requests'] if r['request_id']==rid)
        times=req['token_times_s']
        ax.plot(times,range(1,len(times)+1),lw=1.7,color=color,label=label)
        for e in raw['preemption_events']:
            if e['request_id']==rid and e.get('original_preemption_returned'):
                ax.plot(e['method_entered_s'],e['last_returned_output_count'],
                        marker='x',ms=5,color=color)
        ax.axvline(anchor_s,color=color,ls=':',lw=.8,alpha=.7)
        ax.set_title(title,fontsize=11)
        ax.set_xlabel('Time since measurement origin (s)')
        ax.grid(alpha=.16)
        ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Cumulative actual output tokens')
axes[0].set_xlim(5,35)
axes[0].set_ylim(0,1050)
axes[1].legend(loc='lower right',frameon=False,fontsize=9)
fig.suptitle('One funded recovery: early output, then later native preemptions',fontsize=12)
fig.text(.5,.01,'Independent executions, not a shared physical pre-state. Dotted lines: online selection; crosses: actual preemptions.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,.95))
out=root/'paper_a'/'figures';out.mkdir(exist_ok=True)
for ext in ('pdf','png'):
    fig.savefig(out/f'oldest_admission_r03_progress.{ext}',dpi=170,bbox_inches='tight')
print(out/'oldest_admission_r03_progress.png')
