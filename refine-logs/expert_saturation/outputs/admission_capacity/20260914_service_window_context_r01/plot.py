from pathlib import Path
import json
from bisect import bisect_right
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'tradeoff'
d=json.loads((p/'analysis.json').read_text());by={c['label']:c for c in d['cells']}
fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
for b in (0,1):
 for role,color in zip(['native','most_output','least_progress'],['#7556a8','#ca8a04','#167cad']):
  c=by[f'context-block{b}-{role}'];g=sorted(r['max_gap_s'] for r in c['requests'])
  x=[0,*sorted(set(g)),10.2];y=[bisect_right(g,t)/c['wall_s'] for t in x]
  for ax in axes[:,b]:ax.step(x,y,where='post',label=role,color=color)
 axes[0,b].set_title(f'Execution block {b}: full range');axes[0,b].set_xlim(0,10.2)
 axes[1,b].set_title('Same curves, gap <= 2.3 s');axes[1,b].set_xlim(0,2.3)
 for ax in axes[:,b]:ax.grid(alpha=.2);ax.set_xlabel('Allowed maximum engine-return gap (s)')
for ax in axes[:,0]:ax.set_ylabel('Gap-qualified completions / s')
axes[0,0].legend(loc='lower right',fontsize=9)
fig.suptitle('Heterogeneous context calibration: all thresholds retained')
fig.text(.5,.015,'32 reused documents; fixed 1024 outputs; TTFT/TPOT unconstrained; capture costs included.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.96))
for suffix in ('png','svg'):fig.savefig(p/f'gap_service_curves.{suffix}',dpi=150)
plt.close(fig)
