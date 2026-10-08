"""All 128 requests, three native victim rules; no selected subset."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
d=json.loads((root/'A_NATIVE_RESIDENCY_VICTIM_TRIPLET_RESULT_R01_20261002.json').read_text())
assert d['status']=='COMPLETE_TRIPLET'
fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for key,label,color in [('tail','Tail (ordinary backfill)','#276e9b'),('arrival','Original arrival priority','#d6912e'),('service_density','Residence outputs / KV pages','#598950')]:
 rows=d['arms'][key]['metrics']['requests'];assert len(rows)==128
 for ax,field in zip(axs,('max_gap_s','flow_s','ttft_s')):
  values=sorted(r[field] for r in rows)
  ax.step(values,[(i+1)/128 for i in range(128)],where='post',color=color,label=label)
for ax,label in zip(axs,('Max generation gap per request (s)','Arrival-to-completion time (s)','Time to first token (s)')):
 ax.set(xlabel=label,ylim=(0,1.02));ax.grid(alpha=.18)
axs[0].set_ylabel('Cumulative request fraction');axs[0].legend(frameon=False,fontsize=7,loc='lower right')
fig.suptitle('Pressure-point victim selection: 128 complete seen requests per arm',fontsize=11)
for ext in ('png','pdf'):
 out=root/('A_NATIVE_RESIDENCY_VICTIM_FIGURE_R01_20261002.'+ext);assert not out.exists();fig.savefig(out,dpi=180)
plt.close(fig)
