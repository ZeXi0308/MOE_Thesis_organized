from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
d=json.loads((root/"recovery_residencies.json").read_text())
r=json.loads((root.parent/"A_NATIVE_OLDEST_REPEAT_TRIPLET_RESULT_R02_20261002.json").read_text())
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout="constrained")
for arm,label,col in [("native","Native","#566575"),("queue_only","Queue priority","#d68624"),("queue_fund","Funded Q1","#14796f")]:
 vals=sorted(x["max_gap_s"] for x in r["arms"][arm]["metrics"]["requests"])
 axes[0].step(vals,np.arange(1,len(vals)+1)/len(vals),where="post",label=label,color=col)
axes[0].set(xlabel="Per-request maximum output gap (s)",ylabel="Request fraction",title="(a) Complete request cohort, n=128")
axes[0].legend(fontsize=8,loc="lower right");axes[0].grid(alpha=.18)
for kind,label,col in [("NATIVE_TAIL","Native re-eviction","#385f98"),("FUNDS_OTHER_RECOVERY","Funds another recovery","#bc6841")]:
 es=[e for e in d["episodes"] if e.get("next_eviction_kind")==kind]
 axes[1].scatter([e["new_outputs_until_next_eviction_or_completion"] for e in es],[e["next_eviction_last_to_next_output_gap_s"] for e in es],s=27,label=label,color=col,alpha=.85)
axes[1].set(xscale="log",xlabel="New outputs before re-eviction",ylabel="Next observed output gap (s)",title="(b) 48 re-evicted recoveries")
axes[1].legend(fontsize=8);axes[1].grid(alpha=.18)
es=[e for e in d["episodes"] if e.get("next_eviction_kind")=="NATIVE_TAIL" and e["new_outputs_until_next_eviction_or_completion"]<=9]
labels=[e["source_request_id"].split("-")[-1] for e in es]
axes[2].bar(range(len(es)),[e["target_owned_unused_slots_at_native_next_eviction"] for e in es],color="#385f98")
axes[2].set_xticks(range(len(es)),labels,rotation=40,ha="right")
axes[2].set(xlabel="Request suffix (six events; sources repeat)",ylabel="Unused owned KV token slots",title="(c) Short native re-evictions")
fig.savefig(root/"recovery_residencies.png",dpi=190)
fig.savefig(root/"recovery_residencies.svg")
