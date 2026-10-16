#!/usr/bin/env python3
"""Build F1 as a scientific Matplotlib artifact; measured inputs are optional.

Curve labels name topology, not unmeasured performance. Optional summary CSVs
come from analyze_routes.py and retain their provenance mode in the figure.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,nargs="*",default=[])
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--note",default=None,help="Additional evidence boundary printed on the figure")
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.1),gridspec_kw={"width_ratios":[1.2,1]})
    batches=list(range(1,65))
    shapes=[("OLMoE topology: E=64, k=8",64,8),
            ("Qwen3-30B-A3B topology: E=128, k=8",128,8),
            ("DeepSeek-V2-Lite topology: E=64, k=6",64,6),
            ("Illustrative fine-grained: E=256, k=8",256,8)]
    rows=[]
    for name,e,k in shapes:
        vals=[(1-k/e)**(b-1) for b in batches]
        axes[0].plot(batches,vals,label=name,lw=2)
        rows.extend(dict(topology=name,experts=e,top_k=k,batch_size=b,
                         uniform_singleton_fraction=v) for b,v in zip(batches,vals))
    axes[0].set(title="(a) Uniform iid occupancy model",xlabel="Decode tokens per step, B",
                ylabel="Singleton slots / all selected slots",ylim=(-.02,1.02),xlim=(1,64))
    axes[0].set_xticks([1,8,16,32,64])
    axes[0].legend(fontsize=8,frameon=False)
    axes[0].grid(alpha=.2)
    measured=defaultdict(list)
    for path in args.summary:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                measured[(row["model"],row["mode"],int(row["batch_size"]))].append(row)
    if measured:
        names=sorted({(m,mode) for m,mode,b in measured})
        for i,(name,mode) in enumerate(names):
            bs=sorted(b for m,md,b in measured if (m,md)==(name,mode))
            actual=[]
            fitted=[]
            for b in bs:
                rs=measured[(name,mode,b)]
                slots=sum(int(r["slots"]) for r in rs)
                actual.append(sum(int(r["singleton_slots"]) for r in rs)/slots)
                fitted.append(sum(float(r["iid_fitted_q_singleton_fraction"])*int(r["slots"]) for r in rs)/slots)
            color={"OLMoE":"C0","LLM-jp":"C3"}.get(name,f"C{i}")
            kind="native routes" if mode=="actual-decode" else "route replay"
            axes[1].plot(bs,actual,"o-",color=color,label=f"{name}: {kind}")
            axes[1].plot(bs,fitted,"x--",color=color,label=f"{name}: fitted iid q")
        native=all(mode=="actual-decode" for _,mode in names)
        axes[1].set(title="(b) Native router occupancy" if native else "(b) Historical diagnostic only",
                    xlabel="Captured decode batch, B" if native else "Reconstructed session batch, B",
                    ylim=(-.02,1.02),xticks=sorted({b for _,_,b in measured}))
        axes[1].legend(fontsize=8,frameon=False)
        axes[1].grid(alpha=.2)
        foot=("Native router counts measure occupancy; fitted q is in-sample and assumes iid tokens.\nOccupancy alone does not measure DRAM bytes or serving latency." if native else
              "Historical panel: cached decode route replay with reconstructed batches, not native serving.\nNative B=1–64 capture for the three target models remains to be collected; fitted q is in-sample.")
    else:
        axes[1].axis("off")
        axes[1].text(.05,.65,"Native decode measurements pending",fontsize=13,weight="bold")
        axes[1].text(.05,.4,"Collect batch_id + request_id + decode_step\nwith full top-k routes at each layer.\nText-position packing is diagnostic replay.",linespacing=1.8)
        foot="The left panel is a topology model, not measured routing or hardware bandwidth."
    if args.note:
        foot = args.note + "\n" + foot
    fig.text(.08,.015,foot,fontsize=8,color=".35",va="bottom")
    fig.tight_layout(rect=(0,.12,1,1))
    for ext in ["pdf","png","svg"]:
        fig.savefig(args.output_dir/f"F1_occupancy.{ext}",dpi=180)
    plt.close(fig)
    with (args.output_dir/"F1_uniform_model.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    main()
