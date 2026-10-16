#!/usr/bin/env python3
"""Plot measured F2/F3/F4/F6 results without manufacturing missing experiments.

Examples, from research_K_expert_skip (matplotlib required for nonempty plots):
  python code/plot_results.py --bench runs/bench_r01 --quality runs/quality_b1.json \
    runs/quality_b8.json --microbench runs/policy_microbench.json \
    --output-dir figures/results_r01

Bench input accepts measurements.jsonl (current writer), summary.jsonl, or a
summary JSON list / {rows|measurements|results: [...]}. Quality input is the
quality_decode.py manifest plus its adjacent *_artifacts or directory artifacts.
Remote absolute artifact paths are remapped by basename to adjacent artifacts.

F2 pairs each candidate with native in the same run/repeat/batch. Footprints use
logical_weight_bytes_after, including native/global-native identity replay; they
are not hardware DRAM counters. F3 pairs dense counters and candidate counters
within the same teacher-forced group, then uses common groups across shown arms.
Metrics are token-weighted; uncertainty units remain runs, never tokens/groups.
F4 uses NVML sampled-power integration as primary; <3 s windows are diagnostics.
F6 distinguishes eager host dispatch from captured device execution per call.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import textwrap


def number(value):
    return float(value) if isinstance(value,(int,float)) and math.isfinite(value) else None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    path=Path(path)
    if path.suffix==".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data=read_json(path)
    if isinstance(data,list):return data
    for key in ("rows","measurements","results"):
        if isinstance(data.get(key),list):return data[key]
    if "batch" in data and "arm" in data:return [data]
    raise ValueError(f"unrecognized bench summary schema: {path}")


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"))


def slug(text):
    return re.sub(r"[^a-zA-Z0-9_-]+","_",text).strip("_")[:60] or "condition"


def label(arm,spec):
    spec=spec or {}
    mode=spec.get("mode")
    if mode=="global_native":return f"{arm} (native k)"
    if mode=="global":return f"{arm} (post-topk)"
    return arm


class Sources:
    def __init__(self):
        self.paths={};self.warnings=[]

    def use(self,path):
        path=Path(path).resolve()
        if str(path) not in self.paths:
            self.paths[str(path)]={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        return str(path)

    def warn(self,message):
        if message not in self.warnings:self.warnings.append(message)


def discover(paths,kind,sources):
    found=[]
    for raw in paths:
        path=Path(raw)
        if path.is_file():found.append(path)
        elif path.is_dir():
            candidates=list(path.rglob("*.json"))+list(path.rglob("*.jsonl"))
            if kind=="bench":
                priority={"measurements.jsonl":0,"summary.jsonl":1,"summary.json":2}
                merged_roots=set()
                for item in candidates:
                    if item.name=="config.json":
                        try:
                            if read_json(item).get("process_protocol"):merged_roots.add(item.parent)
                        except (ValueError,OSError,AttributeError):pass
                parents=defaultdict(list)
                for item in candidates:
                    if any(owner in item.parents and item.parent!=owner for owner in merged_roots):continue
                    if item.name in priority:parents[item.parent].append(item)
                for items in parents.values():
                    for item in sorted(items,key=lambda x:priority[x.name]):
                        try:rows=read_rows(item)
                        except (ValueError,AttributeError):continue
                        if rows and all("arm" in x and "batch" in x for x in rows):
                            found.append(item);break
            else:
                for item in candidates:
                    if item.suffix!=".json":continue
                    try:data=read_json(item)
                    except (ValueError,OSError):continue
                    if not isinstance(data,dict):continue
                    if kind=="quality" and "batches" in data and "quality_by_arm" in data:found.append(item)
                    if kind=="microbench" and any("eager_wall_us_per_call" in r for r in data.get("rows",[]) if isinstance(r,dict)):
                        found.append(item)
        else:sources.warn(f"Input missing: {path}")
    return sorted(set(p.resolve() for p in found))


def condition(args,hardware,kind,run_path):
    model=args.get("model","unspecified-model")
    fields={"model":model,"dtype":args.get("dtype","unspecified"),
            "quantization":args.get("quantization") or "none",
            "policy_backend":args.get("backend",args.get("policy_backend","unspecified"))}
    if kind=="bench":
        fields.update(hardware=hardware,prompt_tokens=args.get("prompt_tokens"),newtokens=args.get("newtokens"),
                      execution=args.get("execution","eager"),
                      graph_counter_pass=args.get("graph_counter_pass",False))
        # Do not pool unlabeled hardware from unrelated runs.
        if hardware=="unspecified":fields["unverified_hardware_run"]=str(run_path)
    else:
        fields.update(prefix_tokens=args.get("prefix_tokens"),decode_tokens=args.get("tokens"),
                      articles=args.get("articles"),gate_threshold=args.get("gate_threshold"),tail_rank=args.get("tail_rank"))
    key=canonical(fields)
    title=f"{Path(model).name} | {fields['dtype']} | quant={fields['quantization']} | {fields['policy_backend']}"
    if kind=="bench":title+=f" | {fields['execution']} | {hardware}"
    return key,title,fields


def unwrap_snapshot(data):
    if isinstance(data,dict) and "snapshot" in data:data=data["snapshot"]
    if isinstance(data,list):
        if len(data)!=1:raise ValueError("expected one worker snapshot")
        data=data[0]
    return data


def counter_values(data):
    data=unwrap_snapshot(data)
    block=data.get("layers",{})
    if isinstance(block,dict) and isinstance(block.get("totals"),dict):return block["totals"]
    if isinstance(data.get("totals"),dict):return data["totals"]
    return {}


def bench_counters(row,root,sources):
    cell=root/row.get("folder","")
    mode=(row.get("spec") or {}).get("mode")
    if row.get("spec") is None and row["arm"]=="native":prefix="native"
    elif mode=="global_native":prefix="global_native"
    else:prefix=None
    candidates=([cell/f"{prefix}_counters.json"] if prefix else [])+[cell/"counters.json"]
    for path in candidates:
        if path.is_file():
            data=counter_values(read_json(path))
            if number(data.get("logical_weight_bytes_after")) is None:continue
            if prefix and path.name==f"{prefix}_counters.json":
                identity=cell/f"{prefix}_counter_identity.json"
                if identity.is_file():
                    sources.use(identity)
                    if read_json(identity).get("identical_output_tokens") is not True:
                        sources.warn(f"Rejected nonidentical counter replay: {identity}");return {},str(path)
                else:sources.warn(f"Identity replay certificate missing; counter source retained as reported: {path}")
            return data,sources.use(path)
    return row.get("logical_counters") or {},"summary row logical_counters"


def load_bench(paths,sources):
    rows=[]
    for path in discover(paths,"bench",sources):
        sources.use(path);root=path.parent
        config_path=root/"config.json"
        config=read_json(config_path) if config_path.exists() else {}
        if config_path.exists():sources.use(config_path)
        args=config.get("args",{})
        device_path=root/"power_device.json"
        device=read_json(device_path) if device_path.exists() else {}
        if device_path.exists():sources.use(device_path)
        hardware=device.get("name",config.get("gpu","unspecified"))
        key,title,details=condition(args,hardware,"bench",root)
        state_path=root/"status.json"
        state=read_json(state_path).get("status","unknown") if state_path.exists() else "unknown"
        if state_path.exists():sources.use(state_path)
        try:input_rows=read_rows(path)
        except (ValueError,AttributeError) as error:sources.warn(str(error));continue
        for index,row in enumerate(input_rows,1):
            if not {"arm","batch"}.issubset(row):
                sources.warn(f"Skip unsupported bench row {path}:{index}");continue
            counters,counter_source=bench_counters(row,root,sources)
            power_source="summary row"
            power_path=root/row.get("folder","")/"energy.json"
            power=(read_json(power_path).get("power") or {}) if power_path.exists() else {}
            if power_path.exists():power_source=sources.use(power_path)
            duration=number((power.get("boundary") or {}).get("workload_elapsed_s",row.get("energy_window_s")))
            denom=number(power.get("decode_output_tokens",row.get("decode_output_tokens")))
            sampled=number(power.get("sampled_energy_j"))
            energy=sampled/denom if sampled is not None and denom and denom>0 else None
            if energy is None and row.get("decode_energy_primary_method")=="nvml_power_trapezoid":
                energy=number(row.get("decode_j_per_token"))
            if row.get("decode_j_per_token") is not None and energy is None:
                sources.warn(f"No verified sampled-power integration; omitted F4 energy: {path}:{index}")
            tokens=number(row.get("decode_output_tokens"))
            actual=number(counters.get("logical_weight_bytes_after"))
            rows.append({"condition":key,"condition_title":title,"run_id":str(root),"run_status":state,
                "repeat":int(row.get("repeat",0)),"batch":int(row["batch"]),"arm":row["arm"],
                "label":label(row["arm"],row.get("spec")),"spec":canonical(row.get("spec")),
                "diagnostic":bool(row.get("diagnostic",False)),"tpot_ms":number(row.get("mean_tpot_ms")),
                "decode_output_tokens":tokens,"logical_weight_bytes_after":actual,
                "logical_weight_bytes_per_decode_token":actual/tokens if actual is not None and tokens and tokens>0 else None,
                "counter_tokens_across_layers":number(counters.get("tokens")),
                "logical_counter_source":counter_source,"j_per_decode_token":energy,
                "energy_window_s":duration,"energy_steady_window":duration is not None and duration>=3,
                "energy_source":power_source,"summary_source":str(path),"summary_record":index,
                "scope":args.get("execution","eager")+" fixed-output decode; logical footprint not DRAM bytes","condition_details":canonical(details)})
    baseline={}
    for row in rows:
        if row["arm"]=="native":
            key=(row["run_id"],row["repeat"],row["batch"])
            if key in baseline:raise ValueError(f"ambiguous native baseline: {key}")
            baseline[key]=row
    for row in rows:
        base=baseline.get((row["run_id"],row["repeat"],row["batch"]))
        row.update(tpot_saving_fraction=None,logical_footprint_saving_fraction=None,native_summary_record=None,native_counter_source=None)
        if base is None:
            sources.warn(f"No matched native baseline for {row['run_id']} repeat={row['repeat']} B={row['batch']}");continue
        row["native_summary_record"]=base["summary_record"]
        row["native_counter_source"]=base["logical_counter_source"]
        if row["tpot_ms"] is not None and base["tpot_ms"] and base["tpot_ms"]>0:
            row["tpot_saving_fraction"]=1-row["tpot_ms"]/base["tpot_ms"]
        sizes=[row["logical_weight_bytes_after"],base["logical_weight_bytes_after"]]
        same_tokens=row["decode_output_tokens"]==base["decode_output_tokens"] and row["decode_output_tokens"] is not None
        same_counter_tokens=row["counter_tokens_across_layers"]==base["counter_tokens_across_layers"] and row["counter_tokens_across_layers"] is not None
        if all(x is not None for x in sizes) and sizes[1]>0 and same_tokens and same_counter_tokens:
            row["logical_footprint_saving_fraction"]=1-sizes[0]/sizes[1]
        elif row["arm"]!="native":
            sources.warn(f"Footprint ratio unavailable/mismatched decode population: {row['run_id']} {row['arm']} B={row['batch']} repeat={row['repeat']}")
    return rows


def artifact_path(manifest,recorded):
    path=Path(recorded)
    candidates=[path,manifest.parent/path.name,manifest.with_name(manifest.stem+"_artifacts")/path.name]
    for item in candidates:
        if item.is_file():return item.resolve()
    matches=list(manifest.parent.glob(f"*_artifacts/{path.name}"))
    if len(matches)==1:return matches[0].resolve()
    raise FileNotFoundError(f"quality artifact missing/ambiguous: {recorded} (manifest {manifest})")


def load_quality(paths,sources):
    paired=[];aggregate=[]
    for path in discover(paths,"quality",sources):
        sources.use(path);data=read_json(path);args=data.get("args",{})
        key,title,details=condition(args,"not-pooled-for-timing","quality",path)
        records={(r["group"],r["arm"]):r for r in data.get("batches",[]) if r.get("valid") is True}
        # Explicit arms files may reuse names with different budgets. Group by
        # the executed specs as well as CLI defaults, so those runs cannot pool.
        details["executed_arm_specs"]=sorted({canonical({"arm":arm,"spec":r.get("spec")})
                                             for (_,arm),r in records.items()})
        key=canonical(details)
        arms=sorted({arm for _,arm in records if arm not in {"native","dense"}})
        if not arms:
            sources.warn(f"No valid nonidentity quality arm: {path}");continue
        groups_by_arm={arm:{g for g,a in records if a==arm} for arm in ["native","dense",*arms]}
        groups=set.intersection(*groups_by_arm.values())
        if not groups:
            sources.warn(f"No common quality groups with native quality + dense counters + candidate arms: {path}");continue
        if any(groups!=gs for gs in groups_by_arm.values()):
            sources.warn(f"Quality comparison restricted to {len(groups)} groups common to all present arms: {path}")
        valid_groups=[]
        for group in sorted(groups):
            group_rows=[]
            try:
                native=records[(group,"native")];dense=records[(group,"dense")]
                dense_path=artifact_path(path,dense["policy_counter_file"])
                sources.use(dense_path);dc=counter_values(read_json(dense_path))
                dense_bytes=number(dc.get("logical_weight_bytes_after"));dense_tokens=number(dc.get("tokens"))
                if not dense_bytes or dense_bytes<=0 or dense_tokens is None:
                    raise ValueError("missing dense observed footprint/token counters")
                nq=native["quality"];count=int(nq["tokens"])
                if count<=0 or number(nq.get("mean_nll")) is None:raise ValueError("invalid native quality population")
                for arm in ["native",*arms]:
                    record=records[(group,arm)];q=record["quality"]
                    if int(q["tokens"])!=count:raise ValueError(f"quality token denominator mismatch: {arm}")
                    if arm=="native":cp,cc=dense_path,dc
                    else:
                        cp=artifact_path(path,record["policy_counter_file"]);sources.use(cp);cc=counter_values(read_json(cp))
                    actual=number(cc.get("logical_weight_bytes_after"))
                    if actual is None or number(cc.get("tokens"))!=dense_tokens:
                        raise ValueError(f"counter population mismatch: {arm}")
                    if number(q.get("mean_kl")) is None or number(q.get("mean_nll")) is None:
                        raise ValueError(f"missing finite full-token quality metric: {arm}")
                    group_rows.append({"condition":key,"condition_title":title,"run_id":str(path),
                        "run_status":data.get("status","unknown"),"group":group,"batch":int(args["batch"]),
                        "arm":arm,"label":label(arm,record.get("spec")),"spec":canonical(record.get("spec")),
                        "quality_tokens":count,"mean_kl_nats_per_token":float(q["mean_kl"]),
                        "mean_nll":float(q["mean_nll"]),"native_mean_nll":float(nq["mean_nll"]),
                        "logical_weight_bytes_after":actual,"native_dense_logical_weight_bytes_after":dense_bytes,
                        "logical_footprint_saving_fraction":1-actual/dense_bytes,
                        "counter_tokens_across_layers":dense_tokens,"counter_source":str(cp),
                        "native_dense_counter_source":str(dense_path),"quality_source":str(path),
                        "quality_record":f"batches[group={group},arm={arm}].quality",
                        "scope":"same teacher-forced decode windows; full vocabulary KL(native||policy); not free-generation task accuracy",
                        "condition_details":canonical(details)})
                paired.extend(group_rows);valid_groups.append(group)
            except (ValueError,FileNotFoundError,KeyError) as error:
                sources.warn(f"Omitted quality group {path} {group}: {error}")
        for arm in ["native",*arms]:
            rows=[r for r in paired if r["run_id"]==str(path) and r["arm"]==arm]
            if not rows:continue
            tokens=sum(r["quality_tokens"] for r in rows)
            nll=sum(r["mean_nll"]*r["quality_tokens"] for r in rows)/tokens
            bnll=sum(r["native_mean_nll"]*r["quality_tokens"] for r in rows)/tokens
            ppl=math.exp(nll) if nll<709 else None;bppl=math.exp(bnll) if bnll<709 else None
            base_bytes=sum(r["native_dense_logical_weight_bytes_after"] for r in rows)
            after=sum(r["logical_weight_bytes_after"] for r in rows)
            aggregate.append({k:rows[0][k] for k in ("condition","condition_title","run_id","run_status","batch","arm","label","spec","scope")}|{
                "common_groups":len(valid_groups),"quality_tokens":tokens,
                "mean_kl_nats_per_token":sum(r["mean_kl_nats_per_token"]*r["quality_tokens"] for r in rows)/tokens,
                "ppl":ppl,"native_ppl":bppl,"ppl_delta":ppl-bppl if ppl is not None and bppl is not None else None,
                "mean_nll_delta":nll-bnll,"logical_weight_bytes_after":after,
                "native_dense_logical_weight_bytes_after":base_bytes,
                "logical_footprint_saving_fraction":1-after/base_bytes,"quality_source":str(path),
                "groups":canonical(valid_groups)})
    return paired,aggregate


def load_microbench(paths,sources):
    rows=[]
    for path in discover(paths,"microbench",sources):
        sources.use(path);data=read_json(path)
        for index,row in enumerate(data.get("rows",[]),1):
            if "eager_wall_us_per_call" not in row:continue
            gpu=data.get("gpu","unspecified")
            details={"experts":row["experts"],"top_k":row["top_k"],"gpu":gpu,"torch":data.get("torch")}
            if gpu=="unspecified":details["unverified_hardware_run"]=str(path)
            rows.append({"condition":canonical(details),"condition_title":f"E={row['experts']}, k={row['top_k']} | {gpu}",
                "run_id":str(path),"run_status":data.get("status","unknown"),"batch":int(row["batch"]),
                "backend":row["backend"],"eager_wall_us_per_call":number(row.get("eager_wall_us_per_call")),
                "captured_device_us_per_call":number(row.get("captured_device_us_per_call")),
                "within_run_calls":row.get("repetitions"),"calls_per_graph":row.get("calls_per_graph"),
                "source":str(path),"source_record":index,
                "scope":"synthetic routing post-select + counters; captured time excludes Python/allocation/launch dispatch; no MoE/service claim"})
    return rows


def write_csv(path,rows):
    if not rows:return False
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    return True


def axis_batches(ax,bs):
    bs=sorted(set(bs));ax.set_xscale("log",base=2);ax.set_xticks(bs,[str(b) for b in bs])
    if len(bs)==1:ax.set_xlim(bs[0]/1.3,bs[0]*1.3)
    ax.set_xlabel("Decode batch, B");ax.grid(alpha=.2)


def plot_run_points(ax,rows,value,group_key="label",scale=1):
    labels=sorted({r[group_key] for r in rows})
    for i,name in enumerate(labels):
        chosen=[r for r in rows if r[group_key]==name and r[value] is not None]
        if not chosen:continue
        color=f"C{i%10}";batches=sorted({r["batch"] for r in chosen})
        ax.scatter([r["batch"] for r in chosen],[r[value]*scale for r in chosen],s=22,alpha=.5,color=color)
        means=[statistics.mean(r[value]*scale for r in chosen if r["batch"]==b) for b in batches]
        ax.plot(batches,means,marker="o",ms=4,lw=1.5,color=color,label=name)
    axis_batches(ax,[r["batch"] for r in rows]);ax.legend(frameon=False,fontsize=8)


def save_figure(fig,out,stem,caption,generated,captions):
    width,height=fig.get_size_inches()
    title=fig._suptitle
    if title is not None:
        title_text="\n".join(textwrap.wrap(title.get_text(),width=max(40,int(width*12)),break_long_words=False))
        title.set_text(title_text)
        title_lines=len(title_text.splitlines())
    else:title_lines=0
    title_inches=.12+.18*title_lines
    legend_rows=getattr(fig,"_results_legend_rows",0)
    if legend_rows:
        fig.legends[-1].set_bbox_to_anchor((.5,1-title_inches/height))
    top=1-(title_inches+(.06+.18*legend_rows if legend_rows else .04))/height
    wrapped="\n".join("\n".join(textwrap.wrap(line,width=max(35,int(width*14)))) for line in caption.splitlines())
    lines=len(wrapped.splitlines());bottom=min(.35,(.20+.14*lines)/height)
    fig.text(.02,.015,wrapped,fontsize=8,color=".30",va="bottom")
    fig.tight_layout(rect=(0,bottom,1,top))
    for ext in ("pdf","png","svg"):
        path=out/f"{stem}.{ext}";fig.savefig(path,dpi=180);generated.append(path.name)
    captions[stem]=caption


def group_conditions(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row["condition"]].append(row)
    return grouped


def figure_stem(prefix,rows):
    return f"{prefix}_{slug(rows[0]['condition_title'])}_{hashlib.sha256(rows[0]['condition'].encode()).hexdigest()[:8]}"


def status_note(rows):
    states=sorted({r['run_status'] for r in rows if r['run_status'] not in ('PASS','COMPLETE')})
    return " Source runs include partial / unreported status: "+", ".join(states)+"." if states else ""


def render(bench,quality,micro,out,sources,generated,captions):
    if not (bench or quality or micro):return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    for _,rows in group_conditions(bench).items():
        candidates=[r for r in rows if r["arm"]!="native" and not r["diagnostic"]]
        metrics=[(k,title) for k,title in (("tpot_saving_fraction","TPOT reduction vs native (%)"),
                    ("logical_footprint_saving_fraction","Observed logical-footprint reduction (%)")) if any(r[k] is not None for r in candidates)]
        if metrics:
            fig,axes=plt.subplots(1,len(metrics),figsize=(5.5*len(metrics),4.5),squeeze=False)
            for ax,(field,ylabel) in zip(axes[0],metrics):
                usable=[r for r in candidates if r[field] is not None]
                plot_run_points(ax,usable,field,scale=100);ax.axhline(0,color=".5",lw=.7);ax.set_ylabel(ylabel)
            fig.suptitle("F2  "+rows[0]["condition_title"],fontsize=11)
            cap="Matched native within run / repeat / B. Faint points: observed repeats; connected points: their mean.\nFootprint is counted distinct expert weight bytes, not hardware DRAM traffic; eager fixed-length generation."
            cap+=status_note(rows)
            save_figure(fig,out,figure_stem("F2",rows),cap,generated,captions);plt.close(fig)
        energy=[r for r in rows if r["j_per_decode_token"] is not None and not r["diagnostic"]]
        if energy:
            fig,ax=plt.subplots(figsize=(7.7,4.8));names=sorted({r['label'] for r in energy})
            for i,name in enumerate(names):
                selected=[r for r in energy if r['label']==name];long=[r for r in selected if r['energy_steady_window']]
                short=[r for r in selected if not r['energy_steady_window']];color=f"C{i%10}"
                if long:
                    ax.scatter([r['batch'] for r in long],[r['j_per_decode_token'] for r in long],color=color,s=22,alpha=.5)
                    bs=sorted({r['batch'] for r in long})
                    ax.plot(bs,[statistics.mean(r['j_per_decode_token'] for r in long if r['batch']==b) for b in bs],"o-",ms=4,color=color,label=name+" (>=3 s)")
                if short:
                    ax.scatter([r['batch'] for r in short],[r['j_per_decode_token'] for r in short],facecolors="none",edgecolors=color,s=40,label=name+" (short / unknown window)")
            axis_batches(ax,[r['batch'] for r in energy]);ax.set_ylabel("Integrated board energy / decode token (J)")
            ax.legend(frameon=False,fontsize=8);fig.suptitle("F4  "+rows[0]['condition_title'],fontsize=11)
            cap="NVML power-sample trapezoids; window starts after every first token, ends after final decode output.\nHollow points are <3 s / unknown-length diagnostics, excluded from steady-window mean lines; whole-board energy."
            cap+=status_note(rows)
            save_figure(fig,out,figure_stem("F4",rows),cap,generated,captions);plt.close(fig)
    for _,rows in group_conditions(quality).items():
        bs=sorted({r['batch'] for r in rows if r['batch'] in (1,8,32)})
        if not bs:
            sources.warn(f"F3 requested panels B=1/8/32 absent for {rows[0]['condition_title']}");continue
        fig,axes=plt.subplots(2,len(bs),figsize=(5.1*len(bs),7.8),squeeze=False)
        names=sorted({r['label'] for r in rows});colors={n:f"C{i%10}" for i,n in enumerate(names)}
        for col,b in enumerate(bs):
            for line,field in enumerate(("mean_kl_nats_per_token","ppl_delta")):
                ax=axes[line,col];means=[]
                for name in names:
                    chosen=[r for r in rows if r['batch']==b and r['label']==name and r[field] is not None]
                    if not chosen:continue
                    xs=[r['logical_footprint_saving_fraction']*100 for r in chosen];ys=[r[field] for r in chosen]
                    ax.scatter(xs,ys,color=colors[name],alpha=.45,s=25)
                    x,y=statistics.mean(xs),statistics.mean(ys);means.append((x,y))
                    ax.scatter([x],[y],color=colors[name],s=55,label=name,marker="D")
                frontier=sorted((x,y) for x,y in means if not any(x2>=x and y2<=y and (x2>x or y2<y) for x2,y2 in means))
                if len(frontier)>1:ax.plot(*zip(*frontier),ls="--",color=".55",lw=.8)
                ax.set(title=f"B={b}",xlabel="Observed logical-footprint reduction (%)",
                       ylabel="Full-vocabulary KL(native || policy), nats/token" if line==0 else "Selected-window pooled PPL change")
                ax.grid(alpha=.2)
        handles,legend_labels=axes[0,0].get_legend_handles_labels()
        columns=min(3,len(legend_labels))
        fig.legend(handles,legend_labels,loc="upper center",ncol=columns,frameon=False,fontsize=8)
        fig._results_legend_rows=math.ceil(len(legend_labels)/columns)
        missing=sorted({1,8,32}-set(bs))
        fig.suptitle("F3  "+rows[0]['condition_title'],fontsize=11)
        cap="Bytes and quality share teacher-forced groups; all shown arms use their common groups. Dots: run aggregates; diamonds: run means.\nKL/PPL are token-weighted metrics, not token-level repetitions. Dashed: nondominated measured means; not task accuracy."
        if missing:cap+=" Missing B panels: "+", ".join(map(str,missing))+"."
        cap+=status_note(rows)
        save_figure(fig,out,figure_stem("F3",rows),cap,generated,captions);plt.close(fig)
    for _,rows in group_conditions(micro).items():
        metrics=[(key,title) for key,title in (("eager_wall_us_per_call","Eager wall time / policy call (us)"),
               ("captured_device_us_per_call","Captured device time / call (us)")) if any(r[key] is not None for r in rows)]
        if not metrics:continue
        fig,axes=plt.subplots(1,len(metrics),figsize=(5.5*len(metrics),4.5),squeeze=False)
        for ax,(field,title) in zip(axes[0],metrics):
            usable=[r for r in rows if r[field] is not None]
            plot_run_points(ax,usable,field,group_key="backend")
            ax.set(xlabel="Routing matrix rows, B",ylabel=title)
            if usable and all(r[field]>0 for r in usable):ax.set_yscale("log")
        fig.suptitle("F6  "+rows[0]['condition_title'],fontsize=11)
        cap="Measured synthetic routing policy + counters microbenchmark; not MoE/service speedup. Each source execution is one run.\nEager wall time includes host dispatch, launch and device work; it is not pure CPU time. Wall minus device time is not a CPU decomposition.\nCaptured-device timing excludes Python, allocation and individual launch dispatch; inner-loop calls are not independent repeats."
        cap+=status_note(rows)
        save_figure(fig,out,figure_stem("F6",rows),cap,generated,captions);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench",nargs="+",type=Path,default=[])
    p.add_argument("--quality",nargs="+",type=Path,default=[])
    p.add_argument("--microbench",nargs="+",type=Path,default=[])
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--replace-derived",action="store_true",help="replace only this tool's prior derived files in the output directory")
    args=p.parse_args();out=args.output_dir
    if out.exists() and any(out.iterdir()):
        if not args.replace_derived:raise FileExistsError(f"Choose new output or --replace-derived: {out}")
        previous=out/"results_manifest.json"
        if not previous.exists():raise ValueError("refusing replacement without this tool's manifest")
        for name in read_json(previous).get("generated_files",[]):
            path=out/name
            if path.parent.resolve()!=out.resolve():raise ValueError("derived output is not directly inside output directory")
            if path.exists():path.unlink()
    out.mkdir(parents=True,exist_ok=True)
    sources=Sources();generated=[];captions={}
    bench=load_bench(args.bench,sources)
    qgroups,quality=load_quality(args.quality,sources)
    micro=load_microbench(args.microbench,sources)
    for name,rows in (("bench_points.csv",bench),("quality_paired_groups.csv",qgroups),
                      ("quality_run_points.csv",quality),("microbench_run_points.csv",micro)):
        if write_csv(out/name,rows):generated.append(name)
    render(bench,quality,micro,out,sources,generated,captions)
    manifest={"schema_version":1,"status":"DERIVED" if any(n.endswith('.pdf') for n in generated) else "NO_PLOTTABLE_DATA",
        "counts":{"bench_rows":len(bench),"quality_groups":len(qgroups),"quality_run_points":len(quality),"microbench_rows":len(micro)},
        "generated_files":[*generated,"results_manifest.json"],"captions":captions,"sources":list(sources.paths.values()),
        "warnings":sources.warnings,"missing_figures":[f for f in ("F2","F3","F4","F6") if not any(n.startswith(f+'_') and n.endswith('.pdf') for n in generated)],
        "evidence_rules":["No model-estimated performance or fabricated measurements.",
            "No confidence intervals treating tokens, groups, inner-loop calls or layers as independent runs.",
            "Short energy windows are diagnostic only; sampled power integration is primary.",
            "Logical expert weight footprint is not hardware DRAM traffic.",
            "Partial-run points retain source run status in CSV; absent experiments stay absent."]}
    (out/"results_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":manifest["status"],"counts":manifest["counts"],"missing_figures":manifest["missing_figures"],
                      "output_dir":str(out),"warnings":sources.warnings},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
