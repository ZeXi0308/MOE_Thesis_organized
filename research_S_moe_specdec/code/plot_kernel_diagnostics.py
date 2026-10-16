#!/usr/bin/env python3
"""Add held-out simple-model, tile and timing-boundary diagnostics to a kernel fit.

  python code/plot_kernel_diagnostics.py --fit-dir runs/kernel_cost_fit_r03 \
    --figures figures/kernel_cost_r03 --legacy-profile-no-turnover

Reads the SHA-verified measurements referenced by fit_kernel_cost.py's report;
no profiler trace re-parsing is needed. Pure-D, pure-S and constant comparisons
fit the same calibration snapshots by nonnegative least squares. Wall and
single-call profiler targets receive separate diagnostic fits, never pooled
with event intervals. All test outcomes remain excluded from fitting.

The legacy flag records a separately source-verified fact about r03: the
profiler call did not repeat the unrelated-buffer cache turnover. Otherwise
missing profile-cache metadata stays unknown. The three timing scopes must
not be subtracted as a causal decomposition. Neither a nor c is a measured HBM
bandwidth or full-model fixed cost. Plots use local PGFPlots and Poppler.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from fit_kernel_cost import (FIT_BATCHES, TEST_BATCHES, active_term, error_metrics,
    fit_domain, make_ratio_grid, nonnegative_line, prediction)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def metric_rows(rows, parameters, model, metric):
    output = []
    for row in rows:
        value = prediction(parameters, row["distinct_experts"], row["tokens"])
        error = value - row["observed_ms"]
        output.append({k: row[k] for k in ("domain_id", "case_id", "source_file", "source_line", "batch", "gamma", "tokens", "distinct_experts", "tile_id", "observed_ms")}
            | {"metric": metric, "model": model, "split": "calibration" if row["batch"] in FIT_BATCHES else "test" if row["batch"] in TEST_BATCHES else "outside_split",
               "predicted_ms": value, "signed_error_ms": error, "absolute_error_ms": abs(error),
               "relative_error": error/row["observed_ms"], "active_term": active_term(parameters,row["distinct_experts"],row["tokens"])})
    return output


def fit_simple_models(rows, max_parameters, metric):
    train = [r for r in rows if r["batch"] in FIT_BATCHES]
    y = [r["observed_ms"] for r in train]
    parameters = {"max_D_S_plus_c": max_parameters}
    for name, field in (("D_plus_c", "distinct_experts"), ("S_plus_c", "tokens")):
        slope, intercept, loss = nonnegative_line([r[field] for r in train], y)
        parameters[name] = {"a_ms_per_distinct_expert": slope if field == "distinct_experts" else 0,
                            "b_ms_per_token_slot": slope if field == "tokens" else 0,
                            "c_ms": intercept, "fit_sse_ms2": loss}
    parameters["constant"] = {"a_ms_per_distinct_expert": 0,"b_ms_per_token_slot": 0,"c_ms": statistics.mean(y)}
    summaries, predictions = [], []
    for model, p in parameters.items():
        predicted = metric_rows(rows, p, model, metric)
        predictions.extend(predicted)
        for split in ("calibration", "test"):
            metrics = error_metrics([r for r in predicted if r["split"] == split])
            summaries.append({"domain_id": rows[0]["domain_id"], "metric": metric, "model": model,
                "split": split, **{k:p[k] for k in ("a_ms_per_distinct_expert","b_ms_per_token_slot","c_ms")}, **metrics})
    return summaries, predictions


def describe(values):
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)} if values else {"count":0}


def analyze(fit_dir, legacy_no_turnover=False):
    report_path=fit_dir/"kernel_cost_fit.json"
    report=json.loads(report_path.read_text())
    raw={}
    for entry in report["input_files"]:
        path=Path(entry["path"]);content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=entry["sha256"]:
            raise ValueError(f"input changed after fitting: {path}")
        for index,line in enumerate(content.decode().splitlines(),1):
            if line.strip(): raw[(str(path),index)]=json.loads(line)
    observations=[json.loads(line) for line in (fit_dir/"kernel_cost_observations.jsonl").read_text().splitlines() if line]
    grouped=defaultdict(list)
    for row in observations:
        row=dict(row);r=raw[(row["source_file"],row["source_line"])]
        if r.get("case_id")!=row["case_id"]: raise ValueError("case identity mismatch")
        row["raw"]=r
        grouped[row["domain_id"]].append(row)
    domains={};comparisons=[];predictions=[];timings=[];residuals=[];tiles=[];profiles=[]
    for domain_id,rows in sorted(grouped.items()):
        primary=report["domains"][domain_id]
        params=primary["parameters"]
        tile_values={json.dumps(r["raw"].get("resolved_tile_configuration",{}).get("parameters",{}),sort_keys=True) for r in rows}
        def tile_sort(value):
            p=json.loads(value)
            return tuple(p.get(k,0) for k in ("BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","num_warps","num_stages"))
        tile_ids={value:f"T{i+1}" for i,value in enumerate(sorted(tile_values,key=tile_sort))}
        for value,tile_id in tile_ids.items():
            p=json.loads(value)
            tiles.append({"domain_id":domain_id,"tile_id":tile_id,"parameters":p,
                "label":"M%d N%d K%d W%d st%d" % tuple(p.get(k,0) for k in ("BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","num_warps","num_stages")),
                "boundary":"resolved host configuration, not a measured physical padding counter"})
        for row in rows:
            r=row["raw"]; profile=r.get("profile",{})
            row["tile_id"]=tile_ids[json.dumps(r.get("resolved_tile_configuration",{}).get("parameters",{}),sort_keys=True)]
            turn=profile.get("cache_turnover_before_profile")
            if turn is None and legacy_no_turnover: turn=False
            event=float(r["cuda_event_ms_median"]);wall=float(r["wall_ms_median"])
            prof=float(profile["kernel_duration_sum_us"])/1000 if profile.get("kernel_duration_sum_us") is not None else None
            timings.append({"domain_id":domain_id,"case_id":row["case_id"],"batch":row["batch"],"gamma":row["gamma"],
                "tokens":row["tokens"],"distinct_experts":row["distinct_experts"],"tile_id":row["tile_id"],
                "event_ms":event,"wall_ms":wall,"profile_kernel_sum_ms":prof,"wall_over_event":wall/event,
                "profile_over_event":prof/event if prof is not None else None,
                "profile_kernel_count":profile.get("kernel_count"),"profile_trace_file":profile.get("trace_file"),
                "execution_mode":r.get("execution_mode","eager"),"timing_cache_policy":r.get("cache_policy"),
                "cache_turnover_before_profile":turn,"profile_timing_scope":profile.get("timing_scope"),
                "comparison_boundary":"separate calls and timing scopes; differences are not an additive causal decomposition"})
        targets={"cuda_event_ms_median":rows}
        targets["wall_ms_median"]=[{**r,"observed_ms":float(r["raw"]["wall_ms_median"])} for r in rows]
        profiled=[{**r,"observed_ms":float(r["raw"]["profile"]["kernel_duration_sum_us"])/1000} for r in rows
                  if r["raw"].get("profile",{}).get("kernel_duration_sum_us",0)>0]
        if profiled: targets["profile_kernel_sum_ms"]=profiled
        fits={}
        for metric,target in targets.items():
            fit,_=fit_domain(target)
            if metric=="cuda_event_ms_median" and fit["parameters"]!=params:
                raise ValueError("recomputed event coefficients differ from preserved primary fit")
            fits[metric]=fit
            if fit["parameters"]:
                summary,pred=fit_simple_models(target,fit["parameters"],metric)
                comparisons.extend(summary);predictions.extend(pred)
        train=[r for r in rows if r["batch"] in FIT_BATCHES]
        ratios=make_ratio_grid(train,801)
        # The fitted grid starts at the first observed D/S breakpoint. Extend
        # only the diagnostic curve into the equivalent all-D region, so the
        # non-identifiability is visible without changing any fit or selection.
        first_breakpoint=min(r["distinct_experts"]/r["tokens"] for r in train)
        ratios=sorted(set(ratios)|{first_breakpoint*f for f in (.01,.02,.05,.1,.2,.5,.75)})
        for ratio in ratios:
            slope,c,loss=nonnegative_line([max(r["distinct_experts"],ratio*r["tokens"]) for r in train],[r["observed_ms"] for r in train])
            profiles.append({"domain_id":domain_id,"r_b_over_a":ratio,"fit_sse_ms2":loss,"a_ms_per_expert":slope,"b_ms_per_slot":slope*ratio,"c_ms":c})
        selected=[r for r in predictions if r["domain_id"]==domain_id and r["metric"]=="cuda_event_ms_median" and r["model"]=="max_D_S_plus_c"]
        for split in ("calibration","test"):
            for tile_id in tile_ids.values():
                group=[r for r in selected if r["split"]==split and r["tile_id"]==tile_id]
                if not group: continue
                residuals.append({"domain_id":domain_id,"split":split,"tile_id":tile_id,
                    "actual_S_min":min(r["tokens"] for r in group),"actual_S_max":max(r["tokens"] for r in group),
                    "actual_D_min":min(r["distinct_experts"] for r in group),"actual_D_max":max(r["distinct_experts"] for r in group),**error_metrics(group)})
        iden={"calibration_active_terms":params["calibration_active_terms"],
              "test_active_terms":dict(Counter(r["active_term"] for r in selected if r["split"]=="test")),
              "interpretation":"max form may select a boundary; inactive coefficients are not identified by this data"}
        if params["candidate_kind"]=="pure_D":
            rmax=min(r["distinct_experts"]/r["tokens"] for r in train)
            iden.update(inactive_b_interval_ms_per_slot=[0,params["a_ms_per_distinct_expert"]*rmax],
                        flat_r_interval=[0,rmax],
                        interval_boundary="holding fitted a and c fixed, every b in this interval gives identical calibration predictions; not a confidence interval")
        t=[r for r in timings if r["domain_id"]==domain_id]
        domains[domain_id]={"domain":primary["domain"],"snapshot_count":len(rows),
            "actual_S":describe([r["tokens"] for r in rows]),"actual_D":describe([r["distinct_experts"] for r in rows]),
            "fit_count":len(train),"test_count":sum(r["batch"] in TEST_BATCHES for r in rows),
            "resolved_tile_count":len(tile_ids),"identification":iden,"fits_by_timing_scope":fits,
            "timing_descriptions":{key:describe([r[key] for r in t if r[key] is not None]) for key in ("event_ms","wall_ms","profile_kernel_sum_ms","wall_over_event","profile_over_event")},
            "profile_cache_metadata_counts":dict(Counter(str(r["cache_turnover_before_profile"]) for r in t)),
            "profile_target_boundary":"one separately profiled call per case; not the five-repeat timing median and not an independent system repeat"}
    diagnostics={"primary_fit_file":str(report_path.resolve()),"primary_fit_sha256":hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "fit_batches":list(FIT_BATCHES),"test_batches":list(TEST_BATCHES),"domains":domains,
        "model_comparison_rule":"all models fit the same calibration snapshots, equal weights, squared-error objective; holdout is untouched",
        "legacy_profile_no_turnover_assertion":legacy_no_turnover,
        "boundaries":report["boundaries"]+[
            "Profiler sums come from separate calls with their own cache metadata; event/profile gaps cannot isolate host launch time. The legacy r03 profiler did not repeat turnover when explicitly asserted by the analysis flag.",
            "Tile residual summaries are descriptive, not a causal attribution or a newly selected kernel configuration.",
            "An inactive S branch does not establish zero compute cost; the fitted intercept is for this layer/timing boundary, not full-model c0."],
        "analysis_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return diagnostics,comparisons,predictions,timings,residuals,tiles,profiles


def coords(points):
    return " ".join(f"({x:.12g},{y:.12g})" for x,y in points if x is not None and y is not None and math.isfinite(x) and math.isfinite(y))


def document(axes,title,note):
    from plot_energy_progress import tex_escape
    return "\n".join([r"\documentclass[border=5pt]{standalone}",r"\usepackage[T1]{fontenc}",r"\usepackage{lmodern}",
        r"\usepackage{pgfplots}",r"\usepgfplotslibrary{groupplots}",r"\pgfplotsset{compat=1.18}",
        r"\begin{document}\sffamily\begin{tikzpicture}",
        r"\definecolor{cal}{HTML}{3478A5}\definecolor{test}{HTML}{D76832}\definecolor{profile}{HTML}{3B8A5A}",
        r"\begin{groupplot}[group style={group size=%d by 1,horizontal sep=1.9cm},width=8.0cm,height=6.0cm,scale only axis,axis lines*=left,grid=major,grid style={gray!20},tick label style={font=\small},label style={font=\small},title style={font=\small},legend style={font=\scriptsize,draw=none},unbounded coords=jump]" % len(axes),
        *axes,r"\end{groupplot}",
        r"\path (group c1r1.north west) -- node[above=0.8cm,font=\large] {%s} (group c%dr1.north east);"%(tex_escape(title),len(axes)),
        r"\path (group c1r1.south west) -- node[below=1.25cm,align=center,font=\small] {%s} (group c%dr1.south east);"%(note,len(axes)),
        r"\end{tikzpicture}\end{document}"])+"\n"


def scatter(rows,x,y,split_labels=True):
    output=[]
    for split,color,mark in (("calibration","cal","o"),("test","test","triangle*")):
        points=[(x(r),y(r)) for r in rows if r["split"]==split]
        output.append(r"\addplot[only marks,color=%s,mark=%s,mark size=1.8pt,opacity=0.75] coordinates {%s};"%(color,mark,coords(points)))
        if split_labels: output.append(r"\addlegendentry{"+("Fit B=1,4,16,64" if split=="calibration" else "Held out B=2,8,32")+"}")
    return "\n".join(output)


def render(figures,diagnostics,comparisons,predictions,timings,residuals,tiles,profiles):
    from plot_energy_progress import render_pgf,tex_escape
    figures.mkdir(parents=True,exist_ok=True);files=[]
    for domain_id,domain in diagnostics["domains"].items():
        rows=[r for r in predictions if r["domain_id"]==domain_id and r["metric"]=="cuda_event_ms_median" and r["model"]=="max_D_S_plus_c"]
        if not rows: continue
        tag=domain_id[:8];mode=domain["domain"].get("execution_mode","eager")
        maximum=1.05*max(max(r["predicted_ms"],r["observed_ms"]) for r in rows)
        axes=[r"\nextgroupplot[title={(a) Event-interval prediction},xlabel={Measured event interval (ms)},ylabel={Predicted interval (ms)},xmin=0,xmax=%s,ymin=0,ymax=%s,legend pos=north west]"%(maximum,maximum)+
              scatter(rows,lambda r:r["observed_ms"],lambda r:r["predicted_ms"])+r"\addplot[gray,dashed,forget plot,domain=0:%s] {x};"%maximum,
              r"\nextgroupplot[title={(b) Held-out simple-model comparison},ylabel={Mean absolute percentage error (\%)},ybar,bar width=16pt,xtick={1,2,3,4},xticklabels={Max,D+c,S+c,Constant},ymin=0,xmin=0.5,xmax=4.5,nodes near coords,point meta=y,every node near coord/.append style={font=\small,/pgf/number format/fixed,/pgf/number format/precision=2}]"+
              r"\addplot[fill=test!70,draw=test] coordinates {%s};"%coords([(i+1,next(r["mape_percent"] for r in comparisons if r["domain_id"]==domain_id and r["metric"]=="cuda_event_ms_median" and r["split"]=="test" and r["model"]==name)) for i,name in enumerate(("max_D_S_plus_c","D_plus_c","S_plus_c","constant"))])]
        selected_kind=domain["fits_by_timing_scope"]["cuda_event_ms_median"]["parameters"]["candidate_kind"]
        boundary_note=(r"Max$(aD,bS)+c$ selects the pure-$D$ boundary here. " if selected_kind=="pure_D" else "")
        note=r"Each point is one route snapshot's median of five within-process repeat means; no independent-service confidence interval.\\"+boundary_note+r"Coefficients are local effective costs, not measured HBM bandwidth."
        files+=render_pgf(document(axes,f"OLMoE layer cost: {mode} held-out validation",note),figures,f"kernel_cost_validation_{tag}",["pdf","svg","png"])
        tile_rows=[r for r in tiles if r["domain_id"]==domain_id]
        ids=[r["tile_id"] for r in tile_rows]
        axes=[r"\nextgroupplot[title={(a) Residual versus actual token slots},xlabel={Actual token slots $S$},ylabel={Prediction error (\%)},xmode=log,log basis x=2,xtick={1,2,4,8,16,32,64,128},xticklabels={1,2,4,8,16,32,64,128},legend pos=south west]"+
              scatter(rows,lambda r:r["tokens"],lambda r:100*r["relative_error"])+r"\addplot[gray,dashed,forget plot,domain=1:%s] {0};"%max(r["tokens"] for r in rows),
              r"\nextgroupplot[title={(b) Residual versus resolved tile},xlabel={Resolved host tile configuration},ylabel={Prediction error (\%%)},xtick={%s},xticklabels={%s},xmin=0.6,xmax=%s]"%(",".join(str(i+1) for i in range(len(ids))),",".join(ids),len(ids)+.4)+
              scatter(rows,lambda r:ids.index(r["tile_id"])+1+(.08 if r["split"]=="test" else -.08),lambda r:100*r["relative_error"],False)+r"\addplot[gray,dashed,forget plot,domain=1:%s] {0};"%len(ids)]
        short=[tex_escape(r["tile_id"]+": "+r["label"]) for r in tile_rows]
        note="; ".join(short[:3])+r"\\"+"; ".join(short[3:])+r". Host tile parameters are not physical padding counters; residual groups are descriptive."
        files+=render_pgf(document(axes,f"OLMoE layer: {mode} event-fit residuals",note),figures,f"kernel_cost_residuals_{tag}",["pdf","svg","png"])
        timing=[r for r in timings if r["domain_id"]==domain_id]
        first=r"\nextgroupplot[title={(a) Distinct timing scopes},xlabel={Measured distinct experts $D$},ylabel={Time (ms)},legend pos=north west]"
        cache_policy=domain["domain"].get("cache_policy","unknown cache policy")
        for key,label,color,mark in (("event_ms",f"Event interval, {cache_policy}","cal","o"),("wall_ms",f"Wall interval, {cache_policy}","test","triangle"),("profile_kernel_sum_ms","Separate profiler kernel sum","profile","x")):
            first+=r"\addplot[only marks,color=%s,mark=%s,opacity=0.6,mark size=1.7pt] coordinates {%s};\addlegendentry{%s}"%(color,mark,coords([(r["distinct_experts"],r[key]) for r in timing]),label)
        second=r"\nextgroupplot[title={(b) Descriptive profiler/event ratio},xlabel={Measured distinct experts $D$},ylabel={Profiler kernel sum / event interval},ymin=0,ymax=1]"+r"\addplot[only marks,profile,mark=o,mark size=1.7pt,opacity=0.7] coordinates {%s};"%coords([(r["distinct_experts"],r["profile_over_event"]) for r in timing])
        no_turnover=domain["profile_cache_metadata_counts"].get("False",0)
        cache_note=("Profiler call did not repeat cache turnover." if no_turnover else "Profile cache handling is recorded separately in the diagnostics.")
        cache_bytes=domain["domain"].get("cache_turnover_bytes")
        timing_note=(f" Event/wall use per-call turnover of a {cache_bytes/(1024**2):g} MiB unrelated buffer outside timers." if cache_policy=="turnover" and cache_bytes else f" Event/wall cache policy: {cache_policy}.")
        note=tex_escape(cache_note+timing_note)+r"\\Separate calls and cache states prevent interpreting timing differences as additive CPU launch overhead or HBM traffic."
        files+=render_pgf(document([first,second],f"OLMoE layer: {mode} timing boundaries",note),figures,f"kernel_timing_scopes_{tag}",["pdf","svg","png"])
        curve=[r for r in profiles if r["domain_id"]==domain_id and r["r_b_over_a"]>0]
        axis=(r"\nextgroupplot[title={Calibration loss profile},xlabel={Ratio $r=b/a$ (log scale)},ylabel={Calibration squared error (ms$^2$)},xmode=log,log basis x=10]"+
             r"\addplot[cal,no marks,line width=1pt] coordinates {%s};"%coords([(r["r_b_over_a"],r["fit_sse_ms2"]) for r in curve]))
        interval=domain["identification"].get("flat_r_interval")
        note=(r"For $0\leq r\leq %.4f$, all calibration points retain the $D$ branch with the same fitted predictions.\\The selected $b=0$ is a boundary tie preference, not evidence of zero compute cost; this is not a confidence interval."%interval[1]) if interval else r"Only calibration outcomes enter this profile; active branches determine which parameters can be identified."
        files+=render_pgf(document([axis],"Kernel model parameter identification",note),figures,f"kernel_parameter_identification_{tag}",["pdf","svg","png"])
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-dir",type=Path,required=True)
    parser.add_argument("--figures",type=Path)
    parser.add_argument("--legacy-profile-no-turnover",action="store_true")
    args=parser.parse_args()
    data=analyze(args.fit_dir,args.legacy_profile_no_turnover)
    diagnostics,comparisons,predictions,timings,residuals,tiles,profiles=data
    for name,rows in (("model_comparison.csv",comparisons),("diagnostic_predictions.csv",predictions),("timing_scopes.csv",timings),
                      ("residuals_by_tile.csv",residuals),("resolved_tile_configurations.csv",tiles),("calibration_loss_profile.csv",profiles)):
        write_csv(args.fit_dir/name,rows)
    diagnostics["figures"]=render(args.figures,*data) if args.figures else []
    (args.fit_dir/"kernel_diagnostics.json").write_text(json.dumps(diagnostics,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"fit_dir":str(args.fit_dir),"domains":len(diagnostics["domains"]),"figure_artifacts":len(diagnostics["figures"])}))


if __name__=="__main__":
    main()
