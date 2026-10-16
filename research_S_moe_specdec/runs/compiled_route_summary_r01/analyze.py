import collections,csv,gc,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/"runs/olmoe_compiled_routes_r01"
out=Path(__file__).resolve().parent
figures=ROOT/"figures/compiled_routes_r01"
all_results=[]; step_rows=[]; layer_rows=[]
def write_csv(path, rows):
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with path.open("w",newline="") as stream:
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
  writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

for path in sorted(base.glob("gamma*/routes-b*.json")):
 digest=hashlib.sha256()
 with path.open("rb") as stream:
  for chunk in iter(lambda:stream.read(8<<20),b""):digest.update(chunk)
 obj=json.loads(path.read_text())
 episodes=obj.get("episodes",[obj])
 errors=[]; modes=collections.Counter(); pairs=collections.Counter(); phases=collections.Counter()
 counts=collections.Counter(); decode=[]; case_steps=[]; padded_examples=[]; mappings=[]; fingerprints=set()
 def check(ok,message):
  if not ok and len(errors)<30:errors.append(message)
 requests=[r for e in episodes for r in e["requests"]]
 check(obj.get("population")==64,"outer population !=64")
 check(len(requests)==64 and len({r["source_index"] for r in requests})==64,"request/source population incomplete")
 check(all(r["finished"] and len(r["output_token_ids"])==32 for r in requests),"output completion/length")
 for ep in episodes:
  p=ep["plugin"]; steps=p["steps"]; valid=p["routes"]; physical=p["graph_padding_routes"]
  mapping=p["native_layer_mapping"]; layers={r["layer"] for r in mapping}; mappings.extend(mapping)
  check(len(mapping)==16 and len(layers)==16,"native layer mapping !=16")
  check({m["native_layer_id"] for m in mapping}==set(range(16)),"native layer IDs")
  check(p["summary"]["trace_complete"] and not p["summary"]["route_errors"],"trace errors: "+repr(p["summary"]["route_errors"][:2]))
  check(len(ep["host_steps"])==len(steps),"host/plugin step alignment")
  counts["episodes"]+=1;counts["steps"]+=len(steps);counts["valid_records"]+=len(valid);counts["physical_records"]+=len(physical)
  vr={(r["step"],r["layer"]):r for r in valid}; pr={(r["step"],r["layer"]):r for r in physical}
  check(len(vr)==len(valid) and len(pr)==len(physical),"duplicate layer record")
  check(len(vr)==16*len(steps) and len(pr)==16*len(steps),"step/layer coverage")
  for s in steps:
   index=s["step"]; S=s["scheduled_tokens"]; P=s.get("target_cudagraph_padded_tokens"); mode=s.get("target_cudagraph_runtime_mode","missing")
   phases[s["phase"]]+=1;modes[(s["phase"],mode)]+=1;pairs[(s["phase"],S,P)]+=1
   check(type(P) is int and P>=S,"invalid S/P")
   if s["phase"]=="decode":
    step_row={"gamma":obj["gamma"],"batch":obj["batch"],"episode":ep["label"],"step":index,
      "actual_decode_batch":s["decode_batch"],"step_gamma":s["gamma"],"S":S,"P":P,"padding_rows":P-S,
      "mode":mode,"scheduled_drafts":s["scheduled_drafts"]}
    case_steps.append(step_row);step_rows.append(step_row)
   check(s.get("native_route_rows")==S and s.get("native_physical_route_rows")==P,"native boundary row mismatch")
   check({l for i,l in vr if i==index}==layers and {l for i,l in pr if i==index}==layers,"missing layer")
   if P>S:counts[s["phase"]+"_padded_steps"]+=1
   for layer in layers:
    key=(index,layer)
    if key not in vr or key not in pr:continue
    v,h=vr[key],pr[key];a=v["expert_token_counts"];b=h["padding_expert_token_counts"];c=h["physical_expert_token_counts"];k=v["top_k"]
    check(v["tokens"]==S and len(a)==64 and k==8,"valid dimensions")
    check(sum(a)==S*k and max(a)<=S and not v["invalid_expert_assignments"],"valid histogram conservation")
    ids=v.get("topk_ids")
    check(ids is not None and len(ids)==S,"exact valid token routes missing")
    if ids is not None:
     histogram=collections.Counter(e for row in ids for e in row)
     check(all(len(row)==8 and len(set(row))==8 and all(0<=e<64 for e in row) for row in ids),"exact topk population")
     check([histogram[i] for i in range(64)]==a,"exact topk/hist mismatch")
    check(h["complete"] and h["scheduled_tokens"]==S and h["physical_tokens"]==P and h["padding_tokens"]==P-S,"physical dimensions/completeness")
    check(len(b)==len(c)==64 and sum(b)==(P-S)*k and sum(c)==P*k,"physical histogram conservation")
    check(max(b)<=P-S and not h["invalid_padding_expert_assignments"],"padding invalid/repeated")
    check(c==[x+y for x,y in zip(a,b)],"physical=valid+pad")
    new=[i for i in range(64) if a[i]==0 and b[i]>0]
    check(h["padding_only_new_expert_ids"]==new and h["padding_only_new_experts"]==len(new),"new expert set")
    dv=sum(x>0 for x in a);dp=sum(x>0 for x in c)
    check(v["distinct_experts"]==dv and h["physical_distinct_experts"]==dp and dp-dv==len(new),"D arithmetic")
    check(h["target_cudagraph_runtime_mode"]==mode,"mode mismatch")
    if s["phase"]=="decode":
     decode.append((S,P,dv,dp,len(new)))
     layer_rows.append({"gamma":obj["gamma"],"batch":obj["batch"],"episode":ep["label"],"step":index,
      "layer":layer,"native_layer_id":h["native_layer_id"],"actual_decode_batch":s["decode_batch"],"step_gamma":s["gamma"],
      "S":S,"P":P,"valid_D":dv,"physical_D":dp,"padding_only_new_experts":len(new),"mode":mode})
     if P>S and len(new)>0 and len(padded_examples)<3:
      padded_examples.append({"episode":ep["label"],"step":index,"layer":layer,"S":S,"P":P,"valid_D":dv,"physical_D":dp,"new_experts":new,"mode":mode})
  fingerprints.add(json.dumps(mapping,sort_keys=True))
 check(len(fingerprints)==1,"layer mapping changed across episodes")
 padded=[r for r in decode if r[1]>r[0]]
 result={"file":str(path.relative_to(base)),"sha256":digest.hexdigest(),"population":len(requests),"unique_sources":len({r["source_index"] for r in requests}),
 "all_32_outputs":all(len(r["output_token_ids"])==32 for r in requests),"counts":dict(counts),
 "phase_steps":dict(phases),"runtime_modes":[{"phase":k[0],"mode":k[1],"steps":v} for k,v in sorted(modes.items())],
 "decode_S_range":[min(r[0] for r in decode),max(r[0] for r in decode)],"decode_P_range":[min(r[1] for r in decode),max(r[1] for r in decode)],
 "decode_valid_D_mean":statistics.mean(r[2] for r in decode),"decode_physical_D_mean":statistics.mean(r[3] for r in decode),
 "decode_layer_rows":len(decode),"padded_decode_layer_rows":len(padded),"padding_adds_experts_rows":sum(r[4]>0 for r in decode),
 "new_D_mean_all_decode":statistics.mean(r[4] for r in decode),"new_D_mean_padded_decode":statistics.mean(r[4] for r in padded) if padded else 0,"new_D_max":max(r[4] for r in decode),
 "S_P_counts_decode":[{"S":k[1],"P":k[2],"steps":v} for k,v in sorted(pairs.items()) if k[0]=="decode"],
 "mapping_bytes_per_expert":sorted({m["bytes_per_expert"] for m in mappings}),"bytes_basis":sorted({m["bytes_basis"] for m in mappings}),
 "errors":errors,"padded_examples":padded_examples}
 result.update({
  "gamma":obj["gamma"],"batch":obj["batch"],"engine_max_num_seqs":obj["engine_max_num_seqs"],
  "trace_complete":not errors,"validation_error_count":len(errors),
  "decode_S_mean":statistics.mean(r[0] for r in decode),
  "decode_P_mean":statistics.mean(r[1] for r in decode),
  "decode_padding_rows_mean":statistics.mean(r[1]-r[0] for r in decode),
  "decode_actual_B_range":[min(r["actual_decode_batch"] for r in case_steps),max(r["actual_decode_batch"] for r in case_steps)],
  "decode_actual_B_mean":statistics.mean(r["actual_decode_batch"] for r in case_steps),
  "decode_step_gamma_counts":dict(sorted(collections.Counter(r["step_gamma"] for r in case_steps).items())),
  "decode_runtime_mode_counts":dict(sorted(collections.Counter(r["mode"] for r in case_steps).items())),
  "decode_padded_step_fraction":counts["decode_padded_steps"]/len(case_steps),
  "decode_physical_to_valid_D_ratio_of_means":statistics.mean(r[3] for r in decode)/statistics.mean(r[2] for r in decode),
  "decode_P_to_S_ratio_of_means":statistics.mean(r[1] for r in decode)/statistics.mean(r[0] for r in decode),
  "native_layer_mapping":json.loads(next(iter(fingerprints)))
 })
 all_results.append(result)
 (out/f"gamma{obj['gamma']}_b{obj['batch']}.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({k:result[k] for k in ["gamma","batch","trace_complete","decode_S_mean","decode_P_mean","decode_valid_D_mean","decode_physical_D_mean","decode_runtime_mode_counts"]}),flush=True)
 del obj,episodes,requests,decode,padded,valid,physical,vr,pr,p,steps,mapping,mappings
 gc.collect()

boundaries=[
 "Instrumented route diagnostic, not performance data. Each case uses 64 unique requests with 32 output tokens, wave admission, engine max_num_seqs=32, 16 MoE layers.",
 "D means weight every decode layer-step equally. S/P means use the same weighting (16 layers per step); shrinking active batches near completion are retained. Prefill is excluded from plotted means but included in integrity checks.",
 "Steps, layers and waves within one condition are correlated observations, not independent system repeats. No confidence interval is estimated.",
 "S covers every scheduled verification row, including subsequently rejected draft rows. P covers every model-forward row including CUDA Graph padding; it excludes expert GEMM tile padding.",
 "Physical D denotes the union of experts selected by all P model-forward rows. Selected parameter storage is not measured HBM traffic; padding expert increments alone do not establish a wall-time penalty.",
 "Runtime mode comes from the actual forward_context dispatch. Graph capture versus replay was not individually instrumented.",
 "enable_return_routed_experts adds native writes, D2H and CPU slot storage. The plugin adds GPU histogram/copies and deferred drain. Do not use diagnostic timing as trace-off performance.",
 "For this OLMoE configuration both route-on and route-off default to V1. The native flag and plugin do not explicitly alter its graph dispatch or capture sizes. On other default-V2 architectures the native flag can instead force V1.",
 "V1 FULL decode graph configuration rounds shared capture sizes up to multiples of gamma+1. Mixed/piecewise dispatch reuses the same table, so gamma=4 and S=1 can dispatch PIECEWISE with P=5.",
 "Applying P rules to a trace-off run requires the same resolved runner, backend, graph sizes, gamma, and batch structure. This diagnostic uses max_num_seqs=32; the planned performance matrix uses 64. Neither physical D nor time transfers without matched evidence."
]
flat=[]
for r in all_results:
 flat.append({k:v for k,v in r.items() if k not in ("native_layer_mapping","padded_examples","S_P_counts_decode","runtime_modes","bytes_basis")})
source_notes={
 "remote_sources_read_only":[
  {"path":"/root/miniconda3/lib/python3.12/site-packages/vllm/config/compilation.py","sha256":"267f4ca04fdcd2f0cf55c334782dd498c8051b9500ad088673d79518fc712db0","lines":"1459-1474,1503-1548","finding":"V1 FULL decode rounds shared capture sizes to gamma+1 multiples"},
  {"path":"/root/miniconda3/lib/python3.12/site-packages/vllm/v1/cudagraph_dispatcher.py","sha256":"51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63","lines":"82-91,132-155,166-231,302-318","finding":"S maps to next capture P; uniform full key preferred, relaxed piecewise key otherwise"}],
 "local_sources":[
  {"path":"runs/environment_20261016/source/vllm/config/vllm.py","lines":"68-77,550-622,819-835,955-975,2188-2190","finding":"OLMoE defaults V1; route flag can disable default V2 elsewhere; dynamic-SD graph override is unrelated to this fixed ngram config"},
  {"path":"runs/environment_20261016/source/v1/worker/gpu_model_runner.py","lines":"858,3857-3975,4059-4075,4346-4398,7115-7128,7637","finding":"uniform decode length gamma+1, graph descriptor P passed to preprocessing and forward; padding slot -1 masks KV writes, not MoE rows"},
  {"path":"runs/olmoe_compiled_routes_r01/source/vllm_s_plugin.py","lines":"416-458,806-845","finding":"reads existing forward mode/P and captures buffer after target forward; does not change graph keys"}
 ]
}
for r in source_notes["local_sources"]:
 r["sha256"]=hashlib.sha256((ROOT/r["path"]).read_bytes()).hexdigest()
report={"input_run":str(base.relative_to(ROOT)),"analysis_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 "aggregation":"equal decode layer-step weights; no independent confidence intervals",
 "validation_all_passed":all(not r["errors"] for r in all_results),"boundaries":boundaries,"source_notes":source_notes,"cases":all_results}
(out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
write_csv(out/"summary.csv",flat)
write_csv(out/"decode_steps.csv",step_rows)
write_csv(out/"decode_layer_steps.csv",layer_rows)
write_csv(out/"plot_data.csv",[{k:r[k] for k in ("gamma","batch","decode_S_mean","decode_P_mean","decode_valid_D_mean","decode_physical_D_mean","new_D_mean_all_decode")} for r in all_results])
(out/"README.txt").write_text("Reproduce: python3 research_S_moe_specdec/runs/compiled_route_summary_r01/analyze.py\n\n"+"\n\n".join(boundaries)+"\n")
if not report["validation_all_passed"]:raise RuntimeError("Raw trace validation failed")
# Reuse the existing local PGFPlots/Poppler helper; this performs no GPU work.
sys.path.insert(0,str(ROOT/"code"))
from plot_energy_progress import render_pgf
from plot_kernel_diagnostics import document,coords
figures.mkdir(parents=True,exist_ok=True)
cases=sorted(all_results,key=lambda r:(r["gamma"],r["batch"]))
def panel(title,ylabel,first,second,ymax):
 positions=[(i+1,r) for i,r in enumerate(cases)]
 ax=r"\nextgroupplot[title={%s},ylabel={%s},ymin=0,ymax=%s,xmin=0.4,xmax=6.6,xtick={1,2,3,4,5,6},xticklabels={{0/1},{0/8},{0/32},{4/1},{4/8},{4/32}},xlabel={Nominal $\\gamma$ / wave batch $B$},ybar,bar width=10pt,legend pos=north west]"%(title,ylabel,ymax)
 ax+=r"\addplot[fill=cal!75,draw=cal] coordinates {%s};\addlegendentry{Valid rows}"%coords([(i,r[first]) for i,r in positions])
 ax+=r"\addplot[fill=test!75,draw=test] coordinates {%s};\addlegendentry{Physical rows}"%coords([(i,r[second]) for i,r in positions])
 return ax
axes=[panel("(a) Selected distinct experts","Mean distinct experts", "decode_valid_D_mean","decode_physical_D_mean",68),
      panel("(b) Model-forward rows","Mean token rows","decode_S_mean","decode_P_mean",52)]
note=r"Decode layer-step means; 64 requests $\\times$ 32 output tokens per case, 16 layers. Active batch can shrink below $B$.\\\\Instrumented route diagnostic, not performance data; correlated steps, no independent confidence intervals."
tex=document(axes,"OLMoE compiled route diagnostic: valid versus physical work",note)
# Collapse string escapes introduced by constructing this standalone script.
tex=tex.replace("\\\\gamma","\\gamma").replace("\\\\times","\\times")
files=render_pgf(tex,figures,"compiled_routes",["pdf","png","svg"])
(figures/"provenance.json").write_text(json.dumps({"source_summary":str((out/"summary.json").relative_to(ROOT)),"files":files,"aggregation":report["aggregation"],"boundaries":boundaries},indent=2)+"\n")
print("Artifacts:",str(out),str(figures))

