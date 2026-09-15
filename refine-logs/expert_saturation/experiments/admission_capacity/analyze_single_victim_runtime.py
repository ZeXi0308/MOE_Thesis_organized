#!/usr/bin/env python3
"""Analyze six native runs of one frozen victim replacement; never synthesize it."""
import argparse, copy, hashlib, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True

FUNDING_SHA="8137eebffa6e42567f3cc43dda79c414061bbfcc219e78d44dd69cebcf800b7b"
PINNED={
 "absence_rotation.py":"515362bb4f7923637bfcfedd3e2fcc22db6e6c2bb5805a83c6e8f1d765db4000",
 "rotation_native.py":"b6088577fe88e056a7b2072391a33aa3a19200da16349faa5bb1e4913cec0b86",
 "single_victim_event.py":"d7ad18b3fe3a25695343f0cd0d4eef361d8ccaabcc63cea9255b4a05eaac45e6",
 "event_contract.json":"d70105a51c833fa41ec7fc81c09ecb54e289da59d6ad1c355a238199baff5641",
 "run_probe.py":"ef77ab155e757345097434315c663794fd8e00b47f66e71bd29549f66063a251"}
VARIANTS=("most_output","least_feasible","single_victim")
CELLS=tuple(dict(label=f"victim-block{b}-{v}",block=b,variant=v,
 victim_order="most_output" if v=="most_output" else "least_progress",
 filter_victims_by_funding=v!="most_output",event_enabled=v=="single_victim")
 for b,order in ((0,VARIANTS),(1,VARIANTS[::-1])) for v in order)
PAIRS=tuple((f"victim-block{b}-{a}",f"victim-block{b}-{c}") for b in (0,1) for a,c in
 (("most_output","least_feasible"),("least_feasible","single_victim"),("most_output","single_victim")))+tuple(
 (f"victim-block0-{v}",f"victim-block1-{v}") for v in VARIANTS)
SCOPE=("Exploratory reused 32-document workload, one 6656-block pool and two reversed blocks. "
 "The single-victim arm may replace one matched step-891 proposal; all later state is actually "
 "executed. Full costs, output differences, unmatched completed cells and all nine pairs remain. "
 "No quality, significance, Oracle, fresh-workload, production, general-policy or method-GO claim.")

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def jvalue(v):return json.loads(json.dumps(v,allow_nan=False))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);require(spec and spec.loader,f"cannot load {path}")
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def dependencies(funding,context,library):
 require(sha(funding)==FUNDING_SHA,"reviewed funding analyzer changed")
 base=load("single_victim_funding_analysis",funding);ctx,lib=base.import_dependencies(context,library)
 return base,ctx,lib

def replay_event(base,ctx,lib,event_mod,contract,variant,raw,decisions,selector,enabled=None,waiting_order=None):
 holder={};aliases=raw["internal_to_source"]
 def states():
  rows=raw["memory_trace"][holder["step"]]["before"]["requests"]
  return {rid:dict(computed_tokens=s["computed_tokens"],prompt_tokens=s["prompt_tokens"],
   output_tokens=s["output_tokens"],max_tokens=1024,num_tokens=s["prompt_tokens"]+s["output_tokens"],
   owned_blocks=s["block_counts"][0]) for rid,s in rows.items()}
 event=event_mod.SingleVictimEvent(contract=contract,enabled=variant=="single_victim" if enabled is None else enabled,
                                  request_state_provider=states)
 def factory(config,*,victim_order):
  tracker=event.factory(config,victim_order=victim_order);decide=tracker.decide
  def observed(step,running,waiting,free,needed=None,*,released_blocks=None):
   holder["step"]=step
   if step==contract["event"]["step"] and waiting_order:
    by_source={aliases[r]:r for r in waiting}
    if set(by_source)==set(waiting_order):waiting=[by_source[r] for r in waiting_order]
   return decide(step,running,waiting,free,needed,released_blocks=released_blocks)
  tracker.decide=observed;return tracker
 proxy=SimpleNamespace(AbsenceRotation=factory,RotationConfig=selector.RotationConfig,RequestView=selector.RequestView)
 replay=ctx.rotation_replay(lib,"most_output") if variant=="most_output" else base.filtered_rotation_replay(ctx,lib)
 return replay(raw,decisions,proxy),event.snapshot()

def inspector(base,ctx,lib,event_mod,contract,contract_sha,control):
 replayed=[]
 def validate(cfg,raw,decisions,variant):
  funding,enabled=variant!="most_output",variant=="single_victim"
  order="most_output" if variant=="most_output" else "least_progress"
  require(variant in VARIANTS and cfg["variant"]==variant and cfg["cap"]==raw["target_cap"]==32
   and cfg["rotation_victim_order"]==order and cfg["completion_policy"]=="rotate"
   and cfg["policy_family"]=="strong_simple_rotation","arm/cap differs")
  require(cfg.get("filter_victims_by_funding") is funding and cfg.get("event_enabled") is enabled
   and cfg.get("event_contract_sha256")==contract_sha,"event/funding config differs")
  require(all((d.get("filter_victims_by_funding") is True) if funding
   else "filter_victims_by_funding" not in d for d in decisions),"decision funding flag differs")
 def factory(_ctx,_lib,variant):
  def checked(raw,decisions,selector):
   saved=control["saved"];attempt=saved.get("attempt_state") or {}
   result,snapshot=replay_event(base,ctx,lib,event_mod,contract,variant,raw,decisions,selector,
                                waiting_order=attempt.get("waiting_order"))
   replayed.append(snapshot);return result
  return checked
 base.validate_arm_contract,base.replay_factory=validate,factory
 return base.make_inspector(ctx),replayed

def proposal(proposal,aliases):
 p=proposal or {};return tuple(aliases.get(p.get(k),p.get(k)) for k in ("resume_id","victim_id"))
def attach(bundle,row,replayed,contract):
 folder=bundle/"execution/readback/results"/row["label"]
 saved,raw,decisions=(read(folder/n) for n in ("single-victim-event.json","raw.json","headroom-decisions.json"))
 require(saved==jvalue(replayed),"saved event differs from present-state replay")
 variant,aliases=row["variant"],raw["internal_to_source"];enabled=variant=="single_victim";bad=[]
 if saved.get("enabled") is not enabled or saved.get("applied_count")!=int(enabled):bad.append("enable/apply count differs")
 if variant=="least_feasible" and saved.get("status")!="MATCHED_DISABLED":bad.append("control event unmatched")
 native=None
 if enabled:
  event=contract["event"];step=event["step"]
  target,default,replacement=(event[k] for k in ("target_source_id","default_victim_source_id","replacement_source_id"))
  decision=decisions[step];events=[e for e in raw["preemption_events"] if e["attempted_step"]==step
   and aliases[e["victim_internal_request_id"]]==replacement and e["original_preemption_called"] and e["original_preemption_returned"]]
  native=(saved.get("status")=="APPLIED" and saved.get("repeat_attempt_count")==0
   and saved.get("requires_native_forced_event_reconciliation") is True
   and proposal(saved.get("original_proposal"),aliases)==(target,default)
   and proposal(saved.get("returned_proposal"),aliases)==proposal(decision.get("proposal"),aliases)==(target,replacement)
   and [aliases[r] for r in decision.get("forced_preempted",[])]==[replacement] and len(events)==1)
  if not native:bad.append("replacement did not reconcile to one native forced preemption")
 row.update(diagnostic_status="UNMATCHED" if bad else "MATCHED",diagnostic_mismatches=bad,
  single_victim_event=saved,native_forced_event_reconciled=native,
  single_victim_event_sha256=sha(folder/"single-victim-event.json"))

DROP={"start_s","end_s","host_start_perf_counter_s","host_end_perf_counter_s"}
def normalize(v,aliases):
 if isinstance(v,dict):return {aliases.get(k,k):normalize(x,aliases) for k,x in v.items() if k not in DROP}
 if isinstance(v,list):return [normalize(x,aliases) for x in v]
 return aliases.get(v,v) if isinstance(v,str) else v
def prefix(a,b,step):
 def hashes(row):
  raw=read(Path(row["raw_path"]));aliases=raw["internal_to_source"]
  require(len(raw["memory_trace"])>step and len(raw["scheduler_steps"])>step,"event outside trace")
  before=raw["memory_trace"][step]["before"]["requests"];requests={r["request_id"]:r for r in raw["requests"]}
  counts=[{aliases[r]:s["output_tokens"] for r,s in m["before"]["requests"].items()} for m in raw["memory_trace"][:step+1]]
  tokens={aliases[r]:requests[aliases[r]]["output_token_ids"][:s["output_tokens"]] for r,s in before.items()}
  return dict(memory_prefix_sha256=digest(normalize(raw["memory_trace"][:step],aliases)),
   schedule_prefix_sha256=digest(normalize(raw["scheduler_steps"][:step],aliases)),
   output_count_prefix_sha256=digest(counts),pre_event_output_token_prefix_sha256=digest(tokens),
   event_before_state_sha256=digest(normalize(raw["memory_trace"][step]["before"],aliases)))
 left,right=hashes(a),hashes(b)
 return dict(block=a["block"],step=step,matched=left==right,
  token_prefix_semantics="Posthoc identity slice at retained pre-event counts; never an action input.",
  least_feasible=left,single_victim=right)

def analyze(bundle,funding,context,library):
 base,ctx,lib=dependencies(funding,context,library);meta=read(bundle/"preparation/preparation.json")
 source=base.package_source(bundle,meta);campaign=read(source/"campaign.json");contract=read(source/"event_contract.json")
 contract_sha=sha(source/"event_contract.json");checks=read(bundle/"CPU_CHECKS.json")
 require(meta["status"]=="CPU_PREPARED_GPU_UNRUN" and checks["status"]=="CPU_QUALIFIED"
  and sha(bundle/"CPU_CHECKS.json")==meta["cpu_checks_sha256"] and meta.get("selector_ast_matches_frozen_source") is True,
  "CPU qualification/source binding differs")
 require({n:sha(source/n) for n in PINNED}==PINNED,"pinned event execution source differs")
 require(campaign["cells"]==meta["cells"]==list(CELLS),"six-cell inventory differs")
 require(campaign["event_contract_sha256"]==meta["event_contract_sha256"]==contract_sha,"contract binding differs")
 workload=read(source/"inputs_preparation/prepared/heterogeneous/workload.json")
 input_cfg=read(source/"inputs_preparation/prepared/heterogeneous/config.json")
 require(hashlib.sha256(json.dumps(workload,sort_keys=True).encode()).hexdigest()==input_cfg["workload_sha256"],"workload hash differs")
 metrics=lib.module("single_victim_metrics",source/"metrics.py");selector=lib.module("single_victim_selector",source/"absence_rotation.py")
 sys.modules["absence_rotation"]=selector;event_mod=load("single_victim_event_runtime",source/"single_victim_event.py")
 control={};inspect,replayed=inspector(base,ctx,lib,event_mod,contract,contract_sha,control);cells=[]
 for spec in CELLS:
  folder=bundle/"execution/readback/results"/spec["label"];prior=len(replayed)
  control["saved"]=read(folder/"single-victim-event.json") if (folder/"raw.json").exists() else {}
  row=inspect(bundle,spec,lib,meta,source,workload,input_cfg,metrics,selector)
  if row["status"]=="COMPLETE":
   try:
    require(len(replayed)==prior+1,"event replay missing");require(row["gpu_uuid"]==meta["expected_gpu_uuid"],"GPU differs")
    attach(bundle,row,replayed[-1],contract)
   except (OSError,KeyError,ValueError,TypeError,IndexError,AttributeError) as exc:row.update(status="INVALID_OR_INCOMPLETE",eligible=False,error=str(exc))
  cells.append(row)
 comparisons,prefixes=[],[]
 if all(r["eligible"] for r in cells):
  for block in (0,1):
   rows={r["variant"]:r for r in cells if r["block"]==block}
   comparisons += [ctx.compare(rows[a],rows[b]) for a,b in (("most_output","least_feasible"),
    ("least_feasible","single_victim"),("most_output","single_victim"))]
   receipt=prefix(rows["least_feasible"],rows["single_victim"],contract["event"]["step"]);prefixes.append(receipt)
   if not receipt["matched"]:
    for variant in ("least_feasible","single_victim"):
     rows[variant]["diagnostic_status"]="UNMATCHED";rows[variant]["diagnostic_mismatches"].append("pre-event prefix differs")
  for variant in VARIANTS:
   first,second=[r for r in cells if r["variant"]==variant];comparisons.append(ctx.compare(first,second))
  require([(r["baseline"],r["action"]) for r in comparisons]==list(PAIRS),"comparison order differs")
 status="MEASUREMENT_ONLY" if comparisons else "UNRUN" if all(r["status"]=="UNRUN" for r in cells) else "INCOMPLETE"
 diagnostic="UNRUN" if status=="UNRUN" else "INCOMPLETE" if status=="INCOMPLETE" else (
  "MATCHED" if all(r.get("diagnostic_status")=="MATCHED" for r in cells if r["variant"]!="most_output") else "UNMATCHED")
 result=dict(status=status,diagnostic_status=diagnostic,scope=SCOPE,producer_sha256=sha(Path(__file__)),
  funding_analyzer_sha256=sha(funding),reused_context_analyzer={"path":str(context),"sha256":sha(context)},
  reused_analysis={str(Path(m.__file__)):sha(Path(m.__file__)) for m in (lib,lib.rotation,lib.token,lib.base,lib.headroom)},
  prepared_metadata_sha256=sha(bundle/"preparation/preparation.json"),event_contract_sha256=contract_sha,
  event_waiting_order_source="executed event snapshot; raw membership and all reconstructable fields replayed",
  comparison_contract=[dict(baseline=a,action=b) for a,b in PAIRS],pre_event_prefix_receipts=prefixes,
  cells=cells,comparisons=comparisons)
 for row in cells:row.pop("outputs",None)
 return result

def cpu_checks(bundle,legacy,funding,context,library):
 base,ctx,lib=dependencies(funding,context,library);contract=read(bundle/"event_contract.json")
 require(sha(bundle/"event_contract.json")==PINNED["event_contract.json"],"CPU contract differs")
 source=legacy/"preparation/pkg";selector=lib.module("single_victim_cpu_selector",source/"absence_rotation.py")
 sys.modules["absence_rotation"]=selector;event_mod=load("single_victim_event_cpu",Path(__file__).with_name("single_victim_event.py"))
 require(sha(Path(event_mod.__file__))==PINNED["single_victim_event.py"],"CPU event module differs")
 results=legacy/"execution/readback/results";rows=[]
 for label,variant in (("funding-block0-most_output","most_output"),("funding-block0-least_feasible","least_feasible"),
  ("funding-block1-least_feasible","least_feasible"),("funding-block1-most_output","most_output")):
  folder=results/label;raw,decisions=read(folder/"raw.json"),read(folder/"headroom-decisions.json")
  plain=(ctx.rotation_replay(lib,"most_output") if variant=="most_output" else base.filtered_rotation_replay(ctx,lib))(raw,decisions,selector)
  observed,event=replay_event(base,ctx,lib,event_mod,contract,variant,raw,decisions,selector,enabled=False,
                              waiting_order=contract["expected"]["waiting_order"])
  require(plain==observed and event["applied_count"]==0,"disabled observer changed old path")
  require(jvalue(jvalue(event))==jvalue(event),"snapshot JSON form unstable")
  rows.append(dict(label=label,raw_sha256=sha(folder/"raw.json"),result_equal=True,
                   snapshot_json_stable=True,event_status=event["status"]))
 forged=copy.deepcopy(contract);target=forged["event"]["target_source_id"]
 forged["expected"]["request_states"][target]["output_tokens"]+=1;folder=results/"funding-block0-least_feasible"
 raw,decisions=read(folder/"raw.json"),read(folder/"headroom-decisions.json")
 _,rejected=replay_event(base,ctx,lib,event_mod,forged,"single_victim",raw,decisions,selector,enabled=True,
                         waiting_order=contract["expected"]["waiting_order"])
 require(rejected["status"]=="UNMATCHED" and rejected["applied_count"]==0
  and rejected["original_proposal"]==rejected["returned_proposal"],"forged state accepted")
 old=prefix(dict(block=0,raw_path=str(results/"funding-block0-least_feasible/raw.json")),
            dict(block=1,raw_path=str(results/"funding-block1-least_feasible/raw.json")),contract["event"]["step"])
 require(old["least_feasible"]["pre_event_output_token_prefix_sha256"]==old["single_victim"]["pre_event_output_token_prefix_sha256"],
         "old same-role token prefix differs")
 return dict(status="CPU_QUALIFIED_GPU_UNRUN",scope="Old raw checks only; no new-arm performance generated.",
  disabled_old_actual_paths=rows,old_same_role_pre_event_token_prefix_match=True,
  forged_event_rejection=dict(field=f"expected.request_states.{target}.output_tokens",event_status="UNMATCHED",applied_count=0),
  actual_new_gpu_cells="UNRUN",dependencies=dict(analyzer_sha256=sha(Path(__file__)),funding_analyzer_sha256=sha(funding),
   context_analyzer_sha256=sha(context),event_module_sha256=sha(Path(event_mod.__file__)),event_contract_sha256=sha(bundle/"event_contract.json")))

def write(path,value):
 require(not path.exists(),f"refuse output overwrite: {path}");path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("bundle","funding-analyzer","context-analyzer","analysis-library"):p.add_argument("--"+name,required=True,type=Path)
 mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--output",type=Path);mode.add_argument("--cpu-checks-output",type=Path)
 p.add_argument("--legacy-bundle",type=Path);a=p.parse_args()
 if a.cpu_checks_output:
  require(a.legacy_bundle is not None,"--legacy-bundle required");result=cpu_checks(a.bundle,a.legacy_bundle,a.funding_analyzer,a.context_analyzer,a.analysis_library);write(a.cpu_checks_output,result)
 else:
  require(a.legacy_bundle is None,"--legacy-bundle is CPU-only");result=analyze(a.bundle,a.funding_analyzer,a.context_analyzer,a.analysis_library);write(a.output,result)
 print(json.dumps(dict(status=result["status"],diagnostic_status=result.get("diagnostic_status"),
  cells=len(result.get("cells",[])),comparisons=len(result.get("comparisons",[])))))
if __name__=="__main__":main()
