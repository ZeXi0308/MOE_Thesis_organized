#!/usr/bin/env python3
"""CPU-only preparation of the five already selected controller evaluations.

Run from the project root AFTER olmoe_fullbudget_calibration_r01 completes:
  python code/prepare_controller_evaluation.py \
    --calibration-run runs/olmoe_fullbudget_calibration_r01 \
    --direct-run runs/olmoe_ngram_matrix_r02 \
    --output runs/controller_preparation_eager_r01

Reuses build_calibration.py and select_controller_baselines.py. No run_campaign
invocation is executed here. It emits evaluation_commands.json for a separate
serialized coordinator, preserving the existing runner's GPU lock/budgets.
The original 64 direct-draft selection files and protocol are hash pinned;
held-out data cannot change the choices. Incomplete campaigns are refused
before any output directory is created. --check-direct-only verifies the frozen
selection locally without requiring full-budget measurements or GPU software.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import unittest

from cost_model import AcceptanceEWMA, CalibrationTable
from plot_results import REGIME_FIELDS, _context, _requests, load_runs, resolve_regime, route_instrumented
from select_controller_baselines import select_regime, validate_protocol, write_selection


ROOT=Path(__file__).resolve().parents[1]
BATCHES=[1,4,16,64]
GAMMAS=list(range(8))
PROTOCOL_SHA="1acd1735e7c4935e74163463dd8bb1231e728a11339f8d68f7f5623ebe0fb6db"
DIRECT_INPUT_SHA="8b3a9fb14cb192224988272e760aceeba2d62fe634c2b2a82bb2a9d64db53823"
CRITICAL_SOURCE=("run_matrix.py","run_campaign.py","vllm_s_plugin.py","cost_model.py","power.py")
EXPECTED_TABLE=[[1,3,7],[4,15,5],[16,64,6]]
EXPECTED_THRESHOLD=[[1,64,6]]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def behavior_sha(path):
    """Ignore documentation-only differences; compare both files on one Python."""
    tree=ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body=node.body[1:]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def require_complete_status(status):
    cells=status.get("cells",[])
    if status.get("status")!="complete" or len(cells)!=8 or sorted(c.get("gamma") for c in cells)!=GAMMAS or any(c.get("returncode")!=0 for c in cells):
        raise ValueError("campaign must be complete with exactly eight successful gamma cells")


def expected_choice(selection):
    if (selection.get("status")!="selected_pending_evaluation"
        or selection.get("static",{}).get("selected_gamma")!=6
        or selection.get("native_unrestricted",{}).get("table")!=EXPECTED_TABLE
        or selection.get("native_threshold",{}).get("table")!=EXPECTED_THRESHOLD
        or selection.get("native_threshold",{}).get("active_gamma")!=6
        or selection.get("native_threshold",{}).get("disable_above_batch")!=64):
        raise ValueError("reconstruction differs from frozen static6/native-table/threshold6-cutoff64 choices")


def inspect_campaign(directory, *, fullbudget):
    directory=Path(directory).resolve()
    require_complete_status(read(directory/"status.json"))
    plan=read(directory/"plan.json")
    expected_repeats,expected_length=(1,64) if fullbudget else (2,128)
    if plan.get("gammas")!=GAMMAS or plan.get("repeats")!=expected_repeats or plan.get("measured_requests")!=64 or plan.get("output_tokens")!=expected_length:
        raise ValueError("campaign plan differs from the frozen calibration population/repeats/output length")
    if fullbudget and plan.get("batches")!=BATCHES:
        raise ValueError("full-budget calibration must contain exactly B1/4/16/64")
    if not fullbudget and directory.name!="olmoe_ngram_matrix_r02":
        raise ValueError("only the frozen r02 development matrix may choose baselines")
    workload=ROOT/plan["workload"]
    expected_ids=sorted(r["source_index"] for r in read(workload)["records"][:64])
    expected_workload_sha=sha(workload)
    paths=[];sources=[];envs=[];domains=[];source_sets=[]
    for gamma in GAMMAS:
        folder=directory/f"gamma{gamma}"
        if read(folder/"status.json").get("status")!="complete":
            raise ValueError(f"gamma{gamma} did not finish")
        config=read(folder/"configuration.json");engine=read(folder/"engine_args.json");env=read(folder/"environment.json")
        if config.get("workload_sha256")!=expected_workload_sha or int(config.get("plan",{}).get("input_offset",0))!=0:
            raise ValueError("calibration workload/hash/offset is not the development population")
        if config["args"].get("mode")!=("static" if fullbudget else "native"):
            raise ValueError("calibration execution mode does not match its intended role")
        spec=engine.get("speculative_config") or {}
        budget=7 if fullbudget else gamma
        if int(spec.get("num_speculative_tokens",0))!=budget or spec.get("num_speculative_tokens_per_batch_size") is not None:
            raise ValueError("wrong engine proposal budget or unexpected native batch table")
        if not engine.get("enforce_eager") or engine.get("enable_return_routed_experts",False):
            raise ValueError("controller calibration requires eager route-off performance engines")
        if env.get("vllm")!="0.26.0": raise ValueError("vLLM version differs from frozen experiment")
        source_hashes={}
        for name in CRITICAL_SOURCE:
            captured=directory/"source"/name
            source_hashes[name]=sha(captured)
            if env.get("source_sha256",{}).get(name)!=source_hashes[name]:
                raise ValueError(f"captured source differs from actual engine source: {name}")
            if fullbudget and behavior_sha(captured)!=behavior_sha(ROOT/"code"/name):
                raise ValueError(f"current executable behavior changed after full-budget calibration: {name}")
        source_sets.append(source_hashes)
        envs.append({k:env.get(k) for k in ("vllm","torch","cuda","gpu")})
        required=[]
        for repeat in range(expected_repeats):
            for batch in BATCHES:
                path=folder/f"perf-r{repeat}-b{batch}.json";required.append(path)
                record=read(path);context=_context(path)
                if route_instrumented(record,context) or not record.get("all_finished"):
                    raise ValueError(f"trace-on or incomplete performance population: {path}")
                if (record.get("batch"),record.get("gamma"),record.get("proposal_budget"))!=(batch,gamma,budget):
                    raise ValueError(f"wrong B/gamma/proposal budget: {path}")
                requests=_requests(record)
                if len(requests)!=64 or sorted(r.get("source_index") for r in requests)!=expected_ids:
                    raise ValueError(f"wrong actual development prompt IDs: {path}")
                if any(not r.get("finished") or len(r.get("output_token_ids",[]))!=expected_length or r.get("effective_tpot_ms") is None for r in requests):
                    raise ValueError(f"incomplete request outputs or effective TPOT: {path}")
                if record.get("output_tokens")!=64*expected_length:
                    raise ValueError(f"wrong total output token count: {path}")
                for episode in record.get("episodes",[record]):
                    if len(episode.get("host_steps",[]))!=len(episode.get("plugin",{}).get("steps",[])):
                        raise ValueError(f"host/plugin step mismatch: {path}")
                paths.append(path)
                sources.append([str(path.relative_to(directory)),sha(path)])
                domains.append(resolve_regime(record,context))
        if fullbudget and set(folder.glob("perf-*.json"))!=set(required):
            raise ValueError("unexpected or incomplete performance files in full-budget calibration")
    if any(d!=domains[0] for d in domains) or any(e!=envs[0] for e in envs) or any(s!=source_sets[0] for s in source_sets):
        raise ValueError("mixed source/environment/execution domain within a calibration campaign")
    inputs_sha=hashlib.sha256(json.dumps(sorted(sources),separators=(",",":")).encode()).hexdigest()
    if not fullbudget and inputs_sha!=DIRECT_INPUT_SHA:
        raise ValueError("direct calibration inputs differ from the hash-pinned r02 selection files")
    return {"path":str(directory),"files":paths,"source_inputs_sha256":inputs_sha,
            "source_files":sources,"regime":domains[0],"environment":envs[0],"captured_source":source_sets[0],
            "plan_sha256":sha(directory/"plan.json"),"status_sha256":sha(directory/"status.json")}


def reconstruct_direct(directory,protocol):
    evidence=inspect_campaign(directory,fullbudget=False)
    rows,excluded=load_runs(evidence["files"])
    if excluded or len(rows)!=64: raise ValueError("some frozen objective runs were excluded")
    result=select_regime(rows,protocol)
    expected_choice(result)
    if len(result["measured_cells"])!=32 or any(r["n_runs"]!=2 for r in result["measured_cells"]):
        raise ValueError("direct selection must retain two raw run values in all 32 choice cells")
    return evidence,rows,result


def evaluation_config_checks(config_path,full_evidence):
    plan=read(config_path)
    required={"batches":BATCHES,"repeats":2,"measured_requests":64,"output_tokens":128,
        "input_offset":64,"warmup_offset":0,"shape_warmup_offset":0,"warmup_each_batch":True,"enforce_eager":True}
    if any(plan.get(k)!=v for k,v in required.items()): raise ValueError("evaluation config differs from the frozen held-out/warmup protocol")
    regime=full_evidence["regime"]
    if plan["model"]!=regime["model"] or plan["draft_source"]!=regime["draft_source"] or sha(ROOT/plan["workload"])!=regime["workload_sha256"]:
        raise ValueError("evaluation model/draft/workload differs from calibrated domain")
    expected={"dtype":plan.get("dtype","bfloat16"),"quantization":plan.get("quantization","none"),
        "kv_cache_memory_bytes":plan.get("kv_cache_memory_bytes",6*1024**3),"max_model_len":plan.get("max_model_len",1024),
        "admission_mode":plan.get("admission_mode","wave"),"engine_max_num_seqs":plan.get("max_num_seqs",max(plan["batches"]))}
    if any(str(v)!=str(regime[k]) for k,v in expected.items()): raise ValueError("evaluation resource/admission domain differs from calibration")
    expected_kernel={"kernel_config":plan.get("kernel_config",{"moe_backend":"triton"})}
    if json.loads(regime["kernel_backend"])!=expected_kernel or plan.get("native_route_return",False):
        raise ValueError("evaluation kernel or instrumentation differs from calibration")
    if AcceptanceEWMA().snapshot().get("alpha_estimator")!="exposure_ratio":
        raise ValueError("current controller does not use the fixed exposure_ratio estimator")


def make_commands(output,selection_dir,config_path):
    definitions=[("ar","olmoe_controller_eval_ar_r01",0,0,"native",[]),
        ("static","olmoe_controller_eval_static6_r01",6,6,"native",[]),
        ("native_unrestricted","olmoe_controller_eval_native_table_r01",7,7,"native",["--native-table",str(selection_dir/"native_unrestricted_table.json")]),
        ("native_threshold","olmoe_controller_eval_native_threshold_r01",7,7,"native",["--native-table",str(selection_dir/"native_threshold_table.json")]),
        ("adaptive","olmoe_controller_eval_adaptive_r01",7,7,"adaptive",["--calibration",str(output/"calibration/calibration.json")])]
    commands=[]
    for role,run,gamma,budget,mode,extra in definitions:
        if (ROOT/"runs"/run).exists(): raise FileExistsError(f"evaluation run already exists; do not resubmit: {run}")
        argv=[sys.executable,"-u","code/run_campaign.py","--config",str(config_path),"--run",run,
            "--gammas",str(gamma),"--mode",mode,"--proposal-budget",str(budget),
            "--wait-lock-seconds","21600","--skip-trace",*extra]
        commands.append({"role":role,"run":run,"config":str(config_path),"argv":argv,"cwd":str(ROOT),
                         "status_file":str(ROOT/"runs"/run/"status.json"),"expected_performance_run_records":8})
    return commands


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration-run",default="runs/olmoe_fullbudget_calibration_r01")
    parser.add_argument("--direct-run",default="runs/olmoe_ngram_matrix_r02")
    parser.add_argument("--output")
    parser.add_argument("--check-direct-only",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test: self_test();return
    protocol_path=ROOT/"configs/controller_protocol.json"
    if sha(protocol_path)!=PROTOCOL_SHA: raise ValueError("selection protocol changed; frozen selection cannot be silently redefined")
    protocol=read(protocol_path);validate_protocol(protocol)
    direct_path=Path(args.direct_run).resolve()
    if args.check_direct_only:
        _,_,selection=reconstruct_direct(direct_path,protocol)
        print(json.dumps({"status":"frozen_direct_selection_verified","source_files":64,"static_gamma":6,
            "native_unrestricted":selection["native_unrestricted"]["table"],"native_threshold":selection["native_threshold"]["table"]}));return
    if not args.output: parser.error("--output is required")
    output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError("choose a fresh preparation output directory")
    # Check the still-running dependency first; no partial output or GPU action.
    full=inspect_campaign(args.calibration_run,fullbudget=True)
    direct,rows,_=reconstruct_direct(direct_path,protocol)
    if full["regime"]!=direct["regime"] or full["environment"]!=direct["environment"]:
        raise ValueError("full-budget cost and frozen direct selection differ in execution/environment domain")
    eval_config=ROOT/"configs/olmoe_ngram_controller_evaluation.json"
    evaluation_config_checks(eval_config,full)
    output.mkdir()
    subprocess.run([sys.executable,str(ROOT/"code/build_calibration.py"),str(Path(args.calibration_run).resolve()),"--output",str(output/"calibration")],check=True,cwd=ROOT)
    table_path=output/"calibration/calibration.json"
    table=read(table_path);records=table["records"]
    expected={(b,g,7) for b in BATCHES for g in GAMMAS}
    observed=Counter((int(r["batch"]),int(r["gamma"]),r.get("proposal_budget")) for r in records)
    if set(observed)!=expected or any(n!=1 for n in observed.values()) or read(output/"calibration/exclusions.json"):
        raise ValueError("complete-step analysis did not produce exactly 32 unique budget-7 measured cells")
    if any(r.get("evidence_kind")!="measured_complete_step_cost" or r.get("eligible_steps",0)<=0 for r in records):
        raise ValueError("unmeasured or empty complete-step cost cell")
    CalibrationTable(records,table["metadata"])
    manifest=write_selection(output/"selection",protocol,protocol_path,rows,[],[])
    if len(manifest["regimes"])!=1: raise ValueError("unexpected mixed selection regimes")
    selection_path=Path(manifest["regimes"][0]["selection"])
    expected_choice(read(selection_path))
    commands=make_commands(output,selection_path.parent,eval_config)
    evidence={"status":"ready_for_serialized_evaluation","commands":commands,
        "calibration_table":str(table_path),"calibration_sha256":sha(table_path),"calibration_cells":32,
        "selection":str(selection_path),"selection_sha256":sha(selection_path),
        "frozen_protocol_sha256":PROTOCOL_SHA,"frozen_direct_input_sha256":DIRECT_INPUT_SHA,
        "calibration_evidence":{k:v for k,v in full.items() if k!="files"},
        "selection_evidence":{k:v for k,v in direct.items() if k!="files"},
        "evaluation_config_sha256":sha(eval_config),"current_source_sha256":{n:sha(ROOT/"code"/n) for n in CRITICAL_SOURCE},
        "budget":{"group_gpu_budget_seconds":read(eval_config)["group_gpu_budget_seconds"],"cumulative_gpu_budget_seconds":read(eval_config)["total_gpu_budget_seconds"],"enforced_by":"existing run_campaign.py; shared GPU lock and cumulative accounting unchanged"},
        "execution_policy":"this CPU script generates artifacts only; coordinator executes these five unique commands sequentially and requires actual status complete after each",
        "boundaries":["Baseline choices are reconstructed only from hash-pinned r02 development files; no held-out reselection.","Full-budget gamma0 is not AR; AR uses native gamma0 and proposal_budget0.","Native threshold [[1,64,6]] uses engine max7 with native per-batch production6; it is distinct from post-proposer dense_heuristic.","No natural-stop campaign is added by this preparation."]}
    (output/"evaluation_commands.json").write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+"\n")
    (output/"evaluation_commands.txt").write_text("# Reference only; generated by CPU preparation, not executed here.\n"+"\n".join(shlex.join(c["argv"]) for c in commands)+"\n")
    print(json.dumps({"status":evidence["status"],"calibration_cells":32,"selection":str(selection_path),"commands_file":str(output/"evaluation_commands.json")}))


def self_test():
    class PrepChecks(unittest.TestCase):
        def test_evaluation_commands_preserve_queue_and_config_interface(self):
            from unittest.mock import patch
            with patch.object(Path,"exists",return_value=False):
                commands=make_commands(Path("/tmp/preparation"),Path("/tmp/selection"),Path("/tmp/evaluation.json"))
            self.assertEqual([c["role"] for c in commands],
                ["ar","static","native_unrestricted","native_threshold","adaptive"])
            self.assertEqual(len({c["run"] for c in commands}),5)
            for command in commands:
                argv=command["argv"]
                self.assertEqual(argv[:3],[sys.executable,"-u","code/run_campaign.py"])
                self.assertEqual(argv.count("--wait-lock-seconds"),1)
                self.assertEqual(argv[argv.index("--wait-lock-seconds")+1],"21600")
                self.assertTrue(all(flag not in argv for flag in ("--batches","--output-tokens","--repeats")))
        def test_incomplete_campaign_is_not_ready(self):
            complete={"status":"complete","cells":[{"gamma":g,"returncode":0} for g in GAMMAS]}
            require_complete_status(complete)
            for broken in ({**complete,"status":"running"},{**complete,"cells":complete["cells"][:-1]},
                           {**complete,"cells":[{"gamma":g,"returncode":1 if g==7 else 0} for g in GAMMAS]}):
                with self.assertRaises(ValueError): require_complete_status(broken)
        def test_frozen_choice_cannot_silently_change(self):
            selection={"status":"selected_pending_evaluation","static":{"selected_gamma":6},
                "native_unrestricted":{"table":EXPECTED_TABLE},"native_threshold":{"table":EXPECTED_THRESHOLD,"active_gamma":6,"disable_above_batch":64}}
            expected_choice(selection)
            selection["static"]["selected_gamma"]=7
            with self.assertRaises(ValueError): expected_choice(selection)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PrepChecks)
    if not unittest.TextTestRunner().run(suite).wasSuccessful(): raise SystemExit(1)


if __name__=="__main__": main()
