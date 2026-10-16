#!/usr/bin/env python3
"""Single loaded vLLM 0.26 eager engine: native decode TPOT, energy and routes.

Requires an external GPU lock (run_locked.py). Does not obtain or bypass a lock.
Performance and energy/trace are separate passes. Forced-length greedy decoding
is a workload-controlled benchmark, not natural termination or task quality.
NVML measures whole-board decode energy, not process-only power or HBM bytes.

v0.26 API checked against vllm/v1/engine/llm_engine.py at tag v0.26.0:
add_request returns an internal ID; outputs retain the caller's external ID.
V1 multiprocessing and asynchronous scheduling are disabled before loading.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import time


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")


def quantile(values, q):
    if not values:return None
    xs=sorted(values);p=(len(xs)-1)*q;i=int(p)
    return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(p-i)


def select_prompts(path, tokenizer, count, length, offset=0):
    """Select distinct source records with >= length tokens; never pad/repeat."""
    prompts, provenance=[],[]
    seen=set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number,line in enumerate(handle,1):
            if line_number<=offset or not line.strip():continue
            row=json.loads(line)
            text=row if isinstance(row,str) else row.get("text",row.get("prompt"))
            if not isinstance(text,str):
                if isinstance(row,dict) and isinstance(row.get("messages"),list):
                    text="\n".join(m["content"] for m in row["messages"] if isinstance(m.get("content"),str))
                else:raise ValueError(f"{path}:{line_number} needs text, prompt, or text messages")
            ids=tokenizer.encode(text,add_special_tokens=True)
            if len(ids)<length:continue
            ids=ids[:length]
            token_hash=hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()
            if token_hash in seen:continue
            seen.add(token_hash)
            prompts.append({"prompt_token_ids":ids})
            provenance.append({"source_line":line_number,"text_sha256":hashlib.sha256(text.encode()).hexdigest(),
                               "prompt_token_sha256":token_hash,"source_token_count":len(tokenizer.encode(text,add_special_tokens=True))})
            if len(prompts)==count:break
    if len(prompts)<count:
        raise ValueError(f"need {count} distinct source records with >= {length} tokens; got {len(prompts)}")
    return prompts,provenance


def one_worker(llm, method, *args):
    results=llm.collective_rpc(method,args=args)
    if len(results)!=1:raise RuntimeError("single-GPU benchmark expected exactly one worker")
    return results[0]


def run_batch(engine, prompts, sampling, prefix, meter=None, failure_path=None):
    """Measure host-visible one-token steps. Start energy only after all TTFTs.

Synchronous fixed cohorts must emit one token for every request on each step;
if scheduling changes this invariant, preserve the failed run and fix the setup.
The energy window excludes all prefill and first-token generation.
"""
    if engine.has_unfinished_requests():raise RuntimeError("engine has outstanding requests")
    submitted=time.perf_counter_ns()
    states={}
    steps=[];sampling_active=False;power=None
    try:
        for i,prompt in enumerate(prompts):
            rid=f"{prefix}-{i}"
            internal_id=engine.add_request(rid,prompt,sampling)
            if rid in states:raise RuntimeError("duplicate request ID")
            states[rid]={"request_id":rid,"internal_id":internal_id,"prompt_index":i,"token_ids":[],"output_count":0,"visible_ns":[],
                         "finished":False,"finish_reason":None,"stop_reason":None}
        while engine.has_unfinished_requests():
            start=time.perf_counter_ns()
            outputs=engine.step()
            end=time.perf_counter_ns()
            emitted=0;active_before=sum(not s["finished"] for s in states.values())
            seen_this_step=set()
            for output in outputs:
                if output.request_id not in states:raise RuntimeError("unexpected output request")
                if len(output.outputs)!=1:raise RuntimeError("benchmark requires n=1")
                state=states[output.request_id];seq=output.outputs[0]
                ids=seq.token_ids;delta=len(ids)-state["output_count"]
                if delta!=1:raise RuntimeError(f"need exactly one visible token per output; got {delta}")
                # Keep the current cumulative view; copy once after completion.
                # Avoid scanning/copying the growing output inside every timed step.
                state["token_ids"]=ids;state["output_count"]=len(ids);state["visible_ns"].append(end)
                state["finished"]=bool(output.finished)
                state["finish_reason"]=seq.finish_reason;state["stop_reason"]=seq.stop_reason
                emitted+=delta;seen_this_step.add(output.request_id)
            # Full prompt cohort fits in one scheduler prefill batch by construction.
            # Empty warm scheduling steps may precede the first output only.
            any_first=any(s["visible_ns"] for s in states.values())
            if any_first and len(seen_this_step)!=active_before:
                raise RuntimeError("partial/empty cohort step: expected full synchronous decode batch")
            phase="decode" if len(steps) and all(len(s["token_ids"])>=2 for s in states.values()) else "prefill_or_first_token"
            steps.append({"index":len(steps),"start_ns":start,"end_ns":end,
                          "engine_step_ms":(end-start)/1e6,"output_tokens":emitted,
                          "active_before":active_before,"phase":phase})
            if meter is not None and not sampling_active and all(len(s["token_ids"])==1 for s in states.values()):
                meter.start();sampling_active=True
        if meter is not None and sampling_active:
            power=meter.stop();sampling_active=False
    except BaseException as error:
        cleanup_errors=[]
        if sampling_active:
            try:
                power=meter.stop();power["workload_failed"]=True
            except BaseException as cleanup_error:
                cleanup_errors.append(f"power stop: {type(cleanup_error).__name__}: {cleanup_error}")
        pending=[rid for rid,s in states.items() if not s["finished"]]
        if pending:
            try:engine.abort_request(pending)
            except BaseException as cleanup_error:
                cleanup_errors.append(f"request abort: {type(cleanup_error).__name__}: {cleanup_error}")
        if failure_path is not None:
            write_json(failure_path,{"error":f"{type(error).__name__}: {error}",
                                     "partial_requests":list(states.values()),"steps":steps,"power":power,
                                     "cleanup_errors":cleanup_errors})
        raise
    end=time.perf_counter_ns()
    intervals=[]
    requests=[]
    for state in states.values():
        state["token_ids"]=list(state["token_ids"])
        stamps=state["visible_ns"]
        itl=[(b-a)/1e6 for a,b in zip(stamps,stamps[1:])]
        intervals.extend(itl)
        requests.append({**state,"output_tokens":len(state["token_ids"]),
            "ttft_ms":(stamps[0]-submitted)/1e6 if stamps else None,
            "tpot_ms":statistics.mean(itl) if itl else None,"inter_token_ms":itl})
        if not state["finished"] or len(state["token_ids"])!=sampling.max_tokens:
            raise RuntimeError("forced-length request did not finish at max_tokens")
    decode_tokens=sum(max(r["output_tokens"]-1,0) for r in requests)
    if power is not None:
        sampled_j=power.get("sampled_energy_j")
        counter_j=power.get("counter_energy_j")
        power.update({"decode_output_tokens":decode_tokens,"includes_prefill":False,
            "joules_per_decode_token":sampled_j/decode_tokens if sampled_j is not None and decode_tokens else None,
            "primary_energy_method":"nvml_power_trapezoid",
            "counter_joules_per_decode_token":counter_j/decode_tokens if counter_j is not None and decode_tokens else None,
            "counter_comparison_valid":bool(power.get("counter_valid",False)),
            "minimum_3s_window_met":power["boundary"]["workload_elapsed_s"]>=3.0,
            "window_semantics":"after all B first tokens are visible, through completion of remaining decode tokens; includes host loop and NVML boundary overhead"})
    return {"batch_size":len(prompts),"wall_s":(end-submitted)/1e9,"steps":steps,
        "requests":requests,"decode_output_tokens":decode_tokens,
        "mean_tpot_ms":statistics.mean(intervals) if intervals else None,
        "p50_inter_token_ms":quantile(intervals,.50),"p95_inter_token_ms":quantile(intervals,.95),
        "p99_inter_token_ms":quantile(intervals,.99),"power":power,
        "timing_semantics":"host-visible cumulative engine outputs; intervals include harness bookkeeping; excludes first token from TPOT",
        "uncertainty_semantics":"per-token quantiles are descriptive, not independent run confidence intervals"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True);p.add_argument("--prompts",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--arms-file",type=Path,default=Path(__file__).resolve().parents[1]/"configs/default_arms.json")
    p.add_argument("--arms",nargs="+");p.add_argument("--batches",type=int,nargs="+",default=[1,2,4,8,16,32,64])
    p.add_argument("--repeats",type=int,default=3);p.add_argument("--newtokens",type=int,default=256)
    p.add_argument("--prompt-tokens",type=int,default=128);p.add_argument("--prompt-offset",type=int,default=0)
    p.add_argument("--warmup-seconds",type=float,default=3.0);p.add_argument("--warmup-tokens",type=int,default=16)
    p.add_argument("--dtype",default="bfloat16");p.add_argument("--quantization",default=None)
    p.add_argument("--max-model-len",type=int,default=1024);p.add_argument("--gpu-memory-utilization",type=float,default=.65)
    p.add_argument("--backend",choices=["torch","triton"],default="triton")
    p.add_argument("--seed",type=int,default=1701);p.add_argument("--nvml-device",default="0")
    p.add_argument("--no-energy",action="store_true");p.add_argument("--traceonly",action="store_true")
    p.add_argument("--trace-steps",type=int,default=16);p.add_argument("--capture-traces",action="store_true")
    p.add_argument("--dense-counter-check",action="store_true",help="add dense counted identity as diagnostic tenth arm")
    p.add_argument("--execution",choices=["eager","graph"],default="eager")
    p.add_argument("--graph-counter-pass",action="store_true",
                   help="fixed-arm graph accounting process; not the primary latency pass")
    args=p.parse_args()
    if min(*args.batches,args.repeats,args.prompt_tokens,args.warmup_tokens,args.trace_steps)<1 or args.newtokens<2:
        p.error("require positive dimensions and newtokens >= 2")
    if args.prompt_tokens+max(args.newtokens,args.warmup_tokens,args.trace_steps+1)>args.max_model_len:
        p.error("prompt plus generation exceeds max-model-len")
    if args.output_dir.exists():raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
    if importlib.metadata.version("vllm").split("+")[0]!="0.26.0":raise RuntimeError("requires vLLM 0.26.0")
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    arms=json.loads(args.arms_file.read_text())
    if args.arms:
        wanted=set(args.arms);arms=[a for a in arms if a["name"] in wanted]
        if wanted!={a["name"] for a in arms}:raise ValueError("unknown requested arm")
    if args.dense_counter_check:arms.append({"name":"dense_counted","spec":{"mode":"dense"},"diagnostic":True})
    if not arms:raise ValueError("empty arm list")
    for arm in arms:
        if arm["spec"] is not None:arm["spec"]["backend"]=args.backend
    graph=args.execution=="graph"
    if args.graph_counter_pass and not graph:
        raise ValueError("graph-counter-pass requires graph execution")
    if graph and (len(arms)!=1 or args.capture_traces or args.traceonly or args.dense_counter_check):
        raise ValueError("graph execution requires exactly one immutable arm and no Python traces")
    if graph and arms[0]["spec"] and arms[0]["spec"].get("mode")=="global_native":
        arms[0]["spec"]["count_routes"]=args.graph_counter_pass
    model_kwargs=dict(model=args.model,dtype=args.dtype,tensor_parallel_size=1,enforce_eager=True,
        compilation_config={"mode":0,"cudagraph_mode":"NONE"},worker_cls="skip_worker.ExpertSkipWorker",
        moe_backend="triton",enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
        max_model_len=args.max_model_len,max_num_seqs=max(args.batches),
        max_num_batched_tokens=max(args.prompt_tokens*max(args.batches),args.max_model_len),
        gpu_memory_utilization=args.gpu_memory_utilization,seed=args.seed,disable_log_stats=True)
    if graph:
        os.environ["K_SKIP_GRAPH_CONFIG"]=json.dumps({"spec":arms[0]["spec"],
            "count_routes":args.graph_counter_pass,"log_runtime":True})
        model_kwargs.update(enforce_eager=False,worker_cls="skip_graph_worker.ExpertSkipGraphWorker",
            compilation_config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",
                "cudagraph_capture_sizes":sorted(set(args.batches)),
                "max_cudagraph_capture_size":max(args.batches)})
    if args.quantization:model_kwargs["quantization"]=args.quantization
    write_json(args.output_dir/"config.json",{"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "arms":arms,"model_kwargs":model_kwargs,"vllm_version":"0.26.0",
        "source_sha256":{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                         for name in ("bench_decode.py","skip_worker.py","skip_graph_worker.py","expert_skip.py","power.py")},
        "generation_semantics":"greedy, ignore_eos=true, fixed output length; no natural quality claim",
        "byte_semantics":"device policy counters report logical distinct expert weight footprint, not measured DRAM bytes"})
    llm=LLM(**model_kwargs);engine=llm.llm_engine
    installation=one_worker(llm,"skip_set_policy",arms[0]["spec"] if graph else None,"decode",0,False)
    write_json(args.output_dir/"installation.json",installation)
    prompts,source=select_prompts(args.prompts,llm.get_tokenizer(),max(args.batches),args.prompt_tokens,args.prompt_offset)
    write_json(args.output_dir/"prompts.json",{"source":str(args.prompts),"source_sha256":hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                                             "records":source,"prompts":prompts})
    def sampling(n):
        return SamplingParams(temperature=0,max_tokens=n,ignore_eos=True,seed=args.seed,
                              output_kind=RequestOutputKind.CUMULATIVE,detokenize=False)
    meter=None
    if not args.no_energy and not args.traceonly:
        from power import PowerSampler
        device=int(args.nvml_device) if args.nvml_device.isdigit() else args.nvml_device
        meter=PowerSampler(device=device)
        write_json(args.output_dir/"power_device.json",meter.device)
    completed=[]
    try:
        if args.capture_traces or args.traceonly:
            for b in args.batches:
                folder=args.output_dir/f"trace_native_b{b}";folder.mkdir()
                one_worker(llm,"skip_set_policy",None,"decode",len(installation["layers"])*args.trace_steps,False)
                batch=run_batch(engine,prompts[:b],sampling(args.trace_steps+1),f"trace-b{b}")
                snapshot=one_worker(llm,"skip_snapshot")
                write_json(folder/"snapshot.json",snapshot)
                write_json(folder/"outputs.json",batch)
                print(json.dumps({"kind":"trace","batch":b,"rows":len(snapshot["trace"])}),flush=True)
        if args.traceonly:
            write_json(args.output_dir/"status.json",{"status":"COMPLETE","kind":"trace_only","batches":args.batches})
            return
        rng=random.Random(args.seed)
        for repeat in range(args.repeats):
            order=[(b,a) for b in args.batches for a in arms];rng.shuffle(order)
            for order_index,(b,arm) in enumerate(order):
                tag=f"r{repeat:02d}_b{b:02d}_{arm['name']}";folder=args.output_dir/tag;folder.mkdir()
                one_worker(llm,"skip_set_policy",arm["spec"],"decode",0,False)
                warm_start=time.perf_counter();warm_rounds=0
                while warm_rounds==0 or time.perf_counter()-warm_start<args.warmup_seconds:
                    run_batch(engine,prompts[:b],sampling(args.warmup_tokens),f"warm-{tag}-{warm_rounds}",
                              failure_path=folder/"failed_warmup.json")
                    warm_rounds+=1
                warm_elapsed=time.perf_counter()-warm_start
                # Clear warmup counts without rebuilding policy allocations/JIT state.
                one_worker(llm,"skip_snapshot",True)
                result=run_batch(engine,prompts[:b],sampling(args.newtokens),f"perf-{tag}",
                                 failure_path=folder/"failed_performance.json")
                snap=one_worker(llm,"skip_snapshot")
                write_json(folder/"performance.json",result);write_json(folder/"counters.json",snap)
                logical_snapshot=snap
                counter_source="same_performance_pass"
                if graph:
                    runtime=[r for r in snap["graph"]["runtime_rows"] if r["phase"]=="decode"]
                    if len(runtime)!=args.newtokens-1 or any(r["runtime_mode"]!="FULL" or
                        r["actual_tokens"]!=b or r["padded_tokens"]!=b for r in runtime):
                        raise RuntimeError("graph performance requires FULL replay with no padding for every decode step")
                    counter_source=("separate_graph_counter_process" if args.graph_counter_pass else
                                    "pending_separate_graph_counter_process")
                native_global=(arm["spec"] or {}).get("mode")=="global_native"
                if not graph and (arm["spec"] is None or (native_global and not arm["spec"].get("count_routes",True))):
                    # Keep postprocessing and counters out of the native TPOT pass.
                    # Re-run the exact workload with dense identity counters, and
                    # require identical tokens before associating its byte counts.
                    counted_spec=({**arm["spec"],"count_routes":True} if native_global else
                                  {"mode":"dense","backend":args.backend})
                    counter_prefix="global_native" if native_global else "native"
                    one_worker(llm,"skip_set_policy",counted_spec,"decode",0,False)
                    counted=run_batch(engine,prompts[:b],sampling(args.newtokens),f"native-count-{tag}",
                                      failure_path=folder/"failed_native_counter.json")
                    logical_snapshot=one_worker(llm,"skip_snapshot")
                    write_json(folder/f"{counter_prefix}_counters.json",logical_snapshot)
                    write_json(folder/f"{counter_prefix}_counter_outputs.json",counted)
                    same=[r["token_ids"] for r in counted["requests"]]==[r["token_ids"] for r in result["requests"]]
                    write_json(folder/f"{counter_prefix}_counter_identity.json",{"identical_output_tokens":same,
                        "byte_semantics":"separate identical-token counter pass; native timing excludes this pass; compare actual footprint against the dense native counter pass"})
                    if not same:raise RuntimeError("native byte-counter replay changed output tokens")
                    counter_source=f"separate_{counter_prefix}_identity_pass_same_tokens"
                    one_worker(llm,"skip_set_policy",arm["spec"],"decode",0,False)
                energy=None
                if meter is not None:
                    one_worker(llm,"skip_snapshot",True)
                    energy=run_batch(engine,prompts[:b],sampling(args.newtokens),f"energy-{tag}",meter=meter,
                                     failure_path=folder/"failed_energy.json")
                    write_json(folder/"energy.json",energy)
                    write_json(folder/"energy_counters.json",one_worker(llm,"skip_snapshot"))
                row={"repeat":repeat,"order_index":order_index,"batch":b,"arm":arm["name"],
                     "execution":args.execution,"graph_counter_pass":args.graph_counter_pass,
                     "diagnostic":arm.get("diagnostic",False),"spec":arm["spec"],"warmup_rounds":warm_rounds,
                     "warmup_s":warm_elapsed,"mean_tpot_ms":result["mean_tpot_ms"],
                     "p95_inter_token_ms":result["p95_inter_token_ms"],"decode_output_tokens":result["decode_output_tokens"],
                     "logical_counters":logical_snapshot.get("layers",{}).get("totals"),
                     "logical_counter_source":counter_source,
                     "decode_j_per_token":energy["power"]["joules_per_decode_token"] if energy else None,
                     "decode_counter_j_per_token":energy["power"]["counter_joules_per_decode_token"] if energy else None,
                     "decode_energy_primary_method":"nvml_power_trapezoid" if energy else None,
                     "energy_window_s":energy["power"]["boundary"]["workload_elapsed_s"] if energy else None,
                     "energy_minimum_3s_met":energy["power"]["minimum_3s_window_met"] if energy else None,
                     "output_tokens_sha256":hashlib.sha256(json.dumps([r["token_ids"] for r in result["requests"]],separators=(",",":")).encode()).hexdigest(),
                     "folder":tag}
                append_json(args.output_dir/"measurements.jsonl",row);completed.append(row)
                print(json.dumps(row,allow_nan=False),flush=True)
        identity=[]
        for rep in range(args.repeats):
            for b in args.batches:
                rows={r["arm"]:r for r in completed if r["repeat"]==rep and r["batch"]==b}
                if {"native","dense_counted"}.issubset(rows):
                    match=rows["native"]["output_tokens_sha256"]==rows["dense_counted"]["output_tokens_sha256"]
                    identity.append({"repeat":rep,"batch":b,"identical_output_tokens":match})
        write_json(args.output_dir/"dense_identity.json",identity)
        if any(not x["identical_output_tokens"] for x in identity):
            raise RuntimeError("dense counted identity changed output tokens; inspect dense_identity.json")
        write_json(args.output_dir/"status.json",{"status":"COMPLETE","completed_cells":len(completed),"dense_identity_checks":identity})
    except BaseException as error:
        write_json(args.output_dir/"status.json",{"status":"INCOMPLETE","completed_cells":len(completed),
                                               "error":f"{type(error).__name__}: {error}"})
        raise
    finally:
        if meter is not None:meter.close()
        one_worker(llm,"skip_set_policy",arms[0]["spec"] if graph else None,"decode",0,False)


if __name__=="__main__":main()
