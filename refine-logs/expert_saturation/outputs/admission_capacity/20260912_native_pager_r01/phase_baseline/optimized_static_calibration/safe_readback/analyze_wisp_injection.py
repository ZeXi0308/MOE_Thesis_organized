"""Recompute tiny injection results; retain failed cells, warmup and every call."""
import argparse
import json
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return dict(status="MISSING_OR_UNREADABLE", error=str(error))


def request_metrics(row):
    times, arrival = row.get("token_received_s", []), row.get("arrival_s")
    gaps = [b-a for a, b in zip(times, times[1:])]
    completion = row.get("completion_s")
    return dict(status=row.get("status", row.get("finished")), output_count=len(row.get("output_token_ids", [])),
        timing_count=len(times), token_times_s=times, arrival_s=arrival, completion_s=completion,
        ttft_s=times[0]-arrival if times and arrival is not None else None,
        tpot_s=(times[-1]-times[0])/(len(times)-1) if len(times)>1 else None,
        max_itl_s=max(gaps, default=None), completion_latency_s=completion-arrival
        if completion is not None and arrival is not None else None,
        timing_valid=len(times)==len(row.get("output_token_ids", [])) and all(g>=0 for g in gaps)
        and (not times or arrival is not None and times[0]>=arrival))


def call_metrics(calls, trace):
    rows = {c["index"]: dict(index=c["index"], returned=c.get("returned"), error=c.get("error"),
        wall_s=c["return_s"]-c["start_s"] if "return_s" in c else None,
        scheduled=c.get("scheduled"), tokens=c.get("total_scheduled_tokens"),
        layers=0, actual_bytes=0, unique_bytes=0, extra_bytes=0, load_section_ms=0., issues=[], needed_by_layer={}, group_needs_by_layer={}, _host=[]) for c in calls}
    orphan = []
    for layer in trace.get("layers", []):
        index = layer.get("engine_call_index")
        if index not in rows:
            orphan.append(dict(engine_call_index=index, layer_idx=layer.get("layer_idx"))); continue
        out, groups, size = rows[index], layer.get("subgroups", []), layer["expert_weight_bytes"]
        needed = set().union(*(set(g["required_experts"]) for g in groups))
        actual = sum(len(g.get("loaded_experts", []))*size for g in groups)
        lower = len(needed-set(layer["resident_at_entry"]))*size
        out["layers"] += 1
        out["needed_by_layer"][layer["layer_idx"]] = sorted(needed)
        out["group_needs_by_layer"][layer["layer_idx"]] = [g["required_experts"] for g in groups]
        out["_host"].extend((h["start_ns"], h["end_ns"]) for h in layer.get("host_read_spans", []))
        for key, value in (("actual_bytes", actual), ("unique_bytes", lower), ("extra_bytes", actual-lower)):
            out[key] += value
        for group in groups:
            value = group.get("load_section_cuda_ms")
            if value is None: out["issues"].append("unresolved load section")
            else: out["load_section_ms"] += value
        if not layer.get("successful") or actual != layer.get("actual_load_bytes") or lower != layer.get("unique_lower_bound_bytes"):
            out["issues"].append(f"layer {layer.get('layer_idx')} incomplete/accounting mismatch")
    for out in rows.values():
        end, total = 0, 0
        for a,b in sorted(out.pop("_host")):
            total += max(0, b-max(a,end)); end=max(end,b)
        out["host_read_union_s"] = total/1e9
        out["wall_outside_host_read_spans_s"] = out["wall_s"]-total/1e9 if out["wall_s"] is not None else None
        if out["layers"]==0:
            out["issues"].append("TRACE_UNAVAILABLE")
            for key in ("actual_bytes","unique_bytes","extra_bytes","load_section_ms","needed_by_layer","group_needs_by_layer","host_read_union_s","wall_outside_host_read_spans_s"): out[key]=None
        elif "unresolved load section" in out["issues"]: out["load_section_ms"]=None
    return dict(calls=list(rows.values()), orphan_layers=orphan, trace_present=bool(trace),
                duplicate_call_indices=len(rows)!=len(calls), timing_resolved=trace.get("timing_resolved"))


def resources(raw):
    return dict(config=raw.get("engine_config"), runtime=raw.get("runtime"), workers=[dict(
        kv_bytes=w.get("kv_allocated_tensor_bytes"), kv_blocks=w.get("num_gpu_blocks"),
        kv_groups=w.get("kv_groups"), pager_storage=w.get("pager_storage"), layers=[
            {k:s.get(k) for k in ("layer_idx", "cap", "num_experts", "mode", "master_shapes", "master_pinned", "master_bytes", "scratch_bytes")}
            for s in w.get("pager_layers", [])]) for w in raw.get("action", {}).get("before_worker", raw.get("initialization", []))])


def summarize(raw, execution):
    requests, action, steps = raw.get("requests", {}), raw.get("action", {}), raw.get("steps", [])
    old = list(action.get("old_output_tokens", {})); new = [rid for rid in requests if rid not in old]
    index = action.get("engine_call"); first = next((s for s in steps if s["index"]==index), {})
    metrics = {rid:request_metrics(row) for rid, row in requests.items()}
    expected = 2 if raw.get("args", {}).get("no_new") else 2+min(raw.get("args", {}).get("chunk", 0), raw.get("args", {}).get("new_prompt_length", 0))
    arrivals = [r["arrival_s"] for r in metrics.values() if r["arrival_s"] is not None]
    completed = [r["completion_s"] for r in metrics.values() if r["completion_s"] is not None]
    old_done = max((metrics[r]["completion_s"] for r in old if metrics[r]["completion_s"] is not None), default=None)
    warmup = raw.get("warmup", {})
    return dict(status=raw.get("status", "MISSING"), error=raw.get("error"), execution=execution,
        process_wall_s=execution.get("end_unix", 0)-execution["start_unix"] if "end_unix" in execution and "start_unix" in execution else None,
        resources=resources(raw), requests=metrics, old_ids=old, new_ids=new,
        identity_lengths={rid:len(r.get("prompt_token_ids", [])) for rid,r in requests.items()}, identity_valid=len(old)==2 and len(new)==1 and all(len(requests[r].get("prompt_token_ids", []))==32 for r in old) and len(requests[new[0]].get("prompt_token_ids", []))==raw.get("args", {}).get("new_prompt_length"),
        first_action_index=index, first_action_step_s=first.get("return_s", 0)-first["start_s"] if "return_s" in first else None,
        first_action_tokens=first.get("total_scheduled_tokens"), first_shape_valid=first.get("total_scheduled_tokens")==expected and len(old)==2 and all(next((s["tokens"] for s in first.get("scheduled", []) if s["request_id"]==r), None)==1 for r in old),
        old_cross_action_itl_s={rid:metrics[rid]["token_times_s"][4]-metrics[rid]["token_times_s"][3]
            if len(metrics[rid]["token_times_s"])>4 else None for rid in old},
        preaction_old_prefix_valid=len(old)==2 and all(len(action["old_output_tokens"][r])==4 and
            action["old_output_tokens"][r]==requests[r].get("output_token_ids", [])[:4] for r in old),
        action_snapshot_wall_s=action.get("snapshot_end_s", 0)-action["snapshot_start_s"] if "snapshot_end_s" in action else None,
        action_wall_s=action.get("end_s", 0)-action["start_s"] if "end_s" in action else None,
        all_request_makespan_s=max(completed)-min(arrivals) if completed and arrivals else None,
        recorded_wall_s=raw.get("wall_s"), post_old_completion_calls=[s["index"] for s in steps if old_done is not None and s.get("start_s", 0)>old_done],
        complete_expected_requests=bool(requests) and all(r.get("status")=="NOT_INJECTED" or
            r.get("status")=="COMPLETED" and len(r.get("output_token_ids", []))==r.get("max_tokens") for r in requests.values()),
        scheduler_violations=[s["index"] for s in steps if s.get("preempted_request_ids") or any(r.get("computed_adjustment") for r in s.get("scheduled", []))],
        measurement=call_metrics(steps, raw.get("paging_trace", raw.get("partial_trace", {}))),
        warmup=dict(requests=[request_metrics(r) for r in warmup.get("requests", [])],
            recorded_wall_s=warmup.get("wall_s"), measurement=call_metrics(warmup.get("engine_calls", []), warmup.get("paging_trace", {}))),
        compilation="No isolated compile timer; process/warmup/first-action timings retained without exclusions")


def comparison(a, b, sa, sb, relation):
    aa, bb = a.get("action", {}), b.get("action", {})
    pa = [s for w in aa.get("before_worker", []) for s in w.get("pager_execution_state", [])]
    pb = [s for w in bb.get("before_worker", []) for s in w.get("pager_execution_state", [])]
    fields = lambda rows, keys: [{k:r.get(k) for k in keys} for r in rows]
    order = lambda rows: [sorted(range(len(r["lru_tick"])), key=lambda i:(r["lru_tick"][i], i)) for r in rows]
    relative = lambda rows: [[t-r["lru_clock"] for t in r["lru_tick"]] for r in rows]
    ar, br = a.get("requests", {}), b.get("requests", {})
    same_inputs = bool(ar) and ar.keys()==br.keys() and bool(a.get("args", {}).get("no_new"))==bool(b.get("args", {}).get("no_new")) and all(ar[r].get("document_id")==br[r].get("document_id") and
        ar[r].get("prompt_token_ids")==br[r].get("prompt_token_ids") and ar[r].get("max_tokens")==br[r].get("max_tokens") for r in ar)
    ca={c["index"]:c for c in sa["measurement"]["calls"]}; cb={c["index"]:c for c in sb["measurement"]["calls"]}; joined=[]
    for index in sorted(ca.keys()|cb.keys()):
        l,r=ca.get(index,{}),cb.get(index,{})
        observed=bool(l.get("layers")) and bool(r.get("layers"))
        joined.append(dict(index=index,in_both=bool(l) and bool(r), same_scheduled=l.get("scheduled")==r.get("scheduled") if l.get("scheduled") is not None and r.get("scheduled") is not None else None, same_expert_unions=l.get("needed_by_layer")==r.get("needed_by_layer") if observed else None, same_subgroup_experts=l.get("group_needs_by_layer")==r.get("group_needs_by_layer") if observed else None, same_bytes=all(l.get(k)==r.get(k) for k in ("actual_bytes","unique_bytes","extra_bytes")) if observed else None, delta_b_minus_a={k:r[k]-l[k] if l.get(k) is not None and r.get(k) is not None else None for k in ("wall_s","load_section_ms","host_read_union_s","wall_outside_host_read_spans_s")}))
    return dict(relation=relation, resources_same=bool(sa["resources"]["workers"]) and sa["resources"]==sb["resources"],
        call_diagnostics=joined,
        same_input_tasks=same_inputs, old_tokens_same=bool(aa.get("old_output_tokens")) and aa.get("old_output_tokens")==bb.get("old_output_tokens"),
        scheduler_prestate_same=bool(aa.get("before_scheduler")) and aa.get("before_scheduler")==bb.get("before_scheduler"),
        full_pager_state_same=bool(pa) and pa==pb, slot_maps_same=bool(pa) and fields(pa,("layer_idx","slot_to_expert","expert_to_slot"))==fields(pb,("layer_idx","slot_to_expert","expert_to_slot")),
        lru_clocks_a=[r["lru_clock"] for r in pa], lru_clocks_b=[r["lru_clock"] for r in pb],
        relative_lru_ticks_same=bool(pa) and relative(pa)==relative(pb), relative_lru_order_same=bool(pa) and order(pa)==order(pb),
        delta_b_minus_a={key:sb[key]-sa[key] if sa[key] is not None and sb[key] is not None else None
            for key in ("first_action_step_s", "action_snapshot_wall_s", "all_request_makespan_s", "recorded_wall_s")},
        old_cross_itl_delta_b_minus_a={r:sb["old_cross_action_itl_s"][r]-v if v is not None and sb["old_cross_action_itl_s"].get(r) is not None else None
            for r,v in sa["old_cross_action_itl_s"].items()})


def hardware_summary(path, raw, expected_cpu_affinity=None):
    start, duration = raw.get("measurement_origin_unix_s"), raw.get("wall_s")
    if start is None or duration is None or not path.exists(): return dict(status="UNAVAILABLE")
    try:
        samples=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        window=[s for s in samples if start<=s["unix_s"]<=start+duration]
        if not window: return dict(status="NO_SAMPLES_IN_MEASUREMENT_WINDOW", start_unix_s=start, end_unix_s=start+duration)
        clocks={}
        for key in ("sm_clock_mhz","memory_clock_mhz","temperature_c","power_mw","gpu_util_pct"):
            values=[s[key] for s in window if isinstance(s.get(key),(int,float))]
            clocks[key]=dict(samples=len(values), minimum=min(values), maximum=max(values), mean=sum(values)/len(values)) if values else None
        counters=[dict(line.split() for line in s.get("cgroup_cpu_stat", "").splitlines()) for s in window]
        cpu={}
        for key in ("nr_throttled","throttled_usec","usage_usec"):
            a,b=counters[0].get(key),counters[-1].get(key)
            cpu[key]=dict(first=int(a),last=int(b),delta=int(b)-int(a)) if a is not None and b is not None else None
        before={w["pid"] for w in raw.get("action",{}).get("before_worker",[]) if "pid" in w}
        after={w["pid"] for w in raw.get("final_worker",[]) if "pid" in w}; expected=before|after
        pid_rows=[s["compute_pids"] for s in window if isinstance(s.get("compute_pids"),list)]
        isolation=dict(expected_worker_pids=sorted(expected), worker_pids_unchanged=bool(before) and before==after,
            samples_with_pid_info=len(pid_rows), unexpected_compute_pids=sorted(set().union(*(set(r) for r in pid_rows))-expected),
            all_samples_exactly_expected=all(set(r)==expected for r in pid_rows) if expected and len(pid_rows)==len(window) else None)
        placement=[]
        for pid in sorted(expected):
            observations=[p for s in window for p in s.get("process_placement",[]) if p.get("pid")==pid]
            fields=[dict(line.split(":",1) for line in p.get("affinity",[]) if ":" in line) for p in observations]
            masks=sorted({f["Cpus_allowed_list"].strip() for f in fields if "Cpus_allowed_list" in f})
            expand=lambda mask: sorted({n for token in mask.split(",") for n in (range(int(token.split("-")[0]),int(token.split("-")[-1])+1))})
            pages=[p["anon_numa_pages"] for p in observations if isinstance(p.get("anon_numa_pages"),dict)]
            node_ranges={node:dict(minimum=min(p.get(node,0) for p in pages), maximum=max(p.get(node,0) for p in pages)) for node in set().union(*(set(p) for p in pages))}
            placement.append(dict(pid=pid,samples=len(observations),cpu_allowed_lists=masks,
                mems_allowed_lists=sorted({f["Mems_allowed_list"].strip() for f in fields if "Mems_allowed_list" in f}),
                expected_cpu_affinity=expected_cpu_affinity,all_sampled_affinity_matches=all(expand(mask)==sorted(expected_cpu_affinity) for mask in masks)
                if masks and len(fields)==len(window) and all("Cpus_allowed_list" in f for f in fields) and expected_cpu_affinity is not None else None,
                anon_numa_pages=node_ranges,first_anon_numa_pages=pages[0] if pages else None,last_anon_numa_pages=pages[-1] if pages else None))
        return dict(status="SAMPLED", samples=len(window), measurement_start_unix_s=start, measurement_end_unix_s=start+duration,
            mean_sample_interval_s=(window[-1]["unix_s"]-window[0]["unix_s"])/(len(window)-1) if len(window)>1 else None,
            first_sample_unix_s=window[0]["unix_s"],last_sample_unix_s=window[-1]["unix_s"], clocks=clocks,cgroup_cpu=cpu,gpu_process_isolation=isolation,process_placement=placement,
            boundary="Counter delta covers first-to-last samples inside measurement; unsampled edges remain. Clock/PID samples cannot exclude between-sample excursions. Anonymous pages are process totals, not tensor-specific pinned-master placement; affinity is sampled process allowlist.")
    except (ValueError,KeyError,OSError) as error: return dict(status="UNREADABLE",error=str(error))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input-dir", required=True, type=Path); p.add_argument("--out", required=True, type=Path)
    args=p.parse_args(); paths=set(args.input_dir.rglob("result.json")); executions={}; execution_files={}; declarations={}; protocols={}
    for filename,target in (("protocol.json",protocols),("execution.json",execution_files)):
        for path in sorted(args.input_dir.rglob(filename)):
            target[str(path)]=read(path)
            for row in target[str(path)].get("cells", []):
                result=path.parent/row["cell"]/"result.json"; paths.add(result); declarations[str(result.relative_to(args.input_dir).parent)]=row
                if filename=="execution.json": executions[result]=row
    raw={str(path.relative_to(args.input_dir).parent):read(path) for path in sorted(paths)}
    summaries={name:summarize(value, executions.get(args.input_dir/name/"result.json", {})) for name,value in raw.items()}
    source={r["request_id"]:r for r in read(args.input_dir/"workload.json").get("requests", [])}
    for name,value in raw.items():
        summaries[name]["workload_identity_matches"]=len(source)==3 and [len(r["prompt_token_ids"]) for r in source.values()]==[32,32,128] and value.get("requests", {}).keys()==source.keys() and all(r.get("document_id")==source[rid]["document_id"] and r.get("prompt_token_ids")==source[rid]["prompt_token_ids"][:len(r.get("prompt_token_ids", []))] for rid,r in value.get("requests", {}).items())
        summaries[name]["hardware_summary"]=hardware_summary(args.input_dir/name/"hardware.jsonl",value,read((args.input_dir/name).parent/"protocol.json").get("cpu_affinity"))
    missing=[name for name in declarations if not (args.input_dir/name/"result.json").is_file()]
    comparisons=[]; groups={}
    for name,row in declarations.items():
        arm=row.get("name") or ("hold" if row.get("no_new") else ("short" if row.get("new_prompt_length",128)<128 else "long")+str(row.get("chunk")))
        if row.get("repeat") is not None: groups.setdefault((row["repeat"],arm),[]).append(name)
    named={key:values[0] for key,values in groups.items() if len(values)==1}
    for repeat in sorted({r for r,arm in groups}):
        for left,right,relation in (("long8","long32","same-task chunk intervention"),("short8","long8","first-step negative control only"),("hold","long8","two-request hold diagnostic; unequal task")):
            a,b=named.get((repeat,left)),named.get((repeat,right))
            if a and b:
                row=comparison(raw[a],raw[b],summaries[a],summaries[b],relation); row.update(a=a,b=b)
                if left=="short8":
                    ra,rb=raw[a].get("requests", {}),raw[b].get("requests", {})
                    row["same_documents_and_prompt_prefix"]=bool(ra) and ra.keys()==rb.keys() and all(ra[r].get("document_id")==rb[r].get("document_id") and ra[r].get("prompt_token_ids")==rb[r].get("prompt_token_ids", [])[:len(ra[r].get("prompt_token_ids", []))] for r in ra)
                comparisons.append(row)
    for arm in sorted({arm for repeat,arm in groups}):
        repeats=sorted(r for r,name in named if name==arm)
        for left,right in zip(repeats,repeats[1:]):
            a,b=named[(left,arm)],named[(right,arm)]
            row=comparison(raw[a],raw[b],summaries[a],summaries[b],"repeat; every call retained"); row.update(a=a,b=b); comparisons.append(row)
    output=dict(schema_version=2,cells=summaries, missing_declared_cells={n:"UNRUN_OR_MISSING" for n in missing}, duplicate_repeat_arms={str(n):k for n,k in groups.items() if len(k)>1}, protocols=protocols,executions=execution_files, comparisons=comparisons,
        scope="Both repeats and every observed attempt retained; no request SLO or production-tail claim. Bytes shared per batch. Host-read union and CUDA load section overlap: never sum. Wall outside host reads includes compute/host/observation; not an attributed cause.")
    with args.out.open("x") as handle: json.dump(output,handle,indent=2,allow_nan=False); handle.write("\n")


if __name__=="__main__": main()
