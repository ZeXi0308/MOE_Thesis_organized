"""Read-only analysis of retained native pager calls; no inferred missing routes/LRU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def requests(raw, workload):
    result, issues = {}, []
    source = {r["request_id"]:(r, ids) for r,ids in zip(workload["source_requests"],workload["actual_prompt_token_ids"])}
    rebuilt, times = {rid:[] for rid in source}, {rid:[] for rid in source}
    for event in raw["output_events"]:
        rid=event["request_id"]; old=rebuilt[rid]; tokens=event["cumulative_token_ids"]
        if tokens[:len(old)]!=old or len(tokens)-len(old)!=event["chunk_size"]: issues.append("output prefix/chunk mismatch")
        times[rid].extend([event["received_s"]]*(len(tokens)-len(old))); rebuilt[rid]=tokens
    for row in raw["requests"]:
        rid=row["request_id"]; src,ids=source[rid]; ts=row["token_times_s"]; gaps=[b-a for a,b in zip(ts,ts[1:])]
        digest=hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()
        valid=(row["document_id"]==src["document_id"] and row["prompt_token_ids_sha256"]==digest
            and row["prompt_tokens"]==len(ids) and rebuilt[rid]==row["output_token_ids"] and times[rid]==ts
            and len(ts)==len(row["output_token_ids"]) and all(g>=0 for g in gaps) and ts[0]>=row["arrival_s"])
        if rid in result or not valid: issues.append("request identity/timing mismatch: "+rid)
        result[rid]=dict(status=row["status"],identity_and_event_times_valid=valid,arrival_s=row["arrival_s"],
            output_token_ids=row["output_token_ids"],token_times_s=ts,ttft_s=ts[0]-row["arrival_s"],
            tpot_s=(ts[-1]-ts[0])/(len(ts)-1) if len(ts)>1 else None,max_itl_s=max(gaps,default=None),
            completion_latency_s=row["completion_s"]-row["arrival_s"],client_submission_lag_s=row["admission_s"]-row["arrival_s"])
    if set(result)!=set(source): issues.append("request coverage mismatch")
    return result,issues


def signature(records, aliases):
    return [dict(layer=r["layer_name"],rows=r["rows"],phase=r["context"]["phase"],step=r["context"]["step_id"],
        physical_rows=[(aliases.get(x["internal_request_id"],x["internal_request_id"]),x["computed_position"])
                       for x in r["context"].get("rows") or []],
        groups=[{k:g.get(k) for k in ("start","stop","unique_experts","required_experts","loaded_experts",
                                      "reloaded_experts","evicted_experts","miss","evict","weight_copy_bytes")}
                for g in r["groups"]]) for r in records]


def call_accounting(records, layer_bytes):
    out=[]
    for r in records:
        seq=[e for g in r["groups"] for e in g["loaded_experts"]]; size=layer_bytes[r["layer_name"]]
        actual=len(seq)*size; repeated=(len(seq)-len(set(seq)))*size
        out.append(dict(call_id=r["call_id"],layer_name=r["layer_name"],step_id=r["context"]["step_id"],
            status=r["status"],physical_rows=r["rows"],groups=len(r["groups"]),actual_bytes=actual,
            distinct_loaded_bytes=len(set(seq))*size,within_call_repeated_load_bytes=repeated,
            unique_entry_miss_lower_bound_bytes=None,extra_over_unique_entry_lower_bound_bytes=None,
            load_cuda_span_ms=sum(g["load_cuda_span_ms"] for g in r["groups"]),host_apply_ms=r["host_apply_ms"],
            route_to_host_ms=r["route_to_host_ms"],byte_count_valid=actual==r["weight_copy_bytes"] and
            all(len(g["loaded_experts"])==g["miss"] and g["weight_copy_bytes"]==g["miss"]*size for g in r["groups"])))
    return out


def totals(records):
    return {key:sum(r[key] for r in records) for key in ("actual_bytes","within_call_repeated_load_bytes",
            "distinct_loaded_bytes","groups","load_cuda_span_ms","host_apply_ms","route_to_host_ms")}


def cell(path):
    raw=read(path/"raw.json"); workload=read(path/"workload.json"); pager=read(path/"pager_summary.json")
    records=[json.loads(line) for line in (path/"pager/calls.jsonl").read_text().splitlines() if line]
    measured=[r for r in records if r["measurement"]]; pre=[r for r in records if not r["measurement"]]
    warmup=read(path/"warmup_full.json"); mini=read(path/"warmup.json")
    aliases={**mini["internal_to_source"],**warmup["internal_to_source"],**raw["internal_to_source"]}
    layer_bytes={r["layer_name"]:r["pinned_bytes"]//r["num_experts"] for r in pager["layers"]}
    accounted=call_accounting(measured,layer_bytes); reqs,issues=requests(raw,workload); steps=[]
    for i,s in enumerate(raw["scheduler_steps"]):
        calls=[r for r in measured if r["context"]["step_id"]==s["step"]]
        rows=[r for r in accounted if r["step_id"]==s["step"]]
        expected=Counter((r["request_id"],pos) for r in s["scheduled"] for pos in range(r["scheduled_start_computed"],r["computed_after"]))
        aligned=len(calls)==len(pager["layers"]) and all(Counter((aliases.get(x["internal_request_id"]),x["computed_position"])
            for x in r["context"].get("rows") or [])==expected and r["rows"]==s["total_scheduled_tokens"] for r in calls)
        if not aligned: issues.append("native physical row alignment: "+str(s["step"]))
        next_start=raw["scheduler_steps"][i+1]["start_s"] if i+1<len(raw["scheduler_steps"]) else raw["observation_end_s"]
        receipt=[e["received_s"] for e in raw["output_events"] if s["start_s"]<=e["received_s"]<next_start]
        steps.append(dict(step=s["step"],scheduled_tokens=s["total_scheduled_tokens"],physical_alignment_valid=aligned,
            scheduled=[{k:r[k] for k in ("request_id","scheduled_start_computed","computed_after","scheduled_tokens","prefill_tokens","decode_tokens")} for r in s["scheduled"]],
            engine_call_wall_s=None,scheduler_to_last_receipt_s=max(receipt)-s["start_s"] if receipt else None,
            scheduler_to_next_start_or_observation_end_s=next_start-s["start_s"],**totals(rows)))
    resource=read(path/"resources.json"); env=read(path/"environment.json")
    resources=dict(observed_pool_blocks=resource["pool_num_gpu_blocks"],observed_free_blocks_after_init=resource["free_blocks_after_init"],
        block_size=resource["block_size"],requested_kv_bytes=resource["kv_cache_memory_bytes"],actual_kv_tensor_storage_bytes=None,
        expert_cap=resource["expert_cap"],scratch_bytes=sum(r["scratch_bytes"] for r in pager["layers"]),
        pinned_master_bytes=sum(r["pinned_bytes"] for r in pager["layers"]),layer_allocations=pager["layers"])
    report=dict(status=raw["status"],path=str(path),config=read(path/"config.json"),environment=env,issues=issues,
        resources=resources,allocator_observations=resource,engine_args=read(path/"engine_args.json"),
        policy_application=read(path/"policy_application.json"),request_count=len(reqs),requests=reqs,
        completed_count=sum(r["status"]=="completed" for r in reqs.values()),generated_tokens=sum(len(r["output_token_ids"]) for r in reqs.values()),
        whole_observation_wall_s=raw["observation_end_s"],completion_makespan_s=max(r["completion_s"] for r in raw["requests"])-min(r["arrival_s"] for r in raw["requests"]),
        recorded_fields=dict(measurement_initial_cache="measurement_initial_cache" in pager,all_required_expert_sets=all("required_experts" in g for r in measured for g in r["groups"])),
        exact_initial_cache_lru_status="UNAVAILABLE: measurement_initial_cache not recorded",unique_entry_lower_bound_status="UNAVAILABLE: required expert sets and entry residency not recorded",
        warmup_full_wall_s=warmup["observation_end_s"],warmup_full_request_outputs={r["request_id"]:r["output_token_ids"] for r in warmup["requests"]},
        premeasurement_totals=totals(call_accounting(pre,layer_bytes)),episode_totals=totals(accounted),first_call=steps[0],steps=steps,layer_calls=accounted,
        by_layer={name:dict(totals([r for r in accounted if r["layer_name"]==name]),unique_entry_miss_lower_bound_bytes=None,extra_over_unique_entry_lower_bound_bytes=None) for name in layer_bytes},preemptions=raw["preemption_summary"],
        unavailable=["exact route identity","cache slot/map/LRU state","exact entry-miss unique lower bound","engine.step entry/return timestamps","measured KV tensor storage bytes"],
        timing_scope="Load CUDA span and host_apply overlap; route_to_host is nested in host_apply. Do not add them. Scheduler-to-next-start intervals include client submission/bookkeeping.")
    if not all(r["byte_count_valid"] and r["status"]=="complete" for r in accounted): issues.append("pager byte/status mismatch")
    if report["episode_totals"]["actual_bytes"]!=pager["measurement"]["weight_copy_bytes"]: issues.append("summary versus raw group bytes mismatch")
    return report,dict(measured=signature(measured,aliases),pre=signature(pre,aliases),workload=workload)


def compare(a,b,ra,rb,na,nb):
    return dict(a=a,b=b,same_workload=na["workload"]==nb["workload"],same_resources=ra["resources"]==rb["resources"],
        same_recorded_source_hashes=ra["environment"]["sources"]==rb["environment"]["sources"],
        common_warmup_observed_structure_equal=na["pre"]==nb["pre"],common_warmup_outputs_equal=ra["warmup_full_request_outputs"]==rb["warmup_full_request_outputs"],
        exact_initial_cache_lru_equal=None,exact_routes_equal=None,
        measured_rows_groups_misses_victims_equal=na["measured"]==nb["measured"],
        output_tokens_equal={rid:ra["requests"][rid]["output_token_ids"]==rb["requests"][rid]["output_token_ids"] for rid in ra["requests"]},
        whole_wall_delta_b_minus_a_s=rb["whole_observation_wall_s"]-ra["whole_observation_wall_s"],
        whole_wall_change_b_over_a_pct=100*(rb["whole_observation_wall_s"]/ra["whole_observation_wall_s"]-1),
        episode_delta_b_minus_a={k:rb["episode_totals"][k]-ra["episode_totals"][k] for k in ra["episode_totals"]})


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input-dir",required=True,type=Path); p.add_argument("--out",required=True,type=Path); args=p.parse_args()
    execution=read(args.input_dir/"execution.json"); reports,normalized={},{}
    for entry in execution["cells"]:
        name=entry["label"]; reports[name],normalized[name]=cell(args.input_dir/name); reports[name]["execution"]=entry
    pairs=[("default_r0","prefill32_r0"),("default_r1","prefill32_r1"),("default_r0","default_r1"),("prefill32_r0","prefill32_r1")]
    comparisons=[compare(a,b,reports[a],reports[b],normalized[a],normalized[b]) for a,b in pairs]
    output=dict(schema_version=1,cells=reports,comparisons=comparisons,total_request_executions=sum(r["request_count"] for r in reports.values()),
        unique_request_documents=len(set().union(*(set(r["requests"]) for r in reports.values()))),
        workload_scope="4 requests per cell, 4 cells: 16 executions of 4 documents; 64 input/8 output tokens, native eager vLLM0.26, cap24",
        repeated_bytes_scope="Within each layer-call, count repeated appearances in actual loaded_experts lists. This is observed repeated traffic, not total extra over an unavailable entry-cache lower bound. Runtime reloaded_experts instead means ever loaded across past calls.",
        initial_state_scope="Matching warmup configurations and observed groups/misses/outputs do not establish unrecorded exact cache/LRU or full required expert sets.")
    with args.out.open("x") as f: json.dump(output,f,indent=2,allow_nan=False); f.write("\n")


if __name__=="__main__": main()
