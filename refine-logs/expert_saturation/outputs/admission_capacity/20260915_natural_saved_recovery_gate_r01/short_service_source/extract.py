"""Extract only the two already-identified 1–2-output recovery segments."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
execution = here.parent / "execution_weste_26862"
analysis = json.loads((execution / "analysis.json").read_text())
base = execution / "readback/results/diagnostic-current"
raw = json.loads((base / "raw.json").read_text())
selective = json.loads((base / "selective-store.json").read_text())
offload = analysis["diagnostic"]["offload"]
mapping = raw["internal_to_source"]
rows = []
for segment in analysis["diagnostic"]["segments"]:
    if segment["end"] != "repreempted" or not 1 <= segment["useful_outputs"] <= 2:
        continue
    rid, first, end = segment["internal_request_id"], segment["executed_steps"][0]["step"], segment["next_preempt_step"]
    prepares = [e for e in selective["events"] if e.get("event") == "prepare" and e.get("target") == rid and segment["preempt_step"] < e["step"] <= first]
    store_prepare = [e for e in selective["events"] if e.get("event") == "prepare" and e.get("victim") == rid and e["step"] <= segment["preempt_step"]][-1]
    centers = [segment["preempt_step"], prepares[-1]["step"], first, end]
    nearby = lambda step: any(abs(step-center) <= 2 for center in centers)
    snapshots = []
    for snap in selective["eligibility_snapshots"]:
        if not nearby(snap["step"]):
            continue
        peers = []
        for peer in snap["running_ids"]:
            state = snap["requests"][peer]
            held = state["held_blocks"]
            held = held if isinstance(held, int) else sum(held)
            needed = max(0, (state["computed"]+1+15)//16-held)
            if needed:
                peers.append(dict(request_id=mapping[peer], computed=state["computed"], held_blocks=held, blocks_for_one_decode=needed))
        snapshots.append(dict(step=snap["step"], free_blocks=snap["free_blocks"],
            protected_id=snap["protected_id"], protected_reserve=snap["protected_reserve"],
            plan_target=snap["plan_target"], target=snap["requests"].get(rid), decode_growth_candidates=peers))
    preempt = next(p for p in raw["preemption_events"] if p["attempted_step"] == end and p["victim_internal_request_id"] == rid)
    schedule = next(s for s in raw["scheduler_steps"] if s["step"] == end)
    load_ids = {d["job_id"] for d in segment["actual_load_dispatches"]}
    store_jobs = [d for d in offload["dispatch"] if d["request"] == rid and d["is_store"] and d["accepted"] and d["before_perf_s"] < segment["S"]["time_s"]+raw["measurement_origin_perf_counter_s"]]
    retained_ids = load_ids | {store_jobs[-1]["job_id"]}
    rows.append(dict(request_id=segment["request_id"], useful_outputs=segment["useful_outputs"],
        preempt_step=segment["preempt_step"], first_executed_steps=segment["executed_steps"], next_preempt_step=end,
        last_store_prepare=store_prepare, accepted_store=store_jobs[-1], accepted_loads=segment["actual_load_dispatches"],
        completed_jobs=[dict(time_s=e["time_s"], job=j) for e in offload["completed_jobs"] for j in e["jobs"] if j["job_id"] in retained_ids],
        forced_recovery_prepare=prepares[-1],
        selective_events=[e for e in selective["events"] if nearby(e.get("step", -100)) and (e.get("target") == rid or e.get("victim") == rid or e.get("request") == rid)],
        nearby_snapshots=snapshots,
        nearby_selector_decisions=[e for e in selective["selector_decisions"] if abs(e["step"]-end) <= 2],
        next_preemption={k:v for k,v in preempt.items() if not k.startswith("output_token_ids")},
        next_preempt_schedule={k:v for k,v in schedule.items() if k != "scheduled"},
        next_preempt_scheduled=[dict(request_id=x["request_id"], computed_before=x["computed_before"], scheduled_tokens=x["scheduled_tokens"], decode_tokens=x["decode_tokens"], recompute_tokens=x["recompute_tokens"]) for x in schedule["scheduled"]]))
result = dict(source_analysis=str(execution / "analysis.json"), source_raw=str(base / "raw.json"),
    source_selective=str(base / "selective-store.json"), segments=rows,
    boundary="Only the two existing short segments and corresponding +/-2-step snapshots; historical save and completed-job identity links retained.",
    limitation="Failed allocate_slots call arguments are not logged. Peer block demand is derived from same-step states and confirmed decode scheduling, not a logged allocator caller.")
with (here / "source.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(here / "source.json")
