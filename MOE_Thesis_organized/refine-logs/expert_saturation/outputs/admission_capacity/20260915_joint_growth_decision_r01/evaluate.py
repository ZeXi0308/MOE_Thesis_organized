"""Fixed arithmetic resource forecast over every observed protection release."""
import argparse
import json
from collections import Counter
from pathlib import Path

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=here / "result.json")
args = parser.parse_args()
groups = {
    "D": "20260915_natural_saved_recovery_gate_r01",
    "E": "20260915_natural_native_full_gate_r01",
}

def confusion(rows, prediction, label):
    counts = Counter()
    for r in rows:
        counts[("T" if r[prediction] == r[label] else "F") + ("P" if r[prediction] else "N")] += 1
    return {key: counts[key] for key in ("TP", "FP", "FN", "TN")}

results = {"formula": "G(h)=sum(max(0,ceil((computed+h)/16)-held_blocks)); alarm=G(h)>free_blocks", "groups": {}}
for name, directory in groups.items():
    analysis_path = here.parent / directory / "execution_weste_26862/analysis.json"
    analysis = json.loads(analysis_path.read_text())
    raw_path = Path(analysis["sources"]["raw"]["path"])
    selective_path = Path(analysis["sources"]["selective-store"]["path"])
    raw = json.loads(raw_path.read_text())
    selective = json.loads(selective_path.read_text())
    releases = [e for e in selective["events"] if e["event"] == "target_new_output"]
    snapshots = {s["step"]: s for s in selective["eligibility_snapshots"]}
    schedules = {s["step"]: s for s in raw["scheduler_steps"]}
    forced = {(e["step"], e["victim"]) for e in selective["events"] if e["event"] == "commit_check" and e.get("reason") == "READY"}
    native = {}
    for event in raw["preemption_events"]:
        key = (event["attempted_step"], event["victim_internal_request_id"])
        if event.get("original_preemption_returned", True) and key not in forced:
            native.setdefault(key[0], []).append(key[1])
    rows = []
    for event in releases:
        step, rid = event["step"], event["request"]
        snap = snapshots[step]
        running = [snap["requests"][r] for r in snap["running_ids"]]
        reasons = []
        if snap["plan_target"] is not None:
            reasons.append("plan_in_progress")
        if snap["protected_id"] is not None or snap["protected_reserve"] != 0:
            reasons.append("protection_not_released")
        if any(r["status"] != "RUNNING" or r["output"] <= 0 or r["computed"] != r["prompt"]+r["output"]-1 for r in running):
            reasons.append("not_all_running_decode_ready")
        if len(running) > min(analysis["config"]["max_num_batched_tokens"], analysis["config"]["engine_max_num_seqs"]):
            reasons.append("token_or_sequence_budget")
        positions = [[r["computed"], r["held_blocks"] if isinstance(r["held_blocks"], int) else sum(r["held_blocks"])] for r in running]
        growth = lambda h: sum(max(0, (c+h+15)//16-held) for c, held in positions)
        free = snap["free_blocks"]
        victims1 = native.get(step, [])
        victims2 = victims1 + native.get(step+1, [])
        row = dict(step=step, target=raw["internal_to_source"][rid], eligible=not reasons, exclusion_reasons=reasons,
            free_blocks=free, running_count=len(running), computed_and_held_blocks=positions,
            growth1=growth(1), growth2=growth(2), alarm1=growth(1)>free, alarm2=growth(2)>free,
            free_zero=free == 0, native1=bool(victims1), native2=bool(victims2),
            target_native1=rid in victims1, target_native2=rid in victims2,
            native_victims1=[raw["internal_to_source"][v] for v in victims1],
            native_victims2=[raw["internal_to_source"][v] for v in victims2])
        # Outcomes below are explanatory labels only, never forecast inputs.
        row["observed_next_snapshot_free"] = snapshots.get(step+1, {}).get("free_blocks")
        row["observed_current_scheduled_tokens"] = schedules[step]["total_scheduled_tokens"]
        row["observed_next_scheduled_tokens"] = schedules.get(step+1, {}).get("total_scheduled_tokens")
        row["forced_within_two_steps"] = any(s in (step, step+1) for s, _ in forced)
        rows.append(row)
    applicable = [r for r in rows if r["eligible"]]
    results["groups"][name] = dict(source_analysis=str(analysis_path), source_raw=str(raw_path), source_selective=str(selective_path),
        release_boundaries=len(rows), applicable=len(applicable),
        one_step=confusion(applicable, "alarm1", "native1"), free_only_one_step=confusion(applicable, "free_zero", "native1"),
        two_step=confusion(applicable, "alarm2", "native2"), free_only_two_step=confusion(applicable, "free_zero", "native2"),
        one_step_alarms=sum(r["alarm1"] for r in applicable), two_step_alarms=sum(r["alarm2"] for r in applicable),
        rows=rows)
with args.output.open("x") as stream:
    json.dump(results, stream, indent=2)
    stream.write("\n")
for name, result in results["groups"].items():
    print(name, json.dumps({k:v for k,v in result.items() if k not in ("rows", "source_analysis", "source_raw", "source_selective")}, sort_keys=True))
    print("alerts_or_native", json.dumps([{k:v for k,v in r.items() if k != "computed_and_held_blocks"} for r in result["rows"] if r["alarm1"] or r["alarm2"] or r["native1"] or r["native2"]], sort_keys=True))
