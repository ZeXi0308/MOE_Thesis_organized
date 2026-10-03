"""Descriptive pre-action timing localization; no counterfactual outcome replay."""
import importlib.util
import json
from collections import Counter
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("existing", root / "analyze_repeated_kv_service.py")
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)
data = json.loads((root / "analysis/analysis.json").read_text())
result = {"evidence": "DIAGNOSTIC_OBSERVATIONAL", "arms": {}}
for arm in ("diag-off", "diag-on"):
    cell = data["cells"][arm]
    segments = cell["diagnostic"]["segments"]
    groups = {}
    for name, rows in (("all", segments), ("gap_over_1s", [s for s in segments if s["F_s"] - s["L_s"] > 1])):
        gap = sum(s["F_s"] - s["L_s"] for s in rows)
        wait = sum(s["S"]["time_s"] - s["L_s"] for s in rows)
        groups[name] = dict(n=len(rows), summed_request_gap_s=gap, summed_L_to_S_s=wait,
            summed_S_to_F_s=gap-wait, L_to_S_fraction=wait/gap,
            median_gap_s=median(s["F_s"]-s["L_s"] for s in rows),
            median_S_to_F_s=median(s["F_s"]-s["S"]["time_s"] for s in rows))
    result["arms"][arm] = dict(counts=cell["diagnostic"]["counts"], groups=groups,
        repreempted_useful_outputs=sorted(s["useful_outputs"] for s in segments if s["end"] == "repreempted"))

cell = data["cells"]["diag-on"]
base = root / "execution_weste_26862/readback/results/diag-on"
selective = json.loads((base / "selective-store.json").read_text())
raw = json.loads((base / "raw.json").read_text())
origin = raw["measurement_origin_perf_counter_s"]
offload = cell["diagnostic"]["offload"]
completed = {str(j["job_id"]): e["time_s"] for e in offload["completed_jobs"] for j in e["jobs"]}
snapshots = {s["step"]: s for s in selective["eligibility_snapshots"]}
counts, intervals = Counter(), {}
for decision in selective["selector_decisions"]:
    if decision["reason"] != "swap cooldown":
        continue
    snap = snapshots[decision["step"]]
    cfg = snap["tracker"]["config"]
    ids = [r for r in snap["waiting_ids"] + snap["skipped_ids"] if snap["requests"][r]["status"] == "PREEMPTED"]
    ages = [(snap["step"] - snap["tracker"]["absent_since"].get(r, snap["step"]), r) for r in ids]
    if not ages:
        counts["no_preempted_target"] += 1
        continue
    age = max(a for a, _ in ages)
    targets = [r for a, r in ages if a == age]
    if len(targets) != 1:
        counts["age_tie_not_resolved_by_id"] += 1
        continue
    if age < cfg["min_absence_steps"]:
        counts["absence_threshold_not_met"] += 1
        continue
    if snap["plan_target"] is not None or not snap["cohort_active"]:
        counts["plan_or_cohort_blocked"] += 1
        continue
    observed = analyzer.eligibility(snap, targets[0], origin, offload["dispatch"], completed)
    if not (observed["direct"] or observed["fixed_most_funded"]):
        counts["not_resource_funded"] += 1
        continue
    counts["resource_and_recorded_context_candidate"] += 1
    counts["candidate_with_pending_store"] += bool(observed["pending_store_jobs"])
    last = snap["tracker"]["last_swap_step"]
    intervals.setdefault(last, dict(last_swap=last, first_candidate_step=snap["step"],
        steps_since_swap=observed["steps_since_swap"], target_age_steps=age,
        target=raw["internal_to_source"][targets[0]], direct=observed["direct"]))
result["save_on_cooldown_localization"] = dict(counts=dict(counts), intervals=list(intervals.values()))
result["limits"] = [
    "Diagnostic host timing is not the primary lightweight performance comparison.",
    "Summed gaps are overlapping request time, not disjoint episode wall or recoverable wall time.",
    "E is resource-only. Candidate contexts do not establish native queue acceptance or joint prepare-batch growth.",
    "Changing cadence alters future state; these observations are neither a replayed policy nor an outcome upper bound.",
    "S is host submission/call start, not device DMA/kernel start. No actual EOS or future outcome enters candidate detection."
]
output = Path(__file__).with_name("summary.json")
with output.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(output)
