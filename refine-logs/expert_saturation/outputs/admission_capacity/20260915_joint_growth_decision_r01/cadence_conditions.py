"""Reuse existing funding analysis; check necessary conditions, never replay outcomes."""
import argparse
import json
from pathlib import Path

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=here / "cadence_conditions.json")
args = parser.parse_args()
groups = {"D_selected": "20260915_natural_saved_recovery_gate_r01", "E_native_full": "20260915_natural_native_full_gate_r01"}
result = {"scope": "Necessary state conditions on observed current20 trajectories; neither accepted READY nor policy replay.", "groups": {}}
for name, folder in groups.items():
    source = here.parent / folder / "execution_weste_26862/analysis.json"
    analysis = json.loads(source.read_text())
    selective = json.loads(Path(analysis["sources"]["selective-store"]["path"]).read_text())
    snapshots = {s["step"]: s for s in selective["eligibility_snapshots"]}
    cooldown = [r for r in analysis["funding_observations"] if r["gate"] == "selector:swap cooldown"]
    candidates = [r for r in cooldown if not r["direct"] and r["fixed_most_funded"]
        and r["absence_steps"] >= 30 and r["plan_target"] is None]
    intervals, ties, growth_shortfalls, prefix_failures = {}, [], [], []
    first = None
    for row in candidates:
        snap = snapshots[row["step"]]
        cfg = snap["tracker"]["config"]
        waiting = [r for r in snap["waiting_ids"] if r in snap["tracker"]["absent_since"] and snap["requests"][r]["status"] == "PREEMPTED"]
        ages = [snap["step"]-snap["tracker"]["absent_since"][r] for r in waiting]
        eligible_victims = [r for r in snap["running_ids"]
            if max(0, snap["requests"][r]["computed"]-snap["requests"][r]["prompt"])/max(1, snap["requests"][r]["max_output"]) < cfg["protect_progress_fraction"]
            and snap["tracker"]["absence_count"].get(r, 0) < cfg["max_absences_per_request"]
            and snap["step"]-snap["tracker"]["resident_since"].get(r, -10**9) >= cfg["min_residency_steps"]]
        outputs = [snap["requests"][r]["output"] for r in eligible_victims]
        if ages.count(max(ages)) != 1 or outputs.count(max(outputs)) != 1:
            ties.append(snap["step"])
            continue  # This screen never depends on a request-ID tie break.
        victim = snap["requests"][row["victim"]]
        prefix_blocks = victim["computed"]//16
        prefix_ok = prefix_blocks > 0 and victim["held_blocks"] >= prefix_blocks
        if not prefix_ok:
            prefix_failures.append(snap["step"])
        growth = sum(max(0, (snap["requests"][r]["computed"]+16)//16-snap["requests"][r]["held_blocks"]) for r in snap["running_ids"])
        context = dict(step=snap["step"], last_swap=snap["tracker"]["last_swap_step"],
            steps_since_swap=row["steps_since_swap"], absence_steps=row["absence_steps"],
            free_blocks=row["effective_free_blocks"], target_need_blocks=row["need_blocks"], victim_held_blocks=row["victim_held_blocks"],
            joint_prepare_growth_blocks=growth, prefix_count_condition=prefix_ok)
        intervals.setdefault(context["last_swap"], context["step"])
        if first is None:
            first = context
        if growth > snap["free_blocks"]:
            growth_shortfalls.append(context)
    result["groups"][name] = dict(source_analysis=str(source), cooldown_funding_rows=len(cooldown),
        direct_funded_rows=sum(r["direct"] for r in cooldown), mature_fixed_most_rows=len(candidates),
        ranking_tie_rows=len(ties), prefix_count_failure_rows=len(prefix_failures),
        distinct_observed_cooldown_intervals=len(intervals), earliest_context=first,
        pending_store_rows=sum(bool(r["pending_store_jobs"]) for r in candidates),
        prepare_growth_shortfalls=growth_shortfalls, interval_first_contexts=intervals)
result["limits"] = ["All counts are correlated snapshots, not independent trials or a bound on extra rotations.",
    "Direct funding returns native-resume/no-rotation and is excluded from cadence action evidence.",
    "Stored prefixes and pending-store observations do not prove the future commit, transfer completion, or host-cache residency.",
    "One-step prepare-growth arithmetic is a feasibility caveat; the retired G2 release predictor is not reopened.",
    "No EOS, final request length, identifier feature or future rollout cost enters this screen."]
with args.output.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
for name, group in result["groups"].items():
    print(name, json.dumps({k:v for k,v in group.items() if k != "interval_first_contexts"}, sort_keys=True))
