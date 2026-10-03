"""One structural decision screen, not a timing forecast or GPU experiment.

Compare the accepted save-on policy with removal of its global swap cooldown.
Keep target/victim ranking, per-request guards, protection and token share fixed.
The observed fixed output cap is a configured limit, not a predicted natural EOS.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path
from statistics import mean


bundle = Path(__file__).resolve().parents[1]
outputs = bundle.parent
model = outputs / "20260915_repeated_staged_probe_r02/execution_weste_26862/prediction_check/model_skipped_guard.py"
inputs = outputs / "20260914_repeated_staged_model_r01/input.json"
sys.path.insert(0, str(bundle / "pkg"))
from absence_rotation import AbsenceRotation, RequestView

simulate = runpy.run_path(str(model))["simulate"]
data = json.loads(inputs.read_text())


class ReadyWithoutGlobalCooldown(AbsenceRotation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.min_steps_between_swaps = 0


results = {}
for name, cls in (("current_save_on", AbsenceRotation),
                  ("save_on_without_global_cooldown", ReadyWithoutGlobalCooldown)):
    decisions = []
    r = simulate(data["state"], data["history"], "staged_most_on", cls,
                 RequestView, restore_delay_steps=2, decision_log=decisions)
    trace = r["trace"]
    gaps, previous, loads = [], {}, []
    recompute = 0
    for row in trace:
        loads.extend(row.get("loads", []))
        for rid, x in row["scheduled"].items():
            total = data["state"]["requests"][rid]["prompt"] + x["output"]
            recompute += max(0, min(x["tokens"], total - 1 - x["computed"]))
        for rid in row["outputs"]:
            if rid in previous:
                gaps.append(row["step"] - previous[rid])
            previous[rid] = row["step"]
    results[name] = dict(
        status=r["status"], config=vars(cls().config),
        last_step=r["last_step"], total_calls=r["last_step"] + 1,
        mean_completion_step=mean(r["completed"].values()),
        completion_steps=r["completed"],
        max_gap_steps_after_first_post_cutoff_output=max(gaps),
        recompute_positions_after_cutoff=recompute,
        rotations=sum(e["event"] == "commit" for e in r["staging_events"]),
        load_jobs=len(loads), load_positions=sum(x["tokens"] for x in loads),
        stored_positions=r["stored_tokens"],
        modeled_peak_live_saved_bytes=r["peak_host_tokens"] * 131072,
        live_saved_capacity_exceeds_16GiB=r["peak_host_tokens"] * 131072 > 16 * 1024**3,
        decisions=decisions, staging_events=r["staging_events"],
    )

report = dict(
    evidence="STRUCTURAL_CONDITIONAL_ACTION_SCREEN",
    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (model, inputs, bundle / "pkg/absence_rotation.py")},
    question="Does removing only global cooldown create a useful candidate, or mainly more recovery traffic?",
    fixed=dict(save=True, load_delay_steps=2, host_budget_bytes=16 * 1024**3,
               per_request_absence_steps=30, per_request_residency_steps=30,
               victim_order="most_output", target_order="longest_absence",
               protection="until first new output", token_budget=1024),
    results=results,
    limits=[
        "No wall time, actual DMA, routes, token identities, quality or service benefit are predicted.",
        "Each arm independently evolves resource and request state; observed future notifications are not inputs.",
        "No store contention or eviction model. Live saved-state peak is only a necessary capacity check; native cache also retains completed prefixes.",
        "Gap steps exclude intervals crossing the common cutoff and are a structural churn diagnostic, not the primary pause metric.",
        "Only the configured forced-length domain; no natural EOS forecast or request-specific intervention.",
    ],
)
target = Path(__file__).with_name("analysis.json")
with target.open("x") as f:
    json.dump(report, f, indent=2)
    f.write("\n")
for name, r in results.items():
    print(name, json.dumps({k: v for k, v in r.items()
                           if k not in ("decisions", "staging_events", "completion_steps", "config")}))
