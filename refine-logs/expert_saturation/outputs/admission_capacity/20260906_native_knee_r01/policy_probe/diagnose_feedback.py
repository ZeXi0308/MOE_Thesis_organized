#!/usr/bin/env python3
"""Bounded descriptive feedback timing diagnosis; never mutates retained input."""
import bisect
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("policy_analysis", ROOT / "analyze_policy.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def distribution(values):
    values = sorted(values)
    def q(p):
        x = (len(values) - 1) * p
        a = int(x)
        return values[a] + (values[min(a + 1, len(values) - 1)] - values[a]) * (x - a)
    return dict(n=len(values), p50_s=q(.5), p95_s=q(.95), maximum_s=values[-1]) if values else dict(n=0)


analysis = m.read(ROOT / "analysis/analysis.json")
result = dict(episodes=[], limits=[
    "ITLs are host token receipt intervals grouped by their actual scheduler call; first tokens have no ITL.",
    "Each distribution uses one median of existing-request ITLs per completed call; consecutive steps are descriptive, not independent samples.",
    "A mixed step scheduled both prefill and decode; group differences do not isolate prefill causality from batch or context.",
    "Binding windows show waiting requests and the actual admission limit with token-budget room; they do not execute a cap32 counterfactual.",
    "Over-target spans use successive scheduler start timestamps and describe sampled running counts, not exact KV retirement times."])
for cell in analysis["cells"]:
    if cell["arm"] not in ("static32", "shadow32", "feedback32"):
        continue
    raw = m.read(ROOT / "gpu_results" / (cell["id"] + ".json"))
    steps, starts = raw["scheduler_steps"], [s["start_s"] for s in raw["scheduler_steps"]]
    itls = defaultdict(list)
    for request in raw["requests"]:
        for earlier, received in zip(request["token_times_s"], request["token_times_s"][1:]):
            i = bisect.bisect_right(starts, received) - 1
            m.require(i >= 0 and received >= steps[i]["end_s"], "host ITL/step alignment mismatch")
            itls[i].append(received - earlier)
    step_types = {i: "mixed" if any(r["prefill_tokens"] for r in s["scheduled"]) and any(r["decode_tokens"] for r in s["scheduled"])
                  else "pure_decode" if any(r["decode_tokens"] for r in s["scheduled"]) else "prefill_only" for i, s in enumerate(steps)}
    summary = {kind: distribution([median(v) for i, v in itls.items() if step_types[i] == kind]) for kind in ("mixed", "pure_decode", "prefill_only")}
    episode = dict(id=cell["id"], arm=cell["arm"], regime=cell["plan"]["regime"],
        throughput_rps=cell["throughput_rps"], goodput_rps=cell["goodput_rps"], n_slo_pass=cell["n_slo_pass"],
        violation_counts=cell["violation_counts"], step_median_itl=summary)
    if cell["arm"] == "feedback32":
        actions = raw["actions"]
        episode["action_sequence"] = [dict(step=d["decision_index"], applied_s=d["applied_s"],
            change=[d["intent_before"], d["intent_target"]], recent_itl_s=d["recent_step_median_itl_s"],
            active=d["active_before"], waiting=d["waiting_before"],
            window=[dict(step=j - 1, kind=step_types[j - 1], median_itl_s=median(itls[j - 1])) for j in d["window_completed_steps"]]) for d in actions]
        episode["action_bound_outcomes"] = cell["causal_and_action"]["actions"]
        episode["over_target_spans"] = []
        begin = None
        for i in range(len(steps) + 1):
            over = i < len(steps) and steps[i]["actual_active"] > steps[i]["target_cap"]
            if over and begin is None:
                begin = i
            if not over and begin is not None:
                span = steps[begin:i]
                end = steps[i]["start_s"] if i < len(steps) else raw["observation_end_s"]
                episode["over_target_spans"].append(dict(first_step=begin, last_step=i - 1, start_s=span[0]["start_s"],
                    end_s=end, span_s=end - span[0]["start_s"], targets=sorted({s["target_cap"] for s in span}),
                    active_range=[min(s["actual_active"] for s in span), max(s["actual_active"] for s in span)],
                    max_waiting_after=max(s["waiting_requests"] for s in span)))
                begin = None
        config = m.read(ROOT / "gpu_results" / cell["group"] / "engine_args.json")
        binding = [s for s in steps if s["waiting_requests"] > 0 and s["actual_active"] == s["effective_scheduler_limit"] < 32 and
                   config["max_num_batched_tokens"] - s["total_scheduled_tokens"] >= 128]
        first = binding[0] if binding else None
        episode["first_observed_binding_window"] = None if first is None else {k: first[k] for k in (
            "step", "start_s", "end_s", "target_cap", "running_before", "waiting_before", "actual_active", "waiting_requests", "effective_scheduler_limit", "total_scheduled_tokens")}
        episode["binding_steps_with_room_for_one_more_prompt"] = len(binding)
    result["episodes"].append(episode)
output = ROOT / "concise_feedback_diagnostic.json"
with output.open("x") as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write("\n")
print(output)
