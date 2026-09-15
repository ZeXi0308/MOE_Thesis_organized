#!/usr/bin/env python3
"""Find observed FCFS compute-budget exhaustion with an unused request slot.

This is an existence diagnostic on actual pre-schedule state, not an action
replay, latency prediction, or upper bound on an alternative policy.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
records = []
for path in sorted((ROOT / "readback/results").glob("*/*-raw.json")):
    raw = json.loads(path.read_text())
    if raw["phase"] != "cell" or raw["plan"]["waiting_order"] != "fcfs":
        continue
    events, large_steps = [], 0
    for step in raw["scheduler_steps"]:
        large_steps += any(r["prefill_tokens"] > 512 for r in step["scheduled"])
        action = raw["waiting_order_actions"][step["waiting_order_action_index"]]
        running, waiting = action["running_before"], action["before"]
        if len(running) >= step["target_cap"] or not waiting:
            continue
        head = waiting[0]
        if head["prompt_tokens"] != 128 or head["computed_tokens"] != 0:
            continue
        running_ids = {r["request_id"] for r in running}
        large = [r for r in step["scheduled"]
                 if r["internal_request_id"] in running_ids and r["prefill_tokens"] > 512]
        if not large or step["total_scheduled_tokens"] != step["runtime_token_budget"]:
            continue
        # The witness must show that this already-arrived head was not served.
        scheduled_ids = {r["internal_request_id"] for r in step["scheduled"]}
        if head["request_id"] in scheduled_ids:
            continue
        events.append(dict(
            step=step["step"], time_s=step["start_s"],
            running_count=len(running), cap=step["target_cap"],
            budget=step["runtime_token_budget"],
            actual_scheduled_tokens=step["total_scheduled_tokens"],
            waiting_count=len(waiting),
            waiting_head=raw["internal_to_source"][head["request_id"]],
            waiting_head_prompt_tokens=head["prompt_tokens"],
            running_prefills=[dict(request_id=r["request_id"],
                computed_before=r["scheduled_start_computed"],
                actual_prefill_tokens=r["prefill_tokens"]) for r in large]))
    records.append(dict(source=str(path.relative_to(ROOT)), block=path.parent.name,
        cohort=raw["plan"]["cohort"], large_prefill_steps=large_steps,
        budget_exhausted_with_spare_slot_and_short_waiter=len(events), events=events))

result = dict(
    classification="STRUCTURAL_OBSERVED_STATE_NO_COUNTERFACTUAL",
    question="Does one running prefill consume the remaining token budget while a request slot and an arrived short FCFS head are available?",
    threshold_for_identifying_large_steps=512,
    source_reference="runtime_reference.json::v1/core/sched/scheduler.py",
    per_request_limit_source_lines=[507, 508, 861, 863],
    records=records,
    limitations=["No alternative-policy execution or regenerated future state.",
        "An unused request slot does not by itself prove KV allocation would succeed.",
        "Observed event count is neither a latency saving nor a policy-wide headroom bound."])
destination = ROOT / "analysis/prefill_sharing_opportunities.json"
with destination.open("x") as handle:
    json.dump(result, handle, indent=2, allow_nan=False)
    handle.write("\n")
print(json.dumps([{k: r[k] for k in ("block", "cohort", "large_prefill_steps",
    "budget_exhausted_with_spare_slot_and_short_waiter")} for r in records]))
