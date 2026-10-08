#!/usr/bin/env python3
"""Summarize recorded native128 capacity states; no policy or counterfactual."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(run):
    names = ("status.json", "resolved-scheduler.json", "measured-steps.json",
             "measured-outputs.json", "native-drain.json")
    raw = {name: (run / name).read_bytes() for name in names}
    data = {name: json.loads(value) for name, value in raw.items()}
    status, resolved = data["status.json"], data["resolved-scheduler.json"]
    trace, outputs = data["measured-steps.json"], data["measured-outputs.json"]
    drain = data["native-drain.json"]
    if not (status["status"] == "COMPLETE" and status["request_count"] == 128
            and len(outputs) == 128 and all(r["finished"] for r in outputs)
            and len({r["request_id"] for r in outputs}) == 128
            and drain["status"] == "QUALIFIED"
            and resolved == dict(max_num_seqs=128, max_num_batched_tokens=1024,
                                 prefix_caching=True, allocated_kv_blocks=4097,
                                 usable_kv_blocks=4096)):
        raise ValueError("complete native128 runtime/request/drain contract differs")
    calls = trace["scheduler_calls"]
    if not calls or len(calls) != status["schedule_calls"]:
        raise ValueError("schedule observation inventory differs")
    states = [c[phase] for c in calls for phase in ("before", "after")]
    if any(not (0 <= s["used_blocks"] <= 4096 and 0 <= s["running"] <= 128
                and 0 <= s["waiting"] <= 128) for s in states):
        raise ValueError("recorded physical/queue state outside qualified domain")
    if any(c["call"] != i or not 0 <= c["scheduled_tokens_total"] <= 1024
           or not 0 <= c["scheduled_request_count"] <= 128
           for i, c in enumerate(calls)):
        raise ValueError("call identity or native scheduled budget differs")
    preempted = [rid for c in calls for rid in c["preempted_request_ids"]]
    if preempted != trace["preempted_request_ids"] or len(preempted) != status["preemptions"]:
        raise ValueError("native preemption observations disagree")
    peak = max(s["used_blocks"] for s in states)
    free = 4096 - peak
    conclusion = ("NATIVE_PREEMPTIONS_OBSERVED_REQUIRE_CAUSAL_INSPECTION" if preempted
                  else "FULL_POOL_WITHOUT_OBSERVED_PREEMPTION" if free == 0
                  else "NO_PREEMPTION_WITH_OBSERVED_HEADROOM")
    return dict(
        schema="c-instruct-native128-capacity-analysis-v1", status="COMPLETE",
        requests_planned=128, requests_completed=128,
        resolved_scheduler=resolved, schedule_calls=len(calls),
        peak_used_blocks_before_schedule=max(c["before"]["used_blocks"] for c in calls),
        peak_used_blocks_after_schedule=max(c["after"]["used_blocks"] for c in calls),
        peak_used_blocks_after_step=max(s["used_blocks_after_step"] for s in trace["steps"]),
        minimum_observed_free_blocks=free,
        observed_headroom_fraction=free / 4096,
        calls_with_full_pool_before_or_after=sum(
            any(c[phase]["used_blocks"] == 4096 for phase in ("before", "after"))
            for c in calls),
        peak_running=max(s["running"] for s in states),
        peak_waiting=max(s["waiting"] for s in states),
        peak_scheduled_tokens_per_call=max(c["scheduled_tokens_total"] for c in calls),
        total_scheduled_tokens=sum(c["scheduled_tokens_total"] for c in calls),
        preemption_events=len(preempted), unique_preempted_requests=len(set(preempted)),
        output_tokens=sum(len(r["output_token_ids"]) for r in outputs),
        observation_end_s=status["observation_end_s"], conclusion=conclusion,
        no_preemption_with_positive_observed_headroom=(
            conclusion == "NO_PREEMPTION_WITH_OBSERVED_HEADROOM"),
        limitations=[
            "Before/after-schedule and after-step snapshots do not expose every internal allocation transient.",
            "Waiting counts also include token-budget and admission-order effects; they are not classified as KV stalls.",
            "Positive observed headroom alone is not a proof that every waiting prompt could have fit; inspect its magnitude and concurrency before the stopping decision.",
            "Timing is instrumented development evidence, not a 32-vs-128 or equal-work comparison.",
            "A preemption or full pool alone would not establish a policy benefit; quality is scored separately for all128.",
        ],
        input_sha256={name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "peak_used_blocks_after_schedule",
        "minimum_observed_free_blocks", "preemption_events", "conclusion")}))


if __name__ == "__main__":
    main()
