#!/usr/bin/env python3
"""CPU diagnostic, not a policy/performance simulation or a fitted cost model.

At each logged native decision with prefill backlog, active decoders come from
the current scheduled rows (computed_start >= prompt length). Only timestamps
strictly before decision_time_s determine observed output count m and TTFT F.
Lhat=max_tokens is the explicitly known fixed-output contract: oracle length,
an upper-information component baseline, not a length prediction. One global
tau=(20-4)/512=0.03125s is fixed independently of observed outcomes.
"""
import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parents[1]))
import analyze as native

TAU = (20-4)/512


def stats(values):
    return dict(native.describe(values), min=min(values) if values else None,
                median=statistics.median(values) if values else None)


def inspect(path):
    data = path.read_bytes(); raw = json.loads(data)
    requests = {r["request_id"]: r for r in raw["requests"]}
    records, errors, no_decoder = [], [], 0
    for i, step in enumerate(raw["steps"]):
        if step["prefill_backlog_tokens_before"] <= 0:
            continue
        d = step["pulse_decision"]; now = d["decision_time_s"]
        decoding = [r for r in step["requests"] if r["decode_tokens"] > 0
                    and r["computed_start"] >= requests[r["request_id"]]["prompt_tokens"]]
        checks = {"native_D_matches": len(decoding) == d["decode_count"] == step["decode_count_before"],
                  "backlog_matches": d["prefill_backlog_tokens"] == step["prefill_backlog_tokens_before"],
                  "decision_in_completed_host_step": step["end_s"] is not None
                  and step["start_s"] <= now <= step["end_s"] and d.get("forward_completed") is True}
        if not all(checks.values()): errors.append(dict(step=i, checks=checks))
        slacks = []
        for row in decoding:
            q = requests[row["request_id"]]; m = bisect_left(q["token_times_s"], now)
            if not (m > 0 and m == row["computed_start"]-q["prompt_tokens"]+1 and row["decode_tokens"] == 1):
                errors.append(dict(step=i, request_id=row["request_id"], observed_m=m,
                                   expected_m=row["computed_start"]-q["prompt_tokens"]+1))
                continue
            f = q["token_times_s"][0]-q["arrival_s"]
            slack = q["arrival_s"]+max(f, 20-q["max_tokens"]*TAU)+(m+1)*TAU-now
            slacks.append((slack, row["request_id"], m, f, q["max_tokens"]))
        if not decoding: no_decoder += 1
        if len(slacks) != len(decoding) or not all(checks.values()): continue
        minimum = min(slacks) if slacks else None
        records.append(dict(step_index=i, decision_time_s=now, D=len(decoding),
            backlog_tokens=step["prefill_backlog_tokens_before"], current_actual_P=step["prefill_tokens"],
            current_cap=step["budget"], host_step_duration_s=step["end_s"]-step["start_s"],
            postdecision_to_step_end_s=step["end_s"]-now,
            minimum_slack_s=minimum[0] if minimum else None,
            witness_request_id=minimum[1] if minimum else None,
            witness_observed_m=minimum[2] if minimum else None,
            witness_observed_F_s=minimum[3] if minimum else None,
            witness_oracle_Lhat=minimum[4] if minimum else None))
    active = [r for r in records if r["D"] > 0]
    negative = [r for r in active if r["minimum_slack_s"] < 0]
    positive = [r for r in active if r["minimum_slack_s"] > 0]
    zero_p = [r for r in negative if r["D"] >= 128]
    streaks = []
    for r in zero_p:
        if not streaks or r["step_index"] != streaks[-1][-1]["step_index"]+1: streaks.append([])
        streaks[-1].append(r)
    longest = max(streaks, key=len, default=[])
    over = [r for r in positive if r["host_step_duration_s"] > r["minimum_slack_s"]]
    return dict(cell=path.parent.name, is_warmup=path.parent.name.startswith("warm"),
        raw_path=str(path), raw_sha256=hashlib.sha256(data).hexdigest(), policy=raw["policy"],
        extraction_valid=not errors, alignment_error_count=len(errors), alignment_error_examples=errors[:20],
        total_native_decisions=len(raw["steps"]), backlog_decisions=sum(s["prefill_backlog_tokens_before"] > 0 for s in raw["steps"]),
        backlog_decisions_without_decoder=no_decoder, aligned_backlog_decoder_decisions=len(active),
        negative_slack_decisions=len(negative), negative_slack_fraction_of_backlog_decoder_decisions=len(negative)/len(active) if active else None,
        positive_slack_decisions=len(positive), exactly_zero_slack_decisions=sum(r["minimum_slack_s"] == 0 for r in active),
        negative_minimum_slack_s=stats([r["minimum_slack_s"] for r in negative]),
        minimum_total128_implied_P_distribution=dict(Counter(str(max(0, 128-r["D"])) for r in negative)),
        negative_slack_D_at_least128_implied_P0_decisions=len(zero_p),
        implied_P0_fraction_of_negative_decisions=len(zero_p)/len(negative) if negative else None,
        implied_P0_fraction_of_backlog_decoder_decisions=len(zero_p)/len(active) if active else None,
        implied_P0_D=stats([r["D"] for r in zero_p]), implied_P0_backlog_tokens=stats([r["backlog_tokens"] for r in zero_p]),
        observed_current_prefill_tokens_during_implied_P0_decisions=sum(r["current_actual_P"] for r in zero_p),
        implied_P0_streak_count=len(streaks), longest_consecutive_implied_P0=dict(decisions=len(longest),
            first=longest[0] if longest else None, last=longest[-1] if longest else None,
            observed_first_to_last_decision_s=longest[-1]["decision_time_s"]-longest[0]["decision_time_s"] if longest else None,
            backlog_tokens=stats([r["backlog_tokens"] for r in longest])),
        first_negative_slack=negative[0] if negative else None,
        positive_slack_vs_current_host_step=dict(decisions=len(positive), minimum_slack_s=stats([r["minimum_slack_s"] for r in positive]),
            actual_host_duration_s=stats([r["host_step_duration_s"] for r in positive]),
            duration_exceeds_slack_decisions=len(over), fraction=len(over)/len(positive) if positive else None,
            slack_minus_actual_host_duration_s=stats([r["minimum_slack_s"]-r["host_step_duration_s"] for r in positive]),
            postdecision_duration_exceeds_slack_decisions=sum(r["postdecision_to_step_end_s"] > r["minimum_slack_s"] for r in positive)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", nargs="?", type=Path, default=ROOT.parent/"timing01")
    parser.add_argument("--output", type=Path, default=ROOT/"ttlt_slack.json")
    args = parser.parse_args(); rows = [inspect(p) for p in sorted(args.run_directory.rglob("raw.json"))]
    formal = [r for r in rows if not r["is_warmup"]]
    counts = {k: sum(r[k] for r in formal) for k in ("aligned_backlog_decoder_decisions", "negative_slack_decisions", "negative_slack_D_at_least128_implied_P0_decisions")}
    result = dict(formula="arrival_s + max(observed_F_s, 20 - oracle_Lhat*tau) + (observed_m+1)*tau - decision_time_s",
        tau_s=TAU, oracle_length="Known fixed max_tokens contract; upper-information baseline, not predicted length.",
        state_alignment="Native decode rows at computed_start >= prompt length; m uses bisect_left(token_times_s,decision_time_s); checked m=computed_start-prompt+1 and decoder count. No future completion or emitted length is used.",
        scope="CPU trace diagnostic only. The min-total128 fallback implies P=max(0,128-D), without progress adaptation. Current recorded trajectories and host durations are not counterfactual execution, GPU timings, online cost predictions, or fallback performance. Warm runs are labeled and excluded from formal aggregate.",
        formal_run_count=len(formal), all_extractions_valid=bool(rows) and all(r["extraction_valid"] for r in rows),
        formal_aggregate_counts=counts, runs=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(formal_run_count=len(formal), all_extractions_valid=result["all_extractions_valid"], formal_aggregate_counts=counts, output=str(args.output))))


if __name__ == "__main__":
    main()
